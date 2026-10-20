"""Pilot pointwise attention/MLP branch readouts on fit/validation data only.

This follows the frozen residual-probe benchmark. Its published test scores are
already known; this pilot never reads them or uses test examples. Any promoted
method requires a further independent confirmation. A recoverable local memmap
keeps feature extraction independent of the isolated GPU readout processes.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_probes as probes

OUT=data.runtime.OUT/'systemness-component-pilot'
CACHE=data.CACHE/'component-pilot'
SPECS={f'{point}-{layer}':dict(point=point,layer=layer)
       for point in ['mid','attention','mlp'] for layer in [16,20,24]}
JOBS=[('residual',0,8640),('attention',8640,17280),('mlp',17280,25920),('attention+mlp',8640,25920)]


def device_preflight():
    code='''import json, os, cupy as cp, torch
from cuml.linear_model import LogisticRegression
assert cp.cuda.runtime.getDeviceCount()==torch.cuda.device_count()==1
cp.random.seed(7)
x=cp.asfortranarray(cp.random.normal(size=(256,128)).astype(cp.float32));y=cp.arange(256)%5
classifier=LogisticRegression(C=.005,max_iter=300,output_type='cupy');classifier.fit(x,y)
p=classifier.predict_proba(x)
assert bool(cp.isfinite(p).all()) and bool(cp.allclose(p.sum(1),1,atol=1e-6))
with torch.inference_mode():
 q=torch.softmax(torch.nn.functional.linear(torch.from_dlpack(x),torch.from_dlpack(classifier.coef_),torch.from_dlpack(classifier.intercept_).reshape(-1)),dim=1)
 difference=float(cp.max(cp.abs(cp.from_dlpack(q)-p)))
assert difference<2e-5,difference
print(json.dumps(dict(device=os.environ['CUDA_VISIBLE_DEVICES'],one_visible_device=True,finite_normalized_probabilities=True,torch_cuml_probability_difference=difference)))
'''
    workers=[subprocess.Popen([sys.executable,'-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for gpu in range(4)]
    records=[]
    for worker in workers:
        stdout,stderr=worker.communicate()
        assert worker.returncode==0,(stdout,stderr)
        records.append(json.loads(stdout.strip().splitlines()[-1]))
    data.atomic_json(OUT/'isolated-device-preflight.json',dict(gpus=records,all_passed=True,
        check='Implementation/device isolation only; not held-out accuracy evidence.'))
    print('Four isolated GPU device preflights passed.',flush=True)


def digest_array(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def hooks(model,specs,receive):
    handles=[]
    for name,spec in specs.items():
        layer=model.model.layers[spec['layer']]
        if spec['point']=='mid':
            def before(module,args,key=name):receive(key,args[0])
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(before))
        else:
            module=layer.self_attn if spec['point']=='attention' else layer.mlp
            def after(module,args,output,key=name):receive(key,output[0])
            handles.append(module.register_forward_hook(after))
    return handles


def inputs_and_rows():
    rows,_=probes.development()
    docs=sorted(rows[rows.split=='fit'].document_id.unique())[:200]
    rows=rows[(rows.split=='validation')|((rows.split=='fit')&rows.document_id.isin(docs))].copy().reset_index(drop=True)
    rows['source_set']=np.where(rows.sample_ix>=10**9,'supplement','main')
    rows['original_prompt_ix']=rows.prompt_ix
    tokens={};inputs=[];mapping={}
    for source,folder in [('main',data.CACHE),('supplement',data.CACHE/'supplement')]:
        source_inputs=pd.read_parquet(folder/'inputs.parquet')
        source_tokens=json.loads((folder/'token-ids.json').read_text())
        wanted=set(rows[rows.source_set==source].original_prompt_ix)
        for row in source_inputs[source_inputs.prompt_ix.isin(wanted)].itertuples():
            ix=len(inputs);mapping[(source,int(row.prompt_ix))]=ix
            tokens[ix]=source_tokens[row.prompt_ix]
            inputs.append(dict(prompt_ix=ix,length=row.length))
    rows['prompt_ix']=[mapping[(s,int(p))] for s,p in zip(rows.source_set,rows.original_prompt_ix)]
    rows['pilot_row']=np.arange(len(rows))
    assert set(rows.split)=={'fit','validation'} and set(rows.wrapper)=={'native'}
    assert not set(rows[rows.split=='fit'].group_id)&set(rows[rows.split=='validation'].group_id)
    assert rows[rows.split=='fit'].document_id.nunique()==200
    CACHE.mkdir(exist_ok=True)
    rows.to_parquet(CACHE/'rows.parquet',index=False)
    return rows,pd.DataFrame(inputs),tokens


def extract(rows,inputs,tokens):
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts,collate
    runtime=data.runtime
    path=CACHE/'features.f16'
    progress_path=CACHE/'progress.json'
    checkpoint=Path(os.environ.get('ROLE_ANALYSIS_LOAD_PATH','/dev/shm/role-analysis-gptoss120b'))
    signature=dict(rows_sha256=data.digest(CACHE/'rows.parquet'),specs=SPECS,batch_size=8,
        verified_weights_sha256=data.digest(checkpoint/'verified-weights.json'))
    needed=len(rows)*25920*2
    if not path.exists():
        filesystem=os.statvfs(CACHE)
        assert filesystem.f_bavail*filesystem.f_frsize>needed+512*2**20,'Insufficient space for recoverable feature cache.'
    storage=np.memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=(len(rows),25920))
    start_batch=0;written=0
    if progress_path.exists():
        progress=json.loads(progress_path.read_text());assert progress['signature']==signature
        start_batch=progress['next_batch'];written=progress['written_rows']
        if progress.get('complete'):
            complete=json.loads((OUT/'capture-complete.json').read_text())
            assert complete['sha256']==data.digest(path) and written==len(rows)
            print('Recovered complete component feature cache.',flush=True)
            return storage
    runtime.atomic_json=lambda path,value:data.atomic_json(OUT/Path(path).name,value)
    runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
    model=runtime.load_model()
    dataset=Prompts(inputs,tokens)
    loader=DataLoader(dataset,batch_size=8,shuffle=False,collate_fn=collate)
    lookup={int(p):g for p,g in rows.groupby('prompt_ix',sort=False)}
    captured={}
    def receive(name,value):captured[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started=time.time()
    with torch.inference_mode():
        batch=next(iter(loader))
        sample=SimpleNamespace(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0))
        before=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        handles=hooks(model,SPECS,receive)
        after=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        assert torch.equal(before,after) and torch.isfinite(after).all()
        reference=dict(captured)
        for handle in handles:handle.remove()
        del before,after
        # The installer is local to this process. Existing extraction code and
        # previously published midpoint features remain unchanged.
        original=pipeline.install_hooks
        pipeline.install_hooks=hooks
        try:
            for bi,batch,states in pipeline.feature_batches(model,loader,SPECS,start_batch,(sample,reference),
                  lambda result:data.atomic_json(OUT/'pipeline-validation.json',result)):
                width=batch['input_ids'].shape[1];positions=[];targets=[]
                for b,prompt in enumerate(batch['prompt_ix']):
                    group=lookup[prompt];offset=b*width+width-len(tokens[prompt])
                    positions.extend((group.token_ix.to_numpy()+offset).tolist());targets.extend(group.pilot_row.tolist())
                indices=torch.tensor(positions,dtype=torch.long)
                for col,key in enumerate(SPECS):
                    value=states[key][indices]
                    assert torch.isfinite(value).all()
                    storage[targets,col*2880:(col+1)*2880]=value.numpy()
                written+=len(targets);states.clear()
                if (bi+1)%16==0 or bi+1==len(loader):
                    storage.flush()
                    data.atomic_json(progress_path,dict(signature=signature,next_batch=bi+1,
                        written_rows=written,complete=False))
                    record=dict(stage='extracting',batch=bi+1,batches=len(loader),rows=written,seconds=time.time()-started)
                    data.atomic_json(OUT/'status.json',record);print(json.dumps(record),flush=True)
        finally:pipeline.install_hooks=original
    assert written==len(rows)
    for start in range(0,len(storage),4096):
        block=storage[start:start+4096]
        assert np.isfinite(block).all() and (block!=0).any(axis=1).all()
    data.atomic_json(OUT/'capture-complete.json',dict(rows=written,dimensions=25920,specs=SPECS,
        sha256=digest_array(storage),seconds=time.time()-started,all_finite=True,
        hooks_leave_hidden_states_exact=True,input_tokens=int(inputs.length.sum()),cache=str(path),signature=signature))
    data.atomic_json(progress_path,dict(signature=signature,next_batch=len(loader),written_rows=written,complete=True))
    del model,captured,reference,sample
    gc.collect()
    for gpu in range(4):
        with torch.cuda.device(gpu):torch.cuda.empty_cache()
    allocations=[torch.cuda.memory_allocated(gpu) for gpu in range(4)]
    data.atomic_json(OUT/'gpu-memory-after-capture.json',dict(allocated_bytes=allocations))
    assert max(allocations)<2**30,allocations
    return storage


def train(job,gpu,rows,storage):
    name,left,right=job;dims=right-left
    existing=OUT/'candidates'/f'{name}.json'
    if existing.exists():
        record=json.loads(existing.read_text())
        assert record['rows_sha256']==data.digest(CACHE/'rows.parquet')
        assert record['model_sha256']==data.digest(OUT/'models'/f'{name}.npz')
        return name,record['validation']
    import cupy as cp
    import torch
    from cuml.preprocessing import StandardScaler
    from cuml.linear_model import LogisticRegression
    assert cp.cuda.runtime.getDeviceCount()==torch.cuda.device_count()==1
    fit=rows[rows.split=='fit'];valid=rows[rows.split=='validation']
    with cp.cuda.Device(gpu),torch.cuda.device(gpu):
        def load(indices):
            result=cp.empty((len(indices),dims),dtype=cp.float32,order='F')
            for start in range(0,len(indices),4096):
                block=np.asarray(storage[indices[start:start+4096],left:right],dtype=np.float32)
                result[start:start+len(block)]=cp.asarray(block)
            return result
        started=time.time()
        x=load(fit.pilot_row.to_numpy());scaler=StandardScaler(copy=False)
        for start in range(0,len(x),4096):
            if hasattr(scaler,'n_samples_seen_'):scaler.n_samples_seen_=cp.asarray(scaler.n_samples_seen_)
            scaler.partial_fit(x[start:start+4096])
        scaler.transform(x,copy=False)
        xv=load(valid.pilot_row.to_numpy());scaler.transform(xv,copy=False)
        classifier=LogisticRegression(C=.005,max_iter=5000,linesearch_max_iter=100,tol=1e-4,output_type='cupy')
        classifier.fit(x,cp.asarray(probes.labels(fit)),convert_dtype=True)
        p=cp.asnumpy(classifier.predict_proba(xv))
        mean=cp.asnumpy(scaler.mean_).astype('float32');scale=cp.asnumpy(scaler.scale_).astype('float32')
        weight=cp.asnumpy(classifier.coef_);bias=cp.asnumpy(classifier.intercept_).reshape(-1)
        sample=np.linspace(0,len(valid)-1,256,dtype=int)
        raw=np.asarray(storage[valid.iloc[sample].pilot_row.to_numpy(),left:right],dtype=np.float32)
        with torch.inference_mode():
            z=(torch.from_numpy(raw).to(f'cuda:{gpu}')-torch.from_numpy(mean).to(f'cuda:{gpu}'))/torch.from_numpy(scale).to(f'cuda:{gpu}')
            actual=torch.softmax(torch.nn.functional.linear(z,torch.from_numpy(weight).to(f'cuda:{gpu}'),torch.from_numpy(bias).to(f'cuda:{gpu}')),dim=1).cpu().numpy()
        difference=float(np.abs(actual-p[sample]).max());assert difference<5e-4,difference
        metadata=dict(feature=name,columns=[left,right],dimensions=dims,C=.005,fit_documents=200,
                      feature_spec=[SPECS[k] for k in list(SPECS)[left//2880:right//2880]],
                      roles=data.ROLES,source_sha256=data.digest(__file__),seed=data.SEED)
        np.savez_compressed(OUT/'models'/f'{name}.npz',mean=mean,scale=scale,weight=weight,bias=bias,metadata=np.array(json.dumps(metadata)))
        result=probes.validation_metrics(valid,p)
        data.atomic_json(OUT/'candidates'/f'{name}.json',dict(**metadata,validation=result,
            seconds=time.time()-started,portable_probability_max_difference=difference,
            rows_sha256=data.digest(CACHE/'rows.parquet'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            model_sha256=data.digest(OUT/'models'/f'{name}.npz'),test_used_for_selection=False))
        np.savez_compressed(CACHE/f'{name}-validation.npz',p=p,sample_ix=valid.sample_ix.to_numpy())
        print(json.dumps(dict(stage='pilot_complete',feature=name,mean_system_f1=result['mean_system_f1'],
              profiles={k:{m:v[m] for m in ['accuracy','system_precision','system_recall','system_f1']} for k,v in result['profiles'].items()})),flush=True)
        return name,result


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('/logs/\n')
    for folder in ['models','candidates']:(OUT/folder).mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(), 'This pilot is already complete; inspect evidence before repeating.'
    rows,inputs,tokens=inputs_and_rows()
    plan=dict(fit_documents=200,validation='Same complete native validation partition as the 48 residual candidates',
              features=SPECS,jobs=JOBS,seed=data.SEED,language_model='Frozen MXFP4 on four A40s',
              hypothesis='Direct attention/MLP branch outputs may expose role information obscured by the large residual stream.',
              no_history_or_header_features=True,tests_read_by_this_pilot=False,
              previous_benchmark_already_reported=True,fresh_confirmation_required_if_promoted=True,
              rows_sha256=data.digest(CACHE/'rows.parquet'),readout_execution='one isolated process per GPU',
              feature_cache='recoverable memmap in project cache',
              source_sha256=data.digest(__file__),created_unix_time=time.time())
    if (OUT/'plan.json').exists():
        previous=json.loads((OUT/'plan.json').read_text())
        if previous['source_sha256']!=plan['source_sha256']:
            history_path=OUT/'run-history.json'
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            history.append(dict(previous_plan=previous,
                prior_capture=json.loads((OUT/'capture-complete.json').read_text()) if (OUT/'capture-complete.json').exists() else None,
                change=('Replace cross-GPU threads after CUDA illegal-address failure with isolated processes and durable cache.'
                    if not previous.get('readout_execution') else 'Correct runtime/preflight code; data, capture points and solver unchanged.'),
                updated_unix_time=time.time()))
            data.atomic_json(history_path,history)
    data.atomic_json(OUT/'plan.json',plan)
    device_preflight()
    storage=extract(rows,inputs,tokens)
    (OUT/'logs').mkdir(exist_ok=True)
    workers=[]
    for gpu,job in enumerate(JOBS):
        log=(OUT/'logs'/f'{job[0]}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(gpu)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log))
    data.atomic_json(OUT/'status.json',dict(stage='training',workers=[p.pid for p,_ in workers]))
    try:
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in [None,0] for p,_ in workers):raise RuntimeError('Isolated component worker failed; inspect its log.')
            time.sleep(5)
    finally:
        for p,log in workers:
            if p.poll() is None:p.terminate()
            log.close()
    assert all(p.returncode==0 for p,_ in workers)
    results=[(name,json.loads((OUT/'candidates'/f'{name}.json').read_text())['validation']) for name,_,_ in JOBS]
    table=[]
    for name,result in results:
        for profile,m in result['profiles'].items():
            table.append(dict(feature=name,profile=profile,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(table).to_csv(OUT/'validation-metrics.csv',index=False)
    data.atomic_json(OUT/'complete.json',dict(candidates=4,tests_used=False,
        best_feature=max(results,key=lambda r:r[1]['mean_system_f1'])[0],
        source_sha256=data.digest(__file__),all_models_portable=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.worker is None:main()
    else:
        rows=pd.read_parquet(CACHE/'rows.parquet')
        meta=json.loads((OUT/'capture-complete.json').read_text())
        assert meta['signature']['rows_sha256']==data.digest(CACHE/'rows.parquet')
        storage=np.memmap(CACHE/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),25920))
        train(JOBS[args.worker],0,rows,storage)
