"""Matched 200-document local/global attention feature comparison.

The original local attention candidate is frozen. This adds one full-attention
candidate with exactly the same documents, validation rows and linear solver.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import systemness_components as components
import systemness_data as data
import systemness_probes as probes

OUT=data.runtime.OUT/'systemness-global-attention-pilot'
CACHE=data.CACHE/'global-attention-pilot'
SPECS={f'attention-{layer}':dict(point='attention',layer=layer) for layer in [17,21,25]}
NAME='attention-global'


def inputs_and_rows():
    original_cache=components.CACHE
    try:
        # Reuse the unmodified, audited document/validation selector. Only its
        # output directory changes in this process; the previous pilot is intact.
        components.CACHE=CACHE
        rows,inputs,tokens=components.inputs_and_rows()
    finally:components.CACHE=original_cache
    assert data.digest(CACHE/'rows.parquet')==data.digest(original_cache/'rows.parquet')
    return rows,inputs,tokens


def extract(rows,inputs,tokens):
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts,collate
    signature=dict(rows_sha256=data.digest(CACHE/'rows.parquet'),features=SPECS,batch_size=8,
        verified_weights_sha256=data.digest(Path(os.environ['ROLE_ANALYSIS_LOAD_PATH'])/'verified-weights.json'))
    path=CACHE/'features.f16';progress_path=CACHE/'progress.json'
    if not path.exists():
        filesystem=os.statvfs(CACHE)
        assert filesystem.f_bavail*filesystem.f_frsize>len(rows)*8640*2+512*2**20
    array=np.memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=(len(rows),8640))
    start_batch=0;written=0
    if progress_path.exists():
        progress=json.loads(progress_path.read_text());assert progress['signature']==signature
        start_batch=progress['next_batch'];written=progress['written_rows']
        if progress.get('complete'):
            assert data.digest(path)==json.loads((OUT/'capture-complete.json').read_text())['sha256']
            return
    runtime=data.runtime
    runtime.atomic_json=lambda path,value:data.atomic_json(OUT/Path(path).name,value)
    runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
    model=runtime.load_model();loader=DataLoader(Prompts(inputs,tokens),batch_size=8,shuffle=False,collate_fn=collate)
    captured={};lookup={int(p):g for p,g in rows.groupby('prompt_ix',sort=False)}
    def receive(name,value):captured[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started=time.time()
    with torch.inference_mode():
        batch=next(iter(loader));sample=SimpleNamespace(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0))
        before=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        handles=components.hooks(model,SPECS,receive)
        after=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        assert torch.equal(before,after) and torch.isfinite(after).all()
        reference=dict(captured)
        for handle in handles:handle.remove()
        del before,after
        original=pipeline.install_hooks;pipeline.install_hooks=components.hooks
        try:
            for bi,batch,states in pipeline.feature_batches(model,loader,SPECS,start_batch,(sample,reference),
                    lambda result:data.atomic_json(OUT/'pipeline-validation.json',result)):
                width=batch['input_ids'].shape[1];positions=[];targets=[]
                for b,prompt in enumerate(batch['prompt_ix']):
                    group=lookup[prompt];offset=b*width+width-len(tokens[prompt])
                    positions.extend((group.token_ix.to_numpy()+offset).tolist());targets.extend(group.pilot_row.tolist())
                indices=torch.tensor(positions,dtype=torch.long)
                values=torch.cat([states[key][indices] for key in SPECS],dim=1)
                assert values.shape==(len(targets),8640) and torch.isfinite(values).all()
                array[targets]=values.numpy();written+=len(targets);states.clear()
                if (bi+1)%16==0 or bi+1==len(loader):
                    array.flush()
                    data.atomic_json(progress_path,dict(signature=signature,next_batch=bi+1,written_rows=written,complete=False))
                    record=dict(stage='extracting',batch=bi+1,batches=len(loader),rows=written,seconds=time.time()-started)
                    data.atomic_json(OUT/'status.json',record);print(json.dumps(record),flush=True)
        finally:pipeline.install_hooks=original
    assert written==len(rows)
    for start in range(0,len(array),4096):
        block=array[start:start+4096]
        assert np.isfinite(block).all() and (block!=0).any(axis=1).all()
    data.atomic_json(OUT/'capture-complete.json',dict(rows=written,dimensions=8640,features=SPECS,
        sha256=data.digest(path),signature=signature,source_sha256=data.digest(__file__),
        hooks_leave_hidden_states_exact=True,all_finite=True,seconds=time.time()-started))
    data.atomic_json(progress_path,dict(signature=signature,next_batch=len(loader),written_rows=written,complete=True))
    del model,captured,reference,sample
    gc.collect()
    for gpu in range(4):
        with torch.cuda.device(gpu):torch.cuda.empty_cache()
    allocations=[torch.cuda.memory_allocated(gpu) for gpu in range(4)]
    assert max(allocations)<2**30,allocations
    data.atomic_json(OUT/'gpu-memory-after-capture.json',dict(allocated_bytes=allocations))


def train():
    rows=pd.read_parquet(CACHE/'rows.parquet')
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['linear_trainer_sha256']==data.digest(components.__file__)
    # The original pilot's exact C=.005 scaler/solver/export implementation.
    components.OUT=OUT;components.CACHE=CACHE;components.SPECS=SPECS
    array=np.memmap(CACHE/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),8640))
    components.train((NAME,0,8640),0,rows,array)


def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('/logs/\n')
    for folder in ['models','candidates','logs']:(OUT/folder).mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(),'Global attention pilot is complete.'
    config_path=Path(os.environ['ROLE_ANALYSIS_LOAD_PATH'])/'config.json'
    config=json.loads(config_path.read_text())
    assert all(config['layer_types'][layer]=='full_attention' for layer in [17,21,25])
    assert all(config['layer_types'][layer]=='sliding_attention' for layer in [16,20,24])
    rows,inputs,tokens=inputs_and_rows()
    plan=dict(fit_documents=200,features=SPECS,local_control_layers=[16,20,24],global_layers=[17,21,25],
        comparison='Same 200 fit documents and full original validation; C=.005 linear readouts; feature location is the only planned difference.',
        config_sha256=data.digest(config_path),rows_sha256=data.digest(CACHE/'rows.parquet'),
        local_control_sha256=data.digest(components.OUT/'candidates/attention.json'),
        linear_trainer_sha256=data.digest(components.__file__),source_sha256=data.digest(__file__),
        promotion=dict(mean_system_f1_gain_at_least=.02,minimum_each_single_long_gain=.03,maximum_each_multi_natural_loss=.02),
        tests_read=False,previous_benchmark_already_reported=True,fresh_confirmation_required_if_promoted=True,
        inference='Pointwise 8640 hidden features only; no header labels, token IDs or history features.')
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan
    else:data.atomic_json(OUT/'plan.json',plan)
    extract(rows,inputs,tokens)
    with (OUT/'logs/linear.log').open('a') as log:
        worker=subprocess.run([sys.executable,'-u',__file__,'--worker'],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
        assert worker.returncode==0,'Inspect isolated linear worker log.'
    local=json.loads((components.OUT/'candidates/attention.json').read_text())['validation']
    result=json.loads((OUT/'candidates'/f'{NAME}.json').read_text())['validation']
    mean=result['mean_system_f1']-local['mean_system_f1']
    weak=min(result['profiles'][k]['system_f1']-local['profiles'][k]['system_f1'] for k in ['single','long-single'])
    other=max(local['profiles'][k]['system_f1']-result['profiles'][k]['system_f1'] for k in ['multi','natural'])
    promoted=mean>=.02 and weak>=.03 and other<=.02
    table=[]
    for name,v in [('local',local),('global',result)]:
        for profile,m in v['profiles'].items():
            table.append(dict(feature=name,profile=profile,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(table).to_csv(OUT/'validation-metrics.csv',index=False)
    assert plan['source_sha256']==data.digest(__file__)
    complete=dict(mean_system_f1_gain=mean,minimum_each_single_long_f1_gain=weak,maximum_each_multi_natural_loss=other,
        promoted=promoted,tests_read=False,source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'plan.json'),
        fresh_confirmation_required_if_promoted=True)
    data.atomic_json(OUT/'complete.json',complete)
    data.atomic_json(OUT/'status.json',dict(stage='complete',promoted=promoted))
    print(json.dumps(complete),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:train()
    else:main()
