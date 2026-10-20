"""Validation-only lexical nuisance removal, with portable folded readouts.

Use current-token embeddings only to estimate a nuisance subspace on training
examples. Inference consumes the same 8640 hidden features as before: the fixed
projection is folded into the first readout weight matrix, with no token IDs or
role/header metadata used by the consumer.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import systemness_data as data
import systemness_probes as probes

CONFIGS=[dict(rank=r,kind='linear') for r in [16,64,128]]+[dict(rank=64,kind='mlp',hidden=64)]
DESCRIPTOR_WIDTH=128


def prepare_embedding():
    import torch
    from safetensors import safe_open
    path=data.CACHE/'token-descriptors.f16'
    record=data.OUT/'lexical-token-descriptors.json'
    if path.exists() and record.exists():
        old=json.loads(record.read_text())
        assert data.digest(path)==old['descriptor_sha256']
        return
    model=Path(os.environ.get('ROLE_ANALYSIS_LOAD_PATH','/dev/shm/role-analysis-gptoss120b'))
    assert (model/'verified-weights.json').exists()
    index=json.loads((model/'model.safetensors.index.json').read_text())
    shard=model/index['weight_map']['model.embed_tokens.weight']
    with safe_open(shard,framework='pt',device='cpu') as handle:
        embedding=handle.get_tensor('model.embed_tokens.weight')
    assert embedding.shape[1]==2880
    generator=torch.Generator().manual_seed(data.SEED+19)
    projection=torch.randn((2880,DESCRIPTOR_WIDTH),generator=generator,dtype=torch.float32)/np.sqrt(2880)
    torch.set_float32_matmul_precision('highest')
    projected=np.memmap(path,mode='w+',dtype=np.float16,shape=(len(embedding),DESCRIPTOR_WIDTH))
    with torch.inference_mode():
        direction=projection.cuda()
        for start in range(0,len(embedding),8192):
            value=embedding[start:start+8192].to('cuda:0',torch.float32)@direction
            projected[start:start+len(value)]=value.to('cpu',torch.float16).numpy()
        projected.flush()
    data.atomic_json(record,dict(vocabulary=len(embedding),descriptor_width=DESCRIPTOR_WIDTH,
        seed=data.SEED+19,source_checkpoint=str(model),source_shard=shard.name,
        source_shard_sha256=data.digest(shard),embedding_dtype=str(embedding.dtype),
        descriptor_sha256=data.digest(path),source_sha256=data.digest(__file__),
        note='Fixed Gaussian projection of frozen token embeddings; no validation/test data or labels used.'))


def train(job):
    import cupy as cp
    import torch
    from cuml.preprocessing import StandardScaler
    from cuml.linear_model import LogisticRegression
    probes.setup()
    config=CONFIGS[job];rank=config['rank']
    key=f'large-mixed-lexical-r{rank}-{config["kind"]}'
    if (data.OUT/'candidates'/f'{key}.json').exists():return
    assert not (data.OUT/'selection.json').exists()
    meta=json.loads((data.OUT/'lexical-token-descriptors.json').read_text())
    descriptors=np.memmap(data.CACHE/'token-descriptors.f16',mode='r',dtype=np.float16,
                          shape=(meta['vocabulary'],DESCRIPTOR_WIDTH))
    rows,array=probes.development()
    fit=rows[rows.split=='fit'].copy();valid=rows[rows.split=='validation'].copy()
    paired=fit[fit.profile.isin(['single','long-single'])].copy()
    paired['position']=paired.groupby(['document_id','profile','role']).cumcount()
    check=paired.groupby(['document_id','profile','position']).token_id.agg(['nunique','size'])
    assert (check['nunique']==1).all() and (check['size']==5).all(), 'Require word-matched five-role examples.'
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    started=time.time()
    x=probes.gpu_array(array,fit.cache_ix.to_numpy())
    scaler=StandardScaler(copy=False)
    for start in range(0,len(x),8192):
        if hasattr(scaler,'n_samples_seen_'):scaler.n_samples_seen_=cp.asarray(scaler.n_samples_seen_)
        scaler.partial_fit(x[start:start+8192])
    scaler.transform(x,copy=False)
    xv=probes.gpu_array(array,valid.cache_ix.to_numpy());scaler.transform(xv,copy=False)
    mean=cp.asnumpy(scaler.mean_);scale=cp.asnumpy(scaler.scale_)
    ids=np.flatnonzero(fit.profile.isin(['single','long-single']).to_numpy())
    covariance=cp.zeros((DESCRIPTOR_WIDTH,8640),dtype=cp.float64)
    e_sum=cp.zeros(DESCRIPTOR_WIDTH,dtype=cp.float64);x_sum=cp.zeros(8640,dtype=cp.float64)
    token_ids=fit.token_id.to_numpy()
    for start in range(0,len(ids),4096):
        positions=ids[start:start+4096]
        features=x[cp.asarray(positions)]
        words=cp.asarray(np.asarray(descriptors[token_ids[positions]],dtype=np.float32))
        covariance+=(words.T@features).astype(cp.float64)
        e_sum+=words.sum(axis=0,dtype=cp.float64);x_sum+=features.sum(axis=0,dtype=cp.float64)
    covariance=(covariance-e_sum[:,None]*x_sum[None,:]/len(ids))/len(ids)
    _,singular,directions=cp.linalg.svd(covariance.astype(cp.float32),full_matrices=False)
    basis=cp.ascontiguousarray(directions[:rank])
    assert bool(cp.allclose(basis@basis.T,cp.eye(rank,dtype=cp.float32),atol=1e-5))
    for values in [x,xv]:
        for start in range(0,len(values),4096):
            block=values[start:start+4096]
            block-=(block@basis.T)@basis
    details=dict(case='large-mixed',kind='lexical',readout=config['kind'],rank=rank,roles=data.ROLES,
                 feature_spec=data.SPECS,fit_documents=1200,train_tokens=len(fit),
                 nuisance_fit_tokens=len(ids),word_matched_role_pairs_checked=True,
                 removed_singular_values=cp.asnumpy(singular[:rank]).tolist(),
                 descriptor_sha256=meta['descriptor_sha256'],source_sha256=data.digest(__file__),
                 consumer='Projection folded into first weight matrix. Inference needs only original hidden features.',
                 seed=data.SEED,test_used=False)
    y=probes.labels(fit);path=data.OUT/'checkpoints'/f'{key}.npz'
    if config['kind']=='linear':
        classifier=LogisticRegression(C=.005,max_iter=5000,linesearch_max_iter=100,tol=1e-4,output_type='cupy')
        classifier.fit(x,cp.asarray(y),convert_dtype=True)
        p=cp.asnumpy(classifier.predict_proba(xv))
        weights=cp.asarray(classifier.coef_);weights=weights-(weights@basis.T)@basis
        net=probes.make_net('linear',0)
        with torch.no_grad():
            net.head.weight.copy_(torch.from_numpy(cp.asnumpy(weights)))
            net.head.bias.copy_(torch.from_numpy(cp.asnumpy(classifier.intercept_).reshape(-1)))
        probes.save_model(path,net,mean,scale,details|dict(kind='linear',training_kind='lexical-nullspace',C=.005))
        details['iterations']=cp.asnumpy(cp.asarray(classifier.n_iter_)).tolist()
    else:
        torch.manual_seed(data.SEED)
        net=probes.make_net('mlp',64).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        xt=torch.from_dlpack(x);xvt=torch.from_dlpack(xv);yt=torch.as_tensor(y,device='cuda:0')
        bt=torch.from_dlpack(basis)
        generator=torch.Generator(device='cuda:0').manual_seed(data.SEED)
        best=-1;best_epoch=0;history=[]
        for epoch in range(1,46):
            net.train();order=torch.randperm(len(xt),device='cuda:0',generator=generator)
            for start in range(0,len(order),4096):
                positions=order[start:start+4096]
                loss=torch.nn.functional.nll_loss(net(xt[positions].contiguous()),yt[positions])
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            p=probes.predict_standardized(net,xvt)
            result=probes.validation_metrics(valid,p);score=result['mean_system_f1']
            history.append(dict(epoch=epoch,mean_system_f1=score,mean_accuracy=result['mean_accuracy']))
            if score>best+1e-5:
                best=score;best_epoch=epoch
                portable=copy.deepcopy(net).eval()
                with torch.no_grad():
                    weight=portable.encoder[0].weight
                    weight.sub_((weight@bt.T)@bt)
                probes.save_model(path,portable,mean,scale,details|dict(kind='mlp',hidden=64,training_kind='lexical-nullspace',best_epoch=epoch))
                np.save(data.CACHE/f'{key}-best-validation.npy',p)
                del portable
            print(json.dumps(dict(stage='lexical_epoch',candidate=key,epoch=epoch,mean_system_f1=score,best_epoch=best_epoch)),flush=True)
            if epoch>=10 and epoch-best_epoch>=8:break
        p=np.load(data.CACHE/f'{key}-best-validation.npy')
        details.update(best_epoch=best_epoch,history=history)
    # This checks raw features, scaling, folded projection, and exported weights together.
    positions=np.linspace(0,len(valid)-1,256,dtype=int)
    portable=probes.predict_raw(path,array,valid.iloc[positions].cache_ix.to_numpy())
    difference=float(np.abs(portable-p[positions]).max())
    assert difference<5e-4,difference
    result=probes.validation_metrics(valid,p)
    np.savez_compressed(data.CACHE/f'{key}-validation.npz',p=p,sample_ix=valid.sample_ix.to_numpy())
    data.atomic_json(data.OUT/'candidates'/f'{key}.json',dict(candidate=key,**details,validation=result,
        model_sha256=data.digest(path),seconds=time.time()-started,portable_probability_max_difference=difference))
    print(json.dumps(dict(stage='lexical_complete',candidate=key,validation=result)),flush=True)


def search(gpus):
    probes.setup()
    assert not (data.OUT/'selection.json').exists()
    assert (data.OUT/'lexical-token-descriptors.json').exists()
    plan=dict(configurations=CONFIGS,case='large-mixed',seed=data.SEED,
              hypothesis='Role-independent token content can obscure system directions; remove leading hidden/token-embedding cross-covariance directions.',
              fit_rule='Training single/long-single matched five-role tokens only; no validation or test examples used to fit directions.',
              inference_rule='Same hidden features; fixed projection folded into first weights. No token identity, tag or label required at inference.',
              descriptor=data.digest(data.OUT/'lexical-token-descriptors.json'),
              source_sha256=data.digest(__file__),created_unix_time=time.time(),gpus=gpus)
    if not (data.OUT/'lexical-plan.json').exists():data.atomic_json(data.OUT/'lexical-plan.json',plan)
    for start in range(0,len(CONFIGS),len(gpus)):
        workers=[]
        for job,gpu in zip(range(start,min(start+len(gpus),len(CONFIGS))),gpus):
            log=(data.OUT/'logs'/f'lexical-{job}.log').open('a')
            process=subprocess.Popen([sys.executable,'-u',__file__,'--job',str(job)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            workers.append((process,log))
        try:
            while any(p.poll() is None for p,_ in workers):
                if any(p.poll() not in [None,0] for p,_ in workers):
                    raise RuntimeError('Lexical worker failed; inspect log before resuming.')
                time.sleep(5)
        finally:
            for p,log in workers:
                if p.poll() is None:p.terminate()
                log.close()
        assert all(p.returncode==0 for p,_ in workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-embedding',action='store_true')
    parser.add_argument('--job',type=int);parser.add_argument('--gpus',default='0,1,2,3')
    args=parser.parse_args()
    if args.prepare_embedding:prepare_embedding()
    elif args.job is not None:train(args.job)
    else:search([int(x) for x in args.gpus.split(',')])
