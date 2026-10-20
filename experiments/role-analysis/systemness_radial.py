"""Pointwise readouts that separate residual magnitude from direction.

No additional language-model forward pass, token identity, history, or tags are
used. All normalization statistics are fitted on the original fit partition.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import systemness_data as data
import systemness_probes as probes

CONFIGS=[dict(mode='normalized',hidden=0),dict(mode='combined',hidden=0),
         dict(mode='combined',hidden=64),dict(mode='combined',hidden=128)]


def radial_features(x):
    import torch
    blocks=x.reshape(-1,3,2880)
    radius=torch.sqrt(blocks.square().mean(dim=2)+1e-12)
    normalized=(blocks/radius[:,:,None]).reshape(-1,8640)
    return normalized,radius.log()


def make_net(mode,hidden):
    import torch
    from torch import nn
    class RadialProbe(nn.Module):
        def __init__(self):
            super().__init__()
            for prefix,width in [('raw',8640),('norm',8640),('radius',3)]:
                self.register_buffer(prefix+'_mean',torch.zeros(width))
                self.register_buffer(prefix+'_scale',torch.ones(width))
            width=(8640 if mode=='normalized' else 17280)+3
            self.encoder=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(.1)) if hidden else nn.Identity()
            self.head=nn.Linear(hidden or width,5)

        def forward(self,x):
            normalized,radius=radial_features(x)
            values=[(normalized-self.norm_mean)/self.norm_scale,(radius-self.radius_mean)/self.radius_scale]
            if mode=='combined':values.insert(0,(x-self.raw_mean)/self.raw_scale)
            features=torch.cat(values,dim=1)
            return torch.log_softmax(self.head(self.encoder(features)),dim=1)
    return RadialProbe()


def train(job):
    import cupy as cp
    import torch
    from cuml.preprocessing import StandardScaler
    probes.setup()
    assert not (data.OUT/'selection.json').exists()
    config=CONFIGS[job]
    key=f'large-mixed-radial-{config["mode"]}-h{config["hidden"]}'
    if (data.OUT/'candidates'/f'{key}.json').exists():return
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(data.SEED)
    rows,array=probes.development()
    fit=rows[rows.split=='fit'];valid=rows[rows.split=='validation']
    x=probes.gpu_array(array,fit.cache_ix.to_numpy());xv=probes.gpu_array(array,valid.cache_ix.to_numpy())
    xt=torch.from_dlpack(x);xvt=torch.from_dlpack(xv)
    stats={name:StandardScaler(copy=False) for name in ['raw','norm','radius']}
    started=time.time()
    with torch.inference_mode():
        for start in range(0,len(xt),4096):
            block=xt[start:start+4096].contiguous()
            normalized,radius=radial_features(block)
            for name,values in [('raw',block),('norm',normalized),('radius',radius)]:
                scaler=stats[name]
                if hasattr(scaler,'n_samples_seen_'):scaler.n_samples_seen_=cp.asarray(scaler.n_samples_seen_)
                scaler.partial_fit(cp.from_dlpack(values.contiguous()))
    net=make_net(**config).cuda()
    with torch.no_grad():
        for name,scaler in stats.items():
            assert int(scaler.n_samples_seen_)==len(fit)
            getattr(net,name+'_mean').copy_(torch.from_numpy(cp.asnumpy(scaler.mean_)).cuda())
            getattr(net,name+'_scale').copy_(torch.from_numpy(cp.asnumpy(scaler.scale_)).cuda().clamp_min(1e-6))
    yt=torch.as_tensor(probes.labels(fit),device='cuda:0',dtype=torch.long)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    generator=torch.Generator(device='cuda:0').manual_seed(data.SEED)
    metadata=dict(kind='radial',case='large-mixed',**config,roles=data.ROLES,feature_spec=data.SPECS,
                  fit_documents=1200,train_tokens=len(fit),seed=data.SEED,
                  source_sha256=data.digest(__file__),normalization_fit_partition='fit only',
                  input='Original 8640 raw hidden features; per-layer RMS direction and log magnitude computed within the pointwise network.')
    path=data.OUT/'checkpoints'/f'{key}.npz'
    history=[];best=-1;best_epoch=0
    for epoch in range(1,46):
        net.train();order=torch.randperm(len(xt),device='cuda:0',generator=generator)
        total=0.
        for start in range(0,len(order),4096):
            indices=order[start:start+4096]
            loss=torch.nn.functional.nll_loss(net(xt[indices].contiguous()),yt[indices])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(indices)
        p=probes.predict_standardized(net,xvt)
        validation=probes.validation_metrics(valid,p);score=validation['mean_system_f1']
        history.append(dict(epoch=epoch,loss=total/len(xt),mean_system_f1=score,mean_accuracy=validation['mean_accuracy']))
        if score>best+1e-5:
            best=score;best_epoch=epoch
            probes.save_model(path,net,np.zeros(8640),np.ones(8640),metadata|dict(best_epoch=epoch))
            np.save(data.CACHE/f'{key}-best-validation.npy',p)
        print(json.dumps(dict(stage='radial_epoch',candidate=key,**history[-1],best_epoch=best_epoch)),flush=True)
        if epoch>=10 and epoch-best_epoch>=8:break
    p=np.load(data.CACHE/f'{key}-best-validation.npy')
    sample=np.linspace(0,len(valid)-1,256,dtype=int)
    actual=probes.predict_raw(path,array,valid.iloc[sample].cache_ix.to_numpy())
    difference=float(np.abs(actual-p[sample]).max());assert difference<5e-4,difference
    result=probes.validation_metrics(valid,p)
    np.savez_compressed(data.CACHE/f'{key}-validation.npz',p=p,sample_ix=valid.sample_ix.to_numpy())
    data.atomic_json(data.OUT/'candidates'/f'{key}.json',dict(candidate=key,**metadata,validation=result,
        best_epoch=best_epoch,history=history,seconds=time.time()-started,model_sha256=data.digest(path),
        portable_probability_max_difference=difference,test_used=False))
    print(json.dumps(dict(stage='radial_complete',candidate=key,validation=result)),flush=True)


def search():
    probes.setup()
    assert not (data.OUT/'selection.json').exists()
    plan=dict(configurations=CONFIGS,case='large-mixed',seed=data.SEED,
        hypothesis='Explicit residual direction and magnitude can complement raw features without changing pointwise input information.',
        normalization='Per-block RMS in FP32; raw/directional/log-radius scalers fitted on fit rows only.',
        selection='Existing four-profile validation system F1 and FPR rule; no test data read.',
        max_epochs=45,min_epochs=10,patience=8,learning_rate=.001,weight_decay=.001,
        source_sha256=data.digest(__file__),created_unix_time=time.time())
    if not (data.OUT/'radial-plan.json').exists():data.atomic_json(data.OUT/'radial-plan.json',plan)
    workers=[]
    for job in range(4):
        log=(data.OUT/'logs'/f'radial-{job}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',__file__,'--job',str(job)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job)),stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log))
    try:
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in [None,0] for p,_ in workers):raise RuntimeError('Radial worker failed; inspect logs.')
            time.sleep(5)
    finally:
        for p,log in workers:
            if p.poll() is None:p.terminate()
            log.close()
    assert all(p.returncode==0 for p,_ in workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=int)
    args=parser.parse_args()
    if args.job is None:search()
    else:train(args.job)
