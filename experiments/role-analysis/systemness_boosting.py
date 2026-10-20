"""Second validation-only search: nonlinear trees on the large mixed corpus.

The first 36 candidates identify single/long-message systemness as the remaining
weakness. Four GPU jobs compare five-class and dedicated binary-system trees.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

import systemness_data as data
import systemness_probes as probes

CONFIGS=[dict(mode=mode,depth=depth) for mode in ['multiclass','binary'] for depth in [6,10]]


def tree_net(archive,meta,device):
    import torch
    import xgboost as xgb
    booster=xgb.Booster()
    booster.load_model(bytearray(archive['booster'].tobytes()))
    booster.set_param(dict(device='cpu' if str(device)=='cpu' else 'cuda:0'))
    class TreeProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.booster=booster
            if meta['mode']=='binary':
                self.register_buffer('conditional_weight',torch.from_numpy(archive['conditional_weight'].copy()))
                self.register_buffer('conditional_bias',torch.from_numpy(archive['conditional_bias'].copy()))

        def forward(self,x):
            if x.device.type=='cuda':
                import cupy as cp
                values=self.booster.inplace_predict(cp.from_dlpack(x.contiguous()))
                p=torch.from_dlpack(values)
            else:
                p=torch.from_numpy(self.booster.inplace_predict(x.detach().contiguous().numpy()))
            if meta['mode']=='binary':
                system=p.reshape(-1,1)
                conditional=torch.softmax(x@self.conditional_weight.T+self.conditional_bias,dim=1)
                p=torch.cat([system,(1-system)*conditional],dim=1)
            return p.clamp_min(1e-15).log()
    return TreeProbe()


def train(job):
    import cupy as cp
    import torch
    import xgboost as xgb
    probes.setup()
    config=CONFIGS[job]
    case='large-mixed'
    key=f'{case}-xgb-{config["mode"]}-d{config["depth"]}'
    if (data.OUT/'candidates'/f'{key}.json').exists():
        print(key,'already complete.',flush=True)
        return
    assert not (data.OUT/'selection.json').exists(), 'Do not tune after unblinding tests.'
    rows,array=probes.development()
    fit=rows[rows.split=='fit'];validation=rows[rows.split=='validation']
    x=probes.gpu_array(array,fit.cache_ix.to_numpy())
    # Trees need no feature scaling. Keep raw FP16->FP32 values exact so tiny
    # scaler rounding differences cannot move a consumer across a tree split.
    mean=np.zeros(8640,dtype='float32');scale=np.ones(8640,dtype='float32')
    y=probes.labels(fit);yv=probes.labels(validation)
    binary=config['mode']=='binary'
    yt=cp.asarray((y==0).astype('float32') if binary else y)
    yvt=cp.asarray((yv==0).astype('float32') if binary else yv)
    started=time.time()
    dtrain=xgb.QuantileDMatrix(x,label=yt,max_bin=64)
    # Sketching raw training and validation arrays concurrently exceeds the
    # memory available beside the host's existing GPU allocations. Quantile
    # matrices retain compressed data; release the raw fit array first.
    del x
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
    xv=probes.gpu_array(array,validation.cache_ix.to_numpy())
    dvalid=xgb.QuantileDMatrix(xv,label=yvt,max_bin=64,ref=dtrain)
    params=dict(objective='binary:logistic' if binary else 'multi:softprob',
                eval_metric='logloss' if binary else 'mlogloss',device='cuda:0',tree_method='hist',
                max_depth=config['depth'],eta=.06,max_bin=64,subsample=.8,colsample_bytree=.5,
                reg_lambda=10.,min_child_weight=20,max_cached_hist_node=32,seed=data.SEED,nthread=4)
    if not binary:params['num_class']=5
    history={}
    booster=xgb.train(params,dtrain,num_boost_round=400,evals=[(dvalid,'validation')],
                      early_stopping_rounds=30,evals_result=history,verbose_eval=50)
    best_iteration=int(booster.best_iteration)
    booster=booster[:best_iteration+1]
    arrays={}
    if binary:
        base=data.OUT/'checkpoints/large-mixed-linear-C0.005-w1.npz'
        with np.load(base,allow_pickle=False) as reference:
            weights=reference['weight::head.weight'][1:].copy()
            bias=reference['weight::head.bias'][1:].copy()
            # Express the old four-way conditional readout in this scaler's units.
            adjusted=weights*(scale/reference['scale'])
            bias=bias+weights@((mean-reference['mean'])/reference['scale'])
        arrays.update(conditional_weight=adjusted.astype('float32'),conditional_bias=bias.astype('float32'))
    meta=dict(kind='xgboost',mode=config['mode'],depth=config['depth'],case=case,seed=data.SEED,
              params=params,best_iteration=best_iteration,xgboost=xgb.__version__,roles=data.ROLES,
              feature_spec=data.SPECS,fit_documents=1200,train_tokens=len(fit),source_sha256=data.digest(__file__),
              preprocessing='identity on raw FP16-cached features promoted exactly to FP32',
              binary_fusion='P(system) times four-way conditional softmax from fixed linear readout' if binary else None)
    path=data.OUT/'checkpoints'/f'{key}.npz'
    np.savez_compressed(path,mean=mean,scale=scale,metadata=np.array(json.dumps(meta)),
                        booster=np.frombuffer(booster.save_raw(raw_format='ubj'),dtype=np.uint8),**arrays)
    # Serialize/reload before scoring, so these are consumer probabilities.
    net,_,_,_=probes.load_model(path)
    p=probes.predict_standardized(net,torch.from_dlpack(xv))
    validation_result=probes.validation_metrics(validation,p)
    record=dict(candidate=key,case=case,kind='xgboost',mode=config['mode'],depth=config['depth'],
                train_tokens=len(fit),fit_documents=1200,validation=validation_result,
                seconds=time.time()-started,model_sha256=data.digest(path),best_iteration=best_iteration,
                history=history,source_sha256=data.digest(__file__),test_used=False)
    np.savez_compressed(data.CACHE/f'{key}-validation.npz',p=p,sample_ix=validation.sample_ix.to_numpy())
    data.atomic_json(data.OUT/'candidates'/f'{key}.json',record)
    print(json.dumps(dict(stage='boosting_complete',candidate=key,mean_system_f1=validation_result['mean_system_f1'],
             profiles={name:{k:v[k] for k in ['accuracy','system_precision','system_recall','system_f1']} for name,v in validation_result['profiles'].items()})),flush=True)
    del dtrain,dvalid,xv,net,booster
    gc.collect();cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache()


def search():
    import xgboost as xgb
    probes.setup()
    assert not (data.OUT/'selection.json').exists()
    records=[json.loads(p.read_text()) for p in (data.OUT/'candidates').glob('*.json')]
    assert len(records)>=36
    plan=dict(configurations=CONFIGS,case='large-mixed',seed=data.SEED,
              max_rounds=400,early_stopping_rounds=30,selection='Existing four-profile validation F1/FPR rule; no tests read.',
              reason='Initial 36 candidates reach high multi/natural system F1 but remain weaker on single/long controls.',
              initial_candidates={r['candidate']:r['validation']['mean_system_f1'] for r in records if r['kind']!='xgboost'},
              xgboost=xgb.__version__,build=xgb.build_info(),source_sha256=data.digest(__file__),
              consumer_sha256=data.digest(data.HERE/'systemness_probes.py'),created_unix_time=time.time())
    if not (data.OUT/'boosting-plan.json').exists():data.atomic_json(data.OUT/'boosting-plan.json',plan)
    data.atomic_json(data.OUT/'boosting-runtime.json',dict(source_sha256=data.digest(__file__),
        memory_strategy='Construct quantized fit data, release raw fit array, then construct validation data. Same rows, features and parameters.',
        created_unix_time=time.time()))
    workers=[]
    for gpu in range(4):
        log=(data.OUT/'logs'/f'boosting-{gpu}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',__file__,'--job',str(gpu)],
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log))
    try:
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in [None,0] for p,_ in workers):
                raise RuntimeError('Boosting worker failed; inspect the corresponding log before resuming.')
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
