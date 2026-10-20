"""Fit four portable attention-output readouts; select using validation only."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_data as data
import systemness_probes as probes

OUT = attention.OUT
CACHE = attention.CACHE
CONFIGS = {
    'linear-C0.0005':dict(kind='linear',hidden=0,C=.0005),
    'linear-C0.005':dict(kind='linear',hidden=0,C=.005),
    'mlp-h64':dict(kind='mlp',hidden=64),
    'mlp-h128':dict(kind='mlp',hidden=128),
}


def features():
    rows=pd.read_parquet(CACHE/'rows.parquet')
    complete=json.loads((OUT/'extraction-complete.json').read_text())
    assert complete['rows']==len(rows) and complete['rows_sha256']==data.digest(CACHE/'rows.parquet')
    bank=np.memmap(CACHE/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),8640))
    return rows,bank


def standardize():
    rows,bank=features()
    fit=rows[rows.split=='fit']
    assert fit.document_id.nunique()==1200 and set(fit.wrapper)=={'native'}
    count=0;mean=np.zeros(8640,dtype=np.float64);m2=np.zeros_like(mean)
    indices=fit.cache_ix.to_numpy()
    for start in range(0,len(indices),4096):
        block=np.asarray(bank[indices[start:start+4096]],dtype=np.float32)
        assert np.isfinite(block).all()
        n=len(block);batch_mean=block.mean(axis=0,dtype=np.float64)
        batch_m2=block.var(axis=0,dtype=np.float64)*n
        delta=batch_mean-mean;total=count+n
        m2+=batch_m2+delta*delta*(count*n/total)
        mean+=delta*(n/total);count=total
    scale=np.sqrt(m2/count);scale[scale==0]=1
    assert np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all()
    np.savez_compressed(OUT/'standardization.npz',mean=mean.astype('float32'),scale=scale.astype('float32'))
    data.atomic_json(OUT/'standardization.json',dict(fit_rows=count,fit_documents=1200,
        fit_only=True,algorithm='Float64 batch-merged mean and population variance; exported float32.',
        sha256=data.digest(OUT/'standardization.npz'),rows_sha256=data.digest(CACHE/'rows.parquet')))


def worker(key):
    import cupy as cp
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    assert cp.cuda.runtime.getDeviceCount()==torch.cuda.device_count()==1
    plan=json.loads((OUT/'search-plan.json').read_text())
    assert plan['source_sha256']==data.digest(__file__)
    assert plan['standardization_sha256']==data.digest(OUT/'standardization.npz')
    rows,bank=features();fit=rows[rows.split=='fit'];valid=rows[rows.split=='validation'].copy()
    assert not set(fit.group_id)&set(valid.group_id)
    with np.load(OUT/'standardization.npz',allow_pickle=False) as archive:
        mean=archive['mean'].copy();scale=archive['scale'].copy()
    def load(indices):
        result=cp.empty((len(indices),8640),dtype=cp.float32,order='F')
        for start in range(0,len(indices),4096):
            block=np.asarray(bank[indices[start:start+4096]],dtype=np.float32)
            standardized=(block-mean)/scale
            assert np.isfinite(standardized).all()
            result[start:start+len(block)]=cp.asarray(standardized)
        return result
    started=time.time();x=load(fit.cache_ix.to_numpy());xv=load(valid.cache_ix.to_numpy())
    y=probes.labels(fit);config=CONFIGS[key]
    model_path=OUT/'models'/f'{key}.npz';probability_path=CACHE/f'{key}-validation.npz'
    details=dict(**config,roles=data.ROLES,feature_spec=attention.SPECS,fit_documents=1200,
        train_tokens=len(fit),seed=data.SEED,source_sha256=data.digest(__file__),
        search_plan_sha256=data.digest(OUT/'search-plan.json'),
        standardization_sha256=plan['standardization_sha256'],
        inference='Pointwise 8640 attention-output features; no header labels or token IDs.')
    if config['kind']=='linear':
        from cuml.linear_model import LogisticRegression
        clf=LogisticRegression(C=config['C'],max_iter=5000,linesearch_max_iter=100,tol=1e-4,output_type='cupy')
        clf.fit(x,cp.asarray(y));p=cp.asnumpy(clf.predict_proba(xv))
        net=probes.make_net('linear',0)
        with torch.no_grad():
            net.head.weight.copy_(torch.from_numpy(cp.asnumpy(clf.coef_)))
            net.head.bias.copy_(torch.from_numpy(cp.asnumpy(clf.intercept_).reshape(-1)))
        probes.save_model(model_path,net,mean,scale,details)
        details['iterations']=cp.asnumpy(cp.asarray(clf.n_iter_)).tolist()
    else:
        xt=torch.from_dlpack(x);xvt=torch.from_dlpack(xv);yt=torch.from_numpy(y).cuda()
        torch.manual_seed(data.SEED)
        net=probes.make_net('mlp',config['hidden']).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        generator=torch.Generator(device='cuda:0').manual_seed(data.SEED)
        best=-1.;best_epoch=-1;history=[]
        for epoch in range(1,46):
            net.train();order=torch.randperm(len(xt),device='cuda',generator=generator);loss_sum=0.
            for start in range(0,len(order),4096):
                ix=order[start:start+4096]
                loss=torch.nn.functional.cross_entropy(net.raw(xt[ix].contiguous()),yt[ix])
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
                loss_sum+=float(loss.detach())*len(ix)
            p=probes.predict_standardized(net,xvt)
            result=probes.validation_metrics(valid,p);score=result['mean_system_f1']
            history.append(dict(epoch=epoch,loss=loss_sum/len(xt),mean_system_f1=score,
                mean_accuracy=result['mean_accuracy']))
            if score>best+1e-5:
                best=score;best_epoch=epoch
                probes.save_model(model_path,net,mean,scale,details|dict(best_epoch=epoch))
                np.savez_compressed(probability_path,p=p,sample_ix=valid.sample_ix.to_numpy())
            print(json.dumps(dict(stage='epoch',candidate=key,**history[-1],best_epoch=best_epoch)),flush=True)
            if epoch>=10 and epoch-best_epoch>=8:break
        with np.load(probability_path,allow_pickle=False) as archive:
            p=archive['p'].copy();assert np.array_equal(archive['sample_ix'],valid.sample_ix.to_numpy())
        details.update(best_epoch=best_epoch,epochs=len(history),history=history)
    np.savez_compressed(probability_path,p=p,sample_ix=valid.sample_ix.to_numpy())
    portable,m,s,_=probes.load_model(model_path,device='cpu')
    indices=np.linspace(0,len(valid)-1,256,dtype=int)
    raw=torch.from_numpy(np.asarray(bank[valid.iloc[indices].cache_ix.to_numpy()],dtype=np.float32))
    with torch.inference_mode():actual=portable((raw-m)/s).exp().numpy()
    difference=float(np.abs(actual-p[indices]).max());assert difference<5e-4,difference
    assert data.digest(__file__)==plan['source_sha256']
    record=dict(candidate=key,**details,validation=probes.validation_metrics(valid,p),
        seconds=time.time()-started,model_sha256=data.digest(model_path),
        portable_probability_max_difference=difference,tests_read=False,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    data.atomic_json(OUT/'candidates'/f'{key}.json',record)


def select():
    records=[json.loads((OUT/'candidates'/f'{key}.json').read_text()) for key in CONFIGS]
    previous=json.loads((data.OUT/'selection.json').read_text())
    reference=previous['validation']
    eligible=[r for r in records if r['validation']['mean_accuracy']>=reference['mean_accuracy']-.01]
    assert eligible
    best=max(eligible,key=lambda r:(r['validation']['mean_system_f1'],r['validation']['mean_accuracy']))
    v=best['validation']
    weak_gain=min(v['profiles'][p]['system_f1']-reference['profiles'][p]['system_f1'] for p in ['single','long-single'])
    other_loss=max(reference['profiles'][p]['system_f1']-v['profiles'][p]['system_f1'] for p in ['multi','natural'])
    mean_gain=v['mean_system_f1']-reference['mean_system_f1']
    promoted=mean_gain>=.01 and weak_gain>=.02 and other_loss<=.02
    frozen=dict(candidate=best['candidate'],model_sha256=best['model_sha256'],
        model_path=str(OUT/'models'/f'{best["candidate"]}.npz'),system_threshold=v['system_threshold'],
        validation=v,previous_validation=reference,promoted=promoted,
        mean_f1_gain=mean_gain,minimum_single_long_f1_gain=weak_gain,maximum_multi_natural_f1_loss=other_loss,
        test_used_for_selection=False,previous_test_results_already_reported=True,
        fresh_confirmation_required=True,source_sha256=data.digest(__file__),
        search_plan_sha256=data.digest(OUT/'search-plan.json'),selected_unix_time=time.time(),
        candidates={r['candidate']:data.digest(OUT/'candidates'/f'{r["candidate"]}.json') for r in records})
    table=[]
    for r in records:
        for profile,m in r['validation']['profiles'].items():
            table.append(dict(candidate=r['candidate'],profile=profile,
                **{k:v for k,v in m.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(table).to_csv(OUT/'validation-metrics.csv',index=False)
    data.atomic_json(OUT/'selection.json',frozen)
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',candidate=best['candidate'],promoted=promoted))
    print(json.dumps(frozen),flush=True)


def main():
    assert (OUT/'extraction-complete.json').exists()
    assert not (OUT/'selection.json').exists(),'Readout search is already frozen.'
    for folder in ['models','candidates','logs']:(OUT/folder).mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('/logs/\n')
    extraction=json.loads((OUT/'extraction-complete.json').read_text())
    assert extraction['sha256']==data.digest(CACHE/'features.f16')
    if not (OUT/'standardization.json').exists():standardize()
    plan=dict(configurations=CONFIGS,fit_documents=1200,tests_read=False,
        source_sha256=data.digest(__file__),extraction_sha256=data.digest(OUT/'extraction-complete.json'),
        standardization_sha256=data.digest(OUT/'standardization.npz'),
        selection='Highest mean validation profile system F1; mean five-role accuracy >= previous selection minus 1 pp.',
        threshold='Unchanged original global validation threshold, each profile FPR <=5%.',
        neural=dict(max_epochs=45,min_epochs=10,patience=8,batch_size=4096,learning_rate=.001,
                    weight_decay=.001,dropout=.1,gradient_clip=5.,seed=data.SEED),
        promotion=dict(minimum_mean_f1_gain=.01,minimum_each_single_long_f1_gain=.02,
                       maximum_each_multi_natural_f1_loss=.02),
        fresh_confirmation='At least 400 new source documents, excluded by body/full-body/URL hashes.',
        previous_benchmark_already_reported=True)
    if (OUT/'search-plan.json').exists():assert json.loads((OUT/'search-plan.json').read_text())==plan
    else:data.atomic_json(OUT/'search-plan.json',plan)
    workers=[]
    for gpu,key in enumerate(CONFIGS):
        if (OUT/'candidates'/f'{key}.json').exists():continue
        log=(OUT/'logs'/f'{key}.log').open('a')
        proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',key],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        workers.append((proc,log))
    data.atomic_json(OUT/'status.json',dict(stage='training',workers=[p.pid for p,_ in workers]))
    try:
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in [None,0] for p,_ in workers):raise RuntimeError('Attention readout worker failed; inspect logs.')
            time.sleep(5)
    finally:
        for p,log in workers:
            if p.poll() is None:p.terminate()
            log.close()
    codes=[p.wait() for p,_ in workers];assert all(c==0 for c in codes),codes
    select()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=CONFIGS)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:main()
