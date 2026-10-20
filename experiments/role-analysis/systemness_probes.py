"""Compare data size, conversation context, and readout families for systemness.

Use four independent GPU workers after systemness_extract --part development.
Only validation is read during this search; tests require a frozen selection.
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

import systemness_data as data

CASES = ['small-single', 'large-single', 'small-mixed', 'large-mixed']
VALIDATION_PROFILES = ['single','multi','natural','long-single']


def configurations():
    return ([dict(kind='linear',C=c,system_weight=1) for c in [.0005,.005,.05]]+
            [dict(kind='linear',C=.005,system_weight=2)]+
            [dict(kind='pca',C=.005,components=n,system_weight=1) for n in [512,1024]]+
            [dict(kind='mlp',hidden=h,system_weight=1) for h in [64,128]]+
            [dict(kind='hierarchical',hidden=128,system_weight=1)])


def setup():
    for name in ['candidates','checkpoints','logs','models']:
        (data.OUT/name).mkdir(exist_ok=True)
    (data.OUT/'.gitignore').write_text('/cache/\n/logs/\n/checkpoints/\n')


def safe_div(a,b):
    return float(a/b) if b else None


def metrics(p,y,threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,atol=2e-5)
    pred=p.argmax(axis=1)
    matrix=np.bincount(y*5+pred,minlength=25).reshape(5,5)
    diag=matrix.diagonal()
    f1=np.divide(2*diag,matrix.sum(axis=0)+matrix.sum(axis=1),out=np.zeros(5),where=(matrix.sum(axis=0)+matrix.sum(axis=1))>0)
    positive=y==0
    detected=p[:,0]>=threshold
    tp=int(np.sum(positive&detected)); fn=int(np.sum(positive&~detected))
    fp=int(np.sum(~positive&detected)); tn=int(np.sum(~positive&~detected))
    recall=safe_div(tp,tp+fn); specificity=safe_div(tn,tn+fp)
    return dict(tokens=len(y),accuracy=float((pred==y).mean()),macro_f1=float(f1[matrix.sum(axis=1)>0].mean()),
                nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),
                recall={role:safe_div(int(diag[i]),int(matrix[i].sum())) for i,role in enumerate(data.ROLES)},
                confusion=matrix.tolist(),system_threshold=float(threshold),
                system_accuracy=float((positive==detected).mean()),
                system_balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                system_precision=safe_div(tp,tp+fp),system_recall=recall,
                system_f1=safe_div(2*tp,2*tp+fp+fn),system_false_positive_rate=safe_div(fp,fp+tn),
                system_auroc=float(roc_auc_score(positive,p[:,0])) if positive.any() and (~positive).any() else None,
                system_average_precision=float(average_precision_score(positive,p[:,0])) if positive.any() else None,
                system_detection_by_role={role:float(detected[y==i].mean()) if np.any(y==i) else None for i,role in enumerate(data.ROLES)},
                system_confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn))


def labels(rows):
    return rows.role.map({r:i for i,r in enumerate(data.ROLES)}).fillna(-1).to_numpy(dtype=np.int64)


def calibration(rows,p):
    """One common threshold, selected using only the three native validation sets."""
    y=labels(rows)
    assert (y>=0).all() and set(rows.split)=={'validation'} and set(rows.wrapper)=={'native'}
    assert set(rows.profile)==set(VALIDATION_PROFILES)
    thresholds=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(p[:,0],np.linspace(0,1,301)),1.000001])
    scores=[]
    fprs=[]
    for profile in VALIDATION_PROFILES:
        mask=rows.profile.to_numpy()==profile
        truth=y[mask]==0
        prob=p[mask,0]
        positives=np.sort(prob[truth]); negatives=np.sort(prob[~truth])
        tp=len(positives)-np.searchsorted(positives,thresholds,side='left')
        fp=len(negatives)-np.searchsorted(negatives,thresholds,side='left')
        fn=len(positives)-tp
        scores.append(2*tp/np.maximum(2*tp+fp+fn,1))
        fprs.append(fp/max(len(negatives),1))
    means=np.mean(scores,axis=0)
    feasible=np.max(fprs,axis=0)<=.05
    means[~feasible]=-1
    ix=int(np.argmax(means))
    return float(thresholds[ix]),float(means[ix])


def validation_metrics(rows,p):
    threshold,score=calibration(rows,p)
    y=labels(rows)
    result={}
    for profile in VALIDATION_PROFILES:
        mask=rows.profile.to_numpy()==profile
        result[profile]=metrics(p[mask],y[mask],threshold)
    return dict(system_threshold=threshold,mean_system_f1=score,
                mean_accuracy=float(np.mean([x['accuracy'] for x in result.values()])),profiles=result)


def make_net(kind,hidden):
    import torch
    from torch import nn
    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(8640,hidden),nn.GELU(),nn.Dropout(.1)) if hidden else nn.Identity()
            self.head=nn.Linear(hidden or 8640,5)

        def raw(self,x):
            return self.head(self.encoder(x))

        def forward(self,x):
            raw=self.raw(x)
            if kind=='hierarchical':
                return torch.cat([torch.nn.functional.logsigmoid(raw[:,:1]),
                    torch.nn.functional.logsigmoid(-raw[:,:1])+torch.log_softmax(raw[:,1:],dim=1)],dim=1)
            return torch.log_softmax(raw,dim=1)
    return Probe()


def save_model(path,net,mean,scale,metadata):
    arrays={f'weight::{name}':value.detach().cpu().numpy() for name,value in net.state_dict().items()}
    np.savez_compressed(path,mean=np.asarray(mean,dtype=np.float32),scale=np.asarray(scale,dtype=np.float32),
                        metadata=np.array(json.dumps(metadata)),**arrays)


def load_model(path,device='cuda:0'):
    import torch
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        if meta['kind']=='xgboost':
            from systemness_boosting import tree_net
            net=tree_net(archive,meta,device)
        else:
            if meta['kind']=='radial':
                from systemness_radial import make_net as radial_net
                net=radial_net(meta['mode'],meta['hidden'])
            else:
                net=make_net(meta['kind'],meta.get('hidden',0))
            net.load_state_dict({key[8:]:torch.from_numpy(archive[key].copy()) for key in archive.files if key.startswith('weight::')})
        mean=torch.from_numpy(archive['mean'].copy()).to(device)
        scale=torch.from_numpy(archive['scale'].copy()).to(device)
    return net.eval().to(device),mean,scale,meta


def predict_standardized(net,x,batch_size=4096):
    import torch
    chunks=[]
    net.eval()
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            chunks.append(net(x[start:start+batch_size].contiguous()).exp().cpu().numpy())
    return np.concatenate(chunks)


def predict_raw(path,array,indices,batch_size=4096):
    import torch
    net,mean,scale,meta=load_model(path)
    probabilities=[]
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            x=torch.from_numpy(np.asarray(array[indices[start:start+batch_size]],dtype=np.float32)).cuda()
            probabilities.append(net((x-mean)/scale).exp().cpu().numpy())
    del net,mean,scale
    torch.cuda.empty_cache()
    return np.concatenate(probabilities)


def export_baseline():
    import cupy as cp
    import pickle
    import torch
    path=data.OUT/'models/previous-selected.npz'
    if path.exists():
        return
    source=data.runtime.OUT/'feature-search/models/final-document-sucat-mid-16+20+24-0.005.pkl'
    with source.open('rb') as f:
        artifact=pickle.load(f)
    pipeline=artifact['probe']
    scaler=pipeline.named_steps['scaler']; clf=pipeline.named_steps['clf']
    net=make_net('linear',0)
    with torch.no_grad():
        net.head.weight.copy_(torch.from_numpy(cp.asnumpy(clf.coef_)))
        net.head.bias.copy_(torch.from_numpy(cp.asnumpy(clf.intercept_).reshape(-1)))
    save_model(path,net,cp.asnumpy(scaler.mean_),cp.asnumpy(scaler.scale_),
               dict(kind='linear',hidden=0,source=str(source),source_sha256=data.digest(source),
                    roles=data.ROLES,feature_spec=data.SPECS,original_weights_unchanged=True))
    # Check the exported portable readout against the original consumer.
    rng=np.random.RandomState(data.SEED)
    raw=rng.normal(size=(128,8640)).astype(np.float32)
    expected=cp.asnumpy(pipeline.predict_proba(cp.asarray(raw)))
    actual=predict_raw(path,raw,np.arange(len(raw)))
    difference=float(np.max(np.abs(expected-actual)))
    assert difference<2e-4,difference
    data.atomic_json(data.OUT/'baseline-export-validation.json',dict(max_probability_difference=difference,passed=True))


def development():
    frames=[]; arrays=[]
    for part in ['development','augmentation']:
        meta=json.loads((data.OUT/f'extraction-{part}.json').read_text())
        folder=data.FEATURE_CACHE/part
        rows=pd.read_parquet(folder/'rows.parquet')
        assert len(rows)==meta['rows'] and data.digest(folder/'rows.parquet')==meta['rows_sha256']
        frames.append(rows)
        arrays.append(np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),8640)))
    rows=pd.concat(frames,ignore_index=True)
    rows['cache_ix']=np.arange(len(rows))
    class Bank:
        def __init__(self,arrays):
            self.arrays=arrays
            self.bounds=np.r_[0,np.cumsum([len(a) for a in arrays])]
        def __getitem__(self,indices):
            result=np.empty((len(indices),8640),dtype=np.float16)
            for i,array in enumerate(self.arrays):
                mask=(indices>=self.bounds[i])&(indices<self.bounds[i+1])
                result[mask]=array[indices[mask]-self.bounds[i]]
            return result
    return rows,Bank(arrays)


def baseline_validation():
    setup(); export_baseline()
    if (data.OUT/'baseline-validation.json').exists():
        return
    rows,array=development()
    valid=rows[rows.split=='validation'].copy()
    p=predict_raw(data.OUT/'models/previous-selected.npz',array,valid.cache_ix.to_numpy())
    data.atomic_json(data.OUT/'baseline-validation.json',validation_metrics(valid,p))
    np.savez_compressed(data.CACHE/'baseline-validation-probabilities.npz',p=p,sample_ix=valid.sample_ix.to_numpy())
    print(json.dumps(dict(stage='baseline_validation',**json.loads((data.OUT/'baseline-validation.json').read_text()))),flush=True)


def gpu_array(array,indices):
    import cupy as cp
    x=cp.empty((len(indices),8640),dtype=cp.float32,order='F')
    for start in range(0,len(indices),8192):
        block=np.asarray(array[indices[start:start+8192]],dtype=np.float32)
        x[start:start+len(block)]=cp.asarray(block)
    assert bool(cp.isfinite(x).all())
    return x


def train_case(case):
    import cupy as cp
    import torch
    from cuml.preprocessing import StandardScaler
    from cuml.linear_model import LogisticRegression
    setup()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    rows,array=development()
    split=json.loads((data.OUT/'splits.json').read_text())
    train=rows[rows.split=='fit']
    if case.startswith('small'):
        train=train[train.document_id.isin(split['small_fit'])]
    if case.endswith('single'):
        train=train[train.profile.isin(['single','long-single'])]
    valid=rows[rows.split=='validation'].copy()
    x=gpu_array(array,train.cache_ix.to_numpy())
    scaler=StandardScaler(copy=False)
    for start in range(0,len(x),8192):
        if hasattr(scaler,'n_samples_seen_'):
            scaler.n_samples_seen_=cp.asarray(scaler.n_samples_seen_)
        scaler.partial_fit(x[start:start+8192])
    scaler.transform(x,copy=False)
    xv=gpu_array(array,valid.cache_ix.to_numpy())
    scaler.transform(xv,copy=False)
    mean=cp.asnumpy(scaler.mean_); scale=cp.asnumpy(scaler.scale_)
    y=labels(train); yv=labels(valid)
    assert (y>=0).all() and (yv>=0).all()
    xt=torch.from_dlpack(x); xvt=torch.from_dlpack(xv)
    yt=torch.as_tensor(y,device='cuda:0',dtype=torch.long)
    for config in configurations():
        suffix=(f'C{config["C"]:g}-w{config["system_weight"]}' if config['kind'] in ['linear','pca'] else f'h{config["hidden"]}')
        if config['kind']=='pca':suffix=f'k{config["components"]}-'+suffix
        key=f'{case}-{config["kind"]}-'+suffix
        record_path=data.OUT/'candidates'/f'{key}.json'
        if record_path.exists():
            continue
        started=time.time()
        model_path=data.OUT/'checkpoints'/f'{key}.npz'
        details=dict(case=case,**config,roles=data.ROLES,feature_spec=data.SPECS,
                     fit_documents=int(train.document_id.nunique()),train_tokens=len(train),
                     train_profiles=sorted(train.profile.unique().tolist()),seed=data.SEED,
                     runner_sha256=data.digest(__file__),selection_uses_test=False)
        if config['kind'] in ['linear','pca']:
            fit_x,validation_x=x,xv
            if config['kind']=='pca':
                from cuml.decomposition import PCA
                ids=np.random.RandomState(data.SEED).choice(len(x),min(len(x),16384),replace=False)
                pca=PCA(n_components=config['components'],svd_solver='full',whiten=False,copy=True,output_type='cupy')
                pca.fit(x[cp.asarray(ids)])
                def project(values):
                    result=cp.empty((len(values),config['components']),dtype=cp.float32,order='F')
                    for start in range(0,len(values),4096):
                        result[start:start+4096]=pca.transform(values[start:start+4096])
                    return result
                fit_x=project(x);validation_x=project(xv)
                details.update(pca_fit_rows=len(ids),pca_sample_rule='numpy RandomState(seed).choice without replacement',
                               pca_explained_variance_ratio=cp.asnumpy(pca.explained_variance_ratio_).tolist(),
                               pca_export='Projection folded into equivalent 8640-dimensional linear weights.')
            clf=LogisticRegression(C=config['C'],class_weight={i:(config['system_weight'] if i==0 else 1) for i in range(5)},
                                    max_iter=5000,linesearch_max_iter=100,tol=1e-4,output_type='cupy')
            clf.fit(fit_x,cp.asarray(y),convert_dtype=True)
            p=cp.asnumpy(clf.predict_proba(validation_x))
            weights=cp.asarray(clf.coef_);bias=cp.asarray(clf.intercept_).reshape(-1)
            if config['kind']=='pca':
                weights=weights@cp.asarray(pca.components_)
                bias=bias-weights@cp.asarray(pca.mean_)
            net=make_net('linear',0)
            with torch.no_grad():
                net.head.weight.copy_(torch.from_numpy(cp.asnumpy(weights)))
                net.head.bias.copy_(torch.from_numpy(cp.asnumpy(bias)))
            details['iterations']=cp.asnumpy(cp.asarray(clf.n_iter_)).tolist()
            details['actual_classifier_dtype']=str(clf.dtype)
            save_model(model_path,net,mean,scale,details|dict(kind='linear',training_kind=config['kind']))
            exported=predict_standardized(net.cuda(),xvt[:128])
            details['portable_probability_max_difference']=float(np.max(np.abs(exported-p[:128])))
            assert details['portable_probability_max_difference']<2e-4
            del clf,net,fit_x,validation_x,weights,bias
            if config['kind']=='pca':del pca
        else:
            torch.manual_seed(data.SEED)
            net=make_net(config['kind'],config['hidden']).cuda()
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
            generator=torch.Generator(device='cuda:0').manual_seed(data.SEED)
            best=-1; best_epoch=-1; history=[]
            for epoch in range(1,46):
                net.train()
                order=torch.randperm(len(xt),device='cuda:0',generator=generator)
                loss_sum=0.
                for start in range(0,len(order),4096):
                    ids=order[start:start+4096]
                    features=xt[ids].contiguous(); truth=yt[ids]
                    raw=net.raw(features)
                    if config['kind']=='hierarchical':
                        binary=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,0],(truth==0).float(),
                                     pos_weight=torch.tensor(4.,device='cuda:0'))
                        nonsystem=truth!=0
                        conditional=torch.nn.functional.cross_entropy(raw[nonsystem,1:],truth[nonsystem]-1)
                        loss=binary+conditional
                    else:
                        loss=torch.nn.functional.cross_entropy(raw,truth)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                    optimizer.step()
                    loss_sum+=float(loss.detach())*len(ids)
                p=predict_standardized(net,xvt)
                result=validation_metrics(valid,p)
                score=result['mean_system_f1']
                history.append(dict(epoch=epoch,loss=loss_sum/len(xt),mean_system_f1=score,mean_accuracy=result['mean_accuracy']))
                if score>best+1e-5:
                    best=score; best_epoch=epoch
                    save_model(model_path,net,mean,scale,details|dict(best_epoch=epoch))
                    np.save(data.CACHE/f'{key}-best-validation.npy',p)
                print(json.dumps(dict(stage='neural_epoch',candidate=key,**history[-1],best_epoch=best_epoch)),flush=True)
                if epoch>=10 and epoch-best_epoch>=8:
                    break
            p=np.load(data.CACHE/f'{key}-best-validation.npy')
            details.update(best_epoch=best_epoch,epochs=len(history),history=history)
            del net,optimizer
        result=validation_metrics(valid,p)
        record=dict(candidate=key,**details,seconds=time.time()-started,validation=result,
                    model_sha256=data.digest(model_path))
        np.savez_compressed(data.CACHE/f'{key}-validation.npz',p=p,sample_ix=valid.sample_ix.to_numpy())
        data.atomic_json(record_path,record)
        print(json.dumps(dict(stage='candidate_complete',candidate=key,mean_system_f1=result['mean_system_f1'],
                              profiles={name:{k:v[k] for k in ['accuracy','system_precision','system_recall','system_f1']} for name,v in result['profiles'].items()})),flush=True)
        gc.collect(); cp.get_default_memory_pool().free_all_blocks(); torch.cuda.empty_cache()
    data.atomic_json(data.OUT/f'case-{case}-complete.json',dict(case=case,candidates=len(configurations()),train_tokens=len(train)))


def search():
    setup()
    assert (data.OUT/'extraction-development.json').exists()
    assert (data.OUT/'extraction-augmentation.json').exists()
    search_plan=dict(cases=CASES,configurations=configurations(),candidates=len(CASES)*len(configurations()),
                validation_profiles=VALIDATION_PROFILES,threshold='One global threshold; highest mean profile system F1 with each profile FPR<=5%.',
                selection_guardrail='Mean five-role validation accuracy >= previous selected probe minus 1 percentage point.',
                neural=dict(max_epochs=45,min_epochs=10,patience=8,early_stopping='mean validation system F1',
                            batch_size=4096,learning_rate=.001,weight_decay=.001,dropout=.1,gradient_clip=5.,dtype='float32'),
                test_not_read=True,created_unix_time=time.time(),
                sources={name:data.digest(data.HERE/name) for name in ['systemness_data.py','systemness_challenges.py',
                         'systemness_extract.py','systemness_probes.py','systemness_evaluate.py']},
                development_extractions={part:data.digest(data.OUT/f'extraction-{part}.json') for part in ['development','augmentation']})
    if not (data.OUT/'search-plan.json').exists():
        data.atomic_json(data.OUT/'search-plan.json',search_plan)
    baseline_validation()
    workers=[]
    for gpu,case in enumerate(CASES):
        log=(data.OUT/'logs'/f'{case}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',__file__,'--stage','worker','--case',case],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log,case))
    try:
        while any(p.poll() is None for p,_,_ in workers):
            failed=[(case,p.returncode) for p,_,case in workers if p.poll() not in [None,0]]
            if failed:
                raise RuntimeError(f'Workers failed: {failed}; inspect logs before resuming.')
            time.sleep(5)
    finally:
        for p,log,_ in workers:
            if p.poll() is None:
                p.terminate()
            log.close()
    assert all(p.returncode==0 for p,_,_ in workers)


def select():
    import shutil
    setup()
    records=[json.loads(path.read_text()) for path in (data.OUT/'candidates').glob('*.json')]
    extra_plan=data.OUT/'boosting-plan.json'
    extra_count=len(json.loads(extra_plan.read_text())['configurations']) if extra_plan.exists() else 0
    lexical_plan=data.OUT/'lexical-plan.json'
    if lexical_plan.exists():extra_count+=len(json.loads(lexical_plan.read_text())['configurations'])
    radial_plan=data.OUT/'radial-plan.json'
    if radial_plan.exists():extra_count+=len(json.loads(radial_plan.read_text())['configurations'])
    assert len(records)==len(CASES)*len(configurations())+extra_count,len(records)
    baseline=json.loads((data.OUT/'baseline-validation.json').read_text())
    floor=baseline['mean_accuracy']-.01
    eligible=[r for r in records if r['validation']['mean_accuracy']>=floor]
    assert eligible
    selected=max(eligible,key=lambda r:(r['validation']['mean_system_f1'],r['validation']['mean_accuracy']))
    frozen=dict(candidate=selected['candidate'],system_threshold=selected['validation']['system_threshold'],
                validation=selected['validation'],baseline_validation=baseline,
                guardrail_mean_accuracy_floor=floor,test_used_for_selection=False,
                data_plan_sha256=data.digest(data.OUT/'plan.json'),
                data_complete_sha256=data.digest(data.OUT/'data-complete.json'),
                supplement_data_sha256=data.digest(data.OUT/'supplement-data-complete.json'),
                search_plan_sha256=data.digest(data.OUT/'search-plan.json'),
                boosting_plan_sha256=data.digest(extra_plan) if extra_plan.exists() else None,
                lexical_plan_sha256=data.digest(lexical_plan) if lexical_plan.exists() else None,
                radial_plan_sha256=data.digest(radial_plan) if radial_plan.exists() else None,
                candidates={r['candidate']:data.digest(data.OUT/'candidates'/f'{r["candidate"]}.json') for r in records},
                selected_model_sha256=selected['model_sha256'],selected_unix_time=time.time())
    shutil.copy2(data.OUT/'checkpoints'/f'{selected["candidate"]}.npz',data.OUT/'models/selected.npz')
    for case in CASES:
        best=max([r for r in records if r['case']==case],key=lambda r:(r['validation']['mean_system_f1'],r['validation']['mean_accuracy']))
        shutil.copy2(data.OUT/'checkpoints'/f'{best["candidate"]}.npz',data.OUT/f'models/{case}.npz')
        shutil.copy2(data.OUT/'checkpoints'/f'{case}-linear-C0.005-w1.npz',data.OUT/f'models/fixed-{case}.npz')
    frozen['model_hashes']={path.name:data.digest(path) for path in (data.OUT/'models').glob('*.npz')}
    data.atomic_json(data.OUT/'selection.json',frozen)
    flat=[]
    for r in records:
        for name,m in r['validation']['profiles'].items():
            flat.append(dict(candidate=r['candidate'],case=r['case'],kind=r['kind'],profile=name,
                             **{k:v for k,v in m.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(flat).to_csv(data.OUT/'validation-metrics.csv',index=False)
    print(json.dumps(frozen),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['search','worker','select','baseline'],default='search')
    parser.add_argument('--case',choices=CASES)
    args=parser.parse_args()
    if args.stage=='worker':train_case(args.case)
    elif args.stage=='search':search()
    elif args.stage=='select':select()
    else:baseline_validation()
