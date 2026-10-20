"""Matched base/dense readouts for actual 1024-token role coverage.

Model selection uses the original four validation profiles plus long-full.
Neither old test results nor the future confirmation cohort are read.
"""
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
import systemness_long_data as long_data
import systemness_probes as probes

OUT=long_data.OUT
BANK=data.CACHE/'long-full-bank'
ORIGINAL_PROFILES=['single','multi','natural','long-single']
PROFILES=ORIGINAL_PROFILES+['long-full']
CONFIGS={
    'base-mlp64':dict(dense=False,kind='mlp',hidden=64),
    'dense-mlp64':dict(dense=True,kind='mlp',hidden=64),
    'dense-mlp128':dict(dense=True,kind='mlp',hidden=128),
    'dense-linear':dict(dense=True,kind='linear',hidden=0,C=.005),
    'resampled-mlp64':dict(dense=False,kind='mlp',hidden=64,resample=True),
}
RESAMPLED_EXTRA_INSTANCES=164800
BATCH_SIZE=4096
CONTROL_LIMITATION=('Dense MLPs and resampled-mlp64 match instances and optimizer updates per epoch and at the maximum epoch budget; '
    'base-mlp64 remains the original unresampled secondary control. '
    'independent early stopping can yield different best and actual total update counts. '
    'Resampling controls update volume; long-text/profile sampling weights and distinct examples still differ.')
DEPENDENCIES=['systemness_long_probes.py','systemness_probes.py','systemness_data.py',
              'systemness_attention.py','systemness_long_data.py']


def verify_plan(plan):
    assert plan['source_sha256']==data.digest(__file__)
    assert plan['dependencies_sha256']=={name:data.digest(data.HERE/name) for name in DEPENDENCIES}
    assert plan['data_complete_sha256']==data.digest(OUT/'data-complete.json')
    assert plan['extraction_sha256']==data.digest(OUT/'extraction-complete.json')
    assert plan['standardization_sha256']==data.digest(attention.OUT/'standardization.npz')
    assert plan['previous_selection_sha256']==data.digest(attention.OUT/'selection.json')


def candidate_record(key,plan):
    verify_plan(plan)
    record=json.loads((OUT/'candidates'/f'{key}.json').read_text())
    assert record['candidate']==key and all(record[k]==v for k,v in CONFIGS[key].items())
    assert record['source_sha256']==plan['source_sha256']
    assert record['search_plan_sha256']==data.digest(OUT/'search-plan.json')
    assert record['standardization_sha256']==plan['standardization_sha256']
    assert record['feature_spec']==attention.SPECS and record['tests_read'] is False
    assert record['portable_probability_max_difference']<5e-4
    assert record['model_sha256']==data.digest(OUT/'models'/f'{key}.npz')
    assert record['validation_probabilities_sha256']==data.digest(BANK/f'{key}-validation.npz')
    return record


def validation_metrics(rows,p):
    assert set(rows.split)=={'validation'} and set(rows.wrapper)=={'native'}
    assert set(rows.profile)==set(PROFILES)
    y=probes.labels(rows)
    thresholds=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(p[:,0],np.linspace(0,1,301)),1.000001])
    scores=[];fprs=[]
    for profile in PROFILES:
        mask=rows.profile.to_numpy()==profile;truth=y[mask]==0;prob=p[mask,0]
        positive=np.sort(prob[truth]);negative=np.sort(prob[~truth])
        tp=len(positive)-np.searchsorted(positive,thresholds,side='left')
        fp=len(negative)-np.searchsorted(negative,thresholds,side='left')
        scores.append(2*tp/np.maximum(2*tp+fp+len(positive)-tp,1))
        fprs.append(fp/max(len(negative),1))
    means=np.mean(scores,axis=0);means[np.max(fprs,axis=0)>.05]=-1
    threshold=float(thresholds[int(np.argmax(means))])
    results={profile:probes.metrics(p[rows.profile.to_numpy()==profile],y[rows.profile.to_numpy()==profile],threshold) for profile in PROFILES}
    full=rows[rows.profile=='long-full'].copy()
    position=full.token_ix-full.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    tail_indices=full.index[position>=512].to_numpy()
    # rows are reset before this helper is called; indices refer directly to p.
    assert np.array_equal(rows.index.to_numpy(),np.arange(len(rows)))
    tail=probes.metrics(p[tail_indices],y[tail_indices],threshold)
    tail.update(documents=int(full.loc[tail_indices].document_id.nunique()),body_start=512,body_end=1023)
    return dict(system_threshold=threshold,mean_system_f1=float(np.mean([r['system_f1'] for r in results.values()])),
        original_mean_system_f1=float(np.mean([results[k]['system_f1'] for k in ORIGINAL_PROFILES])),
        original_mean_accuracy=float(np.mean([results[k]['accuracy'] for k in ORIGINAL_PROFILES])),
        profiles=results,long_full_tail=tail)


def dataset():
    rows=pd.read_parquet(BANK/'rows.parquet')
    complete=json.loads((OUT/'extraction-complete.json').read_text())
    assert complete['rows']==len(rows) and complete['rows_sha256']==data.digest(BANK/'rows.parquet')
    assert np.array_equal(rows.cache_ix,np.arange(len(rows))) and rows.sample_ix.is_unique
    assert len(rows)==755282
    array=np.memmap(BANK/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),8640))
    return rows,array


def fit_rows(rows,dense):
    fit=rows[rows.split=='fit']
    if not dense:return fit[fit.sample_ix<long_data.SAMPLE_START]
    full_ids=set(fit[fit.profile=='long-full'].document_id)
    assert len(full_ids)==132
    return fit[~((fit.profile=='long-single')&fit.document_id.isin(full_ids))]


def epoch_order(count,generator,device,extra_instances=0):
    import torch
    assert 0<=extra_instances<=count
    order=torch.randperm(count,device=device,generator=generator)
    if extra_instances:
        extra=torch.randperm(count,device=device,generator=generator)[:extra_instances]
        # One flattened sequence preserves the same number and size of batches
        # as dense training; two separate passes would add a second short batch.
        order=torch.cat([order,extra])
    return order


def worker(key):
    import cupy as cp
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    assert cp.cuda.runtime.getDeviceCount()==torch.cuda.device_count()==1
    plan=json.loads((OUT/'search-plan.json').read_text());verify_plan(plan)
    standardization=attention.OUT/'standardization.npz'
    assert data.digest(standardization)==plan['standardization_sha256']
    with np.load(standardization,allow_pickle=False) as archive:mean=archive['mean'].copy();scale=archive['scale'].copy()
    assert mean.shape==scale.shape==(8640,) and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all()
    rows,bank=dataset();config=CONFIGS[key];fit=fit_rows(rows,config['dense'])
    valid=rows[rows.split=='validation'].copy().reset_index(drop=True)
    assert len(fit)==(578090 if config['dense'] else 413290) and len(valid)==173032
    assert fit.document_id.nunique()==1200 and not set(fit.group_id)&set(valid.group_id)
    free_bytes,_=cp.cuda.runtime.memGetInfo()
    assert free_bytes>(len(fit)+len(valid))*8640*4+2*2**30,'Insufficient GPU memory for complete rows plus bounded workspace.'
    def load(indices):
        x=cp.empty((len(indices),8640),dtype=cp.float32,order='F')
        for start in range(0,len(indices),4096):
            block=np.asarray(bank[indices[start:start+4096]],dtype=np.float32)
            z=(block-mean)/scale;assert np.isfinite(z).all()
            x[start:start+len(block)]=cp.asarray(z)
        return x
    started=time.time();x=load(fit.cache_ix.to_numpy());xv=load(valid.cache_ix.to_numpy())
    y=probes.labels(fit);path=OUT/'models'/f'{key}.npz';probability_path=BANK/f'{key}-validation.npz'
    details=dict(**config,roles=data.ROLES,feature_spec=attention.SPECS,fit_documents=1200,
        dense_full_length_documents=132 if config['dense'] else 0,train_tokens=len(fit),seed=data.SEED,
        standardization_sha256=plan['standardization_sha256'],source_sha256=data.digest(__file__),
        search_plan_sha256=data.digest(OUT/'search-plan.json'),inference='Pointwise local attention features only.')
    extra_instances=RESAMPLED_EXTRA_INSTANCES if config.get('resample',False) else 0
    if extra_instances:assert not config['dense'] and len(fit)==413290 and len(fit)+extra_instances==578090
    if config['kind']=='mlp':
        instances_per_epoch=len(fit)+extra_instances
        steps_per_epoch=(instances_per_epoch+BATCH_SIZE-1)//BATCH_SIZE
        details.update(unique_fit_rows=len(fit),extra_instances_per_epoch=extra_instances,
            instances_per_epoch=instances_per_epoch,steps_per_epoch=steps_per_epoch,
            last_batch_instances=(instances_per_epoch-1)%BATCH_SIZE+1,
            control_limitation=CONTROL_LIMITATION)
    if config['kind']=='linear':
        from cuml.linear_model import LogisticRegression
        clf=LogisticRegression(C=.005,max_iter=5000,linesearch_max_iter=100,tol=1e-4,output_type='cupy')
        clf.fit(x,cp.asarray(y));p=cp.asnumpy(clf.predict_proba(xv))
        net=probes.make_net('linear',0)
        with torch.no_grad():
            net.head.weight.copy_(torch.from_numpy(cp.asnumpy(clf.coef_)))
            net.head.bias.copy_(torch.from_numpy(cp.asnumpy(clf.intercept_).reshape(-1)))
        probes.save_model(path,net,mean,scale,details)
        details['iterations']=cp.asnumpy(cp.asarray(clf.n_iter_)).tolist()
    else:
        xt=torch.from_dlpack(x);xvt=torch.from_dlpack(xv);yt=torch.from_numpy(y).cuda()
        torch.manual_seed(data.SEED);net=probes.make_net('mlp',config['hidden']).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        generator=torch.Generator(device='cuda:0').manual_seed(data.SEED)
        best=-1.;best_epoch=-1;history=[]
        for epoch in range(1,46):
            net.train();order=epoch_order(len(xt),generator,'cuda',extra_instances);loss_sum=0.;steps=0
            for start in range(0,len(order),BATCH_SIZE):
                ix=order[start:start+BATCH_SIZE]
                loss=torch.nn.functional.cross_entropy(net.raw(xt[ix].contiguous()),yt[ix]);assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();loss_sum+=float(loss.detach())*len(ix);steps+=1
            assert len(order)==instances_per_epoch and steps==steps_per_epoch
            p=probes.predict_standardized(net,xvt);result=validation_metrics(valid,p);score=result['mean_system_f1']
            history.append(dict(epoch=epoch,loss=loss_sum/len(order),instances=len(order),steps=steps,
                cumulative_steps=epoch*steps_per_epoch,mean_system_f1=score,
                long_full_f1=result['profiles']['long-full']['system_f1'],tail_f1=result['long_full_tail']['system_f1']))
            if score>best+1e-5:
                best=score;best_epoch=epoch
                probes.save_model(path,net,mean,scale,details|dict(best_epoch=epoch,best_steps=epoch*steps_per_epoch))
                np.savez_compressed(probability_path,p=p,sample_ix=valid.sample_ix.to_numpy())
            print(json.dumps(dict(candidate=key,**history[-1],best_epoch=best_epoch)),flush=True)
            if epoch>=10 and epoch-best_epoch>=8:break
        with np.load(probability_path,allow_pickle=False) as archive:
            p=archive['p'].copy();assert np.array_equal(archive['sample_ix'],valid.sample_ix)
        details.update(best_epoch=best_epoch,history=history,epochs=len(history),
            best_steps=best_epoch*steps_per_epoch,actual_steps=sum(item['steps'] for item in history),
            actual_instances=sum(item['instances'] for item in history))
    np.savez_compressed(probability_path,p=p,sample_ix=valid.sample_ix.to_numpy())
    portable,m,s,_=probes.load_model(path,device='cpu');ids=np.linspace(0,len(valid)-1,256,dtype=int)
    raw=torch.from_numpy(np.asarray(bank[valid.iloc[ids].cache_ix.to_numpy()],dtype=np.float32))
    with torch.inference_mode():actual=portable((raw-m)/s).exp().numpy()
    difference=float(np.abs(actual-p[ids]).max());assert difference<5e-4,difference
    verify_plan(plan)
    data.atomic_json(OUT/'candidates'/f'{key}.json',dict(candidate=key,**details,
        validation=validation_metrics(valid,p),seconds=time.time()-started,model_sha256=data.digest(path),
        portable_probability_max_difference=difference,tests_read=False,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        validation_probabilities_sha256=data.digest(probability_path)))


def select():
    plan=json.loads((OUT/'search-plan.json').read_text())
    records=[candidate_record(key,plan) for key in CONFIGS]
    control=next(r for r in records if r['candidate']=='resampled-mlp64')['validation']
    secondary_control=next(r for r in records if r['candidate']=='base-mlp64')['validation']
    previous=json.loads((attention.OUT/'selection.json').read_text())
    accuracy_floor=previous['validation']['mean_accuracy']-.01
    eligible=[r for r in records if r['validation']['original_mean_accuracy']>=accuracy_floor]
    if not eligible:eligible=[next(r for r in records if r['candidate']=='resampled-mlp64')]
    selected=max(eligible,key=lambda r:r['validation']['mean_system_f1']);v=selected['validation']
    mean_gain=v['mean_system_f1']-control['mean_system_f1']
    full_gain=v['profiles']['long-full']['system_f1']-control['profiles']['long-full']['system_f1']
    tail_gain=v['long_full_tail']['system_f1']-control['long_full_tail']['system_f1']
    other_loss=max(control['profiles'][k]['system_f1']-v['profiles'][k]['system_f1'] for k in ['single','multi','natural'])
    promoted=selected['dense'] and mean_gain>=.01 and full_gain>=.03 and tail_gain>=.03 and other_loss<=.02
    frozen=dict(candidate=selected['candidate'],model_path=str(OUT/'models'/f'{selected["candidate"]}.npz'),
        model_sha256=selected['model_sha256'],system_threshold=v['system_threshold'],validation=v,
        matched_control=control,promoted=promoted,mean_f1_gain=mean_gain,long_full_f1_gain=full_gain,
        matched_control_candidate='resampled-mlp64',secondary_control_candidate='base-mlp64',
        secondary_control=secondary_control,control_limitation=CONTROL_LIMITATION,
        secondary_mean_f1_gain=v['mean_system_f1']-secondary_control['mean_system_f1'],
        tail_f1_gain=tail_gain,maximum_other_profile_f1_loss=other_loss,accuracy_floor=accuracy_floor,
        test_used_for_selection=False,confirmation_required=True,selected_unix_time=time.time(),
        source_sha256=data.digest(__file__),search_plan_sha256=data.digest(OUT/'search-plan.json'),
        candidates={r['candidate']:data.digest(OUT/'candidates'/f'{r["candidate"]}.json') for r in records})
    flat=[]
    for r in records:
        for profile,m in dict(r['validation']['profiles'],**{'long-full-tail':r['validation']['long_full_tail']}).items():
            flat.append(dict(candidate=r['candidate'],profile=profile,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False)
    data.atomic_json(OUT/'selection.json',frozen)
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',candidate=selected['candidate'],promoted=promoted))
    print(json.dumps({k:v for k,v in frozen.items() if k not in ['validation','matched_control','secondary_control','candidates']}),flush=True)


def run_workers(pending,plan):
    # Five candidates run in two waves, each with at most four isolated CUDA
    # processes. A later wave starts only after every previous process is reaped.
    for offset in range(0,len(pending),4):
        workers=[]
        try:
            for gpu,key in enumerate(pending[offset:offset+4]):
                log=(OUT/'logs'/f'{key}.log').open('a')
                try:
                    proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',key],
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                workers.append((key,proc,log))
            data.atomic_json(OUT/'status.json',dict(stage='training',wave=offset//4+1,
                workers=[dict(candidate=key,pid=proc.pid,gpu=gpu) for gpu,(key,proc,_) in enumerate(workers)]))
            while True:
                codes=[(key,proc.poll()) for key,proc,_ in workers]
                failed=[(key,code) for key,code in codes if code not in [None,0]]
                if failed:raise RuntimeError(f'Long-full workers failed: {failed}; inspect logs.')
                if all(code==0 for _,code in codes):break
                time.sleep(5)
            for key,proc,_ in workers:
                assert proc.wait()==0
                candidate_record(key,plan)
        finally:
            cleanup_errors=[]
            for key,proc,_ in workers:
                try:
                    if proc.poll() is None:proc.terminate()
                except BaseException as error:cleanup_errors.append((key,repr(error)))
            for key,proc,log in workers:
                try:
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait(timeout=10)
                except BaseException as error:cleanup_errors.append((key,repr(error)))
                finally:log.close()
            if cleanup_errors:raise RuntimeError(f'Could not reap long-full workers: {cleanup_errors}')


def main():
    assert (OUT/'extraction-complete.json').exists() and not (OUT/'selection.json').exists()
    extraction=json.loads((OUT/'extraction-complete.json').read_text())
    assert extraction['sha256']==data.digest(BANK/'features.f16')
    for folder in ['models','candidates','logs']:(OUT/folder).mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('/logs/\n')
    old=json.loads((attention.OUT/'selection.json').read_text())
    plan=dict(configurations=CONFIGS,source_sha256=data.digest(__file__),
        dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        previous_selection_sha256=data.digest(attention.OUT/'selection.json'),
        data_complete_sha256=data.digest(OUT/'data-complete.json'),
        extraction_sha256=data.digest(OUT/'extraction-complete.json'),
        feature_spec=attention.SPECS,validation_profiles=PROFILES,
        standardization_sha256=data.digest(attention.OUT/'standardization.npz'),
        standardization='Shared original base-fit mean/scale; no validation rows and no scaler change between controls.',
        training='Dense variants replace sparse old long-single rows for selected132 documents; both base controls keep the original413290 unique fit rows.',
        matched_control='resampled-mlp64',secondary_control='base-mlp64',control_limitation=CONTROL_LIMITATION,
        resampled_control=dict(unique_fit_rows=413290,extra_instances=RESAMPLED_EXTRA_INSTANCES,
            instances_per_epoch=578090,steps_per_epoch=142,last_batch_instances=554,
            sampling='Concatenate one complete randperm with an independent randperm prefix of164800 original fit rows; then slice4096 batches.',
            loss='Unchanged five-class mean cross entropy; epoch loss denominator is sampled instances.',
            max_epochs=45,maximum_steps=45*142,early_stopping='Independent per candidate; record best_steps and actual_steps.'),
        validation='Original four profiles unchanged plus31 actual1024 docs;10 overlap the original long validation as paired variants.',
        selection='Maximum mean system F1 over5 profiles, original4-profile five-way accuracy guardrail.',
        threshold='One global validation threshold; FPR<=5% in each of5 profiles.',
        promotion=dict(mean_f1_gain=.01,long_full_f1_gain=.03,tail_f1_gain=.03,maximum_other_profile_loss=.02),
        accuracy_floor=old['validation']['mean_accuracy']-.01,
        neural=dict(seed=data.SEED,max_epochs=45,min_epochs=10,patience=8,batch_size=4096,
            learning_rate=.001,weight_decay=.001,dropout=.1,gradient_clip=5.),
        tests_read=False,fresh_confirmation_required=True)
    if (OUT/'search-plan.json').exists():assert json.loads((OUT/'search-plan.json').read_text())==plan
    else:data.atomic_json(OUT/'search-plan.json',plan)
    pending=[]
    for key in CONFIGS:
        if (OUT/'candidates'/f'{key}.json').exists():
            candidate_record(key,plan)
            continue
        pending.append(key)
    run_workers(pending,plan)
    select()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=CONFIGS);args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:main()
