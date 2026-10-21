"""Clean public natural fit versus role-slot-matched old-fit sampling.

Default help; --declare metadata freeze; --train explicit. --preflight uses only
synthetic CPU/tmp fixtures. Exactly8 developer records, only new-max33 eligible.
No test, LM, new sources, plots or current-primary update. Legacy old-natural
contains CoT truth; public natural fit/validation is S/U/A-only.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
import systemness_data as data
import systemness_head_probes as hp
import systemness_long_probes as lp
import systemness_probes as probes
import systemness_long_augmentation_probes as aug
import systemness_multiseed_ensemble as ms
import systemness_natural_dev_extract as natural

OUT=data.runtime.OUT/'systemness-natural-fit-probes'
CACHE=data.CACHE/'natural-fit-probes'
DESIGN=data.runtime.OUT/'systemness-natural-fit-preparation/root-prospective-design.json'
DESIGN_SHA='9e8cae74eef9317265cfe4417b807d856d9d2332ca081695780e657629c39f40'
ARMS=['new','matched'];MEMBERS=[20261021,20261022,20261023,20261024]
RECORDS=[f'{m}-{o}' for m in ['new','matched','oldhead'] for o in ['oldmean','max33']]+['ensemble-original','ensemble-max33']
OLD5=['old-'+p for p in lp.PROFILES]
PUBLIC=['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
FULL=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']
GROUPS=OLD5+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']+PUBLIC
FEASIBLE=OLD5+['new-C4-long-full']+PUBLIC
GUARD_CONTROLS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original']
FIT,EXTRA,OLD_VALID,C4_VALID,PUBLIC_VALID=578090,63403,173032,61440,16713
INSTANCES,BATCH,STEPS,DRAW_SEED=641493,4096,157,20516085
ORIGINAL_TAU,ENSEMBLE_TAU=aug.CURRENT_THRESHOLD,.4747979797979798
ROLE_COUNTS={'fit':{'system':7935,'user':12660,'assistant':42808},'validation':{'system':2032,'user':3596,'assistant':11085}}
ARTIFACTS=['protocol.json','search-plan.json','draw-manifest.json','controls.json','records.json','selection.json','validation-metrics.csv','source-counts.json','report.txt']
DEPENDENCIES=sorted(set(ms.DEPENDENCIES+natural.DEPENDENCIES+['systemness_natural_fit_probes.py']))
require,read,array_sha=hp.require,hp.read,hp.array_digest
LIMITATION=('One shared initialization; repeated developer calibration; role-slot/update matching does not match document/style/body-position distribution. '
    'Old/new/public validation are developer data. Legacy old-natural has CoT truth and tool0; fresh public data is SUA-only. '
    'Inherited4seed control is not4 independent datasets. Different early stopping can change realized total updates. '
    'min33>=.90 is not goal completion: fresh disjoint C4/natural confirmation and fixed tomato plots remain required.')


def protocol():
    return dict(version=1,experiment='natural-fit-role-slot-control',arms=ARMS,records=RECORDS,roles=data.ROLES,
        fit_rows=FIT,natural_fit=EXTRA,validation_rows=dict(old=OLD_VALID,c4=C4_VALID,public=PUBLIC_VALID),
        neural=dict(seed=20261021,kind='mlp',hidden=128,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,
            batch=BATCH,instances=INSTANCES,steps=STEPS,last_batch=2517,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5),
        matched=dict(seed=DRAW_SEED,draw_once=True,replacement=True,role_order=['system','user','assistant'],torch_rng_consumed=False),
        early_stop='Original old5 helper only; best model/GPUoldp/history freeze before C4/public validation predictions.',
        CPU=dict(threads=4,float32_matmul_precision='highest',dtype='float32',chunk=4096),
        parity='All readouts use only fixed first256 clean natural FIT vectors for CPU/GPU parity; no validation parity sample.',
        groups=GROUPS,feasible=FEASIBLE,max_FPR=.05,maximin='ActualF32 fullgrid min33; tie higherold5meanF1 then smalleroriginalF64tau.',
        ensemble=dict(members=MEMBERS,weights=[.25]*4,arithmetic='Fixed-order F64 sum/4 castF32, no normalization.',original_tau=ENSEMBLE_TAU),
        promotion=dict(candidate='new-max33',absolute_min33=.9,strict_controls=GUARD_CONTROLS[:3],guard_controls=GUARD_CONTROLS,
            old4_F1_loss=.02,old4_accuracy_loss=.01,old5_mean_F1_loss=.005,full_tail_F1_nonregression=FULL,
            public_F1_loss=.02,public_accuracy_loss=.01,mandatory_savedGPU_floors=True,tolerance=1e-12,no_fallback=True),
        tests_read=False,primary_changed=False,new_independent_confirmation_required=True,limitation=LIMITATION)


def layout():
    old=pd.read_parquet(hp.BANK/'rows.parquet');fit,ov,part=hp.partitions(old)
    c4=pd.read_parquet(aug.NEW_BANK/'rows.parquet');cv=c4[c4.split.eq('validation')].copy().reset_index(drop=True)
    rows=pd.read_parquet(natural.CACHE/'rows.parquet');nf=rows[rows.split.eq('fit')].copy().reset_index(drop=True);pv=rows[rows.split.eq('validation')].copy().reset_index(drop=True)
    require(len(fit)==FIT and len(ov)==OLD_VALID and len(cv)==C4_VALID and len(rows)==80116 and len(nf)==EXTRA and len(pv)==PUBLIC_VALID
        and np.array_equal(rows.cache_ix,np.arange(len(rows))) and rows.sample_ix.is_unique and rows.sample_ix.min()>=4_000_000_000_000
        and rows.original_cache_ix.is_monotonic_increasing and rows.sample_ix.is_monotonic_increasing
        and rows.wrapper.eq('native').all() and rows.source_set.eq('natural-dev-clean').all() and set(rows.role)=={'system','user','assistant'},'Natural/original development namespace or counts changed.')
    require(nf.eligible_fit.eq(True).all() and pv.eligible_clean_validation.eq(True).all()
        and not rows.old_generated_collision.any() and not rows.cross_split_assistant_collision.any(),'Contaminated natural rows entered training/validation.')
    for key in ['source_uid','group_id','family_id','dependency_component_id']:
        require(not set(nf[key])&set(pv[key]),'Natural fit/public validation leakage: '+key)
    for split,frame,count in [('fit',nf,255),('validation',pv,64)]:
        require(frame.source_uid.nunique()==count and {r:int(frame.role.eq(r).sum()) for r in ROLE_COUNTS[split]}==ROLE_COUNTS[split], 'Natural source/role support changed.')
    require(cv.sample_ix.min()>=6_000_000_000_000 and set(cv.role)==set(data.ROLES) and cv.profile.eq('long-full').all()
        and not set(cv.group_id)&set(fit.group_id) and not set(rows.sample_ix)&set(old.sample_ix),'C4/public/old split overlap.')
    return fit,ov,cv,nf,pv,part


def matched_draw(fit,nf):
    require(set(nf.role)<=set(['system','user','assistant']) and fit.cache_ix.is_unique and fit.split.eq('fit').all()
        and nf.split.eq('fit').all() and nf.sample_ix.is_unique,'Invalid matched role pool or fit/validation leakage.')
    rng=np.random.RandomState(DRAW_SEED);chosen=np.empty(len(nf),np.int64);chosen_sample=np.empty(len(nf),np.int64)
    for role in ['system','user','assistant']:
        slots=np.flatnonzero(nf.role.eq(role));pool=fit[fit.role.eq(role)]
        require(len(pool)>0,'Missing original fit role pool.');draw=rng.choice(len(pool),size=len(slots),replace=True)
        chosen[slots]=pool.iloc[draw].cache_ix.to_numpy();chosen_sample[slots]=pool.iloc[draw].sample_ix.to_numpy()
    slots=pd.DataFrame(dict(extra_slot=np.arange(len(nf)),natural_fit_sample_ix=nf.sample_ix.to_numpy(),natural_source_uid=nf.source_uid.to_numpy(),
        role=nf.role.to_numpy(),matched_old_cache_ix=chosen,matched_old_sample_ix=chosen_sample))
    require(np.array_equal(fit.set_index('cache_ix').loc[chosen].role.to_numpy(),nf.role.to_numpy()),'Matched slot formal-role labels differ.')
    summary=dict(seed=DRAW_SEED,draw_once=True,replacement=True,rows=len(slots),role_counts={r:int(nf.role.eq(r).sum()) for r in ['system','user','assistant']},
        matched_cache_ix_sha256=array_sha(chosen),matched_sample_ix_sha256=array_sha(chosen_sample),natural_fit_sample_ix_sha256=array_sha(nf.sample_ix.to_numpy()),
        extra_labels_sha256=array_sha(probes.labels(nf)),no_validation_rows_sampled=True,torch_rng_consumed=False)
    return slots,summary


def static_snapshot():
    hp.require_hash(DESIGN,DESIGN_SHA,'root natural-fit design');base=ms.static_snapshot()
    files=[ms.OUT/n for n in ['protocol.json','search-plan.json','complete.json','selection.json','ensemble.json','old-control.json','actual-audit/actual-development-audit.json']]
    files += [ms.OUT/sub/f'{s}{extension}' for s in MEMBERS[1:] for sub,extension in [('models','.npz'),('candidates','.json'),('training','.json')]]
    files += [natural.OUT/n for n in ['extraction-plan.json','extraction-complete.json','portable-validation.json','hook-validation.json','model_backend.json']]+[natural.CACHE/'rows.parquet']
    require(all(p.is_file() for p in files),'Await completed natural extraction and audited prior ensemble before declaration.')
    audit=read(ms.OUT/'actual-audit/actual-development-audit.json');complete=read(ms.OUT/'complete.json');selection=read(ms.OUT/'selection.json');spec=read(ms.OUT/'ensemble.json')
    hp.require_hash(ms.OUT/'actual-audit/actual-development-audit.json',read(DESIGN)['root_execution_decision']['negative_seed_actual_audit_sha256'],'prior seed actual audit')
    require(audit['stage']=='actual_completed_multiseed_development_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False
        and audit['trainer_complete_sha256']==data.digest(ms.OUT/'complete.json') and selection['promoted'] is False
        and selection['system_threshold']==ENSEMBLE_TAU and complete['source_sha256']==data.digest(ms.__file__),'Inherited negative4seed control changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(ms.OUT/n,h,'inherited artifact')
    paths={'20261021':str(hp.OUT/'models/joint-mlp128.npz'),**{str(s):str(ms.OUT/'models'/f'{s}.npz') for s in MEMBERS[1:]}}
    require(spec['members']==MEMBERS and spec['weights']==[.25]*4 and spec['model_sha256']==audit['member_model_sha256'],'Inherited member order/weights/model identity changed.')
    for s,path in paths.items():hp.require_hash(path,spec['model_sha256'][s],'inherited portable member')
    nc=read(natural.OUT/'extraction-complete.json');parity=read(natural.OUT/'portable-validation.json')
    require(nc['rows']==80116 and nc['dimensions']==512 and nc['all_rows_written'] is True and nc['all_finite'] is True
        and nc['tests_read'] is False and nc['readout_training'] is False and nc['selected_head_ids']==spec['signature']['selected_head_ids']
        and parity['fit_only'] is True and parity['validation_probabilities_computed'] is False,'Natural bank completion/FIT-only parity boundary changed.')
    hp.require_hash(natural.CACHE/'rows.parquet',nc['rows_sha256'],'natural row mapping')
    fit,ov,cv,nf,pv,part=layout();slots,summary=matched_draw(fit,nf);saved=pd.read_parquet(CACHE/'matched-draw.parquet')
    pd.testing.assert_frame_equal(saved,slots,check_exact=True)
    draw=read(OUT/'draw-manifest.json');require(all(draw[k]==v for k,v in summary.items()) and draw['slots_sha256']==data.digest(CACHE/'matched-draw.parquet'),'Fixed role-slot draw changed.')
    return dict(source_sha256=data.digest(__file__),design_sha256=DESIGN_SHA,base=base,metadata_sha256={str(p):data.digest(p) for p in files},
        dependencies_sha256={n:data.digest(data.HERE/n) for n in DEPENDENCIES},model_paths=paths,model_sha256=spec['model_sha256'],
        initial_weight_sha256=ms.initial_hash(20261021),standardization_sha256=data.digest(hp.OUT/'standardization.npz'),
        draw_manifest_sha256=data.digest(OUT/'draw-manifest.json'),draw_slots_sha256=data.digest(CACHE/'matched-draw.parquet'),original_partitions=part,
        sample_ix_sha256={name:array_sha(rows.sample_ix.to_numpy()) for name,rows in [('fit',fit),('old',ov),('c4',cv),('natural-fit',nf),('public',pv)]},
        old_reference=base['old_reference'],ensemble_reference=dict(old_path=str(ms.CACHE/'ensemble-old-validation.npz'),old_sha256=selection['ensemble_probability_sha256']['old'],
            new_path=str(ms.CACHE/'ensemble-new-validation.npz'),new_sha256=selection['ensemble_probability_sha256']['new']),
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},validation_probability_arrays_read=False)


def declare():
    path=OUT/'protocol.json'
    if path.exists():return verify_declaration()
    require(not CACHE.exists() and not (OUT/'search-plan.json').exists(),'Partial metadata/cache predates declaration; inspect explicitly.')
    hp.require_hash(DESIGN,DESIGN_SHA,'root design')
    require((natural.OUT/'extraction-complete.json').is_file(),'Natural extraction must complete before declaration.')
    fit,_,_,nf,_,_=layout();slots,summary=matched_draw(fit,nf)
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True);slots.to_parquet(CACHE/'matched-draw.parquet',index=False)
    data.atomic_json(OUT/'draw-manifest.json',summary|dict(slots_path=str(CACHE/'matched-draw.parquet'),slots_sha256=data.digest(CACHE/'matched-draw.parquet')))
    value=dict(protocol=protocol(),static=static_snapshot());data.atomic_json(path,value);return value


def verify_declaration():
    require((OUT/'protocol.json').is_file(),'Run --declare before candidate probabilities/training.')
    value=read(OUT/'protocol.json');require(value==dict(protocol=protocol(),static=static_snapshot()),'Frozen source/design/model/draw declaration changed.');return value


def source_signature():
    declaration=verify_declaration();up=aug.source_signature();nc=natural.verified_extraction()
    hp.require_hash(natural.CACHE/'features.f16',nc['features_sha256'],'natural features')
    return dict(source_sha256=data.digest(__file__),protocol_sha256=data.digest(OUT/'protocol.json'),static_sha256=aug.json_digest(declaration['static']),
        upstream=up,natural_complete_sha256=data.digest(natural.OUT/'extraction-complete.json'),natural_signature=nc['signature'],
        natural_features_sha256=nc['features_sha256'],natural_rows_sha256=nc['rows_sha256'],selected_head_ids=up['selected_head_ids'],
        standardization_sha256=declaration['static']['standardization_sha256'])


def verify_plan(plan,full=True):
    declaration=verify_declaration();require(plan['protocol']==protocol() and plan['signature']['protocol_sha256']==data.digest(OUT/'protocol.json')
        and plan['initial_weight_sha256']==declaration['static']['initial_weight_sha256'],'Plan/protocol/init changed.')
    if full:require(plan['signature']==source_signature(),'Feature/data/source signature changed.')
    return declaration


def dataset():
    fit,ov,cv,nf,pv,part=layout();banks={}
    for key,path,count in [('old',hp.BANK,755282),('c4',aug.NEW_BANK,389120),('natural',natural.CACHE,80116)]:
        require((path/'features.f16').stat().st_size==count*512*2,'Missing/truncated raw feature bank.')
        banks[key]=np.memmap(path/'features.f16',mode='r',dtype=np.float16,shape=(count,512))
    return banks,fit,ov,cv,nf,pv,part


def validation_groups(rows):
    old,cv,pub=rows['old'],rows['c4'],rows['public'];base=aug.validation_groups(old,cv);groups={}
    for name,(bank,ix) in base.items():groups[name.replace('new-long','new-C4-long')]=('old' if bank=='old' else 'c4',ix)
    groups[PUBLIC[0]]=('public',np.arange(len(pub)))
    for name,dataset_name in zip(PUBLIC[1:],['oasst','toxicchat0124']):groups[name]=('public',np.flatnonzero(pub.dataset.eq(dataset_name)))
    require(set(groups)==set(GROUPS) and all(len(ix) for _,ix in groups.values()),'Missing max33 group.');return groups


def metrics_at(rows,p,tau):
    groups={g:probes.metrics(p[b][ix],probes.labels(rows[b].iloc[ix]),tau) for g,(b,ix) in validation_groups(rows).items()}
    values=[groups[g][k] for g in GROUPS for k in ['system_precision','system_recall','system_balanced_accuracy']]
    separate={b:probes.metrics(p[b],probes.labels(rows[b]),tau) for b in rows}
    return dict(system_threshold=float(tau),groups=groups,min33=min(values) if all(v is not None and np.isfinite(v) for v in values) else None,
        old5_mean_f1=float(np.mean([groups[g]['system_f1'] for g in OLD5])),separate_bank_descriptive=separate,
        pooled_descriptive_only=probes.metrics(np.concatenate([p[b] for b in rows]),np.concatenate([probes.labels(rows[b]) for b in rows]),tau))


def max33(rows,p):
    groups=validation_groups(rows);grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate([p[b][:,0] for b in ['old','c4','public']]),np.linspace(0,1,301)),1.000001])
    effective=grid.astype(np.float32);curve={};direct={}
    for g,(b,ix) in groups.items():
        hp.validate_probabilities(p[b],len(rows[b]));require(p[b].dtype==np.float32,'Max33 needs actual F32 p.')
        truth=probes.labels(rows[b].iloc[ix])==0;score=p[b][ix,0];pos,neg=np.sort(score[truth]),np.sort(score[~truth]);require(len(pos) and len(neg),'Group lacks both truth supports.')
        tp=len(pos)-np.searchsorted(pos,effective,side='left');fp=len(neg)-np.searchsorted(neg,effective,side='left');fn=len(pos)-tp;tn=len(neg)-fp
        values=dict(P=np.divide(tp,tp+fp,out=np.full(len(grid),np.nan),where=tp+fp>0),R=tp/len(pos),BAcc=(tp/len(pos)+tn/len(neg))/2,F1=2*tp/(2*tp+fp+fn),FPR=fp/len(neg))
        actual={k:[] for k in values}
        for tau in grid:
            detected=score>=float(tau);a=int(np.count_nonzero(truth&detected));bb=int(np.count_nonzero(~truth&detected));c=len(pos)-a;d=len(neg)-bb
            for k,v in dict(P=a/(a+bb) if a+bb else np.nan,R=a/len(pos),BAcc=(a/len(pos)+d/len(neg))/2,F1=2*a/(2*a+bb+c),FPR=bb/len(neg)).items():actual[k].append(v)
        direct[g]={k:np.asarray(v) for k,v in actual.items()};curve[g]=values
        require(all(np.array_equal(values[k],direct[g][k],equal_nan=True) for k in values),'F32 every-grid consumer mismatch.')
    def objective(c):
        minimum=np.min([c[g][k] for g in GROUPS for k in ['P','R','BAcc']],axis=0);minimum[~np.isfinite(minimum)]=-np.inf
        mean=np.mean([c[g]['F1'] for g in OLD5],axis=0);feasible=np.max([c[g]['FPR'] for g in FEASIBLE],axis=0)<=.05
        choices=np.flatnonzero(feasible&np.isfinite(minimum));require(len(choices),'No defined FPR-feasible max33 threshold.')
        best=max(choices,key=lambda i:(float(minimum[i]),float(mean[i]),-float(grid[i])));return minimum,mean,feasible,best
    minimum,mean,feasible,best=objective(curve);dm,df,dfeasible,db=objective(direct)
    require(np.array_equal(minimum,dm) and np.array_equal(mean,df) and np.array_equal(feasible,dfeasible) and best==db,'Full-grid objective/feasibility/argmax differs.')
    return float(grid[best]),dict(grid_thresholds=len(grid),groups=11,primary_statistics=len(grid)*33,additional_F1_FPR_statistics=len(grid)*22,
        maximum_direct_difference=0.,actual_F32_comparison=True,all_grid_defined_semantics_equal=True,all_grid_feasibility_equal=True,global_argmax_equal=True,
        min33=float(minimum[best]),old5_mean_f1=float(mean[best]),feasible_thresholds=int(feasible.sum()))


def guard(result,control,legacy=False):
    tol=1e-12;reasons=[];f1={g:control['groups'][g]['system_f1']-result['groups'][g]['system_f1'] for g in OLD5[:4]}
    accuracy={g:control['groups'][g]['accuracy']-result['groups'][g]['accuracy'] for g in OLD5[:4]}
    gains={g:result['groups'][g]['system_f1']-control['groups'][g]['system_f1'] for g in FULL};meanloss=control['old5_mean_f1']-result['old5_mean_f1']
    if max(f1.values())>.02+tol:reasons.append('old4_system_f1_loss')
    if max(accuracy.values())>.01+tol:reasons.append('old4_five_accuracy_loss')
    if meanloss>.005+tol:reasons.append('old5_mean_f1_loss')
    if any(v< -tol for v in gains.values()):reasons.append('C4_full_tail_f1_regression')
    pubf1={g:control['groups'][g]['system_f1']-result['groups'][g]['system_f1'] for g in PUBLIC};pubacc={g:control['groups'][g]['accuracy']-result['groups'][g]['accuracy'] for g in PUBLIC}
    if not legacy:
        if max(pubf1.values())>.02+tol:reasons.append('public_system_f1_loss')
        if max(pubacc.values())>.01+tol:reasons.append('public_five_accuracy_loss')
    if any(result['groups'][g]['system_false_positive_rate']>.05+tol for g in FEASIBLE):reasons.append('FPR_above_limit')
    if not all(np.isfinite(result['groups'][g][k]) for g in GROUPS for k in ['system_f1','accuracy','system_false_positive_rate']):reasons.append('nonfinite_metrics')
    return dict(eligible=not reasons,rejection_reasons=reasons,old4_f1_losses=f1,old4_accuracy_losses=accuracy,old5_mean_f1_loss=meanloss,
        C4_full_tail_f1_gains=gains,public_f1_losses=pubf1,public_accuracy_losses=pubacc,legacy_savedGPU_only=legacy)


def select_records(records,legacy):
    require(len(records)==8 and {r['record'] for r in records}==set(RECORDS),'Exactly eight records required.')
    index={r['record']:r for r in records};candidate=index['new-max33']['validation'];reasons=[];guards={};gains={};minimum=candidate['min33']
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARD_CONTROLS:
        control=index[name]['validation'];g=guard(candidate,control);guards[name]=g;reasons += [name+':'+r for r in g['rejection_reasons']]
        if name in GUARD_CONTROLS[:3]:
            gain=minimum-control['min33'] if minimum is not None and control['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    saved=guard(candidate,legacy,True);guards['original-savedGPU']=saved;reasons += ['original-savedGPU:'+r for r in saved['rejection_reasons']]
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,
        primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def ppath(model,bank):return CACHE/f'{model}-{bank}-validation.npz'
def read_predictions(model,record,rows):return {b:aug.read_probability(ppath(model,b),record['probability_sha256'][b],frame) for b,frame in rows.items()}
def budget(history,best):return aug.check_history(history,best,steps=STEPS,instances=INSTANCES)


def parity(model,banks,nf):
    require(len(nf)>=256 and nf.split.eq('fit').all(),'Parity must use clean FIT rows only.')
    chosen=nf.iloc[:256];indices=chosen.cache_ix.to_numpy();cpu=hp.predict_raw(model,banks['natural'],indices);gpu=hp.predict_raw(model,banks['natural'],indices,device='cuda:0')
    error=float(np.abs(cpu-gpu).max());require(np.isfinite(error) and 0<=error<5e-4,'FIT-only same-vector portable parity failed.')
    return dict(fit_only=True,validation_probabilities_computed=False,sampled_tokens=256,target_cache_ix_sha256=array_sha(indices),
        sample_ix_sha256=array_sha(chosen.sample_ix.to_numpy()),maximum_probability_difference=error)


def verify_parity(proof,nf):
    require(len(nf)>=256 and nf.split.eq('fit').all(),'FIT parity scope changed.')
    require(proof['fit_only'] is True and proof['validation_probabilities_computed'] is False and proof['sampled_tokens']==256
        and proof['target_cache_ix_sha256']==array_sha(nf.cache_ix.to_numpy()[:256]) and proof['sample_ix_sha256']==array_sha(nf.sample_ix.to_numpy()[:256])
        and np.isfinite(proof['maximum_probability_difference']) and 0<=proof['maximum_probability_difference']<5e-4,'Portable FIT-only proof changed.')


def supervisor_alive():require(os.environ.get('ROLE_NATURAL_FIT_SUPERVISOR_PID')==str(os.getppid()),'Worker needs original live parent lease.')


def worker(arm,gpu):
    supervisor_alive();require(arm in ARMS and gpu in [0,1],'Only two declared arm workers allowed.')
    require(not (OUT/'complete.json').exists() and not (OUT/'candidates'/f'{arm}.json').exists(),'Frozen candidate cannot be overwritten.')
    no_competing_jobs(True);plan=read(OUT/'search-plan.json');verify_plan(plan);ms.configure_torch()
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1 and 'A40' in torch.cuda.get_device_name(0),'Worker requires its one assigned A40.')
    banks,fit,ov,cv,nf,pv,_=dataset();mean,scale=aug.read_scaler();slots=pd.read_parquet(CACHE/'matched-draw.parquet')
    def standardized(bank,indices):
        out=torch.empty((len(indices),512),dtype=torch.float32,device='cuda')
        for start in range(0,len(indices),BATCH):
            x=(np.asarray(banks[bank][indices[start:start+BATCH]],np.float32)-mean)/scale
            require(np.isfinite(x).all(),'Nonfinite standardized feature.');out[start:start+len(x)]=torch.from_numpy(x).cuda()
        return out
    require(torch.cuda.mem_get_info(0)[0]>(INSTANCES+OLD_VALID)*512*4+2*2**30,'Insufficient assigned worker memory.')
    old_x=standardized('old',fit.cache_ix.to_numpy());extra_x=standardized('natural' if arm=='new' else 'old',nf.cache_ix.to_numpy() if arm=='new' else slots.matched_old_cache_ix.to_numpy())
    x=torch.cat([old_x,extra_x]);del old_x,extra_x
    y=torch.from_numpy(np.r_[probes.labels(fit),probes.labels(nf)]).cuda();xv=standardized('old',ov.cache_ix.to_numpy())
    torch.manual_seed(20261021);net=hp.make_net('mlp',hidden=128).cuda();init=aug.init_hash(net);require(init==plan['initial_weight_sha256'],'Arms initialization differs.')
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);generator=torch.Generator(device='cuda:0').manual_seed(20261021)
    path=OUT/'models'/f'{arm}.npz';gpu_path=CACHE/f'{arm}-gpu-old-validation.npz';history=[];best=-1.;best_epoch=-1
    meta=dict(arm=arm,kind='mlp',hidden=128,dimensions=512,heads=8,head_dim=64,roles=data.ROLES,selected_head_ids=plan['signature']['selected_head_ids'],
        signature=plan['signature'],source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),initial_weight_sha256=init,
        standardization_sha256=plan['signature']['standardization_sha256'],tests_read=False)
    for epoch in range(1,46):
        supervisor_alive();net.train();order=lp.epoch_order(len(x),generator,'cuda');loss_sum=0.;steps=0
        require(len(order)==INSTANCES,'Epoch instance budget changed.')
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH];loss=torch.nn.functional.cross_entropy(net.raw(x[ix].contiguous()),y[ix]);require(torch.isfinite(loss),'Nonfinite CE.')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();loss_sum+=float(loss.detach())*len(ix);steps+=1
        require(steps==STEPS,'Epoch update budget changed.')
        p=probes.predict_standardized(net,xv).astype(np.float32);value=lp.validation_metrics(ov,p);score=value['mean_system_f1']
        history.append(dict(epoch=epoch,loss=loss_sum/len(order),instances=len(order),steps=steps,cumulative_steps=epoch*STEPS,old5_mean_f1=score))
        if score>best+1e-5:
            best,best_epoch=score,epoch;hp.save_model(path,net,mean,scale,meta|dict(best_epoch=epoch,best_steps=epoch*STEPS));hp.atomic_npz(gpu_path,p=p,sample_ix=ov.sample_ix.to_numpy())
        print(json.dumps(dict(arm=arm,**history[-1],best_epoch=best_epoch)),flush=True)
        if epoch>=10 and epoch-best_epoch>=8:break
    amounts=budget(history,best_epoch);supervisor_alive()
    training=dict(arm=arm,signature=plan['signature'],plan_sha256=data.digest(OUT/'search-plan.json'),initial_weight_sha256=init,history=history,**amounts,
        model_sha256=data.digest(path),gpu_old_probability_sha256=data.digest(gpu_path),new_C4_predictions_read=False,public_validation_predictions_read=False)
    data.atomic_json(OUT/'training'/f'{arm}.json',training)
    rows=dict(old=ov,c4=cv,public=pv);physical_banks=dict(old='old',c4='c4',public='natural');p={}
    # The preceding best/history/model/GPUoldp freeze is immutable before both new validation cohorts.
    for b,frame in rows.items():
        p[b]=hp.predict_raw(path,banks[physical_banks[b]],frame.cache_ix.to_numpy());hp.atomic_npz(ppath(arm,b),p=p[b],sample_ix=frame.sample_ix.to_numpy())
    gpu_p=aug.read_probability(gpu_path,training['gpu_old_probability_sha256'],ov);difference=float(np.abs(gpu_p-p['old']).max())
    require(np.isfinite(difference) and difference<5e-4,'Full candidate GPUold/CPUold portable drift failed.')
    proof=parity(path,banks,nf);verify_plan(plan);supervisor_alive()
    data.atomic_json(OUT/'candidates'/f'{arm}.json',dict(arm=arm,signature=plan['signature'],source_sha256=data.digest(__file__),
        plan_sha256=data.digest(OUT/'search-plan.json'),initial_weight_sha256=init,model_sha256=training['model_sha256'],training_sha256=data.digest(OUT/'training'/f'{arm}.json'),
        probability_sha256={b:data.digest(ppath(arm,b)) for b in rows},gpu_old_probability_sha256=training['gpu_old_probability_sha256'],
        old_early_stop_validation=lp.validation_metrics(ov,gpu_p),full_old_GPU_CPU_max_difference=difference,portable_evidence=proof,**amounts,
        tests_read=False,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_name=torch.cuda.get_device_name(0)))


def candidate_record(arm,plan,rows,nf):
    record=read(OUT/'candidates'/f'{arm}.json');ep=OUT/'training'/f'{arm}.json';hp.require_hash(ep,record['training_sha256'],'pre-new-validation freeze');e=read(ep)
    amounts=budget(e['history'],e['best_epoch']);path=OUT/'models'/f'{arm}.npz'
    require(record['arm']==e['arm']==arm and record['signature']==e['signature']==plan['signature'] and record['source_sha256']==data.digest(__file__)
        and record['plan_sha256']==e['plan_sha256']==data.digest(OUT/'search-plan.json') and record['initial_weight_sha256']==e['initial_weight_sha256']==plan['initial_weight_sha256']
        and record['model_sha256']==e['model_sha256'] and record['tests_read'] is False
        and e['new_C4_predictions_read'] is False and e['public_validation_predictions_read'] is False
        and all(record[k]==e[k]==v for k,v in amounts.items()),'Arm checkpoint/history/source evidence changed.')
    hp.require_hash(path,record['model_sha256'],'portable candidate');_,mean,scale,meta=hp.load_model(path);m,s=aug.read_scaler()
    require(meta['kind']=='mlp' and meta['hidden']==128 and meta['dimensions']==512 and meta['arm']==arm and meta['signature']==plan['signature']
        and meta['initial_weight_sha256']==plan['initial_weight_sha256'] and meta['source_sha256']==data.digest(__file__)
        and meta['plan_sha256']==data.digest(OUT/'search-plan.json') and meta['selected_head_ids']==plan['signature']['selected_head_ids']
        and meta['standardization_sha256']==plan['signature']['standardization_sha256'] and meta['best_epoch']==e['best_epoch'] and meta['best_steps']==e['best_steps']
        and np.array_equal(mean.numpy(),m) and np.array_equal(scale.numpy(),s),'Portable candidate/scaler provenance changed.')
    p=read_predictions(arm,record,rows);gpu_p=aug.read_probability(CACHE/f'{arm}-gpu-old-validation.npz',record['gpu_old_probability_sha256'],rows['old'])
    value=lp.validation_metrics(rows['old'],gpu_p);error=float(np.abs(gpu_p-p['old']).max())
    require(record['gpu_old_probability_sha256']==e['gpu_old_probability_sha256'] and value==record['old_early_stop_validation']
        and value['mean_system_f1']==e['history'][e['best_epoch']-1]['old5_mean_f1'] and error==record['full_old_GPU_CPU_max_difference'] and 0<=error<5e-4,
        'Best model/GPUoldp/CPU drift evidence changed.')
    verify_parity(record['portable_evidence'],nf);require(record['cuda_visible_devices'] in ['0','1'] and 'A40' in record['cuda_device_name'],'Worker GPU assignment changed.')
    return record,p


def inherited_controls(plan,banks,rows,nf,allow_create=False):
    ms.configure_torch();declaration=verify_plan(plan,False);static=declaration['static'];path=OUT/'controls.json'
    reference={b:aug.read_probability(static['old_reference'][key+'_path'],static['old_reference'][key+'_sha256'],rows[b]) for b,key in [('old','old'),('c4','new')]}
    ensemble_reference={b:aug.read_probability(static['ensemble_reference'][key+'_path'],static['ensemble_reference'][key+'_sha256'],rows[b]) for b,key in [('old','old'),('c4','new')]}
    original_candidate=read(hp.OUT/'candidates/joint-mlp128.json');require(lp.validation_metrics(rows['old'],reference['old'])==original_candidate['validation']
        and original_candidate['validation']['system_threshold']==ORIGINAL_TAU,'Original savedGPU oldp/threshold no longer reproduce exactly.')
    exists=path.exists();record=read(path) if exists else None;member={};member_records={}
    if exists:require(record['signature']==plan['signature'] and record['model_sha256']==static['model_sha256'],'Inherited control closure changed.')
    for seed in MEMBERS:
        name='seed-'+str(seed);model=Path(static['model_paths'][str(seed)])
        if exists:
            r=record['members'][str(seed)];p=read_predictions(name,r,rows);verify_parity(r['portable_evidence'],nf)
        else:
            require(allow_create and not any(ppath(name,b).exists() for b in rows),'Missing/unowned inherited p: read-only verification cannot recompute.')
            _,mean,scale,meta=hp.load_model(model);m,s=aug.read_scaler()
            require(np.array_equal(mean.numpy(),m) and np.array_equal(scale.numpy(),s) and meta['selected_head_ids']==plan['signature']['selected_head_ids'],'Inherited scaler/head order changed.')
            p={b:hp.predict_raw(model,banks['natural' if b=='public' else b],frame.cache_ix.to_numpy()) for b,frame in rows.items()}
            proof=parity(model,banks,nf);r=dict(model_sha256=static['model_sha256'][str(seed)],portable_evidence=proof,probability_sha256={})
            for b,frame in rows.items():hp.atomic_npz(ppath(name,b),p=p[b],sample_ix=frame.sample_ix.to_numpy());r['probability_sha256'][b]=data.digest(ppath(name,b))
        require(r['model_sha256']==static['model_sha256'][str(seed)],'Inherited physical model differs.');member[seed]=p;member_records[str(seed)]=r
    old=member[20261021];ensemble={b:ms.fixed_mean([member[s][b] for s in MEMBERS]) for b in rows}
    for b,frame in rows.items():
        target=ppath('ensemble',b)
        if exists:
            p=aug.read_probability(target,record['ensemble_probability_sha256'][b],frame);require(np.array_equal(p,ensemble[b]),'Saved inherited ensemble differs from fixed four-member mean.')
        else:
            require(not target.exists(),'Unowned partial inherited ensemble.');hp.atomic_npz(target,p=ensemble[b],sample_ix=frame.sample_ix.to_numpy())
    def drift(ref,p,tau):return ms.drift_metrics(rows['old'],rows['c4'],dict(old=ref['old'],new=ref['c4']),dict(old=p['old'],new=p['c4']),tau)
    legacy=dict(old=reference['old'],c4=reference['c4'],public=old['public']);legacy_metrics=metrics_at(rows,legacy,ORIGINAL_TAU)
    content=dict(signature=plan['signature'],model_sha256=static['model_sha256'],members=member_records,member_order=MEMBERS,weights=[.25]*4,
        ensemble_probability_sha256={b:data.digest(ppath('ensemble',b)) for b in rows},oldhead_drift=drift(reference,old,ORIGINAL_TAU),
        ensemble_drift=drift(ensemble_reference,ensemble,ENSEMBLE_TAU),original_savedGPU_validation=legacy_metrics,
        original_savedGPU_reference=static['old_reference'],inherited_ensemble_reference=static['ensemble_reference'],
        public_saved_reference='No historical public64 p; original frozen oldhead commonCPU public p is used.',tests_read=False)
    if exists:require(record==content,'Inherited controls/FIT parity/drift/legacy GPU floors changed.')
    else:data.atomic_json(path,content)
    return old,ensemble,legacy_metrics


def calibrate(plan,allow_create=False):
    ms.configure_torch();banks,fit,ov,cv,nf,pv,_=dataset();rows=dict(old=ov,c4=cv,public=pv);p={}
    for arm in ARMS:_,p[arm]=candidate_record(arm,plan,rows,nf)
    p['oldhead'],p['ensemble'],legacy=inherited_controls(plan,banks,rows,nf,allow_create)
    records=[]
    for name in RECORDS:
        model,objective=name.split('-');value=p[model]
        if objective=='max33':tau,proof=max33(rows,value)
        elif model=='oldhead':tau,proof=ORIGINAL_TAU,None
        elif model=='ensemble':tau,proof=ENSEMBLE_TAU,None
        else:tau,proof=lp.validation_metrics(ov,value['old'])['system_threshold'],None
        result=metrics_at(rows,value,tau)
        if proof is not None:require(result['min33']==proof['min33'] and result['old5_mean_f1']==proof['old5_mean_f1'],'Max33 proof differs from actual selected metrics.')
        records.append(dict(record=name,model=model,objective=objective,validation=result,full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False))
    return records,legacy,rows,fit,nf


def publish(plan):
    records,legacy,rows,fit,nf=calibrate(plan,True);selection=select_records(records,legacy);data.atomic_json(OUT/'records.json',records)
    flat=[]
    for record in records:
        for group,v in dict(record['validation']['groups'],pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=record['record'],group=group,**{k:x for k,x in v.items() if not isinstance(x,(dict,list))},**{'role_recall_'+r:x for r,x in v['recall'].items()}))
    require(len(flat)==96,'8record/11group+pooled CSV coverage changed.');pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False)
    counts=dict(old_fit=len(fit),natural_fit=len(nf),per_epoch=INSTANCES,steps=STEPS,last_batch=2517,
        validation_rows={b:len(frame) for b,frame in rows.items()},natural_fit_source_roots=int(nf.source_uid.nunique()),public_source_roots=int(rows['public'].source_uid.nunique()),
        natural_fit_system_families=int(nf.family_id.nunique()),natural_fit_dependency_components=int(nf.dependency_component_id.nunique()),
        public_system_families=int(rows['public'].family_id.nunique()),public_dependency_components=int(rows['public'].dependency_component_id.nunique()),
        public_dataset_source_roots={d:int(rows['public'].loc[rows['public'].dataset.eq(d),'source_uid'].nunique()) for d in ['oasst','toxicchat0124']},
        natural_fit_role_support={r:int(nf.role.eq(r).sum()) for r in data.ROLES},public_role_support={r:int(rows['public'].role.eq(r).sum()) for r in data.ROLES},
        legacy_natural_role_support={r:int((rows['old'].profile.eq('natural')&rows['old'].role.eq(r)).sum()) for r in data.ROLES},
        reused_development_only=True,tails_and_public_source_groups_overlap=True,limitation=LIMITATION)
    data.atomic_json(OUT/'source-counts.json',counts)
    lines=['Natural fit versus role-slot matched old-fit: development only',LIMITATION,'','Record threshold min33 old5meanF1']
    lines += [f"{r['record']} {r['validation']['system_threshold']:.12g} {r['validation']['min33']} {r['validation']['old5_mean_f1']:.9f}" for r in records]
    lines += ['', 'Promoted: '+str(selection['promoted']),'Reasons: '+json.dumps(selection['rejection_reasons'])];(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    selection.update(source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),records_sha256=data.digest(OUT/'records.json'),
        model_path=str(OUT/'models/new.npz'),model_sha256=read(OUT/'candidates/new.json')['model_sha256'],controls_sha256=data.digest(OUT/'controls.json'),
        system_threshold=next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'))
    data.atomic_json(OUT/'selection.json',selection);verify_plan(plan)
    require(all(p.stat().st_size<50*2**20 for folder in [OUT,CACHE] for p in folder.rglob('*') if p.is_file()),'New artifact exceeds50MiB storage bound.')
    data.atomic_json(OUT/'complete.json',dict(signature=plan['signature'],source_sha256=data.digest(__file__),artifact_sha256={n:data.digest(OUT/n) for n in ARTIFACTS},
        candidate_sha256={a:data.digest(OUT/'candidates'/f'{a}.json') for a in ARMS},tests_read=False,primary_changed=False,new_independent_confirmation_required=True))
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',promoted=selection['promoted'],primary_changed=False))


def verified_completion():
    require((OUT/'complete.json').is_file() and read(OUT/'status.json')['stage']=='validation_complete','Natural-fit trial is not terminal complete.')
    plan=read(OUT/'search-plan.json');verify_plan(plan);complete=read(OUT/'complete.json')
    require(complete['source_sha256']==data.digest(__file__) and complete['signature']==plan['signature'] and complete['tests_read'] is False
        and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and set(complete['artifact_sha256'])==set(ARTIFACTS) and set(complete['candidate_sha256'])==set(ARMS),'Completion source/coverage changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(OUT/n,h,'completed artifact')
    for a,h in complete['candidate_sha256'].items():hp.require_hash(OUT/'candidates'/f'{a}.json',h,'completed candidate')
    records,legacy,*_=calibrate(plan,False);require(records==read(OUT/'records.json'),'Eight records do not reproduce from saved p.')
    expected=select_records(records,legacy);actual=read(OUT/'selection.json');require(all(actual[k]==v for k,v in expected.items()),'Absolute90/no-fallback/mandatory GPU gate changed.')
    require(actual['source_sha256']==data.digest(__file__) and actual['plan_sha256']==data.digest(OUT/'search-plan.json') and actual['records_sha256']==data.digest(OUT/'records.json')
        and actual['controls_sha256']==data.digest(OUT/'controls.json') and actual['model_path']==str(OUT/'models/new.npz')
        and actual['model_sha256']==read(OUT/'candidates/new.json')['model_sha256']
        and actual['system_threshold']==next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'),'Selected model/threshold closure changed.')
    return complete


def no_competing_jobs(allow_own=False):
    ms.no_competing_jobs(True)
    names={'systemness_multiseed_ensemble.py','systemness_natural_dev_extract.py','systemness_natural_transcripts.py'}
    if not allow_own:names.add(Path(__file__).name)
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name in names for a in args if a):active.append(int(entry.name))
    require(not active,'Competing natural-dev capture/training process live: '+str(active))


@contextmanager
def run_lock():
    import fcntl
    with (OUT/'.run.lock').open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another natural-fit supervisor owns the output.') from e
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run_workers(pending):
    require(len(pending)<=2 and len(set(pending))==len(pending) and set(pending)<=set(ARMS),'Only two fixed arm workers allowed.');jobs=[]
    try:
        for gpu,arm in enumerate(pending):
            log=(OUT/'logs'/f'{arm}.log').open('a')
            try:p=subprocess.Popen([sys.executable,'-u',__file__,'--train','--worker',arm,'--gpu',str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ROLE_NATURAL_FIT_SUPERVISOR_PID=str(os.getpid())),stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            jobs.append((arm,p,log,gpu))
        data.atomic_json(OUT/'status.json',dict(stage='training',workers=[dict(arm=a,pid=p.pid,gpu=g) for a,p,_,g in jobs]))
        while jobs:
            codes=[(a,p.poll()) for a,p,_,_ in jobs];require(not [(a,c) for a,c in codes if c not in [None,0]],'Natural-fit worker failed: '+str(codes))
            if all(c==0 for _,c in codes):break
            time.sleep(5)
        for arm,p,_,_ in jobs:require(p.wait()==0,arm+' failed.')
    finally:
        errors=[]
        for _,p,_,_ in jobs:
            try:
                if p.poll() is None:p.terminate()
            except BaseException as e:errors.append(repr(e))
        for _,p,log,_ in jobs:
            try:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            except BaseException as e:errors.append(repr(e))
            finally:log.close()
        require(not errors,'Worker terminate/reap failed: '+repr(errors))


def train(restart_incomplete=False):
    ms.configure_torch();no_competing_jobs();verify_declaration();signature=source_signature()
    with run_lock():
        if (OUT/'complete.json').exists():verified_completion();return
        require(not (OUT/'selection.json').exists(),'Partial frozen selection exists; refusing replacement.')
        for n in ['models','training','candidates','logs']:(OUT/n).mkdir(exist_ok=True)
        path=OUT/'search-plan.json'
        if path.exists():plan=read(path);verify_plan(plan)
        else:
            require({p.name for p in CACHE.iterdir()}=={'matched-draw.parquet'} and all(not any((OUT/n).iterdir()) for n in ['models','training','candidates']),'Unowned candidate/model/probabilities predate plan.')
            plan=dict(protocol=protocol(),signature=signature,initial_weight_sha256=read(OUT/'protocol.json')['static']['initial_weight_sha256']);data.atomic_json(path,plan)
        _,_,ov,cv,nf,pv,_=dataset();rows=dict(old=ov,c4=cv,public=pv);pending=[]
        for arm in ARMS:
            if (OUT/'candidates'/f'{arm}.json').exists():candidate_record(arm,plan,rows,nf)
            else:
                partial=[OUT/'models'/f'{arm}.npz',OUT/'training'/f'{arm}.json',CACHE/f'{arm}-gpu-old-validation.npz']+[ppath(arm,b) for b in rows]
                require(restart_incomplete or not any(p.exists() for p in partial),'Partial candidate requires explicit --restart-incomplete; no progress trusted.')
                if any(p.exists() for p in partial):data.atomic_json(OUT/f'restart-{arm}-{time.time_ns()}.json',dict(arm=arm,from_epoch=1,signature=signature,partial_sha256={str(p):data.digest(p) for p in partial if p.exists()}))
                pending.append(arm)
        if pending:run_workers(pending)
        publish(plan)


def preflight():
    from unittest.mock import patch
    from types import SimpleNamespace
    import copy,py_compile,torch
    torch.set_num_threads(2);folder=Path(tempfile.mkdtemp(prefix='systemness-natural-fit-preflight-',dir='/tmp'));source=data.digest(__file__);checks=[];module=sys.modules[__name__]
    def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError,AssertionError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('CPU fault accepted: '+name)
    def forbidden(*a,**k):raise AssertionError('Formal data/probability/model API invoked in synthetic preflight.')
    with patch.object(aug,'dataset',side_effect=forbidden),patch.object(aug,'source_signature',side_effect=forbidden),patch.object(data.runtime,'load_model',side_effect=forbidden),patch.object(natural,'verified_extraction',side_effect=forbidden):
        torch.set_float32_matmul_precision('medium');ms.configure_torch();check('Common4threads/highest portable settings',torch.get_num_threads()==4 and torch.get_float32_matmul_precision()=='highest')
        # Real63403 slot label geometry, a tiny original-fit sampling pool, no feature bank.
        fit=pd.DataFrame([dict(cache_ix=i,sample_ix=i,role=data.ROLES[i%5],split='fit') for i in range(500)])
        labels=np.concatenate([np.repeat(role,count) for role,count in ROLE_COUNTS['fit'].items()])
        nf=pd.DataFrame(dict(cache_ix=np.arange(EXTRA),sample_ix=4_000_000_000_000+np.arange(EXTRA),source_uid=['fit-'+str(i%255) for i in range(EXTRA)],role=labels,split='fit'))
        state=torch.get_rng_state().clone();slots,summary=matched_draw(fit,nf);again,summary2=matched_draw(fit,nf)
        check('63403 same-role deterministic replacement slots without Torch RNG',slots.equals(again) and summary==summary2 and torch.equal(state,torch.get_rng_state())
            and np.array_equal(fit.set_index('cache_ix').loc[slots.matched_old_cache_ix].role,nf.role) and summary['role_counts']==ROLE_COUNTS['fit'])
        check('Extra slots draw only original fit and repeat by design',set(slots.matched_old_cache_ix)<=set(fit.cache_ix) and slots.matched_old_cache_ix.duplicated().any())
        contaminated=fit.copy();contaminated.loc[0,'split']='validation';reject('Matched sampler refuses validation pool row',lambda:matched_draw(contaminated,nf))
        slots.to_parquet(folder/'draw.parquet',index=False);check('Compact exact Parquet draw safely below50MiB',pd.read_parquet(folder/'draw.parquet').equals(slots) and (folder/'draw.parquet').stat().st_size<50*2**20)
        g1=torch.Generator().manual_seed(20261021);g2=torch.Generator().manual_seed(20261021);o1=lp.epoch_order(INSTANCES,g1,'cpu');o2=lp.epoch_order(INSTANCES,g2,'cpu')
        check('641493 shared permutation /157updates /last2517',torch.equal(o1,o2) and torch.unique(o1).numel()==INSTANCES and len(range(0,INSTANCES,BATCH))==157 and INSTANCES%4096==2517)
        check('Both arm initial weight hashes identical under original seed',ms.initial_hash(20261021)==ms.initial_hash(20261021))
        history=[dict(epoch=i,instances=INSTANCES,steps=STEPS,cumulative_steps=i*STEPS,loss=.5,old5_mean_f1=.8) for i in range(1,11)]
        check('Original earlystop157step history best1 actual10',budget(history,1)['best_steps']==157 and budget(history,1)['actual_steps']==1570)
        broken=copy.deepcopy(history);broken[-1]['steps']=142;reject('Wrong per-epoch budget refused',lambda:budget(broken,1))
        rows={};old=[];c4=[];public=[]
        for pi,profile in enumerate(lp.PROFILES):
            for ri,role in enumerate(data.ROLES if profile!='natural' else data.ROLES[:-1]):
                for off in [0,512,1023]:old.append(dict(cache_ix=len(old),sample_ix=len(old),profile=profile,role=role,token_ix=13+off,body_offset=off,prompt_ix=pi*5+ri,message_ix=0,document_id=pi))
        for ri,role in enumerate(data.ROLES):
            for off in [0,512,1023]:c4.append(dict(cache_ix=len(c4),sample_ix=6_000_000_000_000+len(c4),profile='long-full',role=role,token_ix=13+off,body_offset=off,prompt_ix=ri,message_ix=0,document_id=6000000))
        for dataset_name in ['oasst','toxicchat0124']:
            for role in ['system','user','assistant']:
                for off in range(4):public.append(dict(cache_ix=len(public),sample_ix=4_000_000_100_000+len(public),dataset=dataset_name,role=role,token_ix=off,profile='public-natural',document_id=dataset_name))
        rows=dict(old=pd.DataFrame(old),c4=pd.DataFrame(c4),public=pd.DataFrame(public));p={}
        for b,frame in rows.items():
            value=np.full((len(frame),5),.025,np.float32);value[np.arange(len(frame)),probes.labels(frame)]=.9;p[b]=value
        tau,proof=max33(rows,p);value=metrics_at(rows,p,tau)
        check('11groups33 actualF32 full-grid/direct-proof including public OA/TC',proof['groups']==11 and proof['primary_statistics']==proof['grid_thresholds']*33
            and proof['global_argmax_equal'] and value['min33']==proof['min33']==1. and set(value['groups'])==set(GROUPS))
        check('Public SUA/legacy CoT supports kept distinct',value['groups']['old-natural']['recall']['cot']==1. and value['groups']['old-natural']['recall']['tool'] is None
            and all(value['groups'][g]['recall']['cot'] is None and value['groups'][g]['recall']['tool'] is None for g in PUBLIC))
        check('Old/new/public descriptive banks retained separately',set(value['separate_bank_descriptive'])=={'old','c4','public'} and value['pooled_descriptive_only']['tokens']==sum(len(f) for f in rows.values()))
        def v(n):return dict(min33=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in GROUPS})
        records=[dict(record=name,validation=v(.94 if name=='new-max33' else .92)) for name in RECORDS];legacy=v(.92)
        check('Only new-max33 advances against all controls plus original savedGPU',select_records(records,legacy)['promoted'])
        bad=copy.deepcopy(records);next(r for r in bad if r['record']=='new-max33')['validation']['min33']=.899999999999
        check('Absolute90 hard floor is strict even for near90 candidate',not select_records(bad,legacy)['promoted'])
        for control in GUARD_CONTROLS[:3]:
            bad=copy.deepcopy(records);next(r for r in bad if r['record']==control)['validation']['min33']=.94
            check('Strict max33 tie rejected against '+control,not select_records(bad,legacy)['promoted'])
        high=copy.deepcopy(legacy);high['groups']['old-natural']['system_f1']=.99
        check('Mandatory original savedGPU floor rejects apparent CPU-only gain',not select_records(records,high)['promoted'])
        for group,key,val in [(PUBLIC[1],'system_f1',.88),(PUBLIC[2],'accuracy',.89),(FULL[-1],'system_f1',.91)]:
            bad=copy.deepcopy(records);next(r for r in bad if r['record']=='new-max33')['validation']['groups'][group][key]=val
            check('Protected public/C4 floor '+group+'/'+key,not select_records(bad,legacy)['promoted'])
        proof_fit=dict(fit_only=True,validation_probabilities_computed=False,sampled_tokens=256,target_cache_ix_sha256=array_sha(nf.cache_ix.to_numpy()[:256]),
            sample_ix_sha256=array_sha(nf.sample_ix.to_numpy()[:256]),maximum_probability_difference=1e-7)
        verify_parity(proof_fit,nf);check('FIT-only fixed256 parity scope valid')
        bad=copy.deepcopy(proof_fit);bad['validation_probabilities_computed']=True;reject('Validation parity evidence explicitly refused',lambda:verify_parity(bad,nf))
        with patch.object(module,'OUT',folder/'undeclared'),patch.object(module,'static_snapshot',side_effect=forbidden):reject('Declaration gate precedes formal metadata/p APIs',verify_declaration)
        # Actual declaration code, but every metadata supplier/path is synthetic.
        fake_design=folder/'design.json';data.atomic_json(fake_design,dict(synthetic=True));fake_natural=SimpleNamespace(OUT=folder/'features')
        fake_natural.OUT.mkdir();data.atomic_json(fake_natural.OUT/'extraction-complete.json',dict(synthetic=True))
        with patch.object(module,'OUT',folder/'declared'),patch.object(module,'CACHE',folder/'cache'),patch.object(module,'DESIGN',fake_design),patch.object(module,'DESIGN_SHA',data.digest(fake_design)),\
                patch.object(module,'natural',fake_natural),patch.object(module,'layout',return_value=(fit,None,None,nf,None,{})),patch.object(module,'static_snapshot',return_value=dict(synthetic=True)):
            declaration=declare();check('Metadata-only declaration writes compact manifest before protocol',read(OUT/'protocol.json')==declaration and (CACHE/'matched-draw.parquet').is_file()
                and data.digest(CACHE/'matched-draw.parquet')==read(OUT/'draw-manifest.json')['slots_sha256'])
            check('Same declaration restart is immutable',declare()==declaration)
        small=folder/'p.npz';hp.atomic_npz(small,p=p['old'],sample_ix=rows['old'].sample_ix.to_numpy())
        check('Probability SHA/sample identity roundtrip',np.array_equal(aug.read_probability(small,data.digest(small),rows['old']),p['old']))
        wrong=rows['old'].copy();wrong.sample_ix+=1;reject('Changed probability sample order refused',lambda:aug.read_probability(small,data.digest(small),wrong))
    py_compile.compile(__file__,cfile=str(folder/'runner.pyc'),doraise=True);require(data.digest(__file__)==source,'Preflight altered source.')
    report=dict(source_sha256=source,checks=checks,checks_passed=len(checks),records=8,groups=11,primary_statistics=33,
        matched_manifest_rows=EXTRA,matched_manifest_bytes=(folder/'draw.parquet').stat().st_size,max_additional_updates=14130,
        formal_model_or_probabilities_read=False,actual_banks_or_test_data_read=False,gpu_used=False,formal_outputs_written=False,fixture_directory=str(folder))
    data.atomic_json(folder/'report.json',report);print(json.dumps(dict(report=str(folder/'report.json'),report_sha256=data.digest(folder/'report.json'),**report)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--declare',action='store_true');a.add_argument('--train',action='store_true');a.add_argument('--preflight',action='store_true')
    p.add_argument('--worker',choices=ARMS,help=argparse.SUPPRESS);p.add_argument('--gpu',type=int,help=argparse.SUPPRESS);p.add_argument('--restart-incomplete',action='store_true');args=p.parse_args()
    if args.worker is not None:require(args.train and args.gpu is not None,'Worker needs --train and assigned GPU.');worker(args.worker,args.gpu)
    elif args.preflight:preflight()
    elif args.declare:declare()
    elif args.train:train(args.restart_incomplete)
    else:p.print_help()

if __name__=='__main__':main()
