"""Bounded three-new-seed / four-member MLP128 probability ensemble.

Default help. --declare freezes metadata before candidate probabilities; --train
is explicit. --preflight is synthetic CPU/tmp only. Exactly ensemble-maximin
may advance; singletons remain diagnostic. No test, LM, new feature or primary
update is run. Old-natural is legacy generated data with observed CoT truth.
"""
import argparse
from contextlib import contextmanager
import hashlib
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

OUT=data.runtime.OUT/'systemness-multiseed-ensemble'
CACHE=data.CACHE/'multiseed-ensemble'
DESIGN=data.runtime.OUT/'systemness-multiseed-ensemble-preparation/root-prospective-design.json'
DESIGN_SHA='a1ad395c345ec8af91d5782987a95f0b9cd9034da92b7f40debc2dce1ec521e6'
SEEDS=[20261022,20261023,20261024]
MEMBERS=[20261021]+SEEDS
RECORDS=['old-original','old-cpu-maximin']+[f'seed-{s}-maximin' for s in SEEDS]+['ensemble-maximin']
BATCH,STEPS,FIT_ROWS,OLD_VALID,NEW_VALID=4096,142,578090,173032,61440
ARTIFACTS=['protocol.json','search-plan.json','old-control.json','ensemble.json','records.json','selection.json','validation-metrics.csv','report.txt','source-counts.json']
DEPENDENCIES=sorted(set(aug.DEPENDENCIES+['systemness_multiseed_ensemble.py']))
require,read,array_sha=hp.require,hp.read,hp.array_digest
LIMITATION=('Identical fit rows across seeds are shared data, not independent datasets. The inherited20261021 member was previously selected. '
    'Old5 and new96 are reused developer data; no inspected test is used. Independent early stopping may change actual total updates. '
    'Seed averaging cannot recover absent head information. Legacy old-natural includes CoT truth; only tool has zero support. '
    'Fresh public SUA is a different cohort. Developer promotion never completes the90% long-body/natural confirmation goal.')


def protocol():
    return dict(version=1,experiment='bounded-multiseed-head512-ensemble',new_seeds=SEEDS,members=MEMBERS,records=RECORDS,
        fit_rows=FIT_ROWS,old_valid=OLD_VALID,new_valid=NEW_VALID,dimensions=512,roles=data.ROLES,
        neural=dict(kind='mlp',hidden=128,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=BATCH,
            instances=FIT_ROWS,steps=STEPS,last_batch=554,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5),
        scaler='Unchanged original fit-only512 scaler; exact head_probes.partitions578090 fit.',
        early_stop='Exact original old5 meanF1 helper; best model/history/GPUoldp frozen before new96 predictions.',
        probabilities='Four same-path CPU portable F32 predictions, chunk4096; fixed-order float64 sum /4 castF32, no renormalization.',
        drift='Keep original savedGPU oldp/tau and audited old-head new96 reference. Full old/new CPU drift<5e-4 plus changed argmax/original-tau decisions; same-vector GPU parity required.',
        calibration='Frozen aug maximin actualF32 full-grid proof8groups24 P/R/BAcc; tie old5meanF1 then smaller originalF64tau; six groups FPR<=.05.',
        promotion=dict(candidate='ensemble-maximin',controls=['old-cpu-maximin','old-original'],strict_min24_gain=1e-12,
            guards=aug.protocol()['promotion'],singletons='Diagnostic only; no fallback.'),
        tests_read=False,primary_changed=False,new_independent_confirmation_required=True,limitation=LIMITATION)


def configure_torch():
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    require(torch.get_num_threads()==4 and torch.get_float32_matmul_precision()=='highest','Portable CPU settings differ from declaration.')
    return dict(torch_threads=4,float32_matmul_precision='highest')


def initial_hash(seed):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return aug.init_hash(hp.make_net('mlp',hidden=128))


def static_snapshot():
    """Metadata/portable model hashes only; never open a probability array."""
    hp.require_hash(DESIGN,DESIGN_SHA,'root prospective design')
    original=aug.static_snapshot()
    files=[aug.OUT/name for name in ['protocol.json','search-plan.json','complete.json','selection.json','old-head-control.json',
        'actual-audit/actual-development-audit.json','actual-audit/audit.py']]
    require(all(p.is_file() for p in files),'Prior completed development audit/control is required.')
    audit=read(files[-2]);previous=read(aug.OUT/'complete.json');selection=read(aug.OUT/'selection.json')
    require(audit['stage']=='actual_completed_long_augmentation_development_audit' and audit['audit_passed'] is True
        and audit['test_data_or_results_read'] is False and selection['candidate']=='new-maximin' and selection['promoted'] is False
        and previous['tests_read'] is False and previous['source_sha256']==data.digest(aug.__file__), 'Prior negative audited trial changed.')
    hp.require_hash(files[-2],read(DESIGN)['root_execution_decision']['actual_development_audit_sha256'],'prior actual audit')
    for name,expected in audit['provenance_sha256'].items():
        # Only metadata/code/portable models; no raw features or probability payload is read at declaration.
        p=Path(name)
        if p.suffix in {'.json','.py'} and p.is_file(): hp.require_hash(p,expected,'prior metadata/source')
    old_control=read(aug.OUT/'old-head-control.json')
    for name,key in [('complete.json','trainer_complete_sha256'),('protocol.json','trainer_protocol_sha256'),('selection.json','trainer_selection_sha256')]:
        hp.require_hash(aug.OUT/name,audit[key],'audited prior '+name)
    return dict(source_sha256=data.digest(__file__),design_sha256=DESIGN_SHA,original=original,
        metadata_sha256={str(p):data.digest(p) for p in files},
        dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        initial_weight_sha256={str(seed):initial_hash(seed) for seed in SEEDS},
        old_reference=dict(old_path=original['old_probability_path'],old_sha256=original['old_probability_declared_sha256'],
            new_path=str(aug.CACHE/'old-head-new-validation.npz'),new_sha256=old_control['new_probability_sha256']),
        standardization_sha256=data.digest(hp.OUT/'standardization.npz'),
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn']},
        torch_threads=4,float32_matmul_precision='highest',probability_arrays_read=False)


def declare():
    value=dict(protocol=protocol(),static=static_snapshot());path=OUT/'protocol.json'
    if path.exists():require(read(path)==value,'Frozen declaration changed.');return value
    require(not CACHE.exists() and not (OUT/'search-plan.json').exists(),'Training/cache predates declaration.')
    OUT.mkdir(parents=True,exist_ok=True);data.atomic_json(path,value);return value


def verify_declaration():
    require((OUT/'protocol.json').is_file(),'Run --declare before any candidate p or training.')
    value=read(OUT/'protocol.json');require(value==dict(protocol=protocol(),static=static_snapshot()),'Source/model/protocol declaration changed.');return value


def dataset():
    old_rows,new_rows,banks,parts=aug.dataset();fit,old_valid,new_valid,*_=parts
    require(len(fit)==FIT_ROWS and len(old_valid)==OLD_VALID and len(new_valid)==NEW_VALID
        and not set(fit.cache_ix)&set(old_valid.cache_ix),'Original fit/validation membership changed.')
    return banks,fit,old_valid,new_valid


def source_signature():
    declaration=verify_declaration();upstream=aug.source_signature()
    return dict(source_sha256=data.digest(__file__),protocol_sha256=data.digest(OUT/'protocol.json'),
        static_sha256=aug.json_digest(declaration['static']),upstream=upstream,
        selected_head_ids=upstream['selected_head_ids'],standardization_sha256=declaration['static']['standardization_sha256'])


def verify_plan(plan,full=True):
    declaration=verify_declaration()
    require(plan['protocol']==protocol() and plan['initial_weight_sha256']==declaration['static']['initial_weight_sha256']
        and plan['signature']['protocol_sha256']==data.digest(OUT/'protocol.json'),'Plan/declaration changed.')
    if full:require(plan['signature']==source_signature(),'Feature/development/source signature changed.')
    return declaration


def check_history(history,best_epoch):
    return aug.check_history(history,best_epoch,steps=STEPS,instances=FIT_ROWS)


def fixed_mean(values):
    require(len(values)==4 and all(p.dtype==np.float32 and p.shape==values[0].shape for p in values),'Exactly four aligned F32 arrays required.')
    total=np.zeros_like(values[0],dtype=np.float64)
    for p in values:hp.validate_probabilities(p,len(p));total+=p
    value=(total/4).astype(np.float32);hp.validate_probabilities(value,len(value));return value


def drift_metrics(old_rows,new_rows,reference,cpu,threshold):
    groups=aug.validation_groups(old_rows,new_rows);output={}
    for label,(bank,ix) in dict(groups,**{'all-old':('old',np.arange(len(old_rows))),'all-new':('new',np.arange(len(new_rows)))}).items():
        a,b=reference[bank][ix],cpu[bank][ix];hp.validate_probabilities(a,len(a));hp.validate_probabilities(b,len(b))
        delta=np.abs(a-b);maximum=float(delta.max());require(np.isfinite(maximum) and maximum<5e-4,'Full old portable drift exceeds frozen tolerance: '+label)
        output[label]=dict(tokens=len(ix),maximum_probability_difference=maximum,mean_probability_difference=float(delta.mean()),
            absolute_difference_quantiles={str(q):float(np.quantile(delta,q)) for q in [.5,.9,.99,1.]},
            argmax_changes=int(np.count_nonzero(a.argmax(1)!=b.argmax(1))),
            original_threshold_decision_changes=int(np.count_nonzero((a[:,0]>=threshold)!=(b[:,0]>=threshold))))
    return output


def select_records(records):
    require(len(records)==6 and {r['record'] for r in records}==set(RECORDS),'Exactly six fixed records required.')
    index={r['record']:r for r in records};candidate=index['ensemble-maximin']['validation'];reasons=[];guards={};gains={}
    for name in ['old-cpu-maximin','old-original']:
        control=index[name]['validation'];guard=aug.promotion_guard(candidate,control);guards[name]=guard
        reasons.extend(name+':'+r for r in guard['rejection_reasons'])
        gain=candidate['min24']-control['min24'] if candidate['min24'] is not None and control['min24'] is not None else None
        gains[name]=gain
        if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min24_not_strictly_improved')
    return dict(candidate='ensemble-maximin',promoted=not reasons,no_fallback=True,guards=guards,min24_gains=gains,
        rejection_reasons=reasons,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def probability_path(seed,bank,kind='cpu'):
    return CACHE/f'{seed}-{kind}-{bank}-validation.npz'


def read_pair(seed,record,old_valid,new_valid):
    return {bank:aug.read_probability(probability_path(seed,bank),record[bank+'_probability_sha256'],rows)
        for bank,rows in [('old',old_valid),('new',new_valid)]}


def portable_check(path,banks,old_valid,new_valid,cpu,*,device='cuda:0'):
    result={}
    for bank,rows in [('old',old_valid),('new',new_valid)]:
        ix=np.linspace(0,len(rows)-1,256,dtype=int)
        gpu=hp.predict_raw(path,banks[bank],rows.iloc[ix].cache_ix.to_numpy(),device=device)
        error=float(np.abs(gpu-cpu[bank][ix]).max());require(np.isfinite(error) and 0<=error<5e-4,'Same-vector portable CPU/GPU parity failed.')
        result[bank]=error
    return dict(sampled_tokens=dict(old=256,new=256),maximum_probability_difference=result,CPU_outputs_saved=True)


def supervisor_alive():
    require(os.environ.get('ROLE_MULTI_SEED_SUPERVISOR_PID')==str(os.getppid()),'Worker requires its original live parent lease.')


def worker(seed,gpu):
    supervisor_alive();require(seed in SEEDS and gpu in [0,1,2],'Only three declared seed workers allowed.')
    require(not (OUT/'complete.json').exists() and not (OUT/'candidates'/f'{seed}.json').exists(),'Frozen result cannot be overwritten.')
    no_competing_jobs(allow_own=True);plan=read(OUT/'search-plan.json');verify_plan(plan)
    import torch
    configure_torch()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1 and 'A40' in torch.cuda.get_device_name(0),'Each worker needs one assigned A40.')
    banks,fit,old_valid,new_valid=dataset();mean,scale=aug.read_scaler()
    def standardized(rows):
        out=torch.empty((len(rows),512),dtype=torch.float32,device='cuda')
        for start in range(0,len(rows),BATCH):
            block=(np.asarray(banks['old'][rows.iloc[start:start+BATCH].cache_ix.to_numpy()],np.float32)-mean)/scale
            require(np.isfinite(block).all(),'Nonfinite fit/old-validation feature.');out[start:start+len(block)]=torch.from_numpy(block).cuda()
        return out
    require(torch.cuda.mem_get_info(0)[0]>(FIT_ROWS+OLD_VALID)*512*4+2*2**30,'Insufficient worker GPU memory.')
    x=standardized(fit);xv=standardized(old_valid);y=torch.from_numpy(probes.labels(fit)).cuda()
    torch.manual_seed(seed);net=hp.make_net('mlp',hidden=128).cuda();init=aug.init_hash(net)
    require(init==plan['initial_weight_sha256'][str(seed)],'Seed initialization changed.')
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);generator=torch.Generator(device='cuda:0').manual_seed(seed)
    path=OUT/'models'/f'{seed}.npz';gpupath=probability_path(seed,'old','gpu');history=[];best=-1.;best_epoch=-1
    meta=dict(seed=seed,kind='mlp',hidden=128,dimensions=512,heads=8,head_dim=64,roles=data.ROLES,selected_head_ids=plan['signature']['selected_head_ids'],
        signature=plan['signature'],source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),
        initial_weight_sha256=init,standardization_sha256=plan['signature']['standardization_sha256'],tests_read=False)
    for epoch in range(1,46):
        supervisor_alive();net.train();order=lp.epoch_order(len(x),generator,'cuda');loss_sum=0.;steps=0
        require(len(order)==FIT_ROWS,'Epoch instance budget changed.')
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH];loss=torch.nn.functional.cross_entropy(net.raw(x[ix].contiguous()),y[ix]);require(torch.isfinite(loss),'Nonfinite CE.')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ix);steps+=1
        require(steps==STEPS,'Epoch update budget changed.')
        p=probes.predict_standardized(net,xv).astype(np.float32);result=lp.validation_metrics(old_valid,p);score=result['mean_system_f1']
        history.append(dict(epoch=epoch,loss=loss_sum/len(order),instances=len(order),steps=steps,cumulative_steps=epoch*STEPS,old5_mean_f1=score))
        if score>best+1e-5:
            best,best_epoch=score,epoch;hp.save_model(path,net,mean,scale,meta|dict(best_epoch=epoch,best_steps=epoch*STEPS))
            hp.atomic_npz(gpupath,p=p,sample_ix=old_valid.sample_ix.to_numpy())
        print(json.dumps(dict(seed=seed,**history[-1],best_epoch=best_epoch)),flush=True)
        if epoch>=10 and epoch-best_epoch>=8:break
    budget=check_history(history,best_epoch);supervisor_alive()
    training=dict(seed=seed,signature=plan['signature'],plan_sha256=data.digest(OUT/'search-plan.json'),initial_weight_sha256=init,
        history=history,**budget,model_sha256=data.digest(path),gpu_old_probability_sha256=data.digest(gpupath),new_validation_predictions_read=False)
    data.atomic_json(OUT/'training'/f'{seed}.json',training)
    # No new96 prediction above this immutable best-checkpoint/history evidence.
    cpu={}
    for bank,rows in [('old',old_valid),('new',new_valid)]:
        cpu[bank]=hp.predict_raw(path,banks[bank],rows.cache_ix.to_numpy())
        hp.atomic_npz(probability_path(seed,bank),p=cpu[bank],sample_ix=rows.sample_ix.to_numpy())
    original=aug.read_probability(gpupath,training['gpu_old_probability_sha256'],old_valid)
    old_error=float(np.abs(original-cpu['old']).max());require(np.isfinite(old_error) and old_error<5e-4,'Full old GPU/CPU candidate parity failed.')
    proof=portable_check(path,banks,old_valid,new_valid,cpu)
    verify_plan(plan);supervisor_alive()
    data.atomic_json(OUT/'candidates'/f'{seed}.json',dict(seed=seed,signature=plan['signature'],source_sha256=data.digest(__file__),
        plan_sha256=data.digest(OUT/'search-plan.json'),initial_weight_sha256=init,model_sha256=training['model_sha256'],
        training_sha256=data.digest(OUT/'training'/f'{seed}.json'),gpu_old_probability_sha256=training['gpu_old_probability_sha256'],
        old_probability_sha256=data.digest(probability_path(seed,'old')),new_probability_sha256=data.digest(probability_path(seed,'new')),
        old_early_stop_validation=lp.validation_metrics(old_valid,original),portable_evidence=proof,full_old_GPU_CPU_max_difference=old_error,
        **budget,tests_read=False,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_name=torch.cuda.get_device_name(0)))


def candidate_record(seed,plan,old_valid,new_valid):
    record=read(OUT/'candidates'/f'{seed}.json');evidence_path=OUT/'training'/f'{seed}.json';hp.require_hash(evidence_path,record['training_sha256'],'training freeze')
    evidence=read(evidence_path);budget=check_history(evidence['history'],evidence['best_epoch']);path=OUT/'models'/f'{seed}.npz'
    require(record['seed']==evidence['seed']==seed and record['signature']==evidence['signature']==plan['signature']
        and record['source_sha256']==data.digest(__file__) and record['plan_sha256']==evidence['plan_sha256']==data.digest(OUT/'search-plan.json')
        and record['initial_weight_sha256']==evidence['initial_weight_sha256']==plan['initial_weight_sha256'][str(seed)]
        and evidence['new_validation_predictions_read'] is False and record['tests_read'] is False
        and record['model_sha256']==evidence['model_sha256'] and all(record[k]==evidence[k]==v for k,v in budget.items()),'Seed/checkpoint/history evidence changed.')
    hp.require_hash(path,record['model_sha256'],'seed portable model');_,mean,scale,meta=hp.load_model(path);m,s=aug.read_scaler()
    require(meta['seed']==seed and meta['signature']==plan['signature'] and meta['initial_weight_sha256']==record['initial_weight_sha256']
        and meta['source_sha256']==data.digest(__file__) and meta['plan_sha256']==data.digest(OUT/'search-plan.json')
        and meta['kind']=='mlp' and meta['hidden']==128 and meta['dimensions']==512 and meta['heads']==8 and meta['head_dim']==64
        and meta['standardization_sha256']==plan['signature']['standardization_sha256']
        and meta['best_epoch']==evidence['best_epoch'] and meta['best_steps']==evidence['best_steps']
        and meta['selected_head_ids']==plan['signature']['selected_head_ids'] and np.array_equal(mean.numpy(),m) and np.array_equal(scale.numpy(),s),'Portable seed/scaler provenance changed.')
    cpu=read_pair(seed,record,old_valid,new_valid);gpu=aug.read_probability(probability_path(seed,'old','gpu'),record['gpu_old_probability_sha256'],old_valid)
    result=lp.validation_metrics(old_valid,gpu)
    require(record['gpu_old_probability_sha256']==evidence['gpu_old_probability_sha256'] and result==record['old_early_stop_validation']
        and result['mean_system_f1']==evidence['history'][evidence['best_epoch']-1]['old5_mean_f1'],'Best GPUoldp does not reproduce early stopping.')
    actual=float(np.abs(gpu-cpu['old']).max());proof=record['portable_evidence']
    require(record['full_old_GPU_CPU_max_difference']==actual and 0<=actual<5e-4 and proof['sampled_tokens']==dict(old=256,new=256)
        and proof['CPU_outputs_saved'] is True and set(proof['maximum_probability_difference'])=={'old','new'}
        and all(np.isfinite(v) and 0<=v<5e-4 for v in proof['maximum_probability_difference'].values()),'Portable saved probability evidence changed.')
    require(record['cuda_visible_devices'] in ['0','1','2'] and 'A40' in record['cuda_device_name'],'Assigned worker GPU evidence changed.')
    return record,cpu


def old_control(plan,banks,old_valid,new_valid,allow_create=False):
    configure_torch()
    declaration=verify_plan(plan,False);reference=declaration['static']['old_reference'];oldcandidate=read(hp.OUT/'candidates/joint-mlp128.json')
    saved={bank:aug.read_probability(reference[bank+'_path'],reference[bank+'_sha256'],rows) for bank,rows in [('old',old_valid),('new',new_valid)]}
    require(lp.validation_metrics(old_valid,saved['old'])==oldcandidate['validation']
        and oldcandidate['validation']['system_threshold']==aug.CURRENT_THRESHOLD,'Original savedGPU p/tau do not reproduce exactly.')
    path=OUT/'old-control.json';model=hp.OUT/'models/joint-mlp128.npz'
    if path.exists():
        record=read(path);require(record['signature']==plan['signature'] and record['model_sha256']==aug.CURRENT_MODEL_SHA
            and record['reference']==reference,'Old control signature changed.');cpu=read_pair(20261021,record,old_valid,new_valid)
    else:
        require(allow_create and not any(probability_path(20261021,b).exists() for b in ['old','new']),'Missing/unowned old CPU control; verification cannot silently recompute.')
        cpu={bank:hp.predict_raw(model,banks[bank],rows.cache_ix.to_numpy()) for bank,rows in [('old',old_valid),('new',new_valid)]}
        proof=portable_check(model,banks,old_valid,new_valid,cpu)
        record=dict(signature=plan['signature'],model_sha256=aug.CURRENT_MODEL_SHA,reference=reference,portable_evidence=proof,
            old_probability_sha256='',new_probability_sha256='',drift=drift_metrics(old_valid,new_valid,saved,cpu,aug.CURRENT_THRESHOLD),
            cpu_at_original_threshold=aug.metrics_at(old_valid,new_valid,cpu['old'],cpu['new'],aug.CURRENT_THRESHOLD),tests_read=False)
        for bank,rows in [('old',old_valid),('new',new_valid)]:
            hp.atomic_npz(probability_path(20261021,bank),p=cpu[bank],sample_ix=rows.sample_ix.to_numpy());record[bank+'_probability_sha256']=data.digest(probability_path(20261021,bank))
        data.atomic_json(path,record)
    require(record['tests_read'] is False and record['drift']==drift_metrics(old_valid,new_valid,saved,cpu,aug.CURRENT_THRESHOLD)
        and record['cpu_at_original_threshold']==aug.metrics_at(old_valid,new_valid,cpu['old'],cpu['new'],aug.CURRENT_THRESHOLD),'Full old drift/decision report changed.')
    proof=record['portable_evidence'];require(proof['sampled_tokens']==dict(old=256,new=256) and proof['CPU_outputs_saved'] is True
        and set(proof['maximum_probability_difference'])=={'old','new'} and all(np.isfinite(v) and 0<=v<5e-4 for v in proof['maximum_probability_difference'].values()),'Old portable same-vector evidence changed.')
    return record,saved,cpu


def calibrate(plan,allow_create=False):
    configure_torch()
    banks,fit,old_valid,new_valid=dataset();old_record,saved,oldcpu=old_control(plan,banks,old_valid,new_valid,allow_create)
    member={20261021:oldcpu};model_sha={20261021:aug.CURRENT_MODEL_SHA}
    for seed in SEEDS:r,p=candidate_record(seed,plan,old_valid,new_valid);member[seed]=p;model_sha[seed]=r['model_sha256']
    ensemble={bank:fixed_mean([member[s][bank] for s in MEMBERS]) for bank in ['old','new']}
    spec=dict(signature=plan['signature'],members=MEMBERS,weights=[.25]*4,arithmetic=protocol()['probabilities'],model_sha256={str(s):model_sha[s] for s in MEMBERS},
        member_probability_sha256={str(s):{b:data.digest(probability_path(s,b)) for b in ['old','new']} for s in MEMBERS})
    if allow_create:
        if (OUT/'ensemble.json').exists():require(read(OUT/'ensemble.json')==spec,'Ensemble members/arithmetic changed.')
        else:data.atomic_json(OUT/'ensemble.json',spec)
        for bank,rows in [('old',old_valid),('new',new_valid)]:
            path=CACHE/f'ensemble-{bank}-validation.npz'
            if path.exists():
                existing=aug.read_probability(path,data.digest(path),rows);require(np.array_equal(existing,ensemble[bank]),'Partial ensemble probability changed.')
            else:hp.atomic_npz(path,p=ensemble[bank],sample_ix=rows.sample_ix.to_numpy())
    else:
        require(read(OUT/'ensemble.json')==spec,'Ensemble source/member closure changed.')
        for bank,rows in [('old',old_valid),('new',new_valid)]:
            value=aug.read_probability(CACHE/f'ensemble-{bank}-validation.npz',read(OUT/'selection.json')['ensemble_probability_sha256'][bank],rows)
            require(np.array_equal(value,ensemble[bank]),'Saved ensemble does not equal exact fixed4 probability mean.')
    records=[]
    for name in RECORDS:
        p=saved if name=='old-original' else oldcpu if name=='old-cpu-maximin' else ensemble if name=='ensemble-maximin' else member[int(name.split('-')[1])]
        if name=='old-original':tau,proof=aug.CURRENT_THRESHOLD,None
        else:tau,proof=aug.maximin_threshold(old_valid,new_valid,p['old'],p['new'])
        value=aug.metrics_at(old_valid,new_valid,p['old'],p['new'],tau)
        if proof is not None:require(value['min24']==proof['min24'] and value['old5_mean_f1']==proof['old5_mean_f1'],'Maximin actual-consumer metrics/proof differ.')
        records.append(dict(record=name,validation=value,full_grid_proof=proof,diagnostic=name!='ensemble-maximin',tests_read=False))
    return records,old_valid,new_valid,fit,spec


def publish(plan):
    records,old_valid,new_valid,fit,spec=calibrate(plan,True);selected=select_records(records)
    data.atomic_json(OUT/'records.json',records)
    flat=[]
    for record in records:
        for group,v in dict(record['validation']['groups'],pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=record['record'],group=group,**{k:x for k,x in v.items() if not isinstance(x,(list,dict))},**{'role_recall_'+r:x for r,x in v['recall'].items()}))
    pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False)
    counts=dict(fit=len(fit),old_valid=len(old_valid),new_valid=len(new_valid),fit_documents=int(fit.group_id.nunique()),
        original_partition_sha256=plan['signature']['upstream']['old_partitions'],old_profile_counts={p:int(old_valid.profile.eq(p).sum()) for p in lp.PROFILES},
        old_natural_role_support={r:int(((old_valid.profile=='natural')&(old_valid.role==r)).sum()) for r in data.ROLES},
        new96_is_reused_development=True,tails_are_subsets=True,limitation=LIMITATION)
    data.atomic_json(OUT/'source-counts.json',counts)
    lines=['Bounded multiseed probability ensemble: development only',LIMITATION,'','record threshold min24 old5meanF1']
    lines += [f"{r['record']} {r['validation']['system_threshold']:.12g} {r['validation']['min24']} {r['validation']['old5_mean_f1']:.9f}" for r in records]
    lines += ['', 'Promoted: '+str(selected['promoted']),'Reasons: '+json.dumps(selected['rejection_reasons'])]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    selected.update(source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),records_sha256=data.digest(OUT/'records.json'),
        ensemble_definition_sha256=data.digest(OUT/'ensemble.json'),member_model_sha256=spec['model_sha256'],
        ensemble_probability_sha256={b:data.digest(CACHE/f'ensemble-{b}-validation.npz') for b in ['old','new']},
        system_threshold=next(r['validation']['system_threshold'] for r in records if r['record']=='ensemble-maximin'))
    data.atomic_json(OUT/'selection.json',selected);verify_plan(plan)
    data.atomic_json(OUT/'complete.json',dict(signature=plan['signature'],source_sha256=data.digest(__file__),
        artifact_sha256={n:data.digest(OUT/n) for n in ARTIFACTS},candidate_sha256={str(s):data.digest(OUT/'candidates'/f'{s}.json') for s in SEEDS},
        tests_read=False,primary_changed=False,new_independent_confirmation_required=True))
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',promoted=selected['promoted'],primary_changed=False))


def verified_completion():
    require((OUT/'complete.json').is_file() and read(OUT/'status.json')['stage']=='validation_complete','Multiseed trial not terminal complete.')
    plan=read(OUT/'search-plan.json');verify_plan(plan);complete=read(OUT/'complete.json')
    require(complete['signature']==plan['signature'] and complete['source_sha256']==data.digest(__file__)
        and complete['tests_read'] is False and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and set(complete['artifact_sha256'])==set(ARTIFACTS) and set(complete['candidate_sha256'])==set(map(str,SEEDS)),'Completion source/coverage changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(OUT/n,h,'completed artifact')
    for s,h in complete['candidate_sha256'].items():hp.require_hash(OUT/'candidates'/f'{s}.json',h,'completed seed')
    records,*_=calibrate(plan,False);require(records==read(OUT/'records.json'),'Saved six records do not reproduce.')
    selected=select_records(records);actual=read(OUT/'selection.json');require(all(actual[k]==v for k,v in selected.items()),'No-fallback gate changed.')
    spec=read(OUT/'ensemble.json')
    require(actual['source_sha256']==data.digest(__file__) and actual['plan_sha256']==data.digest(OUT/'search-plan.json')
        and actual['records_sha256']==data.digest(OUT/'records.json') and actual['ensemble_definition_sha256']==data.digest(OUT/'ensemble.json')
        and actual['member_model_sha256']==spec['model_sha256']
        and actual['system_threshold']==next(r['validation']['system_threshold'] for r in records if r['record']=='ensemble-maximin'),
        'Selected ensemble/model/threshold provenance differs.')
    return complete


def no_competing_jobs(allow_own=False):
    names={'systemness_long_augmentation_probes.py','systemness_long_augmentation_extract.py','systemness_aug_confirmation_infer.py','systemness_head_probes.py',
        'systemness_head_extract.py','systemness_head_scan.py','systemness_long_extract.py','systemness_recover_local_bank.py','systemness_head_confirm.py',
        'systemness_natural_test_confirm.py','systemness_head_fusion.py','systemness_long_confirm.py','systemness_attention_confirm.py'}
    if not allow_own:names.add(Path(__file__).name)
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name in names for a in args if a):active.append(int(entry.name))
    require(not active,'Competing training/capture remains live: '+str(active))


@contextmanager
def run_lock():
    import fcntl
    with (OUT/'.run.lock').open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another multiseed supervisor owns this directory.') from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run_workers(pending):
    require(len(pending)<=3 and len(set(pending))==len(pending) and set(pending)<=set(SEEDS),'Only three predetermined seeds allowed.')
    jobs=[]
    try:
        for gpu,seed in enumerate(pending):
            log=(OUT/'logs'/f'{seed}.log').open('a')
            try:proc=subprocess.Popen([sys.executable,'-u',__file__,'--train','--worker',str(seed),'--gpu',str(gpu)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ROLE_MULTI_SEED_SUPERVISOR_PID=str(os.getpid())),stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            jobs.append((seed,proc,log,gpu))
        data.atomic_json(OUT/'status.json',dict(stage='training',workers=[dict(seed=s,pid=p.pid,gpu=g) for s,p,_,g in jobs]))
        while jobs:
            codes=[(s,p.poll()) for s,p,_,_ in jobs];require(not [(s,c) for s,c in codes if c not in [None,0]],'A multiseed worker failed: '+str(codes))
            if all(c==0 for _,c in codes):break
            time.sleep(5)
        for seed,p,_,_ in jobs:require(p.wait()==0,str(seed)+' failed.')
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
    configure_torch()
    no_competing_jobs();verify_declaration();signature=source_signature()
    with run_lock():
        if (OUT/'complete.json').exists():verified_completion();return
        require(not (OUT/'selection.json').exists(),'Partial frozen selection cannot be silently replaced.')
        for n in ['models','training','candidates','logs']:(OUT/n).mkdir(exist_ok=True)
        CACHE.mkdir(parents=True,exist_ok=True);path=OUT/'search-plan.json'
        if path.exists():plan=read(path);verify_plan(plan)
        else:
            require(not any(CACHE.iterdir()) and all(not any((OUT/n).iterdir()) for n in ['models','training','candidates']),'Unowned model/probability cache predates plan.')
            plan=dict(protocol=protocol(),signature=signature,initial_weight_sha256=read(OUT/'protocol.json')['static']['initial_weight_sha256'])
            data.atomic_json(path,plan)
        _,_,old_valid,new_valid=dataset();pending=[]
        for seed in SEEDS:
            if (OUT/'candidates'/f'{seed}.json').exists():candidate_record(seed,plan,old_valid,new_valid)
            else:
                partial=[OUT/'models'/f'{seed}.npz',OUT/'training'/f'{seed}.json',probability_path(seed,'old','gpu'),probability_path(seed,'old'),probability_path(seed,'new')]
                require(restart_incomplete or not any(p.exists() for p in partial),'Partial seed requires explicit --restart-incomplete; no progress trusted.')
                if any(p.exists() for p in partial):data.atomic_json(OUT/f'restart-{seed}-{time.time_ns()}.json',dict(seed=seed,from_epoch=1,signature=signature,partial_sha256={str(p):data.digest(p) for p in partial if p.exists()}))
                pending.append(seed)
        if pending:run_workers(pending)
        publish(plan)


def preflight():
    from unittest.mock import patch
    import copy,py_compile,torch
    torch.set_num_threads(2);folder=Path(tempfile.mkdtemp(prefix='systemness-multiseed-preflight-',dir='/tmp'));source=data.digest(__file__);checks=[]
    def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError,AssertionError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('CPU fault accepted: '+name)
    def forbidden(*a,**kw):raise AssertionError('Formal metadata/probability/model API invoked in synthetic preflight.')
    with patch.object(aug,'dataset',side_effect=forbidden),patch.object(aug,'source_signature',side_effect=forbidden),patch.object(data.runtime,'load_model',side_effect=forbidden):
        torch.set_num_threads(2);torch.set_float32_matmul_precision('medium')
        check('CPU settings corrected and verified before portable prediction',configure_torch()==dict(torch_threads=4,float32_matmul_precision='highest')
            and torch.get_num_threads()==4 and torch.get_float32_matmul_precision()=='highest')
        torch.manual_seed(17);state=torch.get_rng_state().clone();hashes=[initial_hash(s) for s in SEEDS]
        check('Three fixed distinct initializations preserve caller Torch RNG',len(set(hashes))==3 and torch.equal(state,torch.get_rng_state()) and hashes==[initial_hash(s) for s in SEEDS])
        g=torch.Generator().manual_seed(SEEDS[0]);order=lp.epoch_order(FIT_ROWS,g,'cpu')
        check('Exact578090 permutation /142updates /last554',len(order)==FIT_ROWS and torch.unique(order).numel()==FIT_ROWS and len(list(range(0,len(order),BATCH)))==142 and len(order[-554:])==554)
        hist=[dict(epoch=i,instances=FIT_ROWS,steps=STEPS,cumulative_steps=i*STEPS,loss=1.,old5_mean_f1=.8) for i in range(1,11)]
        check('Original min10/patience8 stops at10 with best1',check_history(hist,1)['actual_steps']==1420)
        broken=copy.deepcopy(hist);broken[-1]['steps']=143;reject('Changed update evidence rejected',lambda:check_history(broken,1))
        raw=np.random.default_rng(3).uniform(size=(4,20,5)).astype(np.float32);raw/=raw.sum(2,keepdims=True)
        mean=fixed_mean(list(raw));direct=np.add.reduce(raw.astype(np.float64),axis=0).astype(np.float64)/4
        check('Fixed4 F64 mean castF32 exact',np.array_equal(mean,direct.astype(np.float32)))
        reject('Missing ensemble member rejected',lambda:fixed_mean(list(raw[:3])))
        def value(v):return dict(min24=v,old5_mean_f1=v,groups={g:dict(system_f1=v,accuracy=v,system_false_positive_rate=.01) for g in aug.GROUPS})
        records=[dict(record=n,validation=value(.92 if n=='ensemble-maximin' else .91)) for n in RECORDS]
        check('Only ensemble may advance against dual controls',select_records(records)['promoted'])
        tied=copy.deepcopy(records);tied[-1]['validation']['min24']=.91;reject_value=select_records(tied)
        check('Strict min24 tie rejected without singleton fallback',not reject_value['promoted'] and reject_value['candidate']=='ensemble-maximin')
        for group,key,amount in [('old-natural','system_f1',.03),('old-natural','accuracy',.02),('old-long-full-tail','system_f1',.02),('new-long-full','system_false_positive_rate',.06)]:
            b=copy.deepcopy(records);b[-1]['validation']['groups'][group][key]=.91-amount if key!='system_false_positive_rate' else amount
            check('Guard rejects '+group+'/'+key,not select_records(b)['promoted'])
        # All8 groups with real CoT support in old-natural; no test or actual cache.
        old=[];new=[]
        for pi,p in enumerate(lp.PROFILES):
            for ri,r in enumerate(data.ROLES if p!='natural' else data.ROLES[:-1]):
                for off in [0,512,1023]:old.append(dict(cache_ix=len(old),sample_ix=len(old),profile=p,role=r,token_ix=10+off,body_offset=off,prompt_ix=pi*5+ri,message_ix=0,split='validation',wrapper='native'))
        for ri,r in enumerate(data.ROLES):
            for off in [0,512,1023]:new.append(dict(cache_ix=len(new),sample_ix=6000000000000+len(new),profile='long-full',role=r,token_ix=10+off,body_offset=off,prompt_ix=ri,message_ix=0,split='validation',wrapper='native'))
        old,new=pd.DataFrame(old),pd.DataFrame(new)
        def prob(frame):
            p=np.full((len(frame),5),.025,np.float32);p[np.arange(len(frame)),probes.labels(frame)]=.9;return p
        po,pn=prob(old),prob(new);tau,proof=aug.maximin_threshold(old,new,po,pn);v=aug.metrics_at(old,new,po,pn,tau)
        check('Eightgroup fullF32 grid proof and oldnatural CoT support',proof['groups']==8 and proof['all_grid_undefined_and_count_semantics_equal'] and v['groups']['old-natural']['recall']['cot']==1 and v['groups']['old-natural']['recall']['tool'] is None)
        report=drift_metrics(old,new,dict(old=po,new=pn),dict(old=po.copy(),new=pn.copy()),.5)
        check('Full drift all8+2 groups /exact zeros',len(report)==10 and all(r['maximum_probability_difference']==r['argmax_changes']==r['original_threshold_decision_changes']==0 for r in report.values()))
        bad=po.copy();bad[0,0]-=.001;bad[0,1]+=.001
        reject('Drift beyond5e-4 rejected',lambda:drift_metrics(old,new,dict(old=po,new=pn),dict(old=bad,new=pn),.5))
        target=folder/'p.npz';hp.atomic_npz(target,p=po,sample_ix=old.sample_ix.to_numpy())
        check('Bound p/sample identity roundtrip',np.array_equal(aug.read_probability(target,data.digest(target),old),po))
        altered=old.copy();altered.sample_ix+=1;reject('Wrong sample identity refused',lambda:aug.read_probability(target,data.digest(target),altered))
        with patch.object(sys.modules[__name__],'OUT',folder/'not-declared'):reject('No declaration rejects before training/p',verify_declaration)
    py_compile.compile(__file__,cfile=str(folder/'runner.pyc'),doraise=True);require(data.digest(__file__)==source,'Preflight altered source.')
    report=dict(source_sha256=source,checks=checks,checks_passed=len(checks),actual_probabilities_or_data_read=False,actual_model_or_LM_loaded=False,
        gpu_used=False,formal_outputs_written=False,fixture_directory=str(folder),records=6,new_training_seeds=SEEDS,max_additional_updates=19170)
    data.atomic_json(folder/'report.json',report);print(json.dumps(dict(report=str(folder/'report.json'),report_sha256=data.digest(folder/'report.json'),**report)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--declare',action='store_true');a.add_argument('--train',action='store_true');a.add_argument('--preflight',action='store_true')
    p.add_argument('--worker',type=int,choices=SEEDS,help=argparse.SUPPRESS);p.add_argument('--gpu',type=int,help=argparse.SUPPRESS);p.add_argument('--restart-incomplete',action='store_true')
    args=p.parse_args()
    if args.worker is not None:require(args.train and args.gpu is not None,'Workers require --train and assigned GPU.');worker(args.worker,args.gpu)
    elif args.preflight:preflight()
    elif args.declare:declare()
    elif args.train:train(args.restart_incomplete)
    else:p.print_help()

if __name__=='__main__':main()
