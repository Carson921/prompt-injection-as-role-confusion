#!/usr/bin/env python3
"""Independent synthetic source audit only. No formal slots, banks, weights, probabilities, tests, GPU."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT=Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_long_mix_residual as s
EXPECTED='3518cac9d635fef1ddfdd7b1da7aea7b4f4b91a9423f4dc42f005ee5746b2ea0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=sha(s.__file__);CHECKS=[]
def check(name,ok=True):
    assert bool(ok),name;CHECKS.append(name)
def reject(name,fn):
    try:fn()
    except(RuntimeError,ValueError,AssertionError,KeyError,FileNotFoundError):CHECKS.append(name)
    else:raise AssertionError('Fault accepted: '+name)
def banned(*args,**kwargs):raise AssertionError('Formal data/model/probability/GPU API forbidden in source audit.')
original_namespace=(s.br.OUT,s.br.CACHE)
with patch.object(np,'load',side_effect=banned),patch.object(np,'memmap',side_effect=banned),patch.object(pd,'read_parquet',side_effect=banned),patch.object(s,'read_base',side_effect=banned),patch.object(s.nf,'dataset',side_effect=banned),patch.object(s.data.runtime,'load_model',side_effect=banned),patch.object(s,'upstream_snapshot',side_effect=banned):
    check('Held final source3518 exact',START==EXPECTED)
    # Quarter-size pure metadata fixture preserves the actual132:256 ratio and exact33-per-half selector.
    grid=np.floor(np.arange(256,dtype=np.float64)*1023/255).astype(np.int64)
    def packet(documents,old):
        count=documents*5*256;doc=np.repeat(np.arange(documents),1280);role=np.tile(np.repeat(np.arange(5),256),documents)
        offset=np.tile(grid,documents*5)
        return pd.DataFrame(dict(cache_ix=(1000000 if old else 2000000)+np.arange(count)*3,
            sample_ix=(100000000 if old else 6000000000000)+np.arange(count),
            document_id=(1000 if old else 6000000)+doc,group_id=[('old-' if old else 'new-')+str(d) for d in doc],
            source_set='long-full' if old else 'long-augmentation',profile='long-full',role=np.asarray(s.data.ROLES)[role],
            split='fit',wrapper='native',prompt_ix=doc*5+role,message_ix=0,
            token_ix=offset+17+(doc%3)*11+role*7,body_offset=offset))
    full=packet(33,True);new=packet(64,False)
    distract=full.iloc[:21].copy();distract.cache_ix+=10000000;distract.sample_ix+=10000000
    distract.source_set='other-cache';distract.token_ix=-1000 # Same profile/prompt IDs, different source_set.
    fit=pd.concat([distract.iloc[:10],full,distract.iloc[10:]],ignore_index=True)
    np_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    slots,summary=s._replacement_manifest(fit,new,old_documents=33,new_documents=64,new_per_half=33)
    check('Synthetic33old64new exact production66-observation selector',len(slots)==21120 and summary['new_role_rows_per_document']==66 and summary['old_role_rows_per_document']==128)
    check('Ten role/half buckets2112 each preserve constant long rowweight',summary['role_half_buckets']=={r+':'+str(h):2112 for r in s.data.ROLES for h in [0,1]} and summary['kept_old_full_rows']==summary['selected_new_full_rows']==21120 and summary['total_full_rows']==42240)
    check('Private scatter consumes neither global NumPy nor Torch RNG',torch.equal(torch_state,torch.get_rng_state()) and all(np.array_equal(a,b) for a,b in zip(np_state,np.random.get_state())))
    check('Cache-local prompt collision filtered beforeoffset derivation',set(slots.old_cache_ix)<=set(full.cache_ix) and slots.old_body_offset.between(0,1023).all() and not set(slots.old_cache_ix)&set(distract.cache_ix))
    check('Noncontiguous cache_ix maps to original fit_position not rowlabels',np.array_equal(fit.iloc[slots.fit_position].cache_ix,slots.old_cache_ix) and np.array_equal(fit.iloc[slots.fit_position].sample_ix,slots.old_sample_ix))
    check('Bijection acrossold slots/new source cache/sample IDs',slots.fit_position.is_unique and slots.old_cache_ix.is_unique and slots.new_cache_ix.is_unique and slots.new_sample_ix.is_unique)
    old_kept=np.floor(np.arange(64)*127/63).astype(int);old_removed=np.setdiff1d(np.arange(128),old_kept)
    new_selected=np.floor(np.arange(33)*127/32).astype(int)
    expected_old=np.r_[grid[old_removed],grid[128+old_removed]]
    expected_new=np.r_[grid[new_selected],grid[128+new_selected]]
    old_sets=slots.groupby(['old_document_id','role']).old_body_offset.apply(lambda x:tuple(sorted(x)))
    new_sets=slots.groupby(['new_document_id','new_role']).new_body_offset.apply(lambda x:tuple(sorted(x)))
    check('Old removed slots are exact complement of independent uniform64perhalf',len(old_sets)==33*5 and all(x==tuple(expected_old) for x in old_sets))
    check('All five roles share exact independent uniform33perhalf newoffsets',len(new_sets)==64*5 and all(x==tuple(expected_new) for x in new_sets))
    check('Role+half and entire per-slot supervised array identical acrossarms',np.array_equal(slots.role,slots.new_role) and np.array_equal(slots.half,slots.old_body_offset.ge(512).astype(int)) and np.array_equal(slots.half,slots.new_body_offset.ge(512).astype(int)))
    label=s.probes.labels(fit);replaced=label.copy();replaced[slots.fit_position]=slots.new_role.map(dict(zip(s.data.ROLES,range(5)))).to_numpy()
    check('Replacement labels identical under arbitrary shared epochpermutation',np.array_equal(label,replaced) and np.array_equal(label[::-1],replaced[::-1]))
    check('Full production arithmetic84480/168960/641493/157/last2517',132*64==256*33==8448 and 8448*10==84480 and 409130+84480+84480+63403==s.INSTANCES and s.STEPS==157 and s.INSTANCES%s.BATCH==2517)
    bad=new.copy();bad.loc[0,'split']='validation'
    reject('A validation row cannot enter replacement',lambda:s._replacement_manifest(fit,bad,old_documents=33,new_documents=64,new_per_half=33))
    bad=fit.copy();bad.loc[10,'token_ix']+=1
    reject('Corrupt bodygrid fails after nuisance promptcollision filtering',lambda:s._replacement_manifest(bad,new,old_documents=33,new_documents=64,new_per_half=33))
    reject('Production wrapper refuses synthetic sizes instead of picking real slots',lambda:s.replacement_manifest(fit,new))

    def value(n=.92):return dict(min33=n,old5_mean_f1=.92,groups={g:dict(system_f1=.92,accuracy=.92,system_false_positive_rate=.01) for g in s.GROUPS})
    records=[dict(record=n,validation=value(.94 if n=='new-max33' else .92)) for n in s.RECORDS]
    zero={n:value() for n in ['zero-oldmean','zero-max33']};legacy=value()
    result=s.select_records(records,legacy,zero)
    check('Twelve records plus two zero with5strict8guards andGPU',len(s.RECORDS)==12 and len(s.STRICT_CONTROLS)==5 and len(s.GUARD_CONTROLS)==8 and len(result['guards'])==9 and result['promoted'] and result['candidate']=='new-max33')
    for name in s.STRICT_CONTROLS:
        rr=copy.deepcopy(records);zz=copy.deepcopy(zero)
        target=zz[name] if name in zz else next(r['validation'] for r in rr if r['record']==name)
        target['min33']=.94
        answer=s.select_records(rr,legacy,zz)
        check('Strict tie rejects '+name,not answer['promoted'] and name+':min33_not_strictly_improved' in answer['rejection_reasons'])
    rr=copy.deepcopy(records);next(r['validation'] for r in rr if r['record']=='priorBinary-max33')['groups']['old-long-full-tail']['system_f1']=.99
    check('New priorBinary fulltail mandatoryfloor rejects',not s.select_records(rr,legacy,zero)['promoted'])
    for r in rr:
        if r['record']=='priorBinary-max33':r['validation']=value()
        if r['record'].startswith('naturalCE') or r['record']=='priorBinary-oldmean':r['validation']=value(.999)
    check('NaturalCE and priorBinary oldmean remain diagnostics only',s.select_records(rr,legacy,zero)['promoted'])
    val=dict(groups={g:dict(system_f1=.92,recall={r:None for r in s.data.ROLES}) for g in s.GROUPS},pooled_descriptive_only=dict(system_f1=.92,recall={r:None for r in s.data.ROLES}))
    check('Separate144main24zero total168CSV coverage',len(s.flat_metrics([dict(record=n,validation=val) for n in s.RECORDS]))==144 and len(s.flat_metrics([dict(record=n,validation=val) for n in zero]))==24)

    # Spy actual calibrator lifecycle, replacing all model and probability methods with trivial CPU objects.
    events=[];toyrows=pd.DataFrame(dict(cache_ix=[0],sample_ix=[1],split=['fit']))
    physical={name:{'toy_model':name,'old':None} for name in ['new','matched','oldhead','ensemble','naturalCE','priorBinary']}
    def candidate(arm,*a):events.append('candidate:'+arm);return {},physical[arm]
    def inherit(*a):events.append('inherited');return {n:physical[n] for n in ['oldhead','ensemble','naturalCE','priorBinary']},legacy
    def repeat(*a):events.append('replication');require_order=['candidate:new','candidate:matched','inherited','replication'];assert events==require_order
    def zeros(*a):events.append('zero');assert events[-2]=='replication';return zero
    def metric(rows,p,tau):
        v=value(.8990885416666666 if p['toy_model']=='priorBinary' else .94);v['system_threshold']=tau;return v
    def threshold(rows,p):
        tau=.5453636363636364 if p['toy_model']=='priorBinary' else .6008080808080808 if p['toy_model']=='naturalCE' else .5
        v=metric(rows,p,tau);return tau,dict(min33=v['min33'],old5_mean_f1=v['old5_mean_f1'])
    with patch.object(s,'dataset',return_value=({},toyrows,toyrows,toyrows,toyrows,toyrows,{})),patch.object(s,'verified_replacement',return_value=slots),patch.object(s,'candidate_record',side_effect=candidate),patch.object(s,'inherited_controls',side_effect=inherit),patch.object(s,'replication',side_effect=repeat),patch.object(s,'zero_controls',side_effect=zeros),patch.object(s,'metrics_at',side_effect=metric),patch.object(s,'max33',side_effect=threshold),patch.object(s.lp,'validation_metrics',return_value={'system_threshold':.6}):
        calibrated=s.calibrate({},False)
    check('Both candidates verified then inherited/replication beforezero/calibration',events==['candidate:new','candidate:matched','inherited','replication','zero'] and len(calibrated[0])==12)

    # Only source structure is inspected for training; no worker/declare is run.
    src=Path(s.__file__).read_text();tree=ast.parse(src)
    def function(name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    closed=function('upstream_snapshot')
    dynamic={'source_signature','verified_completion','frozen_snapshot','exclusion_inputs','rglob'}
    check('Closed prior2600/feature69 certificates do notcall global future verifiers',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in dynamic for n in ast.walk(closed)))
    check('Prior source/audit/newbank hardpins match root design',s.PRIOR_AUDIT_SHA=='2600d086c711e65bb320b70e66539fa775164cba8c192b84a2f9c52a75a9aa67' and s.NEW_FEATURES_SHA=='69e3db049f3de168c4c0922cfd1a935bb3bf19f91d697da75af2fe3752269cd8' and s.DESIGN_SHA=='e2fb7dfb369c35f137ab23900b9eb611a8f33d9db8d8686ca4ee2c0961d36698')
    worker=inspect.getsource(s.worker)
    check('Matched dataset repeats previousbinary NEW natural features, no oldSUA draw',"extra_x=standardized('natural',nrows.cache_ix.to_numpy())" in worker and 'matched_old_cache_ix' not in worker and "if arm=='new':old_x[" in worker and 'y=torch.from_numpy(np.r_[probes.labels(fit),probes.labels(nrows)]==0)' in worker)
    check('Per-arm immutablebest freeze before newdevelopment p',worker.index("data.atomic_json(OUT/'training'")<worker.index('predict_raw(path'))
    repeat=inspect.getsource(s.replication)
    check('Matched complete3bank probability drift and allweight equality recorded',all(x in repeat for x in ["['old','c4','public']",'all_weight_arrays_bitexact','all_probability_arrays_bitexact','maximum_probability_difference\']<5e-4','numerical_drift_is_not_source_effect=True']))
    verifier=inspect.getsource(s.verified_completion)
    check('Completed route calibrates read-only False andbinds replicationSHA',"calibrate(plan,False)" in verifier and "actual['replication_sha256']==data.digest(OUT/'replication.json')" in verifier)
    check('Reused pure model functions no br global namespace override',s.make_model is s.br.make_model and s.predict_raw is s.br.predict_raw and s.load_model is s.br.load_model and original_namespace==(s.br.OUT,s.br.CACHE) and not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='br' for t in n.targets) for n in ast.walk(tree)))
    check('Final production source unchanged during synthetic audit',sha(s.__file__)==START)

folder=Path(tempfile.mkdtemp(prefix='feature-review-long-mix-source-audit-',dir='/tmp'));dest=folder/'report.json'
report=dict(stage='independent_long_mix_residual_source_synthetic_audit',audit_passed=True,source_sha256=START,source_sha256_after=sha(s.__file__),
    auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=sha(__file__),checks_passed=len(CHECKS),checks=CHECKS,
    synthetic_old_sources=33,synthetic_new_sources=64,synthetic_replacement_rows=len(slots),actual_slots_selected=False,
    actual_model_weights_or_probabilities_read=False,test_data_read=False,gpu_used=False,formal_outputs_modified=False)
dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(report=str(dest),report_sha256=sha(dest),**report)))
