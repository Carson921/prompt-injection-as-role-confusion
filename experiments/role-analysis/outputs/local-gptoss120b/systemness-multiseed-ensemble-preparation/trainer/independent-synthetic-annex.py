#!/usr/bin/env python3
"""Independent source/synthetic audit only; no formal probability/data access."""
import sys,pathlib,json,hashlib,copy,inspect
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=pathlib.Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_multiseed_ensemble as s
import torch
PATH=pathlib.Path(s.__file__)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
START=sha(PATH);CHECKS=[]
def check(name,ok):
 assert bool(ok),name;CHECKS.append(name)
def reject(name,fn):
 try:fn()
 except (RuntimeError,ValueError,KeyError,AssertionError,FileNotFoundError):CHECKS.append(name)
 else:raise AssertionError('accepted '+name)
def banned(*a,**k):raise AssertionError('Actual IO is forbidden')
with patch.object(np,'load',side_effect=banned),patch.object(pd,'read_parquet',side_effect=banned),patch.object(s.aug,'dataset',side_effect=banned),patch.object(s.aug,'source_signature',side_effect=banned),patch.object(s.data.runtime,'load_model',side_effect=banned):
 check('exact fixed seed/member/record identities',s.SEEDS==[20261022,20261023,20261024] and s.MEMBERS==[20261021,20261022,20261023,20261024] and len(s.RECORDS)==6 and len(set(s.RECORDS))==6)
 check('exact unchanged fit/update budget',(s.FIT_ROWS,s.STEPS,s.BATCH,s.OLD_VALID,s.NEW_VALID)==(578090,142,4096,173032,61440) and s.FIT_ROWS%s.BATCH==554)
 torch.set_num_threads(2);torch.set_float32_matmul_precision('medium');s.configure_torch()
 check('CPU path settings enforce4/highest',torch.get_num_threads()==4 and torch.get_float32_matmul_precision()=='highest')
 rng=np.random.default_rng(42);ps=rng.uniform(size=(4,37,5)).astype(np.float32);ps/=ps.sum(2,keepdims=True)
 expected=np.zeros((37,5),np.float64)
 for member in ps:expected+=member
 expected=(expected/4).astype(np.float32)
 check('four-member fixed-order float64 accumulation thenfloat32 exact',np.array_equal(s.fixed_mean(list(ps)),expected))
 reject('three members cannot produce ensemble',lambda:s.fixed_mean(list(ps[:3])))
 invalid=ps.copy();invalid[2,36,4]=np.nan;reject('anymember lastrow NaN refused',lambda:s.fixed_mean(list(invalid)))
 reject('float64 member refused',lambda:s.fixed_mean([ps[0].astype(np.float64),*ps[1:]]))
 def value(x):return dict(min24=x,old5_mean_f1=x,groups={g:dict(system_f1=x,accuracy=x,system_false_positive_rate=.01) for g in s.aug.GROUPS})
 records=[dict(record=n,validation=value(.92 if n=='ensemble-maximin' else .91)) for n in s.RECORDS]
 check('only ensemble promoted despite stronger singleton',s.select_records(records)['candidate']=='ensemble-maximin' and s.select_records(records)['promoted'])
 missing=copy.deepcopy(records);missing.pop(1);reject('missing originalCPU control refused',lambda:s.select_records(missing))
 duplicate=copy.deepcopy(records);duplicate[0]['record']=duplicate[1]['record'];reject('duplicate record refused',lambda:s.select_records(duplicate))
 for control in ['old-cpu-maximin','old-original']:
  z=copy.deepcopy(records);next(r for r in z if r['record']==control)['validation']['min24']=.92
  check('strict control tie rejects '+control,not s.select_records(z)['promoted'])
  z=copy.deepcopy(records);next(r for r in z if r['record']==control)['validation']['groups']['old-natural']['system_f1']=.99
  check('F1 retention against '+control,not s.select_records(z)['promoted'])
 z=copy.deepcopy(records);z[-1]['validation']['min24']=None
 check('undefined objective rejects without singleton fallback',not s.select_records(z)['promoted'] and s.select_records(z)['no_fallback'])
 old=[];new=[]
 for pi,p in enumerate(s.lp.PROFILES):
  for ri,r in enumerate(s.data.ROLES if p!='natural' else s.data.ROLES[:-1]):
   for off in [0,512,1023]:old.append(dict(cache_ix=len(old),sample_ix=len(old),profile=p,role=r,token_ix=10+off,body_offset=off,prompt_ix=pi*5+ri,message_ix=0,split='validation',wrapper='native'))
 for ri,r in enumerate(s.data.ROLES):
  for off in [0,512,1023]:new.append(dict(cache_ix=len(new),sample_ix=6000000000000+len(new),profile='long-full',role=r,token_ix=10+off,body_offset=off,prompt_ix=ri,message_ix=0,split='validation',wrapper='native'))
 old,new=pd.DataFrame(old),pd.DataFrame(new)
 def probabilities(rows):
  p=np.full((len(rows),5),.025,np.float32);p[np.arange(len(rows)),s.probes.labels(rows)]=.9;return p
 po,pn=probabilities(old),probabilities(new);ref=dict(old=po,new=pn)
 check('drift exact/full8groups+totals',len(s.drift_metrics(old,new,ref,ref,.5))==10)
 # A decision-changing tiny drift in an arbitrary non-first row must remain visible.
 changed=po.copy();j=int(np.flatnonzero(old.profile.eq('natural'))[-1]);po[j]=[.50001,.09999,.1,.2,.1];changed=po.copy();changed[j,0]-=.00002;changed[j,1]+=.00002
 d=s.drift_metrics(old,new,dict(old=po,new=pn),dict(old=changed,new=pn),.5)
 check('full drift tracks threshold decision despite tiny probabilitychange',d['old-natural']['original_threshold_decision_changes']==1 and d['all-old']['original_threshold_decision_changes']==1 and d['all-old']['maximum_probability_difference']<5e-4)
 changed[j,0]-=.001;changed[j,1]+=.001
 reject('any fullarray drift exceeds tolerance refused',lambda:s.drift_metrics(old,new,dict(old=po,new=pn),dict(old=changed,new=pn),.5))
 # Source ordering checks make the no-new96-best-selection boundary reviewable.
 w=inspect.getsource(s.worker)
 check('best evidence frozen before new96 probability loop',w.index("data.atomic_json(OUT/'training'")<w.index("for bank,rows in [('old',old_valid),('new',new_valid)]"))
 check('worker epoch has original-only standardizedvalidation',"xv=standardized(old_valid)" in w and 'xv=standardized(new_valid)' not in w and "lp.validation_metrics(old_valid,p)" in w)
 check('worker optimizer only ordinary fiveclass rawCE',"cross_entropy(net.raw(x[ix].contiguous()),y[ix])" in w)
 check('parentsettings beforecontrolandcalibration', 'configure_torch()' in inspect.getsource(s.old_control) and 'configure_torch()' in inspect.getsource(s.calibrate))
 check('completed verifier recalibrates read-only withsixrecords',"calibrate(plan,False)" in inspect.getsource(s.verified_completion) and "records==read(OUT/'records.json')" in inspect.getsource(s.verified_completion))
 check('missingcontrolprobabilitycannotberecomputedbyverifier',"allow_create and not any(probability_path" in inspect.getsource(s.old_control))
 check('supervisor terminate and reap both branches',all(x in inspect.getsource(s.run_workers) for x in ['p.terminate()','p.kill()','p.wait(timeout=10)','log.close()']))
 assert sha(PATH)==START
result=dict(stage='independent_multiseed_source_synthetic_audit',audit_passed=True,source_sha256=START,source_sha256_after=sha(PATH),checks_passed=len(CHECKS),checks=CHECKS,actual_probabilities_read=False,actual_data_read=False,gpu_used=False,frozen_sources_modified=False,limitations=['Source ordering assertions are structural checks, not an executed GPU worker. Actual complete/schema/mathematics audit remains gated on a future terminal run.'])
p=pathlib.Path('/tmp/feature_review_multiseed_source_audit.json');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=sha(p),script_sha256=sha(__file__),**result)))
