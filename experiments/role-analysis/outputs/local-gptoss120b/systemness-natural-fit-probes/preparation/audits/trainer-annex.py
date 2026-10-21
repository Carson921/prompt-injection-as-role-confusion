#!/usr/bin/env python3
"""Independent synthetic/source audit only; no real rows/p/model/source data."""
import copy,hashlib,inspect,json,pathlib,sys,tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=pathlib.Path('/root/prompt-injection-as-role-confusion');sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_natural_fit_probes as s
P=pathlib.Path(s.__file__)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
START=sha(P);CHECKS=[]
def check(n,x=True):assert bool(x),n;CHECKS.append(n)
def reject(n,fn):
 try:fn()
 except(RuntimeError,ValueError,AssertionError,KeyError,FileNotFoundError):CHECKS.append(n)
 else:raise AssertionError('Accepted '+n)
def banned(*a,**k):raise AssertionError('Actual metadata/data/p/GPU APIs forbidden')
ROLES=['system','user','cot','assistant','tool'];PROFILES=['single','multi','natural','long-single','long-full']
GROUPS=['old-'+p for p in PROFILES]+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail','public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
FEASIBLE=['old-'+p for p in PROFILES]+['new-C4-long-full','public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
def labels(frame):return frame.role.map({r:i for i,r in enumerate(ROLES)}).to_numpy(np.int64)
def group_indices(rows):
 old,c4,pub=rows['old'],rows['c4'],rows['public'];g={'old-'+p:('old',np.flatnonzero(old.profile.eq(p))) for p in PROFILES}
 g.update({'old-long-full-tail':('old',np.flatnonzero(old.profile.eq('long-full')&old.body_offset.ge(512))),
  'new-C4-long-full':('c4',np.arange(len(c4))),'new-C4-long-full-tail':('c4',np.flatnonzero(c4.body_offset.ge(512))),
  'public-natural-all':('public',np.arange(len(pub))),'public-natural-oasst':('public',np.flatnonzero(pub.dataset.eq('oasst'))),
  'public-natural-toxicchat0124':('public',np.flatnonzero(pub.dataset.eq('toxicchat0124')))})
 return g

def independent_max33(rows,p):
 grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.r_[p['old'][:,0],p['c4'][:,0],p['public'][:,0]],np.linspace(0,1,301)),1.000001]);curve={}
 for g,(b,ix) in group_indices(rows).items():
  truth=labels(rows[b].iloc[ix])==0;score=p[b][ix,0];n1,n0=int(truth.sum()),int((~truth).sum());a=[]
  for tau in grid:
   mask=score>=float(tau);tp=int((truth&mask).sum());fp=int((~truth&mask).sum());fn=n1-tp;tn=n0-fp
   a.append([tp/(tp+fp) if tp+fp else np.nan,tp/n1,(tp/n1+tn/n0)/2,2*tp/(2*tp+fp+fn),fp/n0])
  curve[g]=np.asarray(a)
 minimum=np.min([curve[g][:,j] for g in GROUPS for j in [0,1,2]],axis=0);minimum[~np.isfinite(minimum)]=-np.inf
 mean=np.mean([curve['old-'+p][:,3] for p in PROFILES],axis=0);feasible=np.max([curve[g][:,4] for g in FEASIBLE],axis=0)<=.05
 options=np.flatnonzero(feasible&np.isfinite(minimum));assert len(options)
 best=sorted(options,key=lambda i:(-float(minimum[i]),-float(mean[i]),float(grid[i])))[0]
 proof=dict(grid_thresholds=len(grid),groups=11,primary_statistics=len(grid)*33,additional_F1_FPR_statistics=len(grid)*22,
  maximum_direct_difference=0.,actual_F32_comparison=True,all_grid_defined_semantics_equal=True,all_grid_feasibility_equal=True,global_argmax_equal=True,
  min33=float(minimum[best]),old5_mean_f1=float(mean[best]),feasible_thresholds=int(feasible.sum()))
 return float(grid[best]),proof,{g:sha_array(v) for g,v in curve.items()}
def sha_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

with patch.object(np,'load',side_effect=banned),patch.object(pd,'read_parquet',side_effect=banned),patch.object(s,'dataset',side_effect=banned),patch.object(s,'source_signature',side_effect=banned),patch.object(s,'static_snapshot',side_effect=banned),patch.object(s.data.runtime,'load_model',side_effect=banned):
 check('Exact8record roles/2arm/157step budgets',len(s.RECORDS)==8 and len(set(s.RECORDS))==8 and s.GROUPS==GROUPS and s.FEASIBLE==FEASIBLE and (s.FIT,s.EXTRA,s.INSTANCES,s.STEPS,s.BATCH)==(578090,63403,641493,157,4096) and 641493%4096==2517)
 fit=pd.DataFrame([dict(cache_ix=i,sample_ix=100+i,role=ROLES[i%5],split='fit') for i in range(25)])
 nf=pd.DataFrame([dict(cache_ix=50+i,sample_ix=4_000_000_000_000+i,source_uid='toy:'+str(i//3),role=r,split='fit') for i,r in enumerate(['assistant','system','user','assistant','user','assistant','system','assistant','system'])])
 import torch
 state=torch.get_rng_state().clone();slots,manifest=s.matched_draw(fit,nf);rng=np.random.RandomState(20516085);wanted=np.empty(len(nf),np.int64)
 for role in ['system','user','assistant']:
  ii=np.flatnonzero(nf.role.eq(role));pool=fit.loc[fit.role.eq(role)].cache_ix.to_numpy();wanted[ii]=pool[rng.choice(len(pool),size=len(ii),replace=True)]
 check('Independent private same-role draw and allslot labels exact',np.array_equal(slots.matched_old_cache_ix,wanted) and np.array_equal(fit.set_index('cache_ix').loc[wanted].role,nf.role) and torch.equal(state,torch.get_rng_state()))
 check('Matched draw stable/no epoch redraw',slots.equals(s.matched_draw(fit,nf)[0]) and manifest==s.matched_draw(fit,nf)[1] and manifest['extra_labels_sha256']==sha_array(labels(nf)))
 old=[];c4=[];pub=[]
 for pi,profile in enumerate(PROFILES):
  for ri,role in enumerate(ROLES if profile!='natural' else ROLES[:-1]):
   for off in [0,512,1023]:old.append(dict(cache_ix=len(old),sample_ix=len(old),profile=profile,role=role,prompt_ix=pi*5+ri,message_ix=0,token_ix=20+off,body_offset=off))
 for ri,role in enumerate(ROLES):
  for off in [0,512,1023]:c4.append(dict(cache_ix=len(c4),sample_ix=6_000_000_000_000+len(c4),profile='long-full',role=role,prompt_ix=ri,message_ix=0,token_ix=20+off,body_offset=off))
 for dataset in ['oasst','toxicchat0124']:
  for ri,role in enumerate(['system','user','assistant']):
   for off in [0,1,2]:pub.append(dict(cache_ix=len(pub),sample_ix=4_000_000_000_000+len(pub),profile='natural-public-sua',dataset=dataset,role=role,prompt_ix=ri,message_ix=ri,token_ix=20+off,body_offset=off))
 rows=dict(old=pd.DataFrame(old),c4=pd.DataFrame(c4),public=pd.DataFrame(pub));p={}
 for b,f in rows.items():
  a=np.full((len(f),5),.025,np.float32);a[np.arange(len(a)),labels(f)]=.9;p[b]=a
 for g,(b,ix) in group_indices(rows).items():bb,jj=s.validation_groups(rows)[g];assert b==bb and np.array_equal(ix,jj)
 check('All11global groups exact/tails/source strata distinct')
 tau,proof,curves=independent_max33(rows,p);t2,p2=s.max33(rows,p)
 check('Independent fullF32 min33 grid/objective/feasible/globalargmax exact',tau==t2 and proof==p2)
 v=s.metrics_at(rows,p,tau)
 check('Public trueSUA Cot/tool N/A while legacy CoT observed',all(v['groups'][g]['recall']['cot'] is None and v['groups'][g]['recall']['tool'] is None for g in s.PUBLIC) and v['groups']['old-natural']['recall']['cot']==1 and v['groups']['old-natural']['recall']['tool'] is None)
 def value(minimum,score=.94):return dict(min33=minimum,old5_mean_f1=score,groups={g:dict(system_f1=score,accuracy=score,system_false_positive_rate=.01) for g in GROUPS})
 records=[dict(record=n,validation=value(.92,.93)) for n in s.RECORDS];records[1]['validation']=value(.95,.95);legacy=value(.9,.92)
 check('Onlynew max33 dualCPU/legacy guards can promote',s.select_records(records,legacy)['promoted'] and s.select_records(records,legacy)['candidate']=='new-max33')
 bad=copy.deepcopy(records);bad[1]['validation']['min33']=.89999999
 check('Absolute90 floor rejects despite relative gain',not s.select_records(bad,legacy)['promoted'] and 'absolute_min33_below_90_percent' in s.select_records(bad,legacy)['rejection_reasons'])
 for name in ['matched-max33','oldhead-max33','ensemble-max33']:
  bad=copy.deepcopy(records);next(r for r in bad if r['record']==name)['validation']['min33']=.95
  check('Exact tie forbidden against '+name,not s.select_records(bad,legacy)['promoted'])
 for name in ['oldhead-oldmean','ensemble-original']:
  bad=copy.deepcopy(records);next(r for r in bad if r['record']==name)['validation']['groups']['old-natural']['system_f1']=.99
  check('Inheritedfixedtau floor enforced '+name,not s.select_records(bad,legacy)['promoted'])
 badlegacy=copy.deepcopy(legacy);badlegacy['groups']['old-natural']['system_f1']=.99
 check('SavedGPU original floor independently blocksCPU drift credit',not s.select_records(records,badlegacy)['promoted'] and 'original-savedGPU:old4_system_f1_loss' in s.select_records(records,badlegacy)['rejection_reasons'])
 for group in ['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']:
  bad=copy.deepcopy(records);bad[1]['validation']['groups'][group]['accuracy']=.90
  check('Everypublic stratum accuracy protection '+group,not s.select_records(bad,legacy)['promoted'])
 history=[dict(epoch=i,instances=641493,steps=157,cumulative_steps=i*157,loss=.7,old5_mean_f1=.8) for i in range(1,11)]
 budget=s.budget(history,1);check('Best/actual157step and2517last evidence',budget['best_steps']==157 and budget['actual_steps']==1570 and budget['actual_instances']==6414930 and budget['last_batch_instances']==2517)
 bad=copy.deepcopy(history);bad[-1]['steps']=158;reject('Wrong update history refuses',lambda:s.budget(bad,1))
 w=inspect.getsource(s.worker)
 check('Model/history/GPUold freeze precedes anyC4/public prediction',w.index("data.atomic_json(OUT/'training'")<w.index("for b,frame in rows.items()") and "new_C4_predictions_read=False,public_validation_predictions_read=False" in w)
 check('Matchedonlysource vs newnatural features preserve sharedlabels',"'natural' if arm=='new' else 'old'" in w and "probes.labels(fit),probes.labels(nf)" in w)
 check('No new/public earlystop feature standardization',"xv=standardized('old',ov.cache_ix.to_numpy())" in w and "lp.validation_metrics(ov,p)" in w)
 check('CPUsettings applyworker parentandcontrols',all('ms.configure_torch()' in inspect.getsource(f) for f in [s.worker,s.train,s.calibrate,s.inherited_controls]))
 check('FIT-only portability no validation sampling',"chosen=nf.iloc[:256]" in inspect.getsource(s.parity) and "banks['natural']" in inspect.getsource(s.parity))
 check('Readonlycomplete cannot silentlycreate inheritedcontrols',"calibrate(plan,False)" in inspect.getsource(s.verified_completion) and "allow_create and not any(ppath" in inspect.getsource(s.inherited_controls))
 check('Supervisor cleanup terminates/reaps bothworkers',all(x in inspect.getsource(s.run_workers) for x in ['p.terminate()','p.kill()','p.wait(timeout=10)','log.close()']))
 check('Source unchanged',sha(P)==START)
folder=pathlib.Path(tempfile.mkdtemp(prefix='feature-review-natural-fit-source-audit-',dir='/tmp'));target=folder/'report.json'
result=dict(stage='independent_natural_fit_source_synthetic_audit',audit_passed=True,source_sha256=START,source_sha256_after=sha(P),checks_passed=len(CHECKS),checks=CHECKS,independent_grid_proof=proof,
 direct_grid_group_curve_sha256=curves,actual_metadata_or_probabilities_read=False,actual_source_text_read=False,gpu_used=False,formal_outputs_modified=False,
 limitation='Source ordering assertions audit checkpoint-before-new-validation structure; no worker/GPU/actual-result execution is claimed.')
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(target),report_sha256=sha(target),auditor_source_sha256=sha(__file__),**result)))
