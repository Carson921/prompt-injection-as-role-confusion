"""Synthetic/source-only independent annex for frozen prospective ensemble v1.
Never calls run(), reads an NPZ, verifies production completion, or reads results.
"""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
import numpy as np
import pandas as pd
SOURCE=Path('/tmp/feature_review_validation_ensemble.py')
PLAN=Path('/tmp/feature_review_validation_ensemble_protocol.json')
EXPECTED='fc09b16b690e9f45cf31ee80d68c42400f8c38b0dbe1c7c12c39095e17edc6c3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,s):
    if not x:raise RuntimeError(s)
start=sha(SOURCE);need(start==EXPECTED,'Frozen executor changed.')
module=importlib.util.spec_from_file_location('synthetic_ensemble_annex_target',SOURCE);e=importlib.util.module_from_spec(module);module.loader.exec_module(e)
def forbidden(*args,**kwargs):raise RuntimeError('Forbidden real-data/runner API called.')
e.run=forbidden
hp,lp,probes=e.helpers();hp.verified_completion=forbidden
checks=[]
def check(name,value):need(value,name);checks.append(name)
rows=[]
for profile in e.PROFILES:
    for role in e.ROLES:
        prompt=len(rows)//2
        for offset in [0,512]:rows.append(dict(profile=profile,role=role,prompt_ix=prompt,message_ix=0,token_ix=80+offset,document_id=prompt,group_id=f'toy-{prompt}',source_set='synthetic',split='validation',wrapper='native'))
rows=pd.DataFrame(rows);y=probes.labels(rows)
p=np.full((len(rows),5),.04,np.float32);p[np.arange(len(rows)),y]=.84;q=p[:,[0,2,1,4,3]].copy()
check('All-five alpha endpoints preserve exact F32 p',np.array_equal(e.mix(p,q,0),q) and np.array_equal(e.mix(p,q,1),p))
for alpha in e.ALPHAS:
    m=e.mix(p,q,alpha);e.validate_probability(m,len(rows))
    t,_=e.choose_threshold(rows,m,'mean-f1');need(e.metrics_at(rows,m,t)==lp.validation_metrics(rows,m),'Legacy mean threshold helper mismatch.')
check('Legacy mean helper parity all5 alpha weights',True)
g=e.group_indices(rows);expected=rows.index[rows.profile.eq('long-full') & rows.token_ix.eq(592)].to_numpy()
check('Tail uses full-group global row indices and >=512 actual offsets',np.array_equal(g['long-full-tail'],expected))
for alpha in e.ALPHAS:
    m=e.mix(p,q,alpha);grid,values,mean,minimum,feasible=e.threshold_curve(rows,m)
    eligible=np.flatnonzero(feasible & np.isfinite(minimum));best=max(eligible,key=lambda k:(float(minimum[k]),float(mean[k]),-float(grid[k])))
    t,c=e.choose_threshold(rows,m,'maximin-pr-bacc');need(t==float(grid[best]),'Maximin ordering changed.')
check('New maximin uses18 values and stated min/mean/smaller-tau ordering',True)
base=e.metrics_at(rows,p,.5);same=copy.deepcopy(base)
check('Identical result passes both controls',hp.promotion_guard(same,base)['eligible'] and hp.promotion_guard(same,base)['eligible'])
for label,field,drop,reason in [('F1','system_f1',.020001,'original_profile_system_f1_loss'),('Accuracy','accuracy',.010001,'original_profile_five_way_accuracy_loss')]:
    bad=copy.deepcopy(base);bad['profiles']['single'][field]-=drop
    v=hp.promotion_guard(bad,base);need(not v['eligible'] and reason in v['rejection_reasons'],label+' guard relaxed.')
check('Original-profile F1 and five-way accuracy negative guards reject',True)
for name in ['long-full','long-full-tail']:
    bad=copy.deepcopy(base);target=bad['long_full_tail'] if name.endswith('tail') else bad['profiles']['long-full'];target['system_f1']-=.000001
    need(not hp.promotion_guard(bad,base)['eligible'],'Long full/tail regression guard relaxed.')
check('Full and tail any substantive F1 loss reject',True)
bad=copy.deepcopy(base);bad['profiles']['single']['system_false_positive_rate']=.050001
check('Five-profile FPR guard rejects negative result',not hp.promotion_guard(bad,base)['eligible'])
# Independent check of NumPy2 scalar threshold semantics, on adjacent F32 values.
small=[];prob=[]
for profile in e.PROFILES:
    for role in ['system','user']:
        prompt=len(small)//2
        for offset in [0,512]:
            small.append(dict(profile=profile,role=role,prompt_ix=prompt,message_ix=0,token_ix=80+offset,document_id=prompt,group_id=f'small-{prompt}',source_set='synthetic',split='validation',wrapper='native'))
            ps=(np.float32(.5) if offset==0 else np.nextafter(np.float32(.5),np.float32(1))) if role=='system' else np.float32(.01)
            prob.append([ps,np.float32(1)-ps,0,0,0])
small=pd.DataFrame(small);prob=np.asarray(prob,np.float32);grid,values,mean,minimum,feasible=e.threshold_curve(small,prob)
indices=e.group_indices(small);labels=probes.labels(small)
mismatches=[]
for k,t in enumerate(grid):
    if not (float(np.float32(.5))<t<float(np.nextafter(np.float32(.5),np.float32(1)))):continue
    for profile,ix in indices.items():
        truth=labels[ix]==0;detected=prob[ix,0]>=float(t);tp=int(np.count_nonzero(truth&detected));r=tp/int(truth.sum())
        if r!=float(values[profile]['recall'][k]):mismatches.append(dict(threshold=float(t),effective_threshold_float32=float(np.float32(t)),profile=profile,curve_recall=float(values[profile]['recall'][k]),actual_metric_recall=r))
check('Synthetic-only adjacent-F32 threshold disagreement independently reproduced',bool(mismatches))
check('No executor or plan source changed during annex',sha(SOURCE)==start)
report=dict(executor_sha256=start,protocol_sha256=sha(PLAN),numpy_version=np.__version__,checks=checks,checks_passed=len(checks),threshold_boundary_examples=mismatches[:6],
    verdict='Must revise new maximin count semantics in a versioned protocol before treating it as exact actual-metric optimization. Legacy mean objective must preserve existing helper semantics.',
    probabilities='Only toy in-memory arrays; no production NPZ/readout/test artifacts opened.',actual_runner_called=False,verified_completion_called=False,gpu_used=False,formal_files_modified=False)
p=Path('/tmp/confirmation-review-ensemble-annex.json');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(p),checks_passed=len(checks),boundary_disagreements=len(mismatches),example=mismatches[0]),indent=2))
