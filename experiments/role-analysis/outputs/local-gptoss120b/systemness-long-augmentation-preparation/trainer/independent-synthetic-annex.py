"""Independent small CPU annex: no real p/new text/GPU/training."""
from pathlib import Path
from unittest.mock import patch
import ast
import copy
import hashlib
import json
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,'/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import systemness_long_augmentation_probes as t
EXPECTED='220d7d24a8a640198351ce925093fe2ba135af42ab9ccbb797a7d5b3431699e5'
start=t.data.digest(t.__file__);assert start==EXPECTED
folder=Path(tempfile.mkdtemp(prefix='long-data-augmentation-trainer-audit-',dir='/tmp'));checks=[]
def reject(fn,label):
    try:fn()
    except (RuntimeError,AssertionError,FileNotFoundError):checks.append(label)
    else:raise AssertionError(label)
def forbidden(*a,**k):raise AssertionError('Real p/model/dataset read forbidden')

# Actual metadata-only declaration must survive a hard ban on NumPy loading.
with patch.object(t,'OUT',folder/'declared'),patch.object(t,'CACHE',folder/'declared-cache'),patch('numpy.load',side_effect=forbidden):
    declaration=t.declare();t.verify_declaration()
    assert declaration['static']['probability_arrays_read'] is False
checks.append('real_static_snapshot_and_declare_never_numpy_load_probability_or_model')

# Labels/offsets are matched despite shuffled physical cache order.
def frame(documents,fit,samples,offset_id,source):
    entries=[]
    for d in range(documents):
        split='fit' if d<fit else 'validation';n=samples if split=='fit' else 2
        for ri,role in enumerate(t.data.ROLES):
            for off in t.data.uniform_indices(1024,n):
                i=len(entries);entries.append(dict(cache_ix=i,sample_ix=offset_id+i,document_id=(6000000 if source=='long-augmentation' else 0)+d,
                    group_id=source+str(d),role=role,split=split,source_set=source,profile='long-full',wrapper='native',
                    token_ix=20+ri+int(off),body_offset=int(off),prompt_ix=d*5+ri,message_ix=0))
    return pd.DataFrame(entries)
old=frame(4,3,4,2000000000,'long-full');new=frame(7,5,4,6000000000000,'long-augmentation')
before=torch.get_rng_state().clone();rng=np.random.get_state()
parts=t.augmentation_layout(old,new,expected=dict(old_pool=3,new_fit=5,new_valid=2,samples_fit=4,samples_valid=2))
fresh,matched,manifest=parts[3:6]
assert torch.equal(before,torch.get_rng_state()) and all(np.array_equal(a,b) for a,b in zip(rng,np.random.get_state()))
assert np.array_equal(old.iloc[matched].role,new.iloc[fresh].role)
assert np.array_equal(old.iloc[matched].body_offset,new.iloc[fresh].body_offset)
assert all(old.iloc[matched].split=='fit')
r=np.random.RandomState(t.DRAW_SEED);draw=np.r_[r.permutation(3),r.permutation(3)[:2]]
assert [slot['old_document_id'] for slot in manifest['slots']]==draw.tolist()
assert t.augmentation_layout(old,new,expected=dict(old_pool=3,new_fit=5,new_valid=2,samples_fit=4,samples_valid=2))[5]==manifest
r=np.random.RandomState(t.DRAW_SEED);draw=np.r_[r.permutation(132),r.permutation(132)[:124]]
assert len(draw)==256 and np.count_nonzero(np.bincount(draw)==2)==124 and np.count_nonzero(np.bincount(draw)==1)==8
checks.append('independent_fixed132plus124_recipe_and_toy_same_label_offset_slots_no_rng_leak')

# Independently enumerate all actual F32 threshold decisions without trainer
# group/metric helpers, including adjacent representable probability values.
offsets=[0,146,292,438,584,730,876,1023];oldv=[];newv=[]
for pi,profile in enumerate(t.lp.PROFILES):
    for ri,role in enumerate(t.data.ROLES):
        for off in offsets:oldv.append(dict(profile=profile,role=role,prompt_ix=pi*5+ri,message_ix=0,token_ix=100+off,
            split='validation',wrapper='native',sample_ix=len(oldv)))
for ri,role in enumerate(t.data.ROLES):
    for off in offsets:newv.append(dict(profile='long-full',role=role,prompt_ix=100+ri,message_ix=0,token_ix=100+off,
        body_offset=off,split='validation',wrapper='native',sample_ix=6000000000000+len(newv)))
oldv,newv=pd.DataFrame(oldv),pd.DataFrame(newv)
rng=np.random.default_rng(113)
def probability(rows):
    y=rows.role.map({r:i for i,r in enumerate(t.data.ROLES)}).to_numpy();p=np.zeros((len(rows),5),np.float32)
    for i,label in enumerate(y):
        options=np.array([.41,np.nextafter(np.float32(.41),np.float32(1)),.7,.9] if label==0 else [.02,.15,.39,.41],np.float32)
        ps=options[rng.integers(4)];p[i,0]=ps
        p[i,1:]=(1-ps)/4
        if label:p[i,1:]=(1-ps)*.2/3;p[i,label]=(1-ps)*.8
    return p
po,pn=probability(oldv),probability(newv)
actual,proof=t.maximin_threshold(oldv,newv,po,pn)
grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.r_[po[:,0],pn[:,0]],np.linspace(0,1,301)),1.000001])
gs={}
for profile in t.lp.PROFILES:gs['old-'+profile]=(oldv,po,np.flatnonzero(oldv.profile.eq(profile)))
ix=np.flatnonzero(oldv.profile.eq('long-full')&oldv.token_ix.ge(612));gs['old-long-full-tail']=(oldv,po,ix)
gs['new-long-full']=(newv,pn,np.arange(len(newv)));gs['new-long-full-tail']=(newv,pn,np.flatnonzero(newv.body_offset.ge(512)))
assert set(gs)==set(t.GROUPS);valid=[]
for tau in grid:
    values={};minimum=[]
    for name,(rows,p,ix) in gs.items():
        truth=rows.iloc[ix].role.eq('system').to_numpy();d=p[ix,0]>=float(tau)
        a=int((truth&d).sum());b=int((~truth&d).sum());c=int((truth&~d).sum());z=int((~truth&~d).sum())
        precision=a/(a+b) if a+b else None;recall=a/(a+c);ba=(recall+z/(z+b))/2
        values[name]=dict(P=precision,R=recall,BAcc=ba,FPR=b/(b+z),F1=2*a/(2*a+b+c))
        if precision is None:minimum.append(-np.inf)
        else:minimum.extend([precision,recall,ba])
    if any(values[g]['FPR']>.05 for g in t.FEASIBLE_GROUPS) or not np.isfinite(min(minimum)):continue
    mean=float(np.mean([values['old-'+p]['F1'] for p in t.lp.PROFILES]));valid.append((min(minimum),mean,-float(tau)))
best=max(valid);assert actual==-best[2] and proof['min24']==best[0] and proof['old5_mean_f1']==best[1]
checks.append('independent_all8groups_F32_direct_grid_argmax_adjacent_values_and_ties')

# Exact guards/no fallback, independently assembled diagnostic records.
def validation(minimum,mean=.9):
    return dict(min24=minimum,old5_mean_f1=mean,groups={g:dict(system_f1=.9,accuracy=.9,system_false_positive_rate=.01) for g in t.GROUPS})
records=[dict(record=m+'-'+o,validation=validation(.90)) for m in t.MODELS for o in t.OBJECTIVES]
candidate=next(row for row in records if row['record']=='new-maximin');candidate['validation']=validation(.93)
next(row for row in records if row['record']=='matched-maximin')['validation']['min24']=.92
assert t.select_records(records)['promoted']
for group,field,value in [('old-single','system_f1',.879),('old-multi','accuracy',.889),
    ('old-long-full-tail','system_f1',.89999),('new-long-full','system_false_positive_rate',.05001)]:
    bad=copy.deepcopy(records);next(row for row in bad if row['record']=='new-maximin')['validation']['groups'][group][field]=value
    assert not t.select_records(bad)['promoted']
bad=copy.deepcopy(records);next(row for row in bad if row['record']=='new-maximin')['validation']['min24']=.92
next(row for row in bad if row['record']=='new-old-mean')['validation']['min24']=1.
assert not t.select_records(bad)['promoted'] and t.select_records(bad)['candidate']=='new-maximin'
checks.append('all_guard_failures_and_strict_min24_tie_reject_without_diagnostic_fallback')

# Completion branch is read-only, even with all train/publish operations banned.
out=folder/'completed';out.mkdir();(out/'complete.json').write_text('{}');before=(out/'complete.json').read_bytes()
with patch.object(t,'OUT',out),patch.object(t,'no_competing_jobs',return_value=None), \
    patch.object(t,'verify_declaration',return_value={}),patch.object(t,'source_signature',return_value={}), \
    patch.object(t,'verified_completion',return_value={}),patch.object(t,'dataset',side_effect=forbidden), \
    patch.object(t,'run_workers',side_effect=forbidden),patch.object(t,'publish',side_effect=forbidden):t.train()
assert (out/'complete.json').read_bytes()==before
checks.append('completed_train_route_verifies_without_dataset_workers_publish_or_overwrite')

tree=ast.parse(Path(t.__file__).read_text());fn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
static_calls={ast.unparse(n.func) for n in ast.walk(fn['static_snapshot']) if isinstance(n,ast.Call)}
assert not static_calls&{'np.load','read_probability','candidate_record','hp.verified_completion','dataset','source_signature'}
worker=ast.get_source_segment(Path(t.__file__).read_text(),fn['worker'])
assert worker.index("data.atomic_json(OUT / 'training'")<worker.index("new_p = hp.predict_raw")
assert "result = lp.validation_metrics(old_valid, p)" in worker
assert "torch.manual_seed(data.SEED)" in worker and "manual_seed(data.SEED)" in worker
checks.append('source_static_metadata_only_and_old_only_best_evidence_before_new_val_prediction')
assert t.data.digest(t.__file__)==start
report=dict(source_sha256=start,source_end_sha256=t.data.digest(t.__file__),independent_checks=checks,checks_passed=len(checks),
    grid_thresholds=len(grid),independent_grid_selected_threshold=actual,no_must_fix=True,
    real_probability_arrays_read=False,new_source_text_read=False,gpu_used=False,formal_outputs_changed=False)
p=folder/'report.json';p.write_text(json.dumps(report,indent=2));print(json.dumps(dict(report_path=str(p),**report)))
