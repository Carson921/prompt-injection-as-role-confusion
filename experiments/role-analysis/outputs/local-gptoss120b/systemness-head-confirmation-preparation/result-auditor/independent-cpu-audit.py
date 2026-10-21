"""Synthetic/source-only independent audit; never executes the actual audit."""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, log_loss, roc_auc_score)

SOURCE=Path('/tmp/long_data_head_confirmation_result_audit.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial=sha(SOURCE)
spec=importlib.util.spec_from_file_location('c4auditor_under_review',SOURCE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
assert 'systemness_head_confirm' not in sys.modules
checks=['import_has_no_production_confirmation_module_or_data_access'];maximum=0.
rng=np.random.default_rng(20261021)
for n in [10,37,1001]:
    for dtype in [np.float32,np.float64]:
        for tied in [False,True]:
            z=rng.normal(size=(n,5));p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            if tied:p[:]=.2
            p=p.astype(dtype);y=rng.integers(0,5,size=n);y[:5]=np.arange(5)
            threshold=.2 if tied else .41935353535353535
            result=a.token_metrics(p,y,threshold);pred=p.argmax(1);s=y==0;detected=p[:,0]>=threshold
            cm=confusion_matrix(y,pred,labels=range(5))
            truth=dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=np.unique(y),average='macro'),
                system_auroc=roc_auc_score(s,p[:,0]),system_average_precision=average_precision_score(s,p[:,0]),
                system_accuracy=accuracy_score(s,detected),system_f1=f1_score(s,detected),
                nll=log_loss(y,p,labels=range(5)))
            assert result['confusion']==cm.tolist()
            for key,value in truth.items():
                error=abs(result[key]-value);maximum=max(maximum,error)
                assert error<2e-6,(n,dtype,tied,key,result[key],value)
checks.append('12_random_or_tied_F32_F64_multiclass_and_binary_sklearn_parity')
for scores,truth in [(np.array([.2,.2,.3,.3,.8]),np.array([1,0,0,1,0],bool)),
                     (np.ones(5),np.array([0,1,0,0,1],bool))]:
    auc,ap=a.independently_rank_metrics(truth,scores)
    assert abs(auc-roc_auc_score(truth,scores))<1e-12
    assert abs(ap-average_precision_score(truth,scores))<1e-12
checks.append('unequal_tie_blocks_AUROC_AP_exact')
# Preserve global indices through tail/position probability gathers.
parts=[]
for profile,length in [('single',3),('long-full',1024)]:
    for doc in [0,1]:
        for i,role in enumerate(a.ROLES):
            parts.append(pd.DataFrame(dict(group_id='fixture-'+str(doc),document_id=doc,profile=profile,wrapper='native',
                role=role,message_ix=i,body_offset=np.arange(length),token_ix=np.arange(length)+3)))
rows=pd.concat(parts,ignore_index=True);p=np.full((len(rows),5),.025)
y=a.labels(rows);p[np.arange(len(rows)),y]=.9
values={name:p.copy() for name in a.NAMES}
fault=(rows.profile.eq('long-full')&rows.role.eq('system')&rows.body_offset.ge(512)).to_numpy()
values['head-selected'][fault]=[.1,.8,.025,.05,.025]
metrics,docs,projections,positions=a.independently_summarize(rows,values,{name:.5 for name in a.NAMES},strict=False)
tail={r['model']:r for r in metrics if r['profile']=='long-full-tail'}
assert tail['head-selected']['system_recall']==0 and tail['dense128']['system_recall']==1
assert tail['head-selected']['tokens']==2*5*512
assert int(docs[(docs.model=='head-selected')&docs.profile.eq('long-full-tail')].fn.sum())==1024
checks.append('full_tail_uses_global_probability_indices_and_fixed_model_columns')
# Reproduce paired source draws independently for a small unequal-count case.
records=[]
for profile in ['single','long-full-tail']:
    for mi,name in enumerate(a.NAMES):
        for doc in range(4):
            tp=8+doc+mi;fn=4-mi;fp=doc%2;tn=20+doc
            records.append(dict(profile=profile,model=name,group_id='source-'+str(doc),wrapper='native',
                tp=tp,fp=fp,fn=fn,tn=tn,correct=tp+tn-1,tokens=tp+fp+fn+tn))
frame=pd.DataFrame(records);comparisons,intervals,_=a.independent_bootstrap(frame,strict=False)
def calculate(c):
    tp,fp,fn,tn,correct,total=np.moveaxis(c,-1,0)
    return dict(system_precision=tp/(tp+fp),system_recall=tp/(tp+fn),system_f1=2*tp/(2*tp+fp+fn),
        system_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,system_false_positive_rate=fp/(fp+tn),accuracy=correct/total)
for pi,profile in enumerate(sorted(frame.profile.unique())):
    draw=np.random.RandomState(a.SEED+pi).randint(4,size=(2000,4));point={};samples={}
    for name in a.NAMES:
        c=frame[(frame.profile==profile)&(frame.model==name)].sort_values('group_id')[a.FIELDS].to_numpy(float)
        point[name]=calculate(c.sum(0));samples[name]=calculate(c[draw].sum(1))
        for field,value in point[name].items():
            record=next(r for r in intervals if r['profile']==profile and r['model']==name and r['metric']==field)
            lo,hi=np.quantile(samples[name][field],[.025,.975])
            assert np.allclose([record['value'],record['ci95_low'],record['ci95_high']],[value,lo,hi],atol=1e-14,rtol=0)
    for record in [r for r in comparisons if r['profile']==profile]:
        before=record['baseline_model'];after=record['comparison_model'];field=record['metric']
        lo,hi=np.quantile(100*(samples[after][field]-samples[before][field]),[.025,.975])
        assert np.allclose([record['gain_percentage_points'],record['ci95_low'],record['ci95_high']],
            [100*(point[after][field]-point[before][field]),lo,hi],atol=1e-14,rtol=0)
checks.append('paired2000_draws_all_six_metrics_four_models_six_pairs_exact')
# Simulate the actual entry gate using only a temporary fake h module.
folder=Path(tempfile.mkdtemp(prefix='feature-review-c4-auditor-gate-',dir='/tmp'));calls=[]
def forbidden():calls.append('load_data');raise RuntimeError('synthetic post-terminal sentinel')
fake=SimpleNamespace(OUT=folder,NAMES=a.NAMES,BOOTSTRAPS=a.BOOTSTRAPS,
    data=SimpleNamespace(ROLES=a.ROLES,SEED=20261021),load_data=forbidden)
sys.modules['systemness_head_confirm']=fake
try:
    for state in ['no-complete','nonterminal']:
        if state=='nonterminal':
            (folder/'complete.json').write_text('{}');(folder/'status.json').write_text('{"stage":"inference"}')
        try:a.actual()
        except RuntimeError:pass
        else:raise AssertionError('Partial gate accepted')
        assert not calls
    (folder/'status.json').write_text('{"stage":"complete"}')
    try:a.actual()
    except RuntimeError as error:assert str(error)=='synthetic post-terminal sentinel'
    else:raise AssertionError('Terminal sentinel not reached')
    assert calls==['load_data']
finally:sys.modules.pop('systemness_head_confirm')
checks.append('actual_terminal_file_and_status_gates_before_load_data_or_probability_access')
assert sha(SOURCE)==initial
result=dict(source_sha256_at_start=initial,source_sha256_at_end=sha(SOURCE),checks=checks,checks_passed=len(checks),
    maximum_sklearn_difference=maximum,actual_result_audit_run=False,current_c4_artifacts_read=False,
    gpu_run=False,formal_files_modified=False,limitations='Audits source and synthetic arrays only; does not certify actual output yet.')
output=Path('/tmp/feature_review_c4_auditor_audit.json');output.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(dict(report=str(output),sha256=sha(output),**result),indent=2))
