"""Bounded independent source/toy annex only; never read actual artifacts."""
import ast,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
p=Path('/tmp/feature_review_multiseed_results_audit.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=sha(p);assert start=='dd3f7b9328a15b26d2fe0c945b6da8385816e929a19a3eaa60dd94405edee8cb'
spec=importlib.util.spec_from_file_location('independent_multiseed_result_audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-multiseed-auditor-',dir='/tmp'));checks=[]
def check(n,v):assert bool(v),n;checks.append(n)
def forbidden(*a,**kw):raise AssertionError('Formal read/producer helper invoked')
def reject(f):
    try:f()
    except (RuntimeError,AssertionError):return True
    return False
with patch.object(np,'load',side_effect=forbidden),patch.object(pd,'read_parquet',side_effect=forbidden):
    pval=np.array([[.4,.5,.04,.04,.02],[.3,.2,.4,.08,.02],[.6,.1,.05,.2,.05],[.1,.2,.1,.2,.4],[.5,.1,.1,.2,.1]],np.float32)
    y=np.array([0,1,3,3,0]);t=.35;r=m.metrics(pval,y,t);truth=y==0;detected=pval[:,0]>=t;guess=pval.argmax(1)
    matrix=np.zeros((5,5),int)
    for a,b in zip(y,guess):matrix[a,b]+=1
    tp=sum(truth&detected);fp=sum(~truth&detected);fn=sum(truth&~detected);tn=sum(~truth&~detected)
    check('Independent confusion/threshold P/R/F1 and unsupported role handling',r['confusion']==matrix.tolist() and r['system_precision']==tp/(tp+fp)
        and r['system_recall']==tp/(tp+fn) and r['system_f1']==2*tp/(2*tp+fp+fn) and r['recall']['cot'] is r['recall']['tool'] is None)
    check('Tied-score AUROC/AP and truth-supported macro match independent definition',abs(r['system_auroc']-roc_auc_score(truth,pval[:,0]))<1e-14
        and abs(r['system_average_precision']-average_precision_score(truth,pval[:,0]))<1e-14
        and r['macro_f1']==np.mean([2*matrix[i,i]/(matrix[i].sum()+matrix[:,i].sum()) for i in range(5) if matrix[i].sum()]))
    raw=np.random.default_rng(5).uniform(size=(4,19,5)).astype(np.float32);raw/=raw.sum(2,keepdims=True)
    avg=np.mean(raw.astype(np.float64),axis=0).astype(np.float32)
    check('Fixed four-member F64 probability mean cast F32 exact / no missing member',np.array_equal(m.fixed_mean(list(raw)),avg) and reject(lambda:m.fixed_mean(list(raw[:3]))))
    def value(n):return dict(min24=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in m.GROUPS})
    records=[dict(record=n,validation=value(.94 if n=='ensemble-maximin' else .92)) for n in m.RECORDS]
    # Diagnostic singleton being stronger must never alter the predeclared candidate.
    for r0 in records:
        if r0['record'].startswith('seed-'):r0['validation']['min24']=.99
    selected=m.selection(records);tied=json.loads(json.dumps(records));tied[-1]['validation']['min24']=.92
    check('Only ensemble against dual old controls / no best-singleton fallback',selected['candidate']=='ensemble-maximin' and selected['promoted']
        and not m.selection(tied)['promoted'] and m.selection(tied)['candidate']=='ensemble-maximin')
    bad=json.loads(json.dumps(records));bad[-1]['validation']['groups']['old-natural']['accuracy']=.90
    check('Natural five-class hardfloor independently rejects',not m.selection(bad)['promoted'])
    old,new,po,pn=m.synthetic();reference=po.copy();i=int(np.flatnonzero(old.profile.eq('natural'))[0]);reference[i]=[.50001,.09999,.1,.2,.1]
    cpu=reference.copy();cpu[i,0]-=.00002;cpu[i,1]+=.00002
    d=m.drift(old,new,{'old':reference,'new':pn},{'old':cpu,'new':pn},.5)
    check('Full drift preserves small threshold changes / all8 plus2 totals',len(d)==10 and d['old-natural']['original_threshold_decision_changes']==1
        and d['old-natural']['argmax_changes']==0 and d['old-natural']['maximum_probability_difference']<5e-4)
    cpu[i,0]-=.001;cpu[i,1]+=.001
    check('Frozen drift tolerance rejects beyond5e-4',reject(lambda:m.drift(old,new,{'old':reference,'new':pn},{'old':cpu,'new':pn},.5)))
    h=[dict(epoch=i,instances=578090,steps=142,cumulative_steps=i*142,loss=.5,old5_mean_f1=.8) for i in range(1,11)]
    b=m.history_audit(h,1)
    check('142 updates /best and actual counts /earlystop continuation guard',b['best_steps']==142 and b['actual_steps']==1420
        and reject(lambda:m.history_audit(h+[dict(h[-1],epoch=11,cumulative_steps=1562)],1)))
    with patch.object(m,'processes',return_value=[999]),patch.object(m,'read',side_effect=forbidden),patch.object(m,'digest',side_effect=forbidden):
        check('Live-job gate precedes every actual artifact/hash/cache read',reject(lambda:m.actual(folder/'never.json')))
    with patch.object(m,'OUT',folder/'no-output'),patch.object(m,'processes',return_value=[]),patch.object(m,'read',side_effect=forbidden),patch.object(m,'digest',side_effect=forbidden):
        check('No livejobs but missing complete gate still precedes reads',reject(lambda:m.actual(folder/'never.json')))
    actual=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='actual')
    calls=[]
    for n in ast.walk(actual):
        if isinstance(n,ast.Call):
            f=n.func;parts=[]
            while isinstance(f,ast.Attribute):parts.append(f.attr);f=f.value
            if isinstance(f,ast.Name):parts.append(f.id)
            name='.'.join(reversed(parts))
            if name.startswith(('production.','hp.')):calls.append(name)
    check('Actual AST has no production/hp function calls and remains developer-only',not calls and m.RECORDS==['old-original','old-cpu-maximin','seed-20261022-maximin','seed-20261023-maximin','seed-20261024-maximin','ensemble-maximin'])
assert sha(p)==start
result={'source_path':str(p),'source_sha256':start,'source_sha256_after':sha(p),'annex_source_sha256':sha(__file__),
    'checks_passed':len(checks),'checks':checks,'actual_probabilities_or_artifacts_read':False,'gpu_used':False,'formal_outputs_written':False}
(folder/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'report':str(folder/'report.json'),'report_sha256':sha(folder/'report.json'),**result}))
