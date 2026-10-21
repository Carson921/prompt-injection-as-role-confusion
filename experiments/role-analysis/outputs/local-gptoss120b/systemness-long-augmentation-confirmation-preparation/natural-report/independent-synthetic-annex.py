"""Pure synthetic/source-only independent report audit; no actual APIs/data."""
import hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
BASE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis');sys.path.insert(0,str(BASE))
source=BASE/'systemness_aug_natural_report.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
start=sha(source);spec=importlib.util.spec_from_file_location('independent_aug_natural_report',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-aug-natural-report-',dir='/tmp'));checks=[]
def check(name,value):
    assert bool(value),name
    checks.append(name)
def forbidden(*a,**kw):raise AssertionError('Real reader/inference invoked')
def ratios(a):
    a=np.asarray(a,dtype=np.float64)
    tp,fp,fn,tn,correct,total=np.moveaxis(a[...,:6],-1,0)
    def div(n,d):return np.divide(n,d,out=np.full(np.shape(n),np.nan),where=d>0)
    out={'system_precision':div(tp,tp+fp),'system_recall':div(tp,tp+fn),'system_f1':div(2*tp,2*tp+fp+fn),
        'system_balanced_accuracy':(div(tp,tp+fn)+div(tn,tn+fp))/2,'system_false_positive_rate':div(fp,fp+tn),'accuracy':div(correct,total)}
    for k,r in enumerate(m.OBSERVED):out['role_recall_'+r]=div(a[...,6+2*k],a[...,7+2*k])
    return out
maxdiff=0.
with patch.object(np,'load',side_effect=forbidden),patch.object(pd,'read_parquet',side_effect=forbidden):
    # False CoT/tool argmax must remain errors, not fabricated observed truth.
    p=np.array([[.4,.5,.04,.04,.02],[.3,.2,.4,.08,.02],[.6,.1,.05,.2,.05],[.1,.2,.1,.2,.4],[.5,.1,.1,.2,.1]],np.float32)
    roles=['system','user','assistant','assistant','system'];threshold=.35
    result,vector,pred,detected=m.token_metrics(p,roles,threshold);y=np.array([m.data.ROLES.index(r) for r in roles]);matrix=np.zeros((5,5),np.int64)
    for truth,guess in zip(y,p.argmax(1)):matrix[truth,guess]+=1
    positive=y==0;decision=p[:,0]>=threshold
    direct=np.array([sum(positive&decision),sum(~positive&decision),sum(positive&~decision),sum(~positive&~decision),sum(y==p.argmax(1)),len(y)]+
        [v for r in m.OBSERVED for v in (sum((y==m.data.ROLES.index(r))&(p.argmax(1)==y)),sum(y==m.data.ROLES.index(r)))])
    check('Five-class confusion / binary threshold / SUA recalls independently exact',np.array_equal(vector,direct) and result['confusion']==matrix.tolist()
        and all(np.isclose(result[k],v) for k,v in ratios(direct).items()) and result['role_support']['cot']==result['role_support']['tool']==0
        and result['recall']['cot'] is result['recall']['tool'] is None)
    f1=np.divide(2*matrix.diagonal(),matrix.sum(0)+matrix.sum(1),out=np.zeros(5),where=matrix.sum(0)+matrix.sum(1)>0)
    check('Truth-supported macro with false-positive CoT/tool differs from union diagnostic',result['macro_f1']==f1[matrix.sum(1)>0].mean()
        and result['macro_f1']!=f1[(matrix.sum(0)+matrix.sum(1))>0].mean())
    check('Binary system recall is separate from five-class argmax recall',result['system_recall']!=result['role_recall_system'])
    check('System AUROC/AP independently match sklearn',np.isclose(result['system_auroc'],roc_auc_score(positive,p[:,0]))
        and np.isclose(result['system_average_precision'],average_precision_score(positive,p[:,0])))
    metadata,rows,ledger,physical,thresholds=m.synthetic_data();metrics,counts,messages=m.summarize(rows,physical,thresholds)
    check('24 metric groups / aliases share every threshold-independent value',len(metrics)==24 and len(messages)>0 and all(
        next(r for r in metrics if r['stratum']==s and r['dataset']==d and r['model']=='old-head-maximin')[key]
        ==next(r for r in metrics if r['stratum']==s and r['dataset']==d and r['model']=='old-head-original')[key]
        for s,d in m.STRATA for key in ['accuracy','macro_f1','nll','confusion','recall','system_auroc','system_average_precision']))
    pairs,intervals,components=m.bootstrap(counts);check('Six strata /216 intervals /324 pairs',len(intervals)==216 and len(pairs)==324)
    checked=0
    for si,(stratum,dataset) in enumerate(m.STRATA):
        sub=counts[(counts.stratum==stratum)&(counts.dataset==dataset)];arrays={};ids=None
        for r in m.RECORDS:
            aggregate=sub[sub.model==r].groupby('dependency_component_id')[m.COUNT_FIELDS].sum().sort_index()
            if ids is None:ids=aggregate.index.tolist()
            assert ids==aggregate.index.tolist();arrays[r]=aggregate.to_numpy(np.float64)
        draws=np.random.RandomState(m.BOOTSTRAP_SEED+si).randint(len(ids),size=(2000,len(ids)))
        samples={r:ratios(a[draws].sum(1)) for r,a in arrays.items()};points={r:ratios(a.sum(0)) for r,a in arrays.items()}
        for row in [i for i in intervals if (i['stratum'],i['dataset'])==(stratum,dataset)]:
            values=samples[row['model']][row['metric']];finite=np.isfinite(values);assert row['defined_bootstrap_replicates']==finite.sum()
            
            if not finite.any():assert row['ci95_low'] is row['ci95_high'] is None
            else:
                lo,hi=np.quantile(values[finite],[.025,.975]);maxdiff=max(maxdiff,abs(lo-row['ci95_low']),abs(hi-row['ci95_high']))
            checked+=1
        for row in [i for i in pairs if (i['stratum'],i['dataset'])==(stratum,dataset)]:
            a,b,k=row['baseline_model'],row['comparison_model'],row['metric'];good=np.isfinite(samples[a][k])&np.isfinite(samples[b][k])
            values=100*(samples[b][k][good]-samples[a][k][good]);assert row['defined_bootstrap_replicates']==good.sum()
            if not len(values):assert row['ci95_low'] is row['ci95_high'] is None
            else:
                lo,hi=np.quantile(values,[.025,.975]);maxdiff=max(maxdiff,abs(lo-row['ci95_low']),abs(hi-row['ci95_high']))
            checked+=1
    check('All nine metrics /2000 paired component draws independently exact',checked==540 and maxdiff==0)
    bridge=rows.copy();bridge.loc[bridge.family_id.eq('family-5'),'eligible_clean_test']=False
    _,bridged,_=m.summarize(bridge,physical,thresholds)
    check('All/clean/source filters retain prior union component identity',set(bridged.loc[bridged.family_id.eq('family-4'),'dependency_component_id'])=={'component-4'}
        and not len(bridged[(bridged.stratum=='clean-test')&bridged.family_id.eq('family-5')]))
    one=counts.copy();one.dependency_component_id='only';_,ci,_=m.bootstrap(one)
    check('Single component never receives CI',all(i['ci95_low'] is i['ci95_high'] is None and i['ci_available'] is False for i in ci))
    em,er,el,ep,et=m.synthetic_data(True);values,c,proj=m.summarize(er,ep,et);pairs,ci,_=m.bootstrap(c)
    check('All128failed retain24/216/324 observations N/A /zero defined draws',len(values)==24 and len(ci)==216 and len(pairs)==324 and len(el)==128
        and all(i['tokens']==0 and i['system_precision'] is None and i['accuracy'] is None for i in values)
        and all(i['defined_bootstrap_replicates']==0 and not i['ci_available'] for i in ci))
    # Guard the public write path with fake modules: no actual builder/inference imports.
    fakeb=SimpleNamespace(OUT=folder/'fake-builder');fakei=SimpleNamespace(load_context=forbidden,completed_probabilities=forbidden)
    with patch.dict(sys.modules,{'systemness_aug_natural_data':fakeb,'systemness_aug_confirmation_infer':fakei}):
        try:m.write()
        except RuntimeError:pass
        else:raise AssertionError('Nonterminal accepted')
    checks.append('Terminal marker before context/cache/probability reads')
    inference=fakeb.OUT/'inference';inference.mkdir(parents=True);(inference/'inference-complete.json').write_text('{}');(inference/'status.json').write_text(json.dumps({'stage':'inference_complete'}))
    fakei.load_context=lambda _:SimpleNamespace(snapshot={'runtime_files_sha256':{}})
    with patch.dict(sys.modules,{'systemness_aug_natural_data':fakeb,'systemness_aug_confirmation_infer':fakei}):
        try:m.write()
        except RuntimeError:pass
        else:raise AssertionError('Unfrozen report accepted')
    checks.append('Exact report source frozen before any probabilities')
# Independent inspection of owner's synthetic full/empty artifact sets only.
fixture=Path('/tmp/systemness-aug-natural-report-preflight-do_1wkr0')
for label in ['report','empty-report']:
    out=fixture/label;complete=json.loads((out/'complete.json').read_text())
    assert len(list((out/'plots').glob('*.png')))==len(list((out/'plots').glob('*.pdf')))==13
    for name,expected in complete['artifact_sha256'].items():assert sha(out/name)==expected
    status=json.loads((out/'source-status.json').read_text())
    assert len(status['records'])==128
    if label=='empty-report':
        assert all(r['status']=='quality-failed' for r in status['records'])
        assert len(pd.read_csv(out/'source-counts.csv'))==len(pd.read_csv(out/'message-projections.csv.gz'))==0
        assert len(pd.read_csv(out/'metrics.csv'))==24
checks.append('Normal and allfailed toy artifacts:13 PNG/PDF each /128ledger /emptyCSV /allSHA')
assert sha(source)==start,'Owner source changed during annex'
report={'source_sha256':start,'annex_source_sha256':sha(__file__),'checks':checks,'checks_passed':len(checks),'maximum_bootstrap_difference':maxdiff,
    'paired_and_interval_rows_compared':checked,'formal_data_or_probabilities_read':False,'gpu_used':False,'formal_outputs_changed':False}
(folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'report':str(folder/'report.json'),'report_sha256':sha(folder/'report.json'),**report}))
