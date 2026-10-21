"""Independent synthetic/source-only annex; never calls any actual data APIs."""
import hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path('/root/prompt-injection-as-role-confusion')
HERE=ROOT/'experiments/role-analysis'
sys.path.insert(0,str(HERE))
source=HERE/'systemness_aug_confirmation_report.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=sha(source)
spec=importlib.util.spec_from_file_location('review_aug_c4_report',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-aug-c4-report-',dir='/tmp'))
checks=[]
def check(name, yes):
    assert bool(yes), name
    checks.append(name)
def unavailable(*a,**kw): raise AssertionError('Actual model/cache/p API invoked')
def independent_counts(x):
    tp,fp,fn,tn,correct,tokens=np.moveaxis(np.asarray(x,dtype=np.float64),-1,0)
    def ratio(n,d): return np.divide(n,d,out=np.full(np.shape(n),np.nan),where=d>0)
    recall=ratio(tp,tp+fn)
    return {'system_precision':ratio(tp,tp+fp),'system_recall':recall,'system_f1':ratio(2*tp,2*tp+fp+fn),
        'system_balanced_accuracy':(recall+ratio(tn,tn+fp))/2,'system_false_positive_rate':ratio(fp,fp+tn),'accuracy':ratio(correct,tokens)}
# All helper checks are pure synthetic counts; every real probability reader forbidden.
with patch.object(np,'load',side_effect=unavailable),patch.object(pd,'read_parquet',side_effect=unavailable):
    physical={name:np.array([[.9,.025,.025,.025,.025],[.1,.7,.1,.05,.05]],np.float32) for name in m.PHYSICAL}
    rec=m.record_probabilities(physical)
    check('Three physical arrays / four records / exact old two alias',len(rec)==4 and rec['old-head-maximin'] is rec['old-head-original'])
    bad=dict(physical);bad.pop('matched')
    try:m.record_probabilities(bad)
    except RuntimeError:pass
    else:raise AssertionError('Incomplete physical arrays accepted')
    checks.append('Missing physical array rejected')
    example=np.array([[0,0,5,10,8,15],[3,2,1,4,7,10],[1,0,0,0,1,1]],float)
    expected=independent_counts(example);actual=m.binary_values(example)
    check('All six ratios and undefined precision independently exact',all(np.allclose(actual[k],v,equal_nan=True,atol=0,rtol=0) for k,v in expected.items()))
    # Eleven groups, 4 source units, 4 records; one old record has undefined P.
    documents=[]
    base=np.array([[2,1,3,4,5,10],[4,0,1,5,8,10],[1,2,2,5,4,10],[3,1,1,5,7,10]],float)
    arrays={}
    for gi,g in enumerate(m.GROUPS):
        arrays[g]={}
        for ri,r in enumerate(m.RECORDS):
            a=base.copy();a[:,0]+=ri;a[:,2]=np.maximum(a[:,2]-ri,0);a[:,4]+=ri*.1
            if ri==3:a[:,0]=a[:,1]=0
            arrays[g][r]=a
            for di,v in enumerate(a):documents.append(dict(profile=g,record=r,group_id=f'd{di}',document_id=di,**dict(zip(m.FIELDS,v))))
    doc=pd.DataFrame(documents)
    with patch.object(m,'EXPECTED_DOCUMENTS',{g:4 for g in m.GROUPS}):pairs,intervals=m.bootstrap(doc)
    check('Exact 11groups x 4models x6metrics / 6pairs geometry',len(intervals)==264 and len(pairs)==396)
    maxdiff=0.;compared=0
    for gi,g in enumerate(m.GROUPS):
        rng=np.random.RandomState(m.BOOTSTRAP_SEED+gi);draw=rng.randint(4,size=(2000,4))
        p={r:independent_counts(a.sum(0)) for r,a in arrays[g].items()}
        s={r:independent_counts(a[draw].sum(1)) for r,a in arrays[g].items()}
        for row in [v for v in intervals if v['profile']==g]:
            r,k=row['record'],row['metric'];finite=np.isfinite(s[r][k]);v=s[r][k][finite]
            assert row['valid_bootstrap_draws']==finite.sum()
            expected=[None,None] if len(v)==0 else np.quantile(v,[.025,.975])
            for field,value in zip(['ci95_low','ci95_high'],expected):
                if value is None:assert row[field] is None
                else:maxdiff=max(maxdiff,abs(row[field]-value))
            compared+=1
        for row in [v for v in pairs if v['profile']==g]:
            a,b,k=row['baseline_record'],row['comparison_record'],row['metric'];finite=np.isfinite(s[a][k])&np.isfinite(s[b][k])
            assert row['valid_paired_bootstrap_draws']==finite.sum()
            dif=(s[b][k][finite]-s[a][k][finite])*100
            bounds=[None,None] if len(dif)==0 else np.quantile(dif,[.025,.975])
            for field,value in zip(['ci95_low','ci95_high'],bounds):
                if value is None:assert row[field] is None
                else:maxdiff=max(maxdiff,abs(row[field]-value))
            if not np.isfinite(p[a][k]) or not np.isfinite(p[b][k]):assert row['gain_percentage_points'] is None
            else:maxdiff=max(maxdiff,abs(row['gain_percentage_points']-(p[b][k]-p[a][k])*100))
            compared+=1
    check('All 2000 paired-source draws / defined filtering independently exact',maxdiff==0 and compared==660)
    # Marker must gate cache and inference module access.
    fake=SimpleNamespace(OUT=folder/'unprepared')
    with patch.object(m.importlib,'import_module',return_value=fake),patch.object(m,'infer_module',side_effect=unavailable):
        try:m.load_completed()
        except RuntimeError:pass
        else:raise AssertionError('Incomplete inference accepted')
    checks.append('Terminal inference markers precede all context/cache/probability APIs')
    # Every primary group remains in goal decision, so one failing bin cannot disappear.
    main=[dict(record=m.PRIMARY,profile=g,system_precision=.95,system_recall=.95,system_balanced_accuracy=.95) for g in m.PROFILES]
    positions=[dict(record=m.PRIMARY,profile=g,system_precision=.95,system_recall=.95,system_balanced_accuracy=.95) for g in m.BIN_NAMES]
    positions[-1]['system_recall']=.899999
    decision=m.targets(main,positions)
    check('All 11 target groups retained / last-bin failure cannot hide',len(decision)==11 and not decision[m.BIN_NAMES[-1]] and not all(decision.values()))
assert sha(source)==start,'Source changed during independent audit'
result={'source_path':str(source),'source_sha256':start,'checks':checks,'checks_passed':len(checks),'maximum_bootstrap_difference':maxdiff,
        'intervals_and_pair_rows_compared':compared,'bootstrap_replicates':2000,'formal_data_or_probabilities_read':False,'gpu_used':False,
        'formal_outputs_changed':False,'source_only_and_synthetic':True,'annex_source_sha256':sha(__file__)}
(folder/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':str(folder/'report.json'),'report_sha256':sha(folder/'report.json'),**result}))
