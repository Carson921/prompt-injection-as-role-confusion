"""Source/toy-only independent review of six-record auditor; never actual."""
import copy,hashlib,importlib.util,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
SOURCE=Path('/tmp/feature_review_long_augmentation_results_audit.py')
EXPECTED='e06738da7ebe2ae4b2b3a730d587520e3240b5838fe093b000f7d04ac4dc5282'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,m):
    if not x:raise RuntimeError(m)
def forbidden(*a,**kw):raise AssertionError('Actual result/cache read was attempted.')
need(sha(SOURCE)==EXPECTED,'Source differs');spec=importlib.util.spec_from_file_location('auditor_annex',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[];rng=np.random.default_rng(9921);maxerror=0.;total=0
with patch('numpy.load',side_effect=forbidden),patch('pandas.read_parquet',side_effect=forbidden):
    for case in range(12):
        y=rng.integers(0,5,127);p=rng.integers(1,12,size=(127,5)).astype(np.float32);p/=p.sum(1,keepdims=True)
        got=m.metrics(p,y,.3);truth=y==0
        maxerror=max(maxerror,abs(got['system_auroc']-roc_auc_score(truth,p[:,0])),abs(got['system_average_precision']-average_precision_score(truth,p[:,0])))
        need(got['confusion']==confusion_matrix(y,p.argmax(1),labels=np.arange(5)).tolist(),'Confusion disagrees')
    need(maxerror<1e-14,'Tied rankings disagree');checks.append('12tiedAUROC/AP/fiveconfusion sklearn cases')
    old,new,_,_=m.synthetic();ys={'old':m.labels(old),'new':m.labels(new)};groups=m.group_indices(old,new)
    for case in range(2):
        values={}
        for name,rows in [('old',old),('new',new)]:
            ps=np.where(ys[name]==0,rng.uniform(.35,.99,len(rows)),rng.uniform(.01,.45,len(rows))).astype(np.float32)
            ps[np.arange(len(rows))%9==0]=np.nextafter(np.float32(.5),np.float32(1))
            arr=np.empty((len(rows),5),np.float32);arr[:,0]=ps;arr[:,1:]=(1-ps[:,None])*rng.dirichlet(np.ones(4),size=len(rows));values[name]=arr
        po,pn=values['old'],values['new'];tau,proof,annex=m.direct_grid(old,new,po,pn)
        grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.r_[po[:,0],pn[:,0]],np.linspace(0,1,301)),1.000001]);curves={}
        for name,(bank,ix) in groups.items():
            truth=ys[bank][ix]==0;rows=[]
            for t in grid:
                detection=values[bank][ix,0]>=float(t);tp=int(np.sum(truth&detection));fp=int(np.sum(~truth&detection));fn=int(np.sum(truth&~detection));tn=int(np.sum(~truth&~detection))
                rows.append([tp/(tp+fp) if tp+fp else np.nan,tp/(tp+fn),(tp/(tp+fn)+tn/(tn+fp))/2,2*tp/(2*tp+fp+fn),fp/(fp+tn)])
            curves[name]=np.array(rows);need(m.array_sha(curves[name])==annex['direct_curves_sha256'][name],'Full F32 count curve differs')
        low=np.min([curves[g][:,i] for g in m.GROUPS for i in [0,1,2]],axis=0);low[~np.isfinite(low)]=-np.inf
        mean=np.mean([curves['old-'+p][:,3] for p in m.PROFILES],axis=0);feasible=np.max([curves[g][:,4] for g in m.FEASIBLE],axis=0)<=.05
        best=max(np.flatnonzero(feasible&np.isfinite(low)),key=lambda i:(float(low[i]),float(mean[i]),-float(grid[i])))
        need(tau==float(grid[best]) and proof['min24']==float(low[best]) and proof['old5_mean_f1']==float(mean[best]),'Global argmax mismatch')
        total+=len(grid)*24
        need(m.legacy_mean(old,po)==m.production.lp.validation_metrics(old,po)['system_threshold'],'Old-only legacy mismatch')
    checks.append('All8groups/F32curves x24 objective and lexargmax on2random boundary cases');checks.append('Oldmean original helper exact on2varied cases')
    def value(score):return dict(min24=score,old5_mean_f1=score,groups={g:dict(system_f1=score,accuracy=score,system_false_positive_rate=.01) for g in m.GROUPS})
    records=[dict(record=x+'-'+o,validation=value(.9)) for x in m.MODELS for o in m.OBJECTIVES]
    next(r for r in records if r['record']=='new-maximin')['validation']=value(.93)
    next(r for r in records if r['record']=='matched-maximin')['validation']=value(.92)
    need(m.selection(records)['promoted'],'Positive gate failed')
    for edit in [lambda r:r.update(min24=.92),lambda r:r.update(min24=None),lambda r:r.update(old5_mean_f1=.9149),lambda r:r['groups']['old-single'].update(system_f1=.87),lambda r:r['groups']['old-multi'].update(accuracy=.88),lambda r:r['groups']['new-long-full-tail'].update(system_f1=.89),lambda r:r['groups']['new-long-full'].update(system_false_positive_rate=.050001)]:
        bad=copy.deepcopy(records);edit(next(r for r in bad if r['record']=='new-maximin')['validation']);out=m.selection(bad)
        need(not out['promoted'] and out['candidate']=='new-maximin' and out['no_fallback'],'Negative gate/fallback failed')
    checks.append('Strictmin24 tie/None and6negative guard pathways nofallback')
    with patch.object(m,'processes',return_value=[91]),patch.object(m,'read',side_effect=forbidden):
        try:m.actual()
        except RuntimeError:pass
        else:raise RuntimeError('Terminal gate did not reject')
    checks.append('Livegate precedes any result/cache/probability read')
    h=[dict(epoch=i,steps=222,instances=905770,cumulative_steps=i*222,loss=.2,old5_mean_f1=.8) for i in range(1,11)]
    need(m.history_audit(h,1)['actual_steps']==2220,'History steps differ')
    try:m.history_audit(h+[dict(h[-1],epoch=11,cumulative_steps=2442)],1)
    except RuntimeError:pass
    else:raise RuntimeError('Continued earlystop accepted')
    checks.append('Oldearlystop and best/actual222step evidence')
need(sha(SOURCE)==EXPECTED,'Source changed')
report=dict(source_sha256=EXPECTED,checks=checks,checks_passed=len(checks),direct_primary_statistics=total,maximum_sklearn_difference=maxerror,actual_called_only_under_live_rejection_fixture=True,actual_probability_or_results_read=False,gpu_used=False,formal_files_modified=False,source_sha256_at_end=sha(SOURCE))
out=Path('/tmp/confirmation-review-augmentation-auditor-annex.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(out),**report),indent=2))
