"""Independent synthetic CPU annex: never calls --run or opens real p.
Direct NumPy F32 >= Python-float counts audit all threshold/profile positions.
"""
import ast,copy,hashlib,importlib.util,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
V1=Path('/tmp/feature_review_validation_ensemble.py');V2=Path('/tmp/feature_review_validation_ensemble_v2.py');PLAN=Path('/tmp/feature_review_validation_ensemble_v2_protocol.json')
EXPECTED={V1:'fc09b16b690e9f45cf31ee80d68c42400f8c38b0dbe1c7c12c39095e17edc6c3',V2:'055ac5db7f8b8105a41ccb624474f0aa276b18146be0e6dc368132b9be8ed7f3',PLAN:'fb8b4c58e5983f1c00e9fe379b1f58a0fc5e5929bbaa48c21df80c47533d4d1b'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,s):
    if not x:raise RuntimeError(s)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def forbidden(*args,**kwargs):raise RuntimeError('Forbidden real input or run API.')
for path,s in EXPECTED.items():need(sha(path)==s,'Frozen source/protocol mismatch: '+str(path))
a,b=load(V1,'v1_synthetic_annex'),load(V2,'v2_synthetic_annex');a.run=b.run=forbidden
hp,lp,probes=b.helpers();hp.verified_completion=hp.dataset=lp.dataset=forbidden
checks=[];details=[];totals=dict(primary_statistics=0,additional_f1_fpr=0,legacy_grid_values=0,maximum_direct_difference=0.)
def check(name,value=True):need(value,name);checks.append(name)
def equal(x,y,message):
    if isinstance(x,dict):need(set(x)==set(y),message);[equal(x[k],y[k],message) for k in x];return
    if isinstance(x,np.ndarray):need(x.dtype==y.dtype and np.array_equal(x,y,equal_nan=True),message);return
    need(x==y,message)
def make_rows():
    rows=[]
    for pi,profile in enumerate(b.PROFILES):
        for di in range(3):
            for role in b.ROLES:
                if profile=='natural' and role=='tool':continue
                prompt=pi*100+di*5+b.ROLES.index(role)
                for off in [0,31,191,511,512,641,777,1023]:rows.append(dict(profile=profile,role=role,prompt_ix=prompt,message_ix=0,token_ix=100+off,document_id=di,group_id=f'{profile}-{di}',source_set='synthetic',split='validation',wrapper='native',sample_ix=9000000000000+len(rows)))
    return pd.DataFrame(rows)
rows=make_rows();labels=probes.labels(rows)
# Preserve adjacent float32 and ties, while allowing genuinely varied objectives.
rng=np.random.default_rng(901211);bank={}
for key in [64,128]:
    p=np.empty((len(rows),5),np.float32)
    for i,y in enumerate(labels):
        ps=rng.uniform(.48,.98) if y==0 else rng.uniform(.001,.14)
        if i%17==0:ps=float(np.nextafter(np.float32(.5),np.float32(1))) if (i//17)%2 else .5
        if i%37==0:ps=0.
        tail=rng.dirichlet(np.ones(4));p[i,0]=ps;p[i,1:]=(1-p[i,0])*tail
    bank[key]=p

def independent_indices(r):
    groups={k:r.index[r.profile.eq(k)].to_numpy() for k in b.PROFILES}
    full=r[r.profile.eq('long-full')];idx=[]
    for (_, _),part in full.groupby(['prompt_ix','message_ix']):
        first=int(part.token_ix.min());idx.extend(int(ix) for ix,value in part.token_ix.items() if value-first>=512)
    groups['long-full-tail']=np.array(sorted(idx),np.int64);return groups

def direct_curve(r,p,grid):
    result={};ys=probes.labels(r)
    for name,ix in independent_indices(r).items():
        positive=ys[ix]==0;ps=p[ix,0];denp=int(positive.sum());denn=int((~positive).sum());fields={k:[] for k in ['precision','recall','balanced_accuracy','fpr','f1']}
        for cutoff in grid:
            detection=ps>=float(cutoff)
            tp=int(np.count_nonzero(positive&detection));fp=int(np.count_nonzero((~positive)&detection));fn=denp-tp;tn=denn-fp
            for k,v in dict(precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/denp,balanced_accuracy=(tp/denp+tn/denn)/2,fpr=fp/denn,f1=2*tp/(2*tp+fp+fn)).items():fields[k].append(v)
        result[name]={k:np.asarray(v) for k,v in fields.items()}
    mean=np.mean([result[k]['f1'] for k in b.PROFILES],axis=0)
    low=np.min([result[k][metric] for k in b.GROUPS for metric in ['precision','recall','balanced_accuracy']],axis=0);low[~np.isfinite(low)]=-np.inf
    feasible=np.array([all(result[k]['fpr'][i]<=.05 for k in b.PROFILES) for i in range(len(grid))],bool)
    options=[i for i in range(len(grid)) if feasible[i] and np.isfinite(low[i])]
    chosen=sorted(options,key=lambda i:(-low[i],-mean[i],float(grid[i])))[0]
    return result,mean,low,feasible,chosen

def verify_case(name,r,p):
    grid,curve,mean,minimum,feasible=b.threshold_curve(r,p,consumer_aligned=True)
    expected_grid=np.unique(np.concatenate((np.linspace(.001,.999,199),np.quantile(p[:,0],np.linspace(0,1,301)),[1.000001])))
    equal(grid,expected_grid,name+' grid changed')
    direct,dm,dl,df,best=direct_curve(r,p,grid);maximum=0.
    for group in b.GROUPS:
        for field in ['precision','recall','balanced_accuracy','f1','fpr']:
            need(np.array_equal(np.isnan(curve[group][field]),np.isnan(direct[group][field])),name+' undefined flags changed')
            finite=np.isfinite(direct[group][field]);error=float(np.max(np.abs(curve[group][field][finite]-direct[group][field][finite]))) if finite.any() else 0.
            maximum=max(maximum,error);need(error==0.,name+' direct arithmetic mismatch')
    equal(mean,dm,name+' tie mean mismatch');equal(minimum,dl,name+' min18 mismatch');equal(feasible,df,name+' actual FPR feasibility mismatch')
    cutoff,choice=b.choose_threshold(r,p,'maximin-pr-bacc');need(cutoff==float(grid[best]),name+' exact argmax mismatch')
    result=b.metrics_at(r,p,cutoff);need(b.min_pr_bacc(result)==float(dl[best]) and result['mean_system_f1']==float(dm[best]),name+' selected metrics mismatch')
    need(choice['count_semantics']=='actual-F32-consumer',name+' missing count semantics')
    old=a.threshold_curve(r,p);legacy=b.threshold_curve(r,p,consumer_aligned=False)
    for x,y in zip(old,legacy):equal(x,y,name+' legacy curve changed')
    t1,c1=a.choose_threshold(r,p,'mean-f1');t2,c2=b.choose_threshold(r,p,'mean-f1');c2=dict(c2);need(c2.pop('count_semantics')=='unchanged-legacy-mean-helper',name+' mean semantics changed')
    need(t1==t2 and c1==c2 and a.metrics_at(r,p,t1)==b.metrics_at(r,p,t2)==lp.validation_metrics(r,p),name+' legacy mean result changed')
    totals['primary_statistics']+=len(grid)*18;totals['additional_f1_fpr']+=len(grid)*12;totals['legacy_grid_values']+=len(grid)*30;totals['maximum_direct_difference']=max(totals['maximum_direct_difference'],maximum)
    details.append(dict(case=name,rows=len(r),grid=len(grid),cutoff=cutoff,min18=float(dl[best]),mean5F1=float(dm[best]),maximum_direct_difference=maximum,legacy_mean_cutoff=t2))
    return cutoff

with patch('numpy.load',side_effect=forbidden),patch('pandas.read_parquet',side_effect=forbidden):
    for alpha in b.ALPHAS:
        p=b.mix(bank[64],bank[128],alpha);b.validate_probability(p,len(rows));verify_case('mixed-alpha-'+str(alpha),rows,p)
    check('Each of5 candidates every grid x18 plus F1/FPR matches actual F32 scalar predicates exactly')
    check('All5 entire legacy mean curves, thresholds and metrics remain identical to v1 and original helper')
    check('Mixture endpoints remain bitexact',np.array_equal(b.mix(bank[64],bank[128],0),bank[128]) and np.array_equal(b.mix(bank[64],bank[128],1),bank[64]))
    # Dedicated adjacent-F32 boundary counterexample; positives just above .5;
    # a small cluster of negative .5 values makes v1 choose the wrong cutoff.
    boundary=[];pp=[];high=np.nextafter(np.float32(.5),np.float32(1))
    for profile in b.PROFILES:
        for ri,role in enumerate(b.ROLES):
            for pos in range(20):
                boundary.append(dict(profile=profile,role=role,prompt_ix=b.PROFILES.index(profile)*5+ri,message_ix=ri,token_ix=100+pos%10+(512 if pos>=10 else 0),document_id=ri,group_id=f'toy-{ri}',source_set='synthetic',split='validation',wrapper='native'))
                x=np.zeros(5,np.float32)
                if ri==0:x[0]=high;x[1:]=(1-high)/4
                else:x[:]=.01;x[0]=.5 if ri==1 and pos<8 else .1;x[ri]=1-x[0]-.03
                pp.append(x)
    boundary=pd.DataFrame(boundary);pp=np.asarray(pp,np.float32);old_t,old=a.choose_threshold(boundary,pp,'maximin-pr-bacc');old_actual=a.min_pr_bacc(a.metrics_at(boundary,pp,old_t));new_t=verify_case('adjacent-F32-negative-boundary',boundary,pp)
    check('v1 counterexample independently retained and v2 repaired',old['minimum_pr_bacc']==1. and old_actual<.8 and b.min_pr_bacc(b.metrics_at(boundary,pp,new_t))==1.)
    # Feasibility excludes tail as declared, even if tail FPR alone exceeds .05.
    tail_p=np.empty_like(pp)
    for i,y in enumerate(probes.labels(boundary)):
        tail_p[i]=.025 if y==0 else 0.
        if y==0:tail_p[i,0]=.9
        else:tail_p[i,0]=.1;tail_p[i,y]=.9
    full=boundary[boundary.profile.eq('long-full')];negative_tail=full.index[(full.role!='system') & (full.token_ix>=612)]
    tail_p[int(negative_tail[0])]=[.8,.2,0,0,0];tail_p[int(negative_tail[1])]=[.8,.2,0,0,0];tail_p[int(negative_tail[2])]=[.8,.2,0,0,0]
    grid,cv,_,_,fv=b.threshold_curve(boundary,tail_p,consumer_aligned=True);j=int(np.flatnonzero(grid==.5)[0])
    # full negative80/tail40: use3 fp -> full3.75%,tail7.5%.
    check('FPR feasibility uses5 profiles and reports rather than constrains tail',bool(fv[j]) and cv['long-full-tail']['fpr'][j]>.05 and cv['long-full']['fpr'][j]<=.05)
    ids=independent_indices(rows);check('Tail uses true >=512 offsets and global indices',np.array_equal(ids['long-full-tail'],b.group_indices(rows)['long-full-tail']))
    check('Unsupported natural tool truth stays zero',rows[rows.profile.eq('natural')].role.eq('tool').sum()==0)
    def metric_record(value=.9):
        return dict(mean_system_f1=value,profiles={k:dict(system_f1=value,accuracy=value,system_false_positive_rate=.02) for k in b.PROFILES},long_full_tail=dict(system_f1=value))
    base=metric_record();cases=[('same',copy.deepcopy(base),True)]
    for name,where,key,value in [('F1-boundary','single','system_f1',.88),('F1-fail','single','system_f1',.87999999),('accuracy-boundary','multi','accuracy',.89),('accuracy-fail','multi','accuracy',.88999999),('full-fail','long-full','system_f1',.89999999),('FPR-boundary','natural','system_false_positive_rate',.05),('FPR-fail','natural','system_false_positive_rate',.0500001)]:
        result=copy.deepcopy(base);result['profiles'][where][key]=value;cases.append((name,result,'fail' not in name))
    bad=copy.deepcopy(base);bad['long_full_tail']['system_f1']=.89999999;cases.append(('tail-fail',bad,False))
    for name,result,eligible in cases:need(hp.promotion_guard(result,base)['eligible'] is eligible,name+' guard boundary changed')
    check('Dual guard .02 F1/.01 accuracy/FPR .05/full-tail nonregression boundary and negative cases hold')
    weak=metric_record(.8);strong=metric_record(.95);middle=metric_record(.9)
    check('Passing dense alone cannot bypass current-head128 control',hp.promotion_guard(middle,weak)['eligible'] and not hp.promotion_guard(middle,strong)['eligible'])
    # Only AST inspected for runtime scope; run() is never called.
    tree=ast.parse(V2.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run');text=ast.unparse(run)
    outer=[n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='alpha'];need(len(outer)==1,'Alpha loop changed')
    inner=[n for n in ast.walk(outer[0]) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='objective'];need(len(inner)==1,'Objective loop changed')
    objectives=ast.literal_eval(inner[0].iter)
    check('Runtime loop remains exactly5 x2=10 candidate rows',b.ALPHAS==[0.,.25,.5,.75,1.] and objectives==['mean-f1','maximin-pr-bacc'])
    check('Runtime leader filter requires dual guards and every candidate remains unpromoted',"r['dual_guard_eligible']" in text and 'promoted=False' in text and 'new_confirmation_required=True' in text and "[('dense128', baseline), ('current-head128', current)]" in text)
    check('Runtime probability reads remain exactly declared head64/head128 validation NPZ and no broad completion verifier', 'verified_completion' not in text and "plan['probability_inputs'].items()" in text and "{'fit', 'validation'}" in text and "rows.split.eq('validation')" in text and "lp.validation_metrics(rows, probability[key])" in text)
for p,s in EXPECTED.items():need(sha(p)==s,'Frozen source changed during independent audit')
check('Frozenv1/v2 source and protocol SHA unchanged at end')
report=dict(stage='independent_v2_synthetic_CPU_annex',source_sha256=sha(__file__),frozen_input_sha256={str(p):s for p,s in EXPECTED.items()},numpy_version=np.__version__,checks=checks,checks_passed=len(checks),cases=details,**totals,
    v1_boundary=dict(cutoff=old_t,curve_minimum=old['minimum_pr_bacc'],actual_minimum=old_actual),v2_boundary=dict(cutoff=new_t,curve_minimum=1.,actual_minimum=1.),
    actual_run_called=False,actual_probability_files_read=False,verified_completion_called=False,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,
    limitations=['Synthetic boundary/count and frozen source review only; no actual validation result or generalization claim.','Legacy mean F64 sorted-count semantics intentionally retained; only new maximin matches direct F32 consumer counts.'])
out=Path('/tmp/confirmation-review-ensemble-v2-annex.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(out),checks_passed=len(checks),primary_statistics=totals['primary_statistics'],maximum_direct_difference=totals['maximum_direct_difference']),indent=2))
