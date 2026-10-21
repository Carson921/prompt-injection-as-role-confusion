"""Prospective validation-only diagnosis. Default help; no test-path access.

Root approval is required before --run. --preflight reads synthetic arrays only.
No model, threshold, source, selection or production artifact is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
BASE=HERE/'outputs/local-gptoss120b'
HEAD=BASE/'systemness-head-probes'
PLAN=Path('/tmp/feature_review_validation_ensemble_protocol.json')
ALPHAS=[0.,.25,.5,.75,1.]
PROFILES=['single','multi','natural','long-single','long-full']
GROUPS=PROFILES+['long-full-tail']
ROLES=['system','user','cot','assistant','tool']
N=173032

def need(value,message):
    if not value:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()
def write(path,value):
    path=Path(path).resolve();need(path.is_relative_to('/tmp'),'Only /tmp outputs are allowed.')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def helpers():
    sys.path.insert(0,str(HERE))
    import systemness_head_probes as hp
    import systemness_long_probes as lp
    import systemness_probes as probes
    return hp,lp,probes

def validate_probability(p,count):
    need(p.shape==(count,5) and p.dtype==np.float32 and np.isfinite(p).all()
        and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=2e-5,rtol=0),'Invalid five-class probabilities.')

def mix(p64,p128,alpha):
    # Fixed power-of-two weights, explicit F64 accumulation and oneF32 cast.
    return (float(alpha)*p64.astype(np.float64)+(1-float(alpha))*p128.astype(np.float64)).astype(np.float32)

def group_indices(rows):
    need(np.array_equal(rows.index,np.arange(len(rows))),'Validation rows must reset to global probability indices.')
    result={profile:np.flatnonzero(rows.profile.eq(profile).to_numpy()) for profile in PROFILES}
    full=rows[rows.profile.eq('long-full')]
    offset=full.token_ix-full.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    result['long-full-tail']=full.index[offset>=512].to_numpy()
    need(all(len(v)>0 for v in result.values()),'Missing profile or512+tail.')
    return result

def threshold_curve(rows,p):
    _,_,probes=helpers();labels=probes.labels(rows);indices=group_indices(rows)
    grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(p[:,0],np.linspace(0,1,301)),1.000001])
    values={}
    for name,ix in indices.items():
        positive=np.sort(p[ix,0][labels[ix]==0]);negative=np.sort(p[ix,0][labels[ix]!=0])
        need(len(positive)>0 and len(negative)>0,'All six groups require actual positive and negative support.')
        tp=len(positive)-np.searchsorted(positive,grid,side='left');fp=len(negative)-np.searchsorted(negative,grid,side='left')
        recall=tp/len(positive);fpr=fp/len(negative)
        precision=np.divide(tp,tp+fp,out=np.full(len(grid),np.nan),where=tp+fp>0)
        values[name]=dict(precision=precision,recall=recall,fpr=fpr,
            balanced_accuracy=(recall+1-fpr)/2,f1=2*tp/np.maximum(2*tp+fp+len(positive)-tp,1))
    mean=np.mean([values[name]['f1'] for name in PROFILES],axis=0)
    feasible=np.max([values[name]['fpr'] for name in PROFILES],axis=0)<=.05
    minimum=np.min([values[name][field] for name in GROUPS for field in ['precision','recall','balanced_accuracy']],axis=0)
    minimum[~np.isfinite(minimum)]=-np.inf
    return grid,values,mean,minimum,feasible

def choose_threshold(rows,p,objective):
    grid,_,mean,minimum,feasible=threshold_curve(rows,p)
    if objective=='mean-f1':
        score=mean.copy();score[~feasible]=-1;index=int(np.argmax(score))
    else:
        need(objective=='maximin-pr-bacc','Unknown threshold objective.')
        valid=np.flatnonzero(feasible&np.isfinite(minimum));need(len(valid)>0,'No defined FPR-feasible maximin threshold.')
        index=max(valid,key=lambda i:(float(minimum[i]),float(mean[i]),-float(grid[i])))
    return float(grid[index]),dict(mean_f1=float(mean[index]),minimum_pr_bacc=float(minimum[index]),
        grid_size=len(grid),feasible_thresholds=int(feasible.sum()))

def metrics_at(rows,p,threshold):
    _,_,probes=helpers();indices=group_indices(rows);labels=probes.labels(rows)
    records={name:probes.metrics(p[ix],labels[ix],threshold) for name,ix in indices.items()}
    tail=records.pop('long-full-tail');full=rows[rows.profile.eq('long-full')]
    tail.update(documents=int(rows.loc[indices['long-full-tail'],'document_id'].nunique()),body_start=512,body_end=1023)
    return dict(system_threshold=threshold,mean_system_f1=float(np.mean([records[k]['system_f1'] for k in PROFILES])),
        original_mean_system_f1=float(np.mean([records[k]['system_f1'] for k in PROFILES[:4]])),
        original_mean_accuracy=float(np.mean([records[k]['accuracy'] for k in PROFILES[:4]])),profiles=records,long_full_tail=tail)

def min_pr_bacc(result):
    groups=list(result['profiles'].values())+[result['long_full_tail']]
    values=[g[k] for g in groups for k in ['system_precision','system_recall','system_balanced_accuracy']]
    return min(values) if all(v is not None and np.isfinite(v) for v in values) else None

def source_counts(rows):
    output=[]
    for name,ix in group_indices(rows).items():
        group=rows.iloc[ix]
        output.append(dict(profile=name,tokens=len(group),source_groups=int(group.group_id.nunique()),
            document_ids=int(group.document_id.nunique()),source_sets={str(k):int(v) for k,v in group.groupby('source_set').group_id.nunique().items()},
            role_support={r:int(group.role.eq(r).sum()) for r in ROLES},
            independent_new_test_documents=0))
    return output

def run():
    # This is the first possible p access; parent must approve exact protocolSHA
    # and sourceSHA before invoking --run. Default/help/preflight never enter.
    plan=read(PLAN);need(sha(__file__)==plan['executor_sha256'],'Executor changed after prospective freeze.')
    for path,digest in plan['input_metadata_and_sources_sha256'].items():need(sha(path)==digest,'Frozen input metadata/source changed: '+path)
    hp,lp,_=helpers();rows=pd.read_parquet(plan['rows_path'])
    need(sha(plan['rows_path'])==plan['rows_sha256'] and set(rows.split)=={'fit','validation'}
        and set(rows.wrapper)=={'native'} and rows.sample_ix.is_unique,'Row bank identity/split changed.')
    rows=rows[rows.split.eq('validation')].copy().reset_index(drop=True)
    need(len(rows)==N and set(rows.profile)==set(PROFILES),'Exact173032 validation partition required.')
    need(hashlib.sha256(np.ascontiguousarray(rows.sample_ix.to_numpy()).tobytes()).hexdigest()==plan['validation_sample_ix_sha256'],
        'Validation sample order changed.')
    probability={}
    for key,entry in plan['probability_inputs'].items():
        need(sha(entry['path'])==entry['sha256'],'Frozen validation probability SHA changed: '+key)
        with np.load(entry['path'],allow_pickle=False) as archive:
            need(np.array_equal(archive['sample_ix'],rows.sample_ix),'Validation probability sample order changed: '+key)
            probability[key]=archive['p'].copy()
        validate_probability(probability[key],N)
        need(lp.validation_metrics(rows,probability[key])==read(entry['candidate_record'])['validation'],
            'Saved validation probability does not reproduce frozen candidate metrics: '+key)
    baseline=read(plan['baseline_dense_record'])['validation'];current=read(plan['probability_inputs']['joint-mlp128']['candidate_record'])['validation']
    controls=dict(dense128_frozen=baseline,current_head128_frozen=current,
        head64_frozen=read(plan['probability_inputs']['joint-mlp64']['candidate_record'])['validation'])
    records=[];flat=[]
    for alpha in ALPHAS:
        p=mix(probability['joint-mlp64'],probability['joint-mlp128'],alpha);validate_probability(p,N)
        for objective in ['mean-f1','maximin-pr-bacc']:
            threshold,choice=choose_threshold(rows,p,objective);result=metrics_at(rows,p,threshold)
            if objective=='mean-f1':need(result==lp.validation_metrics(rows,p),'Original meanF1 helper parity failed.')
            guards={name:hp.promotion_guard(result,control) for name,control in [('dense128',baseline),('current-head128',current)]}
            eligible=all(v['eligible'] for v in guards.values());minimum=min_pr_bacc(result)
            records.append(dict(alpha_weight_head64=alpha,objective=objective,validation=result,threshold_choice=choice,
                dual_guard_eligible=eligible,guards=guards,minimum_pr_bacc=minimum,
                mean_f1_gain_vs_current=result['mean_system_f1']-current['mean_system_f1'],
                min_pr_bacc_gain_vs_current=minimum-min_pr_bacc(current),promoted=False,new_confirmation_required=True))
            for profile,values in dict(result['profiles'],**{'long-full-tail':result['long_full_tail']}).items():
                flat.append(dict(alpha_weight_head64=alpha,objective=objective,profile=profile,dual_guard_eligible=eligible,
                    **{k:v for k,v in values.items() if not isinstance(v,(dict,list))},
                    **{'role_recall_'+role:value for role,value in values['recall'].items()}))
    # Diagnostic leaders only; no deployable selection/current-model file.
    leaders={}
    for objective in ['mean-f1','maximin-pr-bacc']:
        pool=[r for r in records if r['objective']==objective and r['dual_guard_eligible']]
        key=lambda r:(r['validation']['mean_system_f1'] if objective=='mean-f1' else r['minimum_pr_bacc'],
            r['validation']['mean_system_f1'],-r['validation']['system_threshold'],-r['alpha_weight_head64'])
        leaders[objective]=max(pool,key=key) if pool else None
    for path,digest in plan['input_metadata_and_sources_sha256'].items():need(sha(path)==digest,'Input metadata changed during diagnosis.')
    for entry in plan['probability_inputs'].values():need(sha(entry['path'])==entry['sha256'],'Validation p changed during diagnosis.')
    output=Path(tempfile.mkdtemp(prefix='validation-ensemble-diagnosis-',dir='/tmp'))
    write(output/'report.json',dict(protocol_sha256=sha(PLAN),executor_sha256=sha(__file__),controls=controls,
        source_counts=source_counts(rows),records=records,diagnostic_leaders=leaders,test_data_or_probabilities_read=False,
        current_model_changed=False,models_trained=False,promoted=False,new_independent_confirmation_required=True,
        limitations=plan['limitations']))
    pd.DataFrame(flat).to_csv(output/'metrics.csv',index=False)
    print(json.dumps(dict(report=str(output/'report.json'),report_sha256=sha(output/'report.json'))))

def preflight():
    rows=[]
    for profile in PROFILES:
        for ri,role in enumerate(ROLES):
            for offset in [0,512]:rows.append(dict(profile=profile,role=role,prompt_ix=len(rows)//2,message_ix=ri,
                token_ix=100+offset,document_id=ri,group_id='toy-'+str(ri),source_set='synthetic',split='validation',wrapper='native'))
    rows=pd.DataFrame(rows);p=np.full((len(rows),5),.04,np.float32)
    p[np.arange(len(rows)),rows.role.map({role:i for i,role in enumerate(ROLES)}).to_numpy()]=.84
    other=p[:,[0,2,1,4,3]].copy();need(np.array_equal(mix(p,other,0),other) and np.array_equal(mix(p,other,1),p),'Alpha endpoints changed frozen p.')
    _,lp,_=helpers()
    for alpha in ALPHAS:
        mixed=mix(p,other,alpha);validate_probability(mixed,len(rows))
        t,_=choose_threshold(rows,mixed,'mean-f1');need(metrics_at(rows,mixed,t)==lp.validation_metrics(rows,mixed),'MeanF1 helper parity failed.')
        t,score=choose_threshold(rows,mixed,'maximin-pr-bacc');result=metrics_at(rows,mixed,t)
        need(abs(min_pr_bacc(result)-score['minimum_pr_bacc'])<1e-14,'Maximin curve/count mismatch.')
        need(all(r['system_false_positive_rate']<=.05 for r in result['profiles'].values()),'FPR grid feasibility changed.')
    print(json.dumps(dict(preflight=True,synthetic_only=True,validation_probability_files_read=False,
        source_sha256=sha(__file__),checks=['exact_alpha_endpoints','all5_mix_normalized','original_meanF1_helper_parity',
        'all6groups_maximin18values_count_parity','original_FPR5profile_constraint'],formal_files_modified=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--run',action='store_true');action.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.run:run()
    elif args.preflight:preflight()
    else:parser.print_help()
