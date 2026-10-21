"""Gold bound4 developer-only oracle. Default help; no test or GPU access.

--preflight uses synthetic CPU data. --actual is separately root-authorized and
only reads certified old/C4/public development rows and original raw base model.
The oracle is not deployable, trainable, a candidate, or a promotion result.
"""
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
BASE=ROOT/'outputs/local-gptoss120b'
CERT=Path('/tmp/long_data_binary_residual_results_audit.json')
CERT_SHA='2600d086c711e65bb320b70e66539fa775164cba8c192b84a2f9c52a75a9aa67'
PROTOCOL=BASE/'systemness-binary-residual/protocol.json'
PROTOCOL_SHA='84ebfc2dff0138dab829cb7756660b819e4d734823508c6219ec9de6c93cba1a'
MODEL=BASE/'systemness-head-probes/models/joint-mlp128.npz'
MODEL_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
BANKS={'old':Path('/dev/shm/role-analysis-top8-heads'),'c4':Path('/dev/shm/role-analysis-long-augmentation-heads'),'public':Path('/dev/shm/role-analysis-natural-dev-heads')}
ROLES=['system','user','cot','assistant','tool'];OLD5=['old-'+p for p in ['single','multi','natural','long-single','long-full']]
PUBLIC=['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
GROUPS=OLD5+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']+PUBLIC
FPR_GROUPS=OLD5+['new-C4-long-full']+PUBLIC
LIMITATION='Gold labels choose delta=+4(system)/-4(other): undeployable developer-only oracle. A reachable90% envelope only rules out an absolute bound4 ceiling; it does not demonstrate learnability, optimization success, a deployed classifier, promotion, or independent-test generalization.'
NUMERICS='No reconstruction from clipped savedF32 p0. RawF32 logits are recomputed once on CPU4/highest; F64 logodds retain margins for saturated p0. Actual F32 logsoftmax-exp endpoint scores and real-arithmetic logodds envelopes are reported separately. Float32 endpoint arithmetic is a numerical diagnostic; real-arithmetic monotonicity gives the mathematical upper bound.'
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def groups(rows):
    r={g:('old',np.flatnonzero(rows['old'].profile.eq(p))) for g,p in zip(OLD5,['single','multi','natural','long-single','long-full'])}
    r['new-C4-long-full']=('c4',np.arange(len(rows['c4'])))
    for bank,name in [('old','old-long-full-tail'),('c4','new-C4-long-full-tail')]:
        full=rows[bank][rows[bank].profile.eq('long-full')]
        off=full.body_offset if 'body_offset' in full else full.token_ix-full.groupby(['source_set','prompt_ix','message_ix']).token_ix.transform('min')
        r[name]=(bank,full.index.to_numpy()[off.to_numpy()>=512])
    r[PUBLIC[0]]=('public',np.arange(len(rows['public'])))
    for g,d in zip(PUBLIC[1:],['oasst','toxicchat0124']):r[g]=('public',np.flatnonzero(rows['public'].dataset.eq(d)))
    require(set(r)==set(GROUPS) and all(len(ix) for _,ix in r.values()),'Missing developer group.');return r
def objective(scores,truth,gix,cuts):
    """Exact sorted-event counts; bounded cut blocks, one global threshold."""
    sorted_scores={g:(np.sort(scores[b][ix][truth[b][ix]]),np.sort(scores[b][ix][~truth[b][ix]])) for g,(b,ix) in gix.items()}
    require(all(len(p) and len(n) for p,n in sorted_scores.values()),'Binary support missing.')
    best=None
    for start in range(0,len(cuts),4096):
        c=cuts[start:start+4096];minimum=np.ones(len(c));mean=np.zeros(len(c));feasible=np.ones(len(c),bool)
        for g,(pos,neg) in sorted_scores.items():
            tp=len(pos)-np.searchsorted(pos,c,side='left');fp=len(neg)-np.searchsorted(neg,c,side='left')
            precision=np.divide(tp,tp+fp,out=np.full(len(c),np.nan),where=(tp+fp)>0);recall=tp/len(pos);bacc=(recall+1-fp/len(neg))/2
            minimum=np.minimum(minimum,np.minimum(precision,np.minimum(recall,bacc)))
            if g in OLD5:mean+=(2*tp/(2*tp+fp+len(pos)-tp))/5
            if g in FPR_GROUPS:feasible &= fp/len(neg)<=.05
        valid=np.flatnonzero(feasible&np.isfinite(minimum))
        for j in valid:
            candidate=(float(minimum[j]),float(mean[j]),-float(c[j]),start+int(j))
            if best is None or candidate[:3]>best[:3]:best=candidate
    require(best is not None,'No defined globally FPR-feasible threshold.')
    cut=cuts[best[3]];details={}
    for g,(pos,neg) in sorted_scores.items():
        tp=int(len(pos)-np.searchsorted(pos,cut,side='left'));fp=int(len(neg)-np.searchsorted(neg,cut,side='left'));fn=len(pos)-tp;tn=len(neg)-fp
        details[g]=dict(system_precision=tp/(tp+fp) if tp+fp else None,system_recall=tp/len(pos),system_balanced_accuracy=(tp/len(pos)+tn/len(neg))/2,system_f1=2*tp/(2*tp+fp+fn),system_false_positive_rate=fp/len(neg),tp=tp,fp=fp,fn=fn,tn=tn)
    return dict(min33=best[0],old5_mean_f1=best[1],cut=float(cut),thresholds=len(cuts),FPR_groups=FPR_GROUPS,groups=details)
def logit_thresholds(t):
    result=np.full(len(t),np.inf,np.float64);result[t<=0]=-np.inf;valid=(t>0)&(t<1);result[valid]=np.log(t[valid])-np.log1p(-t[valid]);return result
def sigmoid(a):return np.exp(-np.logaddexp(0.,-np.asarray(a,np.float64)))
def diagnose(oracle_f32,oracle_a,truth,gix):
    # Underlying cohorts included once, overlapping group subsets never duplicated.
    grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate(list(oracle_f32.values())),np.linspace(0,1,301)),1.000001])
    events32=np.unique(np.r_[np.concatenate(list(oracle_f32.values())),np.float32(1.000001)]).astype(np.float32)
    events64=np.unique(np.r_[np.concatenate(list(oracle_a.values())),np.inf])
    result=dict(F32_fixed_production_grid=objective(oracle_f32,truth,gix,grid.astype(np.float32)),F32_all_distinct_score_events=objective(oracle_f32,truth,gix,events32),
        real_logodds_fixed_grid=objective(oracle_a,truth,gix,logit_thresholds(grid)),real_logodds_all_distinct_events=objective(oracle_a,truth,gix,events64))
    for name,r in result.items():
        r['threshold_domain']='logodds' if name.startswith('real') else 'F32_probability';r['threshold_probability_approximation']=float(sigmoid(r['cut'])) if name.startswith('real') else r['cut']
        r['reaches_90_percent']=r['min33']>=.9
    result['interpretation']=('Bound4 is not an absolute binary90% ceiling on these development cohorts.' if result['real_logodds_all_distinct_events']['min33']>=.9 else 'Even gold bound4 cannot reach90% in all33 binary developer statistics with one global threshold; inspect numerical/gap evidence before changing parameterization.')
    return result
def raw_scores(z,y):
    import torch
    a=z[:,0].double()-torch.logsumexp(z[:,1:].double(),1);adjusted=z.clone();adjusted[:,0]+=torch.where(y,4.,-4.)
    with torch.inference_mode():base=torch.log_softmax(z,1).exp()[:,0].numpy().copy();oracle=torch.log_softmax(adjusted,1).exp()[:,0].numpy().copy()
    return base,oracle,a.numpy(),a.numpy()+np.where(y.numpy(),4.,-4.)
def preflight():
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');checks=[]
    def check(s,v):require(v,s);checks.append(s)
    z=torch.tensor([[20.,0.,-2.,-3.,-4.],[30.,0.,-2.,-3.,-4.],[-110.,0.,-2.,-3.,-4.],[0.,1.,-1.,-2.,-3.]],dtype=torch.float32)
    b,o,a,oa=raw_scores(z,torch.tensor([False,False,True,True]))
    check('Saturated savedp1 has distinct finite raw logodds',b[0]==b[1]==1 and np.isfinite(a).all() and abs(a[0]-a[1])>9)
    check('p0underflow never clipped into invented margin',b[2]==0 and np.isfinite(a[2]) and oa[2]==a[2]+4)
    check('Gold endpoints improve positive scores and reduce negative scores',np.all(o[[2,3]]>=b[[2,3]]) and np.all(o[:2]<=b[:2]))
    check('Stable logodds avoids savedprobability infinities',np.isfinite(oa).all() and np.array_equal(logit_thresholds(np.array([0.,1.,1.000001])),[-np.inf,np.inf,np.inf]))
    rng=np.random.RandomState(37);base=rng.normal(size=20);y=np.arange(20)%5==0;gold=base+np.where(y,4.,-4.);random=base+rng.uniform(-4,4,20)
    check('At fixed thresholds oracle TP dominates and FP is no larger',all(np.count_nonzero((gold>=t)&y)>=np.count_nonzero((random>=t)&y) and np.count_nonzero((gold>=t)&~y)<=np.count_nonzero((random>=t)&~y) for t in np.linspace(-6,6,31)))
    gix={g:('old',np.arange(20)) for g in GROUPS};truth={'old':y};f32={'old':sigmoid(gold).astype(np.float32)};scores={'old':gold};result=diagnose(f32,scores,truth,gix)
    check('Fixed grid and all-score envelopes are explicitly separate',set(result)>={'F32_fixed_production_grid','F32_all_distinct_score_events','real_logodds_all_distinct_events'} and result['F32_all_distinct_score_events']['min33']>=result['F32_fixed_production_grid']['min33'])
    tied={'old':np.full(20,.5,np.float32)}
    try:objective(tied,truth,gix,np.array([.5,1.000001],np.float32))
    except RuntimeError:checks.append('All tied scores can be FPR-infeasible; not invented90% success')
    else:raise AssertionError('Tied infeasibility missed')
    events=np.unique(np.r_[f32['old'],np.float32(1.000001)]);sweep=objective(f32,truth,gix,events);best=-1.
    for t in events:
        detect=f32['old']>=float(t);tp=int((detect&y).sum());fp=int((detect&~y).sum());fn=int((~detect&y).sum());tn=int((~detect&~y).sum())
        if tp+fp and fp/(fp+tn)<=.05:best=max(best,min(tp/(tp+fp),tp/(tp+fn),(tp/(tp+fn)+tn/(tn+fp))/2))
    check('Sorted event sweep matches brute actualF32 threshold counts',sweep['min33']==best)
    folder=Path(tempfile.mkdtemp(prefix='bound4-oracle-preflight-',dir='/tmp'));report=dict(stage='synthetic_gold_bound4_oracle_preflight',source_sha256=digest(__file__),checks=checks,checks_passed=len(checks),actual_data_or_predictions_read=False,gpu_used=False,limitation=LIMITATION,numerics=NUMERICS)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),**report)))
def actual(output):
    import torch
    require(Path(output).resolve().is_relative_to(Path('/tmp')),'Only /tmp output.');require(digest(CERT)==CERT_SHA and digest(PROTOCOL)==PROTOCOL_SHA,'Completed audited trial/protocol changed.')
    cert=read(CERT);require(cert['audit_passed'] is True and cert['stage']=='actual_completed_binary_residual_development_audit' and cert['test_data_or_results_read'] is False,'Actual completed developer certificate required.')
    static=read(PROTOCOL)['static'];require(cert['trainer_protocol_sha256']==PROTOCOL_SHA and digest(MODEL)==MODEL_SHA,'Original model/protocol differs.')
    provenance=cert['provenance_sha256'];bindings={str(CERT):CERT_SHA,str(PROTOCOL):PROTOCOL_SHA,str(MODEL):MODEL_SHA}
    with np.load(MODEL,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));require(meta['kind']=='mlp' and meta['hidden']==128 and meta['dimensions']==512 and meta['roles']==ROLES,'Original model geometry changed.');mean=z['mean'].copy();scale=z['scale'].copy();w={k[8:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('weight::')}
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');mt,st=torch.from_numpy(mean),torch.from_numpy(scale)
    rows={};truth={};basep={};orap={};loga={};stats={}
    for bank,total,count in [('old',755282,173032),('c4',389120,61440),('public',80116,16713)]:
        path=BANKS[bank];rp=path/'rows.parquet';fp=path/'features.f16'
        for p in [rp,fp]:require(str(p) in provenance and digest(p)==provenance[str(p)],'Certified feature/row SHA changed: '+str(p));bindings[str(p)]=provenance[str(p)]
        require(fp.stat().st_size==total*512*2,'Feature size changed.');frame=pd.read_parquet(rp);frame=frame[frame.split.eq('validation')].copy().reset_index(drop=True)
        require(len(frame)==count and arrsha(frame.sample_ix.to_numpy())==static['sample_ix_sha256'][bank],'Only exact audited developer validation rows allowed.')
        require(set(frame.role)<=set(ROLES),'Unknown truth.');rows[bank]=frame;truth[bank]=frame.role.eq('system').to_numpy();mem=np.memmap(fp,mode='r',dtype=np.float16,shape=(total,512));bp=[];op=[];aa=[];original_a=[]
        with torch.inference_mode():
            for start in range(0,len(frame),4096):
                ix=frame.cache_ix.to_numpy()[start:start+4096];x=(torch.from_numpy(np.asarray(mem[ix],np.float32))-mt)/st
                zz=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,w['encoder.0.weight'],w['encoder.0.bias'])),w['head.weight'],w['head.bias']);require(torch.isfinite(zz).all(),'Nonfinite raw base logits.')
                bb,oo,al,ol=raw_scores(zz,torch.from_numpy(truth[bank][start:start+len(ix)]));bp.append(bb);op.append(oo);aa.append(ol);original_a.append(al)
        basep[bank]=np.concatenate(bp);orap[bank]=np.concatenate(op);loga[bank]=np.concatenate(aa);a0=np.concatenate(original_a);inner=(basep[bank]>0)&(basep[bank]<1)
        reconstructed=np.log(basep[bank][inner].astype(np.float64))-np.log1p(-basep[bank][inner].astype(np.float64))
        stats[bank]=dict(rows=count,system_support=int(truth[bank].sum()),sample_ix_sha256=arrsha(frame.sample_ix.to_numpy()),base_p0_equal_zero=int((basep[bank]==0).sum()),base_p0_equal_one=int((basep[bank]==1).sum()),
            raw_logodds_min=float(a0.min()),raw_logodds_max=float(a0.max()),interior_savedF32_logit_max_error=float(np.abs(reconstructed-a0[inner]).max()) if inner.any() else None,
            base_F32_score_sha256=arrsha(basep[bank]),oracle_F32_score_sha256=arrsha(orap[bank]),oracle_F64_logodds_sha256=arrsha(loga[bank]))
        del mem
    result=diagnose(orap,loga,truth,groups(rows));require(digest(CERT)==CERT_SHA and digest(PROTOCOL)==PROTOCOL_SHA and digest(MODEL)==MODEL_SHA,'Bound inputs changed during CPU diagnosis.')
    report=dict(stage='actual_gold_bound4_development_oracle_diagnostic',source_sha256=digest(__file__),certificate_sha256=CERT_SHA,bindings=bindings,original_model_sha256=MODEL_SHA,
        rows=sum(len(r) for r in rows.values()),cohorts=stats,groups=GROUPS,bound=4.,results=result,limitation=LIMITATION,numerics=NUMERICS,
        labels_used_for_untrainable_oracle=True,deployable=False,candidate_or_promotion=False,tests_read=False,gpu_used=False,new_sources_or_training=False,probability_NPZ_files_read=False,
        additional_feature_or_raw_logit_cache_written=False,primary_changed=False)
    output=Path(output);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=report['source_sha256'],results={k:v['min33'] for k,v in result.items() if isinstance(v,dict)},interpretation=result['interpretation'])))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--actual',action='store_true');p.add_argument('--output',type=Path,default=Path('/tmp/confirmation_review_bound4_oracle_actual.json'));args=p.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual(args.output)
    else:p.print_help()
