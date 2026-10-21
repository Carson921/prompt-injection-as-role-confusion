"""Independent640 residual results auditor; source/CPU preparation only.

Default help. Actual is fail-closed until root supplies final source/protocol
pins after feature-bank audit. No production imports or global inventory scan.
"""
import argparse,copy,hashlib,importlib.util,importlib.metadata,json,os,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis');BASE=HERE/'outputs/local-gptoss120b'
SOURCE=HERE/'systemness_complementary_head_readout.py';SOURCE_SHA='e74e40889ba49ab05815f0faac68c2a522ccc06627b09ff14379929bc53dbdf6';PROTOCOL_SHA='10c92c0af904b96d42e1ff8c9cbc12f7efb0371da08fdc3962aebd7b8d0d6f54'
DESIGN=BASE/'systemness-complementary-head-readout-preparation/root-prospective-design.json';DESIGN_SHA='f3dae68aae0003daf7382c925978313f984911cf428401d1bdd1eaa3b161c314'
OUT=BASE/'systemness-complementary-head-readout';CACHE=BASE/'systemness-expansion/cache/complementary-head-readout'
PRIOR_OUT=BASE/'systemness-long-mix-residual';PRIOR_AUDIT=PRIOR_OUT/'actual-audit/actual-development-audit.json';PRIOR_AUDIT_SHA='6dadc3a34ff9ea2c5c3415930051876165b529f3983d77eed757456259a1e58e'
LIB=Path('/tmp/long_data_long_mix_residual_results_audit.py');LIB_SHA='7e4fc4d94ec175905fee9d999dbaf94c3fa86df9df6e9633e258cab44cc9fbff'
ORIGINAL=[458,462,2045,460,459,2047,456,997];ADDED=[247,2248];ROLES=['system','user','cot','assistant','tool']
PRIOR_MODEL_SHA='649c16b33f5d7ed51c5a291e001232a0cf3c83506347962b7923004c392a9a8c';PRIOR_TAU=.9435555555555556;PRIOR_MAX_TAU=.7923434343434343
RECORDS=['new-oldmean','new-max33','matched-oldmean','matched-max33','oldhead-oldmean','oldhead-max33','ensemble-original','ensemble-max33','naturalCE-oldmean','naturalCE-max33','priorBinary-oldmean','priorBinary-max33','priorLongMix-oldmean','priorLongMix-max33'];ZERO=['zero-oldmean','zero-max33']
STRICT=['matched-max33','oldhead-max33','ensemble-max33','zero-max33','priorBinary-max33'];GUARDS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33','priorBinary-max33']
EXTRA_OUT=BASE/'systemness-complementary-head-features';EXTRA_BANK=Path('/dev/shm/role-analysis-complementary-heads');EXTRACTOR_SHA='964335b1dbdbea19219cc4e3db15409195376cd92dfd7125f7d4370758a34509'
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def labels(f):
    y=f.role.map(dict(zip(ROLES,range(5))));require(y.notna().all(),'Unknown role.');return y.to_numpy(np.int64)
def validate_p(p,n):require(p.shape==(n,5) and p.dtype==np.float32 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.allclose(p.sum(1),1,atol=2e-6),'Invalid saved F32 probability.')
def libraries():
    require(digest(LIB)==LIB_SHA,'Versioned independent library changed.');spec=importlib.util.spec_from_file_location('complementary_prior_independent',LIB);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib);return lib,lib.pure_math()
def processes():
    live=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name in {'systemness_complementary_head_readout.py','systemness_complementary_head_extract.py','systemness_complementary_head_development_plots.py'} for a in args if a):live.append(int(p.name))
    return live
def terminal_gate():
    require(SOURCE_SHA is not None and PROTOCOL_SHA is not None,'Final trainer/protocol pins unknown; actual forbidden.')
    require(not processes(),'Readout/extractor/gallery live; refuse partial reads.')
    require((OUT/'complete.json').is_file() and (OUT/'status.json').is_file(),'Completion/status absent.')
    require(read(OUT/'status.json').get('stage')=='validation_complete','Readout not terminal complete.')
class Joined:
    def __init__(self,base,added,lookup,duplicate=False):self.base,self.added,self.lookup,self.duplicate=base,added,np.asarray(lookup,np.int64),duplicate;self.shape=(len(base),640)
    def __getitem__(self,ix):
        ix=np.asarray(ix,np.int64);require(ix.ndim==1 and (self.lookup[ix]>=0).all(),'Unmapped C4fit cannot enter model/scaler.');x=np.asarray(self.base[ix]);return np.concatenate([x,x[:,:128] if self.duplicate else self.added[self.lookup[ix]]],1)
def mapping(extra,parents,expected=896838):
    require(len(extra)==expected and np.array_equal(extra.cache_ix,np.arange(expected)) and extra.sample_ix.is_unique and set(extra.parent_bank)=={'old','c4','natural'},'Extra row namespace/order differs.');out={}
    for name,frame in parents.items():
        group=extra[extra.parent_bank.eq(name)];ix=group.parent_cache_ix.to_numpy(np.int64);require(group.parent_cache_ix.is_unique and ((ix>=0)&(ix<len(frame))).all(),'Bad parent row indices.')
        for c in ['sample_ix','role','split']:require(np.array_equal(group[c],frame.iloc[ix][c]),'Parent '+c+' mapping differs.')
        v=np.full(len(frame),-1,np.int64);v[ix]=group.cache_ix.to_numpy(np.int64);out[name]=v
        require(group.split.eq('validation').all() if name=='c4' else (v>=0).all(),'C4fit entered or complete parent rows missing.')
    return out
def fit128(added,lookup,indices,chunk=4096):
    indices=np.asarray(indices,np.int64);require(len(indices)>1 and len(np.unique(indices))==len(indices) and (lookup[indices]>=0).all(),'Wrong original fit scaler scope.');n=0;mean=np.zeros(128,np.float64);m2=mean.copy()
    for s in range(0,len(indices),chunk):
        x=np.asarray(added[lookup[indices[s:s+chunk]]],np.float64);require(np.isfinite(x).all(),'Nonfinite scaler fit.');k=len(x);part=x.mean(0);d=part-mean;total=n+k;m2+=((x-part)**2).sum(0)+d*d*n*k/total;mean+=d*k/total;n=total
    return mean.astype(np.float32),np.maximum(np.sqrt(np.maximum(m2/n,0)),1e-6).astype(np.float32)
def initial640(base,math):
    import torch
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=torch.nn.Module();self.base.encoder=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.GELU(),torch.nn.Dropout(.1));self.base.head=torch.nn.Linear(128,5);self.correction=torch.nn.Sequential(torch.nn.Linear(640,32),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(32,1));torch.nn.init.zeros_(self.correction[-1].weight);torch.nn.init.zeros_(self.correction[-1].bias);self.base.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in base.items()})
    with torch.random.fork_rng(devices=[]):torch.random.default_generator.manual_seed(20261021);net=Net();state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    return state,math.weight_state_sha(state)
def arrays640(path,oldbase,mean,scale,math):
    with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['metadata']));state={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')};m,s=z['mean'].copy(),z['scale'].copy()
    require(meta['kind']=='binary-residual-640' and meta['dimensions']==640 and meta['base_dimensions']==512 and meta['base_hidden']==128 and meta['residual_hidden']==32 and meta['parameters']==20545 and meta['delta_formula']=='4*tanh(g)' and meta['roles']==ROLES and meta['base_model_sha256']==math.CURRENT_SHA,'Portable640 model convention differs.')
    shapes={'base.encoder.0.weight':(128,512),'base.encoder.0.bias':(128,),'base.head.weight':(5,128),'base.head.bias':(5,),'correction.0.weight':(32,640),'correction.0.bias':(32,),'correction.3.weight':(1,32),'correction.3.bias':(1,)}
    require(set(state)==set(shapes) and all(v.shape==shapes[k] and v.dtype==np.float32 and np.isfinite(v).all() for k,v in state.items()),'Malformed640 weights.')
    require(m.dtype==s.dtype==np.float32 and np.array_equal(m.view(np.uint32),mean.view(np.uint32)) and np.array_equal(s.view(np.uint32),scale.view(np.uint32)) and np.isfinite(m).all() and np.isfinite(s).all() and (s>0).all(),'Wrong arm scaler embedded.')
    base={k[5:]:v for k,v in state.items() if k.startswith('base.')}
    require(set(base)==set(oldbase) and all(np.array_equal(v.view(np.uint32),oldbase[k].view(np.uint32)) for k,v in base.items()),'Frozen512 base arrays changed.')
    require(meta['embedded_base_weight_sha256']==math.weight_state_sha(base,True) and meta['mean_sha256']==array_sha(m) and meta['scale_sha256']==array_sha(s),'Portable640 arrays metadata differs.')
    residual={'encoder.0.weight':state['correction.0.weight'],'encoder.0.bias':state['correction.0.bias'],'head.weight':state['correction.3.weight'],'head.bias':state['correction.3.bias']};return meta,base,residual


def replay640(base,residual,mean,scale,bank,rows,*,require_no_swaps=True):
    """Own rawF16->base+delta consumer; no producer loader or predictor.

    Returns full p and compact raw-logit invariance/actual-p transition proof.
    Non-system raw byte equality and raw argmax equality are structural checks.
    Actual F32 probability class swaps are separately measured and fail closed.
    """
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    wb={k:torch.from_numpy(v) for k,v in base.items()};wr={k:torch.from_numpy(v) for k,v in residual.items()}
    mt,st=torch.from_numpy(mean),torch.from_numpy(scale);parts=[];base_parts=[]
    oldhash,newhash=hashlib.sha256(),hashlib.sha256();conditional_changes=0;non_system_swaps=0;cot_assistant=0
    transitions=np.zeros((5,5),np.int64);by_truth={r:np.zeros((5,5),np.int64) for r in ROLES}
    prob_ties=0;underflow=0;non_sys_underflow=0;maxdelta=0.;delta_min=4.;delta_max=-4.;delta_sum=0.;conditional_p_changes=0;conditional_p_ties=0
    with torch.inference_mode():
        for start in range(0,len(rows),4096):
            frame=rows.iloc[start:start+4096];raw=torch.from_numpy(np.asarray(bank[frame.cache_ix.to_numpy()],np.float32));x=(raw-mt)/st
            require(torch.isfinite(x).all(),'Nonfinite residual input.')
            xb=x[:,:512].contiguous()
            z=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(xb,wb['encoder.0.weight'],wb['encoder.0.bias'])),wb['head.weight'],wb['head.bias'])
            hidden=torch.nn.functional.gelu(torch.nn.functional.linear(x,wr['encoder.0.weight'],wr['encoder.0.bias']))
            # eval dropout is identity; explicitly no stochastic operation here.
            score=torch.nn.functional.linear(hidden,wr['head.weight'],wr['head.bias']).reshape(-1)
            delta=4*torch.tanh(score);corrected=z.clone();corrected[:,0]+=delta
            require(torch.isfinite(corrected).all() and torch.isfinite(delta).all() and bool((delta.abs()<=4).all()),'Residual bound/finiteness violated.')
            maxdelta=max(maxdelta,float(delta.abs().max()));dv=delta.numpy()
            delta_min=min(delta_min,float(dv.min()));delta_max=max(delta_max,float(dv.max()));delta_sum+=float(dv.astype(np.float64).sum())
            before,after=z.numpy(),corrected.numpy();a,b=np.ascontiguousarray(before[:,1:]),np.ascontiguousarray(after[:,1:])
            require(a.dtype==b.dtype==np.float32 and np.array_equal(a.view(np.uint32),b.view(np.uint32)),'Non-system raw logits changed at bit level.')
            oldhash.update(a.tobytes());newhash.update(b.tobytes());conditional_changes+=int(np.count_nonzero(a.argmax(1)!=b.argmax(1)))
            oldp=torch.log_softmax(z,dim=1).exp().numpy().astype(np.float32);p=torch.log_softmax(corrected,dim=1).exp().numpy().astype(np.float32)
            validate_p(oldp,len(frame));validate_p(p,len(frame));oldpred,newpred=oldp.argmax(1),p.argmax(1)
            changed=(oldpred>0)&(newpred>0)&(oldpred!=newpred);non_system_swaps+=int(changed.sum());cot_assistant+=int(np.count_nonzero((oldpred==2)&(newpred==3)))
            conditional_p_changes+=int(np.count_nonzero(oldp[:,1:].argmax(1)!=p[:,1:].argmax(1)))
            conditional_p_ties+=int(np.count_nonzero((p[:,1:]==p[:,1:].max(1,keepdims=True)).sum(1)>1))
            non_sys_underflow+=int(np.count_nonzero(p[:,1:]==0))
            np.add.at(transitions,(oldpred,newpred),1)
            for r_ix,role in enumerate(ROLES):
                mask=labels(frame)==r_ix;np.add.at(by_truth[role],(oldpred[mask],newpred[mask]),1)
            # Report all complete-probability ties/zeros, never perturb them.
            order=np.sort(p,axis=1);prob_ties+=int(np.count_nonzero(order[:,-1]==order[:,-2]));underflow+=int(np.count_nonzero(p==0))
            parts.append(p);base_parts.append(oldp)
    proof=dict(rows=len(rows),non_system_raw_before_sha256=oldhash.hexdigest(),non_system_raw_after_sha256=newhash.hexdigest(),
        raw_non_system_bitexact=oldhash.hexdigest()==newhash.hexdigest(),conditional_raw_argmax_changes=conditional_changes,
        p_non_system_different_class_swaps=non_system_swaps,p_cot_to_assistant_swaps=cot_assistant,
        five_class_transition=transitions.tolist(),transitions_by_true_role={r:m.tolist() for r,m in by_truth.items()},
        full_p_argmax_tie_rows=prob_ties,p_zero_elements=underflow,maximum_abs_delta=maxdelta,
        delta_min=delta_min,delta_max=delta_max,delta_mean=delta_sum/len(rows),
        p_conditional_non_system_argmax_changes=conditional_p_changes,p_non_system_tie_rows=conditional_p_ties,
        p_non_system_underflow_values=non_sys_underflow)
    require(proof['raw_non_system_bitexact'] and conditional_changes==0,'Non-system conditional raw classifier changed.')
    if require_no_swaps:require(non_system_swaps==0,'Actual F32 p introduced a non-system class swap; no epsilon repair allowed.')
    return np.concatenate(parts),np.concatenate(base_parts),proof

def select_records(math,records,zero,legacy):
    require(len(records)==14 and {r['record'] for r in records}==set(RECORDS) and set(zero)==set(ZERO),'Fourteen records plus separate zero2 required.');index={r['record']:r['validation'] for r in records};index.update(zero);c=index['new-max33'];minimum=c['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARDS:
        g=math.guard(c,index[name]);guards[name]=g;reasons.extend(name+':'+s for s in g['rejection_reasons'])
        if name in STRICT:
            gain=minimum-index[name]['min33'] if minimum is not None and index[name]['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    g=math.guard(c,legacy,True);guards['original-savedGPU']=g;reasons.extend('original-savedGPU:'+s for s in g['rejection_reasons'])
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,naturalCE_is_diagnostic_only=True,priorLongMix_is_diagnostic_only=True,priorBinary_max33_is_strict_and_guard=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)

def actual(output):
    terminal_gate();output=Path(output);require(output.resolve().is_relative_to(Path('/tmp')),'Auditor writes /tmp only.');lib,math=libraries();bindings={};probability_bindings={};old_state=(math.RECORDS.copy(),math.GUARD_CONTROLS.copy(),math.STRICT_CONTROLS.copy(),math.MAX_DIFFERENCE)
    def bind(p,want=None):
        p=Path(p);require(not str(p).startswith('/workspace') and p.is_file(),'Closed file missing or forbidden.');got=digest(p);require(want is None or want==got,'SHA differs: '+str(p));bindings[str(p.resolve())]=got;return got
    def p_read(p,want,frame):probability_bindings[str(Path(p).resolve())]=bind(p,want);return math.read_probability(p,want,frame)
    bind(__file__);bind(LIB,LIB_SHA);bind(lib.MATH_SOURCE,lib.MATH_SHA);bind(SOURCE,SOURCE_SHA);bind(DESIGN,DESIGN_SHA);bind(OUT/'protocol.json',PROTOCOL_SHA);bind(OUT/'status.json');done_sha=bind(OUT/'complete.json');done=read(OUT/'complete.json');plan_sha=bind(OUT/'search-plan.json');plan=read(OUT/'search-plan.json');decl=read(OUT/'protocol.json');static=decl['static'];proto=decl['protocol'];sig=plan['signature']
    artifacts={'protocol.json','search-plan.json','scaler-record.json','controls.json','zero-controls.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt'}
    require(done['signature']==sig and done['source_sha256']==SOURCE_SHA and done['tests_read'] is False and done['primary_changed'] is False and done['new_independent_confirmation_required'] is True and set(done['artifact_sha256'])==artifacts and set(done['candidate_sha256'])=={'new','matched'},'Terminal completion identity/coverage differs.')
    for n,want in done['artifact_sha256'].items():bind(OUT/n,want)
    require(proto==plan['protocol'] and proto['records']==RECORDS and proto['dimensions']==640 and proto['parameters']==20545 and proto['original_head_ids']==ORIGINAL and proto['added_head_ids']==ADDED
        and proto['promotion']['strict_controls']==STRICT and proto['promotion']['guard_controls']==GUARDS and proto['promotion']['absolute_min33']==.9 and proto['tests_read'] is False and proto['primary_changed'] is False,'Frozen640 protocol differs.')
    lib.close(proto['neural'],dict(seed=20261021,residual_hidden=32,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=4096,instances=641493,steps=157,last_batch=2517,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5),'neural')
    require(proto['groups']==math.GROUPS and proto['feasible']==math.FEASIBLE and proto['max_FPR']==.05 and proto['zero_controls']==ZERO and proto['CPU']==dict(threads=4,float32_matmul_precision='highest',dtype='float32',chunk=4096),'Grid or CPU runtime protocol differs.')
    require(static['source_sha256']==SOURCE_SHA and static['design_sha256']==DESIGN_SHA and static['validation_probability_arrays_read'] is False and static['global_inventory_scanned'] is False and static['selected_head_ids']==ORIGINAL+ADDED,'Static source/scope differs.')
    for n,want in static['dependencies_sha256'].items():bind(HERE/n,want)
    for p,want in static['closed_files_sha256'].items():bind(p,want)
    require(static['packages']=={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},'CPU package versions changed.')
    bind(PRIOR_AUDIT,PRIOR_AUDIT_SHA);prior=read(PRIOR_AUDIT)
    require(prior['stage']=='actual_completed_long_mix_residual_development_audit' and prior['audit_passed'] is True and prior['test_data_or_results_read'] is False and prior['model_sha256']['new']==PRIOR_MODEL_SHA and static['prior_actual_audit_sha256']==PRIOR_AUDIT_SHA,'Prior independent longmix certificate differs.')
    for f,k in [('complete.json','trainer_complete_sha256'),('protocol.json','trainer_protocol_sha256'),('records.json','trainer_records_sha256'),('selection.json','trainer_selection_sha256')]:bind(PRIOR_OUT/f,prior[k])
    # Exact static closure and certificate bind inherited references, without
    # calling prior actual/global verification or reselecting a prior model.
    extra_complete=read(EXTRA_OUT/'extraction-complete.json');bind(EXTRA_OUT/'extraction-complete.json',static['extra_complete_sha256']);bind(EXTRA_BANK/'rows.parquet',extra_complete['rows_sha256']);require(extra_complete['rows']==896838 and extra_complete['dimensions']==128 and extra_complete['all_rows_written'] is True and extra_complete['all_finite'] is True and extra_complete['source_sha256']==EXTRACTOR_SHA and extra_complete['added_head_ids']==ADDED and extra_complete['tests_read'] is False,'Extra128 bank identity differs.')
    audit_paths=[p for p,v in static['closed_files_sha256'].items() if v==static['extra_audit_sha256']];require(len(audit_paths)==1,'Unique pinned extra-bank audit absent.');ea=read(audit_paths[0]);require(ea['stage']=='actual_completed_complementary_head_feature_audit' and ea['audit_passed'] is True and ea['test_data_or_results_read'] is False and ea['extractor_source_sha256']==EXTRACTOR_SHA and ea['extractor_complete_sha256']==static['extra_complete_sha256'] and ea['features_sha256']==extra_complete['features_sha256'] and ea['rows_sha256']==extra_complete['rows_sha256'],'Independent extra-bank certificate differs.')
    tables={};banks={};counts={'old':755282,'c4':389120,'public':80116};folders={'old':BASE/'systemness-head-features','c4':BASE/'systemness-long-augmentation','public':BASE/'systemness-natural-dev-features'}
    import pyarrow.parquet as pq
    for b in counts:
        path=folders[b]/'extraction-complete.json';bind(path);c=read(path);folder=math.BANK_PATHS[b];require(c['rows']==counts[b] and c['dimensions']==512 and c['all_rows_written'] is True and c['all_finite'] is True and c['selected_head_ids']==ORIGINAL,'Original512 parent differs.')
        bind(folder/'rows.parquet',c['rows_sha256']);require((folder/'features.f16').stat().st_size==counts[b]*512*2 and static['feature_files_sha256'][str(folder/'features.f16')]==c['features_sha256'],'Parent512 shape/static hash differs.');bind(folder/'features.f16',c['features_sha256']);columns=[n for n in pq.ParquetFile(folder/'rows.parquet').schema.names if n not in ['prompt','token']];tables[b]=pd.read_parquet(folder/'rows.parquet',columns=columns);banks[b]=np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(counts[b],512))
    fit,rows,natural,part=math.source_partitions(tables['old'],tables['c4'],tables['public']);lib.close(part,static['original_partitions'],'original fit partition')
    for n,f in [('fit',fit),('old',rows['old']),('c4',rows['c4']),('natural-fit',natural),('public',rows['public'])]:require(array_sha(f.sample_ix.to_numpy())==static['sample_ix_sha256'][n],'Fixed sample order differs: '+n)
    ecols=['cache_ix','sample_ix','parent_bank','parent_cache_ix','role','split'];erows=pd.read_parquet(EXTRA_BANK/'rows.parquet',columns=ecols);maps=mapping(erows,{'old':tables['old'],'c4':tables['c4'],'natural':tables['public']});maps['public']=maps.pop('natural')
    ep=EXTRA_BANK/'features.f16';require(ep.stat().st_size==896838*128*2 and static['feature_files_sha256'][str(ep)]==extra_complete['features_sha256'],'Extra128 shape/static hash differs.');bind(ep,extra_complete['features_sha256']);added=np.memmap(ep,mode='r',dtype=np.float16,shape=(896838,128))
    virtual={arm:{b:Joined(bank,added,maps[b],arm=='matched') for b,bank in banks.items()} for arm in ['new','matched']}
    oldscaler=BASE/'systemness-head-probes/standardization.npz';bind(oldscaler,static['standardization_sha256'])
    with np.load(oldscaler,allow_pickle=False) as z:om,oscale=z['mean'].copy(),z['scale'].copy()
    require(om.shape==oscale.shape==(512,) and om.dtype==oscale.dtype==np.float32 and array_sha(om)==static['old512_mean_sha256'] and array_sha(oscale)==static['old512_scale_sha256'],'Old512 fit scaler changed.')
    em,es=fit128(added,maps['old'],fit.cache_ix.to_numpy());scalers=dict(new=(np.r_[om,em].astype(np.float32),np.r_[oscale,es].astype(np.float32)),matched=(np.r_[om,om[:128]].astype(np.float32),np.r_[oscale,oscale[:128]].astype(np.float32)));scalers['zero']=scalers['new']
    sp=CACHE/'standardization.npz';sr=read(OUT/'scaler-record.json');newscaler_sha=bind(sp,sr['standardization_sha256']);require(sr['protocol_sha256']==PROTOCOL_SHA and sr['old512_mean_sha256']==array_sha(om) and sr['old512_scale_sha256']==array_sha(oscale),'640 scaler provenance differs.')
    descriptor=dict(fit_rows=578090,fit_only=True,natural_rows_used=0,validation_rows_used=0,chunk=4096,floor=1e-6,variance='Population F64 parallel-Welford',fit_sample_ix_sha256=array_sha(fit.sample_ix.to_numpy()),extra_fit_cache_ix_sha256=array_sha(maps['old'][fit.cache_ix.to_numpy()]))
    lib.close(sr['fit'],descriptor,'FIT-only new128 scaler')
    with np.load(sp,allow_pickle=False) as z:
        require(set(z.files)=={'new_mean','new_scale','matched_mean','matched_scale'},'Scaler NPZ coverage differs.')
        for arm in ['new','matched']:
            for k,v in zip(['mean','scale'],scalers[arm]):require(np.array_equal(z[arm+'_'+k].view(np.uint32),v.view(np.uint32)),'Independent FIT scaler bits differ: '+arm+'/'+k)
    expected_sig=dict(source_sha256=SOURCE_SHA,protocol_sha256=PROTOCOL_SHA,static_sha256=canonical(static),selected_head_ids=ORIGINAL+ADDED,standardization_sha256=newscaler_sha,base_model_sha256=math.CURRENT_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],feature_files_sha256=static['feature_files_sha256'],extra_rows_sha256=extra_complete['rows_sha256']);lib.close(sig,expected_sig,'640 signature')
    bind(static['base_model_path'],math.CURRENT_SHA);_,base=math.model_arrays(static['base_model_path'],om,oscale);initial,initsha=initial640(base,math);require(initsha==static['initial_weight_sha256']==plan['initial_weight_sha256'] and math.weight_state_sha(base,True)==static['embedded_base_weight_sha256'],'640 full initialization/frozen base differs.')
    probs={};models={};budgets={};replays={};training_sha={}
    def load_probs(name,hashes):require(set(hashes)==set(rows),'Probability bank coverage differs.');return {b:p_read(CACHE/f'{name}-{b}-validation.npz',hashes[b],f) for b,f in rows.items()}
    def replay(name,want,rawproof,extra_meta):
        path=OUT/'models'/f'{name}.npz';models[name]=bind(path,want);mean,scale=scalers[name];meta,bb,rr=arrays640(path,base,mean,scale,math)
        expected=dict(kind='binary-residual-640',arm=name,dimensions=640,base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',roles=ROLES,base_model_sha256=math.CURRENT_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),selected_head_ids=ORIGINAL+(ORIGINAL[:2] if name=='matched' else ADDED),signature=sig,source_sha256=SOURCE_SHA,plan_sha256=plan_sha,initial_weight_sha256=initsha,standardization_sha256=newscaler_sha,tests_read=False,**extra_meta)
        for k,v in expected.items():lib.close(meta[k],v,'portable/'+name+'/'+k)
        replays[name]={}
        for b,frame in rows.items():
            computed,bp,inv=replay640(bb,rr,mean,scale,virtual['matched' if name=='matched' else 'new']['public' if b=='public' else b],frame);difference=float(np.abs(computed-probs[name][b]).max());require(difference<5e-6,'Independent full640 F32 functional replay differs: '+name+'/'+b);lib.raw_evidence(rawproof[b],inv,name+'/'+b)
            replays[name][b]=dict(tokens=len(frame),maximum_probability_difference=difference,independent_probability_sha256=array_sha(computed),saved_probability_sha256=array_sha(probs[name][b]),base_probability_sha256=array_sha(bp),raw_invariance=inv)
        return rr
    for arm in ['new','matched']:
        cp=OUT/'candidates'/f'{arm}.json';bind(cp,done['candidate_sha256'][arm]);candidate=read(cp);training_sha[arm]=bind(OUT/'training'/f'{arm}.json',candidate['training_sha256']);training=read(OUT/'training'/f'{arm}.json');amounts=math.history_audit(training['history'],training['best_epoch']);budgets[arm]=amounts
        for obj in [candidate,training]:
            require(obj['arm']==arm and obj['signature']==sig and obj['plan_sha256']==plan_sha and obj['initial_weight_sha256']==initsha,'Arm ownership/init differs.')
            for k,v in amounts.items():lib.close(obj[k],v,arm+'/budget/'+k)
        require(candidate['source_sha256']==SOURCE_SHA and candidate['model_sha256']==training['model_sha256'] and candidate['gpu_old_probability_sha256']==training['gpu_old_probability_sha256'] and candidate['tests_read'] is False and training['new_C4_predictions_read'] is False and training['public_validation_predictions_read'] is False and training['embedded_base_weight_sha256']==static['embedded_base_weight_sha256'] and training['base_eval_all_epochs'] is True and training['base_gradient_observations']==0 and training['optimizer_only_residual'] is True and candidate['cuda_visible_devices'] in ['0','1'] and 'A40' in candidate['cuda_device_name'],'Best-before-new/frozen base/GPU evidence differs.')
        math.parity_proof(candidate['portable_evidence'],natural);probs[arm]=load_probs(arm,candidate['probability_sha256']);gp=p_read(CACHE/f'{arm}-gpu-old-validation.npz',candidate['gpu_old_probability_sha256'],rows['old']);tau=math.legacy_mean(rows['old'],gp);gv=math.evaluate(rows,dict(probs[arm],old=gp),tau);lib.close(math.old_validation_record(gv,rows['old']),candidate['old_early_stop_validation'],arm+'/old5');lib.close(gv['old5_mean_f1'],training['history'][training['best_epoch']-1]['old5_mean_f1'],arm+'/best');drift=math.probability_drift(gp,probs[arm]['old'],math.CURRENT_TAU);lib.close(drift,candidate['full_old_GPU_CPU_drift'],arm+'/GPUCPU');require(drift['maximum_probability_difference']<5e-4,'GPUold versus CPUold drift exceeds gate.');replay(arm,candidate['model_sha256'],candidate['raw_invariance'],dict(best_epoch=amounts['best_epoch'],best_steps=amounts['best_steps']))
    controls=read(OUT/'controls.json');require(controls['signature']==sig and controls['prior_actual_audit_sha256']==PRIOR_AUDIT_SHA and controls['extra_complete_sha256']==static['extra_complete_sha256'] and controls['extra_audit_sha256']==static['extra_audit_sha256'] and controls['reference_probabilities']==static['reference_probabilities'] and controls['reference_probabilities_recomputed'] is False and controls['tests_read'] is False and controls['naturalCE_diagnostic_only'] is True and controls['priorLongMix_diagnostic_only'] is True and controls['priorBinary_max33_is_strict_and_guard'] is True,'Closed inherited controls differ.')
    require(set(static['reference_probabilities'])=={'oldhead','ensemble','naturalCE','priorBinary','priorLongMix'},'Inherited physical control set differs.')
    for name,items in static['reference_probabilities'].items():
        probs[name]={}
        for b,item in items.items():require(item['sample_ix_sha256']==array_sha(rows[b].sample_ix.to_numpy()) and prior['probability_bindings'][str(Path(item['path']).resolve())]==item['sha256'],'Inherited p not prior-audited.');probs[name][b]=p_read(item['path'],item['sha256'],rows[b])
    lib.close(controls['physical_model_sha256'],dict(oldhead=math.CURRENT_SHA,ensemble=static['prior_model_sha256'],naturalCE='204f4a9dbfb7855460ccb210ca703aca721b974a87059be9134f55c4f7b5cba5',priorBinary=lib.PRIOR_MODEL_SHA,priorLongMix=PRIOR_MODEL_SHA),'Inherited model identities')
    ref=static['old_reference'];gpu={b:p_read(ref[k+'_path'],ref[k+'_sha256'],rows[b]) for b,k in [('old','old'),('c4','new')]};legacy=math.evaluate(rows,dict(gpu,public=probs['oldhead']['public']),math.CURRENT_TAU);lib.close(controls['original_savedGPU_validation'],legacy,'Original savedGPU control')
    zero=read(OUT/'zero-controls.json');require(zero['signature']==sig and zero['both_arm_training_sha256']==training_sha and zero['original_threshold']==math.CURRENT_TAU and zero['trained_record'] is False and zero['tests_read'] is False and zero['model_path']==str(OUT/'models/zero.npz'),'Zero frozen-source metadata differs.');math.parity_proof(zero['portable_evidence'],natural);probs['zero']=load_probs('zero',zero['probability_sha256']);zr=replay('zero',zero['model_sha256'],zero['raw_invariance'],dict(zero_residual=True))
    require(np.array_equal(zr['encoder.0.weight'],initial['correction.0.weight']) and np.array_equal(zr['encoder.0.bias'],initial['correction.0.bias']) and not np.any(zr['head.weight']) and not np.any(zr['head.bias']),'Zero full initialization differs.')
    for b in rows:
        inv=replays['zero'][b]['raw_invariance'];require(inv['delta_min']==inv['delta_max']==inv['delta_mean']==0,'Zero residual changed logit.');drift=math.probability_drift(probs['oldhead'][b],probs['zero'][b],math.CURRENT_TAU);lib.close(drift,zero['oldhead_commonCPU_drift'][b],'Zero commonCPU drift');require(drift['maximum_probability_difference']<5e-4,'Zero drift exceeds gate.')
    ztau,zproof,zannex=math.direct_grid(rows,probs['zero']);zero_records=[dict(record='zero-oldmean',validation=math.evaluate(rows,probs['zero'],math.CURRENT_TAU)),dict(record='zero-max33',validation=math.evaluate(rows,probs['zero'],ztau))];zero_map={r['record']:r['validation'] for r in zero_records};lib.close(zero['records'],zero_map,'Zero records');lib.close(zero['max33_full_grid_proof'],zproof,'Zero full33 grid')
    records=[];grids={'zero':dict(proof=zproof,annex=zannex)};saved=read(OUT/'records.json');index={r['record']:r for r in saved};require(len(saved)==14 and set(index)==set(RECORDS),'Fourteen record identities differ.')
    original_taus={'oldhead':math.CURRENT_TAU,'ensemble':math.ENSEMBLE_TAU,'naturalCE':.7772222222222223,'priorBinary':lib.PRIOR_TAU,'priorLongMix':PRIOR_TAU}
    for name in RECORDS:
        model,objective=name.split('-');p=probs[model]
        if objective=='max33':tau,proof,annex=math.direct_grid(rows,p);grids[model]=dict(proof=proof,annex=annex)
        else:tau=original_taus.get(model);tau=math.legacy_mean(rows['old'],p['old']) if tau is None else tau;proof=None
        record=dict(record=name,model=model,objective=objective,validation=math.evaluate(rows,p,tau),full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False)
        if name=='naturalCE-max33':require(tau==.6008080808080808,'Frozen CE max33 changed.')
        if name=='priorBinary-max33':require(tau==lib.PRIOR_MAX33_TAU and record['validation']['min33']==.8990885416666666,'Prior binary max33 changed.')
        if name=='priorLongMix-max33':require(tau==PRIOR_MAX_TAU,'Prior longmix max33 changed.')
        lib.close(index[name],record,name);records.append(record)
    selection=select_records(math,records,zero_map,legacy);published=read(OUT/'selection.json')
    for k,v in selection.items():lib.close(published[k],v,'Selection/'+k)
    require(published['source_sha256']==SOURCE_SHA and published['plan_sha256']==plan_sha and published['model_path']==str(OUT/'models/new.npz') and published['model_sha256']==models['new'] and published['system_threshold']==index['new-max33']['validation']['system_threshold'],'Selected primary/threshold identity differs.')
    for f,k in [('records.json','records_sha256'),('controls.json','controls_sha256'),('zero-controls.json','zero_controls_sha256'),('scaler-record.json','scaler_record_sha256')]:require(published[k]==done['artifact_sha256'][f],'Selection artifact link differs.')
    lib.verify_csv(pd.read_csv(OUT/'validation-metrics.csv'),records,RECORDS);lib.verify_csv(pd.read_csv(OUT/'zero-control-metrics.csv'),zero_records,ZERO)
    counts=read(OUT/'source-counts.json');binary_counts=read(BASE/'systemness-binary-residual/source-counts.json');expected_counts=binary_counts|dict(experiment='complementary-two-head-640-residual',same_existing_rows=True,same_validation_rows=True,dimensions=640,trainable_parameters=20545,informative_heads=dict(new=10,matched=8),extra128_scaler_fit_rows=578090,extra128_scaler_natural_rows=0,extra128_scaler_validation_rows=0,original_and_natural_fit_unchanged=True,new_arm_bank_rows=dict(old=578090,c4=0,natural=63403),matched_arm_bank_rows=dict(old=578090,c4=0,natural=63403),priorLongMix_diagnostic_only=True,limitation=proto['limitation']);lib.close(counts,expected_counts,'Source counts');require(counts['per_epoch']==641493 and counts['steps']==157 and counts['last_batch']==2517,'Epoch budget differs.')
    source_counts={g:dict(bank=b,tokens=len(ix),sources=int(rows[b].iloc[ix].group_id.nunique()),role_counts={r:int(rows[b].iloc[ix].role.eq(r).sum()) for r in ROLES},sample_ix_sha256=array_sha(rows[b].iloc[ix].sample_ix.to_numpy())) for g,(b,ix) in math.group_indices(rows).items()}
    terminal_gate();require(old_state==(math.RECORDS,math.GUARD_CONTROLS,math.STRICT_CONTROLS,math.MAX_DIFFERENCE),'Frozen independent numerical library mutated.')
    for p,want in bindings.items():require(digest(p)==want,'Artifact changed during audit: '+p)
    report=dict(stage='actual_completed_complementary_head_readout_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),trainer_source_sha256=SOURCE_SHA,trainer_protocol_sha256=PROTOCOL_SHA,trainer_complete_sha256=done_sha,trainer_selection_sha256=bindings[str((OUT/'selection.json').resolve())],trainer_records_sha256=bindings[str((OUT/'records.json').resolve())],trainer_model_record_sha256={a:bindings[str((OUT/'candidates'/f'{a}.json').resolve())] for a in ['new','matched']},prior_actual_audit_sha256=PRIOR_AUDIT_SHA,model_sha256=models,provenance_sha256=bindings,probability_bindings=probability_bindings,fourteen_records=records,zero_records=zero_records,selection=selection,independent_grid_proofs=grids,training_budgets=budgets,full_independent_CPU_predictions=replays,source_counts=source_counts,scaler_fit=descriptor,full_old512_scaler_bitexact=True,matched_duplicate_rank01_scaler_bitexact=True,csv_primary_group_rows=154,csv_descriptive_pool_rows=14,zero_csv_primary_group_rows=22,zero_csv_pool_rows=2,combined_csv_rows=192,only_new_max33_eligible=True,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,reused_development_validation=True,new_independent_confirmation_required=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=report['auditor_source_sha256'],selection=selection)))

def preflight():
    folder=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-readout-audit-'));start=digest(__file__);lib,math=libraries();state=(math.RECORDS.copy(),math.GUARD_CONTROLS.copy(),math.STRICT_CONTROLS.copy(),math.MAX_DIFFERENCE);checks=[]
    def check(name,v=True):require(v,'CPU check failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,AssertionError):checks.append(name);return
        raise RuntimeError('Expected refusal: '+name)
    def forbidden(*a,**k):raise LookupError('Production artifact read forbidden.')
    with patch.dict(globals(),dict(SOURCE_SHA=None,PROTOCOL_SHA=None,read=forbidden)),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden):reject('unknown_final_pins_block_before_artifacts',lambda:actual(folder/'forbidden.json'))
    with patch.dict(globals(),dict(SOURCE_SHA='mock',PROTOCOL_SHA='mock',processes=lambda:[1],read=forbidden)),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden):reject('live_process_gate_before_metadata_and_probabilities',lambda:actual(folder/'forbidden.json'))
    rng=np.random.RandomState(37);basebank=rng.normal(size=(12,512)).astype(np.float16);add=rng.normal(size=(12,128)).astype(np.float16);lookup=np.arange(12)[::-1].copy();joined=Joined(basebank,add,lookup);duplicate=Joined(basebank,add,lookup,True);ix=np.array([9,1,7])
    check('640base_prefix_and_rank01_duplicate_columns_bitexact',np.array_equal(joined[ix][:,:512].view(np.uint16),basebank[ix].view(np.uint16)) and np.array_equal(duplicate[ix][:,512:].view(np.uint16),basebank[ix,:128].view(np.uint16)) and np.array_equal(joined[ix][:,512:],add[lookup[ix]]))
    badmap=lookup.copy();badmap[0]=-1;reject('unmapped_newC4fit_lookup_rejected',lambda:Joined(basebank,add,badmap)[np.array([0])])
    em,es=fit128(add,lookup,np.array([1,3,5,7,9]),chunk=2);expected=add[lookup[[1,3,5,7,9]]].astype(np.float64)
    check('FIT_only_F64_population_scaler_against_direct_formula',np.allclose(em,expected.mean(0).astype(np.float32),atol=1e-7,rtol=1e-7) and np.allclose(es,expected.std(0).astype(np.float32),atol=1e-7,rtol=1e-7))
    zeros=np.zeros((12,128),np.float16);_,zs=fit128(zeros,lookup,np.arange(6),chunk=4);check('constant_extra_coordinates_scale_floor',np.array_equal(zs,np.full(128,1e-6,np.float32)))
    base={'encoder.0.weight':rng.normal(scale=.03,size=(128,512)).astype(np.float32),'encoder.0.bias':rng.normal(scale=.01,size=128).astype(np.float32),'head.weight':rng.normal(scale=.03,size=(5,128)).astype(np.float32),'head.bias':rng.normal(scale=.01,size=5).astype(np.float32)}
    initial,h=initial640(base,math);initial2,h2=initial640(base,math);check('same640_full_seed_initialization_and_zero_last',h==h2 and initial['correction.0.weight'].shape==(32,640) and not np.any(initial['correction.3.weight']) and not np.any(initial['correction.3.bias']))
    mean=np.zeros(640,np.float32);scale=np.ones(640,np.float32);rr={'encoder.0.weight':initial['correction.0.weight'],'encoder.0.bias':initial['correction.0.bias'],'head.weight':initial['correction.3.weight'],'head.bias':initial['correction.3.bias']};frame=pd.DataFrame(dict(cache_ix=np.arange(12),role=[ROLES[i%5] for i in range(12)]))
    p,bp,proof=replay640(base,rr,mean,scale,joined,frame);check('zero640_cpu_functional_base512_contiguous_exact',np.array_equal(p.view(np.uint32),bp.view(np.uint32)) and proof['raw_non_system_bitexact'] and proof['delta_min']==proof['delta_max']==0)
    rr2={k:v.copy() for k,v in rr.items()};rr2['head.weight'][:]=.01;p2,bp2,proof2=replay640(base,rr2,mean,scale,joined,frame);check('nonzero640_functional_delta_rawtail_invariant',np.isfinite(p2).all() and proof2['maximum_abs_delta']<=4 and proof2['raw_non_system_bitexact'] and proof2['conditional_raw_argmax_changes']==0 and proof2['p_non_system_different_class_swaps']==0)
    meta=dict(kind='binary-residual-640',dimensions=640,base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',roles=ROLES,base_model_sha256=math.CURRENT_SHA,embedded_base_weight_sha256=math.weight_state_sha(base,True),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale));path=folder/'toy-portable.npz'
    np.savez_compressed(path,mean=mean,scale=scale,metadata=np.array(json.dumps(meta)),**{'weight::'+k:v for k,v in initial.items()});_,bb,loaded=arrays640(path,base,mean,scale,math);pl,_,_=replay640(bb,loaded,mean,scale,joined,frame);check('portable640_saved_fullbase_and_residual_replay',np.array_equal(pl.view(np.uint32),p.view(np.uint32)))
    wrong={k:v.copy() for k,v in base.items()};wrong['head.bias'][0]+=1;reject('portable_embedded_original_base_replacement_rejected',lambda:arrays640(path,wrong,mean,scale,math))
    rows,ps=math.synthetic();tau,grid,annex=math.direct_grid(rows,ps);value=math.evaluate(rows,ps,tau);check('independent11group_33stat_F32_consumer_grid',len(value['groups'])==11 and grid['min33']==value['min33'] and grid['old5_mean_f1']==value['old5_mean_f1'])
    records=[dict(record=n,model=n.split('-')[0],objective=n.split('-')[1],validation=copy.deepcopy(value),full_grid_proof=None,diagnostic=n!='new-max33',tests_read=False) for n in RECORDS];zero={n:copy.deepcopy(value) for n in ZERO};selected=select_records(math,records,zero,value)
    check('five_strict_eight_guard_savedGPU_and_no_diagnostic_fallback',not selected['promoted'] and selected['no_fallback'] and len(selected['min33_gains'])==5 and len(selected['guards'])==9 and selected['priorLongMix_is_diagnostic_only'])
    records[1]['validation']['min33']=.899999;selected=select_records(math,records,zero,value);check('absolute90_not_rounded_up','absolute_min33_below_90_percent' in selected['rejection_reasons']);records[1]['validation']['min33']=value['min33']
    def flat(rs):
        return pd.DataFrame([dict(record=r['record'],group=g,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))},**{'role_recall_'+role:v for role,v in m['recall'].items()}) for r in rs for g,m in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items()])
    zrecords=[dict(record=n,validation=v) for n,v in zero.items()];f=flat(records);zf=flat(zrecords);lib.verify_csv(f,records,RECORDS);lib.verify_csv(zf,zrecords,ZERO);check('fourteen_plus_zero_two_192CSV_values_and_NA',len(f)==168 and len(zf)==24)
    scores=[.7+i*.005 if i<9 else .74 for i in range(17)];history=[dict(epoch=i+1,loss=.3,instances=641493,steps=157,cumulative_steps=(i+1)*157,old5_mean_f1=v) for i,v in enumerate(scores)];amounts=math.history_audit(history,9);check('641493_157_last2517_earlystop_budget',amounts['epochs']==17 and amounts['best_steps']==9*157 and amounts['actual_steps']==17*157 and 641493%4096==2517)
    check('frozen_independent_library_globals_unchanged',state==(math.RECORDS,math.GUARD_CONTROLS,math.STRICT_CONTROLS,math.MAX_DIFFERENCE));check('source_unchanged',digest(__file__)==start)
    report=dict(stage='synthetic_CPU_complementary640_result_auditor_preflight',audit_passed=True,source_sha256=start,check_count=len(checks),checks=checks,trainer_source_pin=SOURCE_SHA,trainer_protocol_pin=PROTOCOL_SHA,actual_enabled=False,actual_data_or_artifacts_read=False,production_module_imported=False,real_feature_probability_or_weights_read=False,source_text_materialized=False,gpu_used=False,actual_run=False)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),source_sha256=start,checks=len(checks))))

def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group();g.add_argument('--preflight',action='store_true');g.add_argument('--actual',action='store_true');parser.add_argument('--output',default='/tmp/confirmation_review_complementary_readout_results_audit.json');a=parser.parse_args()
    if a.preflight:preflight()
    elif a.actual:actual(a.output)
    else:parser.print_help()
if __name__=='__main__':main()
