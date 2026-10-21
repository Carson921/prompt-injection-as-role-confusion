"""Conditional640-feature frozen-base system residual, one candidate/control.

Default help. --preflight uses only synthetic CPU/tmp. Actual declare/train is
closed until the added128-head bank is complete and independently audited with
root-pinned certificates. No LM, source changes, tests or automatic primary use.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
import systemness_data as data
import systemness_head_probes as hp
import systemness_probes as probes
import systemness_long_probes as lp
import systemness_long_augmentation_probes as aug
import systemness_multiseed_ensemble as ms
import systemness_natural_fit_probes as nf
import systemness_binary_residual as br
import systemness_long_mix_residual as prior
import systemness_complementary_head_extract as extra

OUT=data.runtime.OUT/'systemness-complementary-head-readout'
CACHE=data.CACHE/'complementary-head-readout'
DESIGN=data.runtime.OUT/'systemness-complementary-head-readout-preparation/root-prospective-design.json'
DESIGN_SHA='f3dae68aae0003daf7382c925978313f984911cf428401d1bdd1eaa3b161c314'
EXTRA_COMPLETE_SHA='a48ad6d326f1665e6048952d1274b5f880207908ca46bd135d37c787788515e4'
EXTRA_AUDIT_PATH='/root/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/local-gptoss120b/systemness-complementary-head-features/actual-audit/actual-feature-bank-audit.json'
EXTRA_AUDIT_SHA='e58804dadb6228e5c59a85b19e89c05cfaf4707d8390815067472a3857b89845'
EXTRACTOR_SOURCE_SHA='964335b1dbdbea19219cc4e3db15409195376cd92dfd7125f7d4370758a34509'
PRIOR_AUDIT=prior.OUT/'actual-audit/actual-development-audit.json'
PRIOR_AUDIT_SHA='6dadc3a34ff9ea2c5c3415930051876165b529f3983d77eed757456259a1e58e'
PRIOR_MODEL_SHA='649c16b33f5d7ed51c5a291e001232a0cf3c83506347962b7923004c392a9a8c'
PRIOR_TAU=.9435555555555556
PRIOR_MAX33_TAU=.7923434343434343
ORIGINAL=[458,462,2045,460,459,2047,456,997]
ADDED=[247,2248]
ARMS=['new','matched'];RECORDS=list(prior.RECORDS)+['priorLongMix-oldmean','priorLongMix-max33']
GROUPS,FEASIBLE,PUBLIC,FULL,OLD5=nf.GROUPS,nf.FEASIBLE,nf.PUBLIC,nf.FULL,nf.OLD5
GUARD_CONTROLS=list(prior.GUARD_CONTROLS);STRICT_CONTROLS=list(prior.STRICT_CONTROLS)
FIT,EXTRA,INSTANCES,BATCH,STEPS=578090,63403,641493,4096,157
BASE_SHA,BASE_PATH=br.BASE_SHA,br.BASE_PATH
NATURAL_CE_SHA,NATURAL_CE_TAU=br.NATURAL_CE_SHA,br.NATURAL_CE_TAU
PRIOR_BINARY_SHA,PRIOR_BINARY_TAU=prior.PRIOR_BINARY_SHA,prior.PRIOR_BINARY_TAU
DEPENDENCIES=sorted(set(prior.DEPENDENCIES+['systemness_complementary_head_readout.py','systemness_complementary_head_extract.py','systemness_complementary_head_scan.py']))
ARTIFACTS=['protocol.json','search-plan.json','scaler-record.json','controls.json','zero-controls.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt']
require,read,array_sha=hp.require,hp.read,hp.array_digest
layout,validation_groups,metrics_at,max33,budget=nf.layout,nf.validation_groups,nf.metrics_at,nf.max33,nf.budget
state_hash,read_base=br.state_hash,br.read_base
probability_drift,verify_raw_evidence,predict_net=br.probability_drift,br.verify_raw_evidence,br.predict_net
LIMITATION=('One frozen head-discovery partition and one residual seed on reused development validation. The duplicated-coordinate control matches640input/20545parameters but changes conditioning and effective regularization; it is not a perfect causal isolation of information. NaturalCE/priorLongMix are diagnostic only. No primary change; fresh C4/natural confirmation remains required.')


def protocol():
    value=br.protocol();value.update(experiment='fit-selected-complementary-two-head-640-residual',records=RECORDS,
        original_head_ids=ORIGINAL,added_head_ids=ADDED,dimensions=640,parameters=20545,
        residual='640->32 GELU/dropout.1->1 zero-last, delta=4*tanh(g), no division. Base sees x[:,:512].contiguous().',
        matched='Same original578090+actualclean-natural63403 rows/labels/order, rank0/1 duplicated as trailing128 coordinates; no resampling or long replacement.',
        scaler='Freeze old512; new128 F64Welford population stats on original578090 only, chunk4096/floor1e-6; matched repeats oldrank0/1 scaler.',
        old5_before_new=True,limitation=LIMITATION)
    value['promotion']=dict(value['promotion'],strict_controls=STRICT_CONTROLS,guard_controls=GUARD_CONTROLS,priorLongMix='Two diagnostic records only, no new hardfloor.')
    return value


def bank_gate():
    require(all(v is not None for v in [EXTRA_COMPLETE_SHA,EXTRA_AUDIT_PATH,EXTRA_AUDIT_SHA]),'Complete independently audited extra128 bank is not pinned: actual forbidden.')
    hp.require_hash(extra.__file__,EXTRACTOR_SOURCE_SHA,'frozen extra2 extractor');hp.require_hash(extra.OUT/'extraction-complete.json',EXTRA_COMPLETE_SHA,'extra128 completion');hp.require_hash(EXTRA_AUDIT_PATH,EXTRA_AUDIT_SHA,'independent extra-bank audit')
    done,audit=read(extra.OUT/'extraction-complete.json'),read(EXTRA_AUDIT_PATH)
    require(audit['stage']=='actual_completed_complementary_head_feature_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False and audit['extractor_source_sha256']==EXTRACTOR_SOURCE_SHA
        and audit['extractor_complete_sha256']==EXTRA_COMPLETE_SHA and audit['extractor_plan_sha256']==done['plan_sha256']
        and audit['features_sha256']==done['features_sha256'] and audit['rows_sha256']==done['rows_sha256']
        and audit['rows']==896838 and audit['dimensions']==128 and audit['shape']==[896838,128] and audit['dtype']=='float16'
        and audit['original_head_ids']==ORIGINAL and audit['added_head_ids']==ADDED and done['rows']==896838 and done['dimensions']==128 and done['feature_shape']==[896838,128]
        and done['dtype']=='float16' and done['all_rows_written'] is True and done['all_finite'] is True and done['tests_read'] is False
        and done['added_head_ids']==ADDED and done['original_head_ids']==ORIGINAL,'Extra-bank audit/head/shape/coverage changed.')
    hp.require_hash(extra.CACHE/'rows.parquet',done['rows_sha256'],'extra-bank row map');require((extra.CACHE/'features.f16').stat().st_size==896838*128*2,'Extra feature file truncated.')
    return done,audit


def make_model(base_state,seed=20261021):
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.base=hp.make_net('mlp',hidden=128);self.correction=nn.Sequential(nn.Linear(640,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
            nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias);self.base.load_state_dict({n:torch.as_tensor(v).clone() for n,v in base_state.items()});self.base.eval().requires_grad_(False)
        def train(self,mode=True):super().train(mode);self.base.eval();return self
        def components(self,x):
            require(x.ndim==2 and x.shape[1]==640,'640 features required.')
            with torch.no_grad():base=self.base.raw(x[:,:512].contiguous())
            delta=4*torch.tanh(self.correction(x.contiguous()));raw=base.clone();raw[:,0]+=delta[:,0];return raw,base,delta
        def raw(self,x):return self.components(x)[0]
        def binary_logit(self,x):
            raw=self.raw(x);return raw[:,0]-torch.logsumexp(raw[:,1:],dim=1)
        def forward(self,x):return torch.log_softmax(self.raw(x),dim=1)
    with torch.random.fork_rng(devices=[]):torch.random.default_generator.manual_seed(seed);net=Model()
    return net


def initial_hash(base_state):return aug.init_hash(make_model(base_state))


def save_model(path,net,mean,scale,metadata):
    require(metadata['base_model_sha256']==BASE_SHA and metadata['roles']==data.ROLES and len(mean)==len(scale)==640,'Portable640 identity differs.')
    hp.atomic_npz(path,mean=np.asarray(mean,np.float32),scale=np.asarray(scale,np.float32),metadata=np.array(json.dumps(metadata)),**{'weight::'+n:v.detach().cpu().numpy() for n,v in net.state_dict().items()})


def load_model(path,device='cpu'):
    import torch
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));require(meta['kind']=='binary-residual-640' and meta['dimensions']==640 and meta['base_dimensions']==512 and meta['base_hidden']==128
            and meta['residual_hidden']==32 and meta['parameters']==20545 and meta['delta_formula']=='4*tanh(g)' and meta['roles']==data.ROLES and meta['base_model_sha256']==BASE_SHA,'Portable640 geometry/base/formula changed.')
        state={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')};base={n[5:]:v for n,v in state.items() if n.startswith('base.')};mean,scale=z['mean'].copy(),z['scale'].copy()
    require(state_hash(base)==meta['embedded_base_weight_sha256'] and mean.shape==scale.shape==(640,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Portable640 scaler/base malformed.')
    require(array_sha(mean)==meta['mean_sha256'] and array_sha(scale)==meta['scale_sha256'] and all(v.dtype==np.float32 and np.isfinite(v).all() for v in state.values()),'Portable640 array digests/nonfinite changed.')
    net=make_model(base);net.load_state_dict({n:torch.from_numpy(v) for n,v in state.items()});net.eval().to(device);return net,torch.from_numpy(mean).to(device),torch.from_numpy(scale).to(device),meta


class CombinedBank:
    def __init__(self,base,added,lookup,duplicate=False):self.base,self.added,self.lookup,self.duplicate=base,added,np.asarray(lookup,np.int64),duplicate;self.shape=(len(base),640)
    def __getitem__(self,indices):
        indices=np.asarray(indices,np.int64);require(indices.ndim==1,'Chunked row-index vector required.')
        require((self.lookup[indices]>=0).all(),'Unmapped newC4fit row requested; no new256fit allowed.')
        original=np.asarray(self.base[indices]);tail=original[:,:128] if self.duplicate else np.asarray(self.added[self.lookup[indices]])
        return np.concatenate([original,tail],axis=1)


class ExtraFitBank:
    def __init__(self,added,lookup):self.added,self.lookup=added,lookup;self.shape=(len(lookup),128)
    def __getitem__(self,indices):
        indices=np.asarray(indices,np.int64);require((self.lookup[indices]>=0).all(),'Scaler fit index outside original old bank.');return self.added[self.lookup[indices]]


def extra_mapping(rows,parent_rows,*,expected=896838):
    require(len(rows)==expected and np.array_equal(rows.cache_ix,np.arange(len(rows))) and rows.sample_ix.is_unique and set(rows.parent_bank)==set(parent_rows),'Extra map ownership/namespace changed.')
    lookups={}
    for name,parent in parent_rows.items():
        group=rows[rows.parent_bank.eq(name)];ix=group.parent_cache_ix.to_numpy(dtype=np.int64);require(group.parent_cache_ix.is_unique and ((ix>=0)&(ix<len(parent))).all(),'Duplicate/out-of-range parent index: '+name)
        source=parent.iloc[ix];require(np.array_equal(source.sample_ix,group.sample_ix) and np.array_equal(source.role,group.role) and np.array_equal(source.split,group.split),'Extra128 parent sample/role/split identity changed.')
        lookup=np.full(len(parent),-1,np.int64);lookup[ix]=group.cache_ix.to_numpy();lookups[name]=lookup
        if name=='c4':require(group.split.eq('validation').all(),'New C4fit entered extra-bank fit.')
        else:require((lookup>=0).all(),'Original/natural parent rows missing.')
    return lookups


def dataset(arm='new'):
    require(arm in ['new','matched'],'Unknown640 arm.');banks,fit,ov,cv,nrows,pv,part=nf.dataset()
    parents={'old':pd.read_parquet(hp.BANK/'rows.parquet'),'c4':pd.read_parquet(aug.NEW_BANK/'rows.parquet'),'natural':pd.read_parquet(nf.natural.CACHE/'rows.parquet')}
    mapping=extra_mapping(pd.read_parquet(extra.CACHE/'rows.parquet'),parents);added=np.memmap(extra.CACHE/'features.f16',mode='r',dtype=np.float16,shape=(896838,128))
    virtual={n:CombinedBank(bank,added,mapping[n],arm=='matched') for n,bank in banks.items()}
    return virtual,fit,ov,cv,nrows,pv,part


def upstream_snapshot():
    done,audit=bank_gate();hp.require_hash(DESIGN,DESIGN_SHA,'prospective640 design');hp.require_hash(PRIOR_AUDIT,PRIOR_AUDIT_SHA,'prior longmix independent audit')
    certificate=read(PRIOR_AUDIT);complete=read(prior.OUT/'complete.json');old=read(prior.OUT/'protocol.json')['static'];design=read(DESIGN)
    require(certificate['stage']=='actual_completed_long_mix_residual_development_audit' and certificate['audit_passed'] is True and certificate['test_data_or_results_read'] is False
        and certificate['trainer_complete_sha256']==data.digest(prior.OUT/'complete.json') and certificate['trainer_source_sha256']==data.digest(prior.__file__)
        and certificate['trainer_records_sha256']==data.digest(prior.OUT/'records.json') and certificate['trainer_selection_sha256']==data.digest(prior.OUT/'selection.json')
        and certificate['model_sha256']['new']==PRIOR_MODEL_SHA and read(prior.OUT/'selection.json')['promoted'] is False,'Prior audited negative longmix identity changed.')
    closure={}
    for path,want in certificate['provenance_sha256'].items():
        if path not in certificate['probability_bindings'] and not path.endswith('.f16'):hp.require_hash(path,want,'closed prior dependency');closure[path]=want
    for path in [PRIOR_AUDIT,DESIGN,extra.OUT/'extraction-complete.json',Path(EXTRA_AUDIT_PATH)]:closure[str(path)]=data.digest(path)
    for name,want in complete['artifact_sha256'].items():hp.require_hash(prior.OUT/name,want,'closed longmix artifact')
    base,mean,scale,_=read_base();fit,ov,cv,nrows,pv,part=layout();samples={n:array_sha(r.sample_ix.to_numpy()) for n,r in [('fit',fit),('old',ov),('c4',cv),('natural-fit',nrows),('public',pv)]}
    require(part==design['data']['original_partitions']==old['original_partitions'] and samples==design['data']['sample_ix_sha256']==old['sample_ix_sha256'],'Prospective unchanged641493 membership/order differs.')
    candidate=read(prior.OUT/'candidates/new.json');refs=dict(old['reference_probabilities']);refs['priorLongMix']={b:dict(path=str(prior.ppath('new',b)),sha256=candidate['probability_sha256'][b],sample_ix_sha256=samples[b]) for b in ['old','c4','public']}
    require(all(certificate['probability_bindings'][v['path']]==v['sha256'] for m in refs.values() for v in m.values()),'Inherited reference probabilities not prior-audited.')
    features=dict(old['feature_files_sha256']);features[str(extra.CACHE/'features.f16')]=done['features_sha256']
    return dict(source_sha256=data.digest(__file__),design_sha256=DESIGN_SHA,dependencies_sha256={n:data.digest(data.HERE/n) for n in DEPENDENCIES},closed_files_sha256=closure,
        extra_complete_sha256=EXTRA_COMPLETE_SHA,extra_audit_sha256=EXTRA_AUDIT_SHA,extra_rows_sha256=done['rows_sha256'],feature_files_sha256=features,
        base_model_path=str(BASE_PATH),base_model_sha256=BASE_SHA,embedded_base_weight_sha256=state_hash(base),old512_mean_sha256=array_sha(mean),old512_scale_sha256=array_sha(scale),
        initial_weight_sha256=initial_hash(base),standardization_sha256=old['standardization_sha256'],selected_head_ids=ORIGINAL+ADDED,
        original_partitions=part,sample_ix_sha256=samples,reference_probabilities=refs,prior_model_sha256=old['prior_model_sha256'],old_reference=old['old_reference'],
        prior_actual_audit_sha256=PRIOR_AUDIT_SHA,packages={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},validation_probability_arrays_read=False,global_inventory_scanned=False)


def declare():
    value=dict(protocol=protocol(),static=upstream_snapshot());OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():require(read(OUT/'protocol.json')==value,'Declared640 protocol differs.')
    else:require(not CACHE.exists(),'640 cache predates declaration.');data.atomic_json(OUT/'protocol.json',value)
    return value


def verify_declaration():
    require((OUT/'protocol.json').is_file(),'Root must declare640 protocol after extra-bank independent audit.')
    value=read(OUT/'protocol.json');require(value==dict(protocol=protocol(),static=upstream_snapshot()),'Frozen640 source/design/closure changed.');return value


def prepare_scalers():
    declaration=verify_declaration();base,mean,scale,_=read_base();fit,*_=layout();rows=pd.read_parquet(extra.CACHE/'rows.parquet');oldrows=pd.read_parquet(hp.BANK/'rows.parquet');group=rows[rows.parent_bank.eq('old')]
    lookup=np.full(len(oldrows),-1,np.int64);lookup[group.parent_cache_ix.to_numpy(np.int64)]=group.cache_ix.to_numpy(np.int64);indices=fit.cache_ix.to_numpy()
    require((lookup[indices]>=0).all() and len(indices)==578090 and fit.split.eq('fit').all(),'New scaler fit scope changed.')
    descriptor=dict(fit_rows=578090,fit_only=True,natural_rows_used=0,validation_rows_used=0,chunk=4096,floor=1e-6,variance='Population F64 parallel-Welford',fit_sample_ix_sha256=array_sha(fit.sample_ix.to_numpy()),extra_fit_cache_ix_sha256=array_sha(lookup[indices]))
    CACHE.mkdir(exist_ok=True);path=CACHE/'standardization.npz';record=OUT/'scaler-record.json'
    if path.exists():
        require(record.exists() and read(record)['standardization_sha256']==data.digest(path) and read(record)['fit']==descriptor,'Unowned/changed640 scaler.');return
    added=np.memmap(extra.CACHE/'features.f16',mode='r',dtype=np.float16,shape=(896838,128));em,es=hp.fit_scaler(ExtraFitBank(added,lookup),indices,chunk=4096)
    hp.atomic_npz(path,new_mean=np.r_[mean,em].astype(np.float32),new_scale=np.r_[scale,es].astype(np.float32),matched_mean=np.r_[mean,mean[:128]].astype(np.float32),matched_scale=np.r_[scale,scale[:128]].astype(np.float32))
    data.atomic_json(record,dict(protocol_sha256=data.digest(OUT/'protocol.json'),standardization_sha256=data.digest(path),fit=descriptor,old512_mean_sha256=array_sha(mean),old512_scale_sha256=array_sha(scale)))


def read_scaler(arm):
    require(arm in ARMS+['zero'],'Unknown640 scaler arm.');name='new' if arm=='zero' else arm
    with np.load(CACHE/'standardization.npz',allow_pickle=False) as z:mean,scale=z[name+'_mean'].copy(),z[name+'_scale'].copy()
    require(mean.shape==scale.shape==(640,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Invalid640 scaler arrays.')
    _,oldmean,oldscale,_=read_base();require(np.array_equal(mean[:512].view(np.uint32),oldmean.view(np.uint32)) and np.array_equal(scale[:512].view(np.uint32),oldscale.view(np.uint32)),'Original512 scaler modified.')
    if arm=='matched':require(np.array_equal(mean[512:],oldmean[:128]) and np.array_equal(scale[512:],oldscale[:128]),'Duplicate scaler is not exact oldrank0/1.')
    return mean,scale


def source_signature():
    value=verify_declaration();static=value['static']
    for path,want in static['feature_files_sha256'].items():hp.require_hash(path,want,'certified640 feature source')
    scaler=read(OUT/'scaler-record.json');hp.require_hash(CACHE/'standardization.npz',scaler['standardization_sha256'],'640 scalers')
    return dict(source_sha256=data.digest(__file__),protocol_sha256=data.digest(OUT/'protocol.json'),static_sha256=aug.json_digest(static),
        selected_head_ids=ORIGINAL+ADDED,standardization_sha256=scaler['standardization_sha256'],base_model_sha256=BASE_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],
        feature_files_sha256=static['feature_files_sha256'],extra_rows_sha256=static['extra_rows_sha256'])


def verify_plan(plan,full=True):
    value=verify_declaration();require(plan['protocol']==protocol() and plan['initial_weight_sha256']==value['static']['initial_weight_sha256'] and plan['signature']['protocol_sha256']==data.digest(OUT/'protocol.json'),'640 plan/init changed.')
    if full:require(plan['signature']==source_signature(),'640 source/feature/scaler signature changed.')
    return value


def portable_metadata(plan,arm,**additional):
    static=read(OUT/'protocol.json')['static'];mean,scale=read_scaler(arm)
    return dict(kind='binary-residual-640',arm=arm,dimensions=640,base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',roles=data.ROLES,
        base_model_sha256=BASE_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),
        selected_head_ids=ORIGINAL+(ORIGINAL[:2] if arm=='matched' else ADDED),signature=plan['signature'],source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),
        initial_weight_sha256=plan['initial_weight_sha256'],standardization_sha256=plan['signature']['standardization_sha256'],tests_read=False,**additional)


def verify_portable(path,plan,arm):
    net,mean,scale,meta=load_model(path);base,m,s,_=read_base();require(state_hash({n:v.detach().numpy() for n,v in net.base.state_dict().items()})==state_hash(base),'Embedded640 base differs from original.')
    em,es=read_scaler(arm);require(np.array_equal(mean.numpy().view(np.uint32),em.view(np.uint32)) and np.array_equal(scale.numpy().view(np.uint32),es.view(np.uint32)),'Portable640 arm scaler differs.')
    require(all(meta[k]==v for k,v in portable_metadata(plan,arm).items()),'Portable640 source/head/scaler provenance changed.');return net,mean,scale,meta


def predict_raw(path,bank,rows,device='cpu'):
    net,mean,scale,_=load_model(path,device);return predict_net(net,mean,scale,bank,rows,device)


def parity(path,banks,nrows):
    require(nrows.split.eq('fit').all() and len(nrows)>=256,'Only clean FIT parity.');chosen=nrows.iloc[:256]
    cpu,_=predict_raw(path,banks['natural'],chosen);gpu,_=predict_raw(path,banks['natural'],chosen,'cuda:0');difference=float(np.abs(cpu-gpu).max());require(difference<5e-4,'640 FIT portable parity failed.')
    return dict(fit_only=True,validation_probabilities_computed=False,sampled_tokens=256,target_cache_ix_sha256=array_sha(chosen.cache_ix.to_numpy()),sample_ix_sha256=array_sha(chosen.sample_ix.to_numpy()),maximum_probability_difference=difference)


def ppath(model,bank):return CACHE/f'{model}-{bank}-validation.npz'
def read_predictions(model,record,rows):return {b:aug.read_probability(ppath(model,b),record['probability_sha256'][b],r) for b,r in rows.items()}
def supervisor_alive():require(os.environ.get('ROLE_COMPLEMENTARY_READOUT_SUPERVISOR_PID')==str(os.getppid()),'Worker requires original supervisor lease.')


def worker(arm,gpu):
    supervisor_alive();require(arm in ARMS and gpu in [0,1],'Only declared two arm workers allowed.')
    require(not (OUT/'complete.json').exists() and not (OUT/'candidates'/f'{arm}.json').exists(),'Frozen candidate cannot be overwritten.')
    no_competing_jobs(True);plan=read(OUT/'search-plan.json');verify_plan(plan);ms.configure_torch()
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1 and 'A40' in torch.cuda.get_device_name(0),'Worker requires its assigned A40.')
    banks,fit,ov,cv,nrows,pv,_=dataset(arm);base,_,_,_=read_base();mean,scale=read_scaler(arm)
    def standardized(bank,indices):
        out=torch.empty((len(indices),640),dtype=torch.float32,device='cuda')
        for start in range(0,len(indices),BATCH):
            x=(np.asarray(banks[bank][indices[start:start+BATCH]],np.float32)-mean)/scale
            require(np.isfinite(x).all(),'Nonfinite standardized feature.');out[start:start+len(x)]=torch.from_numpy(x).cuda()
        return out
    require(torch.cuda.mem_get_info(0)[0]>(INSTANCES+173032)*640*4+2*2**30,'Insufficient assigned worker memory.')
    old_x=standardized('old',fit.cache_ix.to_numpy())
    extra_x=standardized('natural',nrows.cache_ix.to_numpy())
    x=torch.cat([old_x,extra_x]);del old_x,extra_x
    y=torch.from_numpy(np.r_[probes.labels(fit),probes.labels(nrows)]==0).float().cuda();xv=standardized('old',ov.cache_ix.to_numpy())
    torch.manual_seed(20261021);net=make_model(base).cuda();init=aug.init_hash(net);require(init==plan['initial_weight_sha256'],'Arms full initial state differs.')
    parameters=list(net.correction.parameters());require({id(p) for p in net.parameters() if p.requires_grad}=={id(p) for p in parameters},'Optimizer includes frozen base parameters.')
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.001);generator=torch.Generator(device='cuda:0').manual_seed(20261021)
    path=OUT/'models'/f'{arm}.npz';gpu_path=CACHE/f'{arm}-gpu-old-validation.npz';history=[];best=-1.;best_epoch=-1;base_sha=state_hash(base)
    for epoch in range(1,46):
        supervisor_alive();net.train();require(not net.base.training and not any(p.requires_grad for p in net.base.parameters()),'Frozen base became trainable.')
        order=lp.epoch_order(len(x),generator,'cuda');loss_sum=0.;steps=0;require(len(order)==INSTANCES,'Epoch budget changed.')
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH];loss=torch.nn.functional.binary_cross_entropy_with_logits(net.binary_logit(x[ix].contiguous()),y[ix]);require(torch.isfinite(loss),'Nonfinite binary loss.')
            optimizer.zero_grad(set_to_none=True);loss.backward();require(all(p.grad is None for p in net.base.parameters()),'Frozen base received gradients.')
            torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();loss_sum+=float(loss.detach())*len(ix);steps+=1
        require(steps==STEPS and state_hash({n:v.detach().cpu().numpy() for n,v in net.base.state_dict().items()})==base_sha,'Updates/base frozen bytes changed.')
        p=probes.predict_standardized(net,xv).astype(np.float32);value=lp.validation_metrics(ov,p);score=value['mean_system_f1']
        history.append(dict(epoch=epoch,loss=loss_sum/len(order),instances=len(order),steps=steps,cumulative_steps=epoch*STEPS,old5_mean_f1=score))
        if score>best+1e-5:
            best,best_epoch=score,epoch;save_model(path,net,mean,scale,portable_metadata(plan,arm,best_epoch=epoch,best_steps=epoch*STEPS));hp.atomic_npz(gpu_path,p=p,sample_ix=ov.sample_ix.to_numpy())
        print(json.dumps(dict(arm=arm,**history[-1],best_epoch=best_epoch)),flush=True)
        if epoch>=10 and epoch-best_epoch>=8:break
    amounts=budget(history,best_epoch);supervisor_alive()
    training=dict(arm=arm,signature=plan['signature'],plan_sha256=data.digest(OUT/'search-plan.json'),initial_weight_sha256=init,history=history,**amounts,
        embedded_base_weight_sha256=base_sha,base_eval_all_epochs=True,base_gradient_observations=0,optimizer_only_residual=True,
        model_sha256=data.digest(path),gpu_old_probability_sha256=data.digest(gpu_path),new_C4_predictions_read=False,public_validation_predictions_read=False)
    data.atomic_json(OUT/'training'/f'{arm}.json',training)
    # Immutable best/model/history/GPUold proof precedes all new-cohort predictions.
    rows=dict(old=ov,c4=cv,public=pv);p={};traces={}
    for b,frame in rows.items():
        p[b],traces[b]=predict_raw(path,banks['natural' if b=='public' else b],frame);hp.atomic_npz(ppath(arm,b),p=p[b],sample_ix=frame.sample_ix.to_numpy())
    gpu_p=aug.read_probability(gpu_path,training['gpu_old_probability_sha256'],ov);drift=probability_drift(gpu_p,p['old'],nf.ORIGINAL_TAU);require(drift['maximum_probability_difference']<5e-4,'Candidate full GPUold/CPUold drift failed.')
    proof=parity(path,banks,nrows);nf.verify_parity(proof,nrows);verify_plan(plan);supervisor_alive()
    data.atomic_json(OUT/'candidates'/f'{arm}.json',dict(arm=arm,signature=plan['signature'],source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),
        initial_weight_sha256=init,model_sha256=training['model_sha256'],training_sha256=data.digest(OUT/'training'/f'{arm}.json'),probability_sha256={b:data.digest(ppath(arm,b)) for b in rows},
        gpu_old_probability_sha256=training['gpu_old_probability_sha256'],old_early_stop_validation=lp.validation_metrics(ov,gpu_p),full_old_GPU_CPU_drift=drift,
        raw_invariance=traces,portable_evidence=proof,**amounts,tests_read=False,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_name=torch.cuda.get_device_name(0)))


def candidate_record(arm,plan,rows,nrows):
    r=read(OUT/'candidates'/f'{arm}.json');ep=OUT/'training'/f'{arm}.json';hp.require_hash(ep,r['training_sha256'],'pre-new-validation checkpoint freeze');e=read(ep);amounts=budget(e['history'],e['best_epoch'])
    require(r['arm']==e['arm']==arm and r['signature']==e['signature']==plan['signature'] and r['source_sha256']==data.digest(__file__)
        and r['plan_sha256']==e['plan_sha256']==data.digest(OUT/'search-plan.json') and r['initial_weight_sha256']==e['initial_weight_sha256']==plan['initial_weight_sha256']
        and r['model_sha256']==e['model_sha256'] and r['tests_read'] is False and e['new_C4_predictions_read'] is False and e['public_validation_predictions_read'] is False
        and e['embedded_base_weight_sha256']==plan['signature']['embedded_base_weight_sha256'] and e['base_eval_all_epochs'] is True and e['base_gradient_observations']==0 and e['optimizer_only_residual'] is True
        and all(r[k]==e[k]==v for k,v in amounts.items()),'Candidate history/source/frozen-base/early-stop changed.')
    path=OUT/'models'/f'{arm}.npz';hp.require_hash(path,r['model_sha256'],'candidate portable');_,_,_,meta=verify_portable(path,plan,arm)
    require(meta['best_epoch']==e['best_epoch'] and meta['best_steps']==e['best_steps'],'Best portable/history differs.')
    p=read_predictions(arm,r,rows);gpu_p=aug.read_probability(CACHE/f'{arm}-gpu-old-validation.npz',r['gpu_old_probability_sha256'],rows['old']);value=lp.validation_metrics(rows['old'],gpu_p)
    require(r['gpu_old_probability_sha256']==e['gpu_old_probability_sha256'] and value==r['old_early_stop_validation']
        and value['mean_system_f1']==e['history'][e['best_epoch']-1]['old5_mean_f1'] and probability_drift(gpu_p,p['old'],nf.ORIGINAL_TAU)==r['full_old_GPU_CPU_drift']
        and 0<=r['full_old_GPU_CPU_drift']['maximum_probability_difference']<5e-4,'Saved best GPUold/CPUold drift changed.')
    require(set(r['raw_invariance'])==set(rows),'All three full cohort raw proofs required.')
    for b in rows:verify_raw_evidence(r['raw_invariance'][b],len(rows[b]))
    nf.verify_parity(r['portable_evidence'],nrows);require(r['cuda_visible_devices'] in ['0','1'] and 'A40' in r['cuda_device_name'],'Assigned GPU evidence changed.');return r,p


def both_frozen(plan):
    hashes={}
    for arm in ARMS:
        path=OUT/'training'/f'{arm}.json';value=read(path)
        require(value['arm']==arm and value['signature']==plan['signature'] and value['plan_sha256']==data.digest(OUT/'search-plan.json')
            and value['new_C4_predictions_read'] is False and value['public_validation_predictions_read'] is False,'Both arm freezes required before zero predictions.')
        hp.require_hash(OUT/'models'/f'{arm}.npz',value['model_sha256'],'frozen arm model');hp.require_hash(CACHE/f'{arm}-gpu-old-validation.npz',value['gpu_old_probability_sha256'],'frozen old GPU probabilities');hashes[arm]=data.digest(path)
    return hashes


def zero_controls(plan,banks,rows,nrows,old,allow_create=False):
    frozen=both_frozen(plan);path=OUT/'zero-controls.json';model=OUT/'models/zero.npz';exists=path.exists();value=read(path) if exists else None
    if exists:
        require(value['signature']==plan['signature'] and value['both_arm_training_sha256']==frozen and value['original_threshold']==nf.ORIGINAL_TAU,'Zero control signature/freeze changed.')
        hp.require_hash(model,value['model_sha256'],'same-path zero portable');net,_,_,_=verify_portable(model,plan,'zero')
        import torch
        require(torch.count_nonzero(net.correction[-1].weight)==0 and torch.count_nonzero(net.correction[-1].bias)==0,'Zero residual is nonzero.')
        p=read_predictions('zero',value,rows);traces=value['raw_invariance'];proof=value['portable_evidence']
    else:
        require(allow_create and not model.exists() and not any(ppath('zero',b).exists() for b in rows),'Missing/unowned partial zero controls: no silent recomputation.')
        base,_,_,_=read_base();mean,scale=read_scaler('zero');net=make_model(base);require(aug.init_hash(net)==plan['initial_weight_sha256'],'Zero full initialization changed.')
        save_model(model,net,mean,scale,portable_metadata(plan,'zero',zero_residual=True));p={};traces={}
        for b,frame in rows.items():
            p[b],traces[b]=predict_raw(model,banks['natural' if b=='public' else b],frame);hp.atomic_npz(ppath('zero',b),p=p[b],sample_ix=frame.sample_ix.to_numpy())
        proof=parity(model,banks,nrows)
    for b in rows:
        verify_raw_evidence(traces[b],len(rows[b]));require(traces[b]['delta_min']==traces[b]['delta_max']==traces[b]['delta_mean']==0,'Zero-path delta is not zero.')
    nf.verify_parity(proof,nrows);drift={b:probability_drift(old[b],p[b],nf.ORIGINAL_TAU) for b in rows}
    require(all(0<=v['maximum_probability_difference']<5e-4 for v in drift.values()),'Zero same-path versus audited CPU base drift failed.')
    tau,grid=max33(rows,p);records={'zero-oldmean':metrics_at(rows,p,nf.ORIGINAL_TAU),'zero-max33':metrics_at(rows,p,tau)}
    require(records['zero-max33']['min33']==grid['min33'],'Zero max33 direct proof changed.')
    content=dict(signature=plan['signature'],both_arm_training_sha256=frozen,model_path=str(model),model_sha256=data.digest(model),
        probability_sha256={b:data.digest(ppath('zero',b)) for b in rows},original_threshold=nf.ORIGINAL_TAU,records=records,max33_full_grid_proof=grid,
        oldhead_commonCPU_drift=drift,raw_invariance=traces,portable_evidence=proof,trained_record=False,tests_read=False)
    if exists:require(value==content,'Zero model/probability/drift/calibration evidence changed.')
    else:data.atomic_json(path,content)
    return records


def flat_metrics(records):
    flat=[]
    for record in records:
        for group,value in dict(record['validation']['groups'],pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=record['record'],group=group,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))},**{'role_recall_'+r:v for r,v in value['recall'].items()}))
    return flat


def no_competing_jobs(allow_own=False):
    nf.no_competing_jobs(True);names={'systemness_natural_fit_probes.py','systemness_binary_residual.py','systemness_long_mix_residual.py','systemness_complementary_head_scan.py','systemness_complementary_head_extract.py'}
    if not allow_own:names.add(Path(__file__).name)
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name in names for a in args if a):active.append(int(entry.name))
    require(not active,'Competing readout/capture process live: '+str(active))


@contextmanager
def run_lock():
    import fcntl
    with (OUT/'.run.lock').open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another complementary640 supervisor owns output.') from e
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run_workers(pending):
    require(len(pending)<=2 and len(set(pending))==len(pending) and set(pending)<=set(ARMS),'Only two fixed arm workers allowed.');jobs=[]
    try:
        for gpu,arm in enumerate(pending):
            log=(OUT/'logs'/f'{arm}.log').open('a')
            try:p=subprocess.Popen([sys.executable,'-u',__file__,'--train','--worker',arm,'--gpu',str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ROLE_COMPLEMENTARY_READOUT_SUPERVISOR_PID=str(os.getpid())),stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            jobs.append((arm,p,log,gpu))
        data.atomic_json(OUT/'status.json',dict(stage='training',workers=[dict(arm=a,pid=p.pid,gpu=g) for a,p,_,g in jobs]))
        while jobs:
            codes=[(a,p.poll()) for a,p,_,_ in jobs];require(not [(a,c) for a,c in codes if c not in [None,0]],'Complementary640 worker failed: '+str(codes))
            if all(c==0 for _,c in codes):break
            time.sleep(5)
        for arm,p,_,_ in jobs:require(p.wait()==0,arm+' failed.')
    finally:
        errors=[]
        for _,p,_,_ in jobs:
            try:
                if p.poll() is None:p.terminate()
            except BaseException as e:errors.append(repr(e))
        for _,p,log,_ in jobs:
            try:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            except BaseException as e:errors.append(repr(e))
            finally:log.close()
        require(not errors,'Worker terminate/reap failed: '+repr(errors))


def train(restart_incomplete=False):
    bank_gate();ms.configure_torch();no_competing_jobs();verify_declaration();prepare_scalers();signature=source_signature()
    with run_lock():
        if (OUT/'complete.json').exists():verified_completion();return
        require(not (OUT/'selection.json').exists(),'Partial frozen selection exists; refusing replacement.')
        for name in ['models','training','candidates','logs']:(OUT/name).mkdir(exist_ok=True)
        path=OUT/'search-plan.json'
        if path.exists():plan=read(path);verify_plan(plan)
        else:
            require({p.name for p in CACHE.iterdir()}=={'standardization.npz'} and all(not any((OUT/n).iterdir()) for n in ['models','training','candidates']),'Unowned models/probabilities predate plan.')
            plan=dict(protocol=protocol(),signature=signature,initial_weight_sha256=read(OUT/'protocol.json')['static']['initial_weight_sha256']);data.atomic_json(path,plan)
        _,_,ov,cv,nrows,pv,_=dataset();rows=dict(old=ov,c4=cv,public=pv);pending=[]
        for arm in ARMS:
            if (OUT/'candidates'/f'{arm}.json').exists():candidate_record(arm,plan,rows,nrows)
            else:
                partial=[OUT/'models'/f'{arm}.npz',OUT/'training'/f'{arm}.json',CACHE/f'{arm}-gpu-old-validation.npz']+[ppath(arm,b) for b in rows]
                require(restart_incomplete or not any(p.exists() for p in partial),'Partial candidate requires explicit --restart-incomplete; no old progress trusted.')
                if any(p.exists() for p in partial):data.atomic_json(OUT/f'restart-{arm}-{time.time_ns()}.json',dict(arm=arm,from_epoch=1,signature=signature,partial_sha256={str(p):data.digest(p) for p in partial if p.exists()}))
                pending.append(arm)
        if pending:run_workers(pending)
        publish(plan)


def select_records(records,legacy,zero):
    require(len(records)==14 and {r['record'] for r in records}==set(RECORDS) and set(zero)=={'zero-oldmean','zero-max33'},'Fourteen records and two separate zero controls required.')
    index={r['record']:r for r in records};candidate=index['new-max33']['validation'];minimum=candidate['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARD_CONTROLS:
        control=zero[name] if name in zero else index[name]['validation'];g=nf.guard(candidate,control);guards[name]=g;reasons += [name+':'+r for r in g['rejection_reasons']]
        if name in STRICT_CONTROLS:
            gain=minimum-control['min33'] if minimum is not None and control['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    gpu=nf.guard(candidate,legacy,True);guards['original-savedGPU']=gpu;reasons += ['original-savedGPU:'+r for r in gpu['rejection_reasons']]
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,
        naturalCE_is_diagnostic_only=True,priorLongMix_is_diagnostic_only=True,priorBinary_max33_is_strict_and_guard=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def calibrate(plan,allow_create=False):
    ms.configure_torch();banks,fit,ov,cv,nrows,pv,_=dataset();rows=dict(old=ov,c4=cv,public=pv);p={}
    for arm in ARMS:_,p[arm]=candidate_record(arm,plan,rows,nrows)
    inherited,legacy=inherited_controls(plan,rows,allow_create);p.update(inherited)
    zero=zero_controls(plan,banks,rows,nrows,p['oldhead'],allow_create);records=[]
    for name in RECORDS:
        model,objective=name.split('-');value=p[model]
        if objective=='max33':tau,proof=max33(rows,value)
        elif model=='oldhead':tau,proof=nf.ORIGINAL_TAU,None
        elif model=='ensemble':tau,proof=nf.ENSEMBLE_TAU,None
        elif model=='naturalCE':tau,proof=NATURAL_CE_TAU,None
        elif model=='priorBinary':tau,proof=PRIOR_BINARY_TAU,None
        elif model=='priorLongMix':tau,proof=PRIOR_TAU,None
        else:tau,proof=lp.validation_metrics(ov,value['old'])['system_threshold'],None
        result=metrics_at(rows,value,tau)
        if proof is not None:require(result['min33']==proof['min33'] and result['old5_mean_f1']==proof['old5_mean_f1'],'Max33 selected metrics differ.')
        if name=='naturalCE-max33':require(tau==.6008080808080808,'Inherited CE threshold changed.')
        if name=='priorBinary-max33':require(tau==.5453636363636364 and result['min33']==.8990885416666666,'Frozen prior binary calibration no longer reproduces.')
        if name=='priorLongMix-max33':require(tau==PRIOR_MAX33_TAU,'Frozen prior longmix calibration changed.')
        records.append(dict(record=name,model=model,objective=objective,validation=result,full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False))
    return records,legacy,zero,rows,fit,nrows


def publish(plan):
    records,legacy,zero,rows,fit,nrows=calibrate(plan,True);selection=select_records(records,legacy,zero);data.atomic_json(OUT/'records.json',records)
    flat=flat_metrics(records);zf=flat_metrics([dict(record=k,validation=v) for k,v in zero.items()]);require(len(flat)==168 and len(zf)==24,'154+14 main/22+2 zero CSV coverage required.')
    pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False);pd.DataFrame(zf).to_csv(OUT/'zero-control-metrics.csv',index=False)
    counts=read(br.OUT/'source-counts.json')|dict(experiment='complementary-two-head-640-residual',same_existing_rows=True,same_validation_rows=True,dimensions=640,trainable_parameters=20545,
        informative_heads=dict(new=10,matched=8),extra128_scaler_fit_rows=FIT,extra128_scaler_natural_rows=0,extra128_scaler_validation_rows=0,original_and_natural_fit_unchanged=True,
        new_arm_bank_rows=dict(old=FIT,c4=0,natural=EXTRA),matched_arm_bank_rows=dict(old=FIT,c4=0,natural=EXTRA),priorLongMix_diagnostic_only=True,limitation=LIMITATION)
    data.atomic_json(OUT/'source-counts.json',counts)
    lines=['Complementary two-head640 residual: reused development only',LIMITATION,'','Record threshold min33 old5meanF1']
    lines += [f"{r['record']} {r['validation']['system_threshold']:.12g} {r['validation']['min33']} {r['validation']['old5_mean_f1']:.9f}" for r in records]
    lines += ['','Same-path zero numerical controls (not14 records)']+[f"{k} {v['system_threshold']:.12g} {v['min33']} {v['old5_mean_f1']:.9f}" for k,v in zero.items()]
    lines += ['','Promoted: '+str(selection['promoted']),'Reasons: '+json.dumps(selection['rejection_reasons'])];(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    selection.update(source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),records_sha256=data.digest(OUT/'records.json'),model_path=str(OUT/'models/new.npz'),
        model_sha256=read(OUT/'candidates/new.json')['model_sha256'],controls_sha256=data.digest(OUT/'controls.json'),zero_controls_sha256=data.digest(OUT/'zero-controls.json'),
        scaler_record_sha256=data.digest(OUT/'scaler-record.json'),system_threshold=next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'))
    data.atomic_json(OUT/'selection.json',selection);verify_plan(plan)
    require(all(p.stat().st_size<50*2**20 for folder in [OUT,CACHE] for p in folder.rglob('*') if p.is_file()),'Artifact exceeds50MiB storage bound.')
    data.atomic_json(OUT/'complete.json',dict(signature=plan['signature'],source_sha256=data.digest(__file__),artifact_sha256={n:data.digest(OUT/n) for n in ARTIFACTS},
        candidate_sha256={a:data.digest(OUT/'candidates'/f'{a}.json') for a in ARMS},tests_read=False,primary_changed=False,new_independent_confirmation_required=True))
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',promoted=selection['promoted'],primary_changed=False))


def verified_completion():
    require((OUT/'complete.json').is_file() and read(OUT/'status.json')['stage']=='validation_complete','Complementary640 residual trial not terminal complete.')
    plan=read(OUT/'search-plan.json');verify_plan(plan);complete=read(OUT/'complete.json')
    require(complete['signature']==plan['signature'] and complete['source_sha256']==data.digest(__file__) and complete['tests_read'] is False
        and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and set(complete['artifact_sha256'])==set(ARTIFACTS) and set(complete['candidate_sha256'])==set(ARMS),'Completion signature/source/coverage changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(OUT/n,h,'completed artifact')
    for a,h in complete['candidate_sha256'].items():hp.require_hash(OUT/'candidates'/f'{a}.json',h,'completed candidate')
    records,legacy,zero,*_=calibrate(plan,False);require(records==read(OUT/'records.json'),'Fourteen saved records do not reproduce from frozen p.')
    expected=select_records(records,legacy,zero);actual=read(OUT/'selection.json');require(all(actual[k]==v for k,v in expected.items()),'Five strict/eight guard/savedGPU/absolute90 gate changed.')
    require(actual['source_sha256']==data.digest(__file__) and actual['plan_sha256']==data.digest(OUT/'search-plan.json') and actual['records_sha256']==data.digest(OUT/'records.json')
        and actual['scaler_record_sha256']==data.digest(OUT/'scaler-record.json') and actual['controls_sha256']==data.digest(OUT/'controls.json') and actual['zero_controls_sha256']==data.digest(OUT/'zero-controls.json')
        and actual['model_path']==str(OUT/'models/new.npz') and actual['model_sha256']==read(OUT/'candidates/new.json')['model_sha256']
        and actual['system_threshold']==next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'),'Selected model/threshold/control closure changed.')
    return complete


def inherited_controls(plan,rows,allow_create=False):
    static=verify_plan(plan,False)['static'];p={}
    for model,items in static['reference_probabilities'].items():p[model]={b:aug.read_probability(v['path'],v['sha256'],rows[b]) for b,v in items.items()}
    previous=read(prior.OUT/'controls.json');legacy=previous['original_savedGPU_validation'];ref=static['old_reference']
    gpu={b:aug.read_probability(ref[k+'_path'],ref[k+'_sha256'],rows[b]) for b,k in [('old','old'),('c4','new')]}
    require(metrics_at(rows,dict(gpu,public=p['oldhead']['public']),nf.ORIGINAL_TAU)==legacy,'Original savedGPU guards changed.')
    content=dict(signature=plan['signature'],prior_actual_audit_sha256=PRIOR_AUDIT_SHA,extra_complete_sha256=EXTRA_COMPLETE_SHA,extra_audit_sha256=EXTRA_AUDIT_SHA,
        reference_probabilities=static['reference_probabilities'],physical_model_sha256=dict(oldhead=BASE_SHA,ensemble=static['prior_model_sha256'],naturalCE=NATURAL_CE_SHA,priorBinary=PRIOR_BINARY_SHA,priorLongMix=PRIOR_MODEL_SHA),
        original_savedGPU_validation=legacy,reference_probabilities_recomputed=False,naturalCE_diagnostic_only=True,priorLongMix_diagnostic_only=True,priorBinary_max33_is_strict_and_guard=True,tests_read=False)
    path=OUT/'controls.json'
    if path.exists():require(read(path)==content,'Inherited640 explicit control closure changed.')
    else:require(allow_create,'Missing inherited640 controls in readonly path.');data.atomic_json(path,content)
    return p,legacy


def preflight():
    import copy,torch
    from unittest.mock import patch
    module=sys.modules[__name__];folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-readout-preflight-',dir='/tmp'));source=data.digest(__file__);checks=[]
    original_globals=(br.OUT,br.CACHE,prior.OUT,prior.CACHE,extra.OUT,extra.CACHE)
    def check(name,value=True):require(bool(value),'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,AssertionError,KeyError,FileNotFoundError):checks.append(name);return
        raise RuntimeError('Expected rejection: '+name)
    def forbidden(*a,**k):raise AssertionError('Actual extra-bank/source/p/model artifact access in synthetic preflight.')
    torch.set_num_threads(2);ms.configure_torch()
    with patch.object(module,'EXTRA_COMPLETE_SHA',None),patch.object(module,'EXTRA_AUDIT_PATH',None),patch.object(module,'EXTRA_AUDIT_SHA',None),patch.object(module,'read',forbidden),patch.object(data,'digest',forbidden):
        reject('unknown_independent_bank_audit_gate_before_IO',bank_gate)
    rng=np.random.RandomState(15);base={'encoder.0.weight':rng.normal(0,.01,(128,512)).astype(np.float32),'encoder.0.bias':rng.normal(0,.01,128).astype(np.float32),'head.weight':rng.normal(0,.01,(5,128)).astype(np.float32),'head.bias':np.array([0.,.1,.2,.3,.4],np.float32)}
    net=make_model(base);other=make_model(base);check('exact20545_trainable_correction_parameters',sum(p.numel() for p in net.parameters() if p.requires_grad)==20545)
    check('same_neural_initial_weight_state_across_arms',aug.init_hash(net)==aug.init_hash(other)==initial_hash(base))
    check('zero_last_layer_and_frozen_base_initialization',torch.count_nonzero(net.correction[-1].weight)==torch.count_nonzero(net.correction[-1].bias)==0 and not any(p.requires_grad for p in net.base.parameters()))
    net.train();check('train_parent_keeps_base_eval_dropout_disabled',not net.base.training and net.correction.training)
    x=torch.from_numpy(rng.normal(size=(12,1280)).astype(np.float32))[:,::2];require(not x.is_contiguous(),'Fixture must use noncontiguous640.')
    observed=[];handle=net.base.encoder[0].register_forward_pre_hook(lambda mod,args:observed.append((tuple(args[0].shape),args[0].is_contiguous())))
    raw,raw_base,delta=net.components(x);handle.remove();check('base_sees_contiguous_first512_only',observed==[((12,512),True)])
    check('zero640_residual_reproduces_original_base_raw',torch.equal(raw,raw_base) and bool((delta==0).all()))
    target=torch.tensor([0,1]*6,dtype=torch.float32);loss=torch.nn.functional.binary_cross_entropy_with_logits(net.binary_logit(x),target);loss.backward()
    check('BCE_backprop_only_correction_no_base_gradients',all(p.grad is None for p in net.base.parameters()) and net.correction[-1].weight.grad.abs().sum()>0)
    before=state_hash({n:v.detach().numpy().copy() for n,v in net.base.state_dict().items()});opt=torch.optim.AdamW(net.correction.parameters(),lr=.001,weight_decay=.001);torch.nn.utils.clip_grad_norm_(net.correction.parameters(),5);opt.step()
    check('optimizer_step_base_raw_weight_bits_unchanged',state_hash({n:v.detach().numpy() for n,v in net.base.state_dict().items()})==before)
    net.eval();raw,raw_base,delta=net.components(x);check('all_four_non_system_raw_logits_bitexact',np.array_equal(raw[:,1:].detach().numpy().view(np.uint32),raw_base[:,1:].detach().numpy().view(np.uint32)) and float(delta.detach().abs().max())<=4)
    # Same architecture/RNG consumes exactly the same dropout stream; means and
    # tails may differ in real arms, but neural initialization/noise is fixed.
    a,b=make_model(base),make_model(base)
    for v in [a,b]:v.train();torch.nn.init.constant_(v.correction[-1].weight,.1)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(19);va=a.correction(x.contiguous());ra=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):torch.manual_seed(19);vb=b.correction(x.contiguous());rb=torch.random.get_rng_state().clone()
    check('same_seed_same_CE_order_dropout_RNG_stream',torch.equal(va,vb) and torch.equal(ra,rb))
    # Parent C4 cache indices are deliberately nonzero; wrong zero-based lookup
    # would silently return another document's128 features without these guards.
    def parent(n,offset,split):return pd.DataFrame(dict(cache_ix=np.arange(n),sample_ix=offset+np.arange(n),role=['system','user','assistant','cot','tool'][:n],split=split))
    parents={'old':parent(5,10,'fit'),'c4':pd.DataFrame(dict(cache_ix=np.arange(327682),sample_ix=6_000_000_000_000+np.arange(327682),role='user',split='fit')),'natural':parent(3,4_000_000_000_000,'fit')};parents['c4'].loc[327680:,'split']='validation'
    pieces=[]
    for name,frame in parents.items():
        chosen=frame.iloc[327680:] if name=='c4' else frame
        for r in chosen.itertuples():pieces.append(dict(parent_bank=name,parent_cache_ix=r.cache_ix,sample_ix=r.sample_ix,role=r.role,split=r.split))
    erows=pd.DataFrame(pieces);erows['cache_ix']=np.arange(len(erows));lookup=extra_mapping(erows,parents,expected=len(erows))
    check('newC4_parent327680_maps_exactly_without_fit_rows',(lookup['c4'][:327680]==-1).all() and lookup['c4'][327680:].tolist()==[5,6])
    bad=erows.copy();bad.loc[5,'sample_ix']=999;reject('wrong_parent_sample_mapping_rejected',lambda:extra_mapping(bad,parents,expected=len(erows)))
    rawbase=rng.normal(size=(5,512)).astype(np.float16);rawextra=rng.normal(size=(len(erows),128)).astype(np.float16)
    new=CombinedBank(rawbase,rawextra,lookup['old']);dup=CombinedBank(rawbase,rawextra,lookup['old'],True);ix=np.array([4,0,2]);n,d=new[ix],dup[ix]
    check('common_original512_features_and_exact_rank01_duplicate',np.array_equal(n[:,:512],d[:,:512]) and np.array_equal(d[:,512:],d[:,:128]) and np.array_equal(n[:,512:],rawextra[lookup['old'][ix]]))
    missing=CombinedBank(rawbase,rawextra,lookup['c4'],True);reject('unmapped_newC4fit_forbidden_in_both_arms',lambda:missing[np.array([0])])
    # Scaler is fitted only on an explicit original-fit index list; extreme
    # unused natural/validation rows cannot change population statistics.
    scaler_bank=rng.normal(size=(18,128)).astype(np.float16);scaler_bank[:,0]=3.;scaler_bank[12:]=10000;fit_ix=np.arange(12);m,s=hp.fit_scaler(scaler_bank,fit_ix,chunk=4)
    expected_mean=scaler_bank[:12].astype(np.float64).mean(0).astype(np.float32);expected_scale=np.maximum(scaler_bank[:12].astype(np.float64).std(0),1e-6).astype(np.float32)
    check('F64_Welford_population_scaler_matches_raw_fit_only',np.allclose(m,expected_mean,rtol=1e-6,atol=1e-7) and np.allclose(s,expected_scale,rtol=1e-6,atol=1e-7) and s[0]==np.float32(1e-6))
    oldmean=rng.normal(size=512).astype(np.float32);oldscale=np.ones(512,np.float32);mean=np.r_[oldmean,m].astype(np.float32);scale=np.r_[oldscale,s].astype(np.float32)
    check('first512_scaler_frozen_and_duplicate_tail_exact',np.array_equal(mean[:512].view(np.uint32),oldmean.view(np.uint32)) and np.array_equal(np.r_[oldmean,oldmean[:128]][512:],oldmean[:128]))
    standardized_new=(n.astype(np.float32)[:,:512]-mean[:512])/scale[:512];standardized_dup=(d.astype(np.float32)[:,:512]-oldmean)/oldscale
    check('new_and_duplicate_original512_standardization_bitexact',np.array_equal(standardized_new.view(np.uint32),standardized_dup.view(np.uint32)))
    model=folder/'synthetic-model.npz';metadata=dict(kind='binary-residual-640',arm='new',dimensions=640,base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',roles=data.ROLES,
        base_model_sha256=BASE_SHA,embedded_base_weight_sha256=state_hash(base),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),tests_read=False)
    clean=make_model(base);save_model(model,clean,mean,scale,metadata);loaded,lm,ls,meta=load_model(model)
    clean.eval();p0=clean((torch.from_numpy(n.astype(np.float32))-torch.from_numpy(mean))/torch.from_numpy(scale)).exp().detach().numpy();p1=loaded((torch.from_numpy(n.astype(np.float32))-lm)/ls).exp().detach().numpy()
    check('portable640_roundtrip_probabilities_bitexact',np.array_equal(p0,p1) and meta==metadata)
    tinybank=CombinedBank(rawbase,rawextra,lookup['old']);tinyrows=pd.DataFrame(dict(cache_ix=np.arange(5),role=data.ROLES));p,proof=predict_raw(model,tinybank,tinyrows)
    check('reused_pure_predictor_supports640_raw_preservation',p.shape==(5,5) and proof['raw_non_system_sha256']==proof['base_raw_non_system_sha256'] and proof['non_system_to_different_non_system_p_argmax']==0)
    # Synthetic grouping/calibration comes from existing pure in-memory fixture,
    # never dataset()/a saved p cache. Fourteen+two records must always be shown.
    frames={b:[] for b in ['old','c4','public']}
    for pi,profile in enumerate(['single','multi','natural','long-single','long-full']):
        for ri,role in enumerate(data.ROLES):
            if profile=='natural' and role=='tool':continue
            for offset in [0,341,682,1023]:frames['old'].append(dict(profile=profile,role=role,token_ix=17+offset,prompt_ix=pi*5+ri,message_ix=0,document_id=pi,group_id='old:'+str(pi),source_set='fixture',cache_ix=len(frames['old']),sample_ix=len(frames['old'])))
    for ri,role in enumerate(data.ROLES):
        for offset in [0,341,682,1023]:frames['c4'].append(dict(profile='long-full',role=role,token_ix=23+offset,body_offset=offset,prompt_ix=ri,message_ix=0,document_id=6000000,group_id='new',source_set='long-augmentation',cache_ix=len(frames['c4']),sample_ix=6000000000000+len(frames['c4'])))
    for di,dataset_name in enumerate(['oasst','toxicchat0124']):
        for role in ['system','user','assistant']:
            for offset in range(4):frames['public'].append(dict(role=role,dataset=dataset_name,source_uid=dataset_name+':'+str(di),group_id=dataset_name+':'+str(di),cache_ix=len(frames['public']),sample_ix=4000000000000+len(frames['public'])))
    rows={k:pd.DataFrame(v) for k,v in frames.items()};p={}
    for bank,frame in rows.items():
        y=probes.labels(frame);values=np.full((len(frame),5),.02,np.float32);values[np.arange(len(frame)),y]=.92;p[bank]=values
    tau,grid=max33(rows,p);value=metrics_at(rows,p,tau);records=[dict(record=name,validation=copy.deepcopy(value)) for name in RECORDS];zero={name:copy.deepcopy(value) for name in ['zero-oldmean','zero-max33']}
    selected=select_records(records,value,zero);check('14_records_plus_zero192CSV_and_five_strict_equality_guards',len(flat_metrics(records))==168 and len(flat_metrics([dict(record=k,validation=v) for k,v in zero.items()]))==24 and len(selected['min33_gains'])==5 and not selected['promoted'])
    check('eight_hard_guards_plus_original_savedGPU',len(selected['guards'])==9 and set(selected['guards'])==set(GUARD_CONTROLS)|{'original-savedGPU'})
    check('priorLongMix_and_naturalCE_remain_diagnostic',selected['priorLongMix_is_diagnostic_only'] and selected['naturalCE_is_diagnostic_only'] and 'priorLongMix-max33' not in GUARD_CONTROLS)
    check('157_updates_last2517_and_641493_instances',INSTANCES//BATCH+1==STEPS and INSTANCES%BATCH==2517)
    check('frozen_module_OUT_CACHE_globals_unchanged',original_globals==(br.OUT,br.CACHE,prior.OUT,prior.CACHE,extra.OUT,extra.CACHE))
    check('source_unchanged_during_preflight',source==data.digest(__file__))
    report=dict(stage='synthetic_complementary640_readout',source_sha256=source,checks=checks,check_count=len(checks),extra_complete_sha256=EXTRA_COMPLETE_SHA,extra_audit_sha256=EXTRA_AUDIT_SHA,
        actual_enabled=False,real_extra_bank_or_probabilities_read=False,real_source_metadata_or_weights_read=False,gpu_used=False,tests_read=False,formal_outputs_written=False)
    output=folder/'report.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=data.digest(output),source_sha256=source,checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--declare',action='store_true');group.add_argument('--train',action='store_true');parser.add_argument('--restart-incomplete',action='store_true');parser.add_argument('--worker',choices=ARMS);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    require(not args.restart_incomplete or args.train,'Explicit restart is training-only.')
    if args.preflight:preflight()
    elif args.declare:declare()
    elif args.train and args.worker:worker(args.worker,args.gpu)
    elif args.train:train(args.restart_incomplete)
    else:parser.print_help()

if __name__=='__main__':main()
