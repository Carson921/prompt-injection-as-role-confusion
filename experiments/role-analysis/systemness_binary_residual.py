"""Bounded system-logit residual with the original portable MLP128 frozen.

Default help; --preflight is synthetic CPU/tmp only. --declare and --train are
explicit, require a root-frozen design, and never run an LM or read any test.
Only original raw logit0 changes: delta=4*tanh(g(x)), g:512->32->1 with zero last
layer. BCE trains system log-odds; all non-system raw logits remain bitexact.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
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

OUT=data.runtime.OUT/'systemness-binary-residual'
CACHE=data.CACHE/'binary-residual'
DESIGN=data.runtime.OUT/'systemness-binary-residual-preparation/root-prospective-design.json'
DESIGN_SHA='c68a32d55e732f678f77a7ef91d8d966a416f118d43663f913b3c9f2d14e09c5'
ARMS=['new','matched'];MEMBERS=list(nf.MEMBERS)
RECORDS=list(nf.RECORDS)+['naturalCE-oldmean','naturalCE-max33']
GROUPS,FEASIBLE,PUBLIC,FULL,OLD5=nf.GROUPS,nf.FEASIBLE,nf.PUBLIC,nf.FULL,nf.OLD5
GUARD_CONTROLS=list(nf.GUARD_CONTROLS)+['zero-oldmean','zero-max33']
STRICT_CONTROLS=list(nf.GUARD_CONTROLS[:3])+['zero-max33']
FIT,EXTRA,INSTANCES,BATCH,STEPS=nf.FIT,nf.EXTRA,nf.INSTANCES,nf.BATCH,nf.STEPS
BASE_SHA=aug.CURRENT_MODEL_SHA
BASE_PATH=hp.OUT/'models/joint-mlp128.npz'
ARTIFACTS=['protocol.json','search-plan.json','draw-manifest.json','controls.json','zero-controls.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt']
DEPENDENCIES=sorted(set(nf.DEPENDENCIES+['systemness_binary_residual.py']))
require,read,array_sha=hp.require,hp.read,hp.array_digest
layout,matched_draw,validation_groups,metrics_at,max33,budget=nf.layout,nf.matched_draw,nf.validation_groups,nf.metrics_at,nf.max33,nf.budget
LIMITATION=(nf.LIMITATION+' Residual only changes system odds, not relative non-system raw scores. Probability-rounding ties may still change p.argmax; '
    'five-class metrics keep the inherited convention and any observed non-system-to-different-non-system winner swap fails. NaturalCE is a strong diagnostic, never an extra hardfloor or fallback.')


def protocol():
    return dict(version=1,experiment='bounded-binary-system-logit-residual',arms=ARMS,records=RECORDS,
        fit_rows=FIT,natural_fit=EXTRA,validation_rows=dict(old=173032,c4=61440,public=16713),roles=data.ROLES,
        neural=dict(seed=20261021,residual_hidden=32,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=BATCH,
            instances=INSTANCES,steps=STEPS,last_batch=2517,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5),
        frozen_base=dict(model_sha256=BASE_SHA,hidden=128,dimensions=512,eval_even_when_parent_train=True,requires_grad=False),
        residual='delta=4*tanh(g), no division by4; g=Linear512x32/GELU/dropout.1/Linear32x1 with final weight+bias exactly zero.',
        loss='BCEWithLogits(raw0+delta-logsumexp(raw1:), truth==system); optimizer contains only correction parameters.',
        early_stop='Original old5 helper only; best/history/model/GPUoldp freeze before all C4/public candidate and zero-control predictions.',
        CPU=dict(threads=4,float32_matmul_precision='highest',dtype='float32',chunk=4096),
        parity='Only fixed first256 clean natural FIT vectors for actual CPU/GPU portable proof. Never validation parity.',
        raw_invariance='Hash raw non-system bitpatterns; conditional raw argmax unchanged. Require zero observed global non-system-to-different-non-system winner swaps; report conditional p ties/underflow. No epsilon or custom decision rule.',
        groups=GROUPS,feasible=FEASIBLE,max_FPR=.05,maximin='Inherited actualF32 fullgrid min33; tie old5meanF1 then smalleroriginalF64tau.',
        promotion=dict(candidate='new-max33',absolute_min33=.9,strict_controls=STRICT_CONTROLS,guard_controls=GUARD_CONTROLS,
            mandatory_original_savedGPU=True,naturalCE='Two diagnostic records only; no hardfloor.',no_fallback=True,tolerance=1e-12),
        zero_controls=['zero-oldmean','zero-max33'],zero_controls_are_trained_records=False,
        tests_read=False,primary_changed=False,new_independent_confirmation_required=True,limitation=LIMITATION)


def state_hash(state):
    return aug.json_digest([dict(name=n,shape=list(v.shape),dtype=str(v.dtype),sha256=array_sha(v)) for n,v in sorted(state.items())])


def make_model(base_state,seed=20261021):
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.base=hp.make_net('mlp',hidden=128)
            self.correction=nn.Sequential(nn.Linear(512,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
            nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
            self.base.load_state_dict({n:torch.as_tensor(v).clone() for n,v in base_state.items()});self.base.eval().requires_grad_(False)
        def train(self,mode=True):
            super().train(mode);self.base.eval();return self
        def components(self,x):
            with torch.no_grad():base=self.base.raw(x)
            delta=4*torch.tanh(self.correction(x));raw=base.clone();raw[:,0]+=delta[:,0]
            return raw,base,delta
        def raw(self,x):return self.components(x)[0]
        def binary_logit(self,x):
            raw=self.raw(x);return raw[:,0]-torch.logsumexp(raw[:,1:],dim=1)
        def forward(self,x):return torch.log_softmax(self.raw(x),dim=1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed);net=Model()
    return net


def read_base(path=BASE_PATH,expected_sha=BASE_SHA):
    hp.require_hash(path,expected_sha,'frozen original base portable')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));require(meta['kind']=='mlp' and meta['hidden']==128 and meta['dimensions']==512 and meta['roles']==data.ROLES,'Original base geometry differs.')
        state={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')};mean,scale=z['mean'].copy(),z['scale'].copy()
    expected={'encoder.0.weight':(128,512),'encoder.0.bias':(128,),'head.weight':(5,128),'head.bias':(5,)}
    require(set(state)==set(expected) and all(v.shape==expected[n] and v.dtype==np.float32 and np.isfinite(v).all() for n,v in state.items()),'Malformed frozen base weights.')
    require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Malformed original embedded scaler.')
    return state,mean,scale,meta


def initial_hash(base_state):return aug.init_hash(make_model(base_state))


def save_model(path,net,mean,scale,metadata):
    require(metadata['base_model_sha256']==BASE_SHA and metadata['roles']==data.ROLES,'Portable base/role identity differs.')
    arrays={'weight::'+n:v.detach().cpu().numpy() for n,v in net.state_dict().items()}
    hp.atomic_npz(path,mean=np.asarray(mean,np.float32),scale=np.asarray(scale,np.float32),metadata=np.array(json.dumps(metadata)),**arrays)


def load_model(path,device='cpu'):
    import torch
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));require(meta['kind']=='binary-residual' and meta['dimensions']==512 and meta['base_hidden']==128
            and meta['residual_hidden']==32 and meta['delta_formula']=='4*tanh(g)' and meta['roles']==data.ROLES and meta['base_model_sha256']==BASE_SHA,'Portable residual geometry/base/formula differs.')
        state={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')};base={n[5:]:v for n,v in state.items() if n.startswith('base.')}
        require(state_hash(base)==meta['embedded_base_weight_sha256'],'Embedded frozen base bytes changed.')
        mean,scale=z['mean'].copy(),z['scale'].copy()
    require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Portable scaler changed.')
    require(array_sha(mean)==meta['mean_sha256'] and array_sha(scale)==meta['scale_sha256'],'Embedded scaler bytes differ.')
    require(all(v.dtype==np.float32 and np.isfinite(v).all() for v in state.values()),'Nonfinite/non-F32 portable residual weights.')
    net=make_model(base);net.load_state_dict({n:torch.from_numpy(v) for n,v in state.items()});net.eval().to(device)
    require(not any(v.requires_grad for v in net.base.parameters()),'Portable base became trainable.')
    return net,torch.from_numpy(mean).to(device),torch.from_numpy(scale).to(device),meta


def raw_rest_sha(raw):return array_sha(np.ascontiguousarray(raw[:,1:]).view(np.uint32))


def predict_net(net,mean,scale,bank,rows,device='cpu'):
    """Pointwise same-consumer p plus exact raw invariance and p-transition evidence."""
    import torch
    parts=[];rest=hashlib.sha256();base_rest=hashlib.sha256();delta_min=4.;delta_max=-4.;delta_sum=0.;count=0
    transitions=np.zeros((5,5),np.int64);raw_changes=0;non_sys_p_changes=0;prob_ties=0;underflow=0
    net.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            raw=torch.from_numpy(np.asarray(bank[rows.cache_ix.to_numpy()[start:start+BATCH]],np.float32)).to(device);x=(raw-mean)/scale
            adjusted,base,delta=net.components(x);require(torch.isfinite(adjusted).all() and torch.isfinite(delta).all() and bool((delta.abs()<=4).all()),'Invalid bounded residual/logits.')
            a,b=adjusted.cpu().numpy(),base.cpu().numpy();rest.update(np.ascontiguousarray(a[:,1:]).view(np.uint32).tobytes());base_rest.update(np.ascontiguousarray(b[:,1:]).view(np.uint32).tobytes())
            require(np.array_equal(a[:,1:].view(np.uint32),b[:,1:].view(np.uint32)),'Residual modified a frozen non-system raw logit.')
            raw_changes+=int(np.count_nonzero(a[:,1:].argmax(1)!=b[:,1:].argmax(1)))
            p=torch.log_softmax(adjusted,dim=1).exp().cpu().numpy().astype(np.float32);pb=torch.log_softmax(base,dim=1).exp().cpu().numpy().astype(np.float32)
            pred,basepred=p.argmax(1),pb.argmax(1);np.add.at(transitions,(basepred,pred),1)
            non_sys_p_changes+=int(np.count_nonzero(p[:,1:].argmax(1)!=pb[:,1:].argmax(1)))
            prob_ties+=int(np.count_nonzero((p[:,1:]==p[:,1:].max(1,keepdims=True)).sum(1)>1))
            underflow+=int(np.count_nonzero(p[:,1:]==0));d=delta.cpu().numpy();delta_min=min(delta_min,float(d.min()));delta_max=max(delta_max,float(d.max()));delta_sum+=float(d.astype(np.float64).sum());count+=len(d);parts.append(p)
    p=np.concatenate(parts);hp.validate_probabilities(p,len(rows));require(rest.hexdigest()==base_rest.hexdigest() and raw_changes==0,'Raw tail identity proof failed.')
    swaps=int(transitions[1:,1:].sum()-np.trace(transitions[1:,1:]));require(swaps==0,'Observed non-system p.argmax changed between non-system classes; no epsilon repair allowed.')
    return p,dict(tokens=count,raw_non_system_sha256=rest.hexdigest(),base_raw_non_system_sha256=base_rest.hexdigest(),raw_non_system_argmax_changes=raw_changes,
        delta_min=delta_min,delta_max=delta_max,delta_mean=delta_sum/count,base_to_residual_p_argmax=transitions.tolist(),
        p_conditional_non_system_argmax_changes=non_sys_p_changes,p_non_system_tie_rows=prob_ties,p_non_system_underflow_values=underflow,non_system_to_different_non_system_p_argmax=swaps,CoT_to_assistant_p_argmax=int(transitions[2,3]),transition_definition='Rows index original base p.argmax; columns index residual p.argmax, regardless of ground-truth role. CoT_to_assistant is a newly introduced winner transition, not total truth-CoT misclassification.',
        non_system_to_system_p_argmax=int(transitions[1:,0].sum()),system_to_non_system_p_argmax=int(transitions[0,1:].sum()),
        metric_convention='Unmodified savedF32 p.argmax; raw-tail bitidentity does not guarantee p rounding preserves non-system argmax.')


def predict_raw(path,bank,rows,device='cpu'):
    net,mean,scale,_=load_model(path,device);return predict_net(net,mean,scale,bank,rows,device)


def parity(path,banks,nfrows):
    require(nfrows.split.eq('fit').all() and len(nfrows)>=256,'Parity must use clean FIT only.')
    chosen=nfrows.iloc[:256];cpu,_=predict_raw(path,banks['natural'],chosen);gpu,_=predict_raw(path,banks['natural'],chosen,'cuda:0')
    error=float(np.abs(cpu-gpu).max());require(np.isfinite(error) and error<5e-4,'FIT-only portable CPU/GPU parity failed.')
    return dict(fit_only=True,validation_probabilities_computed=False,sampled_tokens=256,target_cache_ix_sha256=array_sha(chosen.cache_ix.to_numpy()),
        sample_ix_sha256=array_sha(chosen.sample_ix.to_numpy()),maximum_probability_difference=error)


def select_records(records,legacy,zero):
    require(len(records)==10 and {r['record'] for r in records}==set(RECORDS) and set(zero)=={'zero-oldmean','zero-max33'},'Ten records and two separate zero controls required.')
    index={r['record']:r for r in records};candidate=index['new-max33']['validation'];minimum=candidate['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARD_CONTROLS:
        control=zero[name] if name in zero else index[name]['validation'];g=nf.guard(candidate,control);guards[name]=g;reasons += [name+':'+r for r in g['rejection_reasons']]
        if name in STRICT_CONTROLS:
            gain=minimum-control['min33'] if minimum is not None and control['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    gpu=nf.guard(candidate,legacy,True);guards['original-savedGPU']=gpu;reasons += ['original-savedGPU:'+r for r in gpu['rejection_reasons']]
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,
        naturalCE_is_diagnostic_only=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


PRIOR_AUDIT=nf.OUT/'preparation/audits/actual-development-audit.json'
NATURAL_CE_SHA='204f4a9dbfb7855460ccb210ca703aca721b974a87059be9134f55c4f7b5cba5'
NATURAL_CE_TAU=.7772222222222223


def static_snapshot():
    """Metadata declaration: bind audited reference p hashes without opening p arrays."""
    hp.require_hash(DESIGN,DESIGN_SHA,'root binary-residual design');design=read(DESIGN);prior=nf.static_snapshot()
    hp.require_hash(PRIOR_AUDIT,design['background']['latest_actual_audit_sha256'],'prior natural-fit actual audit')
    audit=read(PRIOR_AUDIT);complete=read(nf.OUT/'complete.json');selection=read(nf.OUT/'selection.json');controls=read(nf.OUT/'controls.json');candidate=read(nf.OUT/'candidates/new.json')
    require(audit['stage']=='actual_completed_natural_fit_development_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False
        and audit['trainer_source_sha256']==data.digest(nf.__file__) and audit['trainer_complete_sha256']==data.digest(nf.OUT/'complete.json')
        and audit['trainer_protocol_sha256']==data.digest(nf.OUT/'protocol.json') and audit['trainer_selection_sha256']==data.digest(nf.OUT/'selection.json')
        and audit['trainer_records_sha256']==data.digest(nf.OUT/'records.json') and selection['promoted'] is False and complete['source_sha256']==data.digest(nf.__file__),
        'Inherited audited natural-fit negative trial changed.')
    for name,digest in complete['artifact_sha256'].items():hp.require_hash(nf.OUT/name,digest,'inherited natural-fit artifact')
    for arm,digest in complete['candidate_sha256'].items():hp.require_hash(nf.OUT/'candidates'/f'{arm}.json',digest,'inherited natural-fit candidate')
    require(candidate['model_sha256']==selection['model_sha256']==audit['model_sha256']['new']==NATURAL_CE_SHA and controls['model_sha256']['20261021']==BASE_SHA,'Original or diagnostic CE model changed.')
    hp.require_hash(nf.OUT/'models/new.npz',NATURAL_CE_SHA,'inherited diagnostic CE portable')
    fit,ov,cv,nrows,pv,part=layout();samples={n:array_sha(r.sample_ix.to_numpy()) for n,r in [('fit',fit),('old',ov),('c4',cv),('natural-fit',nrows),('public',pv)]}
    require(samples==design['data']['source_sample_sha256'] and part==design['data']['original_partitions'],'Prospective exact partitions changed.')
    draw=Path(design['matching']['draw_path']);hp.require_hash(draw,design['matching']['draw_sha256'],'prospective exact prior role-slot draw')
    expected,summary=matched_draw(fit,nrows);pd.testing.assert_frame_equal(pd.read_parquet(draw),expected,check_exact=True)
    base,mean,scale,_=read_base();m,s=aug.read_scaler();require(np.array_equal(mean,m) and np.array_equal(scale,s),'Original embedded/global frozen scaler differ.')
    references={}
    for model,producer,hashes in [('oldhead','seed-20261021',controls['members']['20261021']['probability_sha256']),('ensemble','ensemble',controls['ensemble_probability_sha256']),('naturalCE','new',candidate['probability_sha256'])]:
        references[model]={b:dict(path=str(nf.ppath(producer,b)),sha256=hashes[b],sample_ix_sha256=samples[b]) for b in ['old','c4','public']}
    files=[PRIOR_AUDIT,Path(audit['auditor_source_path']),nf.OUT/'models/new.npz',nf.OUT/'training/new.json',nf.OUT/'candidates/new.json']
    files += [nf.OUT/n for n in ['protocol.json','search-plan.json','complete.json','selection.json','controls.json','records.json','source-counts.json','draw-manifest.json']]
    hp.require_hash(files[1],audit['auditor_source_sha256'],'prior actual auditor source')
    return dict(source_sha256=data.digest(__file__),design_sha256=DESIGN_SHA,prior=prior,prior_actual_audit_sha256=data.digest(PRIOR_AUDIT),
        metadata_sha256={str(p):data.digest(p) for p in files},dependencies_sha256={n:data.digest(data.HERE/n) for n in DEPENDENCIES},
        base_model_path=str(BASE_PATH),base_model_sha256=BASE_SHA,embedded_base_weight_sha256=state_hash(base),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),
        initial_weight_sha256=initial_hash(base),standardization_sha256=prior['standardization_sha256'],original_partitions=part,sample_ix_sha256=samples,
        draw_source_path=str(draw),draw_slots_sha256=data.digest(draw),draw_summary=summary,reference_probabilities=references,
        naturalCE_model_path=str(nf.OUT/'models/new.npz'),naturalCE_model_sha256=NATURAL_CE_SHA,old_reference=prior['old_reference'],
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},validation_probability_arrays_read=False)


def declare():
    if (OUT/'protocol.json').exists():return verify_declaration()
    require(not CACHE.exists() and not (OUT/'search-plan.json').exists(),'Unowned partial binary-residual cache/plan; inspect explicitly.')
    static=static_snapshot();OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True)
    shutil.copyfile(static['draw_source_path'],CACHE/'matched-draw.parquet');hp.require_hash(CACHE/'matched-draw.parquet',static['draw_slots_sha256'],'copied exact prior draw')
    data.atomic_json(OUT/'draw-manifest.json',static['draw_summary']|dict(slots_path=str(CACHE/'matched-draw.parquet'),slots_sha256=static['draw_slots_sha256'],source_path=static['draw_source_path']))
    value=dict(protocol=protocol(),static=static);data.atomic_json(OUT/'protocol.json',value);return value


def verify_declaration():
    require((OUT/'protocol.json').is_file(),'Run --declare before training or candidate probability reads.')
    value=read(OUT/'protocol.json');require(value==dict(protocol=protocol(),static=static_snapshot()),'Frozen binary source/design/reference/rows changed.')
    hp.require_hash(CACHE/'matched-draw.parquet',value['static']['draw_slots_sha256'],'frozen matched slots')
    expected=value['static']['draw_summary']|dict(slots_path=str(CACHE/'matched-draw.parquet'),slots_sha256=value['static']['draw_slots_sha256'],source_path=value['static']['draw_source_path'])
    require(read(OUT/'draw-manifest.json')==expected,'Matched draw descriptor changed.');return value


def source_signature():
    value=verify_declaration();up=nf.source_signature()
    return dict(source_sha256=data.digest(__file__),protocol_sha256=data.digest(OUT/'protocol.json'),static_sha256=aug.json_digest(value['static']),
        inherited=up,selected_head_ids=up['selected_head_ids'],standardization_sha256=value['static']['standardization_sha256'],
        base_model_sha256=BASE_SHA,embedded_base_weight_sha256=value['static']['embedded_base_weight_sha256'])


def verify_plan(plan,full=True):
    value=verify_declaration();require(plan['protocol']==protocol() and plan['signature']['protocol_sha256']==data.digest(OUT/'protocol.json')
        and plan['initial_weight_sha256']==value['static']['initial_weight_sha256'],'Plan/protocol/full-state init changed.')
    if full:require(plan['signature']==source_signature(),'Features/rows/data/source signature changed.')
    return value


def ppath(model,bank):return CACHE/f'{model}-{bank}-validation.npz'
def read_predictions(model,record,rows):return {b:aug.read_probability(ppath(model,b),record['probability_sha256'][b],frame) for b,frame in rows.items()}
def dataset():return nf.dataset()


def portable_metadata(plan,arm,**extra):
    static=read(OUT/'protocol.json')['static']
    return dict(kind='binary-residual',arm=arm,dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=data.ROLES,
        base_model_sha256=BASE_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],mean_sha256=static['mean_sha256'],scale_sha256=static['scale_sha256'],
        selected_head_ids=plan['signature']['selected_head_ids'],signature=plan['signature'],source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),
        initial_weight_sha256=plan['initial_weight_sha256'],standardization_sha256=plan['signature']['standardization_sha256'],tests_read=False,**extra)


def verify_portable(path,plan,arm):
    net,mean,scale,meta=load_model(path);static=read(OUT/'protocol.json')['static'];base,m,s,_=read_base()
    require(state_hash({n:v.detach().numpy() for n,v in net.base.state_dict().items()})==state_hash(base)==static['embedded_base_weight_sha256']
        and np.array_equal(mean.numpy(),m) and np.array_equal(scale.numpy(),s),'Embedded base/scaler differ from actual original portable arrays.')
    require(all(meta[k]==v for k,v in portable_metadata(plan,arm).items()),'Portable residual signature/head order/init provenance changed.')
    return net,mean,scale,meta


def supervisor_alive():require(os.environ.get('ROLE_BINARY_RESIDUAL_SUPERVISOR_PID')==str(os.getppid()),'Worker requires original live supervisor lease.')


def worker(arm,gpu):
    supervisor_alive();require(arm in ARMS and gpu in [0,1],'Only declared two arm workers allowed.')
    require(not (OUT/'complete.json').exists() and not (OUT/'candidates'/f'{arm}.json').exists(),'Frozen candidate cannot be overwritten.')
    no_competing_jobs(True);plan=read(OUT/'search-plan.json');verify_plan(plan);ms.configure_torch()
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1 and 'A40' in torch.cuda.get_device_name(0),'Worker requires its assigned A40.')
    banks,fit,ov,cv,nrows,pv,_=dataset();base,mean,scale,_=read_base();slots=pd.read_parquet(CACHE/'matched-draw.parquet')
    def standardized(bank,indices):
        out=torch.empty((len(indices),512),dtype=torch.float32,device='cuda')
        for start in range(0,len(indices),BATCH):
            x=(np.asarray(banks[bank][indices[start:start+BATCH]],np.float32)-mean)/scale
            require(np.isfinite(x).all(),'Nonfinite standardized feature.');out[start:start+len(x)]=torch.from_numpy(x).cuda()
        return out
    require(torch.cuda.mem_get_info(0)[0]>(INSTANCES+173032)*512*4+2*2**30,'Insufficient assigned worker memory.')
    old_x=standardized('old',fit.cache_ix.to_numpy());extra_x=standardized('natural' if arm=='new' else 'old',nrows.cache_ix.to_numpy() if arm=='new' else slots.matched_old_cache_ix.to_numpy())
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


def probability_drift(reference,p,tau):
    hp.validate_probabilities(reference,len(p));hp.validate_probabilities(p,len(reference));difference=np.abs(reference.astype(np.float64)-p.astype(np.float64))
    return dict(rows=len(p),maximum_probability_difference=float(difference.max()),mean_probability_difference=float(difference.mean()),
        changed_argmax=int(np.count_nonzero(reference.argmax(1)!=p.argmax(1))),changed_original_threshold_decisions=int(np.count_nonzero((reference[:,0]>=tau)!=(p[:,0]>=tau))),threshold=float(tau))


def verify_raw_evidence(proof,count):
    t=np.asarray(proof['base_to_residual_p_argmax']);require(t.shape==(5,5) and np.issubdtype(t.dtype,np.integer) and (t>=0).all() and int(t.sum())==count
        and proof['tokens']==count and proof['raw_non_system_sha256']==proof['base_raw_non_system_sha256'] and len(proof['raw_non_system_sha256'])==64
        and proof['raw_non_system_argmax_changes']==0 and proof['non_system_to_different_non_system_p_argmax']==int(t[1:,1:].sum()-np.trace(t[1:,1:]))==0
        and proof['CoT_to_assistant_p_argmax']==int(t[2,3])==0 and proof['non_system_to_system_p_argmax']==int(t[1:,0].sum())
        and proof['system_to_non_system_p_argmax']==int(t[0,1:].sum()),'Full-bank raw identity/observed non-system winner proof invalid.')
    require(all(np.isfinite(proof[k]) for k in ['delta_min','delta_max','delta_mean']) and -4<=proof['delta_min']<=proof['delta_mean']<=proof['delta_max']<=4
        and all(isinstance(proof[k],int) and proof[k]>=0 for k in ['p_conditional_non_system_argmax_changes','p_non_system_tie_rows','p_non_system_underflow_values']),'Residual bound/tie/underflow evidence invalid.')


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


def inherited_controls(plan,rows,allow_create=False):
    """Use only previously audited CPU arrays, never recompute their probabilities."""
    static=verify_plan(plan,False)['static'];p={}
    for model,bindings in static['reference_probabilities'].items():
        p[model]={b:aug.read_probability(item['path'],item['sha256'],rows[b]) for b,item in bindings.items()}
    controls=read(nf.OUT/'controls.json');legacy=controls['original_savedGPU_validation']
    # Independently reproduce the immutable savedGPU floors from their actual files.
    ref=static['old_reference'];gpu={b:aug.read_probability(ref[k+'_path'],ref[k+'_sha256'],rows[b]) for b,k in [('old','old'),('c4','new')]}
    require(metrics_at(rows,dict(gpu,public=p['oldhead']['public']),nf.ORIGINAL_TAU)==legacy,'Inherited original savedGPU floors no longer reproduce.')
    content=dict(signature=plan['signature'],prior_actual_audit_sha256=static['prior_actual_audit_sha256'],reference_probabilities=static['reference_probabilities'],
        physical_model_sha256=dict(oldhead=BASE_SHA,ensemble=static['prior']['model_sha256'],naturalCE=NATURAL_CE_SHA),
        original_savedGPU_validation=legacy,reference_probabilities_recomputed=False,naturalCE_diagnostic_only=True,tests_read=False)
    path=OUT/'controls.json'
    if path.exists():require(read(path)==content,'Inherited reference controls changed.')
    else:require(allow_create,'Missing inherited control record in read-only verification.');data.atomic_json(path,content)
    return p,legacy


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
        base,mean,scale,_=read_base();net=make_model(base);require(aug.init_hash(net)==plan['initial_weight_sha256'],'Zero full initialization changed.')
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
        else:tau,proof=lp.validation_metrics(ov,value['old'])['system_threshold'],None
        result=metrics_at(rows,value,tau)
        if proof is not None:require(result['min33']==proof['min33'] and result['old5_mean_f1']==proof['old5_mean_f1'],'Full-grid selected metrics differ.')
        if name=='naturalCE-max33':require(tau==.6008080808080808,'Inherited CE calibration no longer reproduces its audited threshold.')
        records.append(dict(record=name,model=model,objective=objective,validation=result,full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False))
    return records,legacy,zero,rows,fit,nrows


def flat_metrics(records):
    flat=[]
    for record in records:
        for group,value in dict(record['validation']['groups'],pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=record['record'],group=group,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))},**{'role_recall_'+r:v for r,v in value['recall'].items()}))
    return flat


def publish(plan):
    records,legacy,zero,rows,fit,nrows=calibrate(plan,True);selection=select_records(records,legacy,zero);data.atomic_json(OUT/'records.json',records)
    flat=flat_metrics(records);zf=flat_metrics([dict(record=k,validation=v) for k,v in zero.items()]);require(len(flat)==120 and len(zf)==24,'110+10 main/22+2 zero CSV rows required.')
    pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False);pd.DataFrame(zf).to_csv(OUT/'zero-control-metrics.csv',index=False)
    counts=read(nf.OUT/'source-counts.json')|dict(experiment='binary-residual',per_epoch=INSTANCES,steps=STEPS,last_batch=2517,
        same_existing_rows=True,inherited_naturalCE_diagnostic_only=True,limitation=LIMITATION);data.atomic_json(OUT/'source-counts.json',counts)
    lines=['Bounded binary residual: reused development validation only',LIMITATION,'','Record threshold min33 old5meanF1']
    lines += [f"{r['record']} {r['validation']['system_threshold']:.12g} {r['validation']['min33']} {r['validation']['old5_mean_f1']:.9f}" for r in records]
    lines += ['','Same-path zero numerical controls (not trained records)']+[f"{k} {v['system_threshold']:.12g} {v['min33']} {v['old5_mean_f1']:.9f}" for k,v in zero.items()]
    lines += ['','NaturalCE min33 differences: '+json.dumps({k:records[1]['validation']['min33']-next(r['validation']['min33'] for r in records if r['record']==k) for k in ['naturalCE-oldmean','naturalCE-max33']}),
        'Promoted: '+str(selection['promoted']),'Reasons: '+json.dumps(selection['rejection_reasons'])];(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    selection.update(source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),records_sha256=data.digest(OUT/'records.json'),model_path=str(OUT/'models/new.npz'),
        model_sha256=read(OUT/'candidates/new.json')['model_sha256'],controls_sha256=data.digest(OUT/'controls.json'),zero_controls_sha256=data.digest(OUT/'zero-controls.json'),
        system_threshold=next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'))
    data.atomic_json(OUT/'selection.json',selection);verify_plan(plan)
    require(all(p.stat().st_size<50*2**20 for folder in [OUT,CACHE] for p in folder.rglob('*') if p.is_file()),'Artifact exceeds50MiB storage bound.')
    data.atomic_json(OUT/'complete.json',dict(signature=plan['signature'],source_sha256=data.digest(__file__),artifact_sha256={n:data.digest(OUT/n) for n in ARTIFACTS},
        candidate_sha256={a:data.digest(OUT/'candidates'/f'{a}.json') for a in ARMS},tests_read=False,primary_changed=False,new_independent_confirmation_required=True))
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',promoted=selection['promoted'],primary_changed=False))


def verified_completion():
    require((OUT/'complete.json').is_file() and read(OUT/'status.json')['stage']=='validation_complete','Binary residual trial not terminal complete.')
    plan=read(OUT/'search-plan.json');verify_plan(plan);complete=read(OUT/'complete.json')
    require(complete['signature']==plan['signature'] and complete['source_sha256']==data.digest(__file__) and complete['tests_read'] is False
        and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and set(complete['artifact_sha256'])==set(ARTIFACTS) and set(complete['candidate_sha256'])==set(ARMS),'Completion signature/source/coverage changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(OUT/n,h,'completed artifact')
    for a,h in complete['candidate_sha256'].items():hp.require_hash(OUT/'candidates'/f'{a}.json',h,'completed candidate')
    records,legacy,zero,*_=calibrate(plan,False);require(records==read(OUT/'records.json'),'Ten saved records do not reproduce from frozen p.')
    expected=select_records(records,legacy,zero);actual=read(OUT/'selection.json');require(all(actual[k]==v for k,v in expected.items()),'Four strict/seven guard/savedGPU/absolute90 gate changed.')
    require(actual['source_sha256']==data.digest(__file__) and actual['plan_sha256']==data.digest(OUT/'search-plan.json') and actual['records_sha256']==data.digest(OUT/'records.json')
        and actual['controls_sha256']==data.digest(OUT/'controls.json') and actual['zero_controls_sha256']==data.digest(OUT/'zero-controls.json')
        and actual['model_path']==str(OUT/'models/new.npz') and actual['model_sha256']==read(OUT/'candidates/new.json')['model_sha256']
        and actual['system_threshold']==next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'),'Selected model/threshold/control closure changed.')
    return complete


def no_competing_jobs(allow_own=False):
    nf.no_competing_jobs(True);names={'systemness_natural_fit_probes.py'}
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
        except BlockingIOError as e:raise RuntimeError('Another binary-residual supervisor owns output.') from e
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run_workers(pending):
    require(len(pending)<=2 and len(set(pending))==len(pending) and set(pending)<=set(ARMS),'Only two fixed arm workers allowed.');jobs=[]
    try:
        for gpu,arm in enumerate(pending):
            log=(OUT/'logs'/f'{arm}.log').open('a')
            try:p=subprocess.Popen([sys.executable,'-u',__file__,'--train','--worker',arm,'--gpu',str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ROLE_BINARY_RESIDUAL_SUPERVISOR_PID=str(os.getpid())),stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            jobs.append((arm,p,log,gpu))
        data.atomic_json(OUT/'status.json',dict(stage='training',workers=[dict(arm=a,pid=p.pid,gpu=g) for a,p,_,g in jobs]))
        while jobs:
            codes=[(a,p.poll()) for a,p,_,_ in jobs];require(not [(a,c) for a,c in codes if c not in [None,0]],'Binary-residual worker failed: '+str(codes))
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
    ms.configure_torch();no_competing_jobs();verify_declaration();signature=source_signature()
    with run_lock():
        if (OUT/'complete.json').exists():verified_completion();return
        require(not (OUT/'selection.json').exists(),'Partial frozen selection exists; refusing replacement.')
        for name in ['models','training','candidates','logs']:(OUT/name).mkdir(exist_ok=True)
        path=OUT/'search-plan.json'
        if path.exists():plan=read(path);verify_plan(plan)
        else:
            require({p.name for p in CACHE.iterdir()}=={'matched-draw.parquet'} and all(not any((OUT/n).iterdir()) for n in ['models','training','candidates']),'Unowned models/probabilities predate plan.')
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


def preflight():
    """Synthetic CPU only; no real banks/models/development p or formal outputs."""
    import copy,py_compile,torch
    from unittest.mock import patch
    folder=Path(tempfile.mkdtemp(prefix='systemness-binary-residual-preflight-',dir='/tmp'));source=data.digest(__file__);checks=[];module=sys.modules[__name__]
    def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError,AssertionError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('CPU fault accepted: '+name)
    def forbidden(*a,**k):raise AssertionError('Real bank/model/probability/source API invoked by synthetic preflight.')
    with patch.object(nf,'dataset',side_effect=forbidden),patch.object(nf,'source_signature',side_effect=forbidden),patch.object(module,'static_snapshot',side_effect=forbidden),patch.object(data.runtime,'load_model',side_effect=forbidden):
        torch.set_num_threads(2);torch.set_float32_matmul_precision('medium');ms.configure_torch()
        check('Parent/worker CPU4threads highest enforced',torch.get_num_threads()==4 and torch.get_float32_matmul_precision()=='highest')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91);base={n:v.detach().numpy().copy() for n,v in hp.make_net('mlp',hidden=128).state_dict().items()}
        state=torch.get_rng_state().clone();net=make_model(base);net2=make_model(base)
        check('Private seed same complete base+residual init without caller RNG',torch.equal(state,torch.get_rng_state()) and aug.init_hash(net)==aug.init_hash(net2)==initial_hash(base))
        check('Exactly16449 trainable residual parameters',sum(v.numel() for v in net.parameters() if v.requires_grad)==16449)
        net.train();check('Parent train keeps frozen base eval/no-grad',not net.base.training and net.correction.training and not any(v.requires_grad for v in net.base.parameters()))
        check('Final correction layer exactly zero',torch.count_nonzero(net.correction[-1].weight)==0 and torch.count_nonzero(net.correction[-1].bias)==0)
        rng=np.random.RandomState(9);raw=rng.normal(size=(48,512)).astype(np.float16);mean=np.zeros(512,np.float32);scale=np.ones(512,np.float32);x=torch.from_numpy(raw.astype(np.float32));y=torch.arange(len(x))%5
        net.eval();adjusted,baseline,delta=net.components(x)
        check('Zero residual same-consumer logits/p bitexact',torch.equal(adjusted,baseline) and torch.count_nonzero(delta)==0 and torch.equal(net(x).exp(),torch.log_softmax(baseline,1).exp()))
        net.correction[-1].weight.data.fill_(.01);net.correction[-1].bias.data.fill_(.02);net.eval()
        torch.nn.functional.binary_cross_entropy_with_logits(net.binary_logit(x),y.eq(0).float()).backward();binary=[v.grad.clone() for v in net.correction.parameters()]
        net.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(net.raw(x),y).backward();ce=[v.grad.clone() for v in net.correction.parameters()]
        check('BCE and frozen-q fiveCE residual gradients equivalent',max(float((a-b).abs().max()) for a,b in zip(binary,ce))<2e-7)
        a,b,d=net.components(x);check('Nonzero delta preserves raw non-system bytes/order',np.array_equal(a.detach().numpy()[:,1:].view(np.uint32),b.detach().numpy()[:,1:].view(np.uint32)) and torch.equal(a[:,1:].argmax(1),b[:,1:].argmax(1)))
        last=net.correction[-1];last.weight.data.zero_();last.bias.data.zero_();net.zero_grad(set_to_none=True);net.eval();net.binary_logit(x).sum().backward()
        check('Zero-point tanh derivative4 not old fusion /4',abs(float(last.bias.grad)-4*len(x))<1e-5)
        check('Zero final layer first-layer grad0 and final finite',torch.count_nonzero(net.correction[0].weight.grad)==0 and torch.isfinite(last.weight.grad).all() and all(v.grad is None for v in net.base.parameters()))
        initial_base=state_hash(base);opt=torch.optim.AdamW(net.correction.parameters(),lr=.001,weight_decay=.001);opt.step()
        check('Optimizer changes only residual, frozen full base byteidentical',state_hash({n:v.detach().numpy() for n,v in net.base.state_dict().items()})==initial_base and {id(v) for group in opt.param_groups for v in group['params']}=={id(v) for v in net.correction.parameters()})
        last.weight.data.zero_();last.bias.data.fill_(100);check('Bounded delta saturates at4',torch.equal(net.components(x)[2],torch.full((len(x),1),4.)))
        net=make_model(base);net.eval();meta=dict(kind='binary-residual',dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=data.ROLES,
            base_model_sha256=BASE_SHA,embedded_base_weight_sha256=state_hash(base),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale))
        model=folder/'portable.npz';save_model(model,net,mean,scale,meta);loaded,m,s,_=load_model(model)
        check('Complete portable synthetic base/residual/scaler parity',torch.equal(loaded(x).exp(),net(x).exp()) and np.array_equal(m.numpy(),mean) and np.array_equal(s.numpy(),scale))
        with np.load(model,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        changed=copy.deepcopy(arrays);changed['weight::base.head.bias'][0]+=.1;hp.atomic_npz(folder/'bad-base.npz',**changed)
        reject('Portable rejects changed embedded frozen base bytes',lambda:load_model(folder/'bad-base.npz'))
        changed=copy.deepcopy(arrays);changed['scale'][0]=2;hp.atomic_npz(folder/'bad-scaler.npz',**changed)
        reject('Portable rejects changed embedded scaler',lambda:load_model(folder/'bad-scaler.npz'))
        changed=copy.deepcopy(arrays);badmeta=dict(meta,delta_formula='4*tanh(g/4)');changed['metadata']=np.array(json.dumps(badmeta));hp.atomic_npz(folder/'bad-formula.npz',**changed)
        reject('Portable refuses old fusion formula',lambda:load_model(folder/'bad-formula.npz'))
        toyrows=pd.DataFrame(dict(cache_ix=np.arange(len(raw)),sample_ix=np.arange(len(raw)),role=[data.ROLES[i%5] for i in range(len(raw))]))
        p,evidence=predict_raw(model,raw,toyrows);verify_raw_evidence(evidence,len(raw))
        check('Full-F16 prediction records raw hashes/transitions/ties/underflow',evidence['raw_non_system_sha256']==evidence['base_raw_non_system_sha256'] and evidence['delta_mean']==0 and evidence['non_system_to_different_non_system_p_argmax']==0)
        broken=copy.deepcopy(evidence);broken['base_to_residual_p_argmax'][1][1]-=1;broken['base_to_residual_p_argmax'][1][3]+=1;broken['non_system_to_different_non_system_p_argmax']=1
        reject('Observed non-system winner swap hard-fails rather than epsilon repair',lambda:verify_raw_evidence(broken,len(raw)))
        check('Old-p drift retains argmax and fixed-tau counts',probability_drift(p,p,nf.ORIGINAL_TAU)['changed_argmax']==0 and probability_drift(p,p,nf.ORIGINAL_TAU)['changed_original_threshold_decisions']==0)
        fit=pd.DataFrame([dict(cache_ix=i,sample_ix=i,role=data.ROLES[i%5],split='fit') for i in range(500)])
        labels=np.concatenate([np.repeat(role,count) for role,count in nf.ROLE_COUNTS['fit'].items()]);nrows=pd.DataFrame(dict(cache_ix=np.arange(EXTRA),sample_ix=4_000_000_000_000+np.arange(EXTRA),source_uid=['fit-'+str(i%255) for i in range(EXTRA)],role=labels,split='fit'))
        state=torch.get_rng_state().clone();slots,summary=matched_draw(fit,nrows);slots.to_parquet(folder/'draw.parquet',index=False)
        check('63403 compact exact same-role slots use private NumPy RNG',torch.equal(state,torch.get_rng_state()) and np.array_equal(fit.set_index('cache_ix').loc[slots.matched_old_cache_ix].role,nrows.role) and (folder/'draw.parquet').stat().st_size<50*2**20)
        g1=torch.Generator().manual_seed(20261021);g2=torch.Generator().manual_seed(20261021);o1=lp.epoch_order(INSTANCES,g1,'cpu');o2=lp.epoch_order(INSTANCES,g2,'cpu')
        check('Both arms identical641493-label epoch order157steps last2517',torch.equal(o1,o2) and len(o1)==INSTANCES and len(range(0,len(o1),BATCH))==STEPS and len(o1)%BATCH==2517)
        history=[dict(epoch=i,loss=.2,instances=INSTANCES,steps=STEPS,cumulative_steps=i*STEPS,old5_mean_f1=.95 if i==1 else .94) for i in range(1,11)]
        check('Earlystop evidence records157 best/actual steps',budget(history,1)['actual_steps']==1570 and budget(history,1)['best_steps']==157)
        old=[];c4=[];public=[]
        for pi,profile in enumerate(lp.PROFILES):
            for ri,role in enumerate(data.ROLES if profile!='natural' else data.ROLES[:-1]):
                for off in [0,512,1023]:old.append(dict(cache_ix=len(old),sample_ix=len(old),profile=profile,role=role,token_ix=13+off,body_offset=off,prompt_ix=pi*5+ri,message_ix=0,document_id=pi))
        for ri,role in enumerate(data.ROLES):
            for off in [0,512,1023]:c4.append(dict(cache_ix=len(c4),sample_ix=6_000_000_000_000+len(c4),profile='long-full',role=role,token_ix=13+off,body_offset=off,prompt_ix=ri,message_ix=0,document_id=6000000))
        for ds in ['oasst','toxicchat0124']:
            for role in ['system','user','assistant']:
                for off in range(4):public.append(dict(cache_ix=len(public),sample_ix=4_000_000_100_000+len(public),dataset=ds,role=role,token_ix=off,profile='public-natural',document_id=ds))
        rows=dict(old=pd.DataFrame(old),c4=pd.DataFrame(c4),public=pd.DataFrame(public));p={}
        for b,f in rows.items():
            value=np.full((len(f),5),.025,np.float32);value[np.arange(len(f)),probes.labels(f)]=.9;p[b]=value
        tau,proof=max33(rows,p);value=metrics_at(rows,p,tau)
        check('Inherited full33/F32 grid and11group9FPR exact',proof['groups']==11 and proof['primary_statistics']==proof['grid_thresholds']*33 and value['min33']==proof['min33']==1. and proof['global_argmax_equal'])
        check('Legacy truth CoT distinct from fresh public SUA',value['groups']['old-natural']['recall']['cot']==1 and all(value['groups'][g]['recall']['cot'] is None and value['groups'][g]['recall']['tool'] is None for g in PUBLIC))
        def v(n):return dict(min33=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in GROUPS})
        records=[dict(record=name,validation=v(.94 if name=='new-max33' else .92)) for name in RECORDS];zero={k:v(.92) for k in ['zero-oldmean','zero-max33']};legacy=v(.92)
        check('Ten records,4strict7guards plus originalGPU permit only new',select_records(records,legacy,zero)['promoted'] and len(GUARD_CONTROLS)==7 and len(STRICT_CONTROLS)==4)
        bad=copy.deepcopy(records);next(r for r in bad if r['record']=='new-max33')['validation']['min33']=.899999999999
        check('Absolute90 floor no numerical tolerance weakening',not select_records(bad,legacy,zero)['promoted'])
        badzero=copy.deepcopy(zero);badzero['zero-max33']['min33']=.94
        check('Same-path zero strict tie rejects candidate',not select_records(records,legacy,badzero)['promoted'])
        badzero=copy.deepcopy(zero);badzero['zero-oldmean']['groups'][PUBLIC[1]]['system_f1']=.99
        check('Fixed-tau zero public floor rejects drift-only gain',not select_records(records,legacy,badzero)['promoted'])
        diag=copy.deepcopy(records)
        for r in diag:
            if r['record'].startswith('naturalCE-'):r['validation']=v(.999)
        check('NaturalCE strong diagnostic is no hardfloor or fallback',select_records(diag,legacy,zero)['promoted'] and select_records(bad,legacy,zero)['candidate']=='new-max33')
        high=copy.deepcopy(legacy);high['groups']['old-natural']['system_f1']=.99
        check('Mandatory savedGPU floor independently enforced',not select_records(records,high,zero)['promoted'])
        fixture_records=[dict(record=name,validation=value) for name in RECORDS]
        check('Main120/zero24 rows preserve all controls outside trained records',len(flat_metrics(fixture_records))==120 and len(flat_metrics([dict(record=k,validation=value) for k in zero]))==24)
        fake=folder/'metadata-only';fakecache=folder/'cache';descriptor=dict(synthetic=True,draw_source_path=str(folder/'draw.parquet'),draw_slots_sha256=data.digest(folder/'draw.parquet'),draw_summary=summary)
        with patch.object(module,'OUT',fake),patch.object(module,'CACHE',fakecache),patch.object(module,'static_snapshot',return_value=descriptor):
            declaration=declare();check('Declaration only copies compact draw and freezes metadata',read(OUT/'protocol.json')==declaration and data.digest(CACHE/'matched-draw.parquet')==descriptor['draw_slots_sha256'])
            check('Restart declaration remains immutable',declare()==declaration)
            reject('Zero predictions require BOTH prior frozen histories',lambda:both_frozen(dict(signature={})))
        with patch.object(module,'OUT',folder/'undeclared'):reject('Missing declaration rejects before source/data/prob APIs',verify_declaration)
        check('Protocol JSON roundtrip stable and zero fixed-original threshold',json.loads(json.dumps(protocol()))==protocol() and protocol()['zero_controls']==['zero-oldmean','zero-max33'] and nf.ORIGINAL_TAU==.620969696969697)
        hp.require_hash(DESIGN,DESIGN_SHA,'root frozen prospective design');design=read(DESIGN)
        check('Frozen design records and4strict7guard exact',design['records']==RECORDS and design['promotion']['strict_controls']==STRICT_CONTROLS and design['promotion']['guards']==GUARD_CONTROLS)
    py_compile.compile(__file__,cfile=str(folder/'runner.pyc'),doraise=True);require(data.digest(__file__)==source,'Preflight altered source.')
    report=dict(source_sha256=source,design_sha256=DESIGN_SHA,checks=checks,checks_passed=len(checks),records=10,zero_controls=2,groups=11,primary_statistics=33,
        matched_manifest_rows=EXTRA,formal_model_or_probabilities_read=False,actual_banks_or_test_data_read=False,gpu_used=False,formal_outputs_written=False,fixture_directory=str(folder))
    data.atomic_json(folder/'report.json',report);print(json.dumps(dict(report=str(folder/'report.json'),report_sha256=data.digest(folder/'report.json'),**report)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--declare',action='store_true');a.add_argument('--train',action='store_true');a.add_argument('--preflight',action='store_true')
    p.add_argument('--worker',choices=ARMS,help=argparse.SUPPRESS);p.add_argument('--gpu',type=int,help=argparse.SUPPRESS);p.add_argument('--restart-incomplete',action='store_true');args=p.parse_args()
    if args.worker is not None:require(args.train and args.gpu is not None,'Worker requires --train and assigned GPU.');worker(args.worker,args.gpu)
    elif args.preflight:preflight()
    elif args.declare:declare()
    elif args.train:train(args.restart_incomplete)
    else:p.print_help()

if __name__=='__main__':main()
