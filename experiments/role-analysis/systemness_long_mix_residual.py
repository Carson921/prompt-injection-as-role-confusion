"""Constant-long-weight source replacement with a frozen binary residual.

Default help; --preflight is synthetic CPU/tmp only. --declare and --train are
explicit. Replace exactly84480 oldfull fit slots; matched repeats previous binary
NEW data. No LM, new sources, test reads, threshold search beyond frozen rules,
or primary change. Source-only implementation is reviewed before formal use.
"""
import argparse
from contextlib import contextmanager
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

OUT=data.runtime.OUT/'systemness-long-mix-residual'
CACHE=data.CACHE/'long-mix-residual'
DESIGN=data.runtime.OUT/'systemness-long-mix-residual-preparation/root-prospective-design.json'
DESIGN_SHA='e2fb7dfb369c35f137ab23900b9eb611a8f33d9db8d8686ca4ee2c0961d36698'
BINARY_SOURCE_SHA='fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9'
BINARY_PROTOCOL_SHA='84ebfc2dff0138dab829cb7756660b819e4d734823508c6219ec9de6c93cba1a'
PRIOR_AUDIT=br.OUT/'actual-audit/actual-development-audit.json'
PRIOR_AUDIT_SHA='2600d086c711e65bb320b70e66539fa775164cba8c192b84a2f9c52a75a9aa67'
EXTRACTION_OUT=data.runtime.OUT/'systemness-long-augmentation'
NEW_BANK_AUDIT=EXTRACTION_OUT/'extraction-audit/actual-bank-audit.json'
NEW_BANK_AUDIT_SHA='a2cf01d28f8b14fdba5e1b2a002d4fff8ac92aa521defdd27363a1d47de3889c'
NEW_EXTRACTION_SHA='ef266142d54ccd1ccd61c93fb08e7a8f7484702e874c3f508d7cac2ca958f6f7'
NEW_FEATURES_SHA='69e3db049f3de168c4c0922cfd1a935bb3bf19f91d697da75af2fe3752269cd8'
PRIOR_BINARY_SHA='269c9d7e7b04248de8ea03a44a2db5484d6c793d3cfd3557d5fc39702b982779'
PRIOR_BINARY_TAU=.7116969696969697
ARMS=['new','matched'];MEMBERS=list(nf.MEMBERS)
RECORDS=list(br.RECORDS)+['priorBinary-oldmean','priorBinary-max33']
GROUPS,FEASIBLE,PUBLIC,FULL,OLD5=nf.GROUPS,nf.FEASIBLE,nf.PUBLIC,nf.FULL,nf.OLD5
GUARD_CONTROLS=list(br.GUARD_CONTROLS)+['priorBinary-max33']
STRICT_CONTROLS=list(br.STRICT_CONTROLS)+['priorBinary-max33']
FIT,EXTRA,INSTANCES,BATCH,STEPS=nf.FIT,nf.EXTRA,nf.INSTANCES,nf.BATCH,nf.STEPS
BASE_SHA,BASE_PATH=br.BASE_SHA,br.BASE_PATH
NATURAL_CE_SHA,NATURAL_CE_TAU=br.NATURAL_CE_SHA,br.NATURAL_CE_TAU
ARTIFACTS=['protocol.json','search-plan.json','replacement-manifest.json','controls.json','zero-controls.json','replication.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt']
DEPENDENCIES=sorted(set(br.DEPENDENCIES+['systemness_long_mix_residual.py']))
require,read,array_sha=hp.require,hp.read,hp.array_digest
layout,validation_groups,metrics_at,max33,budget=nf.layout,nf.validation_groups,nf.metrics_at,nf.max33,nf.budget
# Pure math/portable routines retain frozen br source; no OUT/CACHE overrides.
state_hash,make_model,read_base,initial_hash=br.state_hash,br.make_model,br.read_base,br.initial_hash
save_model,load_model,predict_net,predict_raw,parity=br.save_model,br.load_model,br.predict_net,br.predict_raw,br.parity
probability_drift,verify_raw_evidence=br.probability_drift,br.verify_raw_evidence
LIMITATION=('One fixed residual seed and reused development validation. Constant long-row weight and matched role/half labels do not isolate document breadth: new66/old128 grids, source style, positions and document weights differ. '
    'Matched exactly repeats prior binary NEW data; CPU/GPU drift is technical evidence, not data-effect evidence. Gold-bound/old tests are not candidates. No primary change or goal completion.')

def protocol():
    value=br.protocol();value.update(experiment='constant-long-weight-source-replacement-residual',records=RECORDS,reused_binary_source_sha256=BINARY_SOURCE_SHA,
        replacement=dict(seed=20516086,draw_once=True,old_full_rows=168960,old_kept=84480,replaced=84480,new_fit_sources=256,new_per_source_role=66,old_per_source_role=128,
            role_half_buckets=10,per_bucket=8448,torch_rng_consumed=False,compact_manifest='replacement-slots.parquet'),
        matched='EXACT previous binary NEW data: original578090 order + identical63403 actual clean natural rows. No old-SUA draw.',
        labels='641493 formal labels identical by slot and each epoch permutation; role/half histogram equal, body offsets/doc weights differ.',
        replication='Full matched model arrays and old/C4/public p versus audited prior binary NEW269c; max p drift<5e-4, plus argmax/fixed original and prior thresholds.',
        limitation=LIMITATION)
    value['promotion']=dict(value['promotion'],strict_controls=STRICT_CONTROLS,guard_controls=GUARD_CONTROLS,priorBinary='max33 compulsory strict/guard; original oldmean diagnostic.')
    return value


def _replacement_manifest(fit,newfit,*,old_documents=132,new_documents=256,new_per_half=33):
    """Pure metadata selection; production wrapper fixes132/256/33 before declare.

    Each role/half is bijectively scattered; no Torch RNG or feature reads.
    Smaller document counts/new_per_half are solely for synthetic CPU fixtures.
    """
    require(fit.cache_ix.is_unique and fit.sample_ix.is_unique and fit.split.eq('fit').all(),'Original fit pool invalid/contaminated.')
    full=fit[fit.profile.eq('long-full') & fit.source_set.eq('long-full')].copy()
    require(full.document_id.nunique()==old_documents and len(full)==old_documents*5*256 and full.wrapper.eq('native').all(),'Original full pool ownership/count changed.')
    # Filter before deriving cache-local prompt offsets: other source_sets reuse IDs.
    full['old_body_offset']=full.token_ix-full.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    require(newfit.split.eq('fit').all() and newfit.profile.eq('long-full').all() and newfit.source_set.eq('long-augmentation').all()
        and newfit.wrapper.eq('native').all() and newfit.document_id.nunique()==new_documents and len(newfit)==new_documents*5*256
        and newfit.cache_ix.is_unique and newfit.sample_ix.is_unique and set(newfit.role)==set(data.ROLES),'New full fit pool ownership/count changed.')
    require(not set(full.group_id)&set(newfit.group_id) and not set(full.sample_ix)&set(newfit.sample_ix),'Original/new full source or sample overlap.')
    positions=pd.Series(np.arange(len(fit)),index=fit.cache_ix);full['fit_position']=full.cache_ix.map(positions)
    grid=data.uniform_indices(1024,256);removed=[];selected=[];kept=[]
    for frame,offset,target,choose,old in [(full,'old_body_offset',removed,64,True),(newfit,'body_offset',selected,new_per_half,False)]:
        require(frame.groupby('document_id').role.nunique().eq(5).all(),'Document missing formal role.')
        for (_,role),group in frame.groupby(['document_id','role'],sort=True):
            group=group.sort_values(offset);require(len(group)==256 and np.array_equal(group[offset].to_numpy(),grid),'Exact frozen256-position body grid changed.')
            for half in [0,1]:
                part=group[(group[offset]>=512) if half else (group[offset]<512)].copy();require(len(part)==128,'Body half support changed.')
                ix=data.uniform_indices(128,choose);chosen=part.iloc[ix].copy();chosen['half']=half
                if old:
                    kept.append(chosen);part=part.drop(chosen.index);part['half']=half;target.append(part)
                else:target.append(chosen)
    removed=pd.concat(removed);selected=pd.concat(selected);kept=pd.concat(kept)
    require(old_documents*64==new_documents*new_per_half,'Role/half replacement arithmetic is not equal.')
    rng=np.random.RandomState(20516086);parts=[];buckets={}
    for role in data.ROLES:
        for half in [0,1]:
            old=removed[removed.role.eq(role)&removed.half.eq(half)].sort_values('fit_position')
            new=selected[selected.role.eq(role)&selected.half.eq(half)].sort_values(['document_id','body_offset','sample_ix']);new=new.iloc[rng.permutation(len(new))]
            require(len(old)==len(new)==old_documents*64,'Unequal role/half bucket.')
            parts.append(pd.DataFrame(dict(fit_position=old.fit_position.to_numpy(),old_bank='old',old_cache_ix=old.cache_ix.to_numpy(),old_sample_ix=old.sample_ix.to_numpy(),
                old_document_id=old.document_id.to_numpy(),role=old.role.to_numpy(),old_body_offset=old.old_body_offset.to_numpy(),half=half,new_bank='c4',
                new_cache_ix=new.cache_ix.to_numpy(),new_sample_ix=new.sample_ix.to_numpy(),new_document_id=new.document_id.to_numpy(),new_role=new.role.to_numpy(),new_body_offset=new.body_offset.to_numpy())))
            buckets[role+':'+str(half)]=len(old)
    result=pd.concat(parts).sort_values('fit_position').reset_index(drop=True)
    require(result.fit_position.is_unique and result.old_cache_ix.is_unique and result.new_cache_ix.is_unique and result.new_sample_ix.is_unique
        and np.array_equal(result.role,result.new_role) and np.array_equal(result.half,result.new_body_offset.ge(512).astype(int))
        and np.array_equal(result.half,result.old_body_offset.ge(512).astype(int)),'Replacement lost bijection/formal role/half labels.')
    require(kept.groupby(['document_id','role']).size().eq(128).all() and selected.groupby(['document_id','role']).size().eq(2*new_per_half).all(),'Per-document density changed.')
    summary=dict(seed=20516086,draw_once=True,torch_rng_consumed=False,rows=len(result),old_pool_documents=old_documents,new_pool_documents=new_documents,
        kept_old_full_rows=len(kept),selected_new_full_rows=len(selected),total_full_rows=len(kept)+len(selected),role_half_buckets=buckets,
        fit_position_sha256=array_sha(result.fit_position.to_numpy()),old_cache_ix_sha256=array_sha(result.old_cache_ix.to_numpy()),new_cache_ix_sha256=array_sha(result.new_cache_ix.to_numpy()),
        old_sample_ix_sha256=array_sha(result.old_sample_ix.to_numpy()),new_sample_ix_sha256=array_sha(result.new_sample_ix.to_numpy()),original_fit_labels_sha256=array_sha(probes.labels(fit)),
        replacement_labels_sha256=array_sha(probes.labels(fit.iloc[result.fit_position.to_numpy()])),new_labels_sha256=array_sha(result.new_role.map(dict(zip(data.ROLES,range(5)))).to_numpy()),
        kept_old_sources=old_documents,selected_new_sources=new_documents,new_role_rows_per_document=2*new_per_half,old_role_rows_per_document=128,
        matched_actual_clean_natural=True,matched_previous_binary_arm='new',body_position_histogram_matched=False)
    require(summary['replacement_labels_sha256']==summary['new_labels_sha256'],'Per-slot role array differs.');return result,summary


def replacement_manifest(fit,newfit):
    require(len(fit)==FIT and len(newfit)==327680 and newfit.document_id.min()>=6000000 and newfit.sample_ix.min()>=6000000000000,'Production fit namespace/count changed.')
    return _replacement_manifest(fit,newfit)


def new_fit_rows():
    rows=pd.read_parquet(aug.NEW_BANK/'rows.parquet');return rows[rows.split.eq('fit')].copy().reset_index(drop=True)


def upstream_snapshot():
    """Closed explicit certificates/files: never call historical global verifiers."""
    hp.require_hash(DESIGN,DESIGN_SHA,'root constant-long-weight design');hp.require_hash(br.__file__,BINARY_SOURCE_SHA,'reused frozen binary math')
    hp.require_hash(br.OUT/'protocol.json',BINARY_PROTOCOL_SHA,'prior binary protocol');hp.require_hash(PRIOR_AUDIT,PRIOR_AUDIT_SHA,'prior binary actual audit')
    audit=read(PRIOR_AUDIT);complete=read(br.OUT/'complete.json');selection=read(br.OUT/'selection.json');decl=read(br.OUT/'protocol.json');prior=decl['static'];design=read(DESIGN)
    require(audit['stage']=='actual_completed_binary_residual_development_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False
        and audit['trainer_source_sha256']==BINARY_SOURCE_SHA and audit['trainer_protocol_sha256']==BINARY_PROTOCOL_SHA
        and audit['trainer_complete_sha256']==data.digest(br.OUT/'complete.json') and audit['trainer_selection_sha256']==data.digest(br.OUT/'selection.json')
        and audit['trainer_records_sha256']==data.digest(br.OUT/'records.json') and selection['promoted'] is False,'Prior audited binary negative trial changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(br.OUT/n,h,'prior binary artifact')
    for arm,h in complete['candidate_sha256'].items():hp.require_hash(br.OUT/'candidates'/f'{arm}.json',h,'prior binary candidate')
    candidate=read(br.OUT/'candidates/new.json');hp.require_hash(br.OUT/'models/new.npz',PRIOR_BINARY_SHA,'prior binary NEW control')
    require(candidate['model_sha256']==selection['model_sha256']==audit['model_sha256']['new']==PRIOR_BINARY_SHA
        and prior['original_partitions']==design['data']['original_partitions'] and prior['sample_ix_sha256']==design['data']['original_reference_source_sample_sha256'],'Prospective exact reference data/model changed.')
    hp.require_hash(NEW_BANK_AUDIT,NEW_BANK_AUDIT_SHA,'existing new-long complete bank audit');na=read(NEW_BANK_AUDIT)
    hp.require_hash(EXTRACTION_OUT/'extraction-complete.json',NEW_EXTRACTION_SHA,'existing new-long extraction');nc=read(EXTRACTION_OUT/'extraction-complete.json')
    require(na['stage']=='actual_completed_long_augmentation_extraction_audit' and na['audit_passed'] is True and nc['features_sha256']==NEW_FEATURES_SHA
        and nc['rows']==389120 and nc['dimensions']==512 and nc['all_finite'] is True and nc['all_rows_written'] is True,'Existing newlong bank not completely audited.')
    probability_paths=set(audit['probability_bindings']);closure={}
    # Raw banks/p are audit-attested here; actual feature hashes are checked in source_signature.
    for name,want in audit['provenance_sha256'].items():
        path=Path(name)
        if name not in probability_paths and path.suffix!='.f16':hp.require_hash(path,want,'closed previous certificate dependency');closure[name]=want
    for path in [NEW_BANK_AUDIT,EXTRACTION_OUT/'extraction-complete.json',PRIOR_AUDIT,DESIGN]:closure[str(path)]=data.digest(path)
    base,mean,scale,_=read_base();m,s=aug.read_scaler();require(np.array_equal(mean,m) and np.array_equal(scale,s),'Original scaler arrays changed.')
    refs=dict(prior['reference_probabilities']);refs['priorBinary']={b:dict(path=str(br.ppath('new',b)),sha256=candidate['probability_sha256'][b],sample_ix_sha256=prior['sample_ix_sha256'][b]) for b in ['old','c4','public']}
    require(all(audit['probability_bindings'][v['path']]==v['sha256'] for model in refs.values() for v in model.values()),'Inherited reference p not independently certified.')
    feature_sha256={str(path/'features.f16'):audit['provenance_sha256'][str(path/'features.f16')] for path in [hp.BANK,aug.NEW_BANK,nf.natural.CACHE]}
    require(feature_sha256[str(aug.NEW_BANK/'features.f16')]==NEW_FEATURES_SHA,'Newfit raw feature identity changed.')
    return dict(source_sha256=data.digest(__file__),design_sha256=DESIGN_SHA,reused_binary_source_sha256=BINARY_SOURCE_SHA,
        prior_actual_audit_sha256=PRIOR_AUDIT_SHA,new_bank_audit_sha256=NEW_BANK_AUDIT_SHA,closed_files_sha256=closure,feature_files_sha256=feature_sha256,
        dependencies_sha256={n:data.digest(data.HERE/n) for n in DEPENDENCIES},base_model_path=str(BASE_PATH),base_model_sha256=BASE_SHA,
        embedded_base_weight_sha256=state_hash(base),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),initial_weight_sha256=initial_hash(base),
        standardization_sha256=prior['standardization_sha256'],original_partitions=prior['original_partitions'],sample_ix_sha256=prior['sample_ix_sha256'],
        reference_probabilities=refs,prior_model_sha256=prior['prior']['model_sha256'],old_reference=prior['old_reference'],
        selected_head_ids=complete['signature']['selected_head_ids'],prior_binary_model_path=str(br.OUT/'models/new.npz'),prior_binary_model_sha256=PRIOR_BINARY_SHA,
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},validation_probability_arrays_read=False,global_inventory_scanned=False)


def static_snapshot():
    value=upstream_snapshot();manifest=read(OUT/'replacement-manifest.json');hp.require_hash(CACHE/'replacement-slots.parquet',manifest['slots_sha256'],'frozen replacement slots')
    require(manifest['slots_path']==str(CACHE/'replacement-slots.parquet') and manifest['rows']==84480 and manifest['total_full_rows']==168960
        and manifest['role_half_buckets']=={r+':'+str(h):8448 for r in data.ROLES for h in [0,1]},'Frozen replacement descriptor changed.')
    return value|dict(replacement_manifest_sha256=data.digest(OUT/'replacement-manifest.json'),replacement_slots_sha256=manifest['slots_sha256'])


def declare():
    if (OUT/'protocol.json').exists():return verify_declaration()
    require(not CACHE.exists() and not (OUT/'search-plan.json').exists(),'Unowned partial long-mix cache/plan; inspect explicitly.')
    upstream_snapshot();fit,_,_,_,_,_=layout();slots,summary=replacement_manifest(fit,new_fit_rows())
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True);slots.to_parquet(CACHE/'replacement-slots.parquet',index=False)
    data.atomic_json(OUT/'replacement-manifest.json',summary|dict(slots_path=str(CACHE/'replacement-slots.parquet'),slots_sha256=data.digest(CACHE/'replacement-slots.parquet')))
    value=dict(protocol=protocol(),static=static_snapshot());data.atomic_json(OUT/'protocol.json',value);return value


def verify_declaration():
    require((OUT/'protocol.json').is_file(),'Run --declare before slots/training/candidate p.')
    value=read(OUT/'protocol.json');require(value==dict(protocol=protocol(),static=static_snapshot()),'Frozen source/design/certificate/replacement changed.');return value


def source_signature():
    declaration=verify_declaration();static=declaration['static']
    for path,want in static['feature_files_sha256'].items():hp.require_hash(path,want,'certified raw features')
    return dict(source_sha256=data.digest(__file__),protocol_sha256=data.digest(OUT/'protocol.json'),static_sha256=aug.json_digest(static),
        selected_head_ids=static['selected_head_ids'],standardization_sha256=static['standardization_sha256'],base_model_sha256=BASE_SHA,
        embedded_base_weight_sha256=static['embedded_base_weight_sha256'],feature_files_sha256=static['feature_files_sha256'],replacement_slots_sha256=static['replacement_slots_sha256'])


def verified_replacement(fit):
    value=read(OUT/'replacement-manifest.json');hp.require_hash(CACHE/'replacement-slots.parquet',value['slots_sha256'],'replacement slots')
    got=pd.read_parquet(CACHE/'replacement-slots.parquet');expected,summary=replacement_manifest(fit,new_fit_rows());pd.testing.assert_frame_equal(got,expected,check_exact=True)
    require(all(value[k]==v for k,v in summary.items()),'Replacement descriptor does not reproduce exactly.');return got


def dataset():return nf.dataset()
def ppath(model,bank):return CACHE/f'{model}-{bank}-validation.npz'
def read_predictions(model,record,rows):return {b:aug.read_probability(ppath(model,b),record['probability_sha256'][b],frame) for b,frame in rows.items()}


def verify_plan(plan,full=True):
    value=verify_declaration();require(plan['protocol']==protocol() and plan['signature']['protocol_sha256']==data.digest(OUT/'protocol.json')
        and plan['initial_weight_sha256']==value['static']['initial_weight_sha256'],'Plan/protocol/full-state init changed.')
    if full:require(plan['signature']==source_signature(),'Features/rows/data/source signature changed.')
    return value


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


def supervisor_alive():require(os.environ.get('ROLE_LONG_MIX_SUPERVISOR_PID')==str(os.getppid()),'Worker requires original live supervisor lease.')


def worker(arm,gpu):
    supervisor_alive();require(arm in ARMS and gpu in [0,1],'Only declared two arm workers allowed.')
    require(not (OUT/'complete.json').exists() and not (OUT/'candidates'/f'{arm}.json').exists(),'Frozen candidate cannot be overwritten.')
    no_competing_jobs(True);plan=read(OUT/'search-plan.json');verify_plan(plan);ms.configure_torch()
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1 and 'A40' in torch.cuda.get_device_name(0),'Worker requires its assigned A40.')
    banks,fit,ov,cv,nrows,pv,_=dataset();base,mean,scale,_=read_base();slots=verified_replacement(fit)
    def standardized(bank,indices):
        out=torch.empty((len(indices),512),dtype=torch.float32,device='cuda')
        for start in range(0,len(indices),BATCH):
            x=(np.asarray(banks[bank][indices[start:start+BATCH]],np.float32)-mean)/scale
            require(np.isfinite(x).all(),'Nonfinite standardized feature.');out[start:start+len(x)]=torch.from_numpy(x).cuda()
        return out
    require(torch.cuda.mem_get_info(0)[0]>(INSTANCES+173032)*512*4+2*2**30,'Insufficient assigned worker memory.')
    old_x=standardized('old',fit.cache_ix.to_numpy())
    if arm=='new':old_x[torch.from_numpy(slots.fit_position.to_numpy()).cuda()]=standardized('c4',slots.new_cache_ix.to_numpy())
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


def flat_metrics(records):
    flat=[]
    for record in records:
        for group,value in dict(record['validation']['groups'],pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=record['record'],group=group,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))},**{'role_recall_'+r:v for r,v in value['recall'].items()}))
    return flat


def no_competing_jobs(allow_own=False):
    nf.no_competing_jobs(True);names={'systemness_natural_fit_probes.py','systemness_binary_residual.py'}
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
        except BlockingIOError as e:raise RuntimeError('Another long-mix supervisor owns output.') from e
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run_workers(pending):
    require(len(pending)<=2 and len(set(pending))==len(pending) and set(pending)<=set(ARMS),'Only two fixed arm workers allowed.');jobs=[]
    try:
        for gpu,arm in enumerate(pending):
            log=(OUT/'logs'/f'{arm}.log').open('a')
            try:p=subprocess.Popen([sys.executable,'-u',__file__,'--train','--worker',arm,'--gpu',str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ROLE_LONG_MIX_SUPERVISOR_PID=str(os.getpid())),stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            jobs.append((arm,p,log,gpu))
        data.atomic_json(OUT/'status.json',dict(stage='training',workers=[dict(arm=a,pid=p.pid,gpu=g) for a,p,_,g in jobs]))
        while jobs:
            codes=[(a,p.poll()) for a,p,_,_ in jobs];require(not [(a,c) for a,c in codes if c not in [None,0]],'Long-mix worker failed: '+str(codes))
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
            require({p.name for p in CACHE.iterdir()}=={'replacement-slots.parquet'} and all(not any((OUT/n).iterdir()) for n in ['models','training','candidates']),'Unowned models/probabilities predate plan.')
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


def inherited_controls(plan,rows,allow_create=False):
    static=verify_plan(plan,False)['static'];p={}
    for model,bindings in static['reference_probabilities'].items():p[model]={b:aug.read_probability(v['path'],v['sha256'],rows[b]) for b,v in bindings.items()}
    previous=read(br.OUT/'controls.json');legacy=previous['original_savedGPU_validation'];ref=static['old_reference']
    gpu={b:aug.read_probability(ref[k+'_path'],ref[k+'_sha256'],rows[b]) for b,k in [('old','old'),('c4','new')]}
    require(metrics_at(rows,dict(gpu,public=p['oldhead']['public']),nf.ORIGINAL_TAU)==legacy,'SavedGPU floors changed.')
    content=dict(signature=plan['signature'],prior_actual_audit_sha256=PRIOR_AUDIT_SHA,new_bank_audit_sha256=NEW_BANK_AUDIT_SHA,reference_probabilities=static['reference_probabilities'],
        physical_model_sha256=dict(oldhead=BASE_SHA,ensemble=static['prior_model_sha256'],naturalCE=NATURAL_CE_SHA,priorBinary=PRIOR_BINARY_SHA),
        original_savedGPU_validation=legacy,reference_probabilities_recomputed=False,naturalCE_diagnostic_only=True,priorBinary_max33_is_strict_and_guard=True,tests_read=False)
    path=OUT/'controls.json'
    if path.exists():require(read(path)==content,'Inherited explicit reference closure changed.')
    else:require(allow_create,'Missing inherited controls in read-only route.');data.atomic_json(path,content)
    return p,legacy


def replication(plan,p,allow_create=False):
    """Matched repeats previous binary NEW data; numerical drift is no data gain."""
    static=read(OUT/'protocol.json')['static'];oldpath=Path(static['prior_binary_model_path']);hp.require_hash(oldpath,PRIOR_BINARY_SHA,'prior binary NEW model')
    current,_,_,_=load_model(OUT/'models/matched.npz');previous,_,_,_=load_model(oldpath);weights={}
    for key,value in current.state_dict().items():
        a=value.detach().numpy();b=previous.state_dict()[key].detach().numpy();require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Replication model geometry changed.')
        d=np.abs(a.astype(np.float64)-b.astype(np.float64));weights[key]=dict(array_bitexact=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),maximum_absolute_difference=float(d.max()),mean_absolute_difference=float(d.mean()),current_sha256=array_sha(a),prior_sha256=array_sha(b))
    drift={b:dict(original_oldhead_tau=probability_drift(p['priorBinary'][b],p['matched'][b],nf.ORIGINAL_TAU),prior_binary_original_tau=probability_drift(p['priorBinary'][b],p['matched'][b],PRIOR_BINARY_TAU)) for b in ['old','c4','public']}
    require(all(0<=v['original_oldhead_tau']['maximum_probability_difference']<5e-4 for v in drift.values()),'Matched repeat differs from frozen prior binary beyond5e-4; do not infer a data effect.')
    current_training=read(OUT/'training/matched.json');old_training=read(br.OUT/'training/new.json')
    content=dict(signature=plan['signature'],matched_data='EXACT previous binary NEW original578090+actual63403 natural, same order and labels.',prior_model_sha256=PRIOR_BINARY_SHA,
        current_model_sha256=data.digest(OUT/'models/matched.npz'),prior_training_sha256=data.digest(br.OUT/'training/new.json'),current_training_sha256=data.digest(OUT/'training/matched.json'),
        weights=weights,all_weight_arrays_bitexact=all(v['array_bitexact'] for v in weights.values()),probability_drift=drift,
        all_probability_arrays_bitexact=all(np.array_equal(p['matched'][b].view(np.uint32),p['priorBinary'][b].view(np.uint32)) for b in ['old','c4','public']),
        budget_comparison={name:{k:v[k] for k in ['best_epoch','best_steps','epochs','actual_steps','actual_instances']} for name,v in [('current',current_training),('prior',old_training)]},
        technical_tolerance=5e-4,numerical_drift_is_not_source_effect=True,tests_read=False)
    path=OUT/'replication.json'
    if path.exists():require(read(path)==content,'Matched replication evidence changed.')
    else:require(allow_create,'Missing replication evidence in read-only verification.');data.atomic_json(path,content)
    return content


def select_records(records,legacy,zero):
    require(len(records)==12 and {r['record'] for r in records}==set(RECORDS) and set(zero)=={'zero-oldmean','zero-max33'},'Twelve records and two separate zero controls required.')
    index={r['record']:r for r in records};candidate=index['new-max33']['validation'];minimum=candidate['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARD_CONTROLS:
        control=zero[name] if name in zero else index[name]['validation'];g=nf.guard(candidate,control);guards[name]=g;reasons += [name+':'+r for r in g['rejection_reasons']]
        if name in STRICT_CONTROLS:
            gain=minimum-control['min33'] if minimum is not None and control['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    gpu=nf.guard(candidate,legacy,True);guards['original-savedGPU']=gpu;reasons += ['original-savedGPU:'+r for r in gpu['rejection_reasons']]
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,
        naturalCE_is_diagnostic_only=True,priorBinary_max33_is_strict_and_guard=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def calibrate(plan,allow_create=False):
    ms.configure_torch();banks,fit,ov,cv,nrows,pv,_=dataset();verified_replacement(fit);rows=dict(old=ov,c4=cv,public=pv);p={}
    for arm in ARMS:_,p[arm]=candidate_record(arm,plan,rows,nrows)
    inherited,legacy=inherited_controls(plan,rows,allow_create);p.update(inherited);replication(plan,p,allow_create)
    zero=zero_controls(plan,banks,rows,nrows,p['oldhead'],allow_create);records=[]
    for name in RECORDS:
        model,objective=name.split('-');value=p[model]
        if objective=='max33':tau,proof=max33(rows,value)
        elif model=='oldhead':tau,proof=nf.ORIGINAL_TAU,None
        elif model=='ensemble':tau,proof=nf.ENSEMBLE_TAU,None
        elif model=='naturalCE':tau,proof=NATURAL_CE_TAU,None
        elif model=='priorBinary':tau,proof=PRIOR_BINARY_TAU,None
        else:tau,proof=lp.validation_metrics(ov,value['old'])['system_threshold'],None
        result=metrics_at(rows,value,tau)
        if proof is not None:require(result['min33']==proof['min33'] and result['old5_mean_f1']==proof['old5_mean_f1'],'Max33 selected metrics differ.')
        if name=='naturalCE-max33':require(tau==.6008080808080808,'Inherited CE threshold changed.')
        if name=='priorBinary-max33':require(tau==.5453636363636364 and result['min33']==.8990885416666666,'Frozen prior binary calibration no longer reproduces.')
        records.append(dict(record=name,model=model,objective=objective,validation=result,full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False))
    return records,legacy,zero,rows,fit,nrows


def publish(plan):
    records,legacy,zero,rows,fit,nrows=calibrate(plan,True);selection=select_records(records,legacy,zero);data.atomic_json(OUT/'records.json',records)
    flat=flat_metrics(records);zf=flat_metrics([dict(record=k,validation=v) for k,v in zero.items()]);require(len(flat)==144 and len(zf)==24,'132+12 main/22+2 zero CSV coverage required.')
    pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False);pd.DataFrame(zf).to_csv(OUT/'zero-control-metrics.csv',index=False)
    summary=read(OUT/'replacement-manifest.json');counts=read(br.OUT/'source-counts.json')|dict(experiment='constant-long-weight-source-replacement-residual',same_existing_rows=False,
        same_validation_rows=True,old_full_rows=168960,kept_old_full_rows=84480,new_full_rows=84480,total_full_fraction=168960/INSTANCES,
        new_arm_bank_rows=dict(old=493610,c4=84480,natural=EXTRA),matched_arm_bank_rows=dict(old=FIT,c4=0,natural=EXTRA),replacement=summary,limitation=LIMITATION)
    data.atomic_json(OUT/'source-counts.json',counts)
    lines=['Constant-long-weight source replacement: reused development only',LIMITATION,'','Record threshold min33 old5meanF1']
    lines += [f"{r['record']} {r['validation']['system_threshold']:.12g} {r['validation']['min33']} {r['validation']['old5_mean_f1']:.9f}" for r in records]
    lines += ['','Same-path zero numerical controls (not12 records)']+[f"{k} {v['system_threshold']:.12g} {v['min33']} {v['old5_mean_f1']:.9f}" for k,v in zero.items()]
    lines += ['','Promoted: '+str(selection['promoted']),'Reasons: '+json.dumps(selection['rejection_reasons'])];(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    selection.update(source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'search-plan.json'),records_sha256=data.digest(OUT/'records.json'),model_path=str(OUT/'models/new.npz'),
        model_sha256=read(OUT/'candidates/new.json')['model_sha256'],controls_sha256=data.digest(OUT/'controls.json'),zero_controls_sha256=data.digest(OUT/'zero-controls.json'),
        replication_sha256=data.digest(OUT/'replication.json'),system_threshold=next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'))
    data.atomic_json(OUT/'selection.json',selection);verify_plan(plan)
    require(all(p.stat().st_size<50*2**20 for folder in [OUT,CACHE] for p in folder.rglob('*') if p.is_file()),'Artifact exceeds50MiB storage bound.')
    data.atomic_json(OUT/'complete.json',dict(signature=plan['signature'],source_sha256=data.digest(__file__),artifact_sha256={n:data.digest(OUT/n) for n in ARTIFACTS},
        candidate_sha256={a:data.digest(OUT/'candidates'/f'{a}.json') for a in ARMS},tests_read=False,primary_changed=False,new_independent_confirmation_required=True))
    data.atomic_json(OUT/'status.json',dict(stage='validation_complete',promoted=selection['promoted'],primary_changed=False))


def verified_completion():
    require((OUT/'complete.json').is_file() and read(OUT/'status.json')['stage']=='validation_complete','Long-mix residual trial not terminal complete.')
    plan=read(OUT/'search-plan.json');verify_plan(plan);complete=read(OUT/'complete.json')
    require(complete['signature']==plan['signature'] and complete['source_sha256']==data.digest(__file__) and complete['tests_read'] is False
        and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and set(complete['artifact_sha256'])==set(ARTIFACTS) and set(complete['candidate_sha256'])==set(ARMS),'Completion signature/source/coverage changed.')
    for n,h in complete['artifact_sha256'].items():hp.require_hash(OUT/n,h,'completed artifact')
    for a,h in complete['candidate_sha256'].items():hp.require_hash(OUT/'candidates'/f'{a}.json',h,'completed candidate')
    records,legacy,zero,*_=calibrate(plan,False);require(records==read(OUT/'records.json'),'Twelve saved records do not reproduce from frozen p.')
    expected=select_records(records,legacy,zero);actual=read(OUT/'selection.json');require(all(actual[k]==v for k,v in expected.items()),'Five strict/eight guard/savedGPU/absolute90 gate changed.')
    require(actual['source_sha256']==data.digest(__file__) and actual['plan_sha256']==data.digest(OUT/'search-plan.json') and actual['records_sha256']==data.digest(OUT/'records.json')
        and actual['replication_sha256']==data.digest(OUT/'replication.json') and actual['controls_sha256']==data.digest(OUT/'controls.json') and actual['zero_controls_sha256']==data.digest(OUT/'zero-controls.json')
        and actual['model_path']==str(OUT/'models/new.npz') and actual['model_sha256']==read(OUT/'candidates/new.json')['model_sha256']
        and actual['system_threshold']==next(r['validation']['system_threshold'] for r in records if r['record']=='new-max33'),'Selected model/threshold/control closure changed.')
    return complete


def preflight():
    """Small synthetic metadata/model fixtures only; no formal slots or arrays."""
    import ast,copy,py_compile,torch
    from unittest.mock import patch
    folder=Path(tempfile.mkdtemp(prefix='systemness-long-mix-residual-preflight-',dir='/tmp'));source=data.digest(__file__);checks=[];module=sys.modules[__name__]
    def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,AssertionError,KeyError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('CPU fault accepted: '+name)
    def forbidden(*a,**k):raise AssertionError('Formal slots/bank/model/p/global verifier read in synthetic preflight.')
    original_br_paths=(br.OUT,br.CACHE);torch.set_num_threads(2);torch.set_float32_matmul_precision('medium');ms.configure_torch()
    # Full256-point geometry, only2old/4new synthetic documents.
    def frame(documents,old):
        out=[];grid=data.uniform_indices(1024,256)
        for doc in range(documents):
            for ri,role in enumerate(data.ROLES):
                for off in grid:
                    out.append(dict(cache_ix=len(out),sample_ix=(100000 if old else 6000000000000)+len(out),document_id=(100 if old else 6000000)+doc,
                        group_id=('old-' if old else 'new-')+str(doc),source_set='long-full' if old else 'long-augmentation',profile='long-full',role=role,
                        split='fit',wrapper='native',prompt_ix=doc*5+ri,message_ix=0,token_ix=17+int(off),body_offset=int(off)))
        return pd.DataFrame(out)
    fit=frame(2,True);newfit=frame(4,False)
    # Deliberate other-profile collision in local prompt IDs must be filtered first.
    distract=fit.iloc[:100].copy();distract.cache_ix+=10000;distract.sample_ix+=10000;distract.source_set='main';distract.profile='single';distract.token_ix+=100
    fit=pd.concat([fit,distract],ignore_index=True)
    with patch.object(nf,'dataset',side_effect=forbidden),patch.object(br,'source_signature',side_effect=forbidden),patch.object(nf,'source_signature',side_effect=forbidden),patch.object(data.runtime,'load_model',side_effect=forbidden):
        state=torch.get_rng_state().clone();npstate=np.random.get_state();slots,summary=_replacement_manifest(fit,newfit,old_documents=2,new_documents=4,new_per_half=32)
        again,summary2=_replacement_manifest(fit,newfit,old_documents=2,new_documents=4,new_per_half=32)
        check('Private deterministic scatter consumes neither Torch nor global NumPy RNG',slots.equals(again) and summary==summary2 and torch.equal(state,torch.get_rng_state()) and np.array_equal(npstate[1],np.random.get_state()[1]))
        check('Ten role/half buckets have exact128 slot bijections',summary['role_half_buckets']=={r+':'+str(h):128 for r in data.ROLES for h in [0,1]} and len(slots)==1280 and slots.new_cache_ix.is_unique)
        check('Kept and replaced full-row weight remains constant',summary['kept_old_full_rows']==summary['selected_new_full_rows']==1280 and summary['total_full_rows']==2560 and summary['old_role_rows_per_document']==128 and summary['new_role_rows_per_document']==64)
        check('Production132x64 equals256x33 and total641493 budget',132*64==256*33==8448 and 8448*10==84480 and 409130+84480+84480+EXTRA==INSTANCES and STEPS==157 and INSTANCES%BATCH==2517)
        check('New old/c4/natural bank contributions493610/84480/63403',FIT-84480==493610 and FIT-84480+84480+EXTRA==INSTANCES)
        check('Per-slot labels and half unchanged with broader sources',np.array_equal(fit.iloc[slots.fit_position].role,slots.new_role) and np.array_equal(slots.half,slots.old_body_offset.ge(512).astype(int)) and np.array_equal(slots.half,slots.new_body_offset.ge(512).astype(int)))
        check('Other source/profile prompt collisions do not corrupt offsets',slots.old_cache_ix.max()<10000 and slots.old_body_offset.between(0,1023).all())
        labels=probes.labels(fit);newlabels=labels.copy();newlabels[slots.fit_position]=slots.new_role.map(dict(zip(data.ROLES,range(5)))).to_numpy()
        g1=torch.Generator().manual_seed(20261021);g2=torch.Generator().manual_seed(20261021);p1=lp.epoch_order(len(fit),g1,'cpu');p2=lp.epoch_order(len(fit),g2,'cpu')
        check('Matched prefix exactly originalfit, labels identical under all shared permutations',np.array_equal(labels,newlabels) and torch.equal(p1,p2) and np.array_equal(labels[p1.numpy()],newlabels[p2.numpy()]))
        malformed=newfit.copy();malformed.loc[0,'split']='validation';reject('New validation source row cannot enter replacement',lambda:_replacement_manifest(fit,malformed,old_documents=2,new_documents=4,new_per_half=32))
        malformed=fit.copy();malformed.loc[5,'token_ix']+=1;reject('Corrupt old uniform body grid refused',lambda:_replacement_manifest(malformed,newfit,old_documents=2,new_documents=4,new_per_half=32))
        malformed=newfit.copy();malformed.loc[5,'body_offset']+=1;reject('Corrupt new uniform body grid refused',lambda:_replacement_manifest(fit,malformed,old_documents=2,new_documents=4,new_per_half=32))
        reject('Production wrapper refuses tiny or foreign namespaces',lambda:replacement_manifest(fit,newfit))
        compact=folder/'replacement.parquet';slots.to_parquet(compact,index=False)
        check('Compact exact row manifest roundtrip without large JSON',pd.read_parquet(compact).equals(slots) and compact.stat().st_size<50*2**20 and len(json.dumps(summary))<10000)
        check('Frozen model/portable/prediction functions reused without namespace overrides',make_model is br.make_model and load_model is br.load_model and predict_raw is br.predict_raw and (br.OUT,br.CACHE)==original_br_paths)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3);base={n:v.detach().numpy().copy() for n,v in hp.make_net('mlp',hidden=128).state_dict().items()}
        mean=np.zeros(512,np.float32);scale=np.ones(512,np.float32);net=make_model(base);net.train()
        check('Reused correction zero init, original base always eval/frozen',not net.base.training and not any(v.requires_grad for v in net.base.parameters()) and torch.count_nonzero(net.correction[-1].weight)==0 and torch.count_nonzero(net.correction[-1].bias)==0)
        fake=folder/'own-output';fake.mkdir();static=dict(embedded_base_weight_sha256=state_hash(base),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale));data.atomic_json(fake/'protocol.json',dict(static=static));data.atomic_json(fake/'search-plan.json',dict(synthetic=True))
        plan=dict(signature=dict(selected_head_ids=[458,462,2045,460,459,2047,456,997],standardization_sha256='synthetic'),initial_weight_sha256=initial_hash(base))
        with patch.object(module,'OUT',fake),patch.object(module,'read_base',return_value=(base,mean,scale,{})):
            metadata=portable_metadata(plan,'new');path=folder/'portable.npz';save_model(path,net,mean,scale,metadata);loaded,_,_,m=verify_portable(path,plan,'new')
            check('Own source/plan metadata carried by reused portable model',m['source_sha256']==source and m['arm']=='new' and m['initial_weight_sha256']==plan['initial_weight_sha256'] and not loaded.base.training and br.OUT==original_br_paths[0])
        def v(n):return dict(min33=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in GROUPS})
        records=[dict(record=n,validation=v(.94 if n=='new-max33' else .92)) for n in RECORDS];zero={n:v(.92) for n in ['zero-oldmean','zero-max33']};legacy=v(.92)
        check('12records+zero2 with5strict8guards and savedGPU permit only new',len(RECORDS)==12 and len(GUARD_CONTROLS)==8 and len(STRICT_CONTROLS)==5 and select_records(records,legacy,zero)['promoted'])
        bad=copy.deepcopy(records);next(r for r in bad if r['record']=='priorBinary-max33')['validation']['min33']=.94
        check('Frozen priorBinary min33 strict tie rejects new',not select_records(bad,legacy,zero)['promoted'])
        bad=copy.deepcopy(records);next(r for r in bad if r['record']=='priorBinary-max33')['validation']['groups']['old-long-full-tail']['system_f1']=.99
        check('Frozen priorBinary full-tail guard cannot be bypassed',not select_records(bad,legacy,zero)['promoted'])
        bad=copy.deepcopy(records);next(r for r in bad if r['record']=='new-max33')['validation']['min33']=.9-1e-13
        check('Absolute90 and no fallback unchanged',not select_records(bad,legacy,zero)['promoted'] and select_records(bad,legacy,zero)['candidate']=='new-max33')
        # CSV count helper needs actual metric schema, but no grid or real rows.
        val=dict(groups={g:dict(system_f1=.92,recall={r:None for r in data.ROLES}) for g in GROUPS},pooled_descriptive_only=dict(system_f1=.92,recall={r:None for r in data.ROLES}))
        check('144main+24zero equals168 CSV rows',len(flat_metrics([dict(record=n,validation=val) for n in RECORDS]))==144 and len(flat_metrics([dict(record=n,validation=val) for n in zero]))==24)
        tree=ast.parse(Path(__file__).read_text());closed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='upstream_snapshot')
        check('Closed declaration never calls global source/completion/inventory verifiers',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'source_signature','verified_completion','frozen_snapshot','exclusion_inputs','rglob'} for n in ast.walk(closed)))
        with patch.object(module,'OUT',folder/'undeclared'),patch.object(module,'upstream_snapshot',side_effect=forbidden):reject('Declaration gate before formal slots/metadata APIs',verify_declaration)
        check('Protocol JSON roundtrip and frozen final e2fb design match',json.loads(json.dumps(protocol()))==protocol())
        hp.require_hash(DESIGN,DESIGN_SHA,'root final long-mix design');design=read(DESIGN)
        check('Hardpin matches12records5strict8guards and original math',design['records']==RECORDS and design['promotion']['strict_controls']==STRICT_CONTROLS and design['promotion']['guards']==GUARD_CONTROLS and design['model']['reused_implementation_sha256']==BINARY_SOURCE_SHA)
        hp.require_hash(br.__file__,BINARY_SOURCE_SHA,'original binary source unchanged');check('Original binary namespace unchanged',(br.OUT,br.CACHE)==original_br_paths)
        hp.require_hash(NEW_BANK_AUDIT,NEW_BANK_AUDIT_SHA,'existing extraction audit certificate path')
        hp.require_hash(EXTRACTION_OUT/'extraction-complete.json',NEW_EXTRACTION_SHA,'existing extraction completion certificate path')
        check('Actual existing extraction certificate paths exist and SHA match without p or slots reads')
    py_compile.compile(__file__,cfile=str(folder/'runner.pyc'),doraise=True);require(data.digest(__file__)==source,'Preflight altered source.')
    report=dict(source_sha256=source,design_sha256=DESIGN_SHA,checks=checks,checks_passed=len(checks),records=12,zero_controls=2,CSV_rows=168,
        synthetic_replacement_rows=len(slots),synthetic_manifest_bytes=compact.stat().st_size,existing_extraction_certificates_checked=True,actual_slots_selected=False,actual_models_or_probabilities_read=False,
        actual_banks_or_test_data_read=False,gpu_used=False,formal_outputs_written=False,fixture_directory=str(folder))
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
