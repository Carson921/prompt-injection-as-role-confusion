"""Conditional bounded system-logit correction; development fit/validation only.

Default: help. --preflight writes CPU fixtures only under /tmp. --train is gated
on the completed, unpromoted four-head search and absent head-confirmation OUT.
No language-model weights, old primary, test data or frozen source is changed.
"""
import argparse
from contextlib import contextmanager
import hashlib
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

import systemness_attention as attention
import systemness_data as data
import systemness_head_probes as heads
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-head-fusion'
CACHE = data.CACHE / 'head-fusion'
HEAD_CONFIRM_OUT = data.runtime.OUT / 'systemness-head-confirmation'
DESIGN = data.runtime.OUT / 'systemness-fusion-preparation/conditional-design-v1.json'
DESIGN_SHA256 = '399a547177752f8aa53adbf542569c8b88d1d568bc8bf51963bdb2d8e4eeadc2'
ROWS, FIT_ROWS, VALID_ROWS = 755282, 578090, 173032
HEAD_DIM, BASE_DIM, CLASSES = 512, 8640, 5
BATCH_SIZE, STEPS, BOUND, TOLERANCE = 4096, 142, 4., 5e-4
CONFIGS = {'delta-linear': dict(hidden=0, dropout=0.),
           'delta-mlp32': dict(hidden=32, activation='GELU', dropout=.1)}
DEPENDENCIES = sorted(set(heads.DEPENDENCIES + ['systemness_head_fusion.py',
    'systemness_attention.py', 'run_local_gptoss120b.py']))
LIMITATION = ('Development validation is reused. Both banks have identical rows but different LM extraction batches; '
    'this is not an isolated causal head ablation. Only the system logit changes. The other four raw logits, '
    'conditional probabilities and their internal argmax ordering remain unchanged; absolute probabilities, '
    'global argmax and CoT recall can change. Five-class CE has the same correction gradient as binary '
    'system-vs-nonsystem CE; the conditional non-system class loss is constant. A positive result needs '
    'new independent confirmation. Delta-zero numerical controls are not a third trained candidate.')
require = long_data.require
read = heads.read
array_digest = heads.array_digest
atomic_npz = heads.atomic_npz
validate_probabilities = heads.validate_probabilities


def check_design():
    heads.require_hash(DESIGN, DESIGN_SHA256, 'pre-implementation design')
    value = read(DESIGN)
    require(value['candidates'] == CONFIGS and value['delta'] ==
        '4*tanh(g(h)/4); final weight+bias zero-initialized; correction input=head512 only; five base logits never correction inputs.',
        'Frozen correction architecture differs from the pre-implementation design.')
    expected = dict(seed=data.SEED, fit_rows=FIT_ROWS, validation_rows=VALID_ROWS, optimizer='AdamW',
        learning_rate=.001, weight_decay=.001, batch_size=BATCH_SIZE, updates_per_epoch=STEPS,
        last_batch_instances=554, max_epochs=45, min_epochs=10, patience=8, gradient_clip=5.,
        loss='Full five-class mean CE', maximum_workers=2)
    require(all(value['training'].get(k) == v for k,v in expected.items()), 'Frozen training budget differs from design.')
    require(value['base_cache']['shape'] == [ROWS,CLASSES] and value['base_cache']['dtype'] == 'float32'
        and value['base_cache']['chunk_size'] == BATCH_SIZE and value['base_cache']['cache_bytes'] == ROWS*CLASSES*4
        and value['zero_control']['full_validation_rows'] == VALID_ROWS
        and value['zero_control']['probability_max_abs_difference_limit'] == TOLERANCE
        and value['zero_control']['third_trained_candidate'] is False
        and value['selection']['no_test_tuning'] is True, 'Frozen cache/control/selection design differs.')
    for name,sha in value['frozen_upstream_sources_sha256'].items():
        heads.require_hash(data.HERE/name,sha,'design upstream source')
    return value


def require_fallback(selection, completion=None):
    require(not HEAD_CONFIRM_OUT.exists(), 'Head-confirmation OUT exists; fallback training is forbidden.')
    require(selection.get('promoted') is False and selection.get('test_used_for_selection') is False
        and selection.get('primary_changed') is False and selection.get('confirmation_required') is True,
        'Fallback requires a strictly unpromoted validation-only head search.')
    if completion is not None:
        require(set(completion.get('candidate_sha256',{})) == set(heads.CONFIGS)
            and completion.get('tests_read') is False and completion.get('primary_changed') is False,
            'Fallback requires all four completed head candidates.')


def fallback_gate():
    """Metadata gate before any fusion directory or feature-bank access."""
    check_design()
    selection = read(heads.OUT/'selection.json'); completion = read(heads.OUT/'complete.json')
    require_fallback(selection, completion)
    heads.require_hash(heads.OUT/'selection.json', completion['selection_sha256'], 'completed head selection')
    for key,sha in completion['candidate_sha256'].items():
        path=heads.OUT/'candidates'/f'{key}.json'; heads.require_hash(path,sha,'completed head candidate')
        record=read(path)
        heads.require_hash(heads.OUT/'models'/f'{key}.npz',record['model_sha256'],'head candidate model')
        heads.require_hash(heads.CACHE/f'{key}-validation.npz',record['validation_probabilities_sha256'],'head validation')
    verified=heads.verified_completion()
    require_fallback(read(heads.OUT/'selection.json'),verified)
    return verified


def protocol():
    return dict(version=1,design_path=str(DESIGN),design_sha256=DESIGN_SHA256,candidates=CONFIGS,
        delta_bound=BOUND,delta='4*tanh(g(head512)/4)',zero_final_layer=True,base_dimensions=BASE_DIM,
        head_dimensions=HEAD_DIM,roles=data.ROLES,head_scaler='Byte-copy completed head-probes densefit-only scaler.',
        baseline='Frozen dense-mlp128 raw logits, with its embedded original8640 scaler; only system column receives delta.',
        cache=dict(rows=ROWS,classes=5,dtype='float32',bytes=ROWS*5*4,chunk=BATCH_SIZE,
            progress='Atomic NaN initialization; every16 chunks flush/fsync and exact completed-prefix hashes.'),
        zero_control='Full173032 frozen-p parity and decisions; original-threshold and same-helper-retuned metrics, not a candidate.',
        training=dict(seed=data.SEED,fit_rows=FIT_ROWS,validation_rows=VALID_ROWS,batch_size=BATCH_SIZE,
            steps=STEPS,last_batch_instances=554,optimizer='AdamW',lr=.001,weight_decay=.001,
            clip=5.,max_epochs=45,min_epochs=10,patience=8,loss='Full five-class mean CE',workers=2),
        guards='Against original dense AND delta0-cache-retuned: original4 F1 loss<=.02, accuracy loss<=.01, full/tail F1 nondecrease,5profile FPR<=.05; tolerance1e-12.',
        promotion='Maximum mean F1 among dual-guard eligible candidates, strictly above max(two controls)+1e-12; otherwise diagnostic only.',
        portable='Raw8640+raw512; hash-bound frozen base; full173032 zero and nonzero checks, independent CPU check.',
        tests_read=False,primary_changed=False,fresh_confirmation_required=True,limitation=LIMITATION)


def source_signature():
    completed=fallback_gate()
    base_record=long_probes.candidate_record('dense-mlp128',read(long_data.OUT/'search-plan.json'))
    original=read(long_data.OUT/'extraction-complete.json')
    base_rows=long_probes.BANK/'rows.parquet'; head_rows=heads.BANK/'rows.parquet'
    row_sha=data.digest(base_rows)
    require(row_sha == data.digest(head_rows) == original['rows_sha256'] == completed['signature']['rows_sha256'],
        'The two development banks must have byte-identical row tables.')
    require(original['rows']==ROWS and original['dimensions']==BASE_DIM and original['all_finite']
        and original['tests_read'] is False,'Frozen baseline feature bank is incomplete.')
    base_features=long_probes.BANK/'features.f16'
    require(base_features.stat().st_size == ROWS*BASE_DIM*2,'Baseline bank missing/truncated.')
    heads.require_hash(base_features,original['sha256'],'baseline raw features')
    _,_,_,_,partition=heads.dataset()
    scaler_path=heads.OUT/'standardization.npz'
    heads.require_hash(scaler_path,completed['standardization_sha256'],'completed head scaler')
    base_path=long_data.OUT/'models/dense-mlp128.npz'
    heads.require_hash(base_path,base_record['model_sha256'],'frozen dense128')
    with np.load(base_path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata'])); mean=archive['mean']; scale=archive['scale']
        require(meta['kind']=='mlp' and meta['hidden']==128 and meta['roles']==data.ROLES
            and meta['feature_spec']==attention.SPECS and mean.shape==scale.shape==(BASE_DIM,)
            and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Wrong frozen dense128 geometry.')
        base_scaler=dict(mean_sha256=array_digest(mean),scale_sha256=array_digest(scale))
    return dict(source_sha256=data.digest(__file__),dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        design_sha256=DESIGN_SHA256,head_completion_sha256=data.digest(heads.OUT/'complete.json'),
        head_selection_sha256=data.digest(heads.OUT/'selection.json'),head_search_plan_sha256=data.digest(heads.OUT/'search-plan.json'),
        head_signature=completed['signature'],head_scaler_sha256=data.digest(scaler_path),
        head_features_path=str(heads.BANK/'features.f16'),head_features_sha256=completed['signature']['features_sha256'],
        baseline_extraction_sha256=data.digest(long_data.OUT/'extraction-complete.json'),
        base_features_path=str(base_features),base_features_sha256=original['sha256'],rows_sha256=row_sha,partitions=partition,
        base_model_path=str(base_path),base_model_sha256=base_record['model_sha256'],base_embedded_scaler=base_scaler,
        base_candidate_sha256=data.digest(long_data.OUT/'candidates/dense-mlp128.json'),
        base_probability_path=str(long_probes.BANK/'dense-mlp128-validation.npz'),
        base_probability_sha256=base_record['validation_probabilities_sha256'],baseline_validation=base_record['validation'],
        head_confirmation_absent=True)


def dataset():
    rows,head,fit,valid,info=heads.dataset()
    require(data.digest(heads.BANK/'rows.parquet')==data.digest(long_probes.BANK/'rows.parquet'),'Development bank rows differ.')
    base=np.memmap(long_probes.BANK/'features.f16',mode='r',dtype=np.float16,shape=(ROWS,BASE_DIM))
    return rows,head,base,fit,valid,info


def make_delta(hidden=0,dimensions=HEAD_DIM):
    import torch
    from torch import nn
    class Delta(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(dimensions,hidden),nn.GELU(),nn.Dropout(.1)) if hidden else nn.Identity()
            self.head=nn.Linear(hidden or dimensions,1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        def delta(self,head): return BOUND*torch.tanh(self.head(self.encoder(head))/BOUND)
        def raw(self,base,head): return torch.cat([base[:,:1]+self.delta(head),base[:,1:]],dim=1)
        def forward(self,base,head): return torch.log_softmax(self.raw(base,head),dim=1)
    return Delta()


def load_base(path,expected_sha,device='cpu'):
    heads.require_hash(path,expected_sha,'portable frozen base')
    base,mean,scale,meta=probes.load_model(path,device)
    require(meta['kind']=='mlp' and meta['hidden']==128 and meta['roles']==data.ROLES
        and meta['feature_spec']==attention.SPECS and mean.shape==scale.shape==(BASE_DIM,)
        and bool(mean.isfinite().all()) and bool(scale.isfinite().all()) and bool((scale>0).all()),'Portable baseline changed.')
    return base.eval().requires_grad_(False),mean,scale


def save_model(path,delta,mean,scale,metadata):
    require(np.asarray(mean).shape==np.asarray(scale).shape==(HEAD_DIM,) and np.isfinite(mean).all()
        and np.isfinite(scale).all() and (np.asarray(scale)>0).all(),'Invalid frozen head scaler.')
    atomic_npz(path,mean=np.asarray(mean,np.float32),scale=np.asarray(scale,np.float32),metadata=np.array(json.dumps(metadata)),
        **{'delta::'+name:value.detach().cpu().numpy() for name,value in delta.state_dict().items()})


def load_model(path,device='cpu',base_path=None):
    """Independent raw8640/raw512 consumer; no training-bank dependency."""
    import torch
    from torch import nn
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        require(meta['roles']==data.ROLES and meta['head_dimensions']==HEAD_DIM and meta['base_dimensions']==BASE_DIM
            and meta['bound']==BOUND and meta['configuration']==CONFIGS[meta['candidate']],'Portable fusion geometry changed.')
        correction=make_delta(meta['configuration']['hidden'])
        correction.load_state_dict({key[7:]:torch.from_numpy(archive[key].copy()) for key in archive.files if key.startswith('delta::')})
        mean=np.asarray(archive['mean'],np.float32).copy();scale=np.asarray(archive['scale'],np.float32).copy()
        require(mean.shape==scale.shape==(HEAD_DIM,) and np.isfinite(mean).all() and np.isfinite(scale).all()
            and (scale>0).all() and array_digest(mean)==meta['head_scaler_mean_sha256']
            and array_digest(scale)==meta['head_scaler_scale_sha256'],'Portable head scaler changed.')
    base,bm,bs=load_base(base_path or meta['base_model_path'],meta['base_model_sha256'],device)
    require(array_digest(bm.cpu().numpy())==meta['base_scaler_mean_sha256']
        and array_digest(bs.cpu().numpy())==meta['base_scaler_scale_sha256'],'Portable embedded baseline scaler changed.')
    class Portable(nn.Module):
        def __init__(self):
            super().__init__();self.base=base;self.correction=correction
            self.register_buffer('base_mean',bm);self.register_buffer('base_scale',bs)
            self.register_buffer('head_mean',torch.from_numpy(mean));self.register_buffer('head_scale',torch.from_numpy(scale))
        def train(self,mode=True):
            super().train(mode);self.base.eval();return self
        def raw(self,raw_base,raw_head):
            with torch.no_grad(): frozen=self.base.raw((raw_base-self.base_mean)/self.base_scale)
            return self.correction.raw(frozen,(raw_head-self.head_mean)/self.head_scale)
        def forward(self,raw_base,raw_head): return torch.log_softmax(self.raw(raw_base,raw_head),dim=1)
    return Portable().to(device).eval(),meta


def predict_raw(path,base_bank,head_bank,indices,device='cpu'):
    import torch
    model,_=load_model(path,device);result=[]
    with torch.inference_mode():
        for start in range(0,len(indices),BATCH_SIZE):
            ix=indices[start:start+BATCH_SIZE]
            a=torch.from_numpy(np.asarray(base_bank[ix],np.float32)).to(device)
            b=torch.from_numpy(np.asarray(head_bank[ix],np.float32)).to(device)
            result.append(model(a,b).exp().cpu().numpy())
    result=np.concatenate(result).astype(np.float32);validate_probabilities(result,len(indices));return result


def metric_at_threshold(rows,p,threshold):
    require(np.array_equal(rows.index,np.arange(len(rows))) and np.isfinite(threshold) and 0<=threshold<=1.000001,
        'Fixed-threshold metrics require reset row indices and a finite valid threshold.')
    values={profile:probes.metrics(p[rows.profile.to_numpy()==profile],probes.labels(rows[rows.profile==profile]),threshold)
        for profile in long_probes.PROFILES}
    full=rows[rows.profile=='long-full'];offset=full.token_ix-full.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    ix=full.index[offset>=512].to_numpy();tail=probes.metrics(p[ix],probes.labels(rows)[ix],threshold)
    tail.update(documents=int(full.loc[ix].document_id.nunique()),body_start=512,body_end=1023)
    return dict(system_threshold=float(threshold),mean_system_f1=float(np.mean([r['system_f1'] for r in values.values()])),
        original_mean_system_f1=float(np.mean([values[k]['system_f1'] for k in long_probes.ORIGINAL_PROFILES])),
        original_mean_accuracy=float(np.mean([values[k]['accuracy'] for k in long_probes.ORIGINAL_PROFILES])),profiles=values,long_full_tail=tail)


def dual_guard(result,original,zero):
    a=heads.promotion_guard(result,original);b=heads.promotion_guard(result,zero)
    ceiling=max(original['mean_system_f1'],zero['mean_system_f1'])
    return dict(eligible=bool(a['eligible'] and b['eligible']),original_dense=a,delta0_retuned=b,
        required_mean_f1=ceiling,strict_mean_improvement=bool(result['mean_system_f1']>ceiling+1e-12),
        mean_gain_vs_original=result['mean_system_f1']-original['mean_system_f1'],
        mean_gain_vs_delta0=result['mean_system_f1']-zero['mean_system_f1'])


def choose_candidate(records,original,zero):
    guards={r['candidate']:dual_guard(r['validation'],original,zero) for r in records}
    eligible=[r for r in records if guards[r['candidate']]['eligible']]
    report_best=max(records,key=lambda r:r['validation']['mean_system_f1'])
    selected=max(eligible,key=lambda r:r['validation']['mean_system_f1']) if eligible else report_best
    guard=guards[selected['candidate']];promoted=bool(eligible and guard['strict_mean_improvement'])
    reasons=[]
    if not eligible:reasons.append('no_dual_guard_eligible_candidate')
    for name in ['original_dense','delta0_retuned']:
        reasons.extend(name+':'+reason for reason in guard[name]['rejection_reasons'])
    if not guard['strict_mean_improvement']:reasons.append('mean_f1_not_above_both_controls_plus1e-12')
    return selected,report_best,guards,promoted,reasons


def fsync_file(path):
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())


def logit_bytes(rows=ROWS):return rows*CLASSES*np.dtype(np.float32).itemsize


def initialize_logits(cache,plan_sha,rows_sha,rows=ROWS):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path=cache/'base-logits.f32';marker=cache/'base-logits-initialized.json'
    expected=dict(plan_sha256=plan_sha,rows_sha256=rows_sha,rows=rows,classes=5,dtype='float32')
    if marker.exists():
        require(read(marker)==expected,'Baseline cache ownership changed.')
        require(path.is_file() and path.stat().st_size==logit_bytes(rows),'Missing/truncated baseline cache; refusing saved progress.')
    else:
        temporary=cache/'base-logits.initializing'
        require(not any(p.exists() for p in [path,temporary,cache/'base-logits-progress.json']),
            'Undeclared or interrupted baseline cache exists; no file is silently recreated.')
        value=np.memmap(temporary,mode='w+',dtype=np.float32,shape=(rows,5));value[:]=np.nan
        value.flush();del value;fsync_file(temporary);temporary.replace(path);data.atomic_json(marker,expected)
    return np.memmap(path,mode='r+',dtype=np.float32,shape=(rows,5))


def logit_progress(array,rows,next_chunk,hashes,plan_sha,chunk=BATCH_SIZE):
    chunks=(len(rows)+chunk-1)//chunk
    require(type(next_chunk) is int and 0<=next_chunk<=chunks and len(hashes)==next_chunk,'Invalid baseline prefix cursor.')
    end=min(next_chunk*chunk,len(rows))
    return dict(version=1,plan_sha256=plan_sha,next_chunk=next_chunk,rows_written=end,chunk_size=chunk,
        cache_ix_sha256=array_digest(rows.cache_ix.to_numpy()[:end]),sample_ix_sha256=array_digest(rows.sample_ix.to_numpy()[:end]),
        chunk_sha256=list(hashes))


def restore_logits(cache,array,rows,plan_sha,chunk=BATCH_SIZE):
    path=Path(cache)/'base-logits-progress.json'
    if not path.exists():return []
    record=read(path);hashes=record['chunk_sha256']
    require(record==logit_progress(array,rows,len(hashes),hashes,plan_sha,chunk),'Baseline progress/source prefix differs.')
    for i,digest in enumerate(hashes):
        block=np.asarray(array[i*chunk:min((i+1)*chunk,len(rows))])
        require(np.isfinite(block).all() and array_digest(block)==digest,'Committed baseline logits changed or are unwritten.')
    return hashes.copy()


def predict_logits(array,indices,device='cpu'):
    import torch
    parts=[]
    with torch.inference_mode():
        for start in range(0,len(indices),BATCH_SIZE):
            z=torch.from_numpy(np.asarray(array[indices[start:start+BATCH_SIZE]],np.float32)).to(device)
            parts.append(torch.log_softmax(z,dim=1).exp().cpu().numpy())
    p=np.concatenate(parts).astype(np.float32);validate_probabilities(p,len(indices));return p


def zero_control(valid,p,original_p,original):
    validate_probabilities(p,len(valid));validate_probabilities(original_p,len(valid))
    difference=np.abs(p-original_p);maximum=float(difference.max())
    require(maximum<TOLERANCE,'Full validation baseline recomputation differs from frozen dense128 probabilities.')
    threshold=original['system_threshold']
    frozen=metric_at_threshold(valid,original_p,threshold)
    require(frozen==original,'Frozen dense128 validation metrics do not match its saved probabilities.')
    return dict(validation_rows=len(valid),max_probability_difference=maximum,mean_probability_difference=float(difference.mean()),
        argmax_changes=int((p.argmax(1)!=original_p.argmax(1)).sum()),
        original_threshold_system_decision_changes=int(((p[:,0]>=threshold)!=(original_p[:,0]>=threshold)).sum()),
        original_system_threshold=threshold,original_threshold_metrics=metric_at_threshold(valid,p,threshold),
        retuned_validation=long_probes.validation_metrics(valid,p),third_trained_candidate=False,
        identity_note='Delta zero is the same mathematical function. Actual recomputed floating probabilities are audited with tolerance, not asserted bit-identical.')


def verify_plan(plan,cache_ready=True):
    require(plan['protocol']==protocol() and plan['signature']==source_signature(),'Frozen fusion protocol/source/banks changed.')
    require(read(OUT/'search-plan.json')==plan,'Published fusion search plan changed.')
    heads.require_hash(OUT/'head-standardization.npz',plan['signature']['head_scaler_sha256'],'copied head scaler')
    if cache_ready:verified_base_cache(plan)


def verified_base_cache(plan):
    path=OUT/'base-logits-complete.json';record=read(path)
    require(record['plan_sha256']==data.digest(OUT/'search-plan.json') and record['signature']==plan['signature']
        and record['rows']==ROWS and record['shape']==[ROWS,5] and record['dtype']=='float32'
        and record['all_rows_written'] and record['all_finite'] and record['tests_read'] is False,'Baseline completion changed.')
    logits=CACHE/'base-logits.f32'
    require(logits.is_file() and logits.stat().st_size==logit_bytes(),'Missing/truncated completed baseline cache.')
    heads.require_hash(logits,record['logits_sha256'],'shared baseline logits')
    heads.require_hash(CACHE/'base-logits-progress.json',record['progress_sha256'],'baseline complete prefix')
    heads.require_hash(CACHE/'delta-zero-validation.npz',record['zero_probabilities_sha256'],'baseline numerical control')
    progress=read(CACHE/'base-logits-progress.json')
    require(progress['rows_written']==ROWS and progress['next_chunk']==len(progress['chunk_sha256'])
        ==(ROWS+BATCH_SIZE-1)//BATCH_SIZE,'Incomplete baseline prefix.')
    rows,_,_,valid,_=heads.dataset()
    with np.load(CACHE/'delta-zero-validation.npz',allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'],valid.sample_ix),'Zero-control validation order changed.')
        p=archive['p'].copy()
    validate_probabilities(p,VALID_ROWS)
    with np.load(plan['signature']['base_probability_path'],allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'],valid.sample_ix),'Original dense validation order changed.')
        old=archive['p'].copy()
    require(record['zero_control']==zero_control(valid,p,old,plan['signature']['baseline_validation']),
        'Baseline numerical audit or retuned zero control changed.')
    return record


def prepare_base_cache(plan,device='cuda:0'):
    verify_plan(plan,cache_ready=False)
    if (OUT/'base-logits-complete.json').exists():return verified_base_cache(plan)
    import torch
    rows,_,base,_,valid,_=dataset();plan_sha=data.digest(OUT/'search-plan.json')
    array=initialize_logits(CACHE,plan_sha,plan['signature']['rows_sha256'])
    hashes=restore_logits(CACHE,array,rows,plan_sha)
    net,mean,scale=load_base(plan['signature']['base_model_path'],plan['signature']['base_model_sha256'],device)
    with torch.inference_mode():
        for start in range(len(hashes)*BATCH_SIZE,ROWS,BATCH_SIZE):
            raw=torch.from_numpy(np.asarray(base[start:start+BATCH_SIZE],np.float32)).to(device)
            logits=net.raw((raw-mean)/scale).cpu().numpy().astype(np.float32)
            require(logits.shape==(min(BATCH_SIZE,ROWS-start),5) and np.isfinite(logits).all(),'Invalid frozen baseline raw logits.')
            array[start:start+len(logits)]=logits;hashes.append(array_digest(logits))
            if len(hashes)%16==0 or start+len(logits)==ROWS:
                array.flush();fsync_file(CACHE/'base-logits.f32')
                data.atomic_json(CACHE/'base-logits-progress.json',logit_progress(array,rows,len(hashes),hashes,plan_sha))
                print(json.dumps(dict(stage='shared-base-logits',rows=min(start+len(logits),ROWS),total=ROWS)),flush=True)
    require(len(hashes)==(ROWS+BATCH_SIZE-1)//BATCH_SIZE,'Incomplete baseline logits.')
    for start in range(0,ROWS,BATCH_SIZE):require(np.isfinite(array[start:start+BATCH_SIZE]).all(),'Unwritten baseline row.')
    p=predict_logits(array,valid.cache_ix.to_numpy(),device)
    with np.load(plan['signature']['base_probability_path'],allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'],valid.sample_ix),'Frozen baseline validation identities changed.')
        original_p=archive['p'].copy()
    control=zero_control(valid,p,original_p,plan['signature']['baseline_validation'])
    atomic_npz(CACHE/'delta-zero-validation.npz',p=p,sample_ix=valid.sample_ix.to_numpy())
    array.flush();fsync_file(CACHE/'base-logits.f32');verify_plan(plan,cache_ready=False)
    record=dict(signature=plan['signature'],plan_sha256=plan_sha,rows=ROWS,shape=[ROWS,5],dtype='float32',
        logits_sha256=data.digest(CACHE/'base-logits.f32'),progress_sha256=data.digest(CACHE/'base-logits-progress.json'),
        zero_probabilities_sha256=data.digest(CACHE/'delta-zero-validation.npz'),zero_control=control,
        all_finite=True,all_rows_written=True,base_model_eval=True,base_model_requires_grad=False,tests_read=False)
    data.atomic_json(OUT/'base-logits-complete.json',record)
    del array,net,mean,scale;torch.cuda.empty_cache() if str(device).startswith('cuda') else None
    return verified_base_cache(plan)


def model_metadata(key,plan,mean,scale):
    base_scaler=plan['signature']['base_embedded_scaler']
    return dict(candidate=key,configuration=CONFIGS[key],roles=data.ROLES,head_dimensions=HEAD_DIM,
        base_dimensions=BASE_DIM,bound=BOUND,base_model_path=plan['signature']['base_model_path'],
        base_model_sha256=plan['signature']['base_model_sha256'],base_scaler_mean_sha256=base_scaler['mean_sha256'],
        base_scaler_scale_sha256=base_scaler['scale_sha256'],head_scaler_mean_sha256=array_digest(np.asarray(mean,np.float32)),
        head_scaler_scale_sha256=array_digest(np.asarray(scale,np.float32)),head_scaler_sha256=plan['signature']['head_scaler_sha256'],
        signature=plan['signature'],search_plan_sha256=data.digest(OUT/'search-plan.json'),
        base_cache_complete_sha256=data.digest(OUT/'base-logits-complete.json'),seed=data.SEED,
        train_rows=FIT_ROWS,tests_read=False,primary_changed=False,limitation=LIMITATION)


def predict_delta(net,x,z):
    import torch
    net.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),BATCH_SIZE):parts.append(net(z[start:start+BATCH_SIZE],x[start:start+BATCH_SIZE]).exp().cpu().numpy())
    p=np.concatenate(parts).astype(np.float32);validate_probabilities(p,len(x));return p


def candidate_record(key,plan):
    verify_plan(plan)
    record=read(OUT/'candidates'/f'{key}.json')
    require(record['candidate']==key and record['configuration']==CONFIGS[key] and record['signature']==plan['signature']
        and record['search_plan_sha256']==data.digest(OUT/'search-plan.json') and record['tests_read'] is False
        and record['base_cache_complete_sha256']==data.digest(OUT/'base-logits-complete.json'), 'Fusion candidate provenance changed.')
    model_path=OUT/'models'/f'{key}.npz';prob_path=CACHE/f'{key}-validation.npz'
    heads.require_hash(model_path,record['model_sha256'],'fusion model')
    heads.require_hash(prob_path,record['validation_probabilities_sha256'],'fusion probabilities')
    _,_,_,valid,_=heads.dataset()
    with np.load(prob_path,allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'],valid.sample_ix),'Fusion validation identities changed.')
        p=archive['p'].copy()
    validate_probabilities(p,VALID_ROWS)
    require(long_probes.validation_metrics(valid,p)==record['validation'],'Fusion metrics do not match saved probabilities.')
    _,meta=load_model(model_path)
    with np.load(OUT/'head-standardization.npz',allow_pickle=False) as archive:
        expected=model_metadata(key,plan,archive['mean'],archive['scale'])
    require(all(meta.get(k)==v and record.get(k)==v for k,v in expected.items())
        and meta['best_epoch']==record['best_epoch'] and meta['best_steps']==record['best_steps'],
        'Fusion portable model metadata changed.')
    require_candidate_evidence(record)
    require(record['dual_guard']==dual_guard(record['validation'],plan['signature']['baseline_validation'],
        verified_base_cache(plan)['zero_control']['retuned_validation']),'Candidate dual-control guard changed.')
    return record


def require_candidate_evidence(record):
    for field in ['zero_full_validation_rows','zero_portable_full_validation_rows','portable_full_validation_rows']:
        require(record[field]==VALID_ROWS,'Fusion full-validation evidence is incomplete.')
    require(record['portable_cpu_rows']==256,'Independent CPU portable evidence is incomplete.')
    for field in ['zero_max_probability_difference_to_original','zero_portable_max_probability_difference_to_original',
                  'portable_full_max_probability_difference','portable_cpu_max_probability_difference']:
        require(np.isfinite(record[field]) and 0<=record[field]<TOLERANCE,'Fusion portable/zero-offset parity failed.')
    history=record['history'];epochs=record['epochs'];best_epoch=record['best_epoch']
    require(type(epochs) is int and 10<=epochs<=45 and len(history)==epochs
        and type(best_epoch) is int and 1<=best_epoch<=epochs
        and record['best_steps']==best_epoch*STEPS and record['actual_steps']==epochs*STEPS
        and record['actual_instances']==epochs*FIT_ROWS and record['instances_per_epoch']==FIT_ROWS
        and record['steps_per_epoch']==STEPS and record['last_batch_instances']==554,'Fusion training budget evidence changed.')
    best=-1.;chosen=-1
    for epoch,row in enumerate(history,1):
        require(row['epoch']==epoch and row['instances']==FIT_ROWS and row['steps']==STEPS
            and row['cumulative_steps']==epoch*STEPS and np.isfinite(row['loss'])
            and np.isfinite(row['mean_system_f1']),'Fusion epoch history changed.')
        if row['mean_system_f1']>best+1e-5:best=row['mean_system_f1'];chosen=epoch
        require(epoch<epochs or epoch==45 or (epoch>=10 and epoch-chosen>=8),
            'Fusion stopped before the frozen early-stopping condition.')
        require(epoch==epochs or epoch<10 or epoch-chosen<8,'Fusion continued after its early-stopping condition.')
    require(chosen==best_epoch and best==record['validation']['mean_system_f1'],'Best fusion checkpoint does not match epoch history.')


def worker_lease():
    require(os.environ.get('ROLE_FUSION_SUPERVISOR_PID')==str(os.getppid()),
        'Worker lost its parent supervisor lease; independent/orphan training is forbidden.')


def worker(key,gpu):
    worker_lease();no_competing_jobs(allow_own=True)
    require(not (OUT/'selection.json').exists() and not (OUT/'complete.json').exists()
        and not (OUT/'candidates'/f'{key}.json').exists(),'A frozen search/candidate cannot be overwritten.')
    plan=read(OUT/'search-plan.json');verify_plan(plan);cache_record=verified_base_cache(plan)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    require(gpu in range(2) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
        and torch.cuda.device_count()==1 and 'A40' in torch.cuda.get_device_name(0),'Worker needs its assigned single A40.')
    rows,head,base,fit,valid,partition=dataset()
    require(partition==plan['signature']['partitions'],'Worker development row partition changed.')
    free,_=torch.cuda.mem_get_info(0)
    require(free>(FIT_ROWS+VALID_ROWS)*HEAD_DIM*4+2*2**30,'Insufficient assigned GPU memory for streamed readout work.')
    logits=np.memmap(CACHE/'base-logits.f32',mode='r',dtype=np.float32,shape=(ROWS,5))
    with np.load(OUT/'head-standardization.npz',allow_pickle=False) as archive:
        mean=np.asarray(archive['mean'],np.float32).copy();scale=np.asarray(archive['scale'],np.float32).copy()
    def load(indices,bank,standardize=False):
        width=HEAD_DIM if standardize else 5
        result=torch.empty((len(indices),width),dtype=torch.float32,device='cuda',requires_grad=False)
        for start in range(0,len(indices),BATCH_SIZE):
            value=np.asarray(bank[indices[start:start+BATCH_SIZE]],np.float32)
            if standardize:value=(value-mean)/scale
            require(np.isfinite(value).all(),'Nonfinite frozen training features/logits.')
            result[start:start+len(value)]=torch.from_numpy(value).cuda()
        return result
    x=load(fit.cache_ix.to_numpy(),head,True);xv=load(valid.cache_ix.to_numpy(),head,True)
    z=load(fit.cache_ix.to_numpy(),logits);zv=load(valid.cache_ix.to_numpy(),logits)
    y=torch.from_numpy(probes.labels(fit)).cuda()
    torch.manual_seed(data.SEED);net=make_delta(CONFIGS[key]['hidden']).cuda()
    metadata=model_metadata(key,plan,mean,scale)
    path=OUT/'models'/f'{key}.npz';probability_path=CACHE/f'{key}-validation.npz'
    # This is an actual full-validation initialized-network forward, not merely
    # the algebraic assertion that zero final weights imply zero correction.
    p_zero=predict_delta(net,xv,zv)
    with np.load(CACHE/'delta-zero-validation.npz',allow_pickle=False) as archive:cache_zero=archive['p'].copy()
    with np.load(plan['signature']['base_probability_path'],allow_pickle=False) as archive:original=archive['p'].copy()
    zero_difference=float(np.abs(p_zero-original).max())
    require(zero_difference<TOLERANCE and np.max(np.abs(p_zero-cache_zero))<1e-6,
        'Full173032 initialized delta0 does not reproduce the verified numerical baseline.')
    save_model(path,net,mean,scale,metadata|dict(initialized_zero_check_only=True))
    actual_zero=predict_raw(path,base,head,valid.cache_ix.to_numpy(),device='cuda:0')
    zero_portable=float(np.abs(actual_zero-original).max())
    require(zero_portable<TOLERANCE and np.max(np.abs(actual_zero-p_zero))<TOLERANCE,
        'Full raw8640/raw512 initialized portable delta0 differs from the frozen baseline.')
    worker_lease()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    generator=torch.Generator(device='cuda:0').manual_seed(data.SEED)
    best=-1.;best_epoch=-1;history=[];started=time.time()
    for epoch in range(1,46):
        worker_lease();net.train();order=long_probes.epoch_order(len(x),generator,'cuda')
        require(len(order)==FIT_ROWS,'Epoch sampled instance budget changed.')
        total_loss=0.;steps=0
        for start in range(0,len(order),BATCH_SIZE):
            worker_lease();ix=order[start:start+BATCH_SIZE]
            loss=torch.nn.functional.cross_entropy(net.raw(z[ix],x[ix]),y[ix])
            require(torch.isfinite(loss),'Nonfinite full five-class CE.')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            total_loss+=float(loss.detach())*len(ix);steps+=1
        require(steps==STEPS,'Epoch optimizer update budget changed.')
        p=predict_delta(net,xv,zv);result=long_probes.validation_metrics(valid,p);score=result['mean_system_f1']
        history.append(dict(epoch=epoch,loss=total_loss/len(order),instances=len(order),steps=steps,
            cumulative_steps=epoch*STEPS,mean_system_f1=score,long_full_f1=result['profiles']['long-full']['system_f1'],
            tail_f1=result['long_full_tail']['system_f1'],cot_recall={k:v['recall']['cot'] for k,v in result['profiles'].items()}))
        if score>best+1e-5:
            best=score;best_epoch=epoch
            save_model(path,net,mean,scale,metadata|dict(best_epoch=epoch,best_steps=epoch*STEPS))
            atomic_npz(probability_path,p=p,sample_ix=valid.sample_ix.to_numpy())
        print(json.dumps(dict(candidate=key,**history[-1],best_epoch=best_epoch)),flush=True)
        if epoch>=10 and epoch-best_epoch>=8:break
    worker_lease()
    with np.load(probability_path,allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'],valid.sample_ix),'Best validation checkpoint identity changed.')
        p=archive['p'].copy()
    validate_probabilities(p,VALID_ROWS)
    # Full validation through the separate raw-feature consumer, not a256-row
    # substitute and not the already cached baseline logits.
    portable=predict_raw(path,base,head,valid.cache_ix.to_numpy(),device='cuda:0')
    difference=float(np.abs(portable-p).max());require(difference<TOLERANCE,'Full nonzero portable validation differs from cached worker predictions.')
    ids=np.linspace(0,len(valid)-1,256,dtype=int)
    cpu=predict_raw(path,base,head,valid.iloc[ids].cache_ix.to_numpy(),device='cpu')
    cpu_difference=float(np.abs(cpu-p[ids]).max());require(cpu_difference<TOLERANCE,'Independent CPU portable consumer differs.')
    verify_plan(plan);worker_lease()
    result=long_probes.validation_metrics(valid,p)
    controls=cache_record['zero_control']
    guard=dual_guard(result,plan['signature']['baseline_validation'],controls['retuned_validation'])
    data.atomic_json(OUT/'candidates'/f'{key}.json',metadata|dict(validation=result,dual_guard=guard,
        best_epoch=best_epoch,best_steps=best_epoch*STEPS,epochs=len(history),history=history,
        actual_steps=sum(r['steps'] for r in history),actual_instances=sum(r['instances'] for r in history),
        instances_per_epoch=FIT_ROWS,steps_per_epoch=STEPS,last_batch_instances=554,
        model_sha256=data.digest(path),validation_probabilities_sha256=data.digest(probability_path),
        zero_full_validation_rows=VALID_ROWS,zero_max_probability_difference_to_original=zero_difference,
        zero_portable_full_validation_rows=VALID_ROWS,zero_portable_max_probability_difference_to_original=zero_portable,
        portable_full_validation_rows=VALID_ROWS,portable_full_max_probability_difference=difference,
        portable_cpu_rows=len(ids),portable_cpu_max_probability_difference=cpu_difference,
        seconds=time.time()-started,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))


def no_competing_jobs(allow_own=False):
    names={'systemness_head_scan.py','systemness_head_extract.py','systemness_head_probes.py',
        'systemness_head_confirm.py','systemness_long_extract.py','systemness_recover_local_bank.py',
        'systemness_long_probes.py','systemness_paired_probes.py','systemness_long_confirm.py',
        'systemness_attention_confirm.py','systemness_natural_recheck.py'}
    if not allow_own:names.add(Path(__file__).name)
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg):active.append(int(entry.name))
    require(not active,f'Other experiment processes remain active: {active}.')


@contextmanager
def run_lock():
    import fcntl
    with (OUT/'.run.lock').open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another fusion supervisor owns this output.') from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run_workers(pending,plan):
    require(len(pending)<=2,'At most two fusion workers are permitted.')
    workers=[]
    try:
        for gpu,key in enumerate(pending):
            log=(OUT/'logs'/f'{key}.log').open('a')
            try:
                process=subprocess.Popen([sys.executable,'-u',__file__,'--train','--worker',key,'--gpu',str(gpu)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ROLE_FUSION_SUPERVISOR_PID=str(os.getpid())),
                    stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            workers.append((key,process,log,gpu))
        data.atomic_json(OUT/'status.json',dict(stage='training',workers=[dict(candidate=k,pid=p.pid,gpu=g) for k,p,_,g in workers]))
        while workers:
            codes=[(key,p.poll()) for key,p,_,_ in workers]
            require(not [(k,c) for k,c in codes if c not in [None,0]],f'Fusion worker failed: {codes}.')
            if all(code==0 for _,code in codes):break
            time.sleep(5)
        for key,p,_,_ in workers:require(p.wait()==0,f'{key} failed.');candidate_record(key,plan)
    finally:
        errors=[]
        for key,p,_,_ in workers:
            try:
                if p.poll() is None:p.terminate()
            except BaseException as error:errors.append((key,repr(error)))
        for key,p,log,_ in workers:
            try:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            except BaseException as error:errors.append((key,repr(error)))
            finally:log.close()
        require(not errors,f'Could not reap fusion workers: {errors}.')


def select(plan):
    records=[candidate_record(key,plan) for key in CONFIGS];cache=verified_base_cache(plan)
    original=plan['signature']['baseline_validation'];zero=cache['zero_control']['retuned_validation']
    selected,report_best,guards,promoted,reasons=choose_candidate(records,original,zero)
    controls=dict(original_dense_frozen=original,delta0_original_threshold=cache['zero_control']['original_threshold_metrics'],
        delta0_cache_retuned=zero,delta0_is_trained_candidate=False)
    data.atomic_json(OUT/'controls.json',controls)
    flat=[]
    for record in records:
        for profile,values in dict(record['validation']['profiles'],**{'long-full-tail':record['validation']['long_full_tail']}).items():
            flat.append(dict(candidate=record['candidate'],profile=profile,
                **{k:v for k,v in values.items() if not isinstance(v,(dict,list))},cot_recall=values['recall']['cot']))
    pd.DataFrame(flat).to_csv(OUT/'validation-metrics.csv',index=False)
    frozen=dict(candidate=selected['candidate'],candidate_is_guard_eligible=guards[selected['candidate']]['eligible'],
        model_path=str(OUT/'models'/f'{selected["candidate"]}.npz'),model_sha256=selected['model_sha256'],
        validation_probabilities_sha256=selected['validation_probabilities_sha256'],validation=selected['validation'],
        system_threshold=selected['validation']['system_threshold'],promoted=promoted,report_best_candidate=report_best['candidate'],
        candidate_guards=guards,promotion_rejection_reasons=reasons,controls=controls,signature=plan['signature'],
        source_sha256=data.digest(__file__),search_plan_sha256=data.digest(OUT/'search-plan.json'),
        base_cache_complete_sha256=data.digest(OUT/'base-logits-complete.json'),
        candidates={r['candidate']:data.digest(OUT/'candidates'/f'{r["candidate"]}.json') for r in records},
        test_used_for_selection=False,confirmation_required=True,primary_changed=False,limitation=LIMITATION)
    lines=['Bounded system-logit fusion on reused development validation','',LIMITATION,'',
        f'Selected diagnostic: {selected["candidate"]}; dual-control promoted={promoted}; reasons={reasons}.',
        'Delta-zero is a numerical control, not a third trained candidate.',
        'Candidate          Profile             System F1  Five-role accuracy  CoT recall']
    for row in flat:lines.append(f'{row["candidate"]:18s} {row["profile"]:19s} {row["system_f1"]:.6f}   {row["accuracy"]:.6f}    {row["cot_recall"]}')
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    verify_plan(plan)
    data.atomic_json(OUT/'selection.json',frozen)
    data.atomic_json(OUT/'complete.json',dict(signature=plan['signature'],source_sha256=data.digest(__file__),
        search_plan_sha256=data.digest(OUT/'search-plan.json'),selection_sha256=data.digest(OUT/'selection.json'),
        controls_sha256=data.digest(OUT/'controls.json'),base_cache_complete_sha256=data.digest(OUT/'base-logits-complete.json'),
        candidate_sha256=frozen['candidates'],validation_metrics_sha256=data.digest(OUT/'validation-metrics.csv'),
        report_sha256=data.digest(OUT/'report.txt'),tests_read=False,primary_changed=False,fresh_confirmation_required=True))
    data.atomic_json(OUT/'status.json',dict(stage='complete',promoted=promoted,primary_changed=False))


def train():
    no_competing_jobs();signature=source_signature()  # Every gate precedes mkdir/feature allocation.
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    with run_lock():
        require(not (OUT/'selection.json').exists() and not (OUT/'complete.json').exists(),'A completed fusion search cannot be overwritten.')
        for name in ['models','candidates','logs']:(OUT/name).mkdir(exist_ok=True)
        (OUT/'.gitignore').write_text('/logs/\n/.run.lock\n')
        path=OUT/'search-plan.json';scaler=OUT/'head-standardization.npz'
        if path.exists():
            plan=read(path);verify_plan(plan,cache_ready=False)
            require(plan['signature']==signature,'Fusion restart signature differs.')
        else:
            require(not scaler.exists() and not any((OUT/'models').iterdir()) and not any((OUT/'candidates').iterdir()),
                'Undeclared partial fusion artifacts exist.')
            shutil.copyfile(heads.OUT/'standardization.npz',scaler)
            heads.require_hash(scaler,signature['head_scaler_sha256'],'byte-copied head scaler')
            plan=dict(protocol=protocol(),signature=signature);data.atomic_json(path,plan)
        import torch
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        require(torch.cuda.device_count()>=2 and all('A40' in torch.cuda.get_device_name(i) for i in range(2)),
            'Two available A40s are required after upstream jobs exit.')
        prepare_base_cache(plan);pending=[]
        for key in CONFIGS:
            if (OUT/'candidates'/f'{key}.json').exists():candidate_record(key,plan)
            else:pending.append(key)  # Incomplete candidates restart from zero initialization/epoch1.
        run_workers(pending,plan);select(plan)


def verified_completion():
    """Read-only independent-consumer gate; recheck both controls and candidates."""
    plan=read(OUT/'search-plan.json');verify_plan(plan)
    complete=read(OUT/'complete.json')
    require(complete['signature']==plan['signature'] and complete['source_sha256']==data.digest(__file__)
        and complete['tests_read'] is False and complete['primary_changed'] is False
        and complete['fresh_confirmation_required'] is True,'Completed fusion provenance changed.')
    for path,field in [('search-plan.json','search_plan_sha256'),('selection.json','selection_sha256'),
                       ('controls.json','controls_sha256'),('base-logits-complete.json','base_cache_complete_sha256'),
                       ('validation-metrics.csv','validation_metrics_sha256'),('report.txt','report_sha256')]:
        heads.require_hash(OUT/path,complete[field],'fusion completion '+path)
    require(set(complete['candidate_sha256'])==set(CONFIGS),'Completed fusion candidate coverage changed.')
    records=[]
    for key in CONFIGS:
        heads.require_hash(OUT/'candidates'/f'{key}.json',complete['candidate_sha256'][key],'fusion candidate completion')
        records.append(candidate_record(key,plan))
    cache=verified_base_cache(plan);original=plan['signature']['baseline_validation']
    zero=cache['zero_control']['retuned_validation']
    controls=dict(original_dense_frozen=original,delta0_original_threshold=cache['zero_control']['original_threshold_metrics'],
        delta0_cache_retuned=zero,delta0_is_trained_candidate=False)
    require(read(OUT/'controls.json')==controls,'Completed fusion controls changed.')
    chosen,report_best,guards,promoted,reasons=choose_candidate(records,original,zero)
    selected=read(OUT/'selection.json')
    expected=dict(candidate=chosen['candidate'],candidate_is_guard_eligible=guards[chosen['candidate']]['eligible'],
        model_path=str(OUT/'models'/f'{chosen["candidate"]}.npz'),model_sha256=chosen['model_sha256'],
        validation_probabilities_sha256=chosen['validation_probabilities_sha256'],validation=chosen['validation'],
        system_threshold=chosen['validation']['system_threshold'],promoted=promoted,report_best_candidate=report_best['candidate'],
        candidate_guards=guards,promotion_rejection_reasons=reasons,controls=controls,signature=plan['signature'],
        source_sha256=data.digest(__file__),search_plan_sha256=complete['search_plan_sha256'],
        base_cache_complete_sha256=complete['base_cache_complete_sha256'],candidates=complete['candidate_sha256'],
        test_used_for_selection=False,confirmation_required=True,primary_changed=False,limitation=LIMITATION)
    require(selected==expected,'Completed fusion selection/threshold/promotion rule changed.')
    # Repeat all upstream/absence gates after reading every completed artifact.
    verify_plan(plan)
    return complete


def preflight():
    """Synthetic CPU-only fault tests. No upstream feature/model or test reads."""
    import torch
    import torch.nn.functional as functional
    import copy
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    root=Path(tempfile.mkdtemp(prefix='systemness-head-fusion-preflight-',dir='/tmp'))
    checks=[];observations={};source_sha=data.digest(__file__)
    original_paths=dict(out_exists=OUT.exists(),cache_exists=CACHE.exists())
    def check(name,condition):
        require(bool(condition),'Preflight failed: '+name);checks.append(name)
    def rejects(name,function):
        try:function()
        except (RuntimeError,ValueError,KeyError,IndexError,AssertionError,FileNotFoundError):checks.append(name);return
        raise RuntimeError('Preflight did not reject: '+name)
    @contextmanager
    def patched(**values):
        before={name:globals()[name] for name in values}
        globals().update(values)
        try:yield
        finally:globals().update(before)
    check_design();check('preimplementation_design_sha_and_budget',protocol()['design_sha256']==DESIGN_SHA256)
    torch.manual_seed(data.SEED);rng=np.random.default_rng(data.SEED)
    raw=torch.randn(37,5,dtype=torch.float64);head=torch.randn(37,HEAD_DIM,dtype=torch.float64)
    for key,config in CONFIGS.items():
        net=make_delta(config['hidden']).double().eval()
        check(key+'_actual_zero_final_layer',torch.count_nonzero(net.head.weight)==0 and torch.count_nonzero(net.head.bias)==0
            and torch.equal(net.raw(raw,head),raw) and torch.equal(net(raw,head),torch.log_softmax(raw,dim=1)))
        with torch.no_grad():net.head.weight.fill_(.08);net.head.bias.fill_(.25)
        shifted=net.raw(raw,head);delta=net.delta(head);p=shifted.softmax(1);base=raw.softmax(1)
        error=float(((p[:,0]/p[:,1:].sum(1)).log()-(base[:,0]/base[:,1:].sum(1)).log()-delta[:,0]).abs().max().detach())
        check(key+'_bounded_system_only_log_odds',error<1e-12 and float(delta.abs().max().detach())<=BOUND
            and torch.equal(shifted[:,1:],raw[:,1:])
            and torch.allclose(p[:,1:]/p[:,1:].sum(1,keepdim=True),base[:,1:]/base[:,1:].sum(1,keepdim=True),atol=1e-14,rtol=1e-14)
            and torch.equal(shifted[:,1:].argmax(1),raw[:,1:].argmax(1)))
        observations[key+'_log_odds_max_error']=error
        with torch.no_grad():net.head.weight.zero_();net.head.bias.fill_(1e9)
        check(key+'_saturating_bound',torch.equal(net.delta(head),torch.full((len(head),1),BOUND,dtype=torch.float64)))
    latent=torch.randn(len(raw),1,dtype=torch.float64,requires_grad=True);delta=BOUND*torch.tanh(latent/BOUND)
    y=torch.arange(len(raw))%5;logits=torch.cat([raw[:,:1]+delta,raw[:,1:]],1)
    full=functional.cross_entropy(logits,y)
    binary=functional.binary_cross_entropy_with_logits(raw[:,0]+delta[:,0]-torch.logsumexp(raw[:,1:],1),(y==0).double())
    a=torch.autograd.grad(full,latent,retain_graph=True)[0];b=torch.autograd.grad(binary,latent)[0]
    check('fiveway_ce_has_binary_delta_gradient',torch.allclose(a,b,atol=1e-15,rtol=1e-13))
    observations['fiveway_binary_gradient_max_error']=float((a-b).abs().max())
    # Exercise all173032 rows with the actual correction implementation, in
    #4096-row chunks; this is synthetic CPU proof, never a formal model result.
    z=rng.normal(size=(VALID_ROWS,5)).astype(np.float32);z_tensor=torch.from_numpy(z)
    reference=predict_logits(z,np.arange(VALID_ROWS))
    with torch.inference_mode():
        for key,config in CONFIGS.items():
            net=make_delta(config['hidden']).eval();parts=[]
            for start in range(0,VALID_ROWS,BATCH_SIZE):
                part=z_tensor[start:start+BATCH_SIZE]
                parts.append(net(part,torch.zeros((len(part),HEAD_DIM))).exp().numpy())
            actual=np.concatenate(parts);check(key+'_complete173032_zero_cpu_forward',len(actual)==VALID_ROWS
                and np.array_equal(actual,reference))
    del z_tensor,z,reference,actual,parts
    base_path=root/'frozen-base.npz';base_net=probes.make_net('mlp',128).eval()
    base_mean=rng.normal(size=BASE_DIM).astype(np.float32);base_scale=rng.uniform(.5,2,size=BASE_DIM).astype(np.float32)
    probes.save_model(base_path,base_net,base_mean,base_scale,dict(kind='mlp',hidden=128,roles=data.ROLES,feature_spec=attention.SPECS))
    mean=rng.normal(size=HEAD_DIM).astype(np.float32);scale=rng.uniform(.5,2,size=HEAD_DIM).astype(np.float32)
    rb=torch.from_numpy(rng.normal(size=(29,BASE_DIM)).astype(np.float32));rh=torch.from_numpy(rng.normal(size=(29,HEAD_DIM)).astype(np.float32))
    for key,config in CONFIGS.items():
        correction=make_delta(config['hidden']).eval()
        with torch.no_grad():correction.head.weight.normal_(std=.07);correction.head.bias.fill_(.23)
        metadata=dict(candidate=key,configuration=config,roles=data.ROLES,head_dimensions=HEAD_DIM,base_dimensions=BASE_DIM,
            bound=BOUND,base_model_path=str(base_path),base_model_sha256=data.digest(base_path),
            base_scaler_mean_sha256=array_digest(base_mean),base_scaler_scale_sha256=array_digest(base_scale),
            head_scaler_mean_sha256=array_digest(mean),head_scaler_scale_sha256=array_digest(scale))
        model_path=root/(key+'.npz');save_model(model_path,correction,mean,scale,metadata)
        portable,meta=load_model(model_path)
        with torch.inference_mode():
            expected=correction.raw(base_net.raw((rb-torch.from_numpy(base_mean))/torch.from_numpy(base_scale)),
                (rh-torch.from_numpy(mean))/torch.from_numpy(scale))
            actual=portable.raw(rb,rh);probability=portable(rb,rh).exp().numpy()
        validate_probabilities(probability,len(rb))
        check(key+'_portable_raw8640_raw512_frozen_scalers',torch.equal(actual,expected) and meta==metadata)
        portable.train();check(key+'_base_eval_and_gradient_freeze',not portable.base.training and portable.correction.training
            and all(not parameter.requires_grad for parameter in portable.base.parameters())
            and all(parameter.requires_grad for parameter in portable.correction.parameters()))
        portable.eval()
        bad=metadata|dict(base_model_sha256='0'*64);save_model(root/'bad-base.npz',correction,mean,scale,bad)
        rejects(key+'_reject_base_sha_drift',lambda:load_model(root/'bad-base.npz'))
        save_model(root/'bad-scaler.npz',correction,mean+1,scale,metadata)
        rejects(key+'_reject_head_scaler_drift',lambda:load_model(root/'bad-scaler.npz'))
    # All five development profiles, every role, and body-relative512+ tail.
    rows=[];probabilities=[]
    for profile in long_probes.PROFILES:
        for role_ix,role in enumerate(data.ROLES):
            for position in [0,512]:
                rows.append(dict(profile=profile,split='validation',wrapper='native',role=role,token_ix=100+position,
                    prompt_ix=len(rows)//2,message_ix=role_ix,document_id=role_ix))
                p=np.full(5,.025,np.float32);p[role_ix]=.9;probabilities.append(p)
    valid=pd.DataFrame(rows);p=np.asarray(probabilities);baseline=long_probes.validation_metrics(valid,p)
    control=zero_control(valid,p,p,baseline)
    check('zero_control_frozen_threshold_and_retuned_metrics',control['max_probability_difference']==0
        and control['mean_probability_difference']==0 and control['argmax_changes']==0
        and control['original_threshold_system_decision_changes']==0
        and control['original_threshold_metrics']==baseline and control['retuned_validation']==baseline
        and control['third_trained_candidate'] is False and control['validation_rows']==len(valid))
    altered=p.copy();altered[0,0]-=.001;altered[0,1]+=.001
    rejects('reject_full_validation_zero_probability_drift',lambda:zero_control(valid,altered,p,baseline))
    shifted=valid.copy();shifted.index=np.arange(100,100+len(shifted))
    rejects('reject_unreset_validation_indices',lambda:long_probes.validation_metrics(shifted,p))
    # Deliberately synthetic numerical controls isolate the selection rule.
    original=copy.deepcopy(baseline);zero=copy.deepcopy(baseline)
    for result in [original,zero]:
        for values in result['profiles'].values():values['system_f1']=.75;values['accuracy']=.8;values['system_false_positive_rate']=.01
        result['long_full_tail']['system_f1']=.7
    original['mean_system_f1']=.75;zero['mean_system_f1']=.76
    candidate=copy.deepcopy(zero);candidate['mean_system_f1']=.755
    record=dict(candidate='delta-linear',validation=candidate)
    chosen,_,guards,promoted,_=choose_candidate([record],original,zero)
    check('numeric_control_improvement_cannot_be_claimed_as_delta_gain',chosen==record and guards['delta-linear']['eligible'] and not promoted)
    candidate2=copy.deepcopy(zero);candidate2['mean_system_f1']=.77
    record2=dict(candidate='delta-mlp32',validation=candidate2)
    check('promotion_requires_strict_gain_above_both',choose_candidate([record,record2],original,zero)[3])
    tied=copy.deepcopy(zero);tied['mean_system_f1']=zero['mean_system_f1']+1e-12
    check('strict_mean_epsilon_rejects_tie',not choose_candidate([dict(candidate='delta-linear',validation=tied)],original,zero)[3])
    for name,change in [('fpr',lambda r:r['profiles']['single'].update(system_false_positive_rate=.051)),
                        ('profile_f1',lambda r:r['profiles']['single'].update(system_f1=.72)),
                        ('accuracy',lambda r:r['profiles']['single'].update(accuracy=.78)),
                        ('full',lambda r:r['profiles']['long-full'].update(system_f1=.74)),
                        ('tail',lambda r:r['long_full_tail'].update(system_f1=.69))]:
        bad=copy.deepcopy(candidate2);change(bad)
        check('reject_dual_control_'+name,not dual_guard(bad,original,zero)['eligible'])
    # A strong diagnostic may fail guards; preserve the strongest eligible one.
    bad=copy.deepcopy(candidate2);bad['mean_system_f1']=.9;bad['long_full_tail']['system_f1']=.69
    result=choose_candidate([record2,dict(candidate='delta-linear',validation=bad)],original,zero)
    check('rank_only_dual_guard_eligible_report_all_negatives',result[0]['candidate']=='delta-mlp32'
        and result[1]['candidate']=='delta-linear' and result[3])
    cache=root/'cache';mapping=pd.DataFrame(dict(cache_ix=np.arange(11),sample_ix=np.arange(100,111)))
    array=initialize_logits(cache,'plan','rows',rows=len(mapping));check('new_cache_nan_initialized',np.isnan(array).all())
    array[:4]=rng.normal(size=(4,5));hashes=[array_digest(np.asarray(array[:4]))]
    array[4:6]=17  # Uncommitted partial work must be recomputed, never resumed.
    progress=logit_progress(array,mapping,1,hashes,'plan',chunk=4);data.atomic_json(cache/'base-logits-progress.json',progress)
    check('restore_exact_committed_prefix_only',restore_logits(cache,array,mapping,'plan',chunk=4)==hashes
        and progress['rows_written']==4)
    changed=mapping.copy();changed.loc[0,'sample_ix']+=1
    rejects('reject_cache_sample_ix_order_drift',lambda:restore_logits(cache,array,changed,'plan',chunk=4))
    changed=mapping.copy();changed.loc[0,'cache_ix']+=1
    rejects('reject_cache_ix_order_drift',lambda:restore_logits(cache,array,changed,'plan',chunk=4))
    saved=float(array[0,0]);array[0,0]+=1
    rejects('reject_committed_cache_hash_drift',lambda:restore_logits(cache,array,mapping,'plan',chunk=4));array[0,0]=saved
    array[0,0]=np.nan
    rejects('reject_committed_unwritten_nan',lambda:restore_logits(cache,array,mapping,'plan',chunk=4));array[0,0]=saved
    rejects('reject_cache_plan_drift',lambda:initialize_logits(cache,'new-plan','rows',rows=len(mapping)))
    del array;(cache/'base-logits.f32').unlink()
    rejects('reject_saved_progress_missing_cache',lambda:initialize_logits(cache,'plan','rows',rows=len(mapping)))
    (cache/'base-logits.f32').write_bytes(b'\0'*4)
    rejects('reject_saved_progress_truncated_cache',lambda:initialize_logits(cache,'plan','rows',rows=len(mapping)))
    rejects('reject_progress_cursor_not_prefix',lambda:logit_progress(None,mapping,3,hashes,'plan',chunk=4))
    good=dict(promoted=False,test_used_for_selection=False,primary_changed=False,confirmation_required=True)
    complete=dict(candidate_sha256={key:'x' for key in heads.CONFIGS},tests_read=False,primary_changed=False)
    with patched(HEAD_CONFIRM_OUT=root/'absent-confirm'):
        require_fallback(good,complete);checks.append('strict_unpromoted_complete_four_head_gate')
        for field,value in [('promoted',True),('promoted',0),('test_used_for_selection',True),('primary_changed',True)]:
            rejects('reject_fallback_'+field+'_'+str(value),lambda f=field,v=value:require_fallback(good|{f:v},complete))
        rejects('reject_incomplete_four_head_search',lambda:require_fallback(good,complete|dict(candidate_sha256={})))
    present=root/'present-confirm';present.mkdir()
    with patched(HEAD_CONFIRM_OUT=present):rejects('reject_existing_head_confirmation',lambda:require_fallback(good,complete))
    def gate_failure():raise RuntimeError('synthetic upstream gate failure')
    def forbidden_read():raise AssertionError('Bank must not be read before fallback gate')
    with patched(OUT=root/'never-out',CACHE=root/'never-cache',no_competing_jobs=lambda:None,fallback_gate=gate_failure,dataset=forbidden_read):
        rejects('train_gate_precedes_any_mkdir_or_bank_read',train)
        check('failed_gate_created_no_formal_paths',not OUT.exists() and not CACHE.exists())
    prior=os.environ.get('ROLE_FUSION_SUPERVISOR_PID')
    try:
        os.environ['ROLE_FUSION_SUPERVISOR_PID']='-1';rejects('reject_orphan_worker_lease',worker_lease)
        os.environ['ROLE_FUSION_SUPERVISOR_PID']=str(os.getppid());worker_lease();checks.append('live_parent_pid_lease')
    finally:
        if prior is None:os.environ.pop('ROLE_FUSION_SUPERVISOR_PID',None)
        else:os.environ['ROLE_FUSION_SUPERVISOR_PID']=prior
    generator=torch.Generator().manual_seed(data.SEED);order=long_probes.epoch_order(FIT_ROWS,generator,'cpu')
    check('seeded578090_142step_budget_last554',len(order)==FIT_ROWS and len(torch.unique(order))==FIT_ROWS
        and (len(order)+BATCH_SIZE-1)//BATCH_SIZE==STEPS and len(order)%BATCH_SIZE==554)
    # Candidate evidence is checked independently of formal artifacts.
    history=[dict(epoch=i,instances=FIT_ROWS,steps=STEPS,cumulative_steps=i*STEPS,loss=1.,mean_system_f1=.77) for i in range(1,11)]
    evidence=dict(zero_full_validation_rows=VALID_ROWS,zero_portable_full_validation_rows=VALID_ROWS,
        portable_full_validation_rows=VALID_ROWS,portable_cpu_rows=256,zero_max_probability_difference_to_original=0.,
        zero_portable_max_probability_difference_to_original=0.,portable_full_max_probability_difference=0.,
        portable_cpu_max_probability_difference=0.,history=history,epochs=10,best_epoch=1,best_steps=STEPS,
        actual_steps=10*STEPS,actual_instances=10*FIT_ROWS,instances_per_epoch=FIT_ROWS,steps_per_epoch=STEPS,
        last_batch_instances=554,validation=dict(mean_system_f1=.77))
    require_candidate_evidence(evidence);checks.append('complete_full_zero_nonzero_cpu_and_budget_evidence')
    for field in ['zero_portable_full_validation_rows','portable_full_validation_rows','portable_cpu_rows']:
        rejects('reject_incomplete_'+field,lambda f=field:require_candidate_evidence(evidence|{f:0}))
    for value in [np.nan,-.1,TOLERANCE]:
        rejects('reject_portable_error_'+str(value),lambda v=value:require_candidate_evidence(evidence|dict(portable_cpu_max_probability_difference=v)))
    rejects('reject_missing_candidate_zero_portable_evidence',lambda:require_candidate_evidence({k:v for k,v in evidence.items()
        if k!='zero_portable_max_probability_difference_to_original'}))
    # A launch failure must reap a previously spawned worker, including kill
    # escalation. No child process, CUDA context or formal output is involved.
    from unittest.mock import patch
    class FakeProcess:
        pid=123456
        def __init__(self):self.terminated=False;self.killed=False;self.reaped=False
        def poll(self):return None if not self.reaped else 0
        def terminate(self):self.terminated=True
        def kill(self):self.killed=True
        def wait(self,timeout=None):
            if not self.killed:raise subprocess.TimeoutExpired('synthetic-worker',timeout)
            self.reaped=True;return 0
    child=FakeProcess();launched=[]
    def launch(*args,**kwargs):
        if launched:raise RuntimeError('synthetic second launch failure')
        launched.append(kwargs['stdout']);return child
    work=root/'supervisor';(work/'logs').mkdir(parents=True)
    with patched(OUT=work),patch.object(subprocess,'Popen',launch):
        rejects('supervisor_second_launch_fault',lambda:run_workers(list(CONFIGS),{}))
    check('supervisor_terminates_kills_waits_reaps_and_closes_logs',child.terminated and child.killed and child.reaped
        and launched[0].closed)
    # Exercise the independent completion chain with fully hash-bound synthetic
    # metadata. Statistical routines above are tested separately with real arrays.
    completed=root/'completed';completed.mkdir()
    for name in ['models','candidates']:(completed/name).mkdir()
    synthetic_plan=dict(protocol=protocol(),signature=dict(baseline_validation=original))
    data.atomic_json(completed/'search-plan.json',synthetic_plan)
    synthetic_records=[]
    for key,value in zip(CONFIGS,[candidate,candidate2]):
        path=completed/'models'/f'{key}.npz';path.write_bytes(key.encode())
        rec=dict(candidate=key,validation=value,model_sha256=data.digest(path),validation_probabilities_sha256=key)
        synthetic_records.append(rec);data.atomic_json(completed/'candidates'/f'{key}.json',rec)
    synthetic_cache=dict(zero_control=dict(retuned_validation=zero,original_threshold_metrics=zero))
    data.atomic_json(completed/'base-logits-complete.json',synthetic_cache)
    def verified_plan_fixture(plan,cache_ready=True):require(plan==synthetic_plan,'Synthetic plan changed.')
    def record_fixture(key,plan):
        verified_plan_fixture(plan);return read(OUT/'candidates'/f'{key}.json')
    with patched(OUT=completed,verify_plan=verified_plan_fixture,candidate_record=record_fixture,
                 verified_base_cache=lambda plan:synthetic_cache):
        select(synthetic_plan);verified_completion();checks.append('completed_selection_binds_two_controls_models_probabilities_and_threshold')
        chosen=read(OUT/'selection.json');saved_complete=read(OUT/'complete.json')
        # Update the outer SHA too: semantic verification, not only the hash,
        # must reject a changed frozen threshold or a missing trained candidate.
        data.atomic_json(OUT/'selection.json',chosen|dict(system_threshold=.123))
        data.atomic_json(OUT/'complete.json',saved_complete|dict(selection_sha256=data.digest(OUT/'selection.json')))
        rejects('reject_completed_threshold_even_with_updated_outer_hash',verified_completion)
        data.atomic_json(OUT/'selection.json',chosen)
        data.atomic_json(OUT/'complete.json',saved_complete|dict(candidate_sha256={}))
        rejects('reject_completed_missing_second_candidate',verified_completion)
    check('source_unchanged_during_cpu_preflight',data.digest(__file__)==source_sha)
    check('formal_out_and_cache_untouched',original_paths==dict(out_exists=OUT.exists(),cache_exists=CACHE.exists()))
    report=dict(source_sha256=source_sha,design_sha256=DESIGN_SHA256,checks=checks,checks_passed=len(checks),
        observations=observations,synthetic_only=True,formal_training_started=False,gpu_used=False,tests_read=False,
        formal_paths=original_paths,source_sha256_at_end=data.digest(__file__))
    data.atomic_json(root/'report.json',report);print(json.dumps(dict(report=str(root/'report.json'),checks_passed=len(checks))),flush=True)
    return root/'report.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--train',action='store_true')
    parser.add_argument('--worker',choices=CONFIGS);parser.add_argument('--gpu',type=int,choices=range(2))
    args=parser.parse_args()
    if args.worker is not None or args.gpu is not None:
        if not (args.train and args.worker and args.gpu is not None):parser.error('--worker and --gpu require --train and each other')
        worker(args.worker,args.gpu)
    elif args.preflight:preflight()
    elif args.train:train()
    else:parser.print_help()


if __name__=='__main__':main()
