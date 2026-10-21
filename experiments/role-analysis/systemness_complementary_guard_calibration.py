"""One fixed640 model, guard-constrained global DEV threshold; default help.

--preflight is synthetic CPU/tmp only. Root must explicitly --declare, then
--actual. No LM, training, raw features, new test/source, model prediction or
primary change. The original failed trial and all14+zero controls are read-only.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-complementary-guard-calibration'
DESIGN=BASE/'systemness-complementary-guard-calibration-preparation/root-prospective-design.json'
DESIGN_SHA='cc3f2a695b7fc9cc9ae1aeaf4c610d9695283a2286b422d804465ff4030a06ad'
TRAINER=BASE/'systemness-complementary-head-readout'
AUDIT=TRAINER/'actual-audit/actual-development-audit.json'
SOURCE_SHA='e74e40889ba49ab05815f0faac68c2a522ccc06627b09ff14379929bc53dbdf6'
PROTOCOL_SHA='10c92c0af904b96d42e1ff8c9cbc12f7efb0371da08fdc3962aebd7b8d0d6f54'
COMPLETE_SHA='376ad13999055eb3144a2b3d90d51e318eb30f24d6f64991c819ea29374a2e52'
AUDITOR_SHA='0219af8dbf0dc6587f8ba4d4bc1e4f163ae9e745a2b6c4d99a24560a88c8ae85'
AUDIT_SHA='7bac44572d8ea37fc4cae87d447061228496a218eadaa92583a4a96bb3c4989c'
MODEL_SHA='ef92b421007b7d0c7a11f13e91008e6d2e39b34b730c6ca7d6ecbf03590236ff'
ROLES=['system','user','cot','assistant','tool']
PROFILES=['single','multi','natural','long-single','long-full']
OLD5=['old-'+p for p in PROFILES]
PUBLIC=['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
FULL=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']
GROUPS=OLD5+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']+PUBLIC
FEASIBLE=OLD5+['new-C4-long-full']+PUBLIC
STRICT=['matched-max33','oldhead-max33','ensemble-max33','zero-max33','priorBinary-max33']
GUARDS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33','priorBinary-max33']
RECORDS=[m+'-'+o for m in ['new','matched','oldhead'] for o in ['oldmean','max33']]+['ensemble-original','ensemble-max33']+[m+'-'+o for m in ['naturalCE','priorBinary','priorLongMix'] for o in ['oldmean','max33']]
BANKS=['old','c4','public'];COUNTS=dict(old=173032,c4=61440,public=16713)
ARTIFACTS=['protocol.json','search-plan.json','scaler-record.json','controls.json','zero-controls.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt']
BINARY=['system_threshold','system_accuracy','system_balanced_accuracy','system_precision','system_recall','system_f1','system_false_positive_rate','system_detection_by_role','system_confusion']
TOL=1e-12


def require(value,message):
    if not bool(value):raise RuntimeError(message)
def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def clean(path):
    raw=os.path.abspath(str(path));require(raw!='/workspace' and not raw.startswith('/workspace/'),'Forbidden path before filesystem access.');return Path(raw)
def digest(path):
    h=hashlib.sha256()
    with clean(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(clean(path).read_text())
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_json(path,value):
    path=clean(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
def immutable_json(path,value):
    if clean(path).exists():require(read(path)==value,'Existing calibration artifact differs; no overwrite: '+str(path))
    else:atomic_json(path,value)


def metadata_close(a,b,path='metadata'):
    """Independent JSON metric reprints: exact schema/types, F64 tol only."""
    require(type(a) is type(b),'Independent metadata type differs: '+path)
    if isinstance(a,dict):
        require(set(a)==set(b),'Independent metadata keys differ: '+path)
        for key in a:metadata_close(a[key],b[key],path+'.'+key)
    elif isinstance(a,list):
        require(len(a)==len(b),'Independent metadata length differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)):metadata_close(x,y,path+f'[{i}]')
    elif isinstance(a,float):require(np.isfinite(a) and np.isfinite(b) and abs(a-b)<=TOL,'Independent F64 metric differs: '+path)
    else:require(a==b,'Independent metadata value differs: '+path)


def protocol():
    return dict(version=1,experiment='complementary640-guard-constrained-global-development-calibration',fixed_model_sha256=MODEL_SHA,
        models=1,training_updates=0,threshold_budget=501,grid='Original unique199linspace(.001,.999)+301 pooled DEV quantiles+1.000001; actual F32 >=; no expansion.',
        selection='All absolute90/exactFPR/5strict/8guards+savedGPU first; then min33,old5meanF1,smaller originalF64tau; no fallback.',
        groups=GROUPS,feasible=FEASIBLE,strict_controls=STRICT,guard_controls=GUARDS+['original-savedGPU'],absolute_min33=.9,max_FPR=.05,tolerance=TOL,
        floor=dict(old4_f1_loss=.02,old4_accuracy_loss=.01,old5_mean_f1_loss=.005,full_tail_nonregression=FULL,public_f1_loss=.02,public_accuracy_loss=.01),
        invariants='All argmax5/role recall/confusion/macroF1/NLL/AUC/AP copied from audited new-max33; only binary threshold fields change.',
        declaration_probability_files_opened=False,declaration_row_tables_materialized=False,tests_read=False,primary_changed=False,
        limitation='Additional selection after reused DEV results; no new model/fiveclass gain or independent confirmation. Fresh C4 and public-natural confirmation still required.')


def threshold_grid(p):
    require(set(p)==set(BANKS) and all(v.dtype==np.float32 and v.ndim==2 and v.shape[1]==5 for v in p.values()),'Exact three F32 five-class DEV arrays required.')
    grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate([p[b][:,0] for b in BANKS]),np.linspace(0,1,301)),1.000001])
    require(0<len(grid)<=501 and np.isfinite(grid).all(),'Original threshold budget changed.');return grid


def binary_curves(score,y,grid):
    score=np.asarray(score);y=np.asarray(y);require(score.dtype==np.float32 and score.ndim==y.ndim==1 and len(score)==len(y)>0 and np.isfinite(score).all() and ((score>=0)&(score<=1.00001)).all() and np.isin(y,np.arange(5)).all(),'Invalid binary curve input.')
    positive=y==0;pos,neg=np.sort(score[positive]),np.sort(score[~positive]);require(len(pos)>0 and len(neg)>0,'Each calibration group needs both truth supports.')
    effective=np.asarray(grid,np.float32);tp=len(pos)-np.searchsorted(pos,effective,side='left');fp=len(neg)-np.searchsorted(neg,effective,side='left');fn=len(pos)-tp;tn=len(neg)-fp
    def div(a,b):return np.divide(a,b,out=np.full(len(grid),np.nan),where=b>0)
    return dict(system_precision=div(tp,tp+fp),system_recall=tp/len(pos),system_balanced_accuracy=(tp/len(pos)+tn/len(neg))/2,
        system_f1=div(2*tp,2*tp+fp+fn),system_false_positive_rate=fp/len(neg),system_accuracy=(tp+tn)/len(y),tp=tp,fp=fp,fn=fn,tn=tn,
        detection={r:((len(v)-np.searchsorted(v,effective,side='left'))/len(v) if len(v) else np.full(len(grid),np.nan)) for r in ROLES for v in [np.sort(score[y==ROLES.index(r)])]})


def guard(result,control,legacy=False):
    f1={g:control['groups'][g]['system_f1']-result['groups'][g]['system_f1'] for g in OLD5[:4]};acc={g:control['groups'][g]['accuracy']-result['groups'][g]['accuracy'] for g in OLD5[:4]}
    full={g:result['groups'][g]['system_f1']-control['groups'][g]['system_f1'] for g in FULL};mean=control['old5_mean_f1']-result['old5_mean_f1'];pf={g:control['groups'][g]['system_f1']-result['groups'][g]['system_f1'] for g in PUBLIC};pa={g:control['groups'][g]['accuracy']-result['groups'][g]['accuracy'] for g in PUBLIC};reasons=[]
    for bad,name in [(max(f1.values())>.02+TOL,'old4_system_f1_loss'),(max(acc.values())>.01+TOL,'old4_five_accuracy_loss'),(mean>.005+TOL,'old5_mean_f1_loss'),(any(v< -TOL for v in full.values()),'C4_full_tail_f1_regression'),(not legacy and max(pf.values())>.02+TOL,'public_system_f1_loss'),(not legacy and max(pa.values())>.01+TOL,'public_five_accuracy_loss'),(any(result['groups'][g]['system_false_positive_rate']>.05+TOL for g in FEASIBLE),'FPR_above_limit'),(not all(np.isfinite(result['groups'][g][k]) for g in GROUPS for k in ['system_f1','accuracy','system_false_positive_rate']),'nonfinite_metrics')]:
        if bad:reasons.append(name)
    return dict(eligible=not reasons,rejection_reasons=reasons,old4_f1_losses=f1,old4_accuracy_losses=acc,old5_mean_f1_loss=mean,C4_full_tail_f1_gains=full,public_f1_losses=pf,public_accuracy_losses=pa,legacy_savedGPU_only=legacy)


def light_result(curves,baseline,i):
    groups={g:dict(accuracy=baseline['groups'][g]['accuracy'],**{k:float(curves[g][k][i]) for k in ['system_f1','system_false_positive_rate']}) for g in GROUPS}
    values=[curves[g][k][i] for g in GROUPS for k in ['system_precision','system_recall','system_balanced_accuracy']]
    return dict(groups=groups,min33=float(min(values)) if all(np.isfinite(v) for v in values) else None,old5_mean_f1=float(np.mean([groups[g]['system_f1'] for g in OLD5])))


def constraints(result,controls,legacy):
    require(set(controls)==set(GUARDS) and set(result['groups'])==set(GROUPS),'Fixed guard/group inventory changed.');minimum=result['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    # The original max33 feasibility is exact <=.05, before tolerance-based guards.
    if any(result['groups'][g]['system_false_positive_rate']>.05 for g in FEASIBLE):reasons.append('original_exact_FPR_feasibility')
    for name in GUARDS:
        proof=guard(result,controls[name]);guards[name]=proof;reasons.extend(name+':'+r for r in proof['rejection_reasons'])
        if name in STRICT:
            gain=minimum-controls[name]['min33'] if minimum is not None and controls[name]['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=TOL:reasons.append(name+':min33_not_strictly_improved')
    proof=guard(result,legacy,True);guards['original-savedGPU']=proof;reasons.extend('original-savedGPU:'+r for r in proof['rejection_reasons'])
    return dict(eligible=not reasons,rejection_reasons=reasons,guards=guards,min33_gains=gains)


def constrained_search(curves,baseline,controls,legacy,grid):
    require(set(curves)==set(GROUPS) and len(grid)<=501 and np.all(np.diff(grid)>0),'Fixed threshold/group protocol changed.')
    ledger=[];original=[];admissible=[]
    for i,tau in enumerate(grid):
        result=light_result(curves,baseline,i);gate=constraints(result,controls,legacy)
        fpr=all(result['groups'][g]['system_false_positive_rate']<=.05 for g in FEASIBLE)
        row=dict(index=i,system_threshold=float(tau),effective_F32_threshold=float(np.float32(tau)),min33=result['min33'],old5_mean_f1=result['old5_mean_f1'],original_FPR_feasible=fpr,eligible=gate['eligible'],rejection_reasons=gate['rejection_reasons']);ledger.append(row)
        if fpr and result['min33'] is not None:original.append(i)
        if gate['eligible']:admissible.append(i)
    require(original,'No defined original FPR-feasible threshold.')
    key=lambda i:(ledger[i]['min33'],ledger[i]['old5_mean_f1'],-float(grid[i]))
    original_best=max(original,key=key);best=max(admissible,key=key) if admissible else None
    return best,dict(grid_thresholds=len(grid),threshold_budget=501,admissible_thresholds=len(admissible),original_best_index=original_best,original_threshold=float(grid[original_best]),original_min33=ledger[original_best]['min33'],original_old5_mean_f1=ledger[original_best]['old5_mean_f1'],selected_index=best,no_fallback=True),ledger


def group_indices(rows):
    groups={'old-'+p:('old',np.flatnonzero(rows['old'].profile.eq(p))) for p in PROFILES}
    for bank,name,frame in [('old','old',rows['old'][rows['old'].profile.eq('long-full')]),('c4','new-C4',rows['c4'])]:
        offset=frame.body_offset if 'body_offset' in frame else frame.token_ix-frame.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
        groups[name+'-long-full-tail']=(bank,frame.index[offset>=512].to_numpy())
        if bank=='c4':groups[name+'-long-full']=(bank,frame.index.to_numpy())
    groups[PUBLIC[0]]=('public',np.arange(len(rows['public'])))
    for name,dataset in zip(PUBLIC[1:],['oasst','toxicchat0124']):groups[name]=('public',np.flatnonzero(rows['public'].dataset.eq(dataset)))
    require(set(groups)==set(GROUPS) and all(len(ix)>0 for _,ix in groups.values()),'Missing original DEV group.');return groups


def update_binary(value,curve,i,tau):
    out=copy.deepcopy(value)
    for k in BINARY:
        if k=='system_threshold':out[k]=float(tau)
        elif k=='system_confusion':out[k]={v:int(curve[v][i]) for v in ['tp','fn','fp','tn']}
        elif k=='system_detection_by_role':out[k]={r:float(v[i]) if np.isfinite(v[i]) else None for r,v in curve['detection'].items()}
        else:out[k]=float(curve[k][i]) if np.isfinite(curve[k][i]) else None
    return out


def evaluate(baseline,curves,descriptive,i,tau):
    out=copy.deepcopy(baseline);out['system_threshold']=float(tau)
    out['groups']={g:update_binary(baseline['groups'][g],curves[g],i,tau) for g in GROUPS}
    out['separate_bank_descriptive']={b:update_binary(baseline['separate_bank_descriptive'][b],descriptive[b],i,tau) for b in BANKS}
    out['pooled_descriptive_only']=update_binary(baseline['pooled_descriptive_only'],descriptive['pooled'],i,tau)
    light=light_result(curves,baseline,i);out['min33']=light['min33'];out['old5_mean_f1']=light['old5_mean_f1'];return out


def fixed_spec(auditor_source=None):
    return dict(source=Path(__file__).resolve(),design=DESIGN,trainer_source=HERE/'systemness_complementary_head_readout.py',trainer_out=TRAINER,audit=AUDIT,auditor_source=auditor_source,
        pins=dict(source=SOURCE_SHA,protocol=PROTOCOL_SHA,complete=COMPLETE_SHA,auditor=AUDITOR_SHA,audit=AUDIT_SHA,model=MODEL_SHA,design=DESIGN_SHA),
        rows_paths=dict(old=Path('/dev/shm/role-analysis-top8-heads/rows.parquet'),c4=Path('/dev/shm/role-analysis-long-augmentation-heads/rows.parquet'),public=Path('/dev/shm/role-analysis-natural-dev-heads/rows.parquet')),
        probability_paths={b:BASE/'systemness-expansion/cache/complementary-head-readout'/f'new-{b}-validation.npz' for b in BANKS})


def pin_gate():
    require(all(valid_sha(v) for v in [SOURCE_SHA,PROTOCOL_SHA,COMPLETE_SHA,AUDITOR_SHA,AUDIT_SHA,MODEL_SHA,DESIGN_SHA]),'Final dependency pins unknown; formal IO forbidden.')


def collect_static(spec):
    """Metadata plus portable-byte hash only: never open p or row-table files."""
    pins=spec['pins'];require(set(pins)=={'source','protocol','complete','auditor','audit','model','design'} and all(valid_sha(v) for v in pins.values()),'Final calibration dependency pins unknown; formal IO forbidden.')
    files={}
    def bind(path,want=None,audited=False):
        path=clean(path);got=digest(path);require(want is None or want==got,'Closed SHA changed: '+str(path))
        if audited:require(audit['provenance_sha256'].get(str(path))==got,'File absent from completed independent audit: '+str(path))
        files[str(path)]=got;return got
    bind(spec['audit'],pins['audit']);audit=read(spec['audit'])
    require(audit['stage']=='actual_completed_complementary_head_readout_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False and audit['gpu_used'] is False and audit['auditor_source_sha256']==pins['auditor'],'Completed independent DEV certificate required.')
    audsrc=clean(spec['auditor_source'] or audit['auditor_source_path']);bind(audsrc,pins['auditor']);bind(spec['source']);bind(spec['design'],pins['design']);bind(spec['trainer_source'],pins['source'],True)
    folder=clean(spec['trainer_out']);require(read(folder/'status.json')['stage']=='validation_complete','Upstream training must be terminal complete.');bind(folder/'status.json',audited=True)
    bind(folder/'complete.json',pins['complete'],True);done=read(folder/'complete.json');bind(folder/'protocol.json',pins['protocol'],True)
    require(audit['trainer_source_sha256']==pins['source'] and audit['trainer_protocol_sha256']==pins['protocol'] and audit['trainer_complete_sha256']==pins['complete'] and done['source_sha256']==pins['source'] and done['tests_read'] is False and done['primary_changed'] is False and set(done['artifact_sha256'])==set(ARTIFACTS) and set(done['candidate_sha256'])=={'new','matched'},'Upstream completed source/protocol differs.')
    for name,want in done['artifact_sha256'].items():bind(folder/name,want,True)
    for arm,want in done['candidate_sha256'].items():require(arm in ['new','matched'],'Unexpected arm.');bind(folder/'candidates'/f'{arm}.json',want,True)
    model=folder/'models/new.npz';bind(model,pins['model'],True);new=read(folder/'candidates/new.json');bind(folder/'training/new.json',new['training_sha256'],True)
    require(new['model_sha256']==audit['model_sha256']['new']==pins['model'] and new['tests_read'] is False,'The fixed audited model changed.')
    records=read(folder/'records.json');index={r['record']:r for r in records};zero=read(folder/'zero-controls.json');controls=read(folder/'controls.json');selection=read(folder/'selection.json');decl=read(folder/'protocol.json')
    # File/source/data hashes stay exact. Independent metric reprints alone
    # allow the already established 1e-12 F64 tolerance, never schema changes.
    metadata_close(records,audit['fourteen_records'],'fourteen_records');metadata_close(zero['records'],{r['record']:r['validation'] for r in audit['zero_records']},'zero_records')
    require(len(records)==14 and set(index)==set(RECORDS) and set(zero['records'])=={'zero-oldmean','zero-max33'} and selection['promoted'] is False and selection['model_sha256']==pins['model'],'Read-only original14+zero/negative selection changed.')
    require(digest(folder/'records.json')==audit['trainer_records_sha256'] and digest(folder/'selection.json')==audit['trainer_selection_sha256'] and decl['protocol']['groups']==GROUPS and decl['protocol']['feasible']==FEASIBLE and decl['protocol']['promotion']['strict_controls']==STRICT and decl['protocol']['promotion']['guard_controls']==GUARDS,'Original grid/groups/controls differ.')
    source_rows={};probs={}
    for b in BANKS:
        rp=clean(spec['rows_paths'][b]);pp=clean(spec['probability_paths'][b]);rh=audit['provenance_sha256'].get(str(rp));ph=audit['probability_bindings'].get(str(pp))
        require(valid_sha(rh) and valid_sha(ph) and ph==new['probability_sha256'][b],'Explicit row/p certificates missing from independent audit.')
        source_rows[b]=dict(path=str(rp),sha256=rh,sample_ix_sha256=decl['static']['sample_ix_sha256'][b],rows=COUNTS[b]);probs[b]=dict(path=str(pp),sha256=ph)
    metadata_close(audit['selection'],{k:selection[k] for k in audit['selection']},'selection')
    base=index['new-max33']['validation'];guard_values={g:(zero['records'][g] if g in zero['records'] else index[g]['validation']) for g in GUARDS}
    return dict(source_sha256=digest(spec['source']),source_path=str(clean(spec['source'])),design_sha256=pins['design'],files_sha256=files,auditor_source_path=str(audsrc),model_path=str(model),model_sha256=pins['model'],upstream_actual_audit_sha256=pins['audit'],upstream_complete_sha256=pins['complete'],upstream_protocol_sha256=pins['protocol'],
        source_rows=source_rows,probabilities=probs,baseline_validation=base,controls=guard_values,legacy=controls['original_savedGPU_validation'],original_selection=selection,
        original_records_sha256=digest(folder/'records.json'),zero_controls_sha256=digest(folder/'zero-controls.json'),all14_plus_zero_untouched=True,
        packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow']},probability_files_opened=False,row_tables_materialized=False,raw_features_needed=False)


def declare(auditor_source=None):
    pin_gate()
    value=dict(protocol=protocol(),static=collect_static(fixed_spec(auditor_source)))
    require(not (OUT/'complete.json').exists() or (OUT/'protocol.json').is_file(),'Unowned completed calibration.')
    immutable_json(OUT/'protocol.json',value);return value


def verify_declaration():
    pin_gate()
    require((OUT/'protocol.json').is_file(),'Root must --declare before any DEV probability reads.')
    value=read(OUT/'protocol.json');expected=dict(protocol=protocol(),static=collect_static(fixed_spec(value['static']['auditor_source_path'])))
    require(value==expected,'Frozen calibration source/model/certificate/protocol changed.');return value


def load_development(static):
    """Actual-only fixed DEV row descriptors and three already audited p arrays."""
    import pyarrow.parquet as pq
    rows={};p={};bindings={}
    fields=['cache_ix','sample_ix','split','wrapper','role','profile','body_offset','token_ix','prompt_ix','message_ix','dataset','group_id']
    for b in BANKS:
        rp=static['source_rows'][b];path=clean(rp['path']);require(digest(path)==rp['sha256'],'DEV row bytes changed.');bindings[str(path)]=rp['sha256']
        columns=[n for n in fields if n in pq.ParquetFile(path).schema.names];table=pd.read_parquet(path,columns=columns)
        require(np.array_equal(table.cache_ix,np.arange(len(table))) and table.sample_ix.is_unique and set(table.split)=={'fit','validation'} and table.wrapper.eq('native').all(),'DEV row ownership/scope changed.')
        frame=table[table.split.eq('validation')].copy().reset_index(drop=True)
        require(len(frame)==rp['rows'] and array_sha(frame.sample_ix.to_numpy())==rp['sample_ix_sha256'] and set(frame.role)<=set(ROLES),'DEV validation sample order changed.')
        if b=='public':require(set(frame.role)=={'system','user','assistant'},'Public validation must remain SUA-only.')
        rows[b]=frame;item=static['probabilities'][b];pp=clean(item['path']);require(digest(pp)==item['sha256'],'DEV p bytes changed.');bindings[str(pp)]=item['sha256']
        with np.load(pp,allow_pickle=False) as z:
            require(set(z.files)=={'p','sample_ix'} and np.array_equal(z['sample_ix'],frame.sample_ix.to_numpy()),'DEV p sample/namespace changed.');v=z['p'].copy()
        require(v.shape==(len(frame),5) and v.dtype==np.float32 and np.isfinite(v).all() and (v>=0).all() and (v<=1.00001).all() and np.allclose(v.sum(1),1,atol=2e-5,rtol=0),'Invalid existing five-class DEV p.');p[b]=v
    return rows,p,bindings


def build_curves(rows,p,grid):
    labels={b:rows[b].role.map({r:i for i,r in enumerate(ROLES)}).to_numpy(np.int64) for b in BANKS}
    curves={g:binary_curves(p[b][ix,0],labels[b][ix],grid) for g,(b,ix) in group_indices(rows).items()}
    descriptive={b:binary_curves(p[b][:,0],labels[b],grid) for b in BANKS};descriptive['pooled']=binary_curves(np.concatenate([p[b][:,0] for b in BANKS]),np.concatenate([labels[b] for b in BANKS]),grid)
    return curves,descriptive


def same_binary(a,b):
    for k in BINARY:
        if k in ['system_detection_by_role','system_confusion']:require(a[k]==b[k],'Original binary metric changed: '+k)
        elif a[k] is None or b[k] is None:require(a[k] is b[k],'Original undefined metric changed.')
        else:require(abs(a[k]-b[k])<=TOL,'Original binary metric changed: '+k)


def run_actual():
    declaration=verify_declaration();static=declaration['static'];rows,p,bindings=load_development(static);grid=threshold_grid(p);curves,descriptive=build_curves(rows,p,grid)
    best,proof,ledger=constrained_search(curves,static['baseline_validation'],static['controls'],static['legacy'],grid)
    old=static['baseline_validation'];original=proof['original_best_index'];require(float(grid[original])==old['system_threshold'] and proof['original_min33']==old['min33'] and proof['original_old5_mean_f1']==old['old5_mean_f1'],'Original audited unconstrained max33 does not reproduce.')
    restored=evaluate(old,curves,descriptive,original,grid[original])
    for key in GROUPS:same_binary(restored['groups'][key],old['groups'][key])
    for key in BANKS:same_binary(restored['separate_bank_descriptive'][key],old['separate_bank_descriptive'][key])
    same_binary(restored['pooled_descriptive_only'],old['pooled_descriptive_only'])
    original_gate=constraints(light_result(curves,old,original),static['controls'],static['legacy'])
    require(original_gate['rejection_reasons']==static['original_selection']['rejection_reasons'] and original_gate['guards']==static['original_selection']['guards'] and original_gate['min33_gains']==static['original_selection']['min33_gains'],'Original failed max33 guards do not reproduce.')
    selected=evaluate(old,curves,descriptive,best,grid[best]) if best is not None else None
    gate=constraints(light_result(curves,old,best),static['controls'],static['legacy']) if best is not None else None
    require(gate is None or gate['eligible'],'Selected inadmissible threshold.')
    selection=dict(candidate='new-guard-max33',eligible=best is not None,no_fallback=True,system_threshold=float(grid[best]) if best is not None else None,validation=selected,gate=gate,grid_proof=proof,model_path=static['model_path'],model_sha256=static['model_sha256'],source_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),upstream_actual_audit_sha256=static['upstream_actual_audit_sha256'],original_trial_unchanged=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)
    paths=['threshold-ledger.json','selection.json','validation-metrics.csv','report.txt']
    if (OUT/'complete.json').exists():
        complete=read(OUT/'complete.json');require(complete['source_sha256']==digest(__file__) and complete['protocol_sha256']==digest(OUT/'protocol.json') and complete['artifact_sha256']=={name:digest(OUT/name) for name in paths} and read(OUT/'selection.json')==selection and read(OUT/'threshold-ledger.json')==ledger,'Completed calibration changed; no overwrite.');return complete
    require(not any((OUT/name).exists() for name in paths),'Partial calibration output exists; inspect rather than overwrite.')
    immutable_json(OUT/'threshold-ledger.json',ledger);immutable_json(OUT/'selection.json',selection)
    flat=[]
    if selected is not None:
        for group,value in dict(selected['groups'],pooled_descriptive_only=selected['pooled_descriptive_only']).items():flat.append(dict(record='new-guard-max33',group=group,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(flat,columns=None if flat else ['record','group']).to_csv(OUT/'validation-metrics.csv',index=False)
    text=['DEV-only constrained global calibration; no new weights or five-class accuracy change.',protocol()['limitation'],f"Original threshold: {old['system_threshold']}; original min33: {old['min33']}",f"Admissible thresholds: {proof['admissible_thresholds']}/{len(grid)}",f"Eligible: {best is not None}; threshold: {selection['system_threshold']}",f"min33: {selected['min33'] if selected else None}", 'No fallback. Original14+zero, failed selection and primary remain unchanged.']
    (OUT/'report.txt').write_text('\n'.join(text)+'\n');verify_declaration()
    complete=dict(stage='completed_guard_constrained_development_calibration',source_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),artifact_sha256={name:digest(OUT/name) for name in paths},probability_bindings={v['path']:v['sha256'] for v in static['probabilities'].values()},data_bindings=bindings,eligible=best is not None,training_updates=0,gpu_used=False,tests_read=False,primary_changed=False,new_independent_confirmation_required=True)
    immutable_json(OUT/'complete.json',complete);return complete


def preflight():
    """Only manufactured arrays and files; never call fixed_spec/actual/declaration."""
    import ast
    from unittest.mock import patch
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-guard-calibration-preflight-',dir='/tmp'));start=digest(__file__);checks=[]
    def check(name,value=True):require(value,'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,KeyError,FileNotFoundError):checks.append(name);return
        raise RuntimeError('Expected fixture refusal: '+name)
    # Tied F32 scores, including upward F64 alias, versus direct F32 consumer.
    score=np.asarray([0.,.5,.5,np.nextafter(np.float32(.5),np.float32(1)),1.,.5,.1,.7,.9,1.],np.float32);y=np.asarray([0,0,0,0,0,1,2,3,4,1]);grid=np.asarray([0.,.1,.5,np.nextafter(.5,np.inf),float(np.nextafter(np.float32(.5),np.float32(1))),1.,1.000001]);curve=binary_curves(score,y,grid)
    for i,tau in enumerate(grid):
        detected=score>=float(tau);truth=y==0;tp=int((truth&detected).sum());fp=int((~truth&detected).sum());fn=int((truth&~detected).sum());tn=int((~truth&~detected).sum())
        for k,w in dict(tp=tp,fp=fp,fn=fn,tn=tn,system_precision=tp/(tp+fp) if tp+fp else np.nan,system_recall=tp/5,system_balanced_accuracy=(tp/5+tn/5)/2,system_f1=2*tp/(2*tp+fp+fn),system_false_positive_rate=fp/5,system_accuracy=(tp+tn)/10).items():check('direct-tied-F32-'+str(i)+'-'+k,np.array_equal(np.asarray(curve[k][i]),np.asarray(w),equal_nan=True))
    fakep={b:np.tile(np.asarray([[.2,.2,.2,.2,.2]],np.float32),(8,1)) for b in BANKS};g=threshold_grid(fakep);check('same-original-grid-budget',len(g)<=501 and np.array_equal(g,np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate([v[:,0] for v in fakep.values()]),np.linspace(0,1,301)),1.000001])))
    def metric(f1=.98):return dict(system_f1=f1,accuracy=.99,system_false_positive_rate=.01,system_auroc=.987,system_average_precision=.981,recall={r:(None if r in ['cot','tool'] else .9) for r in ROLES},confusion=[[1]*5 for _ in ROLES],macro_f1=.9,nll=.1)
    baseline=dict(groups={g:metric() for g in GROUPS},old5_mean_f1=.98,min33=.95,system_threshold=.5,separate_bank_descriptive={b:metric() for b in BANKS},pooled_descriptive_only=metric())
    control=copy.deepcopy(baseline);control['min33']=.90;controls={g:copy.deepcopy(control) for g in GUARDS};legacy=copy.deepcopy(control)
    gs=np.asarray([.4,.5,.6]);curves={g:dict(system_precision=np.array([.94,.95,.93]),system_recall=np.array([.94,.95,.93]),system_balanced_accuracy=np.array([.94,.95,.93]),system_f1=np.array([.99,.99,.99]),system_false_positive_rate=np.array([.01,.01,.01])) for g in GROUPS};curves['old-natural']['system_f1']=np.array([.97,.955,.99])
    best,proof,ledger=constrained_search(curves,baseline,controls,legacy,gs);check('constrained-optimum-before-ranking',best==0 and proof['original_best_index']==1 and ledger[1]['eligible'] is False and any('old4_system_f1_loss' in r for r in ledger[1]['rejection_reasons']))
    check('exact-five-strict-nine-guards',set(constraints(light_result(curves,baseline,0),controls,legacy)['guards'])==set(GUARDS)|{'original-savedGPU'} and len(STRICT)==5)
    bad=copy.deepcopy(curves)
    for v in bad.values():v['system_recall'][:]=.89
    best,proof,_=constrained_search(bad,baseline,controls,legacy,gs);check('no-admissible-no-fallback',best is None and proof['admissible_thresholds']==0)
    result=light_result(curves,baseline,0);blocked=copy.deepcopy(controls);blocked[STRICT[0]]['min33']=result['min33']-1e-13;check('strict-gain-tolerance-kept',not constraints(result,blocked,legacy)['eligible'])
    bad=copy.deepcopy(curves);bad[FEASIBLE[0]]['system_false_positive_rate'][:]=.05+5e-13;best,_,_=constrained_search(dict(bad,**{FEASIBLE[0]:dict(bad[FEASIBLE[0]],system_false_positive_rate=np.array([.01,.05+5e-13,.01]))}),baseline,controls,legacy,gs);check('original-exact-FPR-no-new-tolerance',best!=1)
    tie=copy.deepcopy(curves)
    for v in tie.values():
        for key in v:v[key][:]=v[key][0]
    tg=np.asarray([.5,np.nextafter(.5,np.inf),.6]);best,_,_=constrained_search(tie,baseline,controls,legacy,tg);check('F32-alias-preserves-original-F64-low-tie',best==0 and np.float32(tg[0])==np.float32(tg[1]))
    rich=binary_curves(score,y,grid);value=update_binary(metric(),rich,2,grid[2]);check('only-threshold-dependent-fields-change',all(value[k]==metric()[k] for k in metric() if k not in BINARY) and value['system_confusion']['tp']==int(rich['tp'][2]))
    # Undefined role and precision stay N/A; never invented CoT/tool support.
    suay=np.asarray([0,0,1,1,3,3]);suascore=np.asarray([.8,.9,.1,.2,.3,.4],np.float32);suac=binary_curves(suascore,suay,np.asarray([.5,1.000001]));check('public-unsupported-role-and-zero-detection-NAs',update_binary(metric(),suac,1,1.000001)['system_precision'] is None and update_binary(metric(),suac,0,.5)['system_detection_by_role']['cot'] is None)
    # Unknown dependency gate rejects before any file read. No actual paths used.
    with patch.dict(globals(),digest=lambda p:(_ for _ in ()).throw(LookupError('Forbidden IO'))):reject('unknown-pins-before-any-IO',lambda:collect_static({'pins':dict(source=None)}))
    with patch.dict(globals(),SOURCE_SHA=None,digest=lambda p:(_ for _ in ()).throw(LookupError('Forbidden IO'))):
        reject('public-declare-unknown-pins-before-IO',declare);reject('public-actual-unknown-pins-before-IO',run_actual)
    # Declare/verify are metadata-only: their call trees do not touch np.load,
    # Parquet readers, probability hash loop or model/feature predictors.
    tree=ast.parse(Path(__file__).read_text());names={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    segment=ast.get_source_segment(Path(__file__).read_text(),names['collect_static']);check('declaration-certifies-data-without-opening-p-or-rows','np.load' not in segment and 'read_parquet' not in segment and "digest(pp)" not in segment and "digest(rp)" not in segment)
    check('no-production-verifier-or-predictor-import',not any(isinstance(n,ast.Import) and any(a.name.startswith('systemness_') for a in n.names) for n in ast.walk(tree)))
    check('original-policy-cardinality',len(GROUPS)==11 and len(FEASIBLE)==9 and len(GUARDS)==8 and len(STRICT)==5 and len(RECORDS)==14)
    metadata_close({'v':.9,'i':1,'b':True,'n':None},{'v':.9+1e-16,'i':1,'b':True,'n':None});check('independent-metric-last-bit-tolerance-only')
    for value,name in [(dict(v=.9+2e-12,i=1,b=True,n=None),'F64-over-tolerance'),(dict(v=.9,i=1.,b=True,n=None),'integer-type'),(dict(v=.9,i=1,b=1,n=None),'boolean-type'),(dict(v=.9,i=1,b=True),'schema')]:reject('independent-metadata-rejects-'+name,lambda value=value:metadata_close({'v':.9,'i':1,'b':True,'n':None},value))
    check('final-source-stable',digest(__file__)==start)
    report=dict(stage='synthetic_CPU_guard_constrained_calibration',source_sha256=start,checks=len(checks),check_names=checks,actual_run=False,actual_probability_model_feature_or_rows_read=False,actual_metadata_read=False,gpu_used=False,training_updates=0,new_test_or_source_read=False)
    atomic_json(folder/'report.json',report);print(json.dumps(dict(report=str(folder/'report.json'),report_sha256=digest(folder/'report.json'),source_sha256=start,checks=len(checks))));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--declare',action='store_true');group.add_argument('--actual',action='store_true');parser.add_argument('--auditor-source',type=Path,help='Optional archived copy of the exact pinned independent auditor source; declare only.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.declare:declare(args.auditor_source);print(json.dumps(dict(protocol_sha256=digest(OUT/'protocol.json'),declared=True)))
    elif args.actual:require(args.auditor_source is None,'Auditor source is frozen at declaration.');print(json.dumps(run_actual()))
    else:parser.print_help()
if __name__=='__main__':main()
