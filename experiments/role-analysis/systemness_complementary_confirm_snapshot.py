"""Conditional closed snapshot for complementary640 independent confirmation.

Default help; --preflight uses synthetic CPU/tmp only. Unknown final trainer,
protocol, completion and actual-audit pins reject before formal artifact reads.
No production verifier/predictor import, source selection, LM, test or raw-bank
read occurs here. Future test manifests never enter an old completed inventory.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-complementary-confirmation-preparation'
DESIGN=OUT/'root-prospective-design.json'
DESIGN_SHA='6999ea73bb0207ef3d6a00610d96e47de9e05cca38bac5b5ef08bbd2c17bd824'
TRAINER=BASE/'systemness-complementary-head-readout'
TRAINER_SOURCE=HERE/'systemness_complementary_head_readout.py'
TRAINER_SOURCE_SHA=None
TRAINER_PROTOCOL_SHA=None
TRAINER_COMPLETE_SHA=None
APPROVED_AUDITOR_SOURCE_SHA=None
ACTUAL_AUDIT_PATH=None
ACTUAL_AUDIT_SHA=None
AUDIT_STAGE='actual_completed_complementary_head_readout_audit'
OLD_HEAD=BASE/'systemness-head-probes'
OLD_MODEL_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
OLD_THRESHOLD=.620969696969697
EXTRA_OUT=BASE/'systemness-complementary-head-features'
CHECKPOINT=Path('/dev/shm/role-analysis-gptoss120b')
SCALER=BASE/'systemness-expansion/cache/complementary-head-readout/standardization.npz'
ORIGINAL=[458,462,2045,460,459,2047,456,997]
ADDED=[247,2248]
CAPTURE=ORIGINAL+ADDED
ROLES=['system','user','cot','assistant','tool']
RECORD_TO_MODEL={'new-max33':'new','matched-max33':'matched','oldhead-max33':'old-head','oldhead-original':'old-head'}
SOURCE_RECORD={name:('oldhead-oldmean' if name=='oldhead-original' else name) for name in RECORD_TO_MODEL}
DEVELOPMENT_RECORDS=[m+'-'+o for m in ['new','matched','oldhead'] for o in ['oldmean','max33']]+['ensemble-original','ensemble-max33']+[m+'-'+o for m in ['naturalCE','priorBinary','priorLongMix'] for o in ['oldmean','max33']]
STRICT=['matched-max33','oldhead-max33','ensemble-max33','zero-max33','priorBinary-max33']
GUARDS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33','priorBinary-max33']
ARTIFACTS=['protocol.json','search-plan.json','scaler-record.json','controls.json','zero-controls.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt']
EXCLUSION_NAMES=['feature-search/fresh/document-keys.json','systemness-expansion/document-keys.json','systemness-long-full/document-keys.json','systemness-long-full/confirmation/document-keys.json','systemness-head-confirmation/document-keys.json','systemness-long-augmentation/document-keys.json']
ORIGINAL249=[BASE/'training_inputs.parquet',BASE.parent/'local-gptoss20b/raw_training_data.json']
FUTURE_DATA=[(BASE/'systemness-complementary-confirmation',BASE/'systemness-expansion/cache/complementary-confirmation'),(BASE/'systemness-complementary-natural-confirmation',BASE/'systemness-expansion/cache/complementary-natural-confirmation')]
VIEWS={'new':dict(kind='binary-residual-640',dimensions=640,view='capture640',selected_head_ids=CAPTURE),
    'matched':dict(kind='binary-residual-640',dimensions=640,view='concatenate(capture[:512],capture[:128]); ignores real capture[512:]',selected_head_ids=ORIGINAL+ORIGINAL[:2]),
    'old-head':dict(kind='mlp',dimensions=512,view='capture[:512].contiguous()',selected_head_ids=ORIGINAL)}


def require(value,message):
    if not bool(value):raise RuntimeError(message)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def clean_path(path):
    raw=str(path);require(raw!='/workspace' and not raw.startswith('/workspace/'),'Forbidden /workspace path before filesystem access.')
    value=Path(os.path.abspath(raw));require(str(value)!='/workspace' and not str(value).startswith('/workspace/'),'Forbidden /workspace path.');return value

def digest(path):
    h=hashlib.sha256()
    with clean_path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(clean_path(path).read_text())
def equal(a,b):
    if isinstance(a,dict) or isinstance(b,dict):return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) or isinstance(b,list):return isinstance(a,list) and isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,bool) or isinstance(b,bool):return type(a)==type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return np.isfinite(a) and np.isfinite(b) and abs(a-b)<=1e-12
    return a==b

def pin_gate():
    """Pure strings only: before stat/hash/JSON/NPZ on any formal action."""
    require(all(valid_sha(v) for v in [TRAINER_SOURCE_SHA,TRAINER_PROTOCOL_SHA,TRAINER_COMPLETE_SHA,APPROVED_AUDITOR_SOURCE_SHA,ACTUAL_AUDIT_SHA]) and ACTUAL_AUDIT_PATH is not None,
        'Final640 trainer/protocol/complete/independent-auditor/actual-certificate pins unknown; actual forbidden.')
    require(str(ACTUAL_AUDIT_PATH)!='/workspace' and not str(ACTUAL_AUDIT_PATH).startswith('/workspace/'),'Forbidden actual-audit path.')


def protocol_spec():
    value=dict(version=1,experiment='closed-complementary640-confirmation-snapshot',root_design_sha256=DESIGN_SHA,
        gate='new-max33 promoted, absolute min33>=.90, five strict/eight guards/original savedGPU, completed independently audited added128 bank and640 development; no fallback.',
        approved_independent_auditor_source_sha256=APPROVED_AUDITOR_SOURCE_SHA,physical_models=list(VIEWS),record_to_model=RECORD_TO_MODEL,source_records=SOURCE_RECORD,
        original_old_head_sha256=OLD_MODEL_SHA,original_threshold=OLD_THRESHOLD,capture_head_ids=CAPTURE,capture_dimensions=640,physical_feature_views=VIEWS,
        raw_artifacts='Only certified SHA/shape/size/dtype provenance; no raw development feature/probability load or runtime dependency.',
        closure='Audited models/scalers/certificates plus independently fresh-frozen explicit source-exclusion/execution-code files before any new source IDs/body; no rglob, growing inventory or old dynamic verifier.',
        reserved_sources=dict(c4=528,general=400,true1024=128,paired_general_long=64,window=[12303,20494],natural=128,oasst=64,toxicchat0124=64,system_families=32),
        fixed_plot_pairs=dict(c4=17,natural=13),normal_system_tags=True,new_test_source_body_read=False,test_probabilities_read=False,primary_changed=False,
        limitation='Role classification, not behavioral system compliance; both genuinely freshC4 and fresh natural confirmation are required. Public replies were not120B generations under assigned system wording; CoT/tool natural truth absent.')
    return copy.deepcopy(value)


def model_paths():return {'new':TRAINER/'models/new.npz','matched':TRAINER/'models/matched.npz','old-head':OLD_HEAD/'models/joint-mlp128.npz'}

def audit_input(value=None):
    pin_gate()
    if value is None:value=dict(audit_path=ACTUAL_AUDIT_PATH)
    if isinstance(value,(str,Path)):value=dict(audit_path=value)
    require(isinstance(value,dict) and clean_path(value['audit_path'])==clean_path(ACTUAL_AUDIT_PATH),'Actual audit must be the exact root-pinned certificate.')
    path=clean_path(value['audit_path']);require(digest(path)==ACTUAL_AUDIT_SHA,'Pinned actual audit bytes changed.');audit=read(path)
    source=clean_path(value.get('audit_source_path',audit['auditor_source_path']))
    require(audit['stage']==AUDIT_STAGE and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False and audit['gpu_used'] is False
        and digest(source)==audit['auditor_source_sha256']==APPROVED_AUDITOR_SOURCE_SHA,'A completed independently audited640 development certificate is required.')
    provenance=audit['provenance_sha256'];require(provenance and all(valid_sha(v) for v in provenance.values()),'Missing actual-audit provenance.')
    return value,path,source,audit,{str(clean_path(p)):v for p,v in provenance.items()}


def records_and_gate(audit):
    pin_gate();require(digest(TRAINER_SOURCE)==TRAINER_SOURCE_SHA and digest(TRAINER/'protocol.json')==TRAINER_PROTOCOL_SHA and digest(TRAINER/'complete.json')==TRAINER_COMPLETE_SHA,'Pinned final trainer/source/protocol/completion changed.')
    require(read(TRAINER/'status.json')['stage']=='validation_complete','Trainer not terminal complete.')
    done,decl,plan,selection,records=[read(TRAINER/n) for n in ['complete.json','protocol.json','search-plan.json','selection.json','records.json']]
    static=decl['static'];require(done['source_sha256']==static['source_sha256']==TRAINER_SOURCE_SHA and done['signature']==plan['signature'] and plan['protocol']==decl['protocol']
        and done['tests_read'] is False and done['primary_changed'] is False and done['new_independent_confirmation_required'] is True and set(done['artifact_sha256'])==set(ARTIFACTS) and set(done['candidate_sha256'])=={'new','matched'},'Final640 completion ownership differs.')
    for field,path in [('trainer_source_sha256',TRAINER_SOURCE),('trainer_protocol_sha256',TRAINER/'protocol.json'),('trainer_complete_sha256',TRAINER/'complete.json'),('trainer_selection_sha256',TRAINER/'selection.json'),('trainer_records_sha256',TRAINER/'records.json')]:require(audit[field]==digest(path),'Actual audit names another '+field)
    for name,want in done['artifact_sha256'].items():require(digest(TRAINER/name)==want,'Closed trainer artifact changed: '+name)
    index={r['record']:r for r in records};require(len(records)==14 and set(index)==set(DEVELOPMENT_RECORDS),'Exactly14 audited development records required.')
    require(equal(audit['fourteen_records'],records),'Actual audit record metrics differ from trainer.')
    proto=decl['protocol'];require(proto['promotion']['candidate']=='new-max33' and proto['promotion']['absolute_min33']==.9 and proto['promotion']['strict_controls']==STRICT and proto['promotion']['guard_controls']==GUARDS
        and proto['promotion']['mandatory_original_savedGPU'] is True and proto['original_head_ids']==ORIGINAL and proto['added_head_ids']==ADDED and proto['dimensions']==640 and proto['parameters']==20545,'Declared640 geometry or fixed gates differ.')
    require(selection['candidate']=='new-max33' and selection['promoted'] is True and selection['no_fallback'] is True and selection['rejection_reasons']==[]
        and selection['primary_changed'] is False and selection['tests_read'] is False and selection['new_independent_confirmation_required'] is True
        and set(selection['guards'])==set(GUARDS)|{'original-savedGPU'} and set(selection['min33_gains'])==set(STRICT)
        and all(g['eligible'] is True and g['rejection_reasons']==[] for g in selection['guards'].values())
        and all(np.isfinite(v) and v>1e-12 for v in selection['min33_gains'].values()),'Only a promoted absolute90/five-strict/eight-guard/originalGPU candidate can be snapshotted.')
    require(all(equal(selection[k],v) for k,v in audit['selection'].items()),'Actual audit promotion proof differs.')
    candidate=index['new-max33']['validation'];require(np.isfinite(candidate['min33']) and candidate['min33']>=.9,'Absolute min33 below90.')
    zero=read(TRAINER/'zero-controls.json')['records']
    for name,gain in selection['min33_gains'].items():
        control=zero[name] if name in zero else index[name]['validation'];require(equal(gain,candidate['min33']-control['min33']),'Strict gain not the audited candidate/control difference.')
    thresholds={record:float(index[source]['validation']['system_threshold']) for record,source in SOURCE_RECORD.items()}
    require(all(np.isfinite(v) and 0<=v<=1.00001 for v in thresholds.values()) and thresholds['oldhead-original']==OLD_THRESHOLD and selection['system_threshold']==thresholds['new-max33'],'Four frozen thresholds differ.')
    for arm in ['new','matched']:
        path=TRAINER/'candidates'/f'{arm}.json';r=read(path);training=TRAINER/'training'/f'{arm}.json'
        require(digest(path)==done['candidate_sha256'][arm]==audit['trainer_model_record_sha256'][arm] and digest(training)==r['training_sha256']
            and r['model_sha256']==audit['model_sha256'][arm] and r['source_sha256']==TRAINER_SOURCE_SHA and r['tests_read'] is False,'Audited candidate/history/model ownership changed.')
    require(selection['source_sha256']==TRAINER_SOURCE_SHA and selection['records_sha256']==digest(TRAINER/'records.json') and selection['model_sha256']==audit['model_sha256']['new']
        and clean_path(selection['model_path'])==clean_path(model_paths()['new']),'Selected640 portable identity changed.')
    return done,decl,plan,selection,index,thresholds


def portable_metadata(path,name,old,scalers,plan,static):
    """NPZ-only geometry/scaler/base validation; no torch or inference."""
    pin_gate()
    with np.load(clean_path(path),allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));mean,scale=z['mean'].copy(),z['scale'].copy();weights={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')}
    require(meta['roles']==ROLES and meta['selected_head_ids']==VIEWS[name]['selected_head_ids'],'Portable role or per-model head/view order differs.')
    dim=VIEWS[name]['dimensions'];require(meta['kind']==VIEWS[name]['kind'] and meta['dimensions']==dim and mean.shape==scale.shape==(dim,) and mean.dtype==scale.dtype==np.float32
        and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Malformed portable geometry/scaler.')
    if name=='old-head':
        expected={'encoder.0.weight':(128,512),'encoder.0.bias':(128,),'head.weight':(5,128),'head.bias':(5,)}
        require(meta['hidden']==128 and meta['heads']==8 and meta['head_dim']==64,'Original512 geometry differs.');expected_mean,expected_scale=scalers['old-head'];base=weights
    else:
        expected={'base.encoder.0.weight':(128,512),'base.encoder.0.bias':(128,),'base.head.weight':(5,128),'base.head.bias':(5,),'correction.0.weight':(32,640),'correction.0.bias':(32,),'correction.3.weight':(1,32),'correction.3.bias':(1,)}
        require(meta['base_dimensions']==512 and meta['base_hidden']==128 and meta['residual_hidden']==32 and meta['parameters']==20545 and meta['delta_formula']=='4*tanh(g)' and meta['base_model_sha256']==OLD_MODEL_SHA
            and meta['arm']==name and meta['signature']==plan['signature'] and meta['source_sha256']==TRAINER_SOURCE_SHA and meta['plan_sha256']==digest(TRAINER/'search-plan.json')
            and meta['initial_weight_sha256']==plan['initial_weight_sha256'] and meta['standardization_sha256']==plan['signature']['standardization_sha256'] and meta['tests_read'] is False,'Residual640 formula/arm/training provenance differs.')
        base={k[5:]:v for k,v in weights.items() if k.startswith('base.')};expected_mean,expected_scale=scalers[name]
        require(set(base)==set(old) and all(np.array_equal(v.view(np.uint32),old[k].view(np.uint32)) for k,v in base.items()),'Residual embedded base is not original frozen512.')
        require(meta['embedded_base_weight_sha256']==static['embedded_base_weight_sha256'] and meta['mean_sha256']==array_sha(mean) and meta['scale_sha256']==array_sha(scale),'Portable embedded array identity differs.')
    require(set(weights)==set(expected) and all(v.shape==expected[k] and v.dtype==np.float32 and np.isfinite(v).all() for k,v in weights.items()),'Missing,extra,wrong-shape or nonfinite portable weight array.')
    require(np.array_equal(mean.view(np.uint32),expected_mean.view(np.uint32)) and np.array_equal(scale.view(np.uint32),expected_scale.view(np.uint32)),'Portable scaler differs from original/arm fit-only certificate.')
    return dict(metadata_sha256=canonical(meta),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),weight_sha256={k:array_sha(v) for k,v in weights.items()},**copy.deepcopy(VIEWS[name])),base


def collect_material(audit_request=None):
    pin_gate();request,audit_path,audit_source,audit,provenance=audit_input(audit_request);done,decl,plan,selection,records,thresholds=records_and_gate(audit);static=decl['static']
    require(digest(DESIGN)==DESIGN_SHA,'Root reserved confirmation design changed.')
    raw=set(audit['probability_bindings'])|set(static['feature_files_sha256'])
    def raw_artifact(path):
        p=clean_path(path);return str(p) in raw or p.suffix in ['.f16','.npy'] or p.name.endswith('moments.npz') or p.name=='portable-fit-sample.npz'
    certificates={};runtime={};certified_raw={}
    def bind(path,want=None,runtime_file=False,audited=True):
        p=clean_path(path);key=str(p)
        if raw_artifact(p):
            require(want is not None and valid_sha(want),'Raw provenance requires an already-certified SHA.');certified_raw[key]=want;return
        got=digest(p);require(want is None or want==got,'Closed metadata/runtime hash changed: '+key)
        if audited:require(provenance.get(key)==got,'Actual audit did not bind required metadata/runtime file: '+key)
        (runtime if runtime_file else certificates)[key]=got
    # Audited explicit map only; raw feature/p arrays are never opened/stat'ed.
    for path,want in provenance.items():bind(path,want,runtime_file=Path(path).suffix=='.py')
    for path,want in static['closed_files_sha256'].items():bind(path,want,runtime_file=Path(path).suffix=='.py')
    bind(DESIGN,DESIGN_SHA,audited=False);bind(audit_path,ACTUAL_AUDIT_SHA,audited=False);bind(audit_source,APPROVED_AUDITOR_SOURCE_SHA,runtime_file=True,audited=False);bind(__file__,runtime_file=True,audited=False)
    for path in [TRAINER/n for n in ARTIFACTS]+[TRAINER/'status.json',TRAINER_SOURCE]+[TRAINER/f'{kind}/{arm}.json' for kind in ['candidates','training'] for arm in ['new','matched']]:bind(path)
    # Future source exclusions are prospectively frozen here, not claimed to be
    # direct inputs of the previous development-result audit. Only these eight
    # named inventory paths are exempted from that audit-membership requirement.
    future_exclusions={}
    for path in [BASE/n for n in EXCLUSION_NAMES]+ORIGINAL249:
        bind(path,audited=False);future_exclusions[str(clean_path(path))]=certificates[str(clean_path(path))]
    for name,want in static['dependencies_sha256'].items():bind(HERE/name,want,runtime_file=True)
    # All three portables and both explicit scaler archives are real runtime inputs.
    paths=model_paths();old_scaler=OLD_HEAD/'standardization.npz';bind(old_scaler,static['standardization_sha256'],True);bind(SCALER,plan['signature']['standardization_sha256'],True)
    with np.load(old_scaler,allow_pickle=False) as z:om,oscale=z['mean'].copy(),z['scale'].copy()
    require(om.shape==oscale.shape==(512,) and om.dtype==oscale.dtype==np.float32 and np.isfinite(om).all() and np.isfinite(oscale).all() and (oscale>0).all()
        and array_sha(om)==static['old512_mean_sha256'] and array_sha(oscale)==static['old512_scale_sha256'],'Original frozen512 scaler differs.')
    sr=read(TRAINER/'scaler-record.json');fit=sr['fit'];require(fit['fit_rows']==578090 and fit['fit_only'] is True and fit['natural_rows_used']==fit['validation_rows_used']==0 and fit['chunk']==4096
        and fit['floor']==1e-6 and fit['variance']=='Population F64 parallel-Welford' and fit['fit_sample_ix_sha256']==static['sample_ix_sha256']['fit'] and equal(fit,audit['scaler_fit'])
        and sr['old512_mean_sha256']==array_sha(om) and sr['old512_scale_sha256']==array_sha(oscale) and sr['standardization_sha256']==digest(SCALER) and sr['protocol_sha256']==TRAINER_PROTOCOL_SHA,'New128 scaler was not certified original578090 FIT-only.')
    with np.load(SCALER,allow_pickle=False) as z:
        require(set(z.files)=={'new_mean','new_scale','matched_mean','matched_scale'},'Arm scaler archive coverage differs.');scalers={arm:(z[arm+'_mean'].copy(),z[arm+'_scale'].copy()) for arm in ['new','matched']}
    scalers['old-head']=(om,oscale)
    for m,s in scalers.values():require(np.array_equal(m[:512].view(np.uint32),om.view(np.uint32)) and np.array_equal(s[:512].view(np.uint32),oscale.view(np.uint32)),'Original512 standardization changed.')
    require(np.array_equal(scalers['matched'][0][512:].view(np.uint32),om[:128].view(np.uint32)) and np.array_equal(scalers['matched'][1][512:].view(np.uint32),oscale[:128].view(np.uint32)),'Matched scaler is not exact duplicated rank0/1.')
    portable={};models={};old={}
    # Original portable must be checked before examining residual embedded bases.
    for name in ['old-head','new','matched']:
        path=paths[name];want=OLD_MODEL_SHA if name=='old-head' else audit['model_sha256'][name];bind(path,want,True);models[name]=want
        portable[name],base=portable_metadata(path,name,old,scalers,plan,static)
        if name=='old-head':old=base
    require(selection['model_sha256']==models['new'],'Selected new model is not audited portable.')
    # Independently audited extra-bank certificate; no feature rows/bytes needed.
    extra_path=EXTRA_OUT/'extraction-complete.json';bind(extra_path,static['extra_complete_sha256']);extra=read(extra_path)
    candidates=[Path(p) for p,w in static['closed_files_sha256'].items() if w==static['extra_audit_sha256']];require(len(candidates)==1,'Unique actual extra-bank audit absent.');ep=candidates[0];bind(ep,static['extra_audit_sha256']);ea=read(ep)
    require(ea['stage']=='actual_completed_complementary_head_feature_audit' and ea['audit_passed'] is True and ea['test_data_or_results_read'] is False and ea['extractor_complete_sha256']==static['extra_complete_sha256']
        and ea['extractor_source_sha256']==extra['source_sha256'] and ea['features_sha256']==extra['features_sha256'] and ea['rows_sha256']==extra['rows_sha256']
        and extra['rows']==896838 and extra['dimensions']==128 and extra['feature_shape']==[896838,128] and extra['dtype']=='float16' and extra['added_head_ids']==ADDED and extra['original_head_ids']==ORIGINAL
        and extra['all_rows_written'] is True and extra['all_finite'] is True and extra['tests_read'] is False,'Audited extra128 completion changed.')
    checkpoint=extra['signature']['checkpoint_path'];require(checkpoint==str(CHECKPOINT),'Exact verified /dev/shm checkpoint required before any checkpoint access.')
    source_files=extra['signature']['selection']['files_sha256'];assets={Path(p).name:w for p,w in source_files.items() if str(Path(p).parent)==checkpoint}
    required={'config.json','verified-weights.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'};require(required<=set(assets),'Verified checkpoint/config/tokenizer asset closure incomplete.')
    for name,want in assets.items():bind(CHECKPOINT/name,want,runtime_file=True,audited=False)
    config=read(CHECKPOINT/'config.json');require(config['quantization_config']['quant_method']=='mxfp4' and (config['num_hidden_layers'],config['num_attention_heads'],config['head_dim'])==(36,64,64),'Frozen native MXFP4 config changed.')
    for path in request.get('extra_runtime_sources',[]):
        require(str(path).endswith('.py'),'Future inference/builder/report dependency must be a source file.');bind(path,runtime_file=True,audited=False)
    # Raw bank geometry is an audited claim, not runtime availability or rehash.
    raw_banks={}
    for path,shape in [(str(BASE/'systemness-head-features/extraction-complete.json'),[755282,512]),(str(BASE/'systemness-long-augmentation/extraction-complete.json'),[389120,512]),(str(BASE/'systemness-natural-dev-features/extraction-complete.json'),[80116,512]),(str(extra_path),[896838,128])]:
        bind(path);c=read(path);require(c['feature_shape']==shape and c['dtype']=='float16' and c['all_rows_written'] is True and c['all_finite'] is True,'Parent feature geometry differs.')
        fp=c['features_path'];require(static['feature_files_sha256'][fp]==c['features_sha256'] and provenance.get(str(clean_path(fp)))==c['features_sha256'],'Raw bank was not independently audited.')
        raw_banks[fp]=dict(sha256=c['features_sha256'],shape=shape,dtype='float16',size_bytes=int(np.prod(shape))*2,required_at_inference=False,raw_bytes_rehashed_by_helper=False)
    return dict(protocol=protocol_spec(),helper_source_sha256=digest(__file__),gate=dict(candidate='new-max33',promoted=True,independent_actual_development_audit_passed=True,
        trainer_source_sha256=TRAINER_SOURCE_SHA,trainer_protocol_sha256=TRAINER_PROTOCOL_SHA,trainer_complete_sha256=TRAINER_COMPLETE_SHA,audit_path=str(audit_path),audit_sha256=ACTUAL_AUDIT_SHA,audit_source_path=str(audit_source),audit_source_sha256=APPROVED_AUDITOR_SOURCE_SHA),
        physical_model_paths={n:str(clean_path(p)) for n,p in paths.items()},physical_model_sha256=models,record_thresholds=thresholds,record_to_physical=dict(RECORD_TO_MODEL),source_records=dict(SOURCE_RECORD),
        selected_head_ids=list(CAPTURE),capture_head_ids=list(CAPTURE),capture_dimensions=640,physical_feature_views=copy.deepcopy(VIEWS),portable_metadata=portable,
        old_scaler_path=str(clean_path(old_scaler)),arm_scaler_path=str(clean_path(SCALER)),runtime_files_sha256=runtime,certificate_files_sha256=certificates,
        future_exclusion_files_sha256=future_exclusions,exclusion_inputs_sha256={str(clean_path(BASE/n)):digest(BASE/n) for n in EXCLUSION_NAMES},original249_inputs_sha256={str(clean_path(p)):digest(p) for p in ORIGINAL249},
        audited_artifact_provenance=dict(feature_banks=raw_banks,raw_files={p:dict(sha256=w,required_at_inference=False,raw_bytes_rehashed_by_helper=False) for p,w in certified_raw.items()}),
        checkpoint_path=checkpoint,kernel_revision=extra['signature']['kernel_revision'],runtime_rehashes_all_model_shards=False,raw_feature_or_probability_arrays_loaded=False,new_test_source_body_read=False,primary_changed=False)


def build_conditional_snapshot(audited_dev_snapshot=None):
    pin_gate();value=collect_material(audited_dev_snapshot);value['snapshot_sha256']=canonical(value);return value


def verify_closed_snapshot(snapshot):
    """Rebuild only the same explicit audited closure, never an inventory."""
    pin_gate();value=dict(snapshot);expected=value.pop('snapshot_sha256');require(canonical(value)==expected,'Snapshot content digest differs.')
    source=snapshot['gate']['audit_source_path'];extras=[p for p in snapshot['runtime_files_sha256'] if p.endswith('.py') and p not in read(ACTUAL_AUDIT_PATH)['provenance_sha256'] and p not in [str(clean_path(__file__)),source]]
    rebuilt=build_conditional_snapshot(dict(audit_path=snapshot['gate']['audit_path'],audit_source_path=source,extra_runtime_sources=extras))
    require(snapshot==rebuilt,'Frozen closed model/threshold/scaler/source/feature-view snapshot changed.')
    return {k:Path(v) for k,v in snapshot['physical_model_paths'].items()},dict(snapshot['record_thresholds']),list(snapshot['selected_head_ids'])


def declare_snapshot(audited_dev_snapshot=None,*,snapshot=None,out=None):
    pin_gate();require(snapshot is None or audited_dev_snapshot is None,'Pass either audit input or a built snapshot.')
    value=build_conditional_snapshot(audited_dev_snapshot) if snapshot is None else snapshot;verify_closed_snapshot(value)
    folder=clean_path(OUT if out is None else out);path=folder/'readout-snapshot.json'
    if path.exists():require(read(path)==value,'Existing snapshot is immutable.');return value
    for meta,cache in FUTURE_DATA:
        require(not any((meta/n).exists() for n in ['data-complete.json','document-keys.json','natural-source-keys.json','source-freeze.json'])
            and not any((cache/n).exists() for n in ['inputs.parquet','rows.parquet','token-ids.json']),'Test source data/IDs predate the required closed snapshot.')
    folder.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.json.tmp')
    with tmp.open('w') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path);fd=os.open(folder,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return value


def preflight():
    """Synthetic closed files/NPZ faults only; no real model/bank/p/source IO."""
    from unittest.mock import patch
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-confirm-snapshot-preflight-',dir='/tmp'));start=digest(__file__);checks=[]
    def check(name,value=True):require(value,'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,KeyError,ValueError):checks.append(name);return
        raise RuntimeError('Expected refusal: '+name)
    def forbidden(*a,**k):raise LookupError('Formal artifact access forbidden in unknown-pin fixture.')
    # Patch own pins even after root fills them in a future reviewed revision.
    with patch.dict(globals(),dict(TRAINER_SOURCE_SHA=None,TRAINER_PROTOCOL_SHA=None,TRAINER_COMPLETE_SHA=None,APPROVED_AUDITOR_SOURCE_SHA=None,ACTUAL_AUDIT_PATH=None,ACTUAL_AUDIT_SHA=None,read=forbidden,digest=forbidden)),patch.object(np,'load',forbidden):
        reject('unknown_pins_build_blocks_before_formal_IO',build_conditional_snapshot)
        reject('unknown_pins_verify_blocks_before_formal_IO',lambda:verify_closed_snapshot({}))
        reject('unknown_pins_declare_blocks_before_formal_IO',lambda:declare_snapshot(snapshot={}))
    fake=folder/'metadata';train=fake/'systemness-complementary-head-readout';old=fake/'systemness-head-probes';ext=fake/'systemness-complementary-head-features';checkpoint=folder/'checkpoint';scaler=folder/'cache/scalers.npz';source=folder/'readout.py';auditor=folder/'auditor.py';auditpath=folder/'actual-audit.json';design=folder/'design.json'
    future=[(folder/'future-c4',folder/'cache/c4'),(folder/'future-natural',folder/'cache/natural')]
    def put(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n');return path
    def binary(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**value)
    source.write_text('# synthetic640 producer only\n');auditor.write_text('# synthetic independent auditor only\n');put(design,dict(synthetic=True))
    oldm=np.linspace(-1,1,512,dtype=np.float32);olds=np.ones(512,np.float32);base={'encoder.0.weight':np.zeros((128,512),np.float32),'encoder.0.bias':np.zeros(128,np.float32),'head.weight':np.zeros((5,128),np.float32),'head.bias':np.arange(5,dtype=np.float32)}
    binary(old/'standardization.npz',dict(mean=oldm,scale=olds));oldmeta=dict(kind='mlp',hidden=128,dimensions=512,heads=8,head_dim=64,selected_head_ids=ORIGINAL,roles=ROLES)
    binary(old/'models/joint-mlp128.npz',dict(mean=oldm,scale=olds,metadata=np.array(json.dumps(oldmeta)),**{'weight::'+k:v for k,v in base.items()}));oldsha=digest(old/'models/joint-mlp128.npz')
    newm=np.r_[oldm,np.arange(128,dtype=np.float32)/32];news=np.r_[olds,np.full(128,2,np.float32)];dupm=np.r_[oldm,oldm[:128]];dups=np.r_[olds,olds[:128]]
    binary(scaler,dict(new_mean=newm,new_scale=news,matched_mean=dupm,matched_scale=dups))
    # Fake source inventories are metadata only; original249 text is never parsed.
    for name in EXCLUSION_NAMES:put(fake/name,dict(synthetic_manifest=True))
    originals=[fake/'training_inputs.parquet',folder/'legacy/raw_training_data.json']
    for path in originals:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic byte inventory, intentionally not JSON')
    for name in ['config.json','verified-weights.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
        put(checkpoint/name,dict(quantization_config=dict(quant_method='mxfp4'),num_hidden_layers=36,num_attention_heads=64,head_dim=64) if name=='config.json' else dict(synthetic=True,name=name))
    extraaudit=folder/'extra-audit.json';extra_path=ext/'extraction-complete.json';raw_paths={};parent_completes=[]
    for bank,shape in [('systemness-head-features',[755282,512]),('systemness-long-augmentation',[389120,512]),('systemness-natural-dev-features',[80116,512]),('systemness-complementary-head-features',[896838,128])]:
        fp=folder/'nonexistent-banks'/bank/'features.f16';raw_paths[str(fp)]='a'*64;cp=fake/bank/'extraction-complete.json';parent_completes.append(cp)
        put(cp,dict(features_path=str(fp),features_sha256='a'*64,feature_shape=shape,dtype='float16',all_rows_written=True,all_finite=True))
    extra=read(extra_path)|dict(rows=896838,dimensions=128,rows_sha256='b'*64,source_sha256='c'*64,original_head_ids=ORIGINAL,added_head_ids=ADDED,tests_read=False,
        signature=dict(checkpoint_path=str(checkpoint),kernel_revision='synthetic-kernel',selection=dict(files_sha256={str(checkpoint/name):digest(checkpoint/name) for name in ['config.json','verified-weights.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']})))
    put(extra_path,extra);put(extraaudit,dict(stage='actual_completed_complementary_head_feature_audit',audit_passed=True,test_data_or_results_read=False,extractor_complete_sha256=digest(extra_path),extractor_source_sha256='c'*64,features_sha256='a'*64,rows_sha256='b'*64))
    static=dict(source_sha256=digest(source),embedded_base_weight_sha256='d'*64,standardization_sha256=digest(old/'standardization.npz'),old512_mean_sha256=array_sha(oldm),old512_scale_sha256=array_sha(olds),
        sample_ix_sha256=dict(fit='e'*64),extra_complete_sha256=digest(extra_path),extra_audit_sha256=digest(extraaudit),dependencies_sha256={},closed_files_sha256={str(extraaudit):digest(extraaudit)},feature_files_sha256=raw_paths)
    proto=dict(promotion=dict(candidate='new-max33',absolute_min33=.9,strict_controls=STRICT,guard_controls=GUARDS,mandatory_original_savedGPU=True),original_head_ids=ORIGINAL,added_head_ids=ADDED,dimensions=640,parameters=20545)
    put(train/'protocol.json',dict(protocol=proto,static=static));sig=dict(standardization_sha256=digest(scaler));plan=dict(signature=sig,protocol=proto,initial_weight_sha256='f'*64);put(train/'search-plan.json',plan)
    fit=dict(fit_rows=578090,fit_only=True,natural_rows_used=0,validation_rows_used=0,chunk=4096,floor=1e-6,variance='Population F64 parallel-Welford',fit_sample_ix_sha256='e'*64)
    put(train/'scaler-record.json',dict(standardization_sha256=digest(scaler),protocol_sha256=digest(train/'protocol.json'),fit=fit,old512_mean_sha256=array_sha(oldm),old512_scale_sha256=array_sha(olds)))
    models={}
    for arm,mean,scale in [('new',newm,news),('matched',dupm,dups)]:
        meta=dict(kind='binary-residual-640',arm=arm,dimensions=640,base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',roles=ROLES,base_model_sha256=oldsha,
            selected_head_ids=ORIGINAL+(ADDED if arm=='new' else ORIGINAL[:2]),signature=sig,source_sha256=digest(source),plan_sha256=digest(train/'search-plan.json'),initial_weight_sha256='f'*64,
            standardization_sha256=digest(scaler),tests_read=False,embedded_base_weight_sha256='d'*64,mean_sha256=array_sha(mean),scale_sha256=array_sha(scale))
        weights={'weight::base.'+k:v for k,v in base.items()}|{'weight::correction.0.weight':np.zeros((32,640),np.float32),'weight::correction.0.bias':np.zeros(32,np.float32),'weight::correction.3.weight':np.zeros((1,32),np.float32),'weight::correction.3.bias':np.zeros(1,np.float32)}
        binary(train/f'models/{arm}.npz',dict(metadata=np.array(json.dumps(meta)),mean=mean,scale=scale,**weights));models[arm]=digest(train/f'models/{arm}.npz')
        put(train/f'training/{arm}.json',dict(synthetic=True));put(train/f'candidates/{arm}.json',dict(training_sha256=digest(train/f'training/{arm}.json'),model_sha256=models[arm],source_sha256=digest(source),tests_read=False))
    records=[dict(record=name,validation=dict(min33=.95 if name=='new-max33' else .91,system_threshold=OLD_THRESHOLD if name=='oldhead-oldmean' else .5)) for name in DEVELOPMENT_RECORDS]
    put(train/'records.json',records);put(train/'zero-controls.json',dict(records={k:dict(min33=.91) for k in ['zero-oldmean','zero-max33']}))
    selection=dict(candidate='new-max33',promoted=True,no_fallback=True,rejection_reasons=[],primary_changed=False,tests_read=False,new_independent_confirmation_required=True,
        guards={k:dict(eligible=True,rejection_reasons=[]) for k in GUARDS+['original-savedGPU']},min33_gains={k:.95-.91 for k in STRICT},source_sha256=digest(source),records_sha256=digest(train/'records.json'),model_sha256=models['new'],model_path=str(train/'models/new.npz'),system_threshold=.5)
    put(train/'selection.json',selection)
    for name in ARTIFACTS:
        if not (train/name).exists():put(train/name,dict(synthetic=True,name=name))
    done=dict(source_sha256=digest(source),signature=sig,tests_read=False,primary_changed=False,new_independent_confirmation_required=True,artifact_sha256={n:digest(train/n) for n in ARTIFACTS},candidate_sha256={a:digest(train/f'candidates/{a}.json') for a in ['new','matched']});put(train/'complete.json',done);put(train/'status.json',dict(stage='validation_complete'))
    files=[train/n for n in ARTIFACTS]+[train/'complete.json',train/'status.json',source,scaler,old/'standardization.npz',old/'models/joint-mlp128.npz',extraaudit]+[train/f'{kind}/{arm}.json' for kind in ['candidates','training'] for arm in ['new','matched']]+[train/f'models/{arm}.npz' for arm in ['new','matched']]+[fake/n for n in EXCLUSION_NAMES]+originals+parent_completes
    future_exclusion_paths={str(p) for p in [fake/n for n in EXCLUSION_NAMES]+originals}
    provenance={str(p):digest(p) for p in files if str(p) not in future_exclusion_paths}|raw_paths;probpath=folder/'nonexistent-p/new-old-validation.npz';provenance[str(probpath)]='1'*64
    audit=dict(stage=AUDIT_STAGE,audit_passed=True,test_data_or_results_read=False,gpu_used=False,auditor_source_path=str(auditor),auditor_source_sha256=digest(auditor),trainer_source_sha256=digest(source),trainer_protocol_sha256=digest(train/'protocol.json'),trainer_complete_sha256=digest(train/'complete.json'),trainer_selection_sha256=digest(train/'selection.json'),trainer_records_sha256=digest(train/'records.json'),
        trainer_model_record_sha256=done['candidate_sha256'],fourteen_records=records,selection=selection,model_sha256=models,scaler_fit=fit,provenance_sha256=provenance,probability_bindings={str(probpath):'1'*64});put(auditpath,audit)
    own=dict(BASE=fake,TRAINER=train,TRAINER_SOURCE=source,TRAINER_SOURCE_SHA=digest(source),TRAINER_PROTOCOL_SHA=digest(train/'protocol.json'),TRAINER_COMPLETE_SHA=digest(train/'complete.json'),APPROVED_AUDITOR_SOURCE_SHA=digest(auditor),ACTUAL_AUDIT_PATH=auditpath,ACTUAL_AUDIT_SHA=digest(auditpath),DESIGN=design,DESIGN_SHA=digest(design),OLD_HEAD=old,OLD_MODEL_SHA=oldsha,EXTRA_OUT=ext,CHECKPOINT=checkpoint,SCALER=scaler,ORIGINAL249=originals,FUTURE_DATA=future)
    with patch.dict(globals(),own):
        snapshot=build_conditional_snapshot();paths,thresholds,heads=verify_closed_snapshot(snapshot)
        check('complete_toy_build_three_models_four_thresholds_and_ten_capture_heads',set(paths)==set(VIEWS) and len(thresholds)==4 and heads==CAPTURE and thresholds['oldhead-original']==OLD_THRESHOLD)
        check('returned_protocol_and_snapshot_do_not_alias_constant_views',snapshot['physical_feature_views'] is not VIEWS and snapshot['physical_feature_views']['new']['selected_head_ids'] is not CAPTURE and snapshot['protocol']['physical_feature_views'] is not VIEWS and snapshot['selected_head_ids'] is not CAPTURE)
        check('eight_future_exclusions_frozen_without_false_training_audit_claim',set(snapshot['future_exclusion_files_sha256'])==future_exclusion_paths and not future_exclusion_paths&set(audit['provenance_sha256']))
        check('model_specific640_new_duplicate_and512_old_geometry',snapshot['physical_feature_views']==VIEWS and snapshot['portable_metadata']['matched']['selected_head_ids']==ORIGINAL+ORIGINAL[:2] and snapshot['portable_metadata']['old-head']['dimensions']==512)
        check('embedded_original512_and_duplicate_rank01_scalers_checked',snapshot['portable_metadata']['new']['mean_sha256']==array_sha(newm) and snapshot['portable_metadata']['matched']['mean_sha256']==array_sha(dupm))
        check('raw_banks_and_probabilities_are_not_runtime_dependencies',not any(Path(p).exists() for p in list(raw_paths)+[str(probpath)]) and not set(raw_paths)&set(snapshot['runtime_files_sha256']) and str(probpath) not in snapshot['certificate_files_sha256'])
        saved=declare_snapshot(snapshot=snapshot,out=folder/'snapshot');check('fsynced_snapshot_immutable_roundtrip',read(folder/'snapshot/readout-snapshot.json')==saved)
        put(future[0][0]/'document-keys.json',dict(future_unrelated_manifest=True));verify_closed_snapshot(snapshot);declare_snapshot(snapshot=snapshot,out=folder/'snapshot');check('later_test_manifest_does_not_change_old_closed_snapshot')
        reject('first_declaration_after_test_ID_artifact_is_rejected',lambda:declare_snapshot(snapshot=snapshot,out=folder/'late-snapshot'))
        bad=copy.deepcopy(snapshot);bad['record_thresholds']['new-max33']=.2;bad.pop('snapshot_sha256');bad['snapshot_sha256']=canonical(bad);reject('recomputed_outer_digest_cannot_change_threshold',lambda:verify_closed_snapshot(bad))
        bad=copy.deepcopy(snapshot);bad['physical_feature_views']['matched']['view']='capture640';bad.pop('snapshot_sha256');bad['snapshot_sha256']=canonical(bad);reject('duplicate_control_cannot_consume_real_extra_heads',lambda:verify_closed_snapshot(bad))
        bad=copy.deepcopy(snapshot);bad['runtime_files_sha256'].pop(str(train/'models/new.npz'));bad.pop('snapshot_sha256');bad['snapshot_sha256']=canonical(bad);reject('recomputed_outer_digest_cannot_omit_audited_model',lambda:verify_closed_snapshot(bad))
        extra_source=folder/'future-runtime.py';extra_source.write_text('# explicit future runtime source\n');with_extra=build_conditional_snapshot(dict(audit_path=auditpath,extra_runtime_sources=[extra_source]));verify_closed_snapshot(with_extra);check('explicit_future_builder_infer_report_source_closure_supported',str(extra_source) in with_extra['runtime_files_sha256'])
        data=(train/'selection.json').read_bytes();put(train/'selection.json',selection|dict(promoted=False));reject('failed_promotion_no_fallback',build_conditional_snapshot);(train/'selection.json').write_bytes(data)
        exclusion=fake/EXCLUSION_NAMES[0];data=exclusion.read_bytes();put(exclusion,dict(changed_future_exclusion=True));reject('fresh_frozen_future_exclusion_tamper_rejected',lambda:verify_closed_snapshot(snapshot));exclusion.write_bytes(data)
        data=auditor.read_bytes();auditor.write_text('# changed auditor\n');reject('changed_actual_auditor_source_rejected',lambda:verify_closed_snapshot(snapshot));auditor.write_bytes(data)
        data=(checkpoint/'config.json').read_bytes();put(checkpoint/'config.json',dict(quantization_config=dict(quant_method='bf16')));reject('checkpoint_config_tamper_rejected',lambda:verify_closed_snapshot(snapshot));(checkpoint/'config.json').write_bytes(data)
        with np.load(train/'models/matched.npz',allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
        badpayload=dict(payload);badpayload['mean']=newm;binary(folder/'bad-scaler.npz',badpayload)
        scalers={'old-head':(oldm,olds),'new':(newm,news),'matched':(dupm,dups)}
        reject('matched_embedded_scaler_cannot_use_new128_scaler',lambda:portable_metadata(folder/'bad-scaler.npz','matched',base,scalers,plan,static))
        badpayload=dict(payload);meta=json.loads(str(payload['metadata']));meta['selected_head_ids']=CAPTURE;badpayload['metadata']=np.array(json.dumps(meta));binary(folder/'bad-head-order.npz',badpayload)
        reject('matched_head_metadata_cannot_claim_capture10_unique',lambda:portable_metadata(folder/'bad-head-order.npz','matched',base,scalers,plan,static))
        badpayload=dict(payload);badpayload['weight::base.head.bias']=np.zeros(5,np.float32);binary(folder/'bad-base.npz',badpayload)
        reject('embedded_non_system_base_weight_tamper_rejected',lambda:portable_metadata(folder/'bad-base.npz','matched',base,scalers,plan,static))
    check('own_formal_pins_restored_to_unknown',all(v is None for v in [TRAINER_SOURCE_SHA,TRAINER_PROTOCOL_SHA,TRAINER_COMPLETE_SHA,APPROVED_AUDITOR_SOURCE_SHA,ACTUAL_AUDIT_PATH,ACTUAL_AUDIT_SHA]))
    reject('forbidden_workspace_rejected_lexically',lambda:clean_path('/workspace/config.json'))
    check('source_unchanged_during_CPU_preflight',digest(__file__)==start)
    report=dict(stage='synthetic_CPU_complementary640_closed_snapshot_preflight',source_sha256=start,checks=checks,checks_passed=len(checks),fixture_directory=str(folder),actual_enabled=False,actual_development_artifacts_read=False,
        real_portable_models_read=False,raw_feature_or_probability_arrays_read=False,test_source_or_ID_selection_run=False,actual_snapshot_declared=False,gpu_used=False)
    output=folder/'report.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=start,checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--declare',action='store_true');mode.add_argument('--verify',type=Path);parser.add_argument('--audit',type=Path);parser.add_argument('--audit-source',type=Path);parser.add_argument('--runtime-source',type=Path,action='append',default=[]);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.verify:pin_gate();verify_closed_snapshot(read(args.verify))
    elif args.declare:
        pin_gate();request=dict(audit_path=args.audit or ACTUAL_AUDIT_PATH,extra_runtime_sources=args.runtime_source)
        if args.audit_source:request['audit_source_path']=args.audit_source
        declare_snapshot(request,out=args.out)
    else:parser.print_help()

if __name__=='__main__':main()
