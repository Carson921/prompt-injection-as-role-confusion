"""Conditional guard-first snapshot for unchanged complementary640 readouts.

Default help; --preflight uses synthetic CPU/tmp only. Unknown final calibration
source/protocol/decision and actual-audit pins reject before formal artifact reads.
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
HELPER_PREPARATION=BASE/'systemness-complementary-guard-confirmation-preparation'
DESIGN=HELPER_PREPARATION/'conditional-helper-design-v2.json'
DESIGN_SHA='0ba1e616c9b53def2460db677c473a675d44703f7bdfda1590a4ccd89f4ede20'
SOURCE_RESERVATION_DESIGN=BASE/'systemness-complementary-confirmation-preparation/root-prospective-design.json'
SOURCE_RESERVATION_SHA='6999ea73bb0207ef3d6a00610d96e47de9e05cca38bac5b5ef08bbd2c17bd824'
TRAINER=BASE/'systemness-complementary-head-readout'
TRAINER_SOURCE=HERE/'systemness_complementary_head_readout.py'
TRAINER_SOURCE_SHA='e74e40889ba49ab05815f0faac68c2a522ccc06627b09ff14379929bc53dbdf6'
TRAINER_PROTOCOL_SHA='10c92c0af904b96d42e1ff8c9cbc12f7efb0371da08fdc3962aebd7b8d0d6f54'
TRAINER_COMPLETE_SHA='376ad13999055eb3144a2b3d90d51e318eb30f24d6f64991c819ea29374a2e52'
APPROVED_AUDITOR_SOURCE_SHA='0219af8dbf0dc6587f8ba4d4bc1e4f163ae9e745a2b6c4d99a24560a88c8ae85'
ACTUAL_AUDIT_PATH=TRAINER/'actual-audit/actual-development-audit.json'
ACTUAL_AUDIT_SHA='7bac44572d8ea37fc4cae87d447061228496a218eadaa92583a4a96bb3c4989c'
AUDIT_STAGE='actual_completed_complementary_head_readout_audit'
ORIGINAL_SELECTION_SHA='8d66554032f9325ce2e6110e2881c320d1853ac514098e4c7b887af7253a0193'
ORIGINAL_RECORDS_SHA='9d9d4f70a877a4e2f83ce40b04dfee6f824e39798e54333411e3b4b464184b0f'
NEW_MODEL_SHA='ef92b421007b7d0c7a11f13e91008e6d2e39b34b730c6ca7d6ecbf03590236ff'
CALIBRATION=BASE/'systemness-complementary-guard-calibration'
CALIBRATION_SOURCE=HERE/'systemness_complementary_guard_calibration.py'
CALIBRATION_SOURCE_SHA='70c6ae046cbbb5485047733b593afe23a1d61376f047b2ad29e75c692e7bb85d'
CALIBRATION_PROTOCOL_SHA='7edc1b5ba2c55a21ebe01bc5af9f850c4865a933bb9bc4b1274520d6b00ec64a'
CALIBRATION_COMPLETE_SHA='9d5b032c1b055eda8ccb5620620f873b2e4775cb533729b787d86436857c2e94'
CALIBRATION_DECISION_SHA='e220bda41077f5931dc81b951f10e55e1160904e303f3299a89dfdc300430a0c'
CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA='3296c36da7efffe9ab45a75c12d8ebf1640b18f126cd64166674ae19002bc323'
CALIBRATION_ACTUAL_AUDIT_PATH='/root/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/local-gptoss120b/systemness-complementary-guard-calibration/actual-audit/actual-calibration-audit.json'
CALIBRATION_ACTUAL_AUDIT_SHA='1a6e545db44a2d76c8da30886409d7c3b8376e670281a130b464a2e8821f4667'
CALIBRATION_AUDIT_STAGE='actual_completed_guard_constrained_calibration_audit'
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
RECORD_TO_MODEL={'new-guard-max33':'new','matched-max33':'matched','oldhead-max33':'old-head','oldhead-original':'old-head'}
SOURCE_RECORD={name:dict(origin='guard-calibration-decision' if name=='new-guard-max33' else 'original-readout-records',record='oldhead-oldmean' if name=='oldhead-original' else name) for name in RECORD_TO_MODEL}
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
    """Pure-string final pins; no formal IO before both certificate chains exist."""
    original=[TRAINER_SOURCE_SHA,TRAINER_PROTOCOL_SHA,TRAINER_COMPLETE_SHA,APPROVED_AUDITOR_SOURCE_SHA,ACTUAL_AUDIT_SHA]
    final=[CALIBRATION_SOURCE_SHA,CALIBRATION_PROTOCOL_SHA,CALIBRATION_COMPLETE_SHA,CALIBRATION_DECISION_SHA,CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA,CALIBRATION_ACTUAL_AUDIT_SHA]
    require(all(valid_sha(v) for v in original+final) and ACTUAL_AUDIT_PATH is not None and CALIBRATION_ACTUAL_AUDIT_PATH is not None,'Final guard-calibration/source/protocol/decision/independent-audit pins unknown; actual forbidden.')
    clean_path(ACTUAL_AUDIT_PATH);clean_path(CALIBRATION_ACTUAL_AUDIT_PATH)


def protocol_spec():
    value=dict(version=1,experiment='closed-guard-first-complementary640-confirmation-snapshot',root_design_sha256=DESIGN_SHA,
        gate='Original640 trial remains promotedFalse. Independently audited new-guard-max33 eligibleTrue and gate eligibleTrue, absolute90, originalfive strict/eight guards plus savedGPU; no fallback.',
        approved_independent_auditor_source_sha256=APPROVED_AUDITOR_SOURCE_SHA,
        calibration_certificate_contract=dict(stage=CALIBRATION_AUDIT_STAGE,decision_field='decision',eligible_not_promoted=True,source_sha256=CALIBRATION_SOURCE_SHA,protocol_sha256=CALIBRATION_PROTOCOL_SHA,complete_sha256=CALIBRATION_COMPLETE_SHA,decision_sha256=CALIBRATION_DECISION_SHA,auditor_source_sha256=CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA,actual_audit_sha256=CALIBRATION_ACTUAL_AUDIT_SHA),physical_models=list(VIEWS),record_to_model=RECORD_TO_MODEL,source_records=SOURCE_RECORD,
        original_trial_promoted=False,calibration_uses_eligible_not_promoted=True,source_reservation_design_sha256=SOURCE_RESERVATION_SHA,original_old_head_sha256=OLD_MODEL_SHA,original_threshold=OLD_THRESHOLD,capture_head_ids=CAPTURE,capture_dimensions=640,physical_feature_views=VIEWS,
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
    require(digest(TRAINER/'selection.json')==ORIGINAL_SELECTION_SHA and digest(TRAINER/'records.json')==ORIGINAL_RECORDS_SHA,'Original negative decision or records changed.')
    require(selection['candidate']=='new-max33' and selection['promoted'] is False and selection['no_fallback'] is True
        and selection['rejection_reasons'] and selection['primary_changed'] is False and selection['tests_read'] is False
        and selection['new_independent_confirmation_required'] is True and set(selection['guards'])==set(GUARDS)|{'original-savedGPU'}
        and set(selection['min33_gains'])==set(STRICT),'Original audited failed trial must stay negative and unchanged.')
    require(all(equal(selection[k],v) for k,v in audit['selection'].items()),'Original actual-audit failure proof differs.')
    require(selection['model_sha256']==audit['model_sha256']['new']==NEW_MODEL_SHA,'Fixed unchanged new640 weights differ.')
    thresholds={'matched-max33':float(index['matched-max33']['validation']['system_threshold']),
        'oldhead-max33':float(index['oldhead-max33']['validation']['system_threshold']),
        'oldhead-original':float(index['oldhead-oldmean']['validation']['system_threshold'])}
    require(all(np.isfinite(v) and 0<=v<=1.00001 for v in thresholds.values()) and thresholds['oldhead-original']==OLD_THRESHOLD,'Original three control thresholds differ.')
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
    calibration=calibration_material(request,audit,decl,index=records)
    for p,w in calibration['provenance_sha256'].items():
        require(p not in provenance or provenance[p]==w,'Original/calibration provenance conflicts.');provenance[p]=w
    thresholds={'new-guard-max33':calibration['decision']['system_threshold'],**thresholds}
    require(digest(DESIGN)==DESIGN_SHA,'Root reserved confirmation design changed.')
    raw=set(audit['probability_bindings'])|set(calibration['probability_bindings'])|set(static['feature_files_sha256'])
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
    bind(DESIGN,DESIGN_SHA,audited=False);bind(SOURCE_RESERVATION_DESIGN,SOURCE_RESERVATION_SHA,audited=False);bind(audit_path,ACTUAL_AUDIT_SHA,audited=False);bind(audit_source,APPROVED_AUDITOR_SOURCE_SHA,runtime_file=True,audited=False);bind(__file__,runtime_file=True,audited=False)
    for path in [TRAINER/n for n in ARTIFACTS]+[TRAINER/'status.json',TRAINER_SOURCE]+[TRAINER/f'{kind}/{arm}.json' for kind in ['candidates','training'] for arm in ['new','matched']]:bind(path)
    # Future source exclusions are prospectively frozen here, not claimed to be
    # direct inputs of the previous development-result audit. Only these eight
    # named inventory paths are exempted from that audit-membership requirement.
    future_exclusions={}
    for path in [BASE/n for n in EXCLUSION_NAMES]+ORIGINAL249:
        bind(path,audited=False);future_exclusions[str(clean_path(path))]=certificates[str(clean_path(path))]
    for name,want in static['dependencies_sha256'].items():bind(HERE/name,want,runtime_file=True)
    for path,want in calibration['required_files_sha256'].items():bind(path,want,runtime_file=Path(path).suffix=='.py',audited=False)
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
    return dict(protocol=protocol_spec(),helper_source_sha256=digest(__file__),original_failed_trial=dict(promoted=False,selection_sha256=ORIGINAL_SELECTION_SHA,records_sha256=ORIGINAL_RECORDS_SHA,rejection_reasons=selection['rejection_reasons']),
        calibrated_decision=calibration['snapshot_certificate'],gate=dict(candidate='new-guard-max33',eligible=True,original_trial_promoted=False,independent_actual_development_audit_passed=True,
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
    rebuilt=build_conditional_snapshot(dict(audit_path=snapshot['gate']['audit_path'],audit_source_path=source,extra_runtime_sources=extras,calibration_audit_source_path=snapshot['calibrated_decision']['audit_source_path']))
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



def validate_calibrated_decision(decision,protocol,index,zero,original_savedGPU):
    """Pure metadata gate; no probability evaluation or calibration is run here."""
    require(protocol['models']==1 and protocol['training_updates']==0 and protocol['fixed_model_sha256']==NEW_MODEL_SHA and protocol['threshold_budget']==501
        and protocol['strict_controls']==STRICT and protocol['guard_controls']==GUARDS+['original-savedGPU'] and protocol['absolute_min33']==.9 and protocol['max_FPR']==.05
        and protocol['tolerance']==1e-12 and protocol['tests_read'] is False and protocol['primary_changed'] is False,'Guard-first protocol changed model, budget or constraints.')
    require(protocol['floor']==dict(old4_f1_loss=.02,old4_accuracy_loss=.01,old5_mean_f1_loss=.005,
        full_tail_nonregression=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail'],public_f1_loss=.02,public_accuracy_loss=.01),'Original hard floors changed.')
    require(decision['candidate']=='new-guard-max33' and decision['eligible'] is True and decision['no_fallback'] is True
        and decision['model_sha256']==NEW_MODEL_SHA and clean_path(decision['model_path'])==clean_path(model_paths()['new'])
        and decision['original_trial_unchanged'] is True and decision['primary_changed'] is False and decision['tests_read'] is False
        and decision['new_independent_confirmation_required'] is True,'Only an eligible independently audited unchanged-model guard-first decision may enter confirmation.')
    gate,value=decision['gate'],decision['validation'];tau=decision['system_threshold']
    require(np.isfinite(tau) and 0<=tau<=1.00001 and value['system_threshold']==tau and np.isfinite(value['min33']) and value['min33']>=.9,'Frozen guard-first threshold or absolute90 target differs.')
    require(set(value['groups'])==set(protocol['groups']) and len(protocol['groups'])==11 and len(protocol['feasible'])==9,'Original11group33stat scope changed.')
    stats=[value['groups'][g][k] for g in protocol['groups'] for k in ['system_precision','system_recall','system_balanced_accuracy']]
    require(all(np.isfinite(x) and x>=.9 for x in stats) and equal(value['min33'],min(stats))
        and all(np.isfinite(value['groups'][g]['system_false_positive_rate']) and value['groups'][g]['system_false_positive_rate']<=.05 for g in protocol['feasible']),'Absolute min33/exact FPR feasibility fails.')
    require(gate['eligible'] is True and gate['rejection_reasons']==[] and set(gate['guards'])==set(GUARDS)|{'original-savedGPU'} and set(gate['min33_gains'])==set(STRICT)
        and all(p['eligible'] is True and p['rejection_reasons']==[] for p in gate['guards'].values()),'Originalfive strict/eight guards plus savedGPU must all pass.')
    for name,gain in gate['min33_gains'].items():
        control=zero[name] if name in zero else index[name]['validation']
        require(np.isfinite(gain) and gain>1e-12 and equal(gain,value['min33']-control['min33']),'Original fixed strict gain changed.')
    # The independently audited certificate proves binary counts and all guards.
    # Structural threshold-free identities are also enforced without reading p.
    original=index['new-max33']['validation']
    for group in protocol['groups']:
        for field in ['accuracy','macro_f1','nll','recall','confusion','system_auroc','system_average_precision']:
            require(equal(value['groups'][group][field],original['groups'][group][field]),'Calibration changed threshold-free classifier evidence: '+group+'/'+field)
    require(gate['guards']['original-savedGPU']['legacy_savedGPU_only'] is True,'Original savedGPU guard scope changed.')
    for name in GUARDS:require(gate['guards'][name]['legacy_savedGPU_only'] is False,'A normal guard was changed to legacy-only.')
    grid=decision['grid_proof'];require(grid['no_fallback'] is True and type(grid['selected_index']) is int and 0<=grid['selected_index']<grid['grid_thresholds']<=501
        and grid['threshold_budget']==501 and 0<grid['admissible_thresholds']<=grid['grid_thresholds'],'Guard-first complete grid/selected index proof differs.')
    return tau


def calibration_material(request,original_audit,decl,*,index):
    """Closed metadata only. Never imports the calibrator or opens DEV p/rows."""
    pin_gate();files={}
    def bind(path,want):
        path=clean_path(path);got=digest(path);require(got==want,'Pinned calibration metadata/source changed: '+str(path));files[str(path)]=got;return path
    bind(CALIBRATION_SOURCE,CALIBRATION_SOURCE_SHA);bind(CALIBRATION/'protocol.json',CALIBRATION_PROTOCOL_SHA)
    bind(CALIBRATION/'complete.json',CALIBRATION_COMPLETE_SHA);bind(CALIBRATION/'selection.json',CALIBRATION_DECISION_SHA)
    auditpath=bind(CALIBRATION_ACTUAL_AUDIT_PATH,CALIBRATION_ACTUAL_AUDIT_SHA);audit=read(auditpath)
    auditorsource=clean_path(request.get('calibration_audit_source_path',audit['auditor_source_path']));bind(auditorsource,CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA)
    require(audit['stage']==CALIBRATION_AUDIT_STAGE and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False and audit['gpu_used'] is False
        and audit['auditor_source_sha256']==CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA and audit['calibration_source_sha256']==CALIBRATION_SOURCE_SHA
        and audit['calibration_protocol_sha256']==CALIBRATION_PROTOCOL_SHA and audit['calibration_complete_sha256']==CALIBRATION_COMPLETE_SHA
        and audit['calibration_selection_sha256']==CALIBRATION_DECISION_SHA and audit['upstream_actual_audit_sha256']==ACTUAL_AUDIT_SHA
        and audit['model_sha256']==NEW_MODEL_SHA,'A precise independent guard-calibration actual certificate is required.')
    provenance={str(clean_path(p)):w for p,w in audit['provenance_sha256'].items()}
    require(provenance and all(valid_sha(w) for w in provenance.values()),'Calibration actual-audit provenance absent.')
    done,definition,decision=[read(CALIBRATION/n) for n in ['complete.json','protocol.json','selection.json']];protocol,static=definition['protocol'],definition['static']
    require(done['stage']=='completed_guard_constrained_development_calibration' and done['source_sha256']==CALIBRATION_SOURCE_SHA and done['protocol_sha256']==CALIBRATION_PROTOCOL_SHA
        and done['eligible'] is True and done['training_updates']==0 and done['gpu_used'] is False and done['tests_read'] is False and done['primary_changed'] is False
        and done['new_independent_confirmation_required'] is True and set(done['artifact_sha256'])=={'threshold-ledger.json','selection.json','validation-metrics.csv','report.txt'},'Completed calibration scope/coverage differs.')
    require(done['artifact_sha256']['selection.json']==CALIBRATION_DECISION_SHA and decision['source_sha256']==CALIBRATION_SOURCE_SHA and decision['protocol_sha256']==CALIBRATION_PROTOCOL_SHA
        and decision['upstream_actual_audit_sha256']==ACTUAL_AUDIT_SHA and equal(audit['decision'],decision),'Independent calibration decision ownership differs.')
    for name,want in done['artifact_sha256'].items():
        path=bind(CALIBRATION/name,want);require(provenance.get(str(path))==want,'Calibration actual audit omitted an artifact.')
    for path in [CALIBRATION_SOURCE,CALIBRATION/'protocol.json',CALIBRATION/'complete.json',CALIBRATION/'selection.json']:
        require(provenance.get(str(clean_path(path)))==files[str(clean_path(path))],'Calibration actual audit omitted source/protocol/complete/decision.')
    require(static['upstream_actual_audit_sha256']==ACTUAL_AUDIT_SHA and static['upstream_complete_sha256']==TRAINER_COMPLETE_SHA and static['upstream_protocol_sha256']==TRAINER_PROTOCOL_SHA
        and static['original_records_sha256']==ORIGINAL_RECORDS_SHA and static['all14_plus_zero_untouched'] is True and static['model_sha256']==NEW_MODEL_SHA
        and clean_path(static['model_path'])==clean_path(model_paths()['new']) and equal(static['original_selection'],read(TRAINER/'selection.json')) and static['original_selection']['promoted'] is False
        and static['source_sha256']==CALIBRATION_SOURCE_SHA
        and static['probability_files_opened'] is False and static['row_tables_materialized'] is False and static['raw_features_needed'] is False,'Calibration did not inherit the exact negative original trial/model.')
    require(protocol['groups']==decl['protocol']['groups'] and protocol['feasible']==decl['protocol']['feasible'],'Calibration groups/grid scope differ from original.')
    oldzero=read(TRAINER/'zero-controls.json')['records'];controls={n:(oldzero[n] if n in oldzero else index[n]['validation']) for n in GUARDS}
    require(equal(static['baseline_validation'],index['new-max33']['validation']) and equal(static['controls'],controls)
        and equal(static['legacy'],read(TRAINER/'controls.json')['original_savedGPU_validation']),'Calibration reference controls changed.')
    validate_calibrated_decision(decision,protocol,index,oldzero,static['legacy'])
    ledger=read(CALIBRATION/'threshold-ledger.json');grid=decision['grid_proof'];require(len(ledger)==grid['grid_thresholds'] and len(ledger)<=501,'Calibration threshold ledger truncated.')
    chosen=ledger[grid['selected_index']];require(chosen['index']==grid['selected_index'] and chosen['eligible'] is True and chosen['rejection_reasons']==[]
        and chosen['system_threshold']==decision['system_threshold'] and chosen['effective_F32_threshold']==float(np.float32(decision['system_threshold']))
        and equal(chosen['min33'],decision['validation']['min33']),'Selected decision does not match the certified threshold ledger.')
    probabilities=done['probability_bindings'];expected={str(clean_path(static['probabilities'][b]['path'])):static['probabilities'][b]['sha256'] for b in ['old','c4','public']}
    require(probabilities==expected==audit['probability_bindings'] and len(expected)==3 and all(original_audit['probability_bindings'].get(p)==w for p,w in expected.items()),'Calibration used different DEV probability files.')
    expected_data=dict(expected)
    for bank,count in [('old',173032),('c4',61440),('public',16713)]:
        row=static['source_rows'][bank];path=str(clean_path(row['path']));require(row['rows']==count and row['sample_ix_sha256']==decl['static']['sample_ix_sha256'][bank] and original_audit['provenance_sha256'].get(path)==row['sha256'],'Calibration row identity does not match original certificate.');expected_data[path]=row['sha256']
    require(len(expected_data)==6 and done['data_bindings']==expected_data,'Exactly3row/3probability certificates required.')
    for p,w in done['data_bindings'].items():require(provenance.get(str(clean_path(p)))==w,'Calibration row/probability certificate absent.')
    return dict(decision=decision,probability_bindings=probabilities,provenance_sha256=provenance,required_files_sha256=files,
        snapshot_certificate=dict(candidate='new-guard-max33',eligible=True,original_trial_promoted=False,decision_path=str(clean_path(CALIBRATION/'selection.json')),
            source_path=str(clean_path(CALIBRATION_SOURCE)),source_sha256=CALIBRATION_SOURCE_SHA,protocol_sha256=CALIBRATION_PROTOCOL_SHA,complete_sha256=CALIBRATION_COMPLETE_SHA,
            decision_sha256=CALIBRATION_DECISION_SHA,audit_path=str(auditpath),audit_sha256=CALIBRATION_ACTUAL_AUDIT_SHA,audit_source_path=str(auditorsource),
            audit_source_sha256=CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA,upstream_actual_audit_sha256=ACTUAL_AUDIT_SHA,model_sha256=NEW_MODEL_SHA,
            system_threshold=decision['system_threshold'],strict_controls=STRICT,guard_controls=GUARDS+['original-savedGPU'],no_fallback=True))


def preflight():
    """Only added certificate/eligibility/threshold ownership; fake metadata files."""
    from unittest.mock import patch
    import py_compile
    folder=Path(tempfile.mkdtemp(prefix='complementary-guard-snapshot-preflight-',dir='/tmp'));start=digest(__file__);checks=[]
    def check(name,v=True):require(v,'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,KeyError,ValueError):checks.append(name);return
        raise RuntimeError('Expected refusal: '+name)
    def forbidden(*a,**k):raise LookupError('Formal IO forbidden in unknown-pin CPU fixture.')
    final=['CALIBRATION_SOURCE_SHA','CALIBRATION_PROTOCOL_SHA','CALIBRATION_COMPLETE_SHA','CALIBRATION_DECISION_SHA','CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA','CALIBRATION_ACTUAL_AUDIT_SHA','CALIBRATION_ACTUAL_AUDIT_PATH']
    initial_final_pins={name:globals()[name] for name in final}
    valid={k:('a'*64 if k!='CALIBRATION_ACTUAL_AUDIT_PATH' else Path('/tmp/fake-audit.json')) for k in final}
    for missing in final:
        with patch.dict(globals(),dict(valid,**{missing:None,'read':forbidden,'digest':forbidden})),patch.object(np,'load',forbidden):
            reject('unknown_'+missing+'_before_formal_IO',build_conditional_snapshot)
    with patch.dict(globals(),dict(CALIBRATION_SOURCE_SHA=None,read=forbidden,digest=forbidden)),patch.object(np,'load',forbidden):
        reject('unknown_calibration_pins_verify_before_IO',lambda:verify_closed_snapshot({}))
        reject('unknown_calibration_pins_declare_before_IO',lambda:declare_snapshot(snapshot={}))
    groups=['old-'+g for g in ['single','multi','natural','long-single','long-full']]+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail','public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
    feasible=groups[:5]+['new-C4-long-full']+groups[-3:]
    metric=dict(system_precision=.94,system_recall=.94,system_balanced_accuracy=.94,system_false_positive_rate=.01,system_f1=.98,accuracy=.97,
        macro_f1=.96,nll=.1,recall={r:.9 for r in ROLES},confusion=[[1]*5 for _ in ROLES],system_auroc=.99,system_average_precision=.99)
    value=dict(min33=.94,system_threshold=.6,groups={g:copy.deepcopy(metric) for g in groups},old5_mean_f1=.98)
    index={n:dict(record=n,validation=copy.deepcopy(value)|dict(min33=.91)) for n in DEVELOPMENT_RECORDS};index['new-max33']['validation']['min33']=.95
    zero={n:copy.deepcopy(value)|dict(min33=.91) for n in ['zero-oldmean','zero-max33']};legacy=copy.deepcopy(value)
    protocol=dict(models=1,training_updates=0,fixed_model_sha256=NEW_MODEL_SHA,threshold_budget=501,strict_controls=STRICT,guard_controls=GUARDS+['original-savedGPU'],absolute_min33=.9,max_FPR=.05,tolerance=1e-12,
        tests_read=False,primary_changed=False,groups=groups,feasible=feasible,floor=dict(old4_f1_loss=.02,old4_accuracy_loss=.01,old5_mean_f1_loss=.005,full_tail_nonregression=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail'],public_f1_loss=.02,public_accuracy_loss=.01))
    gate=dict(eligible=True,rejection_reasons=[],guards={n:dict(eligible=True,rejection_reasons=[],legacy_savedGPU_only=n=='original-savedGPU') for n in GUARDS+['original-savedGPU']},min33_gains={n:.94-.91 for n in STRICT})
    grid=dict(no_fallback=True,selected_index=0,grid_thresholds=2,threshold_budget=501,admissible_thresholds=1)
    decision=dict(candidate='new-guard-max33',eligible=True,no_fallback=True,system_threshold=.6,validation=value,gate=gate,grid_proof=grid,
        model_sha256=NEW_MODEL_SHA,model_path=str(model_paths()['new']),original_trial_unchanged=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)
    check('eligible_only_decision_without_promoted_is_supported',validate_calibrated_decision(decision,protocol,index,zero,legacy)==.6 and 'promoted' not in decision)
    bad=copy.deepcopy(decision);bad['eligible']=False;reject('ineligible_calibration_cannot_enable_snapshot',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    bad=copy.deepcopy(decision);bad['gate']['guards']['original-savedGPU']['eligible']=False;reject('one_failed_original_GPU_guard_blocks',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    bad=copy.deepcopy(decision);bad['gate']['min33_gains'][STRICT[0]]=0;reject('strict_tie_blocks_no_fallback',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    bad=copy.deepcopy(decision);bad['validation']['groups'][groups[0]]['system_precision']=.89999;reject('any_one_absolute90_stat_fails',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    bad=copy.deepcopy(decision);bad['validation']['groups'][feasible[0]]['system_false_positive_rate']=.05+1e-13;reject('original_exact_FPR_not_relaxed',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    bad=copy.deepcopy(decision);bad['validation']['groups'][groups[0]]['accuracy']=.98;reject('calibration_cannot_claim_fiveway_gain',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    bad=copy.deepcopy(decision);bad['candidate']='new-max33';reject('old_failed_record_cannot_be_relabelled_eligible',lambda:validate_calibrated_decision(bad,protocol,index,zero,legacy))
    # Full new certificate adapter roundtrip with metadata-only /tmp files.
    train=folder/'original';cal=folder/'calibration';cal.mkdir();train.mkdir();source=folder/'calibration.py';source.write_text('# synthetic source only\n');audsource=folder/'auditor.py';audsource.write_text('# synthetic independent auditor\n');audpath=folder/'actual-audit.json'
    def put(path,v):path=Path(path);path.write_text(json.dumps(v,allow_nan=False)+'\n');return digest(path)
    put(train/'zero-controls.json',dict(records=zero));put(train/'controls.json',dict(original_savedGPU_validation=legacy));oldselection=dict(promoted=False,rejection_reasons=['original-guard-failed']);put(train/'selection.json',oldselection)
    decision['model_path']=str(train/'models/new.npz');decl=dict(protocol=dict(groups=groups,feasible=feasible),static=dict(sample_ix_sha256={b:'f'*64 for b in ['old','c4','public']}))
    probabilities={b:dict(path=str(folder/(b+'-p.npz')),sha256='a'*64) for b in ['old','c4','public']};source_rows={b:dict(path=str(folder/(b+'-rows.parquet')),sha256='b'*64,sample_ix_sha256='f'*64,rows=n) for b,n in [('old',173032),('c4',61440),('public',16713)]}
    rawpaths={v['path'] for v in list(probabilities.values())+list(source_rows.values())}
    oldaudit=dict(selection=oldselection,probability_bindings={v['path']:v['sha256'] for v in probabilities.values()},provenance_sha256={v['path']:v['sha256'] for v in source_rows.values()})
    static=dict(upstream_actual_audit_sha256=ACTUAL_AUDIT_SHA,upstream_complete_sha256=TRAINER_COMPLETE_SHA,upstream_protocol_sha256=TRAINER_PROTOCOL_SHA,original_records_sha256=ORIGINAL_RECORDS_SHA,
        all14_plus_zero_untouched=True,model_sha256=NEW_MODEL_SHA,model_path=decision['model_path'],original_selection=oldselection,source_sha256=digest(source),probability_files_opened=False,row_tables_materialized=False,raw_features_needed=False,
        baseline_validation=index['new-max33']['validation'],controls={n:zero[n] if n in zero else index[n]['validation'] for n in GUARDS},legacy=legacy,probabilities=probabilities,source_rows=source_rows)
    psha=put(cal/'protocol.json',dict(protocol=protocol,static=static));decision.update(source_sha256=digest(source),protocol_sha256=psha,upstream_actual_audit_sha256=ACTUAL_AUDIT_SHA)
    dsha=put(cal/'selection.json',decision);put(cal/'threshold-ledger.json',[dict(index=0,eligible=True,rejection_reasons=[],system_threshold=.6,effective_F32_threshold=float(np.float32(.6)),min33=.94),dict(index=1,eligible=False)])
    (cal/'validation-metrics.csv').write_text('synthetic only\n');(cal/'report.txt').write_text('synthetic metadata\n');prob={v['path']:v['sha256'] for v in probabilities.values()};data={**prob,**{v['path']:v['sha256'] for v in source_rows.values()}}
    complete=dict(stage='completed_guard_constrained_development_calibration',source_sha256=digest(source),protocol_sha256=psha,eligible=True,training_updates=0,gpu_used=False,tests_read=False,primary_changed=False,new_independent_confirmation_required=True,
        artifact_sha256={n:digest(cal/n) for n in ['selection.json','threshold-ledger.json','validation-metrics.csv','report.txt']},probability_bindings=prob,data_bindings=data)
    csha=put(cal/'complete.json',complete);provenance={str(p):digest(p) for p in [source,cal/'protocol.json',cal/'complete.json']+[cal/n for n in complete['artifact_sha256']]}|data
    certificate=dict(stage=CALIBRATION_AUDIT_STAGE,audit_passed=True,test_data_or_results_read=False,gpu_used=False,auditor_source_path=str(audsource),auditor_source_sha256=digest(audsource),calibration_source_sha256=digest(source),
        calibration_protocol_sha256=psha,calibration_complete_sha256=csha,calibration_selection_sha256=dsha,upstream_actual_audit_sha256=ACTUAL_AUDIT_SHA,model_sha256=NEW_MODEL_SHA,decision=decision,probability_bindings=prob,provenance_sha256=provenance)
    asha=put(audpath,certificate)
    own=dict(TRAINER=train,CALIBRATION=cal,CALIBRATION_SOURCE=source,CALIBRATION_SOURCE_SHA=digest(source),CALIBRATION_PROTOCOL_SHA=psha,CALIBRATION_COMPLETE_SHA=csha,CALIBRATION_DECISION_SHA=dsha,
        CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA=digest(audsource),CALIBRATION_ACTUAL_AUDIT_PATH=audpath,CALIBRATION_ACTUAL_AUDIT_SHA=asha)
    real_digest=digest
    def metadata_digest(path):
        require(str(path) not in rawpaths,'Raw DEV file unexpectedly opened.');return real_digest(path)
    with patch.dict(globals(),own|dict(digest=metadata_digest)),patch.object(np,'load',forbidden):
        result=calibration_material({},oldaudit,decl,index=index)
        check('exact_new_certificate_chain_metadata_only_roundtrip',result['decision']['system_threshold']==.6 and not any(Path(p).exists() for p in rawpaths))
        check('new_threshold_record_owned_by_newdecision_control_records_original',SOURCE_RECORD['new-guard-max33']['origin']=='guard-calibration-decision' and all(SOURCE_RECORD[n]['origin']=='original-readout-records' for n in RECORD_TO_MODEL if n!='new-guard-max33'))
        check('original_trial_negative_metadata_remains_unmodified',read(train/'selection.json')==oldselection and result['snapshot_certificate']['original_trial_promoted'] is False)
        with patch.dict(globals(),dict(CALIBRATION_ACTUAL_AUDIT_SHA='0'*64)):reject('wrong_exact_new_actual_certificate_SHA_rejected',lambda:calibration_material({},oldaudit,decl,index=index))
        with patch.dict(globals(),dict(CALIBRATION_DECISION_SHA='0'*64)):reject('wrong_exact_new_threshold_decision_SHA_rejected',lambda:calibration_material({},oldaudit,decl,index=index))
        saved=(cal/'selection.json').read_bytes();bad=copy.deepcopy(decision);bad['system_threshold']=.5;put(cal/'selection.json',bad)
        reject('same_model_different_threshold_cannot_replace_frozen_decision',lambda:calibration_material({},oldaudit,decl,index=index));(cal/'selection.json').write_bytes(saved)
        saved=audpath.read_bytes();bad=copy.deepcopy(certificate);bad['upstream_actual_audit_sha256']='0'*64;badsha=put(audpath,bad)
        with patch.dict(globals(),dict(CALIBRATION_ACTUAL_AUDIT_SHA=badsha)):reject('rehashed_certificate_cannot_switch_original7bac',lambda:calibration_material({},oldaudit,decl,index=index))
        audpath.write_bytes(saved)
    check('reserved_builder_design_path_preserved_without_new_source_selection',OUT/'root-prospective-design.json'==SOURCE_RESERVATION_DESIGN and DESIGN==HELPER_PREPARATION/'conditional-helper-design-v2.json')
    check('all_final_calibration_pins_restored_original',{name:globals()[name] for name in final}==initial_final_pins)
    check('source_unchanged',digest(__file__)==start);py_compile.compile(__file__,cfile=str(folder/'helper.pyc'),doraise=True)
    report=dict(stage='synthetic_CPU_guard_first_closed_snapshot_additional_certificate_preflight',source_sha256=start,checks=checks,checks_passed=len(checks),fixture_directory=str(folder),
        original_helper_source_sha256='6d9e91633b74697b0fba9de42eaa6b354439658afb4508e7103a57e8413703fc',eligible_only_schema=True,
        actual_snapshot_declared=False,actual_development_metadata_read=False,raw_feature_or_probability_or_row_arrays_read=False,real_models_or_scalers_or_tokenizer_loaded=False,
        real_source_IDs_or_bodies_selected=False,gpu_used=False,formal_outputs_written=False,old_geometry_tests_repeated=False)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),source_sha256=start,checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group();m.add_argument('--preflight',action='store_true');m.add_argument('--declare',action='store_true');m.add_argument('--verify',type=Path)
    parser.add_argument('--audit',type=Path,help='Original failed readout actual-audit certificate.');parser.add_argument('--audit-source',type=Path);parser.add_argument('--calibration-audit-source',type=Path);parser.add_argument('--runtime-source',type=Path,action='append',default=[]);parser.add_argument('--out',type=Path);a=parser.parse_args()
    if a.preflight:preflight()
    elif a.verify:pin_gate();verify_closed_snapshot(read(a.verify))
    elif a.declare:
        pin_gate();request=dict(audit_path=a.audit or ACTUAL_AUDIT_PATH,extra_runtime_sources=a.runtime_source)
        if a.audit_source:request['audit_source_path']=a.audit_source
        if a.calibration_audit_source:request['calibration_audit_source_path']=a.calibration_audit_source
        declare_snapshot(request,out=a.out)
    else:parser.print_help()


if __name__=='__main__':main()
