"""Independent closed-helper review using only the owner's existing CPU/tmp toy."""
import ast,copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
P=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/systemness_complementary_confirm_snapshot.py');PIN='6d9e91633b74697b0fba9de42eaa6b354439658afb4508e7103a57e8413703fc'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)==PIN
spec=importlib.util.spec_from_file_location('snapshot_annex',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(n,v=True):assert bool(v),n;checks.append(n)
def reject(n,f):
    try:f()
    except(RuntimeError,ValueError,KeyError):checks.append(n);return
    raise AssertionError(n)
def forbidden(*a,**k):raise LookupError('Formal artifact access forbidden')
# All six final-pin failures are proven before JSON/hash/NPZ/stat calls.
fields=['TRAINER_SOURCE_SHA','TRAINER_PROTOCOL_SHA','TRAINER_COMPLETE_SHA','APPROVED_AUDITOR_SOURCE_SHA','ACTUAL_AUDIT_SHA','ACTUAL_AUDIT_PATH']
for missing in fields:
    values={f:('a'*64 if f!='ACTUAL_AUDIT_PATH' else Path('/tmp/synthetic-audit.json')) for f in fields};values[missing]=None
    with patch.dict(m.__dict__,values|{'read':forbidden,'digest':forbidden}),patch.object(np,'load',forbidden):
        reject('missing_'+missing+'_rejects_before_any_formal_IO',m.build_conditional_snapshot)
# Locate only the published SYNTHETIC fixture, never any production result.
report_path=Path('/tmp/systemness-complementary-confirm-snapshot-preflight-t_nfyxva/report.json');assert sha(report_path)=='9703e64b96968cc27860a325f22ff110d524374306b0d6f9a44e2453e28df91c'
r=json.loads(report_path.read_text());fixture=Path(r['fixture_directory']);assert str(fixture).startswith('/tmp/')
load=lambda p:json.loads(Path(p).read_text())
snap=load(fixture/'snapshot/readout-snapshot.json');gate=snap['gate'];fake=fixture/'metadata';train=fake/'systemness-complementary-head-readout';old=fake/'systemness-head-probes';auditpath=fixture/'actual-audit.json';audit=load(auditpath);proto=load(train/'protocol.json');plan=load(train/'search-plan.json')
orig=[fake/'training_inputs.parquet',fixture/'legacy/raw_training_data.json'];raw=set(audit['probability_bindings'])|set(proto['static']['feature_files_sha256'])
assert all(not Path(p).exists() for p in raw)
own=dict(BASE=fake,TRAINER=train,TRAINER_SOURCE=fixture/'readout.py',TRAINER_SOURCE_SHA=gate['trainer_source_sha256'],TRAINER_PROTOCOL_SHA=gate['trainer_protocol_sha256'],TRAINER_COMPLETE_SHA=gate['trainer_complete_sha256'],APPROVED_AUDITOR_SOURCE_SHA=gate['audit_source_sha256'],ACTUAL_AUDIT_PATH=auditpath,ACTUAL_AUDIT_SHA=gate['audit_sha256'],DESIGN=fixture/'design.json',DESIGN_SHA=sha(fixture/'design.json'),OLD_HEAD=old,OLD_MODEL_SHA=snap['physical_model_sha256']['old-head'],EXTRA_OUT=fake/'systemness-complementary-head-features',CHECKPOINT=fixture/'checkpoint',SCALER=fixture/'cache/scalers.npz',ORIGINAL249=orig,FUTURE_DATA=[(fixture/'future-c4',fixture/'cache/c4'),(fixture/'future-natural',fixture/'cache/natural')])
with patch.dict(m.__dict__,own):
    real_digest=m.digest;real_read=m.read
    def safe_digest(p):
        assert str(Path(p)) not in raw,'Raw bank/probability path unexpectedly read or hashed'
        return real_digest(p)
    def safe_read(p):
        assert Path(p) not in orig,'Historical249 content unexpectedly parsed as JSON'
        return real_read(p)
    with patch.object(m,'digest',safe_digest),patch.object(m,'read',safe_read):
        paths,tau,heads=m.verify_closed_snapshot(snap)
        check('complete_existing_tmp_snapshot_rebuilt_without_any_rawbank_or_pfile',set(paths)=={'new','matched','old-head'} and len(tau)==4 and heads==m.ORIGINAL+m.ADDED)
        check('oldhead_two_tau_records_alias_one_probability_model',snap['record_to_physical']['oldhead-max33']==snap['record_to_physical']['oldhead-original']=='old-head')
        check('three_feature_views_have_exact640_duplicate640_and512_orders',snap['physical_feature_views']['matched']['selected_head_ids']==m.ORIGINAL+m.ORIGINAL[:2] and snap['physical_feature_views']['new']['selected_head_ids']==m.ORIGINAL+m.ADDED and snap['physical_feature_views']['old-head']['dimensions']==512 and 'ignores real capture[512:]' in snap['physical_feature_views']['matched']['view'])
        check('historical249_bytes_never_JSON_parsed',not any(p in snap['runtime_files_sha256'] for p in map(str,orig)))
        for field,mut in [('record_thresholds',lambda v:v.__setitem__('new-max33',.1)),('physical_feature_views',lambda v:v['matched'].__setitem__('view','capture640')),('physical_model_sha256',lambda v:v.__setitem__('new','1'*64))]:
            bad=copy.deepcopy(snap);mut(bad[field]);bad.pop('snapshot_sha256');bad['snapshot_sha256']=m.canonical(bad)
            reject('recomputed_snapshot_digest_cannot_replace_'+field,lambda bad=bad:m.verify_closed_snapshot(bad))
        bad=copy.deepcopy(snap);bad['runtime_files_sha256'].pop(str(train/'models/new.npz'));bad.pop('snapshot_sha256');bad['snapshot_sha256']=m.canonical(bad)
        reject('recomputed_snapshot_digest_cannot_omit_actual_audited_model',lambda:m.verify_closed_snapshot(bad))
    # The root-discovered real closure gap is represented: none of the eight
    # future-only exclusions is in the training audit provenance.
    exclusion_paths={str(fake/n) for n in m.EXCLUSION_NAMES}|set(map(str,orig))
    check('eight_future_only_exclusions_absent_from_training_audit_but_frozen',len(exclusion_paths)==8 and not exclusion_paths&set(audit['provenance_sha256']) and set(snap['future_exclusion_files_sha256'])==exclusion_paths and all(snap['future_exclusion_files_sha256'][p]==sha(p) for p in exclusion_paths))
    exclusion=Path(next(iter(sorted(exclusion_paths))));before=exclusion.read_bytes()
    try:
        exclusion.write_bytes(before+b' changed')
        reject('future_only_exclusion_frozen_bytes_cannot_change',lambda:m.verify_closed_snapshot(snap))
    finally:exclusion.write_bytes(before)
    # This exception must not extend to model/scaler/certificate ownership.
    audit_bytes=auditpath.read_bytes();bad_audit=copy.deepcopy(audit)
    bad_audit['provenance_sha256'].pop(str(old/'standardization.npz'))
    try:
        auditpath.write_text(json.dumps(bad_audit)+'\n')
        with patch.object(m,'ACTUAL_AUDIT_SHA',sha(auditpath)):
            reject('model_scaler_still_requires_direct_approved_audit_provenance',m.build_conditional_snapshot)
    finally:auditpath.write_bytes(audit_bytes)

    # Portable tiny archives only: actual weights are never opened.
    with np.load(old/'models/joint-mlp128.npz',allow_pickle=False) as z:base={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')};om,os=z['mean'].copy(),z['scale'].copy()
    with np.load(fixture/'cache/scalers.npz',allow_pickle=False) as z:scalers={a:(z[a+'_mean'].copy(),z[a+'_scale'].copy()) for a in ['new','matched']}
    scalers['old-head']=(om,os)
    info,_=m.portable_metadata(train/'models/matched.npz','matched',base,scalers,plan,proto['static'])
    check('portable_duplicate640_mean_scale_and_base_bits_checked',info['selected_head_ids']==m.ORIGINAL+m.ORIGINAL[:2])
    with np.load(train/'models/matched.npz',allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
    local=Path(tempfile.mkdtemp(prefix='feature-review-closed640-toy-',dir='/tmp'))
    def write(values):p=local/'fault.npz';np.savez_compressed(p,**values);return p
    bad={k:v.copy() for k,v in payload.items()};bad['mean'][512:]=scalers['new'][0][512:]
    reject('matched_cannot_embed_real_added128_scaler',lambda:m.portable_metadata(write(bad),'matched',base,scalers,plan,proto['static']))
    bad={k:v.copy() for k,v in payload.items()};bad['weight::base.head.bias'][2]=np.nextafter(bad['weight::base.head.bias'][2],np.float32(np.inf))
    reject('single_bit_original512_base_change_rejected',lambda:m.portable_metadata(write(bad),'matched',base,scalers,plan,proto['static']))
    # Approved audit bytes must match exact root pin, not merely compatible schema.
    with patch.object(m,'ACTUAL_AUDIT_SHA','0'*64):reject('wrong_exact_actual_certificate_SHA_rejected',m.audit_input)
    with patch.object(m,'APPROVED_AUDITOR_SOURCE_SHA','0'*64):reject('wrong_exact_actual_auditor_source_SHA_rejected',m.audit_input)
# Static source-path and promotion closure, no production helpers imported/called.
tree=ast.parse(P.read_text());funs={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
check('all14_records_and_zero_with5strict8guard_savedGPU_promotion',all(s in funs['records_and_gate'] for s in ['len(records) == 14',"selection['promoted'] is True",'set(selection[\'guards\']) == set(GUARDS) | {\'original-savedGPU\'}',"candidate['min33'] >= 0.9",'set(selection[\'min33_gains\']) == set(STRICT)']) and len(m.STRICT)==5 and len(m.GUARDS)==8)
check('raw_paths_certified_without_stat_hash_or_load',all(s in funs['collect_material'] for s in ["if raw_artifact(p):",'certified_raw[key] = want','raw_bytes_rehashed_by_helper=False','required_at_inference=False']) and 'np.memmap' not in funs['collect_material'])
check('closed_explicit_lists_no_growing_inventory_or_production_verifier',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['rglob','glob','verified_completion','source_signature','frozen_snapshot','verify_exclusions'] for n in ast.walk(tree)))
reject('workspace_path_rejected_before_access',lambda:m.clean_path('/workspace/config.json'))
check('source_unchanged',sha(P)==PIN)
folder=Path(tempfile.mkdtemp(prefix='feature-review-closed640-snapshot-annex-',dir='/tmp'));out=folder/'report.json';value=dict(stage='independent_source_tiny_CPU_closed640_snapshot_review',passed=True,mustfix=[],source_path=str(P),source_sha256_before=PIN,source_sha256_after=sha(P),synthetic_fixture_report=str(report_path),synthetic_fixture_report_sha256=sha(report_path),check_count=len(checks),checks=checks,final_pins_unknown=True,supersedes_1f757_known_future_exclusion_closure_failure=True,actual_snapshot_built_or_declared=False,actual_source_IDs_or_body_read=False,real_portable_models_read=False,raw_devbank_or_probability_files_accessed=False,production_modules_imported=False,gpu_used=False,formal_outputs_written=False)
out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(report=str(out),report_sha256=sha(out),script_sha256=sha(__file__),source_sha256=sha(P),checks=len(checks))))
