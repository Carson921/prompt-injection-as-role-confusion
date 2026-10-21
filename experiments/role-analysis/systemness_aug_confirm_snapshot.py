"""Conditional closed readout snapshot for a future528-source confirmation.

Default help; --preflight is synthetic CPU /tmp only. Real declaration requires
new-maximin promotion AND an independently completed development audit. No test
source is read here. Old dynamic verifiers/global manifest discovery are never
called. Raw feature/probability arrays are audit provenance, not runtime inputs.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / 'outputs/local-gptoss120b'
OUT = BASE / 'systemness-long-augmentation-confirmation-preparation'
TRAINER = BASE / 'systemness-long-augmentation-probes'
OLD_HEAD = BASE / 'systemness-head-probes'
NEW_DATA = BASE / 'systemness-long-augmentation'
OLD_BANK = Path('/dev/shm/role-analysis-top8-heads')
NEW_BANK = Path('/dev/shm/role-analysis-long-augmentation-heads')
CHECKPOINT = Path('/dev/shm/role-analysis-gptoss120b')
FUTURE_TEST_OUT = BASE / 'systemness-long-augmentation-confirmation'
FUTURE_TEST_CACHE = BASE / 'systemness-expansion/cache/long-augmentation-confirmation'
APPROVED_AUDITOR_SOURCE_SHA = 'e06738da7ebe2ae4b2b3a730d587520e3240b5838fe093b000f7d04ac4dc5282'
OLD_MODEL_SHA = '27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
OLD_THRESHOLD = .620969696969697
AUDIT_STAGE = 'actual_completed_long_augmentation_development_audit'
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
RECORD_TO_MODEL = {'new-maximin': 'new', 'matched-maximin': 'matched',
    'old-head-maximin': 'old-head', 'old-head-original': 'old-head'}
SOURCE_RECORD = {name: ('old-head-old-mean' if name == 'old-head-original' else name) for name in RECORD_TO_MODEL}
TRAIN_ARTIFACTS = ['protocol.json', 'search-plan.json', 'complete.json', 'selection.json', 'records.json',
    'draw-manifest.json', 'old-head-control.json', 'source-counts.json', 'validation-metrics.csv', 'report.txt',
    'candidates/new.json', 'candidates/matched.json', 'training/new.json', 'training/matched.json']
EXCLUSION_NAMES = ['feature-search/fresh/document-keys.json', 'systemness-expansion/document-keys.json',
    'systemness-long-full/document-keys.json', 'systemness-long-full/confirmation/document-keys.json',
    'systemness-head-confirmation/document-keys.json', 'systemness-long-augmentation/document-keys.json']


def require(value, message):
    if not value: raise RuntimeError(message)


def clean_path(path):
    value = str(path)
    require(not (value == '/workspace' or value.startswith('/workspace/')), 'Forbidden inaccessible /workspace path.')
    result = Path(path).resolve()
    require(not str(result).startswith('/workspace/'), 'Dependency resolves to forbidden /workspace.')
    return result


def digest(path):
    path = clean_path(path); value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(clean_path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def protocol_spec():
    return dict(version=1, experiment='conditional-closed-augmentation-confirmation-snapshot', future_source_documents=528,
        gate='Formal trainer completion + candidate new-maximin promotedTrue + actual independent development-result audit passed; never fallback.',
        approved_independent_auditor_source_sha256=APPROVED_AUDITOR_SOURCE_SHA,
        physical_models=['new', 'matched', 'old-head'], record_to_model=RECORD_TO_MODEL, source_records=SOURCE_RECORD,
        original_old_head_sha256=OLD_MODEL_SHA, original_threshold=OLD_THRESHOLD,
        model_geometry=dict(kind='mlp', hidden=128, dimensions=512, heads=8, head_dim=64, roles=ROLES),
        closure='Only explicit immutable SHA maps frozen now; no rglob, old exclusion_inputs/frozen_snapshot, source_signature, load_verified_data or verified_completion.',
        raw_artifacts='Audited features/probabilities carry declared SHA, size, shape and dtype only; never reread to verify inference runtime closure.',
        checkpoint='Actual hashes of config/tokenizer/verified-weights manifest; no claim to rehash all model shards.',
        statement='All model/threshold/head/scaler/source dependencies freeze before any new test-source body access.',
        primary_changed=False, test_source_body_read=False, test_probabilities_read=False,
        old_results_reselected=False, old_raw_banks_deleted=False)


def metadata_files():
    files = [TRAINER / name for name in TRAIN_ARTIFACTS]
    files += [OLD_HEAD / name for name in ['search-plan.json', 'complete.json', 'selection.json', 'candidates/joint-mlp128.json']]
    files += [BASE / 'systemness-head-features' / name for name in ['plan.json', 'extraction-complete.json']]
    files += [NEW_DATA / name for name in ['plan.json', 'data-complete.json', 'document-keys.json',
        'extraction-plan.json', 'extraction-complete.json', 'exclusion-audit.json', 'scan-audit.json', 'independent-validation.json']]
    files += [BASE / 'systemness-expansion' / name for name in ['plan.json', 'data-complete.json', 'supplement-data-complete.json']]
    files += [BASE / 'systemness-long-full' / name for name in ['plan.json', 'data-complete.json', 'extraction-complete.json']]
    files += [BASE / 'systemness-head-scan' / name for name in ['plan.json', 'scan-complete.json', 'validation-complete.json', 'head-selection.json']]
    files += [BASE / name for name in EXCLUSION_NAMES]
    files += [OLD_BANK / 'rows.parquet', NEW_BANK / 'rows.parquet', BASE / 'systemness-expansion/cache/long-augmentation/rows.parquet',
        BASE / 'training_inputs.parquet', BASE.parent / 'local-gptoss20b/raw_training_data.json', BASE / 'checkpoint-cache-verification.json']
    return sorted(set(clean_path(p) for p in files))


def models_and_scaler():
    return {'new': TRAINER / 'models/new.npz', 'matched': TRAINER / 'models/matched.npz',
        'old-head': OLD_HEAD / 'models/joint-mlp128.npz'}, OLD_HEAD / 'standardization.npz'


def portable_metadata(path, scaler_mean, scaler_scale, selected, *, arm=None, plan=None):
    """Inspect portable NPZ only. No network, torch, LM or prediction API."""
    with np.load(clean_path(path), allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'])); mean = z['mean'].copy(); scale = z['scale'].copy()
        require(meta['roles'] == ROLES and meta['kind'] == 'mlp' and meta['hidden'] == 128
            and (meta['dimensions'], meta['heads'], meta['head_dim']) == (512, 8, 64)
            and meta['selected_head_ids'] == selected, 'Portable roles/geometry/ranked head order changed.')
        require(mean.shape == scale.shape == (512,) and mean.dtype == scale.dtype == np.float32
            and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()
            and np.array_equal(mean, scaler_mean) and np.array_equal(scale, scaler_scale), 'Embedded scaler differs from the original frozen fit-only scaler.')
        required = {'weight::encoder.0.weight', 'weight::encoder.0.bias', 'weight::head.weight', 'weight::head.bias'}
        require(required <= set(z.files), 'Portable MLP weight arrays are missing.')
    if arm is not None:
        require(meta['arm'] == arm and meta['signature'] == plan['signature']
            and meta['plan_sha256'] == digest(TRAINER / 'search-plan.json')
            and meta['initial_weight_sha256'] == plan['initial_weight_sha256'], 'New/matched portable training ownership changed.')
    return dict(metadata_sha256=canonical(meta), mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
        scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(), roles=ROLES, dimensions=512, hidden=128,
        selected_head_ids=selected, kind='mlp')


def audit_input(value):
    if isinstance(value, (str, Path)): value = dict(audit_path=str(value))
    require(isinstance(value, dict) and 'audit_path' in value, 'Provide an explicit completed actual-audit JSON path.')
    path = clean_path(value['audit_path']); audit = read(path)
    require(audit['stage'] == AUDIT_STAGE and audit['audit_passed'] is True
        and audit['test_data_or_results_read'] is False and audit['gpu_used'] is False,
        'A completed independent actual development audit is required, not a synthetic/failed audit.')
    source = clean_path(value.get('audit_source_path', audit['auditor_source_path']))
    require(digest(source) == audit['auditor_source_sha256'] == APPROVED_AUDITOR_SOURCE_SHA, 'Actual auditor source SHA is not the prospectively approved version.')
    provenance = audit['provenance_sha256']
    require(isinstance(provenance, dict) and provenance and all(valid_sha(x) for x in provenance.values()), 'Audit provenance map is invalid.')
    return value, path, source, audit, {str(clean_path(p)): s for p, s in provenance.items()}


def records_and_gate(audit):
    require((TRAINER / 'complete.json').is_file(), 'Trainer has not completed; no conditional snapshot may be declared.')
    complete = read(TRAINER / 'complete.json'); selection = read(TRAINER / 'selection.json'); records = read(TRAINER / 'records.json')
    require(selection['candidate'] == 'new-maximin' and selection['promoted'] is True
        and selection['no_fallback'] is True and selection['primary_changed'] is False
        and selection['tests_read'] is False and selection['new_independent_confirmation_required'] is True,
        'new-maximin was not promoted; no fallback snapshot is allowed.')
    require(complete['tests_read'] is False and complete['primary_changed'] is False
        and complete['new_independent_confirmation_required'] is True, 'Development completion contract differs.')
    expected = {'trainer_complete_sha256': TRAINER / 'complete.json', 'trainer_protocol_sha256': TRAINER / 'protocol.json',
        'trainer_selection_sha256': TRAINER / 'selection.json', 'trainer_records_sha256': TRAINER / 'records.json'}
    for field, path in expected.items(): require(audit[field] == digest(path), 'Actual audit belongs to a different ' + field)
    require(audit['trainer_source_sha256'] == digest(HERE / 'systemness_long_augmentation_probes.py')
        == complete['source_sha256'], 'Actual audit/trainer source differs.')
    index = {r['record']: r for r in records}
    require(len(records) == len(index) == 6 and set(index) == {m + '-' + o for m in ['old-head', 'new', 'matched'] for o in ['old-mean', 'maximin']}, 'Development six-record coverage changed.')
    for name, s in complete['artifact_sha256'].items(): require(digest(TRAINER / name) == s, 'Trainer completion artifact changed: ' + name)
    for arm in ['new', 'matched']:
        require(audit['trainer_model_record_sha256'][arm] == complete['candidate_sha256'][arm]
            == digest(TRAINER / 'candidates' / (arm + '.json')), 'Actual audit/candidate record changed.')
        candidate = read(TRAINER / 'candidates' / (arm + '.json'))
        require(digest(TRAINER / 'training' / (arm + '.json')) == candidate['training_complete_sha256'], 'Best-model/early-stop evidence changed.')
    thresholds = {name: index[source]['validation']['system_threshold'] for name, source in SOURCE_RECORD.items()}
    require(all(isinstance(t, (float, int)) and np.isfinite(t) and 0 <= t <= 1.00001 for t in thresholds.values())
        and thresholds['old-head-original'] == OLD_THRESHOLD
        and selection['system_threshold'] == thresholds['new-maximin'], 'Frozen four-record thresholds changed.')
    require(selection['promotion_rejection_reasons'] == [] and all(g['eligible'] is True for g in selection['guards'].values())
        and set(selection['guards']) == {'matched-maximin', 'old-head-maximin', 'old-head-old-mean'}
        and all(np.isfinite(g) and g > 1e-12 for g in selection['min24_gains'].values())
        and set(selection['min24_gains']) == {'matched-maximin', 'old-head-maximin'}, 'Promotion guard proof is inconsistent.')
    return complete, selection, index, thresholds


def dependency_sources(objects):
    """Walk already named metadata dictionaries only, never filesystem trees."""
    output = {str(clean_path(__file__)): digest(__file__)}
    def add(name, expected):
        if not isinstance(name, str) or not name.endswith('.py') or not valid_sha(expected): return
        path = clean_path(name if Path(name).is_absolute() else HERE / name)
        require(path.is_file() and digest(path) == expected, 'Closed source dependency changed: ' + str(path))
        prior = output.setdefault(str(path), expected); require(prior == expected, 'Conflicting frozen source revisions.')
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ['dependencies_sha256', 'dependency_sha256'] and isinstance(item, dict):
                    for name, expected in item.items(): add(name, expected)
                if key == 'source_sha256' and isinstance(item, dict):
                    for name, expected in item.items(): add(name, expected)
                walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    for obj in objects: walk(obj)
    return output


def build_conditional_snapshot(audited_dev_snapshot):
    """Pure read-only metadata/portable construction; never declare implicitly."""
    # Promotion/terminal gate precedes reading any source exclusion inventory.
    require((TRAINER / 'complete.json').is_file(), 'Formal trainer completion is absent.')
    require(read(TRAINER / 'selection.json')['promoted'] is True, 'Conditional new-maximin gate failed.')
    request, audit_path, audit_source, audit, provenance = audit_input(audited_dev_snapshot)
    complete, selection, index, thresholds = records_and_gate(audit)
    files = metadata_files(); metadata = {}
    required_audit = [TRAINER / n for n in ['protocol.json', 'search-plan.json', 'complete.json', 'selection.json', 'records.json', 'old-head-control.json',
        'candidates/new.json', 'candidates/matched.json', 'training/new.json', 'training/matched.json']]
    for path in required_audit:
        require(provenance.get(str(clean_path(path))) == digest(path), 'Independent audit did not bind required trainer metadata: ' + str(path))
    for path in files:
        require(path.is_file(), 'Missing closed metadata dependency: ' + str(path))
        actual = digest(path)
        if str(path) in provenance: require(provenance[str(path)] == actual, 'Audited metadata changed: ' + str(path))
        metadata[str(path)] = actual
    plan = read(TRAINER / 'search-plan.json'); old_features = read(BASE / 'systemness-head-features/extraction-complete.json')
    new_features = read(NEW_DATA / 'extraction-complete.json'); selected = new_features['selected_head_ids']
    require(len(selected) == len(set(selected)) == 8 and all(type(h) is int and 0 <= h < 36 * 64 for h in selected)
        and selected == old_features['selected_head_ids'] == plan['signature']['selected_head_ids']
        == read(BASE / 'systemness-head-scan/head-selection.json')['selected_head_ids'], 'Frozen scanner ranked eight heads changed.')
    require(plan['signature']['new_extraction_sha256'] == digest(NEW_DATA / 'extraction-complete.json')
        and plan['signature']['old_extraction_sha256'] == digest(BASE / 'systemness-head-features/extraction-complete.json')
        and plan['signature']['new_data_sha256'] == digest(NEW_DATA / 'data-complete.json')
        and plan['signature']['old_rows_sha256'] == digest(OLD_BANK / 'rows.parquet')
        and plan['signature']['new_rows_sha256'] == digest(NEW_BANK / 'rows.parquet'), 'Audited data/feature/row completion chain differs.')
    paths, scaler = models_and_scaler(); runtime = {str(clean_path(scaler)): digest(scaler)}
    require(provenance.get(str(clean_path(scaler))) == digest(scaler), 'Independent audit did not bind old fit-only scaler.')
    with np.load(scaler, allow_pickle=False) as z: mean, scale = z['mean'].copy(), z['scale'].copy()
    models = {}; portable = {}
    for name, path in paths.items():
        path = clean_path(path); value = digest(path)
        require(value == audit['model_sha256'][name] == index[('old-head' if name == 'old-head' else name) + '-maximin']['model_sha256']
            and provenance.get(str(path)) == value, 'Audited portable model SHA changed: ' + name)
        if name == 'old-head': require(value == OLD_MODEL_SHA, 'Original frozen head128 changed.')
        else: require(value == read(TRAINER / 'candidates' / (name + '.json'))['model_sha256'], 'Candidate portable ownership differs.')
        runtime[str(path)] = value; models[name] = value
        portable[name] = portable_metadata(path, mean, scale, selected, arm=name if name in ['new', 'matched'] else None, plan=plan)
    require(selection['model_sha256'] == models['new'] and str(clean_path(selection['model_path'])) == str(clean_path(paths['new'])), 'Selected new portable path changed.')
    checkpoint = clean_path(new_features['signature']['checkpoint_path'])
    require(checkpoint == clean_path(CHECKPOINT), 'Verified checkpoint path changed.')
    assets = {'config.json': new_features['signature']['config_sha256'],
        'verified-weights.json': new_features['signature']['verified_weights_sha256'], **new_features['signature']['tokenizer_sha256']}
    for name, expected in assets.items():
        path = checkpoint / name; require(digest(path) == expected, 'Checkpoint metadata changed: ' + name)
        runtime[str(path)] = expected
    config = read(checkpoint / 'config.json')
    require(config['quantization_config']['quant_method'] == 'mxfp4'
        and (config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (36, 64, 64), 'Native MXFP4 geometry changed.')
    # Original249 text inventories are byte-hashed only, never JSON-decoded.
    objects = [read(p) for p in files if p.suffix == '.json' and p.name != 'raw_training_data.json']
    runtime.update(dependency_sources(objects))
    for path in request.get('extra_runtime_sources', []):
        path = clean_path(path); require(path.suffix == '.py', 'Extra runtime dependency must be a Python source.')
        runtime[str(path)] = digest(path)
    metadata[str(audit_path)] = digest(audit_path); runtime[str(audit_source)] = digest(audit_source)
    exclusions = {str(clean_path(BASE / name)): digest(BASE / name) for name in EXCLUSION_NAMES}
    original = {str(clean_path(p)): digest(p) for p in [BASE / 'training_inputs.parquet', BASE.parent / 'local-gptoss20b/raw_training_data.json']}
    artifacts = dict(audited_feature_banks={}, audited_probability_files={})
    for name, evidence, shape in [('old', old_features, [755282, 512]), ('new', new_features, [389120, 512])]:
        artifacts['audited_feature_banks'][name] = dict(sha256=evidence['features_sha256'], shape=shape, dtype='float16',
            size_bytes=int(np.prod(shape)) * 2, raw_bytes_rehashed_by_helper=False, required_at_inference=False)
    for path, expected in provenance.items():
        if 'probabilit' in Path(path).name or Path(path).name.endswith('-validation.npz'):
            artifacts['audited_probability_files'][path] = dict(sha256=expected, raw_bytes_read_by_helper=False, required_at_inference=False)
    result = dict(protocol=protocol_spec(), helper_source_sha256=digest(__file__),
        gate=dict(candidate='new-maximin', promoted=True, independent_actual_development_audit_passed=True,
            audit_path=str(audit_path), audit_sha256=digest(audit_path), audit_source_path=str(audit_source),
            audit_source_sha256=digest(audit_source), trainer_complete_sha256=digest(TRAINER / 'complete.json'),
            trainer_selection_path=str(TRAINER / 'selection.json'), trainer_records_path=str(TRAINER / 'records.json')),
        physical_model_paths={name: str(clean_path(path)) for name, path in paths.items()}, physical_model_sha256=models,
        record_thresholds={k: float(v) for k, v in thresholds.items()}, record_to_physical=RECORD_TO_MODEL,
        source_records=SOURCE_RECORD, selected_head_ids=selected, old_scaler_path=str(clean_path(scaler)), portable_metadata=portable,
        runtime_files_sha256=runtime, certificate_files_sha256=metadata, exclusion_inputs_sha256=exclusions,
        original249_inputs_sha256=original, audited_artifact_provenance=artifacts,
        checkpoint_path=str(checkpoint), runtime_rehashes_all_model_shards=False,
        raw_feature_or_probability_arrays_loaded=False, new_test_source_body_read=False, primary_changed=False)
    result['snapshot_sha256'] = canonical(result)
    verify_closed_snapshot(result)
    return result


def verify_closed_snapshot(snapshot):
    """Rehash only the frozen maps. Later document-key manifests are irrelevant."""
    value = dict(snapshot); expected = value.pop('snapshot_sha256')
    require(canonical(value) == expected and snapshot['protocol'] == protocol_spec()
        and snapshot['helper_source_sha256'] == digest(__file__), 'Closed snapshot contents/source/protocol changed.')
    for kind in ['runtime_files_sha256', 'certificate_files_sha256']:
        require(snapshot[kind] and all(valid_sha(v) for v in snapshot[kind].values()), 'Closed SHA map is missing/invalid.')
        for path, sha in snapshot[kind].items(): require(digest(path) == sha, 'Closed dependency changed: ' + path)
    _, _, _, audit, _ = audit_input(dict(audit_path=snapshot['gate']['audit_path'], audit_source_path=snapshot['gate']['audit_source_path']))
    _, selection, records, thresholds = records_and_gate(audit)
    require(snapshot['gate']['promoted'] is True and snapshot['gate']['independent_actual_development_audit_passed'] is True
        and snapshot['gate']['audit_sha256'] == digest(snapshot['gate']['audit_path'])
        and snapshot['gate']['audit_source_sha256'] == digest(snapshot['gate']['audit_source_path']), 'Actual audited gate differs.')
    require(snapshot['record_to_physical'] == RECORD_TO_MODEL and snapshot['source_records'] == SOURCE_RECORD
        and snapshot['record_thresholds'] == thresholds, 'Four-record mapping or frozen thresholds changed.')
    required_runtime = {str(clean_path(__file__)), str(clean_path(snapshot['old_scaler_path'])), snapshot['gate']['audit_source_path']}
    required_runtime.update(str(clean_path(p)) for p in snapshot['physical_model_paths'].values())
    feature = read(NEW_DATA / 'extraction-complete.json'); checkpoint = clean_path(snapshot['checkpoint_path'])
    assets = {'config.json': feature['signature']['config_sha256'], 'verified-weights.json': feature['signature']['verified_weights_sha256'], **feature['signature']['tokenizer_sha256']}
    require(checkpoint == clean_path(CHECKPOINT), 'Closed checkpoint path changed.')
    required_runtime.update(str(checkpoint / name) for name in assets)
    require(required_runtime <= set(snapshot['runtime_files_sha256']), 'Critical inference runtime dependencies were omitted.')
    for name, expected_sha in assets.items(): require(digest(checkpoint / name) == expected_sha, 'Checkpoint asset differs from audited feature declaration.')
    require(set(str(clean_path(p)) for p in metadata_files()) <= set(snapshot['certificate_files_sha256']), 'Critical development certificate dependency omitted.')
    require(snapshot['exclusion_inputs_sha256'] == {str(clean_path(BASE / name)): snapshot['certificate_files_sha256'][str(clean_path(BASE / name))] for name in EXCLUSION_NAMES}, 'Closed six-manifest exclusion inventory differs.')
    with np.load(snapshot['old_scaler_path'], allow_pickle=False) as z: mean, scale = z['mean'].copy(), z['scale'].copy()
    plan = read(TRAINER / 'search-plan.json'); paths = {name: clean_path(path) for name, path in snapshot['physical_model_paths'].items()}
    require(set(paths) == {'new', 'matched', 'old-head'} and snapshot['physical_model_sha256']['old-head'] == OLD_MODEL_SHA,
        'Three physical models are required.')
    for name, path in paths.items():
        actual_model_sha = digest(path)
        source_record = ('old-head' if name == 'old-head' else name) + '-maximin'
        require(actual_model_sha == snapshot['physical_model_sha256'][name] == audit['model_sha256'][name]
            == records[source_record]['model_sha256']
            and snapshot['runtime_files_sha256'].get(str(path)) == actual_model_sha, 'Physical portable is not the independently audited readout.')
        if name != 'old-head':
            candidate = read(TRAINER / 'candidates' / (name + '.json'))
            require(candidate['model_sha256'] == actual_model_sha
                and clean_path(path) == clean_path(TRAINER / 'models' / (name + '.npz')), 'Candidate audited portable path or SHA differs.')
        meta = portable_metadata(path, mean, scale, snapshot['selected_head_ids'], arm=name if name != 'old-head' else None, plan=plan)
        require(meta == snapshot['portable_metadata'][name], 'Closed portable metadata/scaler changed.')
    require(selection['model_sha256'] == snapshot['physical_model_sha256']['new']
        and clean_path(selection['model_path']) == paths['new'], 'Selected audited new path/SHA differs.')
    return paths, dict(snapshot['record_thresholds']), list(snapshot['selected_head_ids'])


def declare_snapshot(audited_dev_snapshot=None, *, snapshot=None, out=None):
    """Explicit root-only future action; never invoked by default or preflight."""
    require((snapshot is None) != (audited_dev_snapshot is None), 'Provide exactly one audit input or built snapshot.')
    value = build_conditional_snapshot(audited_dev_snapshot) if snapshot is None else snapshot
    verify_closed_snapshot(value)
    folder = clean_path(OUT if out is None else out); path = folder / 'readout-snapshot.json'
    if path.exists(): require(read(path) == value, 'An existing readout statement is immutable.'); return value
    require(not any((FUTURE_TEST_OUT / name).exists() for name in ['data-complete.json', 'document-keys.json'])
        and not any((FUTURE_TEST_CACHE / name).exists() for name in ['inputs.parquet', 'rows.parquet', 'token-ids.json'])
        and not (folder / 'data-complete.json').exists() and not (folder / 'document-keys.json').exists(),
        'New test data exists before readout freeze; refusing retrospective declaration.')
    folder.mkdir(parents=True, exist_ok=True); temporary = folder / 'readout-snapshot.json.tmp'
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(folder, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return value


def preflight():
    """Synthetic certificate/portable faults under /tmp; no actual gate access."""
    import copy
    import py_compile
    folder = Path(tempfile.mkdtemp(prefix='systemness-aug-confirm-snapshot-preflight-', dir='/tmp'))
    source_sha = digest(__file__); checks = []
    original = {name: globals()[name] for name in ['HERE', 'BASE', 'OUT', 'TRAINER', 'OLD_HEAD', 'NEW_DATA',
        'OLD_BANK', 'NEW_BANK', 'CHECKPOINT', 'FUTURE_TEST_OUT', 'FUTURE_TEST_CACHE', 'OLD_MODEL_SHA', 'APPROVED_AUDITOR_SOURCE_SHA']}
    formal = dict(out_exists=OUT.exists(), future_test_out_exists=FUTURE_TEST_OUT.exists())
    def check(name, value=True): require(value, name); checks.append(name)
    def reject(name, fn):
        try: fn()
        except (RuntimeError, ValueError, KeyError, FileNotFoundError): checks.append(name)
        else: raise RuntimeError('Fault fixture accepted: ' + name)
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n' if isinstance(value, (dict, list)) else str(value))
    try:
        globals().update(HERE=folder / 'sources', BASE=folder / 'metadata', OUT=folder / 'snapshot',
            TRAINER=folder / 'metadata/systemness-long-augmentation-probes', OLD_HEAD=folder / 'metadata/systemness-head-probes',
            NEW_DATA=folder / 'metadata/systemness-long-augmentation', OLD_BANK=folder / 'old-bank', NEW_BANK=folder / 'new-bank',
            CHECKPOINT=folder / 'checkpoint', FUTURE_TEST_OUT=folder / 'future-test', FUTURE_TEST_CACHE=folder / 'future-cache')
        for path in metadata_files(): put(path, {} if path.suffix == '.json' else 'Synthetic metadata bytes, never real rows or replies.')
        put(HERE / 'systemness_long_augmentation_probes.py', '# Synthetic trainer source witness.\n')
        audit_source = folder / 'synthetic-auditor.py'; put(audit_source, '# Synthetic-only auditor witness.\n')
        globals()['APPROVED_AUDITOR_SOURCE_SHA'] = digest(audit_source)
        selected = [458, 462, 2045, 460, 459, 2047, 456, 997]
        put(CHECKPOINT / 'config.json', dict(quantization_config=dict(quant_method='mxfp4'), num_hidden_layers=36, num_attention_heads=64, head_dim=64))
        put(CHECKPOINT / 'verified-weights.json', dict(synthetic=True, no_model_shards_exist=True))
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']: put(CHECKPOINT / name, dict(synthetic=True, name=name))
        signature = dict(checkpoint_path=str(CHECKPOINT), config_sha256=digest(CHECKPOINT / 'config.json'),
            verified_weights_sha256=digest(CHECKPOINT / 'verified-weights.json'),
            tokenizer_sha256={name: digest(CHECKPOINT / name) for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']})
        put(BASE / 'systemness-head-features/extraction-complete.json', dict(selected_head_ids=selected, features_sha256='1' * 64))
        put(NEW_DATA / 'extraction-complete.json', dict(selected_head_ids=selected, features_sha256='2' * 64, signature=signature))
        put(BASE / 'systemness-head-scan/head-selection.json', dict(selected_head_ids=selected))
        plan = dict(initial_weight_sha256='3' * 64, signature=dict(selected_head_ids=selected,
            new_extraction_sha256=digest(NEW_DATA / 'extraction-complete.json'), old_extraction_sha256=digest(BASE / 'systemness-head-features/extraction-complete.json'),
            new_data_sha256=digest(NEW_DATA / 'data-complete.json'), old_rows_sha256=digest(OLD_BANK / 'rows.parquet'), new_rows_sha256=digest(NEW_BANK / 'rows.parquet')))
        put(TRAINER / 'search-plan.json', plan)
        paths, scaler = models_and_scaler(); scaler.parent.mkdir(parents=True, exist_ok=True)
        mean, scale = np.zeros(512, np.float32), np.ones(512, np.float32)
        np.savez_compressed(scaler, mean=mean, scale=scale)
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(roles=ROLES, kind='mlp', hidden=128, dimensions=512, heads=8, head_dim=64, selected_head_ids=selected)
            if name != 'old-head': meta.update(arm=name, signature=plan['signature'], plan_sha256=digest(TRAINER / 'search-plan.json'), initial_weight_sha256=plan['initial_weight_sha256'])
            weights = {'weight::encoder.0.weight': np.zeros((128, 512), np.float32), 'weight::encoder.0.bias': np.zeros(128, np.float32),
                'weight::head.weight': np.zeros((5, 128), np.float32), 'weight::head.bias': np.zeros(5, np.float32)}
            np.savez_compressed(path, mean=mean, scale=scale, metadata=np.array(json.dumps(meta)), **weights)
        globals()['OLD_MODEL_SHA'] = digest(paths['old-head'])
        model_hashes = {name: digest(path) for name, path in paths.items()}
        for arm in ['new', 'matched']:
            put(TRAINER / 'training' / (arm + '.json'), dict(synthetic=True, model_sha256=model_hashes[arm]))
            put(TRAINER / 'candidates' / (arm + '.json'), dict(model_sha256=model_hashes[arm], training_complete_sha256=digest(TRAINER / 'training' / (arm + '.json'))))
        records = []
        for model in ['old-head', 'new', 'matched']:
            for objective in ['old-mean', 'maximin']:
                tau = OLD_THRESHOLD if model == 'old-head' and objective == 'old-mean' else .3 if model == 'new' else .4
                records.append(dict(record=model + '-' + objective, model_sha256=model_hashes[model], validation=dict(system_threshold=tau)))
        put(TRAINER / 'records.json', records)
        selection = dict(candidate='new-maximin', promoted=True, no_fallback=True, primary_changed=False, tests_read=False,
            new_independent_confirmation_required=True, system_threshold=.3, model_sha256=model_hashes['new'], model_path=str(paths['new']),
            guards={name: dict(eligible=True) for name in ['matched-maximin', 'old-head-maximin', 'old-head-old-mean']},
            promotion_rejection_reasons=[], min24_gains={'matched-maximin': .01, 'old-head-maximin': .02})
        put(TRAINER / 'selection.json', selection)
        complete = dict(source_sha256=digest(HERE / 'systemness_long_augmentation_probes.py'), tests_read=False, primary_changed=False,
            new_independent_confirmation_required=True, candidate_sha256={arm: digest(TRAINER / 'candidates' / (arm + '.json')) for arm in ['new', 'matched']},
            artifact_sha256={name: digest(TRAINER / name) for name in ['protocol.json', 'search-plan.json', 'selection.json', 'records.json', 'old-head-control.json']})
        put(TRAINER / 'complete.json', complete)
        audit = dict(stage=AUDIT_STAGE, audit_passed=True, test_data_or_results_read=False, gpu_used=False,
            auditor_source_path=str(audit_source), auditor_source_sha256=digest(audit_source),
            trainer_source_sha256=digest(HERE / 'systemness_long_augmentation_probes.py'), trainer_complete_sha256=digest(TRAINER / 'complete.json'),
            trainer_protocol_sha256=digest(TRAINER / 'protocol.json'), trainer_selection_sha256=digest(TRAINER / 'selection.json'),
            trainer_records_sha256=digest(TRAINER / 'records.json'), trainer_model_record_sha256=complete['candidate_sha256'], model_sha256=model_hashes,
            provenance_sha256={str(p): digest(p) for p in metadata_files() + list(paths.values()) + [scaler, audit_source]})
        audit_path = folder / 'synthetic-actual-audit-witness.json'; put(audit_path, audit)
        witness = build_conditional_snapshot(audit_path)
        physical, thresholds, heads = verify_closed_snapshot(witness)
        check('Threephysical models fourrecords with sharedoldhead and exact originaltau', len(physical) == 3 and len(thresholds) == 4
            and thresholds['old-head-original'] == OLD_THRESHOLD and witness['source_records']['old-head-original'] == 'old-head-old-mean'
            and witness['record_to_physical']['old-head-original'] == witness['record_to_physical']['old-head-maximin'] == 'old-head' and heads == selected)
        check('Missing rawbanks and classifier probabilities are allowed as certified provenance', not (OLD_BANK / 'features.f16').exists()
            and not (NEW_BANK / 'features.f16').exists() and witness['raw_feature_or_probability_arrays_loaded'] is False)
        declared = declare_snapshot(snapshot=witness, out=OUT); check('Fsynced immutable declaration roundtrip', read(OUT / 'readout-snapshot.json') == declared)
        put(FUTURE_TEST_OUT / 'document-keys.json', [dict(synthetic_future_key=True)])
        verify_closed_snapshot(witness); declare_snapshot(snapshot=witness, out=OUT)
        check('Later newtest keyinventory cannot invalidate closed verifier or immutable restart')
        reject('No firstdeclaration after testsource artifacts', lambda: declare_snapshot(snapshot=witness, out=folder / 'too-late'))
        altered = copy.deepcopy(witness); altered['record_thresholds']['old-head-original'] = .5
        reject('Reject threshold snapshot tamper', lambda: verify_closed_snapshot(altered))
        altered = copy.deepcopy(witness); altered['runtime_files_sha256'].pop(str(paths['new'])); content = dict(altered); content.pop('snapshot_sha256'); altered['snapshot_sha256'] = canonical(content)
        reject('Reject omitted critical runtime model even with recomputed contenthash', lambda: verify_closed_snapshot(altered))
        replacement = folder / 'unapproved-new.npz'
        with np.load(paths['new'], allow_pickle=False) as z: replacement_arrays = {key: z[key].copy() for key in z.files}
        replacement_arrays['weight::head.bias'][0] = .25; np.savez_compressed(replacement, **replacement_arrays)
        substituted = copy.deepcopy(witness); substituted['physical_model_paths']['new'] = str(replacement)
        substituted['physical_model_sha256']['new'] = digest(replacement)
        substituted['runtime_files_sha256'].pop(str(paths['new'])); substituted['runtime_files_sha256'][str(replacement)] = digest(replacement)
        substituted['portable_metadata']['new'] = portable_metadata(replacement, mean, scale, selected, arm='new', plan=plan)
        content = dict(substituted); content.pop('snapshot_sha256'); substituted['snapshot_sha256'] = canonical(content)
        reject('Reject substituted unaudited model despite valid portable metadata and recomputed snapshot hash', lambda: verify_closed_snapshot(substituted))
        saved = (TRAINER / 'selection.json').read_bytes(); changed = dict(selection, promoted=False); put(TRAINER / 'selection.json', changed)
        reject('No fallback when development notpromoted', lambda: build_conditional_snapshot(audit_path)); (TRAINER / 'selection.json').write_bytes(saved)
        failed = folder / 'failed-audit.json'; put(failed, audit | dict(audit_passed=False))
        reject('Reject failed actual audit witness', lambda: build_conditional_snapshot(failed))
        synthetic = folder / 'synthetic-stage.json'; put(synthetic, audit | dict(stage='synthetic_CPU_preflight'))
        reject('Reject synthetic audit stage', lambda: build_conditional_snapshot(synthetic))
        source_bytes = audit_source.read_bytes(); audit_source.write_bytes(source_bytes + b'# changed\n')
        reject('Reject independently audited source revision change', lambda: verify_closed_snapshot(witness)); audit_source.write_bytes(source_bytes)
        bad = folder / 'bad-portable.npz'
        with np.load(paths['new'], allow_pickle=False) as z: payload = {name: z[name].copy() for name in z.files}
        payload['mean'][0] = 1; np.savez_compressed(bad, **payload)
        reject('Reject embedded scaler drift', lambda: portable_metadata(bad, mean, scale, selected))
        payload['mean'][0] = 0; m = json.loads(str(payload['metadata'])); m['roles'] = ROLES[::-1]; payload['metadata'] = np.array(json.dumps(m)); np.savez_compressed(bad, **payload)
        reject('Reject portable truthrole order drift', lambda: portable_metadata(bad, mean, scale, selected))
        payload['metadata'] = np.array(json.dumps(m | dict(roles=ROLES, selected_head_ids=selected[::-1]))); np.savez_compressed(bad, **payload)
        reject('Reject ranked head order drift', lambda: portable_metadata(bad, mean, scale, selected))
        check('Declaration never creates testsource cache or reads a prompt/body', not FUTURE_TEST_CACHE.exists())
    finally:
        globals().update(original)
    require(digest(__file__) == source_sha and formal == dict(out_exists=OUT.exists(), future_test_out_exists=FUTURE_TEST_OUT.exists()), 'Synthetic preflight changed formal/source state.')
    py_compile.compile(__file__, cfile=str(folder / 'helper.pyc'), doraise=True)
    report = dict(stage='synthetic_CPU_closed_snapshot_preflight', helper_source_sha256=source_sha, checks=checks, checks_passed=len(checks),
        physical_models=3, records=4, fixture_directory=str(folder), actual_development_audit_read=False,
        real_portable_models_read=False, actual_declaration_written=False, raw_feature_or_probability_arrays_read=False,
        new_test_source_body_read=False, gpu_used=False, formal_files_changed=False,
        synthetic_actual_audit_witness='Only a /tmp mock certificate tests gates; this is not an actual development audit or promotion.')
    put(folder / 'report.json', report)
    print(json.dumps(dict(report=str(folder / 'report.json'), checks_passed=len(checks), source_sha256=source_sha)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(); action.add_argument('--preflight', action='store_true'); action.add_argument('--declare', type=Path, metavar='ACTUAL_AUDIT_JSON')
    parser.add_argument('--audit-source', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.declare:
        value = dict(audit_path=str(args.declare))
        if args.audit_source: value['audit_source_path'] = str(args.audit_source)
        declare_snapshot(value, out=args.out)
    else: parser.print_help()

if __name__ == '__main__': main()
