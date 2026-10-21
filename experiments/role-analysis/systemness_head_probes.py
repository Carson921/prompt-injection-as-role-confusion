"""Development-only joint top-eight-head readouts; no test is read here.

Default: help. --preflight runs synthetic CPU fixtures under /tmp only. --train
explicitly freezes a separate search after head extraction and scanner validation
have finished. This experiment never replaces the current primary readout.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
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
import systemness_head_scan as scanner
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-head-probes'
CACHE = data.CACHE / 'head-probes'
FEATURE_OUT = data.runtime.OUT / 'systemness-head-features'
BANK = Path('/dev/shm/role-analysis-top8-heads')
DIMENSIONS, HEADS, HEAD_DIM = 512, 8, 64
FIT_ROWS, VALID_ROWS, TOTAL_ROWS = 578090, 173032, 755282
BATCH_SIZE, STEPS = 4096, 142
CONFIGS = {
    'pooled-lda': dict(kind='pooled-lda', hidden=0, shrinkage=.1, standardized=False),
    'joint-linear': dict(kind='linear', hidden=0, C=.005, standardized=True),
    'joint-mlp64': dict(kind='mlp', hidden=64, standardized=True),
    'joint-mlp128': dict(kind='mlp', hidden=128, standardized=True),
}
DEPENDENCIES = ['systemness_head_probes.py', 'systemness_head_extract.py',
    'systemness_head_scan.py', 'systemness_long_probes.py', 'systemness_long_data.py',
    'systemness_probes.py', 'systemness_data.py']
LIMITATION = ('All four candidates fit the identical mixed densefit content distribution. '
    'The pooled LDA uses raw heads, preserving scanner shrinkage semantics; joint linear/MLP '
    'use a new densefit-only scaler. Head selection, feature geometry, classifier family and '
    'scaling differ from earlier readouts, so this is not an isolated causal ablation. '
    'MLPs match 578090 instances/142 updates per epoch and the maximum budget; independent '
    'early stopping can change best and actual total updates. Validation is reused development '
    'data. Any candidate still needs a new independent confirmation before replacing primary.')
EXPECTED = dict(total=TOTAL_ROWS, base=566482, fit=FIT_ROWS, valid=VALID_ROWS,
    old_fit=413290, long_fit=168960, long_valid=19840, long_fit_documents=132,
    long_valid_documents=31, removed=4160, removed_documents=26, fit_documents=1200)
require = long_data.require


def read(path):
    return json.loads(Path(path).read_text())


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_hash(path, expected, name):
    require(Path(path).is_file(), f'Missing frozen {name}: {path}.')
    require(data.digest(path) == expected, f'Frozen {name} SHA changed: {path}.')


def atomic_npz(path, **arrays):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def validate_probabilities(value, count=None):
    require(value.ndim == 2 and value.shape[1] == 5 and (count is None or len(value) == count),
        'Probability row/class dimensions changed.')
    require(np.isfinite(value).all() and (value >= 0).all() and (value <= 1.00001).all()
        and np.allclose(value.sum(1), 1., atol=2e-5, rtol=0), 'Invalid five-class probabilities.')


def partitions(rows, expected=None):
    expected = EXPECTED if expected is None else expected
    require(len(rows) == expected['total'] and np.array_equal(rows.cache_ix, np.arange(len(rows)))
        and rows.sample_ix.is_unique, 'Development cache row order/identity changed.')
    require(set(rows.split) == {'fit', 'validation'} and set(rows.wrapper) == {'native'},
        'Only native fit/validation rows are permitted; test rows are forbidden.')
    require(set(rows.role) == set(data.ROLES), 'Missing or invalid formal role.')
    base = rows[rows.sample_ix < long_data.SAMPLE_START]
    full = rows[rows.sample_ix >= long_data.SAMPLE_START]
    require(len(base) == expected['base'] and np.array_equal(base.cache_ix, np.arange(len(base)))
        and full.profile.eq('long-full').all() and full.source_set.eq('long-full').all(),
        'Original base prefix or appended long-full namespace changed.')
    raw_fit = rows[rows.split == 'fit']
    full_fit = full[full.split == 'fit']; full_valid = full[full.split == 'validation']
    require(len(full_fit) == expected['long_fit'] and len(full_valid) == expected['long_valid']
        and full_fit.document_id.nunique() == expected['long_fit_documents']
        and full_valid.document_id.nunique() == expected['long_valid_documents'], 'Long-full membership changed.')
    require(len(raw_fit[raw_fit.sample_ix < long_data.SAMPLE_START]) == expected['old_fit'],
        'Original fit row count changed.')
    fit = long_probes.fit_rows(rows, True)
    removed = raw_fit[~raw_fit.cache_ix.isin(fit.cache_ix)]
    require(len(removed) == expected['removed'] and removed.document_id.nunique() == expected['removed_documents']
        and removed.profile.eq('long-single').all() and removed.document_id.isin(full_fit.document_id).all(),
        'Densefit must replace exactly the old sparse long-single rows for overlapping fit documents.')
    valid = rows[rows.split == 'validation'].copy().reset_index(drop=True)
    require(len(fit) == expected['fit'] and len(valid) == expected['valid']
        and fit.document_id.nunique() == expected['fit_documents'], 'Densefit/validation counts changed.')
    require(set(valid.profile) == set(long_probes.PROFILES)
        and not set(fit.group_id) & set(valid.group_id)
        and not set(fit.cache_ix) & set(valid.cache_ix)
        and np.all(np.diff(fit.cache_ix.to_numpy()) > 0), 'Split leakage or fit order change.')
    info = dict(fit_rows=len(fit), validation_rows=len(valid), removed_sparse_rows=len(removed),
        removed_sparse_documents=int(removed.document_id.nunique()),
        fit_cache_ix_sha256=array_digest(fit.cache_ix.to_numpy()),
        fit_sample_ix_sha256=array_digest(fit.sample_ix.to_numpy()),
        fit_labels_sha256=array_digest(probes.labels(fit)),
        validation_cache_ix_sha256=array_digest(valid.cache_ix.to_numpy()),
        validation_sample_ix_sha256=array_digest(valid.sample_ix.to_numpy()),
        validation_labels_sha256=array_digest(probes.labels(valid)))
    return fit, valid, info


def make_net(kind, dimensions=DIMENSIONS, hidden=0, heads=HEADS, head_dim=HEAD_DIM):
    """Own dimension-aware portable consumer; never uses probes.make_net."""
    import torch
    from torch import nn
    require(kind in {'linear', 'mlp', 'pooled-lda'} and dimensions > 0, 'Unknown portable model.')
    if kind == 'pooled-lda':
        require(dimensions == heads * head_dim, 'Pooled head geometry changed.')
        class Pooled(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('weight', torch.zeros(heads, 5, head_dim, dtype=torch.float64))
                self.register_buffer('bias', torch.zeros(heads, 5, dtype=torch.float64))

            def forward(self, value):
                logits = torch.einsum('nhd,hkd->nhk', value.double().reshape(-1, heads, head_dim), self.weight) + self.bias
                return torch.logsumexp(torch.log_softmax(logits, -1), dim=1) - np.log(heads)
        return Pooled()
    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dimensions, hidden), nn.GELU(), nn.Dropout(.1)) if hidden else nn.Identity()
            self.head = nn.Linear(hidden or dimensions, 5)

        def raw(self, value):
            return self.head(self.encoder(value))

        def forward(self, value):
            return torch.log_softmax(self.raw(value), dim=1)
    return Joint()


def save_model(path, net, mean, scale, metadata):
    mean = np.asarray(mean, dtype=np.float32); scale = np.asarray(scale, dtype=np.float32)
    require(mean.shape == scale.shape == (metadata['dimensions'],) and np.isfinite(mean).all()
        and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid portable scaler.')
    arrays = {'weight::' + name: value.detach().cpu().numpy() for name, value in net.state_dict().items()}
    atomic_npz(path, mean=mean, scale=scale, metadata=np.array(json.dumps(metadata)), **arrays)


def load_model(path, device='cpu'):
    import torch
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        require(metadata['roles'] == data.ROLES and metadata['dimensions'] == metadata['heads'] * metadata['head_dim'],
            'Portable role order or head dimensions changed.')
        net = make_net(metadata['kind'], metadata['dimensions'], metadata['hidden'], metadata['heads'], metadata['head_dim'])
        net.load_state_dict({key[8:]: torch.from_numpy(archive[key].copy()) for key in archive.files if key.startswith('weight::')})
        mean = torch.from_numpy(archive['mean'].copy()).to(device)
        scale = torch.from_numpy(archive['scale'].copy()).to(device)
        require(mean.shape == scale.shape == (metadata['dimensions'],) and torch.isfinite(mean).all()
            and torch.isfinite(scale).all() and (scale > 0).all(), 'Corrupt portable scaler.')
    return net.eval().to(device), mean, scale, metadata


def predict_raw(path, bank, indices, device='cpu'):
    import torch
    net, mean, scale, _ = load_model(path, device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            raw = torch.from_numpy(np.asarray(bank[indices[start:start + BATCH_SIZE]], dtype=np.float32)).to(device)
            chunks.append(net((raw - mean) / scale).exp().cpu().numpy().astype(np.float32))
    result = np.concatenate(chunks); validate_probabilities(result, len(indices))
    return result


def fit_scaler(bank, indices, chunk=4096):
    """Population mean/variance from densefit only, accumulated in float64."""
    require(len(indices) > 1 and len(np.unique(indices)) == len(indices), 'Invalid scaler fit identities.')
    count = 0; mean = np.zeros(bank.shape[1], np.float64); m2 = np.zeros_like(mean)
    for start in range(0, len(indices), chunk):
        block = np.asarray(bank[indices[start:start + chunk]], dtype=np.float64)
        require(np.isfinite(block).all(), 'Nonfinite fit feature.')
        other_mean = block.mean(0); delta = other_mean - mean; combined = count + len(block)
        m2 += ((block - other_mean) ** 2).sum(0) + delta ** 2 * count * len(block) / combined
        mean += delta * len(block) / combined; count = combined
    scale = np.maximum(np.sqrt(np.maximum(m2 / count, 0)), 1e-6)
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_pooled_lda(bank, indices, labels, heads=HEADS, head_dim=HEAD_DIM, chunk=2048):
    require(len(indices) == len(labels) and len(np.unique(indices)) == len(indices), 'LDA row identities changed.')
    moments = scanner.HeadMoments(heads=heads, dimensions=head_dim)
    for start in range(0, len(indices), chunk):
        values = np.asarray(bank[indices[start:start + chunk]], dtype=np.float64).reshape(-1, heads, head_dim)
        moments.update(values, labels[start:start + len(values)])
    weight, bias, diagnostics = moments.fit(return_diagnostics=True)
    return weight, bias, diagnostics


def dataset():
    rows = pd.read_parquet(BANK / 'rows.parquet')
    fit, valid, info = partitions(rows)
    require((BANK / 'features.f16').stat().st_size == TOTAL_ROWS * DIMENSIONS * 2, 'Feature bank missing/truncated.')
    bank = np.memmap(BANK / 'features.f16', mode='r', dtype=np.float16, shape=(len(rows), DIMENSIONS))
    return rows, bank, fit, valid, info


def source_signature():
    extract = importlib.import_module('systemness_head_extract')
    require(extract.OUT == FEATURE_OUT and extract.CACHE == BANK, 'Head extractor public paths changed.')
    complete = extract.verified_extraction()
    require(complete['rows'] == TOTAL_ROWS and complete['dimensions'] == DIMENSIONS
        and complete['feature_shape'] == [TOTAL_ROWS, DIMENSIONS] and complete['head_dim'] == HEAD_DIM,
        'Head extraction shape changed.')
    require(complete['all_finite'] and complete['all_rows_written'] and complete['tests_read'] is False,
        'Head extraction is incomplete or read a test.')
    require(complete['rows_sha256'] == data.digest(BANK / 'rows.parquet')
        == data.digest(long_probes.BANK / 'rows.parquet'), 'Head bank must retain the byte-identical base row order.')
    require(complete['features_sha256'] == data.digest(BANK / 'features.f16'), 'Extracted head feature hash changed.')
    require(complete['source_sha256'] == data.digest(data.HERE / 'systemness_head_extract.py'), 'Head extraction source changed.')
    scan_plan = read(scanner.OUT / 'plan.json'); scanner.verify_plan(scan_plan)
    scan_done = read(scanner.OUT / 'scan-complete.json'); selection = read(scanner.OUT / 'head-selection.json')
    scan_validation = read(scanner.OUT / 'validation-complete.json')
    selected = complete['selected_head_ids']
    require(len(selected) == len(set(selected)) == HEADS and all(0 <= h < scanner.TOTAL_HEADS for h in selected),
        'Selected eight heads are invalid.')
    require(selected == selection['selected_head_ids']
        and complete['ordered_layers_heads'] == [[int(h // scanner.HEADS), int(h % scanner.HEADS)] for h in selected],
        'Selected feature head order changed.')
    require(scan_done['plan_sha256'] == scan_validation['plan_sha256'] == data.digest(scanner.OUT / 'plan.json')
        and scan_done['head_selection_sha256'] == data.digest(scanner.OUT / 'head-selection.json')
        and scan_done['model_sha256'] == scan_validation['model_sha256'] == data.digest(scanner.OUT / 'models/top8-lda.npz')
        and scan_validation['scan_complete_sha256'] == data.digest(scanner.OUT / 'scan-complete.json'),
        'Scanner completion chain changed.')
    require(scan_validation['probabilities_sha256'] == data.digest(scanner.CACHE / 'validation-probabilities.npz')
        and scan_validation['metrics_sha256'] == data.digest(scanner.OUT / 'validation-metrics.json'),
        'Scanner validation artifacts changed.')
    dense_plan = read(long_data.OUT / 'search-plan.json')
    dense = long_probes.candidate_record('dense-mlp128', dense_plan)
    _, _, _, valid, partition = dataset()
    with np.load(long_probes.BANK / 'dense-mlp128-validation.npz', allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'], valid.sample_ix), 'Dense control validation identities changed.')
        p = archive['p'].copy()
    validate_probabilities(p, len(valid))
    baseline = long_probes.validation_metrics(valid, p)
    require(baseline == dense['validation'], 'Dense control metrics do not match its saved probabilities.')
    return dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        extraction_sha256=data.digest(FEATURE_OUT / 'extraction-complete.json'),
        extraction_signature=complete['signature'], rows_sha256=complete['rows_sha256'],
        features_sha256=complete['features_sha256'], selected_head_ids=selected,
        ordered_layers_heads=complete['ordered_layers_heads'], partitions=partition,
        scanner_plan_sha256=data.digest(scanner.OUT / 'plan.json'),
        scanner_complete_sha256=data.digest(scanner.OUT / 'scan-complete.json'),
        scanner_validation_complete_sha256=data.digest(scanner.OUT / 'validation-complete.json'),
        head_selection_sha256=data.digest(scanner.OUT / 'head-selection.json'),
        scanner_model_sha256=data.digest(scanner.OUT / 'models/top8-lda.npz'),
        dense_search_plan_sha256=data.digest(long_data.OUT / 'search-plan.json'),
        dense_candidate_sha256=data.digest(long_data.OUT / 'candidates/dense-mlp128.json'),
        dense_model_sha256=dense['model_sha256'],
        dense_probabilities_sha256=dense['validation_probabilities_sha256'], baseline_validation=baseline)


def protocol():
    return dict(configurations=CONFIGS, dimensions=DIMENSIONS, heads=HEADS, head_dim=HEAD_DIM,
        roles=data.ROLES, validation_profiles=long_probes.PROFILES, fit_rows=FIT_ROWS, validation_rows=VALID_ROWS,
        neural=dict(seed=data.SEED, optimizer='AdamW', learning_rate=.001, weight_decay=.001,
            dropout=.1, batch_size=BATCH_SIZE, steps_per_epoch=STEPS, last_batch_instances=554,
            max_epochs=45, min_epochs=10, patience=8, gradient_clip=5., loss='Five-class mean CE'),
        pooled_lda='Per-head raw shrinkLDA(.1), empirical five-class priors, uniform mean of8 probabilities; all578090 densefit rows.',
        standardization='Linear/MLP only: population mean/std on the identical578090 densefit rows, float64 streaming moments, scale floor1e-6.',
        threshold='The unchanged long_probes.validation_metrics helper: one threshold maximizing5-profile mean systemF1, each profile FPR<=.05.',
        selection=dict(rank='Maximum5-profile mean system F1 among candidates passing all guards.',
            original_four_maximum_system_f1_loss=.02, original_four_maximum_five_way_accuracy_loss=.01,
        long_full_maximum_system_f1_loss=0., tail_maximum_system_f1_loss=0., comparison_tolerance=1e-12,
            promotion_requires_positive_mean_f1_gain=True, baseline='dense-mlp128'),
        tests_read=False, fresh_confirmation_required=True, primary_changed=False, limitation=LIMITATION)


def verify_plan(plan):
    require(plan['protocol'] == protocol() and plan['signature'] == source_signature(), 'Frozen head-probe search signature changed.')
    scaler_path = OUT / 'standardization.npz'
    require_hash(scaler_path, plan['standardization_sha256'], 'densefit scaler')
    with np.load(scaler_path, allow_pickle=False) as archive:
        require(archive['mean'].shape == archive['scale'].shape == (DIMENSIONS,)
            and np.isfinite(archive['mean']).all() and np.isfinite(archive['scale']).all()
            and (archive['scale'] > 0).all()
            and str(archive['fit_cache_ix_sha256']) == plan['signature']['partitions']['fit_cache_ix_sha256'],
            'Scaler shape/finiteness/fit identity changed.')


def candidate_record(key, plan):
    verify_plan(plan)
    record = read(OUT / 'candidates' / f'{key}.json')
    require(record['candidate'] == key and record['configuration'] == CONFIGS[key]
        and record['source_sha256'] == plan['signature']['source_sha256']
        and record['search_plan_sha256'] == data.digest(OUT / 'search-plan.json')
        and record['standardization_sha256'] == plan['standardization_sha256']
        and record['signature'] == plan['signature'] and record['tests_read'] is False,
        'Candidate configuration or provenance changed.')
    path = OUT / 'models' / f'{key}.npz'; probability_path = CACHE / f'{key}-validation.npz'
    require_hash(path, record['model_sha256'], 'candidate model')
    require_hash(probability_path, record['validation_probabilities_sha256'], 'candidate probabilities')
    require(record['validation_probabilities_path'] == str(probability_path.resolve())
        and record['portable_probability_max_difference'] < 5e-4, 'Candidate portable artifact changed.')
    _, _, _, valid, _ = dataset()
    with np.load(probability_path, allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'], valid.sample_ix), 'Candidate validation identities changed.')
        p = archive['p'].copy()
    validate_probabilities(p, len(valid))
    require(long_probes.validation_metrics(valid, p) == record['validation'], 'Candidate metrics do not match probabilities.')
    _, _, _, metadata = load_model(path)
    require(metadata['signature'] == plan['signature'] and metadata['configuration'] == CONFIGS[key]
        and metadata['standardization_sha256'] == plan['standardization_sha256'], 'Portable model provenance changed.')
    return record


def verified_completion():
    """Read-only downstream entry point; bind completion through all four models."""
    plan = read(OUT / 'search-plan.json'); verify_plan(plan)
    complete = read(OUT / 'complete.json')
    require(complete['signature'] == plan['signature'] and complete['source_sha256'] == data.digest(__file__)
        and complete['tests_read'] is False and complete['primary_changed'] is False
        and complete['fresh_confirmation_required'] is True, 'Completed search provenance changed.')
    require_hash(OUT / 'search-plan.json', complete['search_plan_sha256'], 'search plan')
    require_hash(OUT / 'selection.json', complete['selection_sha256'], 'selection')
    require_hash(OUT / 'standardization.npz', complete['standardization_sha256'], 'scaler')
    require_hash(OUT / 'validation-metrics.csv', complete['validation_metrics_sha256'], 'validation table')
    require(set(complete['candidate_sha256']) == set(CONFIGS), 'Completed search candidate coverage changed.')
    selection = read(OUT / 'selection.json')
    require(selection['candidates'] == complete['candidate_sha256'] and selection['primary_changed'] is False
        and selection['test_used_for_selection'] is False and selection['confirmation_required'] is True
        and selection['source_sha256'] == plan['signature']['source_sha256']
        and selection['search_plan_sha256'] == complete['search_plan_sha256'],
        'Selection candidate coverage or confirmation requirement changed.')
    records = []
    for key, sha in complete['candidate_sha256'].items():
        require_hash(OUT / 'candidates' / f'{key}.json', sha, f'{key} completion')
        record = candidate_record(key, plan)
        records.append(record)
        if key == selection['candidate']:
            require(selection['model_sha256'] == record['model_sha256']
                and selection['validation_probabilities_sha256'] == record['validation_probabilities_sha256']
                and selection['validation'] == record['validation'], 'Selected portable model/probabilities changed.')
    chosen, report_best, guards, promoted, reasons = choose_candidate(records, plan['signature']['baseline_validation'])
    require(selection['candidate'] == chosen['candidate'] and selection['report_best_candidate'] == report_best['candidate']
        and selection['candidate_guards'] == guards and selection['promoted'] == promoted
        and selection['promotion_rejection_reasons'] == reasons
        and selection['system_threshold'] == chosen['validation']['system_threshold'], 'Completed selection rule changed.')
    return complete


def promotion_guard(result, baseline):
    original = long_probes.ORIGINAL_PROFILES
    losses = {p: baseline['profiles'][p]['system_f1'] - result['profiles'][p]['system_f1'] for p in original}
    accuracy_losses = {p: baseline['profiles'][p]['accuracy'] - result['profiles'][p]['accuracy'] for p in original}
    full_gain = result['profiles']['long-full']['system_f1'] - baseline['profiles']['long-full']['system_f1']
    tail_gain = result['long_full_tail']['system_f1'] - baseline['long_full_tail']['system_f1']
    finite = all(np.isfinite(result['profiles'][p][k]) for p in long_probes.PROFILES
        for k in ['system_f1', 'accuracy', 'system_false_positive_rate']) and np.isfinite(result['long_full_tail']['system_f1'])
    finite = finite and np.isfinite(result['mean_system_f1'])
    tolerance = 1e-12
    eligible = finite and max(losses.values()) <= .02 + tolerance and max(accuracy_losses.values()) <= .01 + tolerance
    eligible = eligible and full_gain >= -tolerance and tail_gain >= -tolerance
    eligible = eligible and all(result['profiles'][p]['system_false_positive_rate'] <= .05 + tolerance for p in long_probes.PROFILES)
    reasons = []
    if not finite: reasons.append('nonfinite_validation_metric')
    if max(losses.values()) > .02 + tolerance: reasons.append('original_profile_system_f1_loss')
    if max(accuracy_losses.values()) > .01 + tolerance: reasons.append('original_profile_five_way_accuracy_loss')
    if full_gain < -tolerance: reasons.append('long_full_system_f1_regression')
    if tail_gain < -tolerance: reasons.append('long_full_tail_system_f1_regression')
    if any(result['profiles'][p]['system_false_positive_rate'] > .05 + tolerance for p in long_probes.PROFILES):
        reasons.append('profile_system_fpr_above_limit')
    return dict(eligible=bool(eligible), mean_f1_gain=result['mean_system_f1'] - baseline['mean_system_f1'],
        original_profile_f1_losses=losses, original_profile_accuracy_losses=accuracy_losses,
        long_full_f1_gain=full_gain, tail_f1_gain=tail_gain, rejection_reasons=reasons)


def choose_candidate(records, baseline):
    require(bool(records), 'No completed head-probe candidate.')
    guards = {r['candidate']: promotion_guard(r['validation'], baseline) for r in records}
    eligible = [r for r in records if guards[r['candidate']]['eligible']]
    best_report = max(records, key=lambda r: r['validation']['mean_system_f1'])
    selected = max(eligible, key=lambda r: r['validation']['mean_system_f1']) if eligible else best_report
    guard = guards[selected['candidate']]
    promoted = bool(eligible and guard['mean_f1_gain'] > 0)
    reasons = ([] if promoted else (['no_guard_eligible_candidate'] if not eligible else [])
        + guard['rejection_reasons'] + (['mean_system_f1_not_improved'] if guard['mean_f1_gain'] <= 0 else []))
    return selected, best_report, guards, promoted, reasons


def worker(key, gpu):
    require(os.environ.get('ROLE_HEAD_PROBE_SUPERVISOR_PID') == str(os.getppid()),
        'Workers must be launched by this experiment supervisor, not started independently.')
    require(not (OUT / 'selection.json').exists() and not (OUT / 'complete.json').exists()
        and not (OUT / 'candidates' / f'{key}.json').exists(), 'A frozen search or completed candidate cannot be overwritten.')
    no_competing_jobs(allow_own=True)
    plan = read(OUT / 'search-plan.json'); verify_plan(plan)
    import torch
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    require(gpu in range(4) and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
        and torch.cuda.device_count() == 1 and 'A40' in torch.cuda.get_device_name(0), 'Worker needs its one assigned A40.')
    _, bank, fit, valid, partition = dataset()
    require(partition == plan['signature']['partitions'], 'Worker fit/validation order changed.')
    config = CONFIGS[key]; path = OUT / 'models' / f'{key}.npz'; pp = CACHE / f'{key}-validation.npz'
    with np.load(OUT / 'standardization.npz', allow_pickle=False) as archive:
        fitted_mean = archive['mean'].copy(); fitted_scale = archive['scale'].copy()
    mean = fitted_mean if config['standardized'] else np.zeros(DIMENSIONS, np.float32)
    scale = fitted_scale if config['standardized'] else np.ones(DIMENSIONS, np.float32)
    details = dict(candidate=key, configuration=config, kind=config['kind'], hidden=config['hidden'],
        dimensions=DIMENSIONS, heads=HEADS, head_dim=HEAD_DIM, roles=data.ROLES,
        standardization_sha256=plan['standardization_sha256'], source_sha256=data.digest(__file__),
        search_plan_sha256=data.digest(OUT / 'search-plan.json'), signature=plan['signature'],
        seed=data.SEED, fit_rows=len(fit), fit_documents=int(fit.document_id.nunique()), tests_read=False,
        selected_head_ids=plan['signature']['selected_head_ids'],
        inference='Pointwise selected8 pre-o_proj head vectors; no position/role/pair inputs.', limitation=LIMITATION)
    started = time.time(); labels = probes.labels(fit)
    if config['kind'] == 'pooled-lda':
        weight, bias, diagnostics = fit_pooled_lda(bank, fit.cache_ix.to_numpy(), labels)
        net = make_net('pooled-lda')
        with torch.no_grad():
            net.weight.copy_(torch.from_numpy(weight)); net.bias.copy_(torch.from_numpy(bias))
        details['lda_diagnostics'] = diagnostics
        save_model(path, net, mean, scale, details)
        p = predict_raw(path, bank, valid.cache_ix.to_numpy(), device='cuda:0')
        history = None
    else:
        free, _ = torch.cuda.mem_get_info(0)
        require(free > (len(fit) + len(valid)) * DIMENSIONS * 4 + 2 * 2**30, 'Insufficient assigned GPU memory.')
        def load(indices):
            value = torch.empty((len(indices), DIMENSIONS), dtype=torch.float32, device='cuda', requires_grad=False)
            for start in range(0, len(indices), BATCH_SIZE):
                raw = np.asarray(bank[indices[start:start + BATCH_SIZE]], dtype=np.float32)
                z = (raw - mean) / scale; require(np.isfinite(z).all(), 'Invalid standardized features.')
                value[start:start + len(z)] = torch.from_numpy(z).cuda()
            return value
        x = load(fit.cache_ix.to_numpy()); xv = load(valid.cache_ix.to_numpy())
        if config['kind'] == 'linear':
            import cupy as cp
            from cuml.linear_model import LogisticRegression
            clf = LogisticRegression(C=config['C'], max_iter=5000, linesearch_max_iter=100, tol=1e-4, output_type='cupy')
            clf.fit(cp.from_dlpack(x), cp.asarray(labels))
            require(np.array_equal(cp.asnumpy(clf.classes_), np.arange(5)), 'Linear classifier class order changed.')
            p = cp.asnumpy(clf.predict_proba(cp.from_dlpack(xv))).astype(np.float32)
            net = make_net('linear')
            with torch.no_grad():
                net.head.weight.copy_(torch.from_numpy(cp.asnumpy(clf.coef_)))
                net.head.bias.copy_(torch.from_numpy(cp.asnumpy(clf.intercept_).reshape(-1)))
            details['iterations'] = cp.asnumpy(cp.asarray(clf.n_iter_)).tolist()
            save_model(path, net, mean, scale, details); history = None
        else:
            y = torch.from_numpy(labels).cuda(); torch.manual_seed(data.SEED)
            net = make_net('mlp', hidden=config['hidden']).cuda()
            optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
            generator = torch.Generator(device='cuda:0').manual_seed(data.SEED)
            best = -1.; best_epoch = -1; history = []
            for epoch in range(1, 46):
                net.train(); order = long_probes.epoch_order(len(x), generator, 'cuda'); loss_sum = 0.; steps = 0
                require(len(order) == FIT_ROWS, 'Epoch instances changed.')
                for start in range(0, len(order), BATCH_SIZE):
                    ix = order[start:start + BATCH_SIZE]
                    loss = torch.nn.functional.cross_entropy(net.raw(x[ix].contiguous()), y[ix])
                    require(torch.isfinite(loss), 'Nonfinite CE loss.')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
                    loss_sum += float(loss.detach()) * len(ix); steps += 1
                require(steps == STEPS, 'Epoch optimizer update count changed.')
                p = probes.predict_standardized(net, xv); validate_probabilities(p, len(valid))
                result = long_probes.validation_metrics(valid, p); score = result['mean_system_f1']
                history.append(dict(epoch=epoch, loss=loss_sum / len(order), instances=len(order), steps=steps,
                    cumulative_steps=epoch * STEPS, mean_system_f1=score,
                    long_full_f1=result['profiles']['long-full']['system_f1'], tail_f1=result['long_full_tail']['system_f1']))
                if score > best + 1e-5:
                    best = score; best_epoch = epoch
                    save_model(path, net, mean, scale, details | dict(best_epoch=epoch, best_steps=epoch * STEPS))
                    atomic_npz(pp, p=p, sample_ix=valid.sample_ix.to_numpy())
                print(json.dumps(dict(candidate=key, **history[-1], best_epoch=best_epoch)), flush=True)
                if epoch >= 10 and epoch - best_epoch >= 8: break
            with np.load(pp, allow_pickle=False) as archive:
                require(np.array_equal(archive['sample_ix'], valid.sample_ix), 'Best checkpoint validation order changed.')
                p = archive['p'].copy()
            details.update(best_epoch=best_epoch, best_steps=best_epoch * STEPS, epochs=len(history), history=history,
                actual_steps=sum(r['steps'] for r in history), actual_instances=sum(r['instances'] for r in history),
                instances_per_epoch=FIT_ROWS, steps_per_epoch=STEPS, last_batch_instances=554)
    validate_probabilities(p, len(valid)); atomic_npz(pp, p=p, sample_ix=valid.sample_ix.to_numpy())
    ids = np.linspace(0, len(valid) - 1, 256, dtype=int)
    actual = predict_raw(path, bank, valid.iloc[ids].cache_ix.to_numpy())
    difference = float(np.abs(actual - p[ids]).max()); require(difference < 5e-4, 'Portable CPU consumer differs from worker.')
    verify_plan(plan)
    data.atomic_json(OUT / 'candidates' / f'{key}.json', details | dict(validation=long_probes.validation_metrics(valid, p),
        seconds=time.time() - started, model_sha256=data.digest(path), validation_probabilities_path=str(pp.resolve()),
        validation_probabilities_sha256=data.digest(pp), portable_probability_max_difference=difference,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))


def no_competing_jobs(allow_own=False):
    names = {'systemness_head_scan.py', 'systemness_head_extract.py', 'systemness_long_extract.py',
        'systemness_recover_local_bank.py', 'systemness_long_probes.py', 'systemness_paired_probes.py',
        'systemness_long_confirm.py', 'systemness_attention_confirm.py', 'systemness_natural_recheck.py'}
    if not allow_own: names.add(Path(__file__).name)
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg): active.append(int(entry.name))
    require(not active, f'Other experiment jobs remain active: {active}.')


@contextmanager
def run_lock():
    import fcntl
    with (OUT / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another head-probe supervisor holds this output directory.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_workers(pending, plan):
    require(len(pending) <= 4, 'At most four independent workers are permitted.')
    workers = []
    try:
        for gpu, key in enumerate(pending):
            log = (OUT / 'logs' / f'{key}.log').open('a')
            try:
                proc = subprocess.Popen([sys.executable, '-u', __file__, '--train', '--worker', key, '--gpu', str(gpu)],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), ROLE_HEAD_PROBE_SUPERVISOR_PID=str(os.getpid())),
                    stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close(); raise
            workers.append((key, proc, log, gpu))
        data.atomic_json(OUT / 'status.json', dict(stage='training', workers=[dict(candidate=k, pid=p.pid, gpu=g) for k, p, _, g in workers]))
        while workers:
            codes = [(key, proc.poll()) for key, proc, _, _ in workers]
            failed = [(key, code) for key, code in codes if code not in [None, 0]]
            require(not failed, f'Head-probe worker failed: {failed}; inspect logs.')
            if all(code == 0 for _, code in codes): break
            time.sleep(5)
        for key, proc, _, _ in workers:
            require(proc.wait() == 0, f'{key} failed.'); candidate_record(key, plan)
    finally:
        errors = []
        for key, proc, _, _ in workers:
            try:
                if proc.poll() is None: proc.terminate()
            except BaseException as error: errors.append((key, repr(error)))
        for key, proc, log, _ in workers:
            try:
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=10)
            except BaseException as error: errors.append((key, repr(error)))
            finally: log.close()
        require(not errors, f'Could not reap workers: {errors}.')


def select(plan):
    records = [candidate_record(key, plan) for key in CONFIGS]
    baseline = plan['signature']['baseline_validation']
    selected, best_report, guards, promoted, reasons = choose_candidate(records, baseline)
    guard = guards[selected['candidate']]
    model_path = OUT / 'models' / f'{selected["candidate"]}.npz'
    frozen = dict(candidate=selected['candidate'], candidate_is_guard_eligible=guard['eligible'],
        model_path=str(model_path), model_sha256=selected['model_sha256'],
        validation_probabilities_path=selected['validation_probabilities_path'],
        validation_probabilities_sha256=selected['validation_probabilities_sha256'],
        standardization_sha256=plan['standardization_sha256'], validation=selected['validation'],
        system_threshold=selected['validation']['system_threshold'], promoted=promoted,
        baseline_candidate='dense-mlp128', baseline_validation=baseline, candidate_guards=guards,
        selected_guard=guard, report_best_candidate=best_report['candidate'],
        promotion_rejection_reasons=reasons,
        source_sha256=data.digest(__file__), search_plan_sha256=data.digest(OUT / 'search-plan.json'),
        candidates={r['candidate']: data.digest(OUT / 'candidates' / f'{r["candidate"]}.json') for r in records},
        test_used_for_selection=False, confirmation_required=True, primary_changed=False,
        limitation=LIMITATION, selected_unix_time=time.time())
    flat = []
    for record in records:
        for profile, values in dict(record['validation']['profiles'], **{'long-full-tail': record['validation']['long_full_tail']}).items():
            flat.append(dict(candidate=record['candidate'], profile=profile,
                **{key: value for key, value in values.items() if not isinstance(value, (dict, list))}))
    pd.DataFrame(flat).to_csv(OUT / 'validation-metrics.csv', index=False)
    verify_plan(plan)
    data.atomic_json(OUT / 'selection.json', frozen)
    data.atomic_json(OUT / 'complete.json', dict(search_plan_sha256=data.digest(OUT / 'search-plan.json'),
        selection_sha256=data.digest(OUT / 'selection.json'), standardization_sha256=plan['standardization_sha256'],
        validation_metrics_sha256=data.digest(OUT / 'validation-metrics.csv'), candidate_sha256=frozen['candidates'],
        source_sha256=data.digest(__file__), signature=plan['signature'], tests_read=False,
        primary_changed=False, fresh_confirmation_required=True))
    data.atomic_json(OUT / 'status.json', dict(stage='validation_complete', candidate=selected['candidate'], promoted=promoted))


def train():
    no_competing_jobs()
    signature = source_signature()  # Missing upstream completion fails before creating output.
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    with run_lock():
        require(not (OUT / 'selection.json').exists() and not (OUT / 'complete.json').exists(), 'Head-probe search already frozen.')
        for folder in ['models', 'candidates', 'logs']: (OUT / folder).mkdir(exist_ok=True)
        (OUT / '.gitignore').write_text('/logs/\n/.run.lock\n')
        plan_path = OUT / 'search-plan.json'; scaler_path = OUT / 'standardization.npz'
        if plan_path.exists():
            plan = read(plan_path); verify_plan(plan)
            require(plan['signature'] == signature, 'Resume source signature changed.')
        else:
            require(not scaler_path.exists() and not any((OUT / 'candidates').iterdir())
                and not any((OUT / 'models').iterdir()), 'Partial undeclared search artifacts exist; refusing silent reuse.')
            _, bank, fit, _, info = dataset()
            mean, scale = fit_scaler(bank, fit.cache_ix.to_numpy())
            atomic_npz(scaler_path, mean=mean, scale=scale,
                fit_cache_ix_sha256=np.array(info['fit_cache_ix_sha256']),
                fit_sample_ix_sha256=np.array(info['fit_sample_ix_sha256']), fit_rows=np.array(len(fit)))
            plan = dict(protocol=protocol(), signature=signature, standardization_sha256=data.digest(scaler_path))
            data.atomic_json(plan_path, plan); verify_plan(plan)
        pending = []
        for key in CONFIGS:
            if (OUT / 'candidates' / f'{key}.json').exists(): candidate_record(key, plan)
            else:
                # An interrupted candidate is explicitly restarted from epoch1. Only a
                # complete hash-bound candidate record can authorize reuse of a model.
                if (OUT / 'models' / f'{key}.npz').exists() or (CACHE / f'{key}-validation.npz').exists():
                    print(json.dumps(dict(candidate=key, restart='Incomplete candidate; rerun from initialization, never resume an old probability cursor.')), flush=True)
                pending.append(key)
        run_workers(pending, plan); select(plan)


def preflight():
    """Synthetic CPU fixtures only: no formal metadata, activation bank or test read."""
    import copy
    import torch
    torch.set_num_threads(2); torch.manual_seed(data.SEED)
    folder = Path(tempfile.mkdtemp(prefix='systemness-head-probes-preflight-'))
    checks = []; rng = np.random.RandomState(data.SEED)
    def rejects(action, name):
        try: action()
        except (RuntimeError, ValueError, KeyError, AssertionError): checks.append(name)
        else: raise RuntimeError(f'Fault fixture accepted: {name}')
    # Build the real4160 sparse-row replacement geometry, with small dense bodies.
    entries = []
    def append(document, profile, split, count, augmented=False):
        for j in range(count):
            i = len(entries)
            entries.append(dict(cache_ix=i, sample_ix=long_data.SAMPLE_START + i if augmented else i,
                document_id=document, group_id=f'c4-{document}', profile=profile, split=split,
                wrapper='native', source_set='long-full' if augmented else 'main',
                role=data.ROLES[j % 5], prompt_ix=i, message_ix=0, token_ix=j))
    for document in range(32): append(document, 'long-single', 'fit', 160)
    for i, profile in enumerate(long_probes.ORIGINAL_PROFILES): append(1000 + i, profile, 'validation', 5)
    base_count = len(entries)
    for document in range(132): append(document, 'long-full', 'fit', 5, True)
    for document in range(1100, 1131): append(document, 'long-full', 'validation', 5, True)
    rows = pd.DataFrame(entries)
    expected = dict(total=len(rows), base=base_count, fit=1620, valid=175, old_fit=5120,
        long_fit=660, long_valid=155, long_fit_documents=132, long_valid_documents=31,
        removed=4160, removed_documents=26, fit_documents=138)
    # Exactly26 of132 long-full documents overlap sparse originals, while six
    # sparse originals lie outside the selected set.
    for document in range(26, 32):
        mask = (rows.document_id == document) & (rows.profile == 'long-single')
        rows.loc[mask, 'document_id'] += 200
        rows.loc[mask, 'group_id'] = 'c4-' + rows.loc[mask, 'document_id'].astype(str)
    fit, valid, partition = partitions(rows, expected)
    require(len(fit) == 1620 and partition['removed_sparse_rows'] == 4160, 'Sparse replacement fixture failed.')
    checks.append('exact4160_replacement_fit_only_order_and_disjoint_validation')
    corrupt = rows.copy(); corrupt.loc[0, 'split'] = 'test'
    rejects(lambda: partitions(corrupt, expected), 'reject_test_rows')
    corrupt = rows.copy(); corrupt.loc[corrupt.split == 'validation', 'group_id'] = 'c4-0'
    rejects(lambda: partitions(corrupt, expected), 'reject_group_leakage')
    corrupt = rows.copy(); corrupt.loc[0, 'sample_ix'] = corrupt.loc[1, 'sample_ix']
    rejects(lambda: partitions(corrupt, expected), 'reject_duplicate_sample_identity')
    corrupt = rows.copy(); corrupt.loc[0, 'profile'] = 'single'
    rejects(lambda: partitions(corrupt, expected), 'reject_wrong_sparse_replacement')
    generator = torch.Generator().manual_seed(data.SEED); mirror = torch.Generator().manual_seed(data.SEED)
    order = long_probes.epoch_order(FIT_ROWS, generator, 'cpu')
    require(torch.equal(order, torch.randperm(FIT_ROWS, generator=mirror))
        and torch.unique(order).numel() == FIT_ROWS and int(order.min()) == 0 and int(order.max()) == FIT_ROWS - 1,
        'Epoch order is not one deterministic permutation.')
    lengths = [len(order[i:i + BATCH_SIZE]) for i in range(0, len(order), BATCH_SIZE)]
    require(len(lengths) == STEPS and lengths[-1] == 554 and sum(lengths) == FIT_ROWS, 'Epoch budget changed.')
    require(not torch.equal(order, long_probes.epoch_order(FIT_ROWS, generator, 'cpu')), 'Epoch order did not advance.')
    checks.append('seeded_flat_epoch578090_instances142_steps_last554')
    raw = rng.normal(size=(len(rows), 12)).astype(np.float32); raw[:, -1] = 0
    mean, scale = fit_scaler(raw, fit.cache_ix.to_numpy(), chunk=37)
    reference = raw[fit.cache_ix].astype(np.float64)
    require(np.allclose(mean, reference.mean(0), atol=2e-7) and np.allclose(scale,
        np.maximum(reference.std(0), 1e-6), atol=2e-7), 'Streaming scaler differs from direct fit.')
    changed = raw.copy(); changed[valid.cache_ix] += 100000
    altered = fit_scaler(changed, fit.cache_ix.to_numpy(), chunk=61)
    require(np.array_equal(mean, altered[0]) and np.array_equal(scale, altered[1]), 'Validation contaminated scaler.')
    checks.append('fit_only_scaler_streaming_direct_and_validation_invariance')
    x = rng.normal(size=(251, 12)); y = np.arange(len(x)) % 5
    x[:, -4:] = 0  # One legal all-zero head exercises the scanner ridge floor.
    w, b, diagnostics = fit_pooled_lda(x, np.arange(len(x)), y, heads=3, head_dim=4, chunk=17)
    direct = scanner.HeadMoments(heads=3, dimensions=4); direct.update(x.reshape(-1, 3, 4), y)
    wd, bd = direct.fit()
    shaped = x.reshape(-1, 3, 4); means = np.stack([shaped[y == role].mean(0) for role in range(5)], 1)
    within = sum(np.einsum('nhd,nhe->hde', shaped[y == role] - means[:, role],
        shaped[y == role] - means[:, role]) for role in range(5)) / (len(x) - 5)
    total_cov = np.einsum('nhd,nhe->hde', shaped - shaped.mean(0), shaped - shaped.mean(0)) / (len(x) - 1)
    ridge = np.maximum.reduce([np.trace(within, axis1=1, axis2=2) / 4,
        1e-6 * np.trace(total_cov, axis1=1, axis2=2) / 4, np.full(3, 1e-12)])
    covariance = .9 * within + .1 * ridge[:, None, None] * np.eye(4)
    reference_w = np.linalg.solve(covariance, means.transpose(0, 2, 1)).transpose(0, 2, 1)
    reference_b = -.5 * (means * reference_w).sum(-1) + np.log(np.bincount(y, minlength=5) / len(y))
    require(np.allclose(w, reference_w, atol=1e-10, rtol=0) and np.allclose(b, reference_b, atol=1e-10, rtol=0),
        'Streaming LDA differs from independent centered raw covariance reference.')
    lda_difference = max(float(np.abs(w - wd).max()), float(np.abs(b - bd).max()))
    require(lda_difference < 1e-10 and diagnostics['zero_total_heads'] == 1, 'Raw/streamed LDA or zero-head handling differs.')
    p_lda = scanner.predict_heads(x.reshape(-1, 3, 4), wd, bd).mean(1).astype(np.float32)
    checks.append('raw_vs_streaming_shrink_lda_with_zero_head')
    parity = {}
    for kind, hidden in [('linear', 0), ('mlp', 64), ('mlp', 128), ('pooled-lda', 0)]:
        net = make_net(kind, dimensions=12, hidden=hidden, heads=3, head_dim=4).eval()
        m = mean.copy() if kind != 'pooled-lda' else np.zeros(12, np.float32)
        s = scale.copy() if kind != 'pooled-lda' else np.ones(12, np.float32)
        if kind == 'pooled-lda':
            with torch.no_grad(): net.weight.copy_(torch.from_numpy(w)); net.bias.copy_(torch.from_numpy(b))
        meta = dict(kind=kind, hidden=hidden, dimensions=12, heads=3, head_dim=4, roles=data.ROLES)
        path = folder / f'{kind}{hidden}.npz'; save_model(path, net, m, s, meta)
        with torch.inference_mode(): p = net((torch.from_numpy(x.astype(np.float32)) - torch.from_numpy(m)) / torch.from_numpy(s)).exp().numpy()
        actual = predict_raw(path, x.astype(np.float32), np.arange(len(x)))
        diff = float(np.abs(actual - p).max()); require(diff < 1e-7, 'Portable round trip failed.')
        if kind == 'pooled-lda': require(np.allclose(actual, p_lda, atol=2e-7, rtol=0), 'Pooled portable arithmetic differs.')
        parity[f'{kind}{hidden}'] = diff
    checks.append('own_dimension_aware_portable_all_four_models')
    for key, config in CONFIGS.items():
        net = make_net(config['kind'], hidden=config['hidden']).eval()
        if config['kind'] == 'pooled-lda':
            with torch.no_grad():
                net.weight.normal_(0, .1); net.bias.normal_(0, .1)
        values = rng.normal(size=(31, DIMENSIONS)).astype(np.float32)
        meta = dict(kind=config['kind'], hidden=config['hidden'], dimensions=DIMENSIONS,
            heads=HEADS, head_dim=HEAD_DIM, roles=data.ROLES)
        path = folder / f'512-{key}.npz'
        save_model(path, net, np.zeros(DIMENSIONS, np.float32), np.ones(DIMENSIONS, np.float32), meta)
        with torch.inference_mode(): p = net(torch.from_numpy(values)).exp().numpy()
        actual = predict_raw(path, values, np.arange(len(values)))
        require(np.allclose(actual, p, atol=1e-7, rtol=0), '512-dimensional portable consumer differs.')
    checks.append('actual512_dimensions_all_four_portable_consumers')
    for label in ['model', 'probabilities', 'scaler', 'completion']:
        path = folder / f'hash-{label}.bin'; path.write_bytes(b'original frozen content')
        expected_sha = data.digest(path); require_hash(path, expected_sha, label)
        path.write_bytes(b'changed content')
        rejects(lambda p=path, h=expected_sha, label=label: require_hash(p, h, label), f'reject_changed_{label}_hash')
        path.unlink()
        rejects(lambda p=path, h=expected_sha, label=label: require_hash(p, h, label), f'reject_missing_{label}_file')
    bad = np.ones((7, 5), dtype=np.float32)
    rejects(lambda: validate_probabilities(bad), 'reject_unnormalized_probability')
    bad[0, 0] = np.nan
    rejects(lambda: validate_probabilities(bad), 'reject_nonfinite_probability')
    baseline = dict(mean_system_f1=.7, profiles={p: dict(system_f1=.7, accuracy=.8, system_false_positive_rate=.02)
        for p in long_probes.PROFILES}, long_full_tail=dict(system_f1=.6))
    require(promotion_guard(copy.deepcopy(baseline), baseline)['eligible'], 'Equal guard baseline rejected.')
    boundary = copy.deepcopy(baseline)
    boundary['profiles']['single']['system_f1'] -= .02
    boundary['profiles']['multi']['accuracy'] -= .01
    require(promotion_guard(boundary, baseline)['eligible'], 'Exact decimal guard boundary rejected.')
    checks.append('guard_decimal_boundary_tolerance1e-12')
    for name, mutate in [('original_f1_guard', lambda r: r['profiles']['single'].update(system_f1=.679)),
        ('per_profile_accuracy_guard', lambda r: r['profiles']['natural'].update(accuracy=.789)),
        ('full_nonregression_guard', lambda r: r['profiles']['long-full'].update(system_f1=.699)),
        ('tail_nonregression_guard', lambda r: r['long_full_tail'].update(system_f1=.599)),
        ('all_profile_fpr_guard', lambda r: r['profiles']['multi'].update(system_false_positive_rate=.051))]:
        result = copy.deepcopy(baseline); mutate(result)
        require(not promotion_guard(result, baseline)['eligible'], f'Promotion accepted {name}.'); checks.append(name)
    failed = copy.deepcopy(baseline); failed['profiles']['long-full']['system_f1'] -= .1
    selected, _, guards, promoted, reasons = choose_candidate([dict(candidate='diagnostic', validation=failed)], baseline)
    require(selected['candidate'] == 'diagnostic' and not guards['diagnostic']['eligible']
        and not promoted and 'no_guard_eligible_candidate' in reasons, 'No-eligible fallback pretended to pass guards.')
    unchanged = dict(candidate='equal', validation=copy.deepcopy(baseline))
    require(not choose_candidate([unchanged], baseline)[3], 'Equal mean F1 was promoted.')
    improved = copy.deepcopy(baseline); improved['mean_system_f1'] += .01
    require(choose_candidate([unchanged, dict(candidate='improved', validation=improved)], baseline)[0]['candidate'] == 'improved',
        'Selection failed to rank eligible mean F1.')
    checks.append('no_eligible_diagnostic_retention_and_strict_mean_promotion')
    report = dict(cpu_only=True, formal_outputs_written=False, test_inputs_read=False,
        source_sha256=data.digest(__file__), checks=checks, check_count=len(checks),
        lda_raw_stream_max_difference=lda_difference, portable_max_differences=parity,
        fixture_sparse_rows_removed=4160, epoch_instances=FIT_ROWS, epoch_steps=STEPS, last_batch_instances=554,
        future_upstream_ready=(FEATURE_OUT / 'extraction-complete.json').is_file(), limitation=LIMITATION)
    data.atomic_json(folder / 'report.json', report)
    print(json.dumps(dict(preflight=str(folder / 'report.json'), **report), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true'); parser.add_argument('--train', action='store_true')
    parser.add_argument('--worker', choices=CONFIGS, help=argparse.SUPPRESS)
    parser.add_argument('--gpu', type=int, choices=range(4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(not (args.preflight and args.train), 'Choose either CPU preflight or training.')
    if args.preflight:
        require(args.worker is None and args.gpu is None, 'Worker flags cannot accompany CPU preflight.'); preflight()
    elif args.train:
        if args.worker:
            require(args.gpu is not None, 'Worker needs an assigned GPU.'); worker(args.worker, args.gpu)
        else:
            require(args.gpu is None, '--gpu is an internal worker argument.'); train()
    else:
        require(args.worker is None and args.gpu is None, 'Worker execution requires explicit --train.'); parser.print_help()


if __name__ == '__main__':
    main()
