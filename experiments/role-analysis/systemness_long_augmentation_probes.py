"""Matched long-source augmentation: two MLP128 fits, six development records.

Default help. --preflight writes only synthetic CPU fixtures under /tmp.
--declare freezes the training protocol without reading probability arrays;
--train requires that declaration and completed new development extraction.
Only new-maximin can pass the prospective gate; no primary file is changed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
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
import systemness_long_probes as lp
import systemness_probes as probes

OUT = data.runtime.OUT / 'systemness-long-augmentation-probes'
CACHE = data.CACHE / 'long-augmentation-probes'
NEW_OUT = data.runtime.OUT / 'systemness-long-augmentation'
NEW_BANK = Path('/dev/shm/role-analysis-long-augmentation-heads')
DESIGN = data.runtime.OUT / 'systemness-long-augmentation-preparation/prospective-experiment-design.json'
DESIGN_SHA = '1d75c30d983aba2b7bbedb2ea61aabac74456157a235f76c89068d6c5fe48f85'
CURRENT_MODEL_SHA = '27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
CURRENT_THRESHOLD = .620969696969697
DIMENSIONS, BATCH, STEPS, INSTANCES = 512, 4096, 222, 905770
OLD_FIT, OLD_VALID, EXTRA_FIT, NEW_VALID = 578090, 173032, 327680, 61440
ARTIFACTS = ['protocol.json', 'search-plan.json', 'draw-manifest.json', 'records.json', 'validation-metrics.csv', 'report.txt', 'source-counts.json', 'selection.json', 'old-head-control.json']
ARMS = ['new', 'matched']
MODELS = ['old-head', 'new', 'matched']
OBJECTIVES = ['old-mean', 'maximin']
GROUPS = ['old-' + p for p in lp.PROFILES] + ['old-long-full-tail', 'new-long-full', 'new-long-full-tail']
FEASIBLE_GROUPS = ['old-' + p for p in lp.PROFILES] + ['new-long-full']
DRAW_SEED = data.SEED + 256132
DEPENDENCIES = ['systemness_long_augmentation_probes.py', 'systemness_long_augmentation_data.py',
    'systemness_long_augmentation_extract.py', 'systemness_head_probes.py', 'systemness_head_extract.py',
    'systemness_head_scan.py', 'systemness_long_probes.py', 'systemness_long_data.py',
    'systemness_probes.py', 'systemness_data.py']
LIMITATION = ('One seed and reused development validation; this is not a causal proof. New and matched arms '
    'share initialization, old fit-only scaler, per-epoch instances and maximum budget. Independent early stopping '
    'can change best and actual total updates. Distinct document coverage and example weights still differ. '
    'The new96 validation documents are calibration/development data, never an independent confirmation. '
    'Previously inspected C4 or natural tests are not read or reused; a promoted candidate needs new independent confirmation.')
require = hp.require
read = hp.read
array_digest = hp.array_digest


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def protocol():
    return dict(version=1, experiment='matched-long-source-augmentation', arms=ARMS, models=MODELS,
        trained_candidates=2, diagnostic_records=6, roles=data.ROLES, dimensions=DIMENSIONS,
        fit_rows=INSTANCES, old_fit_rows=OLD_FIT, extra_rows=EXTRA_FIT,
        old_validation_rows=OLD_VALID, new_validation_rows=NEW_VALID,
        neural=dict(seed=data.SEED, hidden=128, optimizer='AdamW', learning_rate=.001, weight_decay=.001,
            dropout=.1, batch_size=BATCH, steps_per_epoch=STEPS, last_batch_instances=554,
            max_epochs=45, min_epochs=10, patience=8, improvement=1e-5, clip=5., loss='Five-class mean CE'),
        scaler='Reuse unchanged original578090-fit-only head512 scaler; never refit on new fit or validation.',
        matched=dict(seed=DRAW_SEED, pool=132, first_permutation=132, second_permutation_prefix=124,
            draw_once=True, packets=256, packet_order='extra_docslot/ROLES/body_offset',
            rows_per_packet=1280, rng='Private NumPy RandomState; never consume torch CE/dropout RNG.'),
        early_stop='Exact old5-profile long_probes.validation_metrics mean-F1; no new validation predictions before best model freeze.',
        objectives=OBJECTIVES, maximin_groups=GROUPS, fpr_feasible_groups=FEASIBLE_GROUPS,
        threshold_grid='unique(linspace(.001,.999,199), quantile(all old+new validation p_system,linspace(0,1,301)),1.000001) for maximin; unchanged old-only helper grid for old-mean.',
        maximin='Minimum actual-F32-consumer P/R/BAcc across8 groups; undefined P infeasible; tie higher old5 meanF1 then smaller originalF64 tau; full-grid direct-count proof.',
        old_mean='Exact original helper on old5 only, including its legacy F64 sorted-count semantics; old-head must reproduce its original saved NPZ and tau.',
        promotion=dict(candidate='new-maximin', no_fallback=True, strict_min24_controls=['matched-maximin', 'old-head-maximin'],
            guard_controls=['matched-maximin', 'old-head-maximin', 'old-head-old-mean'],
            old4_maximum_f1_loss=.02, old4_maximum_five_accuracy_loss=.01,
            old5_maximum_mean_f1_loss=.005, full_tail_f1_nonregression=['old-long-full', 'old-long-full-tail', 'new-long-full', 'new-long-full-tail'],
            fpr_maximum=.05, comparison_tolerance=1e-12),
        output=str(OUT), cache=str(CACHE), tests_read=False, primary_changed=False,
        fresh_confirmation_required=True, limitation=LIMITATION)


def init_hash(net):
    values = []
    for name, value in net.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        values.append(dict(name=name, shape=list(a.shape), dtype=str(a.dtype), sha256=array_digest(a)))
    return json_digest(values)


def expected_initialization():
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(data.SEED)
        net = hp.make_net('mlp', hidden=128)
        return init_hash(net)


def static_snapshot():
    """Source/model/row metadata only. Never open any probability NPZ."""
    hp.require_hash(DESIGN, DESIGN_SHA, 'root prospective design')
    files = [hp.OUT / n for n in ['search-plan.json', 'complete.json', 'selection.json', 'standardization.npz']]
    files += [hp.OUT / 'candidates/joint-mlp128.json', hp.OUT / 'models/joint-mlp128.npz',
        hp.FEATURE_OUT / 'extraction-complete.json', hp.BANK / 'rows.parquet']
    require(all(p.is_file() for p in files), 'Missing original head metadata/model/scaler/row bank.')
    complete = read(hp.OUT / 'complete.json'); selection = read(hp.OUT / 'selection.json')
    candidate = read(hp.OUT / 'candidates/joint-mlp128.json')
    require(selection['candidate'] == 'joint-mlp128' and selection['promoted'] is True
        and complete['source_sha256'] == data.digest(hp.__file__)
        and candidate['model_sha256'] == selection['model_sha256'] == CURRENT_MODEL_SHA
        and candidate['validation']['system_threshold'] == selection['system_threshold'] == CURRENT_THRESHOLD,
        'Original frozen head128 model/threshold changed.')
    for name, key in [('search-plan.json', 'search_plan_sha256'), ('selection.json', 'selection_sha256'),
                      ('standardization.npz', 'standardization_sha256')]:
        hp.require_hash(hp.OUT / name, complete[key], 'original head ' + name)
    hp.require_hash(hp.OUT / 'candidates/joint-mlp128.json', complete['candidate_sha256']['joint-mlp128'], 'old candidate')
    hp.require_hash(hp.OUT / 'models/joint-mlp128.npz', CURRENT_MODEL_SHA, 'old portable model')
    require(candidate['validation_probabilities_sha256'] == selection['validation_probabilities_sha256'], 'Old probability declaration changed.')
    return dict(source_sha256=data.digest(__file__), design_sha256=DESIGN_SHA,
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        metadata_sha256={str(p): data.digest(p) for p in files},
        old_probability_path=candidate['validation_probabilities_path'],
        old_probability_declared_sha256=candidate['validation_probabilities_sha256'],
        initial_weight_sha256=expected_initialization(),
        package_versions={name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'pandas', 'scikit-learn']},
        float32_matmul_precision='highest', probability_arrays_read=False)


def declare():
    value = dict(protocol=protocol(), static=static_snapshot())
    path = OUT / 'protocol.json'
    if path.exists():
        require(read(path) == value, 'Frozen training declaration changed.')
        return value
    require(not CACHE.exists() and not (OUT / 'search-plan.json').exists(), 'Training artifacts predate declaration.')
    OUT.mkdir(parents=True, exist_ok=True); data.atomic_json(path, value)
    return value


def verify_declaration():
    require((OUT / 'protocol.json').is_file(), 'Run --declare before any validation probability read or training.')
    declaration = read(OUT / 'protocol.json')
    require(declaration == dict(protocol=protocol(), static=static_snapshot()), 'Training source/protocol/model freeze changed.')
    return declaration


def packets(frame, documents, samples):
    require(set(frame.role) == set(data.ROLES) and frame.document_id.nunique() == documents
        and len(frame) == documents * 5 * samples, 'Document-packet source/role count changed.')
    docs = sorted(int(x) for x in frame.document_id.unique()); rows = []
    grid = data.uniform_indices(1024, samples)
    for doc in docs:
        indices = []
        part = frame[frame.document_id == doc]
        for role in data.ROLES:
            r = part[part.role == role].sort_values('token_ix')
            offsets = r.body_offset.to_numpy() if 'body_offset' in r else r.token_ix.to_numpy() - r.token_ix.min()
            require(len(r) == samples and np.array_equal(offsets, grid), 'Packet body offsets/role coverage changed.')
            indices.extend(int(x) for x in r.cache_ix)
        rows.append(indices)
    return docs, np.asarray(rows, np.int64), np.repeat(np.arange(5), samples), np.tile(grid, 5)


def augmentation_layout(old_rows, new_rows, *, expected=None):
    expected = expected or dict(old_pool=132, new_fit=256, new_valid=96, samples_fit=256, samples_valid=128)
    old_fit, old_valid, partition = hp.partitions(old_rows) if expected['old_pool'] == 132 else (None, None, {})
    require(np.array_equal(new_rows.cache_ix, np.arange(len(new_rows))) and new_rows.sample_ix.is_unique
        and set(new_rows.split) == {'fit', 'validation'} and set(new_rows.wrapper) == {'native'}
        and set(new_rows.source_set) == {'long-augmentation'} and set(new_rows.profile) == {'long-full'}
        and new_rows.sample_ix.min() >= 6_000_000_000_000 and new_rows.document_id.min() >= 6_000_000,
        'New feature-bank namespace, split, wrapper or role order changed.')
    require(not set(new_rows.sample_ix) & set(old_rows.sample_ix) and not set(new_rows.group_id) & set(old_rows.group_id), 'Old/new source identity collision.')
    pool = old_rows[(old_rows.split == 'fit') & (old_rows.profile == 'long-full') & (old_rows.source_set == 'long-full')]
    old_docs, pool_ix, labels, offsets = packets(pool, expected['old_pool'], expected['samples_fit'])
    nf = new_rows[new_rows.split == 'fit']; nv = new_rows[new_rows.split == 'validation'].copy().reset_index(drop=True)
    new_docs, new_ix, nl, no = packets(nf, expected['new_fit'], expected['samples_fit'])
    packets(nv, expected['new_valid'], expected['samples_valid'])
    require(not set(nf.document_id) & set(nv.document_id) and not set(nf.group_id) & set(nv.group_id), 'New fit/validation leakage.')
    require(np.array_equal(labels, nl) and np.array_equal(offsets, no), 'Arms do not match formal-role/body-position layout.')
    rng = np.random.RandomState(DRAW_SEED); first = rng.permutation(len(old_docs)); second = rng.permutation(len(old_docs))
    needed = expected['new_fit'] - len(old_docs)
    require(0 <= needed <= len(old_docs), 'Matched draw requires one complete permutation plus one prefix.')
    draw = np.r_[first, second[:needed]]; matched = pool_ix[draw].reshape(-1); fresh = new_ix.reshape(-1)
    manifest = dict(seed=DRAW_SEED, draw_once=True, pool_document_ids=old_docs,
        first_permutation=[old_docs[i] for i in first], second_permutation=[old_docs[i] for i in second],
        slots=[dict(extra_docslot=i, old_document_id=old_docs[int(d)], new_document_id=new_docs[i]) for i, d in enumerate(draw)],
        matched_cache_ix_sha256=array_digest(matched), new_cache_ix_sha256=array_digest(fresh),
        extra_labels_sha256=array_digest(np.tile(labels, len(draw))),
        extra_body_offsets_sha256=array_digest(np.tile(offsets, len(draw))),
        extra_rows=len(fresh), distinct_matched_documents=len(set(draw)),
        matched_document_multiplicities={str(old_docs[i]): int(np.count_nonzero(draw == i)) for i in range(len(old_docs))},
        no_validation_rows_sampled=True, torch_rng_consumed=False)
    if old_fit is not None:
        require(len(fresh) == EXTRA_FIT and len(old_fit) + len(fresh) == INSTANCES and len(nv) == NEW_VALID
            and not set(matched) & set(old_valid.cache_ix), 'Fit/validation instances or matched row ownership changed.')
    return old_fit, old_valid, nv, fresh, matched, manifest, partition


def dataset():
    old_rows = pd.read_parquet(hp.BANK / 'rows.parquet'); new_rows = pd.read_parquet(NEW_BANK / 'rows.parquet')
    parts = augmentation_layout(old_rows, new_rows)
    banks = {}
    for name, folder, count in [('old', hp.BANK, hp.TOTAL_ROWS), ('new', NEW_BANK, EXTRA_FIT + NEW_VALID)]:
        require((folder / 'features.f16').is_file() and (folder / 'features.f16').stat().st_size == count * DIMENSIONS * 2,
            'Missing/truncated ' + name + ' feature bank; no saved progress may authorize reuse.')
        banks[name] = np.memmap(folder / 'features.f16', mode='r', dtype=np.float16, shape=(count, DIMENSIONS))
    return old_rows, new_rows, banks, parts


def source_signature():
    verify_declaration()
    extract = importlib.import_module('systemness_long_augmentation_extract')
    require(extract.OUT == NEW_OUT and extract.CACHE == NEW_BANK, 'New extractor interface changed.')
    complete = extract.verified_extraction()
    old = read(hp.FEATURE_OUT / 'extraction-complete.json')
    require(complete['rows'] == EXTRA_FIT + NEW_VALID and complete['dimensions'] == DIMENSIONS
        and complete['all_finite'] is True and complete['all_rows_written'] is True
        and complete['selected_head_ids'] == old['selected_head_ids'], 'New extraction completeness/head rank changed.')
    hp.require_hash(NEW_BANK / 'rows.parquet', complete['rows_sha256'], 'new feature row table')
    hp.require_hash(NEW_BANK / 'features.f16', complete['features_sha256'], 'new features')
    hp.require_hash(hp.BANK / 'rows.parquet', old['rows_sha256'], 'old feature row table')
    hp.require_hash(hp.BANK / 'features.f16', old['features_sha256'], 'old features')
    old_rows, new_rows, banks, parts = dataset()
    fit, valid, new_valid, fresh, matched, manifest, partition = parts
    return dict(protocol_sha256=data.digest(OUT / 'protocol.json'),
        new_data_sha256=data.digest(NEW_OUT / 'data-complete.json'),
        new_extraction_sha256=data.digest(NEW_OUT / 'extraction-complete.json'),
        new_extraction_signature=complete['signature'], old_extraction_sha256=data.digest(hp.FEATURE_OUT / 'extraction-complete.json'),
        old_features_sha256=old['features_sha256'], new_features_sha256=complete['features_sha256'],
        old_rows_sha256=old['rows_sha256'], new_rows_sha256=complete['rows_sha256'],
        selected_head_ids=complete['selected_head_ids'], old_partitions=partition,
        new_validation_sample_ix_sha256=array_digest(new_valid.sample_ix.to_numpy()),
        new_validation_cache_ix_sha256=array_digest(new_valid.cache_ix.to_numpy()),
        new_fit_cache_ix_sha256=array_digest(fresh), matched_cache_ix_sha256=array_digest(matched),
        draw_manifest_sha256=json_digest(manifest), standardization_sha256=data.digest(hp.OUT / 'standardization.npz'))


def verify_plan(plan, full=True):
    declaration = verify_declaration()
    require(plan['protocol_sha256'] == data.digest(OUT / 'protocol.json')
        and plan['initial_weight_sha256'] == declaration['static']['initial_weight_sha256'], 'Training plan/declaration changed.')
    require(read(OUT / 'draw-manifest.json') == plan['draw_manifest']
        and json_digest(plan['draw_manifest']) == plan['signature']['draw_manifest_sha256'], 'Frozen matched draw changed.')
    if full: require(plan['signature'] == source_signature(), 'Training bank/source/partition signature changed.')
    return declaration


def read_scaler():
    with np.load(hp.OUT / 'standardization.npz', allow_pickle=False) as z:
        mean, scale = z['mean'].copy(), z['scale'].copy()
    require(mean.shape == scale.shape == (DIMENSIONS,) and mean.dtype == scale.dtype == np.float32
        and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), 'Original frozen scaler is invalid.')
    return mean, scale


def validation_groups(old_rows, new_rows):
    require(np.array_equal(old_rows.index, np.arange(len(old_rows))) and np.array_equal(new_rows.index, np.arange(len(new_rows))), 'Validation probability order must reset within each bank.')
    groups = {'old-' + p: ('old', np.flatnonzero(old_rows.profile.eq(p))) for p in lp.PROFILES}
    for label, frame in [('old', old_rows[old_rows.profile.eq('long-full')]), ('new', new_rows)]:
        require(frame.profile.eq('long-full').all(), 'Unexpected new validation profile.')
        offset = frame.body_offset if 'body_offset' in frame else frame.token_ix - frame.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')
        groups[label + '-long-full-tail'] = (label, frame.index[offset >= 512].to_numpy())
        if label == 'new': groups['new-long-full'] = ('new', frame.index.to_numpy())
    require(set(groups) == set(GROUPS) and all(len(ix) for _, ix in groups.values()), 'Missing maximin calibration group.')
    return groups


def maximin_threshold(old_rows, new_rows, old_p, new_p, *, return_curve=False):
    """Actual F32 consumer,8 groups24 stats, direct-count proof at every tau."""
    groups = validation_groups(old_rows, new_rows); probabilities = dict(old=old_p, new=new_p)
    ys = dict(old=probes.labels(old_rows), new=probes.labels(new_rows))
    for name, value in probabilities.items(): hp.validate_probabilities(value, len(ys[name])); require(value.dtype == np.float32, 'Maximin needs actual saved F32 probabilities.')
    grid = np.unique(np.r_[np.linspace(.001, .999, 199), np.quantile(np.r_[old_p[:, 0], new_p[:, 0]], np.linspace(0, 1, 301)), 1.000001])
    effective = grid.astype(np.float32); curve = {}; direct_curve = {}; maximum = 0.
    for name, (bank, ix) in groups.items():
        truth = ys[bank][ix] == 0; p = probabilities[bank][ix, 0]
        pos, neg = np.sort(p[truth]), np.sort(p[~truth]); require(len(pos) and len(neg), 'Each maximin group requires actual positive/negative support.')
        tp = len(pos) - np.searchsorted(pos, effective, side='left'); fp = len(neg) - np.searchsorted(neg, effective, side='left')
        values = dict(P=np.divide(tp, tp + fp, out=np.full(len(grid), np.nan), where=tp + fp > 0), R=tp / len(pos),
            BAcc=(tp / len(pos) + (len(neg) - fp) / len(neg)) / 2, FPR=fp / len(neg), F1=2 * tp / (2 * tp + fp + len(pos) - tp))
        direct = {k: [] for k in values}
        for tau in grid:
            detection = p >= float(tau); a = int(np.count_nonzero(truth & detection)); b = int(np.count_nonzero(~truth & detection)); c = len(pos) - a; d = len(neg) - b
            for key, value in dict(P=a / (a + b) if a + b else np.nan, R=a / (a + c), BAcc=(a / (a + c) + d / (b + d)) / 2,
                FPR=b / (b + d), F1=2 * a / (2 * a + b + c)).items(): direct[key].append(value)
        for key in values:
            exact = np.asarray(direct[key]); require(np.array_equal(values[key], exact, equal_nan=True), 'Maximin F32 grid mismatch: ' + name + '/' + key)
        curve[name] = values; direct_curve[name] = {key: np.asarray(value) for key, value in direct.items()}
    minimum = np.min([curve[name][key] for name in GROUPS for key in ['P', 'R', 'BAcc']], axis=0)
    minimum[~np.isfinite(minimum)] = -np.inf
    mean = np.mean([curve['old-' + p]['F1'] for p in lp.PROFILES], axis=0)
    feasible = np.max([curve[name]['FPR'] for name in FEASIBLE_GROUPS], axis=0) <= .05
    candidates = np.flatnonzero(feasible & np.isfinite(minimum)); require(len(candidates), 'No defined FPR-feasible maximin threshold.')
    best = max(candidates, key=lambda i: (float(minimum[i]), float(mean[i]), -float(grid[i])))
    direct_minimum = np.min([direct_curve[name][key] for name in GROUPS for key in ['P', 'R', 'BAcc']], axis=0)
    direct_minimum[~np.isfinite(direct_minimum)] = -np.inf
    direct_mean = np.mean([direct_curve['old-' + profile]['F1'] for profile in lp.PROFILES], axis=0)
    direct_feasible = np.max([direct_curve[name]['FPR'] for name in FEASIBLE_GROUPS], axis=0) <= .05
    require(np.array_equal(minimum, direct_minimum) and np.array_equal(mean, direct_mean)
        and np.array_equal(feasible, direct_feasible), 'Full-grid direct objective/feasibility changed.')
    direct_best = max(np.flatnonzero(direct_feasible & np.isfinite(direct_minimum)),
        key=lambda i: (float(direct_minimum[i]), float(direct_mean[i]), -float(grid[i])))
    require(best == direct_best, 'Full-grid direct maximin argmax changed.')
    proof = dict(grid_thresholds=len(grid), groups=8, primary_statistics=len(grid) * 24,
        additional_f1_fpr_statistics=len(grid) * 16, maximum_direct_difference=maximum,
        all_grid_undefined_and_count_semantics_equal=True, actual_F32_comparison=True,
        all_grid_feasibility_equal=True, global_argmax_equal=True,
        min24=float(minimum[best]), old5_mean_f1=float(mean[best]), feasible_thresholds=int(feasible.sum()))
    if return_curve: return float(grid[best]), proof, (grid, curve, mean, minimum, feasible)
    return float(grid[best]), proof


def metrics_at(old_rows, new_rows, old_p, new_p, threshold):
    output = {}; groups = validation_groups(old_rows, new_rows)
    for name, (bank, ix) in groups.items():
        rows, p = (old_rows, old_p) if bank == 'old' else (new_rows, new_p)
        output[name] = probes.metrics(p[ix], probes.labels(rows.iloc[ix]), threshold)
    values = [output[name][key] for name in GROUPS for key in ['system_precision', 'system_recall', 'system_balanced_accuracy']]
    minimum = min(values) if all(v is not None and np.isfinite(v) for v in values) else None
    pooled = probes.metrics(np.concatenate([old_p, new_p]), np.r_[probes.labels(old_rows), probes.labels(new_rows)], threshold)
    return dict(system_threshold=float(threshold), groups=output,
        old5_mean_f1=float(np.mean([output['old-' + p]['system_f1'] for p in lp.PROFILES])), min24=minimum,
        pooled_descriptive_only=pooled)


def promotion_guard(result, control):
    tol = 1e-12; reasons = []
    f1_loss = {p: control['groups']['old-' + p]['system_f1'] - result['groups']['old-' + p]['system_f1'] for p in lp.ORIGINAL_PROFILES}
    acc_loss = {p: control['groups']['old-' + p]['accuracy'] - result['groups']['old-' + p]['accuracy'] for p in lp.ORIGINAL_PROFILES}
    full = ['old-long-full', 'old-long-full-tail', 'new-long-full', 'new-long-full-tail']
    gains = {p: result['groups'][p]['system_f1'] - control['groups'][p]['system_f1'] for p in full}
    mean_loss = control['old5_mean_f1'] - result['old5_mean_f1']
    finite = all(np.isfinite(result['groups'][g][k]) for g in GROUPS for k in ['system_f1', 'accuracy', 'system_false_positive_rate']) and np.isfinite(result['old5_mean_f1'])
    if not finite: reasons.append('nonfinite_validation_metric')
    if max(f1_loss.values()) > .02 + tol: reasons.append('original4_system_f1_loss')
    if max(acc_loss.values()) > .01 + tol: reasons.append('original4_five_accuracy_loss')
    if any(v < -tol for v in gains.values()): reasons.append('old_or_new_full_tail_f1_regression')
    if mean_loss > .005 + tol: reasons.append('old5_mean_f1_loss')
    if any(result['groups'][g]['system_false_positive_rate'] > .05 + tol for g in FEASIBLE_GROUPS): reasons.append('profile_fpr_above_limit')
    return dict(eligible=not reasons, rejection_reasons=reasons, original4_f1_losses=f1_loss,
        original4_accuracy_losses=acc_loss, full_tail_f1_gains=gains, old5_mean_f1_loss=mean_loss)


def select_records(records):
    index = {r['record']: r for r in records}
    require(set(index) == {m + '-' + o for m in MODELS for o in OBJECTIVES} and len(records) == 6, 'All six frozen records are required.')
    candidate = index['new-maximin']; guards = {name: promotion_guard(candidate['validation'], index[name]['validation'])
        for name in ['matched-maximin', 'old-head-maximin', 'old-head-old-mean']}
    minimum = candidate['validation']['min24']; controls = ['matched-maximin', 'old-head-maximin']
    improvements = {k: (minimum - index[k]['validation']['min24']) if minimum is not None and index[k]['validation']['min24'] is not None else None for k in controls}
    reasons = [k + ':' + reason for k, guard in guards.items() for reason in guard['rejection_reasons']]
    for key, gain in improvements.items():
        if gain is None or not np.isfinite(gain) or gain <= 1e-12: reasons.append(key + ':min24_not_strictly_improved')
    return dict(candidate='new-maximin', promoted=not reasons, no_fallback=True, guards=guards,
        min24_gains=improvements, promotion_rejection_reasons=reasons, primary_changed=False,
        new_independent_confirmation_required=True, tests_read=False)


def check_history(history, best_epoch, *, steps=STEPS, instances=INSTANCES, maximum=45, minimum=10, patience=8):
    require(1 <= len(history) <= maximum and 1 <= best_epoch <= len(history), 'Invalid early-stop history length.')
    best = -1.; chosen = -1
    for i, row in enumerate(history, 1):
        require(row['epoch'] == i and row['instances'] == instances and row['steps'] == steps
            and row['cumulative_steps'] == i * steps and np.isfinite(row['loss'])
            and np.isfinite(row['old5_mean_f1']), 'Epoch instance/update/evidence changed.')
        if row['old5_mean_f1'] > best + 1e-5: best = row['old5_mean_f1']; chosen = i
        if i < len(history): require(not (i >= minimum and i - chosen >= patience), 'Training continued past original early-stop rule.')
    require(chosen == best_epoch and (len(history) == maximum or (len(history) >= minimum and len(history) - chosen >= patience)),
        'Best epoch or terminal early-stop rule changed.')
    return dict(best_epoch=best_epoch, best_steps=best_epoch * steps, epochs=len(history),
        actual_steps=len(history) * steps, actual_instances=len(history) * instances,
        instances_per_epoch=instances, steps_per_epoch=steps, last_batch_instances=instances % BATCH or BATCH)


def read_probability(path, sha256, rows):
    hp.require_hash(path, sha256, 'saved development probabilities')
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {'p', 'sample_ix'} and np.array_equal(z['sample_ix'], rows.sample_ix), 'Probability sample namespace/order changed.')
        p = z['p'].copy()
    hp.validate_probabilities(p, len(rows)); require(p.dtype == np.float32, 'Saved probability dtype changed.')
    return p


def candidate_record(arm, plan, *, full=True):
    verify_plan(plan, full=full)
    record = read(OUT / 'candidates' / (arm + '.json'))
    require(arm in ARMS and record['arm'] == arm and record['plan_sha256'] == data.digest(OUT / 'search-plan.json')
        and record['signature'] == plan['signature'] and record['source_sha256'] == data.digest(__file__)
        and record['initial_weight_sha256'] == plan['initial_weight_sha256'] and record['tests_read'] is False,
        'Candidate ownership/source/initialization changed.')
    evidence = read(OUT / 'training' / (arm + '.json'))
    hp.require_hash(OUT / 'training' / (arm + '.json'), record['training_complete_sha256'], 'best-model pre-newval freeze')
    require(evidence['arm'] == arm and evidence['signature'] == plan['signature']
        and evidence['plan_sha256'] == data.digest(OUT / 'search-plan.json')
        and evidence['initial_weight_sha256'] == plan['initial_weight_sha256'], 'Training evidence arm/source/initialization changed.')
    budget = check_history(evidence['history'], evidence['best_epoch'])
    require(all(record[k] == v and evidence[k] == v for k, v in budget.items())
        and evidence['new_validation_predictions_read'] is False
        and evidence['model_sha256'] == record['model_sha256'], 'Training/early-stop evidence changed.')
    path = OUT / 'models' / (arm + '.npz'); hp.require_hash(path, record['model_sha256'], 'candidate portable model')
    _, _, _, parts = dataset(); _, old_valid, new_valid, *_ = parts
    pold = read_probability(CACHE / (arm + '-old-validation.npz'), record['old_probability_sha256'], old_valid)
    pnew = read_probability(CACHE / (arm + '-new-validation.npz'), record['new_probability_sha256'], new_valid)
    old_validation = lp.validation_metrics(old_valid, pold)
    require(old_validation == record['old_early_stop_validation']
        and old_validation['mean_system_f1'] == evidence['history'][evidence['best_epoch'] - 1]['old5_mean_f1'],
        'Saved best checkpoint does not match old-only early stopping.')
    net, mean, scale, meta = hp.load_model(path)
    expected_mean, expected_scale = read_scaler()
    require(meta['arm'] == arm and meta['kind'] == 'mlp' and meta['hidden'] == 128
        and meta['dimensions'] == DIMENSIONS and meta['signature'] == plan['signature']
        and meta['initial_weight_sha256'] == plan['initial_weight_sha256']
        and meta['best_epoch'] == evidence['best_epoch'] and meta['best_steps'] == evidence['best_steps']
        and meta['source_sha256'] == data.digest(__file__) and meta['plan_sha256'] == data.digest(OUT / 'search-plan.json')
        and np.array_equal(mean.numpy(), expected_mean) and np.array_equal(scale.numpy(), expected_scale), 'Portable model/scaler provenance changed.')
    proof = record['portable_evidence']
    require(proof['model_sha256'] == record['model_sha256'] and proof['sampled_tokens'] == dict(old=256, new=256)
        and set(proof['maximum_probability_difference']) == {'old', 'new'}
        and all(np.isfinite(v) and 0 <= v < 5e-4 for v in proof['maximum_probability_difference'].values()), 'Actual saved-model portable proof changed.')
    return record, pold, pnew


def supervisor_alive():
    require(os.environ.get('ROLE_LONG_AUG_SUPERVISOR_PID') == str(os.getppid()), 'Worker requires the original live supervisor lease.')


def worker(arm, gpu):
    supervisor_alive(); require(arm in ARMS and gpu in [0, 1], 'Only two assigned workers are authorized.')
    require(not (OUT / 'complete.json').exists() and not (OUT / 'selection.json').exists()
        and not (OUT / 'candidates' / (arm + '.json')).exists(), 'A frozen candidate/result cannot be overwritten.')
    no_competing_jobs(allow_own=True); plan = read(OUT / 'search-plan.json'); verify_plan(plan)
    import torch
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1
        and 'A40' in torch.cuda.get_device_name(0), 'Each readout worker requires its one assigned A40.')
    _, _, banks, parts = dataset(); old_fit, old_valid, new_valid, fresh, matched, manifest, _ = parts
    require(manifest == plan['draw_manifest'], 'Worker matched packet manifest changed.')
    mean, scale = read_scaler()
    def standardized(bank, indices):
        value = torch.empty((len(indices), DIMENSIONS), dtype=torch.float32, device='cuda', requires_grad=False)
        for start in range(0, len(indices), BATCH):
            x = (np.asarray(banks[bank][indices[start:start + BATCH]], dtype=np.float32) - mean) / scale
            require(np.isfinite(x).all(), 'Nonfinite standardized development feature.')
            value[start:start + len(x)] = torch.from_numpy(x).cuda()
        return value
    free, _ = torch.cuda.mem_get_info(0)
    require(free > (INSTANCES + OLD_VALID) * DIMENSIONS * 4 + 2 * 2**30, 'Insufficient assigned GPU memory.')
    old_x = standardized('old', old_fit.cache_ix.to_numpy())
    extra_x = standardized('new' if arm == 'new' else 'old', fresh if arm == 'new' else matched)
    x = torch.cat([old_x, extra_x]); del old_x, extra_x
    # Labels and body-position layout are identical in both arms before the same epoch permutation.
    y = torch.from_numpy(np.r_[probes.labels(old_fit), np.tile(np.repeat(np.arange(5), 256), 256)]).cuda()
    xv = standardized('old', old_valid.cache_ix.to_numpy())
    torch.manual_seed(data.SEED); net = hp.make_net('mlp', hidden=128).cuda()
    require(init_hash(net) == plan['initial_weight_sha256'], 'Arms did not begin at the same original initialization.')
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    generator = torch.Generator(device='cuda:0').manual_seed(data.SEED)
    path = OUT / 'models' / (arm + '.npz'); old_path = CACHE / (arm + '-old-validation.npz')
    metadata = dict(arm=arm, kind='mlp', hidden=128, dimensions=DIMENSIONS, heads=8, head_dim=64, roles=data.ROLES,
        source_sha256=data.digest(__file__), signature=plan['signature'], plan_sha256=data.digest(OUT / 'search-plan.json'),
        selected_head_ids=plan['signature']['selected_head_ids'], initial_weight_sha256=plan['initial_weight_sha256'],
        standardization_sha256=plan['signature']['standardization_sha256'], tests_read=False)
    best = -1.; best_epoch = -1; history = []; started = time.time()
    for epoch in range(1, 46):
        supervisor_alive(); net.train(); order = lp.epoch_order(len(x), generator, 'cuda'); loss_sum = 0.; steps = 0
        require(len(order) == INSTANCES, 'Training instance budget changed.')
        for start in range(0, len(order), BATCH):
            ix = order[start:start + BATCH]
            loss = torch.nn.functional.cross_entropy(net.raw(x[ix].contiguous()), y[ix]); require(torch.isfinite(loss), 'Nonfinite training CE.')
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            loss_sum += float(loss.detach()) * len(ix); steps += 1
        require(steps == STEPS, 'Per-epoch optimizer update budget changed.')
        p = probes.predict_standardized(net, xv).astype(np.float32); hp.validate_probabilities(p, len(old_valid))
        result = lp.validation_metrics(old_valid, p); score = result['mean_system_f1']
        history.append(dict(epoch=epoch, loss=loss_sum / len(order), instances=len(order), steps=steps,
            cumulative_steps=epoch * STEPS, old5_mean_f1=score))
        if score > best + 1e-5:
            best, best_epoch = score, epoch
            hp.save_model(path, net, mean, scale, metadata | dict(best_epoch=epoch, best_steps=epoch * STEPS))
            hp.atomic_npz(old_path, p=p, sample_ix=old_valid.sample_ix.to_numpy())
        print(json.dumps(dict(arm=arm, **history[-1], best_epoch=best_epoch)), flush=True)
        if epoch >= 10 and epoch - best_epoch >= 8: break
    budget = check_history(history, best_epoch); supervisor_alive()
    # Best model and all early-stop evidence are frozen BEFORE the first new-val prediction.
    training = dict(arm=arm, signature=plan['signature'], plan_sha256=data.digest(OUT / 'search-plan.json'),
        initial_weight_sha256=plan['initial_weight_sha256'], history=history, **budget,
        model_sha256=data.digest(path), old_probability_sha256=data.digest(old_path),
        new_validation_predictions_read=False, seconds=time.time() - started)
    data.atomic_json(OUT / 'training' / (arm + '.json'), training)
    old_p = read_probability(old_path, training['old_probability_sha256'], old_valid)
    new_p = hp.predict_raw(path, banks['new'], new_valid.cache_ix.to_numpy(), device='cuda:0')
    new_path = CACHE / (arm + '-new-validation.npz'); hp.atomic_npz(new_path, p=new_p, sample_ix=new_valid.sample_ix.to_numpy())
    differences = {}
    for label, bank, rows, values in [('old', banks['old'], old_valid, old_p), ('new', banks['new'], new_valid, new_p)]:
        ix = np.linspace(0, len(rows) - 1, 256, dtype=int)
        cpu = hp.predict_raw(path, bank, rows.iloc[ix].cache_ix.to_numpy())
        difference = float(np.abs(cpu - values[ix]).max()); require(np.isfinite(difference) and difference < 5e-4, 'Saved portable model differs from actual worker predictions.')
        differences[label] = difference
    verify_plan(plan); supervisor_alive()
    data.atomic_json(OUT / 'candidates' / (arm + '.json'), dict(arm=arm, signature=plan['signature'],
        source_sha256=data.digest(__file__), plan_sha256=data.digest(OUT / 'search-plan.json'),
        initial_weight_sha256=plan['initial_weight_sha256'], model_sha256=training['model_sha256'],
        training_complete_sha256=data.digest(OUT / 'training' / (arm + '.json')),
        old_probability_sha256=data.digest(old_path), new_probability_sha256=data.digest(new_path),
        old_early_stop_validation=lp.validation_metrics(old_valid, old_p), **budget, tests_read=False,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_device_name=torch.cuda.get_device_name(0),
        portable_evidence=dict(model_sha256=training['model_sha256'], sampled_tokens=dict(old=256, new=256), maximum_probability_difference=differences)))


def frozen_control(plan, banks, old_valid, new_valid, *, allow_create=False):
    """Original old NPZ is consumed unchanged; only new-val portable p is added."""
    declaration = verify_plan(plan, full=False); old = read(hp.OUT / 'candidates/joint-mlp128.json')
    old_p = read_probability(declaration['static']['old_probability_path'], declaration['static']['old_probability_declared_sha256'], old_valid)
    require(lp.validation_metrics(old_valid, old_p) == old['validation'] and old['validation']['system_threshold'] == CURRENT_THRESHOLD,
        'Original head saved probabilities/threshold do not reproduce exactly.')
    model = hp.OUT / 'models/joint-mlp128.npz'; net, mean, scale, _ = hp.load_model(model)
    m, s = read_scaler(); require(np.array_equal(mean.numpy(), m) and np.array_equal(scale.numpy(), s), 'Original portable embedded scaler differs.')
    path = CACHE / 'old-head-new-validation.npz'; marker = OUT / 'old-head-control.json'
    if marker.exists():
        evidence = read(marker)
        require(evidence['signature'] == plan['signature'] and evidence['model_sha256'] == CURRENT_MODEL_SHA
            and evidence['old_probability_sha256'] == declaration['static']['old_probability_declared_sha256'], 'Frozen control provenance changed.')
        new_p = read_probability(path, evidence['new_probability_sha256'], new_valid)
    else:
        require(allow_create and not path.exists(), 'Missing/unowned frozen-control probability cache; read-only verification cannot recompute it.')
        new_p = hp.predict_raw(model, banks['new'], new_valid.cache_ix.to_numpy())
        hp.atomic_npz(path, p=new_p, sample_ix=new_valid.sample_ix.to_numpy())
        data.atomic_json(marker, dict(signature=plan['signature'], model_sha256=CURRENT_MODEL_SHA,
            old_probability_sha256=declaration['static']['old_probability_declared_sha256'],
            new_probability_sha256=data.digest(path), backend='Frozen portable CPU pointwise head512; no LM inference.', tests_read=False))
    return old_p, new_p


def calibrate(plan, *, allow_create_control=False):
    _, _, banks, parts = dataset(); _, old_valid, new_valid, *_ = parts
    probability = {}; hashes = {}
    for arm in ARMS:
        record, pold, pnew = candidate_record(arm, plan, full=False)
        probability[arm] = (pold, pnew); hashes[arm] = record['model_sha256']
    probability['old-head'] = frozen_control(plan, banks, old_valid, new_valid, allow_create=allow_create_control); hashes['old-head'] = CURRENT_MODEL_SHA
    records = []
    for model in MODELS:
        old_p, new_p = probability[model]
        for objective in OBJECTIVES:
            if objective == 'old-mean': threshold = lp.validation_metrics(old_valid, old_p)['system_threshold']; proof = None
            else: threshold, proof = maximin_threshold(old_valid, new_valid, old_p, new_p)
            result = metrics_at(old_valid, new_valid, old_p, new_p, threshold)
            if objective == 'maximin':
                require(result['min24'] == proof['min24'] and result['old5_mean_f1'] == proof['old5_mean_f1'], 'Selected maximin counts differ from proof.')
            if model == 'old-head' and objective == 'old-mean': require(threshold == CURRENT_THRESHOLD, 'Original threshold was recalibrated.')
            records.append(dict(record=model + '-' + objective, model=model, objective=objective,
                model_sha256=hashes[model], validation=result, full_grid_proof=proof,
                promoted=False, tests_read=False, diagnostic=model != 'new' or objective != 'maximin'))
    return records


def publish(plan):
    records = calibrate(plan, allow_create_control=True); selection = select_records(records)
    flat = []
    for record in records:
        for group, value in dict(record['validation']['groups'], pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=record['record'], group=group,
                **{k: v for k, v in value.items() if not isinstance(v, (dict, list))},
                **{'role_recall_' + r: v for r, v in value['recall'].items()}))
    data.atomic_json(OUT / 'records.json', records); pd.DataFrame(flat).to_csv(OUT / 'validation-metrics.csv', index=False)
    lines = ['Matched long-source augmentation: development only', LIMITATION, '',
        'Only new-maximin may promote; all other5 records are diagnostic. Pooled metrics do not select epochs, thresholds or models.',
        'Record Model/objective threshold min24 old5meanF1']
    lines += [f"{r['record']} {r['validation']['system_threshold']:.12g} {r['validation']['min24']} {r['validation']['old5_mean_f1']:.9f}" for r in records]
    lines += ['', 'Promoted: ' + str(selection['promoted']), 'Reasons: ' + json.dumps(selection['promotion_rejection_reasons'])]
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
    _, _, _, parts = dataset(); old_fit, old_valid, new_valid, fresh, matched, manifest, _ = parts
    counts = dict(old_fit=len(old_fit), old_valid=len(old_valid), new_fit=len(fresh), new_valid=len(new_valid),
        per_epoch=INSTANCES, updates_per_epoch=STEPS, last_batch=554,
        old_fit_documents=int(old_fit.group_id.nunique()), old_validation_documents=int(old_valid.group_id.nunique()),
        new_fit_documents=256, new_validation_documents=96, matched_distinct_documents=132,
        old_validation_profile_counts={p: int(old_valid.profile.eq(p).sum()) for p in lp.PROFILES},
        old_validation_profile_documents={p: int(old_valid[old_valid.profile.eq(p)].group_id.nunique()) for p in lp.PROFILES},
        old_long_single_full_overlap_documents=len(set(old_valid[old_valid.profile.eq('long-single')].group_id) & set(old_valid[old_valid.profile.eq('long-full')].group_id)),
        tails_are_subsets=True, pooled_counts_not_independent_documents=True, limitation=LIMITATION)
    data.atomic_json(OUT / 'source-counts.json', counts)
    verify_plan(plan); selection.update(plan_sha256=data.digest(OUT / 'search-plan.json'), records_sha256=data.digest(OUT / 'records.json'),
        model_path=str(OUT / 'models/new.npz'), model_sha256=next(r['model_sha256'] for r in records if r['record'] == 'new-maximin'),
        system_threshold=next(r['validation']['system_threshold'] for r in records if r['record'] == 'new-maximin'), source_sha256=data.digest(__file__))
    data.atomic_json(OUT / 'selection.json', selection)
    names = ARTIFACTS
    data.atomic_json(OUT / 'complete.json', dict(signature=plan['signature'], source_sha256=data.digest(__file__),
        artifact_sha256={name: data.digest(OUT / name) for name in names},
        candidate_sha256={arm: data.digest(OUT / 'candidates' / (arm + '.json')) for arm in ARMS},
        tests_read=False, primary_changed=False, new_independent_confirmation_required=True))
    data.atomic_json(OUT / 'status.json', dict(stage='validation_complete', promoted=selection['promoted'], primary_changed=False))


def verified_completion():
    plan = read(OUT / 'search-plan.json'); verify_plan(plan); complete = read(OUT / 'complete.json')
    require(complete['signature'] == plan['signature'] and complete['source_sha256'] == data.digest(__file__)
        and complete['tests_read'] is False and complete['primary_changed'] is False
        and complete['new_independent_confirmation_required'] is True, 'Augmentation completion contract changed.')
    require(set(complete['artifact_sha256']) == set(ARTIFACTS), 'Completion artifact coverage changed.')
    for path, value in complete['artifact_sha256'].items(): hp.require_hash(OUT / path, value, 'completion artifact')
    require(set(complete['candidate_sha256']) == set(ARMS), 'Two trained candidates are required.')
    for arm, value in complete['candidate_sha256'].items():
        hp.require_hash(OUT / 'candidates' / (arm + '.json'), value, 'complete candidate'); candidate_record(arm, plan, full=False)
    records = calibrate(plan); require(records == read(OUT / 'records.json'), 'Calibration records do not reproduce from saved probabilities.')
    selected = select_records(records); actual = read(OUT / 'selection.json')
    require(all(actual[k] == v for k, v in selected.items()), 'No-fallback promotion gate changed.')
    return complete


def no_competing_jobs(allow_own=False):
    names = {'systemness_long_augmentation_extract.py', 'systemness_head_scan.py', 'systemness_head_extract.py',
        'systemness_head_probes.py', 'systemness_head_confirm.py', 'systemness_natural_test_confirm.py',
        'systemness_long_extract.py', 'systemness_recover_local_bank.py', 'systemness_long_probes.py',
        'systemness_head_fusion.py', 'systemness_long_confirm.py', 'systemness_attention_confirm.py'}
    if not allow_own: names.add(Path(__file__).name)
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(a.decode(errors='replace')).name in names for a in args if a): active.append(int(entry.name))
    require(not active, 'Competing experiment processes remain active: ' + str(active))


@contextmanager
def run_lock():
    import fcntl
    with (OUT / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another augmentation supervisor holds this directory.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_workers(pending):
    require(len(pending) <= 2 and set(pending) <= set(ARMS), 'Only two readout workers are allowed.')
    jobs = []
    try:
        for gpu, arm in enumerate(pending):
            log = (OUT / 'logs' / (arm + '.log')).open('a')
            try:
                proc = subprocess.Popen([sys.executable, '-u', __file__, '--train', '--worker', arm, '--gpu', str(gpu)],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), ROLE_LONG_AUG_SUPERVISOR_PID=str(os.getpid())),
                    stdout=log, stderr=subprocess.STDOUT)
            except BaseException: log.close(); raise
            jobs.append((arm, proc, log, gpu))
        data.atomic_json(OUT / 'status.json', dict(stage='training', workers=[dict(arm=a, pid=p.pid, gpu=g) for a, p, _, g in jobs]))
        while jobs:
            codes = [(a, p.poll()) for a, p, _, _ in jobs]; failed = [(a, c) for a, c in codes if c not in [None, 0]]
            require(not failed, 'Augmentation worker failed: ' + str(failed))
            if all(c == 0 for _, c in codes): break
            time.sleep(5)
        for arm, proc, _, _ in jobs: require(proc.wait() == 0, arm + ' worker failed.')
    finally:
        errors = []
        for _, proc, _, _ in jobs:
            try:
                if proc.poll() is None: proc.terminate()
            except BaseException as error: errors.append(repr(error))
        for _, proc, log, _ in jobs:
            try:
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
            except BaseException as error: errors.append(repr(error))
            finally: log.close()
        require(not errors, 'Failed to terminate/reap augmentation workers: ' + repr(errors))


def train(restart_incomplete=False):
    no_competing_jobs(); verify_declaration(); signature = source_signature()
    with run_lock():
        if (OUT / 'complete.json').exists(): verified_completion(); return
        require(not (OUT / 'selection.json').exists(), 'Partial frozen selection exists; refusing silent replacement.')
        for name in ['models', 'candidates', 'training', 'logs']: (OUT / name).mkdir(exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        _, _, _, parts = dataset(); manifest = parts[5]; path = OUT / 'search-plan.json'
        if path.exists(): plan = read(path); verify_plan(plan)
        else:
            require(not any((OUT / 'models').iterdir()) and not any((OUT / 'candidates').iterdir())
                and not any((OUT / 'training').iterdir()) and not any(CACHE.iterdir()), 'Undeclared partial probability/model files; refusing reuse.')
            data.atomic_json(OUT / 'draw-manifest.json', manifest)
            plan = dict(protocol_sha256=data.digest(OUT / 'protocol.json'), signature=signature,
                initial_weight_sha256=read(OUT / 'protocol.json')['static']['initial_weight_sha256'], draw_manifest=manifest)
            data.atomic_json(path, plan); verify_plan(plan)
        pending = []
        for arm in ARMS:
            if (OUT / 'candidates' / (arm + '.json')).exists(): candidate_record(arm, plan, full=False)
            else:
                partial = [OUT / 'models' / (arm + '.npz'), OUT / 'training' / (arm + '.json'),
                    CACHE / (arm + '-old-validation.npz'), CACHE / (arm + '-new-validation.npz')]
                require(restart_incomplete or not any(p.exists() for p in partial), 'Incomplete candidate requires explicit --restart-incomplete; no progress is silently trusted.')
                if any(p.exists() for p in partial):
                    data.atomic_json(OUT / ('restart-' + arm + '-' + str(time.time_ns()) + '.json'), dict(arm=arm, from_epoch=1,
                        original_partial_sha256={str(p): data.digest(p) for p in partial if p.exists()}, signature=signature))
                pending.append(arm)
        if pending: run_workers(pending)
        publish(plan)


def preflight():
    """Synthetic CPU/temp-file faults only. No production bank/probability API."""
    from unittest.mock import patch
    import copy
    import inspect
    import py_compile
    import torch
    torch.set_num_threads(2)
    folder = Path(tempfile.mkdtemp(prefix='systemness-long-augmentation-probes-preflight-', dir='/tmp'))
    source_sha = data.digest(__file__); formal = dict(out_exists=OUT.exists(), cache_exists=CACHE.exists())
    checks = []
    def check(name, value=True): require(bool(value), 'CPU fixture failed: ' + name); checks.append(name)
    def reject(name, fn):
        try: fn()
        except (RuntimeError, ValueError, KeyError, AssertionError, FileNotFoundError): checks.append(name)
        else: raise RuntimeError('CPU fault was accepted: ' + name)
    @contextmanager
    def globals_for_fixture(**values):
        before = {k: globals()[k] for k in values}; globals().update(values)
        try: yield
        finally: globals().update(before)
    def forbidden(*a, **kw): raise RuntimeError('Formal model/data/probability access forbidden in CPU preflight.')
    # Test the real132+124 packet recipe cheaply while retaining exact132 docs.
    old = []; new = []
    def append(target, doc, split, count, fresh=False):
        for ri, role in enumerate(data.ROLES):
            for off in data.uniform_indices(1024, count):
                ix = len(target)
                target.append(dict(cache_ix=ix, sample_ix=(6_000_000_000_000 if fresh else 2_000_000_000) + ix,
                    document_id=(6_000_000 if fresh else 0) + doc, group_id=('new-' if fresh else 'old-') + str(doc),
                    role=role, split=split, source_set='long-augmentation' if fresh else 'long-full',
                    profile='long-full', wrapper='native', prompt_ix=doc * 5 + ri, message_ix=0,
                    token_ix=100 + int(off), body_offset=int(off)))
    # Smaller3+2 doc fixture exercises pure sampler without invoking old production partitions.
    for doc in range(4): append(old, doc, 'fit' if doc < 3 else 'validation', 4)
    for doc in range(7): append(new, doc, 'fit' if doc < 5 else 'validation', 4 if doc < 5 else 2, True)
    old, new = pd.DataFrame(old), pd.DataFrame(new)
    expected = dict(old_pool=3, new_fit=5, new_valid=2, samples_fit=4, samples_valid=2)
    torch_before = torch.get_rng_state().clone(); np_before = np.random.get_state()
    parts = augmentation_layout(old, new, expected=expected); fresh, matched, manifest = parts[3:6]
    check('Fixed two-permutation packet draw covers3pooldocs exactly5slots with1or2multiplicity',
        len(manifest['slots']) == 5 and set(manifest['matched_document_multiplicities'].values()) == {1, 2}
        and manifest['distinct_matched_documents'] == 3 and len(fresh) == len(matched) == 100)
    check('Sampler preserves torch and global NumPy RNG', torch.equal(torch_before, torch.get_rng_state())
        and all(np.array_equal(a, b) for a, b in zip(np_before, np.random.get_state())))
    same = augmentation_layout(old, new, expected=expected)
    check('Fixed manifest and indices repeat exactly, never per-epoch redraw', same[5] == manifest and np.array_equal(same[4], matched))
    check('Matched never samples validation and labels/body-offset order equal new', set(matched) <= set(old[old.split.eq('fit')].cache_ix)
        and np.array_equal(probes.labels(old.iloc[matched]), probes.labels(new.iloc[fresh]))
        and np.array_equal(old.iloc[matched].body_offset.to_numpy(), new.iloc[fresh].body_offset.to_numpy()))
    broken = new.copy(); broken.loc[0, 'split'] = 'test'; reject('Reject new test split', lambda: augmentation_layout(old, broken, expected=expected))
    broken = new.copy(); broken.loc[0, 'sample_ix'] = old.sample_ix.iloc[0]; reject('Reject sample namespace/old collision', lambda: augmentation_layout(old, broken, expected=expected))
    broken = new.copy(); broken.loc[0, 'body_offset'] = 3; reject('Reject changed body-position histogram', lambda: augmentation_layout(old, broken, expected=expected))
    broken = new.copy(); broken.loc[0, 'role'] = 'user'; reject('Reject role imbalance/duplicate packet roles', lambda: augmentation_layout(old, broken, expected=expected))
    # The exact production doc counts with tiny four-position packets.
    pool = []; extra = []
    for doc in range(132): append(pool, doc, 'fit', 4)
    for doc in range(257): append(extra, doc, 'fit' if doc < 256 else 'validation', 4 if doc < 256 else 2, True)
    _, pool_ix, _, _ = packets(pd.DataFrame(pool), 132, 4)
    rng = np.random.RandomState(DRAW_SEED); draw = np.r_[rng.permutation(132), rng.permutation(132)[:124]]
    multiplicities = np.bincount(draw, minlength=132)
    check('Real132+124 recipe gives256slots,124twice8once', len(draw) == 256 and np.count_nonzero(multiplicities == 2) == 124 and np.count_nonzero(multiplicities == 1) == 8)
    # Existing authoritative partition helper: exact4160 sparse replacement.
    entries = []
    def old_entry(doc, profile, split, n, augmented=False):
        for j in range(n):
            ix = len(entries); entries.append(dict(cache_ix=ix, sample_ix=(2_000_000_000 if augmented else 0) + ix,
                document_id=doc, group_id='c4-' + str(doc), profile=profile, split=split, wrapper='native',
                source_set='long-full' if augmented else 'main', role=data.ROLES[j % 5], prompt_ix=ix, message_ix=0, token_ix=j))
    for doc in range(32): old_entry(doc if doc < 26 else doc + 200, 'long-single', 'fit', 160)
    for i, profile in enumerate(lp.ORIGINAL_PROFILES): old_entry(1000 + i, profile, 'validation', 5)
    base = len(entries)
    for doc in range(132): old_entry(doc, 'long-full', 'fit', 5, True)
    for doc in range(1100, 1131): old_entry(doc, 'long-full', 'validation', 5, True)
    miniature = pd.DataFrame(entries)
    fit, valid, info = hp.partitions(miniature, dict(total=len(miniature), base=base, fit=1620, valid=175,
        old_fit=5120, long_fit=660, long_valid=155, long_fit_documents=132, long_valid_documents=31,
        removed=4160, removed_documents=26, fit_documents=138))
    check('Old partitions retain exact4160 replacement and disjoint validation', len(fit) == 1620 and info['removed_sparse_rows'] == 4160
        and not set(fit.cache_ix) & set(valid.cache_ix))
    # Flat full order, same across arms, no separate partial batch for the extra pass.
    ga = torch.Generator().manual_seed(data.SEED); gb = torch.Generator().manual_seed(data.SEED)
    order = lp.epoch_order(INSTANCES, ga, 'cpu'); other = lp.epoch_order(INSTANCES, gb, 'cpu')
    lengths = [len(order[i:i + BATCH]) for i in range(0, len(order), BATCH)]
    check('Fullconcat905770 flat permutation matches botharms222steps last554', torch.equal(order, other)
        and torch.unique(order).numel() == INSTANCES and len(lengths) == STEPS and lengths[-1] == 554)
    check('CE generator advances only by epoch randperm', not torch.equal(order, lp.epoch_order(INSTANCES, ga, 'cpu')))
    history = [dict(epoch=i, instances=INSTANCES, steps=STEPS, cumulative_steps=i * STEPS, loss=1., old5_mean_f1=.8) for i in range(1, 11)]
    budget = check_history(history, 1)
    check('Original min10/patience8 stop records best and actual222step budgets', budget['best_steps'] == 222 and budget['actual_steps'] == 2220)
    reject('Reject continued training after original early-stop trigger', lambda: check_history(history + [history[-1] | dict(epoch=11, cumulative_steps=2442)], 1))
    reject('Reject fabricated best epoch', lambda: check_history(history, 2))
    reject('Reject early stopping before min10', lambda: check_history(history[:8], 1))
    # Own full512 portable model, unchanged frozen scaler values and same initialization.
    initial_rng = torch.get_rng_state().clone(); init1 = expected_initialization(); init2 = expected_initialization()
    check('Same original initialization SHA without perturbing RNG', init1 == init2 and torch.equal(initial_rng, torch.get_rng_state()))
    torch.manual_seed(data.SEED); net = hp.make_net('mlp', hidden=128); check('Worker initialization matches declared SHA', init_hash(net) == init1)
    rng = np.random.default_rng(9); raw = rng.normal(size=(37, DIMENSIONS)).astype(np.float16)
    mean = rng.normal(size=DIMENSIONS).astype(np.float32); scale = rng.uniform(.3, 2, size=DIMENSIONS).astype(np.float32)
    portable = folder / 'portable.npz'; metadata = dict(kind='mlp', hidden=128, dimensions=DIMENSIONS, heads=8, head_dim=64, roles=data.ROLES)
    hp.save_model(portable, net, mean, scale, metadata)
    p = hp.predict_raw(portable, raw, np.arange(len(raw)))
    with torch.inference_mode(): direct = net.eval()(torch.from_numpy((raw.astype(np.float32) - mean) / scale)).exp().numpy()
    difference = float(np.abs(p - direct).max()); check('Frozen512 scaler and MLP128 portable probability parity', difference < 1e-7)
    # Calibration: synthetic old5 plus newfull, adjacentF32 boundaries and missing natural tool support.
    oldv = []; newv = []
    for pi, profile in enumerate(lp.PROFILES):
        for ri, role in enumerate(data.ROLES):
            if profile == 'natural' and role == 'tool': continue
            for off in [0, 341, 682, 1023]: oldv.append(dict(profile=profile, role=role, prompt_ix=pi * 5 + ri, message_ix=0,
                token_ix=100 + off, document_id=pi, split='validation', wrapper='native'))
    for ri, role in enumerate(data.ROLES):
        for off in [0, 341, 682, 1023]: newv.append(dict(profile='long-full', role=role, prompt_ix=100 + ri, message_ix=0,
            token_ix=100 + off, body_offset=off, document_id=6_000_000, split='validation', wrapper='native'))
    oldv, newv = pd.DataFrame(oldv), pd.DataFrame(newv)
    def synthetic_p(frame):
        value = np.full((len(frame), 5), .02, np.float32); y = probes.labels(frame)
        value[np.arange(len(frame)), y] = .92
        for i in np.flatnonzero(y == 0)[::3]:
            ps = np.nextafter(np.float32(.5), np.float32(1)); value[i, 0] = ps; value[i, 1:] = (1 - ps) / 4
        return value
    po, pn = synthetic_p(oldv), synthetic_p(newv)
    tau, proof, arrays = maximin_threshold(oldv, newv, po, pn, return_curve=True)
    grid, curve, means, minimum, feasible = arrays
    # Independently brute-force the chosen objective from complete saved probabilities.
    direct_rows = []
    for threshold in grid:
        r = metrics_at(oldv, newv, po, pn, float(threshold))
        if r['min24'] is not None and all(r['groups'][g]['system_false_positive_rate'] <= .05 for g in FEASIBLE_GROUPS):
            direct_rows.append((r['min24'], r['old5_mean_f1'], -float(threshold)))
    brute = max(direct_rows)
    check('All8groups24F32 statistics and globallexicographic argmax exact', proof['primary_statistics'] == len(grid) * 24
        and tau == -brute[2] and proof['min24'] == brute[0] and proof['old5_mean_f1'] == brute[1])
    result = metrics_at(oldv, newv, po, pn, tau)
    check('Old/new remain separate; pooled descriptive only; natural tool recall N/A', set(result['groups']) == set(GROUPS)
        and result['groups']['old-natural']['recall']['tool'] is None and 'pooled_descriptive_only' in result)
    check('Oldmean remains unchanged exact helper', lp.validation_metrics(oldv, po)['system_threshold'] == lp.validation_metrics(oldv.copy(), po.copy())['system_threshold'])
    corrupt = po.copy(); corrupt[0, 0] = np.nan
    reject('Reject nonfinite calibration probabilities', lambda: maximin_threshold(oldv, newv, corrupt, pn))
    reject('Reject probability dtype drift', lambda: maximin_threshold(oldv, newv, po.astype(np.float64), pn))
    def guard_result(value):
        return dict(min24=value, old5_mean_f1=value, groups={g: dict(system_f1=value, accuracy=value, system_false_positive_rate=.01) for g in GROUPS})
    records = [dict(record=m + '-' + o, validation=guard_result(.9)) for m in MODELS for o in OBJECTIVES]
    for r in records:
        if r['record'] == 'new-maximin': r['validation'] = guard_result(.93)
        elif r['record'] == 'matched-maximin': r['validation'] = guard_result(.92)
    check('Onlynewmaximin can pass all3controls and strictmin24', select_records(records)['promoted'])
    bad = copy.deepcopy(records); next(r for r in bad if r['record'] == 'new-maximin')['validation']['min24'] = .92
    check('Strict min24 ties reject with no fallback', not select_records(bad)['promoted'] and select_records(bad)['candidate'] == 'new-maximin')
    bad = copy.deepcopy(records); candidate = next(r for r in bad if r['record'] == 'new-maximin')['validation']; candidate['old5_mean_f1'] = .91
    check('Fixed .005 old5 mean F1 floor rejects', not select_records(bad)['promoted'])
    for label, change in [('originalF1', lambda r: r['groups']['old-single'].update(system_f1=.87)),
        ('fiveaccuracy', lambda r: r['groups']['old-multi'].update(accuracy=.88)),
        ('newtail', lambda r: r['groups']['new-long-full-tail'].update(system_f1=.89)),
        ('newfullFPR', lambda r: r['groups']['new-long-full'].update(system_false_positive_rate=.050001))]:
        bad = copy.deepcopy(records); change(next(r for r in bad if r['record'] == 'new-maximin')['validation'])
        check('Reject ' + label + ' regression despite min24 improvement', not select_records(bad)['promoted'])
    # Hash, namespace and declaration-before-probability faults in /tmp only.
    sample_rows = pd.DataFrame(dict(sample_ix=np.arange(len(p)) + 6_000_000_000_000))
    probability_path = folder / 'probability.npz'; hp.atomic_npz(probability_path, p=p, sample_ix=sample_rows.sample_ix.to_numpy())
    check('Saved probability SHA and sample order roundtrip', np.array_equal(read_probability(probability_path, data.digest(probability_path), sample_rows), p))
    reject('Reject missing saved probability', lambda: read_probability(folder / 'missing.npz', '0' * 64, sample_rows))
    wrong = folder / 'wrong-sample.npz'; hp.atomic_npz(wrong, p=p, sample_ix=sample_rows.sample_ix.to_numpy() + 1)
    reject('Reject probability crossnamespace despite correct file SHA', lambda: read_probability(wrong, data.digest(wrong), sample_rows))
    with globals_for_fixture(OUT=folder / 'declared', CACHE=folder / 'declared-cache'):
        reject('No training before protocol declaration', verify_declaration)
        with patch(__name__ + '.static_snapshot', return_value=dict(synthetic=True)), patch('numpy.load', side_effect=forbidden):
            declare(); verify_declaration(); check('Declaration precedes and never opens any probability NPZ')
            bad = read(OUT / 'protocol.json'); bad['protocol']['neural']['seed'] += 1; data.atomic_json(OUT / 'protocol.json', bad)
            reject('Reject training protocol/source freeze tamper', verify_declaration)
    source = inspect.getsource(worker)
    check('Newval prediction is after fsynced training/best-model evidence', source.index("data.atomic_json(OUT / 'training'") < source.index('new_p = hp.predict_raw'))
    check('Production old partition uses original authoritative hp.partitions', 'hp.partitions(old_rows)' in inspect.getsource(augmentation_layout))
    py_compile.compile(__file__, cfile=str(folder / 'trainer.pyc'), doraise=True)
    require(data.digest(__file__) == source_sha and formal == dict(out_exists=OUT.exists(), cache_exists=CACHE.exists()), 'Preflight changed source or formal outputs.')
    report = dict(stage='synthetic_CPU_preflight', source_sha256=source_sha, checks=checks, checks_passed=len(checks),
        initial_weight_sha256=init1, portable_max_probability_difference=difference, grid_proof=proof,
        old_sparse_replacement_rows=4160, real_epoch_instances=INSTANCES, real_epoch_steps=STEPS, last_batch_instances=554,
        fixture_directory=str(folder), formal_outputs_changed=False, new_source_data_read=False,
        actual_validation_probabilities_read=False, test_inputs_probabilities_or_metrics_read=False, gpu_used=False,
        declaration_prior_probability_fixture=True)
    data.atomic_json(folder / 'report.json', report)
    print(json.dumps(dict(report=str(folder / 'report.json'), checks_passed=len(checks), source_sha256=source_sha)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(); action.add_argument('--declare', action='store_true')
    action.add_argument('--train', action='store_true'); action.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', choices=ARMS); parser.add_argument('--gpu', type=int)
    parser.add_argument('--restart-incomplete', action='store_true')
    args = parser.parse_args()
    require(args.worker is None or (args.train and args.gpu in [0, 1]), 'Workers require --train and assigned --gpu.')
    if args.preflight: preflight()
    elif args.declare: declare()
    elif args.train:
        if args.worker: worker(args.worker, args.gpu)
        else: train(args.restart_incomplete)
    else: parser.print_help()

if __name__ == '__main__': main()
