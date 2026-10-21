"""Independent six-record development audit; synthetic until explicit --actual.

Never calls production calibration/metrics/selection/verified_completion during
actual inspection. No tests, LM/GPU, training, confidence intervals or writes to
formal output directories. Complete and terminal-process gates precede results.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0, str(ROOT / 'experiments/role-analysis'))
import systemness_data as data
import systemness_head_probes as hp
import systemness_long_augmentation_probes as production

SOURCE_SHA = '220d7d24a8a640198351ce925093fe2ba135af42ab9ccbb797a7d5b3431699e5'
PROTOCOL_SHA = 'de02baf0b846d2af4c83f6c469738932e85d77dc85047011d31cb2433a0647d0'
CURRENT_SHA = '27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
CURRENT_TAU = .620969696969697
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
PROFILES = ['single', 'multi', 'natural', 'long-single', 'long-full']
ORIGINAL = PROFILES[:4]
GROUPS = ['old-' + name for name in PROFILES] + ['old-long-full-tail', 'new-long-full', 'new-long-full-tail']
FEASIBLE = ['old-' + name for name in PROFILES] + ['new-long-full']
ARMS, MODELS, OBJECTIVES = ['new', 'matched'], ['old-head', 'new', 'matched'], ['old-mean', 'maximin']
ARTIFACTS = ['protocol.json', 'search-plan.json', 'draw-manifest.json', 'records.json', 'validation-metrics.csv',
    'report.txt', 'source-counts.json', 'selection.json', 'old-head-control.json']
OUT = data.runtime.OUT / 'systemness-long-augmentation-probes'
CACHE = data.CACHE / 'long-augmentation-probes'
NEW = Path('/dev/shm/role-analysis-long-augmentation-heads')
OLD = Path('/dev/shm/role-analysis-top8-heads')
MAX_DIFFERENCE = 0.


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def labels(frame):
    value = frame.role.map(dict(zip(ROLES, range(5)))).to_numpy()
    require(pd.notna(value).all(), 'Unknown formal role.'); return value.astype(np.int64)
def divide(a, b): return float(a / b) if b else None


def close(actual, expected, label='', *, tolerance=1e-7):
    global MAX_DIFFERENCE
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Mapping keys differ: ' + label)
        for key in expected: close(actual[key], expected[key], label + '/' + str(key), tolerance=tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), 'List dimensions differ: ' + label)
        for index, value in enumerate(expected): close(actual[index], value, label + '/' + str(index), tolerance=tolerance)
    elif expected is None:
        require(actual is None, 'Undefined value was replaced: ' + label)
    elif isinstance(expected, (bool, str, int)):
        require(actual == expected, 'Discrete value differs: ' + label)
    else:
        require(actual is not None and np.isfinite(float(actual)) and np.isfinite(float(expected)), 'Nonfinite comparison: ' + label)
        difference = abs(float(actual) - float(expected))
        require(difference <= (2e-6 if label.endswith('/nll') else tolerance), 'Numeric value differs: ' + label + '/' + str(difference))
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, difference)


def validate_p(p, count):
    require(p.shape == (count, 5) and p.dtype == np.float32 and np.isfinite(p).all()
        and (p >= 0).all() and (p <= 1.00001).all() and np.allclose(p.sum(1), 1., atol=2e-5, rtol=0), 'Invalid F32 five-class probability cache.')


def rankings(score, truth):
    """Independent tied-score AUROC/AP from grouped descending thresholds."""
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives: return None, None
    order = np.argsort(-score, kind='stable'); sorted_p, y = score[order], truth[order]
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(y) - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]; fp = ends + 1 - tp
    pos_group, neg_group = np.diff(np.r_[0, tp]), np.diff(np.r_[0, fp])
    ap = float(np.sum((pos_group / positives) * (tp / (ends + 1))))
    auc = float(np.sum(pos_group * (negatives - fp + .5 * neg_group)) / (positives * negatives)) if negatives else None
    return auc, ap


def metrics(p, y, tau):
    validate_p(p, len(y)); require(len(y) and ((y >= 0) & (y < 5)).all() and np.isfinite(tau), 'Invalid labels/threshold.')
    pred = p.argmax(1); matrix = np.zeros((5, 5), np.int64)
    np.add.at(matrix, (y, pred), 1)
    support, predicted, diagonal = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    class_f1 = [2 * int(diagonal[i]) / int(support[i] + predicted[i]) if support[i] + predicted[i] else 0. for i in range(5)]
    positive, detected = y == 0, p[:, 0] >= float(tau)
    tp = int(np.count_nonzero(positive & detected)); fn = int(np.count_nonzero(positive & ~detected))
    fp = int(np.count_nonzero(~positive & detected)); tn = int(np.count_nonzero(~positive & ~detected))
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp); auc, ap = rankings(p[:, 0], positive)
    return dict(tokens=len(y), accuracy=float(np.count_nonzero(pred == y) / len(y)),
        macro_f1=float(np.mean([class_f1[i] for i in range(5) if support[i] > 0])),
        nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        recall={role: divide(int(diagonal[i]), int(support[i])) for i, role in enumerate(ROLES)},
        confusion=matrix.tolist(), system_threshold=float(tau), system_accuracy=float((tp + tn) / len(y)),
        system_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
        system_precision=divide(tp, tp + fp), system_recall=recall, system_f1=divide(2 * tp, 2 * tp + fp + fn),
        system_false_positive_rate=divide(fp, fp + tn), system_auroc=auc, system_average_precision=ap,
        system_detection_by_role={role: float(detected[y == i].mean()) if np.any(y == i) else None for i, role in enumerate(ROLES)},
        system_confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn))


def group_indices(old_rows, new_rows):
    require(np.array_equal(old_rows.index, np.arange(len(old_rows))) and np.array_equal(new_rows.index, np.arange(len(new_rows))), 'Validation row indices must reset within banks.')
    groups = {f'old-{name}': ('old', np.flatnonzero(old_rows.profile.eq(name))) for name in PROFILES}
    for bank, frame in [('old', old_rows), ('new', new_rows)]:
        full = frame[frame.profile.eq('long-full')]
        offsets = full.body_offset.to_numpy() if 'body_offset' in full else (full.token_ix - full.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')).to_numpy()
        groups[bank + '-long-full-tail'] = (bank, full.index.to_numpy()[offsets >= 512])
        if bank == 'new': groups['new-long-full'] = (bank, full.index.to_numpy())
    require(set(groups) == set(GROUPS) and all(len(ix) for _, ix in groups.values()), 'Missing group support.')
    return groups


def evaluate(old_rows, new_rows, po, pn, tau):
    groups = group_indices(old_rows, new_rows); ps, ys = dict(old=po, new=pn), dict(old=labels(old_rows), new=labels(new_rows))
    values = {name: metrics(ps[bank][ix], ys[bank][ix], tau) for name, (bank, ix) in groups.items()}
    primary = [values[name][key] for name in GROUPS for key in ['system_precision', 'system_recall', 'system_balanced_accuracy']]
    return dict(system_threshold=float(tau), groups=values,
        old5_mean_f1=float(np.mean([values['old-' + name]['system_f1'] for name in PROFILES])),
        min24=min(primary) if all(value is not None and np.isfinite(value) for value in primary) else None,
        pooled_descriptive_only=metrics(np.concatenate([po, pn]), np.r_[ys['old'], ys['new']], tau))


def old_validation_record(value, old_rows):
    profiles = {name: value['groups']['old-' + name] for name in PROFILES}
    tail = dict(value['groups']['old-long-full-tail'])
    full = old_rows[old_rows.profile.eq('long-full')]
    offset = full.token_ix - full.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')
    tail.update(documents=int(full.loc[offset.ge(512), 'document_id'].nunique()), body_start=512, body_end=1023)
    return dict(system_threshold=value['system_threshold'], mean_system_f1=value['old5_mean_f1'],
        original_mean_system_f1=float(np.mean([profiles[name]['system_f1'] for name in ORIGINAL])),
        original_mean_accuracy=float(np.mean([profiles[name]['accuracy'] for name in ORIGINAL])), profiles=profiles, long_full_tail=tail)


def legacy_mean(old_rows, p):
    """Preserve original F64 sorted-search grid exactly; do not fix legacy."""
    grid = np.unique(np.r_[np.linspace(.001, .999, 199), np.quantile(p[:, 0], np.linspace(0, 1, 301)), 1.000001])
    scores, fprs = [], []; y = labels(old_rows)
    for profile in PROFILES:
        ix = np.flatnonzero(old_rows.profile.eq(profile)); truth = y[ix] == 0
        pos, neg = np.sort(p[ix, 0][truth]), np.sort(p[ix, 0][~truth])
        tp, fp = len(pos) - np.searchsorted(pos, grid, side='left'), len(neg) - np.searchsorted(neg, grid, side='left')
        scores.append(2 * tp / np.maximum(2 * tp + fp + len(pos) - tp, 1)); fprs.append(fp / max(len(neg), 1))
    ranked = np.mean(scores, axis=0); ranked[np.max(fprs, axis=0) > .05] = -1
    return float(grid[int(np.argmax(ranked))])


def direct_grid(old_rows, new_rows, po, pn):
    """Every original F64 tau is directly consumed by F32 probability arrays."""
    groups = group_indices(old_rows, new_rows); ps, ys = dict(old=po, new=pn), dict(old=labels(old_rows), new=labels(new_rows))
    grid = np.unique(np.r_[np.linspace(.001, .999, 199), np.quantile(np.r_[po[:, 0], pn[:, 0]], np.linspace(0, 1, 301)), 1.000001])
    curves = {}; stat_keys = ['P', 'R', 'BAcc', 'F1', 'FPR']
    for name, (bank, ix) in groups.items():
        truth = ys[bank][ix] == 0; score = ps[bank][ix, 0]; positive, negative = int(truth.sum()), int((~truth).sum())
        require(positive > 0 and negative > 0, 'A grid group lacks positive/negative support.')
        values = []
        for tau in grid:
            detected = score >= float(tau); tp = int(np.count_nonzero(truth & detected)); fp = int(np.count_nonzero(~truth & detected))
            fn, tn = positive - tp, negative - fp
            values.append([tp / (tp + fp) if tp + fp else np.nan, tp / positive,
                (tp / positive + tn / negative) / 2, 2 * tp / (2 * tp + fp + fn), fp / negative])
        curves[name] = np.asarray(values)
    objective = np.min([curves[name][:, index] for name in GROUPS for index in [0, 1, 2]], axis=0)
    objective[~np.isfinite(objective)] = -np.inf
    mean = np.mean([curves['old-' + name][:, 3] for name in PROFILES], axis=0)
    feasible = np.max([curves[name][:, 4] for name in FEASIBLE], axis=0) <= .05
    candidates = np.flatnonzero(feasible & np.isfinite(objective)); require(len(candidates), 'No defined feasible maximin threshold.')
    best = sorted(candidates, key=lambda index: (-float(objective[index]), -float(mean[index]), float(grid[index])))[0]
    proof = dict(grid_thresholds=len(grid), groups=8, primary_statistics=len(grid) * 24,
        additional_f1_fpr_statistics=len(grid) * 16, maximum_direct_difference=0.,
        all_grid_undefined_and_count_semantics_equal=True, actual_F32_comparison=True,
        all_grid_feasibility_equal=True, global_argmax_equal=True,
        min24=float(objective[best]), old5_mean_f1=float(mean[best]), feasible_thresholds=int(feasible.sum()))
    annex = dict(grid_sha256=array_sha(grid), effective_F32_grid_sha256=array_sha(grid.astype(np.float32)),
        direct_curves_sha256={name: array_sha(curves[name]) for name in GROUPS}, stat_order=stat_keys,
        undefined_precision_thresholds={name: int(np.isnan(curves[name][:, 0]).sum()) for name in GROUPS})
    return float(grid[best]), proof, annex


def guard(result, control):
    tolerance = 1e-12
    f1_losses = {p: control['groups']['old-' + p]['system_f1'] - result['groups']['old-' + p]['system_f1'] for p in ORIGINAL}
    accuracy_losses = {p: control['groups']['old-' + p]['accuracy'] - result['groups']['old-' + p]['accuracy'] for p in ORIGINAL}
    full = ['old-long-full', 'old-long-full-tail', 'new-long-full', 'new-long-full-tail']
    gains = {p: result['groups'][p]['system_f1'] - control['groups'][p]['system_f1'] for p in full}
    mean_loss = control['old5_mean_f1'] - result['old5_mean_f1']; reasons = []
    if not all(np.isfinite(result['groups'][g][key]) for g in GROUPS for key in ['system_f1', 'accuracy', 'system_false_positive_rate']) or not np.isfinite(result['old5_mean_f1']): reasons.append('nonfinite_validation_metric')
    if max(f1_losses.values()) > .02 + tolerance: reasons.append('original4_system_f1_loss')
    if max(accuracy_losses.values()) > .01 + tolerance: reasons.append('original4_five_accuracy_loss')
    if min(gains.values()) < -tolerance: reasons.append('old_or_new_full_tail_f1_regression')
    if mean_loss > .005 + tolerance: reasons.append('old5_mean_f1_loss')
    if max(result['groups'][g]['system_false_positive_rate'] for g in FEASIBLE) > .05 + tolerance: reasons.append('profile_fpr_above_limit')
    return dict(eligible=not reasons, rejection_reasons=reasons, original4_f1_losses=f1_losses,
        original4_accuracy_losses=accuracy_losses, full_tail_f1_gains=gains, old5_mean_f1_loss=mean_loss)


def selection(records):
    require(len(records) == 6 and {entry['record'] for entry in records} == {model + '-' + objective for model in MODELS for objective in OBJECTIVES}, 'All six records are required.')
    entries = {entry['record']: entry for entry in records}; candidate = entries['new-maximin']['validation']
    guards = {name: guard(candidate, entries[name]['validation']) for name in ['matched-maximin', 'old-head-maximin', 'old-head-old-mean']}
    gains, reasons = {}, [name + ':' + reason for name, value in guards.items() for reason in value['rejection_reasons']]
    for name in ['matched-maximin', 'old-head-maximin']:
        left, right = candidate['min24'], entries[name]['validation']['min24']; gain = left - right if left is not None and right is not None else None
        gains[name] = gain
        if gain is None or not np.isfinite(gain) or gain <= 1e-12: reasons.append(name + ':min24_not_strictly_improved')
    return dict(candidate='new-maximin', promoted=not reasons, no_fallback=True, guards=guards, min24_gains=gains,
        promotion_rejection_reasons=reasons, primary_changed=False, new_independent_confirmation_required=True, tests_read=False)


def history_audit(history, best_epoch):
    require(1 <= best_epoch <= len(history) <= 45, 'Invalid epoch evidence.')
    best, selected_epoch = -1., -1
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and row['steps'] == 222 and row['instances'] == 905770
            and row['cumulative_steps'] == epoch * 222 and np.isfinite(row['loss']) and np.isfinite(row['old5_mean_f1']), 'Training epoch budget/evidence mismatch.')
        if row['old5_mean_f1'] > best + 1e-5: best, selected_epoch = row['old5_mean_f1'], epoch
        if epoch < len(history): require(not (epoch >= 10 and epoch - selected_epoch >= 8), 'Training continued after original early-stop trigger.')
    require(selected_epoch == best_epoch and (len(history) == 45 or (len(history) >= 10 and len(history) - selected_epoch >= 8)), 'Wrong best/terminal epoch.')
    return dict(best_epoch=best_epoch, best_steps=best_epoch * 222, epochs=len(history), actual_steps=len(history) * 222,
        actual_instances=len(history) * 905770, instances_per_epoch=905770, steps_per_epoch=222, last_batch_instances=554)


def processes():
    names = {'systemness_long_augmentation_probes.py', 'systemness_long_augmentation_extract.py'}; active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(value.decode(errors='replace')).name in names for value in args if value): active.append(int(entry.name))
    return active


def terminal_gate():
    require(not processes(), 'Trainer/extractor is live: no partial result or probability reads.')
    require((OUT / 'complete.json').is_file() and (OUT / 'status.json').is_file(), 'Formal completion/status is missing.')
    require(read(OUT / 'status.json')['stage'] == 'validation_complete', 'Formal trainer is not complete.')


def read_probability(path, expected_sha, rows):
    require(digest(path) == expected_sha, 'Probability byte SHA mismatch.')
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == {'p', 'sample_ix'} and np.array_equal(values['sample_ix'], rows.sample_ix), 'Probability sample order/namespace mismatch.')
        p = values['p'].copy()
    validate_p(p, len(rows)); return p


def source_partitions(old_rows, new_rows):
    require(len(old_rows) == 755282 and len(new_rows) == 389120
        and np.array_equal(old_rows.cache_ix, np.arange(len(old_rows))) and np.array_equal(new_rows.cache_ix, np.arange(len(new_rows)))
        and old_rows.sample_ix.is_unique and new_rows.sample_ix.is_unique and set(old_rows.split) == set(new_rows.split) == {'fit', 'validation'}
        and set(old_rows.wrapper) == set(new_rows.wrapper) == {'native'}, 'Bank row ownership changed.')
    raw_fit = old_rows[old_rows.split.eq('fit')]; dense_ids = set(raw_fit.loc[raw_fit.profile.eq('long-full'), 'document_id'])
    removed = raw_fit[raw_fit.profile.eq('long-single') & raw_fit.document_id.isin(dense_ids)]
    fit = raw_fit[~raw_fit.cache_ix.isin(removed.cache_ix)]
    valid = old_rows[old_rows.split.eq('validation')].copy().reset_index(drop=True)
    nf, nv = new_rows[new_rows.split.eq('fit')], new_rows[new_rows.split.eq('validation')].copy().reset_index(drop=True)
    require(len(fit) == 578090 and len(valid) == 173032 and len(removed) == 4160 and removed.document_id.nunique() == 26
        and len(dense_ids) == 132 and len(nf) == 327680 and len(nv) == 61440
        and not set(fit.group_id) & set(valid.group_id) and not set(nf.group_id) & set(nv.group_id)
        and not set(new_rows.group_id) & set(old_rows.group_id) and not set(new_rows.sample_ix) & set(old_rows.sample_ix), 'Source partition/removal/leakage mismatch.')
    require(set(new_rows.source_set) == {'long-augmentation'} and set(new_rows.profile) == {'long-full'}
        and new_rows.sample_ix.min() >= 6000000000000 and new_rows.document_id.min() >= 6000000,
        'New source namespace changed.')
    partition = dict(fit_rows=len(fit), validation_rows=len(valid), removed_sparse_rows=len(removed), removed_sparse_documents=26,
        fit_cache_ix_sha256=array_sha(fit.cache_ix.to_numpy()), fit_sample_ix_sha256=array_sha(fit.sample_ix.to_numpy()), fit_labels_sha256=array_sha(labels(fit)),
        validation_cache_ix_sha256=array_sha(valid.cache_ix.to_numpy()), validation_sample_ix_sha256=array_sha(valid.sample_ix.to_numpy()), validation_labels_sha256=array_sha(labels(valid)))
    return fit, valid, nf, nv, partition


def packets(frame, documents, samples):
    docs = sorted(int(value) for value in frame.document_id.unique()); require(len(docs) == documents, 'Packet source count mismatch.')
    grid = np.linspace(0, 1023, samples, dtype=int); result = []
    lookup = {(int(doc), role): part.sort_values('token_ix') for (doc, role), part in frame.groupby(['document_id', 'role'], sort=False)}
    for doc in docs:
        packet = []
        for role in ROLES:
            part = lookup[(doc, role)]; offsets = part.body_offset.to_numpy() if 'body_offset' in part else part.token_ix.to_numpy() - int(part.token_ix.min())
            require(len(part) == samples and np.array_equal(offsets, grid), 'Packet uniform body offsets mismatch.')
            packet.extend(int(ix) for ix in part.cache_ix)
        result.append(packet)
    return docs, np.asarray(result, np.int64)


def draw_manifest(old_rows, nf):
    pool = old_rows[old_rows.split.eq('fit') & old_rows.profile.eq('long-full') & old_rows.source_set.eq('long-full')]
    old_docs, old_packets = packets(pool, 132, 256); new_docs, new_packets = packets(nf, 256, 256)
    rng = np.random.RandomState(20517153); first, second = rng.permutation(132), rng.permutation(132)
    draw = np.r_[first, second[:124]]; matched, fresh = old_packets[draw].reshape(-1), new_packets.reshape(-1)
    manifest = dict(seed=20517153, draw_once=True, pool_document_ids=old_docs,
        first_permutation=[old_docs[i] for i in first], second_permutation=[old_docs[i] for i in second],
        slots=[dict(extra_docslot=index, old_document_id=old_docs[int(value)], new_document_id=new_docs[index]) for index, value in enumerate(draw)],
        matched_cache_ix_sha256=array_sha(matched), new_cache_ix_sha256=array_sha(fresh),
        extra_labels_sha256=array_sha(np.tile(np.repeat(np.arange(5), 256), 256)),
        extra_body_offsets_sha256=array_sha(np.tile(np.tile(np.linspace(0, 1023, 256, dtype=int), 5), 256)),
        extra_rows=len(fresh), distinct_matched_documents=len(set(draw)),
        matched_document_multiplicities={str(old_docs[i]): int(np.count_nonzero(draw == i)) for i in range(132)},
        no_validation_rows_sampled=True, torch_rng_consumed=False)
    require(Counter(np.bincount(draw, minlength=132)) == {1: 8, 2: 124}, 'Matched multiplicities are wrong.')
    return manifest, fresh, matched


def initialization_sha():
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261021); net = hp.make_net('mlp', hidden=128)
        values = [dict(name=name, shape=list(value.shape), dtype=str(value.numpy().dtype), sha256=array_sha(value.numpy()))
            for name, value in net.state_dict().items()]
    return canonical(values)


def model_arrays(path, mean, scale, expected_metadata=None):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata'])); require(meta['kind'] == 'mlp' and meta['hidden'] == 128
            and meta['dimensions'] == 512 and meta['heads'] == 8 and meta['head_dim'] == 64 and meta['roles'] == ROLES, 'Portable model geometry/role order mismatch.')
        require(np.array_equal(archive['mean'], mean) and np.array_equal(archive['scale'], scale), 'Frozen old fit-only embedded scaler mismatch.')
        weights = {key[8:]: archive[key].copy() for key in archive.files if key.startswith('weight::')}
        expected = {'encoder.0.weight': (128, 512), 'encoder.0.bias': (128,), 'head.weight': (5, 128), 'head.bias': (5,)}
        require(set(weights) == set(expected) and all(weights[key].shape == shape and weights[key].dtype == np.float32
            and np.isfinite(weights[key]).all() for key, shape in expected.items()), 'Portable weights changed shape/dtype/finiteness.')
        if expected_metadata:
            for key, value in expected_metadata.items(): close(meta[key], value, 'model/' + key)
    return meta, weights


def portable_check(weights, mean, scale, bank, rows, p):
    import torch
    chosen = np.linspace(0, len(rows) - 1, 32, dtype=int)
    value = torch.from_numpy((np.asarray(bank[rows.iloc[chosen].cache_ix.to_numpy()], dtype=np.float32) - mean) / scale)
    with torch.inference_mode():
        hidden = torch.nn.functional.gelu(torch.nn.functional.linear(value, torch.from_numpy(weights['encoder.0.weight']), torch.from_numpy(weights['encoder.0.bias'])))
        actual = torch.softmax(torch.nn.functional.linear(hidden, torch.from_numpy(weights['head.weight']), torch.from_numpy(weights['head.bias'])), dim=1).numpy()
    difference = float(np.abs(actual - p[chosen]).max()); require(difference < 5e-4, 'Independent raw-F16/scaler/weights probability check failed.')
    return dict(sampled_tokens=32, maximum_difference=difference)


def verify_csv(frame, records):
    expected = {(record['record'], name): value for record in records for name, value in
        dict(record['validation']['groups'], pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items()}
    require(len(frame) == 54 and not frame.duplicated(['record', 'group']).any() and set(zip(frame.record, frame.group)) == set(expected), 'CSV must retain48 group and6 descriptive pooled rows.')
    for row in frame.to_dict('records'):
        value = expected[(row['record'], row['group'])]
        for key, truth in value.items():
            if isinstance(truth, (dict, list)): continue
            actual = row[key]; actual = None if pd.isna(actual) else actual
            close(actual, truth, 'csv/' + row['record'] + '/' + row['group'] + '/' + key)
        for role in ROLES:
            actual = row['role_recall_' + role]; actual = None if pd.isna(actual) else actual
            close(actual, value['recall'][role], 'csv/recall/' + role)


def actual():
    terminal_gate()
    require(data.digest(production.__file__) == SOURCE_SHA and data.digest(OUT / 'protocol.json') == PROTOCOL_SHA, 'Frozen training source/protocol changed.')
    bindings = {}; probability_bindings = {}
    def bind(path, expected=None):
        value = data.digest(path)
        if expected: require(value == expected, 'Artifact byte SHA mismatch: ' + str(path))
        bindings[str(path)] = value; return value
    bind(__file__); bind(production.__file__, SOURCE_SHA); bind(OUT / 'protocol.json', PROTOCOL_SHA)
    bind(OUT / 'status.json'); bind(data.runtime.OUT / 'systemness-final-readout/selection.json')
    complete, plan, declaration = read(OUT / 'complete.json'), read(OUT / 'search-plan.json'), read(OUT / 'protocol.json')
    bind(OUT / 'complete.json'); plan_sha = bind(OUT / 'search-plan.json')
    require(complete['source_sha256'] == SOURCE_SHA and complete['signature'] == plan['signature']
        and complete['tests_read'] is False and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and plan['protocol_sha256'] == PROTOCOL_SHA and set(complete['artifact_sha256']) == set(ARTIFACTS)
        and set(complete['candidate_sha256']) == set(ARMS), 'Formal completion contract is incomplete.')
    for name, value in complete['artifact_sha256'].items(): bind(OUT / name, value)
    for path, value in declaration['static']['metadata_sha256'].items(): bind(path, value)
    for name, value in declaration['static']['dependencies_sha256'].items(): bind(data.HERE / name, value)
    bind(production.DESIGN, declaration['static']['design_sha256'])
    require(declaration['static']['probability_arrays_read'] is False and declaration['protocol']['tests_read'] is False
        and declaration['protocol']['primary_changed'] is False and declaration['protocol']['trained_candidates'] == 2,
        'Training declaration boundary changed.')
    require(declaration['static']['package_versions'] == {name: importlib.metadata.version(name)
        for name in ['torch', 'numpy', 'pandas', 'scikit-learn']}
        and declaration['static']['float32_matmul_precision'] == 'highest', 'Frozen numerical packages/precision changed.')
    from systemness_long_augmentation_extract import verified_extraction
    new_complete = verified_extraction()
    extraction_signature = new_complete['signature']
    for name in ['extraction-plan.json', 'extraction-cpu-preflight.json', 'extraction-hook-validation.json', 'augmentation-model_backend.json']:
        bind(production.NEW_OUT / name)
    for name in ['initialized.json', 'progress.json']: bind(NEW / name)
    for name, value in extraction_signature['dependencies_sha256'].items(): bind(data.HERE / name, value)
    checkpoint = Path(extraction_signature['checkpoint_path'])
    bind(checkpoint / 'verified-weights.json', extraction_signature['verified_weights_sha256'])
    bind(checkpoint / 'config.json', extraction_signature['config_sha256'])
    for name, value in extraction_signature['tokenizer_sha256'].items(): bind(checkpoint / name, value)
    bind(data.runtime.OUT / 'checkpoint-cache-verification.json', extraction_signature['checkpoint_verification_sha256'])
    source_meta = read(production.NEW_OUT / 'data-complete.json')
    bind(production.NEW_OUT / 'plan.json', source_meta['plan_sha256'])
    for name, value in source_meta['metadata_sha256'].items(): bind(production.NEW_OUT / name, value)
    for name, value in source_meta['cache_sha256'].items(): bind(Path(source_meta['cache_path']) / name, value)
    for path, value in source_meta['source_signature']['manifests'].items(): bind(path, value['sha256'])
    for path, value in source_meta['source_signature']['original249_sha256'].items(): bind(path, value)
    import systemness_head_scan as scanner
    import systemness_long_data as long_data
    for name in ['plan.json', 'scan-complete.json', 'head-selection.json', 'discovery-complete.json', 'models/top8-lda.npz']:
        bind(scanner.OUT / name)
    scanner_plan = read(scanner.OUT / 'plan.json')
    source_metadata = {
        'long-plan': long_data.OUT / 'plan.json', 'long-data': long_data.OUT / 'data-complete.json',
        'long-keys': long_data.OUT / 'document-keys.json', 'long-extraction': long_data.OUT / 'extraction-complete.json',
        'dense-selection': long_data.OUT / 'selection.json',
        'paired-selection': data.runtime.OUT / 'systemness-paired-pilot/selection.json',
        'frozen-primary': data.runtime.OUT / 'systemness-final-readout/selection.json',
        'original-data': data.OUT / 'data-complete.json', 'supplement-data': data.OUT / 'supplement-data-complete.json',
        'checkpoint-verification': data.runtime.OUT / 'checkpoint-cache-verification.json'}
    require(set(source_metadata) == set(scanner_plan['source_metadata_sha256']), 'Scanner source metadata scope changed.')
    for name, path in source_metadata.items(): bind(path, scanner_plan['source_metadata_sha256'][name])
    old_complete = read(hp.FEATURE_OUT / 'extraction-complete.json')
    for folder, record in [(OLD, old_complete), (NEW, new_complete)]:
        bind(folder / 'rows.parquet', record['rows_sha256']); bind(folder / 'features.f16', record['features_sha256'])
    signature = plan['signature']; bind(production.NEW_OUT / 'data-complete.json', signature['new_data_sha256'])
    require(signature['protocol_sha256'] == PROTOCOL_SHA, 'Dynamic signature references another protocol.')
    bind(production.NEW_OUT / 'extraction-complete.json', signature['new_extraction_sha256'])
    bind(hp.FEATURE_OUT / 'extraction-complete.json', signature['old_extraction_sha256'])
    close(new_complete['signature'], signature['new_extraction_signature'], 'new extraction signature')
    require(signature['new_features_sha256'] == new_complete['features_sha256'] and signature['old_features_sha256'] == old_complete['features_sha256']
        and signature['new_rows_sha256'] == new_complete['rows_sha256'] and signature['old_rows_sha256'] == old_complete['rows_sha256']
        and signature['selected_head_ids'] == new_complete['selected_head_ids'] == old_complete['selected_head_ids'], 'Feature source/head rank signature mismatch.')
    old_rows, new_rows = pd.read_parquet(OLD / 'rows.parquet'), pd.read_parquet(NEW / 'rows.parquet')
    fit, ov, nf, nv, partition = source_partitions(old_rows, new_rows); close(signature['old_partitions'], partition, 'old partition')
    manifest, fresh, matched = draw_manifest(old_rows, nf)
    close(manifest, read(OUT / 'draw-manifest.json'), 'draw manifest'); close(manifest, plan['draw_manifest'], 'plan draw')
    require(canonical(manifest) == signature['draw_manifest_sha256'] and array_sha(fresh) == signature['new_fit_cache_ix_sha256']
        and array_sha(matched) == signature['matched_cache_ix_sha256'] and array_sha(nv.sample_ix.to_numpy()) == signature['new_validation_sample_ix_sha256']
        and array_sha(nv.cache_ix.to_numpy()) == signature['new_validation_cache_ix_sha256'], 'Packet/draw/validation sequence signature mismatch.')
    scaler_sha = bind(hp.OUT / 'standardization.npz', signature['standardization_sha256'])
    with np.load(hp.OUT / 'standardization.npz', allow_pickle=False) as values: mean, scale = values['mean'].copy(), values['scale'].copy()
    require(mean.shape == scale.shape == (512,) and mean.dtype == scale.dtype == np.float32
        and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid old-fit-only scaler.')
    init = initialization_sha(); require(init == plan['initial_weight_sha256'] == declaration['static']['initial_weight_sha256'], 'Initial weights differ from original seeded MLP128.')
    banks = dict(old=np.memmap(OLD / 'features.f16', mode='r', dtype=np.float16, shape=(755282, 512)),
        new=np.memmap(NEW / 'features.f16', mode='r', dtype=np.float16, shape=(389120, 512)))
    probabilities, models, budgets, portable = {}, {}, {}, {}
    for arm in ARMS:
        record_path, training_path = OUT / 'candidates' / (arm + '.json'), OUT / 'training' / (arm + '.json')
        bind(record_path, complete['candidate_sha256'][arm]); record = read(record_path)
        bind(training_path, record['training_complete_sha256']); training = read(training_path)
        budget = history_audit(training['history'], training['best_epoch']); budgets[arm] = budget
        for evidence in [record, training]:
            require(evidence['arm'] == arm and evidence['signature'] == signature and evidence['plan_sha256'] == plan_sha
                and evidence['initial_weight_sha256'] == init, 'Arm training evidence ownership mismatch.')
            for key, value in budget.items(): close(evidence[key], value, arm + '/budget/' + key)
        require(record['source_sha256'] == SOURCE_SHA and record['tests_read'] is False
            and training['new_validation_predictions_read'] is False and training['model_sha256'] == record['model_sha256']
            and training['old_probability_sha256'] == record['old_probability_sha256'], 'Best-model-before-new-validation evidence mismatch.')
        path = OUT / 'models' / (arm + '.npz'); models[arm] = bind(path, record['model_sha256'])
        _, weights = model_arrays(path, mean, scale, dict(arm=arm, source_sha256=SOURCE_SHA, signature=signature,
            plan_sha256=plan_sha, initial_weight_sha256=init, best_epoch=budget['best_epoch'], best_steps=budget['best_steps'],
            selected_head_ids=signature['selected_head_ids'], standardization_sha256=scaler_sha, tests_read=False))
        ps = []
        for bank, frame in [('old', ov), ('new', nv)]:
            p_path = CACHE / (arm + '-' + bank + '-validation.npz'); expected_sha = record[bank + '_probability_sha256']
            probability_bindings[str(p_path)] = bind(p_path, expected_sha); ps.append(read_probability(p_path, expected_sha, frame))
        probabilities[arm] = tuple(ps)
        old_tau = legacy_mean(ov, ps[0]); result = evaluate(ov, nv, *ps, old_tau)
        original = record['old_early_stop_validation']; close(original, old_validation_record(result, ov), arm + '/earlystop')
        close(result['old5_mean_f1'], training['history'][budget['best_epoch'] - 1]['old5_mean_f1'], arm + '/best/history')
        evidence = record['portable_evidence']; require(evidence['model_sha256'] == models[arm]
            and evidence['sampled_tokens'] == {'old': 256, 'new': 256}
            and set(evidence['maximum_probability_difference']) == {'old', 'new'}
            and all(np.isfinite(value) and 0 <= value < 5e-4 for value in evidence['maximum_probability_difference'].values()), 'Worker portable evidence mismatch.')
        portable[arm] = {bank: portable_check(weights, mean, scale, banks[bank], frame, p)
            for bank, frame, p in [('old', ov, ps[0]), ('new', nv, ps[1])]}
    control = read(OUT / 'old-head-control.json'); require(control['signature'] == signature
        and control['model_sha256'] == CURRENT_SHA and control['tests_read'] is False, 'Frozen control ownership mismatch.')
    models['old-head'] = bind(hp.OUT / 'models/joint-mlp128.npz', CURRENT_SHA)
    _, weights = model_arrays(hp.OUT / 'models/joint-mlp128.npz', mean, scale)
    original = read(hp.OUT / 'candidates/joint-mlp128.json')
    old_path = Path(declaration['static']['old_probability_path']); old_sha = declaration['static']['old_probability_declared_sha256']
    require(control['old_probability_sha256'] == old_sha == original['validation_probabilities_sha256'], 'Original NPZ identity changed.')
    probability_bindings[str(old_path)] = bind(old_path, old_sha); po = read_probability(old_path, old_sha, ov)
    new_path = CACHE / 'old-head-new-validation.npz'; probability_bindings[str(new_path)] = bind(new_path, control['new_probability_sha256'])
    pn = read_probability(new_path, control['new_probability_sha256'], nv); probabilities['old-head'] = (po, pn)
    require(legacy_mean(ov, po) == CURRENT_TAU == original['validation']['system_threshold'], 'Frozen original threshold changed.')
    close(original['validation'], old_validation_record(evaluate(ov, nv, po, pn, CURRENT_TAU), ov), 'unchanged-original-control')
    portable['old-head'] = {bank: portable_check(weights, mean, scale, banks[bank], frame, p)
        for bank, frame, p in [('old', ov, po), ('new', nv, pn)]}
    saved = read(OUT / 'records.json'); require(len(saved) == 6, 'Six records missing.'); computed, grid_proofs = [], {}
    saved_by_name = {entry['record']: entry for entry in saved}
    for model in MODELS:
        po, pn = probabilities[model]
        for objective in OBJECTIVES:
            name = model + '-' + objective
            if objective == 'old-mean': tau, proof = legacy_mean(ov, po), None
            else:
                tau, proof, annex = direct_grid(ov, nv, po, pn); grid_proofs[model] = dict(proof=proof, annex=annex)
            value = evaluate(ov, nv, po, pn, tau)
            entry = dict(record=name, model=model, objective=objective, model_sha256=models[model], validation=value,
                full_grid_proof=proof, promoted=False, tests_read=False, diagnostic=model != 'new' or objective != 'maximin')
            close(saved_by_name[name], entry, name); computed.append(entry)
    expected_selection = selection(computed); actual_selection = read(OUT / 'selection.json')
    for key, value in expected_selection.items(): close(actual_selection[key], value, 'selection/' + key)
    require(actual_selection['source_sha256'] == SOURCE_SHA and actual_selection['plan_sha256'] == plan_sha
        and actual_selection['records_sha256'] == complete['artifact_sha256']['records.json']
        and actual_selection['model_sha256'] == models['new'] and actual_selection['model_path'] == str(OUT / 'models/new.npz')
        and actual_selection['system_threshold'] == saved_by_name['new-maximin']['validation']['system_threshold'], 'Published selection model/tau/source mismatch.')
    verify_csv(pd.read_csv(OUT / 'validation-metrics.csv'), computed)
    sources = {}
    for name, (bank, ix) in group_indices(ov, nv).items():
        frame = (ov if bank == 'old' else nv).iloc[ix]
        sources[name] = dict(tokens=len(frame), documents=int(frame.group_id.nunique()),
            source_sets={str(key): int(value) for key, value in frame.source_set.value_counts().items()},
            role_counts={role: int(frame.role.eq(role).sum()) for role in ROLES},
            cache_ix_sha256=array_sha(frame.cache_ix.to_numpy()), sample_ix_sha256=array_sha(frame.sample_ix.to_numpy()))
    counts = read(OUT / 'source-counts.json')
    expected_counts = dict(old_fit=len(fit), old_valid=len(ov), new_fit=len(fresh), new_valid=len(nv), per_epoch=905770,
        updates_per_epoch=222, last_batch=554, old_fit_documents=int(fit.group_id.nunique()), old_validation_documents=int(ov.group_id.nunique()),
        new_fit_documents=256, new_validation_documents=96, matched_distinct_documents=132,
        old_validation_profile_counts={p: int(ov.profile.eq(p).sum()) for p in PROFILES},
        old_validation_profile_documents={p: int(ov[ov.profile.eq(p)].group_id.nunique()) for p in PROFILES},
        old_long_single_full_overlap_documents=len(set(ov[ov.profile.eq('long-single')].group_id) & set(ov[ov.profile.eq('long-full')].group_id)),
        tails_are_subsets=True, pooled_counts_not_independent_documents=True)
    for key, value in expected_counts.items(): close(counts[key], value, 'source-counts/' + key)
    require(sources['old-natural']['role_counts']['tool'] == 0 and all(entry['validation']['groups']['old-natural']['recall']['tool'] is None for entry in computed), 'Natural tool support must remain N/A.')
    terminal_gate()
    for path, expected in bindings.items(): require(data.digest(path) == expected, 'Input changed during actual audit: ' + path)
    report = dict(stage='actual_completed_long_augmentation_development_audit', audit_passed=True,
        auditor_source_path=str(Path(__file__).resolve()), auditor_source_sha256=data.digest(__file__),
        trainer_source_sha256=SOURCE_SHA, trainer_complete_sha256=bindings[str(OUT / 'complete.json')],
        trainer_protocol_sha256=PROTOCOL_SHA, trainer_selection_sha256=bindings[str(OUT / 'selection.json')],
        trainer_records_sha256=bindings[str(OUT / 'records.json')],
        trainer_model_record_sha256={arm: bindings[str(OUT / 'candidates' / (arm + '.json'))] for arm in ARMS},
        model_sha256=models, provenance_sha256=bindings,
        test_data_or_results_read=False, gpu_used=False,
        source_sha256=data.digest(__file__), training_source_sha256=SOURCE_SHA, protocol_sha256=PROTOCOL_SHA,
        bindings=bindings, probability_bindings=probability_bindings, six_records=computed, selection=expected_selection,
        independent_grid_proofs=grid_proofs, source_counts=sources, training_budgets=budgets, portable_checks=portable,
        maximum_metric_numeric_difference=MAX_DIFFERENCE, csv_primary_group_rows=48, csv_descriptive_pool_rows=6,
        old_current_NPZ_unchanged=True, original_old_threshold_unchanged=True,
        reused_development_validation=True, generalization_confidence_intervals_claimed=False,
        new_independent_confirmation_required=True, tests_read=False, gpu_work_run=False, formal_outputs_written=False)
    target = Path('/tmp/feature_review_long_augmentation_results_audit.json'); target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(target), report_sha256=data.digest(target), training_budgets=budgets,
        maximum_metric_numeric_difference=MAX_DIFFERENCE, selection=expected_selection, source_sha256=report['source_sha256'])), flush=True)


def synthetic():
    old_rows, new_rows = [], []
    for profile_ix, profile in enumerate(PROFILES):
        for role_ix, role in enumerate(ROLES):
            if profile == 'natural' and role == 'tool': continue
            for offset in [0, 341, 682, 1023]:
                old_rows.append(dict(profile=profile, role=role, token_ix=17 + offset, prompt_ix=profile_ix * 5 + role_ix,
                    message_ix=0, document_id=profile_ix, group_id='old-' + str(profile_ix), source_set='fixture',
                    cache_ix=len(old_rows), sample_ix=len(old_rows), split='validation', wrapper='native'))
    for role_ix, role in enumerate(ROLES):
        for offset in [0, 341, 682, 1023]:
            new_rows.append(dict(profile='long-full', role=role, token_ix=29 + offset, body_offset=offset,
                prompt_ix=100 + role_ix, message_ix=0, document_id=6000000, group_id='new', source_set='long-augmentation',
                cache_ix=len(new_rows), sample_ix=6000000000000 + len(new_rows), split='validation', wrapper='native'))
    old_rows, new_rows = pd.DataFrame(old_rows), pd.DataFrame(new_rows)
    def probabilities(frame):
        y = labels(frame); p = np.full((len(frame), 5), .02, np.float32); p[np.arange(len(y)), y] = .92
        for index in np.flatnonzero(y == 0)[::3]:
            p[index, 0] = np.nextafter(np.float32(.5), np.float32(1)); p[index, 1:] = (1 - p[index, 0]) / 4
        return p
    return old_rows, new_rows, probabilities(old_rows), probabilities(new_rows)


def preflight():
    checks = {}; source_before = data.digest(__file__)
    with patch(__name__ + '.processes', return_value=[123]), patch(__name__ + '.read', side_effect=AssertionError('Result read before terminal')):
        try: actual()
        except RuntimeError: pass
        else: raise RuntimeError('Live actual-result gate failed.')
    checks['live_process_rejected_before_any_result_read'] = True
    from systemness_probes import metrics as reference
    rng = np.random.default_rng(4711)
    for case in range(8):
        y = rng.integers(0, 5 if case % 2 else 3, 400); p = rng.integers(1, 10, size=(400, 5)).astype(np.float32); p /= p.sum(1)[:, None]
        close(metrics(p, y, .2 + case * .03), reference(p, y, .2 + case * .03), 'synthetic-metrics')
    checks['eight_random_tied_probability_metrics_all_fields'] = True
    p = np.tile(np.array([.1, .1, .6, .1, .1], np.float32), (6, 1)); y = np.array([0, 0, 1, 1, 3, 3])
    value = metrics(p, y, .9); close(value, reference(p, y, .9), 'supported-macro')
    require(value['recall']['cot'] is None and value['recall']['tool'] is None and value['system_precision'] is None, 'Undefined support policy changed.')
    checks['truth_supported_macro_and_undefined_roles_precision'] = True
    old_rows, new_rows, po, pn = synthetic(); tau = legacy_mean(old_rows, po)
    require(tau == production.lp.validation_metrics(old_rows, po)['system_threshold'], 'Legacy old-only threshold changed.')
    close(evaluate(old_rows, new_rows, po, pn, tau), production.metrics_at(old_rows, new_rows, po, pn, tau), 'synthetic-evaluate')
    close(old_validation_record(evaluate(old_rows, new_rows, po, pn, tau), old_rows),
        production.lp.validation_metrics(old_rows, po), 'complete-old-only-helper')
    chosen, proof, _ = direct_grid(old_rows, new_rows, po, pn)
    expected_tau, expected_proof = production.maximin_threshold(old_rows, new_rows, po, pn)
    require(chosen == expected_tau, 'Independent F32 full-grid ranking differs.')
    close(proof, expected_proof, 'full-grid-proof')
    checks['legacy_threshold_exact_and_all8group_metrics'] = True
    checks['all_grid24stats_feasible_objective_lexargmax'] = True
    adjacent = np.nextafter(np.float32(.5), np.float32(1)); boundary = .5 + (float(adjacent) - .5) / 6
    scores = np.array([.5, adjacent], np.float32)
    require(int(np.count_nonzero(scores >= boundary)) == 2
        and len(scores) - np.searchsorted(np.sort(scores), np.array([boundary]), side='left')[0] == 1,
        'NumPy F32 consumer/legacy F64 boundary fixture did not exercise the declared distinction.')
    checks['nextafter_boundary_actual_F32_distinct_from_legacy_F64'] = True
    groups = group_indices(old_rows, new_rows)
    require(np.array_equal(groups['old-long-full-tail'][1], np.flatnonzero(old_rows.profile.eq('long-full') & old_rows.token_ix.ge(17 + 512)))
        and np.array_equal(groups['new-long-full-tail'][1], np.flatnonzero(new_rows.body_offset.ge(512))), 'Global tail indices do not match saved probabilities.')
    checks['global_tail_probability_indices'] = True
    def g(score): return dict(min24=score, old5_mean_f1=score, groups={name: dict(system_f1=score, accuracy=score, system_false_positive_rate=.01) for name in GROUPS})
    records = [dict(record=model + '-' + objective, validation=g(.9)) for model in MODELS for objective in OBJECTIVES]
    next(entry for entry in records if entry['record'] == 'new-maximin')['validation'] = g(.93)
    next(entry for entry in records if entry['record'] == 'matched-maximin')['validation'] = g(.92)
    close(selection(records), production.select_records(records), 'selection-pass')
    require(selection(records)['promoted'], 'Positive promotion fixture failed.')
    changes = [lambda value: value.update(min24=.92), lambda value: value.update(old5_mean_f1=.89),
        lambda value: value['groups']['old-single'].update(system_f1=.85), lambda value: value['groups']['old-multi'].update(accuracy=.87),
        lambda value: value['groups']['new-long-full-tail'].update(system_f1=.89), lambda value: value['groups']['new-long-full'].update(system_false_positive_rate=.051)]
    for change in changes:
        bad = copy.deepcopy(records); change(next(entry for entry in bad if entry['record'] == 'new-maximin')['validation'])
        close(selection(bad), production.select_records(bad), 'selection-rejection'); require(not selection(bad)['promoted'] and selection(bad)['candidate'] == 'new-maximin', 'A guard was bypassed or fallback selected.')
    checks['all_three_controls_six_guard_faults_no_fallback'] = True
    history = [dict(epoch=epoch, instances=905770, steps=222, cumulative_steps=epoch * 222, loss=.5, old5_mean_f1=.8) for epoch in range(1, 11)]
    close(history_audit(history, 1), production.check_history(history, 1), 'history')
    for wrong in [history[:8], history + [dict(history[-1], epoch=11, cumulative_steps=2442)]]:
        try: history_audit(wrong, 1)
        except RuntimeError: pass
        else: raise RuntimeError('Invalid early-stop history accepted.')
    checks['best_actual_steps_patience_and_minimum_guards'] = True
    require(initialization_sha() == production.expected_initialization(), 'Independent initialization hash differs.')
    checks['independent_seeded_initialization_sha'] = True
    full_records = [dict(record=model + '-' + objective, validation=evaluate(old_rows, new_rows, po, pn, tau)) for model in MODELS for objective in OBJECTIVES]
    flat = []
    for entry in full_records:
        for name, value in dict(entry['validation']['groups'], pooled_descriptive_only=entry['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=entry['record'], group=name, **{key: truth for key, truth in value.items() if not isinstance(truth, (dict, list))}, **{'role_recall_' + role: truth for role, truth in value['recall'].items()}))
    frame = pd.DataFrame(flat); verify_csv(frame, full_records)
    broken = frame.copy(); broken.loc[0, 'system_recall'] = .123
    try: verify_csv(broken, full_records)
    except RuntimeError: pass
    else: raise RuntimeError('CSV metric tamper accepted.')
    checks['all54_csv_rows_nan_and_metric_tamper'] = True
    folder = Path(tempfile.mkdtemp(prefix='feature-review-long-augmentation-results-preflight-', dir='/tmp'))
    target = folder / 'probability.npz'; np.savez(target, p=po, sample_ix=old_rows.sample_ix.to_numpy())
    require(np.array_equal(read_probability(target, digest(target), old_rows), po), 'Probability sample identity fixture failed.')
    wrong = old_rows.copy(); wrong['sample_ix'] += 1
    try: read_probability(target, digest(target), wrong)
    except RuntimeError: pass
    else: raise RuntimeError('Wrong probability sample namespace accepted.')
    checks['probability_hash_and_sample_namespace'] = True
    require(data.digest(__file__) == source_before, 'Auditor source changed during preflight.')
    report = dict(source_sha256=source_before, training_source_sha256=SOURCE_SHA, protocol_sha256=PROTOCOL_SHA, checks=checks,
        synthetic_only=True, actual_model_or_probability_files_read=False, formal_outputs_written=False, gpu_work_run=False,
        maximum_synthetic_metric_difference=MAX_DIFFERENCE, direct_grid_proof=proof)
    target = folder / 'report.json'; target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(target), report_sha256=data.digest(target), **report)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true'); action.add_argument('--actual', action='store_true'); args = parser.parse_args()
    if args.preflight: preflight()
    elif args.actual:
        with patch.object(production, 'calibrate', side_effect=AssertionError('Production calibration forbidden')), \
                patch.object(production, 'verified_completion', side_effect=AssertionError('Production result summarizer forbidden')), \
                patch.object(production, 'metrics_at', side_effect=AssertionError('Production metrics forbidden')):
            actual()
    else: parser.print_help()
