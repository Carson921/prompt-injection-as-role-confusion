"""Independent, development-only audit for all four frozen head readouts.

Default prints help. --preflight uses synthetic CPU data under /tmp. --actual
first requires heads.verified_completion(), then independently recomputes saved
validation probabilities, threshold grids, guards, selection and MLP budgets.
It never launches GPU work, reads tests, changes a model or writes formal OUT.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

REPOSITORY = Path('/root/prompt-injection-as-role-confusion')
MODULE_DIRECTORY = REPOSITORY / 'experiments/role-analysis'
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
PROFILES = ['single', 'multi', 'natural', 'long-single', 'long-full']
ORIGINAL = PROFILES[:4]
CANDIDATES = ['pooled-lda', 'joint-linear', 'joint-mlp64', 'joint-mlp128']
FIT_ROWS, VALID_ROWS, TAIL_ROWS, TOTAL_ROWS = 578090, 173032, 9920, 755282
BATCH, STEPS, SEED = 4096, 142, 20261021
GUARD_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_tmp(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(Path('/tmp')), 'Audit output must be under /tmp.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return path


def compare(actual, expected, context='value', tolerance=1e-12):
    """Compare independently calculated full metric/guard dictionaries."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), context + ': dictionary keys differ.')
        return max([0.] + [compare(actual[k], expected[k], context + '.' + k, tolerance) for k in expected])
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context + ': list length differs.')
        return max([0.] + [compare(a, e, context + '[' + str(i) + ']', tolerance)
                          for i, (a, e) in enumerate(zip(actual, expected))])
    if expected is None or isinstance(expected, (str, bool)):
        require(type(actual) is type(expected) and actual == expected, context + ': categorical value differs.')
        return 0.
    if isinstance(expected, (int, np.integer)):
        require(isinstance(actual, (int, np.integer)) and int(actual) == int(expected), context + ': integer differs.')
        return 0.
    require(isinstance(expected, (float, np.floating)) and math.isfinite(float(actual))
            and math.isfinite(float(expected)), context + ': nonfinite or unexpected scalar.')
    difference = abs(float(actual) - float(expected))
    require(difference <= tolerance, context + ': numeric difference ' + str(difference))
    return difference


def validate_probabilities(p, sample_ix, expected_ix):
    require(p.dtype == np.float32 and p.shape == (len(expected_ix), 5), 'Wrong probability dtype/shape.')
    require(sample_ix.dtype == expected_ix.dtype and np.array_equal(sample_ix, expected_ix),
            'Saved probabilities have different validation sample_ix/order/dtype.')
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1.00001).all()
            and np.allclose(p.sum(1), 1., atol=2e-5, rtol=0), 'Nonfinite or unnormalized probabilities.')


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def labels(rows):
    value = rows.role.map({name: i for i, name in enumerate(ROLES)}).to_numpy()
    require(np.isfinite(value).all() and np.isin(value, np.arange(5)).all(), 'Unknown formal row role.')
    return value.astype(np.int64)


def independent_metrics(p, truth, threshold):
    """Confusion/count-based metrics; no experiment metric helper is called."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    truth = np.asarray(truth, np.int64)
    require(len(p) == len(truth) and len(p) > 0, 'Invalid metric rows.')
    prediction = np.argmax(p, axis=1)
    matrix = np.zeros((5, 5), np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    diagonal = np.diag(matrix)
    class_f1 = np.zeros(5, np.float64)
    np.divide(2 * diagonal, support + predicted, out=class_f1, where=(support + predicted) > 0)
    positive = truth == 0
    detection = p[:, 0] >= threshold
    tp = int(np.count_nonzero(positive & detection))
    fn = int(np.count_nonzero(positive & ~detection))
    fp = int(np.count_nonzero(~positive & detection))
    tn = int(np.count_nonzero(~positive & ~detection))
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return dict(tokens=len(truth), accuracy=float(np.count_nonzero(prediction == truth) / len(truth)),
        macro_f1=float(np.mean(class_f1[support > 0])),
        nll=float(-np.log(np.clip(p[np.arange(len(p)), truth], 1e-15, 1)).mean()),
        recall={role: divide(int(diagonal[i]), int(support[i])) for i, role in enumerate(ROLES)},
        confusion=matrix.tolist(), system_threshold=float(threshold),
        system_accuracy=float((tp + tn) / len(truth)),
        system_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
        system_precision=divide(tp, tp + fp), system_recall=recall,
        system_f1=divide(2 * tp, 2 * tp + fp + fn), system_false_positive_rate=divide(fp, fp + tn),
        system_auroc=float(roc_auc_score(positive, p[:, 0])) if positive.any() and (~positive).any() else None,
        system_average_precision=float(average_precision_score(positive, p[:, 0])) if positive.any() else None,
        system_detection_by_role={role: float(detection[truth == i].mean()) if support[i] else None for i, role in enumerate(ROLES)},
        system_confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn))


def independent_threshold(rows, p):
    """Prespecified grid, independently counted via sorted all-class prefixes."""
    grid = np.unique(np.concatenate((np.linspace(.001, .999, 199),
        np.quantile(p[:, 0], np.linspace(0., 1., 301)), np.array([1.000001]))))
    y = labels(rows)
    scores, fprs = [], []
    for profile in PROFILES:
        indices = np.flatnonzero(rows.profile.to_numpy() == profile)
        values, positive = p[indices, 0], y[indices] == 0
        order = np.argsort(values, kind='stable')
        values, positive = values[order], positive[order]
        prefix_positive = np.concatenate(([0], np.cumsum(positive, dtype=np.int64)))
        left = np.searchsorted(values, grid, side='left')
        count_positive, count_negative = int(positive.sum()), int((~positive).sum())
        tp = count_positive - prefix_positive[left]
        fp = len(values) - left - tp
        fn = count_positive - tp
        scores.append((2 * tp) / np.maximum(2 * tp + fp + fn, 1))
        fprs.append(fp / max(count_negative, 1))
    objective = np.mean(np.stack(scores), axis=0)
    fprs = np.stack(fprs)
    feasible = np.max(fprs, axis=0) <= .05
    require(feasible.any(), 'No feasible global threshold.')
    masked = np.where(feasible, objective, -1.)
    best = float(masked.max())
    tied = np.flatnonzero(masked == best)
    index = int(tied[0])
    return float(grid[index]), dict(grid_count=len(grid), selected_grid_index=index,
        feasible_grid_count=int(feasible.sum()), first_maximum_tie_count=len(tied), objective=best,
        grid_sha256=sha_array(grid), objective_sha256=sha_array(masked),
        fpr_by_profile={name: float(fprs[i, index]) for i, name in enumerate(PROFILES)},
        rule='Sorted union:199 linspace(.001,.999),301 linear quantiles of all validation p(system),1.000001; first maximum mean5F1 satisfying every-profile FPR<=.05.')


def tail_indices(rows):
    full = rows[rows.profile == 'long-full']
    offsets = full.token_ix - full.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')
    ix = full.index[offsets >= 512].to_numpy()
    return ix, int(full.loc[ix].document_id.nunique())


def independent_validation(rows, p):
    require(np.array_equal(rows.index, np.arange(len(rows))) and set(rows.split) == {'validation'}
            and set(rows.wrapper) == {'native'} and set(rows.profile) == set(PROFILES),
            'Validation-only native profiles/index contract changed.')
    threshold, calibration = independent_threshold(rows, p)
    y = labels(rows)
    profiles = {profile: independent_metrics(p[rows.profile.to_numpy() == profile],
        y[rows.profile.to_numpy() == profile], threshold) for profile in PROFILES}
    ix, documents = tail_indices(rows)
    tail = independent_metrics(p[ix], y[ix], threshold)
    tail.update(documents=documents, body_start=512, body_end=1023)
    return dict(system_threshold=threshold,
        mean_system_f1=float(np.mean([profiles[k]['system_f1'] for k in PROFILES])),
        original_mean_system_f1=float(np.mean([profiles[k]['system_f1'] for k in ORIGINAL])),
        original_mean_accuracy=float(np.mean([profiles[k]['accuracy'] for k in ORIGINAL])),
        profiles=profiles, long_full_tail=tail), calibration


def independent_guard(result, baseline):
    losses = {p: baseline['profiles'][p]['system_f1'] - result['profiles'][p]['system_f1'] for p in ORIGINAL}
    accuracy_losses = {p: baseline['profiles'][p]['accuracy'] - result['profiles'][p]['accuracy'] for p in ORIGINAL}
    full = result['profiles']['long-full']['system_f1'] - baseline['profiles']['long-full']['system_f1']
    tail = result['long_full_tail']['system_f1'] - baseline['long_full_tail']['system_f1']
    finite = all(np.isfinite(result['profiles'][p][k]) for p in PROFILES
        for k in ['system_f1', 'accuracy', 'system_false_positive_rate'])
    finite = bool(finite and np.isfinite(result['long_full_tail']['system_f1']) and np.isfinite(result['mean_system_f1']))
    reasons = []
    if not finite: reasons.append('nonfinite_validation_metric')
    if max(losses.values()) > .02 + GUARD_TOLERANCE: reasons.append('original_profile_system_f1_loss')
    if max(accuracy_losses.values()) > .01 + GUARD_TOLERANCE: reasons.append('original_profile_five_way_accuracy_loss')
    if full < -GUARD_TOLERANCE: reasons.append('long_full_system_f1_regression')
    if tail < -GUARD_TOLERANCE: reasons.append('long_full_tail_system_f1_regression')
    if any(result['profiles'][p]['system_false_positive_rate'] > .05 + GUARD_TOLERANCE for p in PROFILES):
        reasons.append('profile_system_fpr_above_limit')
    return dict(eligible=not reasons, mean_f1_gain=result['mean_system_f1'] - baseline['mean_system_f1'],
        original_profile_f1_losses=losses, original_profile_accuracy_losses=accuracy_losses,
        long_full_f1_gain=full, tail_f1_gain=tail, rejection_reasons=reasons)


def independent_selection(records, baseline):
    require([r['candidate'] for r in records] == CANDIDATES, 'All four candidates in fixed tie order are required.')
    guards = {r['candidate']: independent_guard(r['validation'], baseline) for r in records}
    eligible = [r for r in records if guards[r['candidate']]['eligible']]
    report_best = max(records, key=lambda r: r['validation']['mean_system_f1'])
    selected = max(eligible, key=lambda r: r['validation']['mean_system_f1']) if eligible else report_best
    guard = guards[selected['candidate']]
    promoted = bool(eligible and guard['mean_f1_gain'] > 0)
    reasons = [] if promoted else ((['no_guard_eligible_candidate'] if not eligible else [])
        + guard['rejection_reasons'] + (['mean_system_f1_not_improved'] if guard['mean_f1_gain'] <= 0 else []))
    return dict(candidate=selected['candidate'], report_best_candidate=report_best['candidate'],
        candidate_is_guard_eligible=guard['eligible'], candidate_guards=guards, promoted=promoted,
        promotion_rejection_reasons=reasons, selected_guard=guard,
        system_threshold=selected['validation']['system_threshold'])


def independent_budget(record, metadata):
    history = record['history']
    epochs, best_epoch = record['epochs'], record['best_epoch']
    require(type(epochs) is int and 10 <= epochs <= 45 and len(history) == epochs, 'Invalid MLP epoch count.')
    best, chosen = -1., -1
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and row['instances'] == FIT_ROWS and row['steps'] == STEPS
                and row['cumulative_steps'] == epoch * STEPS, 'Epoch order/instance/update budget differs.')
        require(math.isfinite(row['loss']) and row['loss'] >= 0 and 0 <= row['mean_system_f1'] <= 1
                and 0 <= row['long_full_f1'] <= 1 and 0 <= row['tail_f1'] <= 1, 'Nonfinite/invalid epoch metrics.')
        if row['mean_system_f1'] > best + 1e-5:
            best, chosen = row['mean_system_f1'], epoch
        if epoch < epochs:
            require(epoch < 10 or epoch - chosen < 8, 'MLP continued after its early-stop condition.')
    require(epochs == 45 or (epochs >= 10 and epochs - chosen >= 8), 'MLP stopped too early.')
    require(type(best_epoch) is int and best_epoch == chosen and metadata['best_epoch'] == chosen,
            'Best epoch differs from independent1e-5 checkpoint events.')
    require(abs(best - record['validation']['mean_system_f1']) <= 1e-12, 'Best checkpoint metric differs from saved p.')
    expected = dict(best_steps=chosen * STEPS, actual_steps=epochs * STEPS,
        actual_instances=epochs * FIT_ROWS, instances_per_epoch=FIT_ROWS, steps_per_epoch=STEPS,
        last_batch_instances=FIT_ROWS - (STEPS - 1) * BATCH)
    require(all(record[k] == v for k, v in expected.items()) and metadata['best_steps'] == expected['best_steps'],
            'MLP instances/steps/best checkpoint metadata differs.')
    require(record['seed'] == SEED, 'MLP seed differs.')
    chosen_row = history[chosen - 1]
    require(abs(chosen_row['long_full_f1'] - record['validation']['profiles']['long-full']['system_f1']) <= 1e-12
            and abs(chosen_row['tail_f1'] - record['validation']['long_full_tail']['system_f1']) <= 1e-12,
            'Best epoch full/tail metrics differ from saved probabilities.')
    return dict(epochs=epochs, best_epoch=chosen, best_mean_system_f1=best,
        stop='maximum45' if epochs == 45 else 'patience8_after_minimum10', **expected)


def model_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        mean, scale = archive['mean'].copy(), archive['scale'].copy()
    require(mean.shape == scale.shape == (512,) and np.isfinite(mean).all() and np.isfinite(scale).all()
            and (scale > 0).all(), 'Invalid portable512 scaler.')
    return meta, mean, scale


def load_npz(path, expected_ix, expected_sha):
    require(sha_file(path) == expected_sha, 'Frozen probability SHA differs: ' + str(path))
    with np.load(path, allow_pickle=False) as archive:
        p, sample_ix = archive['p'].copy(), archive['sample_ix'].copy()
    validate_probabilities(p, sample_ix, expected_ix)
    return p


def actual(output):
    # Gate first. Importing source defines paths but does not touch formal rows.
    sys.path.insert(0, str(MODULE_DIRECTORY))
    import systemness_head_probes as heads
    completed = heads.verified_completion()
    require(list(heads.CONFIGS) == CANDIDATES and heads.data.ROLES == ROLES, 'Frozen source configuration/order differs.')
    for key in CANDIDATES:
        require((heads.OUT/'candidates'/f'{key}.json').is_file()
                and (heads.CACHE/f'{key}-validation.npz').is_file()
                and (heads.OUT/'models'/f'{key}.npz').is_file(), 'Missing actual completed candidate: ' + key)
    import pandas as pd
    import pyarrow.parquet as pq
    plan = read(heads.OUT/'search-plan.json')
    selection = read(heads.OUT/'selection.json')
    require(completed['signature'] == plan['signature'], 'Completion/plan signature differs.')
    provenance = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha_file(path)
        if expected is not None: require(digest == expected, 'Frozen SHA differs: ' + str(path))
        provenance[str(path)] = digest
        return digest
    source_sha = bind(Path(__file__))
    bind(heads.OUT/'complete.json')
    bind(heads.OUT/'search-plan.json', completed['search_plan_sha256'])
    bind(heads.OUT/'selection.json', completed['selection_sha256'])
    bind(heads.OUT/'standardization.npz', completed['standardization_sha256'])
    bind(heads.OUT/'validation-metrics.csv', completed['validation_metrics_sha256'])
    for name, digest in plan['signature']['dependencies_sha256'].items(): bind(MODULE_DIRECTORY/name, digest)
    rows_path = heads.BANK/'rows.parquet'
    bind(rows_path, plan['signature']['rows_sha256'])
    require(pq.ParquetFile(rows_path).metadata.num_rows == TOTAL_ROWS, 'Development bank row count differs.')
    rows = pd.read_parquet(rows_path, filters=[('split', '==', 'validation')]).reset_index(drop=True)
    require(len(rows) == VALID_ROWS and rows.sample_ix.is_unique and rows.cache_ix.is_unique
            and np.all(np.diff(rows.cache_ix.to_numpy()) > 0), 'Validation count/order/identities differ.')
    expected_ix = rows.sample_ix.to_numpy()
    require(sha_array(expected_ix) == plan['signature']['partitions']['validation_sample_ix_sha256']
            and sha_array(rows.cache_ix.to_numpy()) == plan['signature']['partitions']['validation_cache_ix_sha256']
            and sha_array(labels(rows)) == plan['signature']['partitions']['validation_labels_sha256'],
            'Validation identities or formal labels differ from the predeclared signature.')
    tail, tail_docs = tail_indices(rows)
    require(len(tail) == TAIL_ROWS and tail_docs == 31, 'Expected9920 tail rows across31 sources.')
    scaler_path = heads.OUT/'standardization.npz'
    with np.load(scaler_path, allow_pickle=False) as archive:
        fitted_mean, fitted_scale = archive['mean'].copy(), archive['scale'].copy()
        require(str(archive['fit_cache_ix_sha256']) == plan['signature']['partitions']['fit_cache_ix_sha256'],
                'Scaler fit-row identity differs.')
    require(fitted_mean.shape == fitted_scale.shape == (512,) and np.isfinite(fitted_mean).all()
            and np.isfinite(fitted_scale).all() and (fitted_scale > 0).all(), 'Invalid frozen fit-only scaler.')
    dense_path = heads.long_probes.BANK/'dense-mlp128-validation.npz'
    dense_record_path = heads.long_data.OUT/'candidates/dense-mlp128.json'
    dense_record = read(dense_record_path)
    dense_plan_path=heads.long_data.OUT/'search-plan.json'
    bind(dense_plan_path,plan['signature']['dense_search_plan_sha256'])
    dense_plan=read(dense_plan_path)
    dense_scaler_path=heads.long_probes.attention.OUT/'standardization.npz'
    bind(dense_scaler_path,dense_plan['standardization_sha256'])
    bind(dense_record_path, plan['signature']['dense_candidate_sha256'])
    dense_model_path=heads.long_data.OUT/'models/dense-mlp128.npz'
    bind(dense_model_path, plan['signature']['dense_model_sha256'])
    with np.load(dense_model_path,allow_pickle=False) as archive:
        dense_mean,dense_scale=archive['mean'].copy(),archive['scale'].copy()
        dense_metadata=json.loads(str(archive['metadata']))
    with np.load(dense_scaler_path,allow_pickle=False) as archive:
        require(np.array_equal(dense_mean,archive['mean']) and np.array_equal(dense_scale,archive['scale']),
                'Frozen dense model embedded scaler differs from original scaler.')
    require(dense_mean.shape==dense_scale.shape==(8640,) and dense_metadata['kind']=='mlp'
            and dense_metadata['hidden']==128 and dense_metadata['roles']==ROLES
            and np.isfinite(dense_mean).all() and np.isfinite(dense_scale).all() and (dense_scale>0).all(),
            'Dense128 model/scaler geometry differs.')
    bind(dense_path, plan['signature']['dense_probabilities_sha256'])
    dense = load_npz(dense_path, expected_ix, plan['signature']['dense_probabilities_sha256'])
    baseline, baseline_threshold = independent_validation(rows, dense)
    maximum_difference = compare(baseline, dense_record['validation'], 'dense frozen validation')
    maximum_difference = max(maximum_difference, compare(baseline, plan['signature']['baseline_validation'], 'dense signature baseline'))
    require(baseline['system_threshold'] == dense_record['validation']['system_threshold'], 'Dense frozen threshold differs.')
    baseline_output = dict(validation=baseline, independent_threshold_audit=baseline_threshold,
        embedded_scaler_mean_sha256=sha_array(dense_mean),embedded_scaler_scale_sha256=sha_array(dense_scale))
    records, summaries = [], {}
    for key in CANDIDATES:
        record_path = heads.OUT/'candidates'/f'{key}.json'
        bind(record_path, completed['candidate_sha256'][key])
        record = read(record_path)
        model_path = heads.OUT/'models'/f'{key}.npz'
        probability_path = heads.CACHE/f'{key}-validation.npz'
        bind(model_path, record['model_sha256']); bind(probability_path, record['validation_probabilities_sha256'])
        require(record['candidate'] == key and record['configuration'] == heads.CONFIGS[key]
                and record['tests_read'] is False, 'Candidate identity/configuration differs.')
        p = load_npz(probability_path, expected_ix, record['validation_probabilities_sha256'])
        calculated, threshold_audit = independent_validation(rows, p)
        require(calculated['system_threshold'] == record['validation']['system_threshold'], key + ': independent threshold differs.')
        maximum_difference = max(maximum_difference, compare(calculated, record['validation'], key + '.validation'))
        meta, mean, scale = model_metadata(model_path)
        require(meta['signature'] == plan['signature'] and meta['configuration'] == heads.CONFIGS[key]
                and meta['roles'] == ROLES and meta['dimensions'] == 512 and meta['heads'] == 8 and meta['head_dim'] == 64,
                'Portable model identity/feature geometry differs.')
        if heads.CONFIGS[key]['standardized']:
            require(np.array_equal(mean, fitted_mean) and np.array_equal(scale, fitted_scale), 'Portable model scaler differs from frozen fit-only scaler.')
        else:
            require(np.array_equal(mean, np.zeros(512, np.float32)) and np.array_equal(scale, np.ones(512, np.float32)),
                    'Raw pooled-LDA must not apply the standardized scaler.')
        require(math.isfinite(record['portable_probability_max_difference'])
                and 0 <= record['portable_probability_max_difference'] < 5e-4, 'Invalid saved portable evidence.')
        budget = independent_budget(record, meta) if heads.CONFIGS[key]['kind'] == 'mlp' else dict(epoch_based=False)
        records.append(dict(candidate=key, validation=calculated))
        changes = {}
        for profile, values in dict(calculated['profiles'], **{'long-full-tail': calculated['long_full_tail']}).items():
            control = baseline['long_full_tail'] if profile == 'long-full-tail' else baseline['profiles'][profile]
            changes[profile] = {field: values[field] - control[field] for field in
                ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy', 'system_false_positive_rate', 'accuracy']
                if values[field] is not None and control[field] is not None}
            changes[profile]['role_recall'] = {role: values['recall'][role] - control['recall'][role] for role in ROLES}
        summaries[key] = dict(validation=calculated, threshold_audit=threshold_audit, delta_vs_dense128=changes,
            training_budget=budget, portable_evidence_max_difference=record['portable_probability_max_difference'],
            portable_mean_sha256=sha_array(mean), portable_scale_sha256=sha_array(scale))
    decision = independent_selection(records, baseline)
    for key, value in decision.items():
        maximum_difference = max(maximum_difference, compare(value, selection[key], 'selection.' + key))
    chosen = next(r for r in records if r['candidate'] == decision['candidate'])
    maximum_difference = max(maximum_difference, compare(chosen['validation'], selection['validation'], 'selected validation'))
    require(selection['baseline_candidate'] == 'dense-mlp128' and selection['test_used_for_selection'] is False
            and selection['primary_changed'] is False and selection['confirmation_required'] is True,
            'Primary/test/confirmation policy changed.')
    # Audit CSV independently against the same reconstructed profiles/tail.
    table = pd.read_csv(heads.OUT/'validation-metrics.csv')
    require(len(table) == 24 and not table[['candidate', 'profile']].duplicated().any(), 'Validation CSV coverage differs.')
    csv_max_difference = 0.
    for key in CANDIDATES:
        metrics = summaries[key]['validation']
        for profile, values in dict(metrics['profiles'], **{'long-full-tail': metrics['long_full_tail']}).items():
            found = table[(table.candidate == key) & (table.profile == profile)]
            require(len(found) == 1, 'Missing candidate/profile CSV row.')
            for field, value in values.items():
                if isinstance(value, (dict, list)) or value is None: continue
                stored = found.iloc[0][field]
                difference = abs(float(stored) - float(value))
                require(difference < 1e-11, 'CSV numeric metric differs: ' + key + '/' + profile + '/' + field)
                csv_max_difference = max(csv_max_difference, difference)
    for path, digest in provenance.items():
        require(sha_file(path) == digest, 'Bound audit input changed during CPU audit: ' + path)
    report = dict(stage='actual_completed_four_head_validation_audit', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_sha, upstream_gate='heads.verified_completion() passed before result reads/output creation',
        independent_recomputation=True, validation_rows=VALID_ROWS, long_full_tail_rows=TAIL_ROWS, tail_documents=tail_docs,
        row_role_order=ROLES, candidate_tie_order=CANDIDATES,
        validation_sample_ix_sha256=sha_array(expected_ix), validation_cache_ix_sha256=sha_array(rows.cache_ix.to_numpy()),
        validation_labels_sha256=sha_array(labels(rows)), maximum_metric_difference=maximum_difference,
        maximum_csv_difference=csv_max_difference, dense128=baseline_output, candidates=summaries,
        independent_selection=decision, provenance_sha256=provenance, frozen_signature=plan['signature'],
        gpu_used=False, test_data_read=False, models_or_thresholds_changed=False, formal_outputs_written=False,
        limitation='Reused development validation only. All five role recalls are argmax recalls; system detection uses the global threshold. Head selection, feature geometry/classifier/scaling and extraction batches differ from dense128. Saved portable evidence is hash-bound; no raw features or new model forward are used by this independent audit beyond upstream integrity hashing. A promoted candidate still needs fresh independent confirmation.')
    path = write_tmp(output, report)
    print(json.dumps(dict(output=str(path), promoted=decision['promoted'], candidate=decision['candidate'],
        best_diagnostic=decision['report_best_candidate'], maximum_metric_difference=maximum_difference,
        maximum_csv_difference=csv_max_difference), sort_keys=True))
    return report


def preflight(output):
    import copy
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    checks = []
    rng = np.random.default_rng(1701)
    entries = []
    for profile in PROFILES:
        for document in range(3):
            for role in ROLES:
                for position in [0, 256, 512, 900]:
                    entries.append(dict(profile=profile, document_id=document, role=role, split='validation',
                        wrapper='native', prompt_ix=len(entries)//4, message_ix=0, token_ix=20+position))
    rows = pd.DataFrame(entries)
    p = rng.dirichlet(np.ones(5), size=len(rows)).astype(np.float32)
    ix = np.arange(len(rows), dtype=np.int64)
    validate_probabilities(p, ix, ix)
    metrics, threshold = independent_validation(rows, p)
    checks.append('synthetic five profiles and body512 tail')
    y = labels(rows); t = metrics['system_threshold']; detected = p[:, 0] >= t
    one = independent_metrics(p, y, t)
    require(one['confusion'] == confusion_matrix(y, p.argmax(1), labels=range(5)).tolist(), 'sklearn confusion mismatch.')
    require(abs(one['system_f1'] - f1_score(y == 0, detected, zero_division=0)) < 1e-15
            and abs(one['system_balanced_accuracy'] - balanced_accuracy_score(y == 0, detected)) < 1e-15
            and abs(one['system_recall'] - recall_score(y == 0, detected)) < 1e-15, 'sklearn binary metric mismatch.')
    if one['system_precision'] is not None:
        require(abs(one['system_precision'] - precision_score(y == 0, detected, zero_division=0)) < 1e-15, 'sklearn precision mismatch.')
    checks.append('independent counts versus sklearn multiclass and binary metrics')
    # Brute force every threshold to independently verify the prefix-count implementation.
    grid = np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(p[:,0],np.linspace(0,1,301)),1.000001])
    objectives = []
    for cut in grid:
        values=[];feasible=True
        for profile in PROFILES:
            mask=rows.profile.to_numpy()==profile; truth=y[mask]==0; detect=p[mask,0]>=cut
            tp=int(np.count_nonzero(truth & detect));fp=int(np.count_nonzero(~truth & detect));fn=int(np.count_nonzero(truth & ~detect))
            values.append(2*tp/max(2*tp+fp+fn,1));feasible &= fp/max(int((~truth).sum()),1)<=.05
        objectives.append(float(np.mean(values)) if feasible else -1.)
    expected=float(grid[int(np.argmax(objectives))])
    require(t==expected and threshold['objective']==max(objectives), 'Brute-force global threshold mismatch.')
    checks.append('all-grid brute force threshold/FPR/first tie equals sorted-prefix method')
    def reject(fn,name):
        try:fn()
        except (RuntimeError,ValueError,KeyError):checks.append(name)
        else:raise RuntimeError('Fault accepted: '+name)
    reject(lambda:validate_probabilities(p,ix[::-1],ix),'reject probability sample order drift')
    bad=p.copy();bad[0,0]=np.nan
    reject(lambda:validate_probabilities(bad,ix,ix),'reject probability NaN')
    bad=p.copy();bad[0]*=.5
    reject(lambda:validate_probabilities(bad,ix,ix),'reject probability normalization failure')
    reject(lambda:validate_probabilities(p.astype(np.float64),ix,ix),'reject probability dtype drift')
    baseline=copy.deepcopy(metrics)
    for value in baseline['profiles'].values():value.update(system_f1=.7,accuracy=.8,system_false_positive_rate=.02)
    baseline['long_full_tail']['system_f1']=.6;baseline['mean_system_f1']=.7
    candidate=copy.deepcopy(baseline);candidate['mean_system_f1']=.71
    record_list=[dict(candidate=key,validation=copy.deepcopy(candidate)) for key in CANDIDATES]
    decision=independent_selection(record_list,baseline)
    require(decision['candidate']=='pooled-lda' and decision['promoted'], 'Fixed tie order differs.')
    checks.append('guard-passing positive gain and fixed candidate tie order')
    boundary=copy.deepcopy(candidate);boundary['profiles']['single']['system_f1']-=.02;boundary['profiles']['multi']['accuracy']-=.01
    require(independent_guard(boundary,baseline)['eligible'],'Exact2pp/1pp boundary rejected.')
    checks.append('guard decimal boundaries use1e-12 tolerance')
    for name,mutate in [('F1 loss',lambda r:r['profiles']['single'].update(system_f1=.679)),
        ('fiveacc loss',lambda r:r['profiles']['natural'].update(accuracy=.789)),
        ('full regression',lambda r:r['profiles']['long-full'].update(system_f1=.69)),
        ('tail regression',lambda r:r['long_full_tail'].update(system_f1=.59)),
        ('FPR',lambda r:r['profiles']['multi'].update(system_false_positive_rate=.051))]:
        bad=copy.deepcopy(candidate);mutate(bad);require(not independent_guard(bad,baseline)['eligible'],'Guard accepted '+name);checks.append('reject '+name)
    failed=copy.deepcopy(candidate);failed['long_full_tail']['system_f1']=.5
    noeligible=independent_selection([dict(candidate=k,validation=failed) for k in CANDIDATES],baseline)
    require(not noeligible['promoted'] and not noeligible['candidate_is_guard_eligible'],'Diagnostic claimed promoted.')
    checks.append('no eligible candidate remains diagnostic')
    unchanged=independent_selection([dict(candidate=k,validation=baseline) for k in CANDIDATES],baseline)
    require(not unchanged['promoted'],'Equal baseline promoted.')
    checks.append('strict positive mean gain required')
    history=[dict(epoch=e,instances=FIT_ROWS,steps=STEPS,cumulative_steps=e*STEPS,loss=1.,
        mean_system_f1=(.69 if e==1 else .71 if e==2 else .70),long_full_f1=.7,tail_f1=.6) for e in range(1,11)]
    record=dict(history=history,epochs=10,best_epoch=2,best_steps=2*STEPS,actual_steps=10*STEPS,
        actual_instances=10*FIT_ROWS,instances_per_epoch=FIT_ROWS,steps_per_epoch=STEPS,last_batch_instances=554,
        seed=SEED,validation=candidate)
    metadata=dict(best_epoch=2,best_steps=2*STEPS)
    independent_budget(record,metadata);checks.append('MLP142steps554last early-stop and best checkpoint')
    bad=copy.deepcopy(record);bad['actual_steps']+=1
    reject(lambda:independent_budget(bad,metadata),'reject MLP update budget drift')
    bad=copy.deepcopy(record);bad['history'].append(dict(bad['history'][-1],epoch=11,cumulative_steps=11*STEPS))
    bad.update(epochs=11,actual_steps=11*STEPS,actual_instances=11*FIT_ROWS)
    reject(lambda:independent_budget(bad,metadata),'reject MLP continued beyond patience')
    bad=copy.deepcopy(record);bad['best_epoch']=3
    reject(lambda:independent_budget(bad,metadata),'reject MLP incorrect best checkpoint')
    # Verify actual-mode gate ordering with fake in-memory modules; no formal path is read.
    from types import SimpleNamespace
    from unittest.mock import patch
    root=Path(tempfile.mkdtemp(prefix='four-head-audit-gate-',dir='/tmp'))
    never=root/'never-written.json';seen=[]
    def unavailable():
        seen.append('verified_completion')
        raise RuntimeError('synthetic incomplete four-candidate upstream')
    fake=SimpleNamespace(verified_completion=unavailable)
    with patch.dict(sys.modules,{'systemness_head_probes':fake}):
        reject(lambda:actual(never),'actual gate rejects incomplete upstream before paths/results')
    require(seen==['verified_completion'] and not never.exists(),'Actual gate wrote output or skipped completion.')
    fake=SimpleNamespace(verified_completion=lambda:{},CONFIGS={k:None for k in CANDIDATES},
        data=SimpleNamespace(ROLES=ROLES),OUT=root/'empty-out',CACHE=root/'empty-cache')
    with patch.dict(sys.modules,{'systemness_head_probes':fake}):
        reject(lambda:actual(never),'actual requires every candidate model/probability/record file')
    require(not never.exists() and not fake.OUT.exists() and not fake.CACHE.exists(),'Incomplete files created formal paths.')
    report=dict(stage='synthetic_audit_preflight_only',source_sha256=sha_file(__file__),checks=checks,
        checks_passed=len(checks),rows=len(rows),tail_rows=len(tail_indices(rows)[0]),threshold_audit=threshold,
        gpu_used=False,formal_results_read=False,test_data_read=False,actual_four_candidate_results_claimed=False)
    write_tmp(output,report);print(json.dumps(dict(output=str(output),checks_passed=len(checks)),sort_keys=True))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--actual',action='store_true');mode.add_argument('--preflight',action='store_true')
    parser.add_argument('--output',type=Path,help='JSON output under /tmp only')
    args=parser.parse_args()
    if args.actual:actual(args.output or Path('/tmp/systemness-four-head-actual-audit.json'))
    elif args.preflight:preflight(args.output or Path('/tmp/systemness-four-head-audit-preflight.json'))
    else:parser.print_help()


if __name__=='__main__':main()
