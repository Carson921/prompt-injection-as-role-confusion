"""Independent CPU audit of the completed four-readout C4 confirmation.

Default: help. --preflight reads synthetic fixtures only. --actual is required
before any current confirmation metadata/probability/report artifact is opened.
The parent must wait for session53992 terminal exit0 before using --actual.
This script never runs GPU/model inference or changes a production artifact.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata


REPO = Path('/root/prompt-injection-as-role-confusion')
SOURCE_DIR = REPO / 'experiments/role-analysis'
NAMES = ['residual', 'base-attention', 'dense128', 'head-selected']
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
PROFILES = ['single', 'multi', 'long-single', 'long-full', 'long-full-tail']
BINS = [(0, 31), (32, 63), (64, 127), (128, 255), (256, 511), (512, 1023)]
FIELDS = ['tp', 'fp', 'fn', 'tn', 'correct', 'tokens']
BOOTSTRAPS = 2000
SEED = 20261021 + 11464
PRIMARY = 'head-selected'
PRIMARY_SHA = '27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
PRIMARY_THRESHOLD = .620969696969697


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''): h.update(block)
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, dict): return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def assert_same(actual, expected, label, atol=2e-6):
    if expected is None or (isinstance(expected, (float, np.floating)) and np.isnan(expected)):
        require(actual is None or (isinstance(actual, (float, np.floating)) and np.isnan(actual)), label + ': expected N/A')
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': dictionary fields')
        for key in expected: assert_same(actual[key], expected[key], label + '/' + str(key), atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ': sequence length')
        for index, (a, e) in enumerate(zip(actual, expected)): assert_same(a, e, label + '/' + str(index), atol)
    elif isinstance(expected, (float, np.floating)):
        require(np.isfinite(actual) and np.isclose(actual, expected, atol=atol, rtol=1e-8),
            f'{label}: {actual!r} != {expected!r}')
    else: require(actual == expected, f'{label}: {actual!r} != {expected!r}')


def labels(rows):
    mapping = {role: i for i, role in enumerate(ROLES)}
    require(set(rows.role) <= set(mapping), 'Unknown truth role.')
    return rows.role.map(mapping).to_numpy(dtype=np.int64)


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def independently_rank_metrics(truth, scores):
    positives = int(truth.sum()); negatives = len(truth) - positives
    auc = None
    if positives and negatives:
        ranks = rankdata(scores, method='average')
        auc = float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    ap = None
    if positives:
        order = np.argsort(-scores, kind='stable'); sorted_scores = scores[order]
        cumulative = np.cumsum(truth[order], dtype=np.int64)
        endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
        tp = cumulative[endpoints]; previous = np.r_[0, tp[:-1]]
        ap = float(np.sum((tp - previous) / positives * tp / (endpoints + 1)))
    return auc, ap


def token_metrics(p, y, threshold):
    """Independent direct definitions; no production metric function is used."""
    p = np.asarray(p); y = np.asarray(y)
    require(p.shape == (len(y), 5) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        and np.allclose(p.sum(1), 1, atol=2e-5, rtol=0)
        and np.issubdtype(y.dtype, np.integer) and ((y >= 0) & (y < 5)).all()
        and np.isfinite(threshold) and 0 <= threshold <= 1.00001,
        'Invalid five-class probabilities.')
    pred = np.argmax(p, axis=1); detected = p[:, 0] >= threshold; truth = y == 0
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    tp = int(np.count_nonzero(truth & detected)); fp = int(np.count_nonzero(~truth & detected))
    fn = int(np.count_nonzero(truth & ~detected)); tn = int(np.count_nonzero(~truth & ~detected))
    row_support = confusion.sum(1); column_support = confusion.sum(0); diagonal = confusion.diagonal()
    f1 = [2 * diagonal[i] / (row_support[i] + column_support[i]) if row_support[i] + column_support[i] else 0
          for i in range(5)]
    recall = divide(tp, tp + fn); specificity = divide(tn, tn + fp)
    auc, ap = independently_rank_metrics(truth, p[:, 0])
    return dict(tokens=len(y), accuracy=divide(int(np.count_nonzero(pred == y)), len(y)),
        macro_f1=float(np.mean([f1[i] for i in range(5) if row_support[i]])) if len(y) else None,
        nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()) if len(y) else None,
        recall={role: divide(int(diagonal[i]), int(row_support[i])) for i, role in enumerate(ROLES)},
        confusion=confusion.tolist(), system_threshold=float(threshold),
        system_accuracy=divide(tp + tn, len(y)),
        system_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
        system_precision=divide(tp, tp + fp), system_recall=recall, system_f1=divide(2 * tp, 2 * tp + fp + fn),
        system_false_positive_rate=divide(fp, fp + tn), system_auroc=auc, system_average_precision=ap,
        system_detection_by_role={role: float(detected[y == i].mean()) if np.any(y == i) else None
            for i, role in enumerate(ROLES)}, system_confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn))


def count_documents(rows, p, threshold, name, profile=None):
    y = labels(rows); truth = y == 0; detected = p[:, 0] >= threshold
    frame = rows[['group_id', 'profile', 'wrapper']].copy()
    if profile is not None: frame['profile'] = profile
    flags = dict(tp=truth & detected, fp=~truth & detected, fn=truth & ~detected, tn=~truth & ~detected,
        correct=np.argmax(p, 1) == y, tokens=np.ones(len(y), dtype=np.int64))
    for field, value in flags.items(): frame[field] = np.asarray(value, dtype=np.int64)
    return frame.groupby(['profile', 'wrapper', 'group_id'], as_index=False, sort=True)[FIELDS].sum().assign(model=name)


def independently_summarize(rows, probabilities, thresholds, strict=True):
    require(np.array_equal(rows.index, np.arange(len(rows))), 'Global probability row order must be RangeIndex.')
    require(set(probabilities) == set(thresholds) == set(NAMES), 'Four frozen model columns missing.')
    if strict:
        require(rows.groupby('profile').group_id.nunique().to_dict() ==
            {'single': 400, 'multi': 400, 'long-single': 64, 'long-full': 64}, 'Independent profile source counts.')
        require(rows.group_id.nunique() == 464 and set(rows.document_id) == set(range(464)), 'Independent464 source IDs.')
        require(set(rows.loc[rows.profile == 'long-single', 'document_id']) == set(range(64))
            and set(rows.loc[rows.profile == 'long-full', 'document_id']) == set(range(400, 464)),
            'Paired64 variants versus disjoint64 source cohorts.')
    metrics = []; documents = []; positions = []; projections = []
    full_tail = rows[(rows.profile == 'long-full') & rows.body_offset.between(512, 1023)]
    if strict: require(len(full_tail) == 64 * 5 * 512 and full_tail.group_id.nunique() == 64, 'Exact full-tail support.')
    for name in NAMES:
        p = probabilities[name]
        for profile in PROFILES:
            group = full_tail if profile == 'long-full-tail' else rows[rows.profile == profile]
            indices = group.index.to_numpy()
            if group.empty: continue
            result = token_metrics(p[indices], labels(group), thresholds[name])
            metrics.append(dict(model=name, profile=profile, groups=int(group.group_id.nunique()), **result))
        documents.extend([count_documents(rows, p, thresholds[name], name),
            count_documents(full_tail, p[full_tail.index], thresholds[name], name, 'long-full-tail')])
        projection = rows[['group_id', 'profile', 'message_ix', 'role']].copy()
        for ri, role in enumerate(ROLES): projection['p_' + role] = p[:, ri]
        projections.append(projection.groupby(['group_id', 'profile', 'message_ix', 'role'], as_index=False).agg(
            **{'p_' + role: ('p_' + role, 'mean') for role in ROLES}, tokens=('p_system', 'size')).assign(model=name))
        for profile, group in rows.groupby('profile', sort=True):
            for start, end in BINS:
                subset = group[group.body_offset.between(start, end)]
                if subset.empty: continue
                system = subset[subset.role == 'system']; indices = subset.index.to_numpy()
                positions.append(dict(model=name, profile=profile, start=start, end=end,
                    groups=int(subset.group_id.nunique()), system_tokens=len(system), system_groups=int(system.group_id.nunique()),
                    mean_system_probability=float(p[system.index, 0].mean()) if len(system) else None,
                    **token_metrics(p[indices], labels(subset), thresholds[name])))
                if profile == 'long-full': documents.append(count_documents(subset, p[indices], thresholds[name], name,
                    f'long-full-position-{start}-{end}'))
    return metrics, pd.concat(documents, ignore_index=True), pd.concat(projections, ignore_index=True), positions


def count_metrics(counts, zero_compat=False):
    """Mathematical undefined values are NaN; published bootstrap uses zero."""
    a = np.asarray(counts, dtype=np.float64)
    tp, fp, fn, tn, correct, total = np.moveaxis(a, -1, 0)
    def ratio(n, d):
        return np.divide(n, d, out=np.full_like(n, 0. if zero_compat else np.nan, dtype=float), where=d != 0)
    return dict(system_precision=ratio(tp, tp + fp), system_recall=ratio(tp, tp + fn),
        system_f1=ratio(2 * tp, 2 * tp + fp + fn),
        system_balanced_accuracy=(ratio(tp, tp + fn) + ratio(tn, tn + fp)) / 2,
        system_false_positive_rate=ratio(fp, fp + tn), accuracy=ratio(correct, total))


def independent_bootstrap(document, strict=True):
    comparisons = []; intervals = []; undefined = []
    for pi, profile in enumerate(sorted(document.profile.unique())):
        group = document[document.profile == profile]
        arrays = {}; ids = None
        for name in NAMES:
            frame = group[group.model == name].sort_values('group_id')
            require(len(frame) > 0 and frame.group_id.is_unique, 'Missing/duplicate original-source bootstrap unit.')
            if ids is None: ids = frame.group_id.tolist()
            require(frame.group_id.tolist() == ids, 'Unpaired model source membership.')
            arrays[name] = frame[FIELDS].to_numpy(dtype=np.float64)
        if strict: require(len(ids) == (400 if profile in ['single', 'multi'] else 64), 'Bootstrap source-count inflation.')
        # The public report fixes RandomState, profile alphabetical order and
        # seed offset. Reproduce those draws, independently calculate metrics.
        generator = np.random.RandomState(SEED + pi)
        draws_ix = generator.randint(0, len(ids), size=(BOOTSTRAPS, len(ids)))
        points = {}; draws = {}
        for name, array in arrays.items():
            summed = array.sum(0); sampled = array[draws_ix].sum(1)
            points[name] = count_metrics(summed, zero_compat=True)
            draws[name] = count_metrics(sampled, zero_compat=True)
            mathematical_point = count_metrics(summed); mathematical_draws = count_metrics(sampled)
            for metric, value in points[name].items():
                missing = int(np.count_nonzero(~np.isfinite(mathematical_draws[metric])))
                if missing or not np.isfinite(mathematical_point[metric]):
                    undefined.append(dict(profile=profile, model=name, metric=metric,
                        mathematical_point=None if not np.isfinite(mathematical_point[metric]) else float(mathematical_point[metric]),
                        undefined_resamples=missing, bootstrap_replicates=BOOTSTRAPS,
                        published_compatibility='Production bootstrap fills undefined ratios with0; audit labels the mathematical value N/A.'))
                lo, hi = np.quantile(draws[name][metric], [.025, .975])
                intervals.append(dict(profile=profile, model=name, metric=metric, source_documents=len(ids),
                    bootstrap_replicates=BOOTSTRAPS, value=float(value), ci95_low=float(lo), ci95_high=float(hi)))
        for before, after in itertools.combinations(NAMES, 2):
            for metric in points[before]:
                lo, hi = np.quantile(100 * (draws[after][metric] - draws[before][metric]), [.025, .975])
                comparisons.append(dict(profile=profile, baseline_model=before, comparison_model=after, metric=metric,
                    source_documents=len(ids), bootstrap_replicates=BOOTSTRAPS,
                    baseline=float(points[before][metric]), comparison=float(points[after][metric]),
                    gain_percentage_points=float(100 * (points[after][metric] - points[before][metric])),
                    ci95_low=float(lo), ci95_high=float(hi)))
    return comparisons, intervals, undefined


def compare_records(actual, expected, keys, label, atol=2e-6):
    def keyed(records):
        mapping = {tuple(row[key] for key in keys): row for row in records}
        require(len(mapping) == len(records), label + ': duplicated record identity')
        return mapping
    a, e = keyed(actual), keyed(expected)
    require(set(a) == set(e), label + ': missing/extra record identities')
    for identity in e: assert_same(a[identity], e[identity], label + '/' + str(identity), atol)


def compare_frame(actual, expected, keys, label, atol=2e-6):
    require(set(actual.columns) == set(expected.columns), label + ': table fields differ')
    compare_records(actual.to_dict('records'), expected.to_dict('records'), keys, label, atol)


def flat_metrics(records):
    return pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        | {'role_recall_' + key: value for key, value in row.get('recall', {}).items()} for row in records])


def verify_text_report(text, metrics):
    """Check displayed20 rows against independent values rounded to6 decimals."""
    expected = {(row['model'], row['profile']): row for row in metrics}
    found = {}
    for line in text.splitlines():
        cells = line.split()
        if len(cells) < 2 or cells[0] not in NAMES or cells[1] not in PROFILES: continue
        key = (cells[0], cells[1])
        require(key in expected and key not in found and len(cells) == 8, 'Text report row identity/columns.')
        values = [expected[key][field] for field in ['system_precision', 'system_recall', 'system_f1',
            'system_balanced_accuracy', 'accuracy']] + [expected[key]['recall']['cot']]
        for cell, value in zip(cells[2:], values):
            if value is None: require(cell == 'NA', 'Text report unsupported value must display NA.')
            else:
                require(re.fullmatch(r'-?\d+\.\d{6}', cell) is not None
                    and abs(float(cell) - value) <= .500001e-6, 'Text report six-decimal value differs from independent math.')
        found[key] = True
    require(set(found) == set(expected) and len(found) == 20, 'Text report must display all20 model/profile rows.')
    return len(found)


def verify_plot_files(folder, gallery, document_ids, fixture=False):
    from PIL import Image
    require(len(document_ids) == 12 and len(set(document_ids)) == 12, 'Fixed12 gallery IDs.')
    require(gallery['documents'] == document_ids and gallery['five_role_panels'] == 5 and gallery['model_columns'] == 4
        and gallery['png'] == gallery['pdf'] == 13 and gallery['independent_source_documents'] == 464
        and gallery['unsmoothed_probabilities'] is True and gallery['natural_dialogues'] is False,
        'Gallery model/panel/source/provenance declaration.')
    stems = {f'head-confirm-c4-{doc}' for doc in document_ids} | {'long-full-system-by-position'}
    for extension in ['png', 'pdf']:
        require({path.stem for path in folder.glob('*.' + extension)} == stems, 'Missing/extra ' + extension + ' figure.')
    info = []
    for stem in sorted(stems):
        png = folder / (stem + '.png'); pdf = folder / (stem + '.pdf')
        with Image.open(png) as image:
            image.verify()
        with Image.open(png) as image:
            expected = (1280, 560) if stem == 'long-full-system-by-position' else (3200, 1120)
            if not fixture: require(image.size == expected, 'Figure dimensions do not match declared5x4/position layout.')
            dimensions = list(image.size)
        payload = pdf.read_bytes()
        require(payload.startswith(b'%PDF-') and b'%%EOF' in payload[-1024:]
            and len(re.findall(rb'/Type\s*/Page\b', payload)) == 1, 'Malformed/non-single-page PDF figure.')
        info.append(dict(stem=stem, png_sha256=digest(png), pdf_sha256=digest(pdf), dimensions=dimensions))
    return info


def verify_complete_artifacts(folder, complete):
    require(complete['primary_model'] == PRIMARY and complete['primary_changed'] is False
        and complete['tests_used_for_tuning'] is False and complete['validation_promotion_required'] is True
        and complete['independent_source_documents'] == 464 and complete['general_documents'] == 400
        and complete['additional_full_length_documents'] == complete['up_to1024_subset_documents'] == 64
        and complete['bootstrap_replicates'] == BOOTSTRAPS and complete['bootstrap_unit'] == 'Original source document',
        'Complete source/selection/bootstrap flags.')
    required = {'metrics.json', 'metrics.csv', 'document-counts.csv', 'message-projections.csv.gz',
        'system-by-position.json', 'system-by-position.csv', 'paired-comparisons.csv', 'bootstrap-metrics.csv',
        'report.txt', 'gallery.json', 'plots/gallery-index.md', 'data-complete.json', 'inference-complete.json',
        'portable-validation.json', 'hook-validation.json', 'model_backend.json', 'protocol.json', 'frozen-readouts.json'}
    expected_stems = {f'head-confirm-c4-{int(i)}' for i in np.linspace(0, 399, 12, dtype=int)} | {'long-full-system-by-position'}
    required |= {'plots/' + stem + '.' + ext for stem in expected_stems for ext in ['png', 'pdf']}
    require(required <= set(complete['artifact_sha256']), 'Completion manifest omitted a required report/evidence/plot.')
    for name, sha in complete['artifact_sha256'].items():
        path = (folder / name).resolve()
        require(folder.resolve() in path.parents and path.is_file(), 'Unsafe/missing completion artifact: ' + name)
        require(digest(path) == sha, 'Complete artifact hash mismatch: ' + name)
    require(digest(folder / 'inference-complete.json') == complete['inference_sha256']
        and digest(folder / 'report.txt') == complete['report_sha256'], 'Complete inference/report direct SHA.')
    return len(complete['artifact_sha256'])


def actual():
    """Only this explicitly selected function can touch current C4 artifacts."""
    sys.path.insert(0, str(SOURCE_DIR))
    import systemness_head_confirm as h
    # These two terminal markers are deliberately checked before load_data or
    # completed_probabilities can open a source/probability cache.
    require((h.OUT / 'complete.json').is_file(), 'Confirmation has no terminal complete.json; do not inspect partial results.')
    require(read(h.OUT / 'status.json')['stage'] == 'complete', 'Confirmation status is not terminal complete.')
    complete = read(h.OUT / 'complete.json')
    require(h.NAMES == NAMES and h.BOOTSTRAPS == BOOTSTRAPS and h.data.ROLES == ROLES
        and h.data.SEED + 11464 == SEED, 'Frozen audit specification constants changed.')
    metadata, inputs, rows, tokens = h.load_data()
    inference, probability = h.completed_probabilities(metadata, rows)
    frozen = metadata['frozen']; thresholds = frozen['thresholds']
    require(frozen['models'][PRIMARY] == PRIMARY_SHA and thresholds[PRIMARY] == PRIMARY_THRESHOLD
        and complete['frozen'] == frozen and complete['source_sha256'] == digest(h.__file__), 'Exact frozen primary identity.')
    artifact_count = verify_complete_artifacts(h.OUT, complete)
    require(inference['all_finite'] is True and inference['all_normalized'] is True
        and inference['raw_hidden_states_persisted'] is False, 'Inference completion flags.')
    # Independent calculations below never call h.summarize/bootstrap,
    # probes.metrics/labels or evaluation.binary_values/document_counts.
    metrics, documents, messages, positions = independently_summarize(rows, probability, thresholds)
    require(len(metrics) == 20, 'Main table must contain4models ×5groups =20 rows.')
    compare_records(read(h.OUT / 'metrics.json'), metrics, ['model', 'profile'], 'Main independent metrics')
    compare_frame(pd.read_csv(h.OUT / 'metrics.csv'), flat_metrics(metrics), ['model', 'profile'], 'Main CSV')
    compare_frame(pd.read_csv(h.OUT / 'document-counts.csv'), documents,
        ['model', 'profile', 'wrapper', 'group_id'], 'Original-source document counts', atol=0)
    compare_frame(pd.read_csv(h.OUT / 'message-projections.csv.gz'), messages,
        ['model', 'profile', 'group_id', 'message_ix', 'role'], 'Message probability means')
    compare_records(read(h.OUT / 'system-by-position.json'), positions,
        ['model', 'profile', 'start', 'end'], 'Independent position metrics')
    compare_frame(pd.read_csv(h.OUT / 'system-by-position.csv'), flat_metrics(positions),
        ['model', 'profile', 'start', 'end'], 'Position CSV')
    comparisons, intervals, undefined = independent_bootstrap(documents)
    compare_frame(pd.read_csv(h.OUT / 'paired-comparisons.csv'), pd.DataFrame(comparisons),
        ['profile', 'baseline_model', 'comparison_model', 'metric'], 'Six-pair2000-source bootstrap', atol=1e-9)
    compare_frame(pd.read_csv(h.OUT / 'bootstrap-metrics.csv'), pd.DataFrame(intervals),
        ['profile', 'model', 'metric'], '2000-source bootstrap intervals', atol=1e-9)
    require(len(comparisons) == 11 * 6 * 6 and len(intervals) == 11 * 4 * 6,
        'Five primary plus six full-length position bootstrap groups.')
    text_rows = verify_text_report((h.OUT / 'report.txt').read_text(), metrics)
    expected_ids = [int(i) for i in np.linspace(0, 399, 12, dtype=int)]
    plots = verify_plot_files(h.OUT / 'plots', read(h.OUT / 'gallery.json'), expected_ids)
    gallery_index = (h.OUT / 'plots/gallery-index.md').read_text()
    require(all(f'head-confirm-c4-{doc}.png' in gallery_index for doc in expected_ids), 'Gallery index missed fixed IDs.')
    flags = {row['profile']: all(row[key] is not None and row[key] >= .9 for key in
        ['system_precision', 'system_recall', 'system_balanced_accuracy']) for row in metrics if row['model'] == PRIMARY}
    require(complete['working_target_pass'] == flags
        and complete['all_profiles_and_full_tail_meet_90_percent'] == all(flags.values()), 'Working90 flags differ from independent metrics.')
    status = read(h.OUT / 'status.json')
    require(status['targets'] == flags and status['primary_changed'] is False, 'Terminal status flags.')
    result = dict(auditor_sha256=digest(__file__), confirmation_source_sha256=digest(h.__file__),
        complete_sha256=digest(h.OUT / 'complete.json'), inference_sha256=digest(h.OUT / 'inference-complete.json'),
        actual_completed_confirmation_audited=True, production_artifacts_changed=False,
        verification_apis=['h.load_data', 'h.completed_probabilities'], independent_statistical_math=True,
        primary_model_sha256=PRIMARY_SHA, primary_threshold=PRIMARY_THRESHOLD,
        independent_sources=464, general_sources=400, disjoint_full_length_sources=64,
        paired_general_long_variants=64, inputs=len(inputs), probability_rows=len(rows),
        main_metrics_rows=len(metrics), document_count_rows=len(documents), message_projection_rows=len(messages),
        text_report_numeric_rows_verified=text_rows,
        position_rows=len(positions), bootstrap_comparisons=len(comparisons), bootstrap_intervals=len(intervals),
        all_six_pairs=True, source_paired_bootstrap_replicates=BOOTSTRAPS,
        bootstrap_undefined_ratios=undefined, unsupported_roles_reported_NA=True,
        source_counts_by_bootstrap_profile=documents.groupby('profile').group_id.nunique().to_dict(),
        metrics=metrics, position_metrics=positions, working_target_pass=flags,
        all_profiles_and_full_tail_meet_90_percent=all(flags.values()), plots=plots,
        complete_artifacts_verified=artifact_count, hash_portable_backend_hook_chain_verified=True,
        gpu_work_run=False, models_thresholds_modified=False)
    output = Path(tempfile.mkdtemp(prefix='long-data-head-confirmation-actual-audit-', dir='/tmp')) / 'report.json'
    output.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))
    print(json.dumps(dict(report_path=str(output), report_sha256=digest(output), main_metrics_rows=len(metrics),
        bootstrap_comparisons=len(comparisons), working_target_pass=flags, actual_completed_confirmation_audited=True)), flush=True)


def preflight():
    """No production module import and no current OUT read in this path."""
    folder = Path(tempfile.mkdtemp(prefix='long-data-head-confirmation-audit-preflight-', dir='/tmp'))
    # Known hand count: y=[S,S,U,U,A,T], threshold=.5 gives tp1/fp1/fn1/tn3.
    p = np.array([[.8,.05,.05,.05,.05], [.2,.7,.02,.04,.04], [.6,.2,.1,.05,.05],
        [.1,.8,.02,.04,.04], [.1,.05,.05,.75,.05], [.1,.05,.05,.05,.75]], dtype=np.float64)
    y = np.array([0,0,1,1,3,4])
    m = token_metrics(p, y, .5)
    require(m['system_confusion'] == {'tp':1,'fn':1,'fp':1,'tn':3}
        and m['system_precision'] == m['system_recall'] == m['system_f1'] == .5
        and m['system_balanced_accuracy'] == .625 and m['system_false_positive_rate'] == .25
        and m['accuracy'] == 4/6 and m['recall']['cot'] is None, 'Hand-count/N/A fixture.')
    require(m['system_auroc'] == .875 and np.isclose(m['system_average_precision'], 5/6),
        'Hand-computed AUROC/AP fixture.')
    invalid = p.copy(); invalid[0] = [1.2, -.2, 0, 0, 0]
    for bad_p, bad_y, bad_threshold in [(invalid, y, .5), (p, np.full_like(y, 7), .5), (p, y, np.nan)]:
        try: token_metrics(bad_p, bad_y, bad_threshold)
        except RuntimeError: pass
        else: raise RuntimeError('Invalid probabilities/labels/threshold accepted.')
    na_frame = flat_metrics([dict(model='synthetic', profile='unsupported-cot', **m)])
    na_frame.to_csv(folder/'NA-metrics.csv', index=False)
    compare_frame(pd.read_csv(folder/'NA-metrics.csv'), na_frame, ['model','profile'], 'N/A CSV fixture')
    tie_auc, tie_ap = independently_rank_metrics(np.array([True,False,True,False]), np.ones(4))
    require(tie_auc == tie_ap == .5, 'Tie-correct AUC/AP fixture.')
    missing = count_metrics(np.array([0,0,0,3,2,3]))
    require(np.isnan(missing['system_precision']) and np.isnan(missing['system_recall'])
        and np.isnan(missing['system_balanced_accuracy']), 'Zero-support N/A fixture.')
    # Complete464 source geometry, compressed single/multi text but authentic
    # exact1024 full-length/tail support and all six position bins.
    pieces = []
    for profile, docs, length in [('single',range(400),2),('multi',range(400),2),
        ('long-single',range(64),8),('long-full',range(400,464),1024)]:
        for doc in docs:
            for ri, role in enumerate(ROLES):
                pieces.append(pd.DataFrame(dict(group_id=f'head-confirm-c4-{doc}', document_id=doc,
                    profile=profile, wrapper='native', role=role, message_ix=ri,
                    body_offset=np.arange(length), token_ix=np.arange(length)+ri*3)))
    rows = pd.concat(pieces, ignore_index=True); truth = labels(rows)
    probabilities = {}
    for mi, name in enumerate(NAMES):
        matrix = np.full((len(rows),5),.04,dtype=np.float32)
        matrix[np.arange(len(rows)),truth]=.84
        bad = (rows.profile.eq('long-full') & rows.role.eq('system') & rows.body_offset.ge(960+16*mi)).to_numpy()
        matrix[bad,0]=.2; matrix[bad,1]=.68
        probabilities[name]=matrix
    thresholds = {name:.5 for name in NAMES}
    metrics, documents, messages, positions = independently_summarize(rows,probabilities,thresholds)
    require(len(metrics)==20 and len(documents)==5504 and len(positions)>0, 'Full geometry/table fixture.')
    text = '\n'.join(row['model'] + '  ' + row['profile'] + '  ' + '  '.join(
        'NA' if value is None else f'{value:.6f}' for value in [row[field] for field in
        ['system_precision','system_recall','system_f1','system_balanced_accuracy','accuracy']] + [row['recall']['cot']])
        for row in metrics)
    require(verify_text_report(text, metrics) == 20, 'Text display fixture.')
    for changed in [text.replace('.000000', '.123456', 1), '\n'.join(text.splitlines()[:-1])]:
        try: verify_text_report(changed, metrics)
        except RuntimeError: pass
        else: raise RuntimeError('Wrong/missing text display metric row accepted.')
    for mi,name in enumerate(NAMES):
        tail = next(row for row in metrics if row['model']==name and row['profile']=='long-full-tail')
        require(tail['tokens']==64*5*512 and tail['system_recall']==1-(64-mi*16)/512,
            'Global-index tail/model-column fixture.')
    comparisons,intervals,undefined=independent_bootstrap(documents)
    require(len(comparisons)==396 and len(intervals)==264, 'All six pair/profile/metric bootstrap coverage.')
    require(all(row['source_documents']==(400 if row['profile'] in ['single','multi'] else 64)
        and row['bootstrap_replicates']==2000 for row in comparisons), 'Source count bootstrap fixture.')
    identical=documents.copy(); identical['model']='residual'
    identical=pd.concat([identical[identical.model=='residual'].drop_duplicates(['profile','wrapper','group_id']).assign(model=name)
        for name in NAMES],ignore_index=True)
    same_comparisons,_,_=independent_bootstrap(identical)
    require(all(row['gain_percentage_points']==row['ci95_low']==row['ci95_high']==0 for row in same_comparisons),
        'Paired identical model bootstrap must yield exact0 deltas.')
    for faulty in [pd.concat([documents,documents.iloc[:1]],ignore_index=True),documents.iloc[1:].copy()]:
        try: independent_bootstrap(faulty)
        except RuntimeError: pass
        else: raise RuntimeError('Duplicate/unpaired bootstrap source accepted.')
    # Persist/read fixtures only in/tmp to check CSV fields and None behavior.
    flat_metrics(metrics).to_csv(folder/'metrics.csv',index=False)
    documents.to_csv(folder/'document-counts.csv',index=False)
    compare_frame(pd.read_csv(folder/'metrics.csv'),flat_metrics(metrics),['model','profile'],'Fixture CSV')
    compare_frame(pd.read_csv(folder/'document-counts.csv'),documents,
        ['model','profile','wrapper','group_id'],'Fixture document CSV',atol=0)
    mutated=json_ready(metrics); mutated=json.loads(json.dumps(mutated)); mutated[0]['system_recall']+=.1
    try: compare_records(mutated,metrics,['model','profile'],'Deliberately wrong metrics')
    except RuntimeError: pass
    else: raise RuntimeError('Metric tamper not detected.')
    # Tiny standalone figures exercise fixed13+13 naming/magic/page checking.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=folder/'plots'; plots.mkdir()
    ids=[int(i) for i in np.linspace(0,399,12,dtype=int)]
    for stem in [f'head-confirm-c4-{doc}' for doc in ids]+['long-full-system-by-position']:
        fig,ax=plt.subplots(figsize=(2,1)); ax.plot([0,1],[0,1]); ax.set_title('Synthetic CPU only')
        for ext in ['png','pdf']:fig.savefig(plots/(stem+'.'+ext))
        plt.close(fig)
    gallery=dict(documents=ids,five_role_panels=5,model_columns=4,png=13,pdf=13,
        independent_source_documents=464,unsmoothed_probabilities=True,natural_dialogues=False)
    figures=verify_plot_files(plots,gallery,ids,fixture=True)
    (plots/'head-confirm-c4-0.png').write_bytes(b'broken')
    try:verify_plot_files(plots,gallery,ids,fixture=True)
    except Exception:pass
    else:raise RuntimeError('Corrupt PNG accepted.')
    report=dict(auditor_sha256=digest(__file__),synthetic_only=True,production_module_imported=False,
        current_confirmation_artifacts_read=False,actual_result_audited=False,gpu_work_run=False,
        hand_count_P_R_F1_BAcc_FPR_accuracy=True,unsupported_role_NA=True,tied_AUROC_AP=True,
        invalid_probability_label_threshold_guards=True,text_report20_rows_six_decimal_and_tamper_guards=True,
        exact_464_source_geometry=True,main_metrics_rows=20,document_count_rows=len(documents),
        bootstrap_comparisons=len(comparisons),bootstrap_intervals=len(intervals),
        all_six_pairs_2000_source_resamples=True,identical_pair_exact_zero=True,
        duplicate_and_unpaired_source_guards=True,metric_and_csv_tamper_guards=True,
        fixed13_PNG_and13_PDF_files=True,corrupt_plot_rejected=True,
        synthetic_probability_rows=len(rows),synthetic_message_projection_rows=len(messages),
        mathematical_undefined_bootstrap_ratios=undefined,formal_files_modified=False)
    path=folder/'report.json';path.write_text(json.dumps(json_ready(report),indent=2,allow_nan=False))
    print(json.dumps(dict(report_path=str(path),report_sha256=digest(path),**report)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='Synthetic CPU/tmp only; never opens current OUT.')
    action.add_argument('--actual',action='store_true',help='Parent only, after53992 terminal0: audit completed actual C4 artifacts.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual()
    else:parser.print_help()
