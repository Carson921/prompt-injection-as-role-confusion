"""Report a prospectively frozen four-record, three-model C4 confirmation.

Default help. --preflight uses synthetic CPU arrays and /tmp only. --report
requires complete shared inference; it never prepares sources, runs models,
chooses thresholds, selects examples by scores or changes the current primary.
The separate fresh natural companion is still required for the joint goal.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_probes as probes
import systemness_complementary_guard_confirm_snapshot as closed


OUT = data.runtime.OUT / 'systemness-complementary-confirmation/report'
PHYSICAL = ['new', 'matched', 'old-head']
RECORDS = ['new-guard-max33', 'matched-max33', 'oldhead-max33', 'oldhead-original']
RECORD_TO_PHYSICAL = {'new-guard-max33': 'new', 'matched-max33': 'matched',
    'oldhead-max33': 'old-head', 'oldhead-original': 'old-head'}
PRIMARY = 'new-guard-max33'
PROFILES = ['single', 'multi', 'long-single', 'long-full', 'long-full-tail']
BINS = [(0, 31), (32, 63), (64, 127), (128, 255), (256, 511), (512, 1023)]
BIN_NAMES = [f'long-full-bin-{left:04d}-{right:04d}' for left, right in BINS]
GROUPS = PROFILES + BIN_NAMES
FIELDS = ['tp', 'fp', 'fn', 'tn', 'correct', 'tokens']
METRIC_FIELDS = ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy',
    'system_false_positive_rate', 'accuracy']
BOOTSTRAPS, BOOTSTRAP_SEED = 2000, data.SEED + 528123
EXPECTED_DOCUMENTS = dict(single=400, multi=400, **{'long-single': 64, 'long-full': 128, 'long-full-tail': 128},
    **{name: 128 for name in BIN_NAMES})
COLORS = dict(system='#90a1b9', user='#00a6f4', cot='#fd9a00', assistant='#00d492', tool='#a855f7')
DEPENDENCIES = ['systemness_complementary_confirmation_report.py', 'systemness_complementary_confirmation_infer.py',
    'systemness_complementary_confirmation_data.py', 'systemness_complementary_guard_confirm_snapshot.py', 'systemness_data.py', 'systemness_probes.py']
LIMITATION = ('528 independent pinned C4 source documents:400 general and128 disjoint exact1024 bodies. '
    'The first64 general sources have paired up-to1024 variants; single/multi/long-single, tail and bins '
    'are repeated views, never additional independent sources. Constructed native role inputs are not natural '
    'conversations. Exact source-line, URL and full/prefix fingerprint exclusions do not establish semantic '
    'or domain independence. Original249 URL/line provenance remains incomplete. Models, thresholds, source '
    'selection and galleries were fixed prospectively. No test result selects a replacement readout; '
    'the separate fresh natural companion is also required before claiming the joint goal.')


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())
def digest(path): return data.digest(path)
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def protocol_spec():
    return dict(version=1, experiment='systemness-complementary-confirmation-report', physical_models=PHYSICAL,
        records=RECORDS, record_to_physical=RECORD_TO_PHYSICAL, primary_record=PRIMARY,
        main_groups=PROFILES, full_body_position_bins=[list(pair) for pair in BINS], all_groups=GROUPS, source_documents=528,
        expected_source_documents=EXPECTED_DOCUMENTS, metrics='All normal native content tokens; five-class argmax plus frozen-threshold system detection.',
        old_head='One probability array and argmax per old-head; original and max33 records differ only in their system threshold.',
        bootstrap=dict(replicates=BOOTSTRAPS, seed=BOOTSTRAP_SEED, unit='Original source document within each group; same draws paired across4 records.',
            ratios='Undefined ratios remain N/A; quantiles use defined draws only; paired differences require both defined. Fewer2 source units has no CI.',
            comparisons=396, intervals=264, independent_counts_are_not_added_across_profiles=True),
        gallery='12 fixed evenly spaced general multi sources +full system-only ranks0/42/84/127; five raw probability rows x3 physical columns; 1 four-record full position curve. PNG/PDF17pairs.',
        plot_smoothing=False, target='Primary P/R/BAcc>=.90 in every main group and every full-body bin; all must pass. Fresh natural companion still required.',
        thresholds_retuned=False, model_or_example_selection_from_scores=False, primary_changed=False,
        source_preparation_or_model_forward_run=False, limitation=LIMITATION)


def infer_module(): return importlib.import_module('systemness_complementary_confirmation_infer')


def load_completed():
    closed.pin_gate()
    module = importlib.import_module('systemness_complementary_confirmation_data')
    inference_out = Path(module.OUT) / 'inference'
    require((inference_out / 'inference-complete.json').is_file() and (inference_out / 'status.json').is_file()
        and read(inference_out / 'status.json')['stage'] == 'inference_complete', 'Shared inference must be terminal complete before any report cache read.')
    runner = infer_module(); ctx = runner.load_context('c4', module)
    require(ctx.snapshot['runtime_files_sha256'].get(str(Path(__file__).resolve())) == digest(__file__),
        'Report source was not frozen before test preparation.')
    complete, physical = runner.completed_probabilities('c4', context=ctx)
    return ctx, complete, physical


def record_probabilities(physical):
    require(set(physical) == set(PHYSICAL), 'Exactly three physical probability arrays are required.')
    values = {record: physical[name] for record, name in RECORD_TO_PHYSICAL.items()}
    require(values['oldhead-max33'] is values['oldhead-original'], 'Old-head records must alias one physical probability array.')
    return values


def validate_layout(ctx, physical):
    rows = ctx.rows
    require(np.array_equal(rows.index, np.arange(len(rows))) and np.array_equal(rows.cache_ix, np.arange(len(rows)))
        and rows.sample_ix.is_unique and rows.split.eq('test').all() and rows.wrapper.eq('native').all()
        and set(rows.profile) == set(PROFILES[:-1]) and set(rows.role) == set(data.ROLES), 'Report row identity/native profile/formal role coverage changed.')
    require(set(ctx.thresholds) == set(RECORDS) and all(np.isfinite(value) and 0 <= value <= 1.00001 for value in ctx.thresholds.values()), 'Frozen four-record thresholds are missing or invalid.')
    require(ctx.snapshot['record_to_physical'] == RECORD_TO_PHYSICAL, 'Snapshot record/physical mapping differs.')
    for name, p in physical.items():
        require(name in PHYSICAL and p.shape == (len(rows), 5) and p.dtype == np.float32, 'Physical probability geometry/dtype mismatch.')
        for start in range(0, len(p), 65536):
            block = p[start:start + 65536]
            require(np.isfinite(block).all() and (block >= 0).all() and (block <= 1.00001).all()
                and np.allclose(block.sum(1), 1., atol=2e-5, rtol=0), 'Nonfinite or unnormalized report probabilities.')
    full = rows[rows.profile.eq('long-full')]
    require(len(full) == 128 * 5 * 1024 and full.group_id.nunique() == 128
        and full.groupby(['group_id', 'role']).size().eq(1024).all(), 'Exact1024 full-role coverage changed.')
    for _, part in full.groupby(['group_id', 'role'], sort=False):
        require(np.array_equal(part.body_offset.to_numpy(), np.arange(1024)), 'A full body is truncated or its body offsets changed.')
    require(not set(full.group_id) & set(rows.loc[~rows.profile.eq('long-full'), 'group_id']), 'Disjoint full cohort overlaps general sources.')
    require(ctx.metadata['coverage']['all_content_tokens_scored'] is True and ctx.metadata['coverage']['independent_native_labels_match'] is True,
        'All-content independent native parser evidence is missing.')
    return record_probabilities(physical)


def group_indices(rows):
    full = rows.profile.eq('long-full').to_numpy(); offset = rows.body_offset.to_numpy()
    groups = {profile: np.flatnonzero(rows.profile.eq(profile)) for profile in PROFILES[:-1]}
    groups['long-full-tail'] = np.flatnonzero(full & (offset >= 512) & (offset <= 1023))
    groups.update({name: np.flatnonzero(full & (offset >= left) & (offset <= right)) for name, (left, right) in zip(BIN_NAMES, BINS)})
    for name, indices in groups.items():
        require(len(indices) and rows.iloc[indices].group_id.nunique() == EXPECTED_DOCUMENTS[name], 'Independent source count changed: ' + name)
    require(len(groups['long-full-tail']) == 128 * 5 * 512, 'Full512+ tail coverage changed.')
    return groups


def summarize(ctx, physical):
    probabilities = validate_layout(ctx, physical); rows = ctx.rows; groups = group_indices(rows)
    main, positions, documents, source_counts = [], [], [], {}
    for name, indices in groups.items():
        part = rows.iloc[indices]
        source_counts[name] = dict(source_documents=int(part.group_id.nunique()), tokens=len(part),
            role_support={role: int(part.role.eq(role).sum()) for role in data.ROLES}, repeated_view=name not in ['single', 'multi', 'long-full'])
    for record in RECORDS:
        p = probabilities[record]; threshold = ctx.thresholds[record]
        for name, indices in groups.items():
            part = rows.iloc[indices]; labels = probes.labels(part); value = probes.metrics(p[indices], labels, threshold)
            value.update(record=record, physical_model=RECORD_TO_PHYSICAL[record], profile=name,
                source_documents=int(part.group_id.nunique()), role_support=source_counts[name]['role_support'])
            if name in PROFILES: main.append(value)
            else:
                left, right = BINS[BIN_NAMES.index(name)]; system_ix = indices[part.role.eq('system').to_numpy()]
                value.update(start=left, end=right, system_tokens=len(system_ix), system_source_documents=128,
                    mean_system_probability=float(p[system_ix, 0].mean()))
                positions.append(value)
            truth = labels == 0; detected = p[indices, 0] >= float(threshold); pred = p[indices].argmax(1)
            counts = part[['group_id', 'document_id']].copy()
            for field, values in dict(tp=truth & detected, fp=~truth & detected, fn=truth & ~detected,
                tn=~truth & ~detected, correct=pred == labels, tokens=np.ones(len(indices), np.int64)).items(): counts[field] = values
            counts = counts.groupby(['group_id', 'document_id'], as_index=False)[FIELDS].sum()
            documents.append(counts.assign(record=record, physical_model=RECORD_TO_PHYSICAL[record], profile=name))
    # Threshold-free statistics for two old records must be exactly equal.
    all_metrics = main + positions
    for name in GROUPS:
        left = next(row for row in all_metrics if row['record'] == 'oldhead-max33' and row['profile'] == name)
        right = next(row for row in all_metrics if row['record'] == 'oldhead-original' and row['profile'] == name)
        for field in ['accuracy', 'macro_f1', 'nll', 'recall', 'confusion', 'system_auroc', 'system_average_precision', 'role_support']:
            require(left[field] == right[field], 'Old-head threshold-independent statistic changed: ' + name + '/' + field)
    require(len(main) == 20 and len(positions) == 24, 'Report main/bin metric row coverage changed.')
    return main, positions, pd.concat(documents, ignore_index=True), source_counts


def binary_values(counts):
    values = np.asarray(counts, dtype=float); tp, fp, fn, tn, correct, tokens = np.moveaxis(values, -1, 0)
    def divide(left, right): return np.divide(left, right, out=np.full(np.shape(left), np.nan), where=right > 0)
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return dict(system_precision=divide(tp, tp + fp), system_recall=recall,
        system_f1=divide(2 * tp, 2 * tp + fp + fn), system_balanced_accuracy=(recall + specificity) / 2,
        system_false_positive_rate=divide(fp, fp + tn), accuracy=divide(correct, tokens))


def scalar(value): return float(value) if np.isfinite(value) else None


def bootstrap(document):
    intervals, comparisons = [], []
    require(set(document.profile) == set(GROUPS) and set(document.record) == set(RECORDS), 'Bootstrap groups/records incomplete.')
    for index, name in enumerate(GROUPS):
        group = document[document.profile.eq(name)]; arrays, ids = {}, None
        for record in RECORDS:
            frame = group[group.record.eq(record)].sort_values('group_id')
            require(frame.group_id.is_unique and len(frame) == EXPECTED_DOCUMENTS[name], 'Bootstrap document unit duplicated or missing.')
            if ids is None: ids = frame.group_id.tolist()
            require(frame.group_id.tolist() == ids, 'All four records need identical paired source units.')
            arrays[record] = frame[FIELDS].to_numpy(dtype=np.float64)
        rng = np.random.RandomState(BOOTSTRAP_SEED + index); draws = rng.randint(len(ids), size=(BOOTSTRAPS, len(ids)))
        point = {record: binary_values(value.sum(0)) for record, value in arrays.items()}
        samples = {record: binary_values(value[draws].sum(1)) for record, value in arrays.items()}
        for record in RECORDS:
            for metric in METRIC_FIELDS:
                values = samples[record][metric]; defined = np.isfinite(values); count = int(defined.sum())
                bounds = np.quantile(values[defined], [.025, .975]) if count and len(ids) >= 2 else [np.nan, np.nan]
                intervals.append(dict(profile=name, record=record, metric=metric, source_documents=len(ids),
                    bootstrap_unit='original-source-document', bootstrap_replicates=BOOTSTRAPS,
                    value=scalar(point[record][metric]), ci95_low=scalar(bounds[0]), ci95_high=scalar(bounds[1]),
                    valid_bootstrap_draws=count, ci_unavailable_reason=None if count and len(ids) >= 2 else 'undefined_ratio_or_fewer_than_two_sources'))
        for before, after in itertools.combinations(RECORDS, 2):
            for metric in METRIC_FIELDS:
                valid = np.isfinite(samples[before][metric]) & np.isfinite(samples[after][metric]); count = int(valid.sum())
                differences = (samples[after][metric][valid] - samples[before][metric][valid]) * 100
                bounds = np.quantile(differences, [.025, .975]) if count and len(ids) >= 2 else [np.nan, np.nan]
                baseline, comparison = point[before][metric], point[after][metric]
                comparisons.append(dict(profile=name, baseline_record=before, comparison_record=after, metric=metric,
                    source_documents=len(ids), bootstrap_unit='original-source-document', bootstrap_replicates=BOOTSTRAPS,
                    baseline=scalar(baseline), comparison=scalar(comparison), gain_percentage_points=scalar((comparison - baseline) * 100),
                    ci95_low=scalar(bounds[0]), ci95_high=scalar(bounds[1]), valid_paired_bootstrap_draws=count,
                    ci_unavailable_reason=None if count and len(ids) >= 2 else 'undefined_ratio_or_fewer_than_two_sources'))
    require(len(intervals) == 264 and len(comparisons) == 396, '11group/4record/6pair bootstrap output coverage changed.')
    return comparisons, intervals


def fixed_gallery(ctx):
    manifest = read(Path(ctx.module.OUT) / 'gallery-index.json')
    require(manifest['selected_before_probabilities'] is True and manifest['score_based_selection'] is False
        and manifest['old_head_shared_probabilities'] is True, 'Gallery was not fixed prospectively.')
    general, full = manifest['general_multi'], manifest['full_system']
    require(len(general) == 12 and [entry['cohort_rank'] for entry in general] == np.linspace(0, 399, 12, dtype=int).tolist()
        and len(full) == 4 and [entry['cohort_rank'] for entry in full] == [0, 42, 84, 127], 'Fixed source gallery indices changed.')
    return manifest


def plots(ctx, physical, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    manifest = fixed_gallery(ctx); folder = OUT / 'plots'; folder.mkdir(exist_ok=True)
    entries = []; labels = ['Systemness', 'Userness', 'CoTness', 'Asstness', 'Toolness']
    examples = [(f'general-multi-{i:02d}', entry, False) for i, entry in enumerate(manifest['general_multi'])]
    examples += [(f'full-system-{entry["cohort_rank"]:03d}', entry, True) for entry in manifest['full_system']]
    for basename, entry, system_only in examples:
        group = ctx.rows[ctx.rows.group_id.eq(entry['group_id']) & ctx.rows.profile.eq('long-full' if system_only else 'multi')]
        if system_only: group = group[group.role.eq('system')]
        group = group.sort_values('body_offset' if system_only else 'token_ix')
        require(len(group) and (not system_only or (len(group) == 1024 and group.body_offset.tolist() == list(range(1024)))), 'Fixed gallery source or its full system body is missing.')
        indices = group.index.to_numpy(); x = group.body_offset.to_numpy() if system_only else group.token_ix.to_numpy()
        truth = group.role.to_numpy(); fig, axes = plt.subplots(5, 3, figsize=(15, 7), sharex=True, sharey=True)
        for column, model in enumerate(PHYSICAL):
            for role_ix, label in enumerate(labels):
                ax = axes[role_ix, column]
                for role, color in COLORS.items():
                    mask = truth == role
                    if mask.any(): ax.scatter(x[mask], physical[model][indices[mask], role_ix], s=3, color=color, alpha=.7, rasterized=True)
                if role_ix == 0:
                    for record in [record for record in RECORDS if RECORD_TO_PHYSICAL[record] == model]:
                        ax.axhline(ctx.thresholds[record], ls=':' if record.endswith('original') else '--', lw=.8,
                            color='#111827' if record.endswith('original') else '#6b7280')
                    notes = '\n'.join(f'{record}: τ={ctx.thresholds[record]:.6g}' for record in RECORDS if RECORD_TO_PHYSICAL[record] == model)
                    ax.text(.99, .97, notes, transform=ax.transAxes, ha='right', va='top', fontsize=6,
                        bbox=dict(facecolor='white', alpha=.75, edgecolor='none'))
                if not system_only:
                    for boundary in group.groupby('message_ix').token_ix.min(): ax.axvline(boundary, lw=.5, color='#d1d5db', zorder=0)
                ax.set_ylim(-.03, 1.03); ax.set_yticks([0, 1], ['0%', '100%']); ax.spines[['top', 'right']].set_visible(False)
                if column == 0: ax.set_ylabel(label, fontsize=9)
            axes[0, column].set_title(model + (' | shared two-threshold readout' if model == 'old-head' else ''), fontsize=9)
            axes[-1, column].set_xlabel('Body offset0–1023' if system_only else 'Token position in native input')
        if system_only:
            for ax in axes.flat: ax.set_xlim(0, 1023)
        handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=4, label=role)
            for role, color in COLORS.items() if role in set(truth)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1), ncol=5, frameon=False, fontsize=8)
        caption = 'System-only exact1024 body; all five probability rows; truth is system throughout.' if system_only else 'Constructed native multi-message input; all normal content tokens; structural tool_call content excluded.'
        fig.suptitle(f'{entry["group_id"]} | {caption}', y=.944, fontsize=10)
        fig.text(.5, .01, 'Frozen GPT-OSS-120B MXFP4 | fixed source selection | raw unsmoothed probabilities | formal-role colors', ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .03, 1, .91])
        for extension in ['png', 'pdf']: fig.savefig(folder / f'{basename}.{extension}', dpi=140)
        require(len(fig.axes) == 15, 'Raw gallery must have five probability rows by three physical columns.')
        entries.append(dict(basename=basename, group_id=entry['group_id'], document_id=int(entry['document_id']),
            cohort_rank=int(entry['cohort_rank']), system_only=system_only, truth_roles=sorted(set(truth)),
            content_tokens=len(group), x_min=int(x.min()), x_max=int(x.max()), axes=15,
            files=[f'plots/{basename}.png', f'plots/{basename}.pdf']))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for record, color, marker in zip(RECORDS, ['#059669', '#d97706', '#7c3aed', '#475569'], ['o', 's', '^', 'x']):
        curve = sorted([value for value in positions if value['record'] == record], key=lambda value: value['start'])
        require(len(curve) == 6, 'Full position curve is missing a frozen record/bin.')
        ax.plot([f'{value["start"]}–{value["end"]}' for value in curve], [value['system_recall'] for value in curve],
            marker=marker, linestyle=':' if record.endswith('original') else '-', color=color, label=record)
    ax.set_ylim(0, 1.03); ax.set_ylabel('System recall'); ax.set_xlabel('Zero-based full-body position bin')
    ax.set_title('128 original full-length sources | empirical bins, no smoothing', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    for extension in ['png', 'pdf']: fig.savefig(folder / f'full-system-by-position.{extension}', dpi=140)
    plt.close(fig)
    gallery = dict(entries=entries, position_curve_files=['plots/full-system-by-position.png', 'plots/full-system-by-position.pdf'],
        scatter_examples=16, png_pdf_pairs=17, physical_columns=PHYSICAL, threshold_records=RECORDS,
        fixed_before_probabilities=True, score_based_selection=False, smoothing=False,
        old_head_two_thresholds_one_probability_array=True, limitation='Gallery examples are fixed views; primary assessment uses all528 original source documents.')
    atomic_json(OUT / 'gallery.json', gallery)
    (OUT / 'gallery.md').write_text('# Fixed C4 probability gallery\n\n' + '\n'.join(
        f'- [{entry["group_id"]}]({entry["files"][0]})' for entry in entries) + '\n\n[Full position curve](plots/full-system-by-position.png)\n')
    return gallery


def targets(main, positions):
    result = {}
    for value in main + positions:
        if value['record'] != PRIMARY: continue
        result[value['profile']] = all(value[field] is not None and np.isfinite(value[field]) and value[field] >= .9
            for field in ['system_precision', 'system_recall', 'system_balanced_accuracy'])
    require(set(result) == set(GROUPS), 'Primary target assessment must retain every group.')
    return result


def report_signature(ctx, complete):
    return dict(source_sha256=digest(__file__), dependencies_sha256={name: digest(data.HERE / name) for name in DEPENDENCIES},
        protocol=protocol_spec(), data_complete_sha256=digest(Path(ctx.module.OUT) / 'data-complete.json'),
        gallery_index_sha256=digest(Path(ctx.module.OUT) / 'gallery-index.json'),
        inference_complete_sha256=digest(ctx.out / 'inference-complete.json'), inference_signature=ctx.signature,
        frozen_snapshot_sha256=ctx.snapshot['snapshot_sha256'], model_sha256=ctx.snapshot['physical_model_sha256'],
        physical_probability_sha256=complete['probability_sha256'], record_probability_sha256=complete['record_probability_sha256'],
        physical_argmax_sha256=complete['argmax_sha256'], record_argmax_sha256=complete['record_argmax_sha256'],
        record_thresholds=ctx.thresholds, record_to_physical=RECORD_TO_PHYSICAL)


def artifact_names(gallery):
    return ['metrics.json', 'metrics.csv', 'system-by-position.json', 'system-by-position.csv', 'document-counts.csv',
        'source-counts.json', 'paired-comparisons.csv', 'bootstrap-metrics.csv', 'gallery.json', 'gallery.md', 'report.txt'] \
        + [name for entry in gallery['entries'] for name in entry['files']] + gallery['position_curve_files']


def verified_report(ctx=None, inference_complete=None):
    closed.pin_gate()
    require((OUT / 'complete.json').is_file() and (OUT / 'status.json').is_file()
        and read(OUT / 'status.json')['stage'] == 'complete', 'C4 report is not terminal complete.')
    if ctx is None: ctx, inference_complete, _ = load_completed()
    complete = read(OUT / 'complete.json'); signature = report_signature(ctx, inference_complete)
    require(complete['signature'] == signature and complete['source_sha256'] == digest(__file__)
        and complete['primary_record'] == PRIMARY and complete['primary_changed'] is False
        and complete['thresholds_retuned'] is False and complete['test_results_used_for_selection'] is False
        and complete['natural_companion_still_required'] is True, 'C4 completed report frozen signature differs.')
    gallery = read(OUT / 'gallery.json')
    require(len(gallery['entries']) == 16 and gallery['png_pdf_pairs'] == 17
        and set(complete['artifact_sha256']) == set(artifact_names(gallery)), 'Completed report artifact inventory differs.')
    for name, expected in complete['artifact_sha256'].items(): require(digest(OUT / name) == expected, 'Completed report bytes changed: ' + name)
    main, positions = read(OUT / 'metrics.json'), read(OUT / 'system-by-position.json')
    require(len(main) == 20 and len(positions) == 24 and len(pd.read_csv(OUT / 'paired-comparisons.csv')) == 396
        and len(pd.read_csv(OUT / 'bootstrap-metrics.csv')) == 264, 'Completed report metric/bootstrap coverage differs.')
    actual = targets(main, positions)
    require(complete['working_target_pass'] == actual and complete['all_primary_groups_meet_90_percent'] == all(actual.values()), 'Completed report target decision changed.')
    require(read(OUT / 'initialized.json') == dict(signature=signature), 'Completed report owner changed.')
    return complete


@contextmanager
def run_lock():
    import fcntl
    with (OUT / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another C4 report writer holds the output.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def flat_metrics(values):
    return pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        | {f'role_support_{role}': row['role_support'][role] for role in data.ROLES}
        | {f'role_recall_{role}': row['recall'][role] for role in data.ROLES} for row in values])


def write_report(ctx, inference_complete, physical, *, end_verify=None, synthetic=False):
    if not synthetic:closed.pin_gate()
    signature = report_signature(ctx, inference_complete)
    if (OUT / 'complete.json').exists(): return verified_report(ctx, inference_complete)
    OUT.mkdir(parents=True, exist_ok=True)
    with run_lock():
        if (OUT / 'complete.json').exists(): return verified_report(ctx, inference_complete)
        owner = OUT / 'initialized.json'
        if owner.exists(): require(read(owner) == dict(signature=signature), 'Partial report belongs to different frozen inputs.')
        else:
            require(not any(path.name != '.run.lock' for path in OUT.iterdir()), 'Unowned partial C4 report; inspect before retrying.')
            atomic_json(owner, dict(signature=signature))
        main, positions, document, counts = summarize(ctx, physical); comparisons, intervals = bootstrap(document)
        atomic_json(OUT / 'metrics.json', main); flat_metrics(main).to_csv(OUT / 'metrics.csv', index=False)
        atomic_json(OUT / 'system-by-position.json', positions); flat_metrics(positions).to_csv(OUT / 'system-by-position.csv', index=False)
        document.to_csv(OUT / 'document-counts.csv', index=False)
        atomic_json(OUT / 'source-counts.json', dict(groups=counts, independent_source_documents=528,
            general_documents=400, disjoint_full_length_documents=128, paired_general_long_documents=64,
            tails_and_bins_are_repeated_views=True, groups_must_not_be_added_as_independent_sources=True))
        pd.DataFrame(comparisons).to_csv(OUT / 'paired-comparisons.csv', index=False)
        pd.DataFrame(intervals).to_csv(OUT / 'bootstrap-metrics.csv', index=False)
        gallery = plots(ctx, physical, positions); passed = targets(main, positions)
        lines = ['Prospectively frozen C4 confirmation: four threshold records / three physical models', '', LIMITATION, '',
            'Old-head original/max33 share exactly the same probabilities, argmax, AUC and AP; only system thresholds differ.',
            '2000 source-paired bootstrap replicates within each of11 groups; all4 records and6 pairs retained.',
            'Undefined ratios are N/A with valid-draw counts. No pooled-token/generalization inference.', '',
            'Record                 Group                        P         R         F1        BAcc      FPR       FiveAcc']
        for row in main + positions:
            lines.append(f'{row["record"]:22s} {row["profile"]:28s} ' + ' '.join('NA' if row[key] is None else f'{row[key]:.6f}' for key in METRIC_FIELDS))
        lines += ['', 'Primary P/R/BAcc>=.90 by group: ' + json.dumps(passed, sort_keys=True),
            'C4 all-group target pass: ' + str(all(passed.values())), 'Fresh natural companion is still required; this report never promotes or replaces a primary.']
        (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
        if end_verify is not None: end_verify()
        require(report_signature(ctx, inference_complete) == signature, 'Report dependencies changed during calculation.')
        complete = dict(signature=signature, source_sha256=digest(__file__), primary_record=PRIMARY,
            physical_model_sha256=ctx.snapshot['physical_model_sha256'], record_to_physical=RECORD_TO_PHYSICAL,
            record_thresholds=ctx.thresholds, working_target_pass=passed, all_primary_groups_meet_90_percent=all(passed.values()),
            natural_companion_still_required=True, independent_source_documents=528, bootstrap_replicates=BOOTSTRAPS,
            bootstrap_unit='original-source-document', comparisons=396, intervals=264,
            old_head_probability_and_argmax_identical=True, old_head_threshold_free_metrics_identical=True,
            primary_changed=False, thresholds_retuned=False, test_results_used_for_selection=False,
            model_forward_or_source_preparation_run=False, synthetic_only=synthetic,
            artifact_sha256={name: digest(OUT / name) for name in artifact_names(gallery)})
        atomic_json(OUT / 'complete.json', complete); atomic_json(OUT / 'status.json', dict(stage='complete', primary_changed=False))
        verified_report(ctx, inference_complete)
        return complete


def report():
    closed.pin_gate()
    ctx, complete, physical = load_completed()
    require(OUT == Path(ctx.module.OUT) / 'report', 'C4 report output namespace differs from its builder.')
    return write_report(ctx, complete, physical, end_verify=lambda: infer_module().completed_probabilities('c4', context=ctx))


def synthetic_context(folder):
    rows, inputs, cache_ix = [], [], 0
    def add(document, profile, role_list, size):
        nonlocal cache_ix
        prompt_ix = len(inputs); group = f'fixture-c4-{document:03d}'; blocks = []
        for message_ix, role in enumerate(role_list):
            body = np.arange(size); count = len(body)
            blocks.append(pd.DataFrame(dict(cache_ix=np.arange(cache_ix, cache_ix + count), sample_ix=7000000000000 + np.arange(cache_ix, cache_ix + count),
                prompt_ix=prompt_ix, token_ix=13 + message_ix * (size + 11) + body, token_id=100 + body,
                document_id=7000000 + document, group_id=group, role=role, semantic_role=role, message_ix=message_ix,
                profile=profile, split='test', wrapper='native', body_offset=body)))
            cache_ix += count
        rows.extend(blocks); inputs.append(dict(prompt_ix=prompt_ix, group_id=group, document_id=7000000 + document,
            profile=profile, role=role_list[0] if len(role_list) == 1 else 'mixed', length=13 + len(role_list) * (size + 11)))
    for document in range(400):
        for role in data.ROLES: add(document, 'single', [role], 8)
        add(document, 'multi', data.ROLES, 4)
        if document < 64:
            for role in data.ROLES: add(document, 'long-single', [role], 16)
    for document in range(400, 528):
        for role in data.ROLES: add(document, 'long-full', [role], 1024)
    rows, inputs = pd.concat(rows, ignore_index=True), pd.DataFrame(inputs)
    y = probes.labels(rows); physical = {}
    for index, model in enumerate(PHYSICAL):
        p = np.full((len(rows), 5), .02, np.float32); p[np.arange(len(rows)), y] = .92
        # Different full-tail system failures make filtering/alias mistakes observable.
        failure = rows.profile.eq('long-full').to_numpy() & (y == 0) & rows.body_offset.ge(960 - index * 64).to_numpy()
        p[failure, 0] = .3 + index * .05; p[failure, 1] = .64 - index * .05
        physical[model] = p
    builder_out = folder / 'builder'; builder_out.mkdir(); inference_out = builder_out / 'inference'; inference_out.mkdir()
    snapshot = dict(snapshot_sha256='synthetic-snapshot', record_to_physical=RECORD_TO_PHYSICAL,
        physical_model_sha256={name: hashlib.sha256(name.encode()).hexdigest() for name in PHYSICAL},
        runtime_files_sha256={str(Path(__file__).resolve()): digest(__file__)})
    thresholds = {'new-guard-max33': .6, 'matched-max33': .55, 'oldhead-max33': .5, 'oldhead-original': .35}
    metadata = dict(frozen=snapshot, coverage=dict(all_content_tokens_scored=True, independent_native_labels_match=True))
    ctx = SimpleNamespace(module=SimpleNamespace(OUT=builder_out), rows=rows, inputs=inputs, metadata=metadata,
        snapshot=snapshot, thresholds=thresholds, signature=dict(synthetic_only=True), out=inference_out)
    atomic_json(builder_out / 'data-complete.json', metadata)
    general = [dict(group_id=f'fixture-c4-{rank:03d}', document_id=7000000 + int(rank), cohort_rank=int(rank)) for rank in np.linspace(0, 399, 12, dtype=int)]
    full = [dict(group_id=f'fixture-c4-{400 + rank:03d}', document_id=7000400 + rank, cohort_rank=rank) for rank in [0, 42, 84, 127]]
    atomic_json(builder_out / 'gallery-index.json', dict(general_multi=general, full_system=full,
        selected_before_probabilities=True, score_based_selection=False, old_head_shared_probabilities=True))
    shas = {name: array_sha(value) for name, value in physical.items()}
    argmax = {name: array_sha(value.argmax(1).astype(np.int64)) for name, value in physical.items()}
    complete = dict(probability_sha256=shas, argmax_sha256=argmax,
        record_probability_sha256={name: shas[model] for name, model in RECORD_TO_PHYSICAL.items()},
        record_argmax_sha256={name: argmax[model] for name, model in RECORD_TO_PHYSICAL.items()})
    atomic_json(inference_out / 'inference-complete.json', complete)
    return ctx, complete, physical


def _synthetic_preflight():
    from unittest.mock import patch
    folder = Path(tempfile.mkdtemp(prefix='systemness-complementary-confirmation-report-preflight-', dir='/tmp'))
    source_sha = digest(__file__); checks = {}
    ctx, inference_complete, physical = synthetic_context(folder)
    signature = report_signature(ctx, inference_complete)
    require(json.loads(json.dumps(signature)) == signature, 'Report signature is not JSON-roundtrip stable.')
    checks['signature_json_roundtrip_stable'] = True
    require(len(ctx.inputs) == 3360 and len(ctx.rows) == 684480, 'Synthetic full production source geometry changed.')
    checks['528_sources_3360_inputs_684480_rows'] = True
    main, positions, document, counts = summarize(ctx, physical)
    require(len(main) == 20 and len(positions) == 24 and len(document) == 7552
        and all(counts[name]['source_documents'] == EXPECTED_DOCUMENTS[name] for name in GROUPS), 'Synthetic summary/document coverage differs.')
    checks['20main_24bins_7552_paired_document_rows'] = True
    values = {row['record']: row for row in main if row['profile'] == 'long-full-tail'}
    require(values['new-guard-max33']['system_recall'] > values['matched-max33']['system_recall']
        and values['oldhead-original']['system_recall'] > values['oldhead-max33']['system_recall'],
        'Synthetic failures do not distinguish independent models and shared-old thresholds.')
    checks['different_tail_failures_and_same_old_probabilities_two_thresholds'] = True
    comparisons, intervals = bootstrap(document)
    require(len(comparisons) == 396 and len(intervals) == 264 and all(value['bootstrap_replicates'] == 2000 for value in intervals), 'Bootstrap geometry mismatch.')
    old_accuracy = [value for value in comparisons if value['baseline_record'] == 'oldhead-max33'
        and value['comparison_record'] == 'oldhead-original' and value['metric'] == 'accuracy']
    require(all(value['gain_percentage_points'] == value['ci95_low'] == value['ci95_high'] == 0. for value in old_accuracy), 'Shared old argmax lost exact paired identity.')
    require(binary_values(np.array([0, 0, 1, 2, 2, 3]))['system_precision'].item() != binary_values(np.array([0, 0, 1, 2, 2, 3]))['system_precision'].item(), 'Undefined precision must remain NaN internally.')
    checks['2000_source_paired_draws_396comparisons_264intervals'] = True
    checks['shared_old_accuracy_zero_delta_and_undefined_ratios'] = True
    with patch(__name__ + '.OUT', folder / 'report'):
        complete = write_report(ctx, inference_complete, physical, synthetic=True)
        gallery = read(OUT / 'gallery.json')
        require(gallery['png_pdf_pairs'] == 17 and len(gallery['entries']) == 16
            and all(entry['axes'] == 15 for entry in gallery['entries']), 'Synthetic17-pair five-by-three gallery missing.')
        full = [entry for entry in gallery['entries'] if entry['system_only']]
        require(len(full) == 4 and all(entry['truth_roles'] == ['system'] and entry['content_tokens'] == 1024
            and entry['x_min'] == 0 and entry['x_max'] == 1023 for entry in full), 'Full system-only gallery shows mixed truth or truncated positions.')
        require(len(list((OUT / 'plots').glob('*.png'))) == len(list((OUT / 'plots').glob('*.pdf'))) == 17, 'Synthetic gallery PNG/PDF coverage differs.')
        checks['17_png_pdf_pairs_16_scatter5by3_plus4record_curve'] = True
        checks['four_fixed_full_system_only_sources_0through1023'] = True
        require(complete['all_primary_groups_meet_90_percent'] is False, 'Synthetic below-target full tail was hidden.')
        checks['primary_target_all11groups_reports_failure'] = True
        hashes = {name: digest(OUT / name) for name in complete['artifact_sha256']}
        again = write_report(ctx, inference_complete, physical, synthetic=True)
        require(again == complete and hashes == {name: digest(OUT / name) for name in hashes}, 'Completed report restart wrote artifacts.')
        checks['completed_report_restart_is_readonly'] = True
        saved = (OUT / 'metrics.json').read_bytes(); (OUT / 'metrics.json').write_text('[]')
        try: verified_report(ctx, inference_complete)
        except RuntimeError: pass
        else: raise RuntimeError('Changed completed metric bytes were accepted.')
        (OUT / 'metrics.json').write_bytes(saved); checks['completed_artifact_tamper_rejected'] = True
    require(digest(__file__) == source_sha, 'Report source changed during preflight.')
    report = dict(source_sha256=source_sha, checks=checks, fixture_directory=str(folder), synthetic_only=True,
        actual_source_or_test_probabilities_read=False, formal_report_written=False, gpu_used=False,
        synthetic_rows=len(ctx.rows), synthetic_inputs=len(ctx.inputs), comparisons=396, intervals=264,
        png_pdf_pairs=17, production_source_counts=EXPECTED_DOCUMENTS)
    atomic_json(folder / 'report.json', report)
    print(json.dumps(dict(report_path=str(folder / 'report.json'), report_sha256=digest(folder / 'report.json'), **report)), flush=True)


def preflight():
    from unittest.mock import patch
    def forbidden(*a,**k):raise LookupError('Real builder/inference/probability access forbidden in CPU preflight.')
    with patch.object(closed,'TRAINER_SOURCE_SHA',None),patch.object(closed,'read',forbidden),patch.object(closed,'digest',forbidden):
        try:report()
        except RuntimeError:pass
        else:raise RuntimeError('Unknown closed pins failed to block actual report before IO.')
    with patch.object(closed,'pin_gate',lambda:None):return _synthetic_preflight()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Synthetic CPU summaries/bootstrap/17 PNG-PDF pairs under /tmp only.')
    action.add_argument('--report', action='store_true', help='Explicitly report only completed prospective C4 inference.')
    action.add_argument('--verify', action='store_true', help='Read-only completed-report integrity verification.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.report: print(json.dumps(report()), flush=True)
    elif args.verify: print(json.dumps(verified_report()), flush=True)
    else: parser.print_help()
