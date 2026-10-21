"""Report five frozen readouts on reused development validation, never a new test.

Default: help. --preflight renders synthetic CPU fixtures under /tmp only.
--write requires a verified completed head-probe search. This script never
selects a model/threshold and never reads test or confirmation rows/probabilities.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_head_probes as heads
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-head-probes-report'
NAMES = ['dense128'] + list(heads.CONFIGS)
PROFILES = long_probes.PROFILES + ['long-full-tail']
BINS = [(0, 31), (32, 63), (64, 127), (128, 255), (256, 511), (512, 1023)]
DEPENDENCIES = sorted(set(heads.DEPENDENCIES + ['systemness_head_probes_report.py',
    'systemness_attention.py', 'run_local_gptoss120b.py']))
LIMITATION = ('Reused development validation only: this report is not a new independent test. '
    'All four head candidates fit the identical578090 mixed densefit rows; pooled LDA uses raw '
    'heads and joint linear/MLPs use a new fit-only512-dimensional scaler. The dense128 control '
    'uses its original8640-dimensional local features/scaler. Feature type/dimension, classifier, '
    'scaling and extraction batches differ, so cross-representation differences are not isolated '
    'causal ablations. Original four validation profiles are retained, and10 of31 long-full '
    'validation documents overlap the old long-single profile as paired variants. Single-role '
    'plots show64 existing sampled content positions per role, not every body token. '
    'No model/threshold is selected or existing primary changed by this report.')
require = long_data.require


def read(path):
    return json.loads(Path(path).read_text())


def probability_file(path, sample_ix, expected_sha):
    require(Path(path).is_file() and data.digest(path) == expected_sha, f'Probability file changed: {path}.')
    with np.load(path, allow_pickle=False) as archive:
        require(np.array_equal(archive['sample_ix'], sample_ix), f'Validation sample order differs: {path}.')
        p = archive['p'].copy()
    require(p.dtype == np.float32, 'Expected frozen float32 validation probabilities.')
    heads.validate_probabilities(p, len(sample_ix))
    return p


def body_positions(rows):
    return rows.token_ix - rows.groupby(['source_set', 'prompt_ix', 'message_ix']).token_ix.transform('min')


def gallery_sources(rows, inputs, expected_documents=200):
    single = rows[rows.profile == 'single'].copy()
    require(single.document_id.nunique() == expected_documents and single.source_set.eq('main').all()
        and single.split.eq('validation').all() and single.wrapper.eq('native').all(), 'Single-role validation source membership changed.')
    require(single.groupby(['document_id', 'role']).size().eq(64).all()
        and single.groupby('document_id').role.nunique().eq(5).all(), 'Expected64 sampled positions in each of5 native role versions.')
    require(inputs.prompt_ix.is_unique and set(single.prompt_ix) == set(inputs.prompt_ix)
        and set(inputs.split) == {'validation'} and set(inputs.profile) == {'single'}
        and set(inputs.wrapper) == {'native'}, 'Gallery source input/row mapping changed.')
    identity = inputs.set_index('prompt_ix')
    for field in ['document_id', 'group_id']:
        require(np.array_equal(identity.loc[single.prompt_ix, field], single[field]), f'Single input {field} mapping differs.')
    single['body_offset'] = body_positions(single)
    for _, group in single.groupby('document_id', sort=False):
        reference = None
        for role in data.ROLES:
            part = group[group.role == role].sort_values('body_offset')
            values = part[['body_offset', 'token_id']].to_numpy()
            if reference is None: reference = values
            require(np.array_equal(reference, values), 'Five native role versions no longer match sampled body positions/tokens.')
    docs = sorted(int(value) for value in single.document_id.unique())
    selected = [docs[i] for i in np.linspace(0, len(docs) - 1, min(12, len(docs)), dtype=int)]
    return selected, single


def load_validation():
    completion = heads.verified_completion()  # The only completed-search gate, before any output.
    plan = read(heads.OUT / 'search-plan.json'); selection = read(heads.OUT / 'selection.json')
    rows_path = heads.BANK / 'rows.parquet'
    require(data.digest(rows_path) == completion['signature']['rows_sha256'], 'Frozen development rows changed.')
    rows = pd.read_parquet(rows_path, filters=[('split', '=', 'validation')]).reset_index(drop=True)
    require(len(rows) == 173032 and rows.sample_ix.is_unique and set(rows.profile) == set(long_probes.PROFILES)
        and set(rows.split) == {'validation'} and set(rows.wrapper) == {'native'}, 'Expected the unchanged173032 native validation rows.')
    require(heads.array_digest(rows.sample_ix.to_numpy()) == completion['signature']['partitions']['validation_sample_ix_sha256'],
        'Validation identity digest differs from frozen training.')
    records = {}
    for key in heads.CONFIGS:
        path = heads.OUT / 'candidates' / f'{key}.json'
        require(data.digest(path) == completion['candidate_sha256'][key], 'A verified head candidate changed before report loading.')
        records[key] = read(path)
    dense_plan = read(long_data.OUT / 'search-plan.json')
    dense = long_probes.candidate_record('dense-mlp128', dense_plan); records['dense128'] = dense
    paths = {'dense128': long_probes.BANK / 'dense-mlp128-validation.npz',
        **{key: heads.CACHE / f'{key}-validation.npz' for key in heads.CONFIGS}}
    probability = {name: probability_file(paths[name], rows.sample_ix.to_numpy(),
        records[name]['validation_probabilities_sha256']) for name in NAMES}
    thresholds = {name: float(records[name]['validation']['system_threshold']) for name in NAMES}
    metadata = read(data.OUT / 'data-complete.json'); input_path = data.CACHE / 'inputs.parquet'
    require(data.digest(input_path) == metadata['cache_sha256']['inputs.parquet'], 'Original validation input cache changed.')
    inputs = pd.read_parquet(input_path, filters=[('split', '=', 'validation'), ('wrapper', '=', 'native'), ('profile', '=', 'single')])
    chosen, _ = gallery_sources(rows, inputs)
    for key in heads.CONFIGS:
        require(selection['candidate_guards'][key] == heads.promotion_guard(records[key]['validation'], dense['validation']),
            'Frozen candidate promotion guards no longer match the saved validation.')
    snapshot = dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        head_completion_sha256=data.digest(heads.OUT / 'complete.json'),
        head_selection_sha256=data.digest(heads.OUT / 'selection.json'),
        head_search_plan_sha256=data.digest(heads.OUT / 'search-plan.json'),
        head_scaler_sha256=data.digest(heads.OUT / 'standardization.npz'),
        original_data_sha256=data.digest(data.OUT / 'data-complete.json'),
        input_cache_sha256=data.digest(input_path), rows_sha256=data.digest(rows_path),
        validation_sample_ix_sha256=heads.array_digest(rows.sample_ix.to_numpy()),
        validation_rows=len(rows), selected_gallery_source_ids=chosen,
        model_sha256={name: records[name]['model_sha256'] for name in NAMES},
        candidate_record_sha256={'dense128': data.digest(long_data.OUT / 'candidates/dense-mlp128.json'),
            **{key: data.digest(heads.OUT / 'candidates' / f'{key}.json') for key in heads.CONFIGS}},
        probability_sha256={name: records[name]['validation_probabilities_sha256'] for name in NAMES},
        thresholds=thresholds, package_versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'matplotlib', 'scikit-learn']},
        tests_read=False, development_only=True, selection_changed=False, primary_changed=False)
    return rows, inputs, probability, thresholds, records, selection, snapshot


def metrics_at_frozen_thresholds(rows, probability, thresholds, records):
    require(set(probability) == set(thresholds) == set(NAMES), 'Frozen five-model probability/threshold coverage changed.')
    require(all(np.isfinite(thresholds[name]) and 0 <= thresholds[name] <= 1.00001
        and thresholds[name] == records[name]['validation']['system_threshold'] for name in NAMES),
        'Report threshold differs from its frozen validation record.')
    metrics = []; positions = []; tail = rows[rows.profile == 'long-full'].copy()
    tail['body_offset'] = body_positions(tail)
    for name in NAMES:
        groups = [(profile, rows[rows.profile == profile]) for profile in long_probes.PROFILES]
        groups.append(('long-full-tail', tail[tail.body_offset >= 512]))
        for profile, group in groups:
            value = probes.metrics(probability[name][group.index], probes.labels(group), thresholds[name])
            expected = records[name]['validation']['long_full_tail' if profile == 'long-full-tail' else 'profiles']
            if profile != 'long-full-tail': expected = expected[profile]
            for key in ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy', 'accuracy', 'system_false_positive_rate']:
                require(np.isclose(value[key], expected[key], atol=1e-14, rtol=0), f'{name}/{profile} frozen {key} differs from probabilities.')
            require(value['recall'] == expected['recall'], 'Frozen five-role recall differs.')
            metrics.append(dict(model=name, profile=profile, source_documents=int(group.group_id.nunique()),
                threshold=thresholds[name], cot_recall=value['recall']['cot'], **value))
        for start, end in BINS:
            group = tail[tail.body_offset.between(start, end)]
            if group.empty: continue
            value = probes.metrics(probability[name][group.index], probes.labels(group), thresholds[name])
            positions.append(dict(model=name, profile='long-full', start=start, end=end,
                source_documents=int(group.group_id.nunique()), sampled_tokens=len(group),
                system_tokens=int(group.role.eq('system').sum()), threshold=thresholds[name], **value))
    return metrics, positions


def plots(folder, rows, inputs, probability, thresholds, metrics, positions, expected_documents=200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    chosen, single = gallery_sources(rows, inputs, expected_documents)
    folder.mkdir(parents=True, exist_ok=True)
    colors = dict(system='#90a1b9', user='#00a6f4', cot='#fd9a00', assistant='#00d492', tool='#a855f7')
    labels = ['Systemness', 'Userness', 'CoTness', 'Asstness', 'Toolness']
    titles = ['Frozen dense128 control', 'Mixed raw pooled LDA', 'Joint linear', 'Joint MLP64', 'Joint MLP128']
    for doc in chosen:
        group = single[single.document_id == doc].copy(); span = int(group.body_offset.max()) + 1
        group['plot_x'] = group.body_offset + group.role.map({role: i * (span + 8) for i, role in enumerate(data.ROLES)})
        index = group.index.to_numpy(); x = group.plot_x.to_numpy()
        fig, axes = plt.subplots(5, 5, figsize=(23, 7), sharex=True, sharey=True)
        for col, name in enumerate(NAMES):
            for ri, label in enumerate(labels):
                ax = axes[ri, col]
                for role, color in colors.items():
                    mask = group.role.to_numpy() == role
                    ax.scatter(x[mask], probability[name][index[mask], ri], s=3, color=color, alpha=.7, rasterized=True)
                if ri == 0: ax.axhline(thresholds[name], ls='--', lw=.7, color='#4b5563')
                for k in range(1, 5): ax.axvline(k * (span + 8) - 4, lw=.5, color='#d1d5db')
                ax.set_ylim(-.03, 1.03); ax.set_yticks([0, 1], ['0%', '100%']); ax.spines[['top', 'right']].set_visible(False)
                if col == 0: ax.set_ylabel(label, fontsize=9)
            axes[0, col].set_title(titles[col], fontsize=9)
            axes[-1, col].set_xticks([k * (span + 8) + span / 2 for k in range(5)], data.ROLES, fontsize=7)
            axes[-1, col].set_xlabel('Concatenated native role versions', fontsize=8)
        handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=4, label=role) for role, color in colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1), ncol=5, frameon=False, fontsize=8)
        fig.suptitle(f'Validation C4 source {doc} | same body, five native role versions | development only', y=.945, fontsize=11)
        fig.text(.5, .01, 'Reused development validation |64 existing sampled content positions per role | raw unsmoothed probabilities | fixed source IDs', ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .03, 1, .91])
        for ext in ['png', 'pdf']: fig.savefig(folder / f'validation-c4-{doc}.{ext}', dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(PROFILES)); width = .15
    for mi, name in enumerate(NAMES):
        values = [next(r['system_f1'] for r in metrics if r['model'] == name and r['profile'] == p) for p in PROFILES]
        ax.bar(x + (mi - 2) * width, values, width, label=name)
    ax.set_xticks(x, PROFILES); ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('System F1 at each frozen threshold'); ax.set_title('Reused development validation; no new independent test')
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=7, ncol=3); fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(folder / f'validation-system-f1.{ext}', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for name in NAMES:
        curve = sorted([r for r in positions if r['model'] == name], key=lambda r: r['start'])
        ax.plot([f'{r["start"]}–{r["end"]}' for r in curve], [r['system_recall'] for r in curve], 'o-', label=name)
    ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylabel('System recall')
    ax.set_xlabel('Zero-based sampled body-position bin'); ax.set_title('Long-full development validation | empirical bins, no smoothing')
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=7, ncol=3); fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(folder / f'validation-long-full-by-position.{ext}', dpi=150)
    plt.close(fig)
    return dict(source_ids=chosen, source_selection='Equally spaced sorted single-profile validation source IDs; independent of scores.',
        scatter_figures=len(chosen), panels=5, models=5, png=len(chosen) + 2, pdf=len(chosen) + 2,
        sampled_body_positions_per_role=64, full_body_token_coverage_claimed=False, development_only=True, raw_unsmoothed_probabilities=True)


def render(out, rows, inputs, probability, thresholds, records, selection, snapshot, expected_documents=200):
    out.mkdir(parents=True, exist_ok=True)
    metrics, positions = metrics_at_frozen_thresholds(rows, probability, thresholds, records)
    data.atomic_json(out / 'metrics.json', metrics); data.atomic_json(out / 'long-full-by-position.json', positions)
    flat = lambda values: pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in values])
    flat(metrics).to_csv(out / 'metrics.csv', index=False); flat(positions).to_csv(out / 'long-full-by-position.csv', index=False)
    guards = selection['candidate_guards']; decisions = []
    for name in NAMES:
        baseline = name == 'dense128'; record = records[name]
        reasons = [] if baseline else list(guards[name]['rejection_reasons'])
        if not baseline and name != selection['candidate']: reasons.append('Not selected by the already frozen validation selection.')
        if name == selection['candidate'] and not selection['promoted']: reasons += selection['promotion_rejection_reasons']
        decisions.append(dict(model=name, threshold=thresholds[name], model_sha256=record['model_sha256'],
            probability_sha256=record['validation_probabilities_sha256'], baseline_control=baseline,
            frozen_selected_candidate=name == selection['candidate'], eligible=None if baseline else guards[name]['eligible'],
            frozen_selection_promoted=selection['promoted'], nonpromotion_reasons=reasons,
            guard=None if baseline else guards[name], best_epoch=record.get('best_epoch'),
            best_steps=record.get('best_steps'), actual_steps=record.get('actual_steps')))
    data.atomic_json(out / 'candidate-decisions.json', decisions)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ['guard', 'nonpromotion_reasons']}
        | dict(nonpromotion_reasons='; '.join(r['nonpromotion_reasons'])) for r in decisions]).to_csv(out / 'candidate-decisions.csv', index=False)
    gallery = plots(out / 'plots', rows, inputs, probability, thresholds, metrics, positions, expected_documents)
    data.atomic_json(out / 'gallery.json', gallery)
    lines = ['Five frozen readouts on reused development validation', '', LIMITATION, '',
        f'Frozen head selection: {selection["candidate"]}; promoted={selection["promoted"]}. This report does not change that decision.',
        'Model             Profile           Threshold  System P  System R  System F1  Balanced acc  Five-role acc  CoT recall']
    for r in metrics:
        values = [r[k] for k in ['threshold', 'system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy', 'accuracy', 'cot_recall']]
        lines.append(f'{r["model"]:17s} {r["profile"]:17s} ' + '  '.join('NA' if v is None else f'{v:.6f}' for v in values))
    lines += ['', 'Frozen guard outcomes and non-promotion reasons:']
    for r in decisions: lines.append(f'{r["model"]}: eligible={r["eligible"]}; ' + ('; '.join(r['nonpromotion_reasons']) or 'baseline or already frozen selected candidate'))
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    (out / 'plots/gallery-index.md').write_text('Development validation only;64 scored sample positions per role, five separate native versions of the same body. No new independent test.\n\n' +
        '\n'.join(f'- [C4 validation source {doc}](validation-c4-{doc}.png)' for doc in gallery['source_ids']) + '\n')
    data.atomic_json(out / 'source-snapshot.json', snapshot)
    return metrics, gallery


@contextmanager
def write_lock():
    import fcntl
    with (OUT / '.write.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another head-validation report holds this directory.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write():
    rows, inputs, p, thresholds, records, selection, snapshot = load_validation()
    OUT.mkdir(parents=True, exist_ok=True)
    with write_lock():
        require(not (OUT / 'complete.json').exists(), 'A completed validation report cannot be overwritten.')
        signature_path = OUT / 'report-plan.json'
        if signature_path.exists(): require(read(signature_path) == snapshot, 'Report source/upstream signature changed on restart.')
        else: data.atomic_json(signature_path, snapshot)
        (OUT / '.gitignore').write_text('/.write.lock\n')
        render(OUT, rows, inputs, p, thresholds, records, selection, snapshot)
        require(load_validation()[-1] == snapshot, 'Frozen upstream artifacts changed while rendering.')
        artifacts = [path for path in OUT.rglob('*') if path.is_file() and path.name not in ['complete.json', '.write.lock']]
        data.atomic_json(OUT / 'complete.json', dict(source_snapshot=snapshot, tests_read=False,
            development_only=True, selection_changed=False, primary_changed=False, limitation=LIMITATION,
            artifact_sha256={str(path.relative_to(OUT)): data.digest(path) for path in artifacts}))


def preflight():
    """Synthetic frozen probabilities and report only; never use formal artifacts."""
    import copy
    import py_compile
    from unittest import mock
    from matplotlib.figure import Figure
    folder = Path(tempfile.mkdtemp(prefix='systemness-head-probes-report-preflight-', dir='/tmp'))
    checks = []; pieces = []; entries = []; prompt = 0
    for doc in range(12):
        for role in data.ROLES:
            offset = data.uniform_indices(192, 64)
            pieces.append(pd.DataFrame(dict(prompt_ix=prompt, message_ix=0, token_ix=offset + 4,
                token_id=offset + 1000, role=role, document_id=doc, group_id=f'c4-{doc}',
                profile='single', source_set='main', split='validation', wrapper='native')))
            entries.append(dict(prompt_ix=prompt, document_id=doc, group_id=f'c4-{doc}', length=198,
                prompt='SYNTHETIC CPU FIXTURE ONLY', split='validation', wrapper='native', profile='single'))
            prompt += 1
    for di, profile in enumerate(['multi', 'natural', 'long-single', 'long-full']):
        for role in data.ROLES:
            offset = data.uniform_indices(1024 if profile == 'long-full' else 192, 128 if profile == 'long-full' else 64)
            pieces.append(pd.DataFrame(dict(prompt_ix=prompt, message_ix=0, token_ix=offset + 4,
                token_id=offset + 2000, role=role, document_id=100 + di, group_id=f'c4-{100 + di}',
                profile=profile, source_set='long-full' if profile == 'long-full' else 'main', split='validation', wrapper='native')))
            prompt += 1
    rows = pd.concat(pieces, ignore_index=True); rows['sample_ix'] = np.arange(len(rows)) + 800000
    inputs = pd.DataFrame(entries); chosen, single = gallery_sources(rows, inputs, expected_documents=12)
    require(chosen == list(range(12)) and len(single) == 12 * 5 * 64, 'Fixed synthetic gallery identities changed.')
    checks.append('fixed12_sources_five_matched_native_versions64_sampled_positions')
    def rejects(action, label):
        try: action()
        except (RuntimeError, KeyError, ValueError, FileNotFoundError): checks.append(label)
        else: raise RuntimeError(f'Synthetic fault accepted: {label}')
    corrupt = rows.copy(); corrupt.loc[0, 'token_id'] += 1
    rejects(lambda: gallery_sources(corrupt, inputs, 12), 'reject_mismatched_five_role_content_tokens')
    corrupt_inputs = inputs.copy(); corrupt_inputs.loc[0, 'document_id'] = 999
    rejects(lambda: gallery_sources(rows, corrupt_inputs, 12), 'reject_gallery_input_source_mapping')
    rejects(lambda: gallery_sources(rows.iloc[1:], inputs, 12), 'reject_missing_scored_body_position')
    y = probes.labels(rows); full_offset = body_positions(rows[rows.profile == 'long-full'])
    probability = {}; thresholds = {name: .5 for name in NAMES}; records = {}
    for mi, name in enumerate(NAMES):
        p = np.full((len(rows), 5), .04, np.float32); p[np.arange(len(rows)), y] = .84
        if mi:
            error_ids = full_offset[(full_offset >= 512 + mi * 64)
                & rows.loc[full_offset.index, 'role'].eq('system')].index
            p[error_ids, 0] = .20; p[error_ids, 1] = .68
        probability[name] = p
        values = {profile: probes.metrics(p[rows.profile == profile], y[rows.profile == profile], .5) for profile in long_probes.PROFILES}
        tail = full_offset[full_offset >= 512].index
        validation = dict(system_threshold=.5, profiles=values,
            long_full_tail=probes.metrics(p[tail], y[tail], .5),
            mean_system_f1=float(np.mean([values[k]['system_f1'] for k in long_probes.PROFILES])))
        records[name] = dict(validation=validation, model_sha256=data.text_digest('synthetic ' + name),
            validation_probabilities_sha256=data.text_digest('synthetic probability ' + name))
    guards = {name: heads.promotion_guard(records[name]['validation'], records['dense128']['validation']) for name in heads.CONFIGS}
    selection = dict(candidate='joint-mlp128', promoted=False, candidate_guards=guards,
        promotion_rejection_reasons=['no_guard_eligible_candidate', 'mean_system_f1_not_improved'])
    ppath = folder / 'synthetic-probabilities.npz'
    def save_p(p, identities):
        np.savez_compressed(ppath, p=p, sample_ix=identities)
        return data.digest(ppath)
    sha = save_p(probability['dense128'], rows.sample_ix.to_numpy())
    require(np.array_equal(probability_file(ppath, rows.sample_ix.to_numpy(), sha), probability['dense128']), 'Synthetic probability load changed values.')
    checks.append('probability_dtype_finite_normalization_sample_identity_and_sha')
    swapped = rows.sample_ix.to_numpy().copy(); swapped[:2] = swapped[:2][::-1]
    sha = save_p(probability['dense128'], swapped)
    rejects(lambda: probability_file(ppath, rows.sample_ix.to_numpy(), sha), 'reject_probability_sample_order_drift')
    invalid = probability['dense128'].copy(); invalid[0, 0] = np.nan
    sha = save_p(invalid, rows.sample_ix.to_numpy())
    rejects(lambda: probability_file(ppath, rows.sample_ix.to_numpy(), sha), 'reject_nonfinite_probability')
    invalid[0] = 0.; sha = save_p(invalid, rows.sample_ix.to_numpy())
    rejects(lambda: probability_file(ppath, rows.sample_ix.to_numpy(), sha), 'reject_unnormalized_probability')
    rejects(lambda: probability_file(ppath, rows.sample_ix.to_numpy(), '0' * 64), 'reject_probability_file_hash_drift')
    changed_records = copy.deepcopy(records); changed_records['joint-linear']['validation']['profiles']['single']['system_f1'] -= .01
    rejects(lambda: metrics_at_frozen_thresholds(rows, probability, thresholds, changed_records), 'reject_probability_frozen_metric_mismatch')
    changed_thresholds = dict(thresholds, **{'joint-linear': .6})
    rejects(lambda: metrics_at_frozen_thresholds(rows, probability, changed_thresholds, records), 'reject_threshold_drift_even_when_predictions_unchanged')
    changed_thresholds['joint-linear'] = np.nan
    rejects(lambda: metrics_at_frozen_thresholds(rows, probability, changed_thresholds, records), 'reject_nonfinite_frozen_threshold')
    saved_axes = []; original_save = Figure.savefig
    def save_figure(fig, path, *args, **kwargs):
        saved_axes.append(len(fig.axes)); return original_save(fig, path, *args, **kwargs)
    snapshot = dict(synthetic_fixture_only=True, source_sha256=data.digest(__file__), development_only=True,
        tests_read=False, selection_changed=False, primary_changed=False)
    with mock.patch.object(Figure, 'savefig', save_figure):
        metrics, gallery = render(folder / 'synthetic-report', rows, inputs, probability, thresholds,
            records, selection, snapshot, expected_documents=12)
    require(len(metrics) == 5 * 6 and gallery['models'] == gallery['panels'] == 5
        and gallery['source_ids'] == list(range(12)) and gallery['sampled_body_positions_per_role'] == 64,
        'Synthetic report omitted a model/profile or changed fixed source selection.')
    plot_folder = folder / 'synthetic-report/plots'
    require(len(list(plot_folder.glob('*.png'))) == len(list(plot_folder.glob('*.pdf'))) == 14
        and saved_axes.count(25) == 24 and saved_axes.count(1) == 4,
        'Expected12 five-by-five scatter pairs plus2 summary/position figure pairs.')
    checks += ['five_models_six_profiles_at_frozen_thresholds_without_reselection',
        'synthetic12_fixed5x5_tomato_pairs_plus_f1_and_long_full_position_pairs',
        'all_guard_and_nonpromotion_reasons_rendered_without_changing_selection']
    py_compile.compile(__file__, cfile=str(folder / 'systemness_head_probes_report.pyc'), doraise=True)
    report = dict(cpu_only=True, formal_outputs_written=False, test_inputs_read=False, gpu_run=False,
        source_sha256=data.digest(__file__), checks=checks, check_count=len(checks),
        synthetic_only=True, synthetic_rows=len(rows), models=5, profiles=6, fixed_scatter_sources=12,
        panels=5, model_columns=5, png=14, pdf=14,
        upstream_completion_available=(heads.OUT / 'complete.json').is_file(), limitation=LIMITATION)
    data.atomic_json(folder / 'report.json', report)
    print(json.dumps(dict(preflight=str(folder / 'report.json'), **report), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true'); parser.add_argument('--write', action='store_true')
    args = parser.parse_args(); require(not (args.preflight and args.write), 'Choose CPU preflight or explicit write.')
    if args.preflight: preflight()
    elif args.write: write()
    else: parser.print_help()


if __name__ == '__main__':
    main()
