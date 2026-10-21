"""Report a completed diagnostic head scan without changing any model choice.

Default: help. --write requires scan-complete.json and validation-complete.json,
then writes a validation-only text/CSV report and static PNG/PDF figures. It
reads result JSON/CSV only, never activations, probability caches, tests or
confirmation inputs. --preflight runs an in-memory synthetic schema smoke.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = HERE / 'outputs/local-gptoss120b'
OUT = BASE / 'systemness-head-scan'
PROFILES = ['single', 'multi', 'natural', 'long-single', 'long-full', 'long-full-tail']
SCALAR_METRICS = ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy',
    'system_false_positive_rate', 'accuracy']
FIELDS = ['model', 'profile', 'tokens', 'system_threshold', 'mean_five_profile_system_f1',
    'original_four_profile_system_f1', 'system_precision', 'system_recall', 'system_f1',
    'system_balanced_accuracy', 'system_false_positive_rate', 'five_role_accuracy',
    'profile_cot_recall', 'natural_cot_recall', 'model_sha256']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric_for(result, profile):
    return result['long_full_tail'] if profile == 'long-full-tail' else result['profiles'][profile]


def validate_result(result):
    require(set(result['profiles']) == set(PROFILES[:5]), 'Expected the original five native validation profiles.')
    require(math.isclose(result['mean_system_f1'], sum(result['profiles'][name]['system_f1'] for name in PROFILES[:5]) / 5,
                        rel_tol=0., abs_tol=1e-12), 'Five-profile F1 mean differs from its component metrics.')
    require(math.isclose(result['original_mean_system_f1'], sum(result['profiles'][name]['system_f1'] for name in PROFILES[:4]) / 4,
                        rel_tol=0., abs_tol=1e-12), 'Four-profile F1 mean differs.')
    for profile in PROFILES:
        metric = metric_for(result, profile)
        require(metric['system_threshold'] == result['system_threshold'], 'A model used per-profile thresholds.')
        for name in SCALAR_METRICS:
            value = metric[name]
            require(value is None or (math.isfinite(value) and 0 <= value <= 1), f'Invalid {profile}/{name}.')


def load_completed():
    for name in ['scan-complete.json', 'validation-complete.json']:
        require((OUT / name).is_file(), f'Head scan is incomplete: {name}; no report is written.')
    scan = read(OUT / 'scan-complete.json'); complete = read(OUT / 'validation-complete.json')
    selection = read(OUT / 'head-selection.json'); metadata = read(OUT / 'validation-metrics.json')
    require(scan['plan_sha256'] == complete['plan_sha256'] == sha(OUT / 'plan.json'), 'Head scan/validation plan differs.')
    require(scan['head_selection_sha256'] == metadata['head_selection_sha256'] == sha(OUT / 'head-selection.json'),
            'The selected head set changed between scan and validation.')
    require(complete['scan_complete_sha256'] == sha(OUT / 'scan-complete.json')
            and complete['metrics_sha256'] == sha(OUT / 'validation-metrics.json'), 'Completed validation metadata changed.')
    require(scan['model_sha256'] == complete['model_sha256'] == metadata['model_sha256'] == selection['model_sha256'],
            'Scan and validation reference different models.')
    require(metadata['validation_only'] is True and metadata['tests_read'] is False
            and scan['tests_read'] is False and complete['tests_read'] is False
            and selection['test_used_for_selection'] is False, 'Head results are not declared development-only.')
    require(metadata['probabilities_sha256'] == complete['probabilities_sha256'], 'Validation probability references differ.')
    result = metadata['validation']; validate_result(result)
    dense_path = BASE / 'systemness-long-full/candidates/dense-mlp128.json'
    dense = read(dense_path); dense_selection = read(BASE / 'systemness-long-full/selection.json')
    require(dense['candidate'] == 'dense-mlp128' and dense['tests_read'] is False
            and dense_selection['candidates']['dense-mlp128'] == sha(dense_path), 'Frozen dense128 candidate changed.')
    validate_result(dense['validation'])
    scores = pd.read_csv(OUT / 'discovery-head-metrics.csv', float_precision='round_trip').sort_values('head_id').reset_index(drop=True)
    require(len(scores) == 2304 and np.array_equal(scores.head_id, np.arange(2304))
            and np.array_equal(scores.layer, np.arange(2304) // 64)
            and np.array_equal(scores['head'], np.arange(2304) % 64), '36x64 head score mapping changed.')
    columns = ['discovery_doc_macro_auroc', 'discovery_tail_doc_macro_auroc', 'selection_score']
    require(np.isfinite(scores[columns].to_numpy()).all() and scores[columns].ge(0).all().all()
            and scores[columns].le(1).all().all(), 'Invalid head AUROC scores.')
    require(np.allclose(scores.selection_score, .5 * (scores.discovery_doc_macro_auroc + scores.discovery_tail_doc_macro_auroc),
                        atol=1e-12, rtol=0), 'Head score no longer equals mean all/tail AUROC.')
    order = np.lexsort((scores.head_id.to_numpy(), -scores.discovery_tail_doc_macro_auroc.to_numpy(),
                       -scores.selection_score.to_numpy()))
    selected = np.array(selection['selected_head_ids'], dtype=np.int64)
    require(selected.shape == (8,) and np.array_equal(order[:8], selected)
            and np.array_equal(scores.iloc[order]['rank'], np.arange(1, 2305)), 'Selected head ranking differs from the declared rule.')
    require(selection['selected_layer_head_pairs'] == [[int(head // 64), int(head % 64)] for head in selected],
            'Portable layer/head IDs differ from discovery scores.')
    models = {'dense-mlp128': dict(result=dense['validation'], sha256=dense['model_sha256']),
              'top8-head-lda': dict(result=result, sha256=metadata['model_sha256'])}
    primary_path = BASE / 'systemness-final-readout/selection.json'
    primary = read(primary_path) if primary_path.exists() else None
    return models, scores, selected, selection, primary


def summary_rows(models):
    rows = []
    for name, model in models.items():
        result = model['result']
        for profile in PROFILES:
            metric = metric_for(result, profile)
            rows.append(dict(model=name, profile=profile, tokens=metric['tokens'],
                system_threshold=result['system_threshold'], mean_five_profile_system_f1=result['mean_system_f1'],
                original_four_profile_system_f1=result['original_mean_system_f1'],
                **{key: metric[key] for key in SCALAR_METRICS if key != 'accuracy'},
                five_role_accuracy=metric['accuracy'], profile_cot_recall=metric['recall'].get('cot'),
                natural_cot_recall=result['profiles']['natural']['recall'].get('cot'), model_sha256=model['sha256']))
    return rows


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.2f}'


def render(models, scores, selected, selection, primary):
    rows = summary_rows(models)
    dense = models['dense-mlp128']['result']; heads = models['top8-head-lda']['result']
    layer_counts = pd.Series(selected // 64).value_counts().sort_index()
    unique_heads = sorted(set((selected % 64).tolist()))
    lines = ['GPT-OSS-120B pre-projection head diagnostic — VALIDATION ONLY', '',
        'Frozen native MXFP4 model;36 zero-based layers x64 query heads,64 dimensions/head.',
        'LDA statistics fit on101 original fit long documents; actual discovery probabilities use31 other fit documents.',
        'The discovery31 documents and the original long-full validation31 documents are disjoint.',
        'Top8 are selected from2304 heads using average all-position/tail512+ source-doc-macro system AUROC,',
        'then refitted on all132 original long-fit documents. Discovery scores are selected development results, not held-out F1.',
        'Inference averages8 current-token five-class probability vectors. No role label, position or pairing is an inference input.', '',
        'Each model uses one global validation threshold across all five profiles with the same per-profile FPR rule.',
        'Thresholds may differ between models. Mean5 includes long-full and must not be compared directly with an earlier four-profile89% mean.',
        'Values are percentages; differences are percentage points (head minus dense128).', '',
        'Aggregate validation comparison', 'model             Mean5-F1 Original4-F1 Original4-fiveAcc natural-CoT']
    for name, value in models.items():
        result = value['result']
        lines.append(f'{name:17s} {pct(result["mean_system_f1"]):>8s} {pct(result["original_mean_system_f1"]):>12s} '
            f'{pct(result["original_mean_accuracy"]):>17s} {pct(result["profiles"]["natural"]["recall"].get("cot")):>11s}')
    lines += [f'Mean5 F1 difference: {100 * (heads["mean_system_f1"] - dense["mean_system_f1"]):+.2f}pp', '',
        'Per-profile validation metrics', 'model/profile                    tokens      P      R     F1   BAcc fiveAcc    FPR']
    for row in rows:
        label = f'{row["model"]}/{row["profile"]}'
        lines.append(f'{label:34s} {row["tokens"]:7d} {pct(row["system_precision"]):>6s} {pct(row["system_recall"]):>6s} '
            f'{pct(row["system_f1"]):>6s} {pct(row["system_balanced_accuracy"]):>6s} {pct(row["five_role_accuracy"]):>7s} '
            f'{pct(row["system_false_positive_rate"]):>6s}')
    lines += ['', 'Selected query-head pattern (descriptive; no causal attribution)',
        'rank  layer head   discovery-all-AUROC discovery-tail-AUROC selection-score']
    for rank, head in enumerate(selected, 1):
        row = scores.iloc[int(head)]
        lines.append(f'{rank:4d} {int(head // 64):6d} {int(head % 64):4d} '
            f'{pct(row.discovery_doc_macro_auroc):>21s} {pct(row.discovery_tail_doc_macro_auroc):>22s} {pct(row.selection_score):>15s}')
    lines += [f'Selected layers/counts: {json.dumps({int(layer): int(count) for layer, count in layer_counts.items()}, sort_keys=True)}',
        f'Unique selected layers: {len(layer_counts)}; query-head indices: {unique_heads}.',
        'A head index reused at different layers denotes different parameters. Layer concentration is descriptive;',
        'this selection does not establish that a layer or attention type causes systemness.', '',
        'Position, source and selection limitations',
        'Original long-full validation:31 source documents,128 uniformly sampled content positions/role/document;',
        'tail512..1023 contains64 sampled positions/role/document (1984 system positions), not every token in the interval.',
        'The31 long-full documents are a subset of the original200 C4 validation sources;10 also have old long-single versions.',
        'The all/tail summary does not establish a continuous position curve or guarantee every1024-token position is classified correctly.',
        'Natural-role metrics come from the original40-conversation validation profile; CoT and five-role accuracy can trade off with systemness.',
        'Head selection used only long-full fit texts, whereas dense128 used mixed profiles. Architecture, training distribution and objective differ;',
        'the comparison is a diagnostic benchmark, not a matched causal ablation of head selection alone.',
        'Choosing8 out of2304 on31 fit-discovery sources and repeatedly reusing the original validation can overfit.',
        'Tokens/roles from one source are dependent; discovery AUROC, validation F1 and independent confirmation are distinct evidence.', '',
        'No automatic promotion or current-primary change is performed by this report.']
    if primary is not None:
        lines += [f'Existing frozen primary: {primary.get("originating_trial", "unknown")}/{primary["candidate"]}',
            f'Existing primary model: {primary["model_path"]}', f'Existing primary model SHA256: {primary["model_sha256"]}']
    lines += [f'Diagnostic head model: {selection["model_path"]}', f'Diagnostic head model SHA256: {selection["model_sha256"]}',
        'Figures: plots/discovery-head-score-heatmap.{png,pdf}, plots/validation-systemness-comparison.{png,pdf}.',
        'This report verifies result JSON/CSV consistency and hashes, but does not recompute predictions or open model/probability files.']
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(rows)
    return '\n'.join(lines) + '\n', stream.getvalue()


def figures(models, scores, selected):
    configured = os.environ.get('MPLCONFIGDIR', '')
    if not configured or configured.startswith('/workspace'):
        os.environ['MPLCONFIGDIR'] = '/tmp/systemness-head-report-matplotlib'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import matplotlib.patheffects as effects
    folder = OUT / 'plots'; folder.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(14, 10))
    image = axis.imshow(scores.selection_score.to_numpy().reshape(36, 64), aspect='auto',
        interpolation='nearest', cmap='viridis', vmin=0., vmax=1.)
    axis.set_xticks(np.arange(0, 64, 4)); axis.set_yticks(np.arange(36))
    axis.set_xlabel('Query head index (zero based)'); axis.set_ylabel('Decoder layer (zero based)')
    axis.set_title('Fit-discovery head ranking:31 source documents\nMean of source-doc-macro all-position and512+ system AUROC', pad=12)
    for rank, head in enumerate(selected, 1):
        x, y = int(head % 64), int(head // 64)
        axis.scatter(x, y, marker='s', s=150, facecolors='none', edgecolors='white', linewidths=1.4)
        axis.text(x, y, str(rank), ha='center', va='center', color='white', fontsize=8,
            path_effects=[effects.withStroke(linewidth=2., foreground='black')])
    bar = fig.colorbar(image, ax=axis, pad=.02, ticks=[0., .25, .5, .75, 1.]); bar.set_label('Discovery selection score (AUROC)')
    fig.text(.5, .01, 'Outlined cells: frozen top8, ranked1–8. Selected fit-discovery scores are not held-out validation F1.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, 1))
    for suffix in ['png', 'pdf']: fig.savefig(folder / f'discovery-head-score-heatmap.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    colors = {'dense-mlp128': '#476d9f', 'top8-head-lda': '#b74e3e'}
    y = np.arange(len(PROFILES)); metrics = [('system_precision', 'System precision'), ('system_recall', 'System recall'), ('system_f1', 'System F1')]
    for axis, (metric, title) in zip(axes, metrics):
        for index, (name, model) in enumerate(models.items()):
            values = [metric_for(model['result'], profile)[metric] for profile in PROFILES]
            axis.barh(y + (-.18 if index == 0 else .18), [np.nan if value is None else value for value in values],
                height=.32, color=colors[name], label=name)
        axis.set_title(title); axis.set_xlim(0, 1); axis.xaxis.set_major_formatter(PercentFormatter(1.))
        axis.grid(axis='x', alpha=.2); axis.set_axisbelow(True)
    axes[0].set_yticks(y, labels=PROFILES); axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Original five-profile validation plus long-full tail — no automatic promotion', y=.99)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    for suffix in ['png', 'pdf']: fig.savefig(folder / f'validation-systemness-comparison.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def write():
    models, scores, selected, selection, primary = load_completed()
    text, summary = render(models, scores, selected, selection, primary)
    figures(models, scores, selected)
    for name, content in [('head-validation-report.txt', text), ('head-validation-comparison.csv', summary)]:
        path = OUT / name; temporary = OUT / (name + '.tmp')
        temporary.write_text(content); temporary.replace(path)
    print(json.dumps(dict(report=str(OUT / 'head-validation-report.txt'), comparison=str(OUT / 'head-validation-comparison.csv'),
        plots=str(OUT / 'plots'), validation_only=True, selections_or_models_modified=False)))


def preflight():
    def metric():
        return dict(tokens=100, system_threshold=.6, system_precision=.82, system_recall=.78,
            system_f1=.8, system_balanced_accuracy=.86, system_false_positive_rate=.03,
            accuracy=.84, recall=dict(cot=.55))
    result = dict(system_threshold=.6, mean_system_f1=.8, original_mean_system_f1=.8,
        original_mean_accuracy=.84, profiles={profile: metric() for profile in PROFILES[:5]}, long_full_tail=metric())
    validate_result(result)
    models = {name: dict(result=result, sha256='synthetic') for name in ['dense-mlp128', 'top8-head-lda']}
    scores = pd.DataFrame(dict(head_id=np.arange(2304), discovery_doc_macro_auroc=np.full(2304, .5),
        discovery_tail_doc_macro_auroc=np.full(2304, .5), selection_score=np.full(2304, .5)))
    selected = np.arange(8)
    selection = dict(model_path='synthetic-not-a-model', model_sha256='synthetic')
    text, summary = render(models, scores, selected, selection, None)
    rows = list(csv.DictReader(io.StringIO(summary)))
    require(len(rows) == 12 and 'No automatic promotion' in text and '1984 system positions' in text
            and '0.55' == rows[0]['natural_cot_recall'], 'Synthetic report schema smoke failed.')
    print(json.dumps(dict(cpu_only=True, synthetic_profile_rows=12, formal_outputs_written=False, source_sha256=sha(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Synthetic CPU schema smoke; no report or plots written.')
    action.add_argument('--write', action='store_true', help='Write validation reports/figures only after both head stages finish.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.write: write()
    else: parser.print_help()
