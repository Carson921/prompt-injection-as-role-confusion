"""Fixed 16-source development probability plots, after an independent audit.

No fitting, LM, test source access, probability recomputation or model selection.
The source ranks were fixed before binary-residual training. All observations
are reused development data; these plots are not independent confirmation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / 'outputs/local-gptoss120b'
TRAIN = BASE / 'systemness-binary-residual'
PREP = BASE / 'systemness-binary-residual-preparation'
OUT = BASE / 'systemness-binary-residual-development-plots'
PLAN = PREP / 'development-gallery-plan.json'
PLAN_SHA = '1b2cecc4f345295da55aeab5785d026d8c2a08c0f4bdc656fdb4e641651a9466'
DESIGN_SHA = 'c68a32d55e732f678f77a7ef91d8d966a416f118d43663f913b3c9f2d14e09c5'
APPROVED_AUDITOR_SHA = None  # Root pins the reviewed final auditor before rendering.
CACHE = BASE / 'systemness-expansion/cache'
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
MODELS = ['oldhead', 'naturalCE', 'ensemble', 'binary-new', 'binary-matched']
RECORDS = ['oldhead-max33', 'naturalCE-max33', 'ensemble-max33', 'new-max33', 'matched-max33']
COLORS = ['#64748b', '#00a6f4', '#b56bd6', '#00b982', '#e4983f']


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def gate(audit_path):
    require(APPROVED_AUDITOR_SHA is not None, 'Final independent auditor is not pinned.')
    names = {'systemness_binary_residual.py', 'long_data_binary_residual_results_audit.py'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not any(Path(arg.decode(errors='replace')).name in names for arg in args if arg),
                'Trainer/auditor is live; no partial plots.')
    require(digest(PLAN) == PLAN_SHA, 'Predetermined gallery selection changed.')
    require(digest(PREP / 'root-prospective-design.json') == DESIGN_SHA, 'Root design changed.')
    audit = read(audit_path)
    require(audit['stage'] == 'actual_completed_binary_residual_development_audit'
            and audit['audit_passed'] is True
            and audit['auditor_source_sha256'] == APPROVED_AUDITOR_SHA
            and audit['test_data_or_results_read'] is False,
            'A passing approved independent development audit is required.')
    require(digest(TRAIN / 'complete.json') == audit['trainer_complete_sha256']
            and digest(TRAIN / 'protocol.json') == audit['trainer_protocol_sha256']
            and digest(HERE / 'systemness_binary_residual.py') == audit['trainer_source_sha256']
            and read(TRAIN / 'status.json')['stage'] == 'validation_complete',
            'Trainer completion/source/protocol differs from the actual audit.')
    return audit, read(PLAN)


def probability(audit, path, rows):
    path = Path(path)
    require(audit['probability_bindings'].get(str(path.resolve())) == digest(path),
            'Plot input was not bound by the independent actual audit: ' + str(path))
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'p', 'sample_ix'}
                and np.array_equal(saved['sample_ix'], rows.sample_ix), 'Probability row identity changed.')
        p = saved['p'].copy()
    require(p.shape == (len(rows), 5) and p.dtype == np.float32
            and np.isfinite(p).all() and (p >= 0).all()
            and np.allclose(p.sum(1), 1., rtol=0, atol=2e-5), 'Invalid plot probabilities.')
    return p


def counts(p, truth, threshold):
    predicted = p.argmax(1)
    detected = p[:, 0] >= float(threshold)
    system = truth == 0
    tp = int((detected & system).sum())
    fp = int((detected & ~system).sum())
    fn = int((~detected & system).sum())
    cot = truth == 2
    return dict(tokens=len(p), five_accuracy=float((predicted == truth).mean()),
                system_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                system_tp=tp, system_fp=fp, system_fn=fn, cot_support=int(cot.sum()),
                cot_correct=int((cot & (predicted == 2)).sum()),
                cot_to_assistant=int((cot & (predicted == 3)).sum()),
                cot_to_system=int((cot & (predicted == 0)).sum()))


def render(audit_path):
    audit, plan = gate(audit_path)
    require(not (OUT / 'complete.json').exists(), 'Gallery already completed; do not overwrite it.')
    rows = {}
    for bank, path in [('old', Path('/dev/shm/role-analysis-top8-heads/rows.parquet')),
                       ('public', Path('/dev/shm/role-analysis-natural-dev-heads/rows.parquet'))]:
        require(digest(path) == plan['rows_sha256'][bank], 'Gallery row bank changed.')
        frame = pd.read_parquet(path)
        rows[bank] = frame[frame.split.eq('validation')].reset_index(drop=True)
    require(len(rows['old']) == 173032 and len(rows['public']) == 16713, 'Development counts changed.')
    records = {r['record']: r for r in read(TRAIN / 'records.json')}
    require(digest(TRAIN / 'records.json') == audit['trainer_records_sha256'], 'Audited record thresholds changed.')
    thresholds = {model: records[record]['validation']['system_threshold']
                  for model, record in zip(MODELS, RECORDS)}
    paths = {
        'oldhead': (CACHE / 'natural-fit-probes', 'seed-20261021'),
        'naturalCE': (CACHE / 'natural-fit-probes', 'new'),
        'ensemble': (CACHE / 'natural-fit-probes', 'ensemble'),
        'binary-new': (CACHE / 'binary-residual', 'new'),
        'binary-matched': (CACHE / 'binary-residual', 'matched'),
    }
    values = {model: {bank: probability(audit, folder / f'{prefix}-{bank}-validation.npz', rows[bank])
                      for bank in rows} for model, (folder, prefix) in paths.items()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    OUT.mkdir(exist_ok=True)
    table, files = [], []
    labels = ['Systemness', 'Userness', 'CoTness', 'Assistantness', 'Toolness']
    for entry in plan['entries']:
        frame = rows[entry['bank']]
        group = frame[frame.prompt_ix.eq(entry['prompt_ix'])].sort_values('token_ix')
        require(len(group) == entry['observations']
                and array_sha(group.sample_ix.to_numpy()) == entry['sample_ix_sha256'],
                'Fixed gallery source membership/order changed.')
        if 'source_uid' in entry:
            require(group.source_uid.eq(entry['source_uid']).all(), 'Gallery public source changed.')
        x, ix = group.token_ix.to_numpy(), group.index.to_numpy()
        truth = np.asarray([ROLES.index(role) for role in group.role], np.int64)
        fig, axes = plt.subplots(5, 5, figsize=(23, 8), sharex=True, sharey=True)
        for column, model in enumerate(MODELS):
            p = values[model][entry['bank']][ix]
            result = counts(p, truth, thresholds[model])
            table.append(dict(example=entry['slug'], cohort=entry['cohort'], model=model,
                              threshold=thresholds[model], **result))
            for role, label in enumerate(labels):
                ax = axes[role, column]
                for true_role, color in enumerate(COLORS):
                    mask = truth == true_role
                    ax.scatter(x[mask], p[mask, role], s=2.5, color=color, alpha=.7, rasterized=True)
                for boundary in group.groupby('message_ix').token_ix.min():
                    ax.axvline(boundary, lw=.5, color='#d1d5db', zorder=0)
                if role == 0:
                    ax.axhline(thresholds[model], color='#334155', ls='--', lw=.65)
                    if model == 'oldhead':
                        ax.axhline(.620969696969697, color='#64748b', ls=':', lw=.6)
                ax.set_ylim(-.03, 1.03)
                ax.set_yticks([0, 1], ['0%', '100%'])
                ax.spines[['top', 'right']].set_visible(False)
                if column == 0:
                    ax.set_ylabel(label, fontsize=9)
            axes[0, column].set_title(f"{model} | five-class {result['five_accuracy']:.1%}\n"
                                      f"system F1 {result['system_f1']:.1%} | global tau {thresholds[model]:.3f}",
                                      fontsize=9)
            axes[-1, column].set_xlabel('Token position in native input', fontsize=9)
        handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                          markersize=4, label=role) for role, color in zip(ROLES, COLORS)]
        fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=9)
        fig.suptitle('DEVELOPMENT ONLY | ' + entry['slug'] + ' | ' + entry.get('source_uid',
                     'legacy prompt ' + str(entry['prompt_ix'])), y=.943, fontsize=10)
        fig.text(.5, .008, 'Fixed source ranks, unsmoothed content-token probabilities; gaps are not filled. '
                 'Public S/U/A has no CoT/tool truth. Dashed: max33; dotted oldhead: original tau.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .03, 1, .90])
        for extension in ['png', 'pdf']:
            path = OUT / (entry['slug'] + '.' + extension)
            fig.savefig(path, dpi=140)
            files.append(path)
        plt.close(fig)
    require(len(table) == 80 and len(files) == 32, 'Fixed 16-source/5-model gallery is incomplete.')
    pd.DataFrame(table).to_csv(OUT / 'example-metrics.csv', index=False)
    index = ['Fixed development gallery: 16 examples, five readouts', '',
             'Reused development data; not independent confirmation. Every predetermined source is retained.', '']
    index += [f"- [{e['slug']}]({e['slug']}.png) ([PDF]({e['slug']}.pdf))" for e in plan['entries']]
    (OUT / 'gallery-index.md').write_text('\n'.join(index) + '\n')
    files += [OUT / 'example-metrics.csv', OUT / 'gallery-index.md']
    completion = dict(stage='completed_fixed_binary_development_gallery', examples=16, readouts=MODELS,
                      png=16, pdf=16, source_ranks_changed=False, independent_test=False,
                      LM_run=False, probabilities_recomputed=False, audit_sha256=digest(audit_path),
                      trainer_complete_sha256=audit['trainer_complete_sha256'],
                      plot_source_sha256=digest(__file__), gallery_plan_sha256=PLAN_SHA,
                      inputs={str((folder / f'{prefix}-{bank}-validation.npz').resolve()):
                              digest(folder / f'{prefix}-{bank}-validation.npz')
                              for folder, prefix in paths.values() for bank in rows},
                      artifacts={path.name: digest(path) for path in files})
    (OUT / 'complete.json').write_text(json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(out=str(OUT), png=16, pdf=16, metrics=80, complete_sha256=digest(OUT/'complete.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.render:
        require(args.audit is not None, 'Provide the completed independent actual audit.')
        render(args.audit)
    else:
        parser.print_help()
