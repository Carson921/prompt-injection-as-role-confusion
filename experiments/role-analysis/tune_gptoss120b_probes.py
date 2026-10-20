"""Tune cached 120B role probes without loading or changing MXFP4 model weights.

From the repository root:
  HF_HOME=/root/.cache/huggingface-role-probe .venv/bin/python -u \
    experiments/role-analysis/tune_gptoss120b_probes.py

Searches 16/20 x scaling off/on x five C values for SUCA and SUCAT. Candidates
are selected on validation accuracy, then NLL, never on held-out test results.
Two separate protocols retain the original prompt test split and measure a
source-document-disjoint 80/10/10 split. FP64 is a diagnostic of the two selected
document-split probes, not another hyperparameter chosen using test accuracy.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import run_local_gptoss120b as base

HERE, ROOT = base.HERE, base.ROOT
SOURCE = base.OUT
OUT = SOURCE / 'tuning'
LAYERS = [16, 20]
ROLE_SPACES = {'suca': ['system', 'user', 'cot', 'assistant'],
               'sucat': ['system', 'user', 'cot', 'assistant', 'tool']}
CS = [.0005, .005, .05, .5, 5.]
PROTOCOLS = ['legacy', 'document']
atomic_json = base.atomic_json


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def progress(stage, **kwargs):
    value = dict(stage=stage, unix_time=time.time(), **kwargs)
    atomic_json(OUT / 'status.json', value)
    print(json.dumps(value), flush=True)


def prepare():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    meta = json.loads((SOURCE / 'manifest.json').read_text())
    assert json.loads((SOURCE / 'activations/progress.json').read_text())['complete']
    assert all((SOURCE / f'activations/layer-{i}.f16').stat().st_size == meta['tokens']*2880*2
               for i in LAYERS)
    inputs = pd.read_parquet(SOURCE / 'training_inputs.parquet')
    question = inputs.set_index('prompt_ix').question_ix
    with (HERE / 'outputs/probes/gptoss-120b.pkl').open('rb') as f:
        original = pickle.load(f)
    rng = np.random.RandomState(20261020)
    documents = np.sort(inputs.question_ix.unique())
    shuffled = rng.permutation(documents)
    doc_sets = dict(test=sorted(shuffled[:25].tolist()), validation=sorted(shuffled[25:50].tolist()),
                   fit=sorted(shuffled[50:].tolist()))
    split = {}
    for name, roles in ROLE_SPACES.items():
        prior = next(p for p in original if p['role_space'] == roles and p['layer_ix'] == 16)
        train_ids, test_ids = prior['train_prompt_ids'], prior['test_prompt_ids']
        train_docs = sorted({int(question.loc[p]) for p in train_ids})
        fit_docs, val_docs = train_test_split(train_docs, test_size=.15, random_state=20261020)
        split[name] = {
            'legacy': dict(fit=[p for p in train_ids if question.loc[p] in set(fit_docs)],
                           validation=[p for p in train_ids if question.loc[p] in set(val_docs)],
                           test=test_ids, refit=train_ids,
                           note='Original held-out prompt IDs; document-disjoint inner validation. Original outer split can share documents.'),
            'document': {**{part:inputs[(inputs.question_ix.isin(ids)) & inputs.role.isin(roles)].prompt_ix.tolist()
                            for part, ids in doc_sets.items()},
                         'refit':inputs[(~inputs.question_ix.isin(doc_sets['test'])) & inputs.role.isin(roles)].prompt_ix.tolist(),
                         'note':'199/25/25 source documents for fit/validation/test; refit uses 224 documents.'},
        }
        for protocol, s in split[name].items():
            sets = [set(s[p]) for p in ['fit', 'validation', 'test']]
            assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
            assert set(s['refit']) == sets[0] | sets[1]
            groups = [{int(question.loc[p]) for p in s[part]} for part in ['fit', 'validation', 'test']]
            assert not groups[0] & groups[1]
            if protocol == 'document':
                assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
            s['source_documents'] = {part: sorted({int(question.loc[p]) for p in s[part]})
                                     for part in ['fit', 'validation', 'test', 'refit']}
    config = dict(layers=LAYERS, role_spaces=ROLE_SPACES, C=CS, scaling=[False, True],
                  protocols=PROTOCOLS, candidates=80, seed=20261020,
                  selection='Highest internal validation token accuracy, lowest validation NLL, lower C, lower layer; no test selection.',
                  solver=dict(penalty='l2', max_iter=5000, linesearch_max_iter=100, tol=1e-4, fit_intercept=True),
                  language_model_weights='unchanged frozen MXFP4; cached FP16 activations only',
                  source_probes_sha256=digest(HERE / 'outputs/probes/gptoss-120b.pkl'),
                  source_manifest_sha256=digest(SOURCE / 'manifest.json'),
                  source_runner_sha256=digest(Path(__file__)),
                  original_test_previously_reported=True,
                  uncertainty='Single validation split; no guarantee of accuracy improvement.')
    atomic_json(OUT / 'splits.json', split)
    atomic_json(OUT / 'config.json', config)
    progress('prepared', candidates=80)


@contextmanager
def native_log(path):
    sys.stdout.flush()
    sys.stderr.flush()
    old = [os.dup(1), os.dup(2)]
    with Path(path).open('w') as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old[0], 1)
            os.dup2(old[1], 2)
            for fd in old:
                os.close(fd)


def load_features(layer, rows, dtype):
    import cupy as cp
    import torch
    count = json.loads((SOURCE / 'manifest.json').read_text())['tokens']
    mmap = torch.from_file(str(SOURCE / f'activations/layer-{layer}.f16'),
                           shared=False, size=count*2880, dtype=torch.float16).view(-1, 2880)
    values = mmap[rows.sample_ix.tolist()].to(torch.float64 if dtype == 'float64' else torch.float32).to('cuda:0')
    result = cp.asfortranarray(cp.from_dlpack(values))
    del values, mmap
    torch.cuda.empty_cache()
    assert str(result.dtype) == dtype
    return result


def datasets(name, protocol, phase):
    import pandas as pd
    labels = pd.read_parquet(SOURCE / 'training_labels.parquet')
    split = json.loads((OUT / 'splits.json').read_text())[name][protocol]
    fit_part, eval_part = ('fit', 'validation') if phase == 'search' else ('refit', 'test')
    train = labels[labels.prompt_ix.isin(split[fit_part])]
    test = labels[labels.prompt_ix.isin(split[eval_part])]
    assert set(train.role) == set(test.role) == set(ROLE_SPACES[name])
    return train, test


def cleanup():
    import cupy as cp
    import torch
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()


def metrics(clf, x, y, roles):
    import cupy as cp
    import numpy as np
    p = clf.predict_proba(x, convert_dtype=False)
    assert cp.isfinite(p).all() and cp.allclose(p.sum(axis=1), 1, atol=2e-6)
    pred = p.argmax(axis=1)
    k = len(roles)
    matrix = cp.asnumpy(cp.bincount(y.astype(cp.int64)*k+pred, minlength=k*k).reshape(k, k))
    diagonal = matrix.diagonal()
    recall = diagonal / matrix.sum(axis=1)
    f1 = np.divide(2*diagonal, matrix.sum(axis=0)+matrix.sum(axis=1),
                   out=np.zeros(k), where=(matrix.sum(axis=0)+matrix.sum(axis=1)) > 0)
    result = dict(accuracy=float((pred == y).mean()),
                  nll=float(-cp.log(cp.clip(p[cp.arange(len(y)), y], 1e-15, 1)).mean()),
                  macro_f1=float(f1.mean()), recall=dict(zip(roles, map(float, recall))),
                  confusion=matrix.tolist(), tokens=len(y))
    return result, cp.asnumpy(pred)


def train_one(name, protocol, layer, scaling, C, dtype, phase, train, test, x, xt, scaler, tag):
    import cupy as cp
    import pandas as pd
    from cuml.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    roles = ROLE_SPACES[name]
    y = cp.asarray(train.role.map({r:i for i,r in enumerate(roles)}).to_numpy())
    yt = cp.asarray(test.role.map({r:i for i,r in enumerate(roles)}).to_numpy())
    params = json.loads((OUT / 'config.json').read_text())['solver']
    log = OUT / 'logs' / f'{tag}.log'
    started = time.time()
    clf = LogisticRegression(C=C, output_type='cupy', **params)
    with native_log(log):
        # cuML 25.10 preserves either supported X dtype, and must be allowed to
        # cast its internally label-encoded uint8 y to that floating dtype.
        clf.fit(x, y, convert_dtype=True)
    assert str(clf.dtype) == str(clf.coef_.dtype) == str(clf.solver_model.dtype) == dtype
    train_metrics, _ = metrics(clf, x, y, roles)
    eval_metrics, predictions = metrics(clf, xt, yt, roles)
    warnings = [line for line in log.read_text().splitlines() if 'warning' in line.lower() or 'line search' in line.lower()]
    record = dict(role_space=name, protocol=protocol, layer=layer, scaling=scaling, C=C,
                  dtype=dtype, actual_classifier_dtype=str(clf.dtype), phase=phase,
                  iterations=cp.asnumpy(cp.asarray(clf.n_iter_)).tolist(), seconds=time.time()-started,
                  train=train_metrics, evaluation=eval_metrics, optimizer_warnings=warnings)
    if phase != 'search':
        # Scaler transformed this run's arrays in place; consumers get a copy.
        if scaler is not None:
            scaler.copy = True
        pipeline = Pipeline(([('scaler', scaler)] if scaler is not None else []) + [('clf', clf)])
        result = test.assign(pred=[roles[int(i)] for i in predictions])
        result['is_acc'] = result.role == result.pred
        split = json.loads((OUT / 'splits.json').read_text())[name][protocol]
        artifact = dict(probe=pipeline, acc=eval_metrics['accuracy'], nll=eval_metrics['nll'],
                        layer_ix=layer, role_space=roles, roles_map={r:i for i,r in enumerate(roles)},
                        n_inputs=len(train)+len(test), n_train_tokens=len(train), n_test_tokens=len(test),
                        train_prompt_ids=split['refit'], test_prompt_ids=split['test'],
                        acc_by_role=result.groupby(['role','pred'], as_index=False).agg(count=('sample_ix','count')),
                        acc_by_pos=result.groupby('token_in_seg_ix', as_index=False).agg(count=('sample_ix','count'), acc=('is_acc','mean')),
                        tuning=record)
        with (OUT / 'models' / f'{tag}.pkl').open('wb') as f:
            pickle.dump(artifact, f)
    del clf, y, yt
    return record


def data_for_fit(name, protocol, phase, layer, scaling, dtype):
    from cuml.preprocessing import StandardScaler
    train, test = datasets(name, protocol, phase)
    x, xt = load_features(layer, train, dtype), load_features(layer, test, dtype)
    scaler = StandardScaler(copy=False) if scaling else None
    if scaler is not None:
        x = scaler.fit_transform(x)
        xt = scaler.transform(xt)
    assert str(x.dtype) == str(xt.dtype) == dtype
    return train, test, x, xt, scaler


def search_worker(job):
    name, layer = [(n,l) for n in ROLE_SPACES for l in LAYERS][job]
    for protocol in PROTOCOLS:
        for scaling in [False, True]:
            pending = [C for C in CS if not (OUT / 'candidates' / f'{protocol}-{name}-{layer}-{int(scaling)}-{C:g}.json').exists()]
            if not pending:
                continue
            train, test, x, xt, scaler = data_for_fit(name, protocol, 'search', layer, scaling, 'float32')
            for C in pending:
                tag = f'{protocol}-{name}-{layer}-{int(scaling)}-{C:g}'
                result = train_one(name, protocol, layer, scaling, C, 'float32', 'search',
                                   train, test, x, xt, scaler, tag)
                atomic_json(OUT / 'candidates' / f'{tag}.json', result)
                print(json.dumps(dict(job=job, candidate=tag, val_accuracy=result['evaluation']['accuracy'],
                                      train_accuracy=result['train']['accuracy'], seconds=result['seconds'])), flush=True)
                del result
                cleanup()
            del x, xt, scaler
            cleanup()


def select():
    records = [json.loads(p.read_text()) for p in (OUT / 'candidates').glob('*.json')]
    assert len(records) == 80
    selected = {}
    for protocol in PROTOCOLS:
        selected[protocol] = {}
        for name in ROLE_SPACES:
            rows = [r for r in records if r['protocol'] == protocol and r['role_space'] == name]
            assert len(rows) == 20
            best = min(rows, key=lambda r:(-r['evaluation']['accuracy'], r['evaluation']['nll'], r['C'], r['layer']))
            selected[protocol][name] = {k:best[k] for k in ['layer','scaling','C']}
            selected[protocol][name]['validation'] = best['evaluation']
    atomic_json(OUT / 'selection.json', selected)
    progress('selected', selected=selected)


def final_worker(job, precision=False):
    import pandas as pd
    selections = json.loads((OUT / 'selection.json').read_text())
    if precision:
        protocol, name = 'document', list(ROLE_SPACES)[job]
    else:
        protocol, name = [(p,n) for p in PROTOCOLS for n in ROLE_SPACES][job]
    best = selections[protocol][name]
    # Baseline at the same selected layer isolates hyperparameter effects.
    configs = [('selected', best['scaling'], best['C'], 'float64' if precision else 'float32')]
    if not precision:
        configs.append(('baseline', True, .005, 'float32'))
    for label, scaling, C, dtype in configs:
        tag = f'{protocol}-{name}-{label}-{dtype}'
        if (OUT / 'final' / f'{tag}.json').exists():
            continue
        train, test, x, xt, scaler = data_for_fit(name, protocol, 'final', best['layer'], scaling, dtype)
        result = train_one(name, protocol, best['layer'], scaling, C, dtype, 'precision' if precision else label,
                           train, test, x, xt, scaler, tag)
        atomic_json(OUT / 'final' / f'{tag}.json', result)
        print(json.dumps(dict(result=tag, accuracy=result['evaluation']['accuracy'], seconds=result['seconds'])), flush=True)
        del x, xt, scaler, result
        cleanup()


def run_parallel(stage, count):
    workers = []
    for job in range(count):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job), HF_HOME='/root/.cache/huggingface-role-probe')
        log = (OUT / 'logs' / f'{stage}-worker-{job}.log').open('a')
        proc = subprocess.Popen([sys.executable, '-u', __file__, '--stage', stage, '--job', str(job)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        workers.append((proc, log))
    try:
        while any(proc.poll() is None for proc, _ in workers):
            failed = [proc.returncode for proc, _ in workers if proc.returncode not in (None, 0)]
            if failed:
                raise RuntimeError(f'{stage} worker failed: {failed}; see tuning/logs')
            time.sleep(3)
        assert all(proc.returncode == 0 for proc, _ in workers)
    finally:
        for proc, log in workers:
            if proc.poll() is None:
                proc.terminate()
            log.close()


def report():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(p.read_text()) for p in (OUT / 'candidates').glob('*.json')]
    table = pd.DataFrame([dict(protocol=r['protocol'], role_space=r['role_space'], layer=r['layer'],
                             scaling=r['scaling'], C=r['C'], train_accuracy=r['train']['accuracy'],
                             validation_accuracy=r['evaluation']['accuracy'], validation_nll=r['evaluation']['nll'],
                             validation_macro_f1=r['evaluation']['macro_f1'], seconds=r['seconds'],
                             optimizer_warnings=len(r['optimizer_warnings'])) for r in records])
    table.sort_values(['protocol','role_space','layer','scaling','C']).to_csv(OUT / 'search_metrics.csv', index=False)
    results = []
    for p in sorted((OUT / 'final').glob('*.json')):
        r = json.loads(p.read_text())
        results.append(dict(artifact=p.stem, protocol=r['protocol'], role_space=r['role_space'],
                            phase=r['phase'], layer=r['layer'], scaling=r['scaling'], C=r['C'], dtype=r['dtype'],
                            train_accuracy=r['train']['accuracy'], test_accuracy=r['evaluation']['accuracy'],
                            test_nll=r['evaluation']['nll'], test_macro_f1=r['evaluation']['macro_f1'],
                            **{f'recall_{k}':v for k,v in r['evaluation']['recall'].items()},
                            seconds=r['seconds'], optimizer_warnings=len(r['optimizer_warnings'])))
    final = pd.DataFrame(results)
    assert len(final) == 10
    final.to_csv(OUT / 'final_metrics.csv', index=False)
    fig, axes = plt.subplots(2,2,figsize=(11,7), sharex=True)
    for ax, (protocol, name) in zip(axes.flat, [(p,n) for p in PROTOCOLS for n in ROLE_SPACES]):
        group = table[(table.protocol==protocol)&(table.role_space==name)]
        for (layer, scaling), g in group.groupby(['layer','scaling']):
            g=g.sort_values('C')
            ax.plot(g.C, g.validation_accuracy, marker='o', label=f'layer {layer}, scaling {scaling}')
        ax.set(xscale='log', title=f'{protocol}: {name}', xlabel='C', ylabel='Validation accuracy')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'validation-search.png', dpi=180)
    fig.savefig(OUT / 'validation-search.pdf')
    plt.close(fig)
    models=[]
    for p in sorted((OUT / 'models').glob('*-selected-float32.pkl')):
        with p.open('rb') as f:
            models.append(pickle.load(f))
    with (OUT / 'selected-probes.pkl').open('wb') as f:
        pickle.dump(models, f)
    assert digest(HERE / 'outputs/probes/gptoss-120b.pkl') == json.loads((OUT / 'config.json').read_text())['source_probes_sha256']
    atomic_json(OUT / 'complete.json', dict(candidates=80, final_fits=10, selected_fp32_probes=4,
                verified_fp64_diagnostics=2, baseline_unchanged=True,
                selection_used_test=False, source_sha256=digest(Path(__file__))))
    progress('complete', candidates=80, final_fits=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','prepare','search-worker','select','final-worker','precision-worker','report'], default='all')
    parser.add_argument('--job', type=int, default=0)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for folder in ['candidates','final','models','logs']:
        (OUT / folder).mkdir(exist_ok=True)
    if args.stage == 'all':
        if not (OUT / 'config.json').exists():
            subprocess.run([sys.executable, '-u', __file__, '--stage', 'prepare'],
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES='3'), check=True)
        progress('searching', candidates=80, workers=4)
        run_parallel('search-worker',4)
        select()
        progress('refitting_selected_and_baseline', workers=4)
        run_parallel('final-worker',4)
        progress('precision_diagnostics', workers=2)
        run_parallel('precision-worker',2)
        report()
    elif args.stage == 'prepare':
        prepare()
    elif args.stage == 'search-worker':
        search_worker(args.job)
    elif args.stage == 'select':
        select()
    elif args.stage == 'final-worker':
        final_worker(args.job)
    elif args.stage == 'precision-worker':
        final_worker(args.job,precision=True)
    else:
        report()


if __name__ == '__main__':
    main()
