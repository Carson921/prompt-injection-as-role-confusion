"""Compare readout losses on cached, frozen MXFP4 residual features.

Training groups are profile x role x quartile of sampled message positions.
Those labels change training weights only; inference remains pointwise and takes
only the same 8640 hidden-state features as the published residual probe.
Previously reported tests are not read. A promoted method needs fresh testing.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_probes as probes

OUT = data.runtime.OUT / 'systemness-objective-pilot'
JOBS = ['erm', 'balanced', 'focal', 'group-dro']


def training_groups(rows):
    """Use sampled-position quartiles, not true roles/positions at inference."""
    assert set(rows.split) == {'fit'} and set(rows.wrapper) == {'native'}
    ordered = rows.sort_values(['prompt_ix', 'message_ix', 'token_ix'])
    grouped = ordered.groupby(['prompt_ix', 'message_ix'], sort=False)
    quartile = (4 * grouped.cumcount() // grouped.token_ix.transform('size')).clip(upper=3)
    keys = (ordered.profile + ':' + ordered.role + ':' + quartile.astype(str)).reindex(rows.index)
    names = sorted(keys.unique())
    ids = keys.map({name: i for i, name in enumerate(names)}).to_numpy(dtype=np.int64)
    assert (ids >= 0).all() and len(names) == 60
    return ids, names


def score(result):
    return min(result['profiles'][k]['system_f1'] for k in ['single', 'long-single'])


def loss_for(kind, ce, groups, counts, log_q):
    import torch
    group_loss = torch.zeros_like(counts).scatter_add_(0, groups, ce) / counts.clamp_min(1)
    present = counts > 0
    if kind == 'erm':
        return ce.mean(), log_q
    if kind == 'focal':
        focal = (1 - torch.exp(-ce)).square() * ce
        group_loss = torch.zeros_like(counts).scatter_add_(0, groups, focal) / counts.clamp_min(1)
    if kind != 'group-dro':
        return group_loss[present].mean(), log_q
    log_q = log_q + .01 * group_loss.detach()
    log_q = log_q - torch.logsumexp(log_q, dim=0)
    weights = .5 * torch.softmax(log_q, dim=0) + .5 / len(counts)
    weights = weights * present
    return (weights * group_loss).sum() / weights.sum(), log_q


def implementation_check():
    import torch
    # Equally sized groups make balanced CE equal ERM; focal gamma=2 must
    # downweight easy correct samples, and each objective must retain gradients.
    y = torch.tensor([0, 1, 2, 3])
    group = torch.tensor([0, 0, 1, 1])
    counts = torch.tensor([2., 2.])
    logits = torch.tensor([[4., 0., 0., 0., 0.], [0., 4., 0., 0., 0.],
                           [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.]], requires_grad=True)
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    values = {}
    for kind in JOBS:
        loss, q = loss_for(kind, ce, group, counts, torch.zeros(2))
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        values[kind] = float(loss.detach())
    assert abs(values['balanced'] - values['erm']) < 1e-6
    assert values['focal'] < values['balanced']
    return dict(losses=values,finite_nonzero_gradients=True,
                balanced_equals_erm_for_equal_group_sizes=True)


def worker(kind):
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.device_count() == 1
    plan = json.loads((OUT / 'plan.json').read_text())
    assert plan['source_sha256'] == data.digest(__file__)
    source = Path(plan['standardization_model'])
    assert data.digest(source) == plan['standardization_model_sha256']
    rows, bank = probes.development()
    fit = rows[rows.split == 'fit'].copy()
    valid = rows[rows.split == 'validation'].copy()
    groups, names = training_groups(fit)
    assert fit.document_id.nunique() == 1200
    assert not set(fit.group_id) & set(valid.group_id)
    with np.load(source, allow_pickle=False) as archive:
        mean = archive['mean'].copy(); scale = archive['scale'].copy()
    mean_t = torch.from_numpy(mean).cuda(); scale_t = torch.from_numpy(scale).cuda()

    def load(indices):
        x = torch.empty((len(indices), 8640), device='cuda', dtype=torch.float32)
        for start in range(0, len(indices), 4096):
            block = np.asarray(bank[indices[start:start + 4096]], dtype=np.float32)
            assert np.isfinite(block).all()
            x[start:start + len(block)].copy_(torch.from_numpy(block))
        x.sub_(mean_t).div_(scale_t)
        # torch.isfinite on the entire matrix creates matrix-sized temporaries.
        for start in range(0, len(x), 4096):
            assert torch.isfinite(x[start:start + 4096]).all()
        return x

    started = time.time()
    x = load(fit.cache_ix.to_numpy()); xv = load(valid.cache_ix.to_numpy())
    y = torch.from_numpy(probes.labels(fit)).cuda()
    gt = torch.from_numpy(groups).cuda()
    torch.manual_seed(data.SEED)
    net = probes.make_net('mlp', 64).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    generator = torch.Generator(device='cuda:0').manual_seed(data.SEED)
    log_q = torch.zeros(len(names), device='cuda')
    best = -1.; best_epoch = -1; history = []
    model_path = OUT / 'models' / f'{kind}.npz'
    probability_path = OUT / 'cache' / f'{kind}-validation.npz'
    details = dict(kind='mlp', hidden=64, objective=kind, roles=data.ROLES, feature_spec=data.SPECS,
                   fit_documents=1200, train_tokens=len(fit), seed=data.SEED,
                   source_sha256=data.digest(__file__), plan_sha256=data.digest(OUT / 'plan.json'),
                   inference='Pointwise 8640 residual features only',
                   standardization_model_sha256=plan['standardization_model_sha256'])
    for epoch in range(1, 46):
        net.train()
        order = torch.randperm(len(x), device='cuda', generator=generator)
        loss_sum = 0.
        for start in range(0, len(order), 4096):
            ix = order[start:start + 4096]
            ce = torch.nn.functional.cross_entropy(net.raw(x[ix]), y[ix], reduction='none')
            batch_groups = gt[ix]
            counts = torch.bincount(batch_groups, minlength=len(names)).float()
            loss, log_q = loss_for(kind, ce, batch_groups, counts, log_q)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ix)
        p = probes.predict_standardized(net, xv)
        result = probes.validation_metrics(valid, p)
        current = score(result)
        history.append(dict(epoch=epoch,loss=loss_sum / len(x),
                            minimum_single_long_system_f1=current,
                            mean_system_f1=result['mean_system_f1'],mean_accuracy=result['mean_accuracy']))
        if current > best + 1e-5:
            best = current; best_epoch = epoch
            probes.save_model(model_path, net, mean, scale, details | dict(best_epoch=epoch))
            np.savez_compressed(probability_path, p=p, sample_ix=valid.sample_ix.to_numpy())
        print(json.dumps(dict(stage='epoch', objective=kind, **history[-1], best_epoch=best_epoch)), flush=True)
        if epoch >= 10 and epoch - best_epoch >= 8:
            break
    with np.load(probability_path, allow_pickle=False) as archive:
        p = archive['p'].copy()
        assert np.array_equal(archive['sample_ix'], valid.sample_ix.to_numpy())
    # Validate the exported consumer on actual hidden states, including CPU load.
    portable, mean_check, scale_check, _ = probes.load_model(model_path, device='cpu')
    indices = np.linspace(0, len(valid) - 1, 256, dtype=int)
    raw = torch.from_numpy(np.asarray(bank[valid.iloc[indices].cache_ix.to_numpy()], dtype=np.float32))
    with torch.inference_mode():
        actual = portable((raw - mean_check) / scale_check).exp().numpy()
    difference = float(np.abs(actual - p[indices]).max())
    assert difference < 5e-4, difference
    assert data.digest(__file__) == plan['source_sha256']
    data.atomic_json(OUT / 'candidates' / f'{kind}.json',dict(
        **details, validation=probes.validation_metrics(valid, p),
        best_epoch=best_epoch,epochs=len(history),history=history,seconds=time.time()-started,
        groups={name:int((groups == i).sum()) for i,name in enumerate(names)},
        final_dro_weights=dict(zip(names,log_q.softmax(0).cpu().tolist())) if kind=='group-dro' else None,
        portable_probability_max_difference=difference,model_sha256=data.digest(model_path),
        tests_read=False,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))


def summarize():
    records = {kind:json.loads((OUT / 'candidates' / f'{kind}.json').read_text()) for kind in JOBS}
    reference = json.loads((data.OUT / 'selection.json').read_text())['validation']
    control = records['erm']['validation']
    comparisons = []
    table = []
    for kind, record in records.items():
        result = record['validation']
        weak_gain = min(result['profiles'][p]['system_f1'] - control['profiles'][p]['system_f1']
                        for p in ['single','long-single'])
        mean_gain = result['mean_system_f1'] - reference['mean_system_f1']
        other_loss = max(reference['profiles'][p]['system_f1'] - result['profiles'][p]['system_f1']
                         for p in ['multi','natural'])
        eligible = kind != 'erm' and weak_gain >= .03 and mean_gain >= .01 and other_loss <= .02 \
                   and result['mean_accuracy'] >= reference['mean_accuracy'] - .01
        comparisons.append(dict(objective=kind,promoted=eligible,
            minimum_single_long_f1_gain_vs_matched_control=weak_gain,
            mean_f1_gain_vs_published_selection=mean_gain,
            maximum_multi_natural_f1_loss_vs_published_selection=other_loss,
            mean_accuracy=result['mean_accuracy']))
        for profile, metrics in result['profiles'].items():
            table.append(dict(objective=kind,profile=profile,
                              **{k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(table).to_csv(OUT / 'validation-metrics.csv', index=False)
    eligible = [r for r in comparisons if r['promoted']]
    selected = max(eligible,key=lambda r:score(records[r['objective']]['validation']))['objective'] if eligible else None
    complete = dict(candidates=len(records),comparisons=comparisons,promoted=selected,
                    source_sha256=data.digest(__file__),plan_sha256=data.digest(OUT/'plan.json'),
                    tests_used=False,fresh_confirmation_required_if_promoted=True,
                    models={kind:record['model_sha256'] for kind,record in records.items()})
    data.atomic_json(OUT / 'complete.json',complete)
    data.atomic_json(OUT / 'status.json',dict(stage='complete',promoted=selected))
    print(json.dumps(complete),flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    for folder in ['models','candidates','cache','logs']:(OUT/folder).mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('/cache/\n/logs/\n')
    assert not (OUT/'complete.json').exists(),'This objective pilot is complete.'
    standardization = data.OUT / 'models/selected.npz'
    plan = dict(objectives=JOBS,fit_documents=1200,tests_read=False,
        groups='Training profile x role x within-message quartile of sampled positions; 60 groups.',
        inference='Pointwise 8640 residual features only; training groups are never inputs.',
        network=dict(hidden=64,dropout=.1),optimizer=dict(name='AdamW',lr=.001,weight_decay=.001),
        training=dict(seed=data.SEED,batch_size=4096,max_epochs=45,min_epochs=10,patience=8,gradient_clip=5.),
        early_stopping='Maximum minimum of single and long-single validation system F1.',
        threshold='Unchanged: one global validation threshold maximizing mean profile F1, each FPR <=5%.',
        focal_gamma=2,group_dro=dict(log_weight_step=.01,uniform_weight_mixture=.5),
        promotion=dict(minimum_single_long_f1_gain_vs_matched_control=.03,
                       mean_f1_gain_vs_published_selection=.01,
                       maximum_multi_natural_f1_loss_vs_published_selection=.02,
                       maximum_mean_accuracy_loss_vs_published_selection=.01),
        previous_test_results_already_reported=True,fresh_confirmation_required_if_promoted=True,
        references=['https://arxiv.org/abs/1911.08731','https://arxiv.org/abs/1708.02002'],
        standardization_model=str(standardization),standardization_model_sha256=data.digest(standardization),
        sources={name:data.digest(data.HERE/name) for name in ['systemness_probes.py','systemness_data.py']},
        source_sha256=data.digest(__file__),implementation_checks=implementation_check(),created_unix_time=time.time())
    if (OUT/'plan.json').exists():
        previous=json.loads((OUT/'plan.json').read_text())
        if previous['source_sha256']!=plan['source_sha256']:
            assert not list((OUT/'candidates').glob('*.json'))
            history_path=OUT/'attempt-history.json'
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            history.append(dict(previous_plan=previous,
                reason='All workers failed before training: full-matrix finite check allocated oversized CUDA temporaries. Replace it with bounded chunk checks; data, losses and solver unchanged.',
                updated_unix_time=time.time()))
            data.atomic_json(history_path,history)
            data.atomic_json(OUT/'plan.json',plan)
    else:data.atomic_json(OUT/'plan.json',plan)
    workers=[]
    for gpu,kind in enumerate(JOBS):
        if (OUT/'candidates'/f'{kind}.json').exists():continue
        log=(OUT/'logs'/f'{kind}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',__file__,'--worker',kind],
                  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log))
    data.atomic_json(OUT/'status.json',dict(stage='training',workers=[p.pid for p,_ in workers]))
    try:
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in [None,0] for p,_ in workers):
                raise RuntimeError('Objective worker failed; inspect its log.')
            time.sleep(5)
    finally:
        for p,log in workers:
            if p.poll() is None:p.terminate()
            log.close()
    exit_codes=[p.wait() for p,_ in workers]
    assert all(code==0 for code in exit_codes),exit_codes
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=JOBS)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:main()
