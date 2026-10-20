"""Pointwise attention readouts with fit-only, matched-content ranking loss.

Default: print help. --preflight runs a CPU-only row/gradient/RNG audit and writes
only under /tmp. --train declares and runs the separate three-worker experiment
after long-full extraction and its original readout search have completed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_data as data
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-paired-pilot'
CACHE = data.CACHE / 'paired-pilot'
CONFIGS = {'lambda-0': 0., 'lambda-0.1': .1, 'lambda-0.5': .5}
BATCH_SIZE = 4096
FIT_ROWS = 578090
VALID_ROWS = 173032
STEPS = 142
PAIR_GROUPS = 33792
AUX_GROUPS = 128
MARGIN = .5
PAIR_SEED = data.SEED + 9001
DROPOUT_SEED = data.SEED + 41_000_000
DEPENDENCIES = ['systemness_paired_probes.py', 'systemness_long_probes.py',
    'systemness_long_data.py', 'systemness_long_extract.py', 'systemness_attention.py',
    'systemness_probes.py', 'systemness_data.py']
EARLY_STOPPING_LIMIT = ('All candidates match CE instances and optimizer updates per epoch and '
    'at the maximum budget. Independent checkpoint selection and early stopping may produce '
    'different best and actual total update counts; these are reported explicitly. '
    'The auxiliary objective uses long-full fit rows only. Any gain belongs to the complete '
    'paired auxiliary objective; this experiment does not separate matched-pair structure '
    'from ordinary long-text reweighting.')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_pairs(fit):
    """Return matched five-role cache and dense-relative indices, never features."""
    require(set(fit.split) == {'fit'} and set(fit.wrapper) == {'native'},
            'Pair source must contain only native fit rows; validation/test rows are forbidden.')
    require(fit.cache_ix.is_unique and fit.sample_ix.is_unique, 'Duplicate fit row identity.')
    full = fit[(fit.profile == 'long-full') & (fit.source_set == 'long-full')].copy()
    require(len(full) == 168960 and full.document_id.nunique() == 132, 'Long-full fit membership changed.')
    require(full.sample_ix.ge(long_data.SAMPLE_START).all() and set(full.role) == set(data.ROLES),
            'Unexpected long-full identities or role classes.')
    require((full.group_id == 'c4-' + full.document_id.astype(str)).all(), 'Document grouping changed.')
    full['body_offset'] = full.token_ix - full.groupby(
        ['source_set', 'prompt_ix', 'message_ix'], sort=False).token_ix.transform('min')
    require(full.groupby(['document_id', 'role']).size().eq(256).all(), 'Expected 256 positions per role.')
    expected = data.uniform_indices(1024, 256)
    for _, group in full.groupby(['document_id', 'role'], sort=False):
        require(np.array_equal(np.sort(group.body_offset), expected), 'Uniform body-position alignment changed.')
    grouped = full.groupby(['document_id', 'body_offset'], sort=True)
    require(grouped.size().eq(5).all() and grouped.role.nunique().eq(5).all()
            and grouped.token_id.nunique().eq(1).all(), 'A content position lacks five matching-token roles.')
    table = full.pivot(index=['document_id', 'body_offset', 'token_id'], columns='role', values='cache_ix')
    table = table.reindex(columns=data.ROLES)
    require(table.shape == (PAIR_GROUPS, 5) and not table.isna().any().any(), 'Incomplete five-role pairing.')
    dense = long_probes.fit_rows(fit, True)
    require(len(dense) == FIT_ROWS and np.all(np.diff(dense.cache_ix.to_numpy()) > 0),
            'Dense fit order/count changed.')
    removed = fit[(fit.profile == 'long-single') & fit.document_id.isin(full.document_id)]
    require(len(removed) == 4160 and removed.document_id.nunique() == 26, 'Sparse-row replacement changed.')
    cache_ix = table.to_numpy(dtype=np.int64)
    lookup = pd.Series(np.arange(len(dense), dtype=np.int64), index=dense.cache_ix)
    mapped = lookup.reindex(cache_ix.reshape(-1))
    require(not mapped.isna().any(), 'A pair points outside dense fit.')
    dense_ix = mapped.to_numpy(dtype=np.int64).reshape(cache_ix.shape)
    recovered = dense.iloc[dense_ix.reshape(-1)]
    keys = table.index.to_frame(index=False)
    require(np.array_equal(recovered.cache_ix.to_numpy(), cache_ix.reshape(-1)), 'Pair index round trip failed.')
    require(recovered.split.eq('fit').all() and recovered.profile.eq('long-full').all(), 'Pair index leaked outside fit.')
    require(np.array_equal(recovered.role.to_numpy().reshape(-1, 5), np.broadcast_to(data.ROLES, table.shape)),
            'Pair role axis changed.')
    require(np.array_equal(recovered.document_id.to_numpy().reshape(-1, 5),
                          np.repeat(keys.document_id.to_numpy()[:, None], 5, axis=1)), 'Pair document mismatch.')
    require(np.array_equal(recovered.token_id.to_numpy().reshape(-1, 5),
                          np.repeat(keys.token_id.to_numpy()[:, None], 5, axis=1)), 'Pair token mismatch.')
    arrays = dict(cache_ix=cache_ix, dense_fit_ix=dense_ix,
        document_id=keys.document_id.to_numpy(dtype=np.int64),
        body_offset=keys.body_offset.to_numpy(dtype=np.int64), token_id=keys.token_id.to_numpy(dtype=np.int64))
    metadata = dict(groups=PAIR_GROUPS, documents=132, role_order=data.ROLES, roles_per_group=5,
        negative_roles_per_system=4, matched_rows=168960, ranked_edges=PAIR_GROUPS * 4,
        pair_source='Native fit long-full rows only; match document_id/body_offset/token_id.',
        dense_fit_rows=len(dense), removed_sparse_rows=len(removed),
        arrays_sha256={key: array_digest(value) for key, value in arrays.items()})
    return dense, arrays, metadata


def ranking_loss(logits, margin=MARGIN):
    import torch
    require(logits.ndim == 3 and logits.shape[1:] == (5, 5), 'Expected groups x source-role x predicted-role logits.')
    score = logits[:, :, 0] - torch.logsumexp(logits[:, :, 1:], dim=-1)
    return torch.nn.functional.softplus(margin - (score[:, :1] - score[:, 1:])).mean()


def auxiliary_forward(net, inputs, step):
    """Auxiliary dropout varies by step without advancing CE's RNG stream."""
    import torch
    devices = [inputs.device.index] if inputs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        if inputs.is_cuda:
            torch.cuda.manual_seed(DROPOUT_SEED + step)
        else:
            torch.manual_seed(DROPOUT_SEED + step)
        return net.raw(inputs)


def implementation_checks():
    import copy
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    torch.manual_seed(data.SEED)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(7, 11), nn.GELU(), nn.Dropout(.1), nn.Linear(11, 5))

        def raw(self, value):
            return self.layers(value)

    net = Tiny().double().train()
    reference = copy.deepcopy(net)
    inputs = torch.randn(8, 5, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(5).repeat(8)
    initial_rng = torch.get_rng_state().clone()
    ce_only = F.cross_entropy(reference.raw(inputs.detach()).reshape(-1, 5), labels)
    ce_only.backward()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    ce = F.cross_entropy(net.raw(inputs.detach()).reshape(-1, 5), labels)
    before = torch.get_rng_state().clone()
    aux = auxiliary_forward(net, inputs, 1)
    require(torch.equal(before, torch.get_rng_state()), 'Auxiliary forward changed the CE CPU RNG.')
    require(torch.equal(aux, auxiliary_forward(net, inputs, 1)), 'Auxiliary seed is not reproducible.')
    require(not torch.equal(aux, auxiliary_forward(net, inputs, 2)), 'Auxiliary dropout did not vary by step.')
    (ce + 0. * ranking_loss(aux)).backward()
    require(torch.equal(expected_rng, torch.get_rng_state()), 'Lambda-zero auxiliary path changed CE RNG.')
    gradient_difference = max(float((actual.grad - expected.grad).abs().max())
                              for actual, expected in zip(net.parameters(), reference.parameters()))
    require(gradient_difference < 1e-12, 'Lambda-zero gradients differ from CE-only.')
    net.zero_grad(set_to_none=True)
    inputs.grad = None
    aux = auxiliary_forward(net, inputs, 3)
    rank = ranking_loss(aux)
    terms = []
    for group in range(len(aux)):
        positive = aux[group, 0, 0] - torch.logsumexp(aux[group, 0, 1:], dim=0)
        for role in range(1, 5):
            negative = aux[group, role, 0] - torch.logsumexp(aux[group, role, 1:], dim=0)
            terms.append(F.softplus(MARGIN - (positive - negative)))
    scalar_difference = float((rank - torch.stack(terms).mean()).detach().abs())
    require(scalar_difference < 1e-12, 'Vectorized ranking differs from scalar reference.')
    grad, = torch.autograd.grad(rank, aux, retain_graph=True)
    require((grad[:, 0, 0] < 0).all() and (grad[:, 1:, 0] > 0).all(), 'Ranking gradient direction reversed.')
    rank.backward()
    require(torch.isfinite(inputs.grad).all() and (inputs.grad.abs().sum(-1) > 0).all(),
            'Ranking did not reach all five source-role inputs.')
    require(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in net.parameters()), 'Ranking parameter gradients are missing or nonfinite.')
    generator_a = torch.Generator().manual_seed(data.SEED)
    generator_b = torch.Generator().manual_seed(data.SEED)
    pair_generator = torch.Generator().manual_seed(PAIR_SEED)
    first = torch.randperm(100, generator=generator_a)
    torch.randint(PAIR_GROUPS, (AUX_GROUPS,), generator=pair_generator)
    require(torch.equal(first, torch.randperm(100, generator=generator_b)), 'Pair RNG changed CE permutation.')
    return dict(cpu_only=True, lambda_zero_ce_gradient_max_difference=gradient_difference,
        ranking_scalar_reference_difference=scalar_difference, ranking_input_and_parameter_gradients=True,
        ranking_gradient_direction_correct=True, auxiliary_rng_restored=True,
        auxiliary_dropout_reproducible_and_varies_by_step=True, independent_pair_rng=True)


def file_stat(path):
    info = Path(path).stat()
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino)


def signature():
    extraction = read_json(long_data.OUT / 'extraction-complete.json')
    rows_path = long_probes.BANK / 'rows.parquet'
    feature_path = long_probes.BANK / 'features.f16'
    require(extraction['rows'] == 755282 and extraction['dimensions'] == 8640
            and extraction['features'] == attention.SPECS and extraction['all_finite']
            and extraction['all_rows_nonzero'] and extraction['tests_read'] is False,
            'Long-full extraction is incomplete or has a different feature protocol.')
    require(extraction['rows_sha256'] == data.digest(rows_path), 'Extracted row table changed.')
    require(extraction['signature']['rows_sha256'] == extraction['rows_sha256']
            and extraction['signature']['long_data_sha256'] == data.digest(long_data.OUT / 'data-complete.json')
            and extraction['signature']['long_plan_sha256'] == data.digest(long_data.OUT / 'plan.json'),
            'Long-full data/plan no longer match the frozen extraction signature.')
    require(extraction['source_sha256'] == data.digest(data.HERE / 'systemness_long_extract.py'),
            'Long-full extraction source changed after feature capture.')
    require(feature_path.stat().st_size == 755282 * 8640 * 2, 'Feature bank size changed.')
    require((long_data.OUT / 'selection.json').is_file(), 'Complete the original dense/base validation search first.')
    return dict(rows_sha256=data.digest(rows_path),
        extraction_sha256=data.digest(long_data.OUT / 'extraction-complete.json'),
        data_complete_sha256=data.digest(long_data.OUT / 'data-complete.json'),
        dense_selection_sha256=data.digest(long_data.OUT / 'selection.json'),
        standardization_sha256=data.digest(attention.OUT / 'standardization.npz'),
        previous_selection_sha256=data.digest(attention.OUT / 'selection.json'),
        feature_sha256=extraction['sha256'], feature_stat=file_stat(feature_path), feature_spec=attention.SPECS,
        verified_weights_sha256=extraction['signature']['verified_weights_sha256'],
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        source_sha256=data.digest(__file__))


def verify_plan(plan):
    current = signature()
    require(all(plan.get(key) == value for key, value in current.items()), 'Paired source/data/model signature changed.')
    require(plan['configurations'] == CONFIGS and plan['pair_groups_per_step'] == AUX_GROUPS
            and plan['margin'] == MARGIN and plan['ce_rows_per_epoch'] == FIT_ROWS
            and plan['steps_per_epoch'] == STEPS, 'Paired training protocol changed.')


def save_pair_cache(arrays):
    destination = CACHE / 'fit-pairs.npz'
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            require(set(archive.files) == set(arrays) and all(np.array_equal(archive[name], array)
                    for name, array in arrays.items()), 'Existing pair table changed; refusing overwrite.')
    else:
        temporary = CACHE / 'fit-pairs.tmp.npz'
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
    return data.digest(destination)


def candidate_record(key, plan):
    verify_plan(plan)
    record = read_json(OUT / 'candidates' / f'{key}.json')
    require(record['candidate'] == key and record['ranking_lambda'] == CONFIGS[key], 'Candidate configuration changed.')
    require(record['source_sha256'] == plan['source_sha256']
            and record['search_plan_sha256'] == data.digest(OUT / 'search-plan.json')
            and record['signature'] == {key: plan[key] for key in signature()}, 'Candidate provenance changed.')
    require(record['tests_read'] is False and record['portable_probability_max_difference'] < 5e-4,
            'Candidate lacks its portable validation.')
    require(data.digest(OUT / 'models' / f'{key}.npz') == record['model_sha256'], 'Candidate model hash changed.')
    probability_path = CACHE / f'{key}-validation.npz'
    require(record['validation_probabilities_path'] == str(probability_path.resolve())
            and data.digest(probability_path) == record['validation_probabilities_sha256'], 'Candidate probability cache changed.')
    require(record['pair_cache_sha256'] == plan['pair_cache_sha256'] == data.digest(CACHE / 'fit-pairs.npz'),
            'Candidate pair table changed.')
    return record


def worker(key):
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in {'0', '1', '2'}, 'Paired workers are limited to physical GPU 0, 1 or 2.')
    require(torch.cuda.device_count() == 1 and 'A40' in torch.cuda.get_device_name(0), 'Worker requires one isolated A40.')
    plan = read_json(OUT / 'search-plan.json')
    verify_plan(plan)
    rows, bank = long_probes.dataset()
    fit, arrays, pairing = build_pairs(rows[rows.split == 'fit'])
    require(pairing == plan['pairing'] and save_pair_cache(arrays) == plan['pair_cache_sha256'], 'Worker pairing changed.')
    valid = rows[rows.split == 'validation'].copy().reset_index(drop=True)
    require(len(valid) == VALID_ROWS and not set(fit.group_id) & set(valid.group_id), 'Fit/validation membership changed.')
    with np.load(attention.OUT / 'standardization.npz', allow_pickle=False) as archive:
        mean = archive['mean'].copy(); scale = archive['scale'].copy()
    require(mean.shape == scale.shape == (8640,) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid train-only scaler.')
    free, _ = torch.cuda.mem_get_info(0)
    require(free > (FIT_ROWS + VALID_ROWS) * 8640 * 4 + 2 * 2**30, 'Insufficient memory for full rows plus bounded workspace.')
    def load(indices):
        value = torch.empty((len(indices), 8640), device='cuda', dtype=torch.float32, requires_grad=False)
        for start in range(0, len(indices), BATCH_SIZE):
            block = np.asarray(bank[indices[start:start + BATCH_SIZE]], dtype=np.float32)
            standardized = (block - mean) / scale
            require(np.isfinite(standardized).all(), 'Nonfinite feature block.')
            value[start:start + len(block)].copy_(torch.from_numpy(standardized))
        require(not value.requires_grad, 'Frozen feature bank must not allocate input gradients.')
        return value
    started = time.time()
    x = load(fit.cache_ix.to_numpy()); xv = load(valid.cache_ix.to_numpy())
    y = torch.from_numpy(probes.labels(fit)).cuda()
    pair_ix = torch.from_numpy(arrays['dense_fit_ix']).cuda()
    torch.manual_seed(data.SEED)
    net = probes.make_net('mlp', 64).cuda()
    initial_sha = hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes()
                                          for value in net.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    ce_rng = torch.Generator(device='cuda:0').manual_seed(data.SEED)
    pair_rng = torch.Generator(device='cuda:0').manual_seed(PAIR_SEED)
    path = OUT / 'models' / f'{key}.npz'; probability_path = CACHE / f'{key}-validation.npz'
    common = dict(kind='mlp', hidden=64, ranking_lambda=CONFIGS[key], margin=MARGIN,
        roles=data.ROLES, feature_spec=attention.SPECS, fit_documents=1200, train_tokens=FIT_ROWS,
        paired_documents=132, seed=data.SEED, initial_parameters_sha256=initial_sha,
        instances_per_epoch=FIT_ROWS, steps_per_epoch=STEPS, last_batch_instances=554,
        pair_cache_sha256=plan['pair_cache_sha256'], source_sha256=plan['source_sha256'],
        search_plan_sha256=data.digest(OUT / 'search-plan.json'),
        signature={key: plan[key] for key in signature()}, inference='Pointwise attention features only; no pair identities or labels supplied.',
        control_limitation=EARLY_STOPPING_LIMIT)
    history = []; best = -1.; best_epoch = -1; total_steps = 0
    first_epoch_order_sha = None; rng_first_step_checked = False
    for epoch in range(1, 46):
        net.train()
        order = long_probes.epoch_order(len(x), ce_rng, 'cuda')
        if first_epoch_order_sha is None:
            first_epoch_order_sha = array_digest(order.cpu().numpy())
        ce_sum = 0.; rank_sum = 0.; steps = 0
        for start in range(0, len(order), BATCH_SIZE):
            ix = order[start:start + BATCH_SIZE]
            ce = torch.nn.functional.cross_entropy(net.raw(x[ix].contiguous()), y[ix])
            chosen = torch.randint(PAIR_GROUPS, (AUX_GROUPS,), device='cuda', generator=pair_rng)
            auxiliary_inputs = x[pair_ix[chosen].reshape(-1)].contiguous()
            before = torch.cuda.get_rng_state(0) if not rng_first_step_checked else None
            auxiliary = auxiliary_forward(net, auxiliary_inputs, total_steps).reshape(AUX_GROUPS, 5, 5)
            if before is not None:
                require(torch.equal(before, torch.cuda.get_rng_state(0)), 'Auxiliary forward advanced CE CUDA RNG.')
                rng_first_step_checked = True
            rank = ranking_loss(auxiliary)
            loss = ce + CONFIGS[key] * rank
            require(bool(torch.isfinite(loss)), 'Nonfinite paired loss.')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            require(x.grad is None and not x.requires_grad, 'A full-bank gradient was allocated.')
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step(); total_steps += 1; steps += 1
            ce_sum += float(ce.detach()) * len(ix); rank_sum += float(rank.detach())
        require(len(order) == FIT_ROWS and steps == STEPS and len(ix) == 554, 'CE/update budget changed.')
        p = probes.predict_standardized(net, xv)
        result = long_probes.validation_metrics(valid, p); score = result['mean_system_f1']
        history.append(dict(epoch=epoch, ce_loss=ce_sum / FIT_ROWS, ranking_loss=rank_sum / steps,
            instances=FIT_ROWS, steps=steps, cumulative_steps=total_steps, mean_system_f1=score,
            long_full_f1=result['profiles']['long-full']['system_f1'], tail_f1=result['long_full_tail']['system_f1']))
        if score > best + 1e-5:
            best = score; best_epoch = epoch
            probes.save_model(path, net, mean, scale, common | dict(best_epoch=epoch, best_steps=total_steps))
            np.savez_compressed(probability_path, p=p, sample_ix=valid.sample_ix.to_numpy())
        print(json.dumps(dict(candidate=key, **history[-1], best_epoch=best_epoch)), flush=True)
        if epoch >= 10 and epoch - best_epoch >= 8:
            break
    with np.load(probability_path, allow_pickle=False) as archive:
        p = archive['p'].copy()
        require(np.array_equal(archive['sample_ix'], valid.sample_ix), 'Validation probabilities misaligned.')
    portable, m, s, _ = probes.load_model(path, device='cpu')
    indices = np.linspace(0, len(valid) - 1, 256, dtype=int)
    raw = torch.from_numpy(np.asarray(bank[valid.iloc[indices].cache_ix.to_numpy()], dtype=np.float32))
    with torch.inference_mode():
        actual = portable((raw - m) / s).exp().numpy()
    difference = float(np.abs(actual - p[indices]).max())
    require(difference < 5e-4, 'CPU portable probabilities differ from the frozen best checkpoint.')
    verify_plan(plan)
    record = dict(candidate=key, **common, validation=long_probes.validation_metrics(valid, p),
        best_epoch=best_epoch, best_steps=best_epoch * STEPS, actual_steps=total_steps,
        epochs=len(history), actual_ce_instances=len(history) * FIT_ROWS, history=history,
        first_epoch_ce_order_sha256=first_epoch_order_sha, auxiliary_rng_first_step_verified=rng_first_step_checked,
        seconds=time.time() - started, model_sha256=data.digest(path), tests_read=False,
        validation_probabilities_path=str(probability_path.resolve()),
        validation_probabilities_sha256=data.digest(probability_path), portable_probability_max_difference=difference,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    data.atomic_json(OUT / 'candidates' / f'{key}.json', record)


def select(plan):
    records = [candidate_record(key, plan) for key in CONFIGS]
    require(len({record['initial_parameters_sha256'] for record in records}) == 1
            and len({record['first_epoch_ce_order_sha256'] for record in records}) == 1,
            'Candidate initialization or CE order differed.')
    control = next(record for record in records if record['candidate'] == 'lambda-0')
    comparisons = []
    for record in records:
        current = record['validation']; baseline = control['validation']
        mean_gain = current['mean_system_f1'] - baseline['mean_system_f1']
        full_gain = current['profiles']['long-full']['system_f1'] - baseline['profiles']['long-full']['system_f1']
        tail_gain = current['long_full_tail']['system_f1'] - baseline['long_full_tail']['system_f1']
        other_loss = max(baseline['profiles'][key]['system_f1'] - current['profiles'][key]['system_f1']
                         for key in ['single', 'multi', 'natural'])
        promoted = record['ranking_lambda'] > 0 and mean_gain >= .01 and full_gain >= .03 and tail_gain >= .03 \
            and other_loss <= .02 and current['original_mean_accuracy'] >= plan['accuracy_floor']
        comparisons.append(dict(candidate=record['candidate'], promoted=promoted, mean_f1_gain=mean_gain,
            long_full_f1_gain=full_gain, tail_f1_gain=tail_gain, maximum_other_profile_f1_loss=other_loss))
    eligible = {item['candidate'] for item in comparisons if item['promoted']}
    chosen = max((record for record in records if record['candidate'] in eligible),
                 key=lambda record: record['validation']['mean_system_f1']) if eligible else control
    comparison = next(item for item in comparisons if item['candidate'] == chosen['candidate'])
    frozen = dict(candidate=chosen['candidate'], model_path=str(OUT / 'models' / f'{chosen["candidate"]}.npz'),
        model_sha256=chosen['model_sha256'], system_threshold=chosen['validation']['system_threshold'],
        validation=chosen['validation'], promoted=comparison['promoted'], comparisons=comparisons,
        matched_control_candidate='lambda-0', matched_control=control['validation'],
        control_limitation=EARLY_STOPPING_LIMIT, test_used_for_selection=False, confirmation_required=True,
        source_sha256=plan['source_sha256'], search_plan_sha256=data.digest(OUT / 'search-plan.json'),
        candidates={record['candidate']: data.digest(OUT / 'candidates' / f'{record["candidate"]}.json') for record in records},
        selected_unix_time=time.time(), final_gate='An independent final selector must also beat the dense selection and enforce per-profile nonregression before confirmation.')
    flat = []
    for record in records:
        for profile, metrics in dict(record['validation']['profiles'],
                **{'long-full-tail': record['validation']['long_full_tail']}).items():
            flat.append(dict(candidate=record['candidate'], profile=profile,
                **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}))
    pd.DataFrame(flat).to_csv(OUT / 'validation-metrics.csv', index=False)
    data.atomic_json(OUT / 'selection.json', frozen)
    data.atomic_json(OUT / 'status.json', dict(stage='validation_complete', candidate=chosen['candidate'], promoted=frozen['promoted']))
    return frozen


def no_competing_jobs():
    names = ['systemness_long_extract.py', 'systemness_long_probes.py', 'systemness_attention_confirm.py',
             'systemness_long_confirm.py', 'systemness_recover_local_bank.py']
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for name in names:
            if any(Path(arg.decode(errors='replace')).name == name for arg in args if arg):
                active.append(dict(pid=int(entry.name), script=name))
    require(not active, f'Other extraction/training/confirmation jobs are active; wait for completion: {active}')


@contextmanager
def training_lock():
    import fcntl
    with (OUT / '.training.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another paired training supervisor holds this output directory.') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_workers(pending, plan):
    workers = []
    try:
        for gpu, key in enumerate(pending):
            require(gpu < 3, 'Paired experiment permits at most GPU workers 0, 1 and 2.')
            log = (OUT / 'logs' / f'{key}.log').open('a')
            try:
                process = subprocess.Popen([sys.executable, '-u', __file__, '--train', '--worker', key],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append((key, process, log))
        data.atomic_json(OUT / 'status.json', dict(stage='training',
            workers=[dict(candidate=key, pid=process.pid, gpu=gpu) for gpu, (key, process, _) in enumerate(workers)]))
        while True:
            codes = [(key, process.poll()) for key, process, _ in workers]
            failed = [(key, code) for key, code in codes if code not in [None, 0]]
            if failed:
                raise RuntimeError(f'Paired worker failed: {failed}; inspect logs before retrying.')
            if all(code == 0 for _, code in codes):
                break
            time.sleep(5)
        for key, process, _ in workers:
            require(process.wait() == 0, f'Worker exited unsuccessfully: {key}.')
            candidate_record(key, plan)
    finally:
        errors = []
        for key, process, _ in workers:
            try:
                if process.poll() is None:
                    process.terminate()
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append((key, repr(error)))
        for key, process, log in workers:
            try:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            except BaseException as error:
                errors.append((key, repr(error)))
            finally:
                log.close()
        if errors:
            raise RuntimeError(f'Could not reap all paired workers: {errors}')


def train():
    no_competing_jobs()
    current = signature()
    fit = pd.read_parquet(long_probes.BANK / 'rows.parquet', filters=[('split', '=', 'fit')])
    _, arrays, pairing = build_pairs(fit)
    checks = implementation_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['models', 'candidates', 'logs']:
        (OUT / name).mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    (OUT / '.gitignore').write_text('/logs/\n/.training.lock\n')
    with training_lock():
        pair_sha = save_pair_cache(arrays)
        previous = read_json(attention.OUT / 'selection.json')
        plan = dict(**current, configurations=CONFIGS, pairing=pairing, pair_cache_sha256=pair_sha,
            model='Frozen native MXFP4 GPT-OSS-120B; train readout only', kind='mlp', hidden=64,
            margin=MARGIN, pair_groups_per_step=AUX_GROUPS, auxiliary_rows_per_step=AUX_GROUPS * 5,
            ranking='mean softplus(0.5 - (a_system - a_other)); a=z0-logsumexp(z1:); four matched negative roles.',
            ce_rows_per_epoch=FIT_ROWS, steps_per_epoch=STEPS, ce_batch_size=BATCH_SIZE, last_ce_batch=554,
            classification='Exactly the existing dense578090 CE rows each epoch; auxiliary640 rows receive ranking only.',
            auxiliary_scope='Only original132 fit long-full documents. Gains do not isolate matched pairing from ordinary long-text reweighting.',
            neural=dict(seed=data.SEED, pair_index_seed=PAIR_SEED, auxiliary_dropout_seed_base=DROPOUT_SEED,
                learning_rate=.001, weight_decay=.001, dropout=.1, gradient_clip=5., max_epochs=45, min_epochs=10, patience=8),
            rng='Same model initialization and CE generator in all candidates; independent pair generator; fork_rng auxiliary dropout seed=base+global_step including lambda0.',
            threshold='One global validation threshold over all five profiles, FPR<=5% in each.',
            validation_profiles=long_probes.PROFILES, early_stopping=EARLY_STOPPING_LIMIT,
            promotion=dict(mean_f1_gain=.01, long_full_f1_gain=.03, tail_f1_gain=.03, maximum_other_profile_loss=.02),
            accuracy_floor=previous['validation']['mean_accuracy'] - .01,
            final_gate='Paired promotion alone is insufficient; compare against frozen dense selection with the final selector.',
            tests_read=False, fresh_confirmation_required=True, implementation_checks=checks)
        plan_path = OUT / 'search-plan.json'
        if plan_path.exists():
            require(read_json(plan_path) == plan, 'Existing paired plan differs; refusing to overwrite protocol.')
        else:
            data.atomic_json(plan_path, plan)
        verify_plan(plan)
        require(data.digest(long_probes.BANK / 'features.f16') == current['feature_sha256'], 'Frozen feature-bank SHA changed.')
        if (OUT / 'selection.json').exists():
            frozen = read_json(OUT / 'selection.json')
            require(frozen['search_plan_sha256'] == data.digest(plan_path), 'Frozen paired selection plan changed.')
            for key in CONFIGS:
                record = candidate_record(key, plan)
                require(frozen['candidates'][key] == data.digest(OUT / 'candidates' / f'{key}.json'), 'Frozen candidate record changed.')
            require(data.digest(Path(frozen['model_path'])) == frozen['model_sha256'], 'Frozen selected model changed.')
            print(json.dumps(dict(stage='already_complete_verified', candidate=frozen['candidate'])), flush=True)
            return
        pending = []
        for key in CONFIGS:
            if (OUT / 'candidates' / f'{key}.json').exists():
                candidate_record(key, plan)
            else:
                pending.append(key)
        run_workers(pending, plan)
        frozen = select(plan)
        print(json.dumps(dict(candidate=frozen['candidate'], promoted=frozen['promoted'])), flush=True)


def preflight():
    # No formal output directory, signature, model or raw activation is touched.
    fit = pd.read_parquet(long_probes.BANK / 'rows.parquet', filters=[('split', '=', 'fit')])
    dense, arrays, pairing = build_pairs(fit)
    checks = implementation_checks()
    target = fit.index[fit.profile == 'long-full'][0]
    guards = {}
    for name, field, value in [('validation_leak', 'split', 'validation'),
            ('mismatched_token', 'token_id', int(fit.loc[target, 'token_id']) + 1),
            ('duplicate_cache_identity', 'cache_ix', int(fit.iloc[0].cache_ix))]:
        tampered = fit.copy()
        tampered.loc[target, field] = value
        try:
            build_pairs(tampered)
        except RuntimeError:
            guards[name] = True
        else:
            raise RuntimeError(f'Negative pairing guard did not reject {name}.')
    folder = Path(tempfile.mkdtemp(prefix='systemness-paired-preflight-', dir='/tmp'))
    np.savez_compressed(folder / 'fit-pairs.npz', **arrays)
    report = dict(stage='cpu_preflight_passed', pairing=pairing, dense_fit_rows=len(dense),
        implementation_checks=checks, rejection_guards=guards, rows_sha256=data.digest(long_probes.BANK / 'rows.parquet'),
        source_sha256=data.digest(__file__), raw_activations_read=False, validation_rows_materialized=0,
        test_rows_materialized=0, gpu_work_run=False, formal_metadata_written=False,
        control_limitation=EARLY_STOPPING_LIMIT,
        pair_input_f32_mib=AUX_GROUPS * 5 * 8640 * 4 / 2**20,
        pair_indices_i64_mib=PAIR_GROUPS * 5 * 8 / 2**20, report_path=str(folder / 'report.json'))
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='CPU-only checks; temporary output under /tmp.')
    action.add_argument('--train', action='store_true', help='Declare and execute the three isolated GPU workers.')
    parser.add_argument('--worker', choices=CONFIGS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and not args.train:
        parser.error('--worker is internal and requires --train.')
    if args.train:
        worker(args.worker) if args.worker else train()
    elif args.preflight:
        preflight()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
