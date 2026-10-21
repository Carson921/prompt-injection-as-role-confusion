"""Diagnose all pre-projection query heads with online, frozen-MXFP4 LDA.

Default: help. --preflight is CPU-only and writes under /tmp. --stage scan,
validate or all explicitly runs four sharded A40s. Scan uses original fit-only
long-full documents; validation later uses the original five native profiles.
No test/confirmation input, full activation bank, pipeline thread or MLP is used.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-head-scan'
CACHE = data.CACHE / 'head-scan'
LAYERS, HEADS, DIM, CLASSES = 36, 64, 64, 5
TOTAL_HEADS = LAYERS * HEADS
BATCH_SIZE = 4
SHRINKAGE = .1
TOP_K = 8
CHECKPOINT_INTERVAL = 16
DEPENDENCIES = ['systemness_head_scan.py', 'systemness_data.py', 'systemness_long_data.py',
    'systemness_long_probes.py', 'systemness_probes.py', 'run_local_gptoss120b.py']
require = long_data.require


def read(path):
    return json.loads(Path(path).read_text())


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def frame_digest(frame, columns):
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def atomic_npz(path, **arrays):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


class HeadMoments:
    """Per-head float64 class sums and one pooled uncentered sumXXT."""
    def __init__(self, heads=TOTAL_HEADS, dimensions=DIM, classes=CLASSES, arrays=None):
        self.heads, self.dimensions, self.classes = heads, dimensions, classes
        if arrays is None:
            self.n = np.zeros((heads, classes), dtype=np.int64)
            self.sum = np.zeros((heads, classes, dimensions), dtype=np.float64)
            self.xx = np.zeros((heads, dimensions, dimensions), dtype=np.float64)
        else:
            self.n = np.array(arrays['n'], dtype=np.int64, copy=True)
            self.sum = np.array(arrays['sum'], dtype=np.float64, copy=True)
            self.xx = np.array(arrays['xx'], dtype=np.float64, copy=True)
        require(self.n.shape == (heads, classes) and self.sum.shape == (heads, classes, dimensions)
            and self.xx.shape == (heads, dimensions, dimensions), 'Moment shapes changed.')

    def update(self, values, labels):
        values = np.asarray(values, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
        require(values.shape == (len(labels), self.heads, self.dimensions)
                and np.isfinite(values).all() and ((labels >= 0) & (labels < self.classes)).all(), 'Invalid moment batch.')
        self.xx += np.einsum('nhd,nhe->hde', values, values, optimize=True)
        for role in range(self.classes):
            chosen = values[labels == role]
            self.n[:, role] += len(chosen)
            if len(chosen): self.sum[:, role] += chosen.sum(0)

    def arrays(self):
        return dict(n=self.n, sum=self.sum, xx=self.xx)

    def fit(self, indices=None, return_diagnostics=False):
        indices = np.arange(self.heads) if indices is None else np.asarray(indices)
        n, total, xx = self.n[indices], self.sum[indices], self.xx[indices]
        require((n > 0).all() and (n.sum(1) > self.classes).all(), 'Every fitted head needs all five classes.')
        mean = total / n[:, :, None]
        covariance = (xx - np.einsum('hkd,hke,hk->hde', total, total, 1. / n, optimize=True))
        covariance /= (n.sum(1) - self.classes)[:, None, None]
        covariance = .5 * (covariance + covariance.transpose(0, 2, 1))
        tau = np.trace(covariance, axis1=1, axis2=2) / self.dimensions
        grand_sum = total.sum(1); count = n.sum(1)
        total_covariance = (xx - np.einsum('hd,he,h->hde', grand_sum, grand_sum, 1. / count))
        total_covariance /= (count - 1)[:, None, None]
        total_tau = np.trace(total_covariance, axis1=1, axis2=2) / self.dimensions
        require(np.isfinite(covariance).all() and np.isfinite(total_tau).all(), 'Invalid covariance.')
        ridge_base = np.maximum.reduce([tau, 1e-6 * total_tau, np.full(len(tau), 1e-12)])
        covariance *= 1. - SHRINKAGE
        covariance += SHRINKAGE * ridge_base[:, None, None] * np.eye(self.dimensions)
        weight = np.linalg.solve(covariance, mean.transpose(0, 2, 1)).transpose(0, 2, 1)
        prior = n / n.sum(1, keepdims=True)
        bias = -.5 * np.einsum('hkd,hkd->hk', mean, weight) + np.log(prior)
        require(np.isfinite(weight).all() and np.isfinite(bias).all(), 'Invalid LDA classifier.')
        diagnostics = dict(heads=len(indices), zero_within_heads=int(np.count_nonzero(tau <= 0)),
            zero_total_heads=int(np.count_nonzero(total_tau <= 0)),
            floor_applied_heads=int(np.count_nonzero(ridge_base > tau)),
            floor_applied_relative_head_indices=np.flatnonzero(ridge_base > tau).tolist(),
            total_covariance_relative_floor=1e-6, absolute_floor=1e-12, alpha=SHRINKAGE,
            effective_diagonal_ridge_min=float((SHRINKAGE * ridge_base).min()),
            effective_diagonal_ridge_max=float((SHRINKAGE * ridge_base).max()),
            zero_definition='Nonpositive numerical trace/D; small positive traces are counted separately by floor_applied.',
            formula='.9*within+.1*max(trace(within)/D,1e-6*trace(total_cov)/D,1e-12)*I; total_cov denominator N-1.')
        return (weight, bias, diagnostics) if return_diagnostics else (weight, bias)


def predict_heads(values, weight, bias):
    logits = np.einsum('nhd,hkd->nhk', np.asarray(values, dtype=np.float64), weight) + bias[None]
    logits -= logits.max(-1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(-1, keepdims=True)
    return probability


def load_moments(path, heads=TOTAL_HEADS, dimensions=DIM, classes=CLASSES):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        moments = HeadMoments(heads, dimensions, classes,
            arrays={name: archive[name] for name in ['n', 'sum', 'xx']})
    require((moments.n >= 0).all() and np.isfinite(moments.sum).all() and np.isfinite(moments.xx).all(), 'Corrupt moment snapshot.')
    return moments, meta


def model_folder():
    value = os.environ.get('ROLE_ANALYSIS_LOAD_PATH', '/dev/shm/role-analysis-gptoss120b')
    require(not (value == '/workspace' or value.startswith('/workspace/')), 'Do not access the failed /workspace mount.')
    folder = Path(value).resolve()
    require(not (str(folder) == '/workspace' or str(folder).startswith('/workspace/')), 'Model resolves to /workspace.')
    os.environ['ROLE_ANALYSIS_LOAD_PATH'] = str(folder)
    if os.environ.get('HF_HOME', '').startswith('/workspace'):
        os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
    require((folder / 'verified-weights.json').is_file(), 'Verified native weights manifest missing.')
    config = read(folder / 'config.json')
    require((config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (LAYERS, HEADS, DIM),
            'Expected36 layers,64 query heads and64 dimensions per head.')
    return folder


def source_signature():
    folder = model_folder()
    metadata = {
        'long-plan': long_data.OUT / 'plan.json', 'long-data': long_data.OUT / 'data-complete.json',
        'long-keys': long_data.OUT / 'document-keys.json', 'long-extraction': long_data.OUT / 'extraction-complete.json',
        'dense-selection': long_data.OUT / 'selection.json',
        'paired-selection': data.runtime.OUT / 'systemness-paired-pilot/selection.json',
        'frozen-primary': data.runtime.OUT / 'systemness-final-readout/selection.json',
        'original-data': data.OUT / 'data-complete.json', 'supplement-data': data.OUT / 'supplement-data-complete.json',
        'checkpoint-verification': data.runtime.OUT / 'checkpoint-cache-verification.json'}
    return dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        source_metadata_sha256={name: data.digest(path) for name, path in metadata.items()},
        long_inputs_sha256=data.digest(long_data.CACHE / 'inputs.parquet'),
        long_rows_sha256=data.digest(long_data.CACHE / 'rows.parquet'),
        old_bank_rows_sha256=data.digest(long_probes.BANK / 'rows.parquet'),
        model_path=str(folder), verified_weights_sha256=data.digest(folder / 'verified-weights.json'),
        config_sha256=data.digest(folder / 'config.json'),
        tokenizer_sha256=long_data.tokenizer_assets()[1],
        package_versions={name: importlib.metadata.version(name) for name in
                          ['torch', 'transformers', 'accelerate', 'numpy', 'scipy', 'pandas', 'scikit-learn']},
        kernel_revision=data.runtime.KERNEL_REVISION)


def check_source_cache(name, folder, meta_path):
    metadata = read(meta_path)
    # Hash only files used by this runner; mixed split token JSON is never decoded.
    for file in ['inputs.parquet', 'rows.parquet']:
        require(data.digest(folder / file) == metadata['cache_sha256'][file], f'{name} {file} changed.')
    return metadata


class Bundle:
    def __init__(self, inputs, rows, tokenizer):
        self.inputs = inputs.sort_values(['length', 'uid']).reset_index(drop=True)
        self.rows = rows.copy().reset_index(drop=True)
        self.rows['target_ix'] = np.arange(len(rows), dtype=np.int64)
        require(self.inputs.uid.is_unique and set(self.inputs.uid) == set(self.rows.uid), 'Prompt membership mismatch.')
        self.tokens = {}
        self.lookup = {}
        for entry in self.inputs.itertuples():
            ids = tokenizer.encode(entry.prompt, add_special_tokens=False)
            require(len(ids) == entry.length, 'Retokenized input length differs from the frozen cache.')
            group = self.rows[self.rows.uid == entry.uid]
            actual = long_data.parse_native_labels(ids, tokenizer)
            require(all(0 <= row.token_ix < len(ids) and ids[row.token_ix] == row.token_id
                        and actual.get(row.token_ix) == row.role for row in group.itertuples()),
                    'Sampled native role/token identity differs after local retokenization.')
            self.tokens[entry.uid] = ids; self.lookup[entry.uid] = group

    def batches(self):
        return [self.inputs.iloc[start:start + BATCH_SIZE] for start in range(0, len(self.inputs), BATCH_SIZE)]

    def collate(self, batch):
        import torch
        width = int(batch.length.max())
        ids = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        positions = []; targets = []; labels = []
        for index, entry in enumerate(batch.itertuples()):
            tokens = self.tokens[entry.uid]; size = len(tokens)
            ids[index, -size:] = torch.tensor(tokens); mask[index, -size:] = 1
            group = self.lookup[entry.uid]
            positions.extend((group.token_ix.to_numpy() + index * width + width - size).tolist())
            targets.extend(group.target_ix.tolist()); labels.extend(probes.labels(group).tolist())
        require(len(set(targets)) == len(targets) and min(labels) >= 0, 'Duplicate or unlabeled sampled batch rows.')
        return dict(input_ids=ids, attention_mask=mask, positions=np.asarray(positions, dtype=np.int64),
            targets=np.asarray(targets, dtype=np.int64), labels=np.asarray(labels, dtype=np.int64))

    def descriptor(self):
        return dict(prompts=len(self.inputs), rows=len(self.rows), input_tokens=int(self.inputs.length.sum()),
            inputs_sha256=frame_digest(self.inputs, ['uid', 'prompt', 'length']),
            rows_sha256=frame_digest(self.rows, ['sample_ix', 'uid', 'token_ix', 'token_id', 'role', 'document_id']),
            sample_ix_sha256=digest_array(self.rows.sample_ix.to_numpy()))


def scan_data(tokenizer):
    metadata = check_source_cache('long-full', long_data.CACHE, long_data.OUT / 'data-complete.json')
    require(metadata['independent_validation']['all_labels_match'] and metadata['test_inputs_read'] is False,
            'Long-full source was not independently validated.')
    rows = pd.read_parquet(long_data.CACHE / 'rows.parquet', filters=[('split', '=', 'fit')])
    inputs = pd.read_parquet(long_data.CACHE / 'inputs.parquet', filters=[('split', '=', 'fit')])
    require(len(rows) == 168960 and rows.document_id.nunique() == 132 and set(rows.profile) == {'long-full'},
            'Expected only the original132 long-full fit documents.')
    require(set(rows.role) == set(data.ROLES) and rows.groupby(['document_id', 'role']).size().eq(256).all(),
            'Long-full fit class/position counts changed.')
    rows['body_offset'] = rows.token_ix - rows.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')
    expected = data.uniform_indices(1024, 256)
    for _, group in rows.groupby(['document_id', 'role']):
        require(np.array_equal(np.sort(group.body_offset), expected), 'Long-full body positions changed.')
    paired = rows.groupby(['document_id', 'body_offset'])
    require(paired.size().eq(5).all() and paired.token_id.nunique().eq(1).all(), 'Cross-role content pairing differs.')
    keys = read(long_data.OUT / 'document-keys.json')
    fit_ids = {item['document_id'] for item in keys if item['split'] == 'fit'}
    val_ids = {item['document_id'] for item in keys if item['split'] == 'validation'}
    require(set(rows.document_id) == fit_ids and len(val_ids) == 31 and not fit_ids & val_ids, 'Fit/validation document leak.')
    rows['uid'] = 'long-full:' + rows.prompt_ix.astype(str)
    inputs['uid'] = 'long-full:' + inputs.prompt_ix.astype(str)
    result = {}
    for name, is_discovery, count in [('fit', False, 101), ('discovery', True, 31)]:
        selected = rows[(rows.document_id % 5 == 0) == is_discovery].copy()
        require(selected.document_id.nunique() == count and set(selected.split) == {'fit'}, 'Internal source partition changed.')
        chosen_inputs = inputs[inputs.uid.isin(selected.uid)]
        result[name] = Bundle(chosen_inputs, selected, tokenizer)
    require(not set(result['fit'].rows.document_id) & set(result['discovery'].rows.document_id), 'Internal source partitions overlap.')
    return result


def validation_data(tokenizer):
    rows = pd.read_parquet(long_probes.BANK / 'rows.parquet', filters=[('split', '=', 'validation')])
    require(len(rows) == 173032 and set(rows.profile) == set(long_probes.PROFILES)
            and set(rows.wrapper) == {'native'}, 'Original native five-profile validation rows changed.')
    sources = {'main': (data.CACHE, data.OUT / 'data-complete.json'),
        'supplement': (data.CACHE / 'supplement', data.OUT / 'supplement-data-complete.json'),
        'long-full': (long_data.CACHE, long_data.OUT / 'data-complete.json')}
    inputs = []
    for name, (folder, meta_path) in sources.items():
        check_source_cache(name, folder, meta_path)
        wanted = rows[rows.source_set == name]
        frame = pd.read_parquet(folder / 'inputs.parquet', filters=[('split', '=', 'validation'), ('wrapper', '=', 'native')])
        frame = frame[frame.prompt_ix.isin(wanted.prompt_ix)].copy()
        require(set(frame.prompt_ix) == set(wanted.prompt_ix), 'Validation source prompt mapping changed.')
        frame['uid'] = name + ':' + frame.prompt_ix.astype(str); inputs.append(frame)
    rows['uid'] = rows.source_set + ':' + rows.prompt_ix.astype(str)
    return Bundle(pd.concat(inputs, ignore_index=True), rows, tokenizer)


def resource_specification():
    return dict(heads=TOTAL_HEADS, head_dimensions=DIM, classes=CLASSES,
        covariance_f64_mib=TOTAL_HEADS * DIM * DIM * 8 / 2**20,
        class_sum_f64_mib=TOTAL_HEADS * CLASSES * DIM * 8 / 2**20,
        class_count_i64_mib=TOTAL_HEADS * CLASSES * 8 / 2**20,
        persistent_moments_mib=(TOTAL_HEADS * DIM * DIM * 8 + TOTAL_HEADS * CLASSES * DIM * 8 + TOTAL_HEADS * CLASSES * 8) / 2**20,
        discovery_probabilities_bytes=39680 * TOTAL_HEADS * 4,
        discovery_probabilities_mib=39680 * TOTAL_HEADS * 4 / 2**20,
        peak_note='Moment exports/checkpoint compression, LDA factorization, gathered batch activations, and AUROC ranks add bounded workspaces; these persistent counts are not peak RAM/VRAM.',
        cache_note='No full-sequence/all-layer activation cache. Only complete moments, system probabilities and selected8 weights are stored. No existing cache is deleted.')


def declare(bundles):
    current = source_signature()
    checks = preflight_checks()
    value = dict(**current, version=1, roles=data.ROLES, capture='o_proj forward_pre_hook input before4096->2880 projection',
        layers=list(range(LAYERS)), query_heads_per_layer=HEADS, head_dim=DIM, batch_size=BATCH_SIZE,
        runtime='Sequential model.model forward, frozen native MXFP4 on4 sharded A40s with existing synchronous host staging; no pipeline threads.',
        partition='Original132 fit long-full documents: docid%5!=0 statistics101; docid%5==0 discovery31. Original31 validation documents are disjoint.',
        data={name: bundle.descriptor() for name, bundle in bundles.items()},
        lda=dict(shrinkage=SHRINKAGE, covariance='Pooled within-class covariance divided by N-K.',
            ridge='.9*within+.1*max(trace(within)/D,1e-6*trace(total_cov)/D,1e-12)*I; total_cov=(SXXT-sum_all outer/N)/(N-1).',
            total_covariance_relative_floor=1e-6, absolute_floor=1e-12,
            degenerate_heads='Retain zero/constant heads and class-constant role-perfect heads; report zero-within/zero-total/floor-applied counts.',
            priors='Empirical fit class counts'),
        discovery='Actual streamed system probabilities. Macro-average per-source-document all-position and512+ AUROC, average both scores equally; ties use tail AUROC then layer/head ID.',
        top_k=TOP_K, refit='Add statistics-fit and discovery moments, then refit only the frozen top8 on all132 fit documents.',
        aggregation='Arithmetic mean of8 per-token five-class LDA probabilities. Labels/positions/pairs are not inference inputs.',
        validation='Later stream original173032 rows/five native profiles. One global validation threshold and per-profile FPR<=5%; no test-derived choice.',
        restart=dict(fit='Atomic full-moment checkpoint every16 complete batches; restore exact prefix counts and batch identities.',
            discovery='No partial resume: probabilities and discovery moments are committed together only after the complete stage; an interruption reruns all discovery.',
            validation='No partial resume: rerun entire validation if completion marker/probability hash is absent. Status files are informational.'),
        resources=resource_specification(), checkpoint_interval=CHECKPOINT_INTERVAL,
        preflight_checks=checks,
        limitation='Selecting8 out of2304 heads on31 fit-discovery documents can overfit. Discovery is fit development, not held-out validation. Moment separability is not F1; actual validation is required. This diagnostic does not replace the frozen primary or inspect current464/old tests.',
        tests_read=False, confirmation_inputs_read=False)
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(exist_ok=True); (OUT / 'models').mkdir(exist_ok=True)
    (OUT / '.gitignore').write_text('/.run.lock\n')
    path = OUT / 'plan.json'
    if path.exists(): require(read(path) == value, 'Existing head-scan source/data/protocol differs; refusing overwrite.')
    else: data.atomic_json(path, value)
    record = dict(source_sha256=value['source_sha256'], plan_sha256=data.digest(path),
        checks=checks, partitions=value['data'], resources=value['resources'], model_loaded=False,
        gpu_work_run=False, validation_inputs_loaded=False, tests_read=False)
    record_path = OUT / 'cpu-preflight.json'
    if record_path.exists(): require(read(record_path) == record, 'Published preflight record changed.')
    else: data.atomic_json(record_path, record)
    return value


def verify_plan(plan):
    require(all(plan.get(key) == value for key, value in source_signature().items()), 'Head-scan signature changed.')
    preflight = read(OUT / 'cpu-preflight.json')
    require(preflight['plan_sha256'] == data.digest(OUT / 'plan.json')
            and preflight['source_sha256'] == plan['source_sha256']
            and preflight['checks'] == plan['preflight_checks'], 'Published preflight no longer matches the frozen plan.')


def load_model():
    import torch
    require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Exactly4 A40s required.')
    original_atomic, original_status = data.runtime.atomic_json, data.runtime.status
    data.runtime.atomic_json = lambda path, value: data.atomic_json(OUT / Path(path).name, value)
    data.runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
    try:
        model = data.runtime.load_model()
    finally:
        data.runtime.atomic_json, data.runtime.status = original_atomic, original_status
    require(not any(parameter.requires_grad for parameter in model.parameters()), 'Language model is not frozen.')
    return model


class TorchMoments:
    """At most one small F64 statistics block per layer, on its existing GPU."""
    def __init__(self, initial=None):
        self.initial = initial; self.blocks = {}

    def update(self, layer, values, labels):
        import torch
        if layer not in self.blocks:
            if self.initial is None:
                block = (torch.zeros((HEADS, CLASSES), device=values.device, dtype=torch.int64),
                    torch.zeros((HEADS, CLASSES, DIM), device=values.device, dtype=torch.float64),
                    torch.zeros((HEADS, DIM, DIM), device=values.device, dtype=torch.float64))
            else:
                chosen = slice(layer * HEADS, (layer + 1) * HEADS)
                block = tuple(torch.from_numpy(array[chosen]).to(values.device)
                              for array in [self.initial.n, self.initial.sum, self.initial.xx])
            self.blocks[layer] = block
        n, total, xx = self.blocks[layer]
        x = values.to(torch.float64)
        xx.add_(torch.bmm(x.permute(1, 2, 0), x.permute(1, 0, 2)))
        for role in range(CLASSES):
            group = x[labels == role]
            n[:, role].add_(len(group))
            if len(group): total[:, role].add_(group.sum(0))

    def export(self):
        require(set(self.blocks) == set(range(LAYERS)), 'A layer never supplied moment data.')
        arrays = [np.concatenate([self.blocks[layer][index].cpu().numpy() for layer in range(LAYERS)], axis=0)
                  for index in range(3)]
        return HeadMoments(arrays=dict(zip(['n', 'sum', 'xx'], arrays)))


class Collector:
    def __init__(self, moments=None, weight=None, bias=None, probabilities=None, selected=None):
        self.moments, self.weight, self.bias = moments, weight, bias
        self.probabilities = probabilities; self.selected = selected
        self.batch = None; self.seen = set(); self.device_tensors = {}; self.classifiers = {}
        self.enabled = False; self.cpu_probability_difference = 0.
        self.example = None; self.example_targets = None

    def begin(self, batch):
        self.batch = batch; self.seen = set(); self.device_tensors = {}

    def hook(self, layer):
        def receive(module, args):
            import torch
            value = args[0]
            require(value.ndim == 3 and value.shape[-1] == HEADS * DIM, 'Unexpected o_proj input shape.')
            require(layer not in self.seen, 'Layer hook ran twice in one sequential batch.')
            self.seen.add(layer)
            if not self.enabled: return
            key = str(value.device)
            if key not in self.device_tensors:
                self.device_tensors[key] = (torch.from_numpy(self.batch['positions']).to(value.device),
                    torch.from_numpy(self.batch['labels']).to(value.device))
            positions, labels = self.device_tensors[key]
            chosen_heads = None if self.selected is None else np.flatnonzero(self.selected // HEADS == layer)
            x = value.detach().reshape(-1, HEADS, DIM).index_select(0, positions)
            require(bool(torch.isfinite(x).all()), f'Nonfinite sampled pre-o_proj heads at layer{layer}.')
            if self.moments is not None: self.moments.update(layer, x, labels)
            if self.weight is not None:
                if chosen_heads is not None:
                    head_ix = torch.tensor(self.selected[chosen_heads] % HEADS, dtype=torch.long, device=value.device)
                    x = x.index_select(1, head_ix)
                    model_ix = chosen_heads
                else:
                    model_ix = np.arange(layer * HEADS, (layer + 1) * HEADS)
                if layer not in self.classifiers:
                    self.classifiers[layer] = (torch.from_numpy(self.weight[model_ix]).to(value.device),
                        torch.from_numpy(self.bias[model_ix]).to(value.device))
                weight, bias = self.classifiers[layer]
                p = torch.softmax(torch.einsum('nhd,hkd->nhk', x.to(torch.float64), weight) + bias[None], dim=-1)
                targets = self.batch['targets']
                if self.selected is None:
                    self.probabilities[np.ix_(targets, model_ix)] = p[:, :, 0].float().cpu().numpy()
                    if layer not in getattr(self, 'checked_layers', set()):
                        expected = predict_heads(x[:8].double().cpu().numpy(), self.weight[model_ix], self.bias[model_ix])[:, :, 0]
                        difference = float(np.abs(expected - p[:8, :, 0].cpu().numpy()).max())
                        self.cpu_probability_difference = max(self.cpu_probability_difference, difference)
                        if not hasattr(self, 'checked_layers'): self.checked_layers = set()
                        self.checked_layers.add(layer)
                else:
                    self.probabilities[targets] += p.sum(1).cpu().numpy()
                    if self.example is None:
                        self.example = np.full((min(32, len(targets)), TOP_K, DIM), np.nan, dtype=np.float64)
                        self.example_targets = targets[:len(self.example)].copy()
                    if np.array_equal(targets[:len(self.example_targets)], self.example_targets):
                        self.example[:, chosen_heads] = x[:len(self.example)].double().cpu().numpy()
            # A forward_pre_hook must return None; no activation is modified.
        return receive


def install(model, collector):
    layers = list(range(LAYERS)) if collector.selected is None else sorted(set((collector.selected // HEADS).tolist()))
    handles = [model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)) for layer in layers]
    return handles, layers


def forward(model, collector, batch, layers):
    import torch
    collector.begin(batch)
    result = model.model(input_ids=batch['input_ids'].to('cuda:0'),
        attention_mask=batch['attention_mask'].to('cuda:0'), use_cache=False).last_hidden_state
    require(collector.seen == set(layers), 'Sequential forward missed or repeated a head hook.')
    return result


@contextmanager
def checked_hooks(model, collector, bundle, name):
    import torch
    sample = bundle.collate(bundle.batches()[0])
    with torch.inference_mode():
        before = model.model(input_ids=sample['input_ids'].to('cuda:0'),
            attention_mask=sample['attention_mask'].to('cuda:0'), use_cache=False).last_hidden_state.cpu()
        handles, layers = install(model, collector)
        try:
            collector.enabled = False
            after = forward(model, collector, sample, layers).cpu()
            require(torch.equal(before, after) and bool(torch.isfinite(after).all()), 'Head hooks changed hidden states.')
            require(not collector.moments or not collector.moments.blocks, 'Selfcheck accidentally accumulated statistics.')
            data.atomic_json(OUT / f'hook-validation-{name}.json', dict(hidden_states_bitexact=True,
                statistics_disabled=True, layers=layers, sample_rows=len(sample['targets'])))
            del before, after
            collector.enabled = True
            yield layers
        finally:
            for handle in handles: handle.remove()


def checkpoint_meta(bundle, next_batch, phase, plan_sha256=None):
    covered = bundle.batches()[:next_batch]
    uids = [uid for batch in covered for uid in batch.uid]
    prefix = bundle.rows[bundle.rows.uid.isin(uids)]
    return dict(plan_sha256=plan_sha256 or data.digest(OUT / 'plan.json'), phase=phase, next_batch=next_batch,
        rows_written=len(prefix), class_counts=np.bincount(probes.labels(prefix), minlength=CLASSES).tolist(),
        data=bundle.descriptor())


def fit_statistics(get_model, bundle, plan):
    path = CACHE / 'fit-moments.npz'; done = OUT / 'fit-complete.json'
    if done.exists():
        metadata = read(done)
        require(metadata['plan_sha256'] == data.digest(OUT / 'plan.json') and metadata['moments_sha256'] == data.digest(path), 'Completed fit moments changed.')
        moments, saved = load_moments(path)
        require(saved == checkpoint_meta(bundle, len(bundle.batches()), 'fit'), 'Completed fit prefix differs.')
        require(np.array_equal(moments.n, np.broadcast_to(saved['class_counts'], moments.n.shape)), 'Fit counts changed.')
        return moments
    initial = None; start = 0
    if path.exists():
        initial, saved = load_moments(path); start = saved['next_batch']
        require(0 <= start <= len(bundle.batches()) and saved == checkpoint_meta(bundle, start, 'fit'), 'Moment checkpoint does not match the exact completed prefix.')
        require(np.array_equal(initial.n, np.broadcast_to(saved['class_counts'], initial.n.shape)), 'Checkpoint class counts differ from its source prefix.')
    gpu = TorchMoments(initial); collector = Collector(moments=gpu)
    import torch
    model = get_model(); started = time.time()
    with checked_hooks(model, collector, bundle, 'fit') as layers, torch.inference_mode():
        for index, part in enumerate(bundle.batches()):
            if index < start: continue
            batch = bundle.collate(part); output = forward(model, collector, batch, layers); del output
            if (index + 1) % CHECKPOINT_INTERVAL == 0 or index + 1 == len(bundle.batches()):
                snapshot = gpu.export(); meta = checkpoint_meta(bundle, index + 1, 'fit')
                require(np.array_equal(snapshot.n, np.broadcast_to(meta['class_counts'], snapshot.n.shape)), 'Online fit counts differ from sampled source rows.')
                atomic_npz(path, metadata=np.array(json.dumps(meta)), **snapshot.arrays())
                data.atomic_json(OUT / 'status.json', dict(stage='fit-statistics', batch=index + 1,
                    batches=len(bundle.batches()), rows=meta['rows_written'], seconds=time.time() - started))
                print(json.dumps(dict(stage='fit-statistics', batch=index + 1, batches=len(bundle.batches()))), flush=True)
        moments = gpu.export() if gpu.blocks else initial
    verify_plan(plan)
    require(moments is not None, 'No fit statistics were accumulated.')
    data.atomic_json(done, dict(plan_sha256=data.digest(OUT / 'plan.json'), moments_sha256=data.digest(path),
        data=bundle.descriptor(), rows=len(bundle.rows), tests_read=False))
    del collector, gpu; gc.collect()
    return moments


def discovery_statistics(get_model, bundle, fitted, plan):
    probability_path = CACHE / 'discovery-system-probabilities.npy'
    moments_path = CACHE / 'discovery-moments.npz'; done_path = OUT / 'discovery-complete.json'
    if done_path.exists():
        done = read(done_path)
        require(done['plan_sha256'] == data.digest(OUT / 'plan.json')
            and done['probabilities_sha256'] == data.digest(probability_path)
            and done['moments_sha256'] == data.digest(moments_path)
            and done['fit_lda_diagnostics_sha256'] == data.digest(OUT / 'fit-lda-diagnostics.json'),
            'Completed discovery state changed.')
        moments, meta = load_moments(moments_path)
        require(meta == checkpoint_meta(bundle, len(bundle.batches()), 'discovery'), 'Discovery data differs.')
        return np.load(probability_path, mmap_mode='r'), moments, done
    # No saved partial probability cursor is consulted. Any incomplete discovery
    # restarts with fresh full moments and a fully unwritten NaN probability array.
    probability = np.full((len(bundle.rows), TOTAL_HEADS), np.nan, dtype=np.float32)
    weight, bias, fit_diagnostics = fitted.fit(return_diagnostics=True)
    diagnostics_path = OUT / 'fit-lda-diagnostics.json'
    if diagnostics_path.exists(): require(read(diagnostics_path) == fit_diagnostics, 'Fit LDA diagnostics changed.')
    else: data.atomic_json(diagnostics_path, fit_diagnostics)
    gpu = TorchMoments(); collector = Collector(gpu, weight, bias, probability)
    import torch
    model = get_model(); started = time.time()
    with checked_hooks(model, collector, bundle, 'discovery') as layers, torch.inference_mode():
        for index, part in enumerate(bundle.batches()):
            output = forward(model, collector, bundle.collate(part), layers); del output
            if (index + 1) % 8 == 0 or index + 1 == len(bundle.batches()):
                data.atomic_json(OUT / 'status.json', dict(stage='discovery', batch=index + 1,
                    batches=len(bundle.batches()), partial_resume=False, seconds=time.time() - started))
                print(json.dumps(dict(stage='discovery', batch=index + 1, batches=len(bundle.batches()))), flush=True)
    moments = gpu.export(); meta = checkpoint_meta(bundle, len(bundle.batches()), 'discovery')
    require(np.array_equal(moments.n, np.broadcast_to(meta['class_counts'], moments.n.shape)), 'Discovery moment counts differ.')
    for start in range(0, len(probability), 4096):
        block = probability[start:start + 4096]
        require(np.isfinite(block).all() and (block >= 0).all() and (block <= 1).all(), 'Unwritten or invalid discovery probabilities.')
    require(collector.cpu_probability_difference < 1e-8, 'GPU and portable NumPy discovery predictions differ.')
    verify_plan(plan)
    temporary = Path(str(probability_path) + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, probability); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(probability_path)
    atomic_npz(moments_path, metadata=np.array(json.dumps(meta)), **moments.arrays())
    done = dict(plan_sha256=data.digest(OUT / 'plan.json'), data=bundle.descriptor(),
        probabilities_sha256=data.digest(probability_path), moments_sha256=data.digest(moments_path),
        fit_lda_diagnostics_sha256=data.digest(diagnostics_path),
        actual_probabilities_streamed=True, cpu_probability_max_difference=collector.cpu_probability_difference,
        partial_resume=False, tests_read=False)
    data.atomic_json(done_path, done)
    del collector, gpu; gc.collect()
    return probability, moments, done


def document_auc(rows, probability):
    from scipy.stats import rankdata
    require(probability.shape == (len(rows), TOTAL_HEADS), 'Discovery probability row/head order changed.')
    doc_ids = sorted(rows.document_id.unique().tolist()); all_scores = []; tail_scores = []
    for doc in doc_ids:
        member = np.flatnonzero(rows.document_id.to_numpy() == doc)
        scores = []
        for selected in [member, member[rows.iloc[member].body_offset.to_numpy() >= 512]]:
            positive = rows.iloc[selected].role.to_numpy() == 'system'
            ranks = rankdata(np.asarray(probability[selected]), axis=0, method='average')
            n, m = int(positive.sum()), int((~positive).sum())
            require(n > 0 and m > 0, 'Discovery source lacks positive or negative roles.')
            scores.append((ranks[positive].sum(0) - n * (n + 1) / 2) / (n * m))
        all_scores.append(scores[0]); tail_scores.append(scores[1])
    all_scores, tail_scores = np.array(all_scores), np.array(tail_scores)
    all_mean, tail_mean = all_scores.mean(0), tail_scores.mean(0)
    score = .5 * (all_mean + tail_mean)
    order = np.lexsort((np.arange(TOTAL_HEADS), -tail_mean, -score))
    return order, dict(document_ids=np.asarray(doc_ids), all_auc=all_scores, tail_auc=tail_scores), all_mean, tail_mean, score


def scan(get_model, bundles, plan):
    done_path = OUT / 'scan-complete.json'
    if done_path.exists():
        done = read(done_path)
        require(done['plan_sha256'] == data.digest(OUT / 'plan.json')
            and done['model_sha256'] == data.digest(OUT / 'models/top8-lda.npz')
            and done['head_selection_sha256'] == data.digest(OUT / 'head-selection.json'), 'Frozen scan artifacts changed.')
        return done
    fitted = fit_statistics(get_model, bundles['fit'], plan)
    probability, discovery, discovery_done = discovery_statistics(get_model, bundles['discovery'], fitted, plan)
    order, by_doc, all_mean, tail_mean, score = document_auc(bundles['discovery'].rows, probability)
    selected = order[:TOP_K]
    combined = HeadMoments(heads=TOP_K, arrays={name: fitted.arrays()[name][selected] + discovery.arrays()[name][selected]
                                               for name in ['n', 'sum', 'xx']})
    weight, bias, refit_diagnostics = combined.fit(return_diagnostics=True)
    require(np.array_equal(combined.n, np.full((TOP_K, CLASSES), 132 * 256)), 'Refit failed to use all132 fit documents.')
    metadata = dict(kind='headwise-shrunk-lda', roles=data.ROLES, selected_head_ids=selected.tolist(),
        selected_layer_head_pairs=[[int(head // HEADS), int(head % HEADS)] for head in selected],
        head_dim=DIM, shrinkage=SHRINKAGE, aggregation='Uniform mean of8 five-class probabilities.',
        refit_lda_diagnostics=refit_diagnostics,
        fit_documents=132, discovery_documents=31, head_selection_fit_only=True,
        refit_includes_discovery=True, inference='Only current-token selected pre-o_proj head vectors; no label/position/pair inputs.',
        source_sha256=plan['source_sha256'], plan_sha256=data.digest(OUT / 'plan.json'))
    path = OUT / 'models/top8-lda.npz'
    atomic_npz(path, weight=weight, bias=bias, selected_head_ids=selected, metadata=np.array(json.dumps(metadata)))
    with np.load(path, allow_pickle=False) as archive:
        require(np.array_equal(archive['weight'], weight) and np.array_equal(archive['bias'], bias)
                and np.array_equal(archive['selected_head_ids'], selected), 'Portable head model round trip changed.')
    rank = np.empty(TOTAL_HEADS, dtype=int); rank[order] = np.arange(1, TOTAL_HEADS + 1)
    pd.DataFrame(dict(head_id=np.arange(TOTAL_HEADS), layer=np.arange(TOTAL_HEADS) // HEADS,
        head=np.arange(TOTAL_HEADS) % HEADS, discovery_doc_macro_auroc=all_mean,
        discovery_tail_doc_macro_auroc=tail_mean, selection_score=score, rank=rank)).to_csv(OUT / 'discovery-head-metrics.csv', index=False)
    atomic_npz(CACHE / 'discovery-doc-auroc.npz', **by_doc)
    head_selection = dict(**metadata,
        selection_rule='Mean of source-doc-macro all-position and512+ AUROC; tie tail AUROC, then global head ID.',
        selected_discovery_scores=[dict(head_id=int(head), all_auroc=float(all_mean[head]), tail_auroc=float(tail_mean[head]),
            score=float(score[head])) for head in selected],
        probabilities_sha256=discovery_done['probabilities_sha256'], model_path=str(path), model_sha256=data.digest(path),
        validation_pending=True, test_used_for_selection=False,
        limitation='Fit-discovery selection among2304 heads can overfit31 documents; these scores are not held-out validation F1.')
    data.atomic_json(OUT / 'head-selection.json', head_selection)
    verify_plan(plan)
    done = dict(plan_sha256=data.digest(OUT / 'plan.json'), model_sha256=data.digest(path),
        head_selection_sha256=data.digest(OUT / 'head-selection.json'), discovery_complete_sha256=data.digest(OUT / 'discovery-complete.json'),
        selected_head_ids=selected.tolist(), validation_pending=True, tests_read=False)
    data.atomic_json(done_path, done)
    data.atomic_json(OUT / 'status.json', dict(stage='scan_complete', selected_head_ids=selected.tolist(), validation_pending=True))
    return done


def validate(get_model, tokenizer, plan):
    scan_done = read(OUT / 'scan-complete.json'); head_selection = read(OUT / 'head-selection.json')
    model_path = OUT / 'models/top8-lda.npz'
    require(scan_done['model_sha256'] == head_selection['model_sha256'] == data.digest(model_path)
            and scan_done['head_selection_sha256'] == data.digest(OUT / 'head-selection.json'), 'Frozen selected heads/model changed.')
    bundle = validation_data(tokenizer); done_path = OUT / 'validation-complete.json'
    probability_path = CACHE / 'validation-probabilities.npz'
    if done_path.exists():
        done = read(done_path)
        require(done['plan_sha256'] == data.digest(OUT / 'plan.json')
            and done['probabilities_sha256'] == data.digest(probability_path)
            and done['metrics_sha256'] == data.digest(OUT / 'validation-metrics.json')
            and done['data'] == bundle.descriptor(), 'Completed head validation changed.')
        return done
    with np.load(model_path, allow_pickle=False) as archive:
        weight, bias, selected = archive['weight'].copy(), archive['bias'].copy(), archive['selected_head_ids'].copy()
    require(selected.shape == (TOP_K,) and len(set(selected)) == TOP_K, 'Selected head identity changed.')
    probability_sum = np.zeros((len(bundle.rows), CLASSES), dtype=np.float64)
    collector = Collector(weight=weight, bias=bias, probabilities=probability_sum, selected=selected)
    import torch
    model = get_model(); consumer_difference = None
    with checked_hooks(model, collector, bundle, 'validation') as layers, torch.inference_mode():
        for index, part in enumerate(bundle.batches()):
            batch = bundle.collate(part); output = forward(model, collector, batch, layers); del output
            if index == 0:
                require(np.isfinite(collector.example).all(), 'Portable consumer sample missed a selected head.')
                expected = predict_heads(collector.example, weight, bias).mean(1)
                actual = probability_sum[collector.example_targets] / TOP_K
                consumer_difference = float(np.abs(expected - actual).max())
                require(consumer_difference < 1e-8, 'Portable NumPy aggregate differs from actual streamed GPU predictions.')
            if (index + 1) % 8 == 0 or index + 1 == len(bundle.batches()):
                data.atomic_json(OUT / 'status.json', dict(stage='validation', batch=index + 1,
                    batches=len(bundle.batches()), partial_resume=False))
                print(json.dumps(dict(stage='validation', batch=index + 1, batches=len(bundle.batches()))), flush=True)
    probability = (probability_sum / TOP_K).astype(np.float32)
    require(np.isfinite(probability).all() and np.allclose(probability.sum(1), 1., atol=2e-5), 'Invalid pooled validation probabilities.')
    result = long_probes.validation_metrics(bundle.rows.reset_index(drop=True), probability)
    verify_plan(plan)
    atomic_npz(probability_path, p=probability, sample_ix=bundle.rows.sample_ix.to_numpy())
    metrics = dict(validation=result, model_path=str(model_path), model_sha256=data.digest(model_path),
        head_selection_sha256=data.digest(OUT / 'head-selection.json'), plan_sha256=data.digest(OUT / 'plan.json'),
        probabilities_sha256=data.digest(probability_path), data=bundle.descriptor(),
        portable_probability_max_difference=consumer_difference, validation_only=True, tests_read=False,
        selection_note='Heads frozen using fit-discovery only. This stage calibrates one global validation threshold and never changes the selected head model.')
    data.atomic_json(OUT / 'validation-metrics.json', metrics)
    flat = []
    for profile, values in dict(result['profiles'], **{'long-full-tail': result['long_full_tail']}).items():
        flat.append(dict(profile=profile, **{key: value for key, value in values.items() if not isinstance(value, (dict, list))},
            cot_recall=values['recall'].get('cot')))
    pd.DataFrame(flat).to_csv(OUT / 'validation-metrics.csv', index=False)
    done = dict(plan_sha256=data.digest(OUT / 'plan.json'), scan_complete_sha256=data.digest(OUT / 'scan-complete.json'),
        model_sha256=data.digest(model_path), probabilities_sha256=data.digest(probability_path),
        metrics_sha256=data.digest(OUT / 'validation-metrics.json'), data=bundle.descriptor(), tests_read=False)
    data.atomic_json(done_path, done); data.atomic_json(OUT / 'status.json', dict(stage='validation_complete', validation_only=True))
    return done


def no_competing_jobs():
    names = {'systemness_long_extract.py', 'systemness_recover_local_bank.py', 'systemness_long_confirm.py',
        'systemness_attention_confirm.py', 'systemness_long_probes.py', 'systemness_paired_probes.py',
        'systemness_natural_recheck.py', 'systemness_head_scan.py'}
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg): active.append(int(entry.name))
    require(not active, f'Other experiment processes remain active: {active}; wait before loading another model.')


@contextmanager
def run_lock():
    import fcntl
    with (OUT / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another head-scan supervisor holds this output directory.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(stage):
    no_competing_jobs()
    tokenizer = long_data.load_tokenizer(); bundles = scan_data(tokenizer)
    plan = declare(bundles)
    available = os.statvfs(CACHE)
    needs_scan = stage in ['scan', 'all'] and not (OUT / 'scan-complete.json').exists()
    needs_validation = stage in ['validate', 'all'] and not (OUT / 'validation-complete.json').exists()
    needed_bytes = 1024**3 if needs_scan else (64 * 2**20 if needs_validation else 2**20)
    require(available.f_bavail * available.f_frsize > needed_bytes,
        'Insufficient space for the requested remaining stage; no cache will be deleted.')
    model = None
    def get_model():
        nonlocal model
        if model is None: model = load_model()
        return model
    with run_lock():
        verify_plan(plan)
        if stage in ['scan', 'all']: scan(get_model, bundles, plan)
        if stage in ['validate', 'all']:
            require((OUT / 'scan-complete.json').exists(), 'Run scan before validation.')
            validate(get_model, tokenizer, plan)
        verify_plan(plan)
    del model; gc.collect()


def cpu_checks():
    from scipy.stats import rankdata
    rng = np.random.default_rng(data.SEED)
    h, d, k = 3, 4, 5
    labels = np.repeat(np.arange(k), [47, 51, 56, 43, 61])
    values = rng.normal(size=(len(labels), h, d)) + rng.normal(size=(h, k, d))[:, labels].transpose(1, 0, 2)
    moments = HeadMoments(h, d, k)
    order = rng.permutation(len(labels))
    for start in range(0, len(order), 13):
        chosen = order[start:start + 13]; moments.update(values[chosen], labels[chosen])
    raw_mean = np.stack([values[labels == role].mean(0) for role in range(k)], 1)
    raw_cov = np.zeros((h, d, d))
    for role in range(k):
        centered = values[labels == role] - raw_mean[:, role][None]
        raw_cov += np.einsum('nhd,nhe->hde', centered, centered)
    raw_cov /= len(labels) - k
    means = moments.sum / moments.n[:, :, None]
    covariance = (moments.xx - np.einsum('hkd,hke,hk->hde', moments.sum, moments.sum, 1. / moments.n)) / (len(labels) - k)
    require(np.allclose(means, raw_mean, atol=1e-12) and np.allclose(covariance, raw_cov, atol=1e-12), 'Streaming moments differ from raw.')
    weight, bias = moments.fit()
    shrink = .9 * raw_cov + .1 * np.trace(raw_cov, axis1=1, axis2=2)[:, None, None] / d * np.eye(d)
    raw_weight = np.linalg.solve(shrink, raw_mean.transpose(0, 2, 1)).transpose(0, 2, 1)
    prior = np.bincount(labels) / len(labels)
    raw_bias = -.5 * np.einsum('hkd,hkd->hk', raw_mean, raw_weight) + np.log(prior)
    query = rng.normal(size=(9, h, d))
    difference = float(np.abs(predict_heads(query, weight, bias) - predict_heads(query, raw_weight, raw_bias)).max())
    require(difference < 1e-12, 'Streaming LDA probabilities differ from raw.')
    probability = predict_heads(query, weight, bias); aggregate = probability[:, [0, 2]].mean(1)
    require(np.allclose(aggregate.sum(1), 1., atol=1e-14), 'Aggregate not normalized.')
    tied = np.array([.2, .2, .8, .8]); truth = np.array([0, 1, 0, 1], dtype=bool)
    auc = (rankdata(tied)[truth].sum() - 3) / 4
    require(auc == .5, 'AUROC tie handling failed.')
    # Independent padding fixture: original token positions0/2 in lengths3/5.
    padded = np.array([[0, 0, 10, 11, 12], [20, 21, 22, 23, 24]])
    positions = np.array([0 + 5 - 3, 2 + 5 - 3, 1 * 5 + 0, 1 * 5 + 2])
    require(np.array_equal(padded.reshape(-1)[positions], [10, 12, 20, 22]), 'Left-padding sample offsets failed.')
    return dict(cpu_only=True, heads=h, dimensions=d, classes=k,
        mean_max_difference=float(np.abs(means - raw_mean).max()), covariance_max_difference=float(np.abs(covariance - raw_cov).max()),
        probability_max_difference=difference, aggregate_normalized=True, auroc_average_ties=True, left_padding_indices_correct=True)


def cpu_hook_checks():
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(data.SEED + 3)
    raw = [torch.from_numpy(rng.normal(size=(2, 3, HEADS * DIM)).astype(np.float32)) for _ in range(2)]
    batch = dict(positions=np.array([1, 5], dtype=np.int64), targets=np.arange(2, dtype=np.int64),
        labels=np.array([0, 1], dtype=np.int64))
    projections = [torch.nn.Identity(), torch.nn.Identity()]
    moments = TorchMoments(); collector = Collector(moments=moments)
    collector.begin(batch)
    handle = projections[0].register_forward_pre_hook(collector.hook(0))
    with torch.inference_mode():
        collector.enabled = False
        untouched = projections[0](raw[0])
        require(torch.equal(untouched, raw[0]) and not moments.blocks, 'Disabled selfcheck accumulated or modified data.')
        collector.begin(batch); collector.enabled = True
        untouched = projections[0](raw[0])
        require(torch.equal(untouched, raw[0]), 'Enabled head hook modified its projection input.')
    handle.remove()
    selected_values = raw[0].reshape(-1, HEADS, DIM)[batch['positions']].double().numpy()
    reference = HeadMoments(HEADS, DIM, CLASSES); reference.update(selected_values, batch['labels'])
    difference = max(float(np.abs(actual.cpu().numpy() - expected).max())
        for actual, expected in zip(moments.blocks[0], [reference.n, reference.sum, reference.xx]))
    require(difference < 1e-10, 'Torch online accumulation differs from NumPy moments.')
    selected = np.array([1, HEADS + 3, 0, HEADS + 5, 5, HEADS, 9, HEADS + 7], dtype=np.int64)
    weight = rng.normal(size=(TOP_K, CLASSES, DIM)); bias = rng.normal(size=(TOP_K, CLASSES))
    probability_sum = np.zeros((2, CLASSES), dtype=np.float64)
    collector = Collector(weight=weight, bias=bias, probabilities=probability_sum, selected=selected)
    collector.begin(batch); collector.enabled = True
    handles = [projections[layer].register_forward_pre_hook(collector.hook(layer)) for layer in range(2)]
    with torch.inference_mode():
        for layer in range(2): require(torch.equal(projections[layer](raw[layer]), raw[layer]), 'Portable hook mutated input.')
    for handle in handles: handle.remove()
    expected_values = np.stack([raw[int(head // HEADS)].reshape(-1, HEADS, DIM)[batch['positions'], int(head % HEADS)].double().numpy()
                               for head in selected], axis=1)
    expected = predict_heads(expected_values, weight, bias).mean(1)
    aggregate_difference = float(np.abs(probability_sum / TOP_K - expected).max())
    require(aggregate_difference < 1e-10 and np.array_equal(collector.example, expected_values),
            'Multi-layer portable selected-head axes or aggregation differ.')
    return dict(cpu_only=True, selfcheck_statistics_disabled=True, hooks_leave_values_unchanged=True,
        torch_vs_numpy_moment_max_difference=difference, multi_layer_top8_aggregate_max_difference=aggregate_difference,
        unsorted_selected_head_axis_verified=True)


def cpu_degenerate_checks():
    from scipy.stats import rankdata
    labels = np.repeat(np.arange(5), 10)
    values = np.zeros((len(labels), 2, 4), dtype=np.float64)
    values[:, 1, 0] = labels
    moments = HeadMoments(2, 4, 5)
    for start in range(0, len(labels), 7): moments.update(values[start:start + 7], labels[start:start + 7])
    mean = np.stack([values[labels == role].mean(0) for role in range(5)], 1)
    within = np.zeros((2, 4, 4))
    for role in range(5):
        centered = values[labels == role] - mean[:, role][None]
        within += np.einsum('nhd,nhe->hde', centered, centered)
    within /= len(labels) - 5
    centered = values - values.mean(0, keepdims=True)
    total = np.einsum('nhd,nhe->hde', centered, centered) / (len(labels) - 1)
    tau = np.maximum.reduce([np.trace(within, axis1=1, axis2=2) / 4,
        1e-6 * np.trace(total, axis1=1, axis2=2) / 4, np.full(2, 1e-12)])
    raw_covariance = .9 * within + .1 * tau[:, None, None] * np.eye(4)
    raw_weight = np.linalg.solve(raw_covariance, mean.transpose(0, 2, 1)).transpose(0, 2, 1)
    raw_bias = -.5 * np.einsum('hkd,hkd->hk', mean, raw_weight) + np.log(.2)
    weight, bias, diagnostics = moments.fit(return_diagnostics=True)
    probability = predict_heads(values, weight, bias)
    expected = predict_heads(values, raw_weight, raw_bias)
    require(np.array_equal(mean, moments.sum / moments.n[:, :, None])
        and np.isfinite(probability).all() and np.allclose(probability.sum(-1), 1., atol=1e-14),
        'Degenerate fixture means/probabilities invalid.')
    require(np.allclose(probability[:, 0], .2) and np.array_equal(probability[:, 1].argmax(-1), labels),
        'All-zero prior head or class-constant role-perfect head was discarded or corrupted.')
    require(np.allclose(probability, expected, atol=1e-12) and diagnostics['zero_within_heads'] == 2
        and diagnostics['zero_total_heads'] == 1 and diagnostics['floor_applied_heads'] == 2,
        'Degenerate ridge formula or diagnostic counts differ from raw.')
    positive = labels == 0; n, m = int(positive.sum()), int((~positive).sum())
    auc = (rankdata(probability[:, 0, 0], method='average')[positive].sum() - n * (n + 1) / 2) / (n * m)
    require(auc == .5, 'All-zero head ties are not AUROC0.5.')
    return dict(synthetic_only=True, zero_head_prior_probabilities=True, zero_head_tied_auroc=auc,
        class_constant_role_perfect_head_retained=True, raw_probability_max_difference=float(np.abs(probability - expected).max()),
        diagnostics=diagnostics)


def preflight_checks():
    import torch
    threads = torch.get_num_threads()
    try:
        return dict(mathematical_checks=cpu_checks(), synthetic_hook_checks=cpu_hook_checks(),
            synthetic_degenerate_heads=cpu_degenerate_checks())
    finally:
        torch.set_num_threads(threads)


def preflight():
    checks = preflight_checks()
    tokenizer = long_data.load_tokenizer(); bundles = scan_data(tokenizer)
    source = source_signature()
    prefix = checkpoint_meta(bundles['fit'], 1, 'fit', plan_sha256='synthetic-only')
    require(prefix['rows_written'] == BATCH_SIZE * 256 and sum(prefix['class_counts']) == prefix['rows_written'],
            'Checkpoint prefix counts differ from completed source prompts.')
    report = dict(stage='cpu_preflight_only', **checks,
        partitions={name: bundle.descriptor() | dict(documents=int(bundle.rows.document_id.nunique()),
            tail_rows=int(bundle.rows.body_offset.ge(512).sum())) for name, bundle in bundles.items()},
        resources=resource_specification(), signature=source, formal_output_written=False,
        model_loaded=False, gpu_work_run=False, validation_inputs_loaded=False, tests_read=False)
    folder = Path(tempfile.mkdtemp(prefix='systemness-head-scan-preflight-', dir='/tmp'))
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    # Test moment persistence without touching a formal cache or claiming a real scan.
    tiny = HeadMoments(3, 4, 5); tiny.n[:] = prefix['class_counts']
    atomic_npz(folder / 'synthetic-moments.npz', metadata=np.array(json.dumps(prefix)), **tiny.arrays())
    loaded, saved = load_moments(folder / 'synthetic-moments.npz', 3, 4, 5)
    require(np.array_equal(loaded.n, tiny.n) and saved == prefix, 'Atomic complete snapshot/metadata fixture failed.')
    print(json.dumps(dict(report_path=str(folder / 'report.json'), **{key: value for key, value in report.items() if key != 'signature'})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='CPU math/source checks, temporary output only.')
    action.add_argument('--stage', choices=['scan', 'validate', 'all'], help='Explicitly run the requested four-A40 stage.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.stage: run(args.stage)
    else: parser.print_help()
