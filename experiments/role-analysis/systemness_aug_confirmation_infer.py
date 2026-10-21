"""Shared-head probability-only inference for conditional C4/natural tests.

Default help; --preflight is synthetic CPU /tmp only. --infer --dataset c4|natural
requires independently prepared builder data and a closed audited readout freeze.
Capture only the ranked eight pre-o_proj heads into a transient F16 matrix. Three
cached CPU MLPs consume that same matrix; the old-head's two records share one
probability file. No legacy capture, feature bank, calibration or report is run.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

import systemness_aug_confirm_snapshot as closed
import systemness_data as data

BATCH, INTERVAL, DIMENSIONS, PAD_ID = 4, 16, 512, 0
PHYSICAL = ['new', 'matched', 'old-head']
RECORD_TO_PHYSICAL = dict(closed.RECORD_TO_MODEL)
BUILDERS = {'c4': 'systemness_aug_confirmation_data', 'natural': 'systemness_aug_natural_data'}
require = closed.require
read = closed.read
digest = closed.digest


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def protocol():
    return dict(version=1, batch_size=BATCH, checkpoint_interval=INTERVAL, padding='Left zero padding with explicit attention mask.',
        capture='Only selected8 pre-o_proj query heads in frozen rank order; transient per-batch F16 matrix; no legacy hooks.',
        readouts='Three cached CPU pointwise portable MLP128s with original embedded512 scalers; same transient F16 vectors.',
        physical_models=PHYSICAL, record_to_physical=RECORD_TO_PHYSICAL,
        backend='Frozen native MXFP4 GPT-OSS-120B on exactly four A40; sequential shared forward.',
        portable='Actual same-vector CPU/GPU probability parity for all3 readouts on first processed batch, <5e-4.',
        no_features_persisted=True, no_model_or_threshold_selection=True, empty_natural_allowed=True)


def validate_probabilities(values, indices=None):
    for name, value in values.items():
        require(name in PHYSICAL and value.ndim == 2 and value.shape[1] == 5 and value.dtype == np.float32, 'Invalid probability geometry/dtype.')
        for start in range(0, len(value) if indices is None else len(indices), 65536):
            block = value[start:start + 65536] if indices is None else value[indices[start:start + 65536]]
            require(np.isfinite(block).all() and (block >= 0).all() and (block <= 1.00001).all()
                and np.allclose(block.sum(1), 1., atol=2e-5, rtol=0), 'Nonfinite or unnormalized probability rows.')


def records_from_physical(values):
    require(set(values) == set(PHYSICAL), 'Exactly three physical probability arrays are required.')
    return {record: values[physical] for record, physical in RECORD_TO_PHYSICAL.items()}


@dataclass
class Context:
    dataset: str
    module: object
    metadata: dict
    inputs: object
    rows: object
    tokens: list
    snapshot: dict
    paths: dict
    thresholds: dict
    heads: list
    out: Path
    cache: Path
    signature: dict
    bindings: dict


def dataset_module(dataset, module=None):
    require(dataset in BUILDERS, 'Unknown confirmation dataset.')
    return module if module is not None else importlib.import_module(BUILDERS[dataset])


def load_context(dataset, module=None):
    module = dataset_module(dataset, module)
    metadata, inputs, rows, tokens = module.load_verified_data()
    snapshot = metadata['frozen']; paths, thresholds, heads = closed.verify_closed_snapshot(snapshot)
    runtime_files = snapshot['runtime_files_sha256']
    for source in [Path(__file__).resolve(), Path(module.__file__).resolve(), Path(closed.__file__).resolve()]:
        require(runtime_files.get(str(source)) == digest(source), 'Inference/builder/helper source was not frozen before test preparation: ' + str(source))
    require(metadata['frozen_sha256'] == snapshot['snapshot_sha256'] and digest(metadata['snapshot_path']) == metadata['snapshot_file_sha256']
        and read(metadata['snapshot_path']) == snapshot, 'Builder readout statement changed.')
    coverage = metadata['coverage']
    require(coverage['all_content_tokens_scored'] is True and coverage['independent_native_labels_match'] is True,
        'All-content native role parser evidence is missing.')
    require(np.array_equal(inputs.prompt_ix.to_numpy(), np.arange(len(inputs))) and len(tokens) == len(inputs)
        and np.array_equal(rows.cache_ix.to_numpy(), np.arange(len(rows)))
        and np.array_equal(rows.index.to_numpy(), np.arange(len(rows))) and rows.sample_ix.is_unique
        and set(rows.role) <= set(closed.ROLES) and set(rows.split) <= {'test'} and set(rows.wrapper) <= {'native'},
        'Prepared test input/row identity, split or role coverage changed.')
    require(set(rows.prompt_ix) == set(inputs.prompt_ix) and bool(len(inputs)) == bool(len(rows)),
        'A prepared source/input has no scored content or rows have no input.')
    if dataset == 'natural': require(set(rows.role) <= {'system', 'user', 'assistant'}, 'Natural truth must remain S/U/A only.')
    for prompt, part in rows.groupby('prompt_ix', sort=False):
        ids = np.asarray(tokens[int(prompt)], np.int64); positions = part.token_ix.to_numpy(dtype=np.int64)
        require(len(ids) == int(inputs.iloc[int(prompt)].length) and len(np.unique(positions)) == len(positions)
            and (positions >= 0).all() and (positions < len(ids)).all()
            and np.array_equal(ids[positions], part.token_id.to_numpy(dtype=np.int64)), 'Content token identities/positions changed.')
    bindings = {str(Path(module.OUT) / 'data-complete.json'): digest(Path(module.OUT) / 'data-complete.json'),
        str(Path(metadata['snapshot_path'])): metadata['snapshot_file_sha256']}
    for name, expected in metadata['cache_sha256'].items():
        path = Path(module.CACHE) / name; require(digest(path) == expected, 'Builder cache SHA differs: ' + name); bindings[str(path)] = expected
    for name, expected in metadata['dependencies_sha256'].items():
        path = data.HERE / name; require(digest(path) == expected, 'Builder code dependency changed.'); bindings[str(path)] = expected
    require(metadata['source_sha256'] == digest(module.__file__), 'Builder source SHA differs.')
    bindings[str(Path(__file__).resolve())] = digest(__file__)
    signature = dict(source_sha256=digest(__file__), builder_source_sha256=digest(module.__file__), dataset=dataset,
        protocol=protocol(), data_complete_sha256=bindings[str(Path(module.OUT) / 'data-complete.json')],
        snapshot_sha256=snapshot['snapshot_sha256'], snapshot_file_sha256=metadata['snapshot_file_sha256'],
        model_sha256=snapshot['physical_model_sha256'], record_thresholds=thresholds, selected_head_ids=heads,
        cache_sha256=metadata['cache_sha256'], cache_ix_sha256=array_sha(rows.cache_ix.to_numpy()),
        sample_ix_sha256=array_sha(rows.sample_ix.to_numpy()), rows=len(rows), inputs=len(inputs),
        checkpoint_path=snapshot['checkpoint_path'], package_versions={name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'pandas']})
    return Context(dataset, module, metadata, inputs, rows, tokens, snapshot, paths, thresholds, heads,
        Path(module.OUT) / 'inference', Path(module.CACHE) / 'inference', signature, bindings)


def fast_integrity(ctx):
    closed.verify_closed_snapshot(ctx.snapshot)
    for path, expected in ctx.bindings.items(): require(digest(path) == expected, 'Closed source/input cache changed: ' + path)


class Collector:
    """Only pre-o_proj selected heads. No hook changes any model input."""
    def __init__(self, selected):
        self.selected = np.asarray(selected, np.int64)
        require(self.selected.shape == (8,) and len(set(self.selected)) == 8 and ((self.selected >= 0) & (self.selected < 2304)).all(), 'Invalid eight-head rank list.')
        self.layers = sorted(set((self.selected // 64).tolist())); self.features = None; self.enabled = False

    def begin(self, positions, enabled=True):
        self.positions = np.asarray(positions, np.int64)
        require(self.positions.ndim == 1 and len(np.unique(self.positions)) == len(self.positions) and (self.positions >= 0).all(), 'Duplicate/negative sampled content positions.')
        self.enabled = enabled; self.seen = set(); self.indices = {}
        self.features = np.full((len(self.positions), DIMENSIONS), np.nan, np.float16) if enabled else None

    def hook(self, layer):
        def receive(module, args):
            import torch
            value = args[0]
            require(layer not in self.seen and value.ndim == 3 and value.shape[-1] == 4096, 'Repeated/malformed selected head prehook.')
            self.seen.add(layer)
            if not self.enabled: return None
            key = str(value.device)
            if key not in self.indices: self.indices[key] = torch.as_tensor(self.positions, dtype=torch.long, device=value.device)
            ranks = np.flatnonzero(self.selected // 64 == layer)
            heads = torch.as_tensor(self.selected[ranks] % 64, dtype=torch.long, device=value.device)
            sample = value.detach().reshape(-1, 64, 64).index_select(0, self.indices[key]).index_select(1, heads)
            require(bool(torch.isfinite(sample).all()) and bool((sample.float().abs() <= np.finfo(np.float16).max).all()), 'Selected head feature outside finite F16 range; clipping forbidden.')
            block = sample.to('cpu', torch.float16).numpy().reshape(len(self.positions), -1)
            columns = (ranks[:, None] * 64 + np.arange(64)).reshape(-1); self.features[:, columns] = block
            return None
        return receive

    def finish(self):
        require(self.seen == set(self.layers), 'A selected-head prehook was missed.')
        if self.enabled: require(np.isfinite(self.features).all(), 'Unwritten/nonfinite selected feature column.')

    def release(self):
        self.features = None; self.indices = {}


def install_hooks(model, collector):
    return [model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)) for layer in collector.layers]


def batches(ctx):
    order = ctx.inputs.sort_values(['length', 'prompt_ix']).prompt_ix.to_numpy(dtype=np.int64)
    return [order[i:i + BATCH].tolist() for i in range(0, len(order), BATCH)]


def collate(ctx, prompts):
    import torch
    width = max(len(ctx.tokens[p]) for p in prompts)
    ids = torch.full((len(prompts), width), PAD_ID, dtype=torch.long); mask = torch.zeros_like(ids)
    positions, targets = [], []
    lookup = getattr(ctx, '_lookup', None)
    if lookup is None:
        lookup = {int(p): g for p, g in ctx.rows.groupby('prompt_ix', sort=False)}; ctx._lookup = lookup
    for batch_ix, prompt in enumerate(prompts):
        tokens = ctx.tokens[prompt]; offset = width - len(tokens)
        ids[batch_ix, offset:] = torch.as_tensor(tokens, dtype=torch.long); mask[batch_ix, offset:] = 1
        group = lookup[prompt]; positions.extend((group.token_ix.to_numpy() + offset + batch_ix * width).tolist()); targets.extend(group.index.tolist())
    require(len(set(positions)) == len(positions) and len(set(targets)) == len(targets), 'Duplicate content targets in shared forward.')
    return dict(input_ids=ids, attention_mask=mask, positions=np.asarray(positions, np.int64), targets=np.asarray(targets, np.int64))


def completed_indices(ctx, next_batch):
    parts = batches(ctx); require(type(next_batch) is int and 0 <= next_batch <= len(parts), 'Invalid saved batch cursor.')
    prompts = [p for group in parts[:next_batch] for p in group]
    return np.flatnonzero(ctx.rows.prompt_ix.isin(prompts).to_numpy())


def probability_hashes(values, indices):
    output = {}
    for name in PHYSICAL:
        h = hashlib.sha256()
        for start in range(0, len(indices), 65536): h.update(np.ascontiguousarray(values[name][indices[start:start + 65536]]).tobytes())
        output[name] = h.hexdigest()
    return output


def progress_record(ctx, values, next_batch):
    indices = completed_indices(ctx, next_batch); validate_probabilities(values, indices)
    return dict(signature=ctx.signature, next_batch=next_batch, rows_written=len(indices),
        sample_ix_sha256=array_sha(ctx.rows.iloc[indices].sample_ix.to_numpy()),
        cache_ix_sha256=array_sha(ctx.rows.iloc[indices].cache_ix.to_numpy()),
        completed_probability_sha256=probability_hashes(values, indices))


def initialize_probabilities(ctx, *, readonly=False):
    require(len(ctx.rows) > 0, 'Empty datasets have no probability memmap.')
    expected = dict(signature=ctx.signature, models=PHYSICAL, rows=len(ctx.rows), classes=5, dtype='float32')
    owner = ctx.cache / 'initialized.json'; paths = {name: ctx.cache / (name + '-probabilities.f32') for name in PHYSICAL}
    if owner.exists():
        require(read(owner) == expected and (ctx.cache / 'progress.json').is_file(), 'Probability ownership or committed progress is missing/changed.')
    else:
        require(not readonly and not (ctx.cache / 'progress.json').exists() and not any(p.exists() for p in paths.values()), 'Unowned/partial probability cache; refusing silent reuse.')
        ctx.cache.mkdir(parents=True, exist_ok=True)
        for path in paths.values():
            array = np.memmap(path, mode='w+', dtype=np.float32, shape=(len(ctx.rows), 5)); array[:] = np.nan; array.flush(); del array
            with path.open('rb') as stream: os.fsync(stream.fileno())
        atomic_json(owner, expected)
        initial = {name: np.memmap(path, mode='r+', dtype=np.float32, shape=(len(ctx.rows), 5)) for name, path in paths.items()}
        atomic_json(ctx.cache / 'progress.json', progress_record(ctx, initial, 0)); del initial
    require(all(p.is_file() and p.stat().st_size == len(ctx.rows) * 20 for p in paths.values()), 'Saved probability file is missing/truncated; refusing old progress.')
    return {name: np.memmap(path, mode='r' if readonly else 'r+', dtype=np.float32, shape=(len(ctx.rows), 5)) for name, path in paths.items()}


def restore_progress(ctx, values):
    saved = read(ctx.cache / 'progress.json'); expected = progress_record(ctx, values, saved['next_batch'])
    require(saved == expected, 'Committed probability/sample/cache prefix changed; refusing resume.')
    return saved['next_batch']


def checkpoint(ctx, values, next_batch):
    for name, value in values.items():
        value.flush()
        with (ctx.cache / (name + '-probabilities.f32')).open('rb') as stream: os.fsync(stream.fileno())
    atomic_json(ctx.cache / 'progress.json', progress_record(ctx, values, next_batch))


def load_portables(ctx):
    import torch
    from torch import nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(.1)); self.head = nn.Linear(128, 5)
        def forward(self, value): return torch.log_softmax(self.head(self.encoder(value)), dim=1)
    output = {}
    for name in PHYSICAL:
        with np.load(ctx.paths[name], allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'])); require(meta['selected_head_ids'] == ctx.heads and meta['roles'] == closed.ROLES, 'Portable feature/role order differs.')
            net = Net(); net.load_state_dict({key[8:]: torch.from_numpy(z[key].copy()) for key in z.files if key.startswith('weight::')})
            mean = torch.from_numpy(z['mean'].copy()); scale = torch.from_numpy(z['scale'].copy())
        require(mean.shape == scale.shape == (512,) and torch.isfinite(mean).all() and torch.isfinite(scale).all() and bool((scale > 0).all()), 'Invalid portable scaler.')
        output[name] = (net.eval().requires_grad_(False), mean, scale)
    return output


def cpu_probabilities(nets, features):
    import torch
    require(features.dtype == np.float16 and features.ndim == 2 and features.shape[1] == DIMENSIONS and np.isfinite(features).all(), 'Transient input must be the exact finite F16 head matrix.')
    result = {}
    with torch.inference_mode():
        for name, (net, mean, scale) in nets.items():
            parts = []
            for start in range(0, len(features), 4096):
                raw = torch.from_numpy(features[start:start + 4096]).float(); parts.append(net((raw - mean) / scale).exp().numpy().astype(np.float32))
            result[name] = np.concatenate(parts) if parts else np.empty((0, 5), np.float32)
    validate_probabilities(result); return result


def same_vector_gpu_parity(ctx, nets, features, cpu):
    import copy
    import torch
    count = min(32, len(features)); require(count > 0, 'No actual sampled vectors for portable parity.')
    raw = torch.from_numpy(features[:count]).to('cuda:0').float(); differences = {}
    with torch.inference_mode():
        for name, (net, mean, scale) in nets.items():
            gpu = copy.deepcopy(net).to('cuda:0').eval()
            value = gpu((raw - mean.cuda(0)) / scale.cuda(0)).exp().cpu().numpy()
            error = float(np.abs(value - cpu[name][:count]).max()); require(np.isfinite(error) and 0 <= error < 5e-4, 'Same-vector CPU/GPU portable parity failed: ' + name)
            differences[name] = error; del gpu
    return dict(signature=ctx.signature, model_sha256=ctx.snapshot['physical_model_sha256'], sampled_tokens=count,
        max_probability_difference=differences, selected_head_ids=ctx.heads, captured_dtype='float16',
        same_vectors=True, CPU_outputs_are_saved=True, no_features_persisted=True)


def no_competing_jobs():
    names = {'systemness_aug_confirmation_infer.py', 'systemness_aug_confirmation_data.py', 'systemness_aug_natural_data.py',
        'systemness_long_augmentation_extract.py', 'systemness_long_augmentation_probes.py', 'systemness_head_extract.py',
        'systemness_head_scan.py', 'systemness_head_probes.py', 'systemness_head_confirm.py', 'systemness_natural_test_confirm.py',
        'systemness_long_confirm.py', 'systemness_long_extract.py', 'systemness_recover_local_bank.py', 'systemness_head_fusion.py'}
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg): active.append(int(entry.name))
    require(not active, 'Another extraction/training/confirmation process is live: ' + str(active))


@contextmanager
def run_lock(ctx):
    import fcntl
    with (ctx.out / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('This inference namespace is locked by another process.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def record_identity(physical_sha, argmax_sha):
    return dict(record_probability_sha256={name: physical_sha[physical] for name, physical in RECORD_TO_PHYSICAL.items()},
        record_argmax_sha256={name: argmax_sha[physical] for name, physical in RECORD_TO_PHYSICAL.items()},
        record_to_physical=RECORD_TO_PHYSICAL, old_head_two_records_share_probability_and_argmax=True)


def argmax_hashes(values):
    output = {}
    for name, value in values.items():
        h = hashlib.sha256()
        for start in range(0, len(value), 65536): h.update(value[start:start + 65536].argmax(1).astype(np.int64).tobytes())
        output[name] = h.hexdigest()
    return output


def empty_completion(ctx):
    require(ctx.dataset == 'natural' and not len(ctx.inputs) and not len(ctx.rows), 'Only all-failed natural data may have no observations.')
    require(not any((ctx.cache / name).exists() for name in ['initialized.json', 'progress.json'] + [p + '-probabilities.f32' for p in PHYSICAL]), 'A no-observation inference cannot reuse nonempty cache files.')
    empty = {name: None for name in PHYSICAL}
    complete = dict(signature=ctx.signature, rows=0, inputs=0, no_observations=True, gpu_run=False,
        probability_sha256=empty, argmax_sha256=empty, **record_identity(empty, empty),
        all_finite=True, all_normalized=True, no_features_persisted=True, model_forward_run=False,
        portable_validation_sha256=None, hook_validation_sha256=None, model_backend_sha256=None, progress_sha256=None)
    atomic_json(ctx.out / 'inference-complete.json', complete)
    atomic_json(ctx.out / 'status.json', dict(stage='inference_complete', rows=0, no_observations=True, gpu_run=False))
    return complete


def completed_probabilities(dataset, module=None, *, context=None):
    # Terminal marker is checked before any probability mmap or builder-cache read.
    if context is None:
        module = dataset_module(dataset, module); out = Path(module.OUT) / 'inference'
        require((out / 'inference-complete.json').is_file() and (out / 'status.json').is_file(), 'Inference is not complete; partial probability reads are forbidden.')
        require(read(out / 'status.json')['stage'] == 'inference_complete', 'Inference status is not terminal.')
        ctx = load_context(dataset, module)
    else: ctx = context
    require((ctx.out / 'inference-complete.json').is_file() and read(ctx.out / 'status.json')['stage'] == 'inference_complete', 'Inference is not terminal complete.')
    complete = read(ctx.out / 'inference-complete.json'); require(complete['signature'] == ctx.signature and complete['rows'] == len(ctx.rows)
        and complete['inputs'] == len(ctx.inputs) and complete['all_finite'] is True and complete['all_normalized'] is True
        and complete['no_features_persisted'] is True, 'Completed probability signature/coverage differs.')
    require(set(complete['probability_sha256']) == set(complete['argmax_sha256']) == set(PHYSICAL), 'Three-model completed cache coverage differs.')
    if not len(ctx.rows):
        require(ctx.dataset == 'natural' and not len(ctx.inputs) and complete['no_observations'] is True and complete['gpu_run'] is False
            and complete['model_forward_run'] is False and complete['probability_sha256'] == complete['argmax_sha256'] == {name: None for name in PHYSICAL},
            'Empty inference invented model observations.')
        values = {name: np.empty((0, 5), np.float32) for name in PHYSICAL}
    else:
        require(complete['no_observations'] is False and complete['gpu_run'] is True and complete['model_forward_run'] is True, 'Nonempty inference backend proof missing.')
        for name, key in [('portable-validation.json', 'portable_validation_sha256'), ('hook-validation.json', 'hook_validation_sha256'), ('model_backend.json', 'model_backend_sha256')]:
            require(digest(ctx.out / name) == complete[key], 'Actual inference evidence SHA changed: ' + name)
        require(digest(ctx.cache / 'progress.json') == complete['progress_sha256'], 'Committed inference progress SHA differs.')
        portable, hooks, backend = read(ctx.out / 'portable-validation.json'), read(ctx.out / 'hook-validation.json'), read(ctx.out / 'model_backend.json')
        require(portable['signature'] == hooks['signature'] == ctx.signature and portable['model_sha256'] == ctx.snapshot['physical_model_sha256']
            and set(portable['max_probability_difference']) == set(PHYSICAL) and 0 < portable['sampled_tokens'] <= 32
            and all(np.isfinite(v) and 0 <= v < 5e-4 for v in portable['max_probability_difference'].values())
            and portable['same_vectors'] is True and portable['captured_dtype'] == 'float16', 'Same-vector portable proof differs.')
        require(hooks['hidden_states_bitexact'] is True and hooks['all_capture_writes_disabled'] is True
            and hooks['only_pre_o_proj_hooks'] is True and hooks['legacy_capture'] is False and hooks['selected_head_ids'] == ctx.heads,
            'Disabled selected-head hooks changed hidden states or captured legacy activations.')
        require(backend['expert_backends'] == ['Mxfp4GptOssExperts'] and backend['load_path'] == ctx.snapshot['checkpoint_path']
            and len(backend['gpus']) == 4 and all('A40' in name for name in backend['gpus']), 'Actual native MXFP4 four-A40 backend differs.')
        values = initialize_probabilities(ctx, readonly=True)
        for name in PHYSICAL: require(digest(ctx.cache / (name + '-probabilities.f32')) == complete['probability_sha256'][name], 'Completed probability SHA differs: ' + name)
        validate_probabilities(values); require(restore_progress(ctx, values) == len(batches(ctx)), 'Probability prefix is incomplete.')
        require(argmax_hashes(values) == complete['argmax_sha256'], 'Saved argmax provenance differs.')
    identity = record_identity(complete['probability_sha256'], complete['argmax_sha256'])
    require(all(complete[k] == v for k, v in identity.items()), 'Old-head two-record probability/argmax identity was broken.')
    fast_integrity(ctx)
    return complete, values


def infer(dataset, module=None):
    no_competing_jobs(); ctx = load_context(dataset, module)
    ctx.out.mkdir(parents=True, exist_ok=True)
    with run_lock(ctx):
        if (ctx.out / 'inference-complete.json').exists(): completed_probabilities(dataset, context=ctx); return
        if not len(ctx.rows): fast_integrity(ctx); empty_completion(ctx); return
        values = initialize_probabilities(ctx); start = restore_progress(ctx, values)
        if start:
            path = ctx.out / 'portable-validation.json'; require(path.is_file(), 'Committed probability progress lacks portable evidence.')
            proof = read(path); require(proof['signature'] == ctx.signature
                and proof['model_sha256'] == ctx.snapshot['physical_model_sha256']
                and set(proof['max_probability_difference']) == set(PHYSICAL)
                and all(np.isfinite(v) and 0 <= v < 5e-4 for v in proof['max_probability_difference'].values()), 'Saved portable prefix evidence changed.')
        import torch
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Native shared forward requires exactly four visible A40s.')
        require(all(torch.cuda.mem_get_info(i)[0] > 20 * 2**30 for i in range(4)), 'Insufficient free memory on one of four A40s.')
        os.environ['ROLE_ANALYSIS_LOAD_PATH'] = ctx.snapshot['checkpoint_path']
        if os.environ.get('HF_HOME', '').startswith('/workspace'): os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
        nets = load_portables(ctx); collector = Collector(ctx.heads); handles = []; runtime = data.runtime
        previous_atomic, previous_status = runtime.atomic_json, runtime.status
        runtime.atomic_json = lambda path, value: atomic_json(ctx.out / Path(path).name, value)
        runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
        started = time.time(); parity_checked = False; model = None
        try:
            model = runtime.load_model(); require(not any(p.requires_grad for p in model.parameters()), 'Language model parameters are not frozen.')
            parts = batches(ctx)
            with torch.inference_mode():
                sample = collate(ctx, parts[0]); kwargs = dict(input_ids=sample['input_ids'].cuda(0), attention_mask=sample['attention_mask'].cuda(0), use_cache=False)
                before = model.model(**kwargs).last_hidden_state.cpu(); handles = install_hooks(model, collector)
                collector.begin(sample['positions'], enabled=False); after = model.model(**kwargs).last_hidden_state.cpu(); collector.finish()
                require(torch.equal(before, after) and bool(torch.isfinite(after).all()) and collector.features is None, 'Disabled selected-head capture altered hidden states or wrote features.')
                atomic_json(ctx.out / 'hook-validation.json', dict(signature=ctx.signature, hidden_states_bitexact=True,
                    all_capture_writes_disabled=True, selected_head_ids=ctx.heads, only_pre_o_proj_hooks=True, legacy_capture=False))
                del before, after, kwargs, sample
                for bi in range(start, len(parts)):
                    batch = collate(ctx, parts[bi]); collector.begin(batch['positions'])
                    hidden = model.model(input_ids=batch['input_ids'].cuda(0), attention_mask=batch['attention_mask'].cuda(0), use_cache=False).last_hidden_state
                    del hidden; collector.finish(); probabilities = cpu_probabilities(nets, collector.features)
                    if not parity_checked:
                        proof = same_vector_gpu_parity(ctx, nets, collector.features, probabilities)
                        path = ctx.out / 'portable-validation.json'
                        if path.exists():
                            old = read(path); require(old['signature'] == ctx.signature and old['model_sha256'] == ctx.snapshot['physical_model_sha256']
                                and all(np.isfinite(v) and 0 <= v < 5e-4 for v in old['max_probability_difference'].values()), 'Existing portable proof is not valid for resume.')
                        else: atomic_json(path, proof)
                        parity_checked = True
                    for name in PHYSICAL: values[name][batch['targets']] = probabilities[name]
                    collector.release(); del probabilities
                    if (bi + 1) % INTERVAL == 0 or bi + 1 == len(parts):
                        fast_integrity(ctx); checkpoint(ctx, values, bi + 1)
                        status = dict(stage='inference', batch=bi + 1, batches=len(parts), rows=progress_record(ctx, values, bi + 1)['rows_written'], seconds=time.time() - started)
                        atomic_json(ctx.out / 'status.json', status); print(json.dumps(status), flush=True)
            fast_integrity(ctx); validate_probabilities(values)
            require(restore_progress(ctx, values) == len(parts), 'Final committed content-token coverage incomplete.')
            physical_sha = {name: digest(ctx.cache / (name + '-probabilities.f32')) for name in PHYSICAL}; argmax_sha = argmax_hashes(values)
            complete = dict(signature=ctx.signature, rows=len(ctx.rows), inputs=len(ctx.inputs), probability_sha256=physical_sha,
                argmax_sha256=argmax_sha, **record_identity(physical_sha, argmax_sha), no_observations=False, gpu_run=True,
                model_forward_run=True, all_finite=True, all_normalized=True, no_features_persisted=True,
                portable_validation_sha256=digest(ctx.out / 'portable-validation.json'), hook_validation_sha256=digest(ctx.out / 'hook-validation.json'),
                model_backend_sha256=digest(ctx.out / 'model_backend.json'), progress_sha256=digest(ctx.cache / 'progress.json'), seconds=time.time() - started)
            atomic_json(ctx.out / 'inference-complete.json', complete); atomic_json(ctx.out / 'status.json', dict(stage='inference_complete', rows=len(ctx.rows), gpu_run=True))
            completed_probabilities(dataset, context=ctx)
        except BaseException:
            if not (ctx.out / 'inference-complete.json').exists(): atomic_json(ctx.out / 'status.json', dict(stage='failed', next_batch=read(ctx.cache / 'progress.json')['next_batch']))
            raise
        finally:
            for handle in handles: handle.remove()
            collector.release(); runtime.atomic_json, runtime.status = previous_atomic, previous_status
            if model is not None: del model
            gc.collect(); torch.cuda.empty_cache()


def preflight():
    """Synthetic CPU hooks, portable weights and /tmp resume faults only."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import copy
    import py_compile
    import torch
    from torch import nn
    torch.set_num_threads(2)
    folder = Path(tempfile.mkdtemp(prefix='systemness-aug-infer-preflight-', dir='/tmp')); source_sha = digest(__file__); checks = []
    def check(name, value=True): require(bool(value), 'CPU preflight failed: ' + name); checks.append(name)
    def reject(name, fn):
        try: fn()
        except (RuntimeError, ValueError, KeyError, FileNotFoundError): checks.append(name)
        else: raise RuntimeError('CPU fault accepted: ' + name)
    def forbidden(*a, **kw): raise AssertionError('A real builder/model/snapshot API was called by CPU fixture.')
    selected = [455, 0, 325, 65, 390, 130, 260, 195]
    class Attention(nn.Module):
        def __init__(self): super().__init__(); self.o_proj = nn.Identity()
        def forward(self, value): return self.o_proj(value)
    class Layer(nn.Module):
        def __init__(self, ix): super().__init__(); self.self_attn = Attention(); self.ix = ix
        def forward(self, value): return self.self_attn(value) + self.ix / 1000
    layers = nn.ModuleList([Layer(i) for i in range(36)])
    fake = SimpleNamespace(model=SimpleNamespace(layers=layers)); raw = torch.arange(2 * 5 * 4096).reshape(2, 5, 4096).float() / 1000
    def forward():
        x = raw.clone()
        for layer in layers: x = layer(x)
        return x
    baseline = forward(); collector = Collector(selected); positions = np.array([0, 4, 7, 9]); handles = install_hooks(fake, collector)
    try:
        collector.begin(positions, enabled=False); disabled = forward(); collector.finish()
        check('Onlyrankedheads hooks disabledwrites preserve hidden bitexact', torch.equal(baseline, disabled) and collector.features is None and len(handles) == len(collector.layers))
        collector.begin(positions); enabled = forward(); collector.finish()
        x = raw.clone(); expected = {}
        for i in range(36): expected[i] = x.clone(); x = x + i / 1000
        direct = np.concatenate([expected[h // 64].reshape(-1, 64, 64)[positions, h % 64].half().numpy() for h in selected], axis=1)
        check('Unsorted8rank512 F16 columns and sampledpositions exact', np.array_equal(collector.features, direct) and torch.equal(enabled, baseline))
        reject('Reject duplicate selectedlayer activation', lambda: layers[collector.layers[0]].self_attn(raw))
        collector.begin(positions); reject('Reject missing selectedhead hook', collector.finish)
        reject('Reject duplicate content positions', lambda: collector.begin([0, 0]))
    finally:
        for handle in handles: handle.remove()
        collector.release()
    # Small local portable witnesses: no real model/readout files are opened.
    paths = {}; rng = np.random.default_rng(20261021)
    mean = rng.normal(size=512).astype(np.float32); scale = rng.uniform(.3, 2, size=512).astype(np.float32)
    for index, name in enumerate(PHYSICAL):
        path = folder / (name + '.npz'); paths[name] = path
        arrays = {'weight::encoder.0.weight': rng.normal(0, .02, (128, 512)).astype(np.float32),
            'weight::encoder.0.bias': np.zeros(128, np.float32), 'weight::head.weight': rng.normal(0, .02, (5, 128)).astype(np.float32),
            'weight::head.bias': np.zeros(5, np.float32)}
        np.savez_compressed(path, metadata=np.array(json.dumps(dict(roles=closed.ROLES, selected_head_ids=selected))), mean=mean, scale=scale, **arrays)
    frames = []; inputs = []; tokens = []
    for prompt, length in enumerate([5, 8, 6, 9, 7, 10]):
        tokens.append(list(range(100 + prompt * 20, 100 + prompt * 20 + length)))
        inputs.append(dict(prompt_ix=prompt, length=length))
        for ix in range(1, length - 1): frames.append(dict(cache_ix=len(frames), sample_ix=7_000_000_000_000 + len(frames), prompt_ix=prompt,
            token_ix=ix, token_id=tokens[-1][ix], role=closed.ROLES[ix % 5], wrapper='native', split='test'))
    inputs, rows = pd.DataFrame(inputs), pd.DataFrame(frames)
    module_source = folder / 'builder.py'; module_source.write_text('# Synthetic builder only.\n')
    out, cache = folder / 'builder-output', folder / 'builder-cache'; out.mkdir(); cache.mkdir()
    for name in ['inputs.parquet', 'rows.parquet', 'token-ids.json']: (cache / name).write_text('Synthetic cache identity, never real text.')
    frozen = dict(snapshot_sha256='f' * 64, physical_model_sha256={name: digest(path) for name, path in paths.items()},
        checkpoint_path='/synthetic/no-model', runtime_files_sha256={str(Path(p).resolve()): digest(p) for p in [__file__, module_source, closed.__file__]})
    statement = folder / 'readout-snapshot.json'; atomic_json(statement, frozen)
    metadata = dict(frozen=frozen, frozen_sha256=frozen['snapshot_sha256'], snapshot_path=str(statement), snapshot_file_sha256=digest(statement),
        coverage=dict(all_content_tokens_scored=True, independent_native_labels_match=True),
        cache_sha256={p.name: digest(p) for p in cache.iterdir()}, dependencies_sha256={}, source_sha256=digest(module_source))
    atomic_json(out / 'data-complete.json', metadata)
    module = SimpleNamespace(OUT=out, CACHE=cache, __file__=str(module_source), load_verified_data=lambda: (metadata, inputs, rows, tokens))
    thresholds = {record: .5 for record in RECORD_TO_PHYSICAL}
    with patch.object(closed, 'verify_closed_snapshot', return_value=(paths, thresholds, selected)), patch.object(data.runtime, 'load_model', side_effect=forbidden):
        ctx = load_context('c4', module)
        b = collate(ctx, [0, 1]); require(b['positions'].tolist() == [4, 5, 6, 9, 10, 11, 12, 13, 14], 'Left-padding fixture positions changed.')
        check('Leftpadding plus targetcache indices account for eachprompt length', b['targets'].tolist() == rows[rows.prompt_ix.isin([0, 1])].index.tolist())
        check('Inputs sorted deterministically length thenprompt', batches(ctx) == [[0, 2, 4, 1], [3, 5]])
        nets = load_portables(ctx); feature = rng.normal(size=(len(rows), 512)).astype(np.float16); probabilities = cpu_probabilities(nets, feature)
        maxdiff = 0.
        with torch.inference_mode():
            for name in PHYSICAL:
                with np.load(paths[name], allow_pickle=False) as z:
                    x = torch.from_numpy((feature.astype(np.float32) - mean) / scale)
                    hidden = torch.nn.functional.gelu(torch.nn.functional.linear(x, torch.from_numpy(z['weight::encoder.0.weight']), torch.from_numpy(z['weight::encoder.0.bias'])))
                    value = torch.log_softmax(torch.nn.functional.linear(hidden, torch.from_numpy(z['weight::head.weight']), torch.from_numpy(z['weight::head.bias'])), dim=1).exp().numpy()
                maxdiff = max(maxdiff, float(np.abs(value - probabilities[name]).max()))
        check('Cached3CPU portable MLPs match independent linearGELU reference', maxdiff < 1e-7)
        record_p = records_from_physical(probabilities)
        check('Oldtwo records share same physicalp object and identicalargmax', record_p['old-head-maximin'] is record_p['old-head-original']
            and np.array_equal(record_p['old-head-maximin'].argmax(1), record_p['old-head-original'].argmax(1)))
        ctx.out.mkdir(); values = initialize_probabilities(ctx)
        check('NaN3file initialization owns validzero prefix', all(np.isnan(p).all() for p in values.values()) and restore_progress(ctx, values) == 0)
        prefix = completed_indices(ctx, 1)
        for name in PHYSICAL: values[name][prefix] = probabilities[name][prefix]
        checkpoint(ctx, values, 1)
        check('Committed prefix restore retains exactsample/cache/probability SHA', restore_progress(ctx, values) == 1)
        saved = values['new'][prefix[0]].copy(); values['new'][prefix[0]] = np.roll(saved, 1)
        reject('Reject normalized but changed committedprobability prefix', lambda: restore_progress(ctx, values)); values['new'][prefix[0]] = saved
        del values; gc.collect()
        path = ctx.cache / 'new-probabilities.f32'; payload = path.read_bytes(); path.write_bytes(payload[:-4])
        reject('Reject truncated probability along old progress', lambda: initialize_probabilities(ctx, readonly=True)); path.write_bytes(payload)
        parked = ctx.cache / 'held.f32'; path.rename(parked)
        reject('Reject missing probability along old progress', lambda: initialize_probabilities(ctx, readonly=True)); parked.rename(path)
        progress = ctx.cache / 'progress.json'; parked = ctx.cache / 'held-progress.json'; progress.rename(parked)
        reject('Reject ownership without committedprogress file', lambda: initialize_probabilities(ctx, readonly=True)); parked.rename(progress)
        # Complete APIs cannot consult builder/probs before terminal marker.
        previous = module.load_verified_data; module.load_verified_data = forbidden
        reject('Terminalcomplete gate precedes anybuilder/probability read', lambda: completed_probabilities('c4', module)); module.load_verified_data = previous
        bad = {name: p.copy() for name, p in probabilities.items()}; bad['new'][0, 0] = np.nan
        reject('Reject nonfinite saved probabilities', lambda: validate_probabilities(bad))
        bad = {name: p.copy() for name, p in probabilities.items()}; bad['new'][0] *= .5
        reject('Reject unnormalized saved probabilities', lambda: validate_probabilities(bad))
        empty_rows = rows.iloc[:0].copy(); empty_inputs = inputs.iloc[:0].copy()
        empty = Context('natural', module, metadata, empty_inputs, empty_rows, [], frozen, paths, thresholds, selected,
            folder / 'empty-out', folder / 'empty-cache', dict(synthetic=True, rows=0), {})
        empty.out.mkdir(); empty_completion(empty); complete, ep = completed_probabilities('natural', context=empty)
        check('Allfailed natural0observations completes withoutGPU/cache/forward', complete['gpu_run'] is False and complete['model_forward_run'] is False
            and not empty.cache.exists() and all(p.shape == (0, 5) for p in ep.values()))
        er = records_from_physical(ep); check('Emptyoldtwo records share exactarray and nullSHA', er['old-head-maximin'] is er['old-head-original']
            and complete['record_probability_sha256']['old-head-maximin'] is None and complete['record_probability_sha256']['old-head-original'] is None)
        fast_integrity(ctx)
    py_compile.compile(__file__, cfile=str(folder / 'infer.pyc'), doraise=True)
    require(digest(__file__) == source_sha, 'Preflight changed inference source.')
    report = dict(stage='synthetic_CPU_probability_inference_preflight', source_sha256=source_sha, checks=checks, checks_passed=len(checks),
        cached_physical_models=3, records=4, maximum_CPU_portable_difference=maxdiff, fixture_directory=str(folder),
        actual_builder_or_snapshot_read=False, real_readout_or_LM_loaded=False, raw_features_persisted=False,
        test_source_body_read=False, actual_probability_arrays_read=False, gpu_used=False, formal_outputs_written=False,
        native_hook_and_CPU_GPU_parity='Actual four-A40 hidden-state identity and same-vector GPU parity remain mandatory at future inference, never claimed by these CPU fixtures.')
    atomic_json(folder / 'report.json', report); print(json.dumps(dict(report=str(folder / 'report.json'), checks_passed=len(checks), source_sha256=source_sha)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(); action.add_argument('--infer', action='store_true'); action.add_argument('--preflight', action='store_true')
    parser.add_argument('--dataset', choices=sorted(BUILDERS)); args = parser.parse_args()
    if args.preflight: preflight()
    elif args.infer:
        require(args.dataset is not None, '--infer requires --dataset c4|natural.'); infer(args.dataset)
    else: parser.print_help()

if __name__ == '__main__': main()
