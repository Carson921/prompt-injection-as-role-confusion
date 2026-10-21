"""Extract the frozen top-eight query heads into a 512-dimensional bank.

Default: help. --preflight uses CPU and /tmp only. --stage extract explicitly
runs the verified native-MXFP4 checkpoint on four sharded A40s. Only original
native fit/validation prompts are retokenized; mixed-split token JSON is never
decoded. This bank is a readout hypothesis, not a change to the frozen primary.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_head_scan as scanner
import systemness_long_data as long_data
import systemness_long_probes as long_probes


OUT = data.runtime.OUT / 'systemness-head-features'
CACHE = Path('/dev/shm/role-analysis-top8-heads')
BANK = long_probes.BANK
ROWS, DIMENSIONS, HEAD_DIM, TOP_K = 755282, 512, 64, 8
LAYERS, HEADS = 36, 64
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 16
PARITY_TOLERANCE = 5e-5
RESERVE_BYTES = 512 * 2**20
MIN_GPU_FREE_BYTES = 22 * 2**30
DEPENDENCIES = ['systemness_head_extract.py', 'systemness_head_scan.py',
    'systemness_long_data.py', 'systemness_long_extract.py', 'systemness_long_probes.py',
    'systemness_data.py', 'systemness_probes.py', 'run_local_gptoss120b.py']
require = long_data.require


def read(path):
    return json.loads(Path(path).read_text())


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frame_digest(frame, columns):
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def input_sources():
    result = {}
    for name, folder, metadata_path in [
        ('main', data.CACHE, data.OUT / 'data-complete.json'),
        ('supplement', data.CACHE / 'supplement', data.OUT / 'supplement-data-complete.json'),
        ('long-full', long_data.CACHE, long_data.OUT / 'data-complete.json')]:
        metadata = read(metadata_path)
        hashes = {file: data.digest(folder / file) for file in ['inputs.parquet', 'rows.parquet']}
        require(all(metadata['cache_sha256'][file] == digest for file, digest in hashes.items()),
                f'{name} source input/row cache changed.')
        result[name] = dict(folder=str(folder), metadata_sha256=data.digest(metadata_path), cache_sha256=hashes)
    metadata = read(long_data.OUT / 'data-complete.json')
    require(metadata['plan_sha256'] == data.digest(long_data.OUT / 'plan.json')
        and metadata['document_keys_sha256'] == data.digest(long_data.OUT / 'document-keys.json')
        and metadata['independent_validation']['all_labels_match']
        and metadata['test_inputs_read'] is False, 'Long-full native-data verification changed.')
    return result


def original_bank():
    complete = read(long_data.OUT / 'extraction-complete.json')
    require(complete['rows'] == ROWS and complete['all_finite'] and complete['tests_read'] is False,
            'Original fit/validation bank is not complete.')
    require(complete['rows_sha256'] == data.digest(BANK / 'rows.parquet'), 'Original bank row table changed.')
    require(complete['source_sha256'] == data.digest(data.HERE / 'systemness_long_extract.py'),
            'Original bank extraction source changed.')
    return dict(completion_sha256=data.digest(long_data.OUT / 'extraction-complete.json'),
        rows_sha256=complete['rows_sha256'], rows=ROWS)


def selected_identity():
    """CPU metadata/portable-array checks; no tokenizer, raw features or GPU."""
    plan_path = scanner.OUT / 'plan.json'
    plan = read(plan_path); scanner.verify_plan(plan)
    complete_path = scanner.OUT / 'scan-complete.json'
    selection_path = scanner.OUT / 'head-selection.json'
    model_path = scanner.OUT / 'models/top8-lda.npz'
    complete, selection = read(complete_path), read(selection_path)
    require(complete['plan_sha256'] == selection['plan_sha256'] == data.digest(plan_path)
        and complete['model_sha256'] == selection['model_sha256'] == data.digest(model_path)
        and complete['head_selection_sha256'] == data.digest(selection_path)
        and complete['discovery_complete_sha256'] == data.digest(scanner.OUT / 'discovery-complete.json'),
        'Frozen scanner selection/completion/model identities disagree.')
    require(complete['tests_read'] is False and selection['test_used_for_selection'] is False
        and selection['head_selection_fit_only'] and selection['refit_includes_discovery']
        and selection['fit_documents'] == 132 and selection['roles'] == data.ROLES,
        'Selected heads do not have the declared fit-only provenance.')
    selected = np.asarray(selection['selected_head_ids'], dtype=np.int64)
    pairs = [[int(head // HEADS), int(head % HEADS)] for head in selected]
    require(selected.shape == (TOP_K,) and len(set(selected)) == TOP_K
        and ((selected >= 0) & (selected < LAYERS * HEADS)).all()
        and complete['selected_head_ids'] == selected.tolist()
        and selection['selected_layer_head_pairs'] == pairs and selection['head_dim'] == HEAD_DIM,
        'Selected head identities/order changed.')
    with np.load(model_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        require(np.array_equal(archive['selected_head_ids'], selected)
            and archive['weight'].shape == (TOP_K, 5, HEAD_DIM) and archive['bias'].shape == (TOP_K, 5)
            and np.isfinite(archive['weight']).all() and np.isfinite(archive['bias']).all()
            and metadata['selected_head_ids'] == selected.tolist()
            and metadata['plan_sha256'] == data.digest(plan_path)
            and metadata['source_sha256'] == plan['source_sha256'], 'Portable scanner model changed.')
    return dict(plan_sha256=data.digest(plan_path), scan_complete_sha256=data.digest(complete_path),
        head_selection_sha256=data.digest(selection_path), model_path=str(model_path), model_sha256=data.digest(model_path),
        selected_head_ids=selected.tolist(), ordered_layers_heads=pairs, head_dim=HEAD_DIM,
        scanner_source_sha256=plan['source_sha256'])


def source_signature():
    """No tokenization, CUDA, test JSON decoding or feature-bank reads."""
    return dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        selection=selected_identity(), original_bank=original_bank(), input_sources=input_sources(),
        scanner_signature=scanner.source_signature(), cache_path=str(CACHE), rows=ROWS,
        dimensions=DIMENSIONS, dtype='float16', head_dim=HEAD_DIM, batch_size=BATCH_SIZE)


class Bundle:
    """One grouped lookup avoids a full row-table scan for each source prompt."""
    def __init__(self, inputs, rows, tokenizer):
        self.inputs = inputs.sort_values(['length', 'uid']).reset_index(drop=True)
        self.rows = rows.copy().reset_index(drop=True)
        require(self.inputs.uid.is_unique and set(self.inputs.uid) == set(self.rows.uid), 'Prompt membership mismatch.')
        self.tokens = {}
        self.lookup = {uid: group for uid, group in self.rows.groupby('uid', sort=False)}
        for entry in self.inputs.itertuples():
            ids = tokenizer.encode(entry.prompt, add_special_tokens=False)
            require(len(ids) == entry.length, 'Retokenization differs from original input length.')
            group = self.lookup[entry.uid]
            labels = long_data.parse_native_labels(ids, tokenizer)
            require(all(0 <= row.token_ix < len(ids) and ids[row.token_ix] == row.token_id
                and labels.get(row.token_ix) == row.role for row in group.itertuples()),
                'Native role/content-token identity changed after retokenization.')
            self.tokens[entry.uid] = ids
        self.parts = [self.inputs.iloc[start:start + BATCH_SIZE]
                      for start in range(0, len(self.inputs), BATCH_SIZE)]

    def targets(self, part):
        return np.concatenate([self.lookup[entry.uid].cache_ix.to_numpy(dtype=np.int64)
                               for entry in part.itertuples()])

    def collate(self, part):
        import torch
        width = int(part.length.max())
        ids = torch.zeros((len(part), width), dtype=torch.long)
        mask = torch.zeros_like(ids); positions = []
        for index, entry in enumerate(part.itertuples()):
            tokens = self.tokens[entry.uid]; size = len(tokens)
            ids[index, -size:] = torch.tensor(tokens); mask[index, -size:] = 1
            positions.extend((self.lookup[entry.uid].token_ix.to_numpy() + index * width + width - size).tolist())
        targets = self.targets(part)
        require(len(set(targets)) == len(targets), 'Duplicate sampled rows in one batch.')
        positions = np.asarray(positions, dtype=np.int64)
        require(np.array_equal(ids.reshape(-1)[positions].numpy(), self.rows.iloc[targets].token_id.to_numpy()),
                'Padded source token positions do not map to the original cache rows.')
        return dict(input_ids=ids, attention_mask=mask, positions=positions, targets=targets)

    def descriptor(self):
        return dict(rows=len(self.rows), prompts=len(self.inputs), batches=len(self.parts),
            split_rows={key: int(value) for key, value in self.rows.groupby('split').size().items()},
            source_rows={key: int(value) for key, value in self.rows.groupby('source_set').size().items()},
            input_tokens=int(self.inputs.length.sum()), max_batch_rows=max(len(self.targets(part)) for part in self.parts),
            inputs_sha256=frame_digest(self.inputs, ['uid', 'prompt', 'length']),
            row_mapping_sha256=frame_digest(self.rows, ['cache_ix', 'sample_ix', 'uid', 'token_ix', 'token_id', 'role']),
            sample_ix_sha256=array_digest(self.rows.sample_ix.to_numpy()))


def extraction_data(tokenizer):
    original_bank(); sources = input_sources()
    rows = pd.read_parquet(BANK / 'rows.parquet')
    require(len(rows) == ROWS and np.array_equal(rows.cache_ix, np.arange(ROWS)) and rows.sample_ix.is_unique
        and set(rows.split) == {'fit', 'validation'} and set(rows.wrapper) == {'native'}
        and set(rows.source_set) == set(sources), 'Original native development bank identity changed.')
    require(rows.groupby('split').size().to_dict() == {'fit': 582250, 'validation': 173032}
        and not set(rows[rows.split == 'fit'].group_id) & set(rows[rows.split == 'validation'].group_id),
        'Development row counts or source group partition changed.')
    frames = []
    for name, source in sources.items():
        wanted = rows[rows.source_set == name]
        # Parquet filtering precedes prompt decoding. Never load token-ids.json.
        frame = pd.read_parquet(Path(source['folder']) / 'inputs.parquet',
            filters=[('split', 'in', ['fit', 'validation']), ('wrapper', '=', 'native')])
        frame = frame[frame.prompt_ix.isin(wanted.prompt_ix)].copy()
        require(frame.prompt_ix.is_unique and set(frame.prompt_ix) == set(wanted.prompt_ix),
                f'{name} source prompt mapping differs from original bank.')
        identity = frame.set_index('prompt_ix')
        for column in ['split', 'profile', 'group_id']:
            require(np.array_equal(identity.loc[wanted.prompt_ix, column].to_numpy(), wanted[column].to_numpy()),
                    f'{name} prompt/{column} mapping changed.')
        frame['uid'] = name + ':' + frame.prompt_ix.astype(str); frames.append(frame)
    rows['uid'] = rows.source_set + ':' + rows.prompt_ix.astype(str)
    return Bundle(pd.concat(frames, ignore_index=True), rows, tokenizer)


def feature_bytes(rows=ROWS, dimensions=DIMENSIONS):
    return rows * dimensions * np.dtype(np.float16).itemsize


def resources(bundle=None):
    return dict(rows=ROWS, dimensions=DIMENSIONS, feature_bytes=feature_bytes(),
        feature_gib=feature_bytes() / 2**30, reserve_bytes=RESERVE_BYTES,
        minimum_gpu_free_bytes_before_load=MIN_GPU_FREE_BYTES,
        source_rows_copied_byte_exactly=True, cache_path=str(CACHE),
        max_batch_rows=None if bundle is None else bundle.descriptor()['max_batch_rows'],
        memory_note='512 F16 features per sampled token. Only sampled positions from the current selected layer are gathered; no all-layer raw bank. CPU source tokens, row tables, model workspaces and bounded copies add peak memory.',
        retirement='No old cache is deleted or overwritten. Existing mismatched initialization/plan is an error.')


def declare(bundle):
    signature = source_signature()
    plan = dict(version=1, signature=signature, source_sha256=signature['source_sha256'],
        dependencies_sha256=signature['dependencies_sha256'], data=bundle.descriptor(),
        selected_head_ids=signature['selection']['selected_head_ids'],
        ordered_layers_heads=signature['selection']['ordered_layers_heads'],
        head_dim=HEAD_DIM, dimensions=DIMENSIONS, roles=data.ROLES, dtype='float16',
        extraction='Sequential model.model forward on four sharded A40s, frozen native MXFP4. Capture o_proj prehook input, gather sampled content positions, concatenate head vectors in frozen selected rank order.',
        rows='Byte-copy long-full-bank/rows.parquet. Preserve every original source_set, sample_ix, cache_ix, split and prompt/token identity.',
        hook_check='Disable all writes while comparing unhooked and hooked final hidden states bitwise. Capture hooks return None.',
        portability='Compare saved F16 vectors through the portable NumPy and GPU F64 eight-head LDA aggregate. BF16-to-F16 quantization is reported separately, not claimed bitwise exact.',
        portable_probability_tolerance=PARITY_TOLERANCE,
        restart='NaN-initialize atomically, then checkpoint every16 complete batches. Flush/fsync features before atomic progress. Resume only exact completed prompt prefix; independently rehash each committed batch. Uncommitted partial writes are overwritten.',
        zero_vectors='Finite all-zero vectors are valid. Record zero_row_count/all_rows_nonzero truthfully; require all_rows_written and all_finite rather than rejecting a legitimate constant head.',
        resources=resources(bundle), checkpoint_interval=CHECKPOINT_INTERVAL,
        cpu_preflight=cpu_checks(),
        limitation='Heads were screened among2304 on31 fit-discovery documents and refit on132 fit documents. This bank supports a new joint readout hypothesis; it does not promote a model or alter the frozen primary. No test/current464 input is loaded.',
        tests_read=False, confirmation_inputs_read=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '.gitignore').write_text('/.run.lock\n')
    path = OUT / 'plan.json'
    if path.exists(): require(read(path) == plan, 'Extraction plan/source/cache changed; refusing silent overwrite.')
    else: data.atomic_json(path, plan)
    record = dict(source_sha256=plan['source_sha256'], plan_sha256=data.digest(path),
        checks=plan['cpu_preflight'], data=plan['data'], model_loaded=False, gpu_work_run=False, tests_read=False)
    path = OUT / 'cpu-preflight.json'
    if path.exists(): require(read(path) == record, 'Published extraction CPU record differs.')
    else: data.atomic_json(path, record)
    return plan


def verify_plan(plan):
    """CPU-only immutable signature check, public for downstream readouts."""
    require(plan.get('signature') == source_signature(), 'Head extraction signature changed.')
    require(plan['source_sha256'] == plan['signature']['source_sha256']
        and plan['dependencies_sha256'] == plan['signature']['dependencies_sha256'], 'Extraction source declarations disagree.')
    require(read(OUT / 'plan.json') == plan, 'Published extraction plan differs.')
    record = read(OUT / 'cpu-preflight.json')
    require(record['plan_sha256'] == data.digest(OUT / 'plan.json')
        and record['source_sha256'] == plan['source_sha256']
        and record['checks'] == plan['cpu_preflight'], 'Published CPU record changed.')


def check_feature_file(path, rows=ROWS, dimensions=DIMENSIONS):
    require(Path(path).is_file() and Path(path).stat().st_size == feature_bytes(rows, dimensions),
            'Feature file absent/truncated/wrong shape; refusing fabricated resume.')


def inspect_features(array):
    zeros = 0
    for start in range(0, len(array), 4096):
        block = array[start:start + 4096]
        require(np.isfinite(block).all(), 'Nonfinite/unwritten selected-head feature rows.')
        zeros += int(np.all(block == 0, axis=1).sum())
    return dict(all_finite=True, all_rows_nonzero=zeros == 0, zero_row_count=zeros)


def verified_extraction():
    """Public CPU-only completion/file/hash verification; never retokenizes."""
    plan = read(OUT / 'plan.json'); verify_plan(plan)
    complete = read(OUT / 'extraction-complete.json')
    require(complete['plan_sha256'] == data.digest(OUT / 'plan.json')
        and complete['source_sha256'] == plan['source_sha256'] and complete['signature'] == plan['signature']
        and complete['rows'] == ROWS and complete['dimensions'] == DIMENSIONS
        and complete['feature_shape'] == [ROWS, DIMENSIONS] and complete['dtype'] == 'float16'
        and complete['all_finite'] and complete['all_rows_written'] and complete['tests_read'] is False,
        'Selected-head extraction completion does not match the declared protocol.')
    require(complete['selected_head_ids'] == plan['selected_head_ids']
        and complete['ordered_layers_heads'] == plan['ordered_layers_heads'] and complete['head_dim'] == HEAD_DIM,
        'Completed selected-head column order differs from the model.')
    require(complete['rows_path'] == str(CACHE / 'rows.parquet')
        and complete['features_path'] == str(CACHE / 'features.f16'), 'Completed feature-bank paths changed.')
    require(complete['rows_sha256'] == plan['signature']['original_bank']['rows_sha256'] == data.digest(CACHE / 'rows.parquet'),
            'Selected-head bank row identity changed.')
    check_feature_file(CACHE / 'features.f16')
    require(complete['features_sha256'] == data.digest(CACHE / 'features.f16'), 'Selected-head feature bytes changed.')
    rows = pd.read_parquet(CACHE / 'rows.parquet', columns=['cache_ix', 'sample_ix', 'split', 'wrapper'])
    require(len(rows) == ROWS and np.array_equal(rows.cache_ix, np.arange(ROWS)) and rows.sample_ix.is_unique
        and set(rows.split) == {'fit', 'validation'} and set(rows.wrapper) == {'native'}, 'Completed row mapping changed.')
    require(complete['hook_check_disabled_writes'] and complete['hidden_states_bitexact']
        and complete['portable_probability_max_difference'] <= PARITY_TOLERANCE,
        'Completed feature bank lacks hook/portable-consumer validation.')
    require(complete['hook_validation_sha256'] == data.digest(OUT / 'hook-validation.json')
        and complete['progress_sha256'] == data.digest(CACHE / 'progress.json'), 'Completed hook/progress audit changed.')
    progress = read(CACHE / 'progress.json')
    require(progress['plan_sha256'] == complete['plan_sha256'] and progress['rows_written'] == ROWS
        and progress['next_batch'] == plan['data']['batches']
        and len(progress['batch_features_sha256']) == progress['next_batch']
        and progress['portable_consumer'] == complete['portable_consumer'], 'Completed prefix audit is inconsistent.')
    return complete


def fsync_file(path):
    with Path(path).open('rb') as stream: os.fsync(stream.fileno())


def initialize(cache, source_rows, plan_sha256, rows=ROWS, dimensions=DIMENSIONS):
    """An interrupted initialization never masquerades as a resumable bank."""
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    initialized = cache / 'initialized.json'; target_rows = cache / 'rows.parquet'
    feature_path = cache / 'features.f16'
    expected = dict(plan_sha256=plan_sha256, rows=rows, dimensions=dimensions, dtype='float16',
                    rows_sha256=data.digest(source_rows))
    if initialized.exists():
        require(read(initialized) == expected and data.digest(target_rows) == expected['rows_sha256'],
                'Initialized cache ownership/row hashes differ.')
        check_feature_file(feature_path, rows, dimensions)
        return np.memmap(feature_path, mode='r+', dtype=np.float16, shape=(rows, dimensions))
    require(not any(cache.iterdir()), 'Uncommitted initialization exists; inspect it before retrying. No cache is removed automatically.')
    row_temporary = cache / 'rows.parquet.initializing'
    with Path(source_rows).open('rb') as source, row_temporary.open('xb') as destination:
        shutil.copyfileobj(source, destination); destination.flush(); os.fsync(destination.fileno())
    require(data.digest(row_temporary) == expected['rows_sha256'], 'Byte-copied row table changed.')
    row_temporary.replace(target_rows)
    temporary = cache / 'features.f16.initializing'
    array = np.memmap(temporary, mode='w+', dtype=np.float16, shape=(rows, dimensions))
    for start in range(0, rows, 4096): array[start:start + 4096] = np.nan
    array.flush(); del array; fsync_file(temporary); temporary.replace(feature_path)
    data.atomic_json(initialized, expected)
    return np.memmap(feature_path, mode='r+', dtype=np.float16, shape=(rows, dimensions))


def batch_digest(array, targets):
    value = hashlib.sha256()
    for start in range(0, len(targets), 4096):
        block = np.ascontiguousarray(array[targets[start:start + 4096]])
        require(np.isfinite(block).all(), 'Committed batch contains an unwritten feature column.')
        value.update(block.tobytes())
    return value.hexdigest()


def prefix_state(bundle, next_batch):
    require(0 <= next_batch <= len(bundle.parts), 'Checkpoint batch cursor is outside source prompts.')
    targets = [bundle.targets(part) for part in bundle.parts[:next_batch]]
    target = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    require(len(np.unique(target)) == len(target), 'Completed prompt prefix contains duplicate original rows.')
    prompts = bundle.inputs.iloc[:min(next_batch * BATCH_SIZE, len(bundle.inputs))]
    return dict(next_batch=next_batch, rows_written=len(target), cache_ix_sha256=array_digest(target),
        sample_ix_sha256=array_digest(bundle.rows.iloc[target].sample_ix.to_numpy()),
        prompt_prefix_sha256=frame_digest(prompts, ['uid', 'length']))


def save_progress(cache, bundle, plan_sha256, hashes, parity):
    value = dict(plan_sha256=plan_sha256, **prefix_state(bundle, len(hashes)),
        batch_features_sha256=hashes, portable_consumer=parity)
    fsync_file(Path(cache) / 'features.f16')
    data.atomic_json(Path(cache) / 'progress.json', value)
    return value


def restore_progress(cache, bundle, plan_sha256, array):
    path = Path(cache) / 'progress.json'
    if not path.exists(): return [], None
    value = read(path); hashes = value['batch_features_sha256']; state = prefix_state(bundle, len(hashes))
    require(value['plan_sha256'] == plan_sha256 and all(value.get(key) == item for key, item in state.items()),
            'Saved extraction progress does not match the exact source prompt/row prefix.')
    for part, digest in zip(bundle.parts[:len(hashes)], hashes):
        require(batch_digest(array, bundle.targets(part)) == digest, 'Previously committed feature batch bytes changed.')
    require(not hashes or value['portable_consumer'] is not None, 'Committed rows lack portable consumer verification.')
    return hashes.copy(), value['portable_consumer']


class Collector:
    def __init__(self, array, selected, heads=HEADS, dimensions=HEAD_DIM):
        self.array = array; self.selected = np.asarray(selected, dtype=np.int64)
        self.heads, self.dimensions = heads, dimensions
        require(array.shape[1] == len(selected) * dimensions, 'Selected head output dimension differs.')
        self.layers = sorted(set((self.selected // heads).tolist()))
        self.enabled = False; self.batch = None; self.seen = set(); self.device_positions = {}
        self.maximum_f16_roundtrip_difference = 0.

    def begin(self, batch):
        self.batch = batch; self.seen = set(); self.device_positions = {}

    def hook(self, layer):
        def receive(module, args):
            import torch
            value = args[0]
            require(value.ndim == 3 and value.shape[-1] == self.heads * self.dimensions, 'Unexpected pre-o_proj head layout.')
            require(layer not in self.seen, 'Selected layer hook ran twice in one sequential forward.')
            self.seen.add(layer)
            if not self.enabled: return
            chosen = np.flatnonzero(self.selected // self.heads == layer)
            key = str(value.device)
            if key not in self.device_positions:
                self.device_positions[key] = torch.from_numpy(self.batch['positions']).to(value.device)
            positions = self.device_positions[key]
            heads = torch.as_tensor(self.selected[chosen] % self.heads, dtype=torch.long, device=value.device)
            # Gather sampled content tokens before retaining any head activations.
            x = value.detach().reshape(-1, self.heads, self.dimensions).index_select(0, positions).index_select(1, heads)
            require(bool(torch.isfinite(x).all()) and bool((x.float().abs() <= float(np.finfo(np.float16).max)).all()),
                    'Nonfinite/out-of-F16-range sampled head vectors; refusing clipping.')
            block = x.to(torch.float16).cpu().numpy().reshape(len(positions), -1)
            difference = float((x.float() - x.to(torch.float16).float()).abs().max().item())
            self.maximum_f16_roundtrip_difference = max(self.maximum_f16_roundtrip_difference, difference)
            columns = (chosen[:, None] * self.dimensions + np.arange(self.dimensions)).reshape(-1)
            self.array[np.ix_(self.batch['targets'], columns)] = block
            # forward_pre_hook returns None and never alters the model input.
        return receive

    def finish(self):
        require(self.seen == set(self.layers), 'Sequential forward missed a selected-head layer.')


def install(model, collector):
    return [model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer))
            for layer in collector.layers]


def forward(model, collector, batch, device='cuda:0'):
    collector.begin(batch)
    hidden = model.model(input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device), use_cache=False).last_hidden_state
    collector.finish()
    return hidden


@contextmanager
def checked_hooks(model, collector, bundle):
    import torch
    batch = bundle.collate(bundle.parts[0]); old = collector.array[batch['targets']].copy()
    with torch.inference_mode():
        before = model.model(input_ids=batch['input_ids'].to('cuda:0'),
            attention_mask=batch['attention_mask'].to('cuda:0'), use_cache=False).last_hidden_state.cpu()
        handles = install(model, collector)
        try:
            collector.enabled = False
            after = forward(model, collector, batch).cpu()
            require(torch.equal(before, after) and bool(torch.isfinite(after).all()), 'Capture hooks changed final hidden states.')
            require(np.array_equal(old, collector.array[batch['targets']], equal_nan=True), 'Disabled hook wrote feature data.')
            data.atomic_json(OUT / 'hook-validation.json', dict(hidden_states_bitexact=True,
                writes_disabled=True, feature_rows_unchanged=True, layers=collector.layers,
                sample_rows=len(batch['targets'])))
            del before, after, old
            collector.enabled = True
            yield
        finally:
            for handle in handles: handle.remove()


def portable_consumer(array, targets, weight, bias, device):
    import torch
    saved = np.asarray(array[np.asarray(targets[:min(256, len(targets))])], dtype=np.float64).reshape(-1, len(weight), weight.shape[-1])
    require(np.isfinite(saved).all(), 'Saved portable-consumer vectors missed a selected head.')
    expected = scanner.predict_heads(saved, weight, bias).mean(1)
    x = torch.from_numpy(saved).to(device)
    w = torch.from_numpy(np.asarray(weight, dtype=np.float64)).to(device)
    b = torch.from_numpy(np.asarray(bias, dtype=np.float64)).to(device)
    with torch.inference_mode():
        actual = torch.softmax(torch.einsum('nhd,hkd->nhk', x, w) + b[None], dim=-1).mean(1).cpu().numpy()
    difference = float(np.abs(expected - actual).max())
    require(difference <= PARITY_TOLERANCE, 'Saved512-vector portable NumPy/GPU LDA probabilities disagree.')
    return dict(max_difference=difference, tolerance=PARITY_TOLERANCE, rows=len(saved),
        saved_vectors_sha256=array_digest(np.asarray(array[targets[:len(saved)]], dtype=np.float16)),
        same_saved_f16_vectors=True, classifier_device=str(device))


def no_competing_jobs():
    names = {'systemness_long_extract.py', 'systemness_recover_local_bank.py', 'systemness_long_confirm.py',
        'systemness_attention_confirm.py', 'systemness_long_probes.py', 'systemness_paired_probes.py',
        'systemness_natural_recheck.py', 'systemness_head_scan.py', 'systemness_head_extract.py',
        'systemness_head_probes.py'}
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg): active.append(int(entry.name))
    require(not active, f'Other experiment processes remain active: {active}; wait before extracting.')


@contextmanager
def run_lock():
    import fcntl
    with (OUT / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another selected-head extractor holds this output.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_model():
    import torch
    require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Exactly4 A40s required.')
    available = [int(torch.cuda.mem_get_info(i)[0]) for i in range(4)]
    require(min(available) >= MIN_GPU_FREE_BYTES, 'Fewer than22GiB free on an A40 before model loading; wait for other allocations to clear.')
    data.atomic_json(OUT / 'runtime-resources.json', dict(gpu_free_bytes_before_load=available,
        minimum_gpu_free_bytes=MIN_GPU_FREE_BYTES, shared_memory_free_bytes=shutil.disk_usage(CACHE.parent).free))
    original_atomic, original_status = data.runtime.atomic_json, data.runtime.status
    data.runtime.atomic_json = lambda path, value: data.atomic_json(OUT / Path(path).name, value)
    data.runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
    try: model = data.runtime.load_model()
    finally: data.runtime.atomic_json, data.runtime.status = original_atomic, original_status
    require(not any(parameter.requires_grad for parameter in model.parameters()), 'Language model must remain frozen.')
    return model


def run():
    no_competing_jobs()
    # Selection must exist and remain frozen before any new formal output/GPU work.
    selected_identity()
    tokenizer = long_data.load_tokenizer(); bundle = extraction_data(tokenizer)
    plan = declare(bundle)
    with run_lock():
        verify_plan(plan)
        if (OUT / 'extraction-complete.json').exists():
            print(json.dumps(verified_extraction()), flush=True); return
        free = shutil.disk_usage(CACHE.parent).free
        need = RESERVE_BYTES if (CACHE / 'initialized.json').exists() else feature_bytes() + (BANK / 'rows.parquet').stat().st_size + RESERVE_BYTES
        require(free > need, 'Insufficient shared-memory space for bank plus reserve; no existing cache will be removed.')
        array = initialize(CACHE, BANK / 'rows.parquet', data.digest(OUT / 'plan.json'))
        hashes, parity = restore_progress(CACHE, bundle, data.digest(OUT / 'plan.json'), array)
        selected = np.asarray(plan['selected_head_ids'], dtype=np.int64)
        with np.load(plan['signature']['selection']['model_path'], allow_pickle=False) as archive:
            weight, bias = archive['weight'].copy(), archive['bias'].copy()
        collector = Collector(array, selected); model = load_model()
        import torch
        with checked_hooks(model, collector, bundle), torch.inference_mode():
            for index in range(len(hashes), len(bundle.parts)):
                batch = bundle.collate(bundle.parts[index]); result = forward(model, collector, batch); del result
                hashes.append(batch_digest(array, batch['targets']))
                if parity is None: parity = portable_consumer(array, batch['targets'], weight, bias, 'cuda:0')
                if len(hashes) % CHECKPOINT_INTERVAL == 0 or len(hashes) == len(bundle.parts):
                    array.flush(); progress = save_progress(CACHE, bundle, data.digest(OUT / 'plan.json'), hashes, parity)
                    status = dict(stage='extract', batch=len(hashes), batches=len(bundle.parts),
                        rows_written=progress['rows_written'], maximum_f16_roundtrip_difference=collector.maximum_f16_roundtrip_difference)
                    data.atomic_json(OUT / 'status.json', status); print(json.dumps(status), flush=True)
        state = prefix_state(bundle, len(hashes))
        require(state['rows_written'] == ROWS and len(hashes) == len(bundle.parts), 'Incomplete selected-head row coverage.')
        checks = inspect_features(array); array.flush(); fsync_file(CACHE / 'features.f16')
        verify_plan(plan)
        hook_check = read(OUT / 'hook-validation.json')
        complete = dict(signature=plan['signature'], source_sha256=plan['source_sha256'],
            dependencies_sha256=plan['dependencies_sha256'], plan_sha256=data.digest(OUT / 'plan.json'),
            rows=ROWS, dimensions=DIMENSIONS, feature_shape=[ROWS, DIMENSIONS], dtype='float16',
            rows_path=str(CACHE / 'rows.parquet'), features_path=str(CACHE / 'features.f16'),
            rows_sha256=data.digest(CACHE / 'rows.parquet'), features_sha256=data.digest(CACHE / 'features.f16'),
            selected_head_ids=selected.tolist(), ordered_layers_heads=plan['ordered_layers_heads'], head_dim=HEAD_DIM,
            data=bundle.descriptor(), all_rows_written=True, **checks,
            hook_check_disabled_writes=hook_check['writes_disabled'], hidden_states_bitexact=hook_check['hidden_states_bitexact'],
            hook_validation_sha256=data.digest(OUT / 'hook-validation.json'),
            portable_probability_max_difference=parity['max_difference'], portable_consumer=parity,
            maximum_f16_roundtrip_difference_this_run=collector.maximum_f16_roundtrip_difference,
            quantization_note='Parity uses identical saved F16 vectors. Reported activation roundtrip maximum covers this process only; earlier resumed processes may have a larger maximum.',
            progress_sha256=data.digest(CACHE / 'progress.json'), tests_read=False, confirmation_inputs_read=False,
            frozen_primary_changed=False)
        data.atomic_json(OUT / 'extraction-complete.json', complete)
        verified_extraction()
        data.atomic_json(OUT / 'status.json', dict(stage='extraction_complete', rows=ROWS, dimensions=DIMENSIONS, tests_read=False))
        del array, collector, model; gc.collect()


def cpu_checks():
    """Deterministic synthetic fixtures; all persistence stays in /tmp."""
    import torch
    from types import SimpleNamespace
    torch.manual_seed(1109)
    selected = np.array([5, 0])  # Deliberately unsorted across two layers.
    with tempfile.TemporaryDirectory(prefix='systemness-head-extract-fixture-', dir='/tmp') as temporary:
        folder = Path(temporary); source = folder / 'source.parquet'
        pd.DataFrame(dict(cache_ix=np.arange(4), sample_ix=np.arange(100, 104))).to_parquet(source, index=False)
        cache = folder / 'cache'; array = initialize(cache, source, 'synthetic', rows=4, dimensions=8)
        require(np.isnan(array).all(), 'Initialization did not use unwritten NaN sentinels.')
        x = torch.arange(2 * 3 * 12, dtype=torch.float32).reshape(2, 3, 12) / 8
        batch = dict(input_ids=x, attention_mask=torch.ones((2, 3)),
            positions=np.array([5, 0, 2, 3]), targets=np.array([2, 0, 3, 1]))
        class Attention(torch.nn.Module):
            def __init__(self, delta):
                super().__init__(); self.delta = delta; self.o_proj = torch.nn.Identity()
            def forward(self, value): return self.o_proj(value + self.delta)
        class Layer(torch.nn.Module):
            def __init__(self, delta): super().__init__(); self.self_attn = Attention(delta)
            def forward(self, value): return self.self_attn(value)
        class Body(torch.nn.Module):
            def __init__(self): super().__init__(); self.layers = torch.nn.ModuleList([Layer(0), Layer(10)])
            def forward(self, input_ids, attention_mask=None, use_cache=False):
                value = input_ids
                for layer in self.layers: value = layer(value)
                return SimpleNamespace(last_hidden_state=value)
        model = SimpleNamespace(model=Body()); collector = Collector(array, selected, heads=3, dimensions=4)
        before = model.model(x).last_hidden_state; handles = install(model, collector)
        try:
            after = forward(model, collector, batch, 'cpu')
            require(torch.equal(before, after) and np.isnan(array).all(), 'Disabled hooks changed hidden states or wrote features.')
            collector.enabled = True; after = forward(model, collector, batch, 'cpu')
            raw = x.reshape(-1, 3, 4)[batch['positions']].numpy()
            expected = np.concatenate([raw[:, 2] + 10, raw[:, 0]], axis=1).astype(np.float16)
            require(np.array_equal(array[batch['targets']], expected) and torch.equal(before, after),
                    'Unsorted selected-head concatenation or original row mapping failed.')
            collector.begin(batch)
            try: collector.finish()
            except RuntimeError: pass
            else: raise RuntimeError('Missing selected-layer guard failed.')
            collector.begin(batch); collector.hook(0)(None, (x,))
            try: collector.hook(0)(None, (x,))
            except RuntimeError: pass
            else: raise RuntimeError('Duplicate selected-layer guard failed.')
            collector.begin(batch)
            try: collector.hook(0)(None, (torch.full_like(x, 65536, dtype=torch.bfloat16),))
            except RuntimeError: pass
            else: raise RuntimeError('BF16-to-F16 overflow guard failed.')
        finally:
            for handle in handles: handle.remove()
        generator = np.random.default_rng(621)
        weight = generator.normal(size=(2, 5, 4)); bias = generator.normal(size=(2, 5))
        parity = portable_consumer(array, batch['targets'], weight, bias, 'cpu')
        # Zero vectors are legitimate, but NaNs must fail completed-bank checks.
        array[0] = 0
        checks = inspect_features(array)
        require(checks['zero_row_count'] == 1 and not checks['all_rows_nonzero'], 'Legitimate zero-vector diagnostic failed.')
        array[1, 0] = np.nan
        try: inspect_features(array)
        except RuntimeError: pass
        else: raise RuntimeError('Unwritten NaN guard failed.')
        array[batch['targets']] = expected; array.flush()
        digest = batch_digest(array, batch['targets'])
        restored = initialize(cache, source, 'synthetic', rows=4, dimensions=8)
        require(batch_digest(restored, batch['targets']) == digest, 'Feature reopen/hash fixture changed.')
        try: initialize(cache, source, 'wrong-plan', rows=4, dimensions=8)
        except RuntimeError: pass
        else: raise RuntimeError('Different-protocol cache ownership guard failed.')
        class TinyBundle:
            inputs = pd.DataFrame(dict(uid=['a', 'b'], length=[3, 3]))
            rows = pd.DataFrame(dict(sample_ix=np.arange(100, 104)))
            parts = [inputs]
            def targets(self, part): return batch['targets']
        tiny = TinyBundle(); save_progress(cache, tiny, 'synthetic', [digest], parity)
        hashes, saved_parity = restore_progress(cache, tiny, 'synthetic', restored)
        require(hashes == [digest] and saved_parity == parity, 'Exact completed-prefix resume fixture failed.')
        restored[0, 0] += 1; restored.flush()
        try: restore_progress(cache, tiny, 'synthetic', restored)
        except RuntimeError: pass
        else: raise RuntimeError('Resume feature corruption guard failed.')
        del restored, array
    return dict(unsorted_head_order=True, original_cache_mapping=True, hidden_states_bitexact=True,
        writes_disabled_during_hook_check=True, zero_vectors_retained=True, unwritten_nan_rejected=True,
        duplicate_and_missing_hook_guards=True, cache_ownership_guard=True, resume_prefix_and_bytes_verified=True,
        bf16_to_f16_overflow_guard=True,
        portable_probability_max_difference=parity['max_difference'], cpu_only=True)


def preflight():
    checks = cpu_checks(); tokenizer = long_data.load_tokenizer(); bundle = extraction_data(tokenizer)
    # Verify all padded token-to-row mappings, with no model or CUDA allocation.
    visited = np.zeros(ROWS, dtype=bool)
    for part in bundle.parts:
        batch = bundle.collate(part)
        require(not visited[batch['targets']].any(), 'Original cache row appears in multiple batches.')
        visited[batch['targets']] = True
    require(visited.all(), 'Preflight batching missed an original development row.')
    folder = Path(tempfile.mkdtemp(prefix='systemness-head-extract-preflight-', dir='/tmp'))
    report = dict(source_sha256=data.digest(__file__), checks=checks, data=bundle.descriptor(), resources=resources(bundle),
        original_bank=original_bank(), input_sources=input_sources(), scanner_scan_complete=(scanner.OUT / 'scan-complete.json').exists(),
        selected_identity=selected_identity() if (scanner.OUT / 'scan-complete.json').exists() else None,
        all_padded_token_mappings_checked=True, formal_output_written=False, formal_cache_created=False,
        model_loaded=False, gpu_work_run=False, tests_read=False, confirmation_inputs_read=False)
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(report_path=str(folder / 'report.json'), **report)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='CPU fixtures and complete source/row checks; /tmp output only.')
    action.add_argument('--stage', choices=['extract'], help='Explicitly extract on four A40s after the scanner exits.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.stage: run()
    else: parser.print_help()
