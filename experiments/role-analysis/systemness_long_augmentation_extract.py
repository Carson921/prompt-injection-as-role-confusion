"""Extract frozen ranked-eight-head features for new long development sources.

Default: help. --preflight is synthetic CPU work under /tmp only. The separate
data builder prepares the new256 fit and96 validation documents. --stage declare
freezes metadata, source code, native MXFP4 weights and ordered heads without
loading prompts or a model. --stage extract requires that prior declaration.
This module performs no readout training, model selection or test evaluation.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_head_extract as legacy
import systemness_head_scan as scanner
import systemness_long_data as long_data


OUT = data.runtime.OUT / 'systemness-long-augmentation'
CACHE = Path('/dev/shm/role-analysis-long-augmentation-heads')
ROWS, DIMENSIONS, HEAD_DIM, TOP_K = 389120, 512, 64, 8
FIT_ROWS, VALID_ROWS, PROMPTS, DOCUMENTS = 327680, 61440, 1760, 352
BATCH_SIZE, CHECKPOINT_INTERVAL = 4, 16
SAMPLE_START, DOCUMENT_START = 6000000000000, 6000000
PARITY_TOLERANCE = legacy.PARITY_TOLERANCE
RESERVE_BYTES, MIN_GPU_FREE_BYTES = 512 * 2**20, 22 * 2**30
PLAN_NAME, CPU_NAME = 'extraction-plan.json', 'extraction-cpu-preflight.json'
HOOK_NAME, STATUS_NAME = 'extraction-hook-validation.json', 'extraction-status.json'
DEPENDENCIES = sorted(set(legacy.DEPENDENCIES + ['systemness_long_augmentation_extract.py',
    'systemness_long_augmentation_data.py']))
require = long_data.require
read = legacy.read
array_digest = legacy.array_digest
frame_digest = legacy.frame_digest
Collector = legacy.Collector
forward = legacy.forward
initialize = legacy.initialize
batch_digest = legacy.batch_digest
prefix_state = legacy.prefix_state
save_progress = legacy.save_progress
restore_progress = legacy.restore_progress
inspect_features = legacy.inspect_features
portable_consumer = legacy.portable_consumer
fsync_file = legacy.fsync_file


def builder():
    # Lazy import: default help/preflight never invoke the new-source builder.
    import systemness_long_augmentation_data as module
    return module


def protocol_spec():
    return dict(version=1, experiment='systemness-long-augmentation-head-extraction',
        data_builder='systemness_long_augmentation_data.py', documents=DOCUMENTS,
        fit_documents=256, validation_documents=96, prompts=PROMPTS, rows=ROWS,
        fit_rows=FIT_ROWS, validation_rows=VALID_ROWS, roles=data.ROLES,
        body_tokens=1024, fit_sampled_tokens_per_role=256, validation_sampled_tokens_per_role=128,
        feature_shape=[ROWS, DIMENSIONS], dtype='float16', raw_feature_bytes=feature_bytes(),
        selected_heads='Unchanged ranked eight pre-o_proj query heads from frozen scanner; concatenate in selected rank order.',
        inference='One sequential frozen nativeMXFP4 forward per batch on exactly four A40s. No residual/local-attention legacy capture.',
        batch_size=BATCH_SIZE, checkpoint_interval=CHECKPOINT_INTERVAL,
        hooks='Prehooks return None. Compare final hidden states bitwise with writes disabled, proving no bank writes during self-check.',
        portability='Same saved F16 vectors through NumPy and GPU float64 frozen per-head LDA probabilities; max absolute difference <=5e-5.',
        restart='Atomic NaN initialization, byte-copied row table, fsynced exact prompt-prefix checkpoints and independent committed-batch SHA checks.',
        cache_path=str(CACHE), reserve_bytes=RESERVE_BYTES,
        new_document_id_start=DOCUMENT_START, new_sample_ix_start=SAMPLE_START,
        features_only=True, readout_training=False, tests_read=False,
        old_feature_banks_deleted=False, frozen_primary_changed=False)


def feature_bytes(rows=ROWS, dimensions=DIMENSIONS):
    return rows * dimensions * np.dtype(np.float16).itemsize


def verified_metadata():
    """Metadata/cache hashes only; no tokenizer, source text or probabilities."""
    module = builder()
    require(module.OUT == OUT and module.CACHE == data.CACHE / 'long-augmentation', 'Builder output/cache namespace changed.')
    metadata = read(OUT / 'data-complete.json')
    plan = read(OUT / 'plan.json')
    require(metadata['source_sha256'] == data.digest(data.HERE / 'systemness_long_augmentation_data.py')
        and metadata['plan_sha256'] == data.digest(OUT / 'plan.json')
        and metadata['document_keys_sha256'] == data.digest(OUT / 'document-keys.json')
        and metadata['source_signature'] == module.source_signature(), 'Builder data/source/plan/keys changed.')
    require(plan['source_signature'] == metadata['source_signature'] and plan['protocol'] == module.protocol_spec()
        and plan['source_sha256'] == metadata['source_sha256']
        and plan['dependencies_sha256'] == metadata['dependencies_sha256'], 'Builder declared protocol differs from completed data.')
    require(all(data.digest(data.HERE / name) == value for name, value in metadata['dependencies_sha256'].items()),
        'Builder dependency changed after preparation.')
    require(metadata['cache_path'] == str(module.CACHE)
        and set(metadata['cache_sha256']) == {'inputs.parquet', 'rows.parquet', 'token-ids.json'}
        and set(metadata['metadata_sha256']) == {'document-keys.json', 'exclusion-audit.json',
            'scan-audit.json', 'independent-validation.json'}, 'Builder cache/audit declaration changed.')
    for name, digest in metadata['cache_sha256'].items():
        require(data.digest(module.CACHE / name) == digest, 'Builder cache bytes changed: ' + name)
    for name, digest in metadata['metadata_sha256'].items():
        require(data.digest(OUT / name) == digest, 'Builder audit bytes changed: ' + name)
    require(read(module.CACHE / 'initialized.json') == dict(plan_sha256=metadata['plan_sha256'],
        source_sha256=metadata['source_sha256']), 'Builder cache ownership changed.')
    require(metadata['inputs'] == PROMPTS and metadata['rows'] == ROWS
        and metadata['fit_rows'] == FIT_ROWS and metadata['validation_rows'] == VALID_ROWS
        and metadata['documents'] == {'fit': 256, 'validation': 96}
        and metadata['independent_native_labels_match'] is True and metadata['all_body_tokens1024'] is True
        and metadata['test_inputs_prepared'] is False and metadata['test_probabilities_read'] is False
        and metadata['test_metrics_read'] is False and metadata['gpu_work_run'] is False
        and metadata['model_generation_run'] is False and metadata['downloads_run'] is False,
        'Builder coverage or development-only boundary changed.')
    return metadata


def source_signature():
    """CPU metadata-only binding; never loads prompts, features or probabilities."""
    metadata = verified_metadata()
    selection = legacy.selected_identity()
    checkpoint, tokenizer_sha = long_data.tokenizer_assets()
    config = read(checkpoint / 'config.json')
    require(config['quantization_config']['quant_method'] == 'mxfp4'
        and (config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (36, 64, 64),
        'Frozen nativeMXFP4 geometry changed.')
    return dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        protocol=protocol_spec(), selection=selection,
        data_complete_sha256=data.digest(OUT / 'data-complete.json'),
        data_plan_sha256=metadata['plan_sha256'], document_keys_sha256=metadata['document_keys_sha256'],
        builder_source_signature=metadata['source_signature'], builder_dependencies_sha256=metadata['dependencies_sha256'],
        input_cache_path=metadata['cache_path'], input_cache_sha256=metadata['cache_sha256'],
        checkpoint_path=str(checkpoint), verified_weights_sha256=data.digest(checkpoint / 'verified-weights.json'),
        config_sha256=data.digest(checkpoint / 'config.json'), tokenizer_sha256=tokenizer_sha,
        checkpoint_verification_sha256=data.digest(data.runtime.OUT / 'checkpoint-cache-verification.json'),
        package_versions={name: importlib.metadata.version(name) for name in
            ['torch', 'transformers', 'accelerate', 'numpy', 'pandas']},
        kernel_revision=data.runtime.KERNEL_REVISION,
        rows=ROWS, dimensions=DIMENSIONS, dtype='float16', cache_path=str(CACHE))


def validate_layout(inputs, rows, tokens=None, *, expected=None):
    expected = expected or dict(rows=ROWS, prompts=PROMPTS, documents=DOCUMENTS,
        fit_rows=FIT_ROWS, validation_rows=VALID_ROWS, fit_documents=256, validation_documents=96,
        fit_positions=256, validation_positions=128)
    require(len(inputs) == expected['prompts'] and len(rows) == expected['rows']
        and inputs.prompt_ix.is_unique and inputs.uid.is_unique
        and np.array_equal(inputs.prompt_ix, np.arange(len(inputs)))
        and np.array_equal(rows.index, np.arange(len(rows)))
        and np.array_equal(rows.cache_ix, np.arange(len(rows))) and rows.sample_ix.is_unique
        and np.array_equal(rows.sample_ix, SAMPLE_START + np.arange(len(rows))), 'New feature-bank row/prompt namespace changed.')
    require(set(rows.split) == set(inputs.split) == {'fit', 'validation'}
        and set(rows.wrapper) == {'native'} and set(rows.profile) == set(inputs.profile) == {'long-full'}
        and set(rows.source_set) == set(inputs.source_set) == {'long-augmentation'}
        and set(rows.role) == set(inputs.role) == set(data.ROLES)
        and rows.document_id.ge(DOCUMENT_START).all() and inputs.document_id.ge(DOCUMENT_START).all(),
        'Only new native long-full fit/validation sources are permitted.')
    require(set(inputs.uid) == set(rows.uid)
        and inputs.uid.eq('long-augmentation:' + inputs.prompt_ix.astype(str)).all()
        and not rows.duplicated(['prompt_ix', 'token_ix']).any()
        and rows.groupby('split').size().to_dict() == {'fit': expected['fit_rows'], 'validation': expected['validation_rows']},
        'New source membership/counts or content positions changed.')
    require(rows.document_id.nunique() == expected['documents']
        and set(rows.document_id) == set(DOCUMENT_START + np.arange(expected['documents']))
        and rows[rows.split.eq('fit')].document_id.nunique() == expected['fit_documents']
        and rows[rows.split.eq('validation')].document_id.nunique() == expected['validation_documents']
        and not set(rows.loc[rows.split.eq('fit'), 'group_id']) & set(rows.loc[rows.split.eq('validation'), 'group_id'])
        and not set(rows.loc[rows.split.eq('fit'), 'document_id']) & set(rows.loc[rows.split.eq('validation'), 'document_id']),
        'New fit/validation source partition changed.')
    require(inputs.body_tokens.eq(1024).all()
        and rows.body_offset.between(0, 1023).all(), 'Every new body must have exactly1024 tokens.')
    lookup = inputs.set_index('prompt_ix')
    for column in ['uid', 'split', 'document_id', 'group_id', 'role', 'profile', 'source_set']:
        require(np.array_equal(lookup.loc[rows.prompt_ix, column].to_numpy(), rows[column].to_numpy()),
            'Input/row mapping changed: ' + column)
    require(rows.groupby('document_id').role.nunique().eq(5).all(), 'A new source lacks five role variants.')
    groups = {int(key): group for key, group in rows.groupby('prompt_ix', sort=False)}
    if tokens is not None: require(len(tokens) == len(inputs), 'Cached token-list count changed.')
    for entry in inputs.itertuples():
        group = groups[entry.prompt_ix]
        count = expected['fit_positions'] if entry.split == 'fit' else expected['validation_positions']
        offsets = data.uniform_indices(1024, count)
        require(len(group) == count and np.array_equal(group.body_offset.to_numpy(), offsets)
            and np.all(np.diff(group.token_ix.to_numpy()) == np.diff(offsets))
            and group.token_ix.between(0, entry.length - 1).all(), 'Body-offset sampling/position mapping changed.')
        if tokens is not None:
            ids = tokens[entry.prompt_ix]
            require(len(ids) == entry.length and np.array_equal(np.asarray(ids)[group.token_ix.to_numpy()], group.token_id),
                'Cached native token identity changed.')


class Bundle:
    """Sorted batching over cached tokens, preserving bank-local cache_ix."""
    def __init__(self, inputs, rows, tokens=None):
        self.inputs = inputs.sort_values(['length', 'uid']).reset_index(drop=True)
        self.rows = rows.copy().reset_index(drop=True)
        self.tokens = None if tokens is None else {entry.uid: tokens[entry.prompt_ix] for entry in inputs.itertuples()}
        self.lookup = {uid: group for uid, group in self.rows.groupby('uid', sort=False)}
        require(self.inputs.uid.is_unique and set(self.inputs.uid) == set(self.lookup), 'Bundle prompt membership mismatch.')
        self.parts = [self.inputs.iloc[start:start + BATCH_SIZE] for start in range(0, len(self.inputs), BATCH_SIZE)]

    def targets(self, part):
        return np.concatenate([self.lookup[entry.uid].cache_ix.to_numpy(dtype=np.int64) for entry in part.itertuples()])

    def collate(self, part):
        import torch
        require(self.tokens is not None, 'Metadata-only bundle cannot run inference.')
        width = int(part.length.max()); ids = torch.zeros((len(part), width), dtype=torch.long)
        mask = torch.zeros_like(ids); positions = []
        for index, entry in enumerate(part.itertuples()):
            tokens = self.tokens[entry.uid]; size = len(tokens)
            require(size == entry.length, 'Cached native input length changed.')
            ids[index, -size:] = torch.tensor(tokens, dtype=torch.long); mask[index, -size:] = 1
            positions.extend((self.lookup[entry.uid].token_ix.to_numpy() + index * width + width - size).tolist())
        targets = self.targets(part); positions = np.asarray(positions, dtype=np.int64)
        require(len(set(targets)) == len(targets) and np.array_equal(
            ids.reshape(-1)[positions].numpy(), self.rows.iloc[targets].token_id.to_numpy()),
            'Left-padded source positions do not match bank row token identity.')
        return dict(input_ids=ids, attention_mask=mask, positions=positions, targets=targets)

    def descriptor(self):
        return dict(rows=len(self.rows), prompts=len(self.inputs), batches=len(self.parts),
            split_rows={key: int(value) for key, value in self.rows.groupby('split').size().items()},
            source_rows={key: int(value) for key, value in self.rows.groupby('source_set').size().items()},
            input_tokens=int(self.inputs.length.sum()), max_batch_rows=max(len(self.targets(part)) for part in self.parts),
            inputs_sha256=frame_digest(self.inputs, ['uid', 'prompt_ix', 'length']),
            row_mapping_sha256=frame_digest(self.rows, ['cache_ix', 'sample_ix', 'uid', 'token_ix', 'token_id', 'role', 'body_offset']),
            sample_ix_sha256=array_digest(self.rows.sample_ix.to_numpy()))


def metadata_bundle():
    metadata = verified_metadata(); folder = Path(metadata['cache_path'])
    # Column projection deliberately excludes prompt strings and token spelling.
    inputs = pd.read_parquet(folder / 'inputs.parquet', columns=['prompt_ix', 'uid', 'length', 'split',
        'document_id', 'group_id', 'role', 'profile', 'source_set', 'body_tokens'])
    rows = pd.read_parquet(folder / 'rows.parquet', columns=['cache_ix', 'sample_ix', 'prompt_ix', 'uid',
        'split', 'document_id', 'group_id', 'role', 'profile', 'source_set', 'wrapper',
        'token_ix', 'token_id', 'body_offset'])
    validate_layout(inputs, rows)
    return Bundle(inputs, rows)


def load_data():
    metadata, inputs, rows, tokens = builder().load_verified_data()
    require(metadata == verified_metadata(), 'Loaded builder completion differs from frozen metadata.')
    validate_layout(inputs, rows, tokens)
    return Bundle(inputs, rows, tokens)


def declare():
    signature = source_signature(); bundle = metadata_bundle()
    plan = dict(version=1, signature=signature, source_sha256=signature['source_sha256'],
        dependencies_sha256=signature['dependencies_sha256'], protocol=protocol_spec(), data=bundle.descriptor(),
        selected_head_ids=signature['selection']['selected_head_ids'],
        ordered_layers_heads=signature['selection']['ordered_layers_heads'],
        rows=ROWS, dimensions=DIMENSIONS, head_dim=HEAD_DIM, dtype='float16', roles=data.ROLES,
        cpu_preflight=cpu_checks(), tests_read=False, readout_training=False)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / PLAN_NAME
    if path.exists(): require(read(path) == plan, 'Extraction declaration changed; refusing overwrite.')
    else: data.atomic_json(path, plan)
    record = dict(source_sha256=plan['source_sha256'], plan_sha256=data.digest(path), checks=plan['cpu_preflight'],
        data=plan['data'], model_loaded=False, gpu_work_run=False, new_source_text_read=False)
    path = OUT / CPU_NAME
    if path.exists(): require(read(path) == record, 'Extraction CPU declaration changed.')
    else: data.atomic_json(path, record)
    return plan


def verify_plan(plan):
    require(plan.get('signature') == source_signature() and plan['source_sha256'] == plan['signature']['source_sha256']
        and plan['dependencies_sha256'] == plan['signature']['dependencies_sha256']
        and plan['protocol'] == protocol_spec() and read(OUT / PLAN_NAME) == plan,
        'Extraction protocol/source/data/head/weights changed after declaration.')
    record = read(OUT / CPU_NAME)
    require(record['plan_sha256'] == data.digest(OUT / PLAN_NAME) and record['source_sha256'] == plan['source_sha256']
        and record['checks'] == plan['cpu_preflight'] and record['data'] == plan['data'], 'Extraction CPU declaration changed.')


def check_feature_file(path, rows=ROWS, dimensions=DIMENSIONS):
    require(Path(path).is_file() and Path(path).stat().st_size == feature_bytes(rows, dimensions),
        'Feature file absent/truncated/wrong shape; refusing fabricated resume.')


def verified_extraction():
    """CPU-only full completion/row/feature/prefix verification for readouts."""
    plan = read(OUT / PLAN_NAME); verify_plan(plan)
    complete = read(OUT / 'extraction-complete.json'); plan_sha = data.digest(OUT / PLAN_NAME)
    require(complete['plan_sha256'] == plan_sha and complete['signature'] == plan['signature']
        and complete['source_sha256'] == plan['source_sha256']
        and complete['dependencies_sha256'] == plan['dependencies_sha256']
        and complete['rows'] == ROWS and complete['dimensions'] == DIMENSIONS
        and complete['feature_shape'] == [ROWS, DIMENSIONS] and complete['dtype'] == 'float16'
        and complete['all_rows_written'] is True and complete['all_finite'] is True
        and complete['tests_read'] is False and complete['readout_training'] is False,
        'Completed new bank differs from its extraction declaration.')
    require(complete['rows_path'] == str(CACHE / 'rows.parquet')
        and complete['features_path'] == str(CACHE / 'features.f16')
        and complete['rows_sha256'] == plan['signature']['input_cache_sha256']['rows.parquet'] == data.digest(CACHE / 'rows.parquet')
        and complete['selected_head_ids'] == plan['selected_head_ids']
        and complete['ordered_layers_heads'] == plan['ordered_layers_heads'] and complete['head_dim'] == HEAD_DIM,
        'Completed new bank paths/rows/head rank changed.')
    check_feature_file(CACHE / 'features.f16', rows=ROWS, dimensions=DIMENSIONS)
    require(complete['features_sha256'] == data.digest(CACHE / 'features.f16')
        and complete['hook_validation_sha256'] == data.digest(OUT / HOOK_NAME)
        and complete['model_backend_sha256'] == data.digest(OUT / 'augmentation-model_backend.json')
        and complete['progress_sha256'] == data.digest(CACHE / 'progress.json'), 'Completed bank/audit bytes changed.')
    require(complete['hook_check_disabled_writes'] is True and complete['hidden_states_bitexact'] is True
        and np.isfinite(complete['portable_probability_max_difference'])
        and complete['portable_probability_max_difference'] <= PARITY_TOLERANCE,
        'Completed bank lacks disabled-write/bitexact/portable checks.')
    bundle = metadata_bundle(); require(bundle.descriptor() == plan['data'] == complete['data'], 'Completed row descriptor changed.')
    require(data.digest(CACHE / 'rows.parquet') == plan['signature']['input_cache_sha256']['rows.parquet'], 'Copied row bytes changed.')
    require(read(CACHE / 'initialized.json') == dict(plan_sha256=plan_sha, rows=ROWS, dimensions=DIMENSIONS,
        dtype='float16', rows_sha256=complete['rows_sha256']), 'Completed cache initialization ownership changed.')
    array = np.memmap(CACHE / 'features.f16', mode='r', dtype=np.float16, shape=(ROWS, DIMENSIONS))
    hashes, parity = restore_progress(CACHE, bundle, plan_sha, array)
    require(len(hashes) == len(bundle.parts) and prefix_state(bundle, len(hashes))['rows_written'] == ROWS
        and parity == complete['portable_consumer'], 'Completed extraction prefix/portable record is incomplete.')
    require(inspect_features(array) == {key: complete[key] for key in ['all_finite', 'all_rows_nonzero', 'zero_row_count']},
        'Completed finite/zero-vector diagnostics changed.')
    del array
    return complete


@contextmanager
def checked_hooks(model, collector, bundle, device='cuda:0'):
    """Local audit output and exception-safe registration; no legacy OUT writes."""
    import torch
    batch = bundle.collate(bundle.parts[0]); old = collector.array[batch['targets']].copy(); handles = []
    with torch.inference_mode():
        before = model.model(input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device), use_cache=False).last_hidden_state.cpu()
        try:
            for layer in collector.layers:
                handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)))
            collector.enabled = False; after = forward(model, collector, batch, device).cpu()
            require(torch.equal(before, after) and bool(torch.isfinite(after).all()), 'Capture hooks changed final hidden states.')
            require(np.array_equal(old, collector.array[batch['targets']], equal_nan=True), 'Disabled hook wrote new-bank features.')
            data.atomic_json(OUT / HOOK_NAME, dict(hidden_states_bitexact=True, writes_disabled=True,
                feature_rows_unchanged=True, layers=collector.layers, sample_rows=len(batch['targets'])))
            del before, after, old
            collector.enabled = True
            yield
        finally:
            collector.enabled = False
            for handle in handles: handle.remove()


def resources(bundle=None):
    return dict(rows=ROWS, dimensions=DIMENSIONS, feature_bytes=feature_bytes(),
        reserve_bytes=RESERVE_BYTES, minimum_gpu_free_bytes_before_load=MIN_GPU_FREE_BYTES,
        cache_path=str(CACHE), no_old_cache_deleted=True,
        max_batch_rows=None if bundle is None else bundle.descriptor()['max_batch_rows'])


def no_competing_jobs():
    names = {'systemness_long_extract.py', 'systemness_recover_local_bank.py', 'systemness_long_confirm.py',
        'systemness_attention_confirm.py', 'systemness_long_probes.py', 'systemness_paired_probes.py',
        'systemness_natural_recheck.py', 'systemness_head_scan.py', 'systemness_head_extract.py',
        'systemness_head_probes.py', 'systemness_head_confirm.py', 'systemness_head_fusion.py',
        'systemness_natural_test_confirm.py', 'systemness_long_augmentation_extract.py', 'systemness_long_augmentation_probes.py'}
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
    with (OUT / '.extraction.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another augmentation extractor holds this output.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_model():
    import torch
    require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Exactly4 A40s required.')
    available = [int(torch.cuda.mem_get_info(i)[0]) for i in range(4)]
    require(min(available) >= MIN_GPU_FREE_BYTES, 'Fewer than22GiB free per A40 before model loading.')
    data.atomic_json(OUT / 'extraction-runtime-resources.json', dict(gpu_free_bytes_before_load=available,
        minimum_gpu_free_bytes=MIN_GPU_FREE_BYTES, shared_memory_free_bytes=shutil.disk_usage(CACHE.parent).free))
    old_atomic, old_status = data.runtime.atomic_json, data.runtime.status
    data.runtime.atomic_json = lambda path, value: data.atomic_json(OUT / ('augmentation-' + Path(path).name), value)
    data.runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
    try: model = data.runtime.load_model()
    finally: data.runtime.atomic_json, data.runtime.status = old_atomic, old_status
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()), 'Native LM must stay frozen/eval.')
    return model


def run():
    require((OUT / PLAN_NAME).is_file(), 'Run --stage declare and freeze its source/data signature before extraction.')
    plan = read(OUT / PLAN_NAME); verify_plan(plan)
    with run_lock():
        verify_plan(plan)
        if (OUT / 'extraction-complete.json').exists():
            print(json.dumps(verified_extraction()), flush=True); return
        no_competing_jobs()
        bundle = load_data(); require(bundle.descriptor() == plan['data'], 'Loaded batching differs from prior declaration.')
        source_rows = Path(plan['signature']['input_cache_path']) / 'rows.parquet'
        need = RESERVE_BYTES if (CACHE / 'initialized.json').exists() else feature_bytes() + source_rows.stat().st_size + RESERVE_BYTES
        require(shutil.disk_usage(CACHE.parent).free > need, 'Insufficient shared memory for new bank plus512MiB reserve; old banks are retained.')
        array = initialize(CACHE, source_rows, data.digest(OUT / PLAN_NAME), rows=ROWS, dimensions=DIMENSIONS)
        hashes, parity = restore_progress(CACHE, bundle, data.digest(OUT / PLAN_NAME), array)
        selected = np.asarray(plan['selected_head_ids'], dtype=np.int64)
        with np.load(plan['signature']['selection']['model_path'], allow_pickle=False) as archive:
            weight, bias = archive['weight'].copy(), archive['bias'].copy()
        collector = Collector(array, selected); no_competing_jobs(); model = load_model()
        import torch
        with checked_hooks(model, collector, bundle), torch.inference_mode():
            for index in range(len(hashes), len(bundle.parts)):
                batch = bundle.collate(bundle.parts[index]); result = forward(model, collector, batch); del result
                hashes.append(batch_digest(array, batch['targets']))
                if parity is None: parity = portable_consumer(array, batch['targets'], weight, bias, 'cuda:0')
                if len(hashes) % CHECKPOINT_INTERVAL == 0 or len(hashes) == len(bundle.parts):
                    array.flush(); progress = save_progress(CACHE, bundle, data.digest(OUT / PLAN_NAME), hashes, parity)
                    status = dict(stage='extract', batch=len(hashes), batches=len(bundle.parts), rows_written=progress['rows_written'],
                        maximum_f16_roundtrip_difference=collector.maximum_f16_roundtrip_difference)
                    data.atomic_json(OUT / STATUS_NAME, status); print(json.dumps(status), flush=True)
        state = prefix_state(bundle, len(hashes))
        require(state['rows_written'] == ROWS and len(hashes) == len(bundle.parts), 'Incomplete new sampled-token coverage.')
        checks = inspect_features(array); array.flush(); fsync_file(CACHE / 'features.f16'); verify_plan(plan)
        hook = read(OUT / HOOK_NAME)
        complete = dict(signature=plan['signature'], source_sha256=plan['source_sha256'], dependencies_sha256=plan['dependencies_sha256'],
            plan_sha256=data.digest(OUT / PLAN_NAME), rows=ROWS, dimensions=DIMENSIONS, feature_shape=[ROWS, DIMENSIONS], dtype='float16',
            rows_path=str(CACHE / 'rows.parquet'), features_path=str(CACHE / 'features.f16'),
            rows_sha256=data.digest(CACHE / 'rows.parquet'), features_sha256=data.digest(CACHE / 'features.f16'),
            selected_head_ids=selected.tolist(), ordered_layers_heads=plan['ordered_layers_heads'], head_dim=HEAD_DIM,
            data=bundle.descriptor(), all_rows_written=True, **checks,
            hook_check_disabled_writes=hook['writes_disabled'], hidden_states_bitexact=hook['hidden_states_bitexact'],
            hook_validation_sha256=data.digest(OUT / HOOK_NAME), model_backend_sha256=data.digest(OUT / 'augmentation-model_backend.json'),
            portable_probability_max_difference=parity['max_difference'], portable_consumer=parity,
            maximum_f16_roundtrip_difference_this_run=collector.maximum_f16_roundtrip_difference,
            quantization_note='Parity compares identical saved F16 vectors. Roundtrip maximum covers only this process, not earlier resumed processes.',
            progress_sha256=data.digest(CACHE / 'progress.json'), tests_read=False, readout_training=False,
            frozen_primary_changed=False, old_feature_banks_deleted=False)
        data.atomic_json(OUT / 'extraction-complete.json', complete)
        verified_extraction()
        data.atomic_json(OUT / STATUS_NAME, dict(stage='extraction_complete', rows=ROWS, dimensions=DIMENSIONS, tests_read=False))
        del collector, array, model; gc.collect()


def synthetic_data():
    """Full-sized source-free mapping fixture, with varying native prefix lengths."""
    inputs, blocks, tokens, cache_ix = [], [], [], 0
    for document in range(DOCUMENTS):
        split = 'fit' if document < 256 else 'validation'; count = 256 if split == 'fit' else 128
        for role_ix, role in enumerate(data.ROLES):
            prompt_ix = len(inputs); bos = (document + role_ix) % 3
            body_start = 7 + bos + role_ix; ids = list(range(100 + body_start + 1024 + 2))
            common = dict(prompt_ix=prompt_ix, uid=f'long-augmentation:{prompt_ix}', split=split,
                profile='long-full', source_set='long-augmentation', group_id=f'new-{document}',
                document_id=DOCUMENT_START + document, role=role)
            inputs.append(dict(common, length=len(ids), body_tokens=1024)); tokens.append(ids)
            offset = data.uniform_indices(1024, count); positions = body_start + offset
            blocks.append(pd.DataFrame(dict(common, cache_ix=np.arange(cache_ix, cache_ix + count),
                sample_ix=SAMPLE_START + np.arange(cache_ix, cache_ix + count), wrapper='native',
                token_ix=positions, token_id=np.asarray(ids)[positions], body_offset=offset)))
            cache_ix += count
    return pd.DataFrame(inputs), pd.concat(blocks, ignore_index=True), tokens


def cpu_checks():
    """Tiny persistence/math fixtures. No builder/cache/model/real-text access."""
    checks = legacy.cpu_checks()
    require(checks['cpu_only'], 'Inherited selected-head CPU fixture crossed its boundary.')
    checks['feature_bytes_exact'] = feature_bytes() == 398458880
    require(checks['feature_bytes_exact'], 'New bank storage formula changed.')
    return checks


def local_cpu_checks(folder):
    """Exercise this module's hook cleanup and public completion API on toys."""
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    checks = {}
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
            for layer in self.layers:
                layer.self_attn = torch.nn.Module(); layer.self_attn.o_proj = torch.nn.Identity()
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            value = input_ids
            for layer in self.layers: value = layer.self_attn.o_proj(value + 1)
            return SimpleNamespace(last_hidden_state=value)
    class TinyBundle:
        parts = [None]
        def collate(self, part):
            return dict(input_ids=torch.arange(24, dtype=torch.float32).reshape(1, 2, 12),
                attention_mask=torch.ones(1, 2), positions=np.array([1, 0]), targets=np.array([0, 1]))
    model = SimpleNamespace(model=Body()); array = np.full((2, 8), np.nan, dtype=np.float16)
    collector = Collector(array, [5, 0], heads=3, dimensions=4)
    hook_out = folder / 'hooks'; hook_out.mkdir()
    with patch(__name__ + '.OUT', hook_out):
        try:
            with checked_hooks(model, collector, TinyBundle(), 'cpu'):
                require(np.isnan(array).all(), 'Disabled local hook polluted feature bank.')
                forward(model, collector, TinyBundle().collate(None), 'cpu')
                raise ValueError('synthetic body failure')
        except ValueError: pass
        require(not collector.enabled and all(not layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers),
            'Hook handles leaked after body failure.')
        require(np.isfinite(array).all() and read(hook_out / HOOK_NAME)['writes_disabled'], 'Local hook audit is incomplete.')
        class Broken(torch.nn.Identity):
            def register_forward_pre_hook(self, *args, **kwargs): raise ValueError('synthetic registration failure')
        model.model.layers[1].self_attn.o_proj = Broken()
        try:
            with checked_hooks(model, collector, TinyBundle(), 'cpu'): pass
        except ValueError: pass
        else: raise RuntimeError('Synthetic hook registration failure was ignored.')
        require(all(not layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers), 'Partial registration leaked a hook.')
    checks['local_bitexact_disabled_writes_and_cleanup'] = True
    checks['partial_hook_registration_cleanup'] = True
    # Public completion verifier must be read-only, reject every binding fault,
    # and reconstruct all batch hashes rather than trusting final row counts.
    out, cache = folder / 'completion', folder / 'complete-bank'; out.mkdir()
    tiny_inputs = pd.DataFrame(dict(uid=['a', 'b'], prompt_ix=[0, 1], length=[2, 3]))
    tiny_rows = pd.DataFrame(dict(cache_ix=[0, 1, 2, 3], sample_ix=np.arange(SAMPLE_START, SAMPLE_START + 4),
        uid=['a', 'a', 'b', 'b'], token_ix=[0, 1, 1, 2], token_id=[1, 2, 3, 4],
        role=['system'] * 4, body_offset=[0, 1, 0, 1], split=['fit', 'fit', 'validation', 'validation'], source_set=['long-augmentation'] * 4))
    tiny = Bundle(tiny_inputs, tiny_rows)
    source = folder / 'source-rows.parquet'; tiny_rows.to_parquet(source, index=False)
    with patch.multiple(__name__, OUT=out, CACHE=cache, ROWS=4, DIMENSIONS=8, HEAD_DIM=4):
        signature = dict(source_sha256='fixture-source', dependencies_sha256={'fixture': 'hash'},
            input_cache_sha256={'rows.parquet': data.digest(source)}, input_cache_path=str(folder))
        plan = dict(signature=signature, source_sha256='fixture-source', dependencies_sha256=signature['dependencies_sha256'],
            protocol=protocol_spec(), cpu_preflight={'synthetic': True}, data=tiny.descriptor(),
            selected_head_ids=[5, 0], ordered_layers_heads=[[1, 2], [0, 0]])
        data.atomic_json(out / PLAN_NAME, plan)
        data.atomic_json(out / CPU_NAME, dict(plan_sha256=data.digest(out / PLAN_NAME), source_sha256=plan['source_sha256'],
            checks=plan['cpu_preflight'], data=plan['data']))
        array = initialize(cache, source, data.digest(out / PLAN_NAME), rows=4, dimensions=8)
        array[:] = np.arange(32).reshape(4, 8); array.flush()
        hashes = [batch_digest(array, tiny.targets(part)) for part in tiny.parts]
        parity = dict(max_difference=0., rows=4, same_saved_f16_vectors=True)
        save_progress(cache, tiny, data.digest(out / PLAN_NAME), hashes, parity)
        data.atomic_json(out / HOOK_NAME, dict(writes_disabled=True, hidden_states_bitexact=True))
        data.atomic_json(out / 'augmentation-model_backend.json', dict(expert_backends=['synthetic']))
        complete = dict(plan_sha256=data.digest(out / PLAN_NAME), signature=signature,
            source_sha256=plan['source_sha256'], dependencies_sha256=plan['dependencies_sha256'], rows=4, dimensions=8,
            feature_shape=[4, 8], dtype='float16', all_rows_written=True, all_finite=True, all_rows_nonzero=True, zero_row_count=0,
            tests_read=False, readout_training=False, rows_path=str(cache / 'rows.parquet'), features_path=str(cache / 'features.f16'),
            rows_sha256=data.digest(source), features_sha256=data.digest(cache / 'features.f16'),
            selected_head_ids=[5, 0], ordered_layers_heads=plan['ordered_layers_heads'], head_dim=4,
            hook_validation_sha256=data.digest(out / HOOK_NAME), model_backend_sha256=data.digest(out / 'augmentation-model_backend.json'),
            progress_sha256=data.digest(cache / 'progress.json'), hook_check_disabled_writes=True, hidden_states_bitexact=True,
            portable_probability_max_difference=0., portable_consumer=parity, data=plan['data'])
        data.atomic_json(out / 'extraction-complete.json', complete)
        with patch(__name__ + '.source_signature', return_value=signature), patch(__name__ + '.metadata_bundle', return_value=tiny), \
                patch(__name__ + '.initialize', side_effect=AssertionError('Read-only verifier initialized cache')):
            require(verified_extraction() == complete, 'Valid synthetic completion was rejected.')
            checks['public_verifier_readonly_complete_prefix'] = True
            for key, replacement in [('source_sha256', 'wrong'), ('features_sha256', 'wrong'), ('rows_sha256', 'wrong'),
                    ('selected_head_ids', [0, 5]), ('all_rows_written', False), ('hook_check_disabled_writes', False),
                    ('progress_sha256', 'wrong'), ('portable_probability_max_difference', float('nan'))]:
                bad = dict(complete); bad[key] = replacement; data.atomic_json(out / 'extraction-complete.json', bad)
                try: verified_extraction()
                except RuntimeError: pass
                else: raise RuntimeError('Completion tamper was accepted: ' + key)
            data.atomic_json(out / 'extraction-complete.json', complete)
            with patch(__name__ + '.source_signature', return_value=dict(signature, source_sha256='changed-source')):
                try: verified_extraction()
                except RuntimeError: pass
                else: raise RuntimeError('Changed upstream signature was accepted.')
            array[0, 0] += 1; array.flush()
            changed = dict(complete, features_sha256=data.digest(cache / 'features.f16'))
            data.atomic_json(out / 'extraction-complete.json', changed)
            try: verified_extraction()
            except RuntimeError: pass
            else: raise RuntimeError('Changed feature bytes with stale per-batch hashes were accepted.')
        del array
    checks['completion_eight_field_tampers_rejected'] = True
    checks['changed_upstream_signature_rejected'] = True
    checks['full_prefix_batch_digest_independently_rechecked'] = True
    return checks


def preflight():
    """Full389120-row synthetic mapping plus checkpoint/hook failure fixtures."""
    import torch
    folder = Path(tempfile.mkdtemp(prefix='systemness-long-augmentation-extract-preflight-', dir='/tmp'))
    initial_sha = data.digest(__file__); checks = cpu_checks(); checks.update(local_cpu_checks(folder))
    inputs, rows, tokens = synthetic_data(); validate_layout(inputs, rows, tokens)
    bundle = Bundle(inputs, rows, tokens); visited = np.zeros(ROWS, dtype=bool)
    for part in bundle.parts:
        batch = bundle.collate(part); require(not visited[batch['targets']].any(), 'Synthetic mapping repeats a bank row.')
        visited[batch['targets']] = True
    require(visited.all(), 'Synthetic mapping omits bank rows.')
    checks.update(all_389120_rows_mapped=True, all_1760_prompts_checked=True, fit_validation_disjoint=True,
        all_offsets_uniform_and_explicit=True, left_padding_and_bos_mapping=True, no_test_sources=True)
    tiny_inputs = pd.DataFrame(dict(uid=[f'fixture:{i}' for i in range(65)], prompt_ix=np.arange(65), length=np.arange(65) % 5 + 3))
    tiny_rows = pd.DataFrame(dict(cache_ix=np.arange(130), sample_ix=SAMPLE_START + np.arange(130),
        uid=np.repeat(tiny_inputs.uid.to_numpy(), 2), token_ix=np.tile([0, 1], 65),
        token_id=np.tile([1, 2], 65), role='system', body_offset=np.tile([0, 1], 65)))
    tiny = Bundle(tiny_inputs, tiny_rows)
    source = folder / 'rows.parquet'; tiny_rows.to_parquet(source, index=False)
    cache = folder / 'bank'; array = initialize(cache, source, 'fixture', rows=130, dimensions=8)
    array[:] = np.arange(130 * 8, dtype=np.float16).reshape(130, 8)
    hashes = [batch_digest(array, tiny.targets(part)) for part in tiny.parts]
    parity = dict(max_difference=0., rows=8, same_saved_f16_vectors=True)
    for cursor in [0, 1, 16, len(tiny.parts)]:
        save_progress(cache, tiny, 'fixture', hashes[:cursor], parity if cursor else None)
        restored, _ = restore_progress(cache, tiny, 'fixture', array)
        require(restored == hashes[:cursor] and prefix_state(tiny, cursor)['rows_written'] == min(cursor * 8, 130),
            'Checkpoint prefix0/1/16/full mapping failed.')
    checks['prefix_zero_one_sixteen_full'] = True
    save_progress(cache, tiny, 'fixture', hashes, parity)
    array[0, 0] += 1; array.flush()
    try: restore_progress(cache, tiny, 'fixture', array)
    except RuntimeError: pass
    else: raise RuntimeError('Committed-batch corruption was accepted.')
    checks['batch_corruption_rejected'] = True
    del array
    path = cache / 'features.f16'; path.rename(cache / 'missing.f16')
    for absent in [True, False]:
        if not absent: path.write_bytes(b'0')
        try: initialize(cache, source, 'fixture', rows=130, dimensions=8)
        except RuntimeError: pass
        else: raise RuntimeError('Missing/truncated initialized bank was accepted.')
    checks['missing_truncated_resume_rejected'] = True
    # Entry guards must fail before loading builder tokens or touching CUDA.
    from unittest.mock import patch
    with patch(__name__ + '.builder', side_effect=AssertionError('Builder used in synthetic guard')):
        with patch(__name__ + '.OUT', folder / 'undeclared'):
            try: run()
            except RuntimeError: pass
            else: raise RuntimeError('Undeclared extraction was accepted.')
    checks['undeclared_run_rejected_before_builder_gpu'] = True
    bad = rows.copy(); bad.loc[0, 'body_offset'] = 1
    try: validate_layout(inputs, bad, tokens)
    except RuntimeError: pass
    else: raise RuntimeError('Changed sampled offset was accepted.')
    checks['bad_body_offset_rejected'] = True
    require(data.digest(__file__) == initial_sha, 'Source changed during CPU preflight.')
    report = dict(source_sha256=initial_sha, checks=checks, data=bundle.descriptor(), resources=resources(bundle),
        protocol=protocol_spec(), synthetic_only=True, new_source_builder_called=False, new_source_text_read=False,
        formal_output_written=False, formal_cache_created=False, model_loaded=False, gpu_work_run=False, tests_read=False)
    data.atomic_json(folder / 'report.json', report)
    print(json.dumps(dict(report_path=str(folder / 'report.json'), **report)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Synthetic CPU checks only; all writes under /tmp.')
    action.add_argument('--stage', choices=['declare', 'extract'], help='Declare metadata first; extraction requires a frozen declaration.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.stage == 'declare': print(json.dumps(declare()), flush=True)
    elif args.stage == 'extract': run()
    else: parser.print_help()
