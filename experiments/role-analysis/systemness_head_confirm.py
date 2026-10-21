"""Conditional fresh-source confirmation of a promoted joint-head readout.

Default: help. --preflight uses temporary CPU fixtures only. Explicit --stage
prepare freezes all four models before sampling any new source. No model or
threshold is selected here. The existing primary is never modified by this file.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_components as components
import systemness_data as data
import systemness_evaluate as evaluation
import systemness_head_probes as head_probes
import systemness_long_confirm as legacy
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-head-confirmation'
CACHE = data.CACHE / 'head-confirmation'
NAMES = ['residual', 'base-attention', 'dense128', 'head-selected']
PRIMARY = 'head-selected'
PROFILES = ['single', 'multi', 'long-single', 'long-full']
SPECS = dict(data.SPECS, **attention.SPECS)
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
BATCH_SIZE, BOOTSTRAPS = 4, 2000
HASH_FIELDS = ['body_sha256', 'full_body_sha256', 'truncated_body_sha256',
    'short_body_sha256', 'prefix192_sha256', 'prefix1024_sha256', 'full_text_sha256']
DEPENDENCIES = sorted(set(legacy.DEPENDENCIES + head_probes.DEPENDENCIES + [
    'systemness_head_confirm.py', 'systemness_head_extract.py', 'systemness_head_scan.py']))
require = long_data.require
same = legacy.same
read_json = legacy.read_json
parse_labels = legacy.parse_labels


def protocol_spec():
    return dict(protocol_version=1, experiment='systemness-head-confirmation',
        selection='systemness-head-probes/selection.json', require_validation_promoted=True,
        models=NAMES, primary=PRIMARY, general_documents=400, additional_full_length_documents=64,
        independent_source_documents=464, general_long_subset_documents=64,
        full_length_profile='long-full', full_length_body_tokens=1024,
        source_revision='1588ec454efa1a09f29cd18ddd04fe05fc8653a2',
        source_sha256='1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f',
        no_test_tuning=True, native_system_tags_retained=True, all_content_tokens_scored=True,
        bootstrap_replicates=BOOTSTRAPS, bootstrap_seed=data.SEED + 11464,
        bootstrap_unit='Original source document, paired across four models within each profile.',
        gallery='12 fixed equally spaced general source IDs, five role panels by four frozen models, plus full-length position curve.',
        exclusion='Every prior document-key manifest, original249 full/body/prefix fingerprints, and latest464 confirmation source keys; URL/full/body/prefix192/prefix1024/shard-line checks.',
        inference='One sequential frozen native MXFP4 forward on4 A40s; sampled residual/local16/20/24 and frozen selected pre-o_proj heads; F16 capture then F32 standardization; probabilities only.',
        primary_changed=False, natural_dialogues=False,
        limitations='Pinned C4 shard; no semantic-near-duplicate/domain-disjointness claim. Original249 lacks recoverable URL/line provenance. 64 up-to1024 variants are paired subsets, not extra independent sources.')


def plan():
    value = read_json(OUT / 'protocol.json')
    same(value, protocol_spec(), 'New head confirmation protocol')
    return value


def require_promoted(selection):
    require(selection.get('promoted') is True and selection.get('candidate_is_guard_eligible') is True,
        'A completed guard-eligible, validation-promoted head candidate is required before any confirmation preparation.')
    require(selection.get('test_used_for_selection') is False and selection.get('confirmation_required') is True
        and selection.get('primary_changed') is False, 'Head selection lacks validation-only/fresh-confirmation provenance.')


def selections():
    # This reads development completion/probabilities only. No prior test
    # probability, metric or result is consulted to pick any of these models.
    head_complete = head_probes.verified_completion()
    head_path = head_probes.OUT / 'selection.json'
    selected = read_json(head_path); require_promoted(selected)
    same(head_complete['selection_sha256'], data.digest(head_path), 'Promoted head selection')
    residual_path = data.OUT / 'selection.json'; attention_path = attention.OUT / 'selection.json'
    records = {'residual': read_json(residual_path), 'base-attention': read_json(attention_path), PRIMARY: selected}
    require(all(records[k].get('test_used_for_selection') is False for k in records), 'A readout used test selection.')
    same(records['base-attention']['candidate'], 'mlp-h64', 'Published local-attention control')
    dense_plan = read_json(long_data.OUT / 'search-plan.json')
    dense = long_probes.candidate_record('dense-mlp128', dense_plan)
    records['dense128'] = dense
    paths = {'residual': data.OUT / 'models/selected.npz',
        'base-attention': Path(records['base-attention']['model_path']),
        'dense128': long_data.OUT / 'models/dense-mlp128.npz', PRIMARY: Path(selected['model_path'])}
    thresholds = {'residual': records['residual']['system_threshold'],
        'base-attention': records['base-attention']['system_threshold'],
        'dense128': dense['validation']['system_threshold'], PRIMARY: selected['system_threshold']}
    sources = {'residual': residual_path, 'base-attention': attention_path,
        'dense128': long_data.OUT / 'candidates/dense-mlp128.json', PRIMARY: head_path}
    for name, path in paths.items():
        require(not str(path).startswith('/workspace'), 'Do not access the unavailable /workspace mount.')
        expected = records[name]['selected_model_sha256' if name == 'residual' else 'model_sha256']
        same(data.digest(path), expected, f'{name} frozen portable model')
        require(np.isfinite(thresholds[name]) and 0 <= thresholds[name] <= 1.00001, f'Invalid {name} frozen threshold.')
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            same(meta['roles'], data.ROLES, f'{name} role order')
            if name == PRIMARY:
                same(meta['dimensions'], 512, 'Selected head model dimensions')
                same(meta['selected_head_ids'], selected['signature']['selected_head_ids'] if 'signature' in selected
                    else head_complete['signature']['selected_head_ids'], 'Selected head order')
            else:
                same(meta['feature_spec'], data.SPECS if name == 'residual' else attention.SPECS, f'{name} features')
    return paths, {k: float(v) for k, v in thresholds.items()}, sources, records, head_complete


def exclusion_inputs():
    # Include all known key inventories, even prepared-but-not-yet-reported ones.
    # The new directory is excluded from its own immutable restart inventory.
    paths = {data.OUT / 'document-keys.json', long_data.OUT / 'document-keys.json',
        data.runtime.OUT / 'feature-search/fresh/document-keys.json',
        data.runtime.OUT / 'training_inputs.parquet',
        data.runtime.OUT.parent / 'local-gptoss20b/raw_training_data.json',
        legacy.OUT / 'document-keys.json'}
    require((legacy.OUT / 'data-complete.json').is_file() and (legacy.OUT / 'complete.json').is_file(),
        'The latest464 confirmation must have completed and expose its source exclusion keys.')
    paths.update(p for p in data.runtime.OUT.rglob('document-keys.json') if OUT not in p.parents)
    require(all(p.is_file() for p in paths), 'A required prior source exclusion inventory is missing.')
    return {str(path): data.digest(path) for path in sorted(paths)}


def key_fingerprints(keys):
    hashes, urls, lines = set(), set(), set()
    for key in keys:
        require('shard_line' in key and type(key['shard_line']) is int and key['shard_line'] >= 0,
            'A prior C4 key lacks its pinned shard-line identity.')
        lines.add(key['shard_line'])
        if key.get('source_url'): urls.add(key['source_url'])
        values = [key[field] for field in HASH_FIELDS if key.get(field)]
        require(all(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
            for value in values), 'A source fingerprint is not a lowercase SHA256 digest.')
        hashes.update(values)
        require(any(key.get(field) for field in HASH_FIELDS), 'A prior C4 key lacks exact source fingerprints.')
    return hashes, urls, lines


def exclusion_sets(tok, inputs):
    # The frozen helper additionally recovers full/192/1024 fingerprints for
    # known pinned lines and for both original249 text inventories.
    hashes, urls, lines = legacy.exclusion_sets(tok, inputs)
    for path in inputs:
        if Path(path).name == 'document-keys.json':
            h, u, l = key_fingerprints(read_json(path)); hashes.update(h); urls.update(u); lines.update(l)
    return hashes, urls, lines


def frozen_snapshot():
    protocol = plan()
    paths, thresholds, sources, records, complete = selections()
    checkpoint, tokenizer_sha = long_data.tokenizer_assets()
    os.environ['ROLE_ANALYSIS_LOAD_PATH'] = str(checkpoint)
    if os.environ.get('HF_HOME', '').startswith('/workspace'): os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
    config = read_json(checkpoint / 'config.json')
    same(config['quantization_config']['quant_method'], 'mxfp4', 'Checkpoint precision')
    require((config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (36, 64, 64),
        'Expected original36x64x64 model geometry.')
    require(all(config['layer_types'][layer] == 'sliding_attention' for layer in [16, 20, 24]), 'Local attention controls changed.')
    frozen = dict(experiment=protocol['experiment'], models={k: data.digest(p) for k, p in paths.items()},
        thresholds=thresholds, selection_sha256={k: data.digest(p) for k, p in sources.items()},
        primary_candidate=records[PRIMARY]['candidate'], primary_validation_promoted=True,
        selected_head_ids=complete['signature']['selected_head_ids'],
        head_completion_sha256=data.digest(head_probes.OUT / 'complete.json'),
        head_search_plan_sha256=data.digest(head_probes.OUT / 'search-plan.json'),
        head_standardization_sha256=data.digest(head_probes.OUT / 'standardization.npz'),
        confirmation_plan_sha256=data.digest(OUT / 'protocol.json'),
        dependency_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        checkpoint_path=str(checkpoint), tokenizer_sha256=tokenizer_sha,
        checkpoint_config_sha256=data.digest(checkpoint / 'config.json'),
        verified_weights_sha256=data.digest(checkpoint / 'verified-weights.json'),
        prior_confirmation_complete_sha256=data.digest(legacy.OUT / 'complete.json'),
        exclusion_input_sha256=exclusion_inputs())
    return paths, thresholds, frozen


def declare_readouts():
    selections()  # Promotion gate precedes creating even a protocol/preparation file.
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists(): same(read_json(path), protocol_spec(), 'Declared protocol')
    else:
        require(not (OUT / 'data-complete.json').exists() and not CACHE.exists(), 'Undeclared partial confirmation state exists.')
        data.atomic_json(path, protocol_spec())
    _, _, frozen = frozen_snapshot()
    declaration = dict(frozen=frozen, tests_prepared=False, source_sha256=data.digest(__file__))
    path = OUT / 'frozen-readouts.json'
    if path.exists(): same(read_json(path), declaration, 'Pre-preparation four-readout freeze')
    else:
        require(not (OUT / 'data-complete.json').exists() and not CACHE.exists(), 'Test data exists before model freeze.')
        data.atomic_json(path, declaration)


def verify_frozen(metadata=None):
    paths, thresholds, frozen = frozen_snapshot()
    declaration = read_json(OUT / 'frozen-readouts.json')
    same(declaration, dict(frozen=frozen, tests_prepared=False, source_sha256=data.digest(__file__)), 'Four-readout declaration')
    if metadata is not None: same(metadata['frozen'], frozen, 'Frozen models/thresholds/source exclusions')
    return paths, thresholds, frozen


def append_input(messages, profile, doc, bos, tok, inputs, rows, tokens):
    start = len(rows)
    legacy.append_input(messages, profile, doc, bos, tok, inputs, rows, tokens)
    group = f'head-confirm-c4-{doc}'
    for row in rows[start:]: row['group_id'] = group
    inputs[-1]['group_id'] = group


def validate_rows(inputs, rows, tokens, keys):
    legacy.validate_rows(inputs, rows, tokens, keys)
    require((rows.group_id == 'head-confirm-c4-' + rows.document_id.astype(str)).all()
        and (inputs.group_id == 'head-confirm-c4-' + inputs.document_id.astype(str)).all(), 'New source namespace changed.')
    require(all(key['group_id'] == f'head-confirm-c4-{key["document_id"]}' for key in keys), 'New key namespace changed.')


def prepare():
    declare_readouts()
    _prepare_data()


def inference_signature(metadata):
    verify_frozen(metadata)
    return dict(data_sha256=data.digest(OUT / 'data-complete.json'), frozen=metadata['frozen'],
        frozen_declaration_sha256=data.digest(OUT / 'frozen-readouts.json'), specs=SPECS,
        selected_head_ids=metadata['frozen']['selected_head_ids'], batch_size=BATCH_SIZE,
        inference='One sequential shared four-A40 frozen nativeMXFP4 forward; sampled F16 features; four pointwise readouts; probability-only cache.')


def completed_probabilities(metadata, rows):
    complete = read_json(OUT / 'inference-complete.json')
    same(complete['signature'], inference_signature(metadata), 'Completed four-model inference signature')
    same(set(complete['probability_sha256']), set(NAMES), 'Four-model probability coverage')
    same(complete['rows'], len(rows), 'Completed four-model probability rows')
    same(complete['portable_validation_sha256'], data.digest(OUT / 'portable-validation.json'), 'Actual portable consumer evidence')
    same(complete['hook_validation_sha256'], data.digest(OUT / 'hook-validation.json'), 'Disabled-write hidden-state selfcheck')
    same(complete['model_backend_sha256'], data.digest(OUT / 'model_backend.json'), 'Actual native MXFP4 backend')
    portable = read_json(OUT / 'portable-validation.json'); hooks = read_json(OUT / 'hook-validation.json')
    same(portable['signature'], complete['signature'], 'Portable check inference signature')
    same(portable['model_sha256'], metadata['frozen']['models'], 'Portable checked models')
    require(set(portable['max_probability_difference']) == set(NAMES)
        and all(np.isfinite(v) and 0 <= v < 5e-4 for v in portable['max_probability_difference'].values()),
        'Portable probability evidence is missing, nonfinite or outside tolerance.')
    require(hooks['signature'] == complete['signature'] and hooks['hidden_states_bitexact']
        and hooks['all_capture_writes_disabled'], 'Combined hook selfcheck no longer verifies unmodified hidden states.')
    same(read_json(OUT / 'model_backend.json')['expert_backends'], ['Mxfp4GptOssExperts'], 'Actual native expert backend')
    values = open_probabilities(NAMES, len(rows), require_existing=True, readonly=True)
    for name, sha in complete['probability_sha256'].items():
        same(data.digest(CACHE / f'{name}-probabilities.f32'), sha, f'{name} completed probability file')
    validate_probabilities(values)
    return complete, values


class Capture:
    """Retain sampled positions only; concatenate head columns in frozen rank order."""
    def __init__(self, selected):
        self.selected = np.asarray(selected, dtype=np.int64)
        require(self.selected.shape == (8,) and len(set(self.selected)) == 8
            and ((self.selected >= 0) & (self.selected < 36 * 64)).all(), 'Invalid selected-head geometry.')
        self.layers = sorted(set((self.selected // 64).tolist()))
        self.enabled = False; self.values = {}; self.heads = None

    def begin(self, positions, enabled):
        self.positions = np.asarray(positions, dtype=np.int64)
        require(self.positions.ndim == 1 and len(np.unique(self.positions)) == len(self.positions), 'Duplicate sampled activation position.')
        self.enabled = enabled; self.values = {}; self.seen_legacy = set(); self.seen_heads = set(); self.indices = {}
        self.heads = np.full((len(positions), 512), np.nan, np.float16) if enabled else None

    def index(self, value):
        import torch
        key = str(value.device)
        if key not in self.indices: self.indices[key] = torch.from_numpy(self.positions).to(value.device)
        return self.indices[key]

    def legacy(self, name, value):
        import torch
        require(name not in self.seen_legacy, 'Repeated legacy capture hook.')
        self.seen_legacy.add(name)
        if not self.enabled: return
        require(value.shape[-1] == 2880, 'Legacy feature dimensions changed.')
        sampled = value.detach().reshape(-1, 2880).index_select(0, self.index(value))
        self.values[name] = sampled.to('cpu', torch.float16)
        require(bool(torch.isfinite(self.values[name]).all()), 'Legacy F16 features became nonfinite.')

    def head_hook(self, layer):
        def receive(module, args):
            import torch
            value = args[0]
            require(layer not in self.seen_heads and value.ndim == 3 and value.shape[-1] == 4096,
                'Repeated/malformed pre-o_proj head capture.')
            self.seen_heads.add(layer)
            if not self.enabled: return
            chosen = np.flatnonzero(self.selected // 64 == layer)
            head = torch.as_tensor(self.selected[chosen] % 64, dtype=torch.long, device=value.device)
            x = value.detach().reshape(-1, 64, 64).index_select(0, self.index(value)).index_select(1, head)
            require(bool(torch.isfinite(x).all()) and bool((x.float().abs() <= np.finfo(np.float16).max).all()),
                'Selected head features are nonfinite/outside F16 range; clipping is forbidden.')
            block = x.to(torch.float16).cpu().numpy().reshape(len(self.positions), -1)
            columns = (chosen[:, None] * 64 + np.arange(64)).reshape(-1)
            self.heads[:, columns] = block
            # A forward_pre_hook returns None; the model input is never modified.
        return receive

    def finish(self, legacy=True):
        require(self.seen_heads == set(self.layers), 'A selected-head hook was missed.')
        if legacy: require(self.seen_legacy == set(SPECS), 'A residual/local-attention hook was missed.')
        if self.enabled: require(np.isfinite(self.heads).all(), 'A selected-head column was unwritten.')


def batch_positions(batch, lookup, tokens):
    width = batch['input_ids'].shape[1]; positions, targets = [], []
    for slot, prompt in enumerate(batch['prompt_ix']):
        group = lookup[prompt]; offset = slot * width + width - len(tokens[prompt])
        positions.extend((group.token_ix.to_numpy() + offset).tolist()); targets.extend(group.index.tolist())
    require(len(set(targets)) == len(targets), 'A probability target appears twice in a batch.')
    return np.asarray(positions, np.int64), np.asarray(targets, np.int64)


def no_competing_jobs():
    names = {'systemness_head_scan.py', 'systemness_head_extract.py', 'systemness_head_probes.py',
        'systemness_head_confirm.py', 'systemness_long_extract.py', 'systemness_recover_local_bank.py',
        'systemness_long_probes.py', 'systemness_paired_probes.py', 'systemness_long_confirm.py',
        'systemness_attention_confirm.py', 'systemness_natural_recheck.py'}
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg): active.append(int(entry.name))
    require(not active, f'Another experiment remains active: {active}.')


@contextmanager
def run_lock():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another four-model confirmation holds this directory.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def infer():
    no_competing_jobs()
    metadata, inputs, rows, tokens = load_data()
    paths, _, _ = verify_frozen(metadata); signature = inference_signature(metadata)
    if (OUT / 'inference-complete.json').exists(): completed_probabilities(metadata, rows); return
    progress_path = CACHE / 'progress.json'
    probabilities = open_probabilities(NAMES, len(rows), require_existing=progress_path.exists())
    start, written = 0, 0
    if progress_path.exists(): start, written = verify_progress(read_json(progress_path), signature, inputs, rows, probabilities)
    import torch
    from torch.utils.data import DataLoader
    from systemness_extract import Prompts, collate
    require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Exactly four A40s required.')
    nets = {name: (head_probes.load_model if name == PRIMARY else probes.load_model)(path, device='cuda:0') for name, path in paths.items()}
    cpu_nets = {name: (head_probes.load_model if name == PRIMARY else probes.load_model)(path, device='cpu') for name, path in paths.items()}
    runtime = data.runtime; old_atomic, old_status = runtime.atomic_json, runtime.status
    runtime.atomic_json = lambda path, value: data.atomic_json(OUT / Path(path).name, value)
    runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
    handles = []; started = time.time(); portable_checked = False
    capture = Capture(metadata['frozen']['selected_head_ids'])
    try:
        model = runtime.load_model()
        require(not any(p.requires_grad for p in model.parameters()), 'Language model weights are not frozen.')
        loader = DataLoader(Prompts(inputs, tokens), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        lookup = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
        with torch.inference_mode():
            sample = next(iter(loader)); positions, _ = batch_positions(sample, lookup, tokens)
            kwargs = dict(input_ids=sample['input_ids'].cuda(0), attention_mask=sample['attention_mask'].cuda(0), use_cache=False)
            before = model.model(**kwargs).last_hidden_state.cpu()
            handles = components.hooks(model, SPECS, capture.legacy)
            for layer in capture.layers:
                handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(capture.head_hook(layer)))
            capture.begin(positions, False); after = model.model(**kwargs).last_hidden_state.cpu(); capture.finish()
            require(torch.equal(before, after) and bool(torch.isfinite(after).all()), 'Combined capture hooks changed hidden states.')
            require(not capture.values and capture.heads is None, 'Disabled selfcheck retained/wrote activations.')
            data.atomic_json(OUT / 'hook-validation.json', dict(signature=signature, hidden_states_bitexact=True,
                all_capture_writes_disabled=True, selected_layers=capture.layers, sequential_shared_forward=True))
            del before, after, kwargs
            for bi, batch in enumerate(loader):
                if bi < start: continue
                positions, targets = batch_positions(batch, lookup, tokens); capture.begin(positions, True)
                output = model.model(input_ids=batch['input_ids'].cuda(0), attention_mask=batch['attention_mask'].cuda(0), use_cache=False).last_hidden_state
                del output; capture.finish()
                differences = {}
                for name, (net, mean, scale, _) in nets.items():
                    specs = data.SPECS if name == 'residual' else attention.SPECS
                    raw_cpu = torch.from_numpy(capture.heads) if name == PRIMARY else torch.cat([capture.values[k] for k in specs], 1)
                    raw = raw_cpu.cuda(0).float(); p = net((raw - mean) / scale).exp().cpu().numpy().astype(np.float32)
                    validate_probabilities({name: p}); probabilities[name][targets] = p
                    if not portable_checked:
                        cpu, m, s, _ = cpu_nets[name]
                        expected = cpu((raw_cpu[:32].float() - m) / s).exp().numpy()
                        differences[name] = float(np.abs(expected - p[:32]).max())
                        require(differences[name] < 5e-4, f'{name} portable consumer disagrees with actual saved-F16 GPU probabilities.')
                    del raw, p
                if not portable_checked:
                    evidence = dict(signature=signature, model_sha256=metadata['frozen']['models'],
                        sample_batch=bi, sampled_tokens=min(32, len(targets)), max_probability_difference=differences,
                        head_capture_dtype='float16', no_raw_features_persisted=True)
                    path = OUT / 'portable-validation.json'
                    if path.exists():
                        previous = read_json(path); same(previous['signature'], signature, 'Existing actual portable evidence')
                        same(previous['model_sha256'], metadata['frozen']['models'], 'Previously checked portable models')
                        require(set(previous['max_probability_difference']) == set(NAMES)
                            and all(np.isfinite(v) and 0 <= v < 5e-4 for v in previous['max_probability_difference'].values()),
                            'Invalid prior portable evidence.')
                    else: data.atomic_json(path, evidence)
                    portable_checked = True
                written += len(targets); capture.values.clear(); capture.heads = None
                if (bi + 1) % 16 == 0 or bi + 1 == len(loader):
                    for values in probabilities.values(): values.flush()
                    verify_frozen(metadata)
                    data.atomic_json(progress_path, progress_record(signature, inputs, rows, probabilities, bi + 1, written))
                    status = dict(stage='head-confirmation', batch=bi + 1, batches=len(loader), rows=written, seconds=time.time() - started)
                    data.atomic_json(OUT / 'status.json', status); print(json.dumps(status), flush=True)
        same(written, len(rows), 'Complete four-model probability row count')
        require((OUT / 'portable-validation.json').is_file(), 'Actual portable evidence is missing.')
        load_data(); same(inference_signature(metadata), signature, 'Completed shared-forward dependencies')
        for values in probabilities.values(): values.flush()
        validate_probabilities(probabilities)
        data.atomic_json(OUT / 'inference-complete.json', dict(signature=signature, rows=written,
            probability_sha256={name: data.digest(CACHE / f'{name}-probabilities.f32') for name in NAMES},
            portable_validation_sha256=data.digest(OUT / 'portable-validation.json'),
            hook_validation_sha256=data.digest(OUT / 'hook-validation.json'),
            model_backend_sha256=data.digest(OUT / 'model_backend.json'),
            all_finite=True, all_normalized=True, raw_hidden_states_persisted=False, seconds=time.time() - started))
    finally:
        for handle in handles: handle.remove()
        runtime.atomic_json, runtime.status = old_atomic, old_status


def bootstrap(document, expected_counts=True):
    fields = ['tp', 'fp', 'fn', 'tn', 'correct', 'tokens']; comparisons, intervals = [], []
    for pi, (profile, group) in enumerate(document.groupby('profile', sort=True)):
        arrays = {}; ids = None
        for name in NAMES:
            frame = group[group.model == name].sort_values('group_id')
            require(len(frame) and not frame.group_id.duplicated().any(), 'Missing model or duplicate source bootstrap unit.')
            if ids is None: ids = frame.group_id.tolist()
            same(frame.group_id.tolist(), ids, 'Four-model paired source bootstrap units')
            arrays[name] = frame[fields].to_numpy(dtype=np.float64)
        if expected_counts:
            same(len(ids), 400 if profile in ['single', 'multi'] else 64, 'Independent source bootstrap count')
        rng = np.random.RandomState(data.SEED + 11464 + pi)
        ix = rng.randint(len(ids), size=(BOOTSTRAPS, len(ids)))
        point = {name: evaluation.binary_values(a.sum(0)) for name, a in arrays.items()}
        draws = {name: evaluation.binary_values(a[ix].sum(1)) for name, a in arrays.items()}
        for name in NAMES:
            for metric, value in point[name].items():
                lo, hi = np.quantile(draws[name][metric], [.025, .975])
                intervals.append(dict(profile=profile, model=name, metric=metric, source_documents=len(ids),
                    bootstrap_replicates=BOOTSTRAPS, value=float(value), ci95_low=float(lo), ci95_high=float(hi)))
        for before, after in itertools.combinations(NAMES, 2):
            for metric in point[before]:
                lo, hi = np.quantile((draws[after][metric] - draws[before][metric]) * 100, [.025, .975])
                comparisons.append(dict(profile=profile, baseline_model=before, comparison_model=after, metric=metric,
                    source_documents=len(ids), bootstrap_replicates=BOOTSTRAPS, baseline=float(point[before][metric]),
                    comparison=float(point[after][metric]), gain_percentage_points=float(100 * (point[after][metric] - point[before][metric])),
                    ci95_low=float(lo), ci95_high=float(hi)))
    return comparisons, intervals


def report():
    metadata, _, rows, _ = load_data(); frozen = metadata['frozen']; thresholds = frozen['thresholds']
    _, probabilities = completed_probabilities(metadata, rows)
    metrics, document, messages, positions = summarize(rows, probabilities, thresholds)
    comparisons, intervals = bootstrap(document)
    data.atomic_json(OUT / 'metrics.json', metrics); data.atomic_json(OUT / 'system-by-position.json', positions)
    def flat(values):
        return pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            | {f'role_recall_{k}': v for k, v in row.get('recall', {}).items()} for row in values])
    flat(metrics).to_csv(OUT / 'metrics.csv', index=False); flat(positions).to_csv(OUT / 'system-by-position.csv', index=False)
    document.to_csv(OUT / 'document-counts.csv', index=False)
    messages.to_csv(OUT / 'message-projections.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    pd.DataFrame(comparisons).to_csv(OUT / 'paired-comparisons.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUT / 'bootstrap-metrics.csv', index=False)
    lines = ['New conditional four-readout confirmation:464 independent fresh C4 sources', '',
        f'Primary new head candidate: {frozen["primary_candidate"]}; validation promoted before preparation.',
        '400 general sources;64 up-to1024 variants are paired subsets, not additional independent sources.',
        '64 additional disjoint full-length sources:exactly1024 content tokens in each of5 native role versions.',
        'All system tags and all native content tokens scored; four frozen readouts share one four-A40 nativeMXFP4 forward.',
        'Model weights, thresholds, selected heads and protocol frozen before source selection; no test-derived tuning.',
        'All prior source-key inventories, including the preceding464 cohort, excluded by known lines/URLs and full/body/prefix hashes.',
        'Original249 lacks URL/line provenance; no semantic-near-duplicate, domain-shift or natural-dialogue claim.',
        '2000 paired original-source bootstrap resamples per profile/position bin; all four models and six pairwise comparisons reported.',
        'This file reports a candidate; it never changes an existing primary model or selection.', '',
        'Model             Profile             System P  System R  System F1  Balanced acc  Five-role acc  CoT recall']
    for row in metrics:
        values = [row[k] for k in ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy', 'accuracy']]
        values.append(row['recall'].get('cot'))
        lines.append(f'{row["model"]:17s} {row["profile"]:19s} ' + '  '.join('NA' if v is None else f'{v:.6f}' for v in values))
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n'); plots(rows, probabilities, thresholds, positions)
    targets = {row['profile']: all(row[k] is not None and row[k] >= .9 for k in
        ['system_precision', 'system_recall', 'system_balanced_accuracy']) for row in metrics if row['model'] == PRIMARY}
    completed_probabilities(metadata, rows)
    artifacts = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['complete.json', 'status.json', '.run.lock']]
    data.atomic_json(OUT / 'complete.json', dict(frozen=frozen, primary_model=PRIMARY, primary_changed=False,
        independent_source_documents=464, general_documents=400, additional_full_length_documents=64,
        up_to1024_subset_documents=64, tests_used_for_tuning=False, validation_promotion_required=True,
        source_sha256=data.digest(__file__), inference_sha256=data.digest(OUT / 'inference-complete.json'),
        working_target_pass=targets, all_profiles_and_full_tail_meet_90_percent=all(targets.values()),
        bootstrap_unit='Original source document', bootstrap_replicates=BOOTSTRAPS,
        report_sha256=data.digest(OUT / 'report.txt'), artifact_sha256={str(p.relative_to(OUT)): data.digest(p) for p in artifacts}))
    data.atomic_json(OUT / 'status.json', dict(stage='complete', targets=targets, primary_changed=False))
    print((OUT / 'report.txt').read_text(), flush=True)


# Data/cache/summary helpers below are copied from the frozen long confirmation,
# with this module's own globals, source identity, model set and group namespace.

def _prepare_data():
    _, _, frozen = verify_frozen()
    if (OUT / 'data-complete.json').exists():
        load_data()
        return
    require(not (CACHE / 'progress.json').exists() and not (OUT / 'inference-complete.json').exists(),
            'Inference state exists without complete preparation; inspect it before restarting.')
    protocol = plan()
    same(data.digest(long_data.ARCHIVE), protocol['source_sha256'], 'Pinned C4 archive')
    exclusion_sources = exclusion_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = dict(frozen=frozen, exclusion_input_sha256=exclusion_sources,
        archive_sha256=protocol['source_sha256'])
    if (OUT / 'preparation-lock.json').exists():
        same(read_json(OUT / 'preparation-lock.json'), lock, 'Preparation lock')
    else:
        data.atomic_json(OUT / 'preparation-lock.json', lock)
    tok = long_data.load_tokenizer()
    hashes, urls, lines = exclusion_sets(tok, exclusion_sources)
    excluded_hashes, excluded_urls, excluded_lines = set(hashes), set(urls), set(lines)
    documents, keys = [], []
    with gzip.open(long_data.ARCHIVE, 'rt', encoding='utf-8') as source:
        for line_ix, line in enumerate(source):
            if line_ix in lines:
                continue
            article = json.loads(line)
            full_hash = data.text_digest(article['text'])
            url = article.get('url')
            if full_hash in hashes or (url and url in urls):
                continue
            ids = tok.encode(article['text'], add_special_tokens=False)
            if len(ids) < 96:
                continue
            body = tok.decode(ids[:1024], skip_special_tokens=False)
            short = tok.decode(ids[:192], skip_special_tokens=False)
            body_hash, short_hash = data.text_digest(body), data.text_digest(short)
            if '<|' in body or body_hash in hashes or short_hash in hashes:
                continue
            cohort = 'general' if len(documents) < 400 else 'full-length'
            if cohort == 'full-length' and (len(ids) < 1024 or len(tok.encode(body, add_special_tokens=False)) != 1024):
                continue
            doc = len(documents)
            documents.append(dict(document_id=doc, body=body, ids=ids[:1024], cohort=cohort))
            keys.append(dict(document_id=doc, group_id=f'head-confirm-c4-{doc}', cohort=cohort,
                shard_line=line_ix, body_sha256=body_hash, short_body_sha256=short_hash,
                full_body_sha256=full_hash, source_url=url, original_tokens=len(ids),
                body_tokens=len(tok.encode(body, add_special_tokens=False)),
                up_to1024_variant=cohort == 'general' and doc < 64))
            hashes.update([full_hash, body_hash, short_hash])
            if url:
                urls.add(url)
            lines.add(line_ix)
            if len(documents) == 464:
                break
    require(len(documents) == 464, 'Not enough eligible new documents in the pinned shard.')
    require(not {k['shard_line'] for k in keys} & excluded_lines, 'Prior source line reused.')
    require(not {k['source_url'] for k in keys if k['source_url']} & excluded_urls, 'Known prior URL reused.')
    require(not {k[field] for k in keys for field in ['body_sha256', 'short_body_sha256', 'full_body_sha256']} & excluded_hashes,
            'Prior exact or prefix fingerprint reused.')
    inputs, rows, tokens = [], [], []
    instructions = ['Follow the instructions carefully. Use clear language and state uncertainty when necessary.',
        'You are a helpful assistant. Keep the answer focused and respect the requested format.',
        'Answer using the supplied context. Do not invent sources or unsupported details.',
        'Explain your answer step by step when useful. Be concise and accurate.']
    for doc in documents:
        ix, ids = doc['document_id'], doc['ids']
        def add(messages, profile, bos):
            append_input(messages, profile, ix, bos, tok, inputs, rows, tokens)
        if doc['cohort'] == 'full-length':
            for role in data.ROLES:
                add([(role, doc['body'])], 'long-full', bool(ix % 2))
            continue
        body = tok.decode(ids[:192])
        if ix % 4 == 0:
            body = instructions[(ix // 4) % 4] + '\n\nReference text:\n' + body
        for role in data.ROLES:
            add([(role, body)], 'single', bool(ix % 2))
        width = min(48, max(16, len(ids) // 5)); shift = (ix % 5) * width
        chunks = []
        for j in range(5):
            offset = (j * width + shift) % max(width, len(ids) - width)
            chunk = tok.decode(ids[offset:offset + width])
            if ix % 4 == 0:
                chunk = instructions[(ix // 4) % 4] + '\n\n' + chunk
            chunks.append(chunk)
        messages = list(zip(data.ROLES[:4], chunks[:4]))
        messages += [('user', chunks[1]), ('tool_call', '{"query":"context"}'),
            ('tool', chunks[4]), ('cot', chunks[2]), ('assistant', chunks[3])]
        add(messages, 'multi', True)
        if ix < 64:
            for role in data.ROLES:
                add([(role, doc['body'])], 'long-single', bool(ix % 2))
    inputs, rows = pd.DataFrame(inputs), pd.DataFrame(rows)
    rows['body_offset'] = rows.token_ix - rows.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')
    validate_rows(inputs, rows, tokens, keys)
    same(exclusion_inputs(), exclusion_sources, 'Exclusion manifest inventory')
    same(verify_frozen()[2], frozen, 'Frozen preparation inputs')
    inputs.to_parquet(CACHE / FILES[0], index=False)
    rows.to_parquet(CACHE / FILES[1], index=False)
    (CACHE / FILES[2]).write_text(json.dumps(tokens, separators=(',', ':')))
    data.atomic_json(OUT / 'document-keys.json', keys)
    data.atomic_json(OUT / 'data-complete.json', dict(frozen=frozen,
        documents=464, general_documents=400, full_length_documents=64, up_to1024_subset_documents=64,
        independent_source_documents=464, inputs=len(inputs), rows=len(rows), input_tokens=int(inputs.length.sum()),
        all_content_tokens_scored=True, independent_native_label_parser_all_passed=True,
        exclusion_input_sha256=exclusion_sources, exclusion_fingerprints=len(excluded_hashes),
        known_prior_source_lines=len(excluded_lines), known_prior_urls=len(excluded_urls),
        original249_url_line_limitation='Original249 raw data has only text/source. Exclude its full/truncated/prefix hashes; source-line and URL exclusion covers known or recoverable provenance, not unavailable original249 URL metadata.',
        semantic_near_duplicates_or_domain_exclusion_claimed=False,
        cache_sha256={name: data.digest(CACHE / name) for name in FILES},
        document_keys_sha256=data.digest(OUT / 'document-keys.json'),
        source_sha256=data.digest(__file__)))


def load_data():
    metadata = read_json(OUT / 'data-complete.json')
    verify_frozen(metadata)
    same(metadata['exclusion_input_sha256'], metadata['frozen']['exclusion_input_sha256'], 'Prepared versus frozen exclusion inventories')
    require(metadata['independent_source_documents'] == 464 and metadata['general_documents'] == 400
        and metadata['full_length_documents'] == metadata['up_to1024_subset_documents'] == 64
        and metadata['all_content_tokens_scored'] and metadata['independent_native_label_parser_all_passed'],
        'Prepared source/profile/content-coverage protocol changed.')
    same(metadata['source_sha256'], data.digest(__file__), 'Confirmation source')
    for path, sha in metadata['exclusion_input_sha256'].items():
        same(data.digest(path), sha, f'Prior exclusion input {path}')
    for name, sha in metadata['cache_sha256'].items():
        same(data.digest(CACHE / name), sha, f'Input cache {name}')
    same(data.digest(OUT / 'document-keys.json'), metadata['document_keys_sha256'], 'Confirmation source keys')
    rows = pd.read_parquet(CACHE / 'rows.parquet')
    inputs = pd.read_parquet(CACHE / 'inputs.parquet')
    tokens = read_json(CACHE / 'token-ids.json')
    require(len(rows) == metadata['rows'] and len(inputs) == metadata['inputs'], 'Input counts changed.')
    same(rows.sample_ix.tolist(), list(range(len(rows))), 'Input row alignment')
    same(inputs.prompt_ix.tolist(), list(range(len(inputs))), 'Cached prompt/token alignment')
    require(len(inputs) == len(tokens) == 3040 and rows.sample_ix.is_unique
        and set(rows.split) == {'test'} and set(rows.wrapper) == {'native'}
        and (rows.group_id == 'head-confirm-c4-' + rows.document_id.astype(str)).all(), 'Cached new-cohort namespace/split changed.')
    require(all(len(tokens[row.prompt_ix]) == row.length for row in inputs.itertuples()), 'Cached per-prompt token lengths changed.')
    same(rows.groupby('profile').group_id.nunique().to_dict(),
        {'single': 400, 'multi': 400, 'long-single': 64, 'long-full': 64}, 'Cached profile source coverage')
    full = rows[rows.profile == 'long-full']
    require(len(full) == 64 * 5 * 1024 and full.groupby(['group_id', 'role']).size().eq(1024).all()
        and len(full[full.body_offset.between(512, 1023)]) == 64 * 5 * 512, 'Cached exact1024/full-tail coverage changed.')
    return metadata, inputs, rows, tokens


def open_probabilities(names, row_count, *, require_existing=False, readonly=False):
    paths = {name: CACHE / f'{name}-probabilities.f32' for name in names}
    expected = row_count * 5 * np.dtype(np.float32).itemsize
    for path in paths.values():
        require(not (require_existing or readonly) or path.is_file(),
                f'Probability file missing: {path}; refusing saved progress.')
        require(not path.exists() or path.stat().st_size == expected,
                f'Probability file truncated or wrong size: {path}.')
    return {name: np.memmap(path, mode='r' if readonly else 'r+' if path.exists() else 'w+',
        dtype=np.float32, shape=(row_count, 5)) for name, path in paths.items()}


def validate_probabilities(probabilities, indices=None, block_size=65536):
    for name, values in probabilities.items():
        count = len(values) if indices is None else len(indices)
        for start in range(0, count, block_size):
            block = values[start:start + block_size] if indices is None else values[indices[start:start + block_size]]
            require(np.isfinite(block).all() and (block >= 0).all() and (block <= 1).all()
                and np.allclose(block.sum(1), 1, atol=2e-5, rtol=0),
                f'Invalid, unnormalized or unwritten {name} probabilities at block {start}.')


def completed_indices(inputs, rows, next_batch):
    order = inputs.sort_values(['length', 'prompt_ix']).prompt_ix.to_numpy()
    batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE
    require(type(next_batch) is int and 0 <= next_batch <= batches, 'Invalid resume batch index.')
    prompts = order[:min(len(order), next_batch * BATCH_SIZE)]
    return np.flatnonzero(rows.prompt_ix.isin(prompts).to_numpy())


def probability_digests(probabilities, indices):
    result = {}
    for name, values in probabilities.items():
        h = hashlib.sha256()
        for start in range(0, len(indices), 65536):
            h.update(np.ascontiguousarray(values[indices[start:start + 65536]]).tobytes())
        result[name] = h.hexdigest()
    return result


def progress_record(signature, inputs, rows, probabilities, next_batch, written):
    indices = completed_indices(inputs, rows, next_batch)
    same(len(indices), written, 'Written row count versus deterministic batch layout')
    validate_probabilities(probabilities, indices)
    return dict(version=1, signature=signature, next_batch=next_batch, rows_written=written,
        completed_probability_sha256=probability_digests(probabilities, indices),
        portable_validation_sha256=data.digest(OUT / 'portable-validation.json'))


def verify_progress(progress, signature, inputs, rows, probabilities):
    same(progress.get('version'), 1, 'Progress format')
    same(progress.get('signature'), signature, 'Progress signature')
    expected = progress_record(signature, inputs, rows, probabilities,
        progress['next_batch'], progress['rows_written'])
    same(progress, expected, 'Completed probability rows or progress metadata')
    return progress['next_batch'], progress['rows_written']


def summarize(rows, probabilities, thresholds):
    metrics, documents, messages, positions = [], [], [], []
    tail = rows[(rows.profile == 'long-full') & rows.body_offset.between(512, 1023)].copy()
    require(tail.group_id.nunique() == 64 and len(tail) == 64 * 5 * 512, 'Full-length tail coverage changed.')
    tail['profile'] = 'long-full-tail'
    bins = [(0, 31), (32, 63), (64, 127), (128, 255), (256, 511), (512, 1023)]
    for name, p in probabilities.items():
        for profile, group in rows.groupby('profile', sort=False):
            result = probes.metrics(p[group.index], probes.labels(group), thresholds[name])
            metrics.append(dict(model=name, profile=profile, groups=int(group.group_id.nunique()), **result))
        metrics.append(dict(model=name, profile='long-full-tail', groups=64,
            **probes.metrics(p[tail.index], probes.labels(tail), thresholds[name])))
        documents.append(evaluation.document_counts(rows, p, thresholds[name], name))
        documents.append(evaluation.document_counts(tail, p[tail.index], thresholds[name], name))
        frame = rows[['group_id', 'profile', 'message_ix', 'role']].copy()
        for i, role in enumerate(data.ROLES):
            frame[f'p_{role}'] = p[:, i]
        messages.append(frame.groupby(['group_id', 'profile', 'message_ix', 'role'], as_index=False).agg(
            **{f'p_{role}': (f'p_{role}', 'mean') for role in data.ROLES}, tokens=('p_system', 'size')).assign(model=name))
        for profile, group in rows.groupby('profile', sort=False):
            for start, end in bins:
                subset = group[group.body_offset.between(start, end)]
                if subset.empty:
                    continue
                system = subset[subset.role == 'system']
                positions.append(dict(model=name, profile=profile, start=start, end=end,
                    groups=int(subset.group_id.nunique()), system_tokens=len(system),
                    system_groups=int(system.group_id.nunique()),
                    mean_system_probability=float(p[system.index, 0].mean()) if len(system) else None,
                    **probes.metrics(p[subset.index], probes.labels(subset), thresholds[name])))
                if profile == 'long-full':
                    subset = subset.copy()
                    subset['profile'] = f'long-full-position-{start}-{end}'
                    documents.append(evaluation.document_counts(subset, p[subset.index], thresholds[name], name))
    return metrics, pd.concat(documents, ignore_index=True), pd.concat(messages, ignore_index=True), positions


def plots(rows, probabilities, thresholds, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = OUT / 'plots'
    folder.mkdir(exist_ok=True)
    colors = dict(system='#90a1b9', user='#00a6f4', cot='#fd9a00', assistant='#00d492', tool='#a855f7')
    labels = ['Systemness', 'Userness', 'CoTness', 'Asstness', 'Toolness']
    titles = ['Residual control', 'Published attention control', 'Frozen dense128 control', 'New validation-selected head readout']
    documents = sorted(rows[rows.profile == 'multi'].document_id.unique())
    chosen = [int(documents[i]) for i in np.linspace(0, len(documents) - 1, min(12, len(documents)), dtype=int)]
    for doc in chosen:
        group = rows[(rows.profile == 'multi') & (rows.document_id == doc)].sort_values('token_ix')
        indices, x = group.index.to_numpy(), group.token_ix.to_numpy()
        fig, axes = plt.subplots(5, 4, figsize=(20, 7), sharex=True, sharey=True)
        for col, name in enumerate(NAMES):
            for i, label in enumerate(labels):
                ax = axes[i, col]
                for role, color in colors.items():
                    mask = group.role.to_numpy() == role
                    ax.scatter(x[mask], probabilities[name][indices[mask], i], s=3, color=color, alpha=.7, rasterized=True)
                if i == 0:
                    ax.axhline(thresholds[name], ls='--', lw=.7, color='#4b5563')
                for boundary in group.groupby('message_ix').token_ix.min():
                    ax.axvline(boundary, lw=.5, color='#d1d5db', zorder=0)
                ax.set_ylim(-.03, 1.03); ax.set_yticks([0, 1], ['0%', '100%'])
                ax.spines[['top', 'right']].set_visible(False)
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
            axes[0, col].set_title(titles[col], fontsize=10)
            axes[-1, col].set_xlabel('Token position in full input')
        handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=4, label=role)
                   for role, color in colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1), ncol=5, frameon=False, fontsize=8)
        fig.suptitle(f'head-confirm-c4-{doc} | constructed native multi-message input', y=.945, fontsize=11)
        fig.text(.5, .01, 'Frozen GPT-OSS-120B MXFP4 | all content tokens | formal-role colors | unsmoothed probabilities', ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .03, 1, .91])
        for extension in ['png', 'pdf']:
            fig.savefig(folder / f'head-confirm-c4-{doc}.{extension}', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, color in zip(NAMES, ['#64748b', '#059669', '#d97706', '#7c3aed']):
        curve = sorted([r for r in positions if r['model'] == name and r['profile'] == 'long-full'], key=lambda r: r['start'])
        ax.plot([f'{r["start"]}–{r["end"]}' for r in curve], [r['system_recall'] for r in curve], 'o-', color=color, label=name)
    ax.set_ylim(0, 1.03); ax.set_ylabel('System recall'); ax.set_xlabel('Zero-based content position bin')
    ax.set_title('64 independent full-length sources | empirical bins, no smoothing', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(folder / f'long-full-system-by-position.{extension}', dpi=160)
    plt.close(fig)
    (folder / 'gallery-index.md').write_text('12 fixed equally spaced general source document IDs; constructed native multi-message inputs. Each figure has five role panels and four frozen readouts. All 464 sources enter their declared profile metrics. No natural-dialogue claim.\n\n' +
        '\n'.join(f'- [head-confirm-c4-{doc}](head-confirm-c4-{doc}.png)' for doc in chosen) + '\n')
    data.atomic_json(OUT / 'gallery.json', dict(documents=chosen, selection='Fixed equally spaced general document IDs, independent of scores.',
        five_role_panels=5, model_columns=4, png=len(chosen) + 1, pdf=len(chosen) + 1,
        independent_source_documents=464, unsmoothed_probabilities=True, natural_dialogues=False))


def synthetic_report_fixture(folder, checks):
    """Exercise the complete report with explicit synthetic464-source geometry."""
    import copy
    import sys
    from unittest import mock
    from matplotlib.figure import Figure
    global OUT, CACHE
    original_out, original_cache = OUT, CACHE
    OUT = folder / 'synthetic-report'; CACHE = folder / 'synthetic-cache'
    OUT.mkdir(); CACHE.mkdir()
    pieces = []; entries = []; prompt_ix = 0
    def add(doc, profile, length, multiple=False):
        nonlocal prompt_ix
        for ri, role in enumerate(data.ROLES):
            token_ix = np.arange(length) + 4 + (ri * 3 if multiple else 0)
            pieces.append(pd.DataFrame(dict(prompt_ix=prompt_ix, token_ix=token_ix, role=role,
                body_offset=np.arange(length), profile=profile, split='test', wrapper='native',
                group_id=f'head-confirm-c4-{doc}', document_id=doc, message_ix=ri if multiple else 0)))
            if not multiple:
                entries.append(dict(prompt_ix=prompt_ix, prompt='Synthetic CPU fixture only.', length=length + 6,
                    profile=profile, document_id=doc, group_id=f'head-confirm-c4-{doc}')); prompt_ix += 1
        if multiple:
            entries.append(dict(prompt_ix=prompt_ix, prompt='Synthetic CPU fixture only.', length=5 * 3 + 6,
                profile=profile, document_id=doc, group_id=f'head-confirm-c4-{doc}')); prompt_ix += 1
    for doc in range(400):
        add(doc, 'single', 2); add(doc, 'multi', 2, True)
        if doc < 64: add(doc, 'long-single', 8)
    for doc in range(400, 464): add(doc, 'long-full', 1024)
    rows = pd.concat(pieces, ignore_index=True); rows['sample_ix'] = np.arange(len(rows))
    inputs = pd.DataFrame(entries); same(len(inputs), 3040, 'Synthetic3040-input geometry')
    truth = probes.labels(rows); probability = {}
    for mi, name in enumerate(NAMES):
        p = np.full((len(rows), 5), .04, np.float32); p[np.arange(len(rows)), truth] = .84
        error = (rows.profile.eq('long-full') & rows.role.eq('system') & rows.body_offset.ge(960 + mi * 16)).to_numpy()
        p[error, 0] = .20; p[error, 1] = .68; probability[name] = p
    thresholds = {name: .5 for name in NAMES}
    frozen = dict(thresholds=thresholds, primary_candidate='SYNTHETIC_CPU_FIXTURE_ONLY', primary_validation_promoted=True)
    metadata = dict(frozen=frozen, synthetic_fixture_only=True)
    data.atomic_json(OUT / 'inference-complete.json', dict(synthetic_fixture_only=True, gpu_run=False))
    saved_axes = []; original_save = Figure.savefig
    def save_figure(fig, path, *args, **kwargs):
        saved_axes.append((Path(path).name, len(fig.axes)))
        return original_save(fig, path, *args, **kwargs)
    try:
        module = sys.modules[__name__]
        with mock.patch.object(module, 'load_data', return_value=(metadata, inputs, rows, [])), \
             mock.patch.object(module, 'completed_probabilities', return_value=({}, probability)), \
             mock.patch.object(Figure, 'savefig', save_figure):
            report()
        metrics = read_json(OUT / 'metrics.json'); gallery = read_json(OUT / 'gallery.json')
        same(len(metrics), 4 * 5, 'Synthetic four-model/all-profile report coverage')
        require(all(row['groups'] == (400 if row['profile'] in ['single', 'multi'] else 64) for row in metrics),
            'Synthetic report counted paired variants as new source units.')
        for mi, name in enumerate(NAMES):
            tail = next(row for row in metrics if row['model'] == name and row['profile'] == 'long-full-tail')
            same(tail['tokens'], 64 * 5 * 512, 'Synthetic complete tail token count')
            same(tail['system_recall'], 1 - (64 - mi * 16) / 512, 'Synthetic model-column/tail recall identity')
        require(gallery['model_columns'] == 4 and gallery['five_role_panels'] == 5 and len(gallery['documents']) == 12,
            'Synthetic plot gallery omitted a model/panel/source.')
        require(len(list((OUT / 'plots').glob('*.png'))) == len(list((OUT / 'plots').glob('*.pdf'))) == 13,
            'Synthetic report must produce12 scatters plus1 position curve in both formats.')
        require(sum(axes == 20 for _, axes in saved_axes) == 24 and sum(axes == 1 for _, axes in saved_axes) == 2,
            'Synthetic scatter layout is not5x4 or position curve is missing.')
        checks.extend(['synthetic464_sources_four_models_all_profiles_and_exact64x5x512_tail',
            'synthetic12_fixed5x4_scatter_pairs_plus_full_position_curve'])
        report_folder = OUT
        # Isolate load_data freeze/cache guards from unavailable future head
        # completion using a constant frozen_snapshot, never a fake formal run.
        OUT = folder / 'synthetic-data-guards'; CACHE = folder / 'synthetic-data-cache'; OUT.mkdir(); CACHE.mkdir()
        exclusion = folder / 'synthetic-prior-keys.json'
        data.atomic_json(exclusion, [dict(shard_line=1, body_sha256=data.text_digest('CPU fixture'))])
        excluded = {str(exclusion): data.digest(exclusion)}
        frozen = dict(thresholds=thresholds, exclusion_input_sha256=excluded, synthetic_fixture_only=True)
        data.atomic_json(OUT / 'frozen-readouts.json', dict(frozen=frozen, tests_prepared=False, source_sha256=data.digest(__file__)))
        rows.to_parquet(CACHE / 'rows.parquet', index=False); inputs.to_parquet(CACHE / 'inputs.parquet', index=False)
        (CACHE / 'token-ids.json').write_text(json.dumps([[0] * int(n) for n in inputs.length], separators=(',', ':')))
        data.atomic_json(OUT / 'document-keys.json', [dict(document_id=i, synthetic_fixture_only=True) for i in range(464)])
        meta = dict(frozen=frozen, independent_source_documents=464, general_documents=400,
            full_length_documents=64, up_to1024_subset_documents=64, all_content_tokens_scored=True,
            independent_native_label_parser_all_passed=True, source_sha256=data.digest(__file__),
            exclusion_input_sha256=excluded, cache_sha256={name: data.digest(CACHE / name) for name in FILES},
            document_keys_sha256=data.digest(OUT / 'document-keys.json'), rows=len(rows), inputs=len(inputs), synthetic_fixture_only=True)
        path = OUT / 'data-complete.json'; data.atomic_json(path, meta)
        def rejects_load(changed, label):
            data.atomic_json(path, changed)
            try: load_data()
            except (RuntimeError, KeyError, FileNotFoundError): checks.append(label)
            else: raise RuntimeError(f'Synthetic data-complete tamper accepted: {label}')
            finally: data.atomic_json(path, meta)
        with mock.patch.object(module, 'frozen_snapshot', return_value=({}, thresholds, frozen)):
            require(len(load_data()[2]) == len(rows), 'Untampered synthetic cache failed its signed geometry checks.')
            bad = copy.deepcopy(meta); bad['frozen']['thresholds'][PRIMARY] = .25
            rejects_load(bad, 'reject_data_complete_frozen_threshold_tamper')
            bad = copy.deepcopy(meta); bad['exclusion_input_sha256'] = {}
            rejects_load(bad, 'reject_data_complete_exclusion_inventory_omission')
            bad = copy.deepcopy(meta); bad['source_sha256'] = '0' * 64
            rejects_load(bad, 'reject_data_complete_runner_source_tamper')
            bad = copy.deepcopy(meta); bad['independent_source_documents'] = 528
            rejects_load(bad, 'reject_paired_subset_counted_as_extra_sources')
            bad = copy.deepcopy(meta); bad['cache_sha256']['rows.parquet'] = '0' * 64
            rejects_load(bad, 'reject_data_complete_cache_hash_tamper')
            before = exclusion.read_bytes(); exclusion.write_bytes(b'changed prior source inventory')
            rejects_load(meta, 'reject_changed_actual_exclusion_input_bytes'); exclusion.write_bytes(before)
        return dict(synthetic_fixture_only=True, folder=str(report_folder), rows=len(rows), inputs=len(inputs),
            independent_sources=464, exact_full_tail_tokens=64 * 5 * 512, png=13, pdf=13,
            all_four_models_and_six_paired_comparisons=True)
    finally:
        OUT, CACHE = original_out, original_cache


def preflight():
    """Only synthetic CPU/temp files plus the already local tokenizer; no fresh C4."""
    import torch
    import py_compile
    global OUT, CACHE
    original_out, original_cache = OUT, CACHE
    folder = Path(tempfile.mkdtemp(prefix='systemness-head-confirm-preflight-', dir='/tmp'))
    checks = []; torch.set_num_threads(2); torch.manual_seed(data.SEED)
    def rejects(action, label):
        try: action()
        except (RuntimeError, KeyError, ValueError, FileNotFoundError): checks.append(label)
        else: raise RuntimeError(f'Fault fixture accepted: {label}')
    try:
        OUT = folder / 'output'; CACHE = folder / 'cache'; OUT.mkdir(); CACHE.mkdir()
        promoted = dict(promoted=True, candidate_is_guard_eligible=True, test_used_for_selection=False,
            confirmation_required=True, primary_changed=False)
        require_promoted(promoted)
        rejects(lambda: require_promoted(dict(promoted, promoted=False)), 'reject_unpromoted_before_preparation')
        rejects(lambda: require_promoted(dict(promoted, candidate_is_guard_eligible=False)), 'reject_ineligible_selected_candidate')
        rejects(lambda: require_promoted(dict(promoted, promoted=1)), 'reject_nonboolean_promotion')
        digest = data.text_digest('prior original source')
        keys = [dict(shard_line=17, source_url='https://example.invalid/prior', body_sha256=digest,
            prefix192_sha256=data.text_digest('short'), prefix1024_sha256=data.text_digest('long'))]
        hashes, urls, lines = key_fingerprints(keys)
        require(digest in hashes and len(hashes) == 3 and lines == {17} and len(urls) == 1, 'Prior source fingerprint coverage failed.')
        checks.append('full_body_prefix192_prefix1024_url_line_fingerprint_inventory')
        rejects(lambda: key_fingerprints([dict(body_sha256=digest)]), 'reject_missing_prior_line_identity')
        rejects(lambda: key_fingerprints([dict(shard_line=1, body_sha256='corrupt')]), 'reject_invalid_prior_hash')
        inputs = pd.DataFrame(dict(prompt_ix=range(17), length=range(10, 27)))
        rows = pd.DataFrame(dict(prompt_ix=np.repeat(np.arange(17), 2)))
        signature = dict(fixture='CPU four frozen models')
        data.atomic_json(OUT / 'portable-validation.json', dict(signature=signature))
        probabilities = open_probabilities(NAMES, len(rows))
        for value in probabilities.values(): value[:] = [.5, .125, .125, .125, .125]; value.flush()
        progress = progress_record(signature, inputs, rows, probabilities, 2, 16)
        same(verify_progress(progress, signature, inputs, rows, probabilities), (2, 16), 'Four-model probability resume')
        checks.append('four_model_hash_bound_resume_exact_prefix')
        rejects(lambda: verify_progress(progress, {'fixture': 'changed'}, inputs, rows, probabilities), 'reject_inference_signature_drift')
        rejects(lambda: verify_progress(dict(progress, rows_written=15), signature, inputs, rows, probabilities), 'reject_wrong_prefix_row_count')
        probabilities[PRIMARY][0] = [.25, .25, .25, .125, .125]
        rejects(lambda: verify_progress(progress, signature, inputs, rows, probabilities), 'reject_normalized_completed_probability_corruption')
        probabilities[PRIMARY][0] = [np.nan, 0, 0, 0, 0]
        rejects(lambda: validate_probabilities(probabilities), 'reject_nonfinite_probability')
        probabilities[PRIMARY][0] = [0, 0, 0, 0, 0]
        rejects(lambda: validate_probabilities(probabilities), 'reject_unwritten_probability')
        for value in probabilities.values(): value._mmap.close()
        missing = CACHE / 'dense128-probabilities.f32'; content = missing.read_bytes(); missing.unlink()
        rejects(lambda: open_probabilities(NAMES, len(rows), require_existing=True), 'reject_missing_probability_with_old_progress')
        require(not missing.exists(), 'Missing saved probability was silently recreated.')
        missing.write_bytes(content[:-4])
        rejects(lambda: open_probabilities(NAMES, len(rows), require_existing=True), 'reject_truncated_probability_with_old_progress')
        missing.write_bytes(content)
        data.atomic_json(OUT / 'portable-validation.json', dict(signature='tampered'))
        reopened = open_probabilities(NAMES, len(rows), require_existing=True)
        for value in reopened.values(): value[:] = [.5, .125, .125, .125, .125]
        rejects(lambda: verify_progress(progress, signature, inputs, rows, reopened), 'reject_changed_portable_evidence_hash')
        for value in reopened.values(): value._mmap.close()
        selected = np.array([455, 0, 325, 65, 390, 130, 260, 195])
        capture = Capture(selected); positions = np.array([0, 4, 7, 9], np.int64)
        raw = torch.randn(2, 5, 4096); legacy_raw = torch.randn(2, 5, 2880)
        modules = {layer: torch.nn.Identity() for layer in capture.layers}
        handles = [module.register_forward_pre_hook(capture.head_hook(layer)) for layer, module in modules.items()]
        capture.begin(positions, False)
        for layer, module in modules.items(): require(torch.equal(module(raw), raw), 'Disabled head hook changed hidden input.')
        for name in SPECS: capture.legacy(name, legacy_raw)
        capture.finish(); require(not capture.values and capture.heads is None, 'Disabled hook wrote feature arrays.')
        checks.append('combined_hooks_disabled_writes_and_hidden_identity')
        capture.begin(positions, True)
        for layer, module in modules.items(): require(torch.equal(module(raw), raw), 'Enabled prehook changed hidden input.')
        for name in SPECS: capture.legacy(name, legacy_raw)
        capture.finish()
        expected = raw.reshape(-1, 64, 64)[positions][:, selected % 64].half().numpy().reshape(len(positions), 512)
        require(np.array_equal(capture.heads, expected), 'Frozen head-rank order/F16 capture differs.')
        require(all(torch.equal(value, legacy_raw.reshape(-1, 2880)[positions].half()) for value in capture.values.values()),
            'Gather-before-F16 legacy features differ.')
        checks.append('sampled_positions_head_rank_order_and_f16_training_consistency')
        rejects(lambda: modules[capture.layers[0]](raw), 'reject_duplicate_head_hook')
        capture.begin(positions, True)
        rejects(lambda: capture.finish(), 'reject_missing_capture_layers')
        for handle in handles: handle.remove()
        net = head_probes.make_net('mlp', hidden=64).eval()
        path = folder / 'head-fixture.npz'
        meta = dict(kind='mlp', hidden=64, dimensions=512, heads=8, head_dim=64, roles=data.ROLES)
        head_probes.save_model(path, net, np.zeros(512, np.float32), np.ones(512, np.float32), meta)
        actual = head_probes.predict_raw(path, expected, np.arange(len(expected)))
        with torch.inference_mode(): reference = net(torch.from_numpy(expected).float()).exp().numpy()
        portable_difference = float(np.abs(actual - reference).max())
        require(portable_difference < 1e-7, '512-dim portable raw-F16 consumer differs.')
        checks.append('new_head512_portable_raw_f16_probability_parity')
        tok = long_data.load_tokenizer()
        body = tok.decode(tok.encode('tomato ' * 1400, add_special_tokens=False)[:1024])
        fixture_inputs, fixture_rows, fixture_tokens = [], [], []
        for role in data.ROLES: append_input([(role, body)], 'long-full', 400, True, tok, fixture_inputs, fixture_rows, fixture_tokens)
        same(Counter(row['role'] for row in fixture_rows), Counter({role: 1024 for role in data.ROLES}), 'All five native full-content labels')
        require(all(row['group_id'] == 'head-confirm-c4-400' for row in fixture_rows), 'Fixture reused old cohort namespace.')
        checks.append('new_namespace_all5_native_roles1024_and_system_positive')
        calls = [('system', 'Follow the rules.'), ('user', 'Find context.'), ('tool_call', '{"query":"context"}'),
            ('tool', 'Context found.'), ('cot', 'Check context.'), ('assistant', 'Here is the result.')]
        append_input(calls, 'multi', 0, True, tok, fixture_inputs, fixture_rows, fixture_tokens)
        require(set(row['role'] for row in fixture_rows) == set(data.ROLES), 'Tool-call body became a false formal-role class.')
        checks.append('native_tool_call_cycle_all_content_coverage')
        docs = []
        for name in NAMES:
            for doc in range(3): docs.append(dict(model=name, profile='single', group_id=f'fixture-{doc}',
                tp=7, fp=1, fn=2, tn=20, correct=24, tokens=30))
        comparisons, intervals = bootstrap(pd.DataFrame(docs), expected_counts=False)
        require(len(comparisons) == 6 * 6 and len(intervals) == 4 * 6
            and all(r['source_documents'] == 3 and r['bootstrap_replicates'] == 2000 for r in comparisons)
            and all(r['gain_percentage_points'] == r['ci95_low'] == r['ci95_high'] == 0 for r in comparisons),
            'Four-model source-paired bootstrap has wrong pair count or independent units.')
        checks.append('four_models_six_pairs_2000_source_bootstrap_identical_zero_deltas')
        rejects(lambda: bootstrap(pd.DataFrame(docs + [docs[0]]), expected_counts=False), 'reject_duplicate_source_bootstrap_unit')
        rejects(lambda: bootstrap(pd.DataFrame(docs[:-1]), expected_counts=False), 'reject_unpaired_source_bootstrap_membership')
        report_fixture = synthetic_report_fixture(folder, checks)
        py_compile.compile(__file__, cfile=str(folder / 'systemness_head_confirm.pyc'), doraise=True)
        checks.append('compile_to_tmp')
        result = dict(cpu_only=True, formal_prepare_run=False, fresh_source_archive_read=False, gpu_run=False,
            current_or_old_test_probabilities_read=False, source_sha256=data.digest(__file__),
            checks=checks, check_count=len(checks), portable_head_probability_difference=portable_difference,
            future_head_selection_exists=(head_probes.OUT / 'selection.json').is_file(),
            protocol=protocol_spec(), production_gpu_selfchecks_not_yet_run=True)
        result['synthetic_report_fixture'] = report_fixture
        data.atomic_json(folder / 'report.json', result)
        print(json.dumps(dict(preflight=str(folder / 'report.json'), **result), indent=2), flush=True)
    finally:
        OUT, CACHE = original_out, original_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--stage', choices=['prepare', 'infer', 'report', 'all'])
    args = parser.parse_args(); require(not (args.preflight and args.stage), 'Choose CPU preflight or a formal stage.')
    if args.preflight: preflight()
    elif args.stage:
        selections(); no_competing_jobs()  # Gate before creating the protocol/cache/lock.
        if args.stage in ['infer', 'report']: require(OUT.is_dir(), 'Prepare the new cohort before inference/report.')
        with run_lock():
            if args.stage in ['prepare', 'all']: prepare()
            if args.stage in ['infer', 'all']: infer()
            if args.stage in ['report', 'all']: report()
    else: parser.print_help()


if __name__ == '__main__':
    main()
