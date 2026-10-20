"""Confirm a final validation-selected pointwise readout and two frozen controls.

The cohort has 400 general + 64 additional full-length source documents.
Preparation requires the final validation selection to exist. Nothing runs by default;
use --stage explicitly, or --preflight for temporary CPU-only fault fixtures.
Only probabilities are cached during the shared four-A40 native MXFP4 forward.
"""
import argparse
from collections import Counter
import gzip
import hashlib
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
import systemness_long_data as long_data
import systemness_probes as probes


OUT = long_data.OUT / 'confirmation'
CACHE = data.CACHE / 'long-full-confirmation'
FINAL_SELECTION = data.runtime.OUT / 'systemness-final-readout/selection.json'
NAMES = ['residual', 'base-attention', 'selected']
PROFILES = ['single', 'multi', 'long-single', 'long-full']
SPECS = dict(data.SPECS, **attention.SPECS)
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
BATCH_SIZE = 8
BOOTSTRAPS = 2000
DEPENDENCIES = ['systemness_long_confirm.py', 'systemness_long_data.py',
    'systemness_attention.py', 'systemness_components.py', 'systemness_data.py',
    'systemness_probes.py', 'systemness_evaluate.py', 'systemness_extract.py',
    'feature_pipeline_runtime.py', 'run_local_gptoss120b.py']
require = long_data.require


def read_json(path):
    return json.loads(Path(path).read_text())


def same(actual, expected, label):
    require(actual == expected, f'{label} changed; refusing stale confirmation state.')


def plan():
    value = read_json(long_data.OUT / 'confirmation-plan.json')
    expected = dict(protocol_version=2, final_selection='systemness-final-readout/selection.json',
        general_documents=400, additional_full_length_documents=64,
        independent_source_documents=464, general_long_subset_documents=64,
        full_length_profile='long-full', full_length_body_tokens=1024,
        source_revision='1588ec454efa1a09f29cd18ddd04fe05fc8653a2',
        source_sha256='1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f')
    for key, wanted in expected.items():
        same(value.get(key), wanted, f'Confirmation protocol {key}')
    require(value.get('no_test_tuning') is True, 'Confirmation must prohibit test tuning.')
    return value


def selections():
    sources = {'residual': data.OUT / 'selection.json',
        'base-attention': attention.OUT / 'selection.json',
        'selected': FINAL_SELECTION}
    records = {name: read_json(path) for name, path in sources.items()}
    require(all(r.get('test_used_for_selection') is False for r in records.values()),
            'All readouts must be frozen using validation only before preparation.')
    same(records['base-attention']['candidate'], 'mlp-h64', 'Published attention control')
    require(type(records['selected'].get('promoted')) is bool,
            'Final readout experiment validation promotion flag is missing.')
    paths = {'residual': data.OUT / 'models/selected.npz',
        **{name: Path(records[name]['model_path']) for name in NAMES[1:]}}
    thresholds = {name: float(records[name]['system_threshold']) for name in NAMES}
    for name, path in paths.items():
        require(not str(path).startswith('/workspace'), 'Do not access the unavailable /workspace mount.')
        expected = records[name]['selected_model_sha256' if name == 'residual' else 'model_sha256']
        same(data.digest(path), expected, f'{name} frozen model')
        require(np.isfinite(thresholds[name]) and 0 <= thresholds[name] <= 1.00001,
                f'Invalid frozen threshold for {name}.')
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            same(metadata['roles'], data.ROLES, f'{name} role order')
            same(metadata['feature_spec'], data.SPECS if name == 'residual' else attention.SPECS,
                 f'{name} feature specification')
    return paths, thresholds, sources, records


def frozen_snapshot():
    plan()
    paths, thresholds, sources, records = selections()
    checkpoint, tokenizer_sha = long_data.tokenizer_assets()
    # infer/report can run in a fresh process without the prepare process's env.
    # Bind the runtime loader to the verified path instead of its /workspace default.
    os.environ['ROLE_ANALYSIS_LOAD_PATH'] = str(checkpoint)
    if os.environ.get('HF_HOME', '').startswith('/workspace'):
        os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
    config = read_json(checkpoint / 'config.json')
    same(config['quantization_config']['quant_method'], 'mxfp4', 'Checkpoint precision')
    require(all(config['layer_types'][i] == 'sliding_attention' for i in [16, 20, 24]),
            'The attention features must remain the published local layers.')
    frozen = dict(models={name: data.digest(path) for name, path in paths.items()},
        thresholds=thresholds, selection_sha256={name: data.digest(path) for name, path in sources.items()},
        primary_candidate=records['selected']['candidate'],
        primary_validation_promoted=records['selected']['promoted'],
        confirmation_plan_sha256=data.digest(long_data.OUT / 'confirmation-plan.json'),
        dependency_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        checkpoint_path=str(checkpoint), tokenizer_sha256=tokenizer_sha,
        checkpoint_config_sha256=data.digest(checkpoint / 'config.json'),
        verified_weights_sha256=data.digest(checkpoint / 'verified-weights.json'))
    return paths, thresholds, frozen


def verify_frozen(metadata=None):
    paths, thresholds, frozen = frozen_snapshot()
    if metadata is not None:
        same(metadata['frozen'], frozen, 'Models, thresholds, protocol or dependencies')
    return paths, thresholds, frozen


def exclusion_inputs():
    paths = [data.OUT / 'document-keys.json',
        data.runtime.OUT / 'feature-search/fresh/document-keys.json',
        data.runtime.OUT / 'training_inputs.parquet',
        data.runtime.OUT.parent / 'local-gptoss20b/raw_training_data.json']
    # Include completed and already prepared confirmations, which may have been
    # inspected. Exclude this script's own confirmation directory on restarts.
    for candidate in sorted(data.runtime.OUT.glob('**/confirmation/document-keys.json')):
        if candidate.parent == OUT:
            continue
        if any((candidate.parent / marker).is_file() for marker in
               ['data-complete.json', 'inference-complete.json', 'complete.json']):
            paths.append(candidate)
    return {str(path): data.digest(path) for path in paths}


def exclusion_sets(tok, inputs):
    hashes, urls, lines = set(), set(), set()
    for path in inputs:
        path = Path(path)
        if path.name != 'document-keys.json':
            continue
        for key in read_json(path):
            lines.add(int(key['shard_line']))
            if key.get('source_url'):
                urls.add(key['source_url'])
            hashes.update(key[field] for field in ['body_sha256', 'full_body_sha256',
                'truncated_body_sha256', 'short_body_sha256', 'prefix192_sha256'] if key.get(field))
    prior = pd.read_parquet(data.runtime.OUT / 'training_inputs.parquet')
    require(prior.question.nunique() == 249, 'Original training source count changed.')
    raw = read_json(data.runtime.OUT.parent / 'local-gptoss20b/raw_training_data.json')
    require(len(raw) == 249, 'Original raw source count changed.')
    for text in list(prior.question.unique()) + [r['text'] for r in raw]:
        ids = tok.encode(text, add_special_tokens=False)
        hashes.update([data.text_digest(text), data.text_digest(tok.decode(ids[:192])),
                       data.text_digest(tok.decode(ids[:1024]))])
    # Fingerprints only: recovering prefixes is not evaluation of old tests.
    remaining = set(lines)
    with gzip.open(long_data.ARCHIVE, 'rt', encoding='utf-8') as source:
        for line_ix, line in enumerate(source):
            if line_ix not in remaining:
                continue
            article = json.loads(line)
            ids = tok.encode(article['text'], add_special_tokens=False)
            hashes.update([data.text_digest(article['text']), data.text_digest(tok.decode(ids[:192])),
                           data.text_digest(tok.decode(ids[:1024]))])
            if article.get('url'):
                urls.add(article['url'])
            remaining.remove(line_ix)
            if not remaining:
                break
    require(not remaining, 'Previously used source lines are missing from the pinned archive.')
    return hashes, urls, lines


def parse_labels(ids, tok):
    """Native token-state labels independent of serialization-time body spans."""
    special = {s: tok.convert_tokens_to_ids(s) for s in long_data.SPECIAL}
    active, header, in_header, result = None, [], False, {}
    for pos, token in enumerate(ids):
        if token == special['<|start|>']:
            require(active is None and not in_header, 'Nested native message.')
            header, in_header = [], True
        elif token == special['<|message|>']:
            require(in_header, 'Body delimiter without native header.')
            text = tok.decode(header)
            if text.startswith('functions.'):
                active = 'tool'
            elif text.startswith('assistant') and 'to=functions.' in text:
                active = 'tool_call'
            elif text.startswith('assistant'):
                active = 'cot' if '<|channel|>analysis' in text else 'assistant'
            else:
                require(text in ['system', 'user'], 'Unrecognized native header.')
                active = text
            in_header = False
        elif token in [special['<|end|>'], special['<|return|>'], special['<|call|>']]:
            require(active is not None and not in_header, 'Unexpected native end marker.')
            active = None
        elif in_header:
            header.append(token)
        elif active is not None:
            result[pos] = active
    require(active is None and not in_header, 'Unterminated native input.')
    return result


def append_input(messages, profile, doc, bos, tok, inputs, rows, tokens):
    text, spans = data.serialize(messages, bos=bos)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    labels = parse_labels(encoded['input_ids'], tok)
    prompt_ix = len(inputs)
    positions = set()
    for pos, (left, right) in enumerate(encoded['offset_mapping']):
        for span in spans:
            if left < span['end'] and right > span['start']:
                require(left >= span['start'] and right <= span['end'], 'Ambiguous body boundary token.')
                same(labels.get(pos), span['role'], 'Independent native content label')
                rows.append(dict(prompt_ix=prompt_ix, token_ix=pos, role=span['role'],
                    semantic_role=span['role'], message_ix=span['message_ix'], profile=profile,
                    split='test', group_id=f'long-confirm-c4-{doc}', document_id=doc,
                    wrapper='native', sample_ix=len(rows)))
                positions.add(pos)
                break
    same(positions, {pos for pos, role in labels.items() if role != 'tool_call'},
         'All normal native content tokens scored')
    if profile == 'long-full':
        require(len(positions) == 1024 and len(messages) == 1, 'Full-length body is not exactly 1024 tokens.')
    inputs.append(dict(prompt_ix=prompt_ix, prompt=text, length=len(encoded['input_ids']),
        profile=profile, document_id=doc, group_id=f'long-confirm-c4-{doc}'))
    tokens.append(encoded['input_ids'])


def validate_rows(inputs, rows, tokens, keys):
    require(len(keys) == 464 and len({k['shard_line'] for k in keys}) == 464, 'Expected 464 distinct sources.')
    same(Counter(k['cohort'] for k in keys), Counter(general=400, **{'full-length': 64}), 'Source cohorts')
    same(inputs.prompt_ix.tolist(), list(range(len(inputs))), 'Prompt order')
    same(rows.sample_ix.tolist(), list(range(len(rows))), 'Sample order')
    same(set(rows.profile), set(PROFILES), 'Scored profiles')
    require(set(rows.split) == {'test'} and set(rows.wrapper) == {'native'}, 'Unexpected role wrapper or split.')
    require(len(inputs) == len(tokens) == 3040, 'Input count or token cache alignment changed.')
    lookup = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
    tok = long_data.load_tokenizer()
    for row in inputs.itertuples():
        same(len(tokens[row.prompt_ix]), row.length, 'Cached input length')
        labels = parse_labels(tokens[row.prompt_ix], tok)
        group = lookup[row.prompt_ix]
        same(dict(zip(group.token_ix, group.role)), {p: r for p, r in labels.items() if r != 'tool_call'},
             'Cached native labels and full content coverage')
        if row.profile == 'long-full':
            require(len(group) == 1024 and group.role.nunique() == 1, 'Invalid full-length role version.')
    expected = {'single': 400, 'multi': 400, 'long-single': 64, 'long-full': 64}
    same(rows.groupby('profile').group_id.nunique().to_dict(), expected, 'Profile source counts')
    require(not set(rows[rows.profile == 'long-full'].group_id) &
            set(rows[rows.profile != 'long-full'].group_id), 'Full-length sources overlap general sources.')
    for profile in ['single', 'long-single', 'long-full']:
        counts = rows[rows.profile == profile].groupby(['group_id', 'role']).size()
        require(len(counts) == expected[profile] * 5, f'Missing formal role in {profile}.')


def prepare():
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
            keys.append(dict(document_id=doc, group_id=f'long-confirm-c4-{doc}', cohort=cohort,
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
        completed_probability_sha256=probability_digests(probabilities, indices))


def verify_progress(progress, signature, inputs, rows, probabilities):
    same(progress.get('version'), 1, 'Progress format')
    same(progress.get('signature'), signature, 'Progress signature')
    expected = progress_record(signature, inputs, rows, probabilities,
        progress['next_batch'], progress['rows_written'])
    same(progress, expected, 'Completed probability rows or progress metadata')
    return progress['next_batch'], progress['rows_written']


def inference_signature(metadata):
    verify_frozen(metadata)
    return dict(data_sha256=data.digest(OUT / 'data-complete.json'), frozen=metadata['frozen'],
        specs=SPECS, batch_size=BATCH_SIZE, inference='Shared frozen MXFP4 forward; pointwise; probability-only cache.')


def completed_probabilities(metadata, rows):
    complete = read_json(OUT / 'inference-complete.json')
    same(complete['signature'], inference_signature(metadata), 'Completed inference signature')
    same(set(complete['probability_sha256']), set(NAMES), 'Completed model coverage')
    probabilities = open_probabilities(NAMES, len(rows), require_existing=True, readonly=True)
    for name, sha in complete['probability_sha256'].items():
        same(data.digest(CACHE / f'{name}-probabilities.f32'), sha, f'Completed {name} probabilities')
    validate_probabilities(probabilities)
    return complete, probabilities


def infer():
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts, collate
    metadata, inputs, rows, tokens = load_data()
    paths, _, _ = verify_frozen(metadata)
    signature = inference_signature(metadata)
    if (OUT / 'inference-complete.json').exists():
        completed_probabilities(metadata, rows)
        return
    progress_path = CACHE / 'progress.json'
    probabilities = open_probabilities(NAMES, len(rows), require_existing=progress_path.exists())
    start_batch, written = 0, 0
    if progress_path.exists():
        start_batch, written = verify_progress(read_json(progress_path), signature, inputs, rows, probabilities)
    nets = {name: probes.load_model(path, device='cuda:0') for name, path in paths.items()}
    runtime = data.runtime
    old_atomic, old_status = runtime.atomic_json, runtime.status
    runtime.atomic_json = lambda path, value: data.atomic_json(OUT / Path(path).name, value)
    runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
    captured = {}
    def receive(name, value):
        captured[name] = value.detach().to('cpu', torch.float16).reshape(-1, 2880)
    started = time.time()
    original_hooks = pipeline.install_hooks
    try:
        model = runtime.load_model()
        loader = DataLoader(Prompts(inputs, tokens), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        lookup = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
        with torch.inference_mode():
            batch = next(iter(loader))
            sample = SimpleNamespace(input_ids=batch['input_ids'].cuda(0), attention_mask=batch['attention_mask'].cuda(0))
            before = model.model(input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False).last_hidden_state.cpu()
            handles = components.hooks(model, SPECS, receive)
            try:
                after = model.model(input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False).last_hidden_state.cpu()
                require(torch.equal(before, after) and torch.isfinite(after).all(), 'Capture hooks altered the frozen model.')
                reference = dict(captured)
            finally:
                for handle in handles:
                    handle.remove()
            del before, after
            pipeline.install_hooks = components.hooks
            for bi, batch, states in pipeline.feature_batches(model, loader, SPECS, start_batch, (sample, reference),
                    lambda result: data.atomic_json(OUT / 'pipeline-validation.json', result)):
                width = batch['input_ids'].shape[1]; positions, targets = [], []
                for b, prompt in enumerate(batch['prompt_ix']):
                    group = lookup[prompt]
                    offset = b * width + width - len(tokens[prompt])
                    positions.extend((group.token_ix.to_numpy() + offset).tolist())
                    targets.extend(group.index.tolist())
                indices = torch.tensor(positions, dtype=torch.long)
                for name, (net, mean, scale, _) in nets.items():
                    specs = data.SPECS if name == 'residual' else attention.SPECS
                    raw = torch.cat([states[key][indices] for key in specs], dim=1).cuda(0).float()
                    require(torch.isfinite(raw).all(), 'Non-finite captured content features.')
                    p = net((raw - mean) / scale).exp().cpu().numpy()
                    validate_probabilities({name: p})
                    probabilities[name][targets] = p
                    del raw, p
                written += len(targets)
                states.clear()
                if (bi + 1) % 16 == 0 or bi + 1 == len(loader):
                    for value in probabilities.values():
                        value.flush()
                    verify_frozen(metadata)
                    data.atomic_json(progress_path, progress_record(signature, inputs, rows, probabilities, bi + 1, written))
                    record = dict(stage='confirming', batch=bi + 1, batches=len(loader), rows=written,
                        seconds=time.time() - started)
                    data.atomic_json(OUT / 'status.json', record)
                    print(json.dumps(record), flush=True)
        same(written, len(rows), 'Completed probability row count')
        same(inference_signature(metadata), signature, 'Inference dependencies')
        load_data()
        for values in probabilities.values():
            values.flush()
        validate_probabilities(probabilities)
        data.atomic_json(OUT / 'inference-complete.json', dict(signature=signature, rows=written,
            all_finite=True, all_normalized=True, probability_sha256={name: data.digest(CACHE / f'{name}-probabilities.f32') for name in NAMES},
            raw_hidden_states_persisted=False, seconds=time.time() - started))
    finally:
        pipeline.install_hooks = original_hooks
        runtime.atomic_json, runtime.status = old_atomic, old_status


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


def bootstrap(document):
    fields = ['tp', 'fp', 'fn', 'tn', 'correct', 'tokens']
    comparisons, intervals = [], []
    for pi, (profile, group) in enumerate(document.groupby('profile', sort=True)):
        arrays = {}
        ids = None
        for name in NAMES:
            frame = group[group.model == name].sort_values('group_id')
            require(not frame.group_id.duplicated().any(), 'Duplicate source bootstrap unit.')
            if ids is None:
                ids = frame.group_id.tolist()
            same(frame.group_id.tolist(), ids, 'Paired source bootstrap units')
            arrays[name] = frame[fields].to_numpy(dtype=np.float64)
        rng = np.random.RandomState(data.SEED + 464 + pi)
        indices = rng.randint(len(ids), size=(BOOTSTRAPS, len(ids)))
        point = {name: evaluation.binary_values(a.sum(0)) for name, a in arrays.items()}
        sampled = {name: evaluation.binary_values(a[indices].sum(1)) for name, a in arrays.items()}
        for name in NAMES:
            for metric, value in point[name].items():
                lo, hi = np.quantile(sampled[name][metric], [.025, .975])
                intervals.append(dict(profile=profile, model=name, metric=metric, source_documents=len(ids),
                    bootstrap_replicates=BOOTSTRAPS, value=float(value), ci95_low=float(lo), ci95_high=float(hi)))
        for before, after in [('residual', 'base-attention'), ('residual', 'selected'), ('base-attention', 'selected')]:
            for metric in point[before]:
                lo, hi = np.quantile((sampled[after][metric] - sampled[before][metric]) * 100, [.025, .975])
                comparisons.append(dict(profile=profile, baseline_model=before, comparison_model=after,
                    metric=metric, source_documents=len(ids), bootstrap_replicates=BOOTSTRAPS,
                    baseline=float(point[before][metric]), comparison=float(point[after][metric]),
                    gain_percentage_points=float(100 * (point[after][metric] - point[before][metric])),
                    ci95_low=float(lo), ci95_high=float(hi)))
    return comparisons, intervals


def plots(rows, probabilities, thresholds, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = OUT / 'plots'
    folder.mkdir(exist_ok=True)
    colors = dict(system='#90a1b9', user='#00a6f4', cot='#fd9a00', assistant='#00d492', tool='#a855f7')
    labels = ['Systemness', 'Userness', 'CoTness', 'Asstness', 'Toolness']
    titles = ['Residual control', 'Published attention control', 'Final validation-selected readout']
    documents = sorted(rows[rows.profile == 'multi'].document_id.unique())
    chosen = [int(documents[i]) for i in np.linspace(0, len(documents) - 1, min(12, len(documents)), dtype=int)]
    for doc in chosen:
        group = rows[(rows.profile == 'multi') & (rows.document_id == doc)].sort_values('token_ix')
        indices, x = group.index.to_numpy(), group.token_ix.to_numpy()
        fig, axes = plt.subplots(5, 3, figsize=(15, 7), sharex=True, sharey=True)
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
        fig.suptitle(f'long-confirm-c4-{doc} | constructed native multi-message input', y=.945, fontsize=11)
        fig.text(.5, .01, 'Frozen GPT-OSS-120B MXFP4 | all content tokens | formal-role colors | unsmoothed probabilities', ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .03, 1, .91])
        for extension in ['png', 'pdf']:
            fig.savefig(folder / f'long-confirm-c4-{doc}.{extension}', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, color in zip(NAMES, ['#64748b', '#059669', '#7c3aed']):
        curve = sorted([r for r in positions if r['model'] == name and r['profile'] == 'long-full'], key=lambda r: r['start'])
        ax.plot([f'{r["start"]}–{r["end"]}' for r in curve], [r['system_recall'] for r in curve], 'o-', color=color, label=name)
    ax.set_ylim(0, 1.03); ax.set_ylabel('System recall'); ax.set_xlabel('Zero-based content position bin')
    ax.set_title('64 independent full-length sources | empirical bins, no smoothing', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(folder / f'long-full-system-by-position.{extension}', dpi=160)
    plt.close(fig)
    (folder / 'gallery-index.md').write_text('12 fixed equally spaced general source document IDs; constructed native multi-message inputs. Each figure has five role panels and three frozen readouts. All 464 sources enter their declared profile metrics. No natural-dialogue claim.\n\n' +
        '\n'.join(f'- [long-confirm-c4-{doc}](long-confirm-c4-{doc}.png)' for doc in chosen) + '\n')
    data.atomic_json(OUT / 'gallery.json', dict(documents=chosen, selection='Fixed equally spaced general document IDs, independent of scores.',
        five_role_panels=5, model_columns=3, png=len(chosen) + 1, pdf=len(chosen) + 1,
        independent_source_documents=464, unsmoothed_probabilities=True, natural_dialogues=False))


def report():
    metadata, _, rows, _ = load_data()
    _, thresholds, frozen = verify_frozen(metadata)
    inference, probabilities = completed_probabilities(metadata, rows)
    metrics, document, messages, positions = summarize(rows, probabilities, thresholds)
    comparisons, intervals = bootstrap(document)
    data.atomic_json(OUT / 'metrics.json', metrics)
    data.atomic_json(OUT / 'system-by-position.json', positions)
    flatten = lambda values: pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in values])
    flatten(metrics).to_csv(OUT / 'metrics.csv', index=False)
    flatten(positions).to_csv(OUT / 'system-by-position.csv', index=False)
    document.to_csv(OUT / 'document-counts.csv', index=False)
    messages.to_csv(OUT / 'message-projections.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    pd.DataFrame(comparisons).to_csv(OUT / 'paired-comparisons.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUT / 'bootstrap-metrics.csv', index=False)
    lines = ['Frozen three-readout confirmation: 464 independent C4 source documents', '',
        f'Final validation-selected pointwise primary: {frozen["primary_candidate"]}; its experiment validation promotion flag: {frozen["primary_validation_promoted"]}.',
        '400 general sources; 64 up-to1024 long variants are paired subsets, not extra independent documents.',
        '64 additional, mutually disjoint source bodies contain exactly 1024 actual content tokens in each of five native role versions.',
        'All system tags retained. Frozen native MXFP4 model on four A40s; one shared forward, pointwise readouts.',
        'The final validation-selected pointwise readout, both controls and their global validation thresholds were frozen before confirmation preparation.',
        'No model or threshold is re-selected from this confirmation, including when validation promotion was false.',
        'Prior known source lines/URLs and exact full/body/prefix hashes excluded; original249 lacks URL/line metadata.',
        'No semantic near-duplicate, domain-disjointness or new natural-dialogue claim.',
        '2000 paired source-document bootstrap resamples within each profile and full-length position bin.', '',
        'Model             Profile             System P  System R  System F1  Balanced acc  Five-role acc']
    for row in metrics:
        values = ['NA' if row[key] is None else f'{row[key]:.6f}' for key in
                  ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy', 'accuracy']]
        lines.append(f'{row["model"]:17s} {row["profile"]:19s} ' + '  '.join(values))
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
    plots(rows, probabilities, thresholds, positions)
    targets = {r['profile']: all(r[k] is not None and r[k] >= .9 for k in
        ['system_precision', 'system_recall', 'system_balanced_accuracy']) for r in metrics if r['model'] == 'selected'}
    verify_frozen(metadata)
    completed_probabilities(metadata, rows)
    artifacts = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['complete.json', 'status.json']]
    data.atomic_json(OUT / 'complete.json', dict(frozen=frozen, independent_source_documents=464,
        general_documents=400, additional_full_length_documents=64, up_to1024_subset_documents=64,
        validation_promotion_flag=frozen['primary_validation_promoted'], tests_used_for_tuning=False,
        source_sha256=data.digest(__file__), inference_sha256=data.digest(OUT / 'inference-complete.json'),
        working_target_pass=targets, all_profiles_and_full_tail_meet_90_percent=all(targets.values()),
        bootstrap_unit='Original source document', bootstrap_replicates=BOOTSTRAPS,
        report_sha256=data.digest(OUT / 'report.txt'),
        artifact_sha256={str(p.relative_to(OUT)): data.digest(p) for p in artifacts}))
    data.atomic_json(OUT / 'status.json', dict(stage='complete', validation_promoted=frozen['primary_validation_promoted'], targets=targets))
    print((OUT / 'report.txt').read_text(), flush=True)


def preflight():
    """CPU-only temporary fixtures; never prepare formal confirmation inputs."""
    global CACHE
    original = CACHE
    checks = []
    def fails(action, name):
        try:
            action()
        except (RuntimeError, KeyError):
            checks.append(name)
        else:
            raise RuntimeError(f'Fault fixture did not fail: {name}')
    try:
        with tempfile.TemporaryDirectory(prefix='systemness-long-confirm-') as temporary:
            CACHE = Path(temporary)
            inputs = pd.DataFrame(dict(prompt_ix=range(17), length=range(10, 27)))
            rows = pd.DataFrame(dict(prompt_ix=np.repeat(np.arange(17), 2)))
            signature = {'fixture': 'CPU only'}
            probabilities = open_probabilities(NAMES, len(rows))
            for values in probabilities.values():
                values[:] = [.5, .125, .125, .125, .125]
                values.flush()
            progress = progress_record(signature, inputs, rows, probabilities, 1, 16)
            same(verify_progress(progress, signature, inputs, rows, probabilities), (1, 16), 'Fixture valid resume')
            checks.append('valid-resume-with-finished-row-hashes')
            fails(lambda: verify_progress(progress, {'fixture': 'changed'}, inputs, rows, probabilities), 'signature-drift')
            fails(lambda: verify_progress(dict(progress, version=0), signature, inputs, rows, probabilities), 'old-progress-version')
            fails(lambda: verify_progress(dict(progress, rows_written=15), signature, inputs, rows, probabilities), 'wrong-completed-row-count')
            fails(lambda: verify_progress({k: v for k, v in progress.items() if k != 'completed_probability_sha256'}, signature, inputs, rows, probabilities), 'old-progress-without-probability-hash')
            probabilities['selected'][0] = [.25, .25, .25, .125, .125]
            fails(lambda: verify_progress(progress, signature, inputs, rows, probabilities), 'normalized-completed-row-corruption')
            probabilities['selected'][0] = [np.nan, 0, 0, 0, 0]
            fails(lambda: validate_probabilities(probabilities), 'non-finite-probability')
            probabilities['selected'][0] = [0, 0, 0, 0, 0]
            fails(lambda: validate_probabilities(probabilities), 'unwritten-probability-row')
            for name, values in probabilities.items():
                values._mmap.close()
            del probabilities
            missing = CACHE / 'base-attention-probabilities.f32'
            content = missing.read_bytes(); missing.unlink()
            fails(lambda: open_probabilities(NAMES, len(rows), require_existing=True), 'missing-probability-file')
            require(not missing.exists(), 'Missing-file preflight silently created an empty cache.')
            missing.write_bytes(content[:-4])
            fails(lambda: open_probabilities(NAMES, len(rows), require_existing=True), 'truncated-probability-file')
            fails(lambda: same({'threshold': .5}, {'threshold': .6}, 'Frozen fixture threshold'), 'frozen-threshold-drift')
            tok = long_data.load_tokenizer()
            body = tok.decode(tok.encode('tomato ' * 1400, add_special_tokens=False)[:1024])
            fixture_inputs, fixture_rows, fixture_tokens = [], [], []
            for role in data.ROLES:
                append_input([(role, body)], 'long-full', 400, True, tok, fixture_inputs, fixture_rows, fixture_tokens)
            same(Counter(r['role'] for r in fixture_rows), Counter({role: 1024 for role in data.ROLES}), 'Fixture full native labels')
            checks.append('five-native-role-1024-content-parser')
            calls = [('system', 'Follow the rules.'), ('user', 'Find context.'), ('tool_call', '{"query":"context"}'),
                ('tool', 'Context found.'), ('cot', 'Check this context.'), ('assistant', 'Here is the result.')]
            append_input(calls, 'multi', 0, True, tok, fixture_inputs, fixture_rows, fixture_tokens)
            checks.append('native-tool-call-cycle-parser')
            import py_compile
            py_compile.compile(__file__, cfile=str(CACHE / 'script.pyc'), doraise=True)
            checks.append('pycompile-to-temporary-directory')
    finally:
        CACHE = original
    print(json.dumps(dict(cpu_only=True, formal_prepare_run=False, checks=checks,
        source_sha256=data.digest(__file__)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'infer', 'report', 'all'])
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    require(not (args.preflight and args.stage), 'Choose either a formal stage or CPU preflight.')
    if args.preflight:
        preflight()
    elif args.stage:
        if args.stage in ['prepare', 'all']:
            prepare()
        if args.stage in ['infer', 'all']:
            infer()
        if args.stage in ['report', 'all']:
            report()
    else:
        parser.print_help()
