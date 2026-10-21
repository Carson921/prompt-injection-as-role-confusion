"""Matched recheck of already reported natural-dialogue inputs, never a new test.

Run only after the 464-source confirmation finishes with the same final readout.
No model/threshold selection occurs here. Nothing runs without --stage; use
--preflight for CPU-only temporary fixtures. Existing frozen sources are reused.
All-user formal metrics cover every native-parser content token. Original
semantic rows retain their identities; added formal-only rows have no invented
message function and never enter semantic projections or scatter plots.
"""
import argparse
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
import systemness_long_confirm as confirmation
import systemness_long_data as long_data
import systemness_probes as probes


OUT = data.runtime.OUT / 'systemness-natural-recheck'
CACHE = data.CACHE / 'natural-recheck'
NAMES = list(confirmation.NAMES)
SPECS = dict(confirmation.SPECS)
BATCH_SIZE = 8
BOOTSTRAPS = 2000
LEGACY_ROWS = 324798
ADDITIONAL_FORMAL_ROWS = 972
TOTAL_ROWS = LEGACY_ROWS + ADDITIONAL_FORMAL_ROWS
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
SOURCES = {
    'base': dict(folder=data.CACHE, metadata=data.OUT / 'data-complete.json',
        metadata_sha256='45af8da9ef20b87eb81adf026a714aeedfb92315eb73e460ee5098df0ccaa88a',
        profile='natural', wrappers=['native', 'all-user', 'no-tags'], templates=8),
    'unseen': dict(folder=data.CACHE / 'supplement', metadata=data.OUT / 'supplement-data-complete.json',
        metadata_sha256='aa6fde3fc1b8998d746d8ecb49924b8def28d1ed9b31d4ab0d9c43ae0cdb3702',
        profile='natural-unseen-system', wrappers=['native', 'all-user'], templates=32),
}
KEYS = data.OUT / 'natural-conversation-keys.json'
KEYS_SHA256 = '3f64c06cdf103d7d53dc51bf78da36ccdcb0384c8b29828418b82348b40280f5'
DEPENDENCIES = sorted(set(confirmation.DEPENDENCIES + [
    'systemness_natural_recheck.py', 'systemness_challenges.py',
    'systemness_final_select.py', 'systemness_long_probes.py', 'systemness_paired_probes.py']))
COUNT_FIELDS = ['tp', 'fp', 'fn', 'tn', 'correct', 'tokens', 'cot_correct', 'cot_tokens', 'cot_as_tool']
require = long_data.require
same = confirmation.same
validate_probabilities = confirmation.validate_probabilities


def read(path):
    return json.loads(Path(path).read_text())


def checked_manifest(path, expected):
    same(data.digest(path), expected, f'Published legacy manifest {path}')
    return read(path)


def source_snapshot(frozen):
    # These hashes pin the already reported inputs, not a newly built benchmark.
    manifests, cache_hashes = {}, {}
    for name, source in SOURCES.items():
        metadata = checked_manifest(source['metadata'], source['metadata_sha256'])
        manifests[str(source['metadata'])] = source['metadata_sha256']
        for filename in FILES:
            path = source['folder'] / filename
            same(data.digest(path), metadata['cache_sha256'][filename], f'Published {name} cache {filename}')
            cache_hashes[str(path)] = metadata['cache_sha256'][filename]
        if name == 'base':
            same(metadata['natural_test_conversations'], 144, 'Published conversation count')
            natural_csv = data.HERE / 'data/conversations/gptoss-20b.csv'
            same(data.digest(natural_csv), metadata['natural_source_sha256'], 'Published natural dialogue source')
            cache_hashes[str(natural_csv)] = metadata['natural_source_sha256']
        else:
            same(metadata['unseen_system_conversations'], 144, 'Unseen-wording conversation count')
            same(len(metadata['unseen_system_templates']), 32, 'Unseen wording count')
            same(metadata['unseen_system_formats'], ['native', 'all-user'], 'Published unseen formats')
    same(data.digest(KEYS), KEYS_SHA256, 'Published conversation/template keys')
    keys = [k for k in read(KEYS) if k['split'] == 'test']
    require(len(keys) == len({k['group_id'] for k in keys}) == 144, 'Expected the original 144 conversation groups.')
    require(set(k['system_template'] for k in keys) == set(range(8)), 'Missing base system template.')
    _, _, selection_sources, records = confirmation.selections()
    final_plan = confirmation.FINAL_SELECTION.parent / 'plan.json'
    same(data.digest(final_plan), records['selected']['final_plan_sha256'], 'Frozen final selection plan')
    same(data.digest(data.HERE / 'systemness_final_select.py'), records['selected']['source_sha256'],
         'Frozen final selection source')
    # Enforce the requested order without reading any 464-source input or label.
    complete_path = confirmation.OUT / 'complete.json'
    complete = read(complete_path)
    same(complete['frozen'], frozen, 'Completed 464-source confirmation readouts')
    same(complete['independent_source_documents'], 464, 'Completed primary confirmation size')
    return dict(legacy_manifests_sha256=manifests, legacy_cache_sha256=cache_hashes,
        natural_keys_path=str(KEYS), natural_keys_sha256=KEYS_SHA256,
        selection_sources={name: dict(path=str(path), sha256=data.digest(path)) for name, path in selection_sources.items()},
        final_selection_plan_sha256=data.digest(final_plan), primary_confirmation_complete_sha256=data.digest(complete_path),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        source_sha256=data.digest(__file__))


def verify(metadata=None):
    paths, thresholds, frozen = confirmation.verify_frozen(metadata)
    snapshot = source_snapshot(frozen)
    if metadata is not None:
        same(metadata['source_snapshot'], snapshot, 'Matched-recheck source/cache/dependency snapshot')
    return paths, thresholds, frozen, snapshot


def complete_all_user(inputs, legacy_rows, tokens, tok):
    """Append parser-only all-user positions; preserve the complete legacy prefix.

    This pure CPU helper never reads/writes a cache or assigns semantic ownership
    to serialization delimiters omitted by the original semantic span sampler.
    """
    require(legacy_rows.sample_ix.is_unique and legacy_rows.sample_ix.ge(0).all(),
            'Legacy sample identities must be unique and nonnegative.')
    require(not {'legacy_semantic_row', 'legacy_sample_ix'} & set(legacy_rows.columns),
            'Only an unaugmented legacy table may be completed.')
    rows = legacy_rows.copy()
    rows['legacy_semantic_row'] = True
    rows['legacy_sample_ix'] = pd.array(rows.sample_ix, dtype='Int64')
    lookup = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
    extra, added = [], {name: 0 for name in SOURCES}
    for row in inputs[inputs.wrapper == 'all-user'].itertuples():
        selected = lookup[row.prompt_ix]
        ids = tokens[row.prompt_ix]
        labels = confirmation.parse_labels(ids, tok)
        require(set(labels.values()) == {'user'} and selected.role.eq('user').all(),
                'All-user native-parser labels must be user.')
        require(set(selected.token_ix) <= set(labels), 'Legacy all-user position is not formal content.')
        missing = sorted(set(labels) - set(selected.token_ix))
        if not missing:
            continue
        encoded = tok(row.prompt, add_special_tokens=False, return_offsets_mapping=True)
        same(encoded['input_ids'], ids, 'All-user completion exact input/token ids')
        for pos in missing:
            entry = selected.iloc[0].to_dict()
            left, right = encoded['offset_mapping'][pos]
            entry.update(token_ix=pos, token_id=ids[pos], token=row.prompt[left:right],
                role='user', semantic_role=None, message_ix=None,
                sample_ix=-1 - len(extra), legacy_sample_ix=None, legacy_semantic_row=False)
            extra.append(entry)
            added[row.source_set] += 1
    if extra:
        rows = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)
    else:
        rows = rows.reset_index(drop=True)
    rows['sample_ix'] = rows.sample_ix.astype(np.int64)
    rows['legacy_semantic_row'] = rows.legacy_semantic_row.astype(bool)
    rows['legacy_sample_ix'] = pd.array(rows.legacy_sample_ix, dtype='Int64')
    rows['message_ix'] = pd.array(rows.message_ix, dtype='Int64')
    same(rows.iloc[:len(legacy_rows)].sample_ix.tolist(), legacy_rows.sample_ix.tolist(), 'Original legacy sample order')
    for name in legacy_rows.columns:
        same(rows.iloc[:len(legacy_rows)][name].tolist(), legacy_rows[name].tolist(), f'Original legacy column {name}')
    return rows, added


def validate_row_identity(rows, expected_legacy_rows):
    require(rows.sample_ix.is_unique and rows.index.equals(pd.RangeIndex(len(rows))), 'Row identities are not unique/reset.')
    require(rows.legacy_semantic_row.isin([True, False]).all(), 'Invalid legacy row flags.')
    legacy = rows[rows.legacy_semantic_row]
    extra = rows[~rows.legacy_semantic_row]
    same(len(legacy), expected_legacy_rows, 'Original semantic row count')
    require(legacy.index.tolist() == list(range(len(legacy))), 'Legacy rows must remain an unchanged prefix.')
    require(legacy.sample_ix.ge(0).all() and legacy.legacy_sample_ix.notna().all(), 'Invalid original sample namespace.')
    same(legacy.sample_ix.tolist(), legacy.legacy_sample_ix.tolist(), 'Preserved legacy sample identities')
    require(set(legacy.semantic_role) <= set(data.ROLES), 'Unknown original semantic message function.')
    same(extra.sample_ix.tolist(), list(range(-1, -1 - len(extra), -1)), 'Formal-only negative sample namespace')
    require(extra.wrapper.eq('all-user').all() and extra.role.eq('user').all(), 'Formal-only rows must be all-user content.')
    require(extra[['semantic_role', 'message_ix', 'legacy_sample_ix']].isna().all().all(),
            'Formal-only positions must not have invented semantic ownership.')


def validate_inputs(inputs, rows, tokens, tok, *, expected_groups=144, expected_rows=TOTAL_ROWS,
                    expected_legacy_rows=LEGACY_ROWS):
    same(inputs.prompt_ix.tolist(), list(range(len(inputs))), 'Recheck prompt indices')
    require(len(inputs) == len(tokens), 'Input/token list mismatch.')
    validate_row_identity(rows, expected_legacy_rows)
    require(set(rows.split) == {'test'}, 'Recheck must contain only already reported tests.')
    same(len(rows), expected_rows, 'Complete formal/legacy semantic content rows')
    lookup = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
    require(set(lookup) == set(inputs.prompt_ix), 'Missing retained input content.')
    skipped = {}
    for source_name, source in SOURCES.items():
        group = inputs[inputs.source_set == source_name]
        same(set(group.wrapper), set(source['wrappers']), f'{source_name} wrapper coverage')
        require(group.group_id.nunique() == expected_groups, 'Conversation membership changed.')
        for wrapper in source['wrappers']:
            subset = group[group.wrapper == wrapper]
            require(len(subset) == subset.group_id.nunique() == expected_groups, 'Duplicate or missing wrapper input.')
    for row in inputs.itertuples():
        ids = tokens[row.prompt_ix]
        same(len(ids), row.length, 'Cached tokenizer length')
        same(tok.encode(row.prompt, add_special_tokens=False), ids, 'Exact legacy serialized input/token ids')
        selected = lookup[row.prompt_ix]
        require(selected.wrapper.eq(row.wrapper).all() and selected.group_id.eq(row.group_id).all(), 'Row/input join changed.')
        require(selected.token_ix.is_unique and selected.token_ix.between(0, len(ids) - 1).all(), 'Invalid body token positions.')
        same(np.asarray(ids)[selected.token_ix].tolist(), selected.token_id.tolist(), 'Legacy token identities')
        if row.wrapper == 'no-tags':
            require(selected.role.isna().all(), 'Untagged text must not have formal-role labels.')
            require(selected.legacy_semantic_row.all(), 'Untagged inputs may only contain legacy semantic rows.')
            continue
        labels = confirmation.parse_labels(ids, tok)
        for entry in selected.itertuples():
            same(labels.get(entry.token_ix), entry.role, 'Native token-state formal label')
        if row.wrapper == 'all-user':
            require(selected.role.eq('user').all(), 'All-user labels must be formally user.')
        same(set(selected.token_ix), set(labels), 'Complete native-parser formal content coverage')
        key = f'{row.source_set}/{row.wrapper}'
        skipped[key] = skipped.get(key, 0) + len(labels) - len(selected)
    base = set(inputs[inputs.source_set == 'base'].group_id)
    unseen = set(inputs[inputs.source_set == 'unseen'].group_id)
    same(base, unseen, 'Paired unseen-wording conversation membership')
    return skipped


def prepare():
    _, _, frozen, snapshot = verify()
    if (OUT / 'data-complete.json').exists():
        load_data()
        return
    require(not (CACHE / 'progress.json').exists() and not (OUT / 'inference-complete.json').exists(),
            'Inference state exists without a complete matched input cache.')
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    lock = dict(frozen=frozen, source_snapshot=snapshot)
    path = OUT / 'preparation-lock.json'
    if path.exists():
        same(read(path), lock, 'Matched recheck preparation lock')
    else:
        data.atomic_json(path, lock)
    templates = {k['group_id']: k['system_template'] for k in read(KEYS) if k['split'] == 'test'}
    input_parts, row_parts, tokens, next_prompt = [], [], [], 0
    for name, source in SOURCES.items():
        frame = pd.read_parquet(source['folder'] / 'inputs.parquet')
        frame = frame[(frame.split == 'test') & (frame.profile == source['profile'])].copy()
        require(np.all(np.diff(frame.prompt_ix) > 0), 'Preserve original input-file order for template mapping.')
        native = frame[frame.wrapper == 'native']
        # Unseen styles were assigned in this exact native input order. Sorting
        # group_id lexically or by the inference batch would change the mapping.
        mapping = templates if name == 'base' else {group: i % 32 for i, group in enumerate(native.group_id)}
        same(set(mapping), set(templates), f'{name} conversation/template membership')
        frame['source_set'] = name
        frame['original_prompt_ix'] = frame.prompt_ix
        frame['system_template'] = frame.group_id.map(mapping)
        prompt_map = {int(old): next_prompt + i for i, old in enumerate(frame.prompt_ix)}
        original_tokens = read(source['folder'] / 'token-ids.json')
        tokens.extend(original_tokens[int(ix)] for ix in frame.original_prompt_ix)
        frame['prompt_ix'] = frame.prompt_ix.map(prompt_map)
        next_prompt += len(frame)
        selected = pd.read_parquet(source['folder'] / 'rows.parquet')
        selected = selected[(selected.split == 'test') & (selected.profile == source['profile'])].copy()
        require(set(selected.prompt_ix) == set(prompt_map), 'Published input/row subsets disagree.')
        selected['source_set'] = name
        selected['original_prompt_ix'] = selected.prompt_ix
        selected['prompt_ix'] = selected.prompt_ix.map(prompt_map)
        selected['system_template'] = selected.group_id.map(mapping)
        input_parts.append(frame); row_parts.append(selected)
    inputs = pd.concat(input_parts, ignore_index=True)
    legacy_rows = pd.concat(row_parts, ignore_index=True)
    require(len(inputs) == 720 and int(inputs.length.sum()) == 334464, 'Expected exactly the published 720 inputs.')
    tok = long_data.load_tokenizer()
    rows, added = complete_all_user(inputs, legacy_rows, tokens, tok)
    same(added, dict(base=486, unseen=486), 'Additional all-user formal content positions')
    skipped = validate_inputs(inputs, rows, tokens, tok)
    same(verify()[2:], (frozen, snapshot), 'Frozen matched-recheck inputs')
    inputs.to_parquet(CACHE / FILES[0], index=False)
    rows.to_parquet(CACHE / FILES[1], index=False)
    (CACHE / FILES[2]).write_text(json.dumps(tokens, separators=(',', ':')))
    lengths = np.sort(inputs.length.to_numpy(), kind='stable')
    padded = sum(int(lengths[i:i + BATCH_SIZE].max()) * len(lengths[i:i + BATCH_SIZE])
                 for i in range(0, len(lengths), BATCH_SIZE))
    counts = inputs.groupby(['source_set', 'profile', 'wrapper'], as_index=False).agg(
        inputs=('prompt_ix', 'size'), conversations=('group_id', 'nunique'), input_tokens=('length', 'sum'))
    row_counts = rows.assign(formal_row=rows.wrapper.ne('no-tags'),
        additional_formal_row=~rows.legacy_semantic_row).groupby(
            ['source_set', 'profile', 'wrapper'], as_index=False).agg(
            scored_or_projection_rows=('sample_ix', 'size'), legacy_semantic_rows=('legacy_semantic_row', 'sum'),
            formal_rows=('formal_row', 'sum'), additional_formal_rows=('additional_formal_row', 'sum'))
    counts = counts.merge(row_counts, on=['source_set', 'profile', 'wrapper'], validate='one_to_one')
    counts.to_csv(OUT / 'input-counts.csv', index=False)
    data.atomic_json(OUT / 'data-complete.json', dict(frozen=frozen, source_snapshot=snapshot,
        purpose='Matched recheck of already reported tests; not additional independent confirmation; never used for selection.',
        inputs=720, conversations=144, rows=len(rows), legacy_semantic_rows=LEGACY_ROWS,
        additional_formal_rows=ADDITIONAL_FORMAL_ROWS, additional_formal_rows_by_source=added, input_tokens=334464,
        sorted_batch8_padded_tokens=padded, batches=90, templates=dict(base=8, unseen=32),
        unseen_is_paired_wording_variant=True, independent_conversation_count_unchanged=True,
        legacy_semantic_row_identities_preserved=True, semantic_projections_only_legacy_rows=True,
        all_user_formal_content_coverage='Exact equality to independent native-parser position sets.',
        parser_content_tokens_not_in_retained_rows=skipped,
        no_tags_formal_accuracy_claimed=False, tests_used_for_selection=False,
        cache_sha256={name: data.digest(CACHE / name) for name in FILES}))


def load_data():
    metadata = read(OUT / 'data-complete.json')
    verify(metadata)
    for name, sha in metadata['cache_sha256'].items():
        same(data.digest(CACHE / name), sha, f'Matched input cache {name}')
    inputs, rows = (pd.read_parquet(CACHE / name) for name in FILES[:2])
    tokens = read(CACHE / FILES[2])
    require(len(inputs) == len(tokens) == metadata['inputs'] == 720 and len(rows) == metadata['rows'] == TOTAL_ROWS,
            'Matched input/row counts changed.')
    same(metadata['legacy_semantic_rows'], LEGACY_ROWS, 'Cached legacy semantic row count')
    same(metadata['additional_formal_rows'], ADDITIONAL_FORMAL_ROWS, 'Cached formal-only row count')
    validate_row_identity(rows, LEGACY_ROWS)
    return metadata, inputs, rows, tokens


def open_probabilities(row_count, *, require_existing=False, readonly=False):
    paths = {name: CACHE / f'{name}-probabilities.f32' for name in NAMES}
    for path in paths.values():
        require(not (require_existing or readonly) or path.is_file(), f'Missing saved probability file: {path}.')
        require(not path.exists() or path.stat().st_size == row_count * 5 * 4,
                f'Truncated or incorrectly sized probability file: {path}.')
    return {name: np.memmap(path, mode='r' if readonly else 'r+' if path.exists() else 'w+',
        dtype=np.float32, shape=(row_count, 5)) for name, path in paths.items()}


def inference_signature(metadata):
    verify(metadata)
    return dict(data_sha256=data.digest(OUT / 'data-complete.json'), frozen=metadata['frozen'],
        source_snapshot=metadata['source_snapshot'], batch_size=BATCH_SIZE, features=SPECS,
        cache_policy='probabilities only; one shared native MXFP4 forward for three frozen pointwise readouts')


def completed_probabilities(metadata, rows):
    complete = read(OUT / 'inference-complete.json')
    same(complete['signature'], inference_signature(metadata), 'Completed matched inference signature')
    same(set(complete['probability_sha256']), set(NAMES), 'Completed model coverage')
    probabilities = open_probabilities(len(rows), require_existing=True, readonly=True)
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
    same(BATCH_SIZE, confirmation.BATCH_SIZE, 'Shared checkpoint batch layout')
    metadata, inputs, rows, tokens = load_data()
    paths, _, _, _ = verify(metadata)
    signature = inference_signature(metadata)
    if (OUT / 'inference-complete.json').exists():
        completed_probabilities(metadata, rows)
        return
    progress_path = CACHE / 'progress.json'
    probabilities = open_probabilities(len(rows), require_existing=progress_path.exists())
    start_batch, written = 0, 0
    if progress_path.exists():
        start_batch, written = confirmation.verify_progress(read(progress_path), signature, inputs, rows, probabilities)
    nets = {name: probes.load_model(path, device='cuda:0') for name, path in paths.items()}
    runtime = data.runtime
    old_atomic, old_status = runtime.atomic_json, runtime.status
    runtime.atomic_json = lambda path, value: data.atomic_json(OUT / Path(path).name, value)
    runtime.status = lambda stage, **fields: print(json.dumps(dict(stage=stage, **fields)), flush=True)
    original_hooks = pipeline.install_hooks
    captured = {}
    def receive(name, value):
        captured[name] = value.detach().to('cpu', torch.float16).reshape(-1, 2880)
    started = time.time()
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
                require(torch.equal(before, after) and torch.isfinite(after).all(), 'Pointwise capture hooks altered model output.')
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
                    require(torch.isfinite(raw).all(), 'Non-finite matched content activations.')
                    p = net((raw - mean) / scale).exp().cpu().numpy()
                    validate_probabilities({name: p})
                    probabilities[name][targets] = p
                    del raw, p
                written += len(targets); states.clear()
                if (bi + 1) % 16 == 0 or bi + 1 == len(loader):
                    for values in probabilities.values():
                        values.flush()
                    verify(metadata)
                    data.atomic_json(progress_path, confirmation.progress_record(signature, inputs, rows,
                        probabilities, bi + 1, written))
                    status = dict(stage='matched-natural-recheck', batch=bi + 1, batches=len(loader),
                        rows=written, seconds=time.time() - started)
                    data.atomic_json(OUT / 'status.json', status); print(json.dumps(status), flush=True)
        same(written, len(rows), 'Complete matched probability row count')
        same(inference_signature(metadata), signature, 'Matched inference source snapshot')
        load_data()
        for values in probabilities.values():
            values.flush()
        validate_probabilities(probabilities)
        data.atomic_json(OUT / 'inference-complete.json', dict(signature=signature, rows=written,
            probability_sha256={name: data.digest(CACHE / f'{name}-probabilities.f32') for name in NAMES},
            all_finite=True, all_normalized=True, raw_hidden_states_persisted=False, seconds=time.time() - started))
    finally:
        pipeline.install_hooks = original_hooks
        runtime.atomic_json, runtime.status = old_atomic, old_status


def document_counts(rows, p, threshold, model):
    # Formal rows only; untagged projections must not become pseudo-labels.
    formal = rows[rows.wrapper != 'no-tags']
    y = probes.labels(formal); values = p[formal.index]
    frame = formal[['profile', 'wrapper', 'group_id', 'system_template']].copy()
    truth, detected = y == 0, values[:, 0] >= threshold
    predicted = values.argmax(1)
    frame['tp'] = truth & detected; frame['fp'] = ~truth & detected
    frame['fn'] = truth & ~detected; frame['tn'] = ~truth & ~detected
    frame['correct'] = predicted == y; frame['tokens'] = 1
    frame['cot_correct'] = (y == 2) & (predicted == 2)
    frame['cot_tokens'] = y == 2
    frame['cot_as_tool'] = (y == 2) & (predicted == 4)
    return frame.groupby(['profile', 'wrapper', 'group_id', 'system_template'], as_index=False)[COUNT_FIELDS].sum().assign(model=model)


def legacy_all_user_metrics(rows, probabilities, thresholds):
    """Comparable old-denominator metrics, kept separate from full formal scores."""
    metrics = []
    legacy = rows[rows.legacy_semantic_row & rows.wrapper.eq('all-user')]
    for name, p in probabilities.items():
        for profile, group in legacy.groupby('profile', sort=False):
            values = p[group.index]
            metrics.append(dict(model=name, profile=profile, wrapper='all-user',
                metric_mode='legacy-semantic-subset-formal-user-only',
                conversations=int(group.group_id.nunique()), tokens=len(group),
                system_templates=int(group.system_template.nunique()),
                accuracy=float((values.argmax(1) == 1).mean()),
                system_false_positive_rate=float((values[:, 0] >= thresholds[name]).mean()),
                system_threshold=thresholds[name]))
    return metrics


def summarize(rows, probabilities, thresholds):
    metrics, docs, projections = [], [], []
    for name, p in probabilities.items():
        for (profile, wrapper), group in rows.groupby(['profile', 'wrapper'], sort=False):
            record = dict(model=name, profile=profile, wrapper=wrapper,
                conversations=int(group.group_id.nunique()), tokens=len(group),
                system_templates=int(group.system_template.nunique()))
            if wrapper == 'native':
                result = probes.metrics(p[group.index], probes.labels(group), thresholds[name])
                result.update({f'role_recall_{role}': value for role, value in result['recall'].items()})
                cot = group.role.eq('cot').to_numpy()
                result['cot_as_tool_rate'] = float((p[group.index][cot].argmax(1) == 4).mean()) if cot.any() else None
                metrics.append(record | dict(metric_mode='formal-native', **result))
            elif wrapper == 'all-user':
                require(group.role.eq('user').all(), 'All-user formal labels changed.')
                values = p[group.index]
                metrics.append(record | dict(metric_mode='formal-user-only',
                    accuracy=float((values.argmax(1) == 1).mean()),
                    system_false_positive_rate=float((values[:, 0] >= thresholds[name]).mean()),
                    system_threshold=thresholds[name]))
            else:
                require(wrapper == 'no-tags' and group.role.isna().all(), 'Unexpected projection-only labels.')
                metrics.append(record | dict(metric_mode='semantic-projections-only'))
        docs.append(document_counts(rows, p, thresholds[name], name))
        legacy = rows[rows.legacy_semantic_row]
        frame = legacy[['source_set', 'profile', 'wrapper', 'group_id', 'system_template', 'message_ix', 'semantic_role']].copy()
        for i, role in enumerate(data.ROLES):
            frame[f'p_{role}'] = p[legacy.index, i]
        projections.append(frame.groupby(['source_set', 'profile', 'wrapper', 'group_id', 'system_template',
            'message_ix', 'semantic_role'], as_index=False).agg(
            **{f'p_{role}': (f'p_{role}', 'mean') for role in data.ROLES}, tokens=('p_system', 'size')).assign(model=name))
    return metrics, pd.concat(docs, ignore_index=True), pd.concat(projections, ignore_index=True)


def count_values(array):
    values = evaluation.binary_values(array[..., :6])
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    values['cot_recall'] = div(array[..., 6], array[..., 7])
    values['cot_as_tool_rate'] = div(array[..., 8], array[..., 7])
    return values


def bootstrap(document):
    comparisons, intervals, template_counts = [], [], []
    for pi, ((profile, wrapper), group) in enumerate(document.groupby(['profile', 'wrapper'], sort=True)):
        template = group.groupby(['model', 'system_template'], as_index=False)[COUNT_FIELDS].sum()
        template_counts.append(template.assign(profile=profile, wrapper=wrapper))
        arrays, ids = {}, None
        for name in NAMES:
            current = template[template.model == name].sort_values('system_template')
            if ids is None:
                ids = current.system_template.tolist()
            same(current.system_template.tolist(), ids, 'Paired system-template bootstrap units')
            arrays[name] = current[COUNT_FIELDS].to_numpy(dtype=np.float64)
        rng = np.random.RandomState(data.SEED + 720 + pi)
        index = rng.randint(len(ids), size=(BOOTSTRAPS, len(ids)))
        point = {name: count_values(array.sum(0)) for name, array in arrays.items()}
        draws = {name: count_values(array[index].sum(1)) for name, array in arrays.items()}
        fields = ['accuracy', 'system_false_positive_rate'] if wrapper == 'all-user' else list(point[NAMES[0]])
        common = dict(profile=profile, wrapper=wrapper, conversations=int(group.group_id.nunique()),
            bootstrap_units=len(ids), resampling_unit='system-template', bootstrap_replicates=BOOTSTRAPS)
        for name in NAMES:
            for metric in fields:
                lo, hi = np.quantile(draws[name][metric], [.025, .975])
                intervals.append(common | dict(model=name, metric=metric, value=float(point[name][metric]),
                    ci95_low=float(lo), ci95_high=float(hi)))
        for before, after in [('residual', 'base-attention'), ('residual', 'selected'), ('base-attention', 'selected')]:
            for metric in fields:
                lo, hi = np.quantile((draws[after][metric] - draws[before][metric]) * 100, [.025, .975])
                comparisons.append(common | dict(baseline_model=before, comparison_model=after, metric=metric,
                    baseline=float(point[before][metric]), comparison=float(point[after][metric]),
                    gain_percentage_points=float(100 * (point[after][metric] - point[before][metric])),
                    ci95_low=float(lo), ci95_high=float(hi)))
    return comparisons, intervals, pd.concat(template_counts, ignore_index=True)


def plots(rows, probabilities, thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = OUT / 'plots'
    folder.mkdir(exist_ok=True)
    colors = dict(system='#90a1b9', user='#00a6f4', cot='#fd9a00', assistant='#00d492', tool='#a855f7')
    labels = ['Systemness', 'Userness', 'CoTness', 'Asstness', 'Toolness']
    titles = ['Residual control', 'Published attention control', 'Final validation-selected readout']
    wrappers = dict(native='Correct native tags', **{'all-user': 'Everything in user tags', 'no-tags': 'No role tags'})
    base = rows[rows.source_set.eq('base') & rows.legacy_semantic_row]
    groups = sorted(base.group_id.unique())
    chosen = [groups[i] for i in np.linspace(0, len(groups) - 1, min(12, len(groups)), dtype=int)]
    for group_id in chosen:
        for wrapper, title in wrappers.items():
            subset = base[(base.group_id == group_id) & (base.wrapper == wrapper)].sort_values('token_ix')
            indices, x = subset.index.to_numpy(), subset.token_ix.to_numpy()
            fig, axes = plt.subplots(5, 3, figsize=(15, 7), sharex=True, sharey=True)
            for col, name in enumerate(NAMES):
                for i, label in enumerate(labels):
                    ax = axes[i, col]
                    for role, color in colors.items():
                        mask = subset.semantic_role.to_numpy() == role
                        ax.scatter(x[mask], probabilities[name][indices[mask], i], s=3, color=color, alpha=.7, rasterized=True)
                    if i == 0:
                        ax.axhline(thresholds[name], color='#4b5563', lw=.7, ls='--')
                    for boundary in subset.groupby('message_ix').token_ix.min():
                        ax.axvline(boundary, color='#d1d5db', lw=.5, zorder=0)
                    ax.set_ylim(-.03, 1.03); ax.set_yticks([0, 1], ['0%', '100%'])
                    ax.spines[['top', 'right']].set_visible(False)
                    if col == 0:
                        ax.set_ylabel(label, fontsize=9)
                axes[0, col].set_title(titles[col], fontsize=10)
                axes[-1, col].set_xlabel('Token position in full serialized input')
            handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=4, label=role)
                       for role, color in colors.items()]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1), ncol=5, frameon=False, fontsize=8)
            fig.suptitle(f'{group_id} | {title} | matched recheck', y=.945, fontsize=11)
            fig.text(.5, .01, 'Legacy semantic positions only | colors: original message functions | all-user formal labels: user | no-tags: projections only | raw probabilities',
                ha='center', fontsize=8)
            fig.tight_layout(rect=[0, .03, 1, .91])
            for extension in ['png', 'pdf']:
                fig.savefig(folder / f'{group_id}-{wrapper}.{extension}', dpi=160)
            plt.close(fig)
    (folder / 'gallery-index.md').write_text('Matched recheck of already reported natural dialogues. Fixed equally spaced original conversation IDs; three formats and three frozen readouts per figure. Only legacy semantic positions are plotted; colors retain original message functions. Added formal-only all-user positions enter full formal metrics, not these semantic plots. Untagged inputs have no formal accuracy claim.\n\n' +
        '\n'.join(f'- [{group}-{wrapper}]({group}-{wrapper}.png)' for group in chosen for wrapper in wrappers) + '\n')
    data.atomic_json(OUT / 'gallery.json', dict(conversations=chosen, wrappers=list(wrappers), panels=5,
        model_columns=3, png=len(chosen) * 3, pdf=len(chosen) * 3, score_based_selection=False,
        raw_unsmoothed_probabilities=True, old_test_matched_recheck=True, semantic_positions='legacy-only'))


def report():
    metadata, _, rows, _ = load_data()
    _, thresholds, frozen, _ = verify(metadata)
    inference, probabilities = completed_probabilities(metadata, rows)
    metrics, documents, projections = summarize(rows, probabilities, thresholds)
    legacy_metrics = legacy_all_user_metrics(rows, probabilities, thresholds)
    comparisons, intervals, template_counts = bootstrap(documents)
    data.atomic_json(OUT / 'metrics.json', metrics)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in metrics]).to_csv(OUT / 'metrics.csv', index=False)
    data.atomic_json(OUT / 'legacy-all-user-metrics.json', legacy_metrics)
    pd.DataFrame(legacy_metrics).to_csv(OUT / 'legacy-all-user-metrics.csv', index=False)
    documents.to_csv(OUT / 'document-counts.csv', index=False)
    projections.to_csv(OUT / 'message-projections.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    pd.DataFrame(comparisons).to_csv(OUT / 'paired-comparisons.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUT / 'bootstrap-metrics.csv', index=False)
    template_counts.to_csv(OUT / 'system-template-counts.csv', index=False)
    lines = ['Matched recheck of already reported natural-dialogue tests', '',
        'This is not a new independent confirmation and is never used for model or threshold selection.',
        'Same original 144 conversations: 432 base inputs across native/all-user/no-tags, plus 288 paired unseen-wording inputs across native/all-user.',
        '32 unseen system wordings are assigned across the same 144 conversations, not 32 variants per conversation.',
        '720 exact cached inputs, 334464 input tokens; 325770 scored/projection rows = 324798 unchanged legacy semantic rows + 972 additional all-user formal-only positions.',
        'Full all-user formal denominators: base 64437, unseen wording 66618. Legacy-subset denominators: base 63951, unseen wording 66132.',
        'Original GPT-OSS-20B-generated dialogue content remains capped at 192 tokens per message; no new dialogue-generation claim.',
        'Frozen native MXFP4 on four A40s; shared residual/local-attention states; three pointwise readouts with original fixed thresholds.',
        f'Final primary: {frozen["primary_candidate"]}; originating-trial validation promotion flag: {frozen["primary_validation_promoted"]}.',
        'All-user labels are formally user: report user accuracy and system false-positive rate, not system recall/F1.',
        'Formal-only rows have no semantic role/message assignment; formal document counts and template bootstrap include them. Projections and plots use only legacy semantic rows.',
        'No-tags inputs have no formal-role labels: only original-function probability projections and raw scatter plots.',
        '2000 paired bootstrap resamples of system templates: 8 base clusters or 32 unseen clusters; no token/conversation pseudo-replication.',
        'Every all-user formal position matches the independent native parser. The legacy subset is reported separately for direct comparison with the published denominator. Source-domain generalization is not established.', '',
        'Model             Profile                  Wrapper     Accuracy  System P  System R  System F1  System FPR  CoT recall']
    for row in metrics:
        if row['wrapper'] == 'no-tags':
            lines.append(f'{row["model"]:17s} {row["profile"]:24s} no-tags     semantic projections only')
            continue
        values = [row.get(key) for key in ['accuracy', 'system_precision', 'system_recall', 'system_f1', 'system_false_positive_rate']]
        values.append(row.get('recall', {}).get('cot'))
        text = ['NA' if value is None else f'{value:.6f}' for value in values]
        lines.append(f'{row["model"]:17s} {row["profile"]:24s} {row["wrapper"]:11s} ' + '  '.join(text))
    lines.extend(['', 'Legacy semantic subset only: all-user accuracy/FPR on the original published denominator.',
        'Model             Profile                    Tokens  User accuracy  System FPR'])
    for row in legacy_metrics:
        lines.append(f'{row["model"]:17s} {row["profile"]:26s} {row["tokens"]:6d}  {row["accuracy"]:.6f}       {row["system_false_positive_rate"]:.6f}')
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
    plots(rows, probabilities, thresholds)
    verify(metadata); completed_probabilities(metadata, rows)
    artifacts = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['complete.json', 'status.json']]
    data.atomic_json(OUT / 'complete.json', dict(frozen=frozen, source_snapshot=metadata['source_snapshot'],
        matched_recheck_only=True, newly_independent_test=False, tests_used_for_selection=False,
        conversations=144, inputs=720, rows=TOTAL_ROWS, retained_rows=LEGACY_ROWS, legacy_semantic_rows=LEGACY_ROWS,
        additional_formal_rows=ADDITIONAL_FORMAL_ROWS, templates=dict(base=8, unseen=32),
        semantic_projections_only_legacy_rows=True, legacy_all_user_metrics_reported_separately=True,
        bootstrap_unit='system-template', bootstrap_replicates=BOOTSTRAPS,
        no_tags_formal_accuracy_claimed=False, report_sha256=data.digest(OUT / 'report.txt'),
        inference_sha256=data.digest(OUT / 'inference-complete.json'),
        artifact_sha256={str(p.relative_to(OUT)): data.digest(p) for p in artifacts}))
    data.atomic_json(OUT / 'status.json', dict(stage='complete', matched_recheck_only=True))
    print((OUT / 'report.txt').read_text(), flush=True)


def preflight():
    """CPU-only fixtures for formal/semantic separation and shared resume checks."""
    global CACHE
    original = CACHE
    checks = []
    def fails(action, name):
        try:
            action()
        except (RuntimeError, KeyError):
            checks.append(name)
        else:
            raise RuntimeError(f'Fault fixture accepted: {name}')
    try:
        with tempfile.TemporaryDirectory(prefix='natural-recheck-fixture-') as temporary:
            CACHE = Path(temporary)
            path = CACHE / 'manifest.json'; path.write_text('{"fixture": true}')
            expected = data.digest(path); checked_manifest(path, expected)
            path.write_text('{"fixture":false}')
            fails(lambda: checked_manifest(path, expected), 'legacy-manifest-drift')
            tok = long_data.load_tokenizer()
            inputs, entries, tokens = [], [], []
            for source_name, source in SOURCES.items():
                for doc in range(2):
                    messages = [('system', 'Follow the request.'), ('user', 'What is a tomato?'),
                        ('cot', 'Consider the context.'), ('assistant', 'It is a fruit.')]
                    for wrapper in source['wrappers']:
                        text, spans = data.serialize(messages, bos=True, wrapper=wrapper)
                        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                        prompt = len(inputs)
                        for pos, (left, right) in enumerate(encoded['offset_mapping']):
                            for span in spans:
                                if left < span['end'] and right > span['start']:
                                    if left >= span['start'] and right <= span['end']:
                                        entries.append(dict(prompt_ix=prompt, token_ix=pos, role=span['role'], semantic_role=span['semantic_role'],
                                            message_ix=span['message_ix'], profile=source['profile'], split='test', group_id=f'fixture-{doc}',
                                            document_id=-1, wrapper=wrapper, token_id=encoded['input_ids'][pos], token=text[left:right], sample_ix=len(entries),
                                            source_set=source_name, original_prompt_ix=prompt, system_template=doc))
                                    break
                        inputs.append(dict(prompt_ix=prompt, prompt=text, split='test', profile=source['profile'], group_id=f'fixture-{doc}',
                            wrapper=wrapper, length=len(encoded['input_ids']), source_set=source_name, original_prompt_ix=prompt, system_template=doc))
                        tokens.append(encoded['input_ids'])
            inputs, legacy_rows = pd.DataFrame(inputs), pd.DataFrame(entries)
            rows, added = complete_all_user(inputs, legacy_rows, tokens, tok)
            require(sum(added.values()) > 0, 'Fixture must contain missing all-user formal positions.')
            validate_inputs(inputs, rows, tokens, tok, expected_groups=2, expected_rows=len(rows),
                expected_legacy_rows=len(legacy_rows))
            require(rows.iloc[:len(legacy_rows)].sample_ix.tolist() == legacy_rows.sample_ix.tolist(), 'Fixture legacy IDs changed.')
            checks.append('formal-only-negative-namespace-and-null-semantic-ownership')
            dropped = rows.drop(rows[~rows.legacy_semantic_row].index[-1]).reset_index(drop=True)
            fails(lambda: validate_inputs(inputs, dropped, tokens, tok, expected_groups=2, expected_rows=len(dropped),
                expected_legacy_rows=len(legacy_rows)), 'missing-all-user-parser-position')
            checks.append('native-all-user-no-tags-label-and-token-identity')
            y = probes.labels(rows)
            semantic = rows.semantic_role.map({role: i for i, role in enumerate(data.ROLES)}).to_numpy()
            target = np.where(y >= 0, y, semantic).astype(np.int64)
            p = np.full((len(rows), 5), .025, dtype=np.float32)
            p[np.arange(len(rows)), target] = .9
            # Only new formal-only rows are deliberately wrong. Full formal
            # metrics/bootstrap must see them; the old subset remains perfect.
            formal_only = ~rows.legacy_semantic_row.to_numpy()
            p[formal_only] = [.9, .025, .025, .025, .025]
            probabilities = {name: p.copy() for name in NAMES}; thresholds = {name: .5 for name in NAMES}
            metrics, document, projections = summarize(rows, probabilities, thresholds)
            require(len(metrics) == 15 and len(projections) > 0, 'Missing fixture format/model outputs.')
            for metric in metrics:
                if metric['wrapper'] == 'no-tags':
                    require('accuracy' not in metric and 'system_f1' not in metric, 'Untagged formal score leaked.')
                elif metric['wrapper'] == 'all-user':
                    current = rows[(rows.profile == metric['profile']) & rows.wrapper.eq('all-user')]
                    error_rate = float((~current.legacy_semantic_row).mean())
                    require(metric['accuracy'] == 1 - error_rate and metric['system_false_positive_rate'] == error_rate,
                            'Formal-only all-user positions did not affect full formal scores.')
                    require(not {'system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy'} & set(metric),
                            'Undefined all-user system score leaked.')
            legacy_metrics = legacy_all_user_metrics(rows, probabilities, thresholds)
            require(len(legacy_metrics) == 6 and all(row['accuracy'] == 1 and row['system_false_positive_rate'] == 0
                    for row in legacy_metrics), 'Full and legacy all-user denominators were mixed.')
            same(int(projections.tokens.sum()), len(legacy_rows) * len(NAMES), 'Legacy-only semantic projection denominator')
            require(int(document.tokens.sum()) == int(rows.wrapper.ne('no-tags').sum()) * len(NAMES),
                    'Formal-only rows are missing from document counts.')
            comparisons, intervals, template = bootstrap(document)
            require(all(row['bootstrap_units'] == 2 and row['resampling_unit'] == 'system-template' for row in comparisons),
                    'Fixture bootstrap unit changed.')
            require(all(row['gain_percentage_points'] == row['ci95_low'] == row['ci95_high'] == 0 for row in comparisons),
                    'Identical fixture models have nonzero paired gain.')
            require(not any(row['wrapper'] == 'all-user' and row['metric'] not in ['accuracy', 'system_false_positive_rate']
                            for row in intervals), 'All-user bootstrap contains undefined system scores.')
            for row in intervals:
                if row['wrapper'] == 'all-user':
                    metric = next(m for m in metrics if m['model'] == row['model'] and m['profile'] == row['profile'] and m['wrapper'] == 'all-user')
                    same(row['value'], metric[row['metric']], 'Full-formal bootstrap denominator')
            checks.extend(['formal-only-errors-change-full-not-legacy-metrics-or-projections',
                'formal-versus-projection-metric-policy', 'paired-system-template-bootstrap-not-token-or-conversation'])
            saved = open_probabilities(len(rows))
            for name in NAMES:
                saved[name][:] = probabilities[name]; saved[name].flush()
            index = confirmation.completed_indices(inputs, rows, 1)
            signature = {'fixture': True}
            progress = confirmation.progress_record(signature, inputs, rows, saved, 1, len(index))
            same(confirmation.verify_progress(progress, signature, inputs, rows, saved), (1, len(index)), 'Fixture valid resume')
            fails(lambda: confirmation.verify_progress(progress, {'fixture': False}, inputs, rows, saved), 'changed-resume-signature')
            saved['selected'][index[0]] = [.2] * 5
            fails(lambda: confirmation.verify_progress(progress, signature, inputs, rows, saved), 'normalized-completed-probability-corruption')
            saved['selected'][index[0]] = [0] * 5
            fails(lambda: validate_probabilities(saved), 'unwritten-probability-row')
            for values in saved.values():
                values._mmap.close()
            missing = CACHE / 'selected-probabilities.f32'; payload = missing.read_bytes(); missing.unlink()
            fails(lambda: open_probabilities(len(rows), require_existing=True), 'missing-probability-file')
            require(not missing.exists(), 'Missing probability cache was silently recreated.')
            missing.write_bytes(payload[:-4])
            fails(lambda: open_probabilities(len(rows), require_existing=True), 'truncated-probability-file')
            import py_compile
            py_compile.compile(__file__, cfile=str(CACHE / 'script.pyc'), doraise=True)
            checks.append('pycompile-temporary-output')
    finally:
        CACHE = original
    print(json.dumps(dict(cpu_only=True, formal_prepare_run=False, checks=checks, source_sha256=data.digest(__file__)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'infer', 'report', 'all'])
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    require(not (args.stage and args.preflight), 'Choose either a formal stage or CPU preflight.')
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
