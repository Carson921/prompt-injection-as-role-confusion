"""Independent CPU source replay after terminal preparation; no models/probs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0, str(ROOT / 'experiments/role-analysis'))
import systemness_data as data
import systemness_long_augmentation_data as b
import systemness_long_data as old
import systemness_head_confirm as h

EXPECTED_SOURCE = '7e1880eaf5d2f5b4cb955fb34052d1f7a4196b798d0f2c830ea5e56ec15a84fa'
EXPECTED_PLAN = '505352680e9d8191b476b2fdac289cf1e66f2287f36fff4fa875dbff9a706d10'
EXPECTED_COMPLETE = '6fe07499112eb816fb116905d7db245211297ec07b99cc45df5ec7be8efb847e'
FIELDS = ['body_sha256', 'full_body_sha256', 'truncated_body_sha256', 'short_body_sha256',
    'prefix192_sha256', 'prefix1024_sha256', 'full_text_sha256']


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())
def sha(value): return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
def canonical(value): return sha(json.dumps(value, sort_keys=True, separators=(',', ':')))


def active_preparers():
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: args = [(value.decode(errors='replace')) for value in (entry / 'cmdline').read_bytes().split(b'\0') if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if any(Path(value).name == 'systemness_long_augmentation_data.py' for value in args) \
                and ('prepare' in args or '--stage=prepare' in args): active.append(int(entry.name))
    return active


def terminal_gate():
    require(not active_preparers(), 'Builder prepare is still live: refuse partial source/cache inspection.')


def fingerprints(text, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    short = tokenizer.decode(ids[:192], skip_special_tokens=False)
    body = tokenizer.decode(ids[:1024], skip_special_tokens=False)
    return dict(full=sha(text), short=sha(short), body=sha(body)), ids, body


def actual():
    terminal_gate()  # Must precede every actual manifest/cache/source read.
    require(data.digest(b.__file__) == EXPECTED_SOURCE, 'Builder source changed.')
    require(data.digest(b.OUT / 'plan.json') == EXPECTED_PLAN, 'Declared source plan changed.')
    require(data.digest(b.OUT / 'data-complete.json') == EXPECTED_COMPLETE, 'Completed source cache changed.')
    plan, meta = read(b.OUT / 'plan.json'), read(b.OUT / 'data-complete.json')
    require(meta['plan_sha256'] == EXPECTED_PLAN and meta['source_sha256'] == EXPECTED_SOURCE
        and meta['source_signature'] == plan['source_signature'] and plan['protocol'] == b.protocol_spec(), 'Completed protocol identity mismatch.')
    bindings = dict(builder_source_sha256=EXPECTED_SOURCE, plan_sha256=EXPECTED_PLAN, data_complete_sha256=EXPECTED_COMPLETE,
        cache_sha256={}, audit_sha256={}, old_manifests={}, script_sha256=data.digest(__file__),
        parser_source_sha256=data.digest(ROOT / 'experiments/role-analysis/systemness_long_confirm.py'))
    for name, digest in meta['cache_sha256'].items():
        require(data.digest(b.CACHE / name) == digest, 'Cache byte SHA mismatch: ' + name); bindings['cache_sha256'][name] = digest
    for name, digest in meta['metadata_sha256'].items():
        require(data.digest(b.OUT / name) == digest, 'Audit byte SHA mismatch: ' + name); bindings['audit_sha256'][name] = digest
    require(set(meta['cache_sha256']) == {'inputs.parquet', 'rows.parquet', 'token-ids.json'}
        and set(meta['metadata_sha256']) == {'document-keys.json', 'exclusion-audit.json', 'scan-audit.json', 'independent-validation.json'},
        'Incomplete published cache or audit inventory.')
    require(meta['independent_native_labels_match'] is True and meta['all_body_tokens1024'] is True
        and not any(meta[name] for name in ['test_inputs_prepared', 'test_probabilities_read', 'test_metrics_read',
            'gpu_work_run', 'model_generation_run', 'downloads_run']), 'Development-only completion boundary mismatch.')
    for name, digest in meta['dependencies_sha256'].items(): require(data.digest(data.HERE / name) == digest, 'Builder dependency changed.')
    require(read(b.CACHE / 'initialized.json') == dict(plan_sha256=EXPECTED_PLAN, source_sha256=EXPECTED_SOURCE), 'Cache ownership mismatch.')
    require(data.digest(b.ARCHIVE) == b.ARCHIVE_SHA256, 'Pinned archive byte SHA mismatch.')
    bindings['archive_sha256'] = b.ARCHIVE_SHA256
    keys = read(b.OUT / 'document-keys.json'); inputs = pd.read_parquet(b.CACHE / 'inputs.parquet')
    rows = pd.read_parquet(b.CACHE / 'rows.parquet'); tokens = read(b.CACHE / 'token-ids.json')
    require(len(keys) == 352 and Counter(key['split'] for key in keys) == {'fit': 256, 'validation': 96}, 'Fixed source split changed.')
    require(len(inputs) == len(tokens) == 1760 and len(rows) == 389120
        and np.array_equal(inputs.prompt_ix, np.arange(1760)) and np.array_equal(rows.cache_ix, np.arange(389120))
        and np.array_equal(rows.sample_ix, 6000000000000 + np.arange(389120)), 'Cache dimensions/order/namespace mismatch.')
    require(rows.groupby('split').size().to_dict() == {'fit': 327680, 'validation': 61440}
        and set(rows.wrapper) == {'native'} and set(rows.source_set) == {'long-augmentation'}
        and set(rows.profile) == {'long-full'} and set(rows.role) == set(data.ROLES), 'Role/profile/split row counts changed.')
    require([key['document_id'] for key in keys] == list(range(6000000, 6000352))
        and [key['new_document_ix'] for key in keys] == list(range(352)), 'New document render namespace changed.')
    new_lines = {key['shard_line'] for key in keys}; require(len(new_lines) == 352 and min(new_lines) >= 4111 and max(new_lines) <= 12302, 'Fixed source window changed.')
    historical_hashes, historical_urls, historical_lines = set(), set(), set()
    def urls_for(url): return {value for value in [url, b.canonical_url(url)] if value}
    for name, expected in b.MANIFESTS.items():
        path = data.runtime.OUT / name; require(data.digest(path) == expected, 'Historical manifest SHA changed.')
        previous = read(path); bindings['old_manifests'][name] = dict(sha256=expected, records=len(previous))
        for key in previous:
            historical_lines.add(key['shard_line']); historical_hashes.update(key[field] for field in FIELDS if key.get(field))
            historical_urls.update(urls_for(key.get('source_url')))
    require(len(historical_lines) == 2828 and max(historical_lines) == 4110 and not historical_lines & new_lines, 'Historical/new line exclusion failed.')
    tokenizer = old.load_tokenizer(); checkpoint, tokenizer_sha = old.tokenizer_assets()
    bindings['tokenizer_sha256'] = tokenizer_sha; bindings['checkpoint_path'] = str(checkpoint)
    # Stream archive once, recovering all2828 historical originals and all352 new
    # exact originals, rather than only trusting stored (possibly short) hashes.
    wanted = historical_lines | new_lines; replay = {}; recovered = 0
    with gzip.open(b.ARCHIVE, 'rt', encoding='utf-8') as stream:
        for line_ix, line in enumerate(stream):
            if line_ix not in wanted: continue
            article = json.loads(line); hashes, ids, body = fingerprints(article['text'], tokenizer)
            if line_ix in historical_lines:
                historical_hashes.update(hashes.values()); historical_urls.update(urls_for(article.get('url'))); recovered += 1
            if line_ix in new_lines:
                require(len(ids) >= 1024 and tokenizer.encode(body, add_special_tokens=False) == ids[:1024], 'Actual source1024 prefix does not roundtrip.')
                replay[line_ix] = dict(hashes=hashes, ids=ids[:1024], body=body, original_tokens=len(ids), url=article.get('url'))
            wanted.remove(line_ix)
            if not wanted: break
    require(not wanted and recovered == 2828 and len(replay) == 352, 'Pinned source replay was incomplete.')
    bindings['original249_sha256'] = {}
    for path, expected in b.ORIGINAL.items(): require(data.digest(path) == expected, 'Original249 inventory SHA changed.'); bindings['original249_sha256'][str(path)] = expected
    original_inputs = pd.read_parquet(data.runtime.OUT / 'training_inputs.parquet', columns=['question']).question.unique().tolist()
    original_raw = read(data.runtime.OUT.parent / 'local-gptoss20b/raw_training_data.json')
    require(len(original_inputs) == len(original_raw) == 249, 'Original249 inventory count changed.')
    for text in original_inputs + [entry['text'] for entry in original_raw]: historical_hashes.update(fingerprints(text, tokenizer)[0].values())
    # Check dedup across the two new splits as well as every prior recovered text.
    new_hashes, new_urls = set(), set()
    for key in keys:
        source = replay[key['shard_line']]; hashes = source['hashes']; urls = urls_for(source['url'])
        require(key['full_body_sha256'] == key['full_text_sha256'] == hashes['full']
            and key['body_sha256'] == key['prefix1024_sha256'] == hashes['body']
            and key['short_body_sha256'] == key['prefix192_sha256'] == hashes['short']
            and key['source_url'] == source['url'] and key['canonical_source_url'] == b.canonical_url(source['url'])
            and key['original_tokens'] == source['original_tokens'] and key['body_tokens'] == 1024,
            'Source key full/192/1024/url/token fingerprint mismatch.')
        require(not set(hashes.values()) & (historical_hashes | new_hashes) and not urls & (historical_urls | new_urls), 'Historical or within-new exact fingerprint/URL collision.')
        rank = sha('\0'.join(map(str, [b.SELECT_SEED, b.REVISION, b.SHARD, key['shard_line'], hashes['full']])))
        require(key['rank_sha256'] == rank, 'Source selection rank hash mismatch.')
        new_hashes.update(hashes.values()); new_urls.update(urls)
    split_order = sorted(keys, key=lambda key: sha('\0'.join(map(str,
        [b.SPLIT_SEED, b.REVISION, b.SHARD, key['shard_line'], key['full_body_sha256']]))))
    fit_lines = {key['shard_line'] for key in split_order[:256]}
    require({key['shard_line'] for key in keys if key['split'] == 'fit'} == fit_lines, 'Independent seeded split replay mismatch.')
    selection_order = sorted(keys, key=lambda key: (key['rank_sha256'], key['shard_line']))
    require([key['source_rank'] for key in selection_order] == list(range(352)), 'Selection SHA order mismatch.')
    by_doc = {key['document_id']: key for key in keys}; grouped = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
    full_positions, tail_rows, shared = 0, Counter(), Counter()
    for entry in inputs.itertuples():
        key = by_doc[entry.document_id]; source = replay[key['shard_line']]
        ids = tokenizer.encode(entry.prompt, add_special_tokens=False); labels = h.parse_labels(ids, tokenizer)
        require(ids == tokens[entry.prompt_ix] and len(ids) == entry.length and len(labels) == 1024
            and set(labels.values()) == {entry.role}, 'Independent native labels or cached IDs mismatch.')
        body_positions = np.asarray(sorted(labels), dtype=np.int64); body_ids = [ids[int(pos)] for pos in body_positions]
        require(body_ids == source['ids'] and tokenizer.decode(body_ids, skip_special_tokens=False) == source['body'], 'Five-role variants do not share the actual1024 source body.')
        count = 256 if key['split'] == 'fit' else 128
        offset = (np.arange(count, dtype=np.int64) * 1023) // (count - 1)
        require(np.array_equal(offset, np.linspace(0, 1023, count, dtype=int)), 'Independent integer uniform-grid formula mismatch.')
        group = grouped[entry.prompt_ix]
        require(len(group) == count and np.array_equal(group.body_offset, offset)
            and np.array_equal(group.token_ix, body_positions[offset]) and np.array_equal(group.token_id, np.asarray(ids)[body_positions[offset]]),
            'Sampled row/body-offset/actual-native-token mapping mismatch.')
        for field in ['uid', 'split', 'document_id', 'group_id', 'role', 'source_rank', 'new_document_ix']:
            require(group[field].eq(getattr(entry, field)).all(), 'Prompt/row ownership mismatch: ' + field)
        require(entry.uid == f'long-augmentation:{entry.prompt_ix}' and entry.split == key['split'] and entry.group_id == key['group_id']
            and entry.body_tokens == 1024 and entry.sampled_body_tokens == count and bool(entry.bos) == bool(entry.document_id % 2), 'Native prompt/source namespace mismatch.')
        require(not rows.sample_ix.duplicated().any(), 'Sample namespace repeats.') if entry.prompt_ix == 0 else None
        shared[(entry.document_id, entry.role)] += 1; tail_rows[entry.split] += int(np.sum(offset >= 512)); full_positions += len(labels)
    require(len(shared) == 352 * 5 and all(value == 1 for value in shared.values()) and full_positions == 1760 * 1024
        and dict(tail_rows) == {'fit': 163840, 'validation': 30720}, 'Five-role full-body/tail coverage mismatch.')
    require(not set(rows.loc[rows.split.eq('fit'), 'group_id']) & set(rows.loc[rows.split.eq('validation'), 'group_id']), 'Fit/validation source group leakage.')
    # Detect concurrent edits across replay; no production artifact is written.
    terminal_gate()
    require(data.digest(b.__file__) == EXPECTED_SOURCE and data.digest(b.OUT / 'data-complete.json') == EXPECTED_COMPLETE
        and data.digest(b.OUT / 'plan.json') == EXPECTED_PLAN, 'Source binding changed during replay.')
    for name, digest in bindings['cache_sha256'].items(): require(data.digest(b.CACHE / name) == digest, 'Cache changed during replay.')
    report = dict(bindings=bindings, documents=352, fit_documents=256, validation_documents=96, prompts=1760, rows=389120,
        split_rows={'fit': 327680, 'validation': 61440}, full_native_body_positions=full_positions, tail_rows=dict(tail_rows),
        recovered_historical_lines=recovered, original249_inventory_counts=[249, 249],
        exclusion=dict(stored_manifest_count=5, all_known_lines_full_192_1024_recovered=True,
            both249_full_192_1024_recovered=True, no_exact_line_url_full_body_prefix_collision=True,
            historical_hashes_sha256=canonical(sorted(historical_hashes)), historical_urls_sha256=canonical(sorted(historical_urls)),
            original249_url_line_provenance_complete=False, semantic_near_duplicate_or_domain_disjointness_claimed=False),
        checks=dict(all_selected_source_keys_replayed=True, independent_split_replayed=True, native_independent_parser=True,
            all_five_roles_share_actual1024_body=True, every_sample_offset_and_token_checked=True, cache_namespace_disjoint=True,
            source_hashes_unchanged_at_end=True, prepare_process_terminal=True),
        limitation='Audits every selected source and every known old line plus both249 inventories. Does not rescan unselected8192-window candidates or claim semantic/domain deduplication. Old source text is read only for exclusion fingerprints.',
        gpu_work_run=False, model_or_classifier_probabilities_read=False, actual_prepare_rerun=False, formal_outputs_written=False)
    target = Path('/tmp/feature_review_long_augmentation_actual_audit.json'); target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(target), report_sha256=data.digest(target), **report)), flush=True)


def preflight():
    with patch(__name__ + '.active_preparers', return_value=[123]), patch(__name__ + '.read', side_effect=AssertionError('Read before terminal')):
        try: actual()
        except RuntimeError: pass
        else: raise RuntimeError('Active prepare gate failed.')
    for count in [128, 256]:
        require(np.array_equal((np.arange(count, dtype=np.int64) * 1023) // (count - 1), np.linspace(0, 1023, count, dtype=int)), 'Grid formula mismatch.')
    class Chars:
        def encode(self, value, add_special_tokens=False): return list(map(ord, value))
        def decode(self, value, skip_special_tokens=False): return ''.join(map(chr, value))
    text = 'A' * 192 + 'B' * 832 + 'C' * 100
    hashes, ids, body = fingerprints(text, Chars())
    require(hashes == dict(full=sha(text), short=sha(text[:192]), body=sha(text[:1024])) and len(body) == 1024, 'Fingerprint toy mismatch.')
    print(json.dumps(dict(preflight=True, script_sha256=data.digest(__file__), live_gate_before_reads=True,
        integer_uniform_grids_exact=True, full_192_1024_fingerprints=True, actual_data_read=False, gpu_work_run=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true'); action.add_argument('--actual', action='store_true'); args = parser.parse_args()
    if args.preflight: preflight()
    elif args.actual:
        with patch.object(h, 'selections', side_effect=AssertionError('Model selection/probability access forbidden')), \
                patch.object(h.head_probes, 'verified_completion', side_effect=AssertionError('Classifier probability access forbidden')):
            actual()
    else: parser.print_help()
