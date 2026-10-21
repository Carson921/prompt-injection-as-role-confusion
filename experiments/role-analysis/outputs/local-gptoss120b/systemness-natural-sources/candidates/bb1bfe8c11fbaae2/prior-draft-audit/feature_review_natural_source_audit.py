"""Read-only source/key audit; only small audit/official API metadata goes to /tmp.

No dataset download, generation, classifier probabilities, GPU, formal writes.
The existing builder is replayed only up to its summary/write section.
"""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
os.environ['ROLE_ANALYSIS_LOAD_PATH'] = '/dev/shm/role-analysis-gptoss120b'
sys.path.insert(0, '/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
pa.set_cpu_count(1); pa.set_io_thread_count(1); torch.set_num_threads(2)
import systemness_long_data as ld

def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()

base = ld.data.CACHE / 'natural-source-snapshots'
draft_path, availability_path = base / 'candidate-source-keys-draft.json', base / 'availability.json'
draft, availability = json.loads(draft_path.read_text()), json.loads(availability_path.read_text())
assert digest(draft_path.read_bytes()) == 'b9c643ac5b7e6fc5549a0f12dd9e443ea445a77fbf68c1582262819948d1e655'
assert digest(availability_path.read_bytes()) == draft['source_availability_sha256']
file_records, upstream = [], []
for record in availability['files']:
    path = Path(record['path']); raw = path.read_bytes()
    assert len(raw) == record['size']
    sha = digest(raw)
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert sha == record['sha256_verified'] and blob == record['git_blob_sha1_verified']
    if 'sha256' in record: assert sha == record['sha256']
    if 'git_blob_sha1' in record: assert blob == record['git_blob_sha1']
    assert path == base / record['repo'].replace('/', '--') / record['commit'] / record['file']
    expected_url = f"https://huggingface.co/datasets/{record['repo']}/resolve/{record['commit']}/{record['file']}"
    assert record['url'] == expected_url
    file_records.append(dict(repo=record['repo'], revision=record['commit'], file=record['file'], size=len(raw), sha256=sha, git_blob_sha1=blob))
assert sum(record['size'] for record in file_records) == availability['total_snapshot_bytes'] == 54850365

# Fetch only tiny official pinned-commit tree metadata, never a source archive.
for repo, commit in sorted({(r['repo'], r['commit']) for r in availability['files']}):
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{commit}/data?recursive=true&expand=true'
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read(2 * 1024**2 + 1)
        assert len(raw) <= 2 * 1024**2
        metadata = json.loads(raw)
        for file in [r for r in file_records if r['repo'] == repo]:
            entry = next(item for item in metadata if item['path'] == file['file'])
            assert entry['type'] == 'file' and entry['size'] == file['size']
            if entry.get('lfs'):
                assert entry['lfs']['oid'] == file['sha256']
            else:
                assert entry['oid'] == file['git_blob_sha1']
        path = Path('/tmp') / ('feature_review_official_' + repo.replace('/', '--') + '.json')
        path.write_bytes(raw)
        upstream.append(dict(repo=repo, revision=commit, official_metadata_verified=True, url=url,
            metadata_path=str(path), metadata_sha256=digest(raw)))
    except Exception as error:
        upstream.append(dict(repo=repo, revision=commit, official_metadata_verified=False, url=url, error=repr(error)))

script = Path('/tmp/natural_source_dedup_draft.py')
assert digest(script.read_bytes()) == draft['audit_source_sha256']
prefix = script.read_text().split('summary=dict(stage=', 1)[0]
assert 'write_text(' not in prefix and 'urlopen(' not in prefix
namespace = {'__name__': 'read_only_natural_source_replay'}
exec(compile(prefix, str(script), 'exec'), namespace)
assert namespace['selected'] == draft['source_only_keys']
selected = draft['source_only_keys']; tok = namespace['tok']
old, raw, clean = namespace['old'], namespace['raw'], namespace['clean']
oa, tc, tc_old = namespace['oasst'], namespace['tc'], namespace['tc_old']
assert len(old) == len(raw) == 225 and len(clean) == 207
assert old.conv_id.nunique() == raw.conv_id.nunique() == 200 and clean.conv_id.nunique() == 184
assert set(map(tuple, raw[['dataset','conv_id','user_query_ix']].to_numpy())) == set(map(tuple, old[['dataset','conv_id','user_query_ix']].to_numpy()))
failed = set(old.conv_id) - set(clean.conv_id)
assert len(failed) == 16

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()

def canonical_urls(text):
    result = set()
    for match in re.findall(r'https?://[^\s<>\"\'\)\]]+', text):
        part = urllib.parse.urlsplit(match.rstrip('.,;:!?'))
        query = sorted((k,v) for k,v in urllib.parse.parse_qsl(part.query, keep_blank_values=True)
                       if not k.lower().startswith('utm_'))
        result.add(urllib.parse.urlunsplit((part.scheme.lower(), part.netloc.lower(), part.path.rstrip('/'),
            urllib.parse.urlencode(query), '')))
    return result

def independent_keys(text):
    sanitized = text
    for marker in ld.RESERVED:
        sanitized = sanitized.replace(marker, '[' + marker[2:-2] + ']')
    variants = [text, normalize(text), sanitized, normalize(sanitized)]
    result = {('full', digest(value)) for value in variants}
    result |= {('prefix192', digest(np.asarray(tok.encode(value, add_special_tokens=False)[:192], dtype=np.int64).tobytes())) for value in variants}
    result |= {('url', value) for value in canonical_urls(text)}
    return result

old_keys = set().union(*(independent_keys(text) for text in old.user_query))
old_tree_ids = set(namespace['old_oa_ids']); old_tc_ids = set(namespace['old_tc_ids'])
assert len(old_tree_ids) == 100 and len(old_tc_ids) == 108
assert tc.conv_id.is_unique and tc_old.conv_id.is_unique
assert tc.set_index('conv_id').sort_index().equals(tc_old.set_index('conv_id').sort_index())
used = set(); split_ids = collections.defaultdict(set); collisions = collections.Counter()
selected_tree_user_rows = 0
for row in selected:
    ident = (row['dataset'], row['native_id'])
    assert ident not in split_ids[row['split']]
    split_ids[row['split']].add(ident)
    if row['dataset'] == 'oasst':
        assert row['native_id'] not in old_tree_ids
        messages = oa[oa.message_tree_id.eq(row['native_id']) & oa.role.eq('prompter')]
        assert messages.lang.eq('en').all() and messages.tree_state.eq('ready_for_export').all()
        selected_tree_user_rows += len(messages)
        roots = messages[messages.parent_id.isna()]
        assert len(roots) == 1 and roots.iloc[0].message_id == row['root_message_id'] == row['native_id']
        texts = messages.text.tolist()
    else:
        assert row['native_id'] not in old_tc_ids
        texts = tc[tc.conv_id.astype(str).eq(row['native_id'])].user_input.tolist()
        assert len(texts) == 1
    keys = set().union(*(independent_keys(text) for text in texts))
    assert digest(json.dumps(sorted(keys))) == row['exclusion_keys_sha256']
    for kind, value in keys & (old_keys | used): collisions[kind] += 1
    assert not keys & old_keys and not keys & used
    assert row['rank_sha256'] == digest('natural-source-draft-v1:' + row['dataset'] + ':' + row['revision'] + ':' + row['native_id'])
    used |= keys
assert selected_tree_user_rows == 991 and len(selected) == 448
assert all(split_ids[a].isdisjoint(split_ids[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
assert [r['rank_sha256'] for r in selected] == sorted(r['rank_sha256'] for r in selected)
assert collections.Counter((r['split'],r['dataset']) for r in selected) == {
    ('fit','oasst'):128,('fit','toxicchat0124'):128,('validation','oasst'):32,
    ('validation','toxicchat0124'):32,('test','oasst'):64,('test','toxicchat0124'):64}
assert not set().union(*(independent_keys(text) for text in old[old.conv_id.isin(failed)].user_query)) & used
assert digest(draft_path.read_bytes()) == 'b9c643ac5b7e6fc5549a0f12dd9e443ea445a77fbf68c1582262819948d1e655'
report = dict(draft_sha256=digest(draft_path.read_bytes()), availability_sha256=digest(availability_path.read_bytes()),
    source_files=file_records, official_metadata=upstream, all_official_metadata_verified=all(x['official_metadata_verified'] for x in upstream),
    exact_draft_source_keys_replayed=True, old_requested_conversations=200, old_requested_turns=225,
    old_raw_user_turns=225, old_clean_user_turns=207, old_complete_conversations=184, old_failed_conversations=16,
    all_old_failed_prompts_excluded=True, old_native_oasst_trees=100, old_tc_unique_requested_prompts=100,
    old_tc_native_ids_excluded=108, tc_0124_and_1123_train_source_identity=True,
    selected_roots=448, split_roots=dict(fit=256,validation=64,test=128), sources_balanced_each_split=True,
    selected_oasst_all_native_tree_user_rows_covered=991, full_normalized_sanitized_prefix192_url_overlap_counts=dict(collisions),
    all_pairwise_split_native_ids_disjoint=True, selection_sha_rank_only=True, model_scores_read=False,
    gpu_run=False, generation_run=False, formal_writes=False,
    limitations=['Old source revisions remain unknown; exact text recovery is not retroactive source pinning.',
        'Deduplication covers declared exact/normalized/token-prefix/canonical-URL keys, not semantic near-duplicates.',
        'New template catalog and source assignments/generation failure rules still require freezing before generation.'])
Path('/tmp/feature_review_natural_source_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
