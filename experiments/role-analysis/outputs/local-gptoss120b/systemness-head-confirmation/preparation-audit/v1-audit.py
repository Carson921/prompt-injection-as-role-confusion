"""Read-only independent preparation audit; never opens probabilities."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

sys.path.insert(0, '/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import systemness_head_confirm as h
import systemness_long_data as ld
import systemness_data as d

def check(value, label):
    if not value:
        raise RuntimeError(label)

started = time.time()
metadata, inputs, rows, tokens = h.load_data()
keys = h.read_json(h.OUT / 'document-keys.json')
tok = ld.load_tokenizer()
special = {name: tok.convert_tokens_to_ids(name) for name in ld.SPECIAL}
ends = {special[x] for x in ['<|end|>', '<|return|>', '<|call|>']}
lookup = {int(ix): group for ix, group in rows.groupby('prompt_ix', sort=False)}
role_counts = Counter()
for item in inputs.itertuples():
    ids = tokens[item.prompt_ix]
    check(tok.encode(item.prompt, add_special_tokens=False) == ids, 'Retokenized input differs')
    starts = [i for i, value in enumerate(ids) if value == special['<|start|>']]
    expected = {}
    for mi, start in enumerate(starts):
        boundary = starts[mi + 1] if mi + 1 < len(starts) else len(ids)
        delimiters = [i for i in range(start + 1, boundary) if ids[i] == special['<|message|>']]
        terminations = [i for i in range(start + 1, boundary) if ids[i] in ends]
        check(len(delimiters) == len(terminations) == 1, 'Malformed native message')
        delimiter, end = delimiters[0], terminations[0]
        check(delimiter < end, 'End before body')
        header = tok.decode(ids[start + 1:delimiter])
        if header.startswith('functions.'):
            role = 'tool'
        elif header.startswith('assistant'):
            if 'to=functions.' in header:
                continue  # Explicitly outside the five-role content task.
            role = 'cot' if '<|channel|>analysis' in header else 'assistant'
        else:
            check(header in ['system', 'user'], 'Unknown native header')
            role = header
        for i in range(delimiter + 1, end):
            expected[i] = role
    group = lookup[item.prompt_ix]
    actual = dict(zip(group.token_ix, group.role))
    check(len(group) == len(actual) and actual == expected, 'Missing, duplicated, or wrong native content labels')
    check(group.document_id.eq(item.document_id).all(), 'Cross-source row ownership')
    offsets = group.token_ix - group.groupby('message_ix').token_ix.transform('min')
    check((offsets == group.body_offset).all(), 'Body offset differs from actual native boundary')
    role_counts.update(expected.values())

prior_hashes, prior_urls, prior_lines = h.exclusion_sets(tok, metadata['exclusion_input_sha256'])
check(len(prior_hashes) == metadata['exclusion_fingerprints'], 'Prior fingerprint count')
check(len(prior_urls) == metadata['known_prior_urls'] and len(prior_lines) == metadata['known_prior_source_lines'], 'Prior provenance counts')
check(len(keys) == len({k['shard_line'] for k in keys}) == 464, 'Source independence')
check(Counter(k['cohort'] for k in keys) == Counter({'general': 400, 'full-length': 64}), 'Source cohort sizes')
check(not ({k['shard_line'] for k in keys} & prior_lines), 'Reused known source line')
check(not ({k['source_url'] for k in keys if k['source_url']} & prior_urls), 'Reused known URL')
check(not ({k[f] for k in keys for f in ['body_sha256', 'short_body_sha256', 'full_body_sha256']} & prior_hashes), 'Reused full/body/prefix fingerprint')

wanted = {k['shard_line']: k for k in keys}
source_verified = 0
with gzip.open(ld.ARCHIVE, 'rt', encoding='utf-8') as stream:
    for line_ix, line in enumerate(stream):
        if line_ix not in wanted:
            continue
        key = wanted.pop(line_ix)
        article = json.loads(line)
        ids = tok.encode(article['text'], add_special_tokens=False)
        check(article.get('url') == key['source_url'] and len(ids) == key['original_tokens'], 'Pinned line URL/length')
        for field, text in [('full_body_sha256', article['text']), ('body_sha256', tok.decode(ids[:1024], skip_special_tokens=False)), ('short_body_sha256', tok.decode(ids[:192], skip_special_tokens=False))]:
            check(hashlib.sha256(text.encode()).hexdigest() == key[field], 'Pinned line content fingerprint')
        if key['cohort'] == 'full-length':
            check(key['body_tokens'] == 1024 and len(ids) >= 1024, 'Not a real full-length source')
        source_verified += 1
        if not wanted:
            break
check(not wanted and source_verified == 464, 'Missing source line')

full = rows[rows.profile == 'long-full']
tail = full[full.body_offset.between(512, 1023)]
check(len(full) == 64 * 5 * 1024 and len(tail) == 64 * 5 * 512, 'Wrong full/tail denominator')
check(full.groupby(['document_id', 'role']).size().eq(1024).all(), 'Missing full role version')
check(tail.groupby(['document_id', 'role']).size().eq(512).all(), 'Missing tail role version')
check(not (set(full.document_id) & set(rows[rows.profile != 'long-full'].document_id)), 'Full/general overlap')
report = dict(audit_scope='Preparation only; no probability, metric or test-based model selection access.',
    source_sha256=d.digest(__file__), data_sha256=d.digest(h.OUT / 'data-complete.json'),
    protocol_sha256=d.digest(h.OUT / 'protocol.json'), frozen_readouts_sha256=d.digest(h.OUT / 'frozen-readouts.json'),
    document_keys_sha256=d.digest(h.OUT / 'document-keys.json'),
    source_documents=464, general_documents=400, additional_full_length_documents=64, up_to1024_paired_subset=64,
    inputs=len(inputs), input_tokens=int(inputs.length.sum()), scored_rows=len(rows), role_counts=dict(role_counts),
    native_inputs_retokenized=True, all_native_content_labels_independently_segmented=True,
    all_selected_source_lines_replayed=True, known_prior_lines=len(prior_lines), known_prior_urls=len(prior_urls),
    prior_fingerprints=len(prior_hashes), no_known_source_collision=True,
    original249_unavailable_urls_and_lines_not_claimed=True, semantic_or_domain_disjointness_not_claimed=True,
    full_length_rows=len(full), tail_rows=len(tail), frozen_four_models_verified=True,
    gpu_model_loaded=False, probability_files_opened=False, seconds=time.time()-started)
path = Path('/tmp/root-head-confirmation-preparation-audit.json')
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
