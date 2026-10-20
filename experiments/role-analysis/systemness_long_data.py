"""Prepare denser 1024-token coverage from the original fit/validation documents.

This CPU-only helper never loads model weights or test examples. Run --preflight
for a small in-memory tokenizer check. The default action only prints usage;
--prepare explicitly creates the declared, resumable data cache.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

import pandas as pd

import systemness_data as data


OUT = data.runtime.OUT / 'systemness-long-full'
CACHE = data.CACHE / 'long-full'
ARCHIVE = data.runtime.OUT / 'feature-search/fresh/cache/c4-validation.json.gz'
BODY_CAP = 1024
SAMPLE_CAP = {'fit': 256, 'validation': 128}
SAMPLE_START = 2_000_000_000
EXPECTED = {'fit': 132, 'validation': 31}
EXPECTED_OLD = {'fit': 26, 'validation': 10}
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
SPECIAL = ['<|start|>', '<|message|>', '<|end|>', '<|return|>', '<|call|>']
RESERVED = SPECIAL + ['<|channel|>', '<|startoftext|>']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tokenizer_assets():
    # Check strings before touching paths: /workspace is an inaccessible mount.
    checkpoint = os.environ.get('ROLE_ANALYSIS_LOAD_PATH', '/dev/shm/role-analysis-gptoss120b')
    require(not (checkpoint == '/workspace' or checkpoint.startswith('/workspace/')),
            'Use the verified /dev/shm checkpoint; /workspace must not be accessed.')
    folder = Path(checkpoint).resolve()
    require(not (str(folder) == '/workspace' or str(folder).startswith('/workspace/')),
            'Tokenizer checkpoint resolves to forbidden /workspace.')
    names = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    require(all((folder / name).is_file() for name in names), 'Local tokenizer assets are missing.')
    if (folder / 'added_tokens.json').exists():
        names.append('added_tokens.json')
    return folder, {name: data.digest(folder / name) for name in names}


def load_tokenizer():
    folder, _ = tokenizer_assets()
    os.environ['ROLE_ANALYSIS_LOAD_PATH'] = str(folder)
    if os.environ.get('HF_HOME', '').startswith('/workspace'):
        os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
    return data.runtime.tokenizer()


def source_protocol():
    base_path = data.OUT / 'plan.json'
    base = json.loads(base_path.read_text())
    require(base['source_revision'] == '1588ec454efa1a09f29cd18ddd04fe05fc8653a2',
            'The pinned C4 revision changed.')
    require(base['source_shard_sha256'] ==
            '1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f',
            'The pinned C4 archive hash changed.')
    keys_path = data.OUT / 'document-keys.json'
    # Test keys exist in this immutable manifest, but only fit/validation source
    # records are ever decoded or tokenized by this entry point.
    keys = [key for key in json.loads(keys_path.read_text()) if key['split'] in EXPECTED]
    require(dict(Counter(key['split'] for key in keys)) == {'fit': 1200, 'validation': 200},
            'The original fit/validation membership changed.')
    require(len({key['document_id'] for key in keys}) == len(keys), 'Duplicate document ids.')
    require(len({key['shard_line'] for key in keys}) == len(keys), 'Duplicate source lines.')
    for field in ['body_sha256', 'full_body_sha256', 'source_url']:
        fit = {key[field] for key in keys if key['split'] == 'fit' and key.get(field)}
        val = {key[field] for key in keys if key['split'] == 'validation' and key.get(field)}
        require(not fit & val, f'Original fit/validation overlap: {field}.')
    folder, tokenizer_sha = tokenizer_assets()
    protocol = dict(
        version=1, profile='long-full', roles=list(data.ROLES), wrapper='native',
        eligible_documents=EXPECTED, body_minimum_tokens=BODY_CAP, body_cap_tokens=BODY_CAP,
        sampled_content_tokens_per_role=SAMPLE_CAP, sample_ix_start=SAMPLE_START,
        sampling='Stable data.uniform_indices: 256 fit / 128 validation uniform body-token positions including first and last.',
        selection='All original fit/validation documents whose original text has at least 1024 tokenizer tokens.',
        split_policy='Preserve original split, document_id and c4-{document_id} group_id; role variants stay together.',
        bos_policy='bool(document_id % 2)',
        independent_document_count_unchanged=True,
        previous_long_single_eligible_documents=EXPECTED_OLD,
        newly_covered_existing_documents={'fit': 106, 'validation': 21},
        purpose='Length coverage of existing source documents; these are not new independent documents.',
        replacement_policy='During training, replace old long-single rows for these selected documents with long-full rows. Validation retains the original profiles for matched comparisons and adds the long-full cohort; its 10 overlapping long documents are paired variants, not independent new documents.',
        test_policy='No test input, feature, label, prediction, selection result or confirmation set is loaded. Test entries in the fixed split manifest are excluded before source record decoding.',
        source_revision=base['source_revision'], source_shard=base['source_shard'],
        source_shard_sha256=base['source_shard_sha256'],
        parent_plan_sha256=data.digest(base_path), document_keys_sha256=data.digest(keys_path),
        tokenizer_path=str(folder), tokenizer_sha256=tokenizer_sha,
        source_sha256={name: data.digest(data.HERE / name) for name in
                       ['systemness_long_data.py', 'systemness_data.py', 'run_local_gptoss120b.py']})
    return protocol, keys


def immutable_json(path, value):
    if path.exists():
        require(json.loads(path.read_text()) == value,
                f'Existing {path.name} belongs to a different protocol; refusing to overwrite.')
    else:
        data.atomic_json(path, value)


def write_plan(protocol):
    path = OUT / 'plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        require({key: value for key, value in old.items() if key != 'created_unix_time'} == protocol,
                'Long-full source/protocol changed; use a separately declared output directory.')
    else:
        # Written before source tokenization, document selection, or cache data.
        data.atomic_json(path, dict(protocol, created_unix_time=time.time()))
    return data.digest(path)


def parse_native_labels(ids, tok):
    """Independent token-state parser; never uses serialize() content spans."""
    special = {name: tok.convert_tokens_to_ids(name) for name in SPECIAL}
    labels = {}
    active = None
    header = []
    in_header = False
    for pos, token in enumerate(ids):
        if token == special['<|start|>']:
            require(not in_header and active is None, 'Nested native message start.')
            in_header, header = True, []
        elif token == special['<|message|>']:
            require(in_header, 'Native message marker without a header.')
            head = tok.decode(header, skip_special_tokens=False)
            if head.startswith('functions.'):
                active = 'tool'
            elif head.startswith('assistant'):
                active = 'cot' if '<|channel|>analysis' in head else 'assistant'
            elif head in ['system', 'user']:
                active = head
            else:
                raise RuntimeError(f'Unexpected native header: {head!r}.')
            in_header = False
        elif token in [special['<|end|>'], special['<|return|>'], special['<|call|>']]:
            require(active is not None and not in_header, 'Native message end without content.')
            active = None
        elif in_header:
            header.append(token)
        elif active is not None:
            labels[pos] = active
    require(active is None and not in_header, 'Unterminated native message.')
    return labels


def render_document(doc, tok, inputs, rows, token_ids):
    """Append five equally weighted formal-role versions of one original body."""
    require(doc['split'] in EXPECTED, 'Only fit/validation documents are allowed.')
    body = doc['body']
    cap = SAMPLE_CAP[doc['split']]
    body_ids = tok.encode(body, add_special_tokens=False)
    require(len(body_ids) == BODY_CAP, 'Decoded capped body must contain exactly 1024 actual tokens.')
    for role in data.ROLES:
        text, spans = data.serialize([(role, body)], bos=bool(doc['document_id'] % 2), wrapper='native')
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        actual = parse_native_labels(encoded['input_ids'], tok)
        prompt_ix = len(inputs)
        entries = []
        for pos, (left, right) in enumerate(encoded['offset_mapping']):
            for span in spans:
                if left < span['end'] and right > span['start']:
                    require(left >= span['start'] and right <= span['end'], 'Ambiguous body boundary token.')
                    require(actual.get(pos) == span['role'] == role, 'Native formal-role label mismatch.')
                    entries.append(dict(prompt_ix=prompt_ix, token_ix=pos, role=role, semantic_role=role,
                        message_ix=0, profile='long-full', split=doc['split'], group_id=doc['group_id'],
                        document_id=doc['document_id'], wrapper='native',
                        token_id=encoded['input_ids'][pos], token=text[left:right]))
        require(len(entries) == BODY_CAP and len(actual) == BODY_CAP,
                'Native body content does not contain exactly 1024 tokens.')
        require(list(actual.values()) == [role] * BODY_CAP, 'Unexpected extra native body content.')
        chosen = data.uniform_indices(len(entries), cap)
        require(len(chosen) == cap and chosen[0] == 0 and chosen[-1] == len(entries) - 1,
                'Uniform sampling must retain the declared count including first and last.')
        for index in chosen:
            entry = entries[int(index)]
            entry['sample_ix'] = SAMPLE_START + len(rows)
            rows.append(entry)
        inputs.append(dict(prompt_ix=prompt_ix, prompt=text, split=doc['split'], profile='long-full',
            group_id=doc['group_id'], document_id=doc['document_id'], wrapper='native',
            role=role, length=len(encoded['input_ids']), body_tokens=BODY_CAP,
            sampled_body_tokens=cap, bos=bool(doc['document_id'] % 2), small_fit=doc['small_fit']))
        token_ids.append(encoded['input_ids'])


def documents(keys, tok):
    wanted = {key['shard_line']: key for key in keys}
    remaining = set(wanted)
    with gzip.open(ARCHIVE, 'rt', encoding='utf-8') as stream:
        for line_ix, line in enumerate(stream):
            if line_ix not in remaining:
                continue
            key = wanted[line_ix]
            article = json.loads(line)
            require(data.text_digest(article['text']) == key['full_body_sha256'], 'Full-body SHA mismatch.')
            require(article.get('url') == key.get('source_url'), 'Source URL mismatch.')
            original_ids = tok.encode(article['text'], add_special_tokens=False)
            body = tok.decode(original_ids[:BODY_CAP], skip_special_tokens=False)
            require(data.text_digest(body) == key['body_sha256'], 'Capped-body SHA mismatch.')
            require(not any(marker in body for marker in RESERVED), 'Reserved marker in content.')
            remaining.remove(line_ix)
            if len(original_ids) >= BODY_CAP:
                yield dict(key, group_id=f'c4-{key["document_id"]}', body=body,
                    original_tokens=len(original_ids), capped_body_tokens=BODY_CAP)
            if not remaining:
                break
    require(not remaining, 'Original fit/validation source lines are missing.')


def validate_cache(inputs, rows, tokens, keys, tok, expected_counts=None):
    expected_counts = EXPECTED if expected_counts is None else expected_counts
    require(set(inputs.split.unique()) == set(EXPECTED), 'Cache split mismatch.')
    require(inputs.prompt_ix.tolist() == list(range(len(inputs))) and len(inputs) == len(tokens),
            'Prompt/token cache alignment mismatch.')
    require(rows.sample_ix.tolist() == list(range(SAMPLE_START, SAMPLE_START + len(rows))),
            'Sample index range mismatch.')
    require(set(rows.profile) == {'long-full'} and set(rows.wrapper) == {'native'}, 'Cache profile mismatch.')
    key_map = {key['document_id']: key for key in keys}
    require(len(key_map) == len(keys) and dict(Counter(key['split'] for key in keys)) == expected_counts,
            'Selected document count mismatch.')
    prompt_counts = inputs.groupby(['document_id', 'role']).size()
    require(len(inputs) == len(key_map) * len(data.ROLES) and len(prompt_counts) == len(inputs)
            and (prompt_counts == 1).all(), 'Duplicate or missing input role versions.')
    require(set(rows.prompt_ix) == set(inputs.prompt_ix), 'Sampled-row prompt coverage mismatch.')
    counts = rows.groupby(['document_id', 'role']).size()
    require(len(counts) == len(key_map) * len(data.ROLES), 'Missing document/role variant.')
    require(all(count == SAMPLE_CAP[key_map[doc]['split']] for (doc, _), count in counts.items()),
            'Every selected document needs the declared sampled token count for all five roles.')
    checked = 0
    for entry in inputs.itertuples():
        key = key_map[entry.document_id]
        require(entry.split == key['split'] and entry.group_id == f'c4-{entry.document_id}', 'Split/group changed.')
        require(entry.bos == bool(entry.document_id % 2), 'BOS parity changed.')
        require(entry.body_tokens == BODY_CAP and entry.sampled_body_tokens == SAMPLE_CAP[entry.split],
                'Input body/sampling cap metadata changed.')
        encoded = tok(entry.prompt, add_special_tokens=False, return_offsets_mapping=True)
        require(encoded['input_ids'] == tokens[entry.prompt_ix], 'Stored token IDs differ from re-tokenization.')
        actual = parse_native_labels(tokens[entry.prompt_ix], tok)
        require(len(actual) == BODY_CAP and set(actual.values()) == {entry.role}, 'Native body-label mismatch.')
        positions = list(actual)
        start = encoded['offset_mapping'][positions[0]][0]
        end = encoded['offset_mapping'][positions[-1]][1]
        require(data.text_digest(entry.prompt[start:end]) == key['body_sha256'], 'Cached prompt body SHA mismatch.')
        group = rows[rows.prompt_ix == entry.prompt_ix]
        require(group.token_ix.tolist() == [positions[int(ix)] for ix in data.uniform_indices(BODY_CAP, SAMPLE_CAP[entry.split])],
                'Sampled positions differ from the declared uniform policy.')
        for row in group.itertuples():
            require(row.split == entry.split and row.document_id == entry.document_id and
                    row.group_id == entry.group_id and row.role == row.semantic_role == actual[row.token_ix],
                    'Cached sampled role/group mismatch.')
            left, right = encoded['offset_mapping'][row.token_ix]
            require(row.token_id == tokens[entry.prompt_ix][row.token_ix] and row.token == entry.prompt[left:right],
                    'Cached sampled token mismatch.')
        checked += len(group)
    return dict(independent_native_labels_checked=checked, body_sha256_checked_prompts=len(inputs),
                all_labels_match=True, sample_positions_include_first_and_last=True)


def prepare():
    protocol, original_keys = source_protocol()
    plan_sha = write_plan(protocol)
    require(data.digest(ARCHIVE) == protocol['source_shard_sha256'], 'Pinned local archive hash mismatch.')
    tok = load_tokenizer()
    done_path = OUT / 'data-complete.json'
    keys_path = OUT / 'document-keys.json'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        require(done['plan_sha256'] == plan_sha and done['document_keys_sha256'] == data.digest(keys_path),
                'Completed data plan/keys changed.')
        require(done['source_sha256'] == protocol['source_sha256'], 'Completed data source changed.')
        require(all(data.digest(CACHE / name) == sha for name, sha in done['cache_sha256'].items()),
                'Completed long-full cache hash mismatch.')
        result = validate_cache(pd.read_parquet(CACHE / 'inputs.parquet'),
            pd.read_parquet(CACHE / 'rows.parquet'), json.loads((CACHE / 'token-ids.json').read_text()),
            json.loads(keys_path.read_text()), tok)
        require(result == done['independent_validation'], 'Completed cache validation changed.')
        print(json.dumps(dict(stage='already_complete_verified', **result)), flush=True)
        return done
    selected = sorted(documents(original_keys, tok), key=lambda item: item['document_id'])
    actual_counts = dict(Counter(doc['split'] for doc in selected))
    require(actual_counts == EXPECTED, f'Eligibility audit differs: {actual_counts} != {EXPECTED}.')
    original_groups = {split: sorted(key['document_id'] for key in original_keys if key['split'] == split)
                       for split in EXPECTED}
    old_counts = {split: sum(doc['document_id'] in original_groups[split][:200 if split == 'fit' else 50]
                            for doc in selected if doc['split'] == split) for split in EXPECTED}
    require(old_counts == EXPECTED_OLD, 'Overlap with the previous long-single augmentation changed.')
    keys = [{name: value for name, value in doc.items() if name != 'body'} for doc in selected]
    immutable_json(keys_path, keys)
    inputs, rows, tokens = [], [], []
    for doc in selected:
        render_document(doc, tok, inputs, rows, tokens)
    input_frame, row_frame = pd.DataFrame(inputs), pd.DataFrame(rows)
    validation = validate_cache(input_frame, row_frame, tokens, keys, tok)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, frame in [('inputs.parquet', input_frame), ('rows.parquet', row_frame)]:
        temporary = CACHE / (name + '.tmp')
        frame.to_parquet(temporary, index=False)
        destination = CACHE / name
        if destination.exists():
            require(data.digest(destination) == data.digest(temporary), 'Partial cache differs; refusing to overwrite.')
            temporary.unlink()
        else:
            temporary.replace(destination)
    immutable_json(CACHE / 'token-ids.json', tokens)
    # Recheck sources after work, so an in-flight source edit cannot produce a
    # completion manifest falsely attributed to the source loaded at the start.
    require(source_protocol()[0] == protocol, 'Source/protocol changed during preparation.')
    done = dict(created_unix_time=time.time(), plan_sha256=plan_sha,
        document_keys_sha256=data.digest(keys_path), source_sha256=protocol['source_sha256'],
        tokenizer_sha256=protocol['tokenizer_sha256'], source_shard_sha256=protocol['source_shard_sha256'],
        selected_document_counts=actual_counts, original_split_preserved=True,
        added_independent_documents=0, body_tokens=BODY_CAP,
        previous_long_single_eligible_documents=old_counts,
        sampled_tokens_per_role=SAMPLE_CAP, prompts=len(inputs), sampled_rows=len(rows),
        input_tokens=int(input_frame.length.sum()), independent_validation=validation,
        cache_sha256={name: data.digest(CACHE / name) for name in FILES},
        replacement_policy=protocol['replacement_policy'], test_inputs_read=False)
    data.atomic_json(done_path, done)
    print(json.dumps(done), flush=True)
    return done


def preflight():
    """Validate one eligible original fit and validation source in memory only."""
    protocol, keys = source_protocol()
    require(data.digest(ARCHIVE) == protocol['source_shard_sha256'], 'Pinned local archive hash mismatch.')
    tok = load_tokenizer()
    selected = {}
    iterator = documents(keys, tok)
    try:
        for doc in iterator:
            selected.setdefault(doc['split'], doc)
            if set(selected) == set(EXPECTED):
                break
    finally:
        iterator.close()
    require(set(selected) == set(EXPECTED), 'Small preflight could not find both source splits.')
    inputs, rows, tokens = [], [], []
    for doc in selected.values():
        render_document(doc, tok, inputs, rows, tokens)
    expected_rows = sum(SAMPLE_CAP.values()) * len(data.ROLES)
    require(len(inputs) == 10 and len(rows) == expected_rows, 'Small preflight counts differ.')
    require(len(set(row['sample_ix'] for row in rows)) == expected_rows, 'Small preflight sample IDs overlap.')
    validation = validate_cache(pd.DataFrame(inputs), pd.DataFrame(rows), tokens,
        [{key: value for key, value in doc.items() if key != 'body'} for doc in selected.values()], tok,
        expected_counts={'fit': 1, 'validation': 1})
    print(json.dumps(dict(stage='cpu_preflight_passed', cache_written=False, prompts=len(inputs),
        sampled_rows=len(rows), body_tokens=BODY_CAP, roles=data.ROLES,
        documents={split: doc['document_id'] for split, doc in selected.items()},
        independent_validation=validation,
        source_sha256=protocol['source_sha256'], tokenizer_sha256=protocol['tokenizer_sha256'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true', help='Write immutable plan and prepare the complete cache.')
    action.add_argument('--preflight', action='store_true', help='Check two source documents on CPU without writing data.')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
