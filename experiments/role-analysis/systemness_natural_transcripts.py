"""Prepare public S/U/A dev transcripts without generating model responses.

Default: help. --preflight uses synthetic CPU fixtures and /tmp only.
--stage prepare-dev first freezes the protocol, then reads assistant text for
fixed fit256/validation64 roots. --stage verify-dev is a read-only CPU replay.
Test128 assistant fields are never decoded/prepared. No GPU or download is used.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as long_data
import systemness_natural_sources as sources


OUT = data.runtime.OUT / 'systemness-natural-transcripts'
CACHE = data.CACHE / 'natural-transcripts-dev'
BODY_CAP = 192
SAMPLE_START = 4_000_000_000_000
ROLES = ['system', 'user', 'assistant']
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
DEPENDENCIES = ['systemness_natural_transcripts.py', 'systemness_natural_sources.py',
    'systemness_long_data.py', 'systemness_data.py', 'run_local_gptoss120b.py']
MIN_FREE_BYTES = 128 * 2**20
MAX_CACHE_BYTES = 128 * 2**20
require = long_data.require
read = sources.read
sha = sources.sha
canonical_sha = sources.canonical_sha


def source_context():
    """Hash/metadata only. Do not call sources.verify(): it reads source text."""
    signature = sources.signature()
    folder = sources.candidate_folder(signature)
    proposal = read(folder / 'source-proposal.json')
    require(proposal['signature'] == signature and proposal['signature_sha256'] == canonical_sha(signature),
            'Frozen natural source proposal signature changed.')
    for name, digest in proposal['artifacts_sha256'].items():
        require(data.digest(folder / name) == digest, f'Natural source candidate changed: {name}.')
    keys = read(folder / 'natural-source-keys.json')['source_keys']
    require(canonical_sha(keys) == sources.LEGACY_KEYS_SHA256 and len(keys) == 448, 'Fixed448 source IDs changed.')
    all_templates = read(folder / 'system-templates.json')['templates']
    assignments = read(folder / 'template-assignments.json')['assignments']
    dev = [row for row in keys if row['split'] in ['fit', 'validation']]
    require(Counter(row['split'] for row in dev) == {'fit': 256, 'validation': 64}, 'Fixed dev split changed.')
    test = [row for row in keys if row['split'] == 'test']
    require(len(test) == 128, 'Fixed test source metadata changed.')
    templates = {row['template_id']: row for row in all_templates if row['split'] in ['fit', 'validation']}
    assigned = {(row['dataset'], row['native_id']): row for row in assignments
                if row['split'] in ['fit', 'validation']}
    require(len(assigned) == len(dev) and all((row['dataset'], row['native_id']) in assigned for row in dev),
            'Dev template mapping is incomplete.')
    return dict(folder=folder, dev=dev, test=test, templates=templates, assignments=assigned,
        signature=signature, binding=dict(source_proposal_sha256=data.digest(folder / 'source-proposal.json'),
            source_keys_file_sha256=data.digest(folder / 'natural-source-keys.json'),
            source_keys_canonical_sha256=sources.LEGACY_KEYS_SHA256,
            templates_file_sha256=data.digest(folder / 'system-templates.json'),
            assignments_file_sha256=data.digest(folder / 'template-assignments.json'),
            template_audit_sha256=data.digest(folder / 'template-audit.json'),
            dev_keys_sha256=canonical_sha(dev), dev_assignments_sha256=canonical_sha(list(assigned.values())),
            source_builder_sha256=data.digest(sources.__file__), source_signature=signature))


def protocol_value(context):
    return dict(version=1, stage='public-natural-transcripts-dev-only', source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        source_binding=context['binding'], cache_path=str(CACHE), dev_roots=dict(fit=256, validation=64),
        test_roots_excluded=128, sample_ix_start=SAMPLE_START, roles=ROLES, body_cap=BODY_CAP,
        freeze_order='Persist this immutable protocol before materializing any validation assistant text.',
        oasst_read='Read tree/ID/parent/role/provenance metadata without text first. Arrow predicate-filter only dev tree IDs before text is materialized in Python. Physical Parquet row groups may contain other rows; logical decoded text is dev-only.',
        toxicchat_read='Stream quoted CSV bytes for record framing; only decode conv_id until membership is known. Decode user_input/model_output only for dev IDs. No test assistant field is buffered, decoded, persisted or content-counted. Framing transport buffers and file hashes can physically read mixed-source bytes.',
        path='Clean means en, deleted=False, synthetic=False, nonempty. Choose first clean assistant child of root by SHA256(parentID+NUL+childID). Then among clean user children with a clean assistant child choose the first by the same SHA, then its first clean assistant by SHA. Never switch the first assistant to obtain a longer path. At most U-A-U-A; missing second complete pair leaves U-A.',
        toxicchat='Exactly one authentic user_input and recorded model_output. Missing/empty response is a quality failure, never a replacement ID.',
        serialization='BOS fixed true; explicit native system/user/assistant-final headers. Every assistant response uses the native return marker as a serialization boundary. These markers do not claim source generation completion. No invented analysis/tool segment.',
        source_provenance='OASST declared non-synthetic public human text; ToxicChat recorded Vicuna-demo output with exact checkpoint unknown. Replies were not generated by120B and were not generated under the newly assigned system wording.',
        body_processing='Preserve raw SHA, normalize only for duplicate keys, replace reserved Harmony markers in message bodies, cap to a round-trip-stable tokenizer prefix of at most192 tokens, and record raw/clean/capped/serialized body SHA, lengths and truncation. Do not invent a completed source reply when capping.',
        coverage='All content tokens for actual S/U/A messages, no sparse token sampling. Independently parse native token roles and compare every content position against construction spans. CoT/tool support0 and recall N/A.',
        quality='All320 fixed IDs remain in the ledger. Failed IDs have no invented transcript and no replacement. Successful flagged transcripts remain in cache with explicit eligibility fields.',
        contamination='Audit assistant bodies from successfully prepared transcripts: raw/clean/capped full text, normalized text and first192-token variants, within fit, within validation, across splits, and against old generated clean CSV CoT/final plus raw assistant/native assistant segments. No model scores. A fit conversation is eligible_fit=False on any direct assistant collision with old generated content or any prepared validation assistant. Validation colliding with old generated content is eligible_clean_validation=False; retain all-validation separately. Within-split repeats are marked, never silently dropped. Quality-failed IDs have no transcript observations and stay in the failure ledger.',
        dependence='Publish duplicate message clusters and conversation dependency components combining assistant repeats and shared system family. Positive system stimuli are32 fit wordings and16 validation wordings, not256/64 independent roots. Future paired bootstrap must retain those dependencies; no test bootstrap/data is prepared here.',
        excluded_inference_inputs='Source IDs, quality flags, labels, cluster IDs and message positions are metadata for preparation/evaluation, not role-readout features.',
        storage=dict(minimum_free_bytes=MIN_FREE_BYTES, maximum_cache_bytes=MAX_CACHE_BYTES,
                     reuse_existing_snapshots=True, no_snapshot_copy=True),
        gpu_work_run=False, model_generation_run=False, downloads_run=False,
        test_assistant_decoded=False, test_transcripts_prepared=False, primary_changed=False)


def immutable_json(path, value):
    if path.exists(): require(read(path) == value, f'Immutable artifact differs: {path}.')
    else: data.atomic_json(path, value)


def freeze_protocol(context):
    value = protocol_value(context)
    OUT.mkdir(parents=True, exist_ok=True)
    immutable_json(OUT / 'protocol.json', value)
    with (OUT / 'protocol.json').open('rb') as stream: os.fsync(stream.fileno())
    require(read(OUT / 'protocol.json') == value, 'Protocol was not persisted before source text access.')
    return value, data.digest(OUT / 'protocol.json')


def verify_protocol(context, protocol):
    require(protocol == protocol_value(context) and read(OUT / 'protocol.json') == protocol,
            'Dev source/protocol/dependencies changed.')


def guarded_source_read(protocol_sha256):
    require((OUT / 'protocol.json').is_file() and data.digest(OUT / 'protocol.json') == protocol_sha256,
            'A frozen matching protocol is required before assistant text access.')


def uid(row):
    return row['dataset'] + ':' + row['native_id']


def branch_key(parent_id, child_id):
    return sha(str(parent_id) + '\0' + str(child_id))


def clean_oa(row):
    return (row['lang'] == 'en' and row['deleted'] is not None and not bool(row['deleted'])
        and row['synthetic'] is not None and not bool(row['synthetic'])
        and isinstance(row['text'], str) and bool(row['text'].strip()))


def choose_oa_path(records, root_message_id):
    by_id = {str(row['message_id']): row for row in records}
    require(len(by_id) == len(records), 'Duplicate native OASST message IDs.')
    root = by_id.get(str(root_message_id))
    if root is None or root['role'] != 'prompter' or root['parent_id'] is not None or not clean_oa(root):
        return [], 'missing-or-unclean-root'
    children = defaultdict(list)
    for row in records:
        if row['parent_id'] is not None: children[str(row['parent_id'])].append(row)
    def eligible(parent, role):
        return sorted((row for row in children[str(parent['message_id'])] if row['role'] == role and clean_oa(row)),
                      key=lambda row: branch_key(parent['message_id'], row['message_id']))
    assistants = eligible(root, 'assistant')
    if not assistants: return [], 'no-clean-first-assistant'
    first = assistants[0]; path = [root, first]
    users = [row for row in eligible(first, 'prompter') if eligible(row, 'assistant')]
    if users:
        second_user = users[0]; path += [second_user, eligible(second_user, 'assistant')[0]]
    require(len({row['message_id'] for row in path}) == len(path), 'Cyclic native OASST path.')
    return path, None


def selected_csv(path, wanted):
    """RFC-style quoted CSV scanner; nonselected response fields are skipped.

    Only64KiB transport chunks are read. Nonselected body bytes are neither
    accumulated as fields nor UTF8-decoded. This is logical field isolation,
    not a claim that the physical CSV never transports test-record bytes.
    """
    wanted = set(map(str, wanted)); header = None; selected = False; current_id = None
    payload = {}; selected_records = {}; seen_ids = set(); rows_seen = 0
    column = 0; in_quotes = False; pending_quote = False; closed_quote = False
    at_start = True; row_started = False; skip_lf = False; field = bytearray()
    needed = {}; decoded_response_ids = []; captured_body_fields = 0
    def capture():
        return header is None or column == 0 or (selected and column in needed)
    def finish_field():
        nonlocal header, selected, current_id, payload, column, field, captured_body_fields
        if header is None:
            payload[column] = bytes(field).decode('utf-8')
        elif column == 0:
            current_id = bytes(field).decode('utf-8'); selected = current_id in wanted
            if selected: payload = {'conv_id': current_id}
        elif selected and column in needed:
            payload[needed[column]] = bytes(field).decode('utf-8')
            if needed[column] == 'model_output': captured_body_fields += 1
        column += 1; field = bytearray()
    def finish_record():
        nonlocal header, selected, current_id, payload, column, rows_seen, needed
        if header is None:
            header = [payload[index] for index in range(column)]
            require(len(set(header)) == len(header) and header[0] == 'conv_id'
                and {'user_input', 'model_output'} <= set(header), 'Unexpected pinned ToxicChat CSV schema.')
            needed = {header.index(name): name for name in ['user_input', 'model_output']}
        else:
            require(column == len(header), 'Malformed ToxicChat CSV field count.')
            require(current_id not in seen_ids, 'Duplicate ToxicChat native conv_id.')
            seen_ids.add(current_id); rows_seen += 1
            if selected:
                require({'conv_id', 'user_input', 'model_output'} == set(payload), 'Incomplete selected CSV record.')
                selected_records[current_id] = payload
                decoded_response_ids.append(current_id)
        selected = False; current_id = None; payload = {}; column = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b''):
            for byte in chunk:
                if skip_lf:
                    skip_lf = False
                    if byte == 10: continue
                if in_quotes:
                    if pending_quote:
                        if byte == 34:
                            if capture(): field.append(34)
                            pending_quote = False; continue
                        in_quotes = False; pending_quote = False; closed_quote = True
                    elif byte == 34:
                        pending_quote = True; continue
                    else:
                        if capture(): field.append(byte)
                        continue
                if at_start and byte == 34:
                    in_quotes = True; at_start = False; row_started = True; continue
                if byte == 44:
                    finish_field(); at_start = True; closed_quote = False; row_started = True
                elif byte in [10, 13]:
                    require(row_started or column or field, 'Unexpected blank CSV record.')
                    finish_field(); finish_record(); at_start = True; closed_quote = False; row_started = False
                    skip_lf = byte == 13
                else:
                    require(not closed_quote and byte != 34, 'Malformed quoted CSV field.')
                    if capture(): field.append(byte)
                    at_start = False; row_started = True
    require(not in_quotes or pending_quote, 'Unterminated quoted CSV field.')
    if row_started or column or field:
        finish_field(); finish_record()
    require(header is not None and set(selected_records) <= wanted, 'CSV selection escaped dev IDs.')
    return selected_records, dict(source_records_framed=rows_seen, selected_records=len(selected_records),
        decoded_assistant_ids_sha256=canonical_sha(sorted(decoded_response_ids)),
        assistant_fields_decoded=captured_body_fields, nonselected_assistant_fields_decoded=0,
        nonselected_assistant_fields_buffered=0, quoted_newlines_and_quotes_supported=True,
        physical_byte_framing_is_not_logical_text_materialization=True)


def read_development(context, protocol_sha256):
    import pyarrow.dataset as ds
    guarded_source_read(protocol_sha256)
    dev_oa = {row['native_id'] for row in context['dev'] if row['dataset'] == 'oasst'}
    test_oa = {row['native_id'] for row in context['test'] if row['dataset'] == 'oasst'}
    dev_tc = {row['native_id'] for row in context['dev'] if row['dataset'] == 'toxicchat0124'}
    test_tc = {row['native_id'] for row in context['test'] if row['dataset'] == 'toxicchat0124'}
    require(not dev_oa & test_oa and not dev_tc & test_tc, 'Dev/test native source IDs overlap.')
    path = sources.snapshot_path(sources.PINS[0])
    metadata_columns = ['message_tree_id', 'message_id', 'parent_id', 'role', 'lang', 'deleted', 'synthetic', 'model_name']
    metadata = pd.read_parquet(path, columns=metadata_columns)
    require(set(dev_oa) <= set(metadata.message_tree_id), 'A fixed OASST dev tree is absent.')
    # Predicate precedes conversion to Python strings. No full-text pandas read.
    table = ds.dataset(path, format='parquet').to_table(filter=ds.field('message_tree_id').isin(sorted(dev_oa)),
        columns=metadata_columns + ['text'])
    selected = table.to_pandas()
    require(set(selected.message_tree_id) == dev_oa and not set(selected.message_tree_id) & test_oa,
            'OASST logical text materialization includes a non-dev tree.')
    grouped = {str(native): frame.to_dict('records') for native, frame in selected.groupby('message_tree_id', sort=False)}
    toxic, csv_audit = selected_csv(sources.snapshot_path(sources.PINS[1]), dev_tc)
    require(set(toxic) == dev_tc and not set(toxic) & test_tc, 'ToxicChat dev record membership changed.')
    records = []
    for key in context['dev']:
        assignment = context['assignments'][(key['dataset'], key['native_id'])]
        template = context['templates'][assignment['template_id']]
        base = dict(source_uid=uid(key), dataset=key['dataset'], native_id=key['native_id'], split=key['split'],
            revision=key['revision'], root_raw_sha256=key['root_raw_sha256'],
            template_id=assignment['template_id'], family_id=assignment['family_id'],
            source_snapshot_sha256=sources.PINS[0 if key['dataset'] == 'oasst' else 1]['sha256'])
        messages = [dict(role='system', text=template['text'], source_message_id=template['template_id'],
            origin='new fixed system wording; not source generation context', source_model=None)]
        if key['dataset'] == 'oasst':
            chosen, failure = choose_oa_path(grouped[key['native_id']], key['root_message_id'])
            for item in chosen:
                messages.append(dict(role='user' if item['role'] == 'prompter' else 'assistant', text=item['text'],
                    source_message_id=str(item['message_id']), origin='public OASST non-synthetic text',
                    source_model=item['model_name'], source_synthetic=bool(item['synthetic']), source_deleted=bool(item['deleted'])))
        else:
            item = toxic[key['native_id']]
            failure = None if item['user_input'].strip() and item['model_output'].strip() else 'empty-user-or-recorded-assistant'
            for role, field in [('user', 'user_input'), ('assistant', 'model_output')]:
                messages.append(dict(role=role, text=item[field], source_message_id=key['native_id'] + '/' + field,
                    origin='public ToxicChat user' if role == 'user' else 'recorded Vicuna-demo output; exact checkpoint unknown',
                    source_model=None if role == 'user' else 'Vicuna-demo; exact checkpoint unknown'))
        if not failure:
            require(sha(messages[1]['text']) == key['root_raw_sha256'], 'Selected root text no longer matches fixed source proposal.')
        records.append(dict(**base, messages=messages, failure=failure))
    audit = dict(protocol_sha256_before_text_materialization=protocol_sha256,
        oasst_metadata_rows_read_without_text=len(metadata), oasst_logical_text_rows_materialized=len(selected),
        oasst_logical_text_tree_ids_sha256=canonical_sha(sorted(dev_oa)),
        oasst_non_dev_text_rows_materialized=0, oasst_test_assistant_rows_materialized=0,
        parquet_physical_blocks_may_contain_non_dev_bytes=True, toxicchat=csv_audit,
        test_assistant_decoded=False, test_transcripts_prepared=False)
    return records, audit


def capped_body(raw, tokenizer):
    clean = raw
    for marker in long_data.RESERVED: clean = clean.replace(marker, '[' + marker[2:-2] + ']')
    full_ids = tokenizer.encode(clean, add_special_tokens=False)
    ids = full_ids[:BODY_CAP]; removed = 0
    while ids:
        body = tokenizer.decode(ids, skip_special_tokens=False)
        if tokenizer.encode(body, add_special_tokens=False) == ids: break
        ids = ids[:-1]; removed += 1
    body = tokenizer.decode(ids, skip_special_tokens=False)
    require(len(ids) <= BODY_CAP and not any(marker in body for marker in long_data.RESERVED), 'Body cap/marker isolation failed.')
    return body, dict(raw_sha256=sha(raw), normalized_raw_sha256=sha(sources.normalized(raw)),
        clean_sha256=sha(clean), capped_sha256=sha(body), serialized_body_sha256=sha(body),
        raw_characters=len(raw), clean_characters=len(clean), capped_characters=len(body),
        raw_tokens=len(tokenizer.encode(raw, add_special_tokens=False)), clean_tokens=len(full_ids),
        body_tokens=len(ids), truncated=len(ids) < len(full_ids), marker_sanitized=clean != raw,
        cap_roundtrip_tokens_removed=removed), clean


def assistant_keys(raw, clean, body, tokenizer):
    values = [raw, clean, body]
    normalized = [sources.normalized(value) for value in values]
    result = {('full', sha(value)) for value in values}
    result |= {('normalized', sha(value)) for value in normalized}
    for value in values + normalized:
        ids = np.asarray(tokenizer.encode(value, add_special_tokens=False)[:BODY_CAP], dtype='<i8')
        result.add(('prefix192', sha(ids.tobytes())))
    return result


def legacy_generated_keys(tokenizer):
    index = defaultdict(set); messages = 0; memo = {}
    def add(text, reference):
        nonlocal messages
        if not isinstance(text, str) or not text.strip(): return
        if text not in memo:
            body, _, clean = capped_body(text, tokenizer)
            memo[text] = assistant_keys(text, clean, body, tokenizer)
        for key in memo[text]: index[key].add(reference)
        messages += 1
    clean = pd.read_csv(data.HERE / sources.OLD_CSVS['clean'][0], keep_default_na=False,
        usecols=['dataset', 'conv_id', 'user_query_ix', 'assistant', 'cot'])
    for row in clean.itertuples():
        for role in ['assistant', 'cot']:
            add(getattr(row, role), f'old-clean:{row.dataset}:{row.conv_id}:{row.user_query_ix}:{role}')
    raw = pd.read_csv(data.HERE / sources.OLD_CSVS['raw'][0], keep_default_na=False,
        usecols=['dataset', 'conv_id', 'user_query_ix', 'assistant', 'state_text'])
    pattern = re.compile(r'<\|start\|>assistant<\|channel\|>(analysis|final)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|call\|>|$)', re.S)
    for row in raw.itertuples():
        reference = f'old-raw:{row.dataset}:{row.conv_id}:{row.user_query_ix}'
        add(row.assistant, reference + ':assistant')
        for segment, (channel, text) in enumerate(pattern.findall(row.state_text)):
            add(text, reference + f':segment{segment}:{channel}')
    return index, dict(old_body_occurrences=messages, old_unique_raw_bodies=len(memo), old_key_count=len(index),
        old_generated_csv_sha256={name: sources.OLD_CSVS[name][1] for name in ['clean', 'raw']},
        includes_failed_raw_native_segments=True, model_probabilities_read=False)


class Components:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]; node = self.parent[node]
        return node

    def join(self, nodes):
        nodes = sorted(set(nodes))
        if not nodes: return
        first = self.find(nodes[0])
        for node in nodes[1:]: self.parent[self.find(node)] = first

    def groups(self):
        result = defaultdict(list)
        for node in self.parent: result[self.find(node)].append(node)
        return sorted(sorted(group) for group in result.values())


def duplicate_audit(ledger, assistant, old_index, old_proof):
    by_message = {row['message_uid']: row for row in assistant}
    index = defaultdict(set)
    for row in assistant:
        for key in row['keys']: index[key].add(row['message_uid'])
    components = Components(by_message)
    shared = [(key, sorted(messages)) for key, messages in index.items() if len(messages) > 1]
    for key, messages in shared: components.join(messages)
    shared_by_group = defaultdict(list)
    for key, messages in shared: shared_by_group[components.find(messages[0])].append(key)
    clusters = []; message_cluster = {}
    for messages in components.groups():
        if len(messages) < 2: continue
        cluster_id = 'assistant-repeat-' + canonical_sha(messages)[:16]
        refs = [{key: by_message[message][key] for key in ['message_uid', 'source_uid', 'split', 'message_ix']}
                for message in messages]
        matches = shared_by_group[components.find(messages[0])]
        clusters.append(dict(cluster_id=cluster_id, messages=refs, conversations=len({row['source_uid'] for row in refs}),
            split_message_counts=dict(Counter(row['split'] for row in refs)),
            key_types=sorted({kind for kind, digest in matches}), shared_keys_sha256=canonical_sha(sorted(matches))))
        for message in messages: message_cluster[message] = cluster_id
    validation_keys = set().union(*(row['keys'] for row in assistant if row['split'] == 'validation')) if assistant else set()
    ledger_by_id = {row['source_uid']: row for row in ledger}; old_keys = set(old_index)
    for row in ledger:
        row['old_generated_collision'] = False; row['cross_split_assistant_collision'] = False
        row['assistant_duplicate_cluster_ids'] = []; row['old_generated_references'] = []
        row['dependency_component_id'] = None
    for message in assistant:
        row = ledger_by_id[message['source_uid']]
        old_matches = message['keys'] & old_keys
        references = sorted(set().union(*(old_index[key] for key in old_matches))) if old_matches else []
        row['old_generated_collision'] |= bool(old_matches)
        row['old_generated_references'].extend(references)
        cross = any(by_message[other]['split'] != message['split'] for key in message['keys'] for other in index[key])
        row['cross_split_assistant_collision'] |= cross
        if message['message_uid'] in message_cluster:
            row['assistant_duplicate_cluster_ids'].append(message_cluster[message['message_uid']])
        message['old_collision_key_types'] = sorted({kind for kind, digest in old_matches})
        message['collides_with_validation_assistant'] = bool(message['keys'] & validation_keys) if message['split'] == 'fit' else False
    for row in ledger:
        row['old_generated_references'] = sorted(set(row['old_generated_references']))
        row['assistant_duplicate_cluster_ids'] = sorted(set(row['assistant_duplicate_cluster_ids']))
        eligible = row['status'] == 'prepared'
        row['eligible_fit'] = eligible and row['split'] == 'fit' and not row['old_generated_collision'] and not row['cross_split_assistant_collision']
        row['eligible_clean_validation'] = eligible and row['split'] == 'validation' and not row['old_generated_collision']
        row['eligibility_reasons'] = ([] if eligible else ['quality-failed'])
        if row['old_generated_collision']: row['eligibility_reasons'].append('assistant-collision-with-old-generated-content')
        if row['split'] == 'fit' and row['cross_split_assistant_collision']:
            row['eligibility_reasons'].append('assistant-collision-with-validation')
    dependence = {}
    for split in ['fit', 'validation']:
        chosen = [row for row in ledger if row['split'] == split and row['status'] == 'prepared']
        graph = Components([row['source_uid'] for row in chosen])
        families = defaultdict(list)
        for row in chosen: families[row['family_id']].append(row['source_uid'])
        for nodes in families.values(): graph.join(nodes)
        for cluster in clusters: graph.join([row['source_uid'] for row in cluster['messages'] if row['split'] == split])
        groups = graph.groups()
        for nodes in groups:
            component_id = split + '-dependency-' + canonical_sha(nodes)[:16]
            for node in nodes: ledger_by_id[node]['dependency_component_id'] = component_id
        dependence[split] = dict(prepared_conversations=len(chosen), distinct_system_families=len(families),
            dependency_components=len(groups), components=[dict(component_id=split + '-dependency-' + canonical_sha(nodes)[:16],
                source_uids=nodes) for nodes in groups])
    message_flags = [{key: row[key] for key in ['message_uid', 'source_uid', 'split', 'message_ix',
        'old_collision_key_types', 'collides_with_validation_assistant']} for row in assistant]
    return dict(old_generated_reference=old_proof, scope='Assistant messages in successfully prepared dev transcripts; failed IDs have no invented observations.',
        duplicate_key_types=['full', 'normalized', 'prefix192'],
        full_variants='raw, marker-sanitized and actual capped body; normalized variants use NFKC/casefold/whitespace.',
        assistant_messages=len(assistant), duplicate_clusters=clusters,
        within_fit_duplicate_clusters=sum(cluster['split_message_counts'].get('fit', 0) > 1 for cluster in clusters),
        within_validation_duplicate_clusters=sum(cluster['split_message_counts'].get('validation', 0) > 1 for cluster in clusters),
        cross_split_duplicate_clusters=sum(len(cluster['split_message_counts']) > 1 for cluster in clusters),
        conversation_dependency_components=dependence, message_collision_flags=message_flags,
        eligibility_counts={split: dict(all_fixed_ids=sum(row['split'] == split for row in ledger),
            prepared=sum(row['split'] == split and row['status'] == 'prepared' for row in ledger),
            eligible_fit=sum(row['split'] == split and row['eligible_fit'] for row in ledger),
            eligible_clean_validation=sum(row['split'] == split and row['eligible_clean_validation'] for row in ledger),
            old_collision=sum(row['split'] == split and row['old_generated_collision'] for row in ledger))
            for split in ['fit', 'validation']},
        bootstrap='Use paired conversation dependency components combining same system family and assistant repeats. Positive stimuli are at most32 fit /16 validation wordings. Neither tokens nor256/64 roots are independent system stimuli.',
        tests_read=False, no_ids_replaced=True)


def render_native(messages):
    text = '<|startoftext|>'; spans = []
    for index, (role, body) in enumerate(messages):
        require(role in ROLES, 'No invented CoT/tool role is permitted.')
        rendered = data.native_message(role, body, final=role == 'assistant')
        start = len(text) + rendered.index('<|message|>') + len('<|message|>')
        spans.append(dict(start=start, end=start + len(body), role=role, message_ix=index))
        text += rendered
    return text, spans


def content_rows(prompt, spans, tokenizer):
    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding['input_ids']; actual = long_data.parse_native_labels(ids, tokenizer)
    expected = {}; result = []
    for position, (left, right) in enumerate(encoding['offset_mapping']):
        for span in spans:
            if left < span['end'] and right > span['start']:
                require(left >= span['start'] and right <= span['end'], 'A native token crosses a content/header boundary.')
                expected[position] = span['role']
                result.append(dict(token_ix=position, token_id=ids[position], token=prompt[left:right],
                    message_ix=span['message_ix'], role=span['role'], semantic_role=span['role']))
                break
    require(expected == actual and set(actual.values()) <= set(ROLES), 'Independent native role/content coverage mismatch.')
    require(len(result) == len(actual) and len(set(row['token_ix'] for row in result)) == len(result),
            'Content positions were lost or duplicated.')
    return ids, result


def build_dataset(records, tokenizer, old_index, old_proof):
    ledger = []; inputs = []; rows = []; tokens = []; assistant = []
    for record in records:
        entry = {key: value for key, value in record.items() if key not in ['messages', 'failure']}
        entry.update(status='quality-failed' if record['failure'] else 'prepared', quality_failure=record['failure'],
            prompt_ix=None, messages=[], turns=0)
        processed = []
        if not record['failure']:
            for index, message in enumerate(record['messages']):
                body, hashes, clean = capped_body(message['text'], tokenizer)
                metadata = {key: value for key, value in message.items() if key != 'text'}
                entry['messages'].append(dict(message_ix=index, **metadata, **hashes))
                if not body.strip():
                    entry['status'] = 'quality-failed'; entry['quality_failure'] = 'empty-body-after-cap'; break
                processed.append((message['role'], body, clean, message['text']))
        if entry['status'] == 'prepared':
            require([role for role, body, clean, raw in processed] in [
                ['system', 'user', 'assistant'], ['system', 'user', 'assistant', 'user', 'assistant']],
                'Prepared source is not an authentic complete S/U/A path.')
            prompt, spans = render_native([(role, body) for role, body, clean, raw in processed])
            ids, content = content_rows(prompt, spans, tokenizer)
            prompt_ix = len(inputs); entry['prompt_ix'] = prompt_ix
            entry['turns'] = sum(role == 'user' for role, body, clean, raw in processed)
            for index, (role, body, clean, raw) in enumerate(processed):
                selected = [row for row in content if row['message_ix'] == index]
                require(len(selected) == entry['messages'][index]['body_tokens'], 'Serialized body token count changed.')
                if role == 'assistant':
                    assistant.append(dict(message_uid=entry['source_uid'] + '/message' + str(index),
                        source_uid=entry['source_uid'], split=entry['split'], message_ix=index,
                        keys=assistant_keys(raw, clean, body, tokenizer)))
            common = {key: entry[key] for key in ['source_uid', 'dataset', 'native_id', 'split', 'template_id', 'family_id']}
            inputs.append(dict(prompt_ix=prompt_ix, prompt=prompt, length=len(ids), turns=entry['turns'],
                message_count=len(processed), wrapper='native', profile='natural-public-sua', group_id=entry['source_uid'], **common))
            tokens.append(ids)
            for row in content:
                rows.append(dict(cache_ix=len(rows), sample_ix=SAMPLE_START + len(rows), prompt_ix=prompt_ix,
                    wrapper='native', profile='natural-public-sua', group_id=entry['source_uid'], **common, **row))
        ledger.append(entry)
    require(len({row['source_uid'] for row in ledger}) == len(records), 'Fixed source ID duplicated in dev ledger.')
    audit = duplicate_audit(ledger, assistant, old_index, old_proof)
    mapping = {row['source_uid']: row for row in ledger}
    for entry in inputs + rows:
        source = mapping[entry['source_uid']]
        for field in ['eligible_fit', 'eligible_clean_validation', 'old_generated_collision',
                      'cross_split_assistant_collision', 'dependency_component_id']:
            entry[field] = source[field]
    inputs = pd.DataFrame(inputs); rows = pd.DataFrame(rows)
    require(not set(inputs[inputs.split == 'fit'].group_id) & set(inputs[inputs.split == 'validation'].group_id),
            'Native source groups overlap between dev splits.')
    coverage = dict(all_content_tokens_scored=True, independent_native_labels_match=True,
        input_prompts=len(inputs), content_rows=len(rows), role_counts={key: int(value) for key, value in rows.groupby('role').size().items()},
        roles_with_no_observations=dict(cot=dict(support=0, recall=None, status='N/A'),
                                      tool=dict(support=0, recall=None, status='N/A')),
        sample_ix_start=SAMPLE_START, sample_ix_end_exclusive=SAMPLE_START + len(rows),
        no_sparse_sampling=True, no_c4_document_or_shard_line_ids=True)
    return inputs, rows, tokens, ledger, audit, coverage


def verify_cached_content(inputs, rows, tokens, ledger, tokenizer):
    require(inputs.prompt_ix.tolist() == list(range(len(inputs))) and len(tokens) == len(inputs)
        and np.array_equal(rows.cache_ix, np.arange(len(rows)))
        and np.array_equal(rows.sample_ix, SAMPLE_START + np.arange(len(rows))), 'Dev cache row identity changed.')
    lookup = {int(prompt): group for prompt, group in rows.groupby('prompt_ix', sort=False)}
    by_prompt = {row['prompt_ix']: row for row in ledger if row['status'] == 'prepared'}
    pattern = re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>)', re.S)
    for entry in inputs.itertuples():
        ids = tokenizer.encode(entry.prompt, add_special_tokens=False)
        require(ids == tokens[entry.prompt_ix] and len(ids) == entry.length, 'Cached input tokenization changed.')
        actual = long_data.parse_native_labels(ids, tokenizer)
        group = lookup[entry.prompt_ix]
        require(dict(zip(group.token_ix, group.role)) == actual
            and all(ids[row.token_ix] == row.token_id for row in group.itertuples()), 'Cached native content coverage differs.')
        bodies = pattern.findall(entry.prompt)
        record = by_prompt[entry.prompt_ix]
        require(len(bodies) == len(record['messages']), 'Serialized message count differs from provenance ledger.')
        for index, (header, body, ending) in enumerate(bodies):
            role = record['messages'][index]['role']
            require(header == ('assistant<|channel|>final' if role == 'assistant' else role)
                and ending == ('<|return|>' if role == 'assistant' else '<|end|>')
                and sha(body) == record['messages'][index]['serialized_body_sha256']
                and len(tokenizer.encode(body, add_special_tokens=False)) <= BODY_CAP,
                'Cached final channel/body hash/cap differs from ledger.')
    require(set(rows.role) <= set(ROLES) and set(inputs.split) <= {'fit', 'validation'}, 'Non-dev or invented role in cache.')


def save_parquet(path, frame):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '-', suffix='.tmp', dir=path.parent)
    os.close(descriptor); temporary = Path(temporary)
    try:
        frame.to_parquet(temporary, index=False)
        if path.exists(): require(data.digest(path) == data.digest(temporary), f'Existing dev cache differs: {path}')
        else: temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def prepare_dev():
    context = source_context(); protocol, protocol_sha256 = freeze_protocol(context)
    require(shutil.disk_usage(data.CACHE).free >= MIN_FREE_BYTES, 'Less than128MiB free; do not copy source snapshots or delete caches.')
    if (OUT / 'data-complete.json').exists():
        print(json.dumps(verify_dev()), flush=True); return
    records, read_audit = read_development(context, protocol_sha256)
    tokenizer = long_data.load_tokenizer(); old_index, old_proof = legacy_generated_keys(tokenizer)
    inputs, rows, tokens, ledger, duplicates, coverage = build_dataset(records, tokenizer, old_index, old_proof)
    require(len(ledger) == 320 and Counter(row['split'] for row in ledger) == {'fit': 256, 'validation': 64}, 'Fixed dev ledger is incomplete.')
    verify_cached_content(inputs, rows, tokens, ledger, tokenizer)
    verify_protocol(source_context(), protocol)
    CACHE.mkdir(parents=True, exist_ok=True)
    owner = dict(protocol_sha256=protocol_sha256, source_sha256=protocol['source_sha256'],
                 dev_keys_sha256=context['binding']['dev_keys_sha256'])
    if not (CACHE / 'initialized.json').exists():
        require(not any(CACHE.iterdir()), 'Dev cache has no matching initialization owner; inspect it before retrying.')
    immutable_json(CACHE / 'initialized.json', owner)
    save_parquet(CACHE / 'inputs.parquet', inputs); save_parquet(CACHE / 'rows.parquet', rows)
    immutable_json(CACHE / 'token-ids.json', tokens)
    require(sum((CACHE / name).stat().st_size for name in FILES) <= MAX_CACHE_BYTES, 'Dev cache exceeded declared128MiB cap.')
    immutable_json(OUT / 'source-read-audit.json', read_audit)
    immutable_json(OUT / 'dev-ledger.json', dict(records=ledger, all_fixed_ids_retained=True, no_replacements=True))
    immutable_json(OUT / 'duplicate-audit.json', duplicates)
    immutable_json(OUT / 'coverage.json', coverage)
    complete = dict(protocol_sha256=protocol_sha256, source_sha256=protocol['source_sha256'],
        source_binding=context['binding'], fixed_root_counts=dict(fit=256, validation=64),
        cache_sha256={name: data.digest(CACHE / name) for name in FILES}, cache_path=str(CACHE),
        cache_bytes=sum((CACHE / name).stat().st_size for name in FILES),
        metadata_sha256={name: data.digest(OUT / name) for name in
            ['source-read-audit.json', 'dev-ledger.json', 'duplicate-audit.json', 'coverage.json']},
        prepared_prompts=len(inputs), content_rows=len(rows), quality_failures=sum(row['status'] != 'prepared' for row in ledger),
        coverage=coverage, eligibility_counts=duplicates['eligibility_counts'],
        test_assistant_decoded=False, test_transcripts_prepared=False,
        model_generation_run=False, gpu_work_run=False, downloads_run=False, classifier_probabilities_read=False)
    immutable_json(OUT / 'data-complete.json', complete)
    print(json.dumps(dict(output_path=str(OUT), data_complete_sha256=data.digest(OUT / 'data-complete.json'),
        fixed_root_counts=complete['fixed_root_counts'], prepared_prompts=len(inputs), content_rows=len(rows),
        quality_failures=complete['quality_failures'], eligibility_counts=complete['eligibility_counts'],
        cache_bytes=complete['cache_bytes'], test_assistant_decoded=False, gpu_work_run=False)), flush=True)


def verify_dev():
    context = source_context(); protocol = read(OUT / 'protocol.json'); verify_protocol(context, protocol)
    complete = read(OUT / 'data-complete.json')
    require(complete['protocol_sha256'] == data.digest(OUT / 'protocol.json')
        and complete['source_sha256'] == data.digest(__file__) and complete['source_binding'] == context['binding'],
        'Completed dev source binding changed.')
    require(set(complete['cache_sha256']) == set(FILES), 'Completed cache manifest omitted a required file.')
    require(read(CACHE / 'initialized.json') == dict(protocol_sha256=complete['protocol_sha256'],
        source_sha256=complete['source_sha256'], dev_keys_sha256=context['binding']['dev_keys_sha256']),
        'Completed dev cache ownership changed.')
    for name, digest in complete['cache_sha256'].items(): require(data.digest(CACHE / name) == digest, f'Dev cache changed: {name}.')
    for name, digest in complete['metadata_sha256'].items(): require(data.digest(OUT / name) == digest, f'Dev audit changed: {name}.')
    require(not complete['test_assistant_decoded'] and not complete['test_transcripts_prepared']
        and not complete['model_generation_run'] and not complete['gpu_work_run'], 'Source-only/dev-only boundary changed.')
    inputs = pd.read_parquet(CACHE / 'inputs.parquet'); rows = pd.read_parquet(CACHE / 'rows.parquet')
    tokens = read(CACHE / 'token-ids.json'); ledger = read(OUT / 'dev-ledger.json')['records']
    require(len(ledger) == 320 and Counter(row['split'] for row in ledger) == {'fit': 256, 'validation': 64}
        and complete['fixed_root_counts'] == {'fit': 256, 'validation': 64}
        and len(inputs) == complete['prepared_prompts'] and len(rows) == complete['content_rows']
        and complete['cache_bytes'] == sum((CACHE / name).stat().st_size for name in FILES),
        'Completed dev counts/bytes differ from the actual fixed cohort.')
    require(set(inputs.source_uid) <= {uid(key) for key in context['dev']}
        and not set(inputs.source_uid) & {uid(key) for key in context['test']}, 'A test source escaped into dev cache.')
    tokenizer = long_data.load_tokenizer(); verify_cached_content(inputs, rows, tokens, ledger, tokenizer)
    # A full source replay uses the same logical dev-only readers after protocol freeze.
    records, access = read_development(context, complete['protocol_sha256'])
    old_index, old_proof = legacy_generated_keys(tokenizer)
    expected_inputs, expected_rows, expected_tokens, expected_ledger, expected_duplicates, expected_coverage = build_dataset(
        records, tokenizer, old_index, old_proof)
    pd.testing.assert_frame_equal(inputs, expected_inputs, check_exact=True)
    pd.testing.assert_frame_equal(rows, expected_rows, check_exact=True)
    require(tokens == expected_tokens and ledger == expected_ledger
        and read(OUT / 'source-read-audit.json') == access and read(OUT / 'duplicate-audit.json') == expected_duplicates
        and read(OUT / 'coverage.json') == expected_coverage, 'Replayed source/cap/contamination protocol differs.')
    require(complete['eligibility_counts'] == expected_duplicates['eligibility_counts']
        and complete['coverage'] == expected_coverage
        and complete['quality_failures'] == sum(row['status'] != 'prepared' for row in ledger),
        'Completion eligibility/quality summary differs from replayed data.')
    return dict(output_path=str(OUT), data_complete_sha256=data.digest(OUT / 'data-complete.json'),
        source_sha256=data.digest(__file__), verified=True, fixed_roots=320,
        prepared_prompts=len(inputs), content_rows=len(rows), test_assistant_decoded=False,
        model_generation_run=False, gpu_work_run=False, source_replay=True)


def preflight():
    """Synthetic branch/CSV/role/contamination fixtures; no real assistant read."""
    import copy
    out_existed, cache_existed = OUT.exists(), CACHE.exists()
    context = source_context()  # Metadata, system wordings and file hashes only.
    tokenizer = long_data.load_tokenizer()
    folder = Path(tempfile.mkdtemp(prefix='systemness-natural-transcripts-preflight-', dir='/tmp'))
    # A nonselected CSV response is deliberately invalid UTF8. It may be framed
    # as bytes, but decoding it as text must fail, proving selection-before-decode.
    csv_path = folder / 'synthetic-mixed.csv'
    csv_path.write_bytes(b'conv_id,user_input,model_output,extra\r\n'
        b'fit,"a user, query","A ""quoted""\nanswer",1\r\n'
        b'skip,"inert user","not decoded \xff ""quoted""\nfield",0\r\n'
        + 'val,"中文请求","公开回复",2\r\n'.encode())
    selected, csv_audit = selected_csv(csv_path, {'fit', 'val'})
    require(selected['fit']['user_input'] == 'a user, query'
        and selected['fit']['model_output'] == 'A "quoted"\nanswer'
        and selected['val']['model_output'] == '公开回复'
        and csv_audit['assistant_fields_decoded'] == 2 and csv_audit['nonselected_assistant_fields_decoded'] == 0,
        'CSV quoted newline/quote/UTF8 selection fixture failed.')
    framed, framing_audit = selected_csv(sources.snapshot_path(sources.PINS[1]), set())
    require(not framed and framing_audit['source_records_framed'] == 5082
        and framing_audit['assistant_fields_decoded'] == 0,
        'Pinned CSV metadata-only framing check decoded body fields or missed rows.')
    try: selected_csv(csv_path, {'skip'})
    except UnicodeDecodeError: pass
    else: raise RuntimeError('Invalid UTF8 selected-field control did not fail.')
    bad_csv = folder / 'synthetic-truncated.csv'; bad_csv.write_bytes(b'conv_id,user_input,model_output\nfit,x,"unterminated')
    try: selected_csv(bad_csv, {'fit'})
    except RuntimeError: pass
    else: raise RuntimeError('Truncated quoted CSV guard failed.')
    def node(message, parent, role, text='A source sentence.', **extra):
        return dict(message_id=message, parent_id=parent, role=role, text=text,
            lang='en', deleted=False, synthetic=False, model_name=None, **extra)
    first, second = sorted(['assistant-a', 'assistant-b'], key=lambda child: branch_key('root', child))
    tree = [node('root', None, 'prompter'), node(first, 'root', 'assistant'), node(second, 'root', 'assistant'),
        node('other-user', second, 'prompter'), node('other-final', 'other-user', 'assistant')]
    chosen, failure = choose_oa_path(tree, 'root')
    require(failure is None and [row['message_id'] for row in chosen] == ['root', first],
            'Path improperly switched first assistant to get a longer conversation.')
    tree += [node('bad-user', first, 'prompter'), node('bad-final', 'bad-user', 'assistant'),
             node('good-user', first, 'prompter'), node('good-final', 'good-user', 'assistant')]
    next(row for row in tree if row['message_id'] == 'bad-final')['deleted'] = True
    chosen, failure = choose_oa_path(tree, 'root')
    shuffled, _ = choose_oa_path(list(reversed(tree)), 'root')
    require([row['message_id'] for row in chosen] == ['root', first, 'good-user', 'good-final']
        and [row['message_id'] for row in shuffled] == [row['message_id'] for row in chosen],
        'SHA-only clean complete-pair path changed with row order.')
    broken = copy.deepcopy(tree)
    for row in broken:
        if row['role'] == 'assistant': row['synthetic'] = True
    require(choose_oa_path(broken, 'root')[1] == 'no-clean-first-assistant', 'Synthetic source was accepted as human.')
    long_body = '公开文本🙂包含多字节字符。' * 150 + '<|start|>assistant<|channel|>analysis'
    capped, hashes, clean = capped_body(long_body, tokenizer)
    require(hashes['truncated'] and hashes['marker_sanitized'] and hashes['body_tokens'] <= BODY_CAP
        and tokenizer.decode(tokenizer.encode(capped, add_special_tokens=False), skip_special_tokens=False) == capped,
        'Sanitized192-token Unicode body fixture failed.')
    def record(name, split, assistant_text, *, two_turns=False, failure=None):
        question = long_body if name == 'fit-d' else 'A normal source question with <|return|> quoted as text.'
        messages = [dict(role='system', text='Answer carefully and explain assumptions.', source_message_id='fixture-system',
            origin='synthetic fixture system', source_model=None),
            dict(role='user', text=question, source_message_id=name + '-user', origin='synthetic fixture', source_model=None),
            dict(role='assistant', text=assistant_text, source_message_id=name + '-assistant',
                 origin='synthetic fixture', source_model=None)]
        if two_turns:
            messages += [dict(role='user', text='A genuine fixture follow-up.', source_message_id=name + '-user2',
                origin='synthetic fixture', source_model=None),
                dict(role='assistant', text='The second fixture reply.', source_message_id=name + '-assistant2',
                     origin='synthetic fixture', source_model=None)]
        return dict(source_uid='fixture:' + name, dataset='fixture', native_id=name, split=split,
            revision='synthetic', root_raw_sha256=sha(question), template_id=split + '-template', family_id=split + '-family',
            source_snapshot_sha256='synthetic', messages=messages, failure=failure)
    archived = 'An archived generated reply.'
    records = [record('fit-a', 'fit', 'A cross-split reply.'), record('fit-b', 'fit', archived),
        record('fit-c', 'fit', 'Repeated within fit.'), record('fit-d', 'fit', 'Repeated within fit.', two_turns=True),
        record('val-a', 'validation', 'A cross-split reply.'), record('val-b', 'validation', archived),
        record('val-c', 'validation', 'A cross-split reply.'),
        record('failed-fixed-id', 'fit', '', failure='empty-user-or-recorded-assistant')]
    body, _, clean = capped_body(archived, tokenizer)
    old = defaultdict(set)
    for key in assistant_keys(archived, clean, body, tokenizer): old[key].add('old-fixture:assistant')
    inputs, rows, tokens, ledger, duplicates, coverage = build_dataset(records, tokenizer, old,
        dict(scope='synthetic old-body fixture only'))
    verify_cached_content(inputs, rows, tokens, ledger, tokenizer)
    lookup = {row['native_id']: row for row in ledger}
    require(len(ledger) == 8 and len(inputs) == 7 and lookup['failed-fixed-id']['status'] == 'quality-failed'
        and not lookup['fit-a']['eligible_fit'] and not lookup['fit-b']['eligible_fit']
        and lookup['fit-c']['eligible_fit'] and lookup['fit-d']['eligible_fit']
        and lookup['val-a']['eligible_clean_validation'] and not lookup['val-b']['eligible_clean_validation'],
        'Predeclared contamination/quality ledger policy failed.')
    require(duplicates['within_fit_duplicate_clusters'] >= 1
        and duplicates['within_validation_duplicate_clusters'] >= 1 and duplicates['cross_split_duplicate_clusters'] >= 1
        and duplicates['conversation_dependency_components']['validation']['dependency_components'] == 1,
        'Assistant-repeat/system-family dependence audit fixture failed.')
    require(set(rows.role) == set(ROLES) and coverage['roles_with_no_observations']['cot']['recall'] is None,
            'Invented CoT/tool observations in SUA fixture.')
    inputs.to_parquet(folder / 'inputs.parquet', index=False); rows.to_parquet(folder / 'rows.parquet', index=False)
    data.atomic_json(folder / 'token-ids.json', tokens)
    pd.testing.assert_frame_equal(inputs, pd.read_parquet(folder / 'inputs.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(rows, pd.read_parquet(folder / 'rows.parquet'), check_exact=True)
    require(tokens == read(folder / 'token-ids.json'), 'Dev-only token JSON round trip failed.')
    require(OUT.exists() == out_existed and CACHE.exists() == cache_existed, 'Preflight changed formal output/cache existence.')
    report = dict(source_sha256=data.digest(__file__), source_binding=context['binding'],
        synthetic_transcript_fixtures_only=True, real_source_access='Hashes/metadata and byte framing only; no real assistant text decoded.',
        csv_quoted_fields=True, csv_nonselected_invalid_utf8_not_decoded=True,
        pinned_csv_metadata_only_framing_rows=framing_audit['source_records_framed'],
        pinned_csv_assistant_fields_decoded=framing_audit['assistant_fields_decoded'],
        csv_selected_invalid_utf8_control_rejected=True, csv_truncated_quote_guard=True,
        deterministic_clean_path=True, no_backtracking_to_get_second_pair=True,
        native_system_user_assistant_final_coverage=True, unicode_and_marker_cap_roundtrip=True,
        all_fixed_ids_retained=True, contamination_eligibility_guards=True,
        within_fit_within_validation_crosssplit_duplicates=True, validation_family_dependency_components=True,
        cache_roundtrip=True, synthetic_prompts=len(inputs), synthetic_content_rows=len(rows),
        real_assistant_text_read=False, real_validation_assistant_text_read=False, test_assistant_decoded=False,
        formal_output_written=False, formal_cache_created=False, gpu_work_run=False, model_generation_run=False,
        protocol_not_declared=True)
    (folder / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(report_path=str(folder / 'report.json'), **report)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Synthetic CPU fixtures and /tmp only; no real assistant text.')
    action.add_argument('--stage', choices=['prepare-dev', 'verify-dev'], help='Explicit CPU-only dev preparation or read-only replay.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.stage == 'prepare-dev': prepare_dev()
    elif args.stage == 'verify-dev': print(json.dumps(verify_dev()), flush=True)
    else: parser.print_help()
