"""Prepare fixed public S/U/A test transcripts with no response generation.

Default: help. --preflight uses synthetic CPU fixtures and /tmp only.
--stage prepare freezes four validation-selected readouts before materializing
any test assistant text. --stage verify replays this CPU-only preparation.
Neither stage loads model weights, runs inference, tunes a threshold or reads
any C4 confirmation result. Actual preparation is an explicit root-run action.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_head_confirm as head_confirm
import systemness_head_probes as head_probes
import systemness_long_data as long_data
import systemness_natural_sources as sources
import systemness_natural_transcripts as dev


OUT = data.runtime.OUT / 'systemness-natural-test'
CACHE = data.CACHE / 'natural-test'
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
METADATA = ['source-read-audit.json', 'test-ledger.json', 'duplicate-audit.json', 'coverage.json']
SAMPLE_START = 5_000_000_000_000
BODY_CAP = dev.BODY_CAP
ROLES = dev.ROLES
MIN_FREE_BYTES = MAX_CACHE_BYTES = 128 * 2**20
DEPENDENCIES = sorted(set(head_confirm.DEPENDENCIES + dev.DEPENDENCIES + [
    'systemness_natural_test.py', 'systemness_natural_test_confirm.py',
    'systemness_head_confirm.py', 'systemness_head_probes.py',
    'systemness_extract.py', 'systemness_components.py', 'feature_pipeline_runtime.py']))
require = long_data.require
read = sources.read
sha = sources.sha
canonical_sha = sources.canonical_sha
immutable_json = dev.immutable_json


def source_context():
    """Metadata/hash only; never call sources.verify() or dev.verify_dev()."""
    context = dev.source_context()
    folder = context['folder']
    templates = {row['template_id']: row for row in read(folder / 'system-templates.json')['templates']}
    assignments = {(row['dataset'], row['native_id']): row
                   for row in read(folder / 'template-assignments.json')['assignments']}
    test = context['test']
    require(Counter(row['dataset'] for row in test) == {'oasst': 64, 'toxicchat0124': 64}
        and all(row['split'] == 'test' for row in test), 'Fixed128 test source membership changed.')
    chosen = [assignments[(row['dataset'], row['native_id'])] for row in test]
    require(all(value == 4 for value in Counter(row['family_id'] for row in chosen).values()),
        'Each fixed test system family must retain four roots.')
    require(len({row['family_id'] for row in chosen}) == 32
        and len({row['template_id'] for row in chosen}) == 32
        and all(row['split'] == 'test' and templates[row['template_id']]['split'] == 'test' for row in chosen),
        'Fixed32 test wording assignments changed.')
    complete = read(dev.OUT / 'data-complete.json')
    protocol = read(dev.OUT / 'protocol.json')
    dev.verify_protocol(context, protocol)  # Hashes/metadata only, no dev reader.
    require(complete['protocol_sha256'] == data.digest(dev.OUT / 'protocol.json')
        and complete['source_sha256'] == data.digest(dev.__file__)
        and complete['source_binding'] == context['binding']
        and complete['fixed_root_counts'] == {'fit': 256, 'validation': 64}
        and complete['test_assistant_decoded'] is False
        and complete['test_transcripts_prepared'] is False
        and not complete['gpu_work_run'] and not complete['model_generation_run'],
        'Frozen development completion changed or crosses the test boundary.')
    require(set(complete['cache_sha256']) == set(dev.FILES)
        and set(complete['metadata_sha256']) == {'source-read-audit.json', 'dev-ledger.json', 'duplicate-audit.json', 'coverage.json'},
        'Dev completion omitted an artifact.')
    for name, digest in complete['cache_sha256'].items():
        require(data.digest(dev.CACHE / name) == digest, 'Frozen dev cache changed: ' + name)
    for name, digest in complete['metadata_sha256'].items():
        require(data.digest(dev.OUT / name) == digest, 'Frozen dev metadata changed: ' + name)
    binding = dict(context['binding'], test_keys_sha256=canonical_sha(test),
        test_assignments_sha256=canonical_sha(chosen),
        test_templates_sha256=canonical_sha([templates[key] for key in sorted({row['template_id'] for row in chosen})]),
        dev_data_complete_sha256=data.digest(dev.OUT / 'data-complete.json'),
        dev_protocol_sha256=data.digest(dev.OUT / 'protocol.json'),
        dev_source_sha256=data.digest(dev.__file__),
        dev_cache_sha256=complete['cache_sha256'], dev_metadata_sha256=complete['metadata_sha256'])
    return dict(folder=folder, test=test, dev=context['dev'], templates=templates, assignments=assignments,
        binding=binding, dev_complete=complete)


def frozen_snapshot():
    """Verified validation-only readouts; no C4 confirmation result is read."""
    paths, thresholds, selection_paths, records, complete = head_confirm.selections()
    # selections() calls verified_completion() and its promoted/guard gate.
    head_confirm.require_promoted(records[head_confirm.PRIMARY])
    checkpoint, tokenizer_sha = long_data.tokenizer_assets()
    config = read(checkpoint / 'config.json')
    require(config['quantization_config']['quant_method'] == 'mxfp4'
        and (config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (36, 64, 64),
        'Original native MXFP4 model geometry changed.')
    selected = complete['signature']['selected_head_ids']
    require(len(selected) == len(set(selected)) == 8, 'Selected eight-head order changed.')
    frozen = dict(models={name: data.digest(path) for name, path in paths.items()},
        model_paths={name: str(path.resolve()) for name, path in paths.items()}, thresholds=thresholds,
        selection_sha256={name: data.digest(path) for name, path in selection_paths.items()},
        primary_candidate=records[head_confirm.PRIMARY]['candidate'], primary_validation_promoted=True,
        selected_head_ids=selected, ordered_layers_heads=[[int(h // 64), int(h % 64)] for h in selected],
        head_completion_sha256=data.digest(head_probes.OUT / 'complete.json'),
        head_search_plan_sha256=data.digest(head_probes.OUT / 'search-plan.json'),
        head_standardization_sha256=data.digest(head_probes.OUT / 'standardization.npz'),
        checkpoint_path=str(checkpoint), tokenizer_sha256=tokenizer_sha,
        checkpoint_config_sha256=data.digest(checkpoint / 'config.json'),
        verified_weights_sha256=data.digest(checkpoint / 'verified-weights.json'),
        role_order=data.ROLES, test_used_for_selection=False, c4_confirmation_results_read=False)
    import systemness_natural_test_confirm as runner
    runner.validate_frozen(paths, thresholds, frozen)  # Exact primary pin before any test read.
    return paths, thresholds, frozen


def protocol_value(context, frozen):
    import systemness_natural_test_confirm as runner
    return dict(version=1, experiment='public-natural-test-transcripts', source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        source_binding=context['binding'], frozen=frozen, cache_path=str(CACHE),
        runner_protocol=runner.protocol_spec(),
        runner_protocol_sha256=canonical_sha(runner.protocol_spec()),
        test_roots=128, source_counts={'oasst': 64, 'toxicchat0124': 64},
        system_templates=32, roots_per_template=4, sample_ix_start=SAMPLE_START, roles=ROLES, body_cap=BODY_CAP,
        freeze_order='The promoted validation-only completion gate, immutable protocol and four-readout snapshot must pass and be persisted before any test assistant text is materialized.',
        selection='Exactly head_confirm.selections() models/thresholds/head order; no C4 result, natural-test result or test label tunes a model/threshold.',
        source_reads='Independent reader: OASST tree/ID metadata without text, Arrow selected-tree predicate before text-to-Python; quoted streaming ToxicChat conv_id membership before body decode. Separate reference-dev reads only320 fixed dev IDs. Physical Parquet blocks/CSV transport/hash I/O may include other bytes; logical decoded bodies are only the declared subset.',
        path='Frozen dev path: clean en/nondeleted/nonsynthetic/nonempty. First assistant SHA(parent+NUL+child), then first user with a complete clean assistant child and its first assistant by SHA. Do not backtrack first assistant. At most U-A-U-A.',
        serialization='Frozen dev BOS true; native system/user headers with end, assistant final headers with return serialization boundaries. These markers do not claim source generation completion. No invented analysis/tool segment.',
        source_provenance='Authentic OASST non-synthetic human text and ToxicChat recorded Vicuna-demo output (exact checkpoint unknown). Replies were neither generated by120B nor under the newly assigned system wording. Capping is not a claim that a source reply completed.',
        body_processing='Use frozen dev marker sanitation and round-trip-stable prefix cap192; record raw/clean/capped/body hashes and truncation for every message. Native system/user/assistant-final serialization; independent content-role parser covers every actual S/U/A token. CoT/tool support0, recall N/A.',
        quality='All128 fixed roots remain in attempted ledger, no replacements. Quality failures have no invented transcript. Keep every prepared transcript in all-test regardless of audit flag or future prediction.',
        contamination='Audit successfully prepared test assistant raw/clean/capped full, normalized and prefix192 keys against all old generated final/CoT/raw segments and all successfully prepared fixed dev320 assistants, plus test internal repeats. Clean-test excludes direct old/dev collisions only; within-test repeats remain and are clustered. No scores or outcomes enter cleanliness.',
        dependence='Union32 system families and test assistant repeats. Test roots also join when assistant keys link to a common dev or old reference, including transitive reference message components. Publish members and all/clean effective component counts. At most32 positive system stimuli; neither128 roots nor tokens are independent system stimuli.',
        excluded_inference_inputs='Source/template/family IDs, labels, token/message positions, quality/collision flags and dependency components are metadata only, never readout features.',
        gallery_source_uids=[sorted(dev.uid(row) for row in context['test'])[int(i)] for i in np.linspace(0, 127, 12).astype(int)],
        gallery='Fixed12 equally spaced lexicographically sorted source IDs before probabilities; failures keep placeholders, no replacements.',
        storage=dict(minimum_free_bytes=MIN_FREE_BYTES, maximum_cache_bytes=MAX_CACHE_BYTES,
            reuse_existing_snapshots=True, no_snapshot_copy=True),
        gpu_work_run=False, model_generation_run=False, downloads_run=False,
        classifier_probabilities_read=True,
        stored_validation_probabilities_read_for_completion_verification=True,
        test_classifier_probabilities_read=False,
        probability_read_scope='Only pre-existing development validation probability artifacts are read by the mandatory upstream verified_completion gate; no natural/C4 test probability or result is read.',
        primary_changed=False)


def freeze_protocol(context):
    _, _, frozen = frozen_snapshot()  # Mandatory gate before output creation/text.
    protocol = protocol_value(context, frozen)
    declaration = dict(frozen=frozen, source_sha256=data.digest(__file__),
        protocol_sha256=canonical_sha(protocol), test_assistant_materialized=False, test_used_for_selection=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('protocol.json', protocol), ('frozen-readouts.json', declaration)]:
        immutable_json(OUT / name, value)
        with (OUT / name).open('rb') as stream: os.fsync(stream.fileno())
    descriptor = os.open(OUT, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    guarded_source_read(data.digest(OUT / 'protocol.json'))
    return protocol


def verify_protocol(context):
    _, _, frozen = frozen_snapshot()
    protocol = read(OUT / 'protocol.json')
    require(protocol == protocol_value(context, frozen), 'Natural test protocol/source/model dependencies changed.')
    require(read(OUT / 'frozen-readouts.json') == dict(frozen=frozen, source_sha256=data.digest(__file__),
        protocol_sha256=canonical_sha(protocol), test_assistant_materialized=False, test_used_for_selection=False),
        'Natural test pre-read model snapshot changed.')
    return protocol


def guarded_source_read(protocol_sha256):
    require((OUT / 'protocol.json').is_file() and data.digest(OUT / 'protocol.json') == protocol_sha256,
        'Matching persisted natural-test protocol required before source text access.')
    declaration = read(OUT / 'frozen-readouts.json')
    protocol = read(OUT / 'protocol.json')
    require(declaration['protocol_sha256'] == canonical_sha(protocol)
        and declaration['frozen'] == protocol['frozen']
        and declaration['source_sha256'] == protocol['source_sha256'] == data.digest(__file__)
        and declaration['test_assistant_materialized'] is False
        and declaration['test_used_for_selection'] is False,
        'Missing pre-read four-model snapshot.')


def read_sources(context, which, protocol_sha256):
    """Own selected reader: never call dev.read_development or inherit its audit."""
    import pyarrow.dataset as ds
    require(which in ['test', 'reference-dev'], 'Unknown logically selected source subset.')
    guarded_source_read(protocol_sha256)
    keys = context['test'] if which == 'test' else context['dev']
    other = context['dev'] if which == 'test' else context['test']
    oa = {row['native_id'] for row in keys if row['dataset'] == 'oasst'}
    tc = {row['native_id'] for row in keys if row['dataset'] == 'toxicchat0124'}
    require(not {dev.uid(row) for row in keys} & {dev.uid(row) for row in other}, 'Source subset IDs overlap.')
    path = sources.snapshot_path(sources.PINS[0])
    columns = ['message_tree_id', 'message_id', 'parent_id', 'role', 'lang', 'deleted', 'synthetic', 'model_name']
    metadata = pd.read_parquet(path, columns=columns)
    require(oa <= set(metadata.message_tree_id), 'Fixed OASST tree absent.')
    selected = ds.dataset(path, format='parquet').to_table(
        filter=ds.field('message_tree_id').isin(sorted(oa)), columns=columns + ['text']).to_pandas()
    require(set(selected.message_tree_id) == oa, 'OASST selected text escaped declared IDs.')
    grouped = {str(native): frame.to_dict('records') for native, frame in selected.groupby('message_tree_id', sort=False)}
    toxic, csv_audit = dev.selected_csv(sources.snapshot_path(sources.PINS[1]), tc)
    require(set(toxic) == tc, 'ToxicChat selected records escaped declared IDs.')
    records = []
    for key in keys:
        assignment = context['assignments'][(key['dataset'], key['native_id'])]
        template = context['templates'][assignment['template_id']]
        base = dict(source_uid=dev.uid(key), dataset=key['dataset'], native_id=key['native_id'], split=key['split'],
            revision=key['revision'], root_raw_sha256=key['root_raw_sha256'], template_id=assignment['template_id'],
            family_id=assignment['family_id'], source_snapshot_sha256=sources.PINS[0 if key['dataset'] == 'oasst' else 1]['sha256'])
        messages = [dict(role='system', text=template['text'], source_message_id=template['template_id'],
            origin='new fixed system wording; not source generation context', source_model=None)]
        if key['dataset'] == 'oasst':
            chosen, failure = dev.choose_oa_path(grouped[key['native_id']], key['root_message_id'])
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
        if not failure: require(sha(messages[1]['text']) == key['root_raw_sha256'], 'Fixed root raw hash changed.')
        records.append(dict(**base, messages=messages, failure=failure))
    audit = dict(subset=which, protocol_sha256_before_text_materialization=protocol_sha256,
        logical_source_uids_sha256=canonical_sha([dev.uid(row) for row in keys]),
        oasst_metadata_rows_without_text=len(metadata), oasst_logical_text_rows_materialized=len(selected),
        oasst_logical_text_tree_ids_sha256=canonical_sha(sorted(oa)), oasst_nonselected_text_rows_materialized=0,
        parquet_physical_blocks_may_contain_nonselected_bytes=True, toxicchat=csv_audit,
        test_assistant_materialized=which == 'test', reference_dev_assistant_materialized=which == 'reference-dev',
        independent_subset_reader=True, development_reader_called=False)
    return records, audit


def processed_messages(record, tokenizer):
    processed = []; metadata = []
    if record['failure']: return [], [], record['failure']
    for index, message in enumerate(record['messages']):
        body, hashes, clean = dev.capped_body(message['text'], tokenizer)
        metadata.append(dict(message_ix=index, **{key: value for key, value in message.items() if key != 'text'}, **hashes))
        if not body.strip(): return [], metadata, 'empty-body-after-cap'
        processed.append((message['role'], body, clean, message['text']))
    require([row[0] for row in processed] in [['system', 'user', 'assistant'],
        ['system', 'user', 'assistant', 'user', 'assistant']], 'Not an authentic complete S/U/A source path.')
    return processed, metadata, None


def reference_assistants(records, tokenizer, expected_ledger=None):
    expected = {row['source_uid']: row for row in expected_ledger} if expected_ledger is not None else None
    result = []
    for record in records:
        processed, metadata, failure = processed_messages(record, tokenizer)
        if expected is not None:
            prior = expected[record['source_uid']]
            require(prior['status'] == ('quality-failed' if failure else 'prepared')
                and prior['quality_failure'] == failure and prior['messages'] == metadata,
                'Reference dev source replay differs from frozen dev ledger.')
        if failure: continue
        for index, (role, body, clean, raw) in enumerate(processed):
            if role == 'assistant':
                result.append(dict(message_uid=record['source_uid'] + '/message' + str(index),
                    source_uid=record['source_uid'], split=record['split'], message_ix=index,
                    keys=dev.assistant_keys(raw, clean, body, tokenizer)))
    return result


def duplicate_audit(ledger, assistant, reference, old_index, old_proof):
    """Connected duplicate references also preserve indirect bootstrap links."""
    test_by_message = {row['message_uid']: row for row in assistant}
    ref_by_message = {row['message_uid']: row for row in reference}
    require(not set(test_by_message) & set(ref_by_message), 'Test/dev assistant identities overlap.')
    all_messages = dict(ref_by_message, **test_by_message)
    index = defaultdict(set)
    for row in list(reference) + list(assistant):
        for key in row['keys']: index[key].add(row['message_uid'])
    for key, refs in old_index.items(): index[key].update('old-reference:' + ref for ref in refs)
    all_nodes = set().union(*index.values()) if index else set()
    graph = dev.Components(all_nodes)
    shared = [(key, sorted(nodes)) for key, nodes in index.items() if len(nodes) > 1]
    for key, nodes in shared: graph.join(nodes)
    shared_by_group = defaultdict(list)
    for key, nodes in shared: shared_by_group[graph.find(nodes[0])].append(key)
    clusters = []; message_cluster = {}
    for nodes in graph.groups():
        test_nodes = [node for node in nodes if node in test_by_message]
        if not test_nodes or len(nodes) < 2: continue
        cluster_id = 'assistant-repeat-' + canonical_sha(nodes)[:16]
        keys = shared_by_group[graph.find(nodes[0])]
        refs = [{key: all_messages[node][key] for key in ['message_uid', 'source_uid', 'split', 'message_ix']}
                for node in nodes if node in all_messages]
        clusters.append(dict(cluster_id=cluster_id, messages=refs,
            old_generated_references=[node.removeprefix('old-reference:') for node in nodes if node.startswith('old-reference:')],
            test_source_uids=sorted({test_by_message[node]['source_uid'] for node in test_nodes}),
            key_types=sorted({kind for kind, digest in keys}), shared_keys_sha256=canonical_sha(sorted(keys))))
        for node in test_nodes: message_cluster[node] = cluster_id
    ref_index = defaultdict(set)
    for row in reference:
        for key in row['keys']: ref_index[key].add(row['message_uid'])
    mapping = {row['source_uid']: row for row in ledger}
    for entry in ledger:
        entry.update(old_generated_collision=False, dev_assistant_collision=False, within_test_assistant_duplicate=False,
            old_generated_references=[], dev_assistant_references=[], assistant_duplicate_cluster_ids=[],
            dependency_component_id=None)
    message_flags = []
    for message in assistant:
        entry = mapping[message['source_uid']]
        old_keys = message['keys'] & set(old_index)
        dev_keys = message['keys'] & set(ref_index)
        old_refs = sorted(set().union(*(old_index[key] for key in old_keys))) if old_keys else []
        dev_refs = sorted(set().union(*(ref_index[key] for key in dev_keys))) if dev_keys else []
        test_matches = {node for key in message['keys'] for node in index[key] if node in test_by_message}
        internal = len(test_matches) > 1
        entry['old_generated_collision'] |= bool(old_refs); entry['dev_assistant_collision'] |= bool(dev_refs)
        entry['within_test_assistant_duplicate'] |= internal
        entry['old_generated_references'].extend(old_refs); entry['dev_assistant_references'].extend(dev_refs)
        if message['message_uid'] in message_cluster:
            entry['assistant_duplicate_cluster_ids'].append(message_cluster[message['message_uid']])
        message_flags.append(dict(message_uid=message['message_uid'], source_uid=message['source_uid'],
            old_key_types=sorted({kind for kind, digest in old_keys}), dev_key_types=sorted({kind for kind, digest in dev_keys}),
            within_test_duplicate=internal, keys_sha256=canonical_sha(sorted(message['keys']))))
    prepared = [entry for entry in ledger if entry['status'] == 'prepared']
    components = dev.Components([entry['source_uid'] for entry in prepared])
    families = defaultdict(list)
    for entry in prepared: families[entry['family_id']].append(entry['source_uid'])
    for nodes in families.values(): components.join(nodes)
    for cluster in clusters: components.join(cluster['test_source_uids'])
    for entry in ledger:
        for field in ['old_generated_references', 'dev_assistant_references', 'assistant_duplicate_cluster_ids']:
            entry[field] = sorted(set(entry[field]))
        entry['eligible_clean_test'] = entry['status'] == 'prepared' and not entry['old_generated_collision'] and not entry['dev_assistant_collision']
        entry['eligibility_reasons'] = [] if entry['status'] == 'prepared' else ['quality-failed']
        if entry['old_generated_collision']: entry['eligibility_reasons'].append('assistant-collision-with-old-generated-content')
        if entry['dev_assistant_collision']: entry['eligibility_reasons'].append('assistant-collision-with-fixed-development-content')
    groups = []
    for nodes in components.groups():
        component_id = 'test-dependency-' + canonical_sha(nodes)[:16]
        for node in nodes: mapping[node]['dependency_component_id'] = component_id
        groups.append(dict(component_id=component_id, source_uids=nodes,
            clean_source_uids=[node for node in nodes if mapping[node]['eligible_clean_test']],
            system_family_ids=sorted({mapping[node]['family_id'] for node in nodes})))
    counts = dict(all_fixed_ids=len(ledger), prepared=len(prepared), quality_failed=len(ledger) - len(prepared),
        eligible_clean_test=sum(entry['eligible_clean_test'] for entry in ledger),
        old_collision=sum(entry['old_generated_collision'] for entry in ledger),
        dev_collision=sum(entry['dev_assistant_collision'] for entry in ledger),
        internal_repeat=sum(entry['within_test_assistant_duplicate'] for entry in ledger))
    family_counts = dict(fixed=32, all_prepared=len(families),
        clean_prepared=len({entry['family_id'] for entry in prepared if entry['eligible_clean_test']}))
    return dict(duplicate_key_types=['full', 'normalized', 'prefix192'],
        variants='raw, marker-sanitized, actual capped; normalized NFKC/casefold/whitespace and corresponding token prefixes',
        old_generated_reference=old_proof, dev_reference_assistant_messages=len(reference), test_assistant_messages=len(assistant),
        duplicate_clusters=clusters, message_collision_flags=message_flags, eligibility_counts=counts,
        components=groups, dependency_components=len(groups), clean_dependency_components=sum(bool(row['clean_source_uids']) for row in groups),
        distinct_prepared_system_families=len(families), system_family_counts=family_counts,
        positive_system_stimuli_maximum=32,
        no_ids_replaced=True, no_outcome_based_filtering=True, model_probabilities_read=False,
        bootstrap='Paired conversation components union shared system family and assistant repeat links, including common old/dev references; use the same components for clean subset, never split a component after filtering.')


def build_dataset(records, tokenizer, reference, old_index, old_proof):
    ledger = []; inputs = []; rows = []; tokens = []; assistant = []
    for record in records:
        require(record['split'] == 'test', 'Non-test source in test builder.')
        processed, messages, failure = processed_messages(record, tokenizer)
        entry = {key: value for key, value in record.items() if key not in ['messages', 'failure']}
        entry.update(status='quality-failed' if failure else 'prepared', quality_failure=failure,
            messages=messages, prompt_ix=None, turns=0)
        if not failure:
            prompt, spans = dev.render_native([(role, body) for role, body, clean, raw in processed])
            ids, content = dev.content_rows(prompt, spans, tokenizer)
            prompt_ix = len(inputs); entry['prompt_ix'] = prompt_ix
            entry['turns'] = sum(role == 'user' for role, body, clean, raw in processed)
            for index, (role, body, clean, raw) in enumerate(processed):
                require(sum(row['message_ix'] == index for row in content) == messages[index]['body_tokens'],
                    'Actual serialized body token count changed.')
                if role == 'assistant': assistant.append(dict(message_uid=entry['source_uid'] + '/message' + str(index),
                    source_uid=entry['source_uid'], split='test', message_ix=index,
                    keys=dev.assistant_keys(raw, clean, body, tokenizer)))
            common = {key: entry[key] for key in ['source_uid', 'dataset', 'native_id', 'split', 'template_id', 'family_id']}
            inputs.append(dict(prompt_ix=prompt_ix, prompt=prompt, length=len(ids), turns=entry['turns'],
                message_count=len(processed), wrapper='native', profile='natural-public-sua', group_id=entry['source_uid'], **common))
            tokens.append(ids)
            for row in content: rows.append(dict(cache_ix=len(rows), sample_ix=SAMPLE_START + len(rows),
                prompt_ix=prompt_ix, wrapper='native', profile='natural-public-sua', group_id=entry['source_uid'], **common, **row))
        ledger.append(entry)
    require(len({row['source_uid'] for row in ledger}) == len(records), 'Repeated fixed source ID in test ledger.')
    audit = duplicate_audit(ledger, assistant, reference, old_index, old_proof)
    mapping = {entry['source_uid']: entry for entry in ledger}
    for entry in inputs + rows:
        for field in ['eligible_clean_test', 'old_generated_collision', 'dev_assistant_collision',
                      'within_test_assistant_duplicate', 'dependency_component_id']:
            entry[field] = mapping[entry['source_uid']][field]
    common_columns = ['source_uid', 'dataset', 'native_id', 'split', 'template_id', 'family_id']
    audit_columns = ['eligible_clean_test', 'old_generated_collision', 'dev_assistant_collision',
        'within_test_assistant_duplicate', 'dependency_component_id']
    inputs = pd.DataFrame(inputs, columns=['prompt_ix', 'prompt', 'length', 'turns', 'message_count',
        'wrapper', 'profile', 'group_id'] + common_columns + audit_columns)
    rows = pd.DataFrame(rows, columns=['cache_ix', 'sample_ix', 'prompt_ix', 'wrapper', 'profile', 'group_id']
        + common_columns + ['token_ix', 'token_id', 'token', 'message_ix', 'role', 'semantic_role'] + audit_columns)
    role_counts = Counter(rows.role)
    coverage = dict(all_content_tokens_scored=True, independent_native_labels_match=True,
        input_prompts=len(inputs), content_rows=len(rows), role_counts={role: int(role_counts[role]) for role in ROLES},
        no_observations=len(inputs) == 0,
        roles_with_no_observations={role: dict(support=0, recall=None, status='N/A') for role in ['cot', 'tool']},
        sample_ix_start=SAMPLE_START, sample_ix_end_exclusive=SAMPLE_START + len(rows),
        no_sparse_sampling=True, no_c4_document_or_shard_line_ids=True)
    return inputs, rows, tokens, ledger, audit, coverage


def verify_cached_content(inputs, rows, tokens, ledger, tokenizer):
    require(inputs.prompt_ix.tolist() == list(range(len(inputs))) and len(tokens) == len(inputs)
        and np.array_equal(rows.cache_ix, np.arange(len(rows)))
        and np.array_equal(rows.sample_ix, SAMPLE_START + np.arange(len(rows))), 'Test row identity changed.')
    lookup = {int(prompt): group for prompt, group in rows.groupby('prompt_ix', sort=False)}
    by_prompt = {row['prompt_ix']: row for row in ledger if row['status'] == 'prepared'}
    pattern = re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>)', re.S)
    require(set(inputs.split) <= {'test'} and set(rows.split) <= {'test'} and set(rows.role) <= set(ROLES)
        and len(inputs) == sum(entry['status'] == 'prepared' for entry in ledger),
        'Non-test or invented role in test cache.')
    for entry in inputs.itertuples():
        ids = tokenizer.encode(entry.prompt, add_special_tokens=False)
        require(ids == tokens[entry.prompt_ix] and len(ids) == entry.length, 'Test tokenization changed.')
        actual = long_data.parse_native_labels(ids, tokenizer); group = lookup[entry.prompt_ix]
        require(dict(zip(group.token_ix, group.role)) == actual and len(group) == len(actual)
            and all(ids[row.token_ix] == row.token_id for row in group.itertuples()), 'Native test content coverage changed.')
        record = by_prompt[entry.prompt_ix]; bodies = pattern.findall(entry.prompt)
        require(len(bodies) == len(record['messages']), 'Test provenance message count changed.')
        for index, (header, body, ending) in enumerate(bodies):
            message = record['messages'][index]; role = message['role']
            require(header == ('assistant<|channel|>final' if role == 'assistant' else role)
                and ending == ('<|return|>' if role == 'assistant' else '<|end|>')
                and sha(body) == message['serialized_body_sha256']
                and len(tokenizer.encode(body, add_special_tokens=False)) == message['body_tokens'] <= BODY_CAP,
                'Test native header/body hash/cap differs from provenance ledger.')
        require(entry.source_uid == entry.group_id == record['source_uid'], 'Test conversation mapping changed.')
        for field in ['template_id', 'family_id', 'dependency_component_id', 'eligible_clean_test',
                      'old_generated_collision', 'dev_assistant_collision', 'within_test_assistant_duplicate']:
            require(getattr(entry, field) == record[field] and all(group[field] == record[field]),
                'Test audit metadata mapping changed: ' + field)


def prepare():
    context = source_context(); protocol = freeze_protocol(context)
    if (OUT / 'data-complete.json').exists():
        complete, _, _, _ = load_verified_data()
        print(json.dumps(dict(data_complete_sha256=data.digest(OUT / 'data-complete.json'),
            prepared_prompts=complete['prepared_prompts'], verified=True)), flush=True); return
    require(shutil.disk_usage(data.CACHE).free >= MIN_FREE_BYTES, 'Less than128MiB free; do not copy snapshots/delete existing caches.')
    protocol_sha256 = data.digest(OUT / 'protocol.json')
    tokenizer = long_data.load_tokenizer()
    reference_records, reference_read = read_sources(context, 'reference-dev', protocol_sha256)
    reference = reference_assistants(reference_records, tokenizer, read(dev.OUT / 'dev-ledger.json')['records'])
    old_index, old_proof = dev.legacy_generated_keys(tokenizer)
    test_records, test_read = read_sources(context, 'test', protocol_sha256)
    inputs, rows, tokens, ledger, duplicates, coverage = build_dataset(test_records, tokenizer, reference, old_index, old_proof)
    require(len(ledger) == 128 and [row['source_uid'] for row in ledger] == [dev.uid(row) for row in context['test']],
        'Fixed128 ledger membership/order changed.')
    verify_cached_content(inputs, rows, tokens, ledger, tokenizer)
    verify_protocol(source_context())
    CACHE.mkdir(parents=True, exist_ok=True)
    owner = dict(protocol_sha256=protocol_sha256, source_sha256=data.digest(__file__),
        test_keys_sha256=context['binding']['test_keys_sha256'])
    if not (CACHE / 'initialized.json').exists(): require(not any(CACHE.iterdir()), 'Unowned partial test cache; inspect before retrying.')
    immutable_json(CACHE / 'initialized.json', owner)
    dev.save_parquet(CACHE / 'inputs.parquet', inputs); dev.save_parquet(CACHE / 'rows.parquet', rows)
    immutable_json(CACHE / 'token-ids.json', tokens)
    cache_bytes = sum((CACHE / name).stat().st_size for name in FILES)
    require(cache_bytes <= MAX_CACHE_BYTES, 'Test cache exceeded declared128MiB cap.')
    immutable_json(OUT / 'source-read-audit.json', dict(reference_dev=reference_read, test=test_read,
        test_assistant_materialized=True, gpu_work_run=False, model_probabilities_read=False))
    immutable_json(OUT / 'test-ledger.json', dict(records=ledger, all_fixed_ids_retained=True, no_replacements=True))
    immutable_json(OUT / 'duplicate-audit.json', duplicates); immutable_json(OUT / 'coverage.json', coverage)
    complete = dict(protocol_sha256=protocol_sha256, frozen_readouts_sha256=data.digest(OUT / 'frozen-readouts.json'),
        frozen=protocol['frozen'], source_sha256=data.digest(__file__), source_binding=context['binding'],
        source_keys_sha256=context['binding']['test_keys_sha256'],
        cache_path=str(CACHE), cache_sha256={name: data.digest(CACHE / name) for name in FILES}, cache_bytes=cache_bytes,
        metadata_sha256={name: data.digest(OUT / name) for name in METADATA},
        ledger_path=str(OUT / 'test-ledger.json'), ledger_sha256=data.digest(OUT / 'test-ledger.json'),
        duplicate_audit_path=str(OUT / 'duplicate-audit.json'), duplicate_audit_sha256=data.digest(OUT / 'duplicate-audit.json'),
        fixed_roots=128, fixed_system_families=32, prepared_prompts=len(inputs), content_rows=len(rows),
        eligibility_counts=duplicates['eligibility_counts'], components=duplicates['components'],
        system_family_counts=duplicates['system_family_counts'], no_observations=coverage['no_observations'],
        dependency_components=duplicates['dependency_components'], clean_dependency_components=duplicates['clean_dependency_components'],
        coverage=coverage, gallery_source_uids=protocol['gallery_source_uids'],
        test_assistant_materialized=True, source_replies_generated_under_system=False,
        source_replies_generated_by_120b=False, model_generation_run=False, gpu_work_run=False,
        downloads_run=False, classifier_probabilities_read=True,
        stored_validation_probabilities_read_for_completion_verification=True,
        test_classifier_probabilities_read=False, primary_changed=False)
    immutable_json(OUT / 'data-complete.json', complete)
    print(json.dumps(dict(output_path=str(OUT), data_complete_sha256=data.digest(OUT / 'data-complete.json'),
        prepared_prompts=len(inputs), content_rows=len(rows), eligibility_counts=duplicates['eligibility_counts'],
        dependency_components=duplicates['dependency_components'], gpu_work_run=False)), flush=True)


def verified_data():
    """Read-only cache validation; never read raw source or model probabilities."""
    context = source_context(); protocol = verify_protocol(context); complete = read(OUT / 'data-complete.json')
    require(complete['source_sha256'] == data.digest(__file__) and complete['source_binding'] == context['binding']
        and complete['protocol_sha256'] == data.digest(OUT / 'protocol.json')
        and complete['frozen_readouts_sha256'] == data.digest(OUT / 'frozen-readouts.json')
        and complete['frozen'] == protocol['frozen'] and complete['cache_path'] == str(CACHE)
        and complete['source_keys_sha256'] == context['binding']['test_keys_sha256']
        and complete['fixed_roots'] == 128 and complete['fixed_system_families'] == 32,
        'Completed test binding changed.')
    require(set(complete['cache_sha256']) == set(FILES) and set(complete['metadata_sha256']) == set(METADATA),
        'Test completion omitted a required artifact.')
    require(read(CACHE / 'initialized.json') == dict(protocol_sha256=complete['protocol_sha256'],
        source_sha256=data.digest(__file__), test_keys_sha256=context['binding']['test_keys_sha256']), 'Test cache owner changed.')
    for name, digest in complete['cache_sha256'].items(): require(data.digest(CACHE / name) == digest, 'Test cache changed: ' + name)
    for name, digest in complete['metadata_sha256'].items(): require(data.digest(OUT / name) == digest, 'Test audit changed: ' + name)
    require(complete['ledger_path'] == str(OUT / 'test-ledger.json')
        and complete['ledger_sha256'] == complete['metadata_sha256']['test-ledger.json']
        and complete['duplicate_audit_path'] == str(OUT / 'duplicate-audit.json')
        and complete['duplicate_audit_sha256'] == complete['metadata_sha256']['duplicate-audit.json'], 'Test manifest paths changed.')
    require(complete['test_assistant_materialized'] is True and not complete['model_generation_run']
        and not complete['gpu_work_run'] and not complete['downloads_run']
        and complete['classifier_probabilities_read'] is True
        and complete['stored_validation_probabilities_read_for_completion_verification'] is True
        and complete['test_classifier_probabilities_read'] is False
        and not complete['primary_changed'], 'CPU/source-only boundary changed.')
    inputs = pd.read_parquet(CACHE / 'inputs.parquet'); rows = pd.read_parquet(CACHE / 'rows.parquet')
    tokens = read(CACHE / 'token-ids.json'); ledger = read(OUT / 'test-ledger.json')['records']
    duplicates = read(OUT / 'duplicate-audit.json')
    require([row['source_uid'] for row in ledger] == [dev.uid(row) for row in context['test']]
        and all(row['split'] == 'test' for row in ledger) and len(ledger) == 128
        and len(inputs) == complete['prepared_prompts'] and len(rows) == complete['content_rows']
        and complete['eligibility_counts'] == duplicates['eligibility_counts']
        and complete['components'] == duplicates['components']
        and complete['dependency_components'] == duplicates['dependency_components']
        and complete['clean_dependency_components'] == duplicates['clean_dependency_components']
        and complete['system_family_counts'] == duplicates['system_family_counts']
        and complete['no_observations'] == (len(inputs) == 0)
        and complete['coverage'] == read(OUT / 'coverage.json')
        and complete['gallery_source_uids'] == protocol['gallery_source_uids'], 'Test completion/counts changed.')
    verify_cached_content(inputs, rows, tokens, ledger, long_data.load_tokenizer())
    return complete, inputs, rows, tokens, ledger


def load_verified_data():
    """Runner interface: (metadata, inputs, rows, token-ID lists)."""
    metadata, inputs, rows, tokens, _ = verified_data()
    return metadata, inputs, rows, tokens


def verify():
    """Explicit source replay verifies reference/test audits without inference."""
    complete, inputs, rows, tokens, ledger = verified_data()
    context = source_context(); tokenizer = long_data.load_tokenizer()
    reference_records, reference_audit = read_sources(context, 'reference-dev', complete['protocol_sha256'])
    reference = reference_assistants(reference_records, tokenizer, read(dev.OUT / 'dev-ledger.json')['records'])
    old_index, old_proof = dev.legacy_generated_keys(tokenizer)
    test_records, test_audit = read_sources(context, 'test', complete['protocol_sha256'])
    replay = build_dataset(test_records, tokenizer, reference, old_index, old_proof)
    pd.testing.assert_frame_equal(inputs, replay[0], check_exact=True)
    pd.testing.assert_frame_equal(rows, replay[1], check_exact=True)
    require(tokens == replay[2] and ledger == replay[3]
        and read(OUT / 'duplicate-audit.json') == replay[4] and complete['coverage'] == replay[5]
        and read(OUT / 'source-read-audit.json') == dict(reference_dev=reference_audit, test=test_audit,
            test_assistant_materialized=True, gpu_work_run=False, model_probabilities_read=False), 'Test source replay changed.')
    return dict(data_complete_sha256=data.digest(OUT / 'data-complete.json'), prepared_prompts=len(inputs),
        content_rows=len(rows), verified_source_replay=True, gpu_work_run=False, classifier_probabilities_read=True,
        stored_validation_probabilities_read_for_completion_verification=True, test_classifier_probabilities_read=False)


def preflight():
    """Only toy assistant text, metadata/hashes and local tokenizer are read."""
    existed = (OUT.exists(), CACHE.exists())
    context = source_context()
    tokenizer = long_data.load_tokenizer()
    folder = Path(tempfile.mkdtemp(prefix='systemness-natural-test-preflight-', dir='/tmp'))
    # Freeze-order rejection must happen before any source reader is invoked.
    original_out = globals()['OUT']; globals()['OUT'] = folder / 'unfrozen'
    try:
        try: guarded_source_read('not-frozen')
        except RuntimeError: pass
        else: raise RuntimeError('Unfrozen source access guard failed.')
    finally: globals()['OUT'] = original_out
    # Independently exercise the persisted declaration guard without creating
    # formal metadata or using any real source response/model probability.
    globals()['OUT'] = folder / 'frozen-fixture'
    try:
        OUT.mkdir()
        fixture_protocol = dict(frozen={'toy': True}, source_sha256=data.digest(__file__))
        fixture_declaration = dict(frozen=fixture_protocol['frozen'], source_sha256=data.digest(__file__),
            protocol_sha256=canonical_sha(fixture_protocol), test_assistant_materialized=False, test_used_for_selection=False)
        data.atomic_json(OUT / 'protocol.json', fixture_protocol)
        data.atomic_json(OUT / 'frozen-readouts.json', fixture_declaration)
        guarded_source_read(data.digest(OUT / 'protocol.json'))
        changed = copy.deepcopy(fixture_declaration); changed['frozen'] = {'toy': False}
        data.atomic_json(OUT / 'frozen-readouts.json', changed)
        try: guarded_source_read(data.digest(OUT / 'protocol.json'))
        except RuntimeError: pass
        else: raise RuntimeError('Mismatched frozen readout guard failed.')
        changed = copy.deepcopy(fixture_declaration); changed['source_sha256'] = '0' * 64
        data.atomic_json(OUT / 'frozen-readouts.json', changed)
        try: guarded_source_read(data.digest(OUT / 'protocol.json'))
        except RuntimeError: pass
        else: raise RuntimeError('Changed declaration source guard failed.')
    finally: globals()['OUT'] = original_out
    import systemness_natural_test_confirm as runner
    pinned_paths = {name: Path('/tmp/synthetic-only-' + name) for name in head_confirm.NAMES}
    pinned_thresholds = {name: .5 for name in head_confirm.NAMES}
    pinned_thresholds[runner.PRIMARY] = runner.PRIMARY_THRESHOLD
    pinned_frozen = dict(models={name: '0' * 64 for name in head_confirm.NAMES},
        primary_candidate=runner.PRIMARY_CANDIDATE, thresholds=pinned_thresholds)
    pinned_frozen['models'][runner.PRIMARY] = runner.PRIMARY_MODEL_SHA256
    runner.validate_frozen(pinned_paths, pinned_thresholds, pinned_frozen)
    bad_pin = copy.deepcopy(pinned_frozen); bad_pin['models'][runner.PRIMARY] = '0' * 64
    try: runner.validate_frozen(pinned_paths, pinned_thresholds, bad_pin)
    except RuntimeError: pass
    else: raise RuntimeError('Wrong exact primary model pin was accepted.')
    csv = folder / 'fixture.csv'
    csv.write_bytes(b'conv_id,user_input,model_output\r\n'
        b'test,"a, user","A ""quoted""\nreply"\r\n'
        b'nonselected,"ignored","invalid \xff response"\r\n')
    selected, csv_audit = dev.selected_csv(csv, {'test'})
    require(selected['test']['model_output'] == 'A "quoted"\nreply'
        and csv_audit['assistant_fields_decoded'] == 1, 'Selected quoted CSV fixture failed.')
    def record(name, text, family, *, split='test', failure=None, two=False):
        messages = [dict(role='system', text='Answer carefully and make uncertainty clear.', source_message_id='fixture-system',
            origin='synthetic fixture', source_model=None),
            dict(role='user', text='A normal question with <|return|> as quoted text.', source_message_id=name + '-u',
                 origin='synthetic fixture', source_model=None),
            dict(role='assistant', text=text, source_message_id=name + '-a', origin='synthetic fixture', source_model=None)]
        if two: messages += [dict(role='user', text='A real fixture follow-up.', source_message_id=name + '-u2',
            origin='synthetic fixture', source_model=None), dict(role='assistant', text='Second complete fixture reply.',
            source_message_id=name + '-a2', origin='synthetic fixture', source_model=None)]
        return dict(source_uid='fixture:' + name, dataset='fixture', native_id=name, split=split, revision='synthetic',
            root_raw_sha256=sha(messages[1]['text']), template_id=family + '-template', family_id=family,
            source_snapshot_sha256='synthetic', messages=messages, failure=failure)
    old_text = 'An archived generated reply.'
    dev_text = 'A reference development reply.'
    shared = 'A repeated fresh public reply.'
    records = [record('a', old_text, 'fa'), record('b', dev_text, 'fb'), record('c', shared, 'fc', two=True),
        record('d', shared, 'fd'), record('e', 'An independent answer. ' * 400 + '<|start|>assistant', 'fc'),
        record('failure', '', 'ff', failure='empty-user-or-recorded-assistant')]
    reference = reference_assistants([record('dev', dev_text, 'dev-family', split='fit')], tokenizer)
    old = defaultdict(set); body, _, clean = dev.capped_body(old_text, tokenizer)
    for key in dev.assistant_keys(old_text, clean, body, tokenizer): old[key].add('old-fixture:assistant')
    result = build_dataset(records, tokenizer, reference, old, dict(scope='synthetic only'))
    inputs, rows, tokens, ledger, duplicates, coverage = result
    verify_cached_content(inputs, rows, tokens, ledger, tokenizer)
    by_id = {entry['native_id']: entry for entry in ledger}
    require(len(ledger) == 6 and len(inputs) == 5 and by_id['failure']['status'] == 'quality-failed'
        and not by_id['a']['eligible_clean_test'] and not by_id['b']['eligible_clean_test']
        and all(by_id[name]['eligible_clean_test'] for name in ['c', 'd', 'e'])
        and by_id['c']['dependency_component_id'] == by_id['d']['dependency_component_id'] == by_id['e']['dependency_component_id']
        and duplicates['system_family_counts'] == {'fixed': 32, 'all_prepared': 4, 'clean_prepared': 2},
        'Fixed failure/clean/family/duplicate component fixture failed.')
    require(set(rows.role) == set(ROLES) and coverage['roles_with_no_observations']['cot']['recall'] is None
        and by_id['e']['messages'][2]['truncated'] and by_id['e']['messages'][2]['marker_sanitized'],
        'Native coverage, cap or absent-role fixture failed.')
    # Two distinct test bodies can link through one external reference which
    # contains both typed keys. Bootstrap still must retain this dependency.
    k1, k2 = ('full', 'a' * 64), ('normalized', 'b' * 64)
    toy_ledger = [dict(source_uid='t1', family_id='one', status='prepared'), dict(source_uid='t2', family_id='two', status='prepared')]
    toy_assistant = [dict(message_uid='t1/m', source_uid='t1', split='test', message_ix=2, keys={k1}),
        dict(message_uid='t2/m', source_uid='t2', split='test', message_ix=2, keys={k2})]
    toy_ref = [dict(message_uid='dev/m', source_uid='dev', split='validation', message_ix=2, keys={k1, k2})]
    bridge = duplicate_audit(toy_ledger, toy_assistant, toy_ref, {}, {})
    require(bridge['dependency_components'] == 1 and bridge['clean_dependency_components'] == 0,
        'Shared external reference bootstrap bridge was lost.')
    wrong = rows.copy(); wrong.loc[0, 'role'] = 'assistant'
    try: verify_cached_content(inputs, wrong, tokens, ledger, tokenizer)
    except RuntimeError: pass
    else: raise RuntimeError('Tampered content role guard failed.')
    try: build_dataset([record('not-test', 'An answer.', 'fit', split='fit')], tokenizer, [], {}, {})
    except RuntimeError: pass
    else: raise RuntimeError('Non-test split guard failed.')
    empty = build_dataset([record('failed1', '', 'f1', failure='empty-source'),
        record('failed2', '', 'f2', failure='empty-source')], tokenizer, [], {}, {})
    verify_cached_content(empty[0], empty[1], empty[2], empty[3], tokenizer)
    require(len(empty[3]) == 2 and all(entry['status'] == 'quality-failed' for entry in empty[3])
        and empty[0].empty and empty[1].empty and empty[5]['no_observations']
        and empty[4]['dependency_components'] == 0 and empty[4]['system_family_counts']['all_prepared'] == 0,
        'All-failed source ledger/empty cache schema failed.')
    empty[0].to_parquet(folder / 'empty-inputs.parquet', index=False)
    empty[1].to_parquet(folder / 'empty-rows.parquet', index=False)
    pd.testing.assert_frame_equal(empty[0], pd.read_parquet(folder / 'empty-inputs.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(empty[1], pd.read_parquet(folder / 'empty-rows.parquet'), check_exact=True)
    inputs.to_parquet(folder / 'inputs.parquet', index=False); rows.to_parquet(folder / 'rows.parquet', index=False)
    data.atomic_json(folder / 'token-ids.json', tokens)
    pd.testing.assert_frame_equal(inputs, pd.read_parquet(folder / 'inputs.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(rows, pd.read_parquet(folder / 'rows.parquet'), check_exact=True)
    require(tokens == read(folder / 'token-ids.json') and existed == (OUT.exists(), CACHE.exists()), 'Preflight mutated formal state.')
    report = dict(source_sha256=data.digest(__file__), source_binding=context['binding'],
        frozen_source_metadata_only=True, real_assistant_text_read=False, test_assistant_materialized=False,
        synthetic_prompts=len(inputs), synthetic_rows=len(rows), freeze_order_guard=True,
        exact_primary_pin_before_source_read=True, frozen_declaration_source_and_snapshot_tamper_guards=True,
        quoted_csv_invalid_nonselected_utf8_not_decoded=True, native_SUA_final_all_content_parser=True,
        cap_marker_unicode_rules_inherited_from_frozen_dev=True, no_CoT_tool_observations=True,
        fixed_failure_ledger_no_replacements=True, old_dev_internal_duplicate_keys=True,
        all_failed_complete_ledger_and_empty_cache_schema=True, all_clean_system_family_counts=True,
        family_repeat_and_external_reference_components=True, no_outcome_based_clean_filter=True,
        split_and_role_tamper_guards=True, parquet_token_roundtrip=True,
        model_probabilities_read=False, classifier_probabilities_read=False,
        stored_validation_probabilities_read_for_completion_verification=False,
        test_classifier_probabilities_read=False, gpu_work_run=False, model_generation_run=False,
        formal_output_written=False, formal_cache_created=False, protocol_not_declared=True)
    data.atomic_json(folder / 'report.json', report)
    print(json.dumps(dict(report_path=str(folder / 'report.json'), **report)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Synthetic CPU /tmp fixtures; no real assistant text.')
    action.add_argument('--stage', choices=['prepare', 'verify'], help='Explicit CPU-only fixed-test preparation or source replay.')
    args = parser.parse_args()
    if args.preflight: preflight()
    elif args.stage == 'prepare': prepare()
    elif args.stage == 'verify': print(json.dumps(verify()), flush=True)
    else: parser.print_help()
