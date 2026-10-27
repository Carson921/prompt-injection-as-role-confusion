#!/usr/bin/env python3
"""One bounded, aggregate-only CPU audit of immutable formal input JSONL files."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

BASE = Path('/workspace/role-retraining-20261026')
CODE = Path('/root/role-retraining-20261026/code-v7')
RUN = BASE / 'runs-v7/formal'
DATA = BASE / 'data-v7/formal'
MODEL = Path('/root/role-retraining-model')
HARD = datetime(2026, 10, 27, 13, 19, 17, tzinfo=timezone.utc).timestamp()
ROLES = ('system', 'developer', 'user', 'cot', 'assistant', 'tool')
HEADER = {'system': ('S', 'system'), 'developer': ('D', 'developer'), 'user': ('U', 'user'),
          'assistant<|channel|>analysis': ('C', 'cot'), 'assistant<|channel|>final': ('A', 'assistant'),
          'assistant to=functions.lookup<|channel|>commentary': ('Q', None), 'functions.lookup': ('T', 'tool')}
PIN_FREEZE = 'ecaf97a27e5cd8f456279360493e619b0e3de2c9efff4a451b53028d7d94eb97'
PIN_MANIFEST = '6de19b0062d1ea6face9ae877b8f1dd0062dd03700dd792ebb0c9ed2e850cc6b'
PIN_BUILDER = 'cefc02ef9b49ec37fb334d2be98ec75f6b473a66e922191f12e5b460c2a2e07d'

def require(value, code):
    if not value:
        raise AuditError(code)

class AuditError(Exception):
    pass

def read(path):
    return json.loads(path.read_text())

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()

def source(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def deadline():
    ends = [HARD]
    for phase in ('pilot', 'formal'):
        config = read(BASE / 'runs-v7' / phase / 'config.json')
        budget = config['budget']
        start = datetime.fromisoformat(budget['started_utc'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(budget['deadline_utc'].replace('Z', '+00:00'))
        hours = float(budget['authorized_hours'])
        require(config['phase'] == phase and start.tzinfo is not None and end.tzinfo is not None and
                0 < hours <= 24 and 0 < (end-start).total_seconds() <= hours*3600, 'invalid_budget')
        ends.append(end.timestamp())
    result = min(ends)
    require(result > time.time(), 'authorized_deadline_exceeded')
    return result

def id_parse(ids, tokenizer, markers):
    """Fresh ID-level closed parser, independent of renderer and span metadata."""
    cursor, frames = 0, []
    terminal = {markers[name] for name in ('<|end|>', '<|call|>', '<|return|>')}
    framing = terminal | {markers['<|start|>'], markers['<|message|>']}
    while cursor < len(ids):
        require(ids[cursor] == markers['<|start|>'], 'id_message_start')
        header_start = cursor
        cursor += 1
        first = cursor
        while cursor < len(ids) and ids[cursor] != markers['<|message|>']:
            require(ids[cursor] not in framing, 'id_unclosed_header')
            cursor += 1
        require(cursor < len(ids), 'id_missing_message_marker')
        header = tokenizer.decode(ids[first:cursor], skip_special_tokens=False)
        require(header in HEADER, 'id_unknown_header')
        symbol, role = HEADER[header]
        cursor += 1
        body_start = cursor
        while cursor < len(ids) and ids[cursor] not in terminal:
            require(ids[cursor] not in framing, 'id_body_framing_marker')
            cursor += 1
        require(cursor < len(ids) and cursor > body_start, 'id_unclosed_or_empty_body')
        body_end = cursor
        stop = ids[cursor]
        cursor += 1
        expected = markers['<|call|>'] if symbol == 'Q' else markers['<|return|>'] if symbol == 'A' and cursor == len(ids) else markers['<|end|>']
        require(stop == expected, 'id_invalid_stop_marker')
        frames.append({'symbol': symbol, 'role': role, 'header_token_start': header_start,
                       'body_token_start': body_start, 'body_token_end': body_end})
    require(bool(frames), 'id_empty_input')
    return frames

def expected_symbols(record):
    profile, segments = record['profile'], record['round_segments']
    require(profile in ('single', 'one_round', 'multi_round'), 'unknown_profile')
    if profile == 'single':
        require(not segments and record['rounds'] == 0 and record['template_id'] in 'SDUCAT', 'single_template')
        return record['template_id']
    require(all(x in ('UA', 'UCA', 'UCQ', 'UQTCA', 'UQTA') for x in segments) and bool(segments), 'round_segment')
    require(all(x != 'UCQ' for x in segments[:-1]), 'historical_terminal_ucq')
    require(record['rounds'] == len(segments), 'round_count')
    if profile == 'one_round':
        require(len(segments) == 1 and record['template_id'] == 'SD' + segments[0], 'one_round_template')
    else:
        require(2 <= len(segments) <= (6 if record['split'] == 'test' else 4), 'multiround_length')
        require(record['template_id'] == 'SD+' + '+'.join(segments), 'multiround_template')
    messages = [('S', 0), ('D', 0)] + [(symbol, i+1) for i, segment in enumerate(segments) for symbol in segment]
    stage, policy = record['snapshot_stage'], record['history_policy']
    if stage == 'output':
        if policy == 'drop_completed_round_cot':
            messages = [(s, r) for s, r in messages if not (s == 'C' and r < len(segments))]
        else:
            require(policy in ('retain', 'drop_prior_call_cot'), 'output_policy')
    elif stage in ('pre_tool_call', 'tool_reply'):
        require(segments[-1] in ('UQTCA', 'UQTA'), 'call_snapshot_terminal_template')
        symbol = 'Q' if stage == 'pre_tool_call' else 'T'
        stop = next(i for i, (s, r) in enumerate(messages) if s == symbol and r == len(segments))
        messages = messages[:stop+1]
        if stage == 'pre_tool_call':
            require(policy == 'prior_call_cot_visible', 'pre_call_policy')
            messages.insert(len(messages)-1, ('C', len(segments)))
        else:
            require(policy == 'drop_prior_call_cot', 'tool_reply_policy')
    else:
        raise AuditError('snapshot_stage')
    return ''.join(s for s, _ in messages)

def check_record(record, tokenizer, markers, data, diag, counts, violations, chains):
    ids, targets, spans = record['input_ids'], record['targets'], record['message_spans']
    counts['records'] += 1
    counts['targets_present'] += len(targets)
    counts['input_tokens'] += len(ids)
    counts['max_input_tokens'] = max(counts['max_input_tokens'], len(ids))
    counts['profile/' + record['profile']] += 1
    counts['policy/' + record['history_policy']] += 1
    counts['stage/' + record['snapshot_stage']] += 1
    require(0 < len(ids) <= 4096 and record['input_tokens'] == len(ids), 'input_length')
    frames = id_parse(ids, tokenizer, markers)
    parsed = data.parse_native(record['text'])
    require(len(frames) == len(parsed) == len(spans), 'id_text_span_message_count')
    require([(x['symbol'], x['role']) for x in frames] == [(x['symbol'], x['role']) for x in parsed], 'id_text_role_sequence')
    symbols = ''.join(x['symbol'] for x in frames)
    require(symbols == expected_symbols(record), 'visible_snapshot_symbol_sequence')
    if record['profile'] == 'single':
        require(len(frames) == 1 and symbols == record['template_id'], 'single_exception')
        counts['single_message_exceptions_checked'] += 1
    else:
        require(symbols.startswith('SD') and symbols.count('S') == symbols.count('D') == 1, 'dialogue_sd_not_once')
        counts['dialogue_inputs_sd_exactly_once'] += 1
    pure, lengths, mixed = diag._body_layout(record, tokenizer)
    counts['independently_retokenized_inputs'] += 1
    counts['pure_body_tokens'] += sum(lengths.values())
    counts['visible_supervised_body_messages'] += len(lengths)
    counts['excluded_mixed_bpe_tokens'] += len(mixed)
    body_frame = [-1] * len(ids)
    pending, active_round, visible_events = None, 0, []
    events = {}
    for j, (frame, native, span) in enumerate(zip(frames, parsed, spans)):
        symbol = frame['symbol']
        counts['message/' + symbol] += 1
        for ix in range(frame['body_token_start'], frame['body_token_end']):
            body_frame[ix] = j
        require(span['symbol'] == symbol and span['role'] == frame['role'], 'span_id_role')
        require(span['event_id'] not in events, 'duplicate_visible_event')
        if symbol == 'U':
            active_round += 1
        expected_round = 0 if record['profile'] == 'single' or symbol in 'SD' else active_round
        require(span['round_ix'] == expected_round, 'visible_round_index')
        if span.get('tool_definition'):
            require(symbol == 'D', 'tool_definition_outside_developer')
            counts['developer_tool_definitions'] += 1
        if pending is not None:
            require(symbol == 'T', 'nonadjacent_tool_reply')
        if symbol == 'Q':
            require(pending is None and bool(span['call_id']) and span.get('tool_name') == 'functions.lookup', 'tool_call_identity')
            q = json.loads(record['text'][native['body_start']:native['body_end']])
            require(q == {'document_id': record['source_id'], 'query': 'excerpt-' + str(expected_round)}, 'tool_query_arguments')
            pending = span['call_id']
            counts['tool_calls'] += 1
        elif symbol == 'T' and record['profile'] != 'single':
            require(pending == span['call_id'] and span.get('tool_name') == 'functions.lookup', 'tool_reply_pair')
            pending = None
            counts['matched_tool_replies'] += 1
        if symbol == 'Q':
            require(any(s['symbol'] == 'D' and s.get('tool_definition') for s in spans), 'tool_call_without_definition')
        event = span['event_id']
        events[event] = {'symbol': symbol, 'round': span['round_ix'],
                         'body_sha': hashlib.sha256(record['text'][span['body_start']:span['body_end']].encode()).hexdigest()}
        visible_events.append(event)
    if pending is not None:
        require(frames[-1]['symbol'] == 'Q' and (record['snapshot_stage'] == 'pre_tool_call' or
                record['snapshot_stage'] == 'output' and record['round_segments'][-1] == 'UCQ'), 'unmatched_nonterminal_call')
        counts['legal_terminal_pending_calls'] += 1
    logical, removed = record['logical_event_ids'], record['removed_event_ids']
    require(len(logical) == len(set(logical)) and len(removed) == len(set(removed)), 'duplicate_logical_event')
    require(set(visible_events).issubset(logical) and set(removed) == set(logical)-set(visible_events), 'logical_visible_removed_partition')
    if record['history_policy'] == 'drop_completed_round_cot':
        require(not any(s['symbol'] == 'C' and s['round_ix'] < record['rounds'] for s in spans), 'historical_cot_still_visible')
    if record['history_policy'] == 'drop_prior_call_cot':
        require(not any(e.rsplit('/', 1)[-1] == 'C_before' for e in visible_events), 'prior_call_cot_still_visible')
    seen, mixed_set = set(), set(mixed)
    for target in targets:
        bad = set()
        ix = target.get('token_ix')
        if type(ix) is not int or not 0 <= ix < len(ids):
            bad.add('target_index')
        else:
            if ix in seen:
                bad.add('duplicate_target')
            seen.add(ix)
            frame = frames[body_frame[ix]] if body_frame[ix] >= 0 else None
            if frame is None:
                bad.add('header_or_delimiter_target')
            elif frame['symbol'] == 'Q':
                bad.add('q_json_target')
            if ix in mixed_set:
                bad.add('mixed_bpe_target')
            if ix not in pure:
                bad.add('non_pure_body_target')
                if frame and frame['symbol'] == 'D' and ix not in mixed_set:
                    bad.add('developer_tool_definition_target')
            else:
                observed = pure[ix]
                if frame is None or target.get('role') != frame['role'] or target.get('role') != observed['role'] or target.get('label') != ROLES.index(observed['role']):
                    bad.add('parsed_role_label_mismatch')
                if any(target.get(k) != observed[k] for k in ('event_id', 'message_id', 'body_position')):
                    bad.add('target_event_or_body_position')
                if target.get('family_id') != observed['event_id'] + ':' + str(observed['body_position']):
                    bad.add('target_family_identity')
                if target.get('source_id') != record['source_id'] or target.get('event_id') in removed:
                    bad.add('target_source_or_removed_event')
                if target.get('structure') != record['profile'] + ':' + record['template_id']:
                    bad.add('target_structure')
        weight = target.get('weight')
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 < weight <= 1:
            bad.add('target_weight')
        if bad:
            counts['invalid_targets_detected'] += 1
            violations.update(bad)
        else:
            counts['targets_verified'] += 1
            counts['target_role/' + target['role']] += 1
    counts['records_fully_checked'] += 1
    require(record['id'] == record['snapshot_id'], 'snapshot_id_disagreement')
    snapshot = {'id': record['snapshot_id'], 'stage': record['snapshot_stage'], 'policy': record['history_policy'],
                'rounds': record['rounds'], 'events': events, 'visible': visible_events,
                'ids_sha': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                'text_sha': hashlib.sha256(record['text'].encode()).hexdigest()}
    chains[record['call_chain_id']].append(snapshot)

def pair_checks(chains):
    counts, violations = Counter(), Counter()
    for snapshots in chains.values():
        ids = [s['id'] for s in snapshots]
        if len(ids) != len(set(ids)):
            violations['duplicate_snapshot_id'] += 1
        bases = [s for s in snapshots if s['stage'] == 'output' and s['policy'] != 'drop_completed_round_cot']
        for clean in (s for s in snapshots if s['policy'] == 'drop_completed_round_cot'):
            if len(bases) != 1:
                counts['drop_completed_pairs_unavailable'] += 1
                continue
            base = bases[0]
            expected = [e for e in base['visible'] if not (base['events'][e]['symbol'] == 'C' and base['events'][e]['round'] < base['rounds'])]
            good = clean['visible'] == expected and all(clean['events'][e] == base['events'][e] for e in expected)
            if not good:
                violations['drop_completed_pair_not_whole_message_deletion'] += 1
            if clean['ids_sha'] == base['ids_sha'] or clean['text_sha'] == base['text_sha']:
                violations['drop_completed_input_not_distinct'] += 1
            counts['drop_completed_pairs_checked'] += 1
        pres = [s for s in snapshots if s['stage'] == 'pre_tool_call']
        replies = [s for s in snapshots if s['stage'] == 'tool_reply']
        for reply in replies:
            if len(pres) != 1:
                counts['prior_call_pairs_unavailable'] += 1
                continue
            pre = pres[0]
            before = [e for e in pre['visible'] if e.rsplit('/', 1)[-1] == 'C_before']
            expected = [e for e in pre['visible'] if e not in before]
            good = len(before) == 1 and reply['visible'][:-1] == expected and reply['events'][reply['visible'][-1]]['symbol'] == 'T'
            good = good and all(reply['events'][e] == pre['events'][e] for e in expected)
            if not good:
                violations['prior_call_pair_removal_or_shared_body_mismatch'] += 1
            if reply['ids_sha'] == pre['ids_sha'] or reply['text_sha'] == pre['text_sha']:
                violations['prior_call_inputs_not_distinct'] += 1
            counts['prior_call_pairs_checked'] += 1
    return dict(counts), dict(violations)

def main():
    end = deadline()
    if sys.argv[1:] == ['--print-timeout']:
        print(max(1, min(600, int(end-time.time()))))
        return
    require(not sys.argv[1:], 'unknown_arguments')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'cuda_must_be_disabled')
    start = time.time()
    timer = threading.Timer(min(600, end-start), lambda: os.kill(os.getpid(), signal.SIGKILL))
    timer.daemon = True
    timer.start()
    proc = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    print(json.dumps({'audit_started': True, 'pid': os.getpid(), 'start_ticks': int(proc[19]),
                      'maximum_seconds': 600, 'absolute_deadline_unix': end,
                      'script_sha256': sha(Path(__file__))}), file=sys.stderr, flush=True)
    sys.path.insert(0, str(CODE))
    import synthetic_agent_data as data
    import synthetic_agent_position_diagnostics as diag
    require(sha(Path(data.__file__)) == PIN_BUILDER, 'frozen_builder_sha')
    manifest_path = DATA / 'data-manifest.json'
    manifest_source = source(manifest_path)
    require(manifest_source['sha256'] == PIN_MANIFEST, 'formal_manifest_sha')
    manifest, frozen = read(manifest_path), read(RUN/'freeze.json')
    require(sha(RUN/'freeze.json') == PIN_FREEZE, 'frozen_probe_manifest_sha')
    require(sha(RUN/'best.npz') == frozen['probe_sha256'], 'frozen_probe_sha')
    tokenizer, token_binding = diag._load_tokenizer(RUN, frozen['metadata'])
    markers = {}
    for token in ('<|start|>', '<|message|>', '<|end|>', '<|call|>', '<|return|>'):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        require(len(encoded) == 1, 'non_atomic_harmony_marker')
        markers[token] = encoded[0]
    output = {'format_version': 'synthetic-agent-actual-input-mask-audit-v1', 'scope': 'All existing formal FIT/DEV/TEST input records, structural/mask audit only; frozen readout; no predictions.',
              'started_unix': start, 'authorized_deadline_unix': end, 'maximum_seconds': 600,
              'script': source(Path(__file__)), 'formal_manifest': manifest_source,
              'validators': {name: source(CODE/name) for name in ('synthetic_agent_data.py', 'validate_synthetic_agent_data.py', 'synthetic_agent_position_diagnostics.py', 'synthetic_agent_gpu.py')},
              'tokenizer_binding': token_binding, 'freeze': source(RUN/'freeze.json'), 'splits': {}}
    total_violations = 0
    for split in ('fit', 'dev', 'test'):
        path = DATA / ('inputs-' + split + '.jsonl')
        before = path.stat()
        digest, counts, violations, chains = hashlib.sha256(), Counter(), Counter(), defaultdict(list)
        with path.open('rb') as stream:
            for ordinal, raw in enumerate(stream, 1):
                require(time.time() < end, 'authorized_deadline_exceeded')
                digest.update(raw)
                record = None
                try:
                    record = json.loads(raw)
                    require(record['split'] == split, 'input_split')
                    check_record(record, tokenizer, markers, data, diag, counts, violations, chains)
                except Exception as error:
                    code = str(error) if isinstance(error, AuditError) else 'frozen_layout_or_schema_exception_' + type(error).__name__
                    violations[code] += 1
                    counts['records_not_fully_checked'] += 1
                    counts['targets_not_independently_verified_due_to_record_failure'] += len(record.get('targets', [])) if isinstance(record, dict) else 0
                if ordinal % 2048 == 0:
                    print(json.dumps({'audit_progress': split, 'records': ordinal, 'targets_verified': counts['targets_verified'], 'record_failures': counts['records_not_fully_checked']}), file=sys.stderr, flush=True)
        actual_sha = digest.hexdigest()
        require(actual_sha == manifest['sha256'][path.name], 'input_sha_mismatch_' + split)
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'input_changed_' + split)
        paired, pair_violations = pair_checks(chains)
        violations.update(pair_violations)
        statistics = manifest['statistics'][split]
        require(counts['records'] == statistics['snapshots'] and counts['targets_present'] == statistics['rows'] and counts['input_tokens'] == statistics['input_tokens'], 'manifest_count_mismatch_' + split)
        require(counts['excluded_mixed_bpe_tokens'] == statistics['excluded_boundary_tokens'] or counts['records_not_fully_checked'], 'mixed_bpe_count_mismatch_' + split)
        total_violations += sum(violations.values()) + counts['invalid_targets_detected']
        unknown_targets = counts['targets_not_independently_verified_due_to_record_failure']
        requirement_counts = {
            'dialogue_sd_once_violations': violations.get('dialogue_sd_not_once', 0),
            'invalid_targets': None if unknown_targets else counts['invalid_targets_detected'],
            'invalid_targets_individually_detected': counts['invalid_targets_detected'],
            'targets_unverified_due_to_record_failure': unknown_targets,
            'header_or_delimiter_targets': violations.get('header_or_delimiter_target', 0),
            'q_json_targets': violations.get('q_json_target', 0),
            'developer_tool_definition_targets': violations.get('developer_tool_definition_target', 0),
            'mixed_article_structure_bpe_targets': violations.get('mixed_bpe_target', 0),
            'parsed_role_label_mismatches': violations.get('parsed_role_label_mismatch', 0),
            'records_not_fully_checked': counts['records_not_fully_checked'],
            'all_records_retokenized': counts['independently_retokenized_inputs'] == counts['records'],
            'all_targets_verified': counts['targets_verified'] == counts['targets_present'] and not unknown_targets,
        }
        output['splits'][split] = {'input': {'path': str(path), 'sha256': actual_sha, 'bytes': before.st_size},
                                    'counts': dict(counts), 'violations_by_check': dict(violations), 'snapshot_pair_checks': paired,
                                    'requirement_counts': requirement_counts,
                                    'manifest_counts_match': True}
    output.update(status='passed' if total_violations == 0 else 'failed_or_incomplete', finished_unix=time.time(), elapsed_seconds=time.time()-start,
                  violation_events=total_violations, frozen_sha_unchanged=sha(RUN/'freeze.json') == PIN_FREEZE,
                  formal_manifest_sha_unchanged=sha(manifest_path) == PIN_MANIFEST,
                  methods=['Fresh token-ID Harmony parser counts S/D and native roles independently of stored message spans.',
                           'Frozen position diagnostic independently retokenizes every text with official local offsets, compares all input IDs/spans, and recounts pure body/mixed-BPE tokens.',
                           'Each actual target is compared against pure token layout, ID-parsed role, source/event/body ordinal and family; Q JSON, headers, delimiters, tool definitions and mixed BPE are excluded.',
                           'Visible snapshot symbols, call/reply IDs, logical/removed event partition, whole-CoT deletion pairs and changed actual input hashes are independently checked where sibling records exist.'],
                  limitations=['Shared frozen parse_native/TOOL_DEFINITION and position-layout implementation are reused; this is not independent human gold annotation.',
                               'Article provenance, non-dialogue semantics, URL/SimHash exclusions and original article-fragment authenticity are outside this input-only structural audit.',
                               'All input versions are independently retokenized here; this does not prove a separate historical GPU forward for every version. That requires extraction receipts and use_cache=False implementation evidence.',
                               'Sibling records absent after target-row sampling are reported as unavailable; no missing comparison is invented.'],
                  privacy={'raw_text_exported': False, 'token_id_sequences_exported': False, 'source_or_conversation_files_opened': False,
                           'private_hash_ledgers_opened': False, 'predictions_opened': False, 'gpu_or_model_forward_run': False, 'input_or_core_files_modified': False})
    require(output['frozen_sha_unchanged'] and output['formal_manifest_sha_unchanged'], 'binding_changed_during_audit')
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    timer.cancel()

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(json.dumps({'status': 'audit_failed', 'error_type': type(error).__name__,
                          'error_code': str(error) if isinstance(error, AuditError) else 'unexpected_api_or_io_failure'}, sort_keys=True))
        raise SystemExit(1)
