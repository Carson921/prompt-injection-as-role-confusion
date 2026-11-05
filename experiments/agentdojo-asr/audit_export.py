"""Reevaluate every saved trajectory with AgentDojo and align ASR to probe rows."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'provenance'))
from common import BASE, OUTPUT, read, write, sha, serial, signature, phase
from snapshot import restore_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    import numpy as np
    from transformers import AutoTokenizer
    identity = read(BASE / 'agentdojo-source.json')
    source = Path(identity['source'])
    assert all(sha(source / name) == digest for name, digest in identity['files'].items())
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(BASE / 'provenance'))
    from bridge import parse_native
    from spans import render
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.task_suite.load_suites import get_suites
    from agentdojo.task_suite.task_suite import model_output_from_messages, functions_stack_trace_from_messages
    from agentdojo.types import text_content_block_from_string
    suites = get_suites('v1.2.1')
    manifest = read(OUTPUT / 'manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(BASE / 'provenance' / 'tokenizer' if (BASE / 'provenance/tokenizer').exists()
                                            else '/scratch/wz55191/models/gpt-oss-120b', local_files_only=True)
    binding = signature(manifest)
    assert manifest['base_manifest_sha256'] == sha(BASE / 'manifest.json')
    assert manifest['schedule_sha256'] == sha(BASE / 'schedule.json')
    assert manifest['canonical_cases_sha256'] == sha(BASE / 'canonical_cases.json')
    assert manifest['source_sha256'] == sha(BASE / 'agentdojo-source.json')
    assert all(sha(OUTPUT / 'provenance' / name) == digest for name, digest in manifest['continuation_code_sha256'].items())
    assert read(OUTPUT / 'validation.json')['status'] == 'passed'
    snapshot_validation = read(OUTPUT / 'snapshot_validation.json')
    assert snapshot_validation['status'] == 'passed'
    assert snapshot_validation['snapshot_restore_source_sha256'] == sha(Path(__file__).with_name('snapshot.py'))
    assert snapshot_validation['verification_source_sha256'] == sha(Path(__file__).with_name('verify_snapshot.py'))
    assert snapshot_validation['agentdojo_source_sha256'] == sha(BASE / 'agentdojo-source.json')
    schedule = read(BASE / 'schedule.json')
    canonical = read(BASE / 'canonical_cases.json')
    feature_index = {r['case_id']: r for r in read(BASE / 'feature_index.json')}
    assert len(schedule) == len(canonical) == 1898
    assert [r['case_id'] for r in schedule] == [r['case_id'] for r in canonical]
    counts = Counter()
    records = []
    calls = replayed = native = capped = 0
    all_output_paths = set()
    for case_index, (case, origin) in enumerate(zip(schedule, canonical)):
        case_id = case['case_id']
        result_path = OUTPUT / 'cases' / (case_id + '.json')
        if not result_path.exists():
            assert args.partial, f'Missing {case_id}'
            continue
        result = read(result_path)
        assert result['status'] == 'completed', result
        assert result['case_id'] == case_id
        assert type(result['attacker_success']) is bool and type(result['utility']) is bool
        assert result['security'] == result['attacker_success']
        assert result['source_result'] == origin['source_result']
        assert result['source_result_sha256'] == sha(BASE / origin['source_result'])
        previous = read(BASE / origin['source_result'])
        exposed = origin['status'] == 'captured'
        first = int(origin['first_exposure_call'].rsplit('_', 1)[1]) if exposed else None
        assert result['first_exposure_call'] == origin.get('first_exposure_call')
        if exposed:
            entry = feature_index[case_id]
            assert entry['case_index'] == case_index
            assert result['feature_row_start'] == origin['feature_row_start'] == entry['row_start']
            assert result['feature_row_end'] == origin['feature_row_end'] == entry['row_end']
            assert len(result['requests']) > first
            receipt_path = phase(origin) / previous['capture']
            receipt = read(receipt_path)
            for name, digest in receipt['files'].items():
                assert sha(receipt_path.parent / name) == digest
            assert entry['states_sha256'] == sha(BASE / entry['states'])
        else:
            assert len(result['requests']) == len(previous['requests'])
            assert result['security'] == previous['security'] and result['utility'] == previous['utility']
        limited_calls = 0
        for index, request in enumerate(result['requests']):
            call_id = case_id + f'/call_{index:04d}'
            assert request['call_id'] == call_id and request['call_index'] == index
            directory = OUTPUT / 'calls' / call_id
            raw_path, inp_path = directory / 'original_request.json', directory / 'input.json'
            raw, inp = read(raw_path), read(inp_path)
            response_path, parsed_path = directory / 'response.json', directory / 'parsed_response.json'
            response, parsed = read(response_path), read(parsed_path)
            all_output_paths.add(str(raw_path.parent.relative_to(OUTPUT / 'calls')))
            assert request['request_sha256'] == sha(raw_path) == inp['original_request_sha256']
            assert request['response_sha256'] == sha(response_path) == parsed['source_response_sha256']
            assert request['parsed_response_sha256'] == sha(parsed_path)
            assert response['input_sha256'] == sha(inp_path)
            assert inp['binding'] == response['binding'] == binding
            assert inp['call_id'] == response['call_id'] == parsed['call_id'] == call_id
            assert all(raw[k] == v for k, v in case.items())
            ids = inp['input_ids']
            assert len(ids) == inp['input_tokens'] and len(ids) <= 131072
            rendered = render(tokenizer, raw, manifest['generation']['prompt_date'])
            assert rendered == inp['rendered_prompt']
            assert tokenizer.encode(rendered, add_special_tokens=False) == ids
            assert parsed['native_generated_ids'] == response['generated_ids']
            assert parsed['message'] == parse_native(response['generated_ids'], call_id)
            is_replay = not exposed or index < first
            assert request['status'] == response['mode'] == ('prefix_replay' if is_replay else 'native_generation')
            assert inp['is_first_exposure'] == (index == first)
            if is_replay or index == first:
                reference = inp['base_input']
                assert reference['full_token_ids_equal'] and reference['rendered_prompt_equal'] and reference['raw_request_equal']
                old_inp_path = BASE / reference['path']
                assert sha(old_inp_path) == reference['sha256']
                old_inp = read(old_inp_path)
                assert old_inp['input_ids'] == ids and old_inp['rendered_prompt'] == inp['rendered_prompt']
                old_raw_path = BASE / reference['original_request_path']
                assert sha(old_raw_path) == reference['original_request_sha256'] == previous['requests'][index]['request_sha256']
                assert read(old_raw_path) == raw
            else:
                assert inp['base_input'] is None
            if is_replay:
                ref = response['source_response']
                assert ref['sha256'] == sha(BASE / ref['path']) == previous['requests'][index]['response_sha256']
                assert read(BASE / ref['path'])['generated_ids'] == response['generated_ids']
                replayed += 1
            else:
                assert response['source_response'] is None
                key = signature({'binding': binding, 'input_ids': ids})
                assert response['cache_key'] == key
                cache_path = OUTPUT / 'generation_cache' / (key + '.json')
                cache = read(cache_path)
                assert sha(cache_path) == response['cache_sha256']
                assert cache['input_ids'] == ids and cache['generated_ids'] == response['generated_ids'] and cache['binding'] == binding
                assert 1 <= len(response['generated_ids']) <= manifest['generation']['max_tokens']
                if len(response['generated_ids']) == manifest['generation']['max_tokens'] and response['generated_ids'][-1] not in (200002, 200012, 199999):
                    limited_calls += 1
                native += 1
            calls += 1
        invocations = result['trajectory_invocations']
        assert 1 <= len(invocations) <= 3
        trajectories = []
        for invocation in invocations:
            path = OUTPUT / invocation['path']
            assert sha(path) == invocation['sha256']
            trajectory = read(path)
            messages = trajectory['messages']
            restored_messages = []
            for message in messages:
                message = dict(message)
                if message.get('tool_calls'):
                    message['tool_calls'] = [FunctionCall.model_validate(v) for v in message['tool_calls']]
                if message.get('tool_call'):
                    message['tool_call'] = FunctionCall.model_validate(message['tool_call'])
                restored_messages.append(message)
            assert serial(model_output_from_messages(restored_messages)) == serial(trajectory['model_output'])
            assert serial(functions_stack_trace_from_messages(restored_messages)) == serial(trajectory['functions_stack_trace'])
            trajectories.append(trajectory)
        suite = suites[case['suite']]
        env_type = type(suite.load_and_inject_default_environment(case['injections']))
        before = restore_environment(env_type, trajectories[0]['pre_environment'])
        after = restore_environment(env_type, trajectories[-1]['final_environment'])
        final = trajectories[-1]
        stack = [FunctionCall.model_validate(v) for v in final['functions_stack_trace']]
        utility = bool(suite._check_task_result(suite.user_tasks[case['user_task_id']], final['model_output'] or [], before, after, stack))
        security = bool(suite._check_task_result(suite.injection_tasks[case['injection_task_id']], final['model_output'] or [], before, after, stack))
        assert utility == result['utility'] and security == result['attacker_success'], f'Predicate recheck differs: {case_id}'
        last_message = final['messages'][-1] if final['messages'] else {}
        termination = 'native_tool_loop_limit' if last_message.get('role') == 'assistant' and last_message.get('tool_calls') else 'pipeline_returned'
        record = {k: result[k] for k in ['case_id', 'condition', 'suite', 'user_task_id', 'injection_task_id', 'attacker_success', 'utility', 'base_case_status', 'first_exposure_call', 'feature_row_start', 'feature_row_end']}
        record.update(case_index=case_index, payload_exposed=exposed, asr_label=int(security),
                      result=str(result_path.relative_to(OUTPUT)), result_sha256=sha(result_path),
                      model_calls=len(result['requests']), post_exposure_calls=0 if not exposed else len(result['requests']) - first,
                      generation_token_limit_calls=limited_calls, termination=termination,
                      official_predicates_rechecked=True)
        records.append(record)
        counts[(case['condition'], case['suite'], 'all', 'cases')] += 1
        counts[(case['condition'], case['suite'], 'all', 'successes')] += int(security)
        if exposed:
            counts[(case['condition'], case['suite'], 'exposed', 'cases')] += 1
            counts[(case['condition'], case['suite'], 'exposed', 'successes')] += int(security)
        capped += limited_calls
        if len(records) % 100 == 0:
            print(serial({'event': 'audit_progress', 'verified_cases': len(records)}), flush=True)
    if args.partial:
        print(serial({'status': 'partial_verified', 'cases': len(records), 'calls': calls,
                      'attacker_successes': sum(r['attacker_success'] for r in records)}), flush=True)
        return
    assert len(records) == 1898
    assert sum(r['payload_exposed'] for r in records) == 1852
    assert set(p.relative_to(OUTPUT / 'cases').as_posix()[:-5] for p in (OUTPUT / 'cases').rglob('*.json')) == {r['case_id'] for r in records}
    assert set(p.parent.relative_to(OUTPUT / 'calls').as_posix() for p in (OUTPUT / 'calls').rglob('original_request.json')) == all_output_paths
    def summarize(subset):
        n = len(subset)
        successes = sum(r['attacker_success'] for r in subset)
        return {'cases': n, 'attacker_successes': successes, 'asr': successes / n if n else None,
                'utility_successes': sum(r['utility'] for r in subset),
                'utility_rate': sum(r['utility'] for r in subset) / n if n else None}
    summary = {'status': 'complete', 'asr_definition': 'mean(attacker_success) from AgentDojo injection-task predicates',
        'all_cases': summarize(records), 'payload_exposed_cases': summarize([r for r in records if r['payload_exposed']]),
        'unexposed_cases': summarize([r for r in records if not r['payload_exposed']]),
        'by_condition': {}, 'by_condition_and_suite': [],
        'native_limits': {'max_generated_tokens_per_call': 1024, 'tool_loop_max_iters': 15,
                          'task_max_attempts': 3, 'calls_hitting_token_limit': capped,
                          'cases_ending_with_tool_calls': sum(r['termination'] == 'native_tool_loop_limit' for r in records)}}
    for condition in ['plain_delimiter', 'remember_delimiter']:
        selected = [r for r in records if r['condition'] == condition]
        summary['by_condition'][condition] = {'all_cases': summarize(selected), 'payload_exposed_cases': summarize([r for r in selected if r['payload_exposed']])}
        for suite in ['banking', 'slack', 'travel', 'workspace']:
            selected = [r for r in records if r['condition'] == condition and r['suite'] == suite]
            summary['by_condition_and_suite'].append({'condition': condition, 'suite': suite,
                'all_cases': summarize(selected), 'payload_exposed_cases': summarize([r for r in selected if r['payload_exposed']])})
    write(OUTPUT / 'case_results.json', records)
    write(OUTPUT / 'asr_summary.json', summary)
    fields = list(records[0])
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    (OUTPUT / 'case_results.csv').write_text(buf.getvalue())
    label_dir = OUTPUT / 'probe_labels'
    label_dir.mkdir(exist_ok=True)
    labels = np.asarray([r['asr_label'] for r in records], dtype=np.uint8)
    utilities = np.asarray([r['utility'] for r in records], dtype=np.bool_)
    exposed = np.asarray([r['payload_exposed'] for r in records], dtype=np.bool_)
    probe_manifest = read(BASE / 'probe_inputs/manifest.json')
    for name, digest in probe_manifest['SHA256'].items():
        assert sha(BASE / 'probe_inputs' / name) == digest, f'Original probe input changed: {name}'
    case_indices = np.load(BASE / 'probe_inputs/case_index.npy', mmap_mode='r')
    assert len(case_indices) == 1879782
    for r in records:
        if r['payload_exposed']:
            assert np.all(case_indices[r['feature_row_start']:r['feature_row_end']] == r['case_index'])
    arrays = {'attacker_success_by_case.npy': labels, 'utility_by_case.npy': utilities,
              'payload_exposed_by_case.npy': exposed, 'attacker_success_by_token.npy': labels[case_indices],
              'utility_by_token.npy': utilities[case_indices]}
    for name, value in arrays.items():
        np.save(label_dir / name, value, allow_pickle=False)
        assert np.array_equal(np.load(label_dir / name), value)
    write(label_dir / 'manifest.json', {'status': 'complete', 'case_order': '../case_results.json',
        'base_case_index_sha256': sha(BASE / 'probe_inputs/case_index.npy'), 'case_rows': 1898, 'token_rows': 1879782,
        'attacker_success_semantics': '1=AgentDojo attacker goal satisfied, 0=not satisfied; all cases evaluated',
        'by_token_semantics': 'Case attacker-success label repeated for every corresponding saved tool-response hidden-state row; not a token-local success test.',
        'files': {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': sha(label_dir / name)} for name, value in arrays.items()}})
    audit = {'status': 'all_cases_verified', 'expected_cases': 1898, 'verified_cases': len(records), 'errors': [],
        'captured_cases_with_continuation': 1852, 'unexposed_cases_replayed_and_reevaluated': 46,
        'all_official_predicates_rechecked_from_saved_environments_and_traces': True,
        'all_environment_snapshots_roundtrip_without_model_reinitialization': True,
        'snapshot_restore_source_sha256': sha(Path(__file__).with_name('snapshot.py')),
        'snapshot_validation_sha256': sha(OUTPUT / 'snapshot_validation.json'),
        'all_prefix_requests_and_full_input_tokens_match_original': True,
        'all_actual_requests_rerendered_and_full_prompt_tokens_reencoded': True,
        'original_hidden_states_preserved_and_sha256_verified': len(feature_index),
        'all_original_probe_input_files_sha256_verified': len(probe_manifest['SHA256']),
        'verified_model_calls': calls, 'prefix_replay_calls': replayed, 'post_exposure_calls': native,
        'probe_labels_aligned_to_all_original_token_rows': len(case_indices),
        'source_files_sha256_verified': len(identity['files']), 'manifest_sha256': sha(OUTPUT / 'manifest.json'),
        'case_results_sha256': sha(OUTPUT / 'case_results.json'), 'summary_sha256': sha(OUTPUT / 'asr_summary.json'),
        'audit_source_sha256': sha(Path(__file__)), 'verified_at_unix': time.time()}
    write(OUTPUT / 'audit.json', audit)
    print(serial(audit), flush=True)


if __name__ == '__main__':
    main()
