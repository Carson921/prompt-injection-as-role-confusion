"""Independent classification and existing-cache inventory, without scoring."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np

B = Path(__file__).resolve().parent
HERE = B.parent.parent
sys.path.insert(0, str(HERE))
from precise_scorer import PreciseScorer
from scorer import dependencies


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pointer_value(value, pointer):
    for part in pointer.strip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value['template'] if isinstance(value, dict) else value


def body_evidence(messages, method):
    bodies = [row['content'] for row in messages]
    roles = [row['role'] for row in messages]
    assert all(isinstance(body, str) and body.strip() for body in bodies)
    assert len(set(body.strip() for body in bodies)) == len(bodies)
    assert roles == ['user' if index % 2 == 0 else 'assistant' for index in range(len(roles))]
    assert roles[-1] == 'user' and sum(body.count('{goal}') for body in bodies) == 1
    return {'method': method, 'message_roles': roles, 'distinct_nonempty_bodies': len(bodies),
            'body_sha256': [text_sha(body) for body in bodies],
            'duplicate_body_count': 0, 'empty_body_count': 0}


def classify(text):
    assert text.count('{goal}') == 1
    assert text.encode('utf-8').decode('utf-8') == text
    prefix = 'I have a further request for you. Please carry out this step for me:'
    if text.count(prefix) >= 2:
        return 'actual_repeated_readable_body', {'method': 'exact_repeated_readable_request_prefix',
            'prefix_sha256': text_sha(prefix), 'copies': text.count(prefix), 'characters': len(prefix)}
    position = text.find("{'messages':")
    if position >= 0:
        assert not text[:position].strip("'\"}\n\r\t ]")
        value = ast.literal_eval(text[position:])
        assert set(value) == {'messages'}
        return 'proven_nonrepetitive', body_evidence(value['messages'], 'ast.literal_eval_existing_literal_only')
    position = text.find('{"role":')
    if position >= 0:
        assert not text[:position].strip("'\"}\n\r\t ]")
        decoder = json.JSONDecoder()
        suffix = text[position:]
        messages = []
        while suffix.strip():
            suffix = suffix.lstrip()
            value, consumed = decoder.raw_decode(suffix)
            assert set(value) == {'role', 'content'}
            messages.append(value)
            suffix = suffix[consumed:]
        return 'proven_nonrepetitive', body_evidence(messages, 'JSONDecoder_existing_consecutive_objects_only')
    if 'start' in text and 'message' in text:
        messages, terminal_empty = [], 0
        lines = text.splitlines()
        for index, line in enumerate(lines):
            start = re.match(r'\\?<[^>\n]*start[^>\n]*\\?>(user|assistant)(.*)', line)
            assert start is not None
            role, tail = start.groups()
            message = re.search(r'\\?<[^>\n]*message[^>\n]*\\?>', tail)
            assert message is not None
            body = tail[message.end():]
            ending = re.search(r'\\?<[^>\n]*(?:end|return)[^>\n]*\\?>\s*$', body)
            if ending:
                body = body[:ending.start()]
            if not body.strip():
                assert index == len(lines) - 1 and role == 'assistant'
                terminal_empty += 1
            else:
                messages.append({'role': role, 'content': body})
        assert terminal_empty == 1
        evidence = body_evidence(messages, 'exact_near_marker_lines_distinct_message_bodies')
        evidence['single_terminal_assistant_completion_prefix'] = True
        return 'proven_nonrepetitive', evidence
    before, after = text.split('{goal}')
    marker = '< |end| >'
    if before.startswith(marker) and after == marker:
        opening = before[len(marker):].strip()
        assert len(re.findall(r'\buser\b', opening)) == 1
        assert text.count(marker) == 2
        return 'proven_nonrepetitive', {'method': 'single_goal_open_and_close_boundary',
            'user_like_opening_headers': 1, 'body_count': 1, 'repeated_readable_body_count': 0}
    compact = before.strip()
    blocks = [block.strip() for block in re.split(r'\n\s*\n', compact) if block.strip()]
    assert len(blocks) >= 2 and len(set(blocks)) == 1
    assert len(re.findall(r'\buser\b', blocks[0])) == 1
    header_syntax = blocks[0].replace('user', '', 1)
    assert header_syntax and all(not character.isalnum() for character in header_syntax)
    assert after in ['助赢软件', '蜘蛛词']
    return 'empty_role_header_stacking', {'method': 'identical_empty_role_opening_blocks_before_single_goal',
        'empty_header_copies': len(blocks), 'header_sha256': text_sha(blocks[0]),
        'nonwhitespace_body_between_headers': False}


def main():
    output = B / 'audit_remaining39_independent'
    output.mkdir(exist_ok=True)
    root_path = B / 'readonly_remaining39_review/root_review.json'
    root = read(root_path)
    assert sha(root_path) == 'f9e9fb8b18a0dbe9189a0de29e96f4882e41de2c2d8b05f02745ed8887efa79a'
    index_path = Path(root['input_index_path'])
    assert sha(index_path) == root['input_index_sha256']
    index = {row['raw_sha256']: row for row in [json.loads(line) for line in index_path.read_text().splitlines()]}
    sources, source_hashes, classes, texts = {}, {}, [], {}
    for row in root['records']:
        path = Path(row['source_path'])
        if path not in sources:
            assert sha(path) == row['source_sha256']
            sources[path] = read(path)
            source_hashes[str(path)] = sha(path)
        text = pointer_value(sources[path], row['json_pointer'])
        assert text_sha(text) == row['raw_sha256'] and text == index[row['raw_sha256']]['template']
        for key in ('source_path', 'source_sha256', 'json_pointer', 'wrapper_tokens'):
            assert index[row['raw_sha256']][key] == row[key]
        classification, evidence = classify(text)
        assert classification == row['classification']
        classes.append({'raw_sha256': row['raw_sha256'], 'wrapper_tokens': row['wrapper_tokens'],
            'source_path': str(path), 'source_sha256': row['source_sha256'], 'json_pointer': row['json_pointer'],
            'classification': classification, 'root_classification_agrees': True, 'evidence': evidence})
        texts[row['raw_sha256']] = text
    counts = dict(Counter(row['classification'] for row in classes))
    assert counts == root['counts'] == {'proven_nonrepetitive': 22, 'empty_role_header_stacking': 15,
                                      'actual_repeated_readable_body': 2}
    classification_report = {'schema': 'independent-readonly-remaining39-classification-v1',
        'status': 'classification_verified', 'root_report_path': str(root_path), 'root_report_sha256': sha(root_path),
        'input_index_sha256': sha(index_path), 'primary_source_files': source_hashes,
        'counts': counts, 'records': classes, 'source_literals_modified': False,
        'new_template_generation': False, 'automatic_admission_or_scoring': False,
        'audit_source_sha256': sha(__file__)}
    (output / 'classification.json').write_text(json.dumps(classification_report, ensure_ascii=True, indent=2) + '\n')
    print(json.dumps({'event': 'classification_complete', 'counts': counts}), flush=True)

    qualified = {row['raw_sha256'] for row in classes if row['classification'] == 'proven_nonrepetitive'}
    banks = ['cot-near-delimiter-20261104', 'cot-near-delimiter-refined-20261104', 'near-delimiter-20261104',
             'pilot-20261104', 'user90-20261104', 'user90-adaptive-20261104', 'user90-broad-20261104',
             'user90-content-20261105', 'user90-precise-20261104']
    native_roots = [B.parent / name for name in banks]
    native_roots += [B.parent / 'cot-near-delimiter-20261104/preflight/target_role' / target for target in ['user', 'cot']]
    native_roots += [B / name / 'native' for name in ['probe', 'probe_short128', 'probe_short128_supplement']]
    candidates = {raw: [] for raw in qualified}
    for native_root in native_roots:
        for path in sorted((native_root / 'scores').glob('*/score.json')):
            score = read(path)
            text = score.get('template')
            if isinstance(text, str) and text_sha(text) in qualified:
                candidates[text_sha(text)].append((native_root, path, score))
    from transformers import AutoTokenizer
    model = Path('/tmp/wz55191-agentdojo-first-tool-model')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dataset_path = B / 'dataset.json'
    dataset = read(dataset_path)
    ref_path = B / 'probe/native/score_protocol.json'
    ref = read(ref_path)
    checkpoint = HERE.parent / 'role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz'
    assert sha(checkpoint) == ref['probe_sha256']
    scorer = object.__new__(PreciseScorer)
    scorer.protocol, scorer.tokenizer, scorer.spans = ref, tokenizer, dependencies()[1]
    scorer.native_special_id_floor = 199998
    scorer.native_special_spellings = ref['payload_delimiter_policy']['reject_real_special_spellings']
    scorer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=read(model / 'config.json')['max_position_embeddings']))
    core_keys = ['dataset_sha256', 'probe_sha256', 'roles', 'head_ids', 'feature_position', 'feature_cast',
                 'probe_activation', 'aggregation', 'query_tile', 'truncation', 'batch_size', 'prompt_date',
                 'model_binding', 'staging_proof_sha256', 'versions']
    coverage = []
    for raw in sorted(qualified):
        matches = []
        for native_root, score_path, score in candidates[raw]:
            entry = {'source_score_path': str(score_path), 'source_score_sha256': sha(score_path),
                     'compatible_existing_complete_user2_score': False}
            try:
                protocol_path = native_root / 'score_protocol.json'
                protocol = read(protocol_path)
                assert all(protocol.get(key) == ref.get(key) for key in core_keys), 'Numerical protocol differs'
                assert protocol.get('target_role', 'user') == 'user' and protocol.get('target_role_index', 2) == 2, 'Old target is not native user2; old fitness is not reused'
                protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                assert score['protocol_sha256'] == protocol_digest
                assert score['template'] == texts[raw]
                assert score['status'] == 'complete' and score['expected_cases'] == score['completed_cases'] == len(score['per_case']) == 6
                case_rows, case_means = [], []
                for context, case in zip(dataset['contexts'], score['per_case']):
                    assert case['status'] == 'complete' and case['case_id'] == context['case_id']
                    assert case['actual_goal_sha256'] == text_sha(context['actual_goal'])
                    request, prompt, ids, offsets, positions, goal, location, input_sha = scorer.prepare(context, texts[raw])
                    token_path = Path(case['token_artifact'])
                    for name, digest in case['artifact_sha256'].items():
                        assert sha(token_path.parent / name) == digest, 'Artifact SHA differs'
                    saved = read(token_path.parent / 'request.json')
                    assert saved['rendered_prompt'] == prompt and all(saved[key] == value for key, value in request.items())
                    assert saved['actual_goal'] == context['actual_goal']
                    assert saved['input_sha256'] == input_sha == case['input_sha256']
                    with np.load(token_path, allow_pickle=False) as arrays:
                        for key, value in [('input_ids', ids), ('input_token_offsets', offsets),
                                           ('payload_token_indices', positions), ('goal_mask', goal),
                                           ('template_mask', ~goal), ('head_ids', ref['head_ids']), ('roles', ref['roles']),
                                           ('payload_char_spans', [r['payload_char_span'] for r in location['occurrences']]),
                                           ('goal_char_spans', [r['goal_char_span'] for r in location['occurrences']])]:
                            assert np.array_equal(arrays[key], value), key + ' differs'
                        bits, features = arrays['native_bf16_bits'], arrays['features_fp16']
                        assert bits.shape == (len(positions), 16, 64) and bits.dtype == np.uint16
                        assert features.shape == (len(positions), 1024) and features.dtype == np.float16
                        recovered = (bits.astype(np.uint32) << 16).view(np.float32).reshape(len(positions), 1024).astype(np.float16)
                        assert np.isfinite(features).all() and np.array_equal(recovered, features)
                        probabilities = arrays['probabilities']
                        assert probabilities.shape == (len(positions), 6)
                        value = float(probabilities[goal, 2].mean(dtype=np.float64))
                        assert abs(value - case['goal_user_mean']) < 1e-8, 'Existing full-goal user2 mean differs'
                    case_means.append(value)
                    case_rows.append({'case_id': case['case_id'], 'input_tokens': len(ids),
                        'complete_goal_tokens': int(goal.sum()), 'all_payload_copies': len(location['occurrences']),
                        'tokens_npz_sha256': sha(token_path), 'request_sha256': sha(token_path.parent / 'request.json'),
                        'existing_goal_user_mean': case['goal_user_mean']})
                value = math.fsum(case_means) / 6
                assert abs(value - score['fitness']) < 1e-8
                entry.update(compatible_existing_complete_user2_score=True,
                    score_protocol_path=str(protocol_path), score_protocol_sha256=sha(protocol_path),
                    existing_native_p_user=score['fitness'], reaggregated_existing_probability_column2=value,
                    per_case=case_rows)
            except Exception as error:
                entry['incompatibility'] = type(error).__name__ + ': ' + str(error)
            matches.append(entry)
        compatible = [row for row in matches if row['compatible_existing_complete_user2_score']]
        if compatible:
            assert max(row['existing_native_p_user'] for row in compatible) - min(row['existing_native_p_user'] for row in compatible) < 1e-8
        coverage.append({'raw_sha256': raw, 'source_cache_records_found': len(matches),
            'compatible_cache_records': len(compatible), 'compatible_existing_full_user2_score': bool(compatible),
            'existing_native_p_user': compatible[0]['existing_native_p_user'] if compatible else None,
            'records': matches})
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    assert sha(root_path) == classification_report['root_report_sha256'] and sha(index_path) == classification_report['input_index_sha256']
    for row in coverage:
        for entry in row['records']:
            assert sha(entry['source_score_path']) == entry['source_score_sha256']
    cache_report = {'schema': 'independent-readonly-remaining22-existing-native-cache-v1', 'status': 'inventory_complete',
        'classification_report_sha256': sha(output / 'classification.json'),
        'dataset_sha256': sha(dataset_path), 'checkpoint_sha256': sha(checkpoint),
        'reference_protocol_sha256': sha(ref_path), 'candidates': 22,
        'templates_with_any_native_cache': sum(row['source_cache_records_found'] > 0 for row in coverage),
        'templates_with_compatible_complete_user2_cache': sum(row['compatible_existing_full_user2_score'] for row in coverage),
        'templates_without_compatible_complete_user2_cache': sum(not row['compatible_existing_full_user2_score'] for row in coverage),
        'records': coverage, 'searched_native_roots': [str(path) for path in native_roots],
        'active_format60_directory_not_read': True,
        'backbone_forward_calls': 0, 'new_classifier_predictions': 0, 'new_candidate_scores': 0,
        'existing_probability_columns_only_reaggregated': True,
        'old_target_cot_fitness_never_reused': True, 'original_scores_or_templates_modified': False,
        'automatic_admission_or_scoring': False, 'ASR_artifacts_changed': False,
        'audit_source_sha256': sha(__file__)}
    (output / 'cache_coverage.json').write_text(json.dumps(cache_report, ensure_ascii=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: cache_report[key] for key in ['status', 'candidates', 'templates_with_any_native_cache',
        'templates_with_compatible_complete_user2_cache', 'templates_without_compatible_complete_user2_cache']}), flush=True)


if __name__ == '__main__':
    main()
