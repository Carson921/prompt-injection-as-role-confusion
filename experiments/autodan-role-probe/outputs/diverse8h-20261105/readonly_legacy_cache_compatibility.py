"""CPU-only, read-only compatibility inventory; never reuses an old fitness."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

B = Path(__file__).resolve().parent
HERE = B.parent.parent
sys.path.insert(0, str(HERE))
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from precise_scorer import PreciseScorer
from scorer import dependencies
from strict_header_policy import strict_header_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    from transformers import AutoTokenizer
    output = B / 'audit_readonly_legacy_cache_compatibility'
    output.mkdir(exist_ok=True)
    dataset_path = B / 'dataset.json'
    dataset = read(dataset_path)
    ref_path = B / 'probe/native/score_protocol.json'
    ref = read(ref_path)
    model = Path('/tmp/wz55191-agentdojo-first-tool-model')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    library = BroadDelimiterLibrary(model)
    scorer = object.__new__(PreciseScorer)
    scorer.protocol = ref
    scorer.tokenizer = tokenizer
    scorer.spans = dependencies()[1]
    scorer.native_special_id_floor = 199998
    scorer.native_special_spellings = ref['payload_delimiter_policy']['reject_real_special_spellings']
    config = read(model / 'config.json')
    scorer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=config['max_position_embeddings']))
    seen_scored = {json.loads(line)['raw_sha256'] for line in
                   (B / 'probe_short128_supplement/score_records.jsonl').read_text().splitlines() if line.strip()}
    same_keys = ['dataset_sha256', 'probe_sha256', 'roles', 'head_ids', 'feature_position',
                 'feature_cast', 'probe_activation', 'aggregation', 'query_tile', 'truncation',
                 'batch_size', 'prompt_date', 'model_binding', 'staging_proof_sha256', 'versions']
    reports = []
    banks = []
    for name in ['cot-near-delimiter-20261104', 'cot-near-delimiter-refined-20261104',
                 'near-delimiter-20261104', 'pilot-20261104']:
        root = B.parent / name
        protocol_path = root / 'score_protocol.json'
        protocol = read(protocol_path)
        protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        protocol_same = all(protocol.get(key) == ref.get(key) for key in same_keys)
        assert sha(root / 'dataset.json') == sha(dataset_path) == ref['dataset_sha256']
        bank_rows = []
        for path in sorted((root / 'scores').glob('*/score.json')):
            score = read(path)
            text = score['template']
            raw = hashlib.sha256(text.encode()).hexdigest()
            admission = validate_diverse_template(text, tokenizer, library, maximum=128)
            strict = strict_header_metrics(text)
            row = {'source_score_path': str(path.resolve()), 'source_score_sha256': sha(path),
                   'raw_sha256': raw, 'bank': name,
                   'old_target_role': protocol.get('target_role', 'user'),
                   'old_fitness_not_reused': True,
                   'numerical_feature_protocol_matches': protocol_same,
                   'strict_short128_allowed': admission['allowed'] and strict['allowed'],
                   'wrapper_tokens': admission['metrics'].get('wrapper_tokens'),
                   'already_in_1586_complete': raw in seen_scored,
                   'input_goal_feature_exact': False}
            try:
                assert protocol_same, 'Numerical feature protocol differs'
                assert score['protocol_sha256'] == protocol_digest
                assert score['status'] == 'complete' and score['expected_cases'] == score['completed_cases'] == 6
                assert len(score['per_case']) == 6
                case_rows = []
                for context, case in zip(dataset['contexts'], score['per_case']):
                    assert case['status'] == 'complete' and case['case_id'] == context['case_id']
                    assert case['protocol_sha256'] == protocol_digest
                    assert case['actual_goal_sha256'] == hashlib.sha256(context['actual_goal'].encode()).hexdigest()
                    request, prompt, ids, offsets, positions, goal, location, input_sha = scorer.prepare(context, text)
                    token_path = Path(case['token_artifact'])
                    for filename, digest in case['artifact_sha256'].items():
                        assert sha(token_path.parent / filename) == digest, 'Native artifact SHA differs'
                    saved = read(token_path.parent / 'request.json')
                    assert saved['rendered_prompt'] == prompt
                    assert all(saved[key] == value for key, value in request.items())
                    assert saved['actual_goal'] == context['actual_goal']
                    assert saved['input_sha256'] == input_sha == case['input_sha256']
                    with np.load(token_path, allow_pickle=False) as arrays:
                        for key, current in [('input_ids', ids), ('input_token_offsets', offsets),
                                             ('payload_token_indices', positions), ('goal_mask', goal),
                                             ('template_mask', ~goal), ('head_ids', ref['head_ids']),
                                             ('roles', ref['roles']),
                                             ('payload_char_spans', [r['payload_char_span'] for r in location['occurrences']]),
                                             ('goal_char_spans', [r['goal_char_span'] for r in location['occurrences']])]:
                            assert np.array_equal(arrays[key], current), key + ' differs from current complete context'
                        bits, features = arrays['native_bf16_bits'], arrays['features_fp16']
                        assert bits.shape == (len(positions), 16, 64) and bits.dtype == np.uint16
                        assert features.shape == (len(positions), 1024) and features.dtype == np.float16
                        recovered = (bits.astype(np.uint32) << 16).view(np.float32).reshape(len(positions), 1024).astype(np.float16)
                        assert np.isfinite(features).all() and np.array_equal(features, recovered)
                        assert arrays['probabilities'].shape == (len(positions), 6)
                    case_rows.append({'case_id': case['case_id'], 'input_tokens': len(ids),
                                      'complete_goal_tokens': int(goal.sum()),
                                      'payload_copies': len(location['occurrences']),
                                      'tokens_npz_sha256': sha(token_path),
                                      'request_sha256': sha(token_path.parent / 'request.json')})
                row.update(input_goal_feature_exact=True, per_case=case_rows)
            except Exception as error:
                row['incompatibility'] = type(error).__name__ + ': ' + str(error)
            reports.append(row)
            bank_rows.append(row)
        bank = {'bank': name, 'score_records': len(bank_rows), 'protocol_path': str(protocol_path.resolve()),
                'protocol_sha256': sha(protocol_path),
                'old_target_role': protocol.get('target_role', 'user'),
                'numerical_feature_protocol_matches': protocol_same,
                'protocol_different_keys': sorted(key for key in set(ref) | set(protocol) if ref.get(key) != protocol.get(key)),
                'exact_input_goal_features': sum(r['input_goal_feature_exact'] for r in bank_rows),
                'strict_short128_exact': sum(r['input_goal_feature_exact'] and r['strict_short128_allowed'] for r in bank_rows)}
        banks.append(bank)
        print(json.dumps(bank), flush=True)
    qualified = [row for row in reports if row['input_goal_feature_exact'] and row['strict_short128_allowed']]
    new = [row for row in qualified if not row['already_in_1586_complete']]
    report = {'schema': 'readonly-legacy-native-feature-cache-compatibility-v1', 'status': 'inventory_complete',
              'dataset_sha256': sha(dataset_path), 'probe_checkpoint_sha256': ref['probe_sha256'],
              'reference_protocol_path': str(ref_path.resolve()), 'reference_protocol_sha256': sha(ref_path),
              'banks': banks, 'source_score_records': len(reports),
              'exact_input_goal_feature_records': sum(row['input_goal_feature_exact'] for row in reports),
              'strict_short128_exact_records': len(qualified),
              'strict_short128_exact_unique_templates': len({r['raw_sha256'] for r in qualified}),
              'not_in_1586_strict_short128_exact_records': len(new),
              'not_in_1586_strict_short128_exact_unique_templates': len({r['raw_sha256'] for r in new}),
              'records': reports, 'old_target_fitness_reused': False,
              'checkpoint_probability_recomputations': 0, 'model_forward_calls': 0,
              'API_calls': 0, 'candidate_generation': False, 'source_artifacts_modified': False,
              'audit_source_sha256': sha(__file__),
              'note': 'Compatibility only. Old target-role fitness is not a current P(user) score; CPU checkpoint recomputation is still required.'}
    (output / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in report if key not in ('banks', 'records')}), flush=True)


if __name__ == '__main__':
    main()
