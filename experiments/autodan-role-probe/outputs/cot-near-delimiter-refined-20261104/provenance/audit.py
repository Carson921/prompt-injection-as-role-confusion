"""Independently recompute exported probe scores without loading the backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--target-role', help='Optionally require this target in the saved scoring protocol')
    args = parser.parse_args()
    protocol = read(args.output / 'score_protocol.json')
    assert protocol['probe_sha256'] == sha(args.probe)
    assert protocol['dataset_sha256'] == sha(args.dataset)
    contexts = read(args.dataset)['contexts']
    expected_ids = [r['case_id'] for r in contexts]
    assert len(expected_ids) == len(set(expected_ids)) == 6
    with np.load(args.probe, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        mean, scale = archive['mean'].astype(np.float32), archive['scale'].astype(np.float32)
        w0, b0 = archive['weight::0.weight'], archive['weight::0.bias']
        w3, b3 = archive['weight::3.weight'], archive['weight::3.bias']
    roles = metadata['roles']
    assert roles == ['system', 'developer', 'user', 'cot', 'assistant', 'tool']
    assert protocol['roles'] == roles
    target_role = protocol.get('target_role', 'user')
    target_index = roles.index(target_role)
    assert protocol.get('target_role_index', target_index) == target_index
    if args.target_role is not None:
        assert args.target_role == target_role
    new_format = protocol.get('format') == 'full-context-goal-target-role-equal-task-v2'
    protocol_binding = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    erf = np.vectorize(math.erf, otypes=[np.float32])
    max_error, rows, candidates = 0.0, 0, []
    for path in sorted((args.output / 'scores').glob('*/score.json')):
        result = read(path)
        if result.get('status') != 'complete':
            continue
        assert [r['case_id'] for r in result['per_case']] == expected_ids
        assert result['protocol_sha256'] == protocol_binding
        if new_format:
            assert result['target_role'] == target_role and result['target_role_index'] == target_index
        task_means = []
        all_role_means = []
        for context, record in zip(contexts, result['per_case']):
            artifact = Path(record['token_artifact'])
            for name, digest in record['artifact_sha256'].items():
                assert sha(artifact.parent / name) == digest
            request = read(artifact.parent / 'request.json')
            assert request['actual_goal'] == context['actual_goal']
            assert record['actual_goal_sha256'] == hashlib.sha256(context['actual_goal'].encode()).hexdigest()
            assert request['payload_template'] == result['template']
            assert request['payload_text'] == result['template'].replace('{goal}', context['actual_goal'])
            with np.load(artifact, allow_pickle=False) as archive:
                features = archive['features_fp16']
                raw = archive['native_bf16_bits']
                reconstructed = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float16).reshape(-1, 1024)
                assert np.array_equal(features, reconstructed)
                assert archive['head_ids'].tolist() == metadata['heads']
                assert archive['roles'].tolist() == metadata['roles']
                if new_format:
                    assert str(archive['target_role']) == record['target_role'] == target_role
                    assert int(archive['target_role_index']) == record['target_role_index'] == target_index
                    assert record['protocol_sha256'] == protocol_binding
                offsets = archive['input_token_offsets'][archive['payload_token_indices']]
                a, b = archive['goal_char_span']
                goal = (offsets[:, 1] > a) & (offsets[:, 0] < b) & (offsets[:, 1] > offsets[:, 0])
                assert np.array_equal(goal, archive['goal_mask']) and goal.any()
                assert np.array_equal(~goal, archive['template_mask'])
                assert hashlib.sha256(archive['input_ids'].tobytes()).hexdigest() == record['input_sha256']
                x = (features.astype(np.float32) - mean) / scale
                x = x @ w0.T + b0
                x = x * np.float32(.5) * (1 + erf(x / np.float32(math.sqrt(2))))
                x = x @ w3.T + b3
                x -= x.max(1, keepdims=True)
                p = np.exp(x)
                p /= p.sum(1, keepdims=True)
                stored = archive['probabilities']
                error = float(np.max(np.abs(p - stored)))
                assert error < 5e-4, (str(artifact), error)
                max_error = max(max_error, error)
                role_means = stored[goal].mean(axis=0, dtype=np.float64)
                task_mean = float(role_means[target_index])
                assert abs(float(role_means[roles.index('user')]) - record['goal_user_mean']) < 1e-12
                if new_format:
                    assert abs(float(role_means[roles.index('cot')]) - record['goal_cot_mean']) < 1e-12
                    assert abs(task_mean - record['goal_target_mean']) < 1e-12
                else:
                    assert target_role == 'user'
                assert all(abs(float(value) - record['probabilities']['goal'][role]) < 1e-12
                           for role, value in zip(roles, role_means))
                assert int(goal.sum()) == record['goal_tokens']
                task_means.append(task_mean)
                all_role_means.append(role_means)
                rows += len(features)
        assert abs(float(np.mean(task_means)) - result['fitness']) < 1e-12
        if new_format:
            assert abs(float(np.mean(task_means)) - result['goal_target_mean']) < 1e-12
            assert all(abs(float(value) - result['goal_role_means'][role]) < 1e-12
                       for role, value in zip(roles, np.mean(all_role_means, axis=0)))
        candidates.append({'template': result['template'], 'fitness': result['fitness'],
                           'goal_role_means': dict(zip(roles, map(float, np.mean(all_role_means, axis=0)))),
                           'path': str(path)})
    assert candidates
    report = {'status': 'passed', 'independent_numpy_exact_gelu': True,
              'target_role': target_role, 'target_role_index': target_index,
              'candidates': len(candidates), 'verified_token_rows': rows,
              'max_probability_absolute_error': max_error,
              'all_task_goal_masks_verified': True, 'all_six_task_equal_means_verified': True,
              'dataset_sha256': sha(args.dataset), 'probe_sha256': sha(args.probe),
              'ranking': sorted(candidates, key=lambda r: -r['fitness'])}
    (args.output / 'independent_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'ranking'}), flush=True)


if __name__ == '__main__':
    main()
