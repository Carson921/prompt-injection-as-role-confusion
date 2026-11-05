"""Pair already-captured exact-input features with the ASR in their old logs.

This proof of reuse performs no model inference and no ASR generation. It is
explicitly a 142-case matched subset, not a replacement for missing full runs.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

BASE = Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103')
OLD = Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_role_probe_gptoss120b/subdataset_100_seed20261021')
REVIEW = BASE.with_name(BASE.name + '_asr') / 'log_reuse_inspection'
OUTPUT = REVIEW / 'matched142_probe_inputs'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    comparison = read(REVIEW / 'reference100_input_comparison.json')
    selected = [r for r in comparison['cases'] if r['full_ids_equal'] and r['full_rendered_prompt_equal']]
    assert len(selected) == 142
    case_lookup = {r['case_id']: r for r in read(OLD / 'cases.json')}
    capture_index = {r['case_id']: r for r in read(BASE / 'feature_index.json')}
    protocol = read(BASE / 'manifest.json')
    slots = [protocol['head_ids_union'].index(h) for h in protocol['head_ids_final_v7_16']]
    original_matrix = np.load(BASE / 'probe_inputs/X_final_v7_16.npy', mmap_mode='r')
    blocks, ids, positions, case_indices, cases = [], [], [], [], []
    masks = {key: [] for key in ['payload_mask', 'goal_mask', 'template_mask', 'context_mask']}
    cursor = 0
    for i, row in enumerate(selected):
        assert row['source_request_sha256_valid'] and row['source_npz_sha256_valid']
        old_case = case_lookup[row['source_case']]
        source_record = OLD / old_case['source_record']['path']
        source_trace = OLD / old_case['raw_trace']['path']
        assert sha(source_record) == old_case['source_record']['sha256']
        assert sha(source_trace) == old_case['raw_trace']['sha256']
        recorded, trace = read(source_record), read(source_trace)
        assert type(recorded['security']) is bool and type(recorded['utility']) is bool
        assert row['attack_success'] == old_case['security'] == recorded['security'] == trace['security']
        assert row['utility_success'] == old_case['utility'] == recorded['utility'] == trace['utility']
        entry = capture_index[row['new_case']]
        state_path = BASE / entry['states']
        assert sha(state_path) == entry['states_sha256']
        state = load_file(str(state_path), device='cpu')
        value = state['features_union'][:, slots, :].float().numpy().reshape(-1, 1024)
        assert np.array_equal(value, original_matrix[row['row_start']:row['row_end']])
        assert not np.isin(state['token_ids'].numpy(), protocol['delimiter_ids']).any()
        blocks.append(value)
        ids.append(state['token_ids'].numpy())
        positions.append(state['positions'].numpy())
        case_indices.append(np.full(len(value), i, dtype=np.int32))
        for key in masks:
            masks[key].append(state[key].numpy())
        cases.append({'case_index': i, 'log_case_id': row['source_case'], 'feature_case_id': row['new_case'],
            'condition': row['condition'], 'attacker_success': recorded['security'], 'utility': recorded['utility'],
            'row_start': cursor, 'row_end': cursor + len(value), 'source_record': str(source_record),
            'source_record_sha256': sha(source_record), 'source_trace': str(source_trace), 'source_trace_sha256': sha(source_trace),
            'source_request': row['source_request'], 'source_npz': row['source_npz'],
            'original_states': str(state_path), 'original_states_sha256': entry['states_sha256'],
            'complete_rendered_prompt_and_full_input_ids_equal': True,
            'historical_exposure_conditioned': old_case.get('exposure_conditioned', False),
            'historical_selection': old_case.get('selection')})
        cursor += len(value)
    OUTPUT.mkdir(exist_ok=True)
    case_ids = np.concatenate(case_indices)
    labels = np.asarray([r['attacker_success'] for r in cases], dtype=np.uint8)
    utility = np.asarray([r['utility'] for r in cases], dtype=np.bool_)
    arrays = {'X_final_v7_16.npy': np.concatenate(blocks), 'case_index.npy': case_ids,
        'token_ids.npy': np.concatenate(ids), 'positions.npy': np.concatenate(positions),
        'attacker_success_by_case.npy': labels, 'attacker_success_by_token.npy': labels[case_ids],
        'utility_by_case.npy': utility, 'utility_by_token.npy': utility[case_ids],
        **{key + '.npy': np.concatenate(values) for key, values in masks.items()}}
    for name, value in arrays.items():
        np.save(OUTPUT / name, value, allow_pickle=False)
        assert np.array_equal(np.load(OUTPUT / name, mmap_mode='r'), value)
    (OUTPUT / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    (OUTPUT / 'USAGE.txt').write_text('''Verified reuse demonstration: 142 historical log cases, 27,657 whole-tool token rows.
No model/API calls or ASR generation were performed for this export.
X_final_v7_16.npy is unstandardized float32[27657,1024], exact BF16 conversion,
using the requested 16 heads. Native delimiter-token rows are excluded.
attacker_success_by_token.npy is uint8[27657], repeating the historical log's
case verdict for each token. Case-level labels and complete case mapping are
also included. This is a matched subset of the 100-pair reference experiment,
selected by complete-input equality, not by attack outcome. Historical
exposure-conditioned selections remain marked in cases.json.
Original current-run hidden states are reused only when the entire saved
rendered prompt and every input token equal the corresponding historical log.
These are current HF activations at the logged input, not a claim of bitwise
reproduction of the historical inference backend's internal activations.
Missing/different inputs and the rest of the full benchmark are not zero-filled.

X = np.load('X_final_v7_16.npy', mmap_mode='r')
y = np.load('attacker_success_by_token.npy')
assert len(X) == len(y)
''')
    report = {'status': 'partial_log_reuse_verified', 'cases': len(cases), 'rows': cursor,
        'condition_case_counts': dict(Counter(r['condition'] for r in cases)),
        'full_benchmark_complete': False, 'model_or_api_calls': 0, 'asr_rerun': False,
        'head_ids': protocol['head_ids_final_v7_16'], 'matrix_shape': list(arrays['X_final_v7_16.npy'].shape),
        'matrix_dtype': 'float32', 'standardization': False,
        'all_original_record_trace_states_hashes_verified': True, 'all_exported_values_readback_verified': True,
        'source_comparison_sha256': sha(REVIEW / 'reference100_input_comparison.json'),
        'export_source_sha256': sha(Path(__file__)),
        'sha256': {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sha256'}), flush=True)


if __name__ == '__main__':
    main()
