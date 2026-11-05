"""CPU validation of actual Scorer target selection using saved native BF16 captures.

This deliberately reuses historical features only in a separate preflight folder.
The formal search must perform new backbone forwards in its own output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import scorer as module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-output', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=module.DEFAULT_MODEL)
    parser.add_argument('--probe', type=Path, default=module.DEFAULT_PROBE)
    args = parser.parse_args()
    previous = args.previous_output.resolve()
    dataset = previous / 'dataset.json'
    old = json.loads((previous / 'search_protocol.json').read_text())
    templates = {key: old['config']['baselines'][key] for key in ('plain', 'remember')}
    templates['previous_optimized'] = json.loads((previous / 'winner.json').read_text())['template']
    captured, expected = {}, {}
    for label, template in templates.items():
        record = json.loads((previous / 'scores' / module.sha(template)[:24] / 'score.json').read_text())
        role_values = []
        for case in record['per_case']:
            with np.load(case['token_artifact'], allow_pickle=False) as data:
                assert data['roles'].tolist() == list(module.ROLES)
                captured[case['input_sha256']] = torch.from_numpy(data['native_bf16_bits'].copy()).view(torch.bfloat16)
                role_values.append(data['probabilities'][data['goal_mask']].mean(0, dtype=np.float64))
        expected[label] = np.mean(role_values, axis=0)
    deps = module.dependencies()
    collect, _, attention, _ = deps
    configuration = json.loads((args.model / 'config.json').read_text())
    stub_model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=configuration['max_position_embeddings']))

    def replay(model, heads, ids, positions):
        key = module.sha(np.asarray(ids, dtype=np.int64).tobytes())
        assert key in captured, 'Reconstructed native input differs from historical capture'
        native = captured[key]
        assert native.shape[0] == len(positions)
        return native, None

    observations = {}
    with patch.object(module, 'dependencies', return_value=deps), \
         patch.object(collect, 'load_model', return_value=(stub_model, {})), \
         patch.object(collect, 'forward_capture', side_effect=replay), \
         patch.object(attention, 'install_tiled_attention', return_value=None):
        for target in ('user', 'cot'):
            instance = module.Scorer(dataset, args.output / target, model_path=args.model,
                                     probe_path=args.probe, target_role=target)
            index = module.ROLES.index(target)
            observations[target] = {}
            for label, template in templates.items():
                result = instance.score(template)
                assert result['target_role'] == target and result['target_role_index'] == index
                assert abs(result['fitness'] - expected[label][index]) < 2e-6
                assert result['fitness'] == result['goal_target_mean']
                assert abs(result['goal_role_means']['user'] - expected[label][2]) < 2e-6
                assert abs(result['goal_role_means']['cot'] - expected[label][3]) < 2e-6
                for case in result['per_case']:
                    assert case['goal_user_mean'] == case['probabilities']['goal']['user']
                    assert case['goal_cot_mean'] == case['probabilities']['goal']['cot']
                    assert case['goal_target_mean'] == case['probabilities']['goal'][target]
                observations[target][label] = result['fitness']
            instance.closed = True  # The stub installed no attention function or model allocation.
        try:
            module.Scorer(dataset, args.output / 'user', model_path=args.model,
                          probe_path=args.probe, target_role='cot')
        except ValueError as error:
            assert 'different frozen protocol' in str(error)
        else:
            raise AssertionError('A user-role cache was accepted as cot')
    assert observations['cot'] != observations['user']
    report = {'status': 'passed', 'no_backbone_weights_loaded': True,
              'historical_feature_replay_only': True, 'new_search_cache_written': False,
              'target_role_cot_index': module.ROLES.index('cot'),
              'actual_scorer_constructor_prepare_and_aggregation_exercised': True,
              'historical_complete_input_ids_matched': True,
              'real_user_and_cot_fields_preserved': True,
              'cross_role_protocol_cache_rejected': True,
              'observed_means': observations,
              'scorer_sha256': module.sha(Path(module.__file__)),
              'validation_source_sha256': module.sha(Path(__file__))}
    module.write(args.output / 'validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
