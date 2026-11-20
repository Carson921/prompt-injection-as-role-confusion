"""Score named templates on the fixed DeepSeek AgentDojo contexts (GPU; native probe runtime)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ds_common import read_json, write_json
from ds_scorer import Scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--conditions-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    conditions = read_json(args.conditions_json)
    scorer = Scorer(args.dataset, args.output)
    results = {}
    for name, spec in conditions.items():
        scorer.allow_special = bool(spec.get('allow_special'))
        result = scorer.score(spec['template'])
        results[name] = dict(p_user=result['fitness'], suite_means=result['suite_means'], goal_role_means=result['goal_role_means'],
                             per_case={r['case_id']: r['goal_user_mean'] for r in result['per_case']},
                             wrapper_role_means={r['case_id']: r['probabilities']['wrapper'] for r in result['per_case']},
                             seconds=result['elapsed_seconds'], template=spec['template'])
        print(json.dumps(dict(condition=name, p_user=round(result['fitness'], 4),
                              suites={k: round(v, 3) for k, v in result['suite_means'].items()},
                              roles={k: round(v, 3) for k, v in result['goal_role_means'].items()},
                              seconds=round(result['elapsed_seconds'], 1))), flush=True)
    write_json(args.output / 'condition_scores.json', results)
    scorer.close()


if __name__ == '__main__':
    main()
