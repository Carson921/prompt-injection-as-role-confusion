"""Phase 3 warm-start pool: turn successful swarm candidates into {goal} templates for rich evolution.

A candidate becomes a template when the scenario's attacker goal text appears verbatim in it (replaced by
{goal}); otherwise it is kept as a content example only.  Writes <output>/warm_start_pool.jsonl (records
{fitness, template, source, scenario_id, condition}) and <output>/content_examples.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True, help='swarm run dirs (conditions)')
    parser.add_argument('--dataset', type=Path, default=Path(__file__).with_name('outputs') / 'dataset-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra', nargs='*', type=Path, default=[], help='earlier warm-start pools to append (e.g. ipievo top)')
    parser.add_argument('--extra-limit', type=int, default=40)
    args = parser.parse_args()
    goals = {e['scenario_id']: e['attacker_goal'] for e in read_json(args.dataset / 'dataset.json')['scenarios']}
    args.output.mkdir(parents=True, exist_ok=True)
    templates, examples, seen = [], [], set()
    for run in args.runs:
        condition = read_json(run / 'config.json')['name']
        for path in sorted((run / 'scenarios').glob('*/evals.jsonl')):
            sid = path.parent.name
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get('success'):
                    continue
                candidate = row['candidate']
                goal = goals[sid]
                if goal in candidate:
                    template = candidate.replace(goal, '{goal}')
                    if template.count('{goal}') == 1 and template not in seen:
                        seen.add(template)
                        templates.append(dict(fitness=1.0, template=template, source='swarm', scenario_id=sid, condition=condition,
                                              queries=row.get('defender_queries_so_far')))
                else:
                    examples.append(dict(scenario_id=sid, condition=condition, candidate=candidate, goal=goal))
    extra = []
    for path in args.extra:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: -(r.get('fitness') or 0))
        extra += [dict(fitness=r.get('fitness', 0), template=r['template'], source=str(path)) for r in rows[:args.extra_limit]]
    with (args.output / 'warm_start_pool.jsonl').open('w') as stream:
        for row in templates + extra:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(args.output / 'content_examples.json', examples)
    print(json.dumps(dict(templates=len(templates), content_only=len(examples), extra=len(extra))))


if __name__ == '__main__':
    main()
