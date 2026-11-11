"""Pair each domain's validated scenario with an attacker goal from a domain in a different category.

Writes <output>/dataset.json with one entry per task domain: tools of both domains, the golden
trajectory, the attacker goal and the target call.  Pairing is a seeded random assignment
subject to category disjointness; attacker goals are drawn without replacement where possible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from common import DOMAIN_CATEGORY, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    tools = read_json(args.output / 'tools.json')
    scenarios = read_json(args.output / 'scenarios.json')
    rng = random.Random(args.seed)
    task_domains = sorted(d for d, v in scenarios.items() if v['scenario'])
    goal_pool = {d: list(v['attacker_goals']) for d, v in scenarios.items() if v['attacker_goals']}
    for d in goal_pool:
        rng.shuffle(goal_pool[d])
    entries = []
    order = list(task_domains)
    rng.shuffle(order)
    for task in order:
        candidates = [d for d, pool in goal_pool.items() if pool and DOMAIN_CATEGORY[d] != DOMAIN_CATEGORY[task]]
        if not candidates:
            continue
        # prefer target domains whose goals have been used least
        candidates.sort(key=lambda d: (-len(goal_pool[d]), rng.random()))
        target = candidates[0]
        goal = goal_pool[target].pop()
        entries.append(dict(scenario_id=f'{task}__{target}', task_domain=task, target_domain=target,
                            task_category=DOMAIN_CATEGORY[task], target_category=DOMAIN_CATEGORY[target],
                            task_tools=tools[task]['tools'], target_tools=tools[target]['tools'],
                            user_query=scenarios[task]['scenario']['user_query'], golden_turns=scenarios[task]['scenario']['turns'],
                            final_response=scenarios[task]['scenario']['final_response'],
                            attacker_goal=goal['instruction'], target_call=goal['target_call']))
    entries.sort(key=lambda e: e['scenario_id'])
    write_json(args.output / 'dataset.json', dict(schema='benign-goal-synthetic-v1', seed=args.seed, scenarios=entries))
    print(json.dumps(dict(scenarios=len(entries), task_domains=len(task_domains),
                          turns={n: sum(len(e['golden_turns']) == n for e in entries) for n in (2, 3, 4)})))


if __name__ == '__main__':
    main()
