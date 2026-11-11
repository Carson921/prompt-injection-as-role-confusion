"""Aggregate probe/<cond>/<case>.json (written by ipi-arena-richevo/probe_score.py) into probe_summary.json.

probe_score.py's own summary assumes the IPI Arena categories; here categories are the task-domain categories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import read_json, write_json

ROLES = ('system', 'developer', 'user', 'cot', 'assistant', 'tool')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    conditions = read_json(args.output / 'conditions.json')
    protocol = read_json(args.output / 'protocol.json')
    summary = {}
    for condition in conditions:
        rows = [read_json(args.output / 'probe' / condition / f"{r['case_id']}.json") for r in protocol['rows'] if r['condition'] == condition]
        rows = [r for r in rows if r['status'] == 'complete']
        categories = sorted({r['category'] for r in rows})
        summary[condition] = dict(
            p_user_mean=float(np.mean([r['goal_user_mean'] for r in rows])), p_user_mean_all=float(np.mean([r['goal_user_mean'] for r in rows])),
            goal_role_means={role: float(np.mean([r['region_means']['goal'][role] for r in rows])) for role in ROLES},
            per_category={c: float(np.mean([r['goal_user_mean'] for r in rows if r['category'] == c])) for c in categories},
            cases=len(rows), p_user_agentdojo=conditions[condition].get('p_user_agentdojo'))
    write_json(args.output / 'probe_summary.json', summary)
    print({k: round(v['p_user_mean'], 4) for k, v in summary.items()})


if __name__ == '__main__':
    main()
