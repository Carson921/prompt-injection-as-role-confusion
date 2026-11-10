#!/usr/bin/env python3
"""Pick defensive prefixes covering P(tool) bands from the optimizer ledger.

Selection uses only probe scores (no ASR feedback): in every band [lo, hi) take
the highest-P(tool) prefix whose token bag differs from already chosen ones
(Jaccard < 0.5).  The no-defense attack and the hand-written notice are added
as reference conditions.  Output: a conditions manifest for evaluate_defense_asr.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ATTACK, NAIVE_DEFENSES, SLOT  # noqa: E402


def bag(text):
    return set(re.findall(r'\w+|[^\w\s]', text.lower()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--edges', required=True, help='Comma-separated band edges, ascending, e.g. 0.2,0.3,0.4,0.5')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'scores.jsonl').open()]
    rows = [r for r in rows if r.get('status') == 'complete']
    edges = [float(v) for v in args.edges.split(',')]
    bands = list(zip(edges[:-1], edges[1:]))
    baseline = json.loads((args.run / 'baseline.json').read_text())
    naive = {r['template'][:-len(SLOT)]: r for r in rows if r['template'][:-len(SLOT)] in NAIVE_DEFENSES}
    chosen, conditions = [], [dict(condition='no_defense', defense='', attack=ATTACK,
                                   probe=dict(body_tool=baseline['fitness'], body_user=baseline['body_user_mean'],
                                              suite_means=baseline['suite_means']))]
    for defense, row in naive.items():
        conditions.append(dict(condition='naive_notice', defense=defense, attack=ATTACK,
                               probe=dict(body_tool=row['fitness'], body_user=row['body_user_mean'], suite_means=row['suite_means'],
                                          wrapper_tokens=row['wrapper_tokens'])))
        chosen.append(bag(defense))
    for lo, hi in sorted(bands, reverse=True):
        pool = sorted((r for r in rows if lo <= r['fitness'] < hi or (hi >= 1 and r['fitness'] == 1)),
                      key=lambda r: -r['fitness'])
        pick = next((r for r in pool if r['template'][:-len(SLOT)] not in naive
                     and all(jaccard(bag(r['template']), c) < 0.5 for c in chosen)), None)
        pick = pick or next((r for r in pool if r['template'][:-len(SLOT)] not in naive), None)
        if pick is None:
            print(json.dumps(dict(band=[lo, hi], picked=None)))
            continue
        chosen.append(bag(pick['template']))
        conditions.append(dict(condition=f'defense_{lo:.2f}_{hi:.2f}', defense=pick['template'][:-len(SLOT)], attack=ATTACK,
                               probe=dict(band=[lo, hi], body_tool=pick['fitness'], body_user=pick['body_user_mean'],
                                          suite_means=pick['suite_means'], wrapper_tokens=pick['wrapper_tokens'],
                                          evaluation_index=pick['evaluation_index'], source=pick.get('source'),
                                          operator=pick.get('operator'), score_artifact=pick.get('score_artifact'))))
        print(json.dumps(dict(band=[lo, hi], body_tool=round(pick['fitness'], 4), tokens=pick['wrapper_tokens'],
                              defense=pick['template'][:100]), ensure_ascii=False))
    args.output.write_text(json.dumps(conditions, indent=2, ensure_ascii=False) + '\n')
    print('wrote', args.output, len(conditions), 'conditions')


if __name__ == '__main__':
    main()
