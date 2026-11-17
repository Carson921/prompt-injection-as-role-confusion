"""Compare swarm runs (conditions): ARR@k curves, queries to first success, probe statistics, cost; Chinese report + figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import read_json, write_json


def load_run(path: Path) -> dict:
    config = read_json(path / 'config.json')
    results = {p.parent.name: read_json(p) for p in sorted((path / 'scenarios').glob('*/result.json'))}
    evals = {}
    for sid in results:
        lines = (path / 'scenarios' / sid / 'evals.jsonl').read_text().splitlines()
        evals[sid] = [json.loads(l) for l in lines if l.strip()]
    return dict(name=config['name'], config=config, results=results, evals=evals)


def arr_curve(results: dict, ks) -> list[float]:
    n = len(results)
    return [sum(1 for r in results.values() if not (r['queries_to_success'] is not None and r['queries_to_success'] <= k)) / n for k in ks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-k', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in args.runs]
    common = set.intersection(*(set(r['results']) for r in runs))
    ks = list(range(1, args.max_k + 1))
    rows = []
    for run in runs:
        res = {k: v for k, v in run['results'].items() if k in common}
        curve = arr_curve(res, ks)
        qts = [r['queries_to_success'] for r in res.values() if r['queries_to_success'] is not None]
        ev = [e for sid in res for e in run['evals'][sid]]
        probe = [e['probe']['goal_user_mean'] for e in ev if e.get('probe')]
        gated = sum(1 for e in ev if e.get('defender_queried') is False)
        rows.append(dict(name=run['name'], scenarios=len(res), successes=sum(r['success'] for r in res.values()),
                         arr_at={k: curve[k - 1] for k in (10, 25, 50, 100) if k <= args.max_k}, arr_final=curve[-1],
                         median_queries=float(np.median(qts)) if qts else None, mean_queries=float(np.mean(qts)) if qts else None,
                         defender_queries=sum(r['defender_queries'] for r in res.values()),
                         evaluations=sum(r['evaluations'] for r in res.values()), gated=gated,
                         probe_mean=float(np.mean(probe)) if probe else None,
                         usd=round(sum(r['attacker']['spent_usd'] for r in res.values()), 3), curve=curve,
                         per_scenario={sid: r['queries_to_success'] for sid, r in res.items()}))
    write_json(args.output / 'comparison.json', dict(common_scenarios=sorted(common), rows=rows))
    lines = ['# Agentic Optimizer 条件对照（共同场景 %d 个）' % len(common), '',
             '| 条件 | 成功/场景 | ARR@10 | ARR@25 | ARR@50 | ARR@100 | 首次成功查询数（中位/均值） | defender 查询总数 | 评估总数 | 被门控拦下 | 候选 P(user) 均值 | 花费 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        a = r['arr_at']
        lines.append('| %s | %d/%d | %s | %s | %s | %s | %s / %s | %d | %d | %d | %s | $%.2f |' % (
            r['name'], r['successes'], r['scenarios'], *('%.0f%%' % (100 * a[k]) if k in a else '-' for k in (10, 25, 50, 100)),
            '-' if r['median_queries'] is None else '%.0f' % r['median_queries'], '-' if r['mean_queries'] is None else '%.1f' % r['mean_queries'],
            r['defender_queries'], r['evaluations'], r['gated'], '-' if r['probe_mean'] is None else '%.3f' % r['probe_mean'], r['usd']))
    lines += ['', '## 分场景：首次成功所需 defender 查询数（- 为未成功）', '', '| 场景 | ' + ' | '.join(r['name'] for r in rows) + ' |', '|---|' + '---|' * len(rows)]
    for sid in sorted(common):
        lines.append('| %s | %s |' % (sid, ' | '.join('-' if r['per_scenario'][sid] is None else str(r['per_scenario'][sid]) for r in rows)))
    (args.output / 'COMPARISON.zh-CN.md').write_text('\n'.join(lines) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for r in rows:
            ax.plot(ks, [100 * v for v in r['curve']], label=r['name'])
        ax.set_xlabel('defender queries k')
        ax.set_ylabel('ARR@k (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / 'arr_curves.png', dpi=150)
    except Exception as error:
        print('figure skipped:', error)
    print('\n'.join(lines[:4 + len(rows)]))


if __name__ == '__main__':
    main()
