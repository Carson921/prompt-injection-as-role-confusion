"""Tables for the second-round report: baselines, first attempt, genuine-delimiter references, second-round templates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ds_common import read_json, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
RUNS = [  # (run dir, conditions)
    (OUT / 'baseline-20261118', ('goal_only', 'original', 'original_delimiter')),
    (OUT / 'optimized-20261118', ('opt_top1',)),
    (OUT / 'optimized2-20261119', ('ctrl_leadin', 'delim_close_user', 'delim_close_assistant_user', 'opt2_simple', 'opt2_top1',
                                   'opt2_top2', 'opt2_assistant')),
]
AD = ('banking', 'slack', 'travel')
IPI = ('browser', 'coding', 'tool')


def agentdojo(run, condition):
    root = run / 'agentdojo/cases' / condition
    if not root.exists():
        return None
    rows = [read_json(p) for p in sorted(root.glob('*/*/*.json'))]
    malformed = sum(any(c.get('malformed') or c.get('finish_reason') == 'length' for c in r.get('calls', [])) for r in rows)
    return dict(n=len(rows), success=sum(bool(r['security']) for r in rows), utility=sum(bool(r['utility']) for r in rows),
                errors=sum(r['status'] != 'complete' for r in rows), malformed=malformed,
                suites={s: [sum(bool(r['security']) for r in rows if r['suite'] == s), sum(r['suite'] == s for r in rows)] for s in AD},
                cases={r['base_case_id']: bool(r['security']) for r in rows})


def ipi(run, condition):
    root = run / 'ipi/cases' / condition
    if not root.exists():
        return None
    rows = [r for r in (read_json(p) for p in sorted(root.glob('*.json'))) if r.get('primary')]
    return dict(n=len(rows), success=sum(bool(r['success']) for r in rows), errors=sum(r['status'] != 'complete' for r in rows),
                categories={c: [sum(bool(r['success']) for r in rows if r['category'] == c), sum(r['category'] == c for r in rows)] for c in IPI},
                cases={r['case_id']: bool(r['success']) for r in rows})


def main():
    out = Path(sys.argv[1])
    dataset = read_json(OUT / 'dsevo2-20261119/dataset.json')
    panel_ad = {c['case_id'] if 'case_id' in c else c['base_case_id'] for c in dataset['contexts']}
    panel_ipi = set(dataset['ipi_cases'])
    table = {}
    for run, conditions in RUNS:
        for name in conditions:
            a, i = agentdojo(run, name), ipi(run, name)
            row = dict(run=run.name, agentdojo=a, ipi=i)
            if a:
                held = {k: v for k, v in a['cases'].items() if k not in panel_ad}
                row['agentdojo_heldout'] = [sum(held.values()), len(held)]
            if i:
                held = {k: v for k, v in i['cases'].items() if k not in panel_ipi}
                row['ipi_heldout'] = [sum(held.values()), len(held)]
            table[name] = row
    write_json(out / 'report_rows.json', table)
    pct = lambda s, n: f'{100 * s / n:.1f}% ({s}/{n})'
    print('| 条件 | AgentDojo ASR | banking | slack | travel | utility | 截断/异常 | IPI ASR | browser | coding | tool |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for name, row in table.items():
        a, i = row['agentdojo'], row['ipi']
        left = [pct(a['success'], a['n']), *[f'{s}/{n}' for s, n in a['suites'].values()], f"{100 * a['utility'] / a['n']:.1f}%", str(a['malformed'])] if a else ['-'] * 6
        right = [pct(i['success'], i['n']), *[f'{s}/{n}' for s, n in i['categories'].values()]] if i else ['-'] * 4
        print('| ' + ' | '.join([name, *left, *right]) + ' |')
    print()
    print('| 条件 | AgentDojo 留出 | IPI 留出 | 错误 |')
    print('|---|---|---|---|')
    for name, row in table.items():
        a, i = row['agentdojo'], row['ipi']
        print(f"| {name} | {pct(*row['agentdojo_heldout']) if a else '-'} | {pct(*row['ipi_heldout']) if i else '-'} | {(a['errors'] if a else 0) + (i['errors'] if i else 0)} |")
    print(len(panel_ad), len(panel_ipi), file=sys.stderr)


if __name__ == '__main__':
    main()
