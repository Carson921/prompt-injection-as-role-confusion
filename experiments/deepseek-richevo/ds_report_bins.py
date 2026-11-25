"""Tables for the P(user) decile set (ds_select_bins.py): bin -> panel P(user) -> ASR on AgentDojo and IPI Arena.

Conditions whose ASR run is not complete (cut by the end of a GPU job) are shown with their count and left out of the
correlations.  Exact panel P(user) (ds_refs4.py, refs_agentdojo.json / refs_ipi.json) is used when present; otherwise
only the surrogate value of the search is available, and only on the panel of the search the template came from.
usage: ds_report_bins.py [RUN_DIR]   -> prints markdown tables, writes RUN_DIR/report_data.json
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from ds_common import read_json, write_json
from ds_report2 import agentdojo, ipi
from ds_report4 import AD, IPI, f3, mean, pct, spearman, table

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'outputs/decile-20261125'
N_AD, N_IPI = 389, 36
FLAG = re.compile(r'injection|suspicious|malicious', re.I)


def flagged(name):
    rows = [read_json(p) for p in sorted((RUN / 'agentdojo/cases' / name).glob('*/*/*.json'))]
    return sum(any(FLAG.search(str(c.get('completion') or '')) for c in r.get('calls', [])) for r in rows)


def main():
    conditions = read_json(RUN / 'conditions.json')
    refs = {b: read_json(RUN / f'refs_{b}.json') if (RUN / f'refs_{b}.json').exists() else {} for b in ('agentdojo', 'ipi')}
    data = {}
    for name in sorted(conditions):
        c = conditions[name]
        row = dict(search=c['optimized_for'], bin=c['bin'], bin_candidates=c['bin_candidates'], surrogate=c['fast_fitness'],
                   wrapper_tokens=c['wrapper_tokens'], template=c['template'])
        for bench, suites in (('agentdojo', AD), ('ipi', IPI)):
            row[f'exact_{bench}'] = mean(refs[bench][name], suites) if name in refs[bench] else None
        row['agentdojo'], row['ipi'] = agentdojo(RUN, name), ipi(RUN, name)
        for bench in ('agentdojo', 'ipi'):
            if row[bench]:
                row[bench].pop('cases')
        row['flagged'] = flagged(name) if row['agentdojo'] else None
        data[name] = row

    def asr(row, bench, total):
        r = row[bench]
        if not r:
            return 'not run'
        return pct(r['success'], r['n']) if r['n'] == total else f"incomplete ({r['success']}/{r['n']} of {total})"

    for search, label in (('agentdojo', 'AgentDojo search'), ('ipi', 'IPI search')):
        print(f'### {label}\n')
        rows = [(n, r) for n, r in data.items() if r['search'] == search]
        table(['condition', 'bin', 'in bin', 'surrogate P(user)', 'exact AD panel', 'exact IPI panel', 'AgentDojo ASR', 'banking', 'slack', 'travel',
               'flagged', 'IPI ASR', 'browser', 'coding', 'tool', 'tokens'],
              [[n, f"[{r['bin'][0]:.1f}, {r['bin'][1]:.1f})", r['bin_candidates'], f3(r['surrogate']), f3(r['exact_agentdojo']), f3(r['exact_ipi']),
                asr(r, 'agentdojo', N_AD)] + ['/'.join(map(str, r['agentdojo']['suites'][s])) if r['agentdojo'] else '-' for s in AD]
               + [r['flagged'] if r['flagged'] is not None else '-', asr(r, 'ipi', N_IPI)]
               + ['/'.join(map(str, r['ipi']['categories'][s])) if r['ipi'] else '-' for s in IPI] + [r['wrapper_tokens']] for n, r in rows])
    print('### Spearman(P(user), ASR), complete conditions only\n')
    out, correlation = [], []
    for bench, total in (('agentdojo', N_AD), ('ipi', N_IPI)):
        for scope in ('agentdojo', 'ipi', 'all'):
            for kind in ('surrogate', 'exact'):
                rows = [r for r in data.values() if (scope == 'all' or r['search'] == scope) and r[bench] and r[bench]['n'] == total]
                if kind == 'surrogate':
                    rows = [r for r in rows if r['search'] == bench]      # surrogate exists only on the search's own panel
                    if scope != bench:
                        continue
                x = [r['surrogate'] if kind == 'surrogate' else r[f'exact_{bench}'] for r in rows]
                if len(rows) < 4 or any(v is None for v in x):
                    continue
                rho, p = spearman(x, [r[bench]['success'] for r in rows])
                out.append([bench, scope, kind, len(rows), f'{rho:.3f}', f'{p:.2g}'])
                correlation.append(dict(benchmark=bench, templates=scope, p_user=kind, n=len(rows), spearman=rho, p=p))
    table(['benchmark', 'templates from', 'P(user)', 'n', 'Spearman', 'permutation p'], out)
    # keyword-flagged AgentDojo episodes vs ASR; decile set joined with the conditions tested in the earlier rounds
    rows = [r for r in data.values() if r['agentdojo'] and r['agentdojo']['n'] == N_AD]
    rho, p = spearman([r['flagged'] for r in rows], [r['agentdojo']['success'] for r in rows])
    print(f'AgentDojo: Spearman(flagged episodes, ASR) = {rho:.3f} (p={p:.2g}, n={len(rows)})\n')
    extra = dict(flagged=dict(spearman=rho, p=p, n=len(rows)))
    earlier = RUN.parent / 'optimized4-20261119/report_data.json'
    if earlier.exists() and all(r['exact_agentdojo'] is not None and r['exact_ipi'] is not None for r in data.values()):
        old = read_json(earlier)
        print('### Decile set joined with the earlier conditions (exact panel P(user))\n')
        out = []
        for bench, total, high in (('agentdojo', N_AD, 0.9), ('ipi', N_IPI, 0.35)):
            pairs = [(r[f'exact_{bench}'], r[bench]['success'] / total, r['search'] != 'none') for r in data.values()]
            pairs += [(old['panel'][n][bench], a[bench]['success'] / a[bench]['n'], False) for n, a in old['asr'].items()
                      if a.get(bench) and old['panel'].get(n, {}).get(bench) is not None]
            for label, sub in (('all', pairs), (f'P(user) >= {high}', [q for q in pairs if q[0] >= high]), (f'P(user) < {high}', [q for q in pairs if q[0] < high])):
                rho, p = spearman([q[0] for q in sub], [q[1] for q in sub])
                asrs = [q[1] for q in sub]
                out.append([bench, label, len(sub), f'{rho:.3f}', f'{p:.2g}', f'{100 * min(asrs):.1f}% - {100 * max(asrs):.1f}%'])
                extra.setdefault('joined', []).append(dict(benchmark=bench, subset=label, n=len(sub), spearman=rho, p=p, asr_min=min(asrs), asr_max=max(asrs)))
        table(['benchmark', 'subset', 'n', 'Spearman', 'permutation p', 'ASR range'], out)
    write_json(RUN / 'report_data.json', dict(conditions=data, correlation=correlation, **extra))


if __name__ == '__main__':
    main()
