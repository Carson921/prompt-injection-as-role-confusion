"""Tables for the Llama 4 Scout report (probe, reference P(user), ASR of all tested conditions).

usage: ds_report.py            -> prints markdown tables, writes outputs/optimized-20261125/report_data.json
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from ds_common import PROBE_RUN, read_json, write_json
from ds_contexts import first_exposure

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
EVO = OUT / 'l4evo-20261125'
RUNS = [OUT / 'baseline-20261125', OUT / 'reference2-20261125', OUT / 'optimized-20261125', OUT / 'graded-20261125',
        OUT / 'decile-20261125']
RERUN = {'opt_ad_top_rerun': 'opt_ad_top'}  # same template run twice: left out of correlations, compared case by case
PAIRS = [('opt_ad_top', 'delim_close_user'), ('opt_ad_top', 'delim_close_assistant_user'), ('opt_ad_top', 'delim_oneline_assistant_user'),
         ('opt_ad_top', 'delim_oneline_user'), ('opt_ad_top', 'near_simple'), ('opt_ad_top', 'near_oneline'), ('opt_ad_top', 'original'),
         ('opt_ipi_top', 'delim_close_user'), ('opt_ipi_top', 'delim_close_assistant_user'), ('opt_ipi_top', 'near_simple'),
         ('opt_ad_top', 'opt_ipi_top')]


def mcnemar(a, b):
    """Exact two-sided McNemar (binomial on discordant pairs) for two conditions over their shared cases."""
    from math import comb
    keys = sorted(set(a) & set(b))
    only_a, only_b = sum(a[k] and not b[k] for k in keys), sum(b[k] and not a[k] for k in keys)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p
AD, IPI = ('banking', 'slack', 'travel'), ('browser', 'coding', 'tool')
pct = lambda s, n: f'{100 * s / n:.1f}% ({s}/{n})'
f3 = lambda v: '-' if v is None else f'{v:.3f}'


def agentdojo(run, condition):
    rows = [read_json(p) for p in sorted((run / 'agentdojo/cases' / condition).glob('*/*/*.json'))]
    if not rows:
        return None
    exposed = sum(first_exposure(r) is not None for r in rows)  # a model call saw the goal text inside a tool result
    return dict(n=len(rows), success=sum(bool(r['security']) for r in rows), utility=sum(bool(r['utility']) for r in rows),
                errors=sum(r['status'] != 'complete' for r in rows), exposed=exposed,
                malformed=sum(any(c.get('malformed') or c.get('finish_reason') == 'length' for c in r.get('calls', [])) for r in rows),
                suites={s: [sum(bool(r['security']) for r in rows if r['suite'] == s), sum(r['suite'] == s for r in rows)] for s in AD},
                cases={r['base_case_id']: bool(r['security']) for r in rows})


def ipi(run, condition):
    rows = [r for r in (read_json(p) for p in sorted((run / 'ipi/cases' / condition).glob('*.json'))) if r.get('primary')]
    if not rows:
        return None
    return dict(n=len(rows), success=sum(bool(r['success']) for r in rows), errors=sum(r['status'] != 'complete' for r in rows),
                categories={c: [sum(bool(r['success']) for r in rows if r['category'] == c), sum(r['category'] == c for r in rows)] for c in IPI},
                cases={r['case_id']: bool(r['success']) for r in rows})


def spearman(x, y):
    """Spearman rho with tie-averaged ranks; two-sided p from 20000 seeded permutations."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind='stable')
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):
            ranks[v == value] = ranks[v == value].mean()
        return ranks
    a, b = rank(x), rank(y)
    rho = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(0)
    perms = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(20000)])
    return rho, float(((np.abs(perms) >= abs(rho) - 1e-12).sum() + 1) / (len(perms) + 1))


def table(header, rows):
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(str(c) for c in row) + ' |')
    print()


def main():
    data = {}
    print('## probe')
    m = read_json(PROBE_RUN / 'metrics.json')
    t = m['test']
    table(['candidate', 'sha', 'dev acc', 'test acc', 'by domain', *[f'{r} P/R (n)' for r in t['overall']['roles']]],
          [[m['selected_candidate'], m['checkpoint_sha256'][:12], f"{m['selected_dev']['overall']['accuracy']:.4f}", f"{t['overall']['accuracy']:.4f}",
            ' '.join(f"{k} {v['accuracy']:.4f}" for k, v in t['by_domain'].items()),
            *[f"{v['precision']:.3f}/{v['recall']:.3f} ({v['support']})" for v in t['overall']['roles'].values()]]])
    selection = read_json(PROBE_RUN / 'head-selection.json')
    print('heads (layer.head):', ' '.join(f'{h // 40}.{h % 40}' for h in selection['heads']))
    audit = read_json(PROBE_RUN / 'data/manifest.json')['counts']
    print('records:', {k: v for k, v in audit.items() if k.endswith('records')})
    print()

    conditions, asr = {}, {}
    for run in RUNS:
        if not (run / 'conditions.json').exists():
            continue
        for name, spec in read_json(run / 'conditions.json').items():
            conditions[name] = dict(spec, run=run.name)
            asr[name] = dict(agentdojo=agentdojo(run, name), ipi=ipi(run, name))
    data['conditions'], data['asr'] = conditions, asr
    print('## conditions')
    table(['condition', 'run', 'genuine special tokens', 'template'],
          [[n, c['run'], 'yes' if c.get('allow_special') else 'no', '`' + c['template'].replace('\n', '\\n').replace('|', '\\|') + '`'] for n, c in conditions.items()])

    print('## panel P(user) and ASR')
    panels = {b: read_json(EVO / b / 'refs.json') for b in ('agentdojo', 'ipi')}
    # a panel mean needs all 12 contexts (a template whose payload cannot be located in some context gets no panel value)
    full = lambda rows: bool(rows) and len(rows) == 12 and all('goal_user' in r for r in rows.values())
    mean = lambda rows, key='goal_user': float(np.mean([r[key] for r in rows.values()])) if full(rows) else None
    roles = lambda rows: {r: float(np.mean([v['goal_roles'][r] for v in rows.values()])) for r in ('user', 'tool', 'assistant', 'system')} if full(rows) else None
    measured = lambda rows, s: [r['goal_user'] for r in (rows or {}).values() if r['suite'] == s and 'goal_user' in r]
    suite = lambda rows, s: float(np.mean(measured(rows, s))) if measured(rows, s) else None
    rows, data['panel'] = [], {}
    for name in ['true_user_turn', *conditions]:
        a, i = panels['agentdojo'].get(RERUN.get(name, name)), panels['ipi'].get(RERUN.get(name, name))
        data['panel'][name] = dict(agentdojo=mean(a), ipi=mean(i), agentdojo_roles=roles(a), ipi_roles=roles(i),
                                   agentdojo_suites={s: suite(a, s) for s in AD} if a else None, ipi_suites={s: suite(i, s) for s in IPI} if i else None,
                                   agentdojo_unlocated=sum('skipped' in r for r in (a or {}).values()), ipi_unlocated=sum('skipped' in r for r in (i or {}).values()))
        x = asr.get(name, {})
        ra, ri = x.get('agentdojo'), x.get('ipi')
        rows.append([name, f3(mean(a)), pct(ra['success'], ra['n']) if ra else '-', f"{100 * ra['utility'] / ra['n']:.1f}%" if ra else '-',
                     f3(mean(i)), pct(ri['success'], ri['n']) if ri else '-'])
    table(['condition', 'P(user) AD12', 'AD ASR', 'AD utility', 'P(user) IPI12', 'IPI ASR'], rows)
    table(['condition', 'AD banking', 'AD slack', 'AD travel', 'AD tool mass', 'IPI browser', 'IPI coding', 'IPI tool', 'IPI tool mass', 'IPI assistant mass'],
          [[n, *[f3((p['agentdojo_suites'] or {}).get(s)) for s in AD], f3((p['agentdojo_roles'] or {}).get('tool')),
            *[f3((p['ipi_suites'] or {}).get(s)) for s in IPI], f3((p['ipi_roles'] or {}).get('tool')), f3((p['ipi_roles'] or {}).get('assistant'))]
           for n, p in data['panel'].items()])
    data['correlation'] = {}
    for bench in ('agentdojo', 'ipi'):
        names = [n for n in conditions if n not in RERUN and data['panel'][n][bench] is not None and asr[n][bench]]
        x = [data['panel'][n][bench] for n in names]
        y = [asr[n][bench]['success'] / asr[n][bench]['n'] for n in names]
        r, p = spearman(x, y)
        line = f'{bench}: Spearman(P(user), ASR) = {r:.3f} (p={p:.3g}, n={len(names)})'
        data['correlation'][bench] = dict(spearman=r, p=p, n=len(names))
        plain = [n for n in names if not conditions[n].get('allow_special')]
        if len(plain) > 3:
            r2, p2 = spearman([data['panel'][n][bench] for n in plain], [asr[n][bench]['success'] / asr[n][bench]['n'] for n in plain])
            line += f'; without genuine special tokens ({len(plain)}): {r2:.3f} (p={p2:.3g})'
            data['correlation'][bench + '_no_special'] = dict(spearman=r2, p=p2, n=len(plain))
        graded = [n for n in names if n.startswith('grade_')]
        if len(graded) > 3:
            r3, p3 = spearman([data['panel'][n][bench] for n in graded], [asr[n][bench]['success'] / asr[n][bench]['n'] for n in graded])
            line += f'; graded set only ({len(graded)}): {r3:.3f} (p={p3:.3g})'
            data['correlation'][bench + '_graded'] = dict(spearman=r3, p=p3, n=len(graded))
        # decile set: one searched candidate per P(user) bin; "own" = the ten candidates of this benchmark's own search
        for label, members in (('decile set', [n for n in names if n.startswith('bin_')]),
                               ('decile set, own search', [n for n in names if n.startswith('bin_') and conditions[n]['optimized_for'] == bench]),
                               ('graded + decile sets', [n for n in names if n.startswith(('grade_', 'bin_'))])):
            if len(members) > 3:
                r4, p4 = spearman([data['panel'][n][bench] for n in members], [asr[n][bench]['success'] / asr[n][bench]['n'] for n in members])
                line += f'; {label} ({len(members)}): {r4:.3f} (p={p4:.3g})'
                data['correlation'][bench + '_' + label.replace(',', '').replace(' + ', '_').replace(' ', '_')] = dict(spearman=r4, p=p4, n=len(members))
        print(line)
    print()

    decile = [n for n in conditions if n.startswith('bin_')]
    if decile and all(asr[n]['agentdojo'] and asr[n]['ipi'] for n in decile):
        print('## decile set: one searched candidate per surrogate P(user) bin')
        rows = []
        for n in decile:
            c, own = conditions[n], conditions[n]['optimized_for']
            other = 'ipi' if own == 'agentdojo' else 'agentdojo'
            a = asr[n]
            rows.append([n, own, f"[{c['bin'][0]:.1f}, {c['bin'][1]:.1f})", c['bin_candidates'], f"{c['evaluation_index']}/{c['evaluations']}", c['wrapper_tokens'],
                         f"{c['fast_fitness']:.3f}", f3(data['panel'][n][own]), pct(a[own]['success'], a[own]['n']),
                         pct(a['agentdojo']['success'], a['agentdojo']['exposed']) if own == 'agentdojo' and a['agentdojo']['exposed'] else '-',
                         f3(data['panel'][n][other]), pct(a[other]['success'], a[other]['n'])])
        table(['condition', 'search', 'bin', 'candidates in bin', 'evaluation', 'tokens', 'surrogate P(user)', 'exact P(user), own panel', 'ASR, own benchmark',
               'ASR among episodes that reached the injection', 'exact P(user), other panel', 'ASR, other benchmark'], rows)
        data['decile'] = rows
        # every mechanically chosen searched candidate (graded + decile sets), binned by the exact panel P(user) of the benchmark
        print('## graded + decile sets: ASR by exact panel P(user) decile (templates pooled)')
        chosen, rows = [n for n in conditions if n.startswith(('grade_', 'bin_'))], []
        for bench in ('agentdojo', 'ipi'):
            for k in range(10):
                members = [n for n in chosen if data['panel'][n][bench] is not None and (k / 10 <= data['panel'][n][bench] < (k + 1) / 10 or (k == 9 and data['panel'][n][bench] >= 1))]
                rates = [100 * asr[n][bench]['success'] / asr[n][bench]['n'] for n in members]
                s, t = sum(asr[n][bench]['success'] for n in members), sum(asr[n][bench]['n'] for n in members)
                rows.append([bench, f'[{k / 10:.1f}, {(k + 1) / 10:.1f})', len(members), pct(s, t) if t else '-', f'{min(rates):.1f}% - {max(rates):.1f}%' if rates else '-'])
        table(['benchmark', 'exact P(user) bin', 'templates', 'pooled ASR', 'lowest - highest template'], rows)
        data['chosen_deciles'] = rows

    graded = [n for n in conditions if n.startswith('grade_')]
    if graded and all(asr[n]['agentdojo'] and asr[n]['ipi'] for n in graded):
        print('## graded set: ASR by exact panel P(user) bin (templates pooled)')
        edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
        rows = []
        for bench, total in (('agentdojo', 'n'), ('ipi', 'n')):
            for low, high in zip(edges, edges[1:]):
                members = [n for n in graded if data['panel'][n][bench] is not None and low <= data['panel'][n][bench] < high]
                s = sum(asr[n][bench]['success'] for n in members)
                t = sum(asr[n][bench][total] for n in members)
                rows.append([bench, f'[{low:.1f}, {min(high, 1):.1f})', len(members), pct(s, t) if t else '-'])
        table(['benchmark', 'P(user) bin', 'templates', 'pooled ASR'], rows)
        data['graded_bins'] = rows

    print('## paired comparisons (exact McNemar, shared cases)')
    rows, data['paired'] = [], []
    for first, second in PAIRS + [(b, a) for a, b in RERUN.items()]:
        for bench in ('agentdojo', 'ipi'):
            a, b = asr.get(first, {}).get(bench), asr.get(second, {}).get(bench)
            if not a or not b:
                continue
            only_a, only_b, p = mcnemar(a['cases'], b['cases'])
            rows.append([first, second, bench, pct(a['success'], a['n']), pct(b['success'], b['n']), only_a, only_b, f'{p:.3g}'])
            data['paired'].append(dict(first=first, second=second, bench=bench, only_first=only_a, only_second=only_b, p=p))
    table(['A', 'B', 'benchmark', 'A ASR', 'B ASR', 'A only', 'B only', 'p'], rows)

    print('## breakdown')
    panel_ad = {c['case_id'] for c in read_json(EVO / 'agentdojo/dataset.json')['contexts']}
    panel_ipi = set(read_json(EVO / 'ipi/dataset.json')['ipi_cases'])
    rows, held = [], []
    for name in conditions:
        a, i = asr[name]['agentdojo'], asr[name]['ipi']
        rows.append([name, *([f'{s}/{n}' for s, n in a['suites'].values()] + [f"{a['exposed']}/{a['n']}", pct(a['success'], a['exposed']) if a['exposed'] else '-', a['malformed'], a['errors']] if a else ['-'] * 7),
                     *([f'{s}/{n}' for s, n in i['categories'].values()] + [i['errors']] if i else ['-'] * 4)])
        ha = {k: v for k, v in a['cases'].items() if k not in panel_ad} if a else {}
        hi = {k: v for k, v in i['cases'].items() if k not in panel_ipi} if i else {}
        held.append([name, pct(sum(ha.values()), len(ha)) if ha else '-', pct(sum(hi.values()), len(hi)) if hi else '-'])
    table(['condition', 'banking', 'slack', 'travel', 'episodes that reached the injection', 'ASR among those', 'malformed', 'AD errors', 'browser', 'coding', 'tool', 'IPI errors'], rows)
    table(['condition', 'AD held-out (377)', 'IPI held-out (24)'], held)

    print('## AgentDojo episodes whose output contains injection keywords')
    pattern = re.compile(r'injection|suspicious|malicious', re.I)
    rows, data['flagged'] = [], {}
    for name, spec in conditions.items():
        cases = [read_json(p) for p in sorted((OUT / spec['run'] / 'agentdojo/cases' / name).glob('*/*/*.json'))]
        if not cases:
            continue
        flag = [any(pattern.search(c.get('completion') or '') for c in r['calls']) for r in cases]
        ok = [bool(r['security']) for r in cases]
        data['flagged'][name] = dict(flagged=sum(flag), n=len(cases), success_flagged=sum(f and s for f, s in zip(flag, ok)))
        rows.append([name, pct(sum(ok), len(ok)), f'{sum(flag)}/{len(cases)}'])
    table(['condition', 'ASR', 'flagged episodes'], rows)

    print('## evolution')
    rows = []
    for bench in ('agentdojo', 'ipi'):
        if not (EVO / bench / 'summary.json').exists():
            continue
        s = read_json(EVO / bench / 'summary.json')
        scores = [json.loads(line) for line in (EVO / bench / 'scores.jsonl').read_text().splitlines() if line.strip()]
        sources = {}
        for r in scores:
            sources[r.get('source')] = sources.get(r.get('source'), 0) + 1
        top = read_json(EVO / bench / 'exact_top.json') if (EVO / bench / 'exact_top.json').exists() else []
        diffs = [abs(x['exact_fitness'] - x['fast_fitness']) for x in top]
        rows.append([bench, s['scored'], json.dumps(sources), s['llm']['calls'], f"${s['llm']['spent_usd']:.2f}", f"{s['best']['fitness']:.3f}",
                     f"{max(x['exact_fitness'] for x in top):.3f}" if top else '-', f"{np.mean(diffs):.4f} / {max(diffs):.4f}" if top else '-',
                     s['caps_used'], f"{s['elapsed_hours']:.2f}"])
        data.setdefault('evolution', {})[bench] = dict(scored=s['scored'], sources=sources, llm=s['llm'], best_fast=s['best']['fitness'],
                                                       best_exact=max(x['exact_fitness'] for x in top) if top else None)
    table(['objective', 'scored', 'by source', 'LLM calls', 'LLM cost', 'best fast', 'best exact', 'exact-vs-fast mean / max |diff| (verified top)', 'caps used', 'hours'], rows)
    write_json(OUT / 'optimized-20261125/report_data.json', data)


if __name__ == '__main__':
    main()
