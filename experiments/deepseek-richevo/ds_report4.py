"""Tables for the corrected-probe round (developer -> system probe, separate AgentDojo / IPI objectives).

usage: ds_report4.py            -> prints markdown tables, writes outputs/optimized4-20261119/report_data.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ds_common import PROBE_RUN, PROBE_RUN_V1, read_json, write_json
from ds_report2 import agentdojo, ipi

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
EVO, NEW, OLD = OUT / 'dsevo4-20261119', OUT / 'optimized4-20261119', OUT / 'optimized2-20261119'
RUNS = [(OUT / 'baseline-20261118', ('goal_only', 'original', 'original_delimiter')), (OUT / 'optimized-20261118', ('opt_top1',)),
        (OLD, ('ctrl_leadin', 'delim_close_user', 'delim_close_assistant_user', 'opt2_simple', 'opt2_top1', 'opt2_top2', 'opt2_assistant')),
        (NEW, ('opt4_ad_top', 'opt4_ad_short', 'opt4_ipi_top', 'opt4_ipi_short'))]
ORDER = [c for _, cs in RUNS for c in cs]
AD, IPI = ('banking', 'slack', 'travel'), ('browser', 'coding', 'tool')
OLD_KEYS = {'opt_top1': 'near_close_user (opt_top1)', 'original_delimiter': 'delim_user_original (original_delimiter)'}
pct = lambda s, n: f'{100 * s / n:.1f}% ({s}/{n})'
f3 = lambda v: '-' if v is None else f'{v:.3f}'


def mean(rows, suites, key='goal_user'):
    values = [r[key] for r in rows.values() if 'goal_user' in r and r.get('suite', r.get('category')) in suites]
    return float(np.mean(values)) if values else None


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


def auc(pos, neg):
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def table(header, rows):
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(str(c) for c in row) + ' |')
    print()


def main():
    data = {}
    # T1 probe quality
    print('## T1 probe')
    rows = []
    for name, run in (('v1 (developer→user)', PROBE_RUN_V1), ('v2 devsys (developer→system)', PROBE_RUN)):
        m = read_json(run / 'metrics.json')
        t = m['test']
        dom = {k: v['accuracy'] for k, v in t['by_domain'].items()}
        rows.append([name, m['selected_candidate'], m['checkpoint_sha256'][:12], f"{m['selected_dev']['overall']['accuracy']:.4f}",
                     f"{t['overall']['accuracy']:.4f}", *[f"{t['overall']['roles'][r]['precision']:.3f}/{t['overall']['roles'][r]['recall']:.3f}" for r in ('system', 'user', 'tool')],
                     ' '.join(f'{k} {v:.3f}' for k, v in dom.items()), t['overall']['roles']['user']['support'], t['overall']['roles']['system']['support']])
    table(['probe', 'candidate', 'sha', 'dev acc', 'test acc', 'system P/R', 'user P/R', 'tool P/R', 'test by domain', 'user tokens', 'system tokens'], rows)

    # ASR rows
    asr = {}
    for run, conditions in RUNS:
        for name in conditions:
            asr[name] = dict(run=run.name, agentdojo=agentdojo(run, name), ipi=ipi(run, name))
    data['asr'] = asr

    # T2 joint panel, old vs new probe
    print('## T2 joint panel (AD6 + IPI6), old vs new probe')
    new_joint = read_json(EVO / 'refs_joint.json')
    old_target = read_json(OUT / 'dsevo2-20261119/delimiter_target.json')
    old_conditions = read_json(OLD / 'conditions.json')
    rows, data['joint'] = [], {}
    for name in ['true_user_turn', *ORDER]:
        if name not in new_joint:
            continue
        if name in old_conditions:
            old = (old_conditions[name]['agentdojo_mean'], old_conditions[name]['ipi_mean'])
        else:
            o = old_target.get(OLD_KEYS.get(name, name))
            old = (mean(o, AD), mean(o, IPI)) if o else (None, None)
        new = (mean(new_joint[name], AD), mean(new_joint[name], IPI))
        data['joint'][name] = dict(old=old, new=new)
        rows.append([name, f3(old[0]), f3(new[0]), f3(old[1]), f3(new[1])])
    table(['condition', 'AD6 old', 'AD6 new', 'IPI6 old', 'IPI6 new'], rows)
    both = [k for k, v in data['joint'].items() if v['old'][0] is not None and k != 'true_user_turn']
    for i, bench in enumerate(('agentdojo', 'ipi')):
        r, p = spearman([data['joint'][k]['old'][i] for k in both], [data['joint'][k]['new'][i] for k in both])
        print(f'old-vs-new probe rank correlation on joint panel, {bench}: Spearman {r:.3f} (p={p:.3g}, n={len(both)})')
    print()

    # T3 12-context panels (new probe) vs ASR
    print('## T3 12-context panels (new probe) vs ASR')
    panels = {b: read_json(EVO / b / 'refs.json') for b in ('agentdojo', 'ipi')}
    rows, data['panel'] = [], {}
    for name in ['true_user_turn', *ORDER]:
        a, i = panels['agentdojo'].get(name), panels['ipi'].get(name)
        pa, pi = (mean(a, AD) if a else None), (mean(i, IPI) if i else None)
        roles = lambda rs: {r: float(np.mean([v['goal_roles'][r] for v in rs.values()])) for r in ('user', 'assistant', 'tool', 'system', 'cot')} if rs else None
        data['panel'][name] = dict(agentdojo=pa, ipi=pi, agentdojo_suites={s: mean(a, (s,)) for s in AD} if a else None,
                                   ipi_suites={s: mean(i, (s,)) for s in IPI} if i else None, agentdojo_roles=roles(a), ipi_roles=roles(i))
        x = asr.get(name, {})
        ra, ri = x.get('agentdojo'), x.get('ipi')
        rows.append([name, f3(pa), pct(ra['success'], ra['n']) if ra else '-', f"{100 * ra['utility'] / ra['n']:.1f}%" if ra else '-',
                     f3(pi), *[f3(mean(i, (s,))) if i else '-' for s in IPI], pct(ri['success'], ri['n']) if ri else '-'])
    table(['condition', 'P(user) AD12', 'AD ASR', 'AD utility', 'P(user) IPI12', 'browser', 'coding', 'tool', 'IPI ASR'], rows)
    for bench in ('agentdojo', 'ipi'):
        names = [n for n in ORDER if data['panel'][n][bench] is not None and asr[n][bench]]
        if len(names) > 3:
            x = [data['panel'][n][bench] for n in names]
            y = [asr[n][bench]['success'] / asr[n][bench]['n'] for n in names]
            r, p = spearman(x, y)
            high = [n for n in names if data['panel'][n][bench] >= (0.9 if bench == 'agentdojo' else 0.35)]
            line = f'{bench}: Spearman(P(user), ASR) = {r:.3f} (p={p:.3g}, n={len(names)})'
            if len(high) > 3:
                r2, p2 = spearman([data['panel'][n][bench] for n in high], [asr[n][bench]['success'] / asr[n][bench]['n'] for n in high])
                line += f'; high-P(user) subset ({len(high)}): {r2:.3f} (p={p2:.3g})'
            print(line)
            data.setdefault('correlation', {})[bench] = dict(spearman=r, p=p, n=len(names))
    print()
    print('role mass on goal tokens (new probe, 12-panels)')
    table(['condition', 'AD user', 'AD tool', 'AD assistant', 'IPI user', 'IPI tool', 'IPI assistant', 'IPI system'],
          [[n, *[f3((data['panel'][n]['agentdojo_roles'] or {}).get(r)) for r in ('user', 'tool', 'assistant')],
            *[f3((data['panel'][n]['ipi_roles'] or {}).get(r)) for r in ('user', 'tool', 'assistant', 'system')]] for n in ['true_user_turn', *ORDER]])

    # T4 suites / categories and held-out for the new picks and key references
    print('## T4 breakdown')
    panel_ad = {c.get('case_id', c.get('base_case_id')) for c in read_json(EVO / 'agentdojo/dataset.json')['contexts']}
    panel_ipi = set(read_json(EVO / 'ipi/dataset.json')['ipi_cases'])
    rows, held = [], []
    for name in ORDER:
        a, i = asr[name]['agentdojo'], asr[name]['ipi']
        rows.append([name, *([f'{s}/{n}' for s, n in a['suites'].values()] + [a['malformed'], a['errors']] if a else ['-'] * 5),
                     *([f'{s}/{n}' for s, n in i['categories'].values()] + [i['errors']] if i else ['-'] * 4)])
        ha = {k: v for k, v in a['cases'].items() if k not in panel_ad} if a else {}
        hi = {k: v for k, v in i['cases'].items() if k not in panel_ipi} if i else {}
        held.append([name, pct(sum(ha.values()), len(ha)) if ha else '-', pct(sum(hi.values()), len(hi)) if hi else '-'])
        asr[name]['heldout'] = dict(agentdojo=[sum(ha.values()), len(ha)], ipi=[sum(hi.values()), len(hi)])
    table(['condition', 'banking', 'slack', 'travel', 'malformed', 'AD errors', 'browser', 'coding', 'tool', 'IPI errors'], rows)
    table(['condition', 'AD held-out (377)', 'IPI held-out (24)'], held)

    # T5 swap diagnostic
    print('## T5 swap diagnostic (genuine user turn)')
    rows = []
    for label, path in (('old', OUT / 'dsevo3-20261119/diag_ipi_user.json'), ('new', EVO / 'diag_ipi_user.json')):
        if not path.exists():
            continue
        d = read_json(path)
        for bench, suites in (('agentdojo', AD), ('ipi', IPI)):
            for cond in ('own_goal', 'swapped_goal', 'own_goal_capitalized', 'leadin_own_goal'):
                rs = [r for r in d.values() if r['suite'] in suites and r['condition'] == cond]
                if rs:
                    rows.append([label, bench, cond, len(rs), f3(float(np.mean([r['final_user'] for r in rs]))),
                                 f3(float(np.mean([r['original_request_user'] for r in rs]))),
                                 ' '.join(f"{k[:4]} {float(np.mean([r['final_roles'][k] for r in rs])):.2f}" for k in ('user', 'assistant', 'tool', 'system'))])
    data['swap'] = rows
    table(['probe', 'context', 'final turn text', 'n', 'P(user) final turn', 'P(user) original request', 'final-turn role mass'], rows)

    # T6 IPI-36
    path = EVO / 'ipi36_refs.json'
    if path.exists():
        print('## T6 IPI primary behaviors (new probe)')
        d = read_json(path)
        old36 = read_json(OLD / 'ipi36_scores.json')
        rows, data['ipi36'] = [], {}
        for name, rs in d.items():
            done = {k: v for k, v in rs.items() if 'goal_user' in v}
            x = asr.get(name, {}).get('ipi')
            pos = [v['goal_user'] for k, v in done.items() if x and x['cases'].get(k)]
            neg = [v['goal_user'] for k, v in done.items() if x and not x['cases'].get(k)]
            o = old36.get(name)
            data['ipi36'][name] = dict(n=len(done), mean=mean(done, IPI), auc=auc(pos, neg), skipped=sorted(set(rs) - set(done)))
            rows.append([name, len(done), f3(mean(done, IPI)), *[f3(mean(done, (s,))) for s in IPI],
                         f3(float(np.mean([v['goal_user'] for v in o.values() if 'goal_user' in v]))) if o else '-',
                         pct(x['success'], x['n']) if x else '-', f3(float(np.mean(pos))) if pos else '-', f3(float(np.mean(neg))) if neg else '-',
                         f3(auc(pos, neg))])
        table(['condition', 'n', 'P(user)', 'browser', 'coding', 'tool', 'old probe', 'IPI ASR', 'P(user) success', 'P(user) fail', 'AUC'], rows)
        names = [n for n in d if asr.get(n, {}).get('ipi') and data['ipi36'][n]['mean'] is not None]
        if len(names) > 3:
            r, p = spearman([data['ipi36'][n]['mean'] for n in names], [asr[n]['ipi']['success'] / 36 for n in names])
            print(f'IPI-36: Spearman(mean P(user), ASR) across {len(names)} conditions = {r:.3f} (p={p:.3g})')
            data['correlation']['ipi36'] = dict(spearman=r, p=p, n=len(names))
        skipped = sorted({k for v in data['ipi36'].values() for k in v['skipped']})
        print('skipped (too long):', skipped)

    # T7 AgentDojo episodes whose model output (reasoning included) names an injection
    print('\n## T7 AgentDojo episodes with injection keywords in the model output')
    pattern = re.compile(r'injection|suspicious|malicious', re.I)
    rows, data['flagged'] = [], {}
    for run, conditions in RUNS:
        for name in conditions:
            cases = [read_json(p) for p in sorted((run / 'agentdojo/cases' / name).glob('*/*/*.json'))]
            if not cases:
                continue
            flag = [any(pattern.search(c.get('completion') or '') for c in r['calls']) for r in cases]
            ok = [bool(r['security']) for r in cases]
            both, clean = sum(f and s for f, s in zip(flag, ok)), sum((not f) and s for f, s in zip(flag, ok))
            data['flagged'][name] = dict(flagged=sum(flag), n=len(cases), success_flagged=both, success_unflagged=clean)
            rows.append([name, pct(sum(ok), len(ok)), f'{sum(flag)}/{len(cases)}', pct(both, sum(flag)) if sum(flag) else '-',
                         pct(clean, len(cases) - sum(flag)) if len(cases) - sum(flag) else '-'])
    table(['condition', 'ASR', 'flagged episodes', 'ASR among flagged', 'ASR among unflagged'], rows)
    names = list(data['flagged'])
    r, p = spearman([data['flagged'][n]['flagged'] for n in names], [asr[n]['agentdojo']['success'] for n in names])
    print(f'Spearman(flagged episodes, ASR) across {len(names)} conditions = {r:.3f} (p={p:.3g})')
    delim = [n for n in names if n not in ('original', 'original_delimiter', 'ctrl_leadin')]
    r, p = spearman([data['flagged'][n]['flagged'] for n in delim], [asr[n]['agentdojo']['success'] for n in delim])
    print(f'  same, conditions that imitate or use a user delimiter ({len(delim)}): {r:.3f} (p={p:.3g})')

    # evolution stats
    print('\n## evolution')
    rows = []
    for bench in ('agentdojo', 'ipi'):
        s = read_json(EVO / bench / 'summary.json')
        scores = [json.loads(line) for line in (EVO / bench / 'scores.jsonl').read_text().splitlines() if line.strip()]
        exact = [json.loads(line) for line in (EVO / bench / 'exact_scores.jsonl').read_text().splitlines() if line.strip()]
        top = read_json(EVO / bench / 'exact_top.json')
        rows.append([bench, s['scored'], s['fresh'], s['llm']['calls'], f"${s['llm']['spent_usd']:.2f}", f"{s['best']['fitness']:.3f}",
                     f"{max(t['exact_fitness'] for t in top):.3f}", len(exact), s['caps_used'], f"{s['elapsed_hours']:.2f}"])
        data.setdefault('evolution', {})[bench] = dict(scored=s['scored'], fresh=s['fresh'], llm=s['llm'], best_fast=s['best']['fitness'],
                                                       best_exact=max(t['exact_fitness'] for t in top), exact_checks=len(exact), n_scores=len(scores))
        diffs = [abs(t['exact_fitness'] - t['fast_fitness']) for t in top]
        print(f'{bench}: exact-vs-fast on verified top {len(top)}: mean |diff| {np.mean(diffs):.4f}, max {max(diffs):.4f}')
    table(['objective', 'scored', 'fresh', 'LLM calls', 'LLM cost', 'best fast', 'best exact', 'exact checks during search', 'caps used', 'last-leg hours'], rows)
    write_json(NEW / 'report_data.json', data)


if __name__ == '__main__':
    main()
