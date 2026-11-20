"""After the second evolution: exact re-measurement of the leading candidates, selection for ASR,
and exact goal-token P(user) on all 36 primary IPI Arena behaviors (probe server client).

Outputs (in --out): exact_top.json, conditions.json (ASR conditions: optimized picks plus genuine
delimiter references), panel_scores.json and ipi36_scores.json.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import numpy as np

from ds_common import ROLES, read_json, write_json
from ds_fast_scorer import FastScorer
from ds_probe_client import call
import ds_delim_target

AD = ('banking', 'slack', 'travel')
REFERENCES = {  # genuine delimiter tokens, ordinary use
    'delim_close_user': ds_delim_target.CONDITIONS['delim_close_user'],
    'delim_close_assistant_user': ds_delim_target.CONDITIONS['delim_close_assistant_user'],
}


def panel_summary(result):
    cases = result['per_case']
    ad = [c['goal_user_mean'] for c in cases if c['suite'] in AD]
    ipi = [c['goal_user_mean'] for c in cases if c['suite'] not in AD]
    return dict(joint=result['fitness'], agentdojo_mean=float(np.mean(ad)), ipi_mean=float(np.mean(ipi)),
                suite_means=result['suite_means'], per_case={c['case_id']: c['goal_user_mean'] for c in cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evolution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-top', type=int, default=12)
    parser.add_argument('--picks', type=int, default=2)
    parser.add_argument('--extra-conditions', type=Path, help='previous conditions.json to carry along (e.g. opt_top1)')
    parser.add_argument('--skip-ipi36', action='store_true')
    parser.add_argument('--ipi36-max-tokens', type=int, default=10000)
    parser.add_argument('--ipi36-conditions', nargs='*', help='subset of conditions for the IPI-36 measurement')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scorer = FastScorer(args.evolution / 'dataset.json', args.evolution)
    rows = [json.loads(line) for line in (args.evolution / 'scores.jsonl').read_text().splitlines() if line.strip()]
    best = {}
    for r in rows:
        if r.get('status') == 'complete' and (r['template'] not in best or r['fitness'] > best[r['template']]['fitness']):
            best[r['template']] = r
    ranked = sorted(best.values(), key=lambda r: -r['fitness'])
    candidates = [r['template'] for r in ranked[:args.verify_top]]
    exact_log = args.evolution / 'exact_scores.jsonl'
    if exact_log.exists():  # every candidate that was a best during the search
        candidates += [json.loads(line)['template'] for line in exact_log.read_text().splitlines()][-6:]
    exact = []
    for template in dict.fromkeys(candidates):
        measured = scorer.score(template, exact=True)
        exact.append(dict(template=template, fast_fitness=best[template]['fitness'], wrapper_tokens=best[template]['wrapper_tokens'],
                          source=best[template].get('source'), operator=best[template].get('operator'),
                          evaluation_index=best[template].get('evaluation_index'), **panel_summary(measured)))
        print(json.dumps(dict(stage='exact', fast=round(exact[-1]['fast_fitness'], 4), exact=round(exact[-1]['joint'], 4),
                              agentdojo=round(exact[-1]['agentdojo_mean'], 4), ipi=round(exact[-1]['ipi_mean'], 4),
                              template=template[:100]), ensure_ascii=False), flush=True)
    exact.sort(key=lambda r: -r['joint'])
    write_json(args.out / 'exact_top.json', exact)
    conditions, picks = {}, []
    for r in exact:
        if len(picks) >= args.picks:
            break
        if all(difflib.SequenceMatcher(None, r['template'], p['template']).ratio() < 0.75 for p in picks):
            picks.append(r)
            conditions[f'opt2_top{len(picks)}'] = dict(template=r['template'], source='dsevo2 exact rank', **{k: r[k] for k in (
                'joint', 'agentdojo_mean', 'ipi_mean', 'suite_means', 'wrapper_tokens', 'fast_fitness', 'evaluation_index')})
    target = read_json(args.evolution / 'delimiter_target.json')
    for name, template in REFERENCES.items():
        rows_ = next(v for k, v in target.items() if k.split(' ')[0] == name)
        ad = [v['goal_user'] for v in rows_.values() if v['suite'] in AD]
        ipi = [v['goal_user'] for v in rows_.values() if v['suite'] not in AD]
        conditions[name] = dict(template=template, allow_special=True, source='genuine delimiter reference',
                                joint=float(np.mean(ad + ipi)), agentdojo_mean=float(np.mean(ad)), ipi_mean=float(np.mean(ipi)))
    write_json(args.out / 'conditions.json', conditions)
    print(json.dumps({k: dict(joint=round(v['joint'], 4), agentdojo=round(v['agentdojo_mean'], 4), ipi=round(v['ipi_mean'], 4))
                      for k, v in conditions.items()}), flush=True)
    if args.skip_ipi36:
        return
    # exact P(user) on all primary IPI behaviors
    builder = scorer.builder
    primary = [c for c in builder.ipi_all.values() if c['primary']]
    measure = {name: (lambda case, t=spec['template']: builder.render(case, t)) for name, spec in conditions.items()}
    if args.extra_conditions:
        for name, spec in read_json(args.extra_conditions).items():
            measure[name] = (lambda case, t=spec['template']: builder.render(case, t))
    measure['true_user_turn'] = builder.render_true_user
    if args.ipi36_conditions:
        measure = {name: measure[name] for name in args.ipi36_conditions}
    path = args.out / 'ipi36_scores.json'
    scores = read_json(path) if path.exists() else {}
    user = ROLES.index('user')
    call('drop')  # release the surrogate prefix caches: long IPI contexts need the memory
    for name, render in measure.items():
        rows_ = scores.setdefault(name, {})
        for case in primary:
            if case['case_id'] in rows_:
                continue
            request = render(case)
            positions = request['payload_token_indices']
            # causal model: tokens after the payload cannot change its features, so the prompt is cut there
            input_ids = request['input_ids'][:max(positions) + 1]
            if len(input_ids) > args.ipi36_max_tokens:  # the indexer score matrix of longer prompts exceeds GPU memory
                rows_[case['case_id']] = dict(category=case['suite'], skipped='too_long', tokens=len(input_ids))
                write_json(path, scores)
                continue
            result, seconds = call('full', input_ids=input_ids, positions=positions)
            probabilities = np.asarray(result['probabilities'])
            mask = np.isin(positions, request['goal_token_indices'])
            rows_[case['case_id']] = dict(category=case['suite'], goal_user=float(probabilities[mask, user].mean()),
                                          goal_roles=dict(zip(ROLES, map(float, probabilities[mask].mean(0)))), seconds=seconds)
            write_json(path, scores)
        done = [r for r in rows_.values() if 'goal_user' in r]
        print(json.dumps(dict(stage='ipi36', condition=name, measured=len(done), mean=round(float(np.mean([r['goal_user'] for r in done])), 4),
                              per_category={c: round(float(np.mean([r['goal_user'] for r in done if r['category'] == c])), 4)
                                            for c in ('browser', 'coding', 'tool')})), flush=True)


if __name__ == '__main__':
    main()
