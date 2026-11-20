"""Why is goal P(user) low on IPI Arena even in a genuine user turn?  2x2 swap of goal text and context.

Every prompt is a normal conversation: uninjected tool result, assistant reply "Done.", then a genuine
user turn.  Measured (exact full forward): P(user) on that final user turn's tokens and on the ORIGINAL
user request of the same conversation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ds_common import ROLES, SLOT, read_json, write_json
from ds_probe_client import Builder, call

USER = ROLES.index('user')


def span(encoded, text, start=0):
    begin = encoded['prompt'].find(text, start)
    assert begin >= 0, text[:60]
    end = begin + len(text)
    return [i for i, (a, b) in enumerate(encoded['offsets']) if b > a and a < end and b > begin]


def build(builder, case, final_text):
    if 'behavior' in case:
        messages, _ = builder.ipi_common.build_text_messages(case['behavior'], '')
        messages = builder.ipi.encoder_messages(messages)
    else:
        messages, _ = builder.contexts_module.replay(builder.deps, builder.suites[case['suite']], dict(case, goal=''), SLOT)
    original = next(m['content'] for m in messages if m['role'] == 'user' and isinstance(m['content'], str) and m['content'].strip())
    messages = messages + [dict(role='assistant', content='Done.', reasoning_content=''), dict(role='user', content=final_text)]
    encoded = builder.codec.encode(messages, case['tools'])
    final = span(encoded, final_text, encoded['prompt'].rfind(final_text))
    first = span(encoded, original[:300])
    return encoded['input_ids'], final, first, original


def main():
    dataset, out = read_json(sys.argv[1]), Path(sys.argv[2])
    builder = Builder(dataset)
    ad, ipi = builder.contexts, builder.ipi_cases
    results = read_json(out) if out.exists() else {}
    jobs = []
    for i, case in enumerate(ad):
        jobs += [(case, 'own_goal', case['goal']), (case, 'swapped_goal', ipi[i % len(ipi)]['goal'])]
    for i, case in enumerate(ipi):
        jobs += [(case, 'own_goal', case['goal']), (case, 'swapped_goal', ad[i % len(ad)]['goal']),
                 (case, 'own_goal_capitalized', case['goal'][0].upper() + case['goal'][1:] + '.'),
                 (case, 'leadin_own_goal', 'Wait, can you instead ' + case['goal'])]
    for case, name, text in jobs:
        key = f"{case['case_id']}|{name}"
        if key in results:
            continue
        ids, final, first, original = build(builder, case, text)
        positions = sorted(set(final) | set(first))
        result, seconds = call('full', input_ids=ids, positions=positions)
        p = np.asarray(result['probabilities'])
        fm, om = np.isin(positions, final), np.isin(positions, first)
        results[key] = dict(case_id=case['case_id'], suite=case['suite'], condition=name, final_text=text[:120], original_request=original[:120],
                            final_user=float(p[fm, USER].mean()), final_roles=dict(zip(ROLES, map(float, p[fm].mean(0)))),
                            final_user_per_token=[round(float(v), 3) for v in p[fm, USER]],
                            original_request_user=float(p[om, USER].mean()), original_roles=dict(zip(ROLES, map(float, p[om].mean(0)))),
                            input_tokens=len(ids), seconds=seconds)
        write_json(out, results)
        r = results[key]
        print(f"{case['case_id'][:36]:36s} {name:22s} final {r['final_user']:.2f} original-request {r['original_request_user']:.2f} | {text[:60]}", flush=True)
    for bench, suites in (('agentdojo', ('banking', 'slack', 'travel')), ('ipi', ('browser', 'coding', 'tool'))):
        for name in ('own_goal', 'swapped_goal', 'own_goal_capitalized', 'leadin_own_goal'):
            rows = [r for r in results.values() if r['suite'] in suites and r['condition'] == name]
            if rows:
                print(json.dumps(dict(context=bench, condition=name, final_user=round(float(np.mean([r['final_user'] for r in rows])), 3),
                                      original_request_user=round(float(np.mean([r['original_request_user'] for r in rows])), 3))))


if __name__ == '__main__':
    main()
