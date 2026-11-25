"""Select fixed scoring contexts from a finished Qwen3.5 AgentDojo run (CPU only).

Each context is the request of the first model call that saw the injection in a real Qwen3.5
episode.  Per suite, contexts are spread over distinct injection vectors and injection tasks and
prefer short prompts (cheaper forwards).  The replay of every selected context is checked to
reproduce the recorded request exactly.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import ds_agentdojo
import ds_contexts
from ds_common import Codec, MODEL, read_json, sha, write_json

SUITES = ('banking', 'slack', 'travel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--condition', default='original')
    parser.add_argument('--per-suite', type=int, default=2)
    parser.add_argument('--max-prompt-tokens', type=int, default=12000)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ipi-cases', default='empty-box-listing,salesforce-unauthorized-admin-creation,stealth-dependency-append,'
                        'free-the-agents,modify-cicd-workflow,hotel-frontdesk-premature-checkout',
                        help='IPI Arena panel behaviors added to the objective (empty string: AgentDojo only)')
    args = parser.parse_args()
    template = read_json(args.run / 'conditions.json')[args.condition]['template']
    deps = ds_agentdojo.dependencies()
    suites = {name: deps['get_suite'](ds_agentdojo.BENCHMARK_VERSION, name) for name in SUITES}
    codec = Codec(MODEL)
    pool, stats = defaultdict(list), defaultdict(lambda: dict(complete=0, exposed=0))
    for path in sorted((args.run / 'cases' / args.condition).glob('*/*/*.json')):
        record = read_json(path)
        if record['status'] != 'complete':
            continue
        stats[record['suite']]['complete'] += 1
        context = ds_contexts.context_from_record(record)
        if context is None:
            continue
        stats[record['suite']]['exposed'] += 1
        context.update(source_template=template, source_case_sha256=sha(path),
                       prompt_tokens=record['calls'][context['exposure_call']]['prompt_tokens'],
                       exposing_tools=sorted({m.get('name') for m in context['messages'] if m['role'] == 'tool'
                                              and record['goal'][:40] in (m['content'] or '')}))
        if context['prompt_tokens'] <= args.max_prompt_tokens:
            pool[record['suite']].append(context)
    selected = []
    for suite in SUITES:
        groups = defaultdict(list)
        for context in pool[suite]:
            groups[(tuple(context['injection_vectors']), tuple(context['exposing_tools']))].append(context)
        order = sorted(groups, key=lambda k: (-len(groups[k]), k))
        used_injections, used_users, chosen, round_index = set(), set(), [], 0
        while len(chosen) < args.per_suite and round_index < 50:
            for key in order:
                candidates = sorted(groups[key], key=lambda c: (c['injection_task_id'] in used_injections,
                                                                c['user_task_id'] in used_users, c['prompt_tokens'], c['case_id']))
                candidates = [c for c in candidates if c not in chosen]
                if not candidates or len(chosen) >= args.per_suite:
                    continue
                pick = candidates[0]
                chosen.append(pick)
                used_injections.add(pick['injection_task_id'])
                used_users.add(pick['user_task_id'])
            round_index += 1
        assert len(chosen) == args.per_suite, f'not enough exposed contexts in {suite}'
        selected += chosen
    for context in selected:
        messages, _ = ds_contexts.replay(deps, suites[context['suite']], context, template)
        assert messages == context['messages'], context['case_id']
        location = ds_contexts.locate(codec, messages, context['tools'], template, context['goal'])
        context['goal_tokens'] = len(location['goal_token_indices'])
        assert len(location['input_ids']) == context['prompt_tokens'], context['case_id']
    ipi_cases, ipi_goals, categories = [c for c in args.ipi_cases.split(',') if c], {}, []
    if ipi_cases:
        import ds_ipi
        _, _, goals_module, _ = ds_ipi.ipi_modules()
        behaviors = goals_module.load_behaviors()
        goals, _ = goals_module.build_goal_manifest(behaviors)
        ipi_goals = {c: goals[c]['goal'] for c in ipi_cases}
        categories = [c for c in ('browser', 'coding', 'tool') if any(behaviors[k]['category'] == c for k in ipi_cases)]
    dataset = dict(schema='deepseek-agentdojo-contexts-v1', suites=list(SUITES) + categories, ipi_cases=ipi_cases, ipi_goals=ipi_goals,
                   source_run=str(args.run),
                   source_condition=args.condition, source_template=template, per_suite=args.per_suite,
                   exposure_stats={k: dict(v) for k, v in stats.items()}, contexts=selected)
    write_json(args.out, dataset)
    for c in selected:
        print(json.dumps(dict(case=c['case_id'], vectors=c['injection_vectors'], tools=c['exposing_tools'],
                              prompt_tokens=c['prompt_tokens'], goal_tokens=c['goal_tokens'], call=c['exposure_call'])))
    print(json.dumps(dict(exposure=dataset['exposure_stats'])))


if __name__ == '__main__':
    main()
