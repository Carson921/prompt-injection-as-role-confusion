"""Regenerate each scenario's trajectory on-policy with gpt-oss-120b (vLLM) until it matches the golden
task-domain tool-call sequence (paper footnote 2).  Tools of both domains are offered.

Writes <output>/trajectories.json: {scenario_id: {status, messages, match_level, attempts}} where
`messages` are normalised (tool-call arguments as dicts) and end with the final task-domain tool
response -- the injection site.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from common import read_json, write_json
from target import SYSTEM_PROMPT, Target, api_messages, openai_tools

ATTEMPTS = [(0.0, 'exact'), (0.7, 'exact'), (0.7, 'exact'), (0.7, 'name'), (0.7, 'name')]


def attempt_replay(entry: dict, target: Target, temperature: float, level: str, seed: int) -> dict:
    tools = openai_tools(entry['task_tools'] + entry['target_tools'])
    messages = [dict(role='system', content=SYSTEM_PROMPT), dict(role='user', content=entry['user_query'])]
    steps = []
    for k, turn in enumerate(entry['golden_turns']):
        response = target.chat(api_messages(messages), tools, temperature=temperature, seed=seed)
        calls = response['tool_calls']
        golden = turn['tool_call']
        steps.append(dict(step=k, calls=[dict(name=c['name'], arguments=c['arguments']) for c in calls],
                          content=response['content'], finish_reason=response['finish_reason']))
        if len(calls) != 1 or calls[0]['name'] != golden['name']:
            return dict(status='mismatch', step=k, reason='tool name', steps=steps)
        args_equal = calls[0]['arguments'] == golden['arguments']
        if level == 'exact' and not args_equal:
            return dict(status='mismatch', step=k, reason='arguments', steps=steps)
        steps[-1]['args_equal'] = args_equal
        call_id = calls[0]['id'] or f'call_{k}'
        messages.append(dict(role='assistant', content=response['content'] or '',
                             tool_calls=[dict(id=call_id, type='function', function=dict(name=calls[0]['name'], arguments=calls[0]['arguments']))]))
        messages.append(dict(role='tool', content=json.dumps(turn['tool_response']['content'], ensure_ascii=False), tool_call_id=call_id))
    return dict(status='match', messages=messages, steps=steps, match_level=level,
                args_equal_all=all(s.get('args_equal') for s in steps), temperature=temperature)


def replay(entry: dict, target: Target) -> dict:
    attempts = []
    for i, (temperature, level) in enumerate(ATTEMPTS):
        result = attempt_replay(entry, target, temperature, level, seed=i)
        attempts.append({k: v for k, v in result.items() if k != 'messages'} | dict(temperature=temperature, level=level))
        if result['status'] == 'match':
            return dict(status='match', messages=result['messages'], match_level=level, args_equal_all=result['args_equal_all'],
                        temperature=temperature, attempts=attempts)
    return dict(status='failed', attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    dataset = read_json(args.output / 'dataset.json')['scenarios']
    path = args.output / 'trajectories.json'
    done = read_json(path) if path.exists() else {}
    target = Target(args.base_url)
    pending = [e for e in dataset if e['scenario_id'] not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        for entry, result in zip(pending, pool.map(lambda e: replay(e, target), pending)):
            done[entry['scenario_id']] = result
            write_json(path, done)
            print(json.dumps(dict(scenario=entry['scenario_id'], status=result['status'], level=result.get('match_level'),
                                  args_equal=result.get('args_equal_all'), attempts=len(result['attempts']))), flush=True)
    matched = [k for k, v in done.items() if v['status'] == 'match']
    print(json.dumps(dict(scenarios=len(done), matched=len(matched),
                          exact=sum(done[k]['match_level'] == 'exact' for k in matched),
                          failed=[k for k, v in done.items() if v['status'] != 'match'])))


if __name__ == '__main__':
    main()
