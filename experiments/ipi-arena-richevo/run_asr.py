"""Run every prepared episode against the local vLLM gpt-oss-120b service.

Target model: OpenAI-compatible vLLM service (temperature 0, max_tokens 4096,
up to five target steps, as in ipi_arena_bench.runner).  WorldSim: OpenRouter
google/gemini-2.5-flash through the repository's own simulate_tool_response.
Judging: only the repository's deterministic tool judge (judge_tool_usage) on
every tool call the target made.  Tool calls are recorded, never executed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback

import httpx
import openai

from common import HERE, MODEL, read_json, sha, write_json
from goals import load_behaviors

MAX_STEPS = 5
TARGET_MAX_TOKENS = 4096
WORLDSIM_MAX_TOKENS = 2048


def openrouter_key(path=Path.home() / '.bashrc') -> str:
    for line in path.read_text().splitlines():
        match = re.match(r'\s*export\s+(OPENROUTER_API_KEY|OPENROUTER_KEY)\s*=\s*["\']?([^"\'\s]+)', line)
        if match:
            return match.group(2)
    raise ValueError('OPENROUTER_KEY not found in ~/.bashrc')


class Target:
    """Thin OpenAI-compatible client that also records finish_reason and reasoning."""

    def __init__(self, base_url, model):
        self.client = openai.OpenAI(base_url=base_url, api_key='EMPTY', timeout=1800, max_retries=2)
        self.model = model
        self.root = base_url.rsplit('/v1', 1)[0]

    def chat(self, messages, tools):
        kwargs = dict(model=self.model, messages=messages, temperature=0.0, max_tokens=TARGET_MAX_TOKENS, seed=0)
        if tools:
            kwargs.update(tools=tools, tool_choice='auto')
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls = []
        for tc in message.tool_calls or []:
            raw = tc.function.arguments
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError):
                args = {'_raw': raw}
            if not isinstance(args, dict):
                args = {'_raw': raw}
            calls.append(dict(id=tc.id or '', name=tc.function.name, arguments=args, raw_arguments=raw))
        reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
        usage = {}
        if response.usage:
            usage = dict(prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens)
        return dict(content=message.content, reasoning=reasoning, tool_calls=calls, finish_reason=choice.finish_reason,
                    usage=usage, model=response.model)

    def tokenize(self, messages, tools):
        body = dict(model=self.model, messages=messages, add_generation_prompt=True)
        if tools:
            body['tools'] = tools
        response = httpx.post(self.root + '/tokenize', json=body, timeout=120)
        response.raise_for_status()
        return response.json()['tokens']


def blocked_by_provider_text(text: str) -> bool:
    return ('NoneType' in text and 'subscriptable' in text) or 'PROHIBITED_CONTENT' in text \
        or 'content_policy' in text or 'Error code: 403' in text


def blocked_by_provider(error: Exception) -> bool:
    return blocked_by_provider_text(f'{type(error).__name__}: {error}')


def api_messages(messages):
    """Prepared (normalised) messages -> OpenAI API shape (arguments as JSON strings)."""
    result = []
    for message in messages:
        row = dict(role=message['role'], content=message.get('content', ''))
        if message.get('tool_calls'):
            row['tool_calls'] = [dict(id=c['id'], type='function', function=dict(
                name=c['function']['name'], arguments=json.dumps(c['function']['arguments'])))
                for c in message['tool_calls']]
            if row['content'] == '':
                row['content'] = None
        if message['role'] == 'tool':
            row['tool_call_id'] = message.get('tool_call_id', '')
        result.append(row)
    return result


def run_episode(request, behavior, target, worldsim, output, identity):
    from ipi_arena_bench.behavior import build_tools
    from ipi_arena_bench.judges.tool_usage import judge_tool_usage
    from ipi_arena_bench.llm_client import ToolCall
    from ipi_arena_bench.worldsim import build_worldsim_prompt, simulate_tool_response
    started = time.time()
    condition, case_id = request['condition'], request['case_id']
    tools = build_tools(behavior)
    messages = api_messages(request['messages'])
    record = dict(status='running', identity_sha256=identity, condition=condition, case_id=case_id,
                  category=behavior['category'], primary=request['primary'], request_sha256=request['_sha256'],
                  goal=request['goal'], steps=[], tool_calls_made=[], error=None)
    try:
        vllm_ids = target.tokenize(messages, tools)
        record['tokenize'] = dict(vllm_prompt_tokens=len(vllm_ids), prepared_prompt_tokens=len(request['prompt_token_ids']),
                                  match=vllm_ids == request['prompt_token_ids'],
                                  first_mismatch=next((i for i, (a, b) in enumerate(zip(vllm_ids, request['prompt_token_ids'])) if a != b), None))
    except Exception as error:  # the comparison is diagnostic only
        record['tokenize'] = dict(error=f'{type(error).__name__}: {error}')
    all_calls = []
    world_prompt = build_worldsim_prompt(tools, behavior, custom_prompt=behavior.get('world_sim_prompt')) if tools else None
    try:
        final_content, terminal = None, None
        for step in range(1, MAX_STEPS + 1):
            response = target.chat(messages, tools)
            calls = [ToolCall(name=c['name'], arguments=c['arguments'], id=c['id']) for c in response['tool_calls']]
            all_calls.extend(calls)
            record['steps'].append(dict(step=step, finish_reason=response['finish_reason'], usage=response['usage'],
                                        content=response['content'], reasoning=response['reasoning'],
                                        tool_calls=response['tool_calls'], worldsim=[]))
            assistant = dict(role='assistant', content=response['content'])
            if calls:
                assistant['tool_calls'] = [dict(id=c.id, type='function', function=dict(name=c.name, arguments=json.dumps(c.arguments)))
                                           for c in calls]
            messages.append(assistant)
            if not calls:
                final_content = response['content']
                terminal = 'final_response' if response['finish_reason'] != 'length' else 'max_tokens_without_tool_call'
                break
            if step == MAX_STEPS:
                terminal = 'max_steps'
                break
            for call in calls:
                simulated, sim_error, fallback = None, None, False
                for attempt in range(4):
                    try:
                        simulated = simulate_tool_response(worldsim, world_prompt, call, messages)
                        break
                    except Exception as error:
                        sim_error = f'{type(error).__name__}: {error}'
                        if blocked_by_provider(error):
                            break
                        time.sleep(3 * (attempt + 1))
                if simulated is None and sim_error and blocked_by_provider_text(sim_error):
                    # Provider content policy block (OpenRouter returns an error object with no
                    # choices).  Use the repository's documented empty-response fallback.
                    simulated, fallback = '{"status": "success"}', True
                    record.setdefault('worldsim_fallbacks', []).append(dict(step=step, tool=call.name, error=sim_error))
                if simulated is None:
                    raise RuntimeError(f'WorldSim failed: {sim_error}')
                record['steps'][-1]['worldsim'].append(dict(tool=call.name, response=simulated, fallback=fallback))
                messages.append(dict(role='tool', content=simulated, tool_call_id=call.id))
        judge = judge_tool_usage(all_calls, behavior['tool_judge'])
        record.update(status='complete', success=bool(judge['passed']), tool_judge=judge, terminal_reason=terminal,
                      final_response=final_content, n_steps=len(record['steps']))
    except Exception as error:
        record.update(status='error', success=None, error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    record['tool_calls_made'] = [c.to_dict() for c in all_calls]
    record['messages'] = messages
    record['duration_seconds'] = time.time() - started
    write_json(output / 'cases' / condition / f'{case_id}.json', record)
    return record


def summarize(output, protocol):
    cases = {}
    for path in (output / 'cases').glob('*/*.json'):
        row = read_json(path)
        cases[(row['condition'], row['case_id'])] = row
    conditions = read_json(output / 'conditions.json')
    summary = dict(conditions={}, cases=len(cases))
    for condition, spec in conditions.items():
        rows = [r for (c, _), r in cases.items() if c == condition]
        primary = [r for r in rows if r['primary']]
        done = [r for r in primary if r['status'] == 'complete']
        successes = sum(r['success'] for r in done)
        per_category = {}
        for category in ('browser', 'coding', 'tool'):
            sub = [r for r in done if r['category'] == category]
            per_category[category] = dict(scheduled=sum(r['category'] == category for r in primary),
                                          complete=len(sub), successes=sum(r['success'] for r in sub),
                                          asr=(sum(r['success'] for r in sub) / len(sub)) if sub else None)
        supplementary = [r for r in rows if not r['primary'] and r['status'] == 'complete']
        summary['conditions'][condition] = dict(
            p_user_agentdojo=spec['p_user_agentdojo'], asr_agentdojo=spec['asr_agentdojo'],
            scheduled=sum(1 for c in protocol['rows'] if c['condition'] == condition and c['primary']),
            complete=len(done), errors=sum(r['status'] == 'error' for r in primary), successes=successes,
            asr=successes / len(done) if done else None,
            asr_over_scheduled=successes / max(1, sum(1 for c in protocol['rows'] if c['condition'] == condition and c['primary'])),
            per_category=per_category,
            supplementary=dict(complete=len(supplementary), successes=sum(r['success'] for r in supplementary)),
            tokenize_match=sum(bool(r.get('tokenize', {}).get('match')) for r in rows),
            worldsim_fallback_episodes=sum(bool(r.get('worldsim_fallbacks')) for r in rows),
            mean_steps=sum(r.get('n_steps', 0) for r in done) / len(done) if done else None)
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='gpt-oss-120b')
    parser.add_argument('--worldsim-model', default='google/gemini-2.5-flash')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--conditions', nargs='*')
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read_json(output / 'protocol.json')
    behaviors = load_behaviors()
    from ipi_arena_bench.llm_client import LLMClient
    worldsim = LLMClient(provider='openrouter', model=args.worldsim_model, api_key=openrouter_key())
    target = Target(args.base_url, args.model)
    identity = sha(json.dumps(dict(protocol_sha256=sha(output / 'protocol.json'), target=args.model, base_url=args.base_url,
                                   worldsim=args.worldsim_model, temperature=0, max_tokens=TARGET_MAX_TOKENS,
                                   worldsim_max_tokens=WORLDSIM_MAX_TOKENS, max_steps=MAX_STEPS,
                                   worldsim_block_fallback='{"status": "success"}'), sort_keys=True))
    write_json(output / 'asr_identity.json', dict(identity_sha256=identity, target=args.model, base_url=args.base_url,
                                                  worldsim=args.worldsim_model, temperature=0, seed=0,
                                                  max_tokens=TARGET_MAX_TOKENS, max_steps=MAX_STEPS,
                                                  judge='ipi_arena_bench.judges.tool_usage.judge_tool_usage only'))
    pending = []
    for row in protocol['rows']:
        if args.conditions and row['condition'] not in args.conditions:
            continue
        if args.cases and row['case_id'] not in args.cases:
            continue
        case_path = output / 'cases' / row['condition'] / f"{row['case_id']}.json"
        if case_path.exists():
            existing = read_json(case_path)
            if existing.get('status') == 'complete' and existing.get('identity_sha256') == identity:
                continue
        request = read_json(output / row['request'])
        request['_sha256'] = row['request_sha256']
        pending.append(request)
    print(json.dumps(dict(stage='start', pending=len(pending), workers=args.workers)), flush=True)
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_episode, r, behaviors[r['case_id']], target, worldsim, output, identity): r for r in pending}
        for future in as_completed(futures):
            record = future.result()
            with lock:
                done += 1
                print(json.dumps(dict(stage='case', done=done, total=len(pending), condition=record['condition'],
                                      case_id=record['case_id'], status=record['status'], success=record.get('success'),
                                      steps=record.get('n_steps'), tokenize_match=record.get('tokenize', {}).get('match'),
                                      error=record.get('error'), seconds=round(record['duration_seconds'], 1))), flush=True)
    summary = summarize(output, protocol)
    print(json.dumps({k: dict(asr=v['asr'], complete=v['complete'], errors=v['errors']) for k, v in summary['conditions'].items()}), flush=True)


if __name__ == '__main__':
    main()
