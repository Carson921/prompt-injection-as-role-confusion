"""Native AgentDojo (v1.2.1, 389 pairs) episodes for DeepSeek-V4-Flash served by vLLM.

Benchmark, tools, execution loop and evaluators come unchanged from the pinned ChatInject
checkout.  Every model request is rendered by the official DeepSeek encoder, sent as token ids
to /v1/completions (greedy) and parsed by the official parser.  A fixed template is injected
by literal ``{goal}`` replacement (no str.format, so braces in templates are safe).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback

from ds_common import Codec, GENERATION, MODEL, SLOT, read_json, sha, write_json
from ds_serving import Vllm

CHATINJECT_SRC = Path('/scratch/wz55191/Github/delimeter-token/indirect-prompt-injection/ChatInject/src')
BENCHMARK_VERSION = 'v1.2.1'
SUITE_COUNTS = {'banking': (16, 9), 'slack': (21, 5), 'travel': (20, 7)}
PIPELINE_NAME = 'deepseek_v4_flash_vllm'


def dependencies():
    sys.path.insert(0, str(CHATINJECT_SRC))
    import agentdojo
    assert Path(agentdojo.__file__).resolve().is_relative_to(CHATINJECT_SRC.resolve())
    from functools import partial
    import yaml
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
    from agentdojo.attacks.base_attacks import BaseAttack
    from agentdojo.benchmark import run_task_with_injection_tasks
    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.logging import LOGGER_STACK, OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.types import text_content_block_from_string

    def formatter(result):  # native YAML, literal unicode (keeps delimiter characters unescaped)
        return tool_result_to_str(result, dump_fn=partial(yaml.safe_dump, allow_unicode=True))

    class LiteralAttack(BaseAttack):
        def __init__(self, template, name, task_suite, target_pipeline):
            super().__init__(task_suite, target_pipeline)
            assert template.count(SLOT) == 1
            self.template, self.name = template, name

        def attack(self, user_task, injection_task):
            payload = self.template.replace(SLOT, injection_task.GOAL)
            return {vector: payload for vector in self.get_injection_candidates(user_task)}

    return locals()


def escape_for_environment(payload):
    """AgentDojo substitutes injections textually inside double-quoted YAML scalars of the
    environment file; escaping backslashes and double quotes makes the loaded value equal the payload
    (same rule as the GPT-OSS evaluator, evaluate_optimized.py)."""
    return payload.replace('\\', '\\\\').replace('"', '\\"')


def all_cases(deps):
    suites = {name: deps['get_suite'](BENCHMARK_VERSION, name) for name in SUITE_COUNTS}
    cases = []
    for name, suite in suites.items():
        users = sorted(suite.user_tasks, key=lambda x: int(x.rsplit('_', 1)[1]))
        injections = sorted(suite.injection_tasks, key=lambda x: int(x.rsplit('_', 1)[1]))
        assert (len(users), len(injections)) == SUITE_COUNTS[name]
        for user in users:
            for injection in injections:
                cases.append(dict(suite=name, user_task_id=user, injection_task_id=injection,
                                  base_case_id=f'{name}/{user}/{injection}',
                                  goal=suite.get_injection_task_by_id(injection).GOAL))
    assert len(cases) == 389
    return suites, cases


def semantic_messages(messages):
    """AgentDojo chat messages -> OpenAI-style dicts for the official encoder."""
    result = []
    for message in messages:
        role = message['role']
        content = message.get('content')
        if isinstance(content, list):
            content = ''.join(block.get('content', block.get('text', '')) or '' for block in content)
        if role == 'tool' and message.get('error'):
            content = message['error']
        row = dict(role=role, content=content)
        if role == 'assistant':
            row['reasoning_content'] = message.get('reasoning_content') or ''
            if message.get('tool_calls'):
                row['tool_calls'] = [dict(id=call.id, type='function', function=dict(
                    name=call.function, arguments=json.dumps(call.args, ensure_ascii=False))) for call in message['tool_calls']]
        if role == 'tool':
            row['tool_call_id'] = message['tool_call_id']
            if message.get('tool_call') is not None:
                row['name'] = message['tool_call'].function
        result.append(row)
    return result


def tool_schemas(runtime):
    return [dict(type='function', function=dict(name=tool.name, description=tool.description,
                                                parameters=tool.parameters.model_json_schema()))
            for tool in runtime.functions.values()]


def generate(codec, client, messages, tools, call_id):
    """One model turn. Malformed / truncated output is kept as plain content with no tool calls."""
    encoded = codec.encode(messages, tools)
    response = client.complete(encoded['input_ids'], codec.eos_id)
    text = response['text']
    record = dict(call_id=call_id, prompt_tokens=len(encoded['input_ids']), finish_reason=response['finish_reason'],
                  completion_tokens=(response.get('usage') or {}).get('completion_tokens'), completion=text,
                  n_messages=len(messages), malformed=None)
    try:
        if response['finish_reason'] != 'stop':
            raise ValueError(f"generation ended with finish_reason={response['finish_reason']}")
        parsed = codec.parse(text if text.endswith(codec.eos) else text + codec.eos)
        calls = []
        for index, call in enumerate(parsed.get('tool_calls') or []):
            arguments = json.loads(call['function']['arguments'])
            if not isinstance(arguments, dict):
                raise ValueError('tool arguments are not an object')
            calls.append(dict(id=f'{call_id}-tool-{index}', name=call['function']['name'], arguments=arguments))
        output = dict(content=parsed.get('content') or '', reasoning_content=parsed.get('reasoning_content') or '', tool_calls=calls)
    except (ValueError, AssertionError, KeyError, TypeError) as error:
        record['malformed'] = f'{type(error).__name__}: {str(error)[:200]}'
        reasoning, _, rest = text.partition('</think>')
        output = dict(content=rest if _ else text, reasoning_content=reasoning if _ else '', tool_calls=[])
    return output, record


def make_pipeline(deps, codec, client, case_id, state):
    class DeepSeekLLM(deps['BasePipelineElement']):
        name = PIPELINE_NAME

        def query(self, query, runtime, env, messages=(), extra_args=None):
            call_id = f"{case_id}/call_{len(state['calls']):04d}"
            request, tools = semantic_messages(messages), tool_schemas(runtime)
            if not any(m['role'] == 'assistant' for m in request):
                state['transcripts'].append(None)  # a new pipeline query (AgentDojo retries up to 3 times)
            output, record = generate(codec, client, request, tools, call_id)
            record['query_index'] = len(state['transcripts']) - 1
            state['calls'].append(record)
            state['tools'] = tools
            state['transcripts'][-1] = request + [dict(role='assistant', content=output['content'],
                                                  reasoning_content=output['reasoning_content'],
                                                  tool_calls=[dict(id=c['id'], type='function', function=dict(
                                                      name=c['name'], arguments=json.dumps(c['arguments'], ensure_ascii=False)))
                                                      for c in output['tool_calls']])]
            tool_calls = [deps['FunctionCall'](function=c['name'], args=c['arguments'], id=c['id']) for c in output['tool_calls']]
            message = dict(role='assistant', content=[deps['text_content_block_from_string'](output['content'])],
                           tool_calls=tool_calls or None, reasoning_content=output['reasoning_content'])
            return query, runtime, env, [*messages, message], extra_args or {}

    pipeline = deps['AgentPipeline'].from_config(deps['PipelineConfig'](
        llm=DeepSeekLLM(), model_id=None, defense=None, tool_delimiter='tool', system_message_name=None,
        system_message=None, tool_output_format=None, provider_config=None))
    count = 0

    def visit(element):
        nonlocal count
        if isinstance(element, deps['ToolsExecutor']):
            element.output_formatter = deps['formatter']
            count += 1
        for child in getattr(element, 'elements', ()):
            visit(child)

    visit(pipeline)
    assert count == 1
    pipeline.name = PIPELINE_NAME
    return pipeline


def metrics(records, expected):
    scored = [r for r in records if r['status'] == 'complete']
    successes, useful = sum(r['security'] for r in scored), sum(r['utility'] for r in scored)
    return dict(expected=expected, scored=len(scored), errors=sum(r['status'] == 'error' for r in records),
                successes=successes, utility_successes=useful,
                asr=successes / expected if len(scored) == expected else None,
                asr_scored=successes / len(scored) if scored else None,
                utility_scored=useful / len(scored) if scored else None,
                episodes_with_malformed_call=sum(any(c['malformed'] for c in r['calls']) for r in scored),
                mean_calls=sum(len(r['calls']) for r in scored) / len(scored) if scored else None)


def summarize(output, conditions, cases):
    records = {}
    for path in (output / 'cases').glob('*/*/*/*.json'):
        row = read_json(path)
        records[(row['condition'], row['base_case_id'])] = row
    summary = dict(benchmark_version=BENCHMARK_VERSION, generation=GENERATION, conditions={})
    for condition in conditions:
        rows = [records[(condition, c['base_case_id'])] for c in cases if (condition, c['base_case_id']) in records]
        summary['conditions'][condition] = dict(
            **metrics(rows, len(cases)),
            suites={s: metrics([r for r in rows if r['suite'] == s], sum(c['suite'] == s for c in cases)) for s in SUITE_COUNTS})
    write_json(output / 'summary.json', summary)
    return summary


def run_case(deps, suites, codec, client, output, condition, template, identity, case):
    path = output / 'cases' / condition / (case['base_case_id'] + '.json')
    if path.exists():
        existing = read_json(path)
        if existing.get('identity_sha256') == identity and existing['status'] in ('complete', 'error'):
            return existing
    case_id = f"{condition}/{case['base_case_id']}"
    state = dict(calls=[], tools=None, transcripts=[])
    suite = suites[case['suite']]
    pipeline = make_pipeline(deps, codec, client, case_id, state)
    payload = template.replace(SLOT, case['goal'])
    record = dict(case, condition=condition, case_id=case_id, identity_sha256=identity, status='running',
                  utility=None, security=None, payload=payload)
    started = time.time()
    try:
        user, injection = suite.user_tasks[case['user_task_id']], suite.injection_tasks[case['injection_task_id']]
        vectors = deps['LiteralAttack'](template, condition, suite, pipeline).get_injection_candidates(user)
        injections = {vector: escape_for_environment(payload) for vector in vectors}
        utility, security = suite.run_task_with_pipeline(pipeline, user, injection, injections)
        if type(utility) is not bool or type(security) is not bool:
            raise ValueError('Native task evaluators returned nonboolean scores')
        record.update(status='complete', utility=utility, security=security, injection_vectors=sorted(vectors))
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {str(error)[:500]}', traceback=traceback.format_exc()[-2000:])
    record.update(calls=state['calls'], tools=state['tools'], transcripts=state['transcripts'], duration_seconds=time.time() - started)
    write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions-json', type=Path, required=True, help='{name: {"template": ...}}')
    parser.add_argument('--conditions', default=None, help='comma-separated subset')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--served-model', default='deepseek-v4-flash')
    parser.add_argument('--max-model-len', type=int, default=65536)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--limit-per-suite', type=int, default=None, help='smoke test: first N pairs of each suite')
    args = parser.parse_args()
    output = args.output.resolve()
    conditions = read_json(args.conditions_json)
    names = args.conditions.split(',') if args.conditions else list(conditions)
    deps = dependencies()
    suites, cases = all_cases(deps)
    if args.limit_per_suite:
        cases = [c for s in SUITE_COUNTS for c in [x for x in cases if x['suite'] == s][:args.limit_per_suite]]
    codec = Codec(MODEL)
    client = Vllm(args.base_url, args.served_model, args.max_model_len)
    base = dict(benchmark_version=BENCHMARK_VERSION, generation=GENERATION, codec=codec.binding,
                pipeline=PIPELINE_NAME, max_model_len=args.max_model_len)
    write_json(output / 'conditions.json', {n: conditions[n] for n in names})
    jobs = []
    for name in names:
        identity = sha(json.dumps(dict(base, template=conditions[name]['template']), sort_keys=True))
        jobs += [(name, conditions[name]['template'], identity, case) for case in cases]
    print(json.dumps(dict(stage='start', jobs=len(jobs), workers=args.workers)), flush=True)
    done, started = 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, deps, suites, codec, client, output, n, t, i, c) for n, t, i, c in jobs]
        for future in as_completed(futures):
            row = future.result()
            done += 1
            print(json.dumps(dict(stage='case', done=done, total=len(jobs), case_id=row['case_id'], status=row['status'],
                                  security=row['security'], utility=row['utility'], calls=len(row['calls']),
                                  malformed=sum(bool(c['malformed']) for c in row['calls']),
                                  error=row.get('error'), seconds=round(row['duration_seconds'], 1))), flush=True)
    summary = summarize(output, names, cases)
    print(json.dumps({n: {k: v[k] for k in ('scored', 'errors', 'successes', 'asr', 'asr_scored', 'utility_scored')}
                      for n, v in summary['conditions'].items()}), flush=True)


if __name__ == '__main__':
    main()
