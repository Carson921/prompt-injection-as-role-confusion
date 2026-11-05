"""Restore actual AgentDojo tool environments by verified native-output replay.

Every call before the saved first exposure is checked against its original
request and full model tokens by the producer. At first exposure, decoding is
continued with the frozen native engine, then the unmodified AgentDojo loop
and task predicates run to completion.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
import traceback

from common import BASE, OUTPUT, BENCHMARK_PYTHON, read, write, sha, serial, phase


def emit(value):
    sys.__stdout__.write(serial(value) + '\n')
    sys.__stdout__.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    source_identity = read(BASE / 'agentdojo-source.json')
    source = Path(source_identity['source'])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(BASE / 'provenance'))
    from bridge import parse_native
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.errors import AbortAgentError
    from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai, _function_to_openai
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.task_suite.load_suites import get_suites
    from agentdojo.task_suite.task_suite import model_output_from_messages, functions_stack_trace_from_messages
    from agentdojo.types import text_content_block_from_string
    import agentdojo
    assert Path(agentdojo.__file__).resolve().is_relative_to(source)
    assert all(sha(source / name) == expected for name, expected in source_identity['files'].items())
    suites = get_suites('v1.2.1')
    canonical = {r['case_id']: r for r in read(BASE / 'canonical_cases.json')}
    schedule = read(BASE / 'schedule.json')
    if args.max_cases:
        schedule = schedule[:args.max_cases]
    for case in schedule:
        case_id = case['case_id']
        origin = canonical[case_id]
        previous = read(BASE / origin['source_result'])
        result_path = output / 'cases' / (case_id + '.json')
        if result_path.exists() and read(result_path)['status'] == 'completed':
            emit({'event': 'case_reused', 'case_id': case_id})
            continue

        class RequestLLM(BasePipelineElement):
            name = 'gpt-oss-120b-native-first-payload-continuation'

            def __init__(self):
                self.requests = []

            def query(self, query, runtime, env, messages=(), extra_args=None):
                index = len(self.requests)
                call_id = case_id + f'/call_{index:04d}'
                model_messages = [_message_to_openai(m, 'gpt-oss-120b') for m in messages]
                for message in model_messages:
                    if isinstance(message.get('content'), list):
                        message['content'] = ''.join(b['text'] for b in message['content'])
                    if message.get('content') is None:
                        message['content'] = ''
                    for call in message.get('tool_calls', []):
                        if isinstance(call['function']['arguments'], str):
                            call['function']['arguments'] = json.loads(call['function']['arguments'])
                request = {**case, 'call_id': call_id, 'call_index': index,
                           'messages': model_messages, 'raw_agentdojo_messages': messages,
                           'tools': [_function_to_openai(t) for t in runtime.functions.values()],
                           'reasoning_effort': 'medium', 'max_tokens': 1024, 'version': 'v1.2.1'}
                path = output / 'calls' / call_id / 'original_request.json'
                write(path, request)
                item = {'call_id': call_id, 'call_index': index, 'request_sha256': sha(path)}
                self.requests.append(item)
                emit({'event': 'request', 'call_id': call_id, 'path': str(path)})
                line = sys.stdin.readline()
                if not line:
                    raise EOFError('Inference producer closed')
                response = json.loads(line)
                assert response['call_id'] == call_id
                if response.get('error'):
                    raise RuntimeError(response['error'])
                message = parse_native(response['generated_ids'], call_id)
                parsed_path = path.with_name('parsed_response.json')
                write(parsed_path, {'call_id': call_id, 'message': message,
                                   'native_generated_ids': response['generated_ids'],
                                   'source_response_sha256': response['response_sha256']})
                item.update(status=response['mode'], response_sha256=response['response_sha256'],
                            parsed_response_sha256=sha(parsed_path))
                calls = [FunctionCall(function=c['function']['name'], args=json.loads(c['function']['arguments'])
                         if isinstance(c['function']['arguments'], str) else c['function']['arguments'], id=c['id'])
                         for c in (message.get('tool_calls') or [])]
                text = message.get('content')
                assistant = {'role': 'assistant', 'content': [text_content_block_from_string(text)] if text else None,
                             'tool_calls': calls or None}
                return query, runtime, env, [*messages, assistant], extra_args or {}

        llm = RequestLLM()
        pipeline = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=None,
            tool_delimiter='tool', system_message_name=None, system_message=None, tool_output_format=None))

        class RecordingPipeline(BasePipelineElement):
            name = pipeline.name

            def __init__(self):
                self.invocations = []

            def query(self, query, runtime, env, messages=(), extra_args=None):
                index = len(self.invocations)
                pre = env.model_copy(deep=True)
                aborted = False
                try:
                    answer = pipeline.query(query, runtime, env, messages, extra_args or {})
                    final_env, final_messages = answer[2], answer[3]
                except AbortAgentError as error:
                    aborted = True
                    final_env, final_messages = error.task_environment, error.messages
                    raise
                finally:
                    if 'final_env' in locals():
                        path = output / 'trajectories' / case_id / f'pipeline_{index:02d}.json'
                        write(path, {'query': query, 'pre_environment': pre, 'final_environment': final_env,
                                     'messages': final_messages, 'aborted': aborted,
                                     'model_output': model_output_from_messages(final_messages),
                                     'functions_stack_trace': functions_stack_trace_from_messages(final_messages)})
                        self.invocations.append({'path': str(path.relative_to(output)), 'sha256': sha(path)})
                return answer

        recorder = RecordingPipeline()
        started = time.time()
        result = {'case_id': case_id, 'condition': case['condition'], 'suite': case['suite'],
                  'user_task_id': case['user_task_id'], 'injection_task_id': case['injection_task_id'],
                  'status': 'running', 'attacker_success': None, 'security': None, 'utility': None,
                  'base_case_status': origin['status'], 'source_result': origin['source_result'],
                  'source_result_sha256': sha(BASE / origin['source_result']),
                  'first_exposure_call': origin.get('first_exposure_call'),
                  'feature_row_start': origin.get('feature_row_start'), 'feature_row_end': origin.get('feature_row_end')}
        try:
            suite = suites[case['suite']]
            with redirect_stdout(sys.stderr):
                utility, security = suite.run_task_with_pipeline(recorder, suite.user_tasks[case['user_task_id']],
                    suite.injection_tasks[case['injection_task_id']], case['injections'])
            if origin['status'] == 'captured':
                assert len(llm.requests) >= previous['requests'][-1]['call_index'] + 1
                assert llm.requests[previous['requests'][-1]['call_index']]['status'] != 'prefix_replay'
            else:
                assert len(llm.requests) == len(previous['requests'])
                assert bool(utility) == previous['utility'] and bool(security) == previous['security']
                assert all(r['status'] == 'prefix_replay' for r in llm.requests)
            result.update(status='completed', attacker_success=bool(security), security=bool(security), utility=bool(utility))
        except Exception as error:
            result.update(status='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        result.update(requests=llm.requests, trajectory_invocations=recorder.invocations, seconds=time.time() - started)
        write(result_path, result)
        emit({'event': 'case_complete', 'case_id': case_id, 'status': result['status'],
              'attacker_success': result['attacker_success'], 'utility': result['utility'],
              'calls': len(llm.requests), 'seconds': result['seconds'], 'error': result.get('error')})
    emit({'event': 'all_cases_terminal', 'cases': len(schedule)})


if __name__ == '__main__':
    main()
