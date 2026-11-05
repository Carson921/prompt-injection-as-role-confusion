"""Compare payload conditions with one local vLLM engine and native AgentDojo.

Episode environments, tools, the 15-step loop, the three native task attempts,
and success predicates come from the frozen AgentDojo source. Initial errors
retain null scores and remain part of the reported scheduled denominator.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import date, datetime
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback

from prepare import COUNTS, DELT, ROOT, SOURCE, PLAIN, REMEMBER, read, sha

STOP_IDS = [200002, 200012]
BASE_BPE_BOUND = 199998
PROBE_ROLES = ('system', 'developer', 'user', 'cot', 'assistant', 'tool')


def serial(value):
    def default(item):
        if hasattr(item, 'model_dump'):
            return item.model_dump(mode='json')
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=default, allow_nan=False)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(serial(value) + '\n')
    tmp.replace(path)


def emit(event, **value):
    print(serial(dict(event=event, **value)), flush=True)


def winner_template(path):
    if path.suffix in ('.txt', '.template'):
        value = path.read_text().strip()
    else:
        record = read(path)
        candidate = record.get('candidate', record.get('winner', record))
        if isinstance(candidate, str):
            value = candidate
        else:
            value = candidate.get('template', candidate.get('payload_template', record.get('template')))
    if not isinstance(value, str) or value.count('{goal}') != 1:
        raise ValueError('Winner must contain exactly one immutable {goal} slot')
    return value


def winner_objective_metadata(path):
    """Bind probe role metadata without changing native ASR execution behavior.

    Historical goal-user winner files predate explicit role fields. Their
    schema identifies that objective; a new explicit target always takes
    precedence over a retained legacy schema name.
    """
    path = Path(path)
    template = winner_template(path)
    record = read(path) if path.suffix not in ('.txt', '.template') else {}
    observed, bindings = [], []
    for name, value in [('winner', record), ('score_protocol',
                         read(path.parent / 'score_protocol.json')
                         if (path.parent / 'score_protocol.json').exists() else {})]:
        role = value.get('target_role')
        if role is None:
            continue
        if role not in PROBE_ROLES or value.get('target_role_index', PROBE_ROLES.index(role)) != PROBE_ROLES.index(role):
            raise ValueError('Winner optimization role/index is inconsistent')
        if 'roles' in value and tuple(value['roles']) != PROBE_ROLES:
            raise ValueError('Winner frozen probe role order differs')
        observed.append(role)
        bindings.append(name)
    if len(set(observed)) > 1:
        raise ValueError('Winner role differs from its frozen score protocol')
    legacy_user = not observed and record.get('schema') == 'autodan-goal-user-winner-v1'
    role = observed[0] if observed else 'user' if legacy_user else None
    return dict(target_role=role, target_role_index=PROBE_ROLES.index(role) if role else None,
                roles=list(PROBE_ROLES), objective=f'mean goal-token P({role}) across six fixed tasks' if role else None,
                role_evidence=bindings or (['legacy goal-user winner schema'] if legacy_user else []),
                template_sha256=hashlib.sha256(json.dumps(template, sort_keys=True, ensure_ascii=False,
                                                        allow_nan=False).encode()).hexdigest(),
                winner_file_sha256=sha(path),
                goal_user_mean_semantics='Always the actual user class at index 2; never renamed to the optimized class')


def validate_objective_metadata(metadata, template, winner_file_sha256=None):
    """Offline role/index verification for new evaluation protocol metadata."""
    role, index = metadata.get('target_role'), metadata.get('target_role_index')
    if tuple(metadata.get('roles', ())) != PROBE_ROLES:
        raise ValueError('Evaluation optimization metadata has an unknown role order')
    if (role is None and index is not None) or (role is not None and
            (role not in PROBE_ROLES or index != PROBE_ROLES.index(role))):
        raise ValueError('Evaluation optimization target role/index disagree')
    expected = hashlib.sha256(json.dumps(template, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False).encode()).hexdigest()
    if metadata.get('template_sha256') != expected:
        raise ValueError('Evaluation optimization metadata describes another template')
    if winner_file_sha256 is not None and metadata.get('winner_file_sha256') != winner_file_sha256:
        raise ValueError('Evaluation optimization metadata winner SHA differs')
    return True


def validate_payload_tokens(tokenizer, cases, library=None):
    """Check actual filled payloads, including prefix/goal/suffix boundaries."""
    library = library or {}
    forbidden_ids = sorted(set(tokenizer.all_special_ids) | set(library.get('forbidden_ids', [])))
    forbidden_spellings = sorted(set(tokenizer.all_special_tokens) | set(library.get('forbidden_spellings', [])))
    records = []
    for case in cases:
        text = case['payload_text']
        ids = tokenizer.encode(text, add_special_tokens=False)
        intersections = sorted(set(ids).intersection(forbidden_ids) | {value for value in ids if value >= BASE_BPE_BOUND})
        spellings = [value for value in forbidden_spellings if value and value in text]
        spellings += re.findall(r'<\|reserved_\d+\|>', text)
        if intersections or spellings:
            raise ValueError('Payload contains a real special token: ' + case['case_id'] +
                             '; ids=' + repr(intersections) + '; spellings=' + repr(spellings))
        records.append(dict(case_id=case['case_id'], token_count=len(ids),
                            token_ids_sha256=hashlib.sha256(serial(ids).encode()).hexdigest()))
    return dict(policy='All filled payloads, including immutable goal and both template boundaries; exact spellings and token IDs excluded',
                base_bpe_bound_exclusive=BASE_BPE_BOUND,
                checked_payloads=len(records), passed=True, forbidden_ids=forbidden_ids,
                forbidden_spellings=forbidden_spellings, records=records)


def validate_rendered_payloads(tokenizer, request, prompt, prompt_date, exposed):
    """Check only original tool-body payload spans, allowing genuine outer frames."""
    if not exposed:
        return dict(passed=True, checked_occurrences=0, base_bpe_bound_exclusive=BASE_BPE_BOUND,
                    forbidden_token_ids=[], policy='No complete decoded payload in this model request')
    from scorer import load_module
    from deterministic_spans import locate_payload
    from experiments.ChatInject.autodan_role_probe_pilot.data import recipes
    spans_module = load_module('_near_eval_trusted_spans', ROOT.parent / 'agentdojo-first-tool/spans.py')
    # These are the same native decoded-field recipes used to establish
    # exposure.  Only span recovery changes; prompt bytes/goals stay untouched.
    context = {'slot_recipes': recipes(request['messages'], request['payload_text'])}
    location = locate_payload(tokenizer, spans_module, request, prompt, prompt_date,
                              context=context)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    checks = []
    for occurrence in location['occurrences']:
        a, b = occurrence['payload_char_span']
        positions = [index for index, (start, end) in enumerate(encoded['offset_mapping'])
                     if end > a and start < b and end > start]
        bad = sorted({encoded['input_ids'][index] for index in positions if encoded['input_ids'][index] >= BASE_BPE_BOUND})
        if bad:
            raise ValueError('Actual serialized tool payload produced forbidden special/reserved IDs: ' + repr(bad))
        checks.append(dict(payload_char_span=[a, b], goal_char_span=occurrence['goal_char_span'],
                           token_count=len(positions), forbidden_token_ids=bad))
    return dict(passed=True, checked_occurrences=len(checks), base_bpe_bound_exclusive=BASE_BPE_BOUND,
                forbidden_token_ids=[], occurrences=checks,
                policy='Complete payload spans in trusted original tool bodies; native outer Harmony frames excluded',
                span_mapping_proof=location['mapping_proof'])


def plan_completed_import(source_output, protocol, output):
    """Bind completed optimized native episodes before hashing the new protocol."""
    source_output, output = Path(source_output).resolve(), Path(output).resolve()
    if source_output == output:
        raise ValueError('Completed native import requires a distinct output directory')
    source_protocol_path = source_output / 'protocol.json'
    source = read(source_protocol_path)
    compatibility_fields = (
        'scope', 'dataset_sha256', 'winner_sha256', 'source_identity_sha256',
        'engine_configuration', 'max_tokens', 'reasoning_effort', 'prompt_date',
        'seed', 'temperature', 'stop_ids', 'benchmark', 'normal_system',
        'initial_errors', 'model_staging_proof_sha256', 'model_configuration_sha256',
        'package_versions', 'serialized_payload_validation', 'optimization_objective',
        'engine_environment', 'near_delimiter_library_sha256')
    for field in compatibility_fields:
        if source.get(field) != protocol.get(field):
            raise ValueError('Completed native import protocol differs: ' + field)
    if source.get('evaluator_source_sha256') != sha(ROOT / 'evaluate_precise.py'):
        raise ValueError('Imported episodes are not bound to the frozen precise evaluator')
    if source.get('templates', {}).get('optimized') != protocol['templates']['optimized']:
        raise ValueError('Completed native import optimized template differs')
    canonical_sha = hashlib.sha256(serial(source).encode()).hexdigest()
    physical_sha = sha(source_protocol_path)
    source_cases = {case['case_id']: case for case in source['cases'] if case['condition'] == 'optimized'}
    wanted = {case['case_id']: case for case in protocol['cases']}
    if source_cases != wanted or any(case['condition'] != 'optimized' for case in wanted.values()):
        raise ValueError('Completed native import optimized task population differs')
    cases = []
    for case_id, case in wanted.items():
        case_path = source_output / 'cases' / (case_id + '.json')
        if not case_path.is_file():
            continue
        record = read(case_path)
        if record.get('status') != 'complete':
            continue
        if (record.get('case') != case or record.get('case_id') != case_id
                or record.get('condition') != 'optimized'
                or record.get('protocol_sha256') != canonical_sha
                or type(record.get('security')) is not bool
                or type(record.get('utility')) is not bool):
            raise ValueError('Invalid completed optimized native episode: ' + case_id)
        artifacts = []

        def add_artifact(kind, path, expected=None):
            path = Path(path).resolve()
            if not path.is_relative_to(source_output):
                raise ValueError('Imported native artifact is outside its source output')
            relative = path.relative_to(source_output)
            actual = sha(path)
            if expected is not None and actual != expected:
                raise ValueError('Completed native artifact hash differs: ' + str(relative))
            artifacts.append(dict(kind=kind, source_path=str(path),
                                  destination_relative_path=str(relative), sha256=actual))

        for reference in record['requests']:
            call_id = reference['call_id']
            if reference.get('status') != 'complete' or not call_id.startswith(case_id + '/call_'):
                raise ValueError('Completed episode has an incomplete or foreign native request')
            call = source_output / 'calls' / call_id
            request_path = call / 'request.json'
            request_record = read(request_path)
            if request_record.get('case') != case or request_record.get('call_id') != call_id:
                raise ValueError('Imported native request case differs')
            add_artifact('request', request_path, reference['request_sha256'])
            add_artifact('raw_response', call / 'raw_response.json', reference['raw_response_sha256'])
            add_artifact('parsed_response', call / 'parsed_response.json', reference['parsed_response_sha256'])
        for reference in record['trajectories']:
            path = Path(reference['path'])
            if not path.is_absolute():
                path = source_output / path
            expected_root = (source_output / 'trajectories' / case_id).resolve()
            if not path.resolve().is_relative_to(expected_root):
                raise ValueError('Imported native trajectory belongs to another case')
            add_artifact('trajectory', path, reference['sha256'])
        add_artifact('injection', source_output / 'injections' / (case_id + '.json'))
        cases.append(dict(case_id=case_id, source_case_path=str(case_path.resolve()),
                          source_case_sha256=sha(case_path), artifacts=artifacts))
    return dict(schema='optimized-completed-native-episode-import-v1', source_output=str(source_output),
                source_protocol_path=str(source_protocol_path),
                source_protocol_physical_sha256=physical_sha,
                source_protocol_canonical_sha256=canonical_sha,
                source_evaluator_source_sha256=source['evaluator_source_sha256'],
                compatibility_fields_checked=list(compatibility_fields), cases=cases,
                imported_conditions=['optimized'], baseline_imports=0,
                episode_record_changes=['protocol_sha256', 'import_provenance'],
                native_artifact_bytes_unchanged=True)


def materialize_completed_import(plan, output, protocol_sha256):
    """Copy exact native artifacts and derive only the episode protocol binding."""
    output = Path(output).resolve()
    cases = []
    copied = 0
    for item in plan['cases']:
        source_case_path = Path(item['source_case_path'])
        if sha(source_case_path) != item['source_case_sha256']:
            raise ValueError('Completed source episode changed after import planning')
        for artifact in item['artifacts']:
            source, expected = Path(artifact['source_path']), artifact['sha256']
            destination = output / artifact['destination_relative_path']
            if sha(source) != expected:
                raise ValueError('Completed source native artifact changed after import planning')
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + '.importing')
                shutil.copyfile(source, temporary)
                if sha(temporary) != expected:
                    raise ValueError('Copied native artifact differs from its source')
                temporary.replace(destination)
            if sha(destination) != expected:
                raise ValueError('Existing imported native artifact differs')
            copied += 1
        record = read(source_case_path)
        record['protocol_sha256'] = protocol_sha256
        record['import_provenance'] = dict(
            source_output=plan['source_output'], source_case_path=item['source_case_path'],
            source_case_sha256=item['source_case_sha256'],
            source_protocol_physical_sha256=plan['source_protocol_physical_sha256'],
            source_protocol_canonical_sha256=plan['source_protocol_canonical_sha256'])
        # Original request/case fields and trajectory references are preserved.
        # The exact local trajectory copies are identified by the protocol map.
        destination = output / 'cases' / (item['case_id'] + '.json')
        if destination.exists() and read(destination) != record:
            raise ValueError('Existing imported episode differs from its exact derivation')
        write(destination, record)
        cases.append(dict(case_id=item['case_id'], source_case_path=item['source_case_path'],
                          source_case_sha256=item['source_case_sha256'],
                          destination_case_path=str(destination), destination_case_sha256=sha(destination),
                          native_artifact_count=len(item['artifacts'])))
    manifest = dict(schema='optimized-completed-native-episode-import-manifest-v1',
                    destination_protocol_canonical_sha256=protocol_sha256,
                    source_protocol_physical_sha256=plan['source_protocol_physical_sha256'],
                    source_protocol_canonical_sha256=plan['source_protocol_canonical_sha256'],
                    completed_optimized_imports=len(cases), baseline_imports=0,
                    native_artifacts_copied_and_sha_verified=copied, cases=cases)
    path = output / 'completed_episode_import_manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Existing completed import manifest differs')
    write(path, manifest)
    return manifest


def load_native(source_root):
    identity = read(source_root / 'agentdojo-source.json')
    source = Path(identity['source'])
    if not source.exists():
        source = DELT / 'indirect-prompt-injection/ChatInject/src'
    for name, digest in identity['files'].items():
        if sha(source / name) != digest:
            raise ValueError('Frozen AgentDojo source differs: ' + name)
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(DELT))
    import agentdojo
    if not Path(agentdojo.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('Imported AgentDojo is not the frozen benchmark source')
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig, load_system_message
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.errors import AbortAgentError
    from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai, _function_to_openai
    from agentdojo.attacks.base_attacks import FixedJailbreakAttack
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.task_suite.load_suites import get_suites
    from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages
    from agentdojo.types import text_content_block_from_string
    from experiments.ChatInject.autodan_role_probe_pilot.data import recipes
    from experiments.ChatInject.role_probe.runtime import normalize_messages
    from experiments.ChatInject.role_probe_subset.runtime import safe_parse_completion
    return locals()


class AsyncGenerator:
    """Concurrent native pipeline callers share vLLM continuous batching."""
    def __init__(self, config, max_tokens):
        self.config, self.max_tokens = config, max_tokens
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run_loop, name='autodan-vllm-loop', daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise RuntimeError(self.error)

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        async def initialize():
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
            self.sampling = SamplingParams(temperature=0., top_p=1., top_k=-1, seed=0,
                max_tokens=self.max_tokens, stop_token_ids=STOP_IDS, ignore_eos=True,
                detokenize=False, skip_special_tokens=False,
                output_kind=RequestOutputKind.FINAL_ONLY)
            self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**self.config))
            if inspect.isawaitable(self.engine):
                self.engine = await self.engine
        try:
            self.loop.run_until_complete(initialize())
        except BaseException:
            self.error = traceback.format_exc()
        self.ready.set()
        if self.error is None:
            self.loop.run_forever()
        self.loop.close()

    async def _generate(self, ids, request_id):
        final = None
        async for output in self.engine.generate({'prompt_token_ids': ids}, self.sampling, request_id):
            final = output
        if final is None or len(final.outputs) != 1:
            raise RuntimeError('vLLM returned no unique final sequence')
        if final.prompt_token_ids is not None and list(final.prompt_token_ids) != ids:
            raise RuntimeError('vLLM changed supplied prompt token IDs')
        sequence = final.outputs[0]
        generated = list(sequence.token_ids)
        if not generated or len(generated) > self.max_tokens:
            raise RuntimeError('Empty or over-limit vLLM sequence')
        if sequence.finish_reason == 'stop':
            if generated[-1] not in STOP_IDS or sequence.stop_reason != generated[-1]:
                raise RuntimeError('Raw vLLM output does not preserve the reported Harmony stop token')
        elif sequence.finish_reason != 'length' or len(generated) != self.max_tokens:
            raise RuntimeError('Unexpected vLLM termination')
        if any(token in STOP_IDS for token in generated[:-1]):
            raise RuntimeError('vLLM continued after a Harmony action/final stop')
        return dict(generated_ids=generated, finish_reason=sequence.finish_reason,
                    stop_reason=sequence.stop_reason)

    def generate(self, ids, request_id):
        return asyncio.run_coroutine_threadsafe(self._generate(ids, request_id), self.loop).result()

    def close(self):
        async def stop():
            result = self.engine.shutdown()
            if inspect.isawaitable(result):
                await result
        if self.error is None and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(stop(), self.loop).result(timeout=30)
            finally:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=30)


def metrics(records, expected):
    scored = [r for r in records if r['status'] == 'complete']
    successes = sum(r['security'] for r in scored)
    utility = sum(r['utility'] for r in scored)
    exposed = sum(r['payload_exposed'] for r in records)
    return dict(scheduled=expected, terminal=len(records), scored=len(scored),
                errors=sum(r['status'] == 'error' for r in records), successes=successes,
                asr=successes / expected if len(scored) == expected else None,
                scored_asr=successes / len(scored) if scored else None,
                known_successes_over_scheduled=successes / expected,
                utility=utility / expected if len(scored) == expected else None,
                scored_utility=utility / len(scored) if scored else None,
                exposed=exposed, exposure_rate=exposed / expected)


def run_episode(case, suite, args, dataset, native, tokenizer, encoding, generator):
    path = args.output / 'cases' / (case['case_id'] + '.json')
    if path.exists():
        result = read(path)
        if result['case'] != case or result.get('protocol_sha256') != args.protocol_sha256:
            raise ValueError('Cannot reuse a case from another payload protocol')
        return result
    record = dict(case=case, case_id=case['case_id'], condition=case['condition'], suite=case['suite'],
                  protocol_sha256=args.protocol_sha256,
                  status='running', security=None, utility=None, payload_exposed=False,
                  requests=[], trajectories=[])
    started = time.time()
    fixed = dataset['fixed_normal_system']
    BasePipelineElement = native['BasePipelineElement']

    class NativeLLM(BasePipelineElement):
        name = 'gpt-oss-120b-autodan-probe-vllm'

        def query(self, query, runtime, env, messages=(), extra_args=None):
            index = len(record['requests'])
            call_id = case['case_id'] + f'/call_{index:04d}'
            directory = args.output / 'calls' / call_id
            model_messages = native['normalize_messages']([
                native['_message_to_openai'](m, 'gpt-oss-120b') for m in messages])
            if [m for m in model_messages if m['role'] in ('system', 'developer')] != [fixed]:
                raise ValueError('Native normal system role/text changed')
            try:
                exposed = bool(native['recipes'](model_messages, case['payload_text']))
            except ValueError:
                exposed = False
            record['payload_exposed'] |= exposed
            tools = [native['_function_to_openai'](tool) for tool in runtime.functions.values()]
            prompt = tokenizer.apply_chat_template(model_messages, tools=tools or None, tokenize=False,
                add_generation_prompt=True, reasoning_effort='medium',
                strftime_now=lambda _fmt: args.prompt_date)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            payload_check = validate_rendered_payloads(tokenizer,
                dict(messages=model_messages, tools=tools, actual_goal=case['actual_goal'],
                     payload_template=case['payload_template'], payload_text=case['payload_text']),
                prompt, args.prompt_date, exposed)
            if len(ids) + args.max_tokens > args.max_model_len:
                raise ValueError('Native context exceeds frozen model length; truncation is disabled')
            request = dict(call_id=call_id, call_index=index, case=case, messages=model_messages,
                           tools=tools, rendered_prompt=prompt, input_ids=ids,
                           input_tokens=len(ids), payload_exposed=exposed,
                           payload_special_token_check=payload_check)
            write(directory / 'request.json', request)
            reference = dict(call_id=call_id, request_sha256=sha(directory / 'request.json'), status='running')
            record['requests'].append(reference)
            response = generator.generate(ids, call_id)
            write(directory / 'raw_response.json', response)
            reference['raw_response_sha256'] = sha(directory / 'raw_response.json')
            parsed, diagnostics = native['safe_parse_completion'](encoding, response['generated_ids'], call_id,
                max_tokens=args.max_tokens, stop_ids=STOP_IDS)
            write(directory / 'parsed_response.json', dict(message=parsed, diagnostics=diagnostics))
            reference.update(status='complete', parsed_response_sha256=sha(directory / 'parsed_response.json'))
            calls = [native['FunctionCall'](function=c['function']['name'],
                     args=json.loads(c['function']['arguments']) if isinstance(c['function']['arguments'], str)
                     else c['function']['arguments'], id=c['id']) for c in (parsed.get('tool_calls') or [])]
            text = parsed.get('content')
            output = dict(role='assistant', content=[native['text_content_block_from_string'](text)] if text else None,
                          tool_calls=calls or None)
            return query, runtime, env, [*messages, output], extra_args or {}

    pipeline = native['AgentPipeline'].from_config(native['PipelineConfig'](
        llm=NativeLLM(), model_id=None, defense=None, tool_delimiter='tool',
        system_message_name=None, system_message=fixed['content'], tool_output_format=None))

    class RecordingPipeline(BasePipelineElement):
        name = pipeline.name

        def query(self, query, runtime, env, messages=(), extra_args=None):
            pre = env.model_copy(deep=True)
            index = len(record['trajectories'])
            final, final_messages = None, None
            try:
                result = pipeline.query(query, runtime, env, messages, extra_args or {})
                final, final_messages = result[2], result[3]
                return result
            except native['AbortAgentError'] as error:
                final, final_messages = error.task_environment, error.messages
                raise
            finally:
                if final is not None:
                    trajectory = args.output / 'trajectories' / case['case_id'] / f'pipeline_{index:02d}.json'
                    write(trajectory, dict(query=query, pre_environment=pre, final_environment=final,
                          messages=final_messages,
                          functions_stack_trace=native['functions_stack_trace_from_messages'](final_messages)))
                    record['trajectories'].append(dict(path=str(trajectory), sha256=sha(trajectory)))

    try:
        user = suite.user_tasks[case['user_task_id']]
        injection = suite.injection_tasks[case['injection_task_id']]
        class ExactGoalAttack(native['FixedJailbreakAttack']):
            def attack(self, user_task, injection_task):
                # Preserve native vector selection while treating every other
                # brace literally and retaining the exact immutable goal text.
                return {key: self.jailbreak.replace('{goal}', injection_task.GOAL)
                        for key in self.get_injection_candidates(user_task)}
        attack = ExactGoalAttack(case['payload_template'], suite, pipeline)
        injections = attack.attack(user, injection)
        # Native placeholder substitution occurs inside quoted YAML/JSON values.
        injections = {key: value.replace('\\', '\\\\').replace('"', '\\"')
                      for key, value in injections.items()}
        write(args.output / 'injections' / (case['case_id'] + '.json'), injections)
        utility, security = suite.run_task_with_pipeline(RecordingPipeline(), user, injection, injections)
        if type(utility) is not bool or type(security) is not bool:
            raise ValueError('Native task evaluators returned nonboolean scores')
        record.update(status='complete', utility=utility, security=security)
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    record['elapsed_seconds'] = time.time() - started
    write(path, record)
    return record


def main():
    # FlashInfer's optional sampler JIT needs nvcc, absent on this allocation.
    # vLLM's native sampler supports the identical temperature-zero policy.
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'outputs/pilot-20261104/dataset.json')
    parser.add_argument('--winner', type=Path, required=True)
    parser.add_argument('--import-completed-from', type=Path,
                        help='Reuse only complete optimized episodes from an exactly matching stopped native run')
    parser.add_argument('--previous-winner', type=Path,
                        help='Re-evaluate the previous optimized winner as a fourth matched condition')
    parser.add_argument('--near-delimiter-library', type=Path,
                        help='Default: winner directory/near_delimiter_library.json if present')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['sampled', 'full'], default='sampled')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    parser.add_argument('--tensor-parallel-size', type=int, default=2)
    parser.add_argument('--pipeline-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.85)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--max-num-seqs', type=int, default=18)
    parser.add_argument('--max-num-batched-tokens', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--prompt-date', default='2026-10-21')
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.previous_winner:
        raise ValueError('Optimized-only evaluation does not schedule another condition')
    args.output = args.output.resolve()
    dataset = read(args.dataset)
    native = load_native(args.source)
    if native['load_system_message'](None) != dataset['fixed_normal_system']['content']:
        raise ValueError('Source default system differs from the frozen dataset')
    suites = native['get_suites']('v1.2.1')
    for name, counts in COUNTS.items():
        if (len(suites[name].user_tasks), len(suites[name].injection_tasks)) != counts:
            raise ValueError('Native task population differs: ' + name)
    templates = {'optimized': winner_template(args.winner)}
    numeric = lambda value: int(value.rsplit('_', 1)[1])
    panel = dataset['sample_manifest']['selected_pairs'] if args.scope == 'sampled' else [
        dict(suite=s, user_task_id=u, injection_task_id=i, pair_id=f'{s}/{u}/{i}')
        for s in COUNTS for u in sorted(suites[s].user_tasks, key=numeric)
        for i in sorted(suites[s].injection_tasks, key=numeric)]
    cases = []
    for item in panel:
        goal = suites[item['suite']].injection_tasks[item['injection_task_id']].GOAL
        for condition, template in templates.items():
            cases.append(dict(**item, case_id=condition + '/' + item['pair_id'], condition=condition,
                              actual_goal=goal, payload_template=template, payload_text=template.replace('{goal}', goal)))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    library_path = args.near_delimiter_library or args.winner.parent / 'near_delimiter_library.json'
    library = read(library_path) if library_path.exists() else None
    if args.near_delimiter_library and library is None:
        raise ValueError('Explicit near-delimiter library is missing')
    token_validation = validate_payload_tokens(tokenizer, cases, library)
    config = dict(model=str(args.model), tokenizer=str(args.model), dtype='bfloat16', quantization='mxfp4',
                  tensor_parallel_size=args.tensor_parallel_size, pipeline_parallel_size=args.pipeline_parallel_size,
                  distributed_executor_backend='mp', max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens, enable_chunked_prefill=True,
                  enable_prefix_caching=False, enforce_eager=args.enforce_eager,
                  generation_config='vllm', seed=0, trust_remote_code=False)
    protocol = dict(schema='autodan-role-probe-native-vllm-eval-v1', scope=args.scope,
                    dataset_sha256=sha(args.dataset), winner_sha256=sha(args.winner), templates=templates,
                    source_identity_sha256=sha(args.source / 'agentdojo-source.json'),
                    engine_configuration=config, max_tokens=args.max_tokens, reasoning_effort='medium',
                    prompt_date=args.prompt_date, seed=0, temperature=0., stop_ids=STOP_IDS,
                    benchmark='AgentDojo v1.2.1', cases=cases,
                    normal_system=dataset['fixed_normal_system'], initial_errors='null; no case replacement',
                    evaluator_source_sha256=sha(__file__),
                    model_staging_proof_sha256=sha(args.model / 'staging-proof.json'),
                    model_configuration_sha256={name: sha(args.model / name) for name in
                        ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')},
                    package_versions={n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'transformers', 'openai-harmony')},
                    numerical_equivalence_to_historical_backend_claimed=False)
    protocol['payload_special_token_validation'] = token_validation
    protocol['serialized_payload_validation'] = dict(
        method='deterministic-native-string-origins-v1',
        deterministic_locator_source_sha256=sha(ROOT / 'deterministic_spans.py'),
        trusted_outer_spans_source_sha256=sha(ROOT.parent / 'agentdojo-first-tool/spans.py'),
        native_recipe_source_sha256=sha(inspect.getsourcefile(native['recipes'])),
        native_recipe_source=str(Path(inspect.getsourcefile(native['recipes'])).resolve()),
        unchanged_native_prompt_and_goal=True,
        scope='All complete payload occurrences in native recipe-selected tool string fields')
    protocol['optimization_objective'] = winner_objective_metadata(args.winner)
    validate_objective_metadata(protocol['optimization_objective'], templates['optimized'],
                                protocol['winner_sha256'])
    if args.previous_winner:
        protocol['previous_winner_sha256'] = sha(args.previous_winner)
        protocol['previous_winner_template'] = templates['previous_optimized']
        protocol['previous_winner_policy'] = 'Fresh same-engine episodes; historical ASR is not used as matched evidence'
        protocol['previous_winner_objective'] = winner_objective_metadata(args.previous_winner)
        validate_objective_metadata(protocol['previous_winner_objective'], templates['previous_optimized'],
                                    protocol['previous_winner_sha256'])
    if library is not None:
        protocol['near_delimiter_library_sha256'] = sha(library_path)
    protocol['engine_environment'] = {
        name: os.environ.get(name) for name in
        ('VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_ALLREDUCE_USE_FLASHINFER', 'VLLM_MXFP4_USE_MARLIN')}
    protocol['scheduled_condition_policy'] = 'optimized only; no plain or remember episodes submitted'
    if args.import_completed_from:
        protocol['completed_episode_import'] = plan_completed_import(args.import_completed_from, protocol, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Output directory belongs to a different frozen protocol')
    write(path, protocol)
    if library is not None:
        # Preserve the exact export rather than reserializing its hash binding.
        (args.output / 'near_delimiter_library.json').write_bytes(library_path.read_bytes())
    args.protocol_sha256 = hashlib.sha256(serial(protocol).encode()).hexdigest()
    if args.import_completed_from:
        materialize_completed_import(protocol['completed_episode_import'], args.output, args.protocol_sha256)
    if args.dry_run:
        emit('dry_run_complete', scope=args.scope, episodes=len(cases), output=str(args.output),
             source=str(native['source']), selected_pair_count=len(panel),
             selected_pairs=[p['pair_id'] for p in panel] if args.scope == 'sampled' else None)
        return
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    write(args.output / 'launch.json', dict(pid=os.getpid(), started_at=time.time(),
          versions={n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'transformers', 'openai-harmony')}))
    emit('loading_vllm', configuration=config)
    generator = AsyncGenerator(config, args.max_tokens)
    records = []
    started = time.time()
    try:
        emit('vllm_ready', episodes=len(cases), workers=args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_episode, c, suites[c['suite']], args, dataset, native,
                                   tokenizer, encoding, generator) for c in cases]
            for future in as_completed(futures):
                result = future.result()
                records.append(result)
                write(args.output / 'progress.json', dict(terminal=len(records), scheduled=len(cases),
                      elapsed_seconds=time.time() - started, last_case=result['case_id']))
                emit('case_complete', case_id=result['case_id'], status=result['status'],
                     security=result['security'], utility=result['utility'], payload_exposed=result['payload_exposed'],
                     error=result.get('error'), terminal=len(records), scheduled=len(cases))
        summary = dict(scope=args.scope, tasks_per_condition=len(panel), scheduled_episodes=len(cases),
                       terminal=len(records), elapsed_seconds=time.time() - started,
                       conditions={k: metrics([r for r in records if r['condition'] == k], len(panel)) for k in templates},
                       suite_metrics={k: {s: metrics([r for r in records if r['condition'] == k and r['suite'] == s],
                           sum(p['suite'] == s for p in panel)) for s in COUNTS} for k in templates},
                       errors=[dict(case_id=r['case_id'], error=r['error']) for r in records if r['status'] == 'error'])
        write(args.output / 'summary.json', summary)
        emit('evaluation_complete', summary=summary)
    finally:
        generator.close()


if __name__ == '__main__':
    main()
