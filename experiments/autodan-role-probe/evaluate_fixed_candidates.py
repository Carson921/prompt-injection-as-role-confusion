#!/usr/bin/env python3
"""Evaluate three to six already-existing fixed candidates on native AgentDojo.

This module performs no candidate generation, mutation, search or API calls.
The episode runner and vLLM backend are imported unchanged from the frozen
optimized-only evaluator. Every initial error remains null and is not retried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import time

import evaluate_optimized as frozen
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from prepare import COUNTS, PLAIN, REMEMBER, ROOT, SOURCE, read, sha


def require(value, message):
    if not value:
        raise ValueError(message)


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def resolve_relative(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def bind_probe_score(entry, template, manifest_parent, dataset, dataset_sha256):
    """Bind a prior complete numerical measurement, never infer its fitness."""
    score_path = resolve_relative(entry['score_path'], manifest_parent)
    score = read(score_path)
    require(score.get('status') == 'complete' and score.get('complete') is True,
            'Candidate needs a completed native six-context score')
    require(score.get('template') == template, 'Score evidence describes a different template')
    require(score.get('target_role') == 'user' and score.get('target_role_index') == 2,
            'Only actual user-role measurements are eligible')
    require(score.get('expected_cases') == 6 and score.get('completed_cases') == 6,
            'Partial/pruned/surrogate scores are ineligible')
    records = score.get('per_case', [])
    contexts = {context['case_id']: context for context in dataset['contexts']}
    require(len(records) == 6 and len({row['case_id'] for row in records}) == 6,
            'Candidate score lacks six distinct case records')
    require({row['case_id'] for row in records} == set(contexts),
            'Candidate numerical evidence uses a different fixed panel')
    probabilities, artifacts = [], []
    for row in records:
        require(row.get('status') == 'complete' and row.get('template') == template,
                'Incomplete or mismatched numerical case evidence')
        require(row.get('target_role') == 'user' and row.get('target_role_index') == 2,
                'Numerical case user-role index differs')
        require(row.get('protocol_sha256') == score.get('protocol_sha256'),
                'Numerical case protocol differs from aggregate')
        expected_goal = text_sha(contexts[row['case_id']]['actual_goal'])
        require(row.get('actual_goal_sha256') == expected_goal,
                'Numerical score changed the immutable goal')
        value = row.get('goal_user_mean')
        require(isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1, 'Invalid P(user) measurement')
        probabilities.append(float(value))
        artifact = resolve_relative(row['token_artifact'], score_path.parent)
        expected_artifacts = row.get('artifact_sha256')
        require(isinstance(expected_artifacts, dict) and artifact.name in expected_artifacts,
                'Native evidence lacks its token/request artifact hash map')
        for name, expected in expected_artifacts.items():
            require(Path(name).name == name, 'Numerical artifact name contains a path')
            bound_path = artifact.parent / name
            require(bound_path.is_file() and sha(bound_path) == expected,
                    'Native token/request artifact SHA differs or artifact is missing')
            artifacts.append(dict(case_id=row['case_id'], path=str(bound_path), sha256=expected))
    mean = math.fsum(probabilities) / 6
    require(math.isclose(mean, score['fitness'], rel_tol=0, abs_tol=1e-12)
            and math.isclose(mean, score['goal_role_means']['user'], rel_tol=0, abs_tol=1e-12),
            'Numerical evidence is not the equal-six-task mean P(user)')
    protocol_path = resolve_relative(entry['score_protocol_path'], manifest_parent) if entry.get(
        'score_protocol_path') else score_path.parents[2] / 'score_protocol.json'
    protocol = read(protocol_path)
    compact_sha = text_sha(json.dumps(protocol, sort_keys=True, separators=(',', ':')))
    require(compact_sha == score.get('protocol_sha256'), 'Numerical score protocol binding differs')
    require(protocol.get('dataset_sha256') == dataset_sha256
            and protocol.get('target_role') == 'user' and protocol.get('target_role_index') == 2
            and tuple(protocol.get('roles', [])) == frozen.PROBE_ROLES
            and protocol.get('truncation') is False,
            'Numerical score protocol uses a different dataset, role order, or context truncation')
    return dict(goal_user_mean=mean, six_case_goal_user_means=probabilities,
                score_path=str(score_path), score_file_sha256=sha(score_path),
                score_protocol_path=str(protocol_path), score_protocol_file_sha256=sha(protocol_path),
                score_protocol_canonical_sha256=compact_sha,
                probe_checkpoint_sha256=protocol.get('probe_sha256'), token_artifacts=artifacts,
                fitness_source='Existing complete native equal-six-case P(user); no scoring in this evaluator')


def load_candidates(path, tokenizer, library, dataset, dataset_sha256, maximum=1024):
    path = Path(path).resolve()
    manifest = read(path)
    entries = manifest.get('candidates') if isinstance(manifest, dict) else manifest
    require(isinstance(entries, list) and 3 <= len(entries) <= 6,
            'Three to six fixed candidates are required; baseline conditions are not added')
    result, seen, intervals = [], set(), []
    for entry in entries:
        require(isinstance(entry, dict), 'Each fixed candidate must be a JSON object')
        template = entry.get('template')
        if template is None and entry.get('template_path'):
            template_path = resolve_relative(entry['template_path'], path.parent)
            template = frozen.winner_template(template_path)
        require(isinstance(template, str), 'Candidate must provide an existing template or template_path')
        require(template not in (PLAIN, REMEMBER), 'Plain/remember baselines may not be scheduled')
        guard = validate_diverse_template(template, tokenizer, library, maximum=maximum)
        require(guard['allowed'], 'Candidate admission failed: ' + ','.join(guard['reason_codes']))
        digest = text_sha(template)
        require(digest not in seen, 'Candidates must be distinct')
        seen.add(digest)
        probe = bind_probe_score(entry, template, path.parent, dataset, dataset_sha256)
        interval = entry.get('p_user_interval')
        require(isinstance(interval, list) and len(interval) == 2
                and all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in interval),
                'Each candidate needs a measured P(user) interval [lower, upper]')
        lower, upper = map(float, interval)
        require(0 <= lower < upper <= 1, 'Invalid P(user) interval')
        require(lower <= probe['goal_user_mean'] and (probe['goal_user_mean'] < upper
                or upper == 1 and probe['goal_user_mean'] == 1),
                'Measured P(user) lies outside the declared interval')
        require(all(upper <= a or lower >= b for a, b in intervals),
                'The candidate P(user) intervals must not overlap')
        intervals.append((lower, upper))
        result.append(dict(condition='candidate_' + digest[:16], template=template,
                           template_raw_sha256=digest, p_user_interval=[lower, upper],
                           probe_evidence=probe, admission=guard))
    require(len({row['condition'] for row in result}) == len(entries), 'Stable condition ID collision')
    require(len({row['probe_evidence']['probe_checkpoint_sha256'] for row in result}) == 1,
            'Candidates must use the same role-probe checkpoint')
    # Scheduling remains unchanged if the human reorders manifest entries.
    return sorted(result, key=lambda row: row['condition'])


def full_panel(suites):
    numeric = lambda value: int(value.rsplit('_', 1)[1])
    panel = []
    for name, counts in COUNTS.items():
        suite = suites[name]
        require((len(suite.user_tasks), len(suite.injection_tasks)) == counts,
                'Native task population differs: ' + name)
        for user in sorted(suite.user_tasks, key=numeric):
            for injection in sorted(suite.injection_tasks, key=numeric):
                panel.append(dict(suite=name, user_task_id=user, injection_task_id=injection,
                                  pair_id=f'{name}/{user}/{injection}'))
    require(len(panel) == 389, 'Full native population is not 389 tasks')
    return panel


def fixed_cases(panel, suites, candidates):
    require(3 <= len(candidates) <= 6, 'Three to six fixed candidates are required')
    cases = []
    for item in panel:
        goal = suites[item['suite']].injection_tasks[item['injection_task_id']].GOAL
        for candidate in candidates:
            condition, template = candidate['condition'], candidate['template']
            require(template.count('{goal}') == 1, 'Immutable goal slot changed')
            cases.append(dict(**item, case_id=condition + '/' + item['pair_id'],
                              condition=condition, actual_goal=goal, payload_template=template,
                              payload_text=template.replace('{goal}', goal)))
    expected_episodes = len(panel) * len(candidates)
    require(len(cases) == expected_episodes and len({row['case_id'] for row in cases}) == expected_episodes,
            'Fixed-candidate case population differs')
    return cases


def check_existing_records(output, cases, binding):
    expected = {row['case_id']: row for row in cases}
    directory = Path(output) / 'cases'
    for path in directory.rglob('*.json') if directory.exists() else []:
        record = read(path)
        identity = record.get('case_id')
        require(identity in expected and record.get('case') == expected[identity]
                and record.get('protocol_sha256') == binding,
                'Existing episode belongs to another frozen candidate protocol')
        require(record.get('status') in ('complete', 'error'),
                'Existing nonterminal episode requires inspection; it is not automatically retried')
        if record['status'] == 'error':
            require(record.get('security') is None and record.get('utility') is None,
                    'Initial error episode must retain null scores')
        else:
            require(type(record.get('security')) is bool and type(record.get('utility')) is bool,
                    'Completed episode lacks native boolean scores')


def bounded_metrics(records, expected):
    result = frozen.metrics(records, expected)
    unknown = expected - result['scored']
    result.update(unknown=unknown, asr_lower_bound=result['successes'] / expected,
                  asr_upper_bound=(result['successes'] + unknown) / expected,
                  error_policy='Initial errors remain null; no reruns, replacements, or imputed failures')
    return result


def source_bindings(native):
    paths = [Path(__file__), ROOT / 'evaluate_optimized.py', ROOT / 'evaluate_precise.py',
             ROOT / 'prepare.py', ROOT / 'diverse_policy.py', ROOT / 'broad_validation.py',
             ROOT / 'near_delimiters.py', ROOT / 'deterministic_spans.py',
             ROOT.parent / 'agentdojo-first-tool/spans.py',
             Path(inspect.getsourcefile(native['recipes'])),
             Path(inspect.getsourcefile(native['safe_parse_completion']))]
    return {str(path.resolve()): sha(path) for path in paths}


def preserve_sources(output, bindings):
    provenance = output / 'provenance'
    provenance.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (source, digest) in enumerate(bindings.items()):
        path = Path(source)
        destination = provenance / f'{index:02d}_{path.name}'
        require(sha(path) == digest, 'Source changed after protocol freezing')
        if destination.exists():
            require(sha(destination) == digest, 'Existing provenance copy differs')
        else:
            shutil.copyfile(path, destination)
        manifest.append(dict(source=source, frozen_copy=str(destination), sha256=digest))
    frozen.write(provenance / 'source_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True,
                        help='Existing fixed templates and complete numerical score evidence; three to six intervals')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'outputs/near-delimiter-20261104/dataset.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    parser.add_argument('--maximum-wrapper-tokens', type=int, default=1024)
    parser.add_argument('--dry-run', action='store_true', help='Validate and freeze only; no GPU model is loaded')
    args = parser.parse_args()
    args.output, args.model = args.output.resolve(), args.model.resolve()
    args.prompt_date, args.max_tokens, args.max_model_len, args.workers = '2026-10-21', 1024, 65536, 32
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    dataset = read(args.dataset)
    native = frozen.load_native(args.source)
    require(native['load_system_message'](None) == dataset['fixed_normal_system']['content'],
            'Source default system differs from frozen dataset')
    suites = native['get_suites']('v1.2.1')
    panel = full_panel(suites)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    library = BroadDelimiterLibrary(args.model)
    candidates = load_candidates(args.candidates, tokenizer, library, dataset, sha(args.dataset),
                                 maximum=args.maximum_wrapper_tokens)
    templates = {row['condition']: row['template'] for row in candidates}
    cases = fixed_cases(panel, suites, candidates)
    token_guard = frozen.validate_payload_tokens(tokenizer, cases, dict(
        forbidden_ids=list(library.forbidden_token_ids),
        forbidden_spellings=list(library.forbidden_spellings)))
    config = dict(model=str(args.model), tokenizer=str(args.model), dtype='bfloat16', quantization='mxfp4',
                  tensor_parallel_size=2, pipeline_parallel_size=1, distributed_executor_backend='mp',
                  max_model_len=args.max_model_len, gpu_memory_utilization=.85, max_num_seqs=32,
                  max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=False,
                  enforce_eager=True, generation_config='vllm', seed=0, trust_remote_code=False)
    bindings = source_bindings(native)
    protocol = dict(schema='fixed-candidates-native-vllm-eval-v1', scope='full',
                    candidates_manifest_sha256=sha(args.candidates), candidates=candidates,
                    dataset_sha256=sha(args.dataset), templates=templates, cases=cases,
                    source_identity_sha256=sha(args.source / 'agentdojo-source.json'),
                    engine_configuration=config, max_tokens=args.max_tokens, reasoning_effort='medium',
                    prompt_date=args.prompt_date, seed=0, temperature=0., stop_ids=frozen.STOP_IDS,
                    benchmark='AgentDojo v1.2.1', normal_system=dataset['fixed_normal_system'],
                    initial_errors='null; no case replacement',
                    evaluator_source_sha256=sha(__file__), source_bindings=bindings,
                    native_episode_runner_source_sha256=sha(ROOT / 'evaluate_optimized.py'),
                    model_staging_proof_sha256=sha(args.model / 'staging-proof.json'),
                    model_configuration_sha256={name: sha(args.model / name) for name in
                        ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja')},
                    package_versions={name: importlib.metadata.version(name) for name in
                        ('vllm','torch','transformers','openai-harmony')},
                    payload_special_token_validation=token_guard,
                    serialized_payload_validation=dict(
                        method='deterministic-native-string-origins-v1',
                        deterministic_locator_source_sha256=sha(ROOT / 'deterministic_spans.py'),
                        trusted_outer_spans_source_sha256=sha(ROOT.parent / 'agentdojo-first-tool/spans.py'),
                        native_recipe_source_sha256=sha(inspect.getsourcefile(native['recipes'])),
                        unchanged_native_prompt_and_goal=True,
                        scope='Every complete payload occurrence in native recipe-selected tool fields'),
                    scheduled_condition_policy='Three to six fixed candidate conditions; no plain/remember baseline',
                    candidate_generation=False, candidate_optimization=False, external_llm_api_calls=False,
                    numerical_equivalence_to_historical_backend_claimed=False,
                    engine_environment={name: os.environ.get(name) for name in
                        ('VLLM_USE_FLASHINFER_SAMPLER','VLLM_ALLREDUCE_USE_FLASHINFER','VLLM_MXFP4_USE_MARLIN')})
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = args.output / 'protocol.json'
    require(not protocol_path.exists() or read(protocol_path) == protocol,
            'Output directory belongs to a different frozen fixed-candidate protocol')
    frozen.write(protocol_path, protocol)
    args.protocol_sha256 = text_sha(frozen.serial(protocol))
    check_existing_records(args.output, cases, args.protocol_sha256)
    preserve_sources(args.output, bindings)
    if args.dry_run:
        frozen.emit('dry_run_complete', episodes=len(cases), tasks_per_condition=len(panel),
                    conditions=list(templates), candidate_p_user={row['condition']: row['probe_evidence']['goal_user_mean']
                                                                  for row in candidates}, output=str(args.output))
        return
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    frozen.write(args.output / 'launch.json', dict(pid=os.getpid(), started_at=time.time()))
    frozen.emit('loading_vllm', configuration=config)
    generator = frozen.AsyncGenerator(config, args.max_tokens)
    records, started = [], time.time()
    try:
        frozen.emit('vllm_ready', episodes=len(cases), workers=args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(frozen.run_episode, case, suites[case['suite']], args,
                                   dataset, native, tokenizer, encoding, generator) for case in cases]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                frozen.write(args.output / 'progress.json', dict(terminal=len(records), scheduled=len(cases),
                              elapsed_seconds=time.time()-started, last_case=record['case_id']))
                frozen.emit('case_complete', case_id=record['case_id'], status=record['status'],
                            security=record['security'], utility=record['utility'],
                            payload_exposed=record['payload_exposed'], error=record.get('error'),
                            terminal=len(records), scheduled=len(cases))
        summary = dict(scope='full', tasks_per_condition=389, scheduled_episodes=len(cases),
                       terminal=len(records), elapsed_seconds=time.time()-started,
                       conditions={condition: bounded_metrics([row for row in records if row['condition']==condition],389)
                                   for condition in templates},
                       suite_metrics={condition:{suite:bounded_metrics(
                           [row for row in records if row['condition']==condition and row['suite']==suite],
                           sum(item['suite']==suite for item in panel)) for suite in COUNTS} for condition in templates},
                       candidate_p_user={row['condition']:row['probe_evidence']['goal_user_mean'] for row in candidates},
                       errors=[dict(case_id=row['case_id'],error=row['error']) for row in records if row['status']=='error'])
        frozen.write(args.output / 'summary.json', summary)
        frozen.emit('evaluation_complete', summary=summary)
    finally:
        generator.close()


if __name__ == '__main__':
    main()
