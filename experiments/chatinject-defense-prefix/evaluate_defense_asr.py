#!/usr/bin/env python3
"""Native AgentDojo ASR for ChatInject payloads with/without defensive prefixes.

The episode runner, vLLM backend, task panel (full 389 tasks), decoding and
metrics are the frozen ones used by evaluate_fixed_candidates.py.  Only the
payload policy differs: the attack part of the payload deliberately carries
genuine Harmony special tokens (ChatInject), so the "no special token" guard is
replaced by a recorder that confirms every exposed request really contains the
attack opening as genuine special-token IDs.  The defensive prefix itself must
pass the same ordinary-text admission as the optimizer.

Manifest: JSON list of {condition, defense, attack, probe?}; payload template =
defense + attack (attack contains the single {goal} slot).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AUTODAN, DATASET, SLOT  # noqa: E402

sys.path.insert(0, str(AUTODAN))
import evaluate_optimized as frozen  # noqa: E402
import evaluate_fixed_candidates as fixed  # noqa: E402
from broad_validation import BroadDelimiterLibrary  # noqa: E402
from prepare import COUNTS, SOURCE, read, sha  # noqa: E402

FLOOR = frozen.BASE_BPE_BOUND


def attack_opening(attack):
    """Leading genuine-special-token run of the attack (e.g. '<|end|><|start|>user<|message|>')."""
    head = attack.split(SLOT)[0]
    cut = head.rfind('|>')
    return head[:cut + 2] if head.startswith('<|') and cut > 0 else ''


def make_recorder(openings):
    def record(tokenizer, request, prompt, prompt_date, exposed):
        opening = openings.get(request['payload_template'], '')
        if not exposed:
            return dict(passed=True, checked_occurrences=0, policy='No complete decoded payload in this model request')
        if not opening:
            return dict(passed=True, checked_occurrences=0, policy='Text-only attack; no genuine special tokens expected')
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        starts = {start: index for index, (start, end) in enumerate(encoded['offset_mapping']) if end > start}
        found, genuine, cursor = 0, 0, prompt.find(opening)
        while cursor >= 0:
            found += 1
            index = starts.get(cursor)
            if index is not None and encoded['input_ids'][index] >= FLOOR:
                genuine += 1
            cursor = prompt.find(opening, cursor + 1)
        if found == 0 or genuine != found:
            raise ValueError(f'ChatInject opening not rendered as genuine special tokens: found={found} genuine={genuine}')
        return dict(passed=True, checked_occurrences=found, genuine_special_openings=genuine, opening=opening,
                    policy='ChatInject attack carries genuine Harmony tokens by design; every occurrence verified as special IDs')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output, args.model = args.output.resolve(), args.model.resolve()
    args.prompt_date, args.max_tokens, args.max_model_len = '2026-10-21', 1024, 65536
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    dataset = read(args.dataset)
    native = frozen.load_native(args.source)
    fixed.require(native['load_system_message'](None) == dataset['fixed_normal_system']['content'], 'System prompt differs')
    suites = native['get_suites']('v1.2.1')
    panel = fixed.full_panel(suites)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    library = BroadDelimiterLibrary(args.model)
    conditions = read(args.conditions)
    fixed.require(len({c['condition'] for c in conditions}) == len(conditions), 'Duplicate condition names')
    openings, templates, admissions = {}, {}, {}
    for c in conditions:
        defense, attack = c['defense'], c['attack']
        fixed.require(attack.count(SLOT) == 1 and SLOT not in defense, 'Bad goal slot')
        if defense:
            ids = tokenizer.encode(defense, add_special_tokens=False)
            fixed.require(not any(t >= FLOOR for t in ids) and library.validate(defense + SLOT, tokenizer=None),
                          'Defense prefix contains a genuine special token: ' + c['condition'])
        template = defense + attack
        fixed.require(template not in templates.values(), 'Duplicate payload template')
        templates[c['condition']] = template
        openings[template] = attack_opening(attack)
        admissions[c['condition']] = dict(defense_tokens=len(tokenizer.encode(defense, add_special_tokens=False)) if defense else 0,
                                          attack_opening=openings[template])
    frozen.validate_rendered_payloads = make_recorder(openings)
    cases = []
    for item in panel:
        goal = suites[item['suite']].injection_tasks[item['injection_task_id']].GOAL
        for condition, template in templates.items():
            cases.append(dict(**item, case_id=condition + '/' + item['pair_id'], condition=condition, actual_goal=goal,
                              payload_template=template, payload_text=template.replace(SLOT, goal)))
    config = dict(model=str(args.model), tokenizer=str(args.model), dtype='bfloat16', quantization='mxfp4',
                  tensor_parallel_size=2, pipeline_parallel_size=1, distributed_executor_backend='mp',
                  max_model_len=args.max_model_len, gpu_memory_utilization=.85, max_num_seqs=32,
                  max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=False,
                  enforce_eager=True, generation_config='vllm', seed=0, trust_remote_code=False)
    bindings = fixed.source_bindings(native)
    bindings.update({str(HERE / name): sha(HERE / name) for name in ('evaluate_defense_asr.py', 'common.py')})
    protocol = dict(schema='chatinject-defense-native-vllm-eval-v1', scope='full', conditions=conditions,
                    conditions_manifest_sha256=sha(args.conditions), templates=templates, admissions=admissions,
                    dataset_sha256=sha(args.dataset), cases=cases,
                    source_identity_sha256=sha(args.source / 'agentdojo-source.json'),
                    engine_configuration=config, max_tokens=args.max_tokens, reasoning_effort='medium',
                    prompt_date=args.prompt_date, seed=0, temperature=0., stop_ids=frozen.STOP_IDS,
                    benchmark='AgentDojo v1.2.1', normal_system=dataset['fixed_normal_system'],
                    initial_errors='null; no case replacement', source_bindings=bindings,
                    native_episode_runner_source_sha256=sha(AUTODAN / 'evaluate_optimized.py'),
                    model_staging_proof_sha256=sha(args.model / 'staging-proof.json'),
                    package_versions={n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'transformers', 'openai-harmony')},
                    payload_policy='attack part may contain genuine Harmony tokens (ChatInject); defense prefix ordinary text only',
                    engine_environment={n: os.environ.get(n) for n in
                                        ('VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_ALLREDUCE_USE_FLASHINFER', 'VLLM_MXFP4_USE_MARLIN')})
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = args.output / 'protocol.json'
    fixed.require(not protocol_path.exists() or read(protocol_path) == protocol, 'Output belongs to a different protocol')
    frozen.write(protocol_path, protocol)
    args.protocol_sha256 = fixed.text_sha(frozen.serial(protocol))
    fixed.check_existing_records(args.output, cases, args.protocol_sha256)
    fixed.preserve_sources(args.output, bindings)
    if args.dry_run:
        frozen.emit('dry_run_complete', episodes=len(cases), conditions=list(templates), output=str(args.output))
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
            futures = [pool.submit(frozen.run_episode, case, suites[case['suite']], args, dataset, native,
                                   tokenizer, encoding, generator) for case in cases]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                frozen.write(args.output / 'progress.json', dict(terminal=len(records), scheduled=len(cases),
                                                                 elapsed_seconds=time.time() - started, last_case=record['case_id']))
                frozen.emit('case_complete', case_id=record['case_id'], status=record['status'], security=record['security'],
                            utility=record['utility'], payload_exposed=record['payload_exposed'], error=record.get('error'),
                            terminal=len(records), scheduled=len(cases))
        summary = dict(scope='full', tasks_per_condition=len(panel), scheduled_episodes=len(cases), terminal=len(records),
                       elapsed_seconds=time.time() - started,
                       conditions={k: fixed.bounded_metrics([r for r in records if r['condition'] == k], len(panel)) for k in templates},
                       suite_metrics={k: {s: fixed.bounded_metrics([r for r in records if r['condition'] == k and r['suite'] == s],
                                                                   sum(p['suite'] == s for p in panel)) for s in COUNTS} for k in templates},
                       probe={c['condition']: c.get('probe') for c in conditions},
                       errors=[dict(case_id=r['case_id'], error=r['error']) for r in records if r['status'] == 'error'])
        frozen.write(args.output / 'summary.json', summary)
        frozen.emit('evaluation_complete', summary=summary)
    finally:
        generator.close()


if __name__ == '__main__':
    main()
