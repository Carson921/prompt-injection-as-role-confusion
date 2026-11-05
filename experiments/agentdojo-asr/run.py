"""Native GPT-OSS continuation; frozen full inputs and real tool environments."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from common import BASE, OUTPUT, MODEL, PROJECT, BENCHMARK_PYTHON, read, write, sha, serial, signature, phase

os.environ['HF_HOME'] = str(PROJECT / 'experiments/single-message-pre-mlp/outputs/huggingface')
os.environ['TRITON_CACHE_DIR'] = '/tmp/wz55191-agentdojo-first-tool-triton'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)


def emit(event, **value):
    print(serial({'event': event, **value}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(BASE / 'provenance'))
    sys.path.insert(0, str(BASE / 'postprocessing'))
    from spans import render
    from spans_v2 import locate
    spec = importlib.util.spec_from_file_location('frozen_first_payload_engine', BASE / 'provenance/collect.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    original = read(BASE / 'manifest.json')
    identity = read(BASE / 'agentdojo-source.json')
    assert all(sha(Path(identity['source']) / name) == digest for name, digest in identity['files'].items())
    assert all(sha(BASE / 'provenance' / name) == digest for name, digest in original['code_sha256'].items())
    assert all(sha(MODEL / name) == digest for name, digest in original['model_binding']['assets_sha256'].items())
    staging = read(MODEL / 'staging-proof.json')
    assert staging['all_shards_verified']
    assert {s['name']: s['sha256'] for s in staging['shards']} == original['model_binding']['weight_shards_sha256']
    assert all((MODEL / s['name']).stat().st_size == s['bytes'] for s in staging['shards'])
    code = Path(__file__).resolve().parent
    manifest = {'format': 'agentdojo-native-asr-continuation-v1', 'base_collection': str(BASE),
        'base_manifest_sha256': sha(BASE / 'manifest.json'), 'canonical_cases_sha256': sha(BASE / 'canonical_cases.json'),
        'schedule_sha256': sha(BASE / 'schedule.json'), 'source_sha256': sha(BASE / 'agentdojo-source.json'),
        'model_binding': original['model_binding'], 'generation': original['generation'],
        'query_tile': original['query_tile'], 'system_order': original['system_order'],
        'frozen_engine_sha256': sha(BASE / 'provenance/collect.py'),
        'continuation_code_sha256': {p.name: sha(p) for p in sorted(code.glob('*.py'))},
        'restore_policy': 'Reexecute real AgentDojo tools in fresh environments with saved native outputs; compare original request and every full prompt token through first exposure.',
        'continuation_policy': 'Generate at the exact first-exposure input; continue the unmodified AgentDojo pipeline and its native 15-iteration loop and three-attempt task policy.',
        'evaluation': 'AgentDojo v1.2.1 TaskSuite.run_task_with_pipeline; security is attacker success; booleans verified again from saved pre/final environment and function stack trace.',
        'unexposed_policy': 'Replay all saved calls and reevaluate completed cases; assert equality with existing booleans.',
        'cache_policy': 'Exact full token IDs + frozen model/runtime/generation binding only; no approximate reuse.',
        'row_alignment': 'Original canonical case order and unchanged feature_row_start/end; no recollection or rewriting of hidden states.'}
    if (OUTPUT / 'manifest.json').exists():
        assert read(OUTPUT / 'manifest.json') == manifest, 'Frozen continuation binding changed'
    else:
        write(OUTPUT / 'manifest.json', manifest)
    binding = signature(manifest)
    provenance = OUTPUT / 'provenance'
    provenance.mkdir(exist_ok=True)
    for p in code.glob('*.py'):
        if p.resolve() != (provenance / p.name).resolve():
            shutil.copyfile(p, provenance / p.name)
    canonical = {r['case_id']: r for r in read(BASE / 'canonical_cases.json')}
    torch.set_num_threads(2)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    emit('loading_model', cases=len(canonical), devices=2)
    model, device_map = engine.load_model(MODEL)
    write(OUTPUT / 'backend.json', {'device_map': device_map, 'gpus': [torch.cuda.get_device_name(i) for i in range(2)],
        'slurm_job': os.environ.get('SLURM_JOB_ID'), 'native_mxfp4': True, 'activation_dtype': 'bfloat16',
        'base_backend_sha256': sha(BASE / 'backend.json')})
    engine.validate(model, tok, original['head_ids_union'], OUTPUT, original['query_tile'])
    command = [BENCHMARK_PYTHON, str(provenance / 'continuation_bridge.py'), '--output', str(OUTPUT)]
    if args.max_cases:
        command += ['--max-cases', str(args.max_cases)]
    completed = {p.relative_to(OUTPUT / 'cases').as_posix()[:-5]: read(p)
                 for p in (OUTPUT / 'cases').rglob('*.json')} if (OUTPUT / 'cases').exists() else {}
    completed = {k: v for k, v in completed.items() if v['status'] == 'completed'}
    started = time.time()
    with (OUTPUT / 'bridge.log').open('a') as errors:
        bridge = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
        try:
            for line in bridge.stdout:
                event = json.loads(line)
                if event['event'] != 'request':
                    emit(**event)
                    if event['event'] == 'case_complete':
                        assert event['status'] == 'completed', f"Case failed: {event}"
                        completed[event['case_id']] = event
                        write(OUTPUT / 'progress.json', {'stage': 'running', 'completed': len(completed),
                            'total': len(canonical), 'attacker_successes': sum(x.get('attacker_success') is True for x in completed.values()),
                            'elapsed_seconds': time.time() - started, 'last_case': event['case_id'],
                            'pid': os.getpid(), 'updated_at': time.time()})
                    continue
                path = Path(event['path'])
                request = read(path)
                directory = path.parent
                origin = canonical[request['case_id']]
                old_phase = phase(origin)
                old_result = read(BASE / origin['source_result'])
                index = request['call_index']
                exposure_index = int(origin['first_exposure_call'].rsplit('_', 1)[1]) if origin['status'] == 'captured' else None
                replay = origin['status'] == 'unexposed' or index < exposure_index
                original_call = replay or index == exposure_index
                prompt = render(tok, request, original['generation']['prompt_date'])
                ids = tok.encode(prompt, add_special_tokens=False)
                assert len(ids) <= model.config.max_position_embeddings, 'Refusing prompt truncation'
                location = locate(tok, request, prompt, original['generation']['prompt_date']) if original_call else None
                reference = None
                if original_call:
                    old_directory = old_phase / 'calls' / request['call_id']
                    old_request_path = old_directory / 'original_request.json'
                    old_input_path = old_directory / 'input.json'
                    assert serial(request) == serial(read(old_request_path)), 'AgentDojo replay request diverged'
                    old_input = read(old_input_path)
                    assert ids == old_input['input_ids'] and prompt == old_input['rendered_prompt'], 'Replay prompt tokens diverged'
                    previous = old_result['requests'][index]
                    assert sha(old_request_path) == previous['request_sha256']
                    assert bool(location) == (index == exposure_index), 'First exposure boundary differs'
                    reference = {'path': str(old_input_path.relative_to(BASE)), 'sha256': sha(old_input_path),
                                 'original_request_path': str(old_request_path.relative_to(BASE)),
                                 'original_request_sha256': sha(old_request_path), 'full_token_ids_equal': True,
                                 'rendered_prompt_equal': True, 'raw_request_equal': True}
                write(directory / 'input.json', {'call_id': request['call_id'], 'rendered_prompt': prompt, 'input_ids': ids,
                    'binding': binding, 'original_request_sha256': sha(path), 'base_input': reference,
                    'is_first_exposure': index == exposure_index, 'input_tokens': len(ids)})
                call_started = time.time()
                cache_key = signature({'binding': binding, 'input_ids': ids})
                cache_path = OUTPUT / 'generation_cache' / (cache_key + '.json')
                mode = 'prefix_replay' if replay else 'native_generation'
                cache_hit = False
                source_response = None
                if replay:
                    source_path = old_phase / 'calls' / request['call_id'] / 'response.json'
                    cached = read(source_path)
                    assert sha(source_path) == old_result['requests'][index]['response_sha256']
                    generated = cached['generated_ids']
                    source_response = {'path': str(source_path.relative_to(BASE)), 'sha256': sha(source_path)}
                elif cache_path.exists():
                    cached = read(cache_path)
                    assert cached['binding'] == binding and cached['input_ids'] == ids
                    generated = cached['generated_ids']
                    cache_hit = True
                else:
                    emit('native_generation_start', call_id=request['call_id'], input_tokens=len(ids), first_exposure=index == exposure_index)
                    generated = engine.generate(model, ids, original['generation']['max_tokens'])
                    write(cache_path, {'binding': binding, 'input_ids': ids, 'generated_ids': generated,
                                      'first_native_call': request['call_id'], 'seconds': time.time() - call_started})
                record = {'call_id': request['call_id'], 'generated_ids': generated,
                          'generated_text': tok.decode(generated, skip_special_tokens=False), 'mode': mode,
                          'input_sha256': sha(directory / 'input.json'), 'binding': binding,
                          'source_response': source_response, 'cache_key': cache_key if not replay else None,
                          'cache_hit': cache_hit, 'cache_sha256': sha(cache_path) if not replay else None,
                          'seconds': time.time() - call_started}
                write(directory / 'response.json', record)
                if not replay:
                    emit('native_generation_complete', call_id=request['call_id'], output_tokens=len(generated),
                         cache_hit=cache_hit, seconds=record['seconds'])
                answer = {'call_id': request['call_id'], 'generated_ids': generated,
                          'response_sha256': sha(directory / 'response.json'), 'mode': mode}
                bridge.stdin.write(serial(answer) + '\n')
                bridge.stdin.flush()
            assert bridge.wait() == 0, 'AgentDojo bridge exited unsuccessfully'
        finally:
            if bridge.poll() is None:
                bridge.terminate()
                bridge.wait(timeout=30)
    emit('producer_terminal', completed=len(completed), requested=args.max_cases or len(canonical))
    write(OUTPUT / 'producer-terminal.json', {'status': 'terminal', 'completed': len(completed),
        'requested': args.max_cases or len(canonical), 'pid': os.getpid(), 'finished_at': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        emit('fatal_error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
