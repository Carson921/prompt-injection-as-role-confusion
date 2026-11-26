"""Frozen native 120B feature readouts and harmless first-response continuations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT / 'experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz'
PROBE_SHA = 'fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33'
ROLES = ['system', 'developer', 'user', 'cot', 'assistant', 'tool']
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_generation(prompt_ids, generated_ids, tok, maximum):
    """Only a completed, ordinary final message can satisfy the marker goal."""
    tail = prompt_ids[prompt_ids.index(200006, len(prompt_ids) - 3):] if 200006 in prompt_ids[-3:] else []
    text = tok.decode(tail + generated_ids, skip_special_tokens=False)
    frames = []
    pattern = r'<\|start\|>assistant(?P<header>.*?)<\|message\|>(?P<body>.*?)(?P<ending><\|(?:end|return|call)\|>|$)'
    for m in re.finditer(pattern, text, flags=re.S):
        header = m['header']
        kind = 'tool_call' if 'to=' in header else 'analysis' if '<|channel|>analysis' in header else 'final'
        frames.append(dict(kind=kind, header=header, body=m['body'], ending=m['ending']))
    ending = generated_ids[-1] if generated_ids else None
    stop = {200002: 'final', 200012: 'tool_call', 199999: 'eos'}.get(ending, 'length' if len(generated_ids) == maximum else 'other')
    final = '\n'.join(f['body'] for f in frames if f['kind'] == 'final' and f['ending'] == '<|return|>')
    return dict(generated_text=tok.decode(generated_ids, skip_special_tokens=False),
                frames=frames, final_text=final, stop_reason=stop, length_censored=stop == 'length')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=HERE / 'outputs/gptoss120b-20261125')
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--probe', type=Path, default=PROBE)
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--mode', choices=['readouts', 'behavior', 'both'], default='both')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    out = a.output.resolve()
    assert 0 <= a.worker < a.workers
    assert sha(a.probe) == PROBE_SHA
    os.environ['HF_HOME'] = str(ROOT / 'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR'] = f'/tmp/wz55191-cot-forgery-20261125-worker{a.worker}'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(20261125)
    scorer = load_module('_cot_forgery_scorer', ROOT / 'experiments/autodan-role-probe/scorer.py')
    collect, _, attention, classifier = scorer.dependencies()
    probe = classifier.load_probe(a.probe)
    assert probe.metadata['roles'] == ROLES
    binding = probe.metadata['model_binding']
    for name, digest in binding['assets_sha256'].items():
        assert sha(a.model / name) == digest, name
    proof = json.loads((a.model / 'staging-proof.json').read_text())
    assert proof['all_shards_verified']
    assert {r['name']: r['sha256'] for r in proof['shards']} == binding['weight_shards_sha256']
    for r in binding['weights']:
        assert (a.model / r['name']).stat().st_size == r['bytes']
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    start = time.monotonic()
    model, device_map = collect.load_model(a.model)
    collect.validate(model, tok, probe.metadata['heads'], out/f'validation-worker{a.worker}', 256)
    net = probe.torch(device='cpu').eval()
    mean, scale = torch.from_numpy(probe.mean), torch.from_numpy(probe.scale)
    runtime = dict(worker=a.worker, workers=a.workers, device_map=device_map,
                   packages={x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'kernels', 'triton', 'accelerate', 'numpy']},
                   gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   model_load_seconds=time.monotonic()-start, probe_sha256=PROBE_SHA,
                   model_staging_proof_sha256=sha(a.model / 'staging-proof.json'),
                   model_weight_verification='Reused bound staging proof; file sizes checked now; no new full weight digest claim.',
                   code_sha256={str(x.relative_to(ROOT)): sha(x) for x in [Path(__file__), ROOT/'experiments/agentdojo-first-tool/collect.py', ROOT/'experiments/agentdojo-first-tool/attention.py', ROOT/'experiments/role-analysis/synthetic_agent_classifier.py']})
    write(out / f'runtime-worker{a.worker}.json', runtime)
    print(json.dumps(dict(event='model_loaded', **runtime)), flush=True)
    results = []
    if a.mode in ('readouts', 'both'):
        rows = lines(out / 'inputs.jsonl')
        if (out / 'behavior-inputs.jsonl').exists(): rows += lines(out / 'behavior-inputs.jsonl')
        selected = rows[a.worker::a.workers]
        if a.limit: selected = selected[:a.limit]
        for index, row in enumerate(selected):
            t = time.monotonic()
            name = row['input_id']
            positions = np.asarray(sorted({i for s in row['segments'].values() for i in s}), dtype=np.int64)
            signature = dict(input_sha256=object_sha(row), probe_sha256=PROBE_SHA,
                             runtime_code_sha256=runtime['code_sha256'])
            path = out / 'readouts' / (name + '.npz')
            record_path = out / 'records' / (name + '.json')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                assert record['signature'] == signature and sha(path) == record['readout_sha256']
            else:
                assert len(positions) and positions.min() >= 0 and positions.max() < len(row['input_ids'])
                native, _ = collect.forward_capture(model, probe.metadata['heads'], row['input_ids'], positions)
                features = native.to(torch.float16).reshape(len(positions), 1024).numpy()
                with torch.inference_mode():
                    logits = net((torch.from_numpy(features).float()-mean)/scale).numpy()
                    probabilities = torch.from_numpy(logits).softmax(1).numpy()
                assert np.isfinite(features).all() and np.isfinite(logits).all()
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, positions=positions, token_ids=np.asarray(row['input_ids'], np.int64)[positions],
                                    head_features=features, logits=logits, probabilities=probabilities, roles=np.asarray(ROLES))
                temp.replace(path)
                record = dict(status='complete', input_id=name, signature=signature, readout_sha256=sha(path),
                              input_tokens=len(row['input_ids']), scored_tokens=len(positions), seconds=time.monotonic()-t)
                write(record_path, record)
            results.append(record)
            print(json.dumps(dict(event='readout', worker=a.worker, done=index+1, total=len(selected),
                                  input_id=name, seconds=round(time.monotonic()-t, 3))), flush=True)
    if a.mode in ('behavior', 'both'):
        rows = lines(out / 'behavior-inputs.jsonl')[a.worker::a.workers]
        if a.limit: rows = rows[:a.limit]
        behaviors = []
        for index, row in enumerate(rows):
            path = out / 'continuations' / (row['input_id'] + '.json')
            signature = dict(input_sha256=object_sha(row), max_new_tokens=a.max_new_tokens,
                             decoding='greedy', seed=20261125, runtime_code_sha256=runtime['code_sha256'])
            if path.exists():
                result = json.loads(path.read_text())
                assert result['signature'] == signature
            else:
                t = time.monotonic()
                generated = collect.generate(model, row['input_ids'], a.max_new_tokens)
                parsed = parse_generation(row['input_ids'], generated, tok, a.max_new_tokens)
                marker = row['marker']
                result = dict(input_id=row['input_id'], trace_id=row['trace_id'], source_split=row['source_split'],
                              source_suite=row['source_suite'], condition=row['condition'], marker=marker,
                              signature=signature, generated_ids=generated, generated_tokens=len(generated),
                              seconds=time.monotonic()-t, tool_execution=False,
                              exact_final_success=parsed['final_text'].strip() == marker,
                              contains_final_success=marker in parsed['final_text'],
                              contains_anywhere=marker in parsed['generated_text'], **parsed)
                write(path, result)
            behaviors.append(result)
            print(json.dumps(dict(event='continuation', worker=a.worker, done=index+1, total=len(rows),
                                  input_id=row['input_id'], success=result['exact_final_success'],
                                  stop=result['stop_reason'], seconds=round(result['seconds'], 2))), flush=True)
        write(out / f'behavior-worker{a.worker}.json', behaviors)
    write(out / f'worker{a.worker}-manifest.json', dict(status='complete', mode=a.mode, limited=bool(a.limit),
                                                    readouts=len(results), seconds=time.monotonic()-start,
                                                    readout_ids=[r['input_id'] for r in results], runtime_sha256=sha(out/f'runtime-worker{a.worker}.json')))


if __name__ == '__main__': main()
