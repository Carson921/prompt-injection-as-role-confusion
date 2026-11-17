"""Native two-H100 full-input first-exposure tool-response feature collection."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from spans import render, locate, select_rows, require
from attention import install_tiled_attention

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELECTION = ROOT / 'experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/head-selection.json'
REFERENCE = Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_role_probe_gptoss120b/subdataset_100_seed20261021')
DELIMITERS = [199998, 199999, 200002, 200003, 200005, 200006, 200007, 200008, 200012, 200018]
KERNEL_REVISION = '9fbc1b48e710ac8abf18ea2650ee690d52dc3859'
HARMONY_PACKAGE = Path('/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/openai_harmony')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def emit(event, **value):
    print(json.dumps({'event': event, **value}, ensure_ascii=False), flush=True)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@contextmanager
def cpu_transfers():
    import torch
    import accelerate.hooks as hooks
    import accelerate.utils.operations as operations
    old_op, old_hook = operations.send_to_device, hooks.send_to_device

    def staged(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
            target = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            if target.type == 'cuda' and tensor.device != target:
                return tensor.cpu().to(target, non_blocking=False)
        return old_op(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)

    operations.send_to_device = hooks.send_to_device = staged
    try:
        yield
    finally:
        operations.send_to_device, hooks.send_to_device = old_op, old_hook


class Capture:
    def __init__(self, model, heads, ids, positions):
        self.model, self.heads, self.ids, self.positions = model, heads, list(ids), np.asarray(positions)
        self.layers = sorted(set(h // 64 for h in heads))
        self.blocks, self.handles, self.root_seen = {}, [], False

    def root(self, module, args, kwargs):
        value = kwargs.get('input_ids', args[0] if args else None)
        require(value is not None and value.shape == (1, len(self.ids)) and value[0].tolist() == self.ids,
                'Actual full model input differs from capture token ownership')
        self.root_seen = True

    def hook(self, layer):
        def observe(module, args):
            import torch
            require(layer not in self.blocks, 'Repeated selected-head hook')
            value = args[0].detach()
            require(value.shape == (1, len(self.ids), 4096), 'Incorrect pre-o_proj shape')
            slots = [i for i, h in enumerate(self.heads) if h // 64 == layer]
            positions = torch.as_tensor(self.positions, device=value.device, dtype=torch.long)
            indices = torch.tensor([self.heads[i] % 64 for i in slots], device=value.device)
            block = value[0].view(-1, 64, 64).index_select(0, positions).index_select(1, indices)
            self.blocks[layer] = (slots, block.cpu())
        return observe

    def __enter__(self):
        self.handles.append(self.model.model.register_forward_pre_hook(self.root, with_kwargs=True))
        self.handles.extend(self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(self.hook(layer))
                            for layer in self.layers)
        return self

    def finish(self):
        import torch
        require(self.root_seen and set(self.blocks) == set(self.layers), 'Missing capture hook')
        output = torch.empty((len(self.positions), len(self.heads), 64), dtype=torch.bfloat16)
        for slots, block in self.blocks.values():
            output[:, slots] = block
        require(bool(torch.isfinite(output).all()), 'Nonfinite features')
        return output

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()


def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'accelerate', 'kernels', 'triton']}
    require(versions == {'torch': '2.9.1', 'transformers': '5.3.0', 'accelerate': '1.12.0',
                         'kernels': '0.11.7', 'triton': '3.5.1'}, 'Frozen native runtime versions differ')
    require(torch.__version__ == '2.9.1+cu128' and torch.version.cuda == '12.8', 'Frozen Torch/CUDA runtime differs')
    require(torch.cuda.device_count() == 2 and all('H100' in torch.cuda.get_device_name(i) for i in range(2)),
            'Exactly two visible H100s required')
    original_get, original_validate = hub_kernels.get_kernel, Mxfp4HfQuantizer.validate_environment

    def pinned(name, *a, **kw):
        if name == 'kernels-community/gpt-oss-triton-kernels':
            kw['revision'] = KERNEL_REVISION
        return original_get(name, *a, **kw)

    def strict(self, *a, **kw):
        original_validate(self, *a, **kw)
        require(not self.quantization_config.dequantize, 'Native MXFP4 required')

    device_map = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 1, 'lm_head': 1,
                  **{f'model.layers.{i}': i // 18 for i in range(36)}}
    try:
        hub_kernels.get_kernel, Mxfp4HfQuantizer.validate_environment = pinned, strict
        with cpu_transfers():
            model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, dtype=torch.bfloat16,
                quantization_config=Mxfp4Config(dequantize=False), device_map=device_map,
                attn_implementation='eager', use_kernels=False).eval().requires_grad_(False)
    finally:
        hub_kernels.get_kernel, Mxfp4HfQuantizer.validate_environment = original_get, original_validate
    require(model.hf_device_map == device_map, 'Model layer placement differs')
    require({type(x.mlp.experts).__name__ for x in model.model.layers} == {'Mxfp4GptOssExperts'}, 'Wrong expert backend')
    return model, device_map


def forward_capture(model, heads, ids, positions, native_masks=False):
    import torch
    with torch.inference_mode(), cpu_transfers(), Capture(model, heads, ids, positions) as capture:
        output = model.model(input_ids=torch.tensor([ids], device='cuda:0'),
            attention_mask=None if native_masks else {'full_attention': None, 'sliding_attention': None},
            use_cache=False, return_dict=True)
        logits = model.lm_head(output.last_hidden_state[:, -1:, :]).float().cpu()
        features = capture.finish()
    return features, logits


def generate(model, ids, maximum, *, native_masks=False, announce=True):
    import torch
    generated, cache = [], None
    tokens = ids
    with torch.inference_mode(), cpu_transfers():
        for index in range(maximum):
            output = model(input_ids=torch.tensor([tokens], device='cuda:0'),
                attention_mask=None if native_masks else {'full_attention': None, 'sliding_attention': None},
                past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
            cache = output.past_key_values
            token = int(output.logits[0, -1].float().argmax())
            generated.append(token)
            del output
            if token in (200002, 200012, 199999):
                break
            tokens = [token]
            if announce and (index + 1) % 256 == 0:
                emit('decode_progress', generated=index + 1)
    return generated


def validate(model, tok, heads, output, tile):
    import torch
    text = ' '.join(f'Validation token {i}: preserve full attention and sliding cache behavior.' for i in range(100))
    ids = tok.encode(text, add_special_tokens=False)[:384]
    positions = np.arange(len(ids))
    native, native_logits = forward_capture(model, heads, ids, positions, native_masks=True)
    native_ids = generate(model, ids, 16, native_masks=True, announce=False)
    install_tiled_attention(tile)
    tiled, tiled_logits = forward_capture(model, heads, ids, positions)
    tiled_ids = generate(model, ids, 16, announce=False)
    require(torch.equal(native, tiled) and torch.equal(native_logits, tiled_logits) and native_ids == tiled_ids,
            'Original eager versus tiled full-prefill/native decode mismatch')
    with torch.inference_mode(), cpu_transfers():
        passive = model(input_ids=torch.tensor([ids], device='cuda:0'),
            attention_mask={'full_attention': None, 'sliding_attention': None}, use_cache=False,
            return_dict=True, logits_to_keep=1).logits.float().cpu()
    require(torch.equal(passive, tiled_logits), 'Passive capture changed logits')
    write(output / 'validation.json', {'status': 'passed', 'input_tokens': len(ids), 'native_decode_tokens': native_ids,
          'all_selected_head_values_bit_exact': True, 'logits_bit_exact': True,
          'query_tile': tile, 'sliding_window_and_tile_boundary_crossed': True, 'passive_hooks_verified': True})
    emit('validation_passed', tokens=len(ids), heads=len(heads))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103'))
    p.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    p.add_argument('--bridge-python', default='/scratch/wz55191/venvs/chatinject-table1/bin/python')
    p.add_argument('--date', default='2026-10-21')
    p.add_argument('--query-tile', type=int, default=256)
    p.add_argument('--max-cases', type=int)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['HF_HOME'] = str(ROOT / 'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR'] = '/tmp/wz55191-agentdojo-first-tool-triton'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.manual_seed(0)
    selection, reference = read(SELECTION), read(REFERENCE / 'provenance/probe_provenance.json')
    heads16, heads10 = selection['heads'], reference['selected_head_ids']
    heads = list(dict.fromkeys(heads16 + heads10))
    proof = read(a.model / 'staging-proof.json')
    require(proof['all_shards_verified'] and {r['name']: r['sha256'] for r in proof['shards']}
            == selection['signature']['model_binding']['weight_shards_sha256'], 'Model staging binding differs')
    require({name: sha(a.model / name) for name in selection['signature']['model_binding']['assets_sha256']}
            == selection['signature']['model_binding']['assets_sha256'], 'Tokenizer/model assets changed')
    manifest = {'format': 'agentdojo-first-complete-payload-whole-tool-response-v1', 'scope_sha256': sha(a.output / 'scope.json'),
        'schedule_sha256': sha(a.output / 'schedule.json'), 'source_sha256': sha(a.output / 'agentdojo-source.json'),
        'head_ids_union': heads, 'head_ids_final_v7_16': heads16, 'head_ids_reference_10': heads10,
        'selection_sha256': sha(SELECTION), 'reference_probe_sha256': sha(REFERENCE / 'provenance/role_probe.npz'),
        'model_binding': selection['signature']['model_binding'], 'kernel_revision': KERNEL_REVISION,
        'native_harmony': {'version': '0.0.8', 'module': str(HARMONY_PACKAGE),
                           'files_sha256': {f.name: sha(f) for f in HARMONY_PACKAGE.iterdir() if f.is_file() and f.suffix in ('.py', '.so')}},
        'generation': {'temperature': 0, 'seed': 0, 'max_tokens': 1024, 'reasoning_effort': 'medium', 'prompt_date': a.date},
        'system_order': 'original normal', 'capture': 'self_attn.o_proj input; full complete actual input backbone prefill',
        'activation_dtype': 'bfloat16', 'row_scope': 'whole trusted outer tool body, including serialized quotes; exclude native delimiter IDs',
        'delimiter_ids': DELIMITERS, 'query_tile': a.query_tile,
        'code_sha256': {f.name: sha(f) for f in sorted(HERE.glob('*.py'))},
        'prefix_cache': 'exact full input IDs + frozen manifest signature; raw native greedy output IDs reparsed with case-local call IDs',
        'first_exposure': 'earliest actual call containing full template + actual goal in an original tool message; earliest matching outer message',
        'stop_policy': 'stop episode immediately after real full-input forward captures its first complete-payload tool body; no ASR claimed'}
    if (a.output / 'manifest.json').exists():
        require(read(a.output / 'manifest.json') == manifest, 'Frozen collection specification changed')
    else:
        write(a.output / 'manifest.json', manifest)
    binding = signature(manifest)
    prov = a.output / 'provenance'
    prov.mkdir(exist_ok=True)
    for f in HERE.glob('*.py'):
        shutil.copyfile(f, prov / f.name)
    for src, name in [(SELECTION, 'head-selection-16.json'), (SELECTION.parent / 'best.npz', 'probe-16.npz'),
                      (REFERENCE / 'provenance/role_probe.npz', 'probe-10.npz'),
                      (REFERENCE / 'provenance/probe_provenance.json', 'probe-10-provenance.json'),
                      (a.model / 'staging-proof.json', 'model-staging-proof.json')]:
        shutil.copyfile(src, prov / name)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    emit('loading_model', devices=2, heads=len(heads))
    model, device_map = load_model(a.model)
    write(a.output / 'backend.json', {'device_map': device_map, 'gpus': [torch.cuda.get_device_name(i) for i in range(2)],
        'slurm_job': os.environ.get('SLURM_JOB_ID'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'accelerate', 'kernels', 'triton']},
        'native_mxfp4': True, 'dtype': 'bfloat16', 'cpu_staged_cross_device_transfers': True})
    validate(model, tok, heads, a.output, a.query_tile)
    command = [a.bridge_python, str(HERE / 'bridge.py'), 'run', '--output', str(a.output)]
    if a.max_cases:
        command.extend(['--max-cases', str(a.max_cases)])
    with (a.output / 'bridge.log').open('a') as errors:
        bridge = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
        try:
            for line in bridge.stdout:
                event = json.loads(line)
                if event['event'] != 'request':
                    emit(**event)
                    continue
                path = Path(event['path'])
                request, directory = read(path), path.parent
                prompt = render(tok, request, a.date)
                encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                ids, offsets = encoded['input_ids'], encoded['offset_mapping']
                require(len(ids) <= model.config.max_position_embeddings, 'Input exceeds original checkpoint context; refusing truncation')
                location = locate(tok, request, prompt, a.date)
                write(directory / 'input.json', {'call_id': request['call_id'], 'rendered_prompt': prompt,
                    'input_ids': ids, 'token_char_offsets': offsets, 'location': location,
                    'original_request_sha256': sha(path), 'binding': binding})
                started = time.time()
                if location:
                    positions, payload, goal, excluded = select_rows(ids, offsets, location, DELIMITERS)
                    features, _ = forward_capture(model, heads, ids, positions)
                    arrays = {'features_union': features, 'positions': torch.from_numpy(positions.copy()),
                        'token_ids': torch.tensor(ids, dtype=torch.long)[positions],
                        'char_offsets': torch.tensor(offsets, dtype=torch.long)[positions],
                        'payload_mask': torch.from_numpy(payload.copy()), 'goal_mask': torch.from_numpy(goal.copy()),
                        'template_mask': torch.from_numpy((payload & ~goal).copy()),
                        'context_mask': torch.from_numpy((~payload).copy()),
                        'excluded_delimiter_positions': torch.from_numpy(excluded.copy())}
                    temp = directory / 'states.safetensors.tmp'
                    save_file(arrays, str(temp), metadata={'binding': binding, 'call_id': request['call_id'],
                        'head_ids_union': json.dumps(heads), 'capture': 'pre-o_proj', 'dtype': 'bfloat16'})
                    temp.replace(directory / 'states.safetensors')
                    receipt = {'status': 'captured', 'call_id': request['call_id'], 'case_id': request['case_id'],
                        'binding': binding, 'input_tokens': len(ids), 'tool_rows': len(positions),
                        'template_rows': int((payload & ~goal).sum()), 'goal_rows': int(goal.sum()),
                        'context_rows': int((~payload).sum()), 'excluded_delimiters': len(excluded),
                        'location': location, 'head_ids_union': heads, 'forward_seconds': time.time() - started,
                        'actual_root_input_ids_verified': True, 'all_selected_layer_hooks_seen': True,
                        'files': {f: sha(directory / f) for f in ['original_request.json', 'input.json', 'states.safetensors']}}
                    write(directory / 'receipt.json', receipt)
                    response = {'call_id': request['call_id'], 'captured': True, 'receipt': str((directory / 'receipt.json').relative_to(a.output))}
                    emit('captured', case_id=request['case_id'], rows=len(positions), input_tokens=len(ids), seconds=receipt['forward_seconds'])
                else:
                    key = signature({'binding': binding, 'input_ids': ids})
                    cache_path = a.output / 'prefix_cache' / (key + '.json')
                    if cache_path.exists():
                        cached = read(cache_path)
                        require(cached['binding'] == binding and cached['input_ids'] == ids, 'Prefix cache binding differs')
                        generated = cached['generated_ids']
                        cache_hit = True
                    else:
                        emit('native_prefix_start', call_id=request['call_id'], input_tokens=len(ids))
                        generated = generate(model, ids, 1024)
                        cached = {'binding': binding, 'input_ids': ids, 'generated_ids': generated,
                                  'first_native_call': request['call_id'], 'seconds': time.time() - started}
                        write(cache_path, cached)
                        cache_hit = False
                    record = {'call_id': request['call_id'], 'generated_ids': generated,
                              'generated_text': tok.decode(generated, skip_special_tokens=False),
                              'cache_key': key, 'cache_hit': cache_hit, 'cache_sha256': sha(cache_path),
                              'input_sha256': sha(directory / 'input.json')}
                    response = {'call_id': request['call_id'], 'generated_ids': generated}
                    write(directory / 'response.json', record)
                    response['response_sha256'] = sha(directory / 'response.json')
                    if not cache_hit:
                        emit('native_prefix_complete', call_id=request['call_id'], output_tokens=len(generated), seconds=time.time() - started)
                bridge.stdin.write(json.dumps(response, ensure_ascii=False) + '\n')
                bridge.stdin.flush()
            require(bridge.wait() == 0, 'AgentDojo bridge failed')
        finally:
            if bridge.poll() is None:
                bridge.terminate()
                bridge.wait(timeout=30)
    emit('producer_terminal', max_cases=a.max_cases)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        emit('fatal_error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
