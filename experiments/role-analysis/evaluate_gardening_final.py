"""Evaluate the frozen final640 probe on the paper's exact gardening conversation.

This is a new, separate evaluation namespace. It never trains, changes a threshold,
edits an old result, or regenerates any message. Three archived prompt strings and
all 916 native body positions per format are retained, including the complete system
body. Only native-delimiter predictions have original-function accuracy labels.
The older substring archive missed one zero-width emoji continuation token; a
separate paired comparison uses its unchanged 915-token intersection.

Run with the role-probe environment::

    python evaluate_gardening_final.py --stage prepare --model-path /tmp/chatinject-role-probe-model
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_gardening_final.py --stage infer --model-path /tmp/chatinject-role-probe-model
    python evaluate_gardening_final.py --stage report

The infer stage requires four idle A40s and rehashes all official weight shards.
It does not kill, reset, wait for, or otherwise manage other GPU processes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = HERE / 'outputs/local-gptoss120b'
DEFAULT_OUT = BASE / 'gardening-final-checkpoint'
SOURCE = BASE / 'tomato'
PROBE = BASE / 'systemness-complementary-head-readout/models/new.npz'
PROBE_SHA = 'ef92b421007b7d0c7a11f13e91008e6d2e39b34b730c6ca7d6ecbf03590236ff'
INPUT_SHA = '6018b848a7fd12e0bc8d0e62246b805cef68e26e29d43ba87723352d0b167ce3'
MODEL_REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
KERNEL_REVISION = '9fbc1b48e710ac8abf18ea2650ee690d52dc3859'
HEADS = [458, 462, 2045, 460, 459, 2047, 456, 997, 247, 2248]
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
FORMATS = ['proper_tags', 'everything_in_user_tags', 'basic_no_format']
SUPPORT = dict(system=57, user=97, cot=179, assistant=583, tool=0)
OLD_SUPPORT = dict(SUPPORT, assistant=582)
THRESHOLD = 0.5201616161616162
ASSETS = {
    'config.json': '933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566',
    'tokenizer.json': '0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3',
    'tokenizer_config.json': '9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070',
    'special_tokens_map.json': 'dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293',
}
COLORS = dict(system='#90a1b9', user='#00a6f4', cot='#fd9a00', assistant='#00b980', tool='#9b5de5')


def require(value, message):
    if not bool(value):
        raise RuntimeError(message)


def clean(path):
    path = Path(os.path.abspath(str(path)))
    require(str(path) != '/workspace' and not str(path).startswith('/workspace/'), 'Do not access the unavailable /workspace mount.')
    return path


def sha(path):
    h = hashlib.sha256()
    with clean(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(clean(path).read_text())


def atomic_json(path, value):
    path = clean(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def check_assets(model_path):
    model_path = clean(model_path)
    for name, expected in ASSETS.items():
        require(sha(model_path / name) == expected, 'Frozen model/tokenizer asset differs: ' + name)
    config = read(model_path / 'config.json')
    require((config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (36, 64, 64)
            and config['quantization_config']['quant_method'] == 'mxfp4', 'Expected native MXFP4 120B geometry.')
    return dict(ASSETS)


def strict_native_roles(tokenizer, ids, expected_sequence=None):
    """Independent closed-token parser; classify every body position, never a header."""
    marker = {s: tokenizer.encode(s, add_special_tokens=False) for s in
              ['<|startoftext|>', '<|start|>', '<|message|>', '<|end|>', '<|return|>', '<|call|>']}
    require(all(len(v) == 1 for v in marker.values()), 'Expected single-ID Harmony markers.')
    marker = {k: v[0] for k, v in marker.items()}
    require(ids[0] == marker['<|startoftext|>'], 'Exact archived BOS required.')
    position, roles, segments = 1, {}, []
    header_roles = {'system': 'system', 'user': 'user',
                    'assistant<|channel|>analysis': 'cot', 'assistant<|channel|>final': 'assistant'}
    while position < len(ids):
        require(ids[position] == marker['<|start|>'], 'Unexpected token outside a native message.')
        position += 1
        start = position
        while position < len(ids) and ids[position] != marker['<|message|>']:
            require(ids[position] not in marker.values(), 'Malformed native header.')
            position += 1
        require(position < len(ids), 'Unclosed native header.')
        header = tokenizer.decode(ids[start:position], skip_special_tokens=False)
        require(header in header_roles, 'Unexpected archived native header: ' + header)
        role = header_roles[header]
        position += 1
        body_start = position
        while position < len(ids) and ids[position] not in (marker['<|end|>'], marker['<|return|>'], marker['<|call|>']):
            require(ids[position] not in marker.values(), 'Unexpected native marker in body.')
            roles[position] = role
            position += 1
        require(position < len(ids) and position > body_start, 'Unclosed or empty native body.')
        close = ids[position]
        require(close in ([marker['<|end|>'], marker['<|return|>']] if role == 'assistant' else [marker['<|end|>']]), 'Unexpected native closing delimiter.')
        segments.append(dict(role=role, start=body_start, end=position, header=header,
                             closing=tokenizer.decode([close], skip_special_tokens=False)))
        position += 1
    expected_sequence = expected_sequence or ['system', 'user', 'cot', 'assistant', 'user', 'cot', 'assistant']
    require([s['role'] for s in segments] == expected_sequence, 'Native message sequence changed.')
    return roles, segments


def source_binding():
    paths = [SOURCE / n for n in ['inputs.json', 'tokens.parquet', 'labels.parquet']]
    require(sha(paths[0]) == INPUT_SHA and sha(PROBE) == PROBE_SHA, 'Archived prompt or selected checkpoint changed.')
    config = HERE / 'config/tomato.yaml'
    template = HERE.parents[1] / 'utils/chat_templates/gptoss.j2'
    require(sha(config) == '3b9e3a055a764723ae1f0f32c5b0f98f458a4862f97b5bc9130387d22bb1cbff'
            and sha(template) == 'bcbbcd100a56ad723921990706697a40df73870d00f93746f00759f6f87994c9', 'Author-original gardening source or role template changed.')
    paths += [config, template, HERE / 'systemness_complementary_confirmation_infer.py',
              HERE / 'plot_single_message_tomato.py']
    return {str(p): sha(p) for p in paths + [PROBE, Path(__file__)]}


def prepare(args):
    from transformers import AutoTokenizer
    bindings = source_binding()
    check_assets(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(clean(args.model_path), local_files_only=True,
                                             add_bos_token=False, add_eos_token=False, padding_side='left')
    inputs = read(SOURCE / 'inputs.json')
    require(set(r['prompt_key'] for r in inputs) == set(FORMATS), 'Exactly three archived formats required.')
    labels = pd.read_parquet(SOURCE / 'labels.parquet')
    tokens = pd.read_parquet(SOURCE / 'tokens.parquet')
    prepared = []
    for key in FORMATS:
        row = next(r for r in inputs if r['prompt_key'] == key)
        ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        archive_tokens = tokens[tokens.prompt_ix.eq(row['prompt_ix'])].sort_values('token_in_prompt_ix')
        require(archive_tokens.token_in_prompt_ix.tolist() == list(range(len(ids)))
                and archive_tokens.token_id.tolist() == ids and ''.join(archive_tokens.token) == row['prompt'], 'Archived unpadded token sequence differs: ' + key)
        body = labels[labels.prompt_key.eq(key)].sort_values('token_in_prompt_ix')
        require(len(body) == 915 and body.base_message_type.value_counts().to_dict() == {k: v for k, v in OLD_SUPPORT.items() if v}
                and body.token_in_prompt_ix.is_unique, 'Expected the unchanged archived 915 content labels.')
        rows = [dict(token_position=int(r.token_in_prompt_ix), token_id=int(r.token_id), token=r.token,
                     message_ix=int(r.base_message_ix), semantic_role=r.base_message_type,
                     formal_role=r.base_message_type if key == 'proper_tags' else 'user' if key == 'everything_in_user_tags' else None,
                     archive_sample_ix=int(r.sample_ix), in_old_915_comparison=True) for r in body.itertuples()]
        # The notebook's substring labeler drops a real token whose reconstructed
        # string is empty: the continuation byte of the final seedling emoji.
        # It is part of the last assistant body, established by native parsing.
        zero = archive_tokens[archive_tokens.token.eq('')]
        require(len(zero) == 1, 'Expected exactly one archived zero-width emoji continuation.')
        continuation = zero.iloc[0]
        require(int(continuation.token_id) == 109 and int(continuation.token_in_prompt_ix) == rows[-1]['token_position'] + 1,
                'Unexpected emoji continuation identity or position.')
        rows.append(dict(token_position=int(continuation.token_in_prompt_ix), token_id=109, token='', message_ix=6,
                         semantic_role='assistant', formal_role='assistant' if key == 'proper_tags' else 'user' if key == 'everything_in_user_tags' else None,
                         archive_sample_ix=int(continuation.sample_ix), in_old_915_comparison=False))
        require(all(ids[r['token_position']] == r['token_id'] for r in rows), 'Body token/position mismatch.')
        if key == 'proper_tags':
            parsed, segments = strict_native_roles(tokenizer, ids)
            require(parsed == {r['token_position']: r['semantic_role'] for r in rows}, 'Independent native parser and archived substring labels differ.')
        else:
            segments = None
            require(sum(ids.count(tokenizer.encode(s, add_special_tokens=False)[0]) for s in ['<|start|>', '<|message|>', '<|end|>']) == (3 if key == 'everything_in_user_tags' else 0), 'Altered wrapper markers differ.')
            if key == 'everything_in_user_tags':
                formal, segments = strict_native_roles(tokenizer, ids, expected_sequence=['user'])
                require(formal == {r['token_position']: 'user' for r in rows}, 'Full all-user parser body coverage differs.')
            else:
                require([r['token_position'] for r in rows] == list(range(1, len(ids))), 'Untagged body excludes positions other than BOS.')
        prepared.append(dict(prompt_key=key, prompt=row['prompt'], prompt_ix=int(row['prompt_ix']),
                             prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(), token_ids=ids,
                             content=rows, native_segments=segments,
                             restored_emoji_continuation=dict(token_position=int(continuation.token_in_prompt_ix), token_id=109,
                                                              decoded_archive_token='', semantic_role='assistant',
                                                              formal_role=rows[-1]['formal_role'], message_ix=6)))
    reference = prepared[0]['content']
    for sample in prepared[1:]:
        require([(r['message_ix'], r['semantic_role']) for r in sample['content']] == [(r['message_ix'], r['semantic_role']) for r in reference], 'Altered body role order changed.')
        changed = [i for i, (a, b) in enumerate(zip(reference, sample['content'])) if a['token'] != b['token']]
        require(changed == [56, 101, 184, 514, 566, 662]
                and all(sample['content'][i]['token'] == reference[i]['token'] + '\n' for i in changed), 'Only the six known boundary LF token fusions are permitted.')
    manifest = dict(stage='gardening_final_prepared', source_bindings=bindings, model_assets_sha256=ASSETS,
                    checkpoint_path=str(PROBE), checkpoint_sha256=PROBE_SHA, selected_head_ids=HEADS,
                    roles=ROLES, system_threshold=THRESHOLD, official_model_revision=MODEL_REVISION,
                    kernel_revision=KERNEL_REVISION, formats=FORMATS, source_is_exact_archived_paper_gardening=True,
                    complete_system_body=True, tokens_per_format=916, native_support=SUPPORT,
                    old_archived_substring_tokens_per_format=915,
                    old_archive_omission='Final emoji continuation token ID109 has empty reconstructed text and was excluded by substring labels; native body parser restores it.',
                    restored_emoji_continuations={s['prompt_key']: s['restored_emoji_continuation'] for s in prepared},
                    native_independent_parser_agrees=True, model_forward_run=False, no_training=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in [('prepared-inputs.json', prepared), ('prepare-manifest.json', manifest)]:
        p = args.out / name
        require(not p.exists() or read(p) == value, 'Existing prepared artifact differs; use a new output namespace.')
        atomic_json(p, value)
    print(json.dumps(dict(stage='prepared', output=str(args.out), lengths=[len(x['token_ids']) for x in prepared])), flush=True)


def load_prepared(args):
    manifest = read(args.out / 'prepare-manifest.json')
    require(manifest['source_bindings'] == source_binding() and manifest['checkpoint_sha256'] == PROBE_SHA
            and manifest['selected_head_ids'] == HEADS and manifest['system_threshold'] == THRESHOLD, 'Frozen preparation source/checkpoint changed.')
    samples = read(args.out / 'prepared-inputs.json')
    require([s['prompt_key'] for s in samples] == FORMATS and all(len(s['content']) == 916 for s in samples), 'Prepared input coverage differs.')
    return manifest, samples


def verify_weights(model_path, verified_weights=None):
    """Require all 15 official shards; no trust in an arbitrary local self-report."""
    model_path = clean(model_path)
    check_assets(model_path)
    official = read(BASE / 'checkpoint-cache-verification.json')
    require(official['official_repository'] == 'openai/gpt-oss-120b' and official['revision'] == MODEL_REVISION
            and official['published_weights_sha256_verified'] is True and len(official['verified_shards']) == 15, 'Official historical shard manifest differs.')
    local = read(model_path / 'verified-weights.json')
    require(local.get('repository', local.get('official_repository')) == 'openai/gpt-oss-120b'
            and local['revision'] == MODEL_REVISION, 'Local snapshot provenance differs.')
    expected = {r['name']: r for r in official['verified_shards']}
    require(set(read(model_path / 'model.safetensors.index.json')['weight_map'].values()) == set(expected), 'Shard index is incomplete or changed.')
    reused = read(verified_weights) if verified_weights else None
    if reused is not None:
        require(reused['repository'] == 'openai/gpt-oss-120b' and reused['revision'] == MODEL_REVISION
                and clean(reused['cache_path']) == model_path and reused['all_published_shards_sha256_verified'] is True,
                'Independent full-shard verification is for another model/path.')
        reuse_files = {r['file']: r for r in reused['files']}
        require(set(reuse_files) == set(expected), 'Independent full-shard verification is incomplete.')
    proofs = []
    for name, record in sorted(expected.items()):
        p = model_path / name
        stat = p.stat()
        require(stat.st_size == record['size'], 'Official model shard size differs: ' + name)
        if reused is None:
            require(sha(p) == record['sha256'], 'Official model shard differs: ' + name)
        else:
            r = reuse_files[name]
            require(r['size'] == stat.st_size and r['mtime_ns'] == stat.st_mtime_ns
                    and r['inode'] == stat.st_ino and r['sha256'] == record['sha256'],
                    'Previously independently rehashed official shard changed: ' + name)
        proofs.append(dict(file=name, size=record['size'], sha256=record['sha256']))
        print(json.dumps(dict(stage='verified_weight', file=name)), flush=True)
    return dict(repository='openai/gpt-oss-120b', revision=MODEL_REVISION, model_path=str(model_path),
                all_15_shards_sha256_verified=True, shards=proofs, assets=ASSETS,
                shard_index_sha256=sha(model_path / 'model.safetensors.index.json'),
                independent_verification_path=str(verified_weights) if verified_weights else None,
                independent_verification_sha256=sha(verified_weights) if verified_weights else None)


def load_probe():
    import torch
    from systemness_complementary_confirmation_infer import make_portable
    require(sha(PROBE) == PROBE_SHA, 'Selected probe bytes changed.')
    with np.load(PROBE, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        require(meta['roles'] == ROLES and meta['selected_head_ids'] == HEADS and meta['kind'] == 'binary-residual-640'
                and meta['dimensions'] == 640 and meta['delta_formula'] == '4*tanh(g)', 'Selected probe metadata/order changed.')
        mean, scale = z['mean'].copy(), z['scale'].copy()
        state = {k[8:]: torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('weight::')}
    require(mean.shape == scale.shape == (640,) and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), 'Malformed probe scaler.')
    net = make_portable('new')
    net.load_state_dict(state, strict=True)
    return net, torch.from_numpy(mean), torch.from_numpy(scale)


@contextmanager
def model_context(model_path):
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    import accelerate.hooks
    import accelerate.utils.operations as operations
    require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Exactly four visible A40s required.')
    require(all(torch.cuda.mem_get_info(i)[0] >= 22 * 2**30 for i in range(4)), 'Four idle A40s with at least 22 GiB free each are required; no other process is modified.')
    old_get, old_validate = hub_kernels.get_kernel, Mxfp4HfQuantizer.validate_environment
    old_send, old_hook_send = operations.send_to_device, accelerate.hooks.send_to_device
    def pinned(name, *a, **kw):
        if name == 'kernels-community/gpt-oss-triton-kernels':
            kw['revision'] = KERNEL_REVISION
        return old_get(name, *a, **kw)
    def strict(self, *a, **kw):
        old_validate(self, *a, **kw)
        require(not self.quantization_config.dequantize, 'MXFP4 required; automatic BF16 dequantization refused.')
    def staged(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
            target = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            if target.type == 'cuda' and tensor.device != target:
                return tensor.cpu().to(target, non_blocking=False)
        return old_send(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)
    devices = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 3, 'lm_head': 3,
               **{f'model.layers.{i}': i // 9 for i in range(36)}}
    try:
        hub_kernels.get_kernel, Mxfp4HfQuantizer.validate_environment = pinned, strict
        operations.send_to_device, accelerate.hooks.send_to_device = staged, staged
        model = AutoModelForCausalLM.from_pretrained(clean(model_path), local_files_only=True, dtype=torch.bfloat16,
            quantization_config=Mxfp4Config(dequantize=False), device_map=devices, attn_implementation='eager', use_kernels=False).eval().requires_grad_(False)
        require({type(l.mlp.experts).__name__ for l in model.model.layers} == {'Mxfp4GptOssExperts'}
                and len(model.model.layers) == 36 and model.hf_device_map == devices
                and not model.hf_quantizer.quantization_config.dequantize and not any(p.requires_grad for p in model.parameters()), 'Expected frozen native MXFP4 backend.')
        yield model, dict(native_mxfp4=True, expert_backend='Mxfp4GptOssExperts', device_map=devices,
                          gpus=[torch.cuda.get_device_name(i) for i in range(4)], kernel_revision=KERNEL_REVISION,
                          gpu_transfer='Synchronous CPU staging between different CUDA devices')
    finally:
        hub_kernels.get_kernel, Mxfp4HfQuantizer.validate_environment = old_get, old_validate
        operations.send_to_device, accelerate.hooks.send_to_device = old_send, old_hook_send


def infer(args):
    import torch
    torch.set_num_threads(4)
    from systemness_complementary_confirmation_infer import Collector, install_hooks
    manifest, samples = load_prepared(args)
    require(not (args.out / 'inference-complete.json').exists(), 'Completed inference already exists; do not overwrite it.')
    # Fail before reading 65 GB if resources are occupied.
    require(torch.cuda.device_count() == 4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4))
            and all(torch.cuda.mem_get_info(i)[0] >= 22 * 2**30 for i in range(4)), 'Four available A40s with at least 22 GiB free each required.')
    kernel_snapshot = args.kernel_cache / 'models--kernels-community--gpt-oss-triton-kernels/snapshots' / KERNEL_REVISION
    require(kernel_snapshot.is_dir(), 'Exact pinned local MXFP4 kernel snapshot is unavailable.')
    kernel_files = {str(p.relative_to(kernel_snapshot)): sha(p) for p in sorted(kernel_snapshot.rglob('*.py'))}
    require(bool(kernel_files), 'Pinned local MXFP4 kernel source files are missing.')
    started = time.time()
    weights = verify_weights(args.model_path, args.verified_weights)
    atomic_json(args.out / 'weight-verification.json', weights)
    net, mean, scale = load_probe()
    predictions, parity = {}, {}
    with model_context(args.model_path) as (model, backend), torch.inference_mode():
        collector = Collector(HEADS)
        first = samples[0]
        ids = torch.tensor([first['token_ids']], dtype=torch.long, device='cuda:0')
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        reference = model.model(**kwargs).last_hidden_state.cpu()
        handles = install_hooks(model, collector)
        try:
            collector.begin([r['token_position'] for r in first['content']], enabled=False)
            hooked = model.model(**kwargs).last_hidden_state.cpu()
            collector.finish()
            require(torch.equal(reference, hooked) and torch.isfinite(hooked).all() and collector.features is None, 'Disabled pre-o_proj hooks changed model hidden states.')
            parity['disabled_hooks_hidden_bitexact'] = True
            del reference, hooked
            for sample in samples:
                ids = torch.tensor([sample['token_ids']], dtype=torch.long, device='cuda:0')
                collector.begin([r['token_position'] for r in sample['content']])
                hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                require(torch.isfinite(hidden).all(), 'Nonfinite final hidden state.')
                collector.finish()
                features = collector.features
                x = ((torch.from_numpy(features).float() - mean) / scale).contiguous()
                probabilities = net(x).exp().numpy().astype(np.float32)
                require(probabilities.shape == (916, 5) and np.isfinite(probabilities).all()
                        and np.allclose(probabilities.sum(1), 1, atol=2e-5, rtol=0), 'Malformed probe probabilities.')
                if not predictions:
                    import copy
                    gpu_net = copy.deepcopy(net).cuda(0).eval()
                    gpu = gpu_net(x[:32].cuda(0)).exp().cpu().numpy()
                    error = float(np.max(np.abs(gpu - probabilities[:32])))
                    require(error < 5e-4, 'Same-F16 CPU/GPU probe parity failed.')
                    parity['same_F16_CPU_GPU_probe_max_error'] = error
                    parity['same_vector_rows'] = 32
                    del gpu_net
                predictions[sample['prompt_key']] = probabilities
                collector.release()
                del hidden, x, features
                print(json.dumps(dict(stage='inferred', format=sample['prompt_key'], body_tokens=916)), flush=True)
        finally:
            for handle in handles:
                handle.remove()
    np.savez_compressed(args.out / 'probabilities.npz', **predictions)
    atomic_json(args.out / 'inference-complete.json', dict(stage='gardening_final_inference_complete',
        prepare_manifest_sha256=sha(args.out / 'prepare-manifest.json'), prepared_inputs_sha256=sha(args.out / 'prepared-inputs.json'),
        checkpoint_sha256=PROBE_SHA, selected_head_ids=HEADS, feature_point='o_proj forward_pre_hook input; 64 query heads x64, frozen rank order',
        feature_dtype='float16', standardized_dtype='float32', probabilities_sha256=sha(args.out / 'probabilities.npz'),
        weight_verification_sha256=sha(args.out / 'weight-verification.json'), backend=backend, validation=parity,
        local_kernel_source_sha256=kernel_files,
        package_versions={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'accelerate', 'kernels', 'triton', 'numpy', 'pandas']},
        runtime_environment={n: os.environ[n] for n in ['HF_HOME', 'HF_TOKEN_PATH', 'HF_HUB_CACHE', 'KERNELS_CACHE', 'TRITON_CACHE_DIR', 'HF_HUB_OFFLINE']},
        torch_cpu_threads=torch.get_num_threads(),
        no_features_persisted=True, source_messages_regenerated=False, probe_training_updates=0,
        system_threshold=THRESHOLD, seconds=time.time() - started))


def metrics(sample, probabilities, system_threshold=THRESHOLD):
    truth = np.asarray([ROLES.index(r['semantic_role']) for r in sample['content']])
    pred = probabilities.argmax(1)
    native = sample['prompt_key'] == 'proper_tags'
    result = dict(metric='native_role_accuracy' if native else 'semantic_alignment_to_original_function',
                  formal_original_function_accuracy=native, correct=int((truth == pred).sum()), tokens=len(truth),
                  fraction=float((truth == pred).mean()), confusion_matrix_truth_by_prediction=[[int(((truth == i) & (pred == j)).sum()) for j in range(5)] for i in range(5)],
                  per_role={r: dict(correct=int(((truth == i) & (pred == i)).sum()), support=int((truth == i).sum()),
                                    recall=float((pred[truth == i] == i).mean()) if (truth == i).any() else None) for i, r in enumerate(ROLES)})
    if native and system_threshold is not None:
        s, positive = truth == 0, probabilities[:, 0] >= system_threshold
        tp, fp, fn, tn = [int(x.sum()) for x in [s & positive, ~s & positive, s & ~positive, ~s & ~positive]]
        result['system_detection'] = dict(threshold=system_threshold, tp=tp, fp=fp, fn=fn, tn=tn,
            precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None)
    elif sample['prompt_key'] == 'everything_in_user_tags':
        result['formal_all_user_accuracy'] = float((pred == 1).mean())
        result['formal_all_user_correct'] = int((pred == 1).sum())
    return result


def old_predictions(samples):
    """Pair exact current rows with the already published single-message-only probe."""
    import plot_single_message_tomato as old
    require(sha(old.MODEL_PATH) == old.MODEL_SHA256, 'Old matched-source single-message probe changed.')
    archive = old.read_predictions(old.DEFAULT_PROJECTIONS)
    result = {}
    for sample in samples:
        key = sample['prompt_key']
        records = archive[key]
        paired_rows = [r for r in sample['content'] if r['in_old_915_comparison']]
        require(len(records) == len(paired_rows) == 915, 'Old/new paired content coverage differs.')
        for a, b in zip(records, paired_rows):
            require(a['token_in_prompt_ix'] == b['token_position'] and a['token'] == b['token']
                    and a['sample_ix'] == b['archive_sample_ix'] and a['original_function'] == b['semantic_role'], 'Old/new token identity differs.')
        result[key] = np.asarray([[r['probabilities'][role] for role in ROLES] for r in records], np.float32)
    return result, dict(checkpoint_path=str(old.MODEL_PATH), checkpoint_sha256=old.MODEL_SHA256,
                        projection_path=str(old.DEFAULT_PROJECTIONS), projections_sha256=sha(old.DEFAULT_PROJECTIONS),
                        description='Same archived gardening inputs; original single-message-only C4+Dolma3 selected linear probe, not the old-head MLP.')


def draw(samples, values, out, single):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    shown = samples[:1] if single else samples
    titles = {'proper_tags': 'Native delimiters: original roles',
              'everything_in_user_tags': 'All in user: colors = original functions',
              'basic_no_format': 'No delimiters: colors = original functions'}
    fig, axes = plt.subplots(5, len(shown), figsize=(9 if single else 18, 10), squeeze=False, sharey=True)
    for col, sample in enumerate(shown):
        key = sample['prompt_key']
        x = np.asarray([r['token_position'] for r in sample['content']])
        truth = np.asarray([r['semantic_role'] for r in sample['content']])
        boundaries = [sample['content'][i]['token_position'] for i in range(1, len(sample['content']))
                      if sample['content'][i]['message_ix'] != sample['content'][i - 1]['message_ix']]
        for row, role in enumerate(ROLES):
            ax = axes[row, col]
            for true_role, color in COLORS.items():
                mask = truth == true_role
                if mask.any():
                    ax.scatter(x[mask], values[key][mask, row], color=color, s=4, alpha=.75, rasterized=True)
            for boundary in boundaries:
                ax.axvline(boundary, color='#d1d5db', linewidth=.6)
            if row == 0:
                ax.axhline(THRESHOLD, color='#111827', linestyle='--', linewidth=.8)
                ax.set_title(titles[key], fontsize=10)
            ax.set_ylim(-.025, 1.025)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_yticks([0, .5, 1])
            if col == 0:
                ax.set_ylabel(['Systemness', 'Userness', 'CoTness', 'Assistantness', 'Toolness'][row])
            if row == 4:
                ax.set_xlabel('Token position in full input (all 916 body tokens shown)')
    handles = [Line2D([], [], marker='o', linestyle='', color=COLORS[r], label=r) for r in ROLES if SUPPORT[r]]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(.5, .965), frameon=False)
    fig.suptitle('Final checkpoint new — exact paper gardening example; complete system body', fontsize=12, y=.995)
    fig.text(.5, .012, 'Raw probabilities; no smoothing. Dashed line: frozen system threshold 0.520162. No tool response occurs in this conversation.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .93))
    stem = 'gardening-final-new-native' if single else 'gardening-final-new-three-formats'
    for suffix in ['png', 'pdf']:
        fig.savefig(out / (stem + '.' + suffix), dpi=180)
    plt.close(fig)


def report(args):
    _, samples = load_prepared(args)
    done = read(args.out / 'inference-complete.json')
    require(done['checkpoint_sha256'] == PROBE_SHA and done['selected_head_ids'] == HEADS
            and done['probabilities_sha256'] == sha(args.out / 'probabilities.npz')
            and done['prepared_inputs_sha256'] == sha(args.out / 'prepared-inputs.json')
            and done['prepare_manifest_sha256'] == sha(args.out / 'prepare-manifest.json')
            and done['weight_verification_sha256'] == sha(args.out / 'weight-verification.json'), 'Inference/probability provenance differs.')
    with np.load(args.out / 'probabilities.npz', allow_pickle=False) as z:
        require(set(z.files) == set(FORMATS), 'Missing format probabilities.')
        values = {k: z[k].copy() for k in FORMATS}
    require(all(v.shape == (916, 5) and np.isfinite(v).all() and (v >= 0).all()
                and (v <= 1.00001).all() and np.allclose(v.sum(1), 1, atol=2e-5, rtol=0) for v in values.values()), 'Invalid probability rows.')
    old, old_provenance = old_predictions(samples)
    paired_samples = [dict(s, content=[r for r in s['content'] if r['in_old_915_comparison']]) for s in samples]
    paired_values = {s['prompt_key']: values[s['prompt_key']][[r['in_old_915_comparison'] for r in s['content']]] for s in samples}
    summary = dict(stage='gardening_final_report', checkpoint_path=str(PROBE), checkpoint_sha256=PROBE_SHA,
                   complete_system_body=True, original_function_support=SUPPORT, no_tool_truth=True,
                   results={s['prompt_key']: metrics(s, values[s['prompt_key']]) for s in samples},
                   paired_915_final={s['prompt_key']: metrics(s, paired_values[s['prompt_key']]) for s in paired_samples},
                   single_message_only_old={s['prompt_key']: metrics(s, old[s['prompt_key']], system_threshold=None) for s in paired_samples},
                   old_archive_omitted_final_emoji_continuation_restored=True,
                   restored_emoji_continuations={s['prompt_key']: s['restored_emoji_continuation'] for s in samples},
                   old_probe_provenance=old_provenance,
                   comparison='Paired exact same prompt strings, token IDs, positions and original functions; compare two frozen readouts trained with different datasets/features/objectives.',
                   limitation='One historically used gardening example, not a fresh independent test or evidence of behavioral instruction following. Altered formats use original-function alignment only; all-user formal accuracy is reported separately. Tool recall is undefined.',
                   input_sha256=INPUT_SHA, inference_complete_sha256=sha(args.out / 'inference-complete.json'))
    plots = args.out / 'plots'
    plots.mkdir(exist_ok=True)
    draw(samples, values, plots, single=True)
    draw(samples, values, plots, single=False)
    summary['plots_sha256'] = {p.name: sha(p) for p in sorted(plots.glob('gardening-final-new-*')) if p.suffix in ['.png', '.pdf']}
    atomic_json(args.out / 'summary.json', summary)
    with gzip.open(args.out / 'predictions.csv.gz', 'wt', encoding='utf-8', newline='') as handle:
        fields = ['prompt_key', 'token_position', 'token_id', 'token', 'message_ix', 'semantic_role', 'formal_role', 'archive_sample_ix', 'in_old_915_comparison'] + ['p_' + r for r in ROLES] + ['prediction']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            for row, probability in zip(sample['content'], values[sample['prompt_key']]):
                writer.writerow(dict(row, prompt_key=sample['prompt_key'], **{'p_' + r: float(p) for r, p in zip(ROLES, probability)}, prediction=ROLES[int(probability.argmax())]))
    print(json.dumps(dict(stage='reported', native=summary['results']['proper_tags'], plots=str(plots))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'infer', 'report'], required=True)
    parser.add_argument('--model-path', type=clean, default=Path('/tmp/chatinject-role-probe-model'))
    parser.add_argument('--out', type=clean, default=DEFAULT_OUT)
    parser.add_argument('--verified-weights', type=clean, help='Reuse an independently rehashed full-shard certificate only while every shard size, inode and nanosecond mtime is unchanged.')
    parser.add_argument('--kernel-cache', type=clean, default=Path('/tmp/chatinject-role-probe-kernels-cache'))
    args = parser.parse_args()
    # Hub authentication can read HF_HOME/token even in offline mode. Override
    # every inherited cache before importing transformers/kernels so no path
    # silently falls back to the unavailable /workspace FUSE mount.
    os.environ.update(HF_HOME='/tmp/role-analysis-gardening-hf-cache',
                      HF_TOKEN_PATH='/tmp/role-analysis-gardening-hf-cache/token',
                      HF_HUB_CACHE=str(args.kernel_cache), KERNELS_CACHE=str(args.kernel_cache),
                      TRITON_CACHE_DIR='/tmp/role-analysis-gardening-triton-cache',
                      XDG_CACHE_HOME='/tmp/role-analysis-gardening-cache', HF_HUB_OFFLINE='1')
    os.environ.pop('HF_KERNELS_CACHE', None)
    {'prepare': prepare, 'infer': infer, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
