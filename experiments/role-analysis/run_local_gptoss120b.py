"""Train the role-analysis probes with local MXFP4 GPT-OSS-120B on four A40s.

Reuses the 20B experiment's exact cached texts/prompts, including system messages.
The language model is frozen; logistic-regression probes are the trained models.

From the repository root (Python 3.11):
  uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r experiments/role-analysis/requirements-local-4xa40.txt
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u experiments/role-analysis/run_local_gptoss120b.py

Activation extraction and individual probes resume automatically. The activation
cache requires approximately 36 GiB and can be removed after complete.json exists.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'outputs/local-gptoss120b'
MODEL = Path('/workspace/gpt-oss-120b')
KERNEL_REVISION = '9fbc1b48e710ac8abf18ea2650ee690d52dc3859'
LAYERS = list(range(0, 36, 4))
COMBINATIONS = [
    ('user', 'assistant'), ('user', 'assistant', 'tool'),
    ('user', 'cot', 'assistant'), ('user', 'cot', 'assistant', 'tool'),
    ('system', 'user', 'assistant'), ('system', 'user', 'assistant', 'tool'),
    ('system', 'user', 'cot', 'assistant'),
    ('system', 'user', 'cot', 'assistant', 'tool'),
]
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HOME', '/root/.cache/huggingface-role-probe')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def status(stage, **fields):
    value = {'stage': stage, 'unix_time': time.time(), **fields}
    atomic_json(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        Path(os.environ.get('ROLE_ANALYSIS_LOAD_PATH', str(MODEL))), local_files_only=True, add_eos_token=False,
        add_bos_token=False, padding_side='left',
    )


def prepare():
    import numpy as np
    import pandas as pd
    import yaml
    from utils.dataset import ReconstructableTextDataset
    from utils.role_assignments import label_content_roles
    from utils.role_templates import render_single_message

    OUT.mkdir(parents=True, exist_ok=True)
    source = HERE / 'outputs/local-gptoss20b'
    config = yaml.safe_load((HERE / 'config/probe.yaml').read_text())['gptoss-120b']
    assert config['seq_len'] == 1024 and config['train_prefixes'] == ['']
    assert not config['nested_reasoning']
    raw = json.loads((source / 'raw_training_data.json').read_text())
    assert len(raw) == 249
    tok = tokenizer()
    texts = tok.batch_decode(tok([r['text'] for r in raw], add_special_tokens=False,
                                truncation=True, max_length=config['seq_len']).input_ids)
    # Preserve exactly the 20B row order and therefore its prompt-level split.
    inputs = pd.read_parquet(source / 'training_inputs.parquet')
    assert len(inputs) == len(texts) * 5
    assert inputs.prompt_ix.tolist() == list(range(len(inputs)))
    for row in inputs.itertuples():
        assert row.question == texts[row.question_ix]
        assert row.prompt == render_single_message('gptoss-120b', row.role, row.question)
    assert inputs.groupby('role').size().to_dict() == {r: 249 for r in COMBINATIONS[-1]}
    lengths = tok(inputs.prompt.tolist(), add_special_tokens=False).input_ids
    max_length = max(map(len, lengths))
    ds = ReconstructableTextDataset(inputs.prompt.tolist(), tok, max_length=max_length,
                                   prompt_ix=inputs.prompt_ix.tolist())
    records = []
    for i in range(len(ds)):
        for j in ds.attention_mask[i].nonzero().flatten().tolist():
            records.append((i, j, int(ds.input_ids[i, j]), ds.original_tokens[i][j]))
    tokens = pd.DataFrame(records, columns=['prompt_ix', 'token_ix', 'token_id', 'token'])
    tokens['sample_ix'] = np.arange(len(tokens))
    labeled = label_content_roles('gptoss-120b', tokens)
    assert labeled.sample_ix.tolist() == tokens.sample_ix.tolist()
    labels = labeled.merge(inputs[['prompt_ix', 'role']].rename(columns={'role': 'target_role'}),
                           on='prompt_ix', validate='many_to_one')
    labels = labels[labels.is_content & labels.role.notna() & (labels.role == labels.target_role)].copy()
    counts = labels.groupby('role').size().to_dict()
    assert set(counts) == set(COMBINATIONS[-1]) and min(counts.values()) > 0
    assert len(set(counts.values())) == 1, counts
    sys_prompts = inputs[inputs.role == 'system'].prompt
    assert sys_prompts.str.startswith('<|start|>system<|message|>').all()
    load_path = Path(os.environ.get('ROLE_ANALYSIS_LOAD_PATH', str(MODEL)))
    model_config_hash = hashlib.sha256((load_path / 'config.json').read_bytes()).hexdigest()
    if load_path != MODEL:
        verification = json.loads((OUT / 'checkpoint-cache-verification.json').read_text())
        assert verification['local_model_config_sha256'] == model_config_hash
    signature_data = {'config': config, 'layers': LAYERS, 'kernel_revision': KERNEL_REVISION,
                      'gpu_transfer': 'synchronous CPU staging between distinct CUDA devices',
                      'prompts_sha256': hashlib.sha256('\n'.join(inputs.prompt).encode()).hexdigest(),
                      'model_config_sha256': model_config_hash}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    if (OUT / 'manifest.json').exists():
        assert json.loads((OUT / 'manifest.json').read_text())['signature'] == signature
    inputs.to_parquet(OUT / 'training_inputs.parquet', index=False)
    tokens.to_parquet(OUT / 'training_tokens.parquet', index=False)
    labels.to_parquet(OUT / 'training_labels.parquet', index=False)
    atomic_json(OUT / 'manifest.json', {
        **signature_data, 'signature': signature, 'model_path': str(MODEL),
        'weight_format': 'MXFP4', 'activation_dtype': 'bfloat16',
        'stored_activation_dtype': 'float16', 'capture_point': 'post_attention_layernorm output',
        'raw_texts': len(raw), 'source_counts': dict(Counter(r['source'] for r in raw)),
        'prompts': len(inputs), 'prompts_by_role': inputs.groupby('role').size().to_dict(),
        'content_tokens_by_role': counts, 'tokens': len(tokens), 'max_length': max_length,
        'split': 'original cuML prompt_ix split, 90/10, seed=123; base texts may cross split',
        'role_combinations': COMBINATIONS, 'n_probes': len(LAYERS) * len(COMBINATIONS),
        'frozen_language_model': True,
    })
    status('data_ready', prompts=len(inputs), system_prompts=len(sys_prompts), content_tokens_by_role=counts)
    return tok, ds


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    assert torch.cuda.device_count() == 4, 'All four GPUs must be visible'
    original_get = hub_kernels.get_kernel
    def pinned(name, *args, **kwargs):
        if name == 'kernels-community/gpt-oss-triton-kernels':
            kwargs['revision'] = KERNEL_REVISION
        return original_get(name, *args, **kwargs)
    hub_kernels.get_kernel = pinned
    original_validate = Mxfp4HfQuantizer.validate_environment
    def strict(self, *args, **kwargs):
        original_validate(self, *args, **kwargs)
        if self.quantization_config.dequantize:
            raise RuntimeError('MXFP4 is required: refusing automatic BF16 dequantization')
    Mxfp4HfQuantizer.validate_environment = strict
    device_map = {'model.embed_tokens': 0, 'model.rotary_emb': 0,
                  'model.norm': 3, 'lm_head': 3}
    device_map.update({f'model.layers.{i}': i // 9 for i in range(36)})
    status('loading_model', device_map=device_map)
    load_path = Path(os.environ.get('ROLE_ANALYSIS_LOAD_PATH', str(MODEL)))
    if load_path != MODEL:
        # Require verified official weights and the original local model config.
        verification = json.loads((OUT / 'checkpoint-cache-verification.json').read_text())
        assert verification['published_weights_sha256_verified'] and verification['cache_path'] == str(load_path)
        assert (load_path / 'verified-weights.json').exists()
    model = AutoModelForCausalLM.from_pretrained(
        load_path, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=False), device_map=device_map,
        attn_implementation='eager', use_kernels=False,
    ).eval().requires_grad_(False)
    assert model.config.num_hidden_layers == 36
    backends = {type(layer.mlp.experts).__name__ for layer in model.model.layers}
    assert backends == {'Mxfp4GptOssExperts'}, backends
    assert not model.hf_quantizer.quantization_config.dequantize
    # Avoid unreliable peer copies on this A40/PCIe host. Values and dtypes are
    # preserved by blocking GPU -> CPU -> GPU copies at device boundaries.
    import accelerate.hooks
    import accelerate.utils.operations as operations
    original_send = operations.send_to_device
    def staged_send(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
            target = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            if target.type == 'cuda' and tensor.device != target:
                return tensor.cpu().to(target, non_blocking=False)
        return original_send(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)
    operations.send_to_device = staged_send
    accelerate.hooks.send_to_device = staged_send
    atomic_json(OUT / 'model_backend.json', {
        'expert_backends': list(backends), 'device_map': model.hf_device_map,
        'load_path': str(load_path), 'source_model_path': str(MODEL),
        'gpu_transfer': 'synchronous CPU staging between distinct CUDA devices',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(4)],
        'compute_capabilities': [torch.cuda.get_device_capability(i) for i in range(4)],
    })
    return model


def extract(batch_size, pipeline=True):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from utils.dataset import stack_collate
    tok, ds = prepare()
    meta = json.loads((OUT / 'manifest.json').read_text())
    n, width = meta['tokens'], 2880
    cache = OUT / 'activations'
    cache.mkdir(exist_ok=True)
    checkpoint = cache / 'progress.json'
    progress = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'cursor': 0, 'next_batch': 0}
    if progress.get('complete'):
        status('activations_cached', tokens=n)
        return
    if progress['cursor']:
        assert progress['batch_size'] == batch_size, 'Resuming requires the same batch size'
    maps = {i: np.memmap(cache / f'layer-{i}.f16', mode='r+' if progress['cursor'] else 'w+',
                         dtype=np.float16, shape=(n, width)) for i in LAYERS}
    model = load_model()
    current = {}
    def capture(index):
        def hook(module, args, output):
            current[index] = output.detach().to(device='cpu', dtype=torch.float16).reshape(-1, width)
        return hook
    with torch.inference_mode():
        sample = tok('A short forward validation.', return_tensors='pt').to('cuda:0')
        reference = model(**sample, use_cache=False).logits.float().cpu()
        assert torch.isfinite(reference).all() and float(reference.std()) > .01
        handles = [model.model.layers[i].post_attention_layernorm.register_forward_hook(capture(i)) for i in LAYERS]
        captured = model(**sample, use_cache=False).logits.float().cpu()
        difference = (reference.float() - captured.float()).abs().max().item()
        assert torch.equal(reference, captured), difference
        assert set(current) == set(LAYERS)
        atomic_json(OUT / 'forward_validation.json', {'exact_logits': True, 'max_absolute_logit_difference': difference,
                    'logit_std':float(reference.std()), 'logit_min':float(reference.min()), 'logit_max':float(reference.max()),
                    'captured_layers': LAYERS, 'all_finite': all(torch.isfinite(v).all().item() for v in current.values())})
        assert all(torch.isfinite(v).all() for v in current.values())
        del reference, captured
        validation_activations = dict(current)
        for handle in handles:
            handle.remove()
        handles = []
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=stack_collate)
        if pipeline:
            from pipeline_runtime import pipeline_batches
            batches = pipeline_batches(model, dl, LAYERS, progress['next_batch'],
                        validation=(sample, validation_activations),
                        save_validation=lambda value:atomic_json(OUT / 'pipeline_validation.json', value))
        else:
            handles = [model.model.layers[i].post_attention_layernorm.register_forward_hook(capture(i)) for i in LAYERS]
            def serial_batches():
                for bi, batch in enumerate(dl):
                    if bi < progress['next_batch']:
                        continue
                    current.clear()
                    result = model.model(input_ids=batch['input_ids'].to('cuda:0'),
                                         attention_mask=batch['attention_mask'].to('cuda:0'),
                                         use_cache=False, return_dict=True)
                    del result
                    yield bi, batch, dict(current)
            batches = serial_batches()
        started = time.time()
        cursor = progress['cursor']
        for bi, batch, captured_states in batches:
            mask = batch['attention_mask'].reshape(-1).bool()
            count = int(mask.sum())
            assert set(captured_states) == set(LAYERS)
            for layer in LAYERS:
                values = captured_states[layer][mask].numpy()
                assert np.isfinite(values).all(), (bi, layer)
                maps[layer][cursor:cursor+count] = values
            cursor += count
            captured_states.clear()
            if (bi + 1) % 4 == 0 or bi + 1 == len(dl):
                for mm in maps.values():
                    mm.flush()
                atomic_json(checkpoint, {'cursor': cursor, 'next_batch': bi + 1, 'batch_size': batch_size,
                                        'complete': bi + 1 == len(dl)})
                status('extracting', batch=bi+1, total_batches=len(dl), tokens=cursor,
                       seconds=time.time()-started,
                       allocated_gib=[round(torch.cuda.memory_allocated(i)/2**30, 3) for i in range(4)])
        assert cursor == n
    for handle in handles:
        handle.remove()
    atomic_json(OUT / 'extraction_stats.json', {
        'seconds': time.time()-started, 'tokens': n, 'layers': LAYERS,
        'peak_allocated_gib': [torch.cuda.max_memory_allocated(i)/2**30 for i in range(4)],
        'peak_reserved_gib': [torch.cuda.max_memory_reserved(i)/2**30 for i in range(4)],
    })
    del model, maps
    gc.collect()
    for i in range(4):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    status('extraction_complete', tokens=n)


def fit_worker(gpu):
    import cupy as cp
    import cuml
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.pipeline import Pipeline
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    meta = json.loads((OUT / 'manifest.json').read_text())
    progress = json.loads((OUT / 'activations/progress.json').read_text())
    assert progress['complete'] and progress['cursor'] == meta['tokens']
    labels = pd.read_parquet(OUT / 'training_labels.parquet')
    folder = OUT / 'probe-checkpoints'
    folder.mkdir(exist_ok=True)
    cp.get_default_memory_pool().set_limit(size=40*2**30)
    for layer in LAYERS[gpu::4]:
        features = torch.from_file(str(OUT / f'activations/layer-{layer}.f16'), shared=False,
                                   size=meta['tokens']*2880, dtype=torch.float16).view(-1, 2880)
        for roles in COMBINATIONS:
            name = ''.join(r[0] for r in roles)
            path = folder / f'{name}-{layer}.pkl'
            if path.exists():
                continue
            started = time.time()
            selected = labels[labels.role.isin(roles)].copy()
            train_ids, test_ids = cuml.train_test_split(selected.prompt_ix.unique(), test_size=.1, random_state=123)
            train = selected[selected.prompt_ix.isin(train_ids)]
            test = selected[selected.prompt_ix.isin(test_ids)]
            roles_map = {r: i for i, r in enumerate(roles)}
            assert set(train.role) == set(roles) and set(test.role) == set(roles)
            assert not set(train.prompt_ix) & set(test.prompt_ix)
            # Match the original notebook's fast Torch FP16 -> FP32 conversion.
            # DLPack shares the resulting GPU storage without another copy.
            x_train = cp.from_dlpack(features[train.sample_ix.tolist()].float().to('cuda:0'))
            x_test = cp.from_dlpack(features[test.sample_ix.tolist()].float().to('cuda:0'))
            y_train = cp.asarray(train.role.map(roles_map).to_numpy(), blocking=True)
            y_test = cp.asarray(test.role.map(roles_map).to_numpy(), blocking=True)
            steps = []
            if meta['config']['train_params']['add_scaling']:
                steps.append(('scaler', StandardScaler()))
            steps.append(('clf', LogisticRegression(penalty='l2', max_iter=5000,
                          linesearch_max_iter=100, fit_intercept=True, C=meta['config']['train_params']['C'])))
            probe = Pipeline(steps)
            probe.fit(x_train, y_train)
            pred = cp.asnumpy(probe.predict(x_test))
            proba = cp.asnumpy(probe.predict_proba(x_test))
            assert np.isfinite(proba).all()
            results = test.assign(pred=pd.Series(pred, index=test.index).map({v:k for k,v in roles_map.items()}))
            results['is_acc'] = results.role == results.pred
            acc_by_role = results.groupby(['role', 'pred'], as_index=False).agg(count=('sample_ix', 'count'))
            acc_by_pos = results.groupby('token_in_seg_ix', as_index=False).agg(count=('sample_ix', 'count'), acc=('is_acc', 'mean'))
            result = {'probe': probe, 'acc': float(results.is_acc.mean()),
                      'nll': float(-np.log(np.clip(proba[np.arange(len(test)), cp.asnumpy(y_test)], 1e-15, 1)).mean()),
                      'acc_by_role': acc_by_role, 'acc_by_pos': acc_by_pos,
                      'layer_ix': layer, 'role_space': list(roles), 'roles_map': roles_map,
                      'n_inputs': len(selected), 'n_train_tokens': len(train), 'n_test_tokens': len(test),
                      'train_prompt_ids': list(map(int, train_ids)), 'test_prompt_ids': list(map(int, test_ids)),
                      'seconds': time.time()-started,
                      'iterations': cp.asnumpy(cp.asarray(probe.named_steps['clf'].n_iter_)).tolist()}
            with path.with_suffix('.tmp').open('wb') as f:
                pickle.dump(result, f)
            path.with_suffix('.tmp').replace(path)
            print(f'GPU {gpu} {name} layer {layer}: acc={result["acc"]:.6f}, nll={result["nll"]:.6f}, seconds={result["seconds"]:.1f}', flush=True)
            del x_train, x_test, y_train, y_test, probe, result, pred, proba
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()


def fit():
    workers = []
    status('fitting_probes', workers=4, expected_probes=72)
    for gpu in range(4):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
        log = (OUT / f'fit-gpu{gpu}.log').open('a')
        proc = subprocess.Popen([sys.executable, '-u', __file__, '--stage', 'fit-worker', '--gpu', str(gpu)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        workers.append((gpu, proc, log))
    while any(p.poll() is None for _, p, _ in workers):
        failures = [(g, p.returncode) for g, p, _ in workers if p.returncode not in (None, 0)]
        if failures:
            for _, p, _ in workers:
                if p.poll() is None:
                    p.terminate()
            raise RuntimeError(f'Probe worker failed: {failures}; see fit-gpu*.log')
        time.sleep(5)
    for gpu, proc, log in workers:
        log.close()
        assert proc.returncode == 0, (gpu, proc.returncode)
    report()


def report():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    probes = []
    for roles in COMBINATIONS:
        for layer in LAYERS:
            with (OUT / 'probe-checkpoints' / f'{"".join(r[0] for r in roles)}-{layer}.pkl').open('rb') as f:
                probes.append(pickle.load(f))
    assert len(probes) == 72
    canonical = HERE / 'outputs/probes/gptoss-120b.pkl'
    with canonical.with_suffix('.tmp').open('wb') as f:
        pickle.dump(probes, f)
    canonical.with_suffix('.tmp').replace(canonical)
    metrics = pd.DataFrame([{k:v for k,v in p.items() if k in
        ['layer_ix','role_space','acc','nll','n_inputs','n_train_tokens','n_test_tokens','seconds','iterations']} for p in probes])
    metrics['role_space'] = metrics.role_space.map(lambda r: ''.join(x[0] for x in r))
    assert np.isfinite(metrics[['acc','nll']]).all().all()
    metrics.to_csv(OUT / 'training_metrics.csv', index=False)
    for key in ('acc_by_role', 'acc_by_pos'):
        table = pd.concat([p[key].assign(model='gptoss-120b', layer_ix=p['layer_ix'],
                           role_space=','.join(p['role_space'])) for p in probes], ignore_index=True)
        table.to_csv(HERE / f'outputs/probe-training/{key}_gptoss-120b.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in metrics.groupby('role_space'):
        ax.plot(group.layer_ix, group.acc, marker='o', label=name)
    ax.set(xlabel='Layer (zero-based)', ylabel='Held-out prompt token accuracy',
           title='GPT-OSS-120B MXFP4 role probes, including system')
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'training-accuracy-by-layer.png', dpi=160)
    plt.close(fig)
    system_results = []
    for p in probes:
        rows = p['acc_by_role']
        rows = rows[rows.role == 'system']
        if len(rows):
            system_results.append({'layer_ix':p['layer_ix'], 'role_space':','.join(p['role_space']),
                                   'accuracy':float(rows.loc[rows.pred == 'system', 'count'].sum()/rows['count'].sum()),
                                   'test_tokens':int(rows['count'].sum())})
    pd.DataFrame(system_results).to_csv(OUT / 'system-role-accuracy.csv', index=False)
    summary = {'n_probes':72, 'layers':LAYERS, 'accuracy_min':float(metrics.acc.min()),
               'accuracy_max':float(metrics.acc.max()), 'accuracy_mean':float(metrics.acc.mean()),
               'system_accuracy_min':min(r['accuracy'] for r in system_results),
               'system_accuracy_max':max(r['accuracy'] for r in system_results),
               'probe_file':str(canonical), 'unix_time':time.time()}
    atomic_json(OUT / 'complete.json', summary)
    status('complete', **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','prepare','extract','fit','fit-worker','report'], default='all')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sequential', action='store_true', help='Disable overlapping GPU pipeline stages')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare()
    if args.stage in ('all','extract'):
        extract(args.batch_size, pipeline=not args.sequential)
    if args.stage in ('all','fit'):
        fit()
    if args.stage == 'fit-worker':
        fit_worker(args.gpu)
    if args.stage == 'report':
        report()


def monitor_gpus(stop):
    import pynvml
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
    peaks = [0] * 4
    started = time.time()
    try:
        with (OUT / 'gpu-memory.csv').open('a', buffering=1) as f:
            if f.tell() == 0:
                f.write('unix_time,gpu,used_bytes,utilization_percent\n')
            while not stop.is_set():
                for i, handle in enumerate(handles):
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    peaks[i] = max(peaks[i], memory.used)
                    f.write(f'{time.time():.3f},{i},{memory.used},{utilization.gpu}\n')
                atomic_json(OUT / 'gpu-memory-summary.json', {
                    'peak_used_gib': [p/2**30 for p in peaks],
                    'elapsed_seconds': time.time()-started, 'sample_interval_seconds':2,
                })
                stop.wait(2)
    finally:
        pynvml.nvmlShutdown()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = None
    if 'fit-worker' not in sys.argv:
        monitor = threading.Thread(target=monitor_gpus, args=(stop,), daemon=True)
        monitor.start()
    try:
        main()
    except Exception as error:
        if '--stage' not in sys.argv or 'fit-worker' not in sys.argv:
            status('failed', error=repr(error))
        raise
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=5)
