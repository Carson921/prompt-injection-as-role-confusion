"""Resident, isolated native GPU diagnostics; never produces benchmark scores."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import types


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def difference(a, b):
    import numpy as np
    d = a.astype('float64') - b.astype('float64')
    return {'max_abs': float(np.abs(d).max()), 'rms': float(np.sqrt(np.mean(d*d))),
            'equal': bool(np.array_equal(a, b)), 'shape': list(a.shape)}


def run_v2_preflight(adapter, output, *, run_root):
    import importlib
    import shutil
    module = importlib.import_module('deepseek_agentdojo_preflight')
    module = importlib.reload(module)
    shutil.copy2(module.__file__, Path(output) / 'invoked-preflight-source.py')
    return module.run_preflight(adapter, Path(run_root))


def run_trace(adapter, output, lengths=(8, 257, 1029)):
    """Separate same-shape cache behavior from one-token numerical behavior."""
    import numpy as np
    import torch
    from transformers import DynamicCache
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = adapter.model
    device = model.model.embed_tokens.weight.device
    words = adapter.tokenizer.encode('The quick brown fox checks the complete cache prefix. ' * 600,
                                     add_special_tokens=False)
    results = []
    for length in lengths:
        ids = torch.tensor([[0, 128803] + words[:length-4] + [128804, 128821]], device=device)
        values = {}
        tensors = {}
        def invoke(name, tokens, cache=None, use_cache=False, start=0):
            adapter.prepare_query_length(tokens.shape[1])
            records, handles = {}, []
            def last(value):
                if isinstance(value, (tuple, list)):
                    value = next(x for x in value if torch.is_tensor(x))
                return value[:, -1].detach().float().cpu().numpy()
            for index, layer in enumerate(model.model.layers):
                for label, module in [('input_norm', layer.input_layernorm),
                                      ('attention_heads', layer.self_attn.o_a_proj),
                                      ('attention', layer.self_attn),
                                      ('mlp_norm', layer.post_attention_layernorm),
                                      ('mlp', layer.mlp), ('layer', layer)]:
                    key = f'{index:02d}/{label}'
                    if label == 'attention_heads':
                        def before(module, args, key=key):
                            records[key] = last(args[0])
                        handles.append(module.register_forward_pre_hook(before))
                    else:
                        def after(module, args, result, key=key):
                            records[key] = last(result)
                            if key == '00/mlp_norm' and name == 'full_no_cache':
                                tensors['mlp0_input'] = result.detach().clone()
                        handles.append(module.register_forward_hook(after))
            try:
                with torch.inference_mode():
                    result = model(input_ids=tokens, attention_mask=torch.ones((1, start+tokens.shape[1]),
                        dtype=torch.long, device=device), past_key_values=cache, use_cache=use_cache,
                        position_ids=torch.arange(start, start+tokens.shape[1], device=device)[None],
                        logits_to_keep=1, return_dict=True)
                records['logits'] = result.logits.detach().float().cpu().numpy()
                values[name] = records
                np.savez(output / f'length{length}-{name}.npz', **records)
                return result
            finally:
                for handle in handles:
                    handle.remove()
        full = invoke('full_no_cache', ids)
        del full
        repeat = invoke('full_no_cache_repeat', ids)
        del repeat
        cache_full = invoke('full_cache', ids, DynamicCache(config=model.config), True)
        del cache_full
        adapter.prepare_query_length(length-1)
        with torch.inference_mode():
            prefix = model(input_ids=ids[:, :-1], attention_mask=torch.ones((1, length-1), dtype=torch.long, device=device),
                           past_key_values=DynamicCache(config=model.config), use_cache=True,
                           logits_to_keep=1, return_dict=True)
        cache = prefix.past_key_values
        del prefix
        cached = invoke('cached_last', ids[:, -1:], cache, True, length-1)
        del cached, cache
        comparisons = {name: {key: difference(values['full_no_cache'][key], val)
                             for key, val in records.items()}
                       for name, records in values.items() if name != 'full_no_cache'}
        mlp_input = tensors.pop('mlp0_input')
        with torch.inference_mode():
            adapter.prepare_query_length(length)
            many = model.model.layers[0].mlp(mlp_input, input_ids=ids)[:, -1].float().cpu().numpy()
            adapter.prepare_query_length(1)
            single = model.model.layers[0].mlp(mlp_input[:, -1:], input_ids=ids[:, -1:])[:, -1].float().cpu().numpy()
        del mlp_input
        row = {'length': length, 'comparisons': comparisons,
               'standalone_mlp0_identical_input_last_row': difference(many, single),
               'argmax': {name: int(v['logits'].argmax(-1)[0, 0]) for name, v in values.items()}}
        results.append(row)
        write(output / 'results.json', {'status': 'diagnostic_only', 'results': results})
        print(json.dumps({'diagnostic_length_complete': length, 'logits': {name: v['logits'] for name, v in comparisons.items()},
                          'standalone_mlp0': row['standalone_mlp0_identical_input_last_row']}), flush=True)
    return {'status': 'diagnostic_only', 'results_path': str(output / 'results.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731'))
    parser.add_argument('--probe', type=Path, default=Path(__file__).parent / 'outputs/deepseek-v4-flash-0731-role-probe-20261115')
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    jobs = root / 'jobs'; jobs.mkdir(exist_ok=True)
    stopping = False
    def stop(*unused):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    from deepseek_agentdojo_model import DeepSeekAgentDojoModel
    adapter = DeepSeekAgentDojoModel(args.model, args.probe, root)
    write(root / 'status.json', {'stage': 'ready', 'diagnostic_only': True})
    try:
        while not stopping and not (root / 'STOP').exists():
            for job in sorted(jobs.glob('*.json')):
                destination = root / 'results' / job.stem
                if (destination / 'receipt.json').exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                spec = json.loads(job.read_text())
                source_path = Path(spec.get('module', __file__)).resolve()
                source = source_path.read_bytes()
                (destination / 'executed-source.py').write_bytes(source)
                module = types.ModuleType('native_diagnostic_job_' + job.stem)
                module.__file__ = str(source_path)
                receipt = {'job': spec, 'source_sha256': hashlib.sha256(source).hexdigest(), 'started_unix': time.time()}
                try:
                    exec(compile(source, str(source_path), 'exec'), module.__dict__)
                    receipt['result'] = getattr(module, spec.get('function', 'run_trace'))(adapter, destination, **spec.get('kwargs', {}))
                    receipt['status'] = 'diagnostic_finished'
                except Exception as error:
                    receipt.update(status='diagnostic_failed', error=str(error), traceback=traceback.format_exc())
                    print(receipt['traceback'], flush=True)
                receipt['finished_unix'] = time.time()
                write(destination / 'receipt.json', receipt)
            time.sleep(.5)
    finally:
        adapter.close()
        write(root / 'status.json', {'stage': 'closed', 'diagnostic_only': True})


if __name__ == '__main__':
    main()
