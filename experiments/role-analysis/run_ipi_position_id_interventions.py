"""Position-id interventions on the sealed IPI Arena no-payload traces (GPT-OSS-120B).

Reuses the frozen inputs of outputs/ipi-benign-trace-interventions-20261117 (41 traces,
first body of each role, farthest-move rows) and only varies the RoPE position ids
that the backbone receives. Token order decides causal / sliding-window visibility,
position ids decide the rotary phase, so the two can be changed independently:

  baseline_arange  original order, ids 0..N-1 passed explicitly (must equal sealed baseline bit for bit)
  global+1000      original order, every id +1000 (RoPE is relative: numerical control)
  posid_move       original order, every token gets the id it has after the sealed farthest move
  order_move       sealed farthest-move order, every token keeps its original id
  block+D          original order, ids of the full role block (header+body+ending) shifted by D
  gap+1000         original order, ids of the role block and of everything after it +1000

No text is generated, no tool is executed, nothing in the sealed delivery is modified.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

import run_role_signal_factorial as shared

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROLES = shared.ROLES
SOURCE = HERE / 'outputs/ipi-benign-trace-interventions-20261117'
DEFAULT_OUTPUT = HERE / 'outputs/ipi-benign-position-id-interventions-20261125'
BLOCK_SHIFTS = (100, 1000, 10000, -1000)
GAP_SHIFT = GLOBAL_SHIFT = 1000
SEED = 20261125


def signed(value):
    return f'{value:+d}'


def build_variants(source=SOURCE):
    """Derive every variant from the sealed baseline and farthest-move rows."""
    protocol = json.loads((source / 'all/protocol.json').read_text())
    assert shared.digest(source / 'all/inputs.jsonl') == protocol['inputs_sha256']
    baselines, moves = {}, {}
    with (source / 'all/inputs.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['operation'] == 'baseline':
                baselines[row['trace_id']] = row
            elif row['operation'] == 'move':
                moves[row['trace_id'], row['source_role']] = row
    assert len(baselines) == 41 and len(moves) == 239
    variants = []
    for trace_id, base in baselines.items():
        ids, n = base['input_ids'], len(base['input_ids'])
        arange = list(range(n))
        union = base['target_positions']
        masks = base['role_target_positions']

        def add(condition, role, input_ids, position_ids, positions, role_positions, **extra):
            assert len(input_ids) == len(position_ids) == n
            name = condition if role is None else f'{condition}-{role}'
            variants.append(dict(input_id=base['input_id'].rsplit('--', 1)[0] + '--' + name, trace_id=trace_id,
                condition=condition, source_role=role, input_ids=input_ids, position_ids=position_ids,
                target_positions=positions, role_target_positions=role_positions, input_tokens=n,
                **dict(target_original_positions=union) | extra))

        add('baseline_arange', None, ids, arange, union, masks)
        add(f'global{signed(GLOBAL_SHIFT)}', None, ids, [p + GLOBAL_SHIFT for p in arange], union, masks)
        for role in masks:
            move = moves[trace_id, role]
            origin = move['token_source_positions']          # new index -> original index
            assert sorted(origin) == arange and [ids[o] for o in origin] == move['input_ids']
            new_index = [0] * n                               # original index -> new index
            for new, old in enumerate(origin):
                new_index[old] = new
            start = move['original_body_start'] - len(move['applied_header_ids'])
            end = move['original_body_end'] + len(move['applied_ending_ids'])
            insertion = move['move_insertion_position']
            assert end - start == move['moved_block_tokens']
            assert origin[insertion:insertion + end - start] == list(range(start, end))
            assert masks[role] == list(range(move['original_body_start'], move['original_body_end']))
            geometry = dict(block_start=start, block_end=end, original_body_start=move['original_body_start'],
                moved_body_start=move['new_body_start'], move_destination=move['move_destination'],
                move_displacement_tokens=move['move_displacement_tokens'])
            add('posid_move', role, ids, new_index, union, masks, **geometry)
            moved_union = sorted(new_index[p] for p in union)
            moved_masks = {r: [new_index[p] for p in values] for r, values in masks.items()}
            assert moved_masks[role] == move['target_positions']
            add('order_move', role, move['input_ids'], origin, moved_union, moved_masks,
                **geometry | dict(target_original_positions=[origin[p] for p in moved_union]))
            for shift in BLOCK_SHIFTS:
                add(f'block{signed(shift)}', role, ids,
                    [p + shift if start <= p < end else p for p in arange], union, masks, **geometry)
            add(f'gap{signed(GAP_SHIFT)}', role, ids,
                [p + GAP_SHIFT if p >= start else p for p in arange], union, masks, **geometry)
    assert len({v['input_id'] for v in variants}) == len(variants) == 82 + 239 * (3 + len(BLOCK_SHIFTS))
    return variants, protocol


def run(args):
    import torch
    out = args.output
    for folder in ('readouts', 'shards'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    variants, source_protocol = build_variants()
    order = sorted({v['trace_id']: v['input_tokens'] for v in variants}.items(), key=lambda item: (-item[1], item[0]))
    mine = {trace for index, (trace, _) in enumerate(order) if index % args.shards == args.shard}
    if args.traces:
        mine &= set(args.traces)
    variants = [v for v in variants if v['trace_id'] in mine]
    assert shared.digest(args.probe) == shared.SHA == source_protocol['probe_sha256']
    scorer = shared.load_module('_trace_interventions_scorer', ROOT / 'experiments/autodan-role-probe/scorer.py')
    collect, _, attention, classifier = scorer.dependencies()
    probe = classifier.load_probe(args.probe)
    assert tuple(probe.metadata['roles']) == ROLES
    binding = probe.metadata['model_binding']
    for name, sha in binding['assets_sha256'].items():
        assert shared.digest(args.model / name) == sha
    proof = json.loads((args.model / 'staging-proof.json').read_text())
    assert proof['all_shards_verified'] and {r['name']: r['sha256'] for r in proof['shards']} == binding['weight_shards_sha256']
    os.environ['HF_HOME'] = str(ROOT / 'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR'] = f'/tmp/wz55191-ipi-posid-triton-{args.shard}'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    net = probe.torch(device='cpu').eval()
    mean, scale = torch.from_numpy(probe.mean), torch.from_numpy(probe.scale)
    heads = probe.metadata['heads']
    started = time.monotonic()
    model, device_map = collect.load_model(args.model)
    attention.install_tiled_attention(256)
    tag = f'shard-{args.shard}-of-{args.shards}'
    shared.write(out / 'shards' / f'{tag}-runtime.json', dict(device_map=device_map,
        packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'accelerate', 'kernels', 'triton', 'numpy']},
        model_load_seconds=time.monotonic() - started, probe_sha256=shared.SHA,
        source_inputs_sha256=source_protocol['inputs_sha256'], runner_sha256=shared.digest(__file__),
        attention='Original eager equation, 256-query tiles; causal and sliding-window masks follow token order'))

    def forward(row):
        positions = np.asarray(row['target_positions'], np.int64)
        with torch.inference_mode(), collect.cpu_transfers(), collect.Capture(model, heads, row['input_ids'], positions) as capture:
            model.model(input_ids=torch.tensor([row['input_ids']], device='cuda:0'),
                position_ids=torch.tensor([row['position_ids']], device='cuda:0'),
                attention_mask={'full_attention': None, 'sliding_attention': None}, use_cache=False, return_dict=True)
            native = capture.finish()
        assert native.dtype == torch.bfloat16
        features = native.to(torch.float16).reshape(len(positions), 1024).numpy()
        with torch.inference_mode():
            logits = net((torch.from_numpy(features).float() - mean) / scale).numpy()
            probabilities = torch.from_numpy(logits).softmax(1).numpy()
        assert np.isfinite(logits).all() and np.allclose(probabilities.sum(1), 1, atol=1e-6)
        return positions, features, logits, probabilities

    checks = []
    for index, row in enumerate(variants):
        path = out / 'readouts' / (row['input_id'] + '.npz')
        cached = path.exists()
        if not cached:
            positions, features, logits, probabilities = forward(row)
            if row['condition'] == 'baseline_arange':
                # Explicit 0..N-1 ids must reproduce the sealed baseline readout exactly.
                with np.load(SOURCE / 'readouts' / (row['input_id'].rsplit('--', 1)[0] + '--baseline.npz')) as sealed:
                    exact = (np.array_equal(sealed['positions'], positions)
                             and np.array_equal(sealed['head_features'].view(np.uint16), features.view(np.uint16))
                             and np.array_equal(sealed['logits'].view(np.uint32), logits.view(np.uint32)))
                    difference = float(np.abs(sealed['probabilities'] - probabilities).max())
                checks.append(dict(trace_id=row['trace_id'], bit_exact=bool(exact), max_abs_probability_difference=difference))
                assert exact, f'{row["trace_id"]}: explicit arange ids differ from sealed baseline ({difference})'
            temp = path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, positions=positions,
                original_positions=np.asarray(row['target_original_positions'], np.int64),
                position_ids=np.asarray(row['position_ids'], np.int64)[positions],
                token_ids=np.asarray(row['input_ids'], np.int64)[positions], logits=logits, probabilities=probabilities,
                roles=np.asarray(ROLES), condition=row['condition'], source_role=row['source_role'] or '',
                role_names=np.asarray(list(row['role_target_positions'])),
                **{'mask_' + role: np.asarray(values, np.int64) for role, values in row['role_target_positions'].items()})
            temp.replace(path)
        status = dict(done=index + 1, total=len(variants), trace_id=row['trace_id'], condition=row['condition'],
                      source_role=row['source_role'], cached=cached, tokens=row['input_tokens'], seconds=time.monotonic() - started)
        print(json.dumps(status), flush=True)
    shared.write(out / 'shards' / f'{tag}-complete.json', dict(status='complete', inputs=len(variants),
        traces=sorted(mine), seconds=time.monotonic() - started, baseline_checks=checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model', type=Path, default=shared.MODEL)
    p.add_argument('--probe', type=Path, default=shared.PROBE)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--traces', nargs='*', help='Optional trace ids (smoke test)')
    run(p.parse_args())


if __name__ == '__main__':
    main()
