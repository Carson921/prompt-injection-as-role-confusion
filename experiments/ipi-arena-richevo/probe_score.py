"""Score goal-token P(user) of the frozen six-role probe on every prepared IPI Arena request.

Identical measurement to experiments/autodan-role-probe/scorer.py (same probe
checkpoint, same pre-o_proj head capture, same BF16->fp16->fp32 cast, same
tiled eager attention), but the prompt is each prepared IPI Arena initial
request (exact token ids the vLLM service consumed) and the scored tokens are
the goal tokens of its first payload occurrence.  Run after the vLLM service
has released both H100s.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from common import HERE, MODEL, ROOT, read_json, sha, write_json

SCORER_PATH = ROOT / 'experiments/autodan-role-probe/scorer.py'


def load_scorer_module():
    spec = importlib.util.spec_from_file_location('_autodan_scorer', SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Prober:
    def __init__(self, model_path: Path, probe_path: Path, output: Path):
        import torch
        scorer = load_scorer_module()
        self.scorer, self.roles = scorer, scorer.ROLES
        self.collect, self.spans, self.attention, classifier = scorer.dependencies()
        if sha(probe_path) != scorer.PROBE_SHA256:
            raise ValueError('Frozen probe bytes differ')
        self.probe = classifier.load_probe(probe_path)
        self.heads = self.probe.metadata['heads']
        assert tuple(self.probe.metadata['roles']) == self.roles
        binding = self.probe.metadata['model_binding']
        for name, expected in binding['assets_sha256'].items():
            if sha(model_path / name) != expected:
                raise ValueError(f'Native model/tokenizer asset changed: {name}')
        proof = read_json(model_path / 'staging-proof.json')
        if not (proof['all_shards_verified'] and {r['name']: r['sha256'] for r in proof['shards']} == binding['weight_shards_sha256']):
            raise ValueError('Staged native model shard binding differs')
        os.environ.setdefault('HF_HOME', str(ROOT / 'experiments/single-message-pre-mlp/outputs/huggingface'))
        os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/wz55191-autodan-role-probe-triton')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        torch.set_num_threads(2)
        self.probe_net = self.probe.torch(device='cpu')
        self.probe_mean = torch.from_numpy(self.probe.mean)
        self.probe_scale = torch.from_numpy(self.probe.scale)
        self.query_tile = int(os.environ.get('AUTODAN_QUERY_TILE', '256'))
        self.protocol = dict(format='ipi-arena-goal-token-user-probability-v1', probe_sha256=sha(probe_path),
                             roles=list(self.roles), head_ids=self.heads, feature_position='pre-o_proj',
                             target_role='user', target_role_index=self.roles.index('user'),
                             feature_cast='native BF16 -> float16 -> float32', probe_activation='gelu_exact',
                             query_tile=self.query_tile, truncation=False, batch_size=1,
                             scored_tokens='goal tokens of the first rendered payload occurrence',
                             model_binding=binding, staging_proof_sha256=sha(model_path / 'staging-proof.json'),
                             scorer_code_sha256=sha(SCORER_PATH), code_sha256=sha(Path(__file__)))
        write_json(output / 'probe_protocol.json', self.protocol)
        self.model, _ = self.collect.load_model(model_path)
        self.attention.install_tiled_attention(self.query_tile)

    def score(self, request: dict) -> dict:
        import torch
        started = time.monotonic()
        ids = request['prompt_token_ids']
        occurrence = request['payload_occurrences'][request['scored_occurrence']]
        positions = np.asarray(occurrence['token_indices'], dtype=np.int64)
        goal_mask = np.isin(positions, np.asarray(occurrence['goal_token_indices'], dtype=np.int64))
        features_native, _ = self.collect.forward_capture(self.model, self.heads, list(ids), positions)
        assert features_native.dtype == torch.bfloat16
        features = features_native.to(dtype=torch.float16).reshape(len(positions), -1).numpy()
        with torch.inference_mode():
            standardized = (torch.from_numpy(features).float() - self.probe_mean) / self.probe_scale
            probabilities = self.probe_net(standardized).softmax(1).numpy()
        assert probabilities.shape == (len(positions), 6) and np.isfinite(probabilities).all()
        means = {name: dict(zip(self.roles, map(float, probabilities[mask].mean(0, dtype=np.float64))))
                 for name, mask in [('goal', goal_mask), ('full_payload', np.ones(len(positions), bool))]
                 if mask.any()}
        if (~goal_mask).any():
            means['template'] = dict(zip(self.roles, map(float, probabilities[~goal_mask].mean(0, dtype=np.float64))))
        return dict(probabilities=probabilities, positions=positions, goal_mask=goal_mask, means=means,
                    elapsed_seconds=time.monotonic() - started, input_tokens=len(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--probe', type=Path, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    scorer = load_scorer_module()
    probe_path = args.probe or scorer.DEFAULT_PROBE
    protocol = read_json(output / 'protocol.json')
    prober = Prober(args.model, probe_path, output)
    probe_dir = output / 'probe'
    results = []
    for index, row in enumerate(protocol['rows']):
        destination = probe_dir / row['condition'] / f"{row['case_id']}.json"
        if destination.exists():
            existing = read_json(destination)
            if existing.get('status') == 'complete' and existing.get('request_sha256') == row['request_sha256']:
                results.append(existing)
                continue
        request = read_json(output / row['request'])
        if sha(output / row['request']) != row['request_sha256']:
            raise ValueError(f'Request changed since protocol: {row["request"]}')
        result = prober.score(request)
        archive = destination.with_suffix('.npz')
        archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, input_ids=np.asarray(request['prompt_token_ids'], np.int64),
                            payload_token_indices=result['positions'], goal_mask=result['goal_mask'],
                            probabilities=result['probabilities'], roles=np.asarray(prober.roles))
        record = dict(status='complete', condition=row['condition'], case_id=row['case_id'], category=row['category'],
                      primary=row['primary'], request_sha256=row['request_sha256'], input_tokens=result['input_tokens'],
                      goal_tokens=int(result['goal_mask'].sum()), payload_tokens=int(len(result['positions'])),
                      goal_user_mean=result['means']['goal']['user'], region_means=result['means'],
                      elapsed_seconds=result['elapsed_seconds'], archive=str(archive.relative_to(output)),
                      probe_protocol_sha256=sha(json.dumps(prober.protocol, sort_keys=True)))
        write_json(destination, record)
        results.append(record)
        print(json.dumps(dict(stage='probe', index=index + 1, total=len(protocol['rows']), condition=row['condition'],
                              case_id=row['case_id'], p_user=round(record['goal_user_mean'], 4),
                              seconds=round(result['elapsed_seconds'], 1))), flush=True)
    conditions = read_json(output / 'conditions.json')
    summary = {}
    for condition in conditions:
        rows = [r for r in results if r['condition'] == condition]
        primary = [r for r in rows if r['primary']]
        summary[condition] = dict(
            p_user_mean=float(np.mean([r['goal_user_mean'] for r in primary])),
            p_user_mean_all=float(np.mean([r['goal_user_mean'] for r in rows])),
            goal_role_means={role: float(np.mean([r['region_means']['goal'][role] for r in primary])) for role in prober.roles},
            per_category={c: float(np.mean([r['goal_user_mean'] for r in primary if r['category'] == c]))
                          for c in ('browser', 'coding', 'tool')},
            cases=len(primary), p_user_agentdojo=conditions[condition]['p_user_agentdojo'])
    write_json(output / 'probe_summary.json', summary)
    print(json.dumps({k: round(v['p_user_mean'], 4) for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
