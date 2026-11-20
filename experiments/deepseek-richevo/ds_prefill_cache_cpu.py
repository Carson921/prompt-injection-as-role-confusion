"""Reuse the training run's measured native MXFP4 configurations, without CUDA.

Only missing cache entries are published. The kernel, true S=6*tokens, weights,
precision, routing and feature extraction are unchanged. This follows the
training reference-dispatch protocol; arbitrary autotune configurations have
not been proved bitwise equivalent to one another.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from ds_common import PROBE_RUN, PROBE_RUN_V1, read_json, sha, write_json

ROLE_ANALYSIS = Path(__file__).resolve().parents[1] / 'role-analysis'
sys.path.insert(0, str(ROLE_ANALYSIS))
import prepare_deepseek_performance_cache as performance

REFERENCE = PROBE_RUN_V1 / 'performance-cache-gpu-validation.json'  # kernel-cache evidence, independent of the probe
CACHE = Path('/home/wz55191/.cache/triton')
PROTECTED_LENGTHS = {8, 8309}


def validate_reference(cache_dir, runtime_backend=None):
    """Check saved GPU evidence and current static compiler identity, CPU only."""
    from triton import Config
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    from triton.runtime.cache import triton_key

    report = read_json(REFERENCE)
    identity, validation = report['native_identity'], report['validation']
    if report['status'] != 'validation_passed' or validation['status'] != 'passed':
        raise ValueError('Training reference mechanism validation did not pass')
    checks = validation['checks']
    if {(c['projection'], c['tokens']) for c in checks} != {
            (p, n) for p in performance.PROJECTIONS for n in (8, 257, 1029, 8309)}:
        raise ValueError('Reference validation does not cover both projections and all four shapes')
    if not all(c['native_disk_cache_hit'] and not c['search_called'] and c['finite'] and c['bitexact']
               and c['maximum_absolute_difference'] == 0 for c in checks):
        raise ValueError('Same-configuration cache/direct output equivalence failed')
    if sha(Path(performance.__file__)) != report['tool_source_sha256']:
        raise ValueError('Validated CPU cache helper source changed')
    for kind in ('kernel', 'tuner'):
        if sha(Path(identity[f'{kind}_source_path'])) != identity[f'{kind}_source_sha256']:
            raise ValueError(f'Validated native {kind} source changed')
    current_static = dict(triton_key=triton_key(),
                          backend_hash=make_backend(GPUTarget('cuda', 90, 32)).hash(),
                          invalidating_environment_repr=str(sorted(get_cache_invalidating_env_vars().items())))
    if any(current_static[key] != identity[key] for key in current_static):
        raise ValueError('Static Triton/compiler/environment identity differs from training')
    if identity['target'] != dict(backend='cuda', arch=90, warp_size=32):
        raise ValueError('Reference target is not the production H100 SM90 target')
    expected_environment = read_json(PROBE_RUN / 'environment.json')['packages']
    for name in ('torch', 'triton', 'transformers', 'kernels', 'tokenizers'):
        if importlib.metadata.version(name).split('+')[0] != expected_environment[name]['version'].split('+')[0]:
            raise ValueError(f'Runtime package {name} differs from training')

    sources = report['plan']['sources']
    if {s['projection'] for s in sources.values()} != set(performance.PROJECTIONS) or len(sources) != 2:
        raise ValueError('Exactly two measured projection references are required')
    for source in sources.values():
        if source['source_tokens'] != 8 or not Path(source['path']).resolve().is_relative_to(cache_dir.resolve()):
            raise ValueError('Reference is not the measured eight-token entry in the selected cache')
        best = source['best']
        config = Config({k: best[k] for k in ('COMPUTE_MODE', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K')},
                        **{k: v for k, v in best.items() if k not in ('COMPUTE_MODE', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K')})
        if str(config) not in identity['config_grid_repr']:
            raise ValueError('Reference configuration is outside the validated native grid')
        key = performance.tuning_key(source['projection'], 8)
        observed = performance.measured_winner(Path(source['path']), key, [config])
        if observed is None or observed['sha256'] != source['sha256'] or observed['id'] != source['id']:
            raise ValueError('Measured reference cache bytes changed')

    expected_backend = read_json(PROBE_RUN / 'model_backend.json')
    if runtime_backend is not None:
        actual_backend = read_json(runtime_backend)
        fields = ('torch', 'cuda', 'triton', 'kernels', 'transformers', 'backend_source_sha256',
                  'config_sha256', 'index_sha256', 'quantization_config', 'capture',
                  'attention_implementation', 'attention_query_tile_size', 'experts_implementation')
        if any(actual_backend[k] != expected_backend[k] for k in fields):
            raise ValueError('Loaded production model backend differs from training')
        kernel_ids = lambda d: sorted((k['repository'], str(k['version']), k['revision']) for k in d['loaded_kernels'])
        if kernel_ids(actual_backend) != kernel_ids(expected_backend):
            raise ValueError('Loaded production kernel revisions differ from training')
    return report, dict(reference=str(REFERENCE), reference_sha256=sha(REFERENCE),
                        reference_helper_sha256=report['tool_source_sha256'],
                        static_identity_matches=True, native_sources_match=True, measured_references_match=True,
                        runtime_backend=None if runtime_backend is None else str(Path(runtime_backend).resolve()),
                        runtime_backend_sha256=None if runtime_backend is None else sha(Path(runtime_backend)),
                        equivalence_scope='same measured reference config: direct launch versus native true-S cache dispatch at 8/257/1029/8309; not arbitrary configs versus each other')


def prefill_lengths(lengths, audit_path, *, apply=False, cache_dir=CACHE, runtime_backend=None, inputs=None):
    """Reusable exact-length API. Does not inspect a GPU or initialize CUDA."""
    lengths = sorted(set(lengths))
    if not lengths or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in lengths):
        raise ValueError('Expected a nonempty collection of positive token lengths')
    cache_dir, audit_path = Path(cache_dir).resolve(), Path(audit_path).resolve()
    reference, validation = validate_reference(cache_dir, runtime_backend)
    sources, identity = reference['plan']['sources'], reference['native_identity']
    by_projection = {s['projection']: s for s in sources.values()}
    targets = []
    for n in sorted(set(lengths) | PROTECTED_LENGTHS):
        for projection in performance.PROJECTIONS:
            key = performance.tuning_key(projection, n)
            path = performance.native_cache_path(cache_dir, identity, key)
            before = sha(path) if path.is_file() else None
            targets.append(dict(tokens=n, projection=projection, target_key=key, path=str(path),
                                before_sha256=before, source_winner_id=by_projection[projection]['id'],
                                action='protected' if n in PROTECTED_LENGTHS else 'existing' if before else 'create'))
    plan = dict(sources=sources, targets=targets)
    audit = dict(schema='deepseek-cpu-native-reference-cache-prefill-v1', status='dry_run',
                 started_unix=time.time(), helper_sha256=sha(Path(__file__)), cache_dir=str(cache_dir),
                 cuda_api_called=False, model_loaded=False, validation=validation, inputs=inputs or [],
                 requested_length_count=len(lengths), minimum_length=min(lengths), maximum_length=max(lengths),
                 protected_lengths=sorted(PROTECTED_LENGTHS), counts=dict(Counter(t['action'] for t in targets)),
                 concurrency='Atomic no-clobber publication; an already-running native search does not reread disk and may later atomically replace a newly seeded key with its own legitimate result. In-memory existing keys are unchanged.',
                 frozen_probe_sha256=sha(PROBE_RUN / 'checkpoint.npz'), plan=plan)
    write_json(audit_path, audit)
    if not apply:
        return audit
    outcomes = Counter()
    for i, target in enumerate(targets, 1):
        result = performance.apply_plan(dict(sources=sources, targets=[target]))
        outcomes.update(result)
        target['publication_outcome'] = next(iter(result))
        if i % 1000 == 0:
            print(json.dumps(dict(stage='cpu_cache_prefill', processed=i, total=len(targets), outcomes=dict(outcomes))), flush=True)
    existing_changed, seeded_replaced = [], []
    for target in targets:
        path = Path(target['path'])
        after = target['after_sha256'] = sha(path) if path.is_file() else None
        if target['before_sha256'] is not None and after != target['before_sha256']:
            existing_changed.append(target['path'])
        if target['publication_outcome'] == 'created':
            source = sources[target['source_winner_id']]
            seeded_sha = sha(json.dumps(dict(best=source['best'], timings=[])) + '\n')
            if after != seeded_sha:
                seeded_replaced.append(target['path'])
    audit.update(status='applied', finished_unix=time.time(), application=dict(outcomes),
                 existing_changed_externally=existing_changed, seeded_replaced_by_native=seeded_replaced)
    write_json(audit_path, audit)
    print(json.dumps(dict(stage='cpu_cache_prefill_complete', application=dict(outcomes),
                          existing_changed_externally=len(existing_changed), seeded_replaced_by_native=len(seeded_replaced),
                          audit=str(audit_path))), flush=True)
    return audit


def lengths_from_ipi_runs(runs):
    lengths, inputs = set(), []
    for run in runs:
        path = Path(run).resolve() / 'protocol.json'
        protocol = read_json(path)
        lengths.update(int(row['prompt_tokens']) for row in protocol['rows'])
        inputs.append(dict(protocol=str(path), sha256=sha(path), requests=len(protocol['rows'])))
    return lengths, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=int)
    parser.add_argument('--last', type=int)
    parser.add_argument('--lengths-json', type=Path, help='JSON array of exact prompt token lengths')
    parser.add_argument('--ipi-runs', type=Path, nargs='*', default=[])
    parser.add_argument('--runtime-backend', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    lengths, inputs = lengths_from_ipi_runs(args.ipi_runs)
    if args.first is not None or args.last is not None:
        if args.first is None or args.last is None or args.first > args.last:
            parser.error('--first and --last must specify a nonempty inclusive range')
        lengths.update(range(args.first, args.last + 1))
        inputs.append(dict(inclusive_range=[args.first, args.last]))
    if args.lengths_json is not None:
        lengths.update(read_json(args.lengths_json))
        inputs.append(dict(lengths_json=str(args.lengths_json.resolve()), sha256=sha(args.lengths_json)))
    prefill_lengths(lengths, args.audit, apply=args.apply, runtime_backend=args.runtime_backend, inputs=inputs)


if __name__ == '__main__':
    main()
