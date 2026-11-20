"""One native model load: score named templates on the fixed contexts and probe prepared IPI runs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Match the frozen training runtime: reuse the cached kernel revision instead
# of resolving its mutable Hub version ref in this later, separate process.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from ds_common import PROBE_RUN, read_json, sha, write_json
from ds_scorer import Scorer


BACKEND_PARITY_FIELDS = (
    'model_path', 'config_sha256', 'index_sha256', 'backend_source_sha256',
    'torch', 'cuda', 'transformers', 'triton', 'kernels',
    'layers', 'heads_per_layer', 'head_dim', 'device_map',
    'native_packed_fp4_fp8', 'parameters_frozen', 'all_parameters_on_cuda',
    'parameter_bytes_total', 'dtype_numel', 'dtype_bytes',
    'quantization_config', 'official_checkpoint_quantization_config', 'official_checkpoint_expert_dtype',
    'expert_backends', 'experts_implementation', 'attention_implementation',
    'attention_query_tile_size', 'attention_formula', 'capture', 'capture_after_inverse_rope',
    'forward_target', 'use_cache',
)


def audit_backend_parity(current, output, reference_path=PROBE_RUN / 'model_backend.json'):
    """Persist the actual runtime and reject numeric-backend drift before scoring."""
    output, reference_path = Path(output).resolve(), Path(reference_path).resolve()
    snapshot = output / 'final_probe_model_backend.json'
    write_json(snapshot, current)
    current = read_json(snapshot)  # Compare the saved JSON semantics (tuple/list normalize alike).
    expected = read_json(reference_path)
    mismatches = []
    for field in BACKEND_PARITY_FIELDS:
        if field not in current or field not in expected or current[field] != expected[field]:
            mismatches.append(dict(field=field, expected=expected.get(field), actual=current.get(field)))

    def kernel_identity(backend):
        return sorted([dict(repository=kernel.get('repository'), version=kernel.get('version'),
                            revision=kernel.get('revision')) for kernel in backend.get('loaded_kernels', [])],
                      key=lambda row: json.dumps(row, sort_keys=True))

    expected_kernels, actual_kernels = kernel_identity(expected), kernel_identity(current)
    if not expected_kernels or actual_kernels != expected_kernels:
        mismatches.append(dict(field='loaded_kernels.repository/version/revision',
                               expected=expected_kernels, actual=actual_kernels))
    audit = dict(schema='deepseek-final-probe-backend-parity-v1', passed=not mismatches,
                 reference=str(reference_path), reference_sha256=sha(reference_path),
                 actual_snapshot=str(snapshot), actual_snapshot_sha256=sha(snapshot),
                 compared_fields=list(BACKEND_PARITY_FIELDS) + ['loaded_kernels.repository/version/revision'],
                 ignored_runtime_fields=['load_seconds', 'gpu_memory'], mismatches=mismatches,
                 environment={key: os.environ.get(key) for key in
                              ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TRITON_CACHE_DIR')})
    write_json(output / 'final_probe_backend_audit.json', audit)
    if mismatches:
        raise RuntimeError('Final probe backend differs from frozen training: '
                           + ', '.join(row['field'] for row in mismatches))
    return audit


@contextmanager
def prefetch_during_load(output, scores_out):
    """Independent, best-effort reader; its failure never changes probe execution."""
    helper = None
    log = None
    try:
        status_dir = Path(scores_out).resolve().parent / 'final_probe_prefetch' / str(os.getpid())
        status_dir.mkdir(parents=True, exist_ok=True)
        log = (status_dir / 'prefetch.log').open('a')
        command = [sys.executable, '-u', str(Path(__file__).with_name('ds_prefetch_window.py')),
                   '--pid', str(os.getpid()), '--kind', 'final-probe', '--output', str(Path(output).resolve()),
                   '--status-dir', str(status_dir), '--backend-after-unix', str(time.time()), '--already-read']
        helper = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(stage='prefetch', helper_pid=helper.pid, status_dir=str(status_dir))), flush=True)
    except Exception as error:
        print(json.dumps(dict(stage='prefetch-warning', error=f'{type(error).__name__}: {error}')), flush=True)
    try:
        yield
    finally:
        try:
            if helper is not None and helper.poll() is None:
                helper.terminate()
                try:
                    helper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    helper.kill()
                    helper.wait(timeout=5)
        except Exception as error:
            print(json.dumps(dict(stage='prefetch-warning', error=f'{type(error).__name__}: {error}')), flush=True)
        finally:
            if log is not None:
                try:
                    log.close()
                except Exception as error:
                    print(json.dumps(dict(stage='prefetch-warning', error=f'{type(error).__name__}: {error}')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='scorer output directory (cache + protocol)')
    parser.add_argument('--conditions-json', type=Path, nargs='*', default=[])
    parser.add_argument('--scores-out', type=Path, required=True)
    parser.add_argument('--ipi-runs', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.ipi_runs:
        # Prepared requests already contain the exact lengths, including long
        # IPI contexts and JSON escaping. Prime only those keys before CUDA/model
        # initialization, using the same measured dispatch protocol as training.
        from ds_prefill_cache_cpu import lengths_from_ipi_runs, prefill_lengths
        lengths, inputs = lengths_from_ipi_runs(args.ipi_runs)
        prefill_lengths(lengths, args.output / 'final_probe_cache_prefill_audit.json', apply=True, inputs=inputs)
    # This entry point is exec'ed after evolution, so the already-running
    # controller picks up this auxiliary loading support without a restart.
    with prefetch_during_load(args.output, args.scores_out):
        scorer = Scorer(args.dataset, args.output)
    try:
        audit_backend_parity(scorer.prober.model._deepseek_role_probe_backend, args.output)
    except Exception:
        scorer.close()
        raise
    results = {}
    for path in args.conditions_json:
        for name, spec in read_json(path).items():
            scorer.allow_special = bool(spec.get('allow_special'))
            result = scorer.score(spec['template'])
            results[name] = dict(p_user=result['fitness'], suite_means=result['suite_means'], goal_role_means=result['goal_role_means'],
                                 per_case={r['case_id']: r['goal_user_mean'] for r in result['per_case']},
                                 agentdojo_mean=sum(r['goal_user_mean'] for r in result['per_case'] if r['suite'] in ('banking', 'slack', 'travel'))
                                 / max(1, sum(r['suite'] in ('banking', 'slack', 'travel') for r in result['per_case'])),
                                 template=spec['template'])
            print(json.dumps(dict(condition=name, p_user=round(result['fitness'], 4),
                                  suites={k: round(v, 3) for k, v in result['suite_means'].items()},
                                  roles={k: round(v, 3) for k, v in result['goal_role_means'].items()})), flush=True)
            write_json(args.scores_out, results)
    scorer.allow_special = False
    import ds_ipi
    for run in args.ipi_runs:
        ds_ipi.probe_requests(run.resolve(), scorer.prober)
    scorer.close()


if __name__ == '__main__':
    main()
