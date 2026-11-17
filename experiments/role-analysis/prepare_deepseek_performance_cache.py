"""Plan native MXFP4 autotune-cache reuse without changing model computation.

The default is a dry run. ``--apply`` creates absent native performance-cache
entries atomically, using a configuration already measured by the original
kernel at a reference length. It never overwrites an entry or writes protected
lengths (8309 tokens by default). It does not load a model or change input IDs,
routing, quantization, kernel source, tensor dtypes, or launch-grid dimensions.

The exact upstream cache identity includes Triton, backend/SM architecture, JIT
source, invalidating environment, full tuning key and complete configuration
grid. A device ordinal is absent from that upstream identity; identical H100s
share it. Each new entry retains the true target S=6*input_tokens. Empty target
``timings`` explicitly means its reused configuration was not timed there.

``--validate-gpu`` additionally tests native-cache hits and outputs on the one
visible GPU, with no model and a 2 GiB Torch allocator cap. Run it separately
with CUDA_VISIBLE_DEVICES=3 while the production pipeline uses GPU0.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


FILENAME = "mxfp_dynamic_matmul_batched_kernel.bayes_autotune.json"
DTYPES = ["torch.bfloat16", "torch.uint8", "torch.bfloat16", "torch.uint8", "torch.int64"]
PROJECTIONS = {"gate_up": (4096, 4096), "down": (4096, 2048)}
TOP_K = 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def native_cache_hash(identity, tuning_key):
    """Exact BayesianAutotuner.check_disk_cache hash construction, CPU only."""
    pieces = [identity["triton_key"], identity["backend_hash"], identity["jit_cache_key"],
              identity["invalidating_environment_repr"], str(tuple(tuning_key))]
    pieces += identity["config_grid_repr"]
    return hashlib.sha256("-".join(pieces).encode("utf-8")).hexdigest()


def native_cache_path(cache_dir, identity, tuning_key):
    digest = native_cache_hash(identity, tuning_key)
    directory = base64.b32encode(bytes.fromhex(digest)).decode("utf-8").rstrip("=")
    return Path(cache_dir) / directory / FILENAME


def tuning_key(projection, tokens):
    n, k = PROJECTIONS[projection]
    return [n, k, int(tokens) * TOP_K, *DTYPES]


def initialize_runtime():
    """Import official installed code and obtain its identity; no kernel launch."""
    import torch
    import triton
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.compiler.compiler import make_backend
    from triton.runtime.cache import triton_key
    from triton.runtime.driver import driver
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel

    torch.cuda.set_device(0)
    bundle = load_finegrained_fp8_kernel()
    module = importlib.import_module(bundle.batched_matmul.__module__)
    tuner = module.mxfp_dynamic_matmul_batched_kernel
    fn = tuner.fn
    target = driver.active.get_current_target()
    if (target.backend, target.arch, target.warp_size) != ("cuda", 90, 32):
        raise ValueError(f"Expected H100 SM90, got {target}")
    if tuner.keys != ["N", "K", "S"] or fn.arg_names[:5] != ["A", "B", "C", "Bs", "ExpertIds"]:
        raise ValueError("Native kernel key or tensor-argument order changed")
    tuner.nargs = {}
    configs = tuner.prune_configs({})
    tuner.nargs = None
    # Current native GRF pruning is a no-op on CUDA. Refuse to fabricate keys if
    # a later runtime introduces workload-dependent CUDA configuration pruning.
    if list(configs) != list(tuner.configs):
        raise ValueError("CUDA config pruning changed; re-audit native cache schema")
    identity = {
        "triton_key": triton_key(), "backend_hash": make_backend(target).hash(),
        "jit_cache_key": fn.cache_key,
        "invalidating_environment_repr": str(sorted(get_cache_invalidating_env_vars().items())),
        "config_grid_repr": [str(config) for config in configs],
        "target": {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size},
        "torch": torch.__version__, "triton": triton.__version__,
        "kernel_module": module.__name__, "kernel_source_path": str(Path(module.__file__).resolve()),
        "kernel_source_sha256": sha(module.__file__),
        "tuner_source_path": str(Path(importlib.import_module(type(tuner).__module__).__file__).resolve()),
        "tuner_source_sha256": sha(importlib.import_module(type(tuner).__module__).__file__),
    }
    return {"identity": identity, "module": module, "tuner": tuner, "configs": configs, "bundle": bundle}


def measured_winner(path, expected_key, configs):
    """Only reuse a finite winner from a real multi-candidate native search."""
    if not path.is_file():
        return None
    raw = path.read_bytes()
    value = json.loads(raw)
    best = value.get("best")
    matches = [c for c in configs if c.all_kwargs() == best]
    timings = value.get("timings", [])
    measured = [ms for config, ms in timings if config == best and isinstance(ms, (float, int))
                and math.isfinite(ms) and ms > 0]
    if len(matches) != 1 or len(timings) <= 1 or not measured:
        raise ValueError(f"Winner is not a measured native multi-config result: {path}")
    source_id = hashlib.sha256(str(path.resolve()).encode() + b"\0" + raw).hexdigest()
    return {"id": source_id, "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "source_key": list(expected_key), "best": best,
            "measured_candidates": len(timings), "winner_ms": measured[0]}


def input_lengths(data_dir):
    lengths, bindings, split_stats = set(), [], {}
    for path in sorted(Path(data_dir).glob("inputs-*.jsonl")):
        count, digest = Counter(), hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if line.strip():
                    count[len(json.loads(line)["input_ids"])] += 1
        if count:
            lengths.update(count)
            split_stats[path.stem] = {"records": sum(count.values()), "unique_lengths": len(count),
                                      "minimum": min(count), "maximum": max(count)}
        bindings.append({"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    if not lengths:
        raise ValueError(f"No prepared inputs found: {data_dir}")
    return sorted(lengths), bindings, split_stats


def make_plan(identity, configs, cache_dir, lengths, reference_tokens, protected_tokens):
    sources = {}
    by_projection = {}
    for projection in PROJECTIONS:
        candidates = []
        for tokens in sorted(set(reference_tokens)):
            key = tuning_key(projection, tokens)
            path = native_cache_path(cache_dir, identity, key)
            winner = measured_winner(path, key, configs)
            if winner:
                winner.update(projection=projection, source_tokens=tokens)
                sources[winner["id"]] = winner
                candidates.append(winner)
        if not candidates:
            raise ValueError(f"No measured native winner matches full runtime identity for {projection}")
        by_projection[projection] = candidates
    targets = []
    for tokens in sorted(set(lengths)):
        for projection in PROJECTIONS:
            key = tuning_key(projection, tokens)
            path = native_cache_path(cache_dir, identity, key)
            winner = min(by_projection[projection], key=lambda w: abs(math.log2(tokens / w["source_tokens"])))
            state = "protected" if tokens in protected_tokens else ("existing" if path.exists() else "create")
            targets.append({"tokens": tokens, "projection": projection, "target_key": key,
                            "native_cache_sha256": native_cache_hash(identity, key), "path": str(path.resolve()),
                            "source_winner_id": winner["id"], "action": state})
    return {"sources": sources, "targets": targets,
            "counts": dict(Counter(item["action"] for item in targets)),
            "protected_tokens": sorted(protected_tokens), "unique_input_lengths": len(set(lengths))}


def create_absent_cache(path, value):
    """Atomic, no-clobber publication; readers cannot observe partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="cache-reuse-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply_plan(plan):
    outcomes = Counter()
    for target in plan["targets"]:
        if target["action"] != "create":
            outcomes["skipped_" + target["action"]] += 1
            continue
        source = plan["sources"][target["source_winner_id"]]
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Measured source winner changed after planning")
        # No target timings are invented. Native check_disk_cache accepts a
        # config-grid member and restores an empty configs_timings dictionary.
        created = create_absent_cache(target["path"], {"best": source["best"], "timings": []})
        outcomes["created" if created else "skipped_race_existing"] += 1
    return dict(outcomes)


def validate_gpu(runtime, plan, lengths=(8, 257, 1029, 8309), memory_limit_gib=2):
    """Verify official disk-cache loading and native output, with no autotune."""
    import torch
    from triton.runtime import cache as triton_cache

    if torch.cuda.device_count() != 1:
        raise ValueError("Validation requires exactly one visible GPU, e.g. CUDA_VISIBLE_DEVICES=3")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(memory_limit_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    baseline_free, total = torch.cuda.mem_get_info()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    tuner, module = runtime["tuner"], runtime["module"]
    bayes_module = importlib.import_module(type(tuner).__module__)
    original_manager = bayes_module.get_cache_manager
    configs = runtime["configs"]
    results = []
    torch.manual_seed(79031)
    # E=4 changes only routing-table size, not projection N/K or native tuning
    # key. It keeps random packed weights small and exercises all four experts.
    with tempfile.TemporaryDirectory(prefix="native-performance-cache-validation-") as directory:
        def isolated_manager(key):
            class Manager:
                def get_file(self, filename):
                    path = Path(directory) / (key + "-" + filename)
                    return str(path) if path.is_file() else None
                def put(self, *args, **kwargs):
                    raise AssertionError("Validation must hit the prepared cache without searching")
            return Manager()

        try:
            bayes_module.get_cache_manager = isolated_manager
            for projection, (n, k) in PROJECTIONS.items():
                source = next(s for s in plan["sources"].values() if s["projection"] == projection)
                config = next(c for c in configs if c.all_kwargs() == source["best"])
                weights = torch.randint(-128, 128, (4, n, k // 2), dtype=torch.int8, device="cuda")
                scales = torch.full((4, n, k // 32), 118, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
                for tokens in lengths:
                    s = tokens * TOP_K
                    a = torch.randn((s, k), dtype=torch.bfloat16, device="cuda")
                    ids = (torch.arange(s, device="cuda", dtype=torch.int64) * 3) % 4
                    key = tuple(tuning_key(projection, tokens))
                    native_hash = native_cache_hash(runtime["identity"], key)
                    isolated_path = Path(directory) / (native_hash + "-" + FILENAME)
                    isolated_path.write_text(json.dumps({"best": config.all_kwargs(), "timings": []}))
                    # Exercise the original native disk reader, not our own parser.
                    tuner.cache.pop(key, None)
                    searched = []
                    hit = tuner.check_disk_cache(key, configs, lambda: searched.append(True))
                    if not hit or searched or tuner.cache.get(key) is not config:
                        raise AssertionError("Native hash/schema cache hit verification failed")
                    start = time.monotonic()
                    via_cache = runtime["bundle"].batched_matmul(a, weights, scales, ids, [128, 128])
                    torch.cuda.synchronize()
                    cached_seconds = time.monotonic() - start
                    # With a single original Config the official upstream
                    # Autotuner dispatches directly and cannot run a search.
                    prior_configs = tuner.configs
                    tuner.configs = [config]
                    try:
                        direct = runtime["bundle"].batched_matmul(a, weights, scales, ids, [128, 128])
                    finally:
                        tuner.configs = prior_configs
                    torch.cuda.synchronize()
                    # A full-output isfinite materializes several additional
                    # large tensors. Bound verification memory as well as GEMM
                    # memory so the longest real S stays under the 2 GiB cap.
                    for row_start in range(0, s, 512):
                        cached_rows = via_cache[row_start:row_start + 512]
                        direct_rows = direct[row_start:row_start + 512]
                        if not torch.isfinite(cached_rows).all() or not torch.isfinite(direct_rows).all():
                            raise AssertionError("Nonfinite native FP4 output")
                        if not torch.equal(cached_rows, direct_rows):
                            raise AssertionError("Cache reuse changed the original fixed-configuration output")
                    del cached_rows, direct_rows
                    results.append({"projection": projection, "tokens": tokens, "S": s,
                                    "native_disk_cache_hit": hit, "search_called": bool(searched),
                                    "shape": list(via_cache.shape), "finite": True, "bitexact": True,
                                    "maximum_absolute_difference": 0.0, "cached_seconds": cached_seconds,
                                    "edge_routing_ids": ids[[0, s // 2, s - 1]].tolist(),
                                    "source_winner_sha256": source["sha256"], "configuration": config.all_kwargs()})
                    print(json.dumps({"stage": "native_cache_validation", **results[-1]}), flush=True)
                    del a, ids, via_cache, direct
                del weights, scales
                torch.cuda.empty_cache()
        finally:
            bayes_module.get_cache_manager = original_manager
    final_free, _ = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated() - baseline_allocated
    if peak > memory_limit_gib * 2**30:
        raise AssertionError("GPU allocation limit exceeded")
    return {"status": "passed", "checks": results, "allocator_limit_bytes": memory_limit_gib * 2**30,
            "peak_extra_allocated_bytes": peak, "device_used_difference_after_cleanup": baseline_free - final_free,
            "device_total_bytes": total,
            "scope": "Original native fixed Config versus true-S native disk-cache dispatch; no target-length full autotune",
            "attention_mask": "Not used by this MoE kernel; all actual S rows and routing IDs are unchanged"}


def comma_ints(value):
    return [int(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("TRITON_CACHE_DIR", "~/.cache/triton")).expanduser())
    parser.add_argument("--reference-tokens", type=comma_ints, default=[8, 8309])
    parser.add_argument("--protected-tokens", type=comma_ints, default=[8309])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--validate-gpu", action="store_true")
    parser.add_argument("--validation-tokens", type=comma_ints, default=[8, 257, 1029, 8309])
    args = parser.parse_args()
    if args.apply and args.validate_gpu:
        parser.error("Run validation separately before applying an approved plan")
    runtime = initialize_runtime()
    lengths, bindings, statistics = input_lengths(args.data_dir)
    plan = make_plan(runtime["identity"], runtime["configs"], args.cache_dir, lengths,
                     args.reference_tokens, set(args.protected_tokens))
    result = {"format_version": "native-fp4-performance-cache-reuse-v1", "status": "dry_run",
              "tool_source_path": str(Path(__file__).resolve()), "tool_source_sha256": sha(__file__),
              "native_identity": runtime["identity"], "prepared_inputs": bindings,
              "split_statistics": statistics, "plan": plan,
              "invariants": {"original_kernel_source": True, "true_S_and_grid_unchanged": True,
                             "native_FP4_FP8_formats_unchanged": True, "model_not_loaded": True,
                             "input_tokens_and_routing_unchanged": True, "existing_entries_never_overwritten": True}}
    dump(args.output, result)
    if args.validate_gpu:
        try:
            result["validation"] = validate_gpu(runtime, plan, args.validation_tokens)
            result["status"] = "validation_passed"
        except Exception as error:
            result.update(status="validation_failed", validation_error=f"{type(error).__name__}: {error}")
            dump(args.output, result)
            raise
    if args.apply:
        result["application"] = apply_plan(plan)
        result["status"] = "applied"
    dump(args.output, result)
    print(json.dumps({"status": result["status"], "report": str(args.output.resolve()),
                      "counts": plan["counts"], "sources": len(plan["sources"]),
                      "application": result.get("application")}), flush=True)


if __name__ == "__main__":
    main()
