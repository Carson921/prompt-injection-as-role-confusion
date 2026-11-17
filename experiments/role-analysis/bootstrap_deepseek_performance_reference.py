"""Create measured eight-token native MXFP4 performance references only.

This optional cold-cache setup calibrates the unmodified official native kernel
at S=8*6=48 for gate_up and down. It uses synthetic packed FP4 expert matrices
with the real E=256 projection dimensions and native UE8M0 scale format. It is not a
backbone forward, does not read model weights or prepared inputs, and never
searches the 8309-token workload. Run separately on one available H100 before
the reproduction script; CUDA_VISIBLE_DEVICES must select exactly one GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import prepare_deepseek_performance_cache as performance


TOKENS = 8
EXPERTS = 256
MEMORY_LIMIT_GIB = 3


def run(args):
    # Native cache selection must be fixed before importing/initializing Triton.
    args.cache_dir = args.cache_dir.expanduser().resolve()
    os.environ["TRITON_CACHE_DIR"] = str(args.cache_dir)
    result = {"format_version": "native-eight-token-performance-reference-v1", "status": "initializing",
              "scope": "Synthetic native-kernel calibration; not a complete model forward or corpus benchmark",
              "tokens": TOKENS, "S": TOKENS * performance.TOP_K, "experts": EXPERTS,
              "reference_projections": list(performance.PROJECTIONS), "model_loaded": False,
              "model_weights_read": False, "prepared_inputs_read": False,
              "maximum_calibration_tokens": TOKENS, "allocator_limit_bytes": MEMORY_LIMIT_GIB * 2**30,
              "tool_source_sha256": performance.sha(__file__),
              "performance_helper_sha256": performance.sha(performance.__file__),
              "cache_dir": str(args.cache_dir), "started_unix": time.time(), "references": []}
    performance.dump(args.output, result)
    try:
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError("Select exactly one available H100 with CUDA_VISIBLE_DEVICES before calibration")
        runtime = performance.initialize_runtime()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_GIB * 2**30 / total)
        torch.cuda.reset_peak_memory_stats()
        result["native_identity"] = runtime["identity"]
        torch.manual_seed(79031)
        for projection, (n, k) in performance.PROJECTIONS.items():
            if args.only_projection and projection != args.only_projection:
                continue
            key = performance.tuning_key(projection, TOKENS)
            path = performance.native_cache_path(args.cache_dir, runtime["identity"], key)
            # Refuse malformed/unmeasured existing cache entries; never replace
            # them with fabricated timings or an arbitrary guessed configuration.
            winner = performance.measured_winner(path, key, runtime["configs"])
            if winner is not None:
                result["references"].append({**winner, "projection": projection, "source_tokens": TOKENS,
                                             "reused_existing_measured_reference": True})
                performance.dump(args.output, result)
                continue
            started = time.monotonic()
            weights = torch.randint(-128, 128, (EXPERTS, n, k // 2), dtype=torch.int8, device="cuda")
            scales = torch.full((EXPERTS, n, k // 32), 118, dtype=torch.uint8,
                                device="cuda").view(torch.float8_e8m0fnu)
            activations = torch.randn((TOKENS * performance.TOP_K, k), dtype=torch.bfloat16, device="cuda")
            ids = (torch.arange(TOKENS * performance.TOP_K, device="cuda", dtype=torch.int64) * 3) % EXPERTS
            # The original BayesianAutotuner performs its real short-shape
            # search and publishes the native measured timing table itself.
            with torch.inference_mode():
                output = runtime["bundle"].batched_matmul(activations, weights, scales, ids, [128, 128])
            torch.cuda.synchronize()
            if tuple(output.shape) != (TOKENS * performance.TOP_K, n) or not torch.isfinite(output).all():
                raise ValueError("Short native calibration produced invalid output")
            winner = performance.measured_winner(path, key, runtime["configs"])
            if winner is None:
                raise ValueError("Official short-shape autotuner did not publish a measured reference")
            result["references"].append({**winner, "projection": projection, "source_tokens": TOKENS,
                                         "reused_existing_measured_reference": False,
                                         "calibration_seconds": time.monotonic() - started,
                                         "finite": True, "activation_shape": list(activations.shape),
                                         "packed_weight_shape": list(weights.shape),
                                         "scale_shape": list(scales.shape), "output_shape": list(output.shape)})
            performance.dump(args.output, result)
            del output, activations, ids, weights, scales
            gc.collect()
            torch.cuda.empty_cache()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if result["peak_allocated_bytes"] > MEMORY_LIMIT_GIB * 2**30:
            raise ValueError("Short calibration exceeded its documented allocator limit")
        result.update(status="passed", finished_unix=time.time())
        performance.dump(args.output, result)
        print(json.dumps({"status": "passed", "tokens": TOKENS,
              "projections": [row["projection"] for row in result["references"]],
              "scope": result["scope"], "output": str(args.output.resolve())}), flush=True)
        return result
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", finished_unix=time.time())
        performance.dump(args.output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("TRITON_CACHE_DIR", "~/.cache/triton")).expanduser())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-projection", choices=tuple(performance.PROJECTIONS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
