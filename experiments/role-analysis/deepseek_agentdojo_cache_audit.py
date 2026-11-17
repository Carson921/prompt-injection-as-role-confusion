"""Read-only cache-state diagnostics for an already resident DeepSeek V4 model.

This module does not load a model or change any weight, kernel, tolerance or
production source. ``run_gpu_diagnostics(adapter, output)`` performs three
ordinary cached forwards: prefill seven tokens, consume the eighth token, and
prefill all eight into an independent cache. It saves small cache-state samples
and exact structural invariants. Floating differences are observations, never
silently interpreted as an acceptable numerical tolerance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor)
    # BF16 has no native NumPy dtype; this conversion exactly preserves all
    # finite BF16 values, and does not modify the original cache tensor.
    return tensor.detach().float().cpu().numpy()


def snapshot_cache(cache, config, tokens, *, phase, samples):
    """Inspect actual official cache attributes; usable with small CPU fixtures."""
    arrays, layers, checks = {}, [], []
    def check(name, actual, expected):
        checks.append({"name": name, "actual": actual, "expected": expected, "ok": actual == expected})

    check("layer_count", len(cache.layers), config.num_hidden_layers)
    check("global_cumulative_tokens", int(cache.get_seq_length()), tokens)
    for index, layer in enumerate(cache.layers):
        kind = config.layer_types[index]
        expected_class = {"sliding_attention": "DynamicSlidingWindowLayer",
                          "compressed_sparse_attention": "DeepseekV4CSACache",
                          "heavily_compressed_attention": "DeepseekV4HCACache"}[kind]
        prefix = f"layer_{index:03d}"
        item = {"index": index, "layer_type": kind, "cache_class": type(layer).__name__, "tensors": {}}
        check(prefix + "/class", type(layer).__name__, expected_class)
        check(prefix + "/cumulative_length", int(layer.get_seq_length()), tokens)
        check(prefix + "/sliding_window", int(layer.sliding_window), int(config.sliding_window))
        expected_keys = [1, 1, min(tokens, config.sliding_window - 1), config.head_dim]

        def tensor(name, value, expected_shape):
            if value is None:
                actual_shape = None
                item["tensors"][name] = None
            else:
                array = _numpy(value)
                actual_shape = list(array.shape)
                key = prefix + "/" + name
                arrays[key] = array
                flat = array.reshape(-1)
                count = min(64, len(flat))
                indices = np.unique(np.linspace(0, len(flat)-1, count, dtype=np.int64)) if count else np.empty(0, np.int64)
                sample_key = phase + "__" + key.replace("/", "__")
                samples[sample_key] = flat[indices].copy()
                item["tensors"][name] = {"shape": actual_shape, "native_dtype": str(value.dtype),
                    "finite": bool(np.isfinite(array).all()), "sample_array": sample_key,
                    "sample_flat_indices": indices.tolist(), "cpu_copy_dtype": str(array.dtype)}
                check(key + "/finite", bool(np.isfinite(array).all()), True)
            check(prefix + "/" + name + "/shape", actual_shape, expected_shape)

        tensor("keys", layer.keys, expected_keys)
        tensor("values", layer.values, expected_keys)
        if layer.keys is not None and layer.values is not None:
            check(prefix + "/shared_K_equals_V", bool(np.array_equal(arrays[prefix + "/keys"], arrays[prefix + "/values"])), True)
        if kind != "sliding_attention":
            rate = config.compress_rates[kind]
            check(prefix + "/compress_rate", int(layer.compress_rate), int(rate))
            expected_count, remainder = divmod(tokens, rate)
            names = ("compressor", "indexer") if kind == "compressed_sparse_attention" else ("compressor",)
            item["entry_count"] = {name: int(layer.entry_count[name]) for name in names}
            for name in names:
                width = config.index_head_dim if name == "indexer" else config.head_dim
                source_width = width * (2 if kind == "compressed_sparse_attention" else 1)
                check(prefix + "/entry_count/" + name, int(layer.entry_count[name]), expected_count)
                tensor("buffer_kv/" + name, layer.buffer_kv[name], [1, remainder, source_width])
                tensor("buffer_gate/" + name, layer.buffer_gate[name], [1, remainder, source_width])
                tensor("compressed_kv/" + name, layer.compressed_kv[name], [1, expected_count, width])
                if kind == "compressed_sparse_attention":
                    overlap_shape = [1, rate, width] if expected_count else None
                    tensor("overlap_kv/" + name, layer.overlap_kv[name], overlap_shape)
                    tensor("overlap_gate/" + name, layer.overlap_gate[name], overlap_shape)
        layers.append(item)
    return {"phase": phase, "tokens": tokens, "invariants_passed": all(c["ok"] for c in checks),
            "checks": checks, "layers": layers}, arrays


def compare_cache_arrays(whole, split):
    """Quantify the identical semantic cache slots without choosing a tolerance."""
    result = []
    for key in sorted(set(whole) | set(split)):
        left, right = whole.get(key), split.get(key)
        row = {"tensor": key, "same_shape": left is not None and right is not None and left.shape == right.shape}
        if row["same_shape"]:
            finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
            row.update(finite=finite, values_bitexact=bool(left.dtype == right.dtype and
                       np.array_equal(left.view(np.uint8), right.view(np.uint8))))
            if finite:
                difference = left.astype(np.float64) - right.astype(np.float64)
                row.update(max_abs=float(np.max(np.abs(difference))) if difference.size else 0.0,
                           rms=float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
                           elements=int(difference.size), different_elements=int(np.count_nonzero(difference)))
        else:
            row["values_bitexact"] = False
        result.append(row)
    differing_layers = [int(row["tensor"].split("/", 1)[0].split("_")[1])
                        for row in result if not row["values_bitexact"]]
    return {"whole_vs_split_values_bitexact": all(r["values_bitexact"] for r in result),
            "first_layer_with_any_value_difference": min(differing_layers) if differing_layers else None,
            "floating_difference_is_not_an_acceptance_threshold": True, "tensors": result}


def run_gpu_diagnostics(adapter, output, *, token_ids=None):
    """Diagnose existing resident model; ``output`` is a new diagnostic directory.

    A returned ``diagnostic_completed`` status only means all observations were
    collected. Read ``structural_invariants_passed`` separately; no numerical
    equivalence or production-preflight pass is inferred from that status.
    """
    import torch
    from transformers import DynamicCache
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as official_model
    from transformers import cache_utils

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, config = adapter.model, adapter.model.config
    if model.training:
        raise ValueError("Read-only diagnostics require the resident model in frozen eval mode")
    if token_ids is None:
        prose = adapter.tokenizer.encode("The quick brown fox checks the complete cache prefix. " * 600,
                                         add_special_tokens=False)
        token_ids = [0, 128803] + prose[:4] + [128804, 128821]
    if len(token_ids) < 2:
        raise ValueError("Cache-state comparison requires at least two tokens")
    tokens, samples = len(token_ids), {}
    result = {"format_version": "deepseek-agentdojo-cache-state-diagnostic-v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "token_ids": list(map(int, token_ids)),
              "source_sha256": _sha(__file__), "official_model_source_sha256": _sha(inspect.getfile(official_model)),
              "official_cache_source_sha256": _sha(inspect.getfile(cache_utils)),
              "meaning": "Read-only native cache state observations; not a numerical-preflight acceptance report"}
    started = time.monotonic()
    split_cache = whole_cache = None
    try:
        ids = torch.as_tensor([token_ids], dtype=torch.long, device=model.model.embed_tokens.weight.device)
        prepare = getattr(adapter, "prepare_query_length", lambda n: None)
        def forward(values, cache, length):
            prepare(values.shape[1])
            value = model(input_ids=values,
                          attention_mask=torch.ones((1, length), dtype=torch.long, device=ids.device),
                          past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            new_cache = value.past_key_values
            del value
            return new_cache
        with torch.inference_mode():
            split_cache = forward(ids[:, :-1], DynamicCache(config=config), tokens - 1)
            prefix, _ = snapshot_cache(split_cache, config, tokens - 1, phase="split_prefix", samples=samples)
            split_cache = forward(ids[:, -1:], split_cache, tokens)
            split, split_arrays = snapshot_cache(split_cache, config, tokens, phase="split_final", samples=samples)
            # Explicitly release the first GPU cache before building the second.
            split_cache = None
            whole_cache = forward(ids, DynamicCache(config=config), tokens)
            whole, whole_arrays = snapshot_cache(whole_cache, config, tokens, phase="whole_final", samples=samples)
        sample_path = output / "cache-state-samples.npz"
        np.savez_compressed(sample_path, **samples)
        result.update(status="diagnostic_completed", structural_invariants_passed=all(
                          x["invariants_passed"] for x in (prefix, split, whole)),
                      snapshots=[prefix, split, whole], comparison=compare_cache_arrays(whole_arrays, split_arrays),
                      samples={"path": sample_path.name, "sha256": _sha(sample_path),
                               "arrays": len(samples), "maximum_values_per_tensor": 64,
                               "BF16_conversion": "exact CPU float32 values; original cache tensors unchanged"})
    except Exception as error:
        result.update(status="diagnostic_error", error=f"{type(error).__name__}: {error}")
    finally:
        split_cache = whole_cache = None
        result["seconds"] = time.monotonic() - started
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(output / "cache-state-audit.json", result)
    return result
