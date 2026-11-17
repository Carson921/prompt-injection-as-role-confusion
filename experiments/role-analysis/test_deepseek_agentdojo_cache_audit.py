"""Small CPU fixtures for structural cache-state diagnostics; no model load."""
from types import SimpleNamespace
import unittest

import numpy as np

import deepseek_agentdojo_cache_audit as audit


def fixture(tokens):
    config = SimpleNamespace(num_hidden_layers=3, sliding_window=128, head_dim=4, index_head_dim=2,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128})
    layers = []
    for kind, name in zip(config.layer_types, ["DynamicSlidingWindowLayer", "DeepseekV4CSACache", "DeepseekV4HCACache"]):
        layer = type(name, (), {})()
        layer.get_seq_length = lambda: tokens
        layer.sliding_window = 128
        layer.keys = np.ones((1, 1, min(tokens, 127), 4), np.float32)
        layer.values = layer.keys
        if kind != "sliding_attention":
            rate = config.compress_rates[kind]
            count, remainder = divmod(tokens, rate)
            layer.compress_rate = rate
            layer.entry_count, layer.buffer_kv, layer.buffer_gate, layer.compressed_kv = {}, {}, {}, {}
            if kind == "compressed_sparse_attention":
                layer.overlap_kv, layer.overlap_gate = {}, {}
            for entry in ["compressor", "indexer"] if kind == "compressed_sparse_attention" else ["compressor"]:
                width = config.index_head_dim if entry == "indexer" else config.head_dim
                source_width = width * (2 if kind == "compressed_sparse_attention" else 1)
                layer.entry_count[entry] = count
                layer.buffer_kv[entry] = np.ones((1, remainder, source_width), np.float32)
                layer.buffer_gate[entry] = np.ones((1, remainder, source_width), np.float32)
                layer.compressed_kv[entry] = np.ones((1, count, width), np.float32)
                if kind == "compressed_sparse_attention":
                    value = np.ones((1, rate, width), np.float32) if count else None
                    layer.overlap_kv[entry] = value
                    layer.overlap_gate[entry] = value
        layers.append(layer)
    return SimpleNamespace(layers=layers, get_seq_length=lambda: tokens), config


class CacheAuditTests(unittest.TestCase):
    def test_native_counts_buffers_and_overlap_across_seven_eight(self):
        for tokens in [3, 7, 8, 127, 128, 129]:
            cache, config = fixture(tokens)
            samples = {}
            result, arrays = audit.snapshot_cache(cache, config, tokens, phase="test", samples=samples)
            self.assertTrue(result["invariants_passed"], [c for c in result["checks"] if not c["ok"]])
            self.assertTrue(all(len(value) <= 64 for value in samples.values()))
            self.assertEqual(arrays["layer_001/buffer_kv/compressor"].shape[1], tokens % 4)
            self.assertEqual(arrays["layer_002/compressed_kv/compressor"].shape[1], tokens // 128)

    def test_wrong_buffer_or_entry_count_never_passes_structural_check(self):
        cache, config = fixture(7)
        cache.layers[1].buffer_kv["compressor"] = np.ones((1, 2, 8), np.float32)
        cache.layers[2].entry_count["compressor"] = 1
        result, _ = audit.snapshot_cache(cache, config, 7, phase="test", samples={})
        self.assertFalse(result["invariants_passed"])
        self.assertEqual(sum(not c["ok"] for c in result["checks"]), 2)

    def test_value_difference_is_observed_without_an_acceptance_tolerance(self):
        whole = {"layer_000/keys": np.asarray([1, 2, 3], np.float32),
                 "layer_002/keys": np.asarray([4, 5], np.float32)}
        split = {"layer_000/keys": whole["layer_000/keys"].copy(),
                 "layer_002/keys": np.asarray([4, 7], np.float32)}
        result = audit.compare_cache_arrays(whole, split)
        self.assertFalse(result["whole_vs_split_values_bitexact"])
        self.assertEqual(result["first_layer_with_any_value_difference"], 2)
        self.assertTrue(result["floating_difference_is_not_an_acceptance_threshold"])
        row = next(r for r in result["tensors"] if r["tensor"] == "layer_002/keys")
        self.assertEqual(row["max_abs"], 2)
        self.assertEqual(row["different_elements"], 1)


if __name__ == "__main__":
    unittest.main()
