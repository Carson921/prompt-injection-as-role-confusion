"""Tiny official-V4 CPU checks of the resident diagnostic's actual call contract."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import deepseek_agentdojo_kernel_row_diagnostics as diagnostic


class NativeTinyDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
        torch.set_num_threads(2)
        torch.manual_seed(104)
        config = DeepseekV4Config(vocab_size=64, hidden_size=64, moe_intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=1,
            head_dim=32, q_lora_rank=64, o_groups=2, o_lora_rank=64,
            n_routed_experts=2, n_shared_experts=1, num_experts_per_tok=2,
            layer_types=["sliding_attention"], mlp_layer_types=["hash_moe"],
            index_n_heads=4, index_head_dim=16, index_topk=2,
            compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
            sliding_window=8, max_position_embeddings=128, qk_rope_head_dim=8,
            hc_mult=2, hc_sinkhorn_iters=4, use_cache=True)
        config._attn_implementation = "eager"
        cls.model = DeepseekV4ForCausalLM(config).eval().requires_grad_(False)
        cls.adapter = SimpleNamespace(model=cls.model, prepare_query_length=lambda _: None,
                                      tokenizer=SimpleNamespace(encode=lambda *a, **k: list(range(4))))

    def test_native_layer_calls_router_and_exact_same_input_row(self):
        with tempfile.TemporaryDirectory() as directory:
            result = diagnostic.run_gpu_diagnostics(self.adapter, directory, _cpu_test=True)
            self.assertEqual(result["status"], "diagnostic_only")
            report = json.loads(Path(result["results_path"]).read_text())
            self.assertTrue(report["complete"] and report["all_outputs_finite"])
            self.assertTrue(report["model_storage_and_versions_unchanged"])
            self.assertEqual(report["precision_before"], report["precision_after"])
            operations = {item["name"]: item for item in report["operations"]}
            self.assertIn("attention/q_a_proj", operations)
            self.assertIn("attn_hc/native_fp32_linear", operations)
            self.assertIn("mlp/fp4_experts_fixed_routing_and_weights", operations)
            self.assertTrue(report["routing"]["last_row_ids_order_equal"])
            for op in operations.values():
                self.assertTrue(op["single_input_bitexact"])
                self.assertEqual(op["input_shape"][1], 8)
                self.assertEqual(op["single_input_shape"][1], 1)
                for value in op["outputs"].values():
                    self.assertTrue(value["m8_repeat"]["bitexact"])
                    self.assertTrue(value["m1_repeat"]["bitexact"])
            self.assertFalse(any(module._forward_pre_hooks for module in self.model.modules()))

    def test_shape_dependent_native_operation_is_detected_without_mutation(self):
        import torch
        value = torch.randn(1, 8, 11)
        def operation(x):
            return x + (1 if x.shape[1] == 1 else 0)
        before = value.clone()
        result = diagnostic._measure("fixture", value, operation)
        metric = result["outputs"]["output"]
        self.assertFalse(metric["m8_vs_m1"]["bitexact"])
        self.assertAlmostEqual(metric["m8_vs_m1"]["max_abs"], 1, places=6)
        self.assertTrue(metric["m8_repeat"]["bitexact"] and metric["m1_repeat"]["bitexact"])
        self.assertTrue(torch.equal(value, before))

    def test_cpu_requires_explicit_engineering_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "already loaded native CUDA"):
                diagnostic.run_gpu_diagnostics(self.adapter, directory)

    def test_packed_fp4_projection_row_and_expert_contract(self):
        import copy
        import torch
        from unittest.mock import patch
        model = copy.deepcopy(self.model)
        class PackedFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block_size = [128, 128]
                for name, shape in (("gate_up_proj", (2, 128, 32)), ("down_proj", (2, 64, 32))):
                    self.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.int8), requires_grad=False))
                    self.register_parameter(name + "_scale_inv", torch.nn.Parameter(
                        torch.zeros(shape[0], shape[1], 2, dtype=torch.uint8), requires_grad=False))

            def forward(self, x, indices, weights):
                return torch.zeros_like(x)

            def _apply_gate(self, x):
                gate, up = x.chunk(2, dim=-1)
                return torch.nn.functional.silu(gate) * up
        model.model.layers[0].mlp.experts = PackedFixture().eval()
        calls = []
        def native_contract(a, weight, scale, *, block_size, expert_ids):
            self.assertEqual(a.shape[0], expert_ids.numel())
            self.assertEqual(a.shape[1], weight.shape[-1] * 2)
            self.assertEqual(tuple(scale.shape), (weight.shape[0], weight.shape[1], a.shape[1] // 32))
            self.assertEqual(block_size, [128, 128])
            self.assertEqual(expert_ids.dtype, torch.int64)
            calls.append((a.shape[0], weight.shape[1], a.detach().clone(), expert_ids.detach().clone()))
            return a.new_zeros(a.shape[0], weight.shape[1])
        adapter = SimpleNamespace(model=model, tokenizer=self.adapter.tokenizer, prepare_query_length=lambda _: None)
        with tempfile.TemporaryDirectory() as folder, patch(
                "transformers.integrations.finegrained_fp8.load_finegrained_fp8_kernel",
                return_value=SimpleNamespace(batched_matmul=native_contract)):
            result = diagnostic.run_gpu_diagnostics(adapter, folder, _cpu_test=True)
            report = json.loads(Path(result["results_path"]).read_text())
            self.assertTrue(report["native_packed_fp4"])
            names = {op["name"] for op in report["operations"]}
            self.assertIn("mlp/fp4_gate_up_projection_fixed_experts", names)
            self.assertIn("mlp/fp4_down_projection_identical_activated_input_fixed_experts", names)
            self.assertEqual({call[0] for call in calls}, {2, 16})
            for rows, n in ((16, 128), (16, 64)):
                many = next(call for call in calls if call[:2] == (rows, n))
                one = next(call for call in calls if call[:2] == (2, n))
                self.assertTrue(torch.equal(many[2][-2:], one[2]))
                self.assertTrue(torch.equal(many[3][-2:], one[3]))

    def test_layer_three_hc_capture_matches_actual_complete_native_model(self):
        import numpy as np
        import torch
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
        config = DeepseekV4Config.from_dict(self.model.config.to_dict())
        config.num_hidden_layers = 4
        config.layer_types = ["sliding_attention", "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
        config.mlp_layer_types = ["hash_moe", "hash_moe", "moe", "moe"]
        config._attn_implementation = "eager"
        model = DeepseekV4ForCausalLM(config).eval().requires_grad_(False)
        adapter = SimpleNamespace(model=model, tokenizer=self.adapter.tokenizer, prepare_query_length=lambda _: None)
        with tempfile.TemporaryDirectory() as folder:
            result = diagnostic.run_hc_layer_diagnostics(adapter, folder, _cpu_test=True)
            report = json.loads(Path(result["results_path"]).read_text())
            self.assertEqual(report["layer"], 3)
            self.assertFalse(report["cache_used"])
            self.assertTrue(report["all_outputs_finite"] and report["model_storage_and_versions_unchanged"])
            self.assertEqual(len(report["operations"]), 10)
            self.assertTrue(all(item["bitexact"] for row in report["native_full_sequence_direct_replay"].values()
                                for item in row.values()))
            captured = {}
            handles = []
            for name in ("attn_hc", "ffn_hc"):
                def before(module, args, name=name):
                    captured[name] = args[0].detach().clone()
                handles.append(getattr(model.model.layers[3], name).register_forward_pre_hook(before))
            try:
                ids = torch.arange(8)[None]
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            with np.load(report["native_byte_archive"]["path"], allow_pickle=False) as saved:
                for name in ("attn_hc", "ffn_hc"):
                    self.assertTrue(np.array_equal(saved[name+"__captured_input"], diagnostic._bits(captured[name])))
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in model.modules()))

    def test_actual_three_forward_hc_path_causal_contract_and_no_false_claim(self):
        import torch
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
        config = DeepseekV4Config.from_dict(self.model.config.to_dict())
        config.num_hidden_layers = 4
        config.layer_types = ["sliding_attention", "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
        config.mlp_layer_types = ["hash_moe", "hash_moe", "moe", "moe"]
        config._attn_implementation = "eager"
        model = DeepseekV4ForCausalLM(config).eval().requires_grad_(False)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.model.layers[3].post_attention_layernorm.weight.fill_(1)
        adapter = SimpleNamespace(model=model, tokenizer=self.adapter.tokenizer, prepare_query_length=lambda _: None)
        for shape_effect in (False, True):
            if shape_effect:
                class KnownShapeEffect(torch.nn.Module):
                    def __init__(self, native):
                        super().__init__()
                        self.native = native

                    def forward(self, x):
                        post, comb, collapsed = self.native(x)
                        return post, comb, collapsed + (0.001 if x.shape[1] == 1 else 0)
                model.model.layers[3].ffn_hc = KnownShapeEffect(model.model.layers[3].ffn_hc).eval()
            with tempfile.TemporaryDirectory() as folder:
                result = diagnostic.run_hc_path_diagnostics(adapter, folder, _cpu_test=True)
                report = json.loads(Path(result["results_path"]).read_text())
                self.assertEqual(report["complete_native_forward_calls"], 3)
                self.assertEqual([row["query_tokens"] for row in report["calls"]], [8, 7, 1])
                self.assertEqual([row["cache_consumed_tokens"] for row in report["calls"]], [None, 7, 8])
                self.assertTrue(all(report["required_causal_checks"].values()))
                self.assertEqual(report["actual_norm_difference_attributed_to_same_input_shape_effect"], shape_effect)
                self.assertTrue(report["model_storage_and_versions_unchanged"])
                self.assertEqual(report["actual_changed_norm_coordinates"], report["standalone_changed_norm_coordinates"])
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
