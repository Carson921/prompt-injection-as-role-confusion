"""CPU tests of official HF .generate versus the real adapter cached loop."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from deepseek_agentdojo_generate_diagnostics import run_gpu_diagnostics
from deepseek_agentdojo_model import DeepSeekAgentDojoModel, ROLES
from test_deepseek_agentdojo_model import FakeCodec, FakeProbe


class TinyProbe:
    metadata = {"heads": [10, 0, 5]}
    roles = ROLES
    feature_dim = 384

    def predict_proba(self, features):
        return FakeProbe().predict_proba(np.asarray(features)[:, :4])


def tiny_model():
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    config = DeepseekV4Config(vocab_size=64, hidden_size=256, moe_intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1, head_dim=128,
        q_lora_rank=256, o_groups=2, o_lora_rank=256, n_routed_experts=1, n_shared_experts=1,
        num_experts_per_tok=1, layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"], index_n_heads=4, index_head_dim=32,
        index_topk=2, compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=8, max_position_embeddings=64, qk_rope_head_dim=8, hc_mult=2,
        hc_sinkhorn_iters=4, use_cache=False, bos_token_id=0, eos_token_id=1)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(0)
    return DeepseekV4ForCausalLM(config).eval().requires_grad_(False).to(dtype=torch.bfloat16)


class OfficialGenerationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = tiny_model()
        self.adapter = DeepSeekAgentDojoModel(self.root, self.root, self.root / "adapter", _cpu_test=True,
            _model=self.model, _probe=TinyProbe(), _codec=FakeCodec(), max_new_tokens=4, max_model_len=64)

    def tearDown(self):
        self.adapter.close()
        self.temporary.cleanup()
        torch.set_num_threads(self.old_threads)

    def test_actual_hf_generate_matches_cached_adapter_and_keeps_protocol(self):
        initial = self.model.generation_config.to_dict().copy()
        result = run_gpu_diagnostics(self.adapter, self.root / "diagnostics", token_ids=list(range(2, 10)), steps=4)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["generated_ids_equal"])
        self.assertTrue(result["sampling_logits_bitexact"])
        self.assertTrue(result["common_observed_own_features_bitexact"])
        self.assertTrue(result["common_observed_own_probabilities_bitexact"])
        self.assertEqual(self.model.generation_config.to_dict(), initial)
        self.assertEqual(self.adapter.max_new_tokens, 4)
        with np.load(self.root / "diagnostics/generation-diagnostic.npz") as saved:
            np.testing.assert_array_equal(saved["hf_generated_ids"], saved["adapter_generated_ids"])
            np.testing.assert_array_equal(saved["hf_sampling_logits"], saved["adapter_sampling_logits"])
            if saved["adapter_generated_ids"][-1] != self.adapter.codec.eos_token_id:
                self.assertTrue(saved["adapter_own_observed"][-1])
                self.assertFalse(saved["hf_own_observed"][-1])
                self.assertEqual(result["terminal_reference"]["status"], "passed")
                self.assertTrue(result["terminal_reference"]["features"]["bitexact"])
                self.assertTrue(result["terminal_reference"]["probabilities"]["bitexact"])
                np.testing.assert_array_equal(saved["terminal_reference_features"], saved["adapter_own_features"][-1:])
                np.testing.assert_array_equal(saved["terminal_reference_probabilities"], saved["adapter_own_probabilities"][-1:])
                np.testing.assert_array_equal(saved["terminal_reference_token_ids"], saved["adapter_generated_ids"][-1:])
                self.assertEqual(saved["terminal_reference_positions"].tolist(), [8 + len(saved["adapter_generated_ids"]) - 1])
            self.assertTrue(np.isnan(saved["hf_own_probabilities"][~saved["hf_own_observed"]]).all())
            self.assertTrue(np.isfinite(saved["adapter_own_probabilities"][saved["adapter_own_observed"]]).all())
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(self.model._forward_hooks)
        self.assertFalse(any(layer.self_attn.o_a_proj._forward_pre_hooks for layer in self.model.model.layers))
        self.assertFalse(list((self.root / "adapter/calls").rglob("receipt.json")))

    def test_actual_eos_is_unforwarded_in_both_paths(self):
        self.model.lm_head.weight.zero_()
        self.adapter.codec.eos_token_id = 0  # Every zero-logit argmax is this explicit test EOS.
        result = run_gpu_diagnostics(self.adapter, self.root / "eos", token_ids=list(range(2, 10)), steps=4)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["hf_generated_tokens"], 1)
        self.assertFalse(result["steps"][0]["hf_own_state_observed"])
        self.assertFalse(result["steps"][0]["adapter_own_state_observed"])
        self.assertEqual(result["terminal_reference"]["status"], "eos_unforwarded")
        self.assertFalse(result["terminal_reference"]["eos_forwarded"])
        self.assertEqual(result["terminal_reference"]["cache_length_before"], 8)
        self.assertEqual(result["terminal_reference"]["cache_length_after"], 8)
        with np.load(self.root / "eos/generation-diagnostic.npz") as saved:
            self.assertTrue(np.isnan(saved["adapter_own_features"]).all())
            self.assertTrue(np.isnan(saved["hf_own_features"]).all())
            self.assertEqual(saved["terminal_reference_features"].shape, (0, 384))
            self.assertEqual(saved["terminal_reference_probabilities"].shape, (0, 5))
            self.assertEqual(saved["terminal_reference_token_ids"].shape, (0,))

    def test_failed_real_forward_removes_all_observational_hooks(self):
        handle = self.model.model.layers[0].self_attn.q_a_proj.register_forward_pre_hook(
            lambda *args: (_ for _ in ()).throw(RuntimeError("injected test forward failure")))
        try:
            with self.assertRaisesRegex(RuntimeError, "injected test forward failure"):
                run_gpu_diagnostics(self.adapter, self.root / "failure", token_ids=list(range(2, 10)), steps=4)
        finally:
            handle.remove()
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(self.model._forward_hooks)
        self.assertFalse(any(layer.self_attn.o_a_proj._forward_pre_hooks for layer in self.model.model.layers))
        self.assertTrue((self.root / "failure/generation-diagnostic.json").is_file())


if __name__ == "__main__":
    unittest.main()
