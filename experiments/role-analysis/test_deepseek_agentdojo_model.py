"""CPU-only native encoding, hook and own-token alignment engineering tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from deepseek_agentdojo_model import (AdapterInfrastructureError, DeepSeekAgentDojoModel, ForwardCapture, OfficialCodec,
                                     ROLES, as_uint16_bf16, normalize_messages, object_sha)


class FakeCodec:
    binding = {"engineering_test_codec": True}
    eos_token_id = 1
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=[2]),
                                decode=lambda ids, skip_special_tokens=False: "test")

    def encode(self, messages, tools, *args):
        return {"prompt": "abc", "input_ids": [2, 3, 4], "offsets": [(0, 1), (1, 2), (2, 3)],
                "normalized_messages": normalize_messages(messages), "tools": tools}

    def decode(self, ids):
        return ",".join(map(str, ids))

    def parse(self, text, call_id, *args):
        return {"content": text, "reasoning_content": "actual reasoning", "tool_calls": []}


class FakeProbe:
    metadata = {"heads": [3, 0]}
    roles = ROLES
    feature_dim = 4

    def predict_proba(self, features):
        features = np.asarray(features, np.float32)
        value = np.column_stack((features, features.mean(1)))
        value = np.exp(value - value.max(1, keepdims=True))
        return value / value.sum(1, keepdims=True)


class FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, head_dim=2, num_hidden_layers=2)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(16, 2)
        self.model.layers = torch.nn.ModuleList()
        self.records = []
        for layer in range(2):
            module = torch.nn.Module()
            module.self_attn = torch.nn.Module()
            module.self_attn.o_a_proj = torch.nn.Identity()
            self.model.layers.append(module)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values=None, **kwargs):
        start = 0 if past_key_values is None else past_key_values.length
        assert position_ids.tolist() == [list(range(start, start + input_ids.shape[1]))]
        assert attention_mask.shape == (1, start + input_ids.shape[1])
        assert kwargs["use_cache"] is True and kwargs["logits_to_keep"] == 1
        self.records.append(input_ids.tolist()[0])
        for layer, module in enumerate(self.model.layers):
            value = input_ids.to(torch.bfloat16)[..., None, None].expand(-1, -1, 2, 2).clone()
            value += torch.tensor(layer * 8, dtype=torch.bfloat16)
            value += torch.tensor([[0, .125], [.25, .375]], dtype=torch.bfloat16)
            module.self_attn.o_a_proj(value)
        logits = torch.full((1, 1, 16), -100.)
        final = int(input_ids[0, -1])
        logits[0, 0, {4: 5, 5: 6, 6: 1}.get(final, 1)] = 100
        cache = SimpleNamespace(length=start + input_ids.shape[1])
        return SimpleNamespace(logits=logits, past_key_values=cache)


def fake_locator(*args, **kwargs):
    return {"input_token_indices": [2, 0], "marker_proof": {"engineering_test_only": True}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = FakeLM()
        self.adapter = DeepSeekAgentDojoModel(self.root, self.root, self.root / "run", _cpu_test=True,
                     _model=self.model, _probe=FakeProbe(), _codec=FakeCodec(), locator=fake_locator,
                     max_new_tokens=4, max_model_len=32)

    def tearDown(self):
        self.adapter.close()
        self.temp.cleanup()

    def request(self, call="case/0", scope=None):
        metadata = {"case_id": "case", "call_id": call, "condition": "plain", "actual_goal": "do test"}
        if scope is not None:
            metadata["recording_dir"] = str(self.root / "run" / scope / "calls" / call)
        return {"messages": [{"role": "user", "content": "abc"}], "tools": [], "metadata": metadata}

    def test_actual_sampled_tokens_have_own_states_eos_unobserved(self):
        request = self.request()
        response = self.adapter.generate_batch([request])[0]
        folder = self.root / "run" / response["artifact_refs"]["directory"]
        saved = np.load(folder / "output.npz")
        np.testing.assert_array_equal(saved["token_ids"], [5, 6, 1])
        np.testing.assert_array_equal(saved["observed"], [True, True, False])
        self.assertTrue(np.isnan(saved["probabilities"][-1]).all())
        self.assertEqual(self.model.records, [[2, 3, 4], [5], [6]])
        self.assertEqual(response["raw_request_sha256"], object_sha(request))
        self.assertEqual(response["reasoning_content"], "actual reasoning")
        raw = np.load(folder / "payload-heads-bf16-uint16.npy")
        self.assertEqual(raw.dtype, np.uint16)
        self.assertEqual(raw.shape, (2, 2, 2, 2))
        # Input order [2,0] is preserved, and selected order is global [3,0].
        reconstructed = torch.from_numpy(raw).view(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(reconstructed[:, 0, 0, 0], [4, 2])
        selected = np.load(folder / "payload-selected-fp16.npy")
        expected = reconstructed[:, [1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]]
        np.testing.assert_array_equal(selected, expected.astype(np.float16))
        inputs = np.load(folder / "input.npz")
        np.testing.assert_allclose(inputs["probabilities"][[2, 0]], FakeProbe().predict_proba(selected), atol=1e-7)
        # Generation token 5 features are computed by consuming 5, not from 4.
        own_features = np.array([[13.25, 13.375, 5, 5.125]], np.float16)
        np.testing.assert_allclose(saved["probabilities"][:1], FakeProbe().predict_proba(own_features), atol=1e-7)
        for layer in self.model.model.layers:
            self.assertFalse(layer.self_attn.o_a_proj._forward_pre_hooks)

    def test_first_capture_once_per_episode_and_durable_replay(self):
        first = self.adapter.generate_batch([self.request()])[0]
        forwards = len(self.model.records)
        replay = self.adapter.generate_batch([self.request()])[0]
        self.assertEqual(first, replay)
        self.assertEqual(len(self.model.records), forwards)
        second = self.adapter.generate_batch([self.request("case/1")])[0]
        self.assertFalse(second["full_payload_captured"])
        self.assertEqual(self.adapter.counters["full_payload_captures"], 1)

    def test_smoke_and_formal_scope_do_not_share_exposure_or_response(self):
        smoke = self.adapter.generate_batch([self.request(scope="smoke")])[0]
        formal = self.adapter.generate_batch([self.request()])[0]
        self.assertTrue(smoke["full_payload_captured"])
        self.assertTrue(formal["full_payload_captured"])
        self.assertNotEqual(smoke["artifact_refs"]["directory"], formal["artifact_refs"]["directory"])
        self.assertEqual(len(self.model.records), 6)
        self.assertEqual(len(self.adapter._exposure_seen), 2)

    def test_corrupt_committed_artifact_refuses_replay(self):
        response = self.adapter.generate_batch([self.request()])[0]
        folder = self.root / "run" / response["artifact_refs"]["directory"]
        (folder / "completion.txt").write_text("changed")
        with self.assertRaisesRegex(AdapterInfrastructureError, "Saved call file changed"):
            self.adapter.generate_batch([self.request()])

    def test_length_budget_consumes_terminal_normal_token(self):
        self.adapter.max_new_tokens = 1
        response = self.adapter.generate_batch([self.request()])[0]
        self.assertEqual(response["error"]["type"], "IncompleteNativeCompletion")
        self.assertEqual(self.model.records, [[2, 3, 4], [5]])
        saved = np.load(self.root / "run" / response["artifact_refs"]["directory"] / "output.npz")
        self.assertTrue(saved["observed"].all())

    def test_context_overflow_does_not_forward_and_commits_error(self):
        self.adapter.max_model_len = 6
        response = self.adapter.generate_batch([self.request()])[0]
        self.assertEqual(response["finish_reason"], "error")
        self.assertIn("no truncation", response["error"]["message"])
        self.assertFalse(self.model.records)
        self.assertFalse(response["full_payload_captured"])
        self.assertTrue((self.root / "run" / response["artifact_refs"]["receipt"]).is_file())

    def test_parser_failure_retains_all_probabilities(self):
        self.adapter.codec.parse = lambda *args: (_ for _ in ()).throw(ValueError("malformed DSML"))
        response = self.adapter.generate_batch([self.request()])[0]
        self.assertEqual(response["error"]["type"], "NativeCompletionParseError")
        folder = self.root / "run" / response["artifact_refs"]["directory"]
        self.assertEqual(np.load(folder / "output.npz")["probabilities"].shape, (3, 5))
        self.assertTrue(response["full_payload_captured"])
        self.assertEqual(json.loads((folder / "receipt.json").read_text())["status"], "error")

    def test_scope_traversal_rejected(self):
        for recording in (self.root / "outside/calls/case/0", self.root / "run/wrong/case/0"):
            request = self.request()
            request["metadata"]["recording_dir"] = str(recording)
            with self.assertRaises(AdapterInfrastructureError):
                self.adapter.generate_batch([request])

    def test_marker_failure_is_fatal_and_does_not_commit_or_claim_exposure(self):
        self.adapter.locator = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("marker proof differs"))
        with self.assertRaises(AdapterInfrastructureError) as caught:
            self.adapter.generate_batch([self.request()])
        response = caught.exception.response
        self.assertEqual(response["error"]["category"], "infrastructure")
        self.assertFalse((self.root / "run/calls/case/0").exists())
        self.assertTrue((self.root / "run" / response["artifact_refs"]["error"]).is_file())
        self.assertFalse(self.adapter._exposure_seen)

    def test_native_bits_and_invalid_dtype(self):
        values = torch.tensor([[-2.5, 0., 3.125]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(torch.from_numpy(as_uint16_bf16(values)).view(torch.bfloat16), values))
        with self.assertRaisesRegex(ValueError, "native BF16"):
            as_uint16_bf16(values.float())


class OfficialCodecTests(unittest.TestCase):
    model_dir = Path("/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731")

    @unittest.skipUnless((model_dir / "tokenizer.json").is_file(), "Official tokenizer not present")
    def test_official_render_locator_and_interleaved_reasoning(self):
        from deepseek_agentdojo_protocol import build_payload, locate_first_payload
        codec = OfficialCodec(self.model_dir)
        tools = [{"type": "function", "function": {"name": "check", "description": "Check",
                   "parameters": {"type": "object", "properties": {}}}}]
        for condition in ("plain", "plain_delimiter"):
            goal = "transfer 1 dollar"
            messages = [{"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Check account."},
                        {"role": "assistant", "content": "", "reasoning_content": "I should check.",
                         "tool_calls": [{"id": "old", "type": "function", "function": {"name": "check", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "old", "content": "result\n" + build_payload(condition, goal)}]
            original = copy.deepcopy(messages)
            encoded = codec.encode(messages, tools)
            self.assertEqual(messages, original)
            self.assertIn("I should check.", encoded["prompt"])
            payload = locate_first_payload(encoded["normalized_messages"], condition=condition,
                       actual_goal=goal, rendered_prompt=encoded["prompt"], offsets=encoded["offsets"],
                       input_ids=encoded["input_ids"], render_callback=lambda ms: codec.encode(ms, tools)["prompt"])
            self.assertEqual(payload["tool_message_index"], 3)
            self.assertTrue(payload["marker_proof"]["original_render_matches_actual"])
            self.assertEqual(len(payload["wrapper_token_indices"]), 3 if condition == "plain_delimiter" else 0)
        self.assertEqual(codec.special_tokens["<｜User｜>"], [128803])
        parsed = codec.parse("real reasoning</think>answer<｜end▁of▁sentence｜>", "call")
        self.assertEqual(parsed["content"], "answer")
        self.assertEqual(parsed["reasoning_content"], "real reasoning")


class RealTinyV4Tests(unittest.TestCase):
    def test_native_dynamic_cache_and_single_forward_hooks_on_cpu(self):
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, DynamicCache
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            config = DeepseekV4Config(vocab_size=256, hidden_size=256, moe_intermediate_size=256,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1, head_dim=128,
                q_lora_rank=256, o_groups=2, o_lora_rank=256, n_routed_experts=1,
                n_shared_experts=1, num_experts_per_tok=1,
                layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
                mlp_layer_types=["hash_moe", "moe", "moe"], index_n_heads=4, index_head_dim=32,
                index_topk=2, compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
                sliding_window=8, max_position_embeddings=64, qk_rope_head_dim=8, hc_mult=2,
                hc_sinkhorn_iters=4, use_cache=True)
            config._attn_implementation = "eager"
            config._experts_implementation = "eager"
            torch.manual_seed(0)
            model = DeepseekV4ForCausalLM(config).eval().requires_grad_(False).to(dtype=torch.bfloat16)
            ids = torch.arange(2, 14).unsqueeze(0)
            heads = [10, 0, 5]
            geometry = {"num_layers": 3, "num_heads": 4, "head_dim": 128}
            raw_layers = {}
            with torch.inference_mode():
                plain = model(input_ids=ids, use_cache=False, logits_to_keep=1)
                capture = ForwardCapture(model, heads, positions=[11, 0, 7], all_head_positions=[11, 0],
                    all_head_writer=lambda layer, bits: raw_layers.__setitem__(layer, bits.copy()), **geometry)
                with capture:
                    observed = model(input_ids=ids, use_cache=False, logits_to_keep=1)
                self.assertTrue(torch.equal(plain.logits, observed.logits))
                cache = DynamicCache(config=config)
                prefix = model(input_ids=ids[:, :-1], use_cache=True, past_key_values=cache, logits_to_keep=1)
                self.assertEqual(prefix.past_key_values.get_seq_length(), 11)
                cached_capture = ForwardCapture(model, heads, **geometry)
                with cached_capture:
                    output = model(input_ids=ids[:, -1:], attention_mask=torch.ones_like(ids),
                        position_ids=torch.tensor([[11]]), use_cache=True,
                        past_key_values=prefix.past_key_values, logits_to_keep=1)
            self.assertEqual(output.past_key_values.get_seq_length(), 12)
            self.assertEqual([type(x).__name__ for x in output.past_key_values.layers],
                             ["DynamicSlidingWindowLayer", "DeepseekV4CSACache", "DeepseekV4HCACache"])
            self.assertTrue(torch.equal(output.logits.argmax(-1), plain.logits.argmax(-1)))
            torch.testing.assert_close(output.logits, plain.logits, atol=.02, rtol=.03)
            np.testing.assert_allclose(cached_capture.features, capture.features[:1], atol=.02, rtol=.03)
            self.assertEqual(set(raw_layers), {0, 1, 2})
            self.assertTrue(all(value.dtype == np.uint16 and value.shape == (2, 4, 128)
                                for value in raw_layers.values()))
            self.assertFalse(any(layer.self_attn.o_a_proj._forward_pre_hooks for layer in model.model.layers))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
