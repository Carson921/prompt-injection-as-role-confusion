"""CPU checks for exact system rotation, delimiter masks and causal head order."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("system_swap_collect", Path(__file__).with_name("collect.py"))
collection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection)


class FakeTokenizer:
    def __init__(self):
        self.mapping = {s: 100 + i for i, s in enumerate(collection.DELIMITERS)}
        self.mapping.update({"system": 1, "user": 2, "assistant": 3, "analysis": 4, "final": 5})

    def encode(self, text, **kwargs):
        return [self.mapping[text]]

    def decode(self, ids, **kwargs):
        reverse = {v: k for k, v in self.mapping.items()}
        return "".join(reverse.get(i, "content") for i in ids)


class CollectionTests(unittest.TestCase):
    def test_swap_is_exact_rotation_inside_system_only(self):
        before = "<|startoftext|><|start|>system<|message|>"
        a = "You are a model.\n\nReasoning: medium"
        b = "\n\n# Valid channels: analysis, final.\nCalls use commentary.\n"
        after = "<|end|><|start|>user<|message|>Reasoning: medium in user content.<|end|>"
        changed, proof = collection.rotate_system(before + a + b + after)
        self.assertEqual(changed, before + b + a + after)
        self.assertEqual(proof["original_system"], a + b)
        self.assertEqual(proof["swapped_system"], b + a)
        self.assertEqual(len(changed), len(before + a + b + after))

    def test_invalid_or_ambiguous_system_fails(self):
        for text in ["not native", "<|start|>system<|message|>no anchor<|end|>",
                     "<|start|>system<|message|>Reasoning: medium<|end|>",
                     "<|start|>system<|message|>Reasoning: medium Reasoning: medium suffix<|end|>"]:
            with self.assertRaises(RuntimeError):
                collection.rotate_system(text)

    def test_delimiters_removed_headers_retained_bodies_flagged(self):
        tok = FakeTokenizer()
        t = lambda s: tok.mapping[s]
        ids = np.array([t("<|startoftext|>"), t("<|start|>"), t("system"), t("<|message|>"), 21,
                        t("<|end|>"), t("<|start|>"), t("user"), t("<|message|>"), 22,
                        t("<|end|>"), t("<|start|>"), t("assistant"), t("<|channel|>"),
                        t("analysis"), t("<|message|>"), 23, t("<|end|>"), t("<|start|>"),
                        t("assistant"), t("<|channel|>"), t("final"), t("<|message|>"), 24, t("<|return|>")])
        keep, body, roles, delimiters = collection.position_metadata(tok, ids)
        self.assertEqual(np.flatnonzero(body).tolist(), [4, 9, 16, 23])
        self.assertEqual(roles[body].tolist(), ["system", "user", "cot", "assistant"])
        self.assertEqual(np.flatnonzero(keep).tolist(), [2, 4, 7, 9, 12, 14, 16, 19, 21, 23])
        self.assertFalse(np.isin(ids[keep], delimiters).any())

    def test_open_generation_header_carries_into_output(self):
        tok = FakeTokenizer()
        t = lambda s: tok.mapping[s]
        # Prompt ends at assistant; generated output begins with channel/header.
        ids = [t("<|start|>"), t("assistant"), t("<|channel|>"), t("analysis"),
               t("<|message|>"), 22, t("<|return|>")]
        keep, body, roles, _ = collection.position_metadata(tok, ids)
        self.assertEqual(np.flatnonzero(keep).tolist(), [1, 3, 5])
        self.assertEqual(np.flatnonzero(body).tolist(), [5])
        self.assertEqual(roles[5], "cot")

    def test_head_capture_preserves_selection_order_and_values(self):
        class Holder:
            pass
        model = Holder()
        model.model = Holder()
        model.model.layers = []
        for i in range(36):
            layer = Holder()
            layer.self_attn = Holder()
            layer.self_attn.o_proj = torch.nn.Identity()
            model.model.layers.append(layer)
        ids = [1166, 1069, 2296, 1161, 2093, 2088, 2295, 2045,
               2179, 2288, 1274, 897, 1305, 2091, 2094, 1930]
        result = torch.empty((2, 16, 64), dtype=torch.bfloat16)
        with collection.capture_heads(model, ids, [2, 0], result):
            for layer in sorted({h // 64 for h in ids}):
                value = torch.arange(3 * 4096, dtype=torch.float32).reshape(1, 3, 4096) + layer * 100000
                untouched = model.model.layers[layer].self_attn.o_proj(value)
                self.assertTrue(torch.equal(value, untouched))
        for slot, head in enumerate(ids):
            layer, within = divmod(head, 64)
            full = torch.arange(3 * 4096, dtype=torch.float32).reshape(3, 64, 64) + layer * 100000
            expected = full[[2, 0], within].to(torch.bfloat16)
            self.assertTrue(torch.equal(result[:, slot], expected))

    def test_prepare_and_saved_alignment_verification_all_57_tasks(self):
        model = Path("/tmp/wz55191-single-message-pre-mlp-gpt-oss-120b")
        if not model.exists():
            self.skipTest("Local bound checkpoint unavailable")
        from transformers import AutoTokenizer
        from safetensors.torch import save_file
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        prompt = ("<|start|>system<|message|>Model.\nReasoning: medium\n\n"
                  "# Valid channels: analysis, final.<|end|><|start|>user<|message|>Hi!<|end|>"
                  "<|start|>assistant")
        inp = tok.encode(prompt, add_special_tokens=False)
        gen = tok.encode("<|channel|>final<|message|>Hello.<|return|>", add_special_tokens=False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(source=root / "source", output=root / "out", model=model,
                                   selection=collection.DEFAULT_SELECTION, verify_weights=False)
            for suite, count in collection.EXPECTED.items():
                for i in range(count):
                    directory = args.source / suite / f"user_task_{i}" / "call_0000"
                    directory.mkdir(parents=True)
                    collection.atomic_json(directory / "request.json", {
                        "status": "complete", "rendered_prompt": prompt,
                        "input_tokens": len(inp), "generated_ids": gen})
                    if i % 2 == 0:
                        np.savez(directory / "input.npz", input_ids=np.array(inp))
            collection.prepare(args)
            manifest = collection.read(args.output / "manifest.json")
            self.assertEqual(manifest["tasks"], 57)
            self.assertEqual(len(manifest["calls"]), 57)
            validations = set()
            for row in manifest["calls"]:
                path = args.output / "prepared" / row["call_id"] / "request.json"
                request = collection.read(path)
                validations.add(request["original_input_verification"])
                self.assertEqual(request["generated_ids"], gen)
                all_ids = np.array(request["input_ids"] + gen)
                keep, body, _, _ = collection.position_metadata(tok, all_ids)
                positions = np.flatnonzero(keep)
                n = request["input_tokens"]
                saved = {"head_ids": torch.tensor(manifest["head_ids"])}
                for scope, indices, shift in (("input", positions[positions < n], 0),
                                               ("output", positions[positions >= n], n)):
                    saved[scope + "_features"] = torch.zeros((len(indices), 1024), dtype=torch.bfloat16)
                    saved[scope + "_token_ids"] = torch.from_numpy(all_ids[indices])
                    saved[scope + "_positions"] = torch.from_numpy(indices - shift)
                    saved[scope + "_body_mask"] = torch.from_numpy(body[indices])
                directory = args.output / "states" / row["call_id"]
                directory.mkdir(parents=True)
                save_file(saved, str(directory / "states.safetensors"))
                collection.atomic_json(directory / "receipt.json", {
                    "sha256": collection.digest(directory / "states.safetensors"),
                    "manifest_sha256": collection.digest(args.output / "manifest.json")})
            collection.verify(args)
            self.assertEqual(len(validations), 2)
            result = collection.read(args.output / "complete.json")
            self.assertTrue(result["delimiter_filter_and_token_alignment_verified"])
            # Rewrite a receipt-consistent file with an off-by-one output index.
            saved["output_positions"][0] += 1
            save_file(saved, str(directory / "states.safetensors"))
            collection.atomic_json(directory / "receipt.json", {
                "sha256": collection.digest(directory / "states.safetensors"),
                "manifest_sha256": collection.digest(args.output / "manifest.json")})
            with self.assertRaisesRegex(RuntimeError, "token alignment"):
                collection.verify(args)
            self.assertFalse((args.output / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
