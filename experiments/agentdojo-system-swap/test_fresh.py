"""CPU regression checks; fixtures never write actual model activation outputs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fresh_capture", HERE / "run_fresh.py")
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)
torch.set_num_threads(2)


class NativeDecodeTests(unittest.TestCase):
    def test_bounded_attention_matches_native_equation_for_full_and_sliding_cache(self):
        from transformers.models.gpt_oss import modeling_gpt_oss as implementation
        original = implementation.eager_attention_forward
        torch.manual_seed(17)
        try:
            fresh.install_tiled_attention(6)
            for window in (None, 8):
                module = SimpleNamespace(num_key_value_groups=2, training=False,
                                         sliding_window=window, sinks=torch.randn(4, dtype=torch.bfloat16))
                query = torch.randn(1, 4, 9, 8, dtype=torch.bfloat16)
                key = torch.randn(1, 2, 17, 8, dtype=torch.bfloat16)
                value = torch.randn(1, 2, 17, 8, dtype=torch.bfloat16)
                q, k = torch.arange(9)[:, None] + 8, torch.arange(17)[None, :]
                allowed = k <= q
                if window is not None:
                    allowed &= k > q - window
                mask = torch.zeros(9, 17, dtype=query.dtype).masked_fill(~allowed, torch.finfo(query.dtype).min)
                expected = original(module, query, key, value, mask, scaling=8**-.5)[0]
                actual = implementation.eager_attention_forward(module, query, key, value, None, scaling=8**-.5)[0]
                self.assertTrue(torch.equal(expected, actual), float((expected.float()-actual.float()).abs().max()))
        finally:
            implementation.eager_attention_forward = original

    def test_emitted_tokens_get_their_own_forward_states(self):
        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                for layer in range(36):
                    block = torch.nn.Module()
                    block.self_attn = torch.nn.Module()
                    block.self_attn.o_proj = torch.nn.Identity()
                    self.layers.append(block)

            def forward(self, input_ids, past_key_values=None, **kwargs):
                for layer, block in enumerate(self.layers):
                    value = input_ids.float().unsqueeze(-1) + layer + torch.arange(4096).view(1, 1, -1) / 4096
                    block.self_attn.o_proj(value.to(torch.bfloat16))
                return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1),
                                       past_key_values=(past_key_values or 0) + input_ids.shape[1])

        class Model:
            def __init__(self):
                self.model = Core()

            def lm_head(self, hidden):
                last = int(hidden[0, -1, 0])
                target = {3: 4, 4: 5, 5: 200002}[last]
                logits = torch.full((1, 1, 200003), -100.)
                logits[0, 0, target] = 100.
                return logits

        original_tensor, original_empty = torch.tensor, torch.empty
        def cpu_tensor(*a, **kw):
            if kw.get("device") == "cuda:0":
                kw["device"] = "cpu"
            return original_tensor(*a, **kw)
        def cpu_empty(*a, **kw):
            if kw.get("device") == "cuda:0":
                kw["device"] = "cpu"
            return original_empty(*a, **kw)
        heads = [1166, 1069, 2296, 1161, 2093, 2088, 2295, 2045,
                 2179, 2288, 1274, 897, 1305, 2091, 2094, 1930]
        with patch.object(torch, "tensor", cpu_tensor), patch.object(torch, "empty", cpu_empty):
            generated, inp, out = fresh.generate(Model(), [1, 2, 3], heads, [200002], 2, 8)
        self.assertEqual(generated, [4, 5, 200002])
        self.assertEqual(inp.shape, (3, 1024))
        self.assertEqual(out.shape, (2, 1024))
        for slot, head in enumerate(heads):
            layer, within = divmod(head, 64)
            for index, token in enumerate([4, 5]):
                expected = (torch.arange(within * 64, (within + 1) * 64) / 4096 + layer + token).to(torch.bfloat16)
                self.assertTrue(torch.equal(out[index, slot * 64:(slot + 1) * 64], expected))
        self.assertFalse(torch.equal(inp[-1], out[0]))


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        model = Path('/tmp/wz55191-single-message-pre-mlp-gpt-oss-120b')
        if not model.exists():
            raise unittest.SkipTest("Bound local tokenizer unavailable")
        cls.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def test_tool_and_final_frames_parse_without_folding_analysis_into_content(self):
        text = ('<|channel|>analysis<|message|>Need the file.<|end|>'
                '<|start|>assistant to=functions.read_file<|channel|>commentary json<|message|>'
                '{"file_path":"bill-december-2023.txt"}<|call|>')
        response = fresh.parse_response(self.tok, self.tok.encode(text, add_special_tokens=False), "fixture/call_0")
        self.assertEqual(response["tool_calls"][0]["name"], "read_file")
        self.assertEqual(response["tool_calls"][0]["arguments"], {"file_path": "bill-december-2023.txt"})
        self.assertIsNone(response["content"])
        text = '<|channel|>analysis<|message|>Check.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>'
        response = fresh.parse_response(self.tok, self.tok.encode(text, add_special_tokens=False), "fixture/call_1")
        self.assertEqual(response["content"], "Done.")

    def test_real_requests_keep_developer_user_and_tools_outside_system_unchanged(self):
        output = HERE / "outputs/gpt-oss-120b-system-swap-fresh57-20261102"
        manifest = fresh.base.read(output / "manifest.json")
        for row in manifest["initial_requests"]:
            directory = output / "initial_requests" / row["task_key"]
            request = fresh.base.read(directory / "request.json")
            rendered = fresh.render(self.tok, request)
            self.assertEqual(rendered, fresh.base.read(directory / "rendered.json"))
            start, end = (rendered["transformation"][k] for k in ("system_char_start", "system_char_end"))
            self.assertEqual(rendered["original_prompt"][:start], rendered["rendered_prompt"][:start])
            self.assertEqual(rendered["original_prompt"][end:], rendered["rendered_prompt"][end:])
            self.assertEqual(request["injections"], {})


class BridgeTests(unittest.TestCase):
    def test_real_agentdojo_tools_roundtrip_via_json_pipe(self):
        output = HERE / "outputs/gpt-oss-120b-system-swap-fresh57-20261102"
        python = Path('/scratch/wz55191/venvs/chatinject-table1/bin/python')
        if not python.exists():
            self.skipTest("Existing AgentDojo runtime unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(output / "agentdojo-source.json", root / "agentdojo-source.json")
            with (root / "bridge.log").open("w") as errors:
                process = subprocess.Popen([str(python), str(HERE / 'agentdojo_bridge.py'), 'run', '--output', str(root)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True)
                completed, actual_tool_reply = set(), False
                try:
                    for line in process.stdout:
                        event = json.loads(line)
                        if event["event"] == "request":
                            request = fresh.base.read(event["path"])
                            response = {"call_id": event["call_id"], "content": "CPU bridge fixture.", "tool_calls": []}
                            if event["call_id"] == "no_payload/banking/user_task_0/call_0000":
                                response.update(content=None, tool_calls=[{"name": "read_file", "arguments": {"file_path": "bill-december-2023.txt"}, "id": "fixture_read"}])
                            elif event["call_id"] == "no_payload/banking/user_task_0/call_0001":
                                message = request["messages"][-1]
                                self.assertEqual(message["role"], "tool")
                                self.assertIn("Bill for the month of December 2023", message["content"])
                                actual_tool_reply = True
                            process.stdin.write(json.dumps(response) + '\n')
                            process.stdin.flush()
                        elif event["event"] == "task_complete":
                            completed.add(event["task_key"])
                        elif event["event"] == "task_error":
                            self.fail(str(event))
                    self.assertEqual(process.wait(timeout=30), 0, (root / "bridge.log").read_text())
                    self.assertEqual(len(completed), 57)
                    self.assertTrue(actual_tool_reply)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
