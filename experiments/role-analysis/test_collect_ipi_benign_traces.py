"""CPU checks for retaining naturally invalid model output without tool calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_ipi_benign_traces as collector
import ipi_trace_interventions_data as data
from test_ipi_trace_interventions_data import ToyTokenizer


class InvalidNativeResponseTests(unittest.TestCase):
    def test_invalid_header_retains_every_raw_token_and_never_simulates(self):
        tok = ToyTokenizer()
        initial = tok.encode("<|start|>user<|message|>Request<|end|><|start|>assistant")
        generated = tok.encode(
            "<|channel|>analysis<|message|>Planning<|end|>"
            "<|start|>assistant<|channel|>analysis to=functions.browser"
            "<|channel|>commentary<|message|>{\"action\":\"read\"}<|call|>")

        class NativeTarget:
            @staticmethod
            def generate(model, ids, limit):
                self.assertEqual(ids, initial)
                return generated

        class NoSimulator:
            def simulate(self, *args):
                raise AssertionError("An invalid native response must never reach WorldSim")

        fixture = dict(case_id="opaque-test", trace_id="ipi/opaque-test", category="browser",
                       messages=[dict(role="user", content="Request")], tools=[],
                       adaptations=[], initial_input_ids=initial)
        config = dict(max_steps=5, max_new_tokens=4096, date="2026-11-17")
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared.json"
            prepared.write_text(json.dumps(fixture))
            path = Path(directory) / "record.json"
            with patch.object(collector.source, "render_input", return_value=dict(initial_input_ids=initial)):
                record, _ = collector.collect_case(prepared, path, None, tok, NativeTarget, config, NoSimulator())
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["terminal_reason"], "invalid_native_response")
            self.assertEqual(record["steps"][0]["worldsim"], [])
            self.assertNotIn("parsed_response", record["steps"][0])
            trace = record["trace"]
            self.assertEqual(trace["input_ids"], initial + generated)
            self.assertEqual(trace["opaque_terminal_start"], len(initial) - len(tok.encode("<|start|>assistant")))
            self.assertFalse(trace["allow_terminal_body"])
            blocks = data.parse_native_blocks(trace["input_ids"], tok,
                                              opaque_terminal_start=trace["opaque_terminal_start"])
            self.assertEqual(list(data.first_role_targets(blocks)), ["user"])
            self.assertEqual(json.loads(path.read_text()), record)


if __name__ == "__main__":
    unittest.main()
