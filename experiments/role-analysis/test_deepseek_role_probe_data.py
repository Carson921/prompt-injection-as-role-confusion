"""CPU checks for native-role mapping and body-only span transfer."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import deepseek_role_probe_data as data


class CharacterTokenizer:
    def __call__(self, prompt, **kwargs):
        return {"input_ids": list(range(len(prompt))),
                "offset_mapping": [(i, i + 1) for i in range(len(prompt))]}


class NativeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (data.MODEL / "encoding/encoding_dsv4.py").exists():
            raise unittest.SkipTest("Official downloaded DeepSeek encoder is unavailable")
        cls.encoder = data.load_encoder(data.MODEL)

    def test_developer_maps_to_system_and_source_system_is_dropped(self):
        # The official encoder would render role developer exactly like user; that rendering is not used:
        # developer bodies go to the native system slot and are labeled system, GPT-OSS system is dropped.
        kwargs = dict(thinking_mode="thinking", drop_thinking=False, reasoning_effort="low")
        developer = self.encoder.encode_messages([{"role": "developer", "content": "Same ordinary body."}], **kwargs)
        user = self.encoder.encode_messages([{"role": "user", "content": "Same ordinary body."}], **kwargs)
        system = self.encoder.encode_messages([{"role": "system", "content": "Same ordinary body."}], **kwargs)
        self.assertEqual(developer, user)
        self.assertNotEqual(system, user)
        self.assertEqual(data.ROLE_MAPPING["developer"], "system")
        self.assertNotIn("system", data.ROLE_MAPPING)
        self.assertEqual(data.DROPPED_SOURCE_ROLES, ("system",))
        self.assertNotIn("developer", data.ROLES)

    def test_span_transfer_survives_tool_message_merging(self):
        messages = [{"role": "developer", "content": "Task instruction."},
                    {"role": "user", "content": "Find the answer."},
                    {"role": "assistant", "content": "", "reasoning_content": "Read the tool."},
                    {"role": "tool", "tool_call_id": "call1", "content": "First result."},
                    {"role": "tool", "tool_call_id": "call2", "content": "Second result."}]
        fields = [dict(message_index=2, field="reasoning_content", event_id="thinking", source_role="cot"),
                  dict(message_index=3, field="content", event_id="result1", source_role="tool"),
                  dict(message_index=4, field="content", event_id="result2", source_role="tool")]
        prompt, spans = data.render_with_spans(self.encoder, messages, fields)
        self.assertEqual([prompt[s["char_start"]:s["char_end"]] for s in spans],
                         ["Read the tool.", "First result.", "Second result."])
        self.assertIn("<tool_result>First result.</tool_result>", prompt)
        self.assertEqual(prompt, self.encoder.encode_messages(messages, thinking_mode="thinking", drop_thinking=False,
                                                             reasoning_effort="low", add_default_bos_token=True))
        self.assertNotIn("\ue000", prompt)

    def test_only_bodies_get_labels(self):
        messages = [{"role": "user", "content": "BODY"}]
        fields = [dict(message_index=0, field="content", event_id="event", source_role="user")]
        record = data.tokenize_record(CharacterTokenizer(), self.encoder, messages, fields,
                                     dict(id="example", source_cluster="article:one"), max_tokens=1000)
        self.assertEqual(len(record["targets"]), 4)
        self.assertEqual(record["labels"], [data.ROLES.index("user")] * 4)
        self.assertEqual(len(set(t["family_id"] for t in record["targets"])), 4)

    def test_does_not_truncate_long_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be truncated"):
            data.tokenize_record(CharacterTokenizer(), self.encoder, [{"role": "user", "content": "BODY"}],
                                 [dict(message_index=0, field="content", event_id="event", source_role="user")],
                                 dict(id="example", source_cluster="article:one"), max_tokens=2)

    def test_generated_tool_arguments_are_json_strings(self):
        source = {"role": "assistant", "content": None,
                  "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"n": 1}}}]}
        saved = copy.deepcopy(source)
        normalized = data.normalize_message(source)
        self.assertEqual(normalized["tool_calls"][0]["function"]["arguments"], '{"n": 1}')
        self.assertEqual(source, saved)

    def test_tool_event_identity_survives_reset_but_distinguishes_calls(self):
        first = {"role": "tool", "tool_call_id": "call1", "content": "Repeated result."}
        second = {**first, "tool_call_id": "call2"}
        event = data.body_event_id("banking/task1", first, "content", first["content"])
        self.assertEqual(event, data.body_event_id("banking/task1", copy.deepcopy(first), "content", first["content"]))
        self.assertNotEqual(event, data.body_event_id("banking/task1", second, "content", second["content"]))
        self.assertNotEqual(event, data.body_event_id("banking/task2", first, "content", first["content"]))

    def test_bpe_token_crossing_body_boundary_is_excluded(self):
        class CrossingTokenizer:
            def __call__(self, prompt, **kwargs):
                start = prompt.index("BODY")
                return {"input_ids": [1, 2], "offset_mapping": [(start - 1, start + 1), (start + 1, start + 4)]}
        record = data.tokenize_record(CrossingTokenizer(), self.encoder, [{"role": "user", "content": "BODY"}],
                                     [dict(message_index=0, field="content", event_id="event", source_role="user")],
                                     dict(id="example", source_cluster="article:one"), max_tokens=1000)
        self.assertEqual(record["positions"], [1])
        self.assertEqual(record["excluded_boundary_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
