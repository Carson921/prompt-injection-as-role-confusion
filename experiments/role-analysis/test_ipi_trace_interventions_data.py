"""CPU-only IPI design tests; synthetic bodies, real native structural IDs."""
from collections import Counter
import json
import re
import unittest

import ipi_trace_interventions_data as data


class ToyTokenizer:
    special = {"<|start|>": data.START, "<|message|>": data.MESSAGE,
               "<|channel|>": data.CHANNEL, "<|end|>": data.END,
               "<|return|>": data.RETURN, "<|call|>": data.CALL}
    inverse = {value: key for key, value in special.items()}

    def encode(self, text):
        pieces = re.split("(" + "|".join(re.escape(value) for value in self.special) + ")", text)
        return [token for piece in pieces if piece for token in
                ([self.special[piece]] if piece in self.special else [ord(char) + 1 for char in piece])]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inverse[token] if token in self.inverse else chr(token - 1) for token in ids)


class IPIInterventionTests(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()
        self.headers = dict(system="system", developer="developer", user="user",
                            cot="assistant<|channel|>analysis", assistant="assistant<|channel|>final",
                            tool="functions.native_tool")
        self.catalog = {role: dict(header_ids=[data.START] + self.tok.encode(header) + [data.MESSAGE],
                                   ending_ids=[data.RETURN if role == "assistant" else data.END],
                                   source_trace_id="frozen-pilot", source_block_index=index)
                        for index, (role, header) in enumerate(self.headers.items())}

    def block(self, role, body, *, ending=None, header=None):
        return ([data.START] + self.tok.encode(header or self.headers[role]) + [data.MESSAGE] + list(body)
                + ([data.RETURN if role == "assistant" else data.END] if ending is None else ending))

    def trace(self, roles=data.ROLES):
        return dict(trace_id="IPI-toy", input_ids=[token for index, role in enumerate(roles)
                    for token in self.block(role, [1001 + 10 * index, 1002 + 10 * index])])

    def assert_body_ownership(self, trace, rows):
        original = trace["input_ids"]
        for row in rows:
            self.assertEqual(len(row["input_ids"]), len(row["token_source_positions"]))
            for token, origin in zip(row["input_ids"], row["token_source_positions"]):
                if origin != -1:
                    self.assertEqual(token, original[origin])
            self.assertEqual([row["input_ids"][position] for position in row["target_positions"]],
                             [original[position] for position in row["target_original_positions"]])
            self.assertEqual([row["token_source_positions"][position] for position in row["target_positions"]],
                             row["target_original_positions"])
            if row["operation"] == "move":
                self.assertEqual(sorted(row["token_source_positions"]), list(range(len(original))))
        self.assertEqual(json.loads(json.dumps(rows)), rows)

    def test_all_six_sources_preserve_43_native_variants(self):
        trace = self.trace()
        rows = data.build_interventions(trace, self.tok, self.catalog)
        self.assertEqual(Counter(row["operation"] for row in rows),
                         dict(baseline=1, strip=6, swap=30, move=6))
        self.assertTrue(all(row["wrapper_provenance"]["kind"] == "same_trace"
                            for row in rows if row["operation"] == "swap"))
        self.assertTrue(all(value == "present" for value in rows[0]["role_availability"].values()))
        self.assert_body_ownership(trace, rows)

    def test_missing_destination_uses_exact_catalog_and_keeps_five_swaps(self):
        trace = self.trace(("system", "user", "tool"))
        rows = data.build_interventions(trace, self.tok, self.catalog)
        self.assertEqual(len(rows), 22)
        self.assertEqual(Counter(row["source_role"] for row in rows if row["operation"] == "swap"),
                         dict(system=5, user=5, tool=5))
        for row in (row for row in rows if row["operation"] == "swap"):
            provenance = row["wrapper_provenance"]
            if row["destination_role"] in ("cot", "assistant", "developer"):
                expected = self.catalog[row["destination_role"]]
                self.assertEqual(provenance["kind"], "pilot_catalog")
                self.assertEqual(provenance["source_trace_id"], "frozen-pilot")
                self.assertEqual(provenance["header_ids"], expected["header_ids"])
                self.assertEqual(provenance["ending_ids"], expected["ending_ids"])
                external = [token for token, origin in zip(row["input_ids"], row["token_source_positions"])
                            if origin == -1]
                self.assertEqual(external, expected["header_ids"] + expected["ending_ids"])
            else:
                self.assertEqual(provenance["kind"], "same_trace")
                self.assertNotIn(-1, row["token_source_positions"])
        self.assert_body_ownership(trace, rows)

    def test_empty_first_role_is_missing_but_its_real_wrapper_has_priority(self):
        ids = (self.block("user", []) + self.block("user", [1011])
               + self.block("system", [1021]) + self.block("assistant", [1031]))
        trace = dict(trace_id="empty-first", input_ids=ids)
        rows = data.build_interventions(trace, self.tok, self.catalog)
        self.assertEqual(len(rows), 15)
        self.assertEqual(rows[0]["role_availability"]["user"], "empty_first_body")
        self.assertNotIn("user", rows[0]["target_roles"])
        swap = next(row for row in rows if row["operation"] == "swap" and row["destination_role"] == "user")
        self.assertEqual(swap["wrapper_provenance"]["kind"], "same_trace")
        self.assertEqual(swap["wrapper_block_index"], 0)
        self.assert_body_ownership(trace, rows)

    def test_structural_tool_call_never_supplies_assistant_target(self):
        ids = (self.block("system", [1001]) + self.block("assistant", [1011], ending=[data.CALL],
                header="assistant to=functions.native_tool<|channel|>analysis"))
        trace = dict(trace_id="structural", input_ids=ids)
        blocks = data.parse_native_blocks(ids, self.tok)
        self.assertTrue(blocks[-1]["structural_tool_call"])
        self.assertIsNone(blocks[-1]["role"])
        rows = data.build_interventions(trace, self.tok, self.catalog)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["target_roles"], ["system"])
        self.assertEqual(rows[0]["role_availability"]["assistant"], "absent")
        self.assert_body_ownership(trace, rows)

    def test_authorized_terminal_cot_assistant_and_toolcall_preserve_observed_ids(self):
        for role, header in (("cot", None), ("assistant", None),
                             ("assistant", "assistant to=functions.x<|channel|>commentary")):
            with self.subTest(role=role, header=header):
                trace = dict(trace_id="length", input_ids=self.block("system", [1001]) +
                             self.block(role, [1011, 1012], ending=[], header=header))
                with self.assertRaisesRegex(ValueError, "Unterminated native body"):
                    data.parse_native_blocks(trace["input_ids"], self.tok)
                blocks = data.parse_native_blocks(trace["input_ids"], self.tok, allow_terminal_body=True)
                self.assertFalse(blocks[-1]["terminated"])
                self.assertEqual(blocks[-1]["ending_ids"], [])
                trace["allow_terminal_body"] = True
                with self.assertRaisesRegex(ValueError, "length.*provenance"):
                    data.build_interventions(trace, self.tok, self.catalog)
                trace["provenance"] = dict(eof_finish_reason="length")
                rows = data.build_interventions(trace, self.tok, self.catalog)
                self.assert_body_ownership(trace, rows)
                if header is None:
                    moved = next(row for row in rows if row["operation"] == "move" and row["source_role"] == role)
                    self.assertEqual(moved["applied_ending_ids"], [])
                    self.assertEqual(Counter(moved["input_ids"]), Counter(trace["input_ids"]))
                else:
                    self.assertNotIn("assistant", rows[0]["target_roles"])

    def test_malformed_or_unattributed_catalog_fails_closed(self):
        trace = self.trace(("user",))
        for changes in (dict(source_trace_id=""), dict(source_block_index=-1),
                        dict(ending_ids=[data.CALL]), dict(ending_ids=[]),
                        dict(header_ids=self.catalog["tool"]["header_ids"])):
            with self.subTest(changes=changes):
                catalog = {role: dict(value) for role, value in self.catalog.items()}
                catalog["system"].update(changes)
                with self.assertRaises(ValueError):
                    data.build_interventions(trace, self.tok, catalog)
        with self.assertRaisesRegex(ValueError, "Frozen catalog lacks"):
            data.build_interventions(trace, self.tok, {})

    def test_move_maximizes_displacement_among_every_native_boundary(self):
        trace = self.trace()
        frames = data.parse_native_blocks(trace["input_ids"], self.tok)
        rows = data.build_interventions(trace, self.tok, self.catalog)
        for row in (row for row in rows if row["operation"] == "move"):
            source = frames[row["block_index"]]
            cursor, boundaries = 0, [0]
            for frame in frames:
                if frame is not source:
                    cursor += frame["end"] - frame["start"]
                    boundaries.append(cursor)
            candidates = [(boundary + len(source["header_ids"]), boundary) for boundary in boundaries]
            expected, _ = min(candidates, key=lambda item: (-abs(item[0] - source["body_start"]), item[0]))
            self.assertEqual(row["new_body_start"], expected)

    def test_failed_final_response_is_opaque_context_never_a_target_or_wrapper(self):
        prefix = self.trace(("system", "developer", "user", "tool"))["input_ids"]
        # A valid-looking CoT before the malformed header remains in the same
        # rejected response and is conservatively left entirely unannotated.
        suffix = (self.block("cot", [7011, 7012]) + self.block("assistant", [7021], ending=[data.CALL],
                  header="assistant<|channel|>analysis to=functions.native_tool<|channel|>commentary"))
        trace = dict(trace_id="opaque", input_ids=prefix + suffix, opaque_terminal_start=len(prefix),
                     allow_terminal_body=False, eof_finish_reason="stop",
                     provenance=dict(terminal_reason="invalid_native_response", invalid_native_response=dict(
                         type="ValueError", message="Multiple native channels", step=1, opaque_terminal_start=len(prefix))))
        with self.assertRaisesRegex(ValueError, "Multiple native channels"):
            data.parse_native_blocks(trace["input_ids"], self.tok)
        blocks = data.parse_native_blocks(trace["input_ids"], self.tok, opaque_terminal_start=len(prefix))
        opaque = blocks[-1]
        self.assertEqual(opaque["opaque_ids"], suffix)
        self.assertFalse(opaque["complete"])
        self.assertIsNone(opaque["role"])
        self.assertTrue(opaque["structural_tool_call"])
        self.assertEqual(opaque["header_ids"] + opaque["body_ids"] + opaque["ending_ids"], [])
        self.assertEqual(set(data.first_role_targets(blocks)), {"system", "developer", "user", "tool"})
        rows = data.build_interventions(trace, self.tok, self.catalog)
        self.assertEqual(len(rows), 29)
        self.assertEqual(rows[0]["complete_blocks_end"], len(prefix))
        self.assertEqual(rows[0]["role_availability"]["cot"], "absent")
        for row in rows:
            self.assertEqual(row["input_ids"][row["opaque_terminal_start"]:], suffix)
            self.assertEqual(row["token_source_positions"][row["opaque_terminal_start"]:],
                             list(range(len(prefix), len(trace["input_ids"]))))
            self.assertLess(row["target_positions"][-1], row["opaque_terminal_start"])
            if row["operation"] == "swap" and row["destination_role"] in ("cot", "assistant"):
                self.assertEqual(row["wrapper_provenance"]["kind"], "pilot_catalog")
        self.assert_body_ownership(trace, rows)

    def test_opaque_offset_requires_a_closed_prefix_and_authoritative_error_provenance(self):
        trace = self.trace()
        original = trace["input_ids"]
        suffix = [data.START] + self.tok.encode("assistant") + [data.MESSAGE, 1701, data.RETURN]
        trace["input_ids"] = original + suffix
        start = len(original)
        for offset in (-1, True, start + 1, len(trace["input_ids"]), start - 1):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                data.parse_native_blocks(trace["input_ids"], self.tok, opaque_terminal_start=offset)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            data.parse_native_blocks(trace["input_ids"], self.tok, opaque_terminal_start=start, allow_terminal_body=True)
        trace["opaque_terminal_start"] = start
        with self.assertRaisesRegex(ValueError, "error provenance"):
            data.build_interventions(trace, self.tok, self.catalog)
        # A delimiter inside an original body cannot become an authoritative
        # boundary by merely attaching opaque metadata.
        broken = self.block("user", [1701])[:-1] + suffix
        with self.assertRaisesRegex(ValueError, "Unterminated native body"):
            data.parse_native_blocks(broken, self.tok, opaque_terminal_start=len(broken) - len(suffix))


if __name__ == "__main__":
    unittest.main()
