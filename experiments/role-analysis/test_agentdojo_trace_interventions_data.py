"""CPU-only ownership and experimental-design checks using synthetic token IDs."""
from collections import Counter
import json
import re
import unittest

import agentdojo_trace_interventions_data as data


class ToyTokenizer:
    """Character tokenizer with the real six structural IDs; no model assets."""
    special = {"<|start|>": data.START, "<|message|>": data.MESSAGE,
               "<|channel|>": data.CHANNEL, "<|end|>": data.END,
               "<|return|>": data.RETURN, "<|call|>": data.CALL}
    inverse = {value: key for key, value in special.items()}

    def encode(self, text):
        parts = re.split("(" + "|".join(re.escape(part) for part in self.special) + ")", text)
        return [value for part in parts if part for value in
                ([self.special[part]] if part in self.special else [ord(character) + 1 for character in part])]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inverse[value] if value in self.inverse else chr(value - 1) for value in ids)


class NativeInterventionTests(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()

    def block(self, header, body, ending=data.END):
        return [data.START] + self.tok.encode(header) + [data.MESSAGE] + list(body) + [ending]

    def full_trace(self, *, open_header=True):
        blocks = [
            self.block("system", [1011, 1012, 1013]),
            self.block("developer", [1021] * 39),
            self.block("user", [1031, 1032]),
            self.block("assistant<|channel|>analysis", [1041, 1042, 1043, 1044]),
            self.block("assistant to=functions.special_native_name<|channel|>commentary json",
                       [1051, 1052], data.CALL),
            self.block("functions.special_native_name to=assistant<|channel|>commentary", [1061, 1062, 1063]),
            self.block("assistant<|channel|>final", [1071, 1072], data.RETURN),
            self.block("user", [1081, 1082, 1083, 1084]),
            self.block("assistant<|channel|>final", [1091], data.RETURN),
        ]
        ids = [value for block in blocks for value in block]
        if open_header:
            ids += [data.START] + self.tok.encode("assistant")
        return {"trace_id": "banking/user_task_toy", "input_ids": ids}

    def test_real_structural_ids_parse_roles_and_exclude_tool_call(self):
        trace = self.full_trace()
        blocks = data.parse_native_blocks(trace["input_ids"], self.tok)
        self.assertEqual([block["role"] for block in blocks],
                         ["system", "developer", "user", "cot", None, "tool", "assistant", "user", "assistant", "assistant"])
        self.assertTrue(blocks[4]["structural_tool_call"])
        self.assertEqual(blocks[4]["recipient"], "functions.special_native_name")
        self.assertEqual(blocks[5]["recipient"], "assistant")
        self.assertFalse(blocks[-1]["complete"])
        targets = data.first_role_targets(blocks)
        self.assertEqual(targets["assistant"]["block_index"], 6)
        self.assertEqual(targets["user"]["block_index"], 2)
        self.assertEqual([value for block in blocks for name in ("header_ids", "body_ids", "ending_ids")
                          for value in block[name]], trace["input_ids"])

    def test_tool_recipient_excludes_output_constraint_header(self):
        ids = self.block("assistant to=functions.get_users_in_channel<|constrain|>json", [401], data.CALL)
        block = data.parse_native_blocks(ids, self.tok)[0]
        self.assertEqual(block["recipient"], "functions.get_users_in_channel")
        self.assertTrue(block["structural_tool_call"])
        self.assertIsNone(block["role"])

    def test_complete_43_variant_grid_and_original_token_ownership(self):
        trace = self.full_trace()
        ids = trace["input_ids"]
        rows = data.build_interventions(trace, self.tok)
        self.assertEqual(Counter(row["operation"] for row in rows),
                         {"baseline": 1, "strip": 6, "swap": 30, "move": 6})
        self.assertEqual(len({row["input_id"] for row in rows}), 43)
        self.assertEqual(rows[0]["input_ids"], ids)
        self.assertEqual(len(rows[0]["role_target_positions"]), 6)
        self.assertEqual(rows[0]["target_positions"], sorted(set(rows[0]["target_positions"])))
        targets = data.first_role_targets(data.parse_native_blocks(ids, self.tok))
        for row in rows:
            with self.subTest(input_id=row["input_id"]):
                self.assertEqual(row["input_ids"], [ids[index] for index in row["token_source_positions"]])
                self.assertEqual([row["input_ids"][index] for index in row["target_positions"]],
                                 [ids[index] for index in row["target_original_positions"]])
                self.assertEqual([row["token_source_positions"][index] for index in row["target_positions"]],
                                 row["target_original_positions"])
                if row["source_role"] is not None:
                    source = targets[row["source_role"]]
                    self.assertEqual(row["block_index"], source["block_index"])
                    self.assertEqual(row["target_original_positions"], list(range(source["body_start"], source["body_end"])))
        # Serializability is part of the persisted runner-input contract.
        self.assertEqual(json.loads(json.dumps(rows)), rows)
        self.assertEqual(trace, self.full_trace())

    def test_strip_removes_both_wrappers_and_touches_no_other_token(self):
        trace = self.full_trace()
        ids = trace["input_ids"]
        blocks = data.parse_native_blocks(ids, self.tok)
        rows = data.build_interventions(trace, self.tok)
        for row in (value for value in rows if value["operation"] == "strip"):
            block = blocks[row["block_index"]]
            self.assertEqual(row["input_ids"], ids[:block["start"]] + block["body_ids"] + ids[block["end"]:])
            self.assertEqual(row["target_positions"][0], block["start"])
            self.assertEqual(len(ids) - len(row["input_ids"]), len(block["header_ids"]) + len(block["ending_ids"]))
            # The repeated user body remains in place; only its first instance is selected.
            self.assertIn(1081, row["input_ids"])

    def test_every_swap_copies_exact_native_destination_wrapper(self):
        trace = self.full_trace()
        ids = trace["input_ids"]
        targets = data.first_role_targets(data.parse_native_blocks(ids, self.tok))
        swaps = [row for row in data.build_interventions(trace, self.tok) if row["operation"] == "swap"]
        self.assertEqual({(row["source_role"], row["destination_role"]) for row in swaps},
                         {(source, destination) for source in data.ROLES for destination in data.ROLES if source != destination})
        for row in swaps:
            source, destination = targets[row["source_role"]], targets[row["destination_role"]]
            expected = (ids[:source["start"]] + destination["header_ids"] + source["body_ids"]
                        + destination["ending_ids"] + ids[source["end"]:])
            self.assertEqual(row["input_ids"], expected)
            self.assertEqual(row["applied_header_ids"], destination["header_ids"])
            self.assertEqual(row["applied_ending_ids"], destination["ending_ids"])
            self.assertNotIn(data.CALL, row["applied_ending_ids"])
            parsed = data.parse_native_blocks(expected, self.tok)
            self.assertEqual(parsed[source["block_index"]]["role"], row["destination_role"])

    def test_move_is_farthest_among_all_remaining_native_boundaries(self):
        trace = self.full_trace()
        ids = trace["input_ids"]
        blocks = data.parse_native_blocks(ids, self.tok)
        complete = [block for block in blocks if block["complete"]]
        moves = [row for row in data.build_interventions(trace, self.tok) if row["operation"] == "move"]
        for row in moves:
            source = blocks[row["block_index"]]
            other_lengths = [block["end"] - block["start"] for block in complete if block is not source]
            boundaries, cursor = [0], 0
            for length in other_lengths:
                cursor += length
                boundaries.append(cursor)
            candidates = [(boundary + len(source["header_ids"]), boundary) for boundary in boundaries]
            expected_start, _ = sorted(candidates, key=lambda item: (-abs(item[0] - source["body_start"]), item[0]))[0]
            self.assertEqual(row["new_body_start"], expected_start)
            self.assertEqual(sorted(row["token_source_positions"]), list(range(len(ids))))
            self.assertEqual(row["input_ids"][-len(blocks[-1]["header_ids"]):], blocks[-1]["header_ids"])
            parsed = data.parse_native_blocks(row["input_ids"], self.tok)
            moved = next(block for block in parsed if block["body_start"] == row["new_body_start"])
            self.assertEqual(moved["header_ids"] + moved["body_ids"] + moved["ending_ids"], ids[source["start"]:source["end"]])
            other_origins = [index for index in row["token_source_positions"] if not source["start"] <= index < source["end"]]
            self.assertEqual(other_origins, list(range(source["start"])) + list(range(source["end"], len(ids))))

    def test_equal_distance_move_chooses_start(self):
        # Equal complete lengths make the middle user block equidistant from both ends.
        prefix = self.block("system", [2011])
        middle = self.block("user", [2021, 2022, 2023])
        suffix = self.block("system", [2031])
        self.assertEqual(len(prefix), len(middle))
        trace = dict(trace_id="tie", input_ids=prefix + middle + suffix)
        row = next(row for row in data.build_interventions(trace, self.tok, require_all_roles=False)
                   if row["operation"] == "move" and row["source_role"] == "user")
        self.assertEqual(row["move_candidate_distances"]["start"], row["move_candidate_distances"]["end"])
        self.assertEqual(row["move_destination"], "start")

    def test_incomplete_or_ambiguous_sources_fail_closed(self):
        valid = self.block("user", [501, 502])
        malformed = [[], [123] + valid, valid[:-1], [data.START],
                     self.block("user", [501], data.CALL),
                     self.block("system", [501], data.RETURN),
                     self.block("assistant to=functions.x<|channel|>commentary json", [501], data.END),
                     self.block("user", [501, data.START, 502]),
                     self.block("user", [501, data.MESSAGE, 502]),
                     self.block("unrecognized", [501]),
                     valid + [data.START] + self.tok.encode("user")]
        for ids in malformed:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    data.parse_native_blocks(ids, self.tok)
        self.assertEqual(len(data.parse_native_blocks(valid, self.tok)), 1)
        with self.assertRaisesRegex(ValueError, "lacks role targets"):
            data.build_interventions(dict(trace_id="partial", input_ids=valid), self.tok)
        empty_first_user = self.block("user", []) + valid
        with self.assertRaisesRegex(ValueError, "first user body is empty"):
            data.build_interventions(dict(trace_id="empty-first", input_ids=empty_first_user), self.tok, require_all_roles=False)

    def test_authorized_terminal_assistant_keeps_observed_ids_and_all_variants(self):
        trace = self.full_trace(open_header=False)
        # Remove the later repeated user/final frames, then the first final's return.
        blocks = data.parse_native_blocks(trace["input_ids"], self.tok)
        final = blocks[6]
        trace["input_ids"] = trace["input_ids"][:final["body_end"]]
        with self.assertRaisesRegex(ValueError, "Unterminated native body"):
            data.parse_native_blocks(trace["input_ids"], self.tok)
        trace["allow_terminal_body"] = True
        blocks = data.parse_native_blocks(trace["input_ids"], self.tok, allow_terminal_body=True)
        observed = blocks[-1]
        self.assertTrue(observed["complete"])
        self.assertFalse(observed["terminated"])
        self.assertTrue(observed["observed_terminal_body"])
        self.assertEqual(observed["ending_ids"], [])
        rows = data.build_interventions(trace, self.tok)
        self.assertEqual(len(rows), 43)
        ids = trace["input_ids"]
        for row in rows:
            self.assertEqual(row["input_ids"], [ids[index] for index in row["token_source_positions"]])
            self.assertEqual([row["input_ids"][index] for index in row["target_positions"]],
                             [ids[index] for index in row["target_original_positions"]])
            self.assertTrue(row["trace_has_observed_terminal_body"])
            if row["operation"] == "swap" and row["destination_role"] == "assistant":
                self.assertEqual(row["applied_ending_ids"], [])
                self.assertFalse(row["applied_wrapper_terminated"])
            if row["operation"] == "move":
                self.assertEqual(Counter(row["input_ids"]), Counter(ids))
        moved_final = next(row for row in rows if row["operation"] == "move" and row["source_role"] == "assistant")
        self.assertEqual(moved_final["move_destination"], "start")
        self.assertEqual(moved_final["applied_ending_ids"], [])
        self.assertTrue(moved_final["observed_terminal_body"])
        for header in ("user", "system", "assistant<|channel|>analysis",
                       "assistant to=functions.x<|channel|>commentary json"):
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    data.parse_native_blocks(self.block(header, [501])[:-1], self.tok, allow_terminal_body=True)


if __name__ == "__main__":
    unittest.main()
