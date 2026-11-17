"""CPU controls for the independent IPI audit; all traces here are synthetic."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from test_ipi_trace_interventions_data import ToyTokenizer
import validate_ipi_trace_interventions as audit


class IndependentIPIValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()
        self.prefix = [audit.START] + self.tok.encode("assistant")

    def block(self, header, body, ending=audit.RETURN):
        return ([audit.START] + self.tok.encode(header) + [audit.MESSAGE]
                + self.tok.encode(body) + ([] if ending is None else [ending]))

    def step(self, header, body, ending=audit.RETURN, finish="stop"):
        response = self.block(header, body, ending)
        self.assertEqual(response[:len(self.prefix)], self.prefix)
        return dict(generation_prefix_ids=self.prefix, generated_ids=response[len(self.prefix):],
                    finish_reason=finish)

    def parsed(self, step):
        ids = step["generation_prefix_ids"] + step["generated_ids"]
        frames, _, _ = audit.original_frames(ids, self.tok.decode, step["finish_reason"] == "length",
                                             require_targets=False, allow_open_header=False)
        calls, incomplete, cot, final = [], [], [], []
        for frame in frames:
            body_ids = ids[frame["body_start"]:frame["body_end"]]
            content = self.tok.decode(body_ids)
            if frame["structural"] and frame["terminated"]:
                calls.append(dict(name=frame["recipient"].removeprefix("functions."),
                                  arguments=json.loads(content), raw_arguments=content,
                                  body_ids=body_ids, block_index=frame["block_index"]))
            elif frame["structural"]:
                incomplete.append(dict(recipient=frame["recipient"], raw_arguments=content,
                                       body_ids=body_ids, block_index=frame["block_index"]))
            elif frame["role"] == "cot":
                cot.append(content)
            else:
                final.append(content)
        return dict(tool_calls=calls, incomplete_tool_calls=incomplete,
                    content="\n".join(final) if final else None,
                    reasoning="\n".join(cot) if cot else None,
                    generation_prefix_ids=step["generation_prefix_ids"], raw_generated_ids=step["generated_ids"],
                    observed_terminal_body=any(not frame["terminated"] for frame in frames),
                    finish_reason=step["finish_reason"])

    def test_strict_response_positive_and_negative_controls(self):
        for step in (self.step("assistant", "done"),
                     self.step("assistant to=functions.x", '{"a":1}', audit.CALL),
                     self.step("assistant<|channel|>analysis", "thinking", None, "length"),
                     self.step("assistant to=functions.x", "{unfinished", None, "length")):
            with self.subTest(step=step):
                self.assertIsNone(audit.independent_response_error(step, self.tok.decode, self.parsed(step)))
        bad = [
            (self.step("assistant<|channel|>analysis to=functions.x<|channel|>commentary", "{}", audit.CALL), "ValueError"),
            (self.step("assistant to=functions.x to=functions.y", "{}", audit.CALL), "ValueError"),
            (self.step("assistant to=browser.x", "{}", audit.CALL), "ValueError"),
            (self.step("assistant to=functions.x", "[]", audit.CALL), "ValueError"),
            (self.step("assistant to=functions.x", "{broken", audit.CALL), "JSONDecodeError"),
            (self.step("assistant<|channel|>unknown", "x"), "ValueError"),
            (self.step("assistant", "x", audit.CALL), "ValueError"),
            (dict(generation_prefix_ids=self.prefix, generated_ids=[], finish_reason="length"), "ValueError"),
            (dict(generation_prefix_ids=[], generated_ids=self.block("user", "x", audit.END), finish_reason="stop"), "ValueError"),
        ]
        for step, kind in bad:
            with self.subTest(kind=kind, step=step):
                self.assertEqual(audit.independent_response_error(step, self.tok.decode)["type"], kind)

    def test_saved_tool_call_cannot_disagree_with_native_json_or_body(self):
        step = self.step("assistant to=functions.x", '{"a":1}', audit.CALL)
        expected = self.parsed(step)
        for key, value in (("name", "other"), ("arguments", {"a": 2}),
                           ("body_ids", [17]), ("block_index", 3), ("raw_arguments", '{"a":2}')):
            changed = deepcopy(expected)
            changed["tool_calls"][0][key] = value
            with self.subTest(key=key):
                self.assertEqual(audit.independent_response_error(step, self.tok.decode, changed)["type"], "ValueError")

    def synthetic_collection(self, output, invalid):
        case = "SYNTHETIC-case-00"
        initial = self.block("user", "benign", audit.END) + self.prefix
        fixture = dict(attack_text="", initial_input_ids=initial,
                       messages=[dict(role="user", content="benign")],
                       tools=[dict(type="function", function=dict(name="x"))])
        prepared_path = output / "prepared/case.json"
        prepared_path.parent.mkdir(parents=True)
        prepared_path.write_text(json.dumps(fixture))
        rows = [dict(case_id=f"SYNTHETIC-case-{index:02d}", trace_id=f"SYNTHETIC-trace-{index:02d}",
                     category=("tool" if index < 18 else "coding" if index < 33 else "browser"),
                     path="prepared/case.json", sha256=audit.sha(prepared_path)) for index in range(41)]
        (output / "prepared/manifest.json").write_text(json.dumps(dict(status="complete", cases=41, rows=rows)))
        collection = dict(code_sha256={}, prepared_manifest_sha256=audit.sha(output / "prepared/manifest.json"),
                          tool_call_id_policy="opaque step IDs", max_steps=5, max_new_tokens=4096)
        (output / "collection/cases").mkdir(parents=True)
        (output / "collection/protocol.json").write_text(json.dumps(collection))
        step = (self.step("assistant to=functions.x", "{broken", audit.CALL) if invalid
                else self.step("assistant", "done"))
        step.update(step=1, input_ids=initial, input_ids_sha256=audit.obj_sha(initial),
                    messages=fixture["messages"], worldsim=[])
        if invalid:
            step["parse_error"] = audit.independent_response_error(step, self.tok.decode)
        else:
            step["parsed_response"] = self.parsed(step)
        segments = [dict(kind="initial_request", step=1, start=0, end=len(initial),
                         request_start=0, request_end=len(initial)),
                    dict(kind="raw_generated", step=1, start=len(initial),
                         end=len(initial) + len(step["generated_ids"]), generated_start=0,
                         generated_end=len(step["generated_ids"]))]
        signature = audit.obj_sha(dict(config=collection, prepared_sha256=audit.sha(prepared_path)))
        source = dict(trace_id=rows[0]["trace_id"], input_ids=initial + step["generated_ids"],
                      allow_terminal_body=False, eof_finish_reason="stop",
                      provenance=dict(collection_case=case, collection_signature=signature, segments=segments,
                                      raw_generated_tokens=len(step["generated_ids"]),
                                      raw_generated_tokens_preserved_exactly_once=True, real_tool_execution=False,
                                      terminal_reason="invalid_native_response" if invalid else "final_response"))
        record = dict(status="complete", case_id=case, trace_id=source["trace_id"], trace=source,
                      signature=signature, prepared_sha256=audit.sha(prepared_path), steps=[step],
                      messages=fixture["messages"], terminal_reason=source["provenance"]["terminal_reason"])
        if invalid:
            self.mark_opaque(record)
        return record

    def mark_opaque(self, record):
        source, step = record["trace"], record["steps"][-1]
        offset = len(step["input_ids"]) - len(step["generation_prefix_ids"])
        record["terminal_reason"] = source["provenance"]["terminal_reason"] = "invalid_native_response"
        source["opaque_terminal_start"] = offset
        source["provenance"]["invalid_native_response"] = dict(**step["parse_error"], step=1,
                                                                opaque_terminal_start=offset)

    def commit_and_audit(self, output, record):
        # Recommit all altered hashes, so a negative test must fail semantic checks.
        case = record["case_id"]
        path = output / "collection/cases" / (case + ".json")
        path.write_text(json.dumps(record))
        (output / "sources").mkdir(exist_ok=True)
        sources = output / "sources/pilot-traces.jsonl"
        sources.write_text(json.dumps(record["trace"]) + "\n")
        manifest = dict(status="complete", stage="pilot", traces=1,
                        cases={case: dict(sha256=audit.sha(path))},
                        source_records_path="sources/pilot-traces.jsonl", source_records_sha256=audit.sha(sources),
                        collection_protocol_sha256=audit.sha(output / "collection/protocol.json"))
        manifest_path = output / "sources/pilot-collection-manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        protocol = dict(source_records_path="sources/pilot-traces.jsonl",
                        source_collection_manifest_sha256=audit.sha(manifest_path))
        return audit.audit_collection(output, "pilot", protocol, self.tok.decode)

    def test_whole_collection_accepts_real_invalid_suffix_and_rejects_semantic_tampering(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC-ipi-validator-") as directory:
            output = Path(directory)
            valid = self.synthetic_collection(output, invalid=False)
            self.assertEqual(self.commit_and_audit(output, valid)["status"], "passed")
            # A claimed parse failure on valid native text cannot authorize hiding it.
            spoof = deepcopy(valid)
            spoof["steps"][0].pop("parsed_response")
            spoof["steps"][0]["parse_error"] = dict(type="ValueError", message="fabricated")
            self.mark_opaque(spoof)
            with self.assertRaisesRegex(AssertionError, "not independently reproduced"):
                self.commit_and_audit(output, spoof)
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC-ipi-validator-") as directory:
            output = Path(directory)
            invalid = self.synthetic_collection(output, invalid=True)
            result = self.commit_and_audit(output, invalid)
            self.assertEqual(result["opaque_terminal_traces"], [invalid["trace_id"]])
            self.assertTrue(result["no_worldsim_after_native_parse_failure"])
            changed = deepcopy(invalid)
            changed["trace"]["opaque_terminal_start"] += 1
            changed["trace"]["provenance"]["invalid_native_response"]["opaque_terminal_start"] += 1
            with self.assertRaisesRegex(AssertionError, "exactly the current failed response"):
                self.commit_and_audit(output, changed)
            changed = deepcopy(invalid)
            changed["steps"][0]["worldsim"] = [dict(status="complete")]
            with self.assertRaisesRegex(AssertionError, "interpreted or simulated"):
                self.commit_and_audit(output, changed)


if __name__ == "__main__":
    unittest.main()
