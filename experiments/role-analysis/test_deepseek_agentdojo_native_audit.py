"""Independent native CPU replay fixtures; no language model or GPU is loaded."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deepseek_agentdojo_bridge as producer
import deepseek_agentdojo_native_audit as audit
import deepseek_agentdojo_protocol as protocol
from test_deepseek_agentdojo_bridge import fake_http, fixture_response


class PendingTests(unittest.TestCase):
    def test_missing_run_is_running_never_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            report = audit.audit_run(folder)
            self.assertEqual(report["status"], "running")
            self.assertFalse(report["complete"])
            self.assertEqual(report["audited_cases"], 0)

    def test_official_parser_returning_invalid_json_reproduces_post_parse_error(self):
        model = Path('/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731')
        parser = model/'encoding/encoding_dsv4.py'
        if not parser.is_file():
            self.skipTest("Official local encoder is unavailable")
        completion = ('reason</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="f">\n'
                      '<｜DSML｜parameter name="x" string="false">not json</｜DSML｜parameter>\n'
                      '</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>')
        request = {"response": {"raw_completion": completion, "error": {
            "category": "model", "type": "NativeCompletionParseError", "fatal": False,
            "parser_error_type": "JSONDecodeError", "message": "Expecting value: line 1 column 7 (char 6)"}}}
        identity = {"generation": {}, "server_identity": {"model_dir": str(model), "model_binding": {
            "official_encoder_sha256": audit.sha(parser)}}}
        audit.validate_model_error(request, identity)
        request["response"]["error"]["message"] = "wrong error"
        with self.assertRaisesRegex(audit.AuditError, "does not independently reproduce"):
            audit.validate_model_error(request, identity)

    def test_decoded_arguments_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            parser = model/'encoding/encoding_dsv4.py'
            parser.parent.mkdir()
            parser.write_text('def parse_message_from_completion_text(text, thinking_mode):\n'
                              '    return {"tool_calls": [{"function": {"arguments": "[]"}}]}\n')
            identity = {"generation": {}, "server_identity": {"model_dir": str(model), "model_binding": {
                "official_encoder_sha256": audit.sha(parser)}}}
            request = {"response": {"raw_completion": "fixture", "error": {
                "category": "model", "type": "NativeCompletionParseError", "fatal": False,
                "parser_error_type": "ValueError", "message": "Tool arguments must be an object"}}}
            audit.validate_model_error(request, identity)


@unittest.skipUnless(importlib.util.find_spec("agentdojo"), "Use chatinject-table1 native CPU environment")
class NativeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = audit.dependencies(producer.DEFAULT_REPO)
        cls.production = producer.load_dependencies(producer.DEFAULT_REPO)
        cls.cases = producer.smoke_cases(protocol.native_cases(cls.production["get_suite"]))

    def identity(self, root, cases):
        model = Path('/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731')
        return {"mode": "smoke", "generation": {"temperature": 0.0, "seed": 0, "max_tokens": 1024,
                    "max_model_len": 32768, "thinking_mode": "thinking", "reasoning_effort": "low"},
                "server_identity_sha256": "fixture-server", "schedule_sha256": audit.object_sha(cases),
                "agentdojo_source": producer.source_identity(producer.DEFAULT_REPO, protocol),
                "protocol_sha256": audit.sha(Path(protocol.__file__)), "tool_formatter": {"allow_unicode": True},
                "server_identity": {"model_dir": str(model), "model_binding": {
                    "official_encoder_sha256": audit.sha(model/'encoding/encoding_dsv4.py')}}}

    def test_real_email_clock_and_all_six_native_evaluations_reproduce(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = self.identity(root, self.cases)
            producer.write(root/"run_identity.json", identity)
            producer.write(root/"schedule.json", self.cases)

            def generate(payload):
                meta = payload["metadata"]
                response = fixture_response(payload)
                if meta["call_index"] == 0:
                    if meta["suite"] == "travel":
                        # An actual send_email produces the original clock value.
                        name, args = "send_email", {"recipients": ["alice@example.com"], "subject": "CPU fixture", "body": "Fixture only"}
                    else:
                        suite = self.production["get_suite"](protocol.BENCHMARK_VERSION, meta["suite"])
                        task = suite.user_tasks[meta["user_task_id"]]
                        env = task.init_environment(suite.load_and_inject_default_environment({}))
                        call = task.ground_truth(env)[0]
                        name, args = call.function, call.args
                    response.update(content=None, reasoning_content="fixture thought", tool_calls=[{"function": {
                        "name": name, "arguments": json.dumps(args)}}], finish_reason="tool_calls")
                return response

            with fake_http(generate, identity_sha256="fixture-server") as (url, observed, errors):
                client = producer.HttpGenerator(root, url, "fixture-server", retry_seconds=0)
                summary = producer.run_benchmark(root, self.cases, identity, client, self.production, protocol, workers=3)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(errors, [])
                calls_before = len(observed)
                report = audit.audit_run(root, allow_smoke=True)
                self.assertEqual(report["status"], "passed", report)
                self.assertTrue(report["complete"])
                self.assertEqual(report["audited_cases"], 6)
                self.assertEqual(report["model_requests"], 12)
                self.assertEqual(len(observed), calls_before)
                clocks = [binding for row in report["native_results"] for binding in row["clock_bindings"]]
                self.assertEqual(len(clocks), 2)
                self.assertTrue(all(row["actual_mock_tool_executed"] and not row["tool_result_replaced"] for row in clocks))
                # Changing the saved Boolean cannot survive independent replay.
                case = self.cases[0]
                path = root/"cases"/(case["case_id"]+".json")
                record = producer.read(path)
                record["security"] = not record["security"]
                with self.assertRaisesRegex(audit.AuditError, "utility/security"):
                    audit.replay_case(root, case, record, identity, self.native, protocol, {})

    def test_native_error_reproduces_and_hash_consistent_prompt_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = self.cases[0]
            identity = self.identity(root, [case])
            response = {"raw_completion": "missing native thinking boundary", "error": {
                "category": "model", "type": "NativeCompletionParseError", "parser_error_type": "AssertionError",
                "message": "Invalid thinking format: missing </think>", "fatal": False}}
            with fake_http(lambda _: response, identity_sha256="fixture-server") as (url, observed, errors):
                client = producer.HttpGenerator(root, url, "fixture-server", retry_seconds=0)
                producer.run_benchmark(root, [case], identity, client, self.production, protocol, workers=1)
                record_path = root/"cases"/(case["case_id"]+".json")
                record = producer.read(record_path)
                result = audit.replay_case(root, case, record, identity, self.native, protocol, {})
                self.assertTrue(result["native_error_reproduced"])
                self.assertIsNone(result["utility"])
                self.assertIsNone(result["security"])
                self.assertEqual(len(observed), 1)
                self.assertEqual(errors, [])
                # Rebuild every local SHA after changing a semantic system body.
                # Native reconstruction must still reject the changed prompt.
                receipt_path = root/record["requests"][0]["bridge_receipt"]["path"]
                receipt = producer.read(receipt_path)
                request_path = root/receipt["request"]["path"]
                request = producer.read(request_path)
                request["messages"][0]["content"] = "hash-consistent but not native"
                producer.write(request_path, request)
                digest = producer.object_sha(request)
                receipt["request"]["sha256"] = producer.sha(request_path)
                receipt["raw_request_sha256"] = digest
                wrapper_path = root/receipt["response"]["path"]
                wrapper = producer.read(wrapper_path)
                wrapper["raw_request_sha256"] = digest
                producer.write(wrapper_path, wrapper)
                receipt["response"]["sha256"] = producer.sha(wrapper_path)
                producer.write(receipt_path, receipt)
                record["requests"][0]["bridge_receipt"]["sha256"] = producer.sha(receipt_path)
                with self.assertRaisesRegex(audit.AuditError, "native history"):
                    audit.replay_case(root, case, record, identity, self.native, protocol, {})


if __name__ == "__main__":
    unittest.main()
