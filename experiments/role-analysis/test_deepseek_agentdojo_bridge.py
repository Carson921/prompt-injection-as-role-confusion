"""CPU fixtures for HTTP transport, strict replay, and real native evaluators."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deepseek_agentdojo_bridge as bridge
import deepseek_agentdojo_protocol as protocol


@contextmanager
def fake_http(callback, *, fail_once=False, identity_sha256="frozen-server"):
    observed = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = bridge.canonical({"stage": "ready", "identity_sha256": identity_sha256}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append(payload)
            if fail_once and len(observed) == 1:
                self.send_error(503, "Fixture pending")
                return
            try:
                result = callback(payload)
            except Exception as error:
                errors.append(error)
                result = {"error": {"type": type(error).__name__, "message": str(error)}}
            body = bridge.canonical(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/generate", observed, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fixture_response(payload):
    directory = Path(payload["metadata"]["recording_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / "fixture-model-artifact.json"
    bridge.write(p, {"fixture_only": True, "raw_request_sha256": bridge.object_sha(payload)})
    return {"content": "Fixture response", "reasoning_content": "fixture reasoning", "tool_calls": [],
        "finish_reason": "stop", "raw_request_sha256": bridge.object_sha(payload),
        "_role_probe_artifacts": [{"path": str(p), "sha256": bridge.sha(p)}]}


class HttpTests(unittest.TestCase):
    def payload(self, root):
        return ([{"role": "system", "content": "semantic system"}, {"role": "user", "content": "fixture"}], [],
                {"call_id": "plain/banking/user_task_0/injection_task_0/call_0000",
                 "recording_dir": str(root / "calls/fixture")})

    def test_transport_pending_retries_identical_request_then_replays_without_http(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with fake_http(fixture_response, fail_once=True) as (url, observed, errors):
                generator = bridge.HttpGenerator(root, url, "frozen-server", retry_seconds=0)
                args = self.payload(root)
                response, ref = generator(*args)
                self.assertEqual(len(observed), 2)
                self.assertEqual(observed[0], observed[1])
                self.assertEqual(errors, [])
                replay, replay_ref = generator(*args)
                self.assertEqual((replay, replay_ref), (response, ref))
                self.assertEqual(len(observed), 2)
                self.assertFalse((root / "bridge_calls" / args[2]["call_id"] / "transport_pending.json").exists())
                changed = [{"role": "user", "content": "changed"}]
                with self.assertRaisesRegex(bridge.IntegrityError, "request differs"):
                    generator(changed, args[1], args[2])
                artifact_path = Path(response["_role_probe_artifacts"][0]["path"])
                artifact_path.write_text("tampered")
                with self.assertRaisesRegex(bridge.IntegrityError, "checksum"):
                    generator(*args)

    def test_http200_native_error_is_sealed_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with fake_http(lambda _: {"error": {"type": "ParseError", "message": "malformed DSML"}}) as (url, observed, errors):
                generator = bridge.HttpGenerator(root, url, "frozen-server", retry_seconds=0)
                args = self.payload(root)
                response, ref = generator(*args)
                self.assertIn("error", response)
                self.assertEqual(bridge.read(root / ref["path"])["status"], "native_error")
                self.assertEqual(generator(*args)[0], response)
                self.assertEqual(len(observed), 1)
                self.assertEqual(errors, [])

    def test_infrastructure_error_never_seals_a_native_response(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with fake_http(lambda _: {"error": {"category": "infrastructure", "type": "CUDAError", "message": "fixture"}}) as (url, observed, errors):
                generator = bridge.HttpGenerator(root, url, "frozen-server", retry_seconds=0)
                args = self.payload(root)
                with self.assertRaisesRegex(bridge.IntegrityError, "infrastructure"):
                    generator(*args)
                directory = root / "bridge_calls" / args[2]["call_id"]
                self.assertTrue((directory / "original_request.json").exists())
                self.assertFalse((directory / "response.json").exists())
                self.assertFalse((directory / "receipt.json").exists())
                self.assertEqual(len(observed), 1)
                self.assertEqual(errors, [])

    def test_error_denominators_and_unexposed_case_remain_in_expected(self):
        rows = [{"status": "complete", "utility": True, "security": False},
                {"status": "complete", "utility": False, "security": True},
                {"status": "error", "utility": None, "security": None}]
        result = bridge.metrics(rows, 389)
        self.assertIsNone(result["asr"])
        self.assertEqual(result["asr_scored"], .5)
        self.assertEqual(result["missing_cases"], 386)
        self.assertEqual(result["attack_successes"], 1)
        self.assertFalse(result["data_fully_scored"])
        self.assertEqual(bridge.metrics(rows[:2], 2)["asr"], .5)


@unittest.skipUnless(importlib.util.find_spec("agentdojo"), "Use chatinject-table1 CPU environment for native integration")
class NativePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = bridge.load_dependencies(bridge.DEFAULT_REPO)
        cls.suites = {s: cls.dependencies["get_suite"](protocol.BENCHMARK_VERSION, s) for s in ("banking", "slack", "travel")}
        cls.cases = bridge.smoke_cases(protocol.native_cases(cls.dependencies["get_suite"]))

    def test_six_http_episodes_execute_real_tools_evaluators_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seen_reasons = []
            exposed = set()

            def generate(payload):
                metadata = payload["metadata"]
                index = metadata["call_index"]
                suite_name = metadata["suite"]
                self.assertEqual(payload["messages"][0]["role"], "system")
                self.assertEqual(metadata["max_model_len"], 32768)
                self.assertEqual(metadata["thinking_mode"], "thinking")
                self.assertEqual(metadata["reasoning_effort"], "low")
                if index:
                    self.assertEqual(payload["messages"][-1]["role"], "tool")
                    assistants = [m for m in payload["messages"] if m["role"] == "assistant"]
                    seen_reasons.append(assistants[-1]["reasoning_content"])
                    self.assertEqual(assistants[-1]["reasoning_content"], "reasoning-step-" + str(index-1))
                    if any(metadata["payload_text"] in (m.get("content") or "") for m in payload["messages"] if m["role"] == "tool"):
                        exposed.add(metadata["case_id"])
                    if index == 1 and metadata["condition"] == "plain_delimiter":
                        body = "".join(m.get("content") or "" for m in payload["messages"] if m["role"] == "tool")
                        self.assertIn("<｜User｜>", body)
                        self.assertNotIn("\\uFF5C", body)
                suite = self.suites[suite_name]
                task = suite.user_tasks["user_task_0"]
                environment = task.init_environment(suite.load_and_inject_default_environment({}))
                plan = task.ground_truth(environment)
                # Banking executes read_file AND send_money. Other suites execute
                # one real read; their native evaluator determines the outcome.
                n = 2 if suite_name == "banking" else 1
                response = fixture_response(payload)
                response["reasoning_content"] = "reasoning-step-" + str(index)
                if index < n:
                    call = plan[index]
                    response.update(content=None, tool_calls=[{"type": "function", "function": {
                        "name": call.function, "arguments": json.dumps(call.args)}}], finish_reason="tool_calls")
                return response

            with fake_http(generate, identity_sha256="fixture-server") as (url, observed, errors):
                client = bridge.HttpGenerator(root, url, "fixture-server", retry_seconds=0)
                identity = {"fixture_only": True, "schedule_sha256": bridge.object_sha(self.cases)}
                summary = bridge.run_benchmark(root, self.cases, identity, client, self.dependencies, protocol, workers=3)
                self.assertEqual(errors, [])
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["scheduled_cases"], 6)
                self.assertEqual(summary["model_requests"], 14)
                self.assertEqual(len(seen_reasons), 8)
                self.assertTrue(exposed)
                for case in self.cases:
                    record = bridge.read(root / "cases" / (case["case_id"] + ".json"))
                    trace = bridge.read(root / record["raw_trace"]["path"])
                    self.assertEqual(record["utility"], trace["utility"])
                    self.assertEqual(record["security"], trace["security"])
                    self.assertTrue(trace["injections"])
                    self.assertTrue(all(v == case["payload_text"] for v in trace["injections"].values()))
                    if case["suite"] in {"banking", "slack"}:
                        self.assertTrue(record["utility"])
                requests_before = len(observed)
                resumed = bridge.run_benchmark(root, self.cases, identity, client, self.dependencies, protocol, workers=3)
                self.assertEqual(resumed["conditions"], summary["conditions"])
                self.assertEqual(len(observed), requests_before)
                # Crash between native final evaluation and the bridge case commit:
                # recover existing native scores without executing the model/tools.
                banking = next(c for c in self.cases if c["suite"] == "banking")
                case_path = root / "cases" / (banking["case_id"] + ".json")
                case_path.unlink()
                bridge.run_benchmark(root, [banking], identity, client, self.dependencies, protocol, workers=1)
                self.assertTrue(bridge.read(case_path)["recovered_native_terminal_trace"])
                self.assertEqual(len(observed), requests_before)
                # TraceLogger also writes unscored prefixes. Preserve that prefix
                # and replay already sealed requests against a fresh native env.
                slack = next(c for c in self.cases if c["suite"] == "slack")
                case_path = root / "cases" / (slack["case_id"] + ".json")
                original = bridge.read(case_path)
                trace_path = root / original["raw_trace"]["path"]
                partial = bridge.read(trace_path)
                partial.pop("utility")
                partial.pop("security")
                bridge.write(trace_path, partial)
                partial_sha = bridge.sha(trace_path)
                case_path.unlink()
                bridge.run_benchmark(root, [slack], identity, client, self.dependencies, protocol, workers=1)
                recovered = bridge.read(case_path)
                self.assertEqual(recovered["status"], "complete")
                self.assertEqual(recovered["resumed_partial_trace"]["sha256"], partial_sha)
                bridge.verify_reference(root, recovered["resumed_partial_trace"])
                self.assertEqual(len(observed), requests_before)

    def test_native_error_null_scores_terminal_resume_and_strict_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = self.cases[0]
            with fake_http(lambda _: {"error": {"type": "FixtureNativeError", "message": "unrepaired"}}, identity_sha256="fixture-server") as (url, observed, errors):
                client = bridge.HttpGenerator(root, url, "fixture-server", retry_seconds=0)
                identity = {"fixture_only": True}
                summary = bridge.run_benchmark(root, [case], identity, client, self.dependencies, protocol, workers=1)
                self.assertEqual(summary["status"], "complete_with_initial_errors")
                self.assertIsNone(summary["conditions"][case["condition"]]["asr"])
                record = bridge.read(root / "cases" / (case["case_id"] + ".json"))
                self.assertIsNone(record["utility"])
                self.assertIsNone(record["security"])
                bridge.run_benchmark(root, [case], identity, client, self.dependencies, protocol, workers=1)
                self.assertEqual(len(observed), 1)
                self.assertEqual(errors, [])
                (root / "cases" / (case["case_id"] + ".json")).unlink()
                bridge.run_benchmark(root, [case], identity, client, self.dependencies, protocol, workers=1)
                recovered = bridge.read(root / "cases" / (case["case_id"] + ".json"))
                self.assertTrue(recovered["recovered_native_terminal_trace"])
                self.assertIsNone(recovered["security"])
                self.assertEqual(len(observed), 1)
            for arguments in ['{"broken":', '[1,2]', 'null', '"text"']:
                with self.assertRaises(bridge.NativeModelError):
                    bridge.assistant_message({"tool_calls": [{"function": {"name": "read_file", "arguments": arguments}}]}, "fixture-call", self.dependencies)
            response = {"reasoning_content": "retained", "tool_calls": [{"function": {"name": "read_file", "arguments": '{"file_path":"x"}'}}]}
            first = bridge.assistant_message(response, "fixture-call", self.dependencies)
            second = bridge.assistant_message(response, "fixture-call", self.dependencies)
            self.assertEqual(first["tool_calls"][0].id, second["tool_calls"][0].id)
            converted = bridge.semantic_messages([first])
            self.assertEqual(converted[0]["reasoning_content"], "retained")
            self.assertEqual(json.loads(converted[0]["tool_calls"][0]["function"]["arguments"]), {"file_path": "x"})

    def test_infrastructure_error_aborts_episode_without_native_failure_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = self.cases[0]
            response = {"error": {"category": "infrastructure", "type": "OOM", "message": "fixture"}}
            with fake_http(lambda _: response, identity_sha256="fixture-server") as (url, observed, errors):
                client = bridge.HttpGenerator(root, url, "fixture-server", retry_seconds=0)
                with self.assertRaisesRegex(bridge.IntegrityError, "infrastructure"):
                    bridge.run_benchmark(root, [case], {"fixture_only": True}, client,
                                         self.dependencies, protocol, workers=1)
                self.assertTrue(client.stopped.is_set())
                self.assertFalse((root / "cases" / (case["case_id"] + ".json")).exists())
                self.assertFalse((root / "summary.json").exists())
                self.assertEqual(len(observed), 1)
                self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
