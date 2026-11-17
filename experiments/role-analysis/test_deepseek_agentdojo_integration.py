"""Real adapter response -> real queue/HTTP -> bridge receipts, CPU only.

Model arithmetic is explicitly a tiny engineering fixture. Response dictionaries,
transaction receipts, artifact namespaces and HTTP handling are production code.
"""
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import torch

import deepseek_agentdojo_bridge as bridge
from deepseek_agentdojo_model import DeepSeekAgentDojoModel
from run_deepseek_agentdojo_server import ModelQueue, make_handler
from test_deepseek_agentdojo_model import FakeCodec, FakeLM, FakeProbe, fake_locator


@contextmanager
def actual_adapter_http(root):
    model = FakeLM()
    adapter = DeepSeekAgentDojoModel(root, root, root, _cpu_test=True,
                _model=model, _probe=FakeProbe(), _codec=FakeCodec(), locator=fake_locator,
                max_new_tokens=4, max_model_len=32)
    engine = ModelQueue(root, batch_size=1, batch_wait_seconds=0)
    engine.identity_sha256 = "engineering-integration-server"
    engine.start_worker(adapter)
    engine.set_stage("ready")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield adapter, engine, f"http://127.0.0.1:{server.server_port}/generate"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        engine.stop.set()
        engine.worker.join(timeout=5)
        adapter.close()


class ActualResponseIntegrationTests(unittest.TestCase):
    def args(self, scope):
        call_id = "plain/banking/user_task_0/injection_task_0/call_0000"
        return ([{"role": "system", "content": "semantic system"}, {"role": "user", "content": "abc"}], [],
                {"case_id": "plain/banking/user_task_0/injection_task_0", "call_id": call_id,
                 "condition": "plain", "actual_goal": "engineering goal",
                 "recording_dir": str(scope / "calls" / call_id)})

    def verified_model_files(self, scope, bridge_ref):
        receipt = bridge.read(bridge.verify_reference(scope, bridge_ref))
        paths = [bridge.verify_reference(scope, ref) for ref in receipt["artifacts"]]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertTrue(path.is_relative_to(scope))
        model_receipts = [p for p in paths if p.name == "receipt.json"]
        self.assertEqual(len(model_receipts), 1)
        model_receipt = bridge.read(model_receipts[0])
        self.assertEqual(model_receipt["format_version"], "deepseek-agentdojo-call-v1")
        self.assertEqual(model_receipt["raw_request_sha256"], receipt["raw_request_sha256"])
        base = model_receipts[0].parent
        for filename, digest in model_receipt["files"].items():
            self.assertEqual(bridge.sha(base / filename), digest)
            self.assertIn(base / filename, paths)
        return base, model_receipt

    def test_real_response_roundtrip_smoke_formal_namespace_and_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            smoke = root / "smoke"
            smoke.mkdir()
            with actual_adapter_http(root) as (adapter, engine, endpoint):
                smoke_client = bridge.HttpGenerator(smoke, endpoint, engine.identity_sha256,
                                                    server_output_root=root, retry_seconds=0)
                formal_client = bridge.HttpGenerator(root, endpoint, engine.identity_sha256,
                                                     server_output_root=root, retry_seconds=0)
                smoke_response, smoke_ref = smoke_client(*self.args(smoke))
                formal_response, formal_ref = formal_client(*self.args(root))
                self.assertTrue(smoke_response["full_payload_captured"])
                self.assertTrue(formal_response["full_payload_captured"])
                self.assertNotIn("_role_probe_artifacts", smoke_response)
                self.assertIn("artifact_refs", smoke_response)
                self.assertNotEqual(smoke_response["artifact_refs"], formal_response["artifact_refs"])
                self.assertEqual(len(adapter.model.records), 6)
                for scope, client, response, ref in ((smoke, smoke_client, smoke_response, smoke_ref),
                                                       (root, formal_client, formal_response, formal_ref)):
                    base, model_receipt = self.verified_model_files(scope, ref)
                    self.assertEqual(model_receipt["status"], "complete")
                    self.assertEqual(model_receipt["metadata"], self.args(scope)[2])
                    inputs, output = np.load(base / "input.npz"), np.load(base / "output.npz")
                    np.testing.assert_array_equal(output["token_ids"], [5, 6, 1])
                    np.testing.assert_array_equal(output["token_indices"], [3, 4, 5])
                    np.testing.assert_array_equal(output["observed"], [True, True, False])
                    self.assertTrue(np.isnan(output["probabilities"][-1]).all())
                    bits = np.load(base / "payload-heads-bf16-uint16.npy")
                    states = torch.from_numpy(bits).view(torch.bfloat16).float().numpy()
                    features = states[:, [1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]].astype(np.float16)
                    np.testing.assert_array_equal(features, np.load(base / "payload-selected-fp16.npy"))
                    np.testing.assert_allclose(inputs["probabilities"][[2, 0]], FakeProbe().predict_proba(features), atol=1e-7)
                    inputs.close()
                    output.close()
                    # Bridge replay verifies every artifact and does not invoke HTTP/model.
                    self.assertEqual(client(*self.args(scope)), (response, ref))
                self.assertEqual(len(adapter.model.records), 6)
                self.assertEqual(adapter.counters["full_payload_captures"], 2)
                # A changed original head state invalidates durable bridge replay.
                bits_path = root / "calls" / self.args(root)[2]["call_id"] / "payload-heads-bf16-uint16.npy"
                with bits_path.open("r+b") as stream:
                    stream.seek(-2, 2)
                    stream.write(b"xx")
                with self.assertRaises(bridge.IntegrityError):
                    formal_client(*self.args(root))

    def test_actual_context_overflow_error_retains_unforwarded_ids_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with actual_adapter_http(root) as (adapter, engine, endpoint):
                adapter.max_model_len = 6  # Three input plus four reserved generation tokens cannot fit.
                client = bridge.HttpGenerator(root, endpoint, engine.identity_sha256,
                                              server_output_root=root, retry_seconds=0)
                response, ref = client(*self.args(root))
                self.assertEqual(response["error"]["type"], "NativeContextOverflow")
                self.assertEqual(response["error"]["category"], "model")
                base, receipt = self.verified_model_files(root, ref)
                self.assertEqual(receipt["status"], "error")
                self.assertFalse(receipt["forward_started"])
                self.assertFalse(receipt["input_forward_completed"])
                np.testing.assert_array_equal(np.load(base / "unforwarded-input.npz")["token_ids"], [2, 3, 4])
                self.assertFalse((base / "input.npz").exists())
                self.assertFalse(adapter.model.records)
                self.assertEqual(client(*self.args(root)), (response, ref))

    def test_actual_marker_failure_is_server_fatal_and_never_seals_native_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with actual_adapter_http(root) as (adapter, engine, endpoint):
                adapter.locator = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("marker proof changed"))
                client = bridge.HttpGenerator(root, endpoint, engine.identity_sha256,
                                              server_output_root=root, retry_seconds=0)
                with self.assertRaisesRegex(bridge.IntegrityError, "infrastructure"):
                    client(*self.args(root))
                self.assertTrue(engine.stop.wait(3))
                self.assertEqual(engine.stage, "failed")
                self.assertEqual(engine.fatal_error["category"], "infrastructure")
                bridge_dir = root / "bridge_calls" / self.args(root)[2]["call_id"]
                self.assertTrue((bridge_dir / "original_request.json").exists())
                self.assertFalse((bridge_dir / "receipt.json").exists())
                self.assertFalse((root / "calls" / self.args(root)[2]["call_id"]).exists())
                self.assertTrue(list((root / "calls").glob(".tmp-*/error.json")))
                self.assertFalse(adapter._exposure_seen)


if __name__ == "__main__":
    unittest.main()
