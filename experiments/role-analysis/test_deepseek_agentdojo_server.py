import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from run_deepseek_agentdojo_server import ModelQueue, make_handler, request_sha


def request(call="plain/banking/user_task_0/injection_task_0/call_0000"):
    return {"messages": [{"role": "tool", "content": "<｜User｜>template and goal<｜Assistant｜><think>"}],
            "tools": [], "metadata": {"call_id": call}}


class FakeAdapter:
    supported_batch_size = 2

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def generate_batch(self, requests):
        self.calls.append(requests)
        time.sleep(.02)
        if self.error:
            raise self.error
        return [{"content": "ok", "tool_calls": [], "raw_request_sha256": request_sha(r)} for r in requests]


def stop(engine):
    engine.stop.set()
    if engine.worker:
        engine.worker.join(timeout=3)
        assert not engine.worker.is_alive()


def test_canonical_unicode_identity_and_path_validation():
    original = request()
    roundtrip = json.loads(json.dumps(original, ensure_ascii=True))
    assert request_sha(original) == request_sha(roundtrip)
    roundtrip["messages"][0]["content"] += "changed"
    assert request_sha(original) != request_sha(roundtrip)
    with pytest.raises(ValueError):
        request_sha(request("../bad"))
    with pytest.raises(ValueError):
        request_sha({**original, "unbound_field": 1})


def test_duplicate_transport_submission_has_one_model_owner(tmp_path):
    engine = ModelQueue(tmp_path)
    adapter = FakeAdapter()
    engine.start_worker(adapter)
    engine.stage = "ready"
    try:
        first = engine.submit(request())
        duplicate = engine.submit(request())
        assert first is duplicate
        assert first.done.wait(3)
        assert len(adapter.calls) == 1
        assert engine.submit(request()) is first
        assert first.response["raw_request_sha256"] == request_sha(request())
    finally:
        stop(engine)


def test_batch_contract_is_explicit_and_unvalidated_batch_rejected(tmp_path):
    engine = ModelQueue(tmp_path, batch_size=3)
    with pytest.raises(ValueError, match="validated"):
        engine.start_worker(FakeAdapter())
    engine = ModelQueue(tmp_path, batch_size=2, batch_wait_seconds=.1)
    adapter = FakeAdapter()
    engine.start_worker(adapter)
    engine.stage = "ready"
    try:
        items = [engine.submit(request(str(i))) for i in range(2)]
        assert all(x.done.wait(3) for x in items)
        assert len(adapter.calls) == 1 and len(adapter.calls[0]) == 2
    finally:
        stop(engine)


@pytest.mark.parametrize("error,category,fatal", [
    (ValueError("Native output cannot be parsed"), "model", False),
    (RuntimeError("CUDA out of memory"), "infrastructure", True),
])
def test_model_and_infrastructure_errors_remain_distinct(tmp_path, error, category, fatal):
    engine = ModelQueue(tmp_path)
    engine.start_worker(FakeAdapter(error))
    engine.stage = "ready"
    try:
        item = engine.submit(request())
        assert item.done.wait(3)
        assert item.response["error"]["category"] == category
        assert (engine.fatal_error is not None) == fatal
        assert engine.failed_calls == 1
    finally:
        stop(engine)


def test_integrity_failure_is_fatal_and_retains_actual_artifact_refs(tmp_path):
    class IntegrityFailure(RuntimeError):
        category = "infrastructure"
        response = {"artifact_refs": {"receipt": "calls/actual/receipt.json"},
                    "error": {"type": "IntegrityFailure"}}

    engine = ModelQueue(tmp_path)
    engine.start_worker(FakeAdapter(IntegrityFailure("Marker proof differs")))
    engine.stage = "ready"
    try:
        item = engine.submit(request())
        assert item.done.wait(3)
        assert item.response["error"]["category"] == "infrastructure"
        assert item.response["artifact_refs"]["receipt"] == "calls/actual/receipt.json"
        engine.worker.join(timeout=3)
        assert engine.stage == "failed" and engine.fatal_error is not None
    finally:
        stop(engine)


def test_http_health_busy_shutdown_and_request_roundtrip(tmp_path):
    engine = ModelQueue(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def post(path, value):
        req = urllib.request.Request(url + path, data=json.dumps(value).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as response:
            return json.load(response)

    try:
        assert json.load(urllib.request.urlopen(url + "/health"))["stage"] == "starting"
        with pytest.raises(urllib.error.HTTPError) as caught:
            post("/generate", request())
        assert caught.value.code == 503
        engine.start_worker(FakeAdapter())
        engine.identity_sha256 = "expected"
        engine.stage = "ready"
        response = post("/generate", request())
        assert response["content"] == "ok" and response["raw_request_sha256"] == request_sha(request())
        with pytest.raises(urllib.error.HTTPError) as caught:
            post("/shutdown", {"identity_sha256": "wrong"})
        assert caught.value.code == 400
        assert post("/shutdown", {"identity_sha256": "expected"})["status"] == "closing"
    finally:
        stop(engine)
        server.shutdown(); server.server_close(); thread.join(timeout=3)
