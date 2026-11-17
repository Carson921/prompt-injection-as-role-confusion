"""Local, single-owner GPU queue for the native DeepSeek AgentDojo adapter.

AgentDojo runs in its own pinned CPU environment. Only this process imports the
model runtime. Requests, including duplicates after a transport timeout, enter
one model worker; model receipts remain the durable replay authority.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.metadata
import json
from pathlib import Path
import queue
import signal
import threading
import time
import traceback


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def request_sha(request):
    if set(request) != {"messages", "tools", "metadata"}:
        raise ValueError("Requests must contain exactly messages/tools/metadata")
    if not isinstance(request["messages"], list) or not isinstance(request["metadata"], dict):
        raise ValueError("Malformed semantic messages/metadata")
    call_id = request["metadata"].get("call_id")
    if not isinstance(call_id, str) or not call_id or call_id.startswith("/") or ".." in call_id.split("/"):
        raise ValueError("A relative, traversal-free call_id is required")
    return hashlib.sha256(canonical(request).encode()).hexdigest()


@dataclass
class Pending:
    request: dict
    digest: str
    done: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None


class ModelQueue:
    def __init__(self, output, *, batch_size=1, batch_wait_seconds=0.05):
        self.output = Path(output)
        self.batch_size = batch_size
        self.batch_wait_seconds = batch_wait_seconds
        self.adapter = None
        self.stage = "starting"
        self.identity_sha256 = None
        self.pending = {}
        self.completed = OrderedDict()
        self.work = queue.Queue()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self.completed_calls = 0
        self.failed_calls = 0
        self.fatal_error = None

    def health(self):
        with self.lock:
            return {"stage": self.stage, "identity_sha256": self.identity_sha256,
                    "queued_or_running": len(self.pending), "completed_calls": self.completed_calls,
                    "failed_calls": self.failed_calls, "batch_size": self.batch_size,
                    "fatal_error": self.fatal_error}

    def set_stage(self, stage):
        self.stage = stage
        atomic_json(self.output / "server_status.json", {**self.health(), "updated_unix": time.time()})
        print(canonical({"server_stage": stage}), flush=True)

    def start_worker(self, adapter):
        supported = int(getattr(adapter, "supported_batch_size", 1))
        if self.batch_size > supported:
            raise ValueError(f"Requested batch {self.batch_size} exceeds validated adapter limit {supported}")
        self.adapter = adapter
        self.worker = threading.Thread(target=self._run, name="native-model-owner", daemon=True)
        self.worker.start()

    def submit(self, request):
        digest = request_sha(request)
        with self.lock:
            if self.stage != "ready" or self.stop.is_set():
                raise RuntimeError("Model server is not ready: " + self.stage)
            if digest in self.completed:
                item = self.completed.pop(digest)
                self.completed[digest] = item
                return item
            if digest in self.pending:
                return self.pending[digest]
            item = Pending(request, digest)
            self.pending[digest] = item
            self.work.put(item)
            return item

    def _finish(self, item, response):
        if not isinstance(response, dict):
            response = {"error": {"type": "AdapterContractError", "category": "infrastructure",
                                  "message": "Adapter response is not an object"}}
        response.setdefault("raw_request_sha256", item.digest)
        if response["raw_request_sha256"] != item.digest:
            raise ValueError("Adapter response is bound to a different request")
        item.response = response
        with self.lock:
            self.pending.pop(item.digest, None)
            self.completed[item.digest] = item
            while len(self.completed) > 1024:
                self.completed.popitem(last=False)
            self.completed_calls += 1
            self.failed_calls += int("error" in response)
        item.done.set()

    def _run(self):
        while not self.stop.is_set() or not self.work.empty():
            try:
                first = self.work.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.batch_wait_seconds
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.work.get(timeout=max(0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            try:
                responses = self.adapter.generate_batch([item.request for item in batch])
                if not isinstance(responses, list) or len(responses) != len(batch):
                    raise ValueError("Adapter did not return exactly one response per request")
                for item, response in zip(batch, responses):
                    self._finish(item, response)
            except Exception as error:
                text = str(error)
                infrastructure = getattr(error, "category", None) == "infrastructure" or isinstance(error, (MemoryError, SystemError)) or any(
                    fragment in text.lower() for fragment in
                    ("cuda", "out of memory", "device-side", "triton", "illegal memory", "adapter did not",
                     "different request", "adapter response"))
                detail = {"type": type(error).__name__, "message": text,
                          "category": "infrastructure" if infrastructure else "model",
                          "traceback": traceback.format_exc()}
                atomic_json(self.output / "server_errors" / f"{time.time_ns()}.json",
                            {"error": detail, "requests": [item.digest for item in batch]})
                for item in batch:
                    if not item.done.is_set():
                        retained = getattr(error, "response", None)
                        response = dict(retained) if isinstance(retained, dict) and len(batch) == 1 else {}
                        response["error"] = {**response.get("error", {}), **detail}
                        self._finish(item, response)
                if infrastructure:
                    self.fatal_error = detail
                    self.set_stage("failed")
                    self.stop.set()
                    while not self.work.empty():
                        item = self.work.get_nowait()
                        self._finish(item, {"error": detail})
                    return
            if self.completed_calls % 10 == 0:
                atomic_json(self.output / "server_status.json", {**self.health(), "updated_unix": time.time()})
                print(canonical({"server_progress": self.health()}), flush=True)

    def request_shutdown(self, expected_identity):
        with self.lock:
            if expected_identity != self.identity_sha256:
                raise ValueError("Shutdown ownership identity differs")
            if self.pending:
                raise RuntimeError("Requests are still queued or running")
            self.stage = "closing"
            self.stop.set()


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def send_json(self, code, value):
            body = canonical(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The queued request remains owned and can be recovered.

        def do_GET(self):
            self.send_json(200 if self.path == "/health" else 404,
                           engine.health() if self.path == "/health" else {"error": "Unknown endpoint"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError("Invalid JSON request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/shutdown":
                    engine.request_shutdown(request.get("identity_sha256"))
                    self.send_json(200, {"status": "closing"})
                    return
                if self.path != "/generate":
                    self.send_json(404, {"error": "Unknown endpoint"})
                    return
                item = engine.submit(request)
                while not item.done.wait(5):
                    if engine.stage == "failed":
                        self.send_json(503, {"error": engine.fatal_error, "health": engine.health()})
                        return
                self.send_json(200, item.response)
            except ValueError as error:
                self.send_json(400, {"error": {"type": type(error).__name__, "message": str(error)}})
            except RuntimeError as error:
                self.send_json(503, {"error": {"type": type(error).__name__, "message": str(error)},
                                     "health": engine.health()})
    return Handler


def stable_identity(args, adapter):
    from deepseek_role_probe_model import audit_model
    from deepseek_role_probe_train import verify_freeze
    base = Path(__file__).resolve().parent
    frozen = verify_freeze(args.probe_dir)
    features = json.loads((args.probe_dir / "features/manifest.json").read_text())
    files = [Path(__file__), base / "deepseek_agentdojo_model.py", base / "deepseek_agentdojo_protocol.py",
             base / "deepseek_agentdojo_preflight.py",
             base / "deepseek_role_probe_model.py", base / "deepseek_role_probe_train.py"]
    return {"format": "deepseek-agentdojo-server-v1", "model_dir": str(args.model.resolve()),
            "probe_dir": str(args.probe_dir.resolve()), "checkpoint_sha256": frozen["probe_sha256"],
            "probe_freeze_sha256": file_sha(args.probe_dir / "freeze.json"),
            "model_binding": features["model_binding"],
            "backbone_audit": audit_model(adapter.model),
            "generation": {"temperature": 0.0, "seed": 0, "max_new_tokens": args.max_new_tokens,
                           "max_model_len": args.max_model_len, "thinking_mode": "thinking",
                           "reasoning_effort": "low", "preserve_reasoning_history": True},
            "batch_size": args.batch_size, "batch_wait_seconds": args.batch_wait_seconds,
            "package_versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "kernels", "numpy", "triton")},
            "implementation_sha256": {str(path): file_sha(path) for path in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731"))
    parser.add_argument("--probe-dir", type=Path, default=base / "outputs/deepseek-v4-flash-0731-role-probe-20261115")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-wait-seconds", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--preflight-module", default="deepseek_agentdojo_preflight")
    args = parser.parse_args()
    args.output = args.output.resolve(); args.probe_dir = args.probe_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / ".server.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.batch_size < 1 or args.max_new_tokens != 1024 or args.max_model_len != 32768:
        raise ValueError("The fixed reference budget is 1024 output tokens / 32768 total context")
    engine = ModelQueue(args.output, batch_size=args.batch_size, batch_wait_seconds=args.batch_wait_seconds)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    http_thread = threading.Thread(target=httpd.serve_forever, name="local-model-http", daemon=True)
    http_thread.start()
    adapter = None
    exit_code = 1
    try:
        import os
        atomic_json(args.output / "server_launch.json", {"pid": os.getpid(), "started_unix": time.time(),
                    "port": args.port, "source_sha256": file_sha(__file__)})
        engine.set_stage("loading")
        from deepseek_agentdojo_model import DeepSeekAgentDojoModel
        adapter = DeepSeekAgentDojoModel(model_dir=args.model, probe_dir=args.probe_dir, output_dir=args.output,
                    max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
                    thinking_mode="thinking", reasoning_effort="low")
        engine.set_stage("preflight")
        preflight = importlib.import_module(args.preflight_module).run_preflight(adapter, args.output)
        if preflight.get("status") != "passed":
            raise ValueError("Native cached-decode preflight has not passed")
        identity = stable_identity(args, adapter)
        identity_path = args.output / "server_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Server model/probe/implementation identity changed on resume")
        if not identity_path.exists():
            atomic_json(identity_path, identity)
        engine.identity_sha256 = file_sha(identity_path)
        engine.start_worker(adapter)
        engine.set_stage("ready")
        signal.signal(signal.SIGTERM, lambda signum, frame: engine.stop.set())
        while not engine.stop.wait(0.5):
            pass
        if engine.worker:
            engine.worker.join()
        if engine.fatal_error:
            raise RuntimeError("Model worker failed: " + engine.fatal_error["message"])
        from deepseek_role_probe_model import audit_model, memory_report
        atomic_json(args.output / "server_runtime_final.json", {
            "identity_sha256": engine.identity_sha256, "finished_unix": time.time(),
            "backbone_audit": audit_model(adapter.model), "gpu_memory": memory_report(),
            "requests_completed": engine.completed_calls, "requests_failed": engine.failed_calls})
        exit_code = 0
        engine.set_stage("complete")
    except BaseException as error:
        atomic_json(args.output / "server_failure.json", {"type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(), "at_unix": time.time()})
        engine.set_stage("failed")
        raise
    finally:
        engine.stop.set()
        httpd.shutdown(); httpd.server_close()
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()
        atomic_json(args.output / "server_exit.json", {"exit_code": exit_code, "finished_unix": time.time(),
                    "identity_sha256": engine.identity_sha256})


if __name__ == "__main__":
    main()
