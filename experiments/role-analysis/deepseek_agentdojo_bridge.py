"""CPU-only native AgentDojo episodes backed by a separate DeepSeek HTTP server.

The benchmark, tools, execution loop, and evaluators come from the pinned
ChatInject checkout. Only model requests cross HTTP. Native initial errors are
terminal, while unavailable transport leaves the same request pending. No model
or GPU library is imported by this module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from urllib import error as urlerror
from urllib import request as urlrequest

PIPELINE_NAME = "deepseek_v4_flash_role_probe"
DEFAULT_REPO = Path.home() / "Github/delimeter-token"
HERE = Path(__file__).resolve().parent


class IntegrityError(RuntimeError):
    """Invalid configuration or changed artifacts: abort, never score as failure."""


class NativeModelError(RuntimeError):
    """A completed model request was malformed or returned a native error."""


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def artifact(root, value):
    root = Path(root).resolve()
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(resolved.is_relative_to(root), "Artifact path escapes this run")
    return resolved


def reference(root, path):
    path = artifact(root, path)
    return {"path": str(path.relative_to(Path(root).resolve())), "sha256": sha(path)}


def verify_reference(root, ref):
    path = artifact(root, ref["path"])
    require(path.is_file() and sha(path) == ref["sha256"], "Artifact checksum differs: " + str(path))
    return path


def load_dependencies(repo):
    source = Path(repo).resolve() / "indirect-prompt-injection/ChatInject/src"
    sys.path.insert(0, str(source))
    import agentdojo
    require(Path(agentdojo.__file__).resolve().is_relative_to(source), "AgentDojo was imported from another checkout")
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.attacks.base_attacks import FixedJailbreakAttack
    from agentdojo.benchmark import run_task_with_injection_tasks
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.logging import LOGGER_STACK, OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.types import text_content_block_from_string
    return locals()


def source_identity(repo, protocol):
    source = Path(repo).resolve() / "indirect-prompt-injection/ChatInject/src"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(revision == protocol.CHATINJECT_REVISION, "Pinned AgentDojo revision differs")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    require(not dirty.strip(), "Pinned AgentDojo has tracked changes")
    package = source / "agentdojo"
    files = {str(p.relative_to(source)): sha(p) for p in sorted(package.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".yaml", ".yml", ".json", ".txt", ".jinja"}}
    return {"package_root": str(package), "revision": revision, "files_sha256": files,
            "identity_sha256": object_sha({"revision": revision, "files_sha256": files})}


def semantic_messages(messages):
    """Convert original benchmark roles directly, without GPT-OSS role mapping."""
    result = []
    for message in messages:
        role = message["role"]
        require(role in {"system", "user", "assistant", "tool", "developer"}, "Unknown semantic role")
        content = message.get("content")
        if isinstance(content, list):
            require(all(block.get("type") == "text" for block in content), "Nontext AgentDojo message")
            content = "".join(block.get("content", block.get("text", "")) or "" for block in content)
        require(content is None or isinstance(content, str), "Invalid message content")
        if role == "tool" and message.get("error"):
            content = message["error"]
        row = {"role": role, "content": content}
        if "reasoning_content" in message:
            require(isinstance(message["reasoning_content"], str), "Invalid persisted assistant reasoning")
            row["reasoning_content"] = message["reasoning_content"]
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                require(call.id is not None, "Native tool call lacks ID")
                calls.append({"id": call.id, "type": "function", "function": {
                    "name": call.function, "arguments": json.dumps(call.args, ensure_ascii=False)}})
            row["tool_calls"] = calls
        if role == "tool":
            require(message.get("tool_call_id") is not None, "Native tool result lacks call ID")
            row["tool_call_id"] = message["tool_call_id"]
            if message.get("tool_call") is not None:
                row["name"] = message["tool_call"].function
        result.append(row)
    return result


def assistant_message(response, call_id, dependencies):
    """Use model JSON arguments unchanged; malformed arguments stay errors."""
    if response.get("error") is not None:
        raise NativeModelError(canonical(response["error"]))
    content, reasoning = response.get("content"), response.get("reasoning_content", "")
    if content is not None and not isinstance(content, str):
        raise NativeModelError("Model content must be text or null")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        raise NativeModelError("Model reasoning_content must be text")
    calls = []
    for index, call in enumerate(response.get("tool_calls") or []):
        function = call.get("function", call)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError) as error:
                raise NativeModelError("Model tool arguments are invalid JSON") from error
        if not isinstance(arguments, dict):
            raise NativeModelError("Model tool arguments must decode to an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise NativeModelError("Model tool name must be nonempty text")
        identifier = call.get("id") or "call_" + hashlib.sha256((call_id + ":" + str(index)).encode()).hexdigest()[:24]
        if not isinstance(identifier, str):
            raise NativeModelError("Model tool ID must be text")
        calls.append(dependencies["FunctionCall"](function=name, args=arguments, id=identifier))
    if len({call.id for call in calls}) != len(calls):
        raise NativeModelError("Duplicate model tool-call IDs")
    return {"role": "assistant", "content": [dependencies["text_content_block_from_string"](content)]
            if content is not None else None, "tool_calls": calls or None, "reasoning_content": reasoning}


def unicode_tool_formatter(result):
    """Keep the native YAML representation, allowing literal model delimiters."""
    from functools import partial
    import yaml
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    return tool_result_to_str(result, dump_fn=partial(yaml.safe_dump, allow_unicode=True))


def install_unicode_tool_formatter(pipeline):
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
    count = 0

    def visit(element):
        nonlocal count
        if isinstance(element, ToolsExecutor):
            element.output_formatter = unicode_tool_formatter
            count += 1
        for child in getattr(element, "elements", ()):
            visit(child)

    visit(pipeline)
    require(count == 1, "Native pipeline must contain exactly one tool executor")
    return pipeline


class HttpGenerator:
    """Persist calls separately from GPU artifacts, then replay exact receipts."""
    def __init__(self, root, endpoint, identity_sha256, *, server_output_root=None, timeout=3600, retry_seconds=15):
        self.root, self.endpoint = Path(root).resolve(), endpoint
        self.server_output_root = Path(server_output_root).resolve() if server_output_root is not None else self.root
        require(self.root.is_relative_to(self.server_output_root), "Bridge scope must lie inside the server output root")
        self.identity_sha256 = identity_sha256
        self.timeout, self.retry_seconds = timeout, retry_seconds
        self.stopped = threading.Event()

    def abort(self):
        self.stopped.set()

    def _native_only(self, response):
        error = response.get("error")
        require(not (isinstance(error, dict) and (error.get("category") == "infrastructure" or error.get("fatal") is True)),
                "GPU server infrastructure failure; retain request pending: " + canonical(error))

    def _references(self, response, *, optional=False, expected_request=None):
        if "artifact_refs" in response:
            refs = response["artifact_refs"]
            require(isinstance(refs, dict) and "directory" in refs and "receipt" in refs, "Malformed native model artifact_refs")
            folder = artifact(self.server_output_root, refs["directory"])
            receipt_path = artifact(self.server_output_root, refs["receipt"])
            require(folder.is_relative_to(self.root) and receipt_path == folder / "receipt.json", "Native model receipt crosses bridge scope")
            receipt = read(receipt_path)
            require(receipt["format_version"] == "deepseek-agentdojo-call-v1", "Unsupported native model receipt")
            digest = object_sha(expected_request) if expected_request is not None else response.get("raw_request_sha256")
            require(isinstance(digest, str) and receipt["raw_request_sha256"] == response.get("raw_request_sha256") == digest,
                    "Native model response/receipt/request SHA differs")
            if expected_request is not None:
                require(receipt["metadata"] == expected_request["metadata"], "Native model receipt metadata differs")
                require(folder == Path(expected_request["metadata"]["recording_dir"]).resolve(), "Native model recording directory differs")
            require(receipt["status"] == ("error" if response.get("error") is not None else "complete"), "Native model receipt status differs")
            backend = self.server_output_root / "generation-backend.json"
            require(backend.is_file() and receipt["identity_sha256"] == object_sha(read(backend)), "Native adapter backend identity differs")
            files = receipt["files"]
            require(isinstance(files, dict) and {"request.json", "response.json"} <= set(files), "Native receipt omits source request/response")
            result = [reference(self.root, receipt_path)]
            for name, expected in sorted(files.items()):
                path = artifact(folder, name)
                require(path.is_relative_to(self.root) and sha(path) == expected, "Native model artifact checksum differs: " + str(path))
                result.append({"path": str(path.relative_to(self.root)), "sha256": expected})
            actual_files = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
            require(actual_files == set(files) | {"receipt.json"}, "Native model directory has uncommitted/missing files")
            require(read(folder / "response.json") == response, "HTTP response differs from committed native response")
            model_request = read(folder / "request.json")
            require(model_request["raw_request_sha256"] == digest and model_request["metadata"] == receipt["metadata"], "Native model request binding differs")
            for label, value in refs.items():
                if label != "directory":
                    path = artifact(self.server_output_root, value)
                    require(path.parent == folder and (path == receipt_path or path.name in files), "Native response points to an uncommitted artifact")
            return result
        refs = response.get("_role_probe_artifacts", response.get("artifacts"))
        if optional and not refs:
            return []
        require(isinstance(refs, list) and bool(refs), "Completed model response lacks artifact references")
        result = []
        for ref in refs:
            path = verify_reference(self.root, ref)
            result.append(reference(self.root, path))
        require(len({item["path"] for item in result}) == len(result), "Duplicate model artifact reference")
        return result

    def verify_call(self, call_id, expected_request=None):
        folder = artifact(self.root, "bridge_calls/" + call_id)
        receipt = read(folder / "receipt.json")
        require(receipt["server_identity_sha256"] == self.identity_sha256 and receipt["call_id"] == call_id,
                "Completed bridge call belongs to another server/run")
        request = read(verify_reference(self.root, receipt["request"]))
        require(object_sha(request) == receipt["raw_request_sha256"], "Recorded request hash differs")
        if expected_request is not None:
            require(request == expected_request, "Resumed original model request differs")
        wrapper = read(verify_reference(self.root, receipt["response"]))
        require(wrapper["raw_request_sha256"] == receipt["raw_request_sha256"], "Response/request binding differs")
        for ref in receipt["artifacts"]:
            verify_reference(self.root, ref)
        response = wrapper["response"]
        self._native_only(response)
        require(self._references(response, optional=response.get("error") is not None, expected_request=request) == receipt["artifacts"],
                "Response artifact list changed")
        return response, reference(self.root, folder / "receipt.json")

    def _post(self, payload, folder):
        attempts = 0
        while True:
            require(not self.stopped.is_set(), "Bridge stopped after infrastructure/integrity failure")
            attempts += 1
            try:
                health_url = self.endpoint.rsplit("/", 1)[0] + "/health"
                with urlrequest.urlopen(health_url, timeout=min(self.timeout, 30)) as stream:
                    health = json.loads(stream.read())
                require(health.get("identity_sha256") == self.identity_sha256, "HTTP server identity changed")
                require(health.get("stage") != "failed", "GPU server reports failed infrastructure stage")
                require(not self.stopped.is_set(), "Bridge stopped before submitting another model request")
                request = urlrequest.Request(self.endpoint, data=canonical(payload).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urlrequest.urlopen(request, timeout=self.timeout) as stream:
                    response = json.loads(stream.read())
                require(isinstance(response, dict), "Server returned a non-object response")
                self._native_only(response)
                return response
            except (ValueError, TypeError) as error:
                raise IntegrityError("HTTP server returned malformed JSON") from error
            except urlerror.HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise IntegrityError("HTTP server rejected original request: " + str(error)) from error
                detail = str(error)
            except (urlerror.URLError, TimeoutError, ConnectionError, OSError) as error:
                detail = str(error)
            write(folder / "transport_pending.json", {"status": "pending_same_request", "attempts": attempts,
                  "raw_request_sha256": object_sha(payload), "error": detail, "updated_unix": time.time()})
            emit("transport_pending", call_id=payload["metadata"]["call_id"], attempts=attempts, error=detail)
            time.sleep(self.retry_seconds)

    def __call__(self, messages, tools, metadata):
        require(not self.stopped.is_set(), "Bridge stopped after infrastructure/integrity failure")
        payload = {"messages": messages, "tools": tools, "metadata": metadata}
        call_id = metadata["call_id"]
        folder = artifact(self.root, "bridge_calls/" + call_id)
        folder.mkdir(parents=True, exist_ok=True)
        request_path, response_path = folder / "original_request.json", folder / "response.json"
        request_sha = object_sha(payload)
        if (folder / "receipt.json").exists():
            return self.verify_call(call_id, payload)
        if request_path.exists():
            require(read(request_path) == payload, "Interrupted request differs on resume")
        else:
            write(request_path, payload)
        refs = None
        if response_path.exists():
            wrapper = read(response_path)
            require(wrapper["raw_request_sha256"] == request_sha, "Interrupted response belongs to another request")
            response = wrapper["response"]
        else:
            response = self._post(payload, folder)
            if "raw_request_sha256" in response:
                require(response["raw_request_sha256"] == request_sha, "Server request binding differs")
            refs = self._references(response, optional=response.get("error") is not None, expected_request=payload)
            write(response_path, {"raw_request_sha256": request_sha, "response": response})
        self._native_only(response)
        if refs is None:
            refs = self._references(response, optional=response.get("error") is not None, expected_request=payload)
        receipt = {"schema": "deepseek-agentdojo-bridge-call-v1", "call_id": call_id,
            "status": "native_error" if response.get("error") is not None else "complete",
            "raw_request_sha256": request_sha, "server_identity_sha256": self.identity_sha256,
            "request": reference(self.root, request_path), "response": reference(self.root, response_path), "artifacts": refs}
        write(folder / "receipt.json", receipt)
        (folder / "transport_pending.json").unlink(missing_ok=True)
        return response, reference(self.root, folder / "receipt.json")


def make_pipeline(generator, root, case, calls, dependencies):
    class BridgeLLM(dependencies["BasePipelineElement"]):
        name = PIPELINE_NAME

        def query(self, query, runtime, env, messages=(), extra_args=None):
            index = len(calls)
            call_id = case["case_id"] + f"/call_{index:04d}"
            metadata = {**case, "call_index": index, "call_id": call_id,
                "recording_dir": str(Path(root).resolve() / "calls" / call_id),
                "temperature": 0.0, "seed": 0, "max_tokens": 1024, "max_model_len": 32768,
                "thinking_mode": "thinking", "reasoning_effort": "low"}
            tools = [{"type": "function", "function": {"name": tool.name, "description": tool.description,
                       "parameters": tool.parameters.model_json_schema()}} for tool in runtime.functions.values()]
            record = {"call_id": call_id, "call_index": index, "status": "running"}
            calls.append(record)
            response, ref = generator(semantic_messages(messages), tools, metadata)
            record["bridge_receipt"] = ref
            try:
                output = assistant_message(response, call_id, dependencies)
            except NativeModelError as error:
                record.update(status="native_error", error=str(error))
                raise
            record.update(status="complete", finish_reason=response.get("finish_reason"), usage=response.get("usage"))
            return query, runtime, env, [*messages, output], extra_args or {}

    pipeline = dependencies["AgentPipeline"].from_config(dependencies["PipelineConfig"](
        llm=BridgeLLM(), model_id=None, defense=None, tool_delimiter="tool", system_message_name=None,
        system_message=None, tool_output_format=None, provider_config=None))
    return install_unicode_tool_formatter(pipeline)


def metrics(records, expected):
    scored = [r for r in records if r["status"] == "complete"]
    successes = sum(r["security"] for r in scored)
    useful = sum(r["utility"] for r in scored)
    count = len(scored)
    return {"expected_cases": expected, "terminal_cases": len(records), "scored_cases": count,
        "error_cases": sum(r["status"] == "error" for r in records), "missing_cases": expected - len(records),
        "attack_successes": successes, "utility_successes": useful, "data_fully_scored": count == expected,
        "asr": successes / expected if expected and count == expected else None,
        "utility": useful / expected if expected and count == expected else None,
        "asr_scored": successes / count if count else None, "utility_scored": useful / count if count else None}


def smoke_cases(cases):
    """First canonical pair per condition/suite (Slack injection IDs start at 1)."""
    result, seen = [], set()
    for case in cases:
        key = (case["condition"], case["suite"])
        if key not in seen:
            seen.add(key)
            result.append(case)
    require(len(result) == 6, "Expected six canonical smoke episodes")
    return result


def verify_case(root, path, case, spec_sha, generator):
    record = read(path)
    require(record["spec_sha256"] == spec_sha and all(record.get(k) == v for k, v in case.items()), "Terminal case identity differs")
    require(record["status"] in {"complete", "error"}, "Existing case is not terminal")
    if record["status"] == "complete":
        require(type(record["utility"]) is bool and type(record["security"]) is bool, "Invalid terminal native scores")
        require(record.get("raw_trace") and record["requests"], "Scored case lacks native evidence")
    else:
        require(record["utility"] is None and record["security"] is None and record.get("error"), "Native errors must retain null scores")
    if record.get("raw_trace"):
        trace = read(verify_reference(root, record["raw_trace"]))
        if record["status"] == "complete":
            require(not trace.get("error") and trace["utility"] == record["utility"] and trace["security"] == record["security"], "Native trace/evaluation differs")
    for index, call in enumerate(record["requests"]):
        require(call["call_index"] == index and call["call_id"] == case["case_id"] + f"/call_{index:04d}", "Terminal case call sequence differs")
        if "bridge_receipt" in call:
            verify_reference(root, call["bridge_receipt"])
            _, ref = generator.verify_call(call["call_id"])
            require(ref == call["bridge_receipt"], "Terminal call receipt differs")
        else:
            require(record["status"] == "error", "Scored case has an uncommitted request")
    return record


def recover_native_trace(root, trace_path, case, spec_sha, generator, dependencies, protocol):
    """Recover the trace/case commit gap; archive genuine interrupted prefixes.

    Native TraceLogger writes after each turn, before final evaluation. A JSON
    file alone therefore does not make an episode terminal. Only saved native
    scores or an already completed native-error response establish that state.
    """
    trace = read(trace_path)
    expected = {"suite_name": case["suite"], "user_task_id": case["user_task_id"],
        "injection_task_id": case["injection_task_id"], "pipeline_name": PIPELINE_NAME,
        "attack_type": protocol.ATTACK_NAMES[case["condition"]], "benchmark_version": protocol.BENCHMARK_VERSION}
    require(all(trace.get(k) == v for k, v in expected.items()), "Interrupted native trace identity differs")
    require(trace.get("injections") and all(v == case["payload_text"] for v in trace["injections"].values()), "Interrupted native trace payload differs")
    calls, native_error = [], trace.get("error")
    folders = sorted((root / "bridge_calls" / case["case_id"]).glob("call_*"))
    for index, folder in enumerate(folders):
        call_id = case["case_id"] + f"/call_{index:04d}"
        require(folder.name == f"call_{index:04d}", "Interrupted bridge call numbering differs")
        request_path = folder / "original_request.json"
        require(request_path.exists(), "Interrupted bridge call lacks original request")
        request = read(request_path)
        require(all(request["metadata"].get(k) == v for k, v in case.items())
                and request["metadata"]["call_id"] == call_id, "Interrupted bridge request case differs")
        call = {"call_id": call_id, "call_index": index, "status": "running"}
        if (folder / "receipt.json").exists():
            response, ref = generator.verify_call(call_id, request)
            call["bridge_receipt"] = ref
            try:
                assistant_message(response, call_id, dependencies)
                call.update(status="complete", finish_reason=response.get("finish_reason"), usage=response.get("usage"))
            except NativeModelError as error:
                native_error = str(error)
                call.update(status="native_error", error=str(error))
        calls.append(call)
    terminal_scores = type(trace.get("utility")) is bool and type(trace.get("security")) is bool
    if terminal_scores or native_error:
        require(calls, "Native terminal trace lacks bridge calls")
        if not native_error:
            require(all(c["status"] == "complete" for c in calls), "Native scores precede an uncommitted call")
        return {**case, "spec_sha256": spec_sha, "status": "error" if native_error else "complete",
            "utility": None if native_error else trace["utility"], "security": None if native_error else trace["security"],
            **({"error": "Recovered original native error: " + str(native_error)} if native_error else {}),
            "injections": trace["injections"], "raw_trace": reference(root, trace_path), "requests": calls,
            "duration_seconds": trace.get("duration"), "recovered_native_terminal_trace": True}
    digest = sha(trace_path)
    archive = root / "interrupted_traces" / case["case_id"] / (digest + ".json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        require(sha(archive) == digest, "Archived interrupted trace checksum differs")
        trace_path.unlink()
    else:
        trace_path.replace(archive)
    return reference(root, archive)


def run_benchmark(root, cases, identity, generator, dependencies, protocol, *, workers=8):
    root = Path(root).resolve()
    require(workers > 0, "Worker count must be positive")
    spec_sha = object_sha(identity)
    suites = {name: dependencies["get_suite"](protocol.BENCHMARK_VERSION, name) for name in {c["suite"] for c in cases}}

    def execute(case):
        require(not generator.stopped.is_set(), "Bridge stopped before starting another native episode")
        path = root / "cases" / (case["case_id"] + ".json")
        if path.exists():
            return verify_case(root, path, case, spec_sha, generator)
        calls = []
        pipeline = make_pipeline(generator, root, case, calls, dependencies)
        suite = suites[case["suite"]]
        attack = dependencies["FixedJailbreakAttack"](case["payload_template"], suite, pipeline)
        attack.name = protocol.ATTACK_NAMES[case["condition"]]
        trace_path = root / "raw" / PIPELINE_NAME / case["suite"] / case["user_task_id"] / attack.name / (case["injection_task_id"] + ".json")
        partial_trace = None
        if trace_path.exists():
            recovered = recover_native_trace(root, trace_path, case, spec_sha, generator, dependencies, protocol)
            if recovered.get("status") in {"complete", "error"}:
                write(path, recovered)
                return verify_case(root, path, case, spec_sha, generator)
            partial_trace = recovered
        record = {**case, "spec_sha256": spec_sha, "status": "running", "utility": None, "security": None}
        if partial_trace:
            record["resumed_partial_trace"] = partial_trace
        started = time.time()
        token = dependencies["LOGGER_STACK"].set([])
        try:
            with dependencies["OutputLogger"](str(root / "raw")):
                utility, security = dependencies["run_task_with_injection_tasks"](
                    suite, pipeline, suite.user_tasks[case["user_task_id"]], attack, root / "raw", True,
                    [case["injection_task_id"]], protocol.BENCHMARK_VERSION)
            trace = read(trace_path)
            if trace.get("error"):
                raise NativeModelError("Native AgentDojo error: " + str(trace["error"]))
            require(trace["injections"] and all(v == case["payload_text"] for v in trace["injections"].values()), "Native slot payload differs")
            require(calls and all(c["status"] == "complete" for c in calls), "Native scored case lacks complete model calls")
            key = (case["user_task_id"], case["injection_task_id"])
            require(type(utility[key]) is bool and type(security[key]) is bool, "Native evaluator returned nonboolean scores")
            record.update(status="complete", utility=utility[key], security=security[key],
                          injections=trace["injections"])
        except IntegrityError:
            raise
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        finally:
            dependencies["LOGGER_STACK"].reset(token)
            if record["status"] in {"complete", "error"}:
                if trace_path.exists():
                    record["raw_trace"] = reference(root, trace_path)
                record.update(requests=calls, duration_seconds=time.time() - started)
                write(path, record)
        return verify_case(root, path, case, spec_sha, generator)

    records, started = [], time.time()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deepseek-agentdojo-cpu")
    futures = []
    try:
        futures = [executor.submit(execute, case) for case in cases]
        for future in as_completed(futures):
            records.append(future.result())
            write(root / "progress.json", {"stage": "native_benchmark", "finished": len(records),
                "scheduled": len(cases), "errors": sum(r["status"] == "error" for r in records),
                "last_case": records[-1]["case_id"], "updated_unix": time.time()})
            emit("case_terminal", case_id=records[-1]["case_id"], status=records[-1]["status"], finished=len(records), total=len(cases))
    except BaseException:
        generator.abort()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    complete = len(records) == len(cases)
    scored = complete and all(r["status"] == "complete" for r in records)
    summary = {"schema": "deepseek-agentdojo-native-summary-v1", "spec_sha256": spec_sha,
        "status": "complete" if scored else "complete_with_initial_errors" if complete else "running",
        "execution_complete": complete, "data_fully_scored": scored, "scheduled_cases": len(cases),
        "conditions": {c: metrics([r for r in records if r["condition"] == c], sum(v["condition"] == c for v in cases)) for c in protocol.CONDITIONS},
        "suite_metrics": {c: {s: metrics([r for r in records if r["condition"] == c and r["suite"] == s],
             sum(v["condition"] == c and v["suite"] == s for v in cases)) for s in sorted(suites)} for c in protocol.CONDITIONS},
        "model_requests": sum(len(r["requests"]) for r in records), "seconds_this_session": time.time() - started,
        "errors": [{"case_id": r["case_id"], "error": r["error"]} for r in records if r["status"] == "error"]}
    write(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8765/generate")
    parser.add_argument("--server-identity", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--retry-seconds", type=float, default=15)
    args = parser.parse_args()
    require(args.workers > 0 and args.timeout > 0 and args.retry_seconds >= 0, "Invalid HTTP/worker configuration")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import deepseek_agentdojo_protocol as protocol
    dependencies = load_dependencies(args.repo)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".bridge.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    server_path = (args.server_identity or root / "server_identity.json").resolve()
    require(server_path.is_file(), "Frozen server_identity.json is required")
    cases = protocol.native_cases(dependencies["get_suite"])
    require(len(cases) == 778, "Expected all 778 canonical initial episodes")
    if args.mode == "smoke":
        cases = smoke_cases(cases)
    source = source_identity(args.repo, protocol)
    identity = {"schema": "deepseek-agentdojo-native-run-v1", "mode": args.mode,
        "output_root": str(root), "benchmark_version": protocol.BENCHMARK_VERSION, "agentdojo_source": source,
        "server_identity_path": str(server_path), "server_identity_sha256": sha(server_path),
        "server_identity": read(server_path), "server_url": args.server_url, "workers": args.workers,
        "schedule_sha256": object_sha(cases), "initial_episodes": len(cases), "conditions": list(protocol.CONDITIONS),
        "payloads": protocol.PAYLOADS, "protocol_sha256": sha(Path(protocol.__file__)), "bridge_sha256": sha(__file__),
        "protocol_manifest_sha256": sha(root / "protocol.json") if (root / "protocol.json").exists() else None,
        "cpu_runtime": {"executable": str(Path(sys.executable).resolve()), "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("agentdojo", "pydantic", "openai", "pyyaml", "deepdiff")}},
        "generation": {"temperature": 0.0, "seed": 0, "max_tokens": 1024, "max_model_len": 32768,
                       "thinking_mode": "thinking", "reasoning_effort": "low"},
        "role_mapping": "Preserve original AgentDojo semantic system/user/assistant/tool roles; preserve assistant reasoning",
        "tool_formatter": {"native_converter": "agentdojo.agent_pipeline.tool_execution.tool_result_to_str",
            "dump_fn": "yaml.safe_dump", "allow_unicode": True, "other_options": "native defaults including line folding",
            "scope": "identical in both conditions", "reason": "Preserve literal native Unicode delimiters in tool results"},
        "initial_error_policy": "Immutable native errors retain null scores; no replacement or automatic native rerun",
        "transport_error_policy": "Keep identical pending request and wait; do not score transport errors",
        "storage": {"bridge_calls": "bridge_calls/<call_id>", "model_artifacts": "calls/<call_id>",
                    "native_traces": "raw", "native_cases": "cases"}}
    identity_path = root / "run_identity.json"
    if identity_path.exists():
        require(args.resume, "Existing run requires --resume")
        require(read(identity_path) == identity and read(root / "schedule.json") == cases, "Resumed run/source/server/schedule differs")
    else:
        if (root / "schedule.json").exists():
            require(read(root / "schedule.json") == cases, "Prepared canonical schedule differs")
        write(identity_path, identity)
        if not (root / "schedule.json").exists():
            write(root / "schedule.json", cases)
    generator = HttpGenerator(root, args.server_url, identity["server_identity_sha256"], server_output_root=server_path.parent,
                              timeout=args.timeout, retry_seconds=args.retry_seconds)
    try:
        summary = run_benchmark(root, cases, identity, generator, dependencies, protocol, workers=args.workers)
    except IntegrityError as error:
        write(root / "bridge_infrastructure_failure.json", {"status": "blocked_infrastructure_or_integrity",
            "spec_sha256": object_sha(identity), "error": str(error), "updated_unix": time.time(),
            "native_error_scoring_performed": False})
        raise
    emit("benchmark_terminal", status=summary["status"], data_fully_scored=summary["data_fully_scored"], cases=len(cases))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
