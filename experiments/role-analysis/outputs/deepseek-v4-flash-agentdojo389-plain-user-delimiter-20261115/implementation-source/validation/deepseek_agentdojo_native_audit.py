"""Independent CPU replay of native AgentDojo evaluation from saved completions.

No producer converter, metrics implementation, model, or HTTP service is used.
Every native tool actually executes in a fresh mock environment. Only the
external send_email clock is bound to its exact recorded tool-result timestamp.
"""
from __future__ import annotations

import argparse
import copy
import datetime
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REVISION = "c6832ab413ebb4728f7b18640702df542e8b4142"
VERSION = "v1.2.1"


class AuditError(ValueError):
    pass


class NativeModelError(RuntimeError):
    pass


def require(condition, detail):
    if not condition:
        raise AuditError(detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def bound_path(root, reference, bindings):
    root, value = Path(root).resolve(), Path(reference["path"])
    path = (root / value).resolve() if not value.is_absolute() else value.resolve()
    require(path.is_relative_to(root) and path.is_file(), "Reference leaves the run or is missing")
    digest = sha(path)
    require(digest == reference["sha256"], "Referenced file SHA differs: " + str(path))
    bindings[str(path.relative_to(root))] = digest
    return path


def dependencies(repo):
    source = Path(repo).resolve() / "indirect-prompt-injection/ChatInject/src"
    sys.path.insert(0, str(source))
    import agentdojo
    require(Path(agentdojo.__file__).resolve().is_relative_to(source), "Native package comes from another checkout")
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
    from agentdojo.attacks.base_attacks import FixedJailbreakAttack
    from agentdojo.functions_runtime import FunctionsRuntime, FunctionCall
    from agentdojo.logging import Logger, LOGGER_STACK
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.types import text_content_block_from_string
    from agentdojo.default_suites.v1.tools import email_client
    return locals()


def independent_messages(messages):
    """Interpret native messages independently, preserving semantic roles."""
    output = []
    for message in messages:
        role = message["role"]
        body = message.get("content")
        if isinstance(body, list):
            require(all(item.get("type") == "text" for item in body), "Unexpected nontext native message")
            body = "".join((item.get("content", item.get("text", "")) or "") for item in body)
        row = {"role": role, "content": message.get("error") or body if role == "tool" else body}
        if "reasoning_content" in message:
            row["reasoning_content"] = message["reasoning_content"]
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                value = call.model_dump() if hasattr(call, "model_dump") else call
                calls.append({"id": value["id"], "type": "function", "function": {
                    "name": value["function"], "arguments": json.dumps(value["args"], ensure_ascii=False)}})
            row["tool_calls"] = calls
        if role == "tool":
            row["tool_call_id"] = message["tool_call_id"]
            call = message.get("tool_call")
            if call is not None:
                row["name"] = call.function if hasattr(call, "function") else call["function"]
        output.append(row)
    return output


def independent_assistant(response, call_id, native):
    error = response.get("error")
    if error is not None:
        require(not (isinstance(error, dict) and (error.get("category") == "infrastructure" or error.get("fatal") is True)),
                "Infrastructure error was incorrectly sealed as a native case")
        raise NativeModelError(canonical(error))
    content, reasoning = response.get("content"), response.get("reasoning_content") or ""
    if content is not None and not isinstance(content, str):
        raise NativeModelError("Model content must be text or null")
    if not isinstance(reasoning, str):
        raise NativeModelError("Model reasoning_content must be text")
    calls = []
    for index, original in enumerate(response.get("tool_calls") or []):
        function = original.get("function", original)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError) as error:
                raise NativeModelError("Model tool arguments are invalid JSON") from error
        if not isinstance(arguments, dict):
            raise NativeModelError("Model tool arguments must decode to an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise NativeModelError("Model tool name must be nonempty text")
        identifier = original.get("id") or "call_" + hashlib.sha256((call_id + ":" + str(index)).encode()).hexdigest()[:24]
        if not isinstance(identifier, str):
            raise NativeModelError("Model tool ID must be text")
        calls.append(native["FunctionCall"](function=name, args=arguments, id=identifier))
    if len({call.id for call in calls}) != len(calls):
        raise NativeModelError("Duplicate model tool-call IDs")
    return {"role": "assistant", "content": None if content is None else [native["text_content_block_from_string"](content)],
            "tool_calls": calls or None, "reasoning_content": reasoning}


def validate_model_error(request, identity):
    """Reproduce the recorded parser/budget failure from its actual evidence."""
    response = request["response"]
    error = response.get("error")
    if not isinstance(error, dict):
        return
    require(error.get("category") == "model" and error.get("fatal") is not True, "Non-native error cannot receive a native score")
    kind = error["type"]
    generation = identity["generation"]
    if kind == "NativeContextOverflow":
        require(response["input_tokens"] + generation["max_tokens"] > generation["max_model_len"]
                and response.get("forward_started") is False, "Saved context-overflow condition does not reproduce")
    elif kind == "IncompleteNativeCompletion":
        require(response["finish_reason"] == "length" and response["output_tokens"] == generation["max_tokens"]
                and isinstance(response.get("raw_completion"), str), "Saved generation-budget failure does not reproduce")
    elif kind == "NativeCompletionParseError":
        server = identity["server_identity"]
        path = Path(server["model_dir"]) / "encoding/encoding_dsv4.py"
        require(sha(path) == server["model_binding"]["official_encoder_sha256"], "Official parser SHA differs")
        spec = importlib.util.spec_from_file_location("_independent_native_audit_deepseek_encoder", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        caught = None
        try:
            parsed = module.parse_message_from_completion_text(response["raw_completion"], thinking_mode="thinking")
            # OfficialCodec validates the parser's JSON-string arguments after
            # parsing. The official DSML parser can return invalid JSON without
            # throwing, so independently reproduce this part of its contract too.
            for call in parsed.get("tool_calls") or []:
                if not isinstance(json.loads(call["function"]["arguments"]), dict):
                    raise ValueError("Tool arguments must be an object")
        except Exception as failure:
            caught = failure
        require(caught is not None and type(caught).__name__ == error["parser_error_type"] and str(caught) == error["message"],
                "Saved official native parser error does not independently reproduce")
    else:
        raise AuditError("Unsupported native error evidence: " + kind)


class BoundEmailClock:
    def __init__(self, timestamp):
        self.timestamp, self.calls, self.denied = timestamp, 0, False

    def now(self, tz=None):
        if tz is not None or self.timestamp is None or self.calls:
            self.denied = True
            raise AuditError("Unrecorded send_email clock input")
        self.calls += 1
        return self.timestamp


def email_timestamp(message):
    import yaml
    value = yaml.safe_load(message.get("content") or "")
    if not isinstance(value, dict) or "timestamp" not in value:
        return None
    require(value.get("status") == "sent", "Timestamp does not describe a sent email")
    stamp = value["timestamp"]
    if isinstance(stamp, str):
        stamp = datetime.datetime.fromisoformat(stamp)
    require(type(stamp) is datetime.datetime and stamp.tzinfo is None, "Invalid native email clock evidence")
    return stamp


def email_evidence(requests, trace):
    evidence, passes, native_pass = {}, [], -1
    first = requests[0]["messages"] if requests else None

    def add(messages, number, source):
        for index, message in enumerate(messages):
            if message["role"] == "tool" and message.get("name") == "send_email":
                key = (number, message["tool_call_id"])
                if key in evidence:
                    require(evidence[key]["message"] == message, "Repeated original email result differs")
                else:
                    evidence[key] = {"message": message, "source": source, "message_index": index}

    for index, request in enumerate(requests):
        if index == 0 or request["messages"] == first:
            native_pass += 1
        passes.append(native_pass)
        add(request["messages"], native_pass, f"original_request_{index:04d}")
    if trace:
        add([m for m in independent_messages(trace["messages"]) if m["role"] == "tool"], native_pass, "original_native_trace")
    return evidence, passes


def saved_calls(root, case, record, identity, bindings):
    result = []
    require(record["spec_sha256"] == object_sha(identity), "Case run identity hash differs")
    for index, call in enumerate(record["requests"]):
        call_id = case["case_id"] + f"/call_{index:04d}"
        require(call["call_index"] == index and call["call_id"] == call_id, "Saved call order differs")
        receipt = read(bound_path(root, call["bridge_receipt"], bindings))
        require(receipt["call_id"] == call_id and receipt["server_identity_sha256"] == identity["server_identity_sha256"], "Bridge/server binding differs")
        request = read(bound_path(root, receipt["request"], bindings))
        require(object_sha(request) == receipt["raw_request_sha256"], "Saved semantic request hash differs")
        meta = request["metadata"]
        require(all(meta.get(k) == v for k, v in case.items()) and meta["call_index"] == index and meta["call_id"] == call_id,
                "Saved request case/GOAL/payload differs")
        require(all(meta.get(k) == v for k, v in identity["generation"].items()), "Saved generation settings differ")
        response_wrapper = read(bound_path(root, receipt["response"], bindings))
        require(response_wrapper["raw_request_sha256"] == receipt["raw_request_sha256"], "Saved response/request hash differs")
        response = response_wrapper["response"]
        if "raw_request_sha256" in response:
            require(response["raw_request_sha256"] == receipt["raw_request_sha256"], "Model response request SHA differs")
        if identity.get("mode") == "full":
            require("artifact_refs" in response, "Full native evaluation lacks an actual native model receipt")
        if "artifact_refs" in response:
            server_root = Path(identity["server_identity_path"]).resolve().parent
            model_path = (server_root / response["artifact_refs"]["receipt"]).resolve()
            candidates = [ref for ref in receipt["artifacts"] if (Path(root)/ref["path"]).resolve() == model_path]
            require(len(candidates) == 1, "Bridge receipt does not bind the native model receipt")
            model_receipt = read(bound_path(root, candidates[0], bindings))
            require(model_receipt["format_version"] == "deepseek-agentdojo-call-v1"
                    and model_receipt["raw_request_sha256"] == receipt["raw_request_sha256"]
                    and model_receipt["metadata"] == meta, "Native model receipt input binding differs")
            backend = read(server_root/"generation-backend.json")
            require(object_sha(backend) == model_receipt["identity_sha256"], "Native model backend receipt identity differs")
            for filename in ("request.json", "response.json"):
                path = model_path.parent/filename
                recorded = {"path": str(path.relative_to(Path(root).resolve())), "sha256": model_receipt["files"][filename]}
                value = read(bound_path(root, recorded, bindings))
                if filename == "response.json":
                    require(value == response, "Saved bridge/model responses differ")
                else:
                    require(value["raw_request_sha256"] == receipt["raw_request_sha256"], "Native model source request SHA differs")
        result.append({**request, "response": response, "call_id": call_id})
    return result


def replay_case(root, case, record, identity, native, protocol, bindings):
    require(all(record.get(k) == v for k, v in case.items()), "Case differs from canonical schedule")
    require(record["status"] in {"complete", "error"}, "Case is not terminal")
    if record["status"] == "error":
        require(record["utility"] is None and record["security"] is None, "Native error scores must remain null")
    else:
        require(type(record["utility"]) is bool and type(record["security"]) is bool, "Native scores must be booleans")
    trace = read(bound_path(root, record["raw_trace"], bindings)) if record.get("raw_trace") else None
    if record.get("resumed_partial_trace"):
        bound_path(root, record["resumed_partial_trace"], bindings)
    requests = saved_calls(root, case, record, identity, bindings)
    suite = native["get_suite"](VERSION, case["suite"])
    task, injection = suite.user_tasks[case["user_task_id"]], suite.get_injection_task_by_id(case["injection_task_id"])
    require(injection.GOAL == case["actual_goal"], "Native injection GOAL changed")
    evidence, passes = email_evidence(requests, trace)
    clock_bindings = []

    class Capture(native["Logger"]):
        def __init__(self):
            self.messages, self.errors, self.logdir = [], [], None

        def log(self, messages, **kwargs):
            self.messages = json.loads(json.dumps(messages, default=lambda value: value.model_dump()))

        def log_error(self, error):
            self.errors.append(str(error))

    capture = Capture()

    class Replay(native["BasePipelineElement"]):
        name = "independent_deepseek_saved_completion_native_replay"

        def __init__(self):
            self.index, self.current, self.email_index = 0, None, 0

        def query(self, query, runtime, env, messages=(), extra_args=None):
            index = self.index
            require(index < len(requests), "Native replay requested an unsaved model completion")
            request = requests[index]
            require(independent_messages(messages) == request["messages"], f"Actual native history differs at model call {index}")
            actual_tools = [{"type": "function", "function": {"name": tool.name, "description": tool.description,
                "parameters": tool.parameters.model_json_schema()}} for tool in runtime.functions.values()]
            require(actual_tools == request["tools"], "Native tool schema differs")
            self.index += 1
            self.current, self.email_index = index, 0
            validate_model_error(request, identity)
            message = independent_assistant(copy.deepcopy(request["response"]), request["call_id"], native)
            return query, runtime, env, [*messages, message], extra_args or {}

    llm = Replay()

    class Runtime(native["FunctionsRuntime"]):
        def run_function(self, env, function, kwargs, raise_on_error=False):
            if function != "send_email":
                return super().run_function(env, function, kwargs, raise_on_error=raise_on_error)
            require(llm.current is not None, "Email invoked outside saved model action")
            response = independent_assistant(copy.deepcopy(requests[llm.current]["response"]), requests[llm.current]["call_id"], native)
            emails = [call for call in response["tool_calls"] or [] if call.function == "send_email"]
            require(llm.email_index < len(emails), "Extra native email invocation")
            call = emails[llm.email_index]
            llm.email_index += 1
            key = (passes[llm.current], call.id)
            require(key in evidence, "No recorded original email clock input")
            item, stamp = evidence[key], email_timestamp(evidence[key]["message"])
            clock = BoundEmailClock(stamp)
            with patch.object(native["email_client"], "datetime", SimpleNamespace(datetime=clock)):
                value, error = super().run_function(env, function, kwargs, raise_on_error=raise_on_error)
            require(not clock.denied and clock.calls == int(stamp is not None), "Email clock use differs")
            if stamp is not None:
                require(error is None and value.timestamp == stamp, "Actual native email tool did not reproduce its clock input")
                clock_bindings.append({"call_index": llm.current, "native_pass": passes[llm.current], "tool_call_id": call.id,
                    "timestamp": stamp.isoformat(), "source": item["source"], "message_index": item["message_index"],
                    "actual_mock_tool_executed": True, "tool_result_replaced": False})
            return value, error

    pipeline = native["AgentPipeline"].from_config(native["PipelineConfig"](llm=llm, model_id=None, defense=None,
        tool_delimiter="tool", system_message_name=None, system_message=None, tool_output_format=None, provider_config=None))
    import yaml
    count = 0
    for element in pipeline.elements:
        for child in getattr(element, "elements", ()):
            if isinstance(child, native["ToolsExecutor"]):
                child.output_formatter = lambda value: native["tool_result_to_str"](value, dump_fn=partial(yaml.safe_dump, allow_unicode=True))
                count += 1
    require(count == 1 and identity.get("tool_formatter", {}).get("allow_unicode") is True, "Native Unicode YAML protocol is missing")
    attack = native["FixedJailbreakAttack"](case["payload_template"], suite, pipeline)
    injections = attack.attack(task, injection)
    require(injections and all(value == case["payload_text"] for value in injections.values()), "Independent native injection slots differ")
    if trace:
        require(trace["injections"] == injections and trace["injection_task_id"] == case["injection_task_id"]
                and trace["attack_type"] == protocol.ATTACK_NAMES[case["condition"]], "Recorded native attack identity differs")
    token, exception = native["LOGGER_STACK"].set([]), None
    try:
        with capture:
            try:
                utility, security = suite.run_task_with_pipeline(pipeline, task, injection, injections, runtime_class=Runtime)
            except Exception as error:
                exception = error
    finally:
        native["LOGGER_STACK"].reset(token)
    if isinstance(exception, AuditError):
        raise exception
    require(llm.index == len(requests), "Native replay did not consume every saved completion")
    if trace:
        require(capture.messages == trace["messages"], "Final original native message/tool trace differs")
    if record["status"] == "complete":
        require(exception is None, f"Native replay unexpectedly raised: {exception}")
        require(utility is record["utility"] and security is record["security"], "Independent native utility/security differs")
    else:
        require(exception is not None, "Original native error did not reproduce")
        expected_error = record.get("error", "")
        actual_error = f"{type(exception).__name__}: {exception}"
        require(actual_error == expected_error or (expected_error.startswith("Recovered original native error: ")
                and str(exception) == expected_error[len("Recovered original native error: "):]), "Replayed native error differs")
        utility = security = None
    return {"case_id": case["case_id"], "valid": True, "utility": utility, "security": security,
        "native_calls_replayed": llm.index, "native_passes": max(passes, default=-1)+1,
        "native_error_reproduced": exception is not None, "actual_mock_tools_executed": True,
        "clock_bindings": clock_bindings, "tool_results_replaced": False, "model_generation_performed": False}


def audit_run(root, repo=Path.home()/"Github/delimeter-token", *, allow_smoke=False):
    root = Path(root).resolve()
    result = {"schema": "deepseek-agentdojo-independent-native-evaluation-v1", "status": "running", "complete": False,
        "scheduled_cases": 778, "audited_cases": 0, "model_requests": 0, "cases_sha256": {}, "native_results": [],
        "started_unix": time.time(), "audit_source_sha256": sha(__file__), "model_generation_performed": False}
    missing = [name for name in ("run_identity.json", "schedule.json") if not (root/name).exists()]
    if missing:
        result["pending"] = missing
        return result
    try:
        identity, cases = read(root/"run_identity.json"), read(root/"schedule.json")
        result.update(run_identity_sha256=sha(root/"run_identity.json"), schedule_sha256=sha(root/"schedule.json"), scheduled_cases=len(cases))
        result["server_identity_sha256"] = identity["server_identity_sha256"]
        require(identity["schedule_sha256"] == object_sha(cases), "Canonical schedule hash differs")
        if identity.get("server_identity_path"):
            server_path = Path(identity["server_identity_path"])
            require(sha(server_path) == identity["server_identity_sha256"] and read(server_path) == identity["server_identity"], "Frozen server/model/probe identity differs")
        else:
            require(allow_smoke, "Full audit lacks a bound server identity file")
        if identity.get("bridge_sha256"):
            require(sha(HERE/"deepseek_agentdojo_bridge.py") == identity["bridge_sha256"], "Frozen bridge source differs")
        if identity.get("protocol_manifest_sha256"):
            require(sha(root/"protocol.json") == identity["protocol_manifest_sha256"], "Protocol manifest differs")
        if allow_smoke:
            require(identity["mode"] == "smoke" and len(cases) == 6, "Smoke audit requires exact canonical six-case scope")
        else:
            require(identity["mode"] == "full" and len(cases) == 778, "Full audit requires all canonical 778 initial episodes")
        missing = [c["case_id"] for c in cases if not (root/"cases"/(c["case_id"]+".json")).exists()]
        if missing:
            result["pending_cases"] = missing
            return result
        native = dependencies(repo)
        import deepseek_agentdojo_protocol as protocol
        canonical_cases = protocol.native_cases(native["get_suite"])
        if allow_smoke:
            selected, seen = [], set()
            for case in canonical_cases:
                key = (case["condition"], case["suite"])
                if key not in seen:
                    seen.add(key); selected.append(case)
            canonical_cases = selected
        require(cases == canonical_cases, "Schedule differs from actual pinned native versioned GOALs")
        source = Path(native["agentdojo"].__file__).resolve().parent
        require(subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() == REVISION, "Native checkout revision differs")
        require(not subprocess.check_output(["git","-C",str(source),"status","--porcelain","--untracked-files=no"],text=True).strip(), "Native tracked source is dirty")
        source_bindings = identity["agentdojo_source"]["files_sha256"]
        require(source_bindings, "Native source snapshot is absent")
        for name, digest in source_bindings.items():
            require(sha(source.parent/name) == digest, "Native source SHA differs: "+name)
        result["source"] = {"revision": REVISION, "files_sha256": source_bindings,
            "protocol_sha256": sha(Path(protocol.__file__))}
        require(result["source"]["protocol_sha256"] == identity["protocol_sha256"], "Protocol source changed")
        bindings = {}
        for case in cases:
            path = root/"cases"/(case["case_id"]+".json")
            result["cases_sha256"][case["case_id"]] = sha(path)
            try:
                row = replay_case(root, case, read(path), identity, native, protocol, bindings)
            except Exception as error:
                row = {"case_id": case["case_id"], "valid": False, "error": f"{type(error).__name__}: {error}"}
            result["native_results"].append(row)
            result["audited_cases"] += 1
            result["model_requests"] += row.get("native_calls_replayed", 0)
        require(all(sha(root/"cases"/(case_id+".json")) == digest for case_id,digest in result["cases_sha256"].items()), "Case changed during independent audit")
        require(sha(root/"run_identity.json") == result["run_identity_sha256"] and sha(root/"schedule.json") == result["schedule_sha256"], "Run identity changed during audit")
        result["request_and_trace_files_sha256"] = bindings
        result["complete"] = len(result["native_results"]) == len(cases) and all(r["valid"] for r in result["native_results"])
        result["status"] = "passed" if result["complete"] else "failed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result["finished_unix"] = time.time()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.home()/"Github/delimeter-token")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = audit_run(args.output, args.repo, allow_smoke=args.allow_smoke)
    destination = args.output/"native-evaluation-audit.json"
    write(destination, report)
    print(json.dumps({"status": report["status"], "audited_cases": report["audited_cases"], "output": str(destination)}))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
