"""Obtain and execute the 57 no-injection requests using delimeter-token code.

Run in the existing ChatInject environment. A JSON-lines pipe connects this
benchmark-only process to the separate native-MXFP4 inference process. No model
provider or external API is used here. Tool effects are AgentDojo simulation.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

EXPECTED = {"banking": 16, "slack": 21, "travel": 20}
VERSION = "v1.2.1"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=lambda v: v.model_dump())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: v.model_dump()) + "\n")
    temp.replace(path)


def emit(value):
    sys.__stdout__.write(encode(value) + "\n")
    sys.__stdout__.flush()


class InitialRequestCaptured(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "run"])
    p.add_argument("--repo", type=Path, default=Path.home() / "Github/delimeter-token")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.repo.resolve() / "indirect-prompt-injection/ChatInject/src"
    sys.path.insert(0, str(source))
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai, _function_to_openai
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.types import text_content_block_from_string
    import agentdojo
    if not Path(agentdojo.__file__).resolve().is_relative_to(source):
        raise RuntimeError("AgentDojo must be imported from the requested delimeter-token code")
    package = source / "agentdojo"
    identity = {"repo": str(args.repo.resolve()), "agentdojo_module": str(Path(agentdojo.__file__).resolve()),
                "benchmark_version": VERSION,
                "git_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {str(f.relative_to(source)): sha(f) for f in sorted(package.rglob("*"))
                                  if f.is_file() and f.suffix in (".py", ".yaml", ".yml", ".json")}}
    identity["identity_sha256"] = hashlib.sha256(encode(identity).encode()).hexdigest()
    if args.stage == "prepare":
        write(args.output / "agentdojo-source.json", identity)
    else:
        expected = json.loads((args.output / "agentdojo-source.json").read_text())
        if identity != expected:
            raise RuntimeError("AgentDojo source changed after initial request preparation")

    class RequestLLM(BasePipelineElement):
        name = "gpt-oss-120b-native-system-swap"

        def __init__(self, suite_name, task_id):
            self.suite_name, self.task_id, self.index = suite_name, task_id, 0

        def query(self, query, runtime, env, messages=(), extra_args=None):
            index = self.index
            self.index += 1
            model_messages = [_message_to_openai(m, "gpt-oss-120b") for m in messages]
            for m in model_messages:
                if isinstance(m.get("content"), list):
                    m["content"] = "".join(part["text"] for part in m["content"])
                if m.get("content") is None:
                    m["content"] = ""
                for call in m.get("tool_calls", []):
                    arguments = call["function"]["arguments"]
                    if isinstance(arguments, str):
                        call["function"]["arguments"] = json.loads(arguments)
            call_id = f"no_payload/{self.suite_name}/{self.task_id}/call_{index:04d}"
            request = {"call_id": call_id, "suite": self.suite_name, "user_task_id": self.task_id,
                       "injections": {}, "messages": model_messages,
                       "tools": [_function_to_openai(t) for t in runtime.functions.values()],
                       "reasoning_effort": "medium", "tool_choice": "auto",
                       "benchmark_version": VERSION, "model": "openai/gpt-oss-120b"}
            if args.stage == "prepare":
                path = args.output / "initial_requests" / self.suite_name / self.task_id / "request.json"
                write(path, request)
                raise InitialRequestCaptured()
            path = args.output / "calls" / call_id / "original_request.json"
            if path.exists() and json.loads(path.read_text()) != request:
                raise RuntimeError("Resumed benchmark request differs: " + call_id)
            write(path, request)
            emit({"event": "request", "call_id": call_id, "path": str(path.resolve())})
            line = sys.stdin.readline()
            if not line:
                raise EOFError("Inference process closed before returning a response")
            response = json.loads(line)
            if response.get("call_id") != call_id:
                raise RuntimeError("Response/call alignment differs")
            if "error" in response:
                raise RuntimeError(response["error"])
            calls = [FunctionCall(function=c["name"], args=c["arguments"], id=c["id"])
                     for c in response.get("tool_calls", [])]
            content = response.get("content")
            output = {"role": "assistant", "content": [text_content_block_from_string(content)] if content else None,
                      "tool_calls": calls or None}
            return query, runtime, env, [*messages, output], extra_args or {}

    tasks = []
    for suite_name, count in EXPECTED.items():
        suite = get_suite(VERSION, suite_name)
        if set(suite.user_tasks) != {f"user_task_{i}" for i in range(count)}:
            raise RuntimeError("The 57-task suite coverage differs")
        for i in range(count):
            task = suite.user_tasks[f"user_task_{i}"]
            llm = RequestLLM(suite_name, task.ID)
            pipeline = AgentPipeline.from_config(PipelineConfig(
                llm=llm, model_id=None, defense=None, tool_delimiter="tool",
                system_message_name=None, system_message=None, tool_output_format=None))
            key = f"{suite_name}/{task.ID}"
            result_path = args.output / "tasks" / suite_name / task.ID / "result.json"
            if args.stage == "run" and result_path.exists():
                emit({"event": "task_reused", "task_key": key, "path": str(result_path.resolve())})
                continue
            started = time.time()
            try:
                with redirect_stdout(sys.stderr):
                    utility, security = suite.run_task_with_pipeline(
                        pipeline, task, injection_task=None, injections={})
            except InitialRequestCaptured:
                if args.stage != "prepare" or llm.index != 1:
                    raise
                tasks.append({"task_key": key, "suite": suite_name, "user_task_id": task.ID,
                              "prompt": task.PROMPT, "request_sha256": sha(args.output / "initial_requests" / key / "request.json")})
                continue
            except Exception as error:
                emit({"event": "task_error", "task_key": key, "error": f"{type(error).__name__}: {error}"})
                raise
            result = {"task_key": key, "utility": bool(utility), "security": bool(security),
                      "injections": {}, "calls": llm.index, "seconds": time.time() - started}
            write(result_path, result)
            emit({"event": "task_complete", **result})
    if args.stage == "prepare":
        if len(tasks) != 57:
            raise RuntimeError("Expected 57 initial requests")
        write(args.output / "initial-requests-manifest.json", {
            "status": "complete", "tasks": tasks, "suite_tasks": EXPECTED, "task_count": 57,
            "injections": {}, "new_model_generation": False, "source_identity_sha256": identity["identity_sha256"]})
        emit({"event": "prepared", "tasks": len(tasks), "injections": {}})
    else:
        emit({"event": "all_tasks_complete", "tasks": 57})


if __name__ == "__main__":
    main()
