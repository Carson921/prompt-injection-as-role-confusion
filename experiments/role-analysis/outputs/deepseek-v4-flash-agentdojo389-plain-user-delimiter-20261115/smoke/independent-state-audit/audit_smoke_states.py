"""Read-only CPU audit of committed native smoke states; no ASR claims.

Run without --final while the six episodes are active. --final requires all six
canonical case commits and no uncommitted smoke requests. Only this directory's
source-bindings.json and audit.json are written. Core and rollout artifacts are
never changed; no model or CUDA runtime is imported.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def normalized_calls(calls):
    return [{"name": call["function"]["name"],
             "arguments": json.loads(call["function"]["arguments"])} for call in calls or []]


def audit(scope, source_dir, *, final=False):
    scope, source_dir = Path(scope).resolve(), Path(source_dir).resolve()
    run = scope.parent
    out = Path(__file__).resolve().parent
    result = {"format_version": "deepseek-smoke-independent-state-audit-v1", "status": "running",
              "expected_cases": 6, "terminal_cases": 0, "all_six_terminal": False, "final_requested": final,
              "scope": str(scope), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_scope": "CPU validation of committed real smoke states; not native evaluator replay or ASR validation",
              "no_gpu_forward_or_new_generation": True, "calls": [], "cases": [], "paired_preinjection": [],
              "artifact_sha256": {}, "limitations": [
                  "Other-token hidden states are intentionally not persisted. Their probabilities are checked for actual token coverage, numerical validity and frozen source/preflight provenance.",
                  "First complete real tool payload probabilities are independently recomputed from original BF16 head bits through ordered FP16 heads and scalar-erf CPU probe math."]}
    try:
        backend = read(run / "generation-backend.json")
        cases = read(scope / "schedule.json")
        require(len(cases) == 6 and len({c["case_id"] for c in cases}) == 6, "Expected the canonical six-case smoke schedule")
        case_map = {c["case_id"]: c for c in cases}
        sources = [Path(__file__).resolve(), source_dir / "deepseek_agentdojo_report.py",
                   source_dir / "deepseek_role_probe_report.py", source_dir / "deepseek_agentdojo_protocol.py",
                   source_dir / "deepseek_agentdojo_model.py", source_dir / "deepseek_role_probe_model.py",
                   source_dir / "deepseek_role_probe_train.py", run / "generation-backend.json",
                   run / "server_identity.json", run / "native-generation-preflight.json",
                   scope / "schedule.json", scope / "run_identity.json",
                   Path(backend["probe_dir"]) / "checkpoint.npz", Path(backend["probe_dir"]) / "freeze.json",
                   Path(backend["model_dir"]) / "tokenizer.json",
                   Path(backend["model_dir"]) / "encoding/encoding_dsv4.py"]
        bindings = {str(path): sha(path) for path in sources}
        binding_path = out / "source-bindings.json"
        if binding_path.exists():
            require(read(binding_path) == bindings, "An independently checked source/config/probe binding changed")
        else:
            write(binding_path, bindings)
        result["source_sha256"] = bindings
        sys.path.insert(0, str(source_dir))
        from deepseek_agentdojo_report import CpuEncoding, verify_payload, selected_from_native, validate_probabilities
        from deepseek_role_probe_report import checkpoint, independent_probabilities
        codec = CpuEncoding(backend["model_dir"])
        probe = checkpoint(Path(backend["probe_dir"]) / "checkpoint.npz")
        require(tuple(probe["metadata"]["roles"]) == ("system", "user", "cot", "assistant", "tool"), "Wrong probe role ordering")
        require(probe["metadata"]["heads"] == backend["heads"] and
                bindings[str(Path(backend["probe_dir"]) / "checkpoint.npz")] == backend["probe_checkpoint_sha256"],
                "Frozen probe/head binding differs")
        by_case = {key: [] for key in case_map}
        first_inputs = {}
        previous_responses = {key: [] for key in case_map}
        exposed_cases = set()
        for receipt_path in sorted((scope / "calls").rglob("receipt.json")):
            folder = receipt_path.parent
            receipt, response = read(receipt_path), read(folder / "response.json")
            request, original = read(folder / "request.json"), read(folder / "original_request.json")
            meta = receipt["metadata"]
            case_id, call_id = meta["case_id"], meta["call_id"]
            require(case_id in case_map and all(meta.get(k) == v for k, v in case_map[case_id].items()), "Call has a substituted case/GOAL/payload")
            index = len(by_case[case_id])
            require(meta["call_index"] == index and call_id == case_id + f"/call_{index:04d}", "Call order has a gap or duplicate")
            require(folder == scope / "calls" / call_id and Path(meta["recording_dir"]).resolve() == folder,
                    "Actual recording directory crosses scope")
            require(receipt["format_version"] == "deepseek-agentdojo-call-v1" and
                    receipt["identity_sha256"] == object_sha(backend) and
                    receipt["raw_request_sha256"] == response["raw_request_sha256"] == request["raw_request_sha256"] == object_sha(original),
                    "Call/backend/original-request binding differs")
            require(meta == request["metadata"] == original["metadata"] and
                    receipt["episode_key"] == [str(scope), meta["condition"], case_id], "Semantic metadata/namespace differs")
            require(not response.get("error") or response["error"].get("category") == "model", "Infrastructure failure cannot count as a native call")
            files = receipt["files"]
            require({str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} == set(files) | {"receipt.json"}, "Call file inventory differs")
            for name, digest in files.items():
                path = (folder / name).resolve()
                require(path.is_relative_to(folder) and sha(path) == digest, "Changed committed call artifact: " + name)
                result["artifact_sha256"][str(path.relative_to(scope))] = digest
            result["artifact_sha256"][str(receipt_path.relative_to(scope))] = sha(receipt_path)
            row = {"call_id": call_id, "case_id": case_id, "status": receipt["status"],
                   "request_sha256": receipt["raw_request_sha256"], "committed_files": len(files),
                   "input_tokens": response["input_tokens"], "generated_tokens": response["output_tokens"],
                   "payload_exposed": False, "full_payload_captured": False, "seconds": response["seconds"]}
            if response.get("error", {}).get("type") == "NativeContextOverflow":
                require(not receipt["forward_started"] and not receipt["input_forward_completed"] and
                        not receipt["payload_exposed"] and not receipt["full_payload_captured"], "Overflow claimed real forward state")
                prompt = codec.render(request["messages"])
                require(prompt == request["rendered_prompt"] == (folder / "prompt.txt").read_text(), "Overflow native prompt differs")
                encoded = codec.tokenizer.encode(prompt, add_special_tokens=False)
                with np.load(folder / "unforwarded-input.npz", allow_pickle=False) as arrays:
                    require(np.array_equal(arrays["token_ids"], encoded.ids) and np.array_equal(arrays["offsets"], encoded.offsets), "Overflow IDs/offsets differ")
                require(len(encoded.ids) + backend["generation"]["max_new_tokens"] > backend["generation"]["max_model_len"] and
                        not (folder / "input.npz").exists() and not (folder / "output.npz").exists(), "Invalid native overflow evidence")
                row["not_forwarded_context_overflow"] = True
            else:
                with np.load(folder / "input.npz", allow_pickle=False) as inp, np.load(folder / "output.npz", allow_pickle=False) as generated:
                    require(set(inp.files) == {"token_ids", "token_indices", "offsets", "probabilities", "observed", "roles"} and
                            set(generated.files) == {"token_ids", "token_indices", "probabilities", "observed", "roles"}, "Unrequested hidden states or missing probability arrays")
                    codec.verify_input(request, inp)
                    row["input_readouts"] = validate_probabilities(inp)
                    row["generated_readouts"] = validate_probabilities(generated, input_count=len(inp["token_ids"]), eos_id=codec.eos_id)
                    decoded = codec.tokenizer.decode(generated["token_ids"].tolist(), skip_special_tokens=False)
                    require(decoded == response["raw_completion"] == (folder / "completion.txt").read_text(), "Actual completion tokens/text differ")
                    if not response.get("error"):
                        parsed = codec.encoder.parse_message_from_completion_text(decoded, thinking_mode="thinking")
                        for j, call in enumerate(parsed.get("tool_calls") or []):
                            call["id"] = call_id + "-tool-" + str(j)
                        require(all((parsed.get(k) if k != "tool_calls" else parsed.get(k) or []) == response[k]
                                    for k in ("content", "reasoning_content", "tool_calls")), "Official native parse differs")
                    elif response["error"]["type"] == "IncompleteNativeCompletion":
                        require(response["finish_reason"] == "length" and len(generated["token_ids"]) == backend["generation"]["max_new_tokens"] and
                                not np.any(generated["token_ids"] == codec.eos_id) and generated["observed"].all(), "Invalid budget-exhaustion own-state evidence")
                    elif response["error"]["type"] == "NativeCompletionParseError":
                        require(generated["token_ids"][-1] == codec.eos_id, "Native parser error lacks actual EOS")
                    else:
                        raise ValueError("Unclassified model response error")
                    parts = verify_payload(codec, request, inp["token_ids"], inp["offsets"], meta)
                    exposed = parts is not None
                    require(exposed == receipt["payload_exposed"] == response["payload_exposed"], "A real complete tool payload was omitted or fabricated")
                    capture_expected = exposed and case_id not in exposed_cases
                    require(capture_expected == receipt["full_payload_captured"] == response["full_payload_captured"], "First actual exposure was skipped or recaptured")
                    row["payload_exposed"], row["full_payload_captured"] = exposed, capture_expected
                    hidden_files = {"payload-capture.json", "payload-heads-bf16-uint16.npy", "payload-selected-fp16.npy"}
                    require(hidden_files <= set(files) if capture_expected else not hidden_files.intersection(files), "Full hidden states persisted outside the first exposure")
                    if capture_expected:
                        positions = np.asarray(parts["full_payload"], np.int64)
                        capture = read(folder / "payload-capture.json")
                        require(capture["positions"] == positions.tolist() and capture["proof"] == request["payload"] and
                                capture["shape"] == [len(positions), 43, 64, 512] and capture["selected_heads"] == backend["heads"] and
                                capture["capture"] == "attention-head-output-before-o_a_proj" and capture["capture_after_inverse_rope"] is True and
                                capture["native_dtype"] == "bfloat16" and capture["storage_dtype"] == "uint16", "Wrong native full-head capture semantics")
                        native_bits = np.load(folder / "payload-heads-bf16-uint16.npy", mmap_mode="r", allow_pickle=False)
                        selected = np.load(folder / "payload-selected-fp16.npy", allow_pickle=False)
                        rebuilt = selected_from_native(native_bits, backend["heads"], selected)
                        require(bits_equal(rebuilt, selected), "Native BF16 bits fail exact ordered FP16 reconstruction")
                        saved = inp["probabilities"][positions]
                        error = float(np.abs(independent_probabilities(probe, rebuilt) - saved).max(initial=0))
                        require(error <= 3e-5 and bits_equal(np.asarray(capture["role_probabilities"], np.float32), saved), "Independent scalar-erf payload probe probabilities differ")
                        row["first_payload"] = {"positions": positions.tolist(), "shape": list(native_bits.shape),
                            "roles": list(probe["metadata"]["roles"]), "ordered_heads": backend["heads"],
                            "native_bf16_all_finite": True, "ordered_fp16_bits_equal": True,
                            "independent_probability_max_abs_error": error,
                            "component_token_counts": {k: len(v) for k, v in parts.items()},
                            "component_labels": request["payload"]["component_labels"],
                            "original_tool_message_index": request["payload"]["tool_message_index"],
                            "marker_proof": request["payload"]["marker_proof"],
                            "full_heads_sha256": files["payload-heads-bf16-uint16.npy"]}
                        exposed_cases.add(case_id)
                    if index == 0:
                        first_inputs[case_id] = {"prompt": request["rendered_prompt"], "input_ids": inp["token_ids"].copy(),
                            "input_probabilities": inp["probabilities"].copy(), "generated_ids": generated["token_ids"].copy(),
                            "generated_probabilities": generated["probabilities"].copy(), "response": response}
                for assistant in (m for m in original["messages"] if m["role"] == "assistant"):
                    require(any((assistant.get("content") or "") == (prior.get("content") or "") and
                                (assistant.get("reasoning_content") or "") == (prior.get("reasoning_content") or "") and
                                normalized_calls(assistant.get("tool_calls")) == normalized_calls(prior.get("tool_calls"))
                                for prior in previous_responses[case_id]), "Interleaved actual assistant content/reasoning/tool calls were changed")
            by_case[case_id].append(row)
            previous_responses[case_id].append(response)
            result["calls"].append(row)
        for base in dict.fromkeys(c["base_case_id"] for c in cases):
            pair = [c["case_id"] for c in cases if c["base_case_id"] == base]
            require(len(pair) == 2, "Unpaired smoke scope")
            row = {"base_case_id": base, "conditions": [case_map[c]["condition"] for c in pair], "status": "pending"}
            if all(case_id in first_inputs for case_id in pair):
                left, right = [first_inputs[key] for key in pair]
                require(left["prompt"] == right["prompt"] and all(bits_equal(left[key], right[key]) for key in
                        ("input_ids", "input_probabilities", "generated_ids", "generated_probabilities")), "Paired pre-injection prompts/token IDs/probabilities differ")
                require(all(left["response"][key] == right["response"][key] for key in ("content", "reasoning_content")) and
                        normalized_calls(left["response"]["tool_calls"]) == normalized_calls(right["response"]["tool_calls"]),
                        "Paired pre-injection actual semantic generation differs")
                row.update(status="passed", prompt_exact=True, actual_input_ids_and_probabilities_bitexact=True,
                           actual_generated_ids_and_probabilities_bitexact=True, tool_call_ids="deterministic per condition call_id, excluded from semantic comparison")
            result["paired_preinjection"].append(row)
        actual_case_files = {str(p.relative_to(scope / "cases"))[:-5]: p for p in (scope / "cases").rglob("*.json")}
        require(set(actual_case_files) <= set(case_map), "Unexpected or duplicated terminal case")
        for case in cases:
            key = case["case_id"]
            row = {"case_id": key, "status": "running", "committed_calls": len(by_case[key]),
                   "payload_exposed": key in exposed_cases}
            if key in actual_case_files:
                path = actual_case_files[key]
                record = read(path)
                require(all(record.get(k) == v for k, v in case.items()) and record["status"] in ("complete", "error"), "Native terminal case differs from its canonical schedule")
                require([x["call_id"] for x in record["requests"]] == [x["call_id"] for x in by_case[key]], "Terminal case has missing/uncommitted model requests")
                for call in record["requests"]:
                    ref = call["bridge_receipt"]
                    p = (scope / ref["path"]).resolve()
                    require(p.is_relative_to(scope) and sha(p) == ref["sha256"], "Bridge terminal receipt changed")
                    bridge_receipt = read(p)
                    require(bridge_receipt["call_id"] == call["call_id"] and
                            bridge_receipt["raw_request_sha256"] == next(x["request_sha256"] for x in by_case[key] if x["call_id"] == call["call_id"]), "Bridge/model request binding differs")
                    result["artifact_sha256"][str(p.relative_to(scope))] = sha(p)
                row["status"] = record["status"]
                result["terminal_cases"] += 1
                result["artifact_sha256"][str(path.relative_to(scope))] = sha(path)
            result["cases"].append(row)
        result["all_six_terminal"] = result["terminal_cases"] == 6
        result["uncommitted_smoke_requests"] = []
        for path in (run / "calls").glob(".tmp-*/original_request.json"):
            raw = read(path)
            target = Path(raw.get("metadata", {}).get("recording_dir", "/")).resolve()
            if target.is_relative_to(scope):
                result["uncommitted_smoke_requests"].append(raw["metadata"]["call_id"])
        if final:
            require(result["all_six_terminal"] and not result["uncommitted_smoke_requests"] and
                    all(row["status"] == "passed" for row in result["paired_preinjection"]), "Six native terminal episodes and all paired/state checks are not complete")
            result["status"] = "passed"
        result["committed_calls_checked"] = len(result["calls"])
        result["first_payloads_checked"] = len(exposed_cases)
        require(all(sha(path) == digest for path, digest in bindings.items()), "A bound source changed during the CPU audit")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    write(out / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = audit(args.scope, args.source_dir, final=args.final)
    print(json.dumps({key: result.get(key) for key in ("status", "expected_cases", "terminal_cases",
        "committed_calls_checked", "first_payloads_checked", "paired_preinjection", "error")}, ensure_ascii=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
