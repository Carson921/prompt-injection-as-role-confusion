"""Read-only CPU validation of explicitly selected immutable formal cases.

Only this diagnostic directory is written. This is a subset state/provenance
audit, not the full 778-case validator or an independent native score replay.
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


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def equal_bits(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def semantic_calls(calls):
    return [{"id": c["id"], "name": c["function"]["name"],
             "arguments": json.loads(c["function"]["arguments"])} for c in calls or []]


def audit(root, source, selected):
    root, source = Path(root).resolve(), Path(source).resolve()
    output = Path(__file__).resolve().parent
    result = {"format": "deepseek-formal-incremental-state-audit-v1", "status": "verifying_subset",
        "subset_verified": False, "full778complete": False, "expected_formal_cases": 778,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "no_gpu_or_new_generation": True, "native_evaluator_replay_performed": False,
        "cases": [], "calls": [], "artifact_sha256": {},
        "limitations": ["Only the explicitly selected committed formal cases are checked; no global ASR/completion claim.",
            "Only first complete payload hidden states are stored and independently CPU-rescored. Other actual token probabilities are checked for coverage, numerical validity and frozen source/preflight provenance."]}
    try:
        identity, schedule, backend = (read(root / name) for name in
            ("run_identity.json", "schedule.json", "generation-backend.json"))
        require(identity["mode"] == "full" and identity["initial_episodes"] == len(schedule) == 778 and
                Path(identity["output_root"]).resolve() == root and identity["schedule_sha256"] == object_sha(schedule),
                "Not the exact formal 778-case namespace")
        require(len(selected) == len(set(selected)), "Duplicate requested case")
        canonical_cases = {case["case_id"]: case for case in schedule}
        require(set(selected) <= set(canonical_cases), "Requested case is outside the canonical schedule")
        snapshots = {}
        # Freeze the requested terminal bytes before inspecting any live tree.
        for key in selected:
            path = root / "cases" / (key + ".json")
            payload = path.read_bytes()
            snapshot = output / "case-snapshots" / (key + ".json")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if snapshot.exists():
                require(snapshot.read_bytes() == payload, "Previously snapshotted terminal case changed")
            else:
                with snapshot.open("xb") as stream:
                    stream.write(payload)
            snapshots[key] = (path, snapshot, hashlib.sha256(payload).hexdigest())
        sources = [Path(__file__).resolve()] + [source / name for name in
            ("deepseek_agentdojo_report.py", "deepseek_role_probe_report.py", "deepseek_agentdojo_protocol.py",
             "deepseek_agentdojo_model.py", "deepseek_role_probe_model.py", "deepseek_role_probe_train.py",
             "finalize_deepseek_agentdojo.py")]
        sources += [root / name for name in ("run_identity.json", "schedule.json", "protocol.json", "server_identity.json",
            "generation-backend.json", "native-generation-preflight.json", "implementation-source/core-manifest.json",
            "implementation-source/validation-manifest.json")]
        sources += [Path(backend["probe_dir"]) / name for name in ("checkpoint.npz", "freeze.json")]
        sources += [Path(backend["model_dir"]) / name for name in ("tokenizer.json", "encoding/encoding_dsv4.py")]
        bindings = {str(path): sha(path) for path in sources}
        binding_path = output / "source-bindings.json"
        if binding_path.exists():
            require(read(binding_path) == bindings, "Bound independent validation/model/probe source changed")
        else:
            write(binding_path, bindings)
        result["source_sha256"] = bindings
        require(sha(root / "server_identity.json") == identity["server_identity_sha256"] and
                read(root / "server_identity.json") == identity["server_identity"], "Frozen server byte identity differs")
        sys.path.insert(0, str(source))
        import deepseek_agentdojo_report as independent
        import deepseek_role_probe_report as probe_cpu
        import finalize_deepseek_agentdojo as finalizer
        finalizer.verify_sources(root, source)
        codec = independent.CpuEncoding(backend["model_dir"])
        probe = probe_cpu.checkpoint(Path(backend["probe_dir"]) / "checkpoint.npz")
        require(probe["metadata"]["heads"] == backend["heads"] and tuple(probe["metadata"]["roles"]) ==
                tuple(backend["roles"]) == ("system", "user", "cot", "assistant", "tool") and
                sha(Path(backend["probe_dir"]) / "checkpoint.npz") == backend["probe_checkpoint_sha256"], "Probe binding/order differs")

        def bound(path, digest):
            path = Path(path).resolve()
            require(path.is_relative_to(root) and not path.is_relative_to(root / "smoke") and sha(path) == digest,
                    "Artifact changed or crosses formal scope: " + str(path))
            result["artifact_sha256"][str(path.relative_to(root))] = digest
            return path

        def reference(item):
            return bound(root / item["path"], item["sha256"])

        for key in selected:
            path, snapshot, digest = snapshots[key]
            case, record = canonical_cases[key], read(snapshot)
            bound(path, digest)
            require(all(record.get(k) == v for k, v in case.items()) and record["spec_sha256"] == object_sha(identity),
                    "Terminal case metadata differs")
            require(record["status"] == "complete" and type(record["security"]) is bool and type(record["utility"]) is bool,
                    "This diagnostic selection requires a scored terminal case")
            for name in ("raw_trace", "resumed_partial_trace"):
                if record.get(name):
                    reference(record[name])
            previous, captured_first = [], False
            case_calls = []
            for index, call in enumerate(record["requests"]):
                call_id = key + f"/call_{index:04d}"
                require(call["call_id"] == call_id and call["call_index"] == index, "Call sequence differs")
                bridge = read(reference(call["bridge_receipt"]))
                require(bridge["call_id"] == call_id and bridge["server_identity_sha256"] == identity["server_identity_sha256"],
                        "Bridge call/server binding differs")
                original = read(reference(bridge["request"]))
                wrapper = read(reference(bridge["response"]))
                request_sha = object_sha(original)
                response = wrapper["response"]
                require(bridge["raw_request_sha256"] == wrapper["raw_request_sha256"] == request_sha,
                        "Bridge raw request binding differs")
                folder = root / "calls" / call_id
                artifact_paths = {reference(item) for item in bridge["artifacts"]}
                receipt_path = folder / "receipt.json"
                require(receipt_path in artifact_paths, "Bridge omitted the native receipt")
                receipt = read(receipt_path)
                require(receipt["format_version"] == "deepseek-agentdojo-call-v1" and receipt["identity_sha256"] == object_sha(backend) and
                        receipt["raw_request_sha256"] == request_sha and receipt["metadata"] == original["metadata"] and
                        receipt["episode_key"] == [str(root), case["condition"], key], "Formal episode namespace differs or reuses smoke")
                files = {bound(folder / name, digest) for name, digest in receipt["files"].items()}
                require(artifact_paths == files | {receipt_path} == {p for p in folder.rglob("*") if p.is_file()},
                        "Native file inventory is incomplete or contains unbound files")
                request = read(folder / "request.json")
                require(read(folder / "original_request.json") == original and read(folder / "response.json") == response and
                        request["raw_request_sha256"] == response["raw_request_sha256"] == request_sha and
                        request["metadata"] == original["metadata"] and
                        all(original["metadata"].get(k) == v for k, v in case.items()) and
                        original["metadata"]["recording_dir"] == str(folder), "Original request/response/recording scope differs")
                independent._verify_semantic_normalization(original, request)
                require((folder / "prompt.txt").read_text() == request["rendered_prompt"] and
                        not response.get("error") and receipt["status"] == bridge["status"] == call["status"] == "complete" and
                        response["execution_batch_size"] == 1 and receipt["forward_started"] is receipt["input_forward_completed"] is True,
                        "Non-native or incomplete actual forward")
                assistant_count = 0
                for message in original["messages"]:
                    if message["role"] != "assistant":
                        continue
                    assistant_count += 1
                    require(any((message.get("content") or "") == (prior.get("content") or "") and
                                (message.get("reasoning_content") or "") == (prior.get("reasoning_content") or "") and
                                semantic_calls(message.get("tool_calls")) == semantic_calls(prior.get("tool_calls"))
                                for prior in previous), "Actual interleaved assistant history/reasoning/tool IDs changed")
                with np.load(folder / "input.npz", allow_pickle=False) as inp, np.load(folder / "output.npz", allow_pickle=False) as out:
                    require(set(inp.files) == {"token_ids", "token_indices", "offsets", "probabilities", "observed", "roles"} and
                            set(out.files) == {"token_ids", "token_indices", "probabilities", "observed", "roles"},
                            "Unexpected persisted hidden states or omitted probability arrays")
                    codec.verify_input(request, inp)
                    input_info = independent.validate_probabilities(inp)
                    output_info = independent.validate_probabilities(out, input_count=input_info["tokens"], eos_id=codec.eos_id)
                    require(request["input_tokens"] == response["input_tokens"] == input_info["tokens"] and
                            response["output_tokens"] == output_info["tokens"] and response["finish_reason"] == "stop" and
                            int(out["token_ids"][-1]) == codec.eos_id and output_info["unobserved"] == 1,
                            "Actual own-token coverage/counts or unforwarded EOS differs")
                    completion = codec.tokenizer.decode(out["token_ids"].tolist(), skip_special_tokens=False)
                    require(completion == (folder / "completion.txt").read_text() == response["raw_completion"], "Completion IDs/text differ")
                    independent._verify_parse(codec, completion, call_id, response)
                    parts = independent.verify_payload(codec, request, inp["token_ids"], inp["offsets"], case)
                    exposed, captured = parts is not None, parts is not None and not captured_first
                    require(receipt["payload_exposed"] is exposed and response["payload_exposed"] is exposed and
                            receipt["full_payload_captured"] is captured and response["full_payload_captured"] is captured,
                            "Actual first payload exposure accounting differs")
                    hidden = {"payload-heads-bf16-uint16.npy", "payload-selected-fp16.npy", "payload-capture.json"}
                    require(hidden <= set(receipt["files"]) if captured else not hidden.intersection(receipt["files"]),
                            "Full hidden states omitted or persisted beyond first exposure")
                    row = {"call_id": call_id, "scope": str(root), "request_sha256": request_sha,
                        "receipt_sha256": sha(receipt_path), "input_readouts": input_info, "generated_readouts": output_info,
                        "original_assistant_history_entries_verified": assistant_count,
                        "official_input_offsets_parse_verified": True, "payload_exposed": exposed, "first_payload_captured": captured}
                    if captured:
                        capture = read(folder / "payload-capture.json")
                        positions = parts["full_payload"]
                        native = np.load(folder / "payload-heads-bf16-uint16.npy", mmap_mode="r", allow_pickle=False)
                        selected_features = np.load(folder / "payload-selected-fp16.npy", mmap_mode="r", allow_pickle=False)
                        features = independent.selected_from_native(native, backend["heads"], selected_features)
                        require(equal_bits(features, selected_features) and capture["positions"] == positions and
                                capture["proof"] == request["payload"] and capture["selected_heads"] == backend["heads"] and
                                capture["shape"] == list(native.shape) == [len(positions),43,64,512] and
                                capture["native_dtype"] == "bfloat16" and capture["storage_dtype"] == "uint16" and
                                capture["storage_layout"] == "[payload_token, layer, head, head_dimension]" and
                                capture["capture"] == "attention-head-output-before-o_a_proj" and capture["capture_after_inverse_rope"] is True,
                                "Raw native head capture semantics/proof differs")
                        actual = inp["probabilities"][positions]
                        error = float(np.abs(probe_cpu.independent_probabilities(probe, features) - actual).max(initial=0))
                        require(error <= 3e-5 and equal_bits(np.asarray(capture["role_probabilities"], np.float32), actual),
                                "Independent scalar-erf probe reconstruction differs")
                        row["first_payload"] = {"positions": positions, "shape": list(native.shape), "ordered_heads": backend["heads"],
                            "native_bf16_all_finite": True, "ordered_fp16_bits_equal": True,
                            "independent_probability_max_abs_error": error, "roles": backend["roles"],
                            "component_token_counts": {name:len(indices) for name,indices in parts.items()},
                            "wrapper_token_ids": inp["token_ids"][parts["wrapper"]].tolist(),
                            "marker_proof": request["payload"]["marker_proof"],
                            "original_tool_message_index": request["payload"]["tool_message_index"],
                            "full_heads_sha256": receipt["files"]["payload-heads-bf16-uint16.npy"]}
                        captured_first = True
                previous.append(response); case_calls.append(call_id); result["calls"].append(row)
            require(sha(path) == sha(snapshot) == digest, "Terminal case changed after initial snapshot")
            result["cases"].append({"case_id": key, "case_sha256": digest, "snapshot": str(snapshot.relative_to(output)),
                "status": record["status"], "model_calls": case_calls, "first_payload_captured": captured_first,
                "native_scores_not_recomputed_by_this_state_audit": True})
        require({str(path):sha(path) for path in sources} == bindings, "Source/probe binding changed during audit")
        result.update(status="subset_verified", subset_verified=True, verified_terminal_cases=len(selected),
            calls_verified=len(result["calls"]), first_payloads_verified=sum(c["first_payload_captured"] for c in result["calls"]),
            input_tokens=sum(c["input_readouts"]["tokens"] for c in result["calls"]),
            generated_tokens=sum(c["generated_readouts"]["tokens"] for c in result["calls"]),
            generated_observed=sum(c["generated_readouts"]["observed"] for c in result["calls"]),
            unforwarded_eos_tokens=sum(c["generated_readouts"]["unobserved"] for c in result["calls"]))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    write(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--case-id", action="append", required=True)
    args = parser.parse_args()
    result = audit(args.output, args.source_dir, args.case_id)
    print(json.dumps({k:result.get(k) for k in ("status", "subset_verified", "full778complete", "verified_terminal_cases",
        "calls_verified", "first_payloads_verified", "input_tokens", "generated_tokens", "generated_observed", "unforwarded_eos_tokens", "error")}))
    return 0 if result["subset_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
