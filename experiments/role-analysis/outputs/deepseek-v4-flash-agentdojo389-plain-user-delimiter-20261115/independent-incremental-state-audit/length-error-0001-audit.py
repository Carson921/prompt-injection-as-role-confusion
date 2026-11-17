"""CPU-only evidence audit of the first real 1024-token native length error.

Writes only new length-error-0001* diagnostic files; never changes scores,
committed generation files, core sources, or the frozen smoke audit.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import audit_formal_subset as helpers
import numpy as np


def main():
    output = Path(__file__).resolve().parent
    root, source = output.parent, output.parents[2]
    case_id = "plain/banking/user_task_2/injection_task_0"
    call_id = case_id + "/call_0002"
    folder = root / "calls" / call_id
    result = {"format": "deepseek-native-length-error-state-subset-v1", "status": "verifying_subset",
        "subset_verified": False, "full778complete": False, "expected_formal_cases": 778,
        "case_id": case_id, "call_id": call_id, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_gpu_http_or_new_generation": True, "native_scores_recomputed": False, "artifact_sha256": {},
        "limitations": ["This validates one immutable real length-error call, not completion of all 778 episodes.",
            "Raw hidden states are intentionally retained only at the episode's first complete payload exposure. The final normal token's stored probability is directly checked for position/coverage/numerics; own-token alignment is supported by frozen source and separately CPU-revalidated native terminal-reference artifacts, rather than recomputing this token's missing raw hidden state."]}
    try:
        case_path = root / "cases" / (case_id + ".json")
        # The terminal case and native receipt are snapshotted before traversal.
        snapshots = []
        for name, path in (("case", case_path), ("receipt", folder / "receipt.json")):
            destination = output / ("length-error-0001-" + name + ".json")
            payload = path.read_bytes()
            if destination.exists():
                helpers.require(destination.read_bytes() == payload, "Immutable length-error snapshot changed")
            else:
                with destination.open("xb") as stream:
                    stream.write(payload)
            snapshots.append((path, destination, helpers.sha(destination)))
        identity, backend, schedule = (helpers.read(root / name) for name in
            ("run_identity.json", "generation-backend.json", "schedule.json"))
        case = next(c for c in schedule if c["case_id"] == case_id)
        record, receipt = (helpers.read(path) for path in (snapshots[0][1], snapshots[1][1]))
        helpers.require(identity["mode"] == "full" and len(schedule) == identity["initial_episodes"] == 778 and
            identity["schedule_sha256"] == helpers.object_sha(schedule) and
            helpers.sha(root / "server_identity.json") == identity["server_identity_sha256"] and
            helpers.read(root / "server_identity.json") == identity["server_identity"], "Formal schedule/server identity differs")
        helpers.require(all(record.get(k) == v for k,v in case.items()) and record["spec_sha256"] == helpers.object_sha(identity) and
            record["status"] == "error" and record["security"] is record["utility"] is None and len(record["requests"]) == 3 and
            "IncompleteNativeCompletion" in record["error"], "Original native error or null scores changed")
        sys.path.insert(0, str(source))
        import deepseek_agentdojo_report as independent
        import deepseek_role_probe_report as probe_cpu
        import finalize_deepseek_agentdojo as finalizer
        sources = [Path(__file__).resolve(), Path(helpers.__file__).resolve()] + [source/name for name in
            ("deepseek_agentdojo_model.py", "deepseek_agentdojo_report.py", "deepseek_role_probe_report.py",
             "deepseek_agentdojo_protocol.py", "deepseek_role_probe_model.py", "deepseek_role_probe_train.py",
             "deepseek_agentdojo_generate_diagnostics.py", "deepseek_agentdojo_preflight.py")]
        sources += [root/name for name in ("run_identity.json", "schedule.json", "generation-backend.json", "server_identity.json",
             "native-generation-preflight.json", "implementation-source/core-manifest.json", "implementation-source/validation-manifest.json")]
        sources += [Path(backend["probe_dir"])/name for name in ("checkpoint.npz", "freeze.json")]
        sources += [Path(backend["model_dir"])/name for name in ("tokenizer.json", "encoding/encoding_dsv4.py")]
        bindings = {str(path):helpers.sha(path) for path in sources}
        result["source_sha256"] = bindings
        result["frozen_manifest_and_snapshot_bindings"] = finalizer.verify_sources(root, source)
        codec = independent.CpuEncoding(backend["model_dir"])
        probe = probe_cpu.checkpoint(Path(backend["probe_dir"])/"checkpoint.npz")
        freeze = helpers.read(Path(backend["probe_dir"])/"freeze.json")
        roles = ("system", "user", "cot", "assistant", "tool")
        helpers.require(tuple(probe["metadata"]["roles"]) == tuple(backend["roles"]) == roles and
            probe["metadata"]["heads"] == backend["heads"] and
            helpers.sha(Path(backend["probe_dir"])/"checkpoint.npz") == backend["probe_checkpoint_sha256"] == freeze["probe_sha256"] and
            helpers.sha(Path(backend["probe_dir"])/"freeze.json") == backend["probe_freeze_sha256"], "Frozen checkpoint/roles/head order differs")

        def bound(path, digest):
            path = path.resolve()
            helpers.require(path.is_relative_to(root) and not path.is_relative_to(root/"smoke") and helpers.sha(path) == digest,
                "Changed or out-of-scope committed artifact: " + str(path))
            result["artifact_sha256"][str(path.relative_to(root))] = digest
            return path

        def reference(item):
            return bound(root/item["path"], item["sha256"])

        bound(case_path, snapshots[0][2])
        raw_trace = helpers.read(reference(record["raw_trace"]))
        # The native logger's partial trace may keep error=null when the bridge
        # raises. The immutable case and original model/bridge response own the
        # native error; the independent native replay uses the same contract.
        helpers.require(raw_trace["suite_name"]==case["suite"] and raw_trace["user_task_id"]==case["user_task_id"] and
            raw_trace["injection_task_id"]==case["injection_task_id"], "Native partial trace case binding differs")
        result["native_partial_trace_error_field"] = raw_trace.get("error")
        previous_responses, prior_capture = [], None
        for index, call in enumerate(record["requests"]):
            expected_call = case_id+f"/call_{index:04d}"
            helpers.require(call["call_index"] == index and call["call_id"] == expected_call, "Case call order differs")
            bridge = helpers.read(reference(call["bridge_receipt"]))
            helpers.require(bridge["call_id"] == expected_call and bridge["server_identity_sha256"] == identity["server_identity_sha256"],
                "Bridge call/server binding differs")
            original = helpers.read(reference(bridge["request"]))
            wrapper = helpers.read(reference(bridge["response"]))
            this_folder = root/"calls"/expected_call
            model_receipt_path = this_folder/"receipt.json"
            model_receipt = helpers.read(model_receipt_path)
            artifact_paths = {reference(item) for item in bridge["artifacts"]}
            helpers.require(model_receipt_path in artifact_paths and
                model_receipt["identity_sha256"] == helpers.object_sha(backend) and
                model_receipt["episode_key"] == [str(root),case["condition"],case_id] and
                model_receipt["metadata"] == original["metadata"] and
                original["metadata"]["recording_dir"] == str(this_folder), "Native formal scope/request binding differs")
            committed = {bound(this_folder/name,digest) for name,digest in model_receipt["files"].items()}
            helpers.require(artifact_paths == committed|{model_receipt_path} == {p for p in this_folder.rglob("*") if p.is_file()},
                "Native call inventory differs")
            request_sha = helpers.object_sha(original)
            request, response = (helpers.read(this_folder/name) for name in ("request.json","response.json"))
            helpers.require(bridge["raw_request_sha256"] == wrapper["raw_request_sha256"] == model_receipt["raw_request_sha256"] ==
                request["raw_request_sha256"] == response["raw_request_sha256"] == request_sha and
                wrapper["response"] == response and helpers.read(this_folder/"original_request.json") == original and
                all(original["metadata"].get(k) == v for k,v in case.items()), "Raw original request/response binding differs")
            independent._verify_semantic_normalization(original,request)
            if index < 2:
                helpers.require(not response.get("error") and model_receipt["status"] == bridge["status"] == call["status"] == "complete",
                    "Earlier actual model call changed status")
                if model_receipt["full_payload_captured"]:
                    with np.load(this_folder/"input.npz",allow_pickle=False) as prior_input:
                        codec.verify_input(request,prior_input)
                        parts = independent.verify_payload(codec,request,prior_input["token_ids"],prior_input["offsets"],case)
                        native = np.load(this_folder/"payload-heads-bf16-uint16.npy",mmap_mode="r",allow_pickle=False)
                        selected = np.load(this_folder/"payload-selected-fp16.npy",mmap_mode="r",allow_pickle=False)
                        features = independent.selected_from_native(native,backend["heads"],selected)
                        error = float(np.abs(probe_cpu.independent_probabilities(probe,features)-prior_input["probabilities"][parts["full_payload"]]).max(initial=0))
                        helpers.require(error<=3e-5, "Episode first-payload checkpoint reconstruction differs")
                        prior_capture = {"call_id":expected_call,"shape":list(native.shape),
                            "native_bf16_to_ordered_fp16_bitexact":helpers.equal_bits(features,selected),
                            "independent_probability_max_abs_error":error}
                previous_responses.append(response)
                continue
            helpers.require(model_receipt == receipt and response["error"]["type"] == "IncompleteNativeCompletion" and
                response["error"]["category"] == "model" and response["finish_reason"] == "length" and
                model_receipt["status"] == "error" and bridge["status"] == call["status"] == "native_error" and
                model_receipt["forward_started"] is model_receipt["input_forward_completed"] is True and
                response["forward_started"] is response["input_forward_completed"] is True,
                "Length-error type/category/forward status differs")
            helpers.require(request["generation"] == backend["generation"] and request["generation"]["max_new_tokens"] == 1024 and
                request["generation"]["temperature"] == 0 and request["generation"]["seed"] == 0 and
                response["execution_batch_size"] == 1, "Actual generation/checkpoint budget differs")
            assistants = [m for m in original["messages"] if m["role"]=="assistant"]
            for message in assistants:
                helpers.require(any((message.get("content") or "")== (prior.get("content") or "") and
                    (message.get("reasoning_content") or "")== (prior.get("reasoning_content") or "") and
                    helpers.semantic_calls(message.get("tool_calls"))==helpers.semantic_calls(prior.get("tool_calls"))
                    for prior in previous_responses), "Actual interleaved reasoning/tool history changed")
            with np.load(this_folder/"input.npz",allow_pickle=False) as inp, np.load(this_folder/"output.npz",allow_pickle=False) as out:
                codec.verify_input(request,inp)
                inp_info = independent.validate_probabilities(inp)
                out_info = independent.validate_probabilities(out,input_count=2609,eos_id=codec.eos_id)
                helpers.require(inp_info == {"tokens":2609,"observed":2609,"unobserved":0} and
                    out_info == {"tokens":1024,"observed":1024,"unobserved":0} and
                    response["input_tokens"] == request["input_tokens"] == 2609 and response["output_tokens"]==1024 and
                    len(inp["token_ids"])+1024<=32768, "Input/output coverage/counts differ")
                for arrays in (inp,out):
                    helpers.require(arrays["token_ids"].dtype==np.int64 and arrays["token_indices"].dtype==np.int64 and
                        arrays["probabilities"].dtype==np.float32 and arrays["observed"].dtype==np.bool_ and
                        tuple(arrays["roles"].tolist())==roles and arrays["observed"].all() and
                        np.isfinite(arrays["probabilities"]).all(), "Token/probability/role dtypes or finite own-state coverage differs")
                helpers.require(inp["offsets"].dtype==np.int64 and inp["offsets"].shape==(2609,2) and
                    np.array_equal(inp["token_indices"],np.arange(2609)) and
                    np.array_equal(out["token_indices"],np.arange(2609,3633)) and
                    not np.any(out["token_ids"]==codec.eos_id) and bool(out["observed"][-1]),
                    "Last normal sampled token lacks its own position/readout or contains EOS")
                decoded = codec.tokenizer.decode(out["token_ids"].tolist(),skip_special_tokens=False)
                helpers.require(decoded==response["raw_completion"]==(this_folder/"completion.txt").read_text() and
                    request["rendered_prompt"]==(this_folder/"prompt.txt").read_text(), "Actual input/completion text differs from IDs")
                parts = independent.verify_payload(codec,request,inp["token_ids"],inp["offsets"],case)
                helpers.require(parts is not None and prior_capture is not None and receipt["payload_exposed"] is True and
                    response["payload_exposed"] is True and receipt["full_payload_captured"] is response["full_payload_captured"] is False,
                    "Re-exposure/first-capture bookkeeping differs")
                helpers.require(not any(name.startswith("payload-") for name in receipt["files"]), "Later exposure persisted duplicate raw hidden states")
                result.update(input_readouts=inp_info,generated_readouts=out_info,input_positions=[0,2608],generated_positions=[2609,3632],
                    roles=list(roles),original_semantic_role_counts=dict(Counter(m["role"] for m in original["messages"])),
                    original_interleaved_assistants_verified=len(assistants),official_input_ids_offsets_render_exact=True,
                    actual_output_ids_decode_exact=True,eos_token_id=codec.eos_id,eos_token_count=0,
                    probability_sum_max_abs_error={name:float(np.abs(arrays["probabilities"].sum(1,dtype=np.float64)-1).max(initial=0))
                        for name,arrays in (("input",inp),("generated",out))},
                    last_normal_generated_token={"row":1023,"absolute_position":int(out["token_indices"][-1]),
                        "token_id":int(out["token_ids"][-1]),"decoded_token":codec.tokenizer.decode([int(out["token_ids"][-1])],skip_special_tokens=False),
                        "observed":bool(out["observed"][-1]),"probabilities":out["probabilities"][-1].tolist(),
                        "probabilities_finite":bool(np.isfinite(out["probabilities"][-1]).all())},
                    first_payload_checkpoint_crosscheck=prior_capture)
        # Revalidate saved current native references, including their independent
        # last-normal-token forward through the old frozen HeadCapture.
        preflight = independent.verify_preflight_v2(root,backend)
        result["current_native_terminal_reference_cpu_validation"] = preflight
        adapter_source = (source/"deepseek_agentdojo_model.py").read_text()
        fragment = "logits, cache, features = self._forward([token], cache, len(ids) + step)\n                generated_probabilities[-1] = self._probabilities(features)[0]\n                observed[-1] = True"
        helpers.require(fragment in adapter_source and all(r["terminal_reference"]=="passed" for r in preflight["references"]),
            "Frozen terminal normal-token consumption path/reference changed")
        result["terminal_alignment_evidence"] = {"frozen_adapter_sha256":helpers.sha(source/"deepseek_agentdojo_model.py"),
            "source_lines":[637,641],"rule":"Every non-EOS sampled token, including step1023, is consumed before its own probability row is marked observed.",
            "native_reference_prefix_lengths":[r["tokens"] for r in preflight["references"]],
            "independent_normal_terminal_old_HeadCapture_states_and_probabilities_bitexact":True,
            "this_actual_last_token_raw_hidden_state_available_for_direct_recompute":False}
        for path,snapshot,digest in snapshots:
            helpers.require(helpers.sha(path)==helpers.sha(snapshot)==digest,"Terminal case/receipt changed during audit")
        helpers.require({str(path):helpers.sha(path) for path in sources}==bindings,"Source/model/probe binding changed during audit")
        result.update(status="subset_verified",subset_verified=True,full778complete=False,original_native_error_retained=True,
            original_native_scores={"utility":None,"security":None},committed_target_call_file_count=len(receipt["files"]),
            checkpoint_sha256=backend["probe_checkpoint_sha256"])
    except Exception as error:
        result.update(status="failed",error=f"{type(error).__name__}: {error}")
    helpers.write(output/"length-error-0001-audit.json",result)
    print(__import__("json").dumps({k:result.get(k) for k in ("status","subset_verified","full778complete","input_readouts",
        "generated_readouts","last_normal_generated_token","probability_sum_max_abs_error","error")}))
    return 0 if result["subset_verified"] else 1


if __name__=="__main__":
    raise SystemExit(main())
