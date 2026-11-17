"""Compare immutable same-case smoke/formal records without rerunning either."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import audit_formal_subset as verifier
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parent
    case_id = "plain/banking/user_task_0/injection_task_0"
    result = {"format": "deepseek-formal-smoke-scope-comparison-v1", "status": "verifying_subset",
        "full778complete": False, "case_id": case_id, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_gpu_or_new_generation": True, "smoke_verifier_rerun": False, "calls": [], "input_sha256": {},
        "limitations": ["Separate filesystem objects, scope-specific signed requests and frozen capture accounting establish namespace isolation. CPU observation cannot itself prove execution of a new GPU forward; that depends on the frozen native server/adapter and their execution records."]}
    try:
        formal = verifier.read(folder / "audit.json")
        smoke_path = root / "smoke/independent-state-audit/audit.json"
        smoke = verifier.read(smoke_path)
        verifier.require(formal["status"] == "subset_verified" and any(c["case_id"] == case_id for c in formal["cases"]),
                         "Formal same-case independent subset audit is absent")
        verifier.require(smoke["status"] == "passed" and smoke["terminal_cases"] == 6 and
                         verifier.sha(smoke_path) == "085594a18c50848abf7387cc46516e635d7c2955b924ddcf9bd522652c7086f1",
                         "Frozen final smoke audit changed")
        for path in (Path(__file__).resolve(), Path(verifier.__file__).resolve(), folder / "audit.json",
                     folder / "source-bindings.json", smoke_path, smoke_path.with_name("source-bindings.json")):
            result["input_sha256"][str(path.relative_to(root))] = verifier.sha(path)

        def bound(path, scope, inventory):
            relative = str(path.relative_to(scope))
            verifier.require(relative in inventory and verifier.sha(path) == inventory[relative], "Previously audited artifact changed")
            result["input_sha256"][str(path.relative_to(root))] = inventory[relative]
            return verifier.read(path)

        formal_case_path = root / "cases" / (case_id + ".json")
        smoke_case_path = root / "smoke/cases" / (case_id + ".json")
        formal_case = bound(formal_case_path, root, formal["artifact_sha256"])
        smoke_case = bound(smoke_case_path, root / "smoke", smoke["artifact_sha256"])
        verifier.require(len(formal_case["requests"]) == len(smoke_case["requests"]) == 2, "Unexpected same-case call counts")
        for index in range(2):
            call_id = case_id + f"/call_{index:04d}"
            paths = [root / "calls" / call_id, root / "smoke/calls" / call_id]
            inventories = [formal["artifact_sha256"], smoke["artifact_sha256"]]
            scopes = [root, root / "smoke"]
            receipts, requests, originals, responses = [], [], [], []
            for path, inventory, scope in zip(paths, inventories, scopes):
                receipts.append(bound(path / "receipt.json", scope, inventory))
                requests.append(bound(path / "request.json", scope, inventory))
                originals.append(bound(path / "original_request.json", scope, inventory))
                responses.append(bound(path / "response.json", scope, inventory))
                verifier.require(receipts[-1]["episode_key"] == [str(scope), "plain", case_id] and
                    originals[-1]["metadata"]["recording_dir"] == str(path), "First-exposure namespace crosses smoke/formal")
                verifier.require(receipts[-1]["full_payload_captured"] is (index == 1), "Scope failed to independently capture first payload")
            verifier.require(receipts[0]["raw_request_sha256"] != receipts[1]["raw_request_sha256"] and
                originals[0]["messages"] == originals[1]["messages"] and originals[0]["tools"] == originals[1]["tools"],
                "Scoped requests were reused or semantic inputs changed")
            verifier.require(requests[0]["rendered_prompt"] == requests[1]["rendered_prompt"], "Official same-case prompt differs")
            for name in ("content", "reasoning_content", "tool_calls", "raw_completion", "finish_reason"):
                verifier.require(responses[0][name] == responses[1][name], "Actual deterministic same-case response differs")
            arrays = {}
            file_names = ["input.npz", "output.npz"] + (["payload-heads-bf16-uint16.npy", "payload-selected-fp16.npy"] if index == 1 else [])
            for name in file_names:
                file_paths = [path / name for path in paths]
                for path, inventory, scope in zip(file_paths, inventories, scopes):
                    relative = str(path.relative_to(scope))
                    verifier.require(verifier.sha(path) == inventory[relative], "Compared state array changed")
                    result["input_sha256"][str(path.relative_to(root))] = inventory[relative]
                stats = [path.stat() for path in file_paths]
                verifier.require((stats[0].st_dev,stats[0].st_ino) != (stats[1].st_dev,stats[1].st_ino),
                                 "Formal/smoke arrays share one filesystem object")
                if name.endswith(".npz"):
                    with np.load(file_paths[0],allow_pickle=False) as left, np.load(file_paths[1],allow_pickle=False) as right:
                        verifier.require(set(left.files) == set(right.files) and all(verifier.equal_bits(left[k], right[k]) for k in left.files),
                                         "Same-case actual token/probability arrays differ")
                else:
                    left, right = [np.load(path,mmap_mode="r",allow_pickle=False) for path in file_paths]
                    verifier.require(verifier.equal_bits(left,right), "Same-case native state array differs")
                arrays[name] = {"bitexact": True, "distinct_filesystem_objects": True,
                    "formal_inode": stats[0].st_ino, "smoke_inode": stats[1].st_ino,
                    "formal_mtime_ns": stats[0].st_mtime_ns, "smoke_mtime_ns": stats[1].st_mtime_ns}
            result["calls"].append({"call_id": call_id, "scopes": [str(scope) for scope in scopes],
                "formal_raw_request_sha256": receipts[0]["raw_request_sha256"], "smoke_raw_request_sha256": receipts[1]["raw_request_sha256"],
                "independent_first_exposure_capture": index == 1, "prompt_and_semantic_response_bitexact": True, "arrays": arrays})
        result.update(status="subset_scope_verified", namespace_isolation_verified=True,
            same_case_actual_token_probability_and_native_states_bitexact=True)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    verifier.write(folder / "formal-smoke-scope-comparison.json", result)
    print(json.dumps({k:result.get(k) for k in ("status","full778complete","namespace_isolation_verified",
        "same_case_actual_token_probability_and_native_states_bitexact","error")}))
    return 0 if result["status"] == "subset_scope_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
