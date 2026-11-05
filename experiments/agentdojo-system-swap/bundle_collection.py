"""Bundle a verified full collection and its directly usable system-probe input."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def bundle(args):
    source, probe = args.source.resolve(), args.probe_dir.resolve()
    output = args.output.expanduser().absolute()
    require(output.name.endswith(".tar.gz") and not output.exists(), "Expected a new .tar.gz destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    complete, manifest = read(source / "complete.json"), read(source / "manifest.json")
    pm = read(probe / "manifest.json")
    export = read(probe.with_name(probe.name + ".export-complete.json"))
    source_sha = digest(source / "manifest.json")
    order = manifest.get("system_order", "swapped")
    require(complete["status"] == "complete" and complete["manifest_sha256"] == source_sha
            and complete["tasks"] == manifest["task_count"], "Full source collection is not verified")
    require(complete["all_features_finite"] and complete["fresh_model_generation"], "Source validation is incomplete")
    system_gate = "all_original_system_prompts_and_token_filters_verified" if order == "normal" else "all_system_swaps_and_token_filters_verified"
    require(complete[system_gate], "System order / token filtering was not verified")
    require(export["status"] == "complete" and pm["source_manifest_sha256"] == source_sha
            and pm["tasks"] == complete["tasks"] and pm["calls"] == complete["calls"]
            and pm["suite_tasks"] == manifest["suite_tasks"] and pm["system_order"] == order,
            "System-probe export does not cover the complete source collection")
    require(pm["head_ids"] == complete["head_ids"] == manifest["head_ids"], "Probe head order differs")
    files = {}

    def add(name, path, expected=None):
        require(name not in files and not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid or duplicate bundle path")
        require(path.is_file() and not path.is_symlink(), "Expected a regular source file: " + str(path))
        files[name] = {"path": path, "sha256": expected or digest(path), "bytes": path.stat().st_size}

    for name in ("manifest.json", "complete.json", "backend.json", "agentdojo-source.json", "initial-requests-manifest.json", "progress.json"):
        add("collection/" + name, source / name)
    for name in ("comparison-preflight.json", "launch.json"):
        if (source / name).is_file():
            add("collection/" + name, source / name)
    initial = {r["task_key"]: r for r in manifest["initial_requests"]}
    results = {read(p)["task_key"]: p for p in (source / "tasks").glob("*/*/result.json")}
    require(set(initial) == set(results) and len(results) == complete["tasks"], "Task coverage differs")
    committed = {item["call_id"]: item["states_sha256"] for item in complete["files"]}
    require(len(committed) == complete["calls"], "Duplicate or missing committed call")
    seen = set()
    for task_key in sorted(results):
        result = read(results[task_key])
        add("collection/tasks/" + task_key + "/result.json", results[task_key])
        for name, key in (("request.json", "request_sha256"), ("rendered.json", "rendered_sha256")):
            add("collection/initial_requests/" + task_key + "/" + name,
                source / "initial_requests" / task_key / name, initial[task_key][key])
        for i in range(result["calls"]):
            call_id = f"no_payload/{task_key}/call_{i:04d}"
            directory = source / "calls" / call_id
            receipt = read(directory / "receipt.json")
            require(receipt["manifest_sha256"] == source_sha and receipt["call_id"] == call_id
                    and receipt["files"]["states.safetensors"] == committed[call_id], "Call receipt differs")
            require(set(receipt["files"]) == {"states.safetensors", "request.json", "response.json", "original_request.json"}, "Unexpected call artifact set")
            for name, expected in receipt["files"].items():
                add("collection/calls/" + call_id + "/" + name, directory / name, expected)
            add("collection/calls/" + call_id + "/receipt.json", directory / "receipt.json")
            seen.add(call_id)
    require(seen == set(committed), "Call coverage differs")
    for name, expected in manifest["code_sha256"].items():
        add("capture_code/" + name, args.collector.parent / name, expected)
    add("read_states.py", HERE / "read_states.py")
    add("bundle_collection.py", Path(__file__).resolve())
    probe_paths = {str(p.relative_to(probe)): p for p in probe.rglob("*") if p.is_file()}
    require(set(probe_paths) == set(export["package_file_sha256"]), "Probe package file set differs")
    for name, path in probe_paths.items():
        add("system_probe/" + name, path, export["package_file_sha256"][name])
    add("system_probe/export-complete.json", probe.with_name(probe.name + ".export-complete.json"))
    description = {"format": "agentdojo-full-hidden-states-with-system-probe-v1", "status": "complete",
                   "system_order": order, "suite_tasks": manifest["suite_tasks"], "tasks": complete["tasks"],
                   "calls": complete["calls"], "input_rows": complete["input_rows"], "output_rows": complete["output_rows"],
                   "benchmark_utility_successes": complete["utility_successes"],
                   "max_token_terminations": complete["max_token_terminations"],
                   "nonstandard_parse_calls": complete["nonstandard_parse_calls"],
                   "raw_capture": "BF16 pre-o_proj; ordered 16 heads, 1024 columns; saved native delimiters excluded",
                   "system_probe_shape": pm["feature_shape"], "system_probe_dtype": "float32, exact BF16 conversion, unstandardized",
                   "source_manifest_sha256": source_sha, "source_complete_sha256": digest(source / "complete.json"),
                   "probe_manifest_sha256": digest(probe / "manifest.json"), "created_at_unix": time.time(),
                   "files": {name: {k: v for k, v in item.items() if k != "path"} for name, item in files.items()}}
    memory = {"BUNDLE.json": json_bytes(description), "USAGE.txt": (
        f"System order: {order}. Completed tasks: {complete['tasks']}; calls: {complete['calls']}.\n"
        "collection/ contains the full committed input/output states, raw requests, responses, token IDs and masks.\n"
        "capture_code/ contains the exact frozen sources used for inference. Model weights are identified by hashes in the collection manifest.\n"
        "read_states.py reads one call from collection/, optionally selecting body-only tokens or a single head.\n"
        "system_probe/X.npy is direct N x 1024 raw float32 probe input; np.load(..., mmap_mode='r', allow_pickle=False).\n"
        "system_probe/system_states.safetensors contains the same system-body values in BF16.\n"
        "No feature standardization or pooling has been applied. The frozen probe applies its FIT mean/scale internally.\n"
        "system_probe/ contains aligned token IDs, original input positions, task/call indices, labels and first-call mask.\n"
        "See BUNDLE.json and collection/complete.json for exact coverage and validation results.\n"
        "Verify extracted files using sha256sum -c SHA256SUMS.\n").encode()}
    hashes = {name: item["sha256"] for name, item in files.items()}
    hashes.update({name: hashlib.sha256(value).hexdigest() for name, value in memory.items()})
    memory["SHA256SUMS"] = "".join(f"{value}  {name}\n" for name, value in sorted(hashes.items())).encode()
    hashes["SHA256SUMS"] = hashlib.sha256(memory["SHA256SUMS"]).hexdigest()
    top = output.name[:-7]
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        for name, item in sorted(files.items()):
            tar.add(item["path"], arcname=top + "/" + name, recursive=False)
        for name, value in sorted(memory.items()):
            info = tarfile.TarInfo(top + "/" + name)
            info.size, info.mode = len(value), 0o644
            tar.addfile(info, io.BytesIO(value))
    observed = set()
    with tarfile.open(temporary, "r:gz") as tar:
        for member in tar:
            require(member.isfile(), "Unexpected archive member type")
            name = str(Path(member.name).relative_to(top))
            require(name in hashes and name not in observed, "Unexpected or duplicate archive member")
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    h.update(block)
            require(h.hexdigest() == hashes[name], "Archive bytes differ from the committed result: " + name)
            observed.add(name)
    require(observed == set(hashes), "Archive is missing committed artifacts")
    temporary.rename(output)
    archive_sha = digest(output)
    output.with_name(output.name + ".sha256").write_text(f"{archive_sha}  {output.name}\n")
    receipt = {k: v for k, v in description.items() if k != "files"}
    receipt.update({"archive": str(output), "archive_sha256": archive_sha, "archive_bytes": output.stat().st_size,
                    "packaged_files": len(hashes), "all_archive_files_readback_verified": True,
                    "verified_at_unix": time.time()})
    output.with_name(top + ".bundle-complete.json").write_bytes(json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    bundle(p.parse_args())
