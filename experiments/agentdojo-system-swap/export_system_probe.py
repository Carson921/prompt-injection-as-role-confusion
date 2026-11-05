"""Export selected completed suites' system-body states as N x 1024 probe input."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELECTION = ROOT / "experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/head-selection.json"
CLASSIFIER = ROOT / "experiments/role-analysis/synthetic_agent_classifier.py"
EXPECTED = {"banking": 16, "slack": 21, "travel": 20}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(value, message):
    if not value:
        raise ValueError(message)


LOADER = '''"""Load raw, unstandardized system-token features for the bound probe."""
import json
from pathlib import Path
import numpy as np

def load_input(directory=None, *, initial_only=False):
    root = Path(directory) if directory is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    x = np.load(root / "X.npy", mmap_mode="r", allow_pickle=False)
    if x.dtype != np.float32 or x.shape != (manifest["rows"], 1024):
        raise ValueError("Unexpected probe input shape or dtype")
    if initial_only:
        x = x[np.load(root / "initial_call_mask.npy", allow_pickle=False)]
    return x

if __name__ == "__main__":
    x = load_input()
    print(json.dumps({"shape": list(x.shape), "dtype": str(x.dtype), "standardized": False}))
'''


def export(args):
    torch.set_num_threads(1)
    source, target = args.source.resolve(), args.output.expanduser().absolute()
    archive = target.with_name(target.name + ".tar.gz")
    require(not target.exists() and not archive.exists(), "Export destination already exists")
    require(len(args.suites) == len(set(args.suites)), "Duplicate suite selection")
    selected = {name: EXPECTED[name] for name in args.suites}
    expected_tasks = sum(selected.values())
    native = module("native_system_export_source", args.collector)
    manifest = read(source / "manifest.json")
    source_sha = sha(source / "manifest.json")
    selection = read(args.selection)
    require(manifest["code_sha256"] == native.sources(), "Frozen capture sources changed")
    require(manifest["selection_sha256"] == sha(args.selection), "Head selection changed")
    require(manifest["head_ids"] == selection["heads"], "Ordered heads differ")
    require(manifest["model_identity"]["model_binding"] == selection["signature"]["model_binding"], "Model binding differs")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    classifier = module("system_export_probe", CLASSIFIER)
    binding = {"heads": manifest["head_ids"], "feature_dim": 1024, "feature_position": "pre-o_proj",
               "selection_sha256": manifest["selection_sha256"],
               "model_binding": manifest["model_identity"]["model_binding"]}
    probe = classifier.load_probe(args.probe, expected_binding=binding)

    stage = target.with_name(target.name + f".building-{os.getpid()}")
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "provenance").mkdir()
    blocks, tokens, positions, row_calls, row_tasks, initial = [], [], [], [], [], []
    calls, tasks, systems, offsets = [], [], {}, [0]
    task_offsets, expected_receipts = [0], set()
    for suite, task_count in selected.items():
        for task_number in range(task_count):
            task_key = f"{suite}/user_task_{task_number}"
            result_path = source / "tasks" / task_key / "result.json"
            result = read(result_path)
            require(result["task_key"] == task_key and result["injections"] == {} and result["calls"] > 0,
                    "Task is not a committed no-payload result: " + task_key)
            task_id = len(tasks)
            task_start = offsets[-1]
            for call_number in range(result["calls"]):
                directory = source / "calls/no_payload" / task_key / f"call_{call_number:04d}"
                expected_receipts.add(directory / "receipt.json")
                receipt = read(directory / "receipt.json")
                require(receipt["manifest_sha256"] == source_sha, "Receipt belongs to another capture run")
                for name, expected in receipt["files"].items():
                    require(sha(directory / name) == expected, "Source file hash mismatch: " + str(directory / name))
                request, raw = read(directory / "request.json"), read(directory / "original_request.json")
                require(raw["injections"] == {} and receipt["raw_request_sha256"] == native.signature(raw), "Raw request differs")
                rendered = native.render(tok, raw)
                for key in ("original_prompt", "rendered_prompt", "input_ids", "transformation"):
                    require(request[key] == rendered[key], "System rotation or actual input differs")
                require(request["model_input_ids_verified_per_forward"] and request["prefill_mode"] == "entire_sequence",
                        "Missing native prefill provenance")
                z = load_file(str(directory / "states.safetensors"), device="cpu")
                require(z["head_ids"].tolist() == manifest["head_ids"], "Stored head ordering differs")
                ids = np.asarray(request["input_ids"], dtype=np.int64)
                keep, body, roles = native.metadata(tok, ids, manifest["delimiter_ids"])
                retained = np.flatnonzero(keep)
                require(z["input_positions"].tolist() == retained.tolist()
                        and z["input_token_ids"].tolist() == ids[retained].tolist()
                        and z["input_body_mask"].tolist() == body[retained].tolist()
                        and z["input_role_ids"].tolist() == roles[retained].tolist(), "Input alignment differs")
                mask = z["input_body_mask"] & (z["input_role_ids"] == 0)
                part = z["input_features"][mask].contiguous()
                part_ids, part_positions = z["input_token_ids"][mask], z["input_positions"][mask]
                require(part.dtype == torch.bfloat16 and part.ndim == 2 and part.shape[1] == 1024
                        and len(part) > 0 and bool(torch.isfinite(part).all()), "Invalid system features")
                text = request["transformation"]["swapped_system"]
                require(tok.decode(part_ids.tolist(), skip_special_tokens=False) == text, "Selected tokens are not exactly the system body")
                require(not np.isin(part_ids.numpy(), manifest["delimiter_ids"]).any(), "Delimiter leaked into system input")
                text_sha = hashlib.sha256(text.encode()).hexdigest()
                systems[text_sha] = {"system_body": text, "transformation": request["transformation"]}
                call_id = len(calls)
                start, end = offsets[-1], offsets[-1] + len(part)
                blocks.append(part)
                tokens.append(part_ids.numpy().astype(np.int32))
                positions.append(part_positions.numpy().astype(np.int64))
                row_calls.append(np.full(len(part), call_id, dtype=np.int32))
                row_tasks.append(np.full(len(part), task_id, dtype=np.int32))
                initial.append(np.full(len(part), call_number == 0, dtype=bool))
                offsets.append(end)
                calls.append({"index": call_id, "call_id": receipt["call_id"], "task_index": task_id,
                              "call_number": call_number, "row_start": start, "row_end": end,
                              "system_text_sha256": text_sha, "source_directory": str(directory),
                              "receipt_sha256": sha(directory / "receipt.json"),
                              "source_files_sha256": receipt["files"]})
            tasks.append({"index": task_id, "task_key": task_key, "calls": result["calls"],
                          "row_start": task_start, "row_end": offsets[-1],
                          "result_sha256": sha(result_path), "result": result})
            task_offsets.append(offsets[-1])
            print(json.dumps({"exported_task": task_key, "tasks": len(tasks), "calls": len(calls), "rows": offsets[-1]}), flush=True)

    actual_receipts = {p for suite in selected for p in (source / "calls/no_payload" / suite).glob("*/call_*/receipt.json")}
    require(actual_receipts == expected_receipts and len(tasks) == expected_tasks, "Selected call/task coverage differs")
    bf16 = torch.cat(blocks, dim=0)
    x = bf16.float().numpy()
    require(x.shape == (offsets[-1], 1024) and np.isfinite(x).all(), "Invalid concatenated probe input")
    arrays = {"X": x, "y": np.zeros(len(x), dtype=np.int64), "token_ids": np.concatenate(tokens),
              "positions": np.concatenate(positions), "call_index": np.concatenate(row_calls),
              "task_index": np.concatenate(row_tasks), "initial_call_mask": np.concatenate(initial),
              "call_offsets": np.asarray(offsets, dtype=np.int64), "task_offsets": np.asarray(task_offsets, dtype=np.int64),
              "head_ids": np.asarray(manifest["head_ids"], dtype=np.int64)}
    for name, value in arrays.items():
        np.save(stage / f"{name}.npy", value, allow_pickle=False)
    save_file({"features": bf16}, str(stage / "system_states.safetensors"),
              metadata={"capture": "pre-o_proj", "role": "system", "scope": "input_body", "dtype": "bfloat16"})
    write(stage / "calls.json", calls)
    write(stage / "tasks.json", tasks)
    write(stage / "system_messages.json", systems)
    for name in ("manifest.json", "backend.json", "agentdojo-source.json"):
        shutil.copyfile(source / name, stage / "provenance" / name)
    shutil.copyfile(args.selection, stage / "provenance/head-selection.json")
    shutil.copyfile(__file__, stage / "provenance/export_system_probe.py")
    (stage / "load_input.py").write_text(LOADER)

    # Read back the actual deliverable and compare every row to its source block.
    saved_x = np.load(stage / "X.npy", mmap_mode="r", allow_pickle=False)
    saved_bf16 = load_file(str(stage / "system_states.safetensors"), device="cpu")["features"]
    for call, block in zip(calls, blocks):
        a, b = call["row_start"], call["row_end"]
        require(torch.equal(saved_bf16[a:b], block) and np.array_equal(saved_x[a:b], block.float().numpy()),
                "Exported values differ from the actual source system states")
    sample = np.unique(np.r_[np.arange(min(8, len(x))), np.arange(max(0, len(x) - 8), len(x))])
    smoke_backend = "numpy_scipy_cpu"
    try:
        probabilities = probe.predict_proba(saved_x[sample])
    except ModuleNotFoundError as error:
        if error.name != "scipy":
            raise
        # Use the existing probe's own Torch consumer, with its frozen FIT
        # standardizer applied once to only the smoke sample. X stays raw.
        smoke_backend = "frozen_probe_torch_cpu"
        net = probe.torch(device="cpu")
        scaled = (np.asarray(saved_x[sample], dtype=np.float32) - probe.mean) / probe.scale
        with torch.no_grad():
            probabilities = torch.softmax(net(torch.from_numpy(scaled)), dim=1).numpy()
    require(probabilities.shape == (len(sample), 6) and np.isfinite(probabilities).all()
            and np.allclose(probabilities.sum(1), 1, atol=1e-6), "Actual frozen probe did not accept the exported input")
    package = {"format": "agentdojo-system-body-probe-input-v1", "status": "complete_subset_export",
               "scope": "all committed calls of all completed tasks in the selected suites",
               "suite_tasks": selected, "tasks": len(tasks), "calls": len(calls), "rows": len(x),
               "system_order": manifest.get("system_order", "swapped"),
               "initial_call_rows": int(arrays["initial_call_mask"].sum()), "feature_shape": list(x.shape),
               "probe_input": "X.npy", "probe_input_dtype": "float32", "raw_bfloat16": "system_states.safetensors:features",
               "float32_conversion": "exact BF16-to-float32; no standardization, pooling, head reordering or deduplication",
               "selection": "input_body_mask AND input_role_ids == 0; headers and all delimiters excluded",
               "labels": "y.npy contains system=0 for every row", "roles": list(classifier.ROLES),
               "head_ids": manifest["head_ids"], "feature_dim": 1024, "capture": "pre-o_proj",
               "probe_binding": binding, "compatible_probe": str(args.probe.resolve()), "compatible_probe_sha256": sha(args.probe),
               "source_run": str(source), "source_manifest_sha256": source_sha,
               "source_task_count": manifest["task_count"],
               "source_collection_complete_at_export": (source / "complete.json").exists(),
               "checks": {"all_source_file_hashes": True, "all_system_swaps_and_ids": True,
                          "exact_system_body_only": True, "every_exported_row_equal_to_source": True,
                          "all_finite": True, "exact_selected_suite_coverage": True,
                          "actual_probe_input_smoke_rows": len(sample), "probe_smoke_backend": smoke_backend,
                          "probe_binding_verified": True},
               "created_at_unix": time.time()}
    write(stage / "manifest.json", package)
    (stage / "USAGE.txt").write_text(
        "Selected completed suites: raw system-body states for probe input.\n"
        "X.npy: N x 1024 float32; values are exact conversions of the captured BF16 states.\n"
        "No scaling/standardization has been applied. The existing probe applies its own FIT mean/scale.\n"
        "system_states.safetensors retains the same values in BF16 under key 'features'.\n"
        "All calls are retained separately, including repeated system text. See manifest.json for exact suite coverage.\n"
        "call_index.npy / task_index.npy join rows to calls.json / tasks.json.\n"
        "positions.npy refers to positions in the full original model input, before delimiter filtering.\n"
        "token_ids.npy is aligned with X rows. y.npy is the system role label (0) for every row.\n"
        "Use X[initial_call_mask] for just the first original request of each selected task.\n\n"
        "Python (after extracting this archive):\n"
        "  from load_input import load_input\n"
        "  X = load_input()  # memory-mapped raw N x 1024 input\n"
        "  # Or: X = np.load('X.npy', mmap_mode='r', allow_pickle=False)\n"
        "  probabilities = probe.predict_proba(X)\n\n"
        "The compatible frozen probe path/hash, ordered heads and model binding are in manifest.json.\n"
        "SHA256SUMS verifies this complete system-body feature export.\n")
    paths = sorted(p for p in stage.rglob("*") if p.is_file())
    checksums = {str(p.relative_to(stage)): sha(p) for p in paths}
    (stage / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in checksums.items()))
    stage.rename(target)
    checksums["SHA256SUMS"] = sha(target / "SHA256SUMS")
    if args.directory_only:
        receipt = {"status": "complete", "directory": str(target), "tasks": len(tasks),
                   "calls": len(calls), "system_token_rows": len(x), "feature_shape": list(x.shape),
                   "system_order": package["system_order"], "package_file_sha256": checksums,
                   "verified_at_unix": time.time()}
        write(target.with_name(target.name + ".export-complete.json"), receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "package_file_sha256"}, ensure_ascii=False), flush=True)
        return
    temporary_archive = archive.with_name(archive.name + f".tmp-{os.getpid()}")
    with tarfile.open(temporary_archive, "w:gz", compresslevel=1) as bundle:
        bundle.add(target, arcname=target.name)
    seen = set()
    with tarfile.open(temporary_archive, "r:gz") as bundle:
        for item in bundle:
            if item.isdir():
                continue
            require(item.isfile(), "Unexpected archive member type")
            name = str(Path(item.name).relative_to(target.name))
            require(name in checksums, "Unexpected archive member")
            h = hashlib.sha256()
            with bundle.extractfile(item) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    h.update(block)
            require(h.hexdigest() == checksums[name], "Archive readback hash differs")
            seen.add(name)
    require(seen == set(checksums), "Archive is missing files")
    temporary_archive.rename(archive)
    archive_sha = sha(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{archive_sha}  {archive.name}\n")
    receipt = {"status": "complete", "directory": str(target), "archive": str(archive),
               "archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
               "tasks": len(tasks), "calls": len(calls), "system_token_rows": len(x),
               "feature_shape": list(x.shape), "package_file_sha256": checksums,
               "all_archive_files_readback_verified": True, "verified_at_unix": time.time()}
    write(target.with_name(target.name + ".export-complete.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "outputs/gpt-oss-120b-system-swap-fresh57-20261102")
    parser.add_argument("--output", type=Path, default=Path.home() / "agentdojo-system-swap-system-probe-banking-slack-20261102")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--collector", type=Path, default=HERE / "run_fresh.py")
    parser.add_argument("--suites", choices=list(EXPECTED), nargs="+", default=["banking", "slack"])
    parser.add_argument("--directory-only", action="store_true")
    parser.add_argument("--probe", type=Path, default=SELECTION.parent / "best.npz")
    parser.add_argument("--model", type=Path, default=Path("/tmp/wz55191-single-message-pre-mlp-gpt-oss-120b"))
    export(parser.parse_args())
