"""Three-A40 extraction for the fixed-head six-hour warm-start experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

import synthetic_agent_gpu as runtime
from synthetic_agent_classifier import Probe

FORMAT = "synthetic-role-warm-features-v1"


def check_budget(path):
    value = json.loads(Path(path).read_text())
    start, end = float(value["started_unix"]), float(value["deadline_unix"])
    runtime.require(0 < end - start <= 21600.01, "Invalid six-hour GPU authorization")
    runtime.require(start <= time.time() < end, "Six-hour GPU window is not active")
    runtime.require(value.get("boot_id") == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "GPU authorization belongs to another host boot")
    runtime.require(0 < float(value["deadline_monotonic"]) - float(value["started_monotonic"]) <= 21600.01,
                    "Invalid monotonic six-hour authorization")
    runtime.require(time.monotonic() < float(value["deadline_monotonic"]), "Monotonic six-hour budget expired")
    return value


def arm_budget(path):
    path = Path(path)
    if path.exists():
        return check_budget(path)
    now = time.time()
    monotonic = time.monotonic()
    value = {"format_version": "role-warmstart-six-hour-budget-v1",
             "authorized_hours": 6, "devices": "0,1,2", "started_unix": now,
             "deadline_unix": now + 21600,
             "started_monotonic": monotonic, "deadline_monotonic": monotonic + 21600,
             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             "started_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
             "deadline_utc": datetime.fromtimestamp(now + 21600, timezone.utc).isoformat(),
             "clock_origin": "immediately before first CUDA model loading"}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".pending-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return check_budget(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return value
    finally:
        temporary.unlink(missing_ok=True)


def validate_test_freeze(path, parent):
    path = Path(path)
    frozen = json.loads(path.read_text())
    runtime.require(frozen.get("format_version") == "synthetic-role-warm-freeze-v1"
                    and frozen.get("selection_split") == "dev", "Not an article-DEV freeze")
    for key, name in (("probe_sha256", "best.npz"), ("training_sha256", "training.json"),
                      ("selection_sha256", "selection.json")):
        runtime.require(frozen.get(key) == runtime.digest(path.parent / name), "Frozen artifact changed: " + name)
    runtime.require(frozen["parent_probe_sha256"] == runtime.digest(parent.path), "Frozen parent identity differs")
    selected = Probe(path.parent / "best.npz")
    for key in ("roles", "heads", "feature_dim", "feature_position", "model_binding"):
        runtime.require(frozen["feature_binding"][key] == parent.metadata[key] == selected.metadata[key],
                        "Frozen feature identity differs: " + key)
    runtime.require(selected.metadata["warm_start"]["parent_probe_sha256"] == frozen["parent_probe_sha256"],
                    "Frozen weight lineage differs")
    return frozen


def _capture(model, ids, positions, heads):
    import torch
    capture = runtime.HeadCapture(positions, selected=heads)
    with torch.inference_mode(), capture.install(model):
        hidden = runtime._forward(model, ids)
    return hidden, capture.matrix()


def preflight(args):
    import torch
    from transformers import AutoTokenizer

    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    parent = Probe(args.parent_probe)
    binding = runtime.model_binding(args.model_dir)
    runtime.require(binding == parent.metadata["model_binding"], "Parent/backbone identities differ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    head_ids = parent.metadata["heads"]
    report = {"status": "running", "model_binding": binding, "heads": head_ids,
              "parent_probe_sha256": runtime.digest(args.parent_probe), "devices": [0, 1, 2],
              "runtime_sha256": runtime.digest(runtime.__file__), "extractor_sha256": runtime.digest(__file__),
              "maximum_input_tokens": 4096, "forward_batch_size": 1,
              "all_head_scan": False, "started_unix": time.time()}
    runtime.atomic_json(root / "preflight.json", report)
    with runtime.MemoryMonitor(3) as monitor:
        model = runtime.load_model(args.model_dir, root, devices="0,1,2", cache_dir=args.cache_dir)
        for index in range(3):
            torch.cuda.reset_peak_memory_stats(index)
        body = tokenizer.encode("Forests contain many kinds of plants and animals. ", add_special_tokens=False)
        prefix = tokenizer.encode("<|start|>system<|message|>", add_special_tokens=False)
        suffix = tokenizer.encode("<|end|>", add_special_tokens=False)
        checks = []
        for length in (256, 2048, 4096):
            check_budget(args.budget_file)
            ids = prefix + (body * (length // len(body) + 1))[:length - len(prefix) - len(suffix)] + suffix
            positions = np.unique(np.linspace(len(prefix), length - len(suffix) - 1, 96, dtype=int))
            with torch.inference_mode():
                reference = runtime._forward(model, ids).cpu()
            features = []
            for repetition in range(3):
                hidden, x = _capture(model, ids, positions, head_ids)
                runtime.require(torch.equal(reference, hidden.cpu()), "A hook or repeated forward changed hidden states")
                runtime.require(np.isfinite(x).all(), "Nonfinite selected-head features")
                if features:
                    runtime.require(np.array_equal(features[0], x), "Fixed-shape features are not bitexact")
                features.append(x)
                if repetition == 0:
                    with torch.inference_mode():
                        logits = model.lm_head(hidden[:, -1:, :]).float().cpu()
                    runtime.require(bool(torch.isfinite(logits).all()), "Nonfinite final-token logits")
                del hidden
            probabilities = parent.predict_proba(features[0])
            runtime.require(np.isfinite(probabilities).all(), "Invalid parent readout")
            checks.append({"input_tokens": length, "repetitions": 3,
                           "hidden_states_bitexact": True, "features_bitexact": True,
                           "feature_sha256": hashlib.sha256(features[0].tobytes()).hexdigest(),
                           "feature_rows": len(positions), "hook_logits_finite": True})
            print(json.dumps({"stage": "preflight_forward", **checks[-1]}), flush=True)
        peaks = [{"device": index, "allocated_bytes": torch.cuda.max_memory_allocated(index),
                  "reserved_bytes": torch.cuda.max_memory_reserved(index)} for index in range(3)]
    runtime.require(not monitor.errors, "GPU memory monitoring failed")
    runtime.require(max(monitor.peak) <= 40 * 1024, "Preflight exceeded the 40-GiB device-wide limit")
    report.update(status="passed", checks=checks, pytorch_peak=peaks,
                  device_wide_peak_mib=monitor.peak, memory_monitor_errors=monitor.errors,
                  finished_unix=time.time(),
                  note="Device-wide sampled peaks include all contexts; counters reset after loading. Cached AgentDojo forward stability is not revalidated here.")
    runtime.atomic_json(root / "preflight.json", report)
    return report


def _write_shard(root, split, index, arrays, signature, next_record):
    folder = root / f"{split}/shard-{index:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in arrays.items():
        path = folder / f"{name}.npy"
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        files[path.name] = runtime.digest(path)
    runtime.atomic_json(folder / "complete.json", {"rows": len(arrays["y"]), "split": split,
                        "files": files, "signature": signature, "next_record": next_record})
    return {"path": str(folder.relative_to(root)), "rows": len(arrays["y"]),
            "sha256": runtime.digest(folder / "complete.json")}


def extract(args):
    import torch

    check_budget(args.budget_file)
    parent = Probe(args.parent_probe)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run = Path(args.run_dir)
    proof = json.loads((run / "preflight.json").read_text())
    runtime.require(proof.get("status") == "passed" and proof.get("devices") == [0, 1, 2], "Three-card preflight missing")
    binding = runtime.model_binding(args.model_dir)
    runtime.require(binding == parent.metadata["model_binding"] == proof["model_binding"], "Backbone binding changed")
    runtime.require(proof.get("runtime_sha256") == runtime.digest(runtime.__file__)
                    and proof.get("extractor_sha256") == runtime.digest(__file__)
                    and proof.get("parent_probe_sha256") == runtime.digest(args.parent_probe)
                    and proof.get("heads") == parent.metadata["heads"], "Preflight implementation/head identity changed")
    if args.split == "test":
        runtime.require(args.freeze is not None and Path(args.freeze).is_file(), "TEST extraction requires a frozen candidate")
        validate_test_freeze(args.freeze, parent)
    signature = {"input_sha256": runtime.digest(args.inputs), "heads": parent.metadata["heads"],
                 "model_binding": binding, "parent_probe_sha256": runtime.digest(args.parent_probe),
                 "extractor_sha256": runtime.digest(__file__),
                 "runtime_sha256": runtime.digest(runtime.__file__), "layout": "3xa40-12layers-b1-no-cache"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        runtime.require(manifest["parent_probe_sha256"] == signature["parent_probe_sha256"], "Parent cache identity changed")
        runtime.require(manifest["model_binding"] == binding, "Feature cache model changed")
    else:
        manifest = {"format_version": FORMAT, "roles": runtime.ROLES,
                    "feature_dim": 1024, "feature_position": "pre-o_proj",
                    "heads": parent.metadata["heads"], "model_binding": binding,
                    "parent_probe_sha256": signature["parent_probe_sha256"],
                    "splits": {}, "inputs": {}, "progress": {}}
    previous = manifest["inputs"].get(args.split)
    runtime.require(previous is None or previous["signature"] == signature, "Input/extractor binding changed during resume")
    manifest["inputs"][args.split] = {"path": str(Path(args.inputs).resolve()), "signature": signature,
                                      "sha256": signature["input_sha256"]}
    progress = manifest["progress"].get(args.split, {"next_record": 0, "complete": False})
    if progress["complete"]:
        return progress
    manifest["splits"].setdefault(args.split, [])
    # Recover only atomically committed shards. Ignore any uncommitted folder.
    for entry in manifest["splits"][args.split]:
        folder = root / entry["path"]
        runtime.require(runtime.digest(folder / "complete.json") == entry["sha256"], "Committed feature shard changed")
        complete = json.loads((folder / "complete.json").read_text())
        runtime.require(complete["signature"] == signature, "Committed feature signature differs")
        for name, sha in complete["files"].items():
            runtime.require(runtime.digest(folder / name) == sha, "Committed feature array changed")
    runtime.atomic_json(manifest_path, manifest)
    t0 = time.monotonic()
    model = runtime.load_model(args.model_dir, run, devices="0,1,2", cache_dir=args.cache_dir)
    model_load_seconds = time.monotonic() - t0
    bucket_times = {str(limit): {"records": 0, "input_tokens": 0, "forward_seconds": 0.0}
                    for limit in (512, 1024, 2048, 4096)}
    pending = {key: [] for key in ("x", "y", "source", "family", "structure", "weight", "groups")}
    rows = 0
    next_record = int(progress["next_record"])
    input_tokens = 0
    processed = 0

    def commit():
        nonlocal rows
        if not rows:
            return
        arrays = {"x": np.concatenate(pending["x"]).astype(np.float16, copy=False),
                  "y": np.asarray(pending["y"], dtype=np.int64),
                  "weight": np.asarray(pending["weight"], dtype=np.float32)}
        for name in ("source", "family", "structure", "groups"):
            arrays[name] = np.asarray(pending[name], dtype=str)
        entry = _write_shard(root, args.split, len(manifest["splits"][args.split]), arrays, signature, next_record)
        manifest["splits"][args.split].append(entry)
        manifest["progress"][args.split] = {"next_record": next_record, "complete": False}
        runtime.atomic_json(manifest_path, manifest)
        for values in pending.values():
            values.clear()
        rows = 0

    with runtime.MemoryMonitor(3) as monitor, Path(args.inputs).open() as stream:
        for index, line in enumerate(stream):
            if index < next_record:
                continue
            check_budget(args.budget_file)
            record = json.loads(line)
            targets = record["targets"]
            ids = record["input_ids"]
            runtime.require(0 < len(ids) <= 4096 and bool(targets), "Invalid complete article input")
            positions = [int(target["token_ix"]) for target in targets]
            runtime.require(len(set(positions)) == len(positions), "Duplicated target token")
            forward_started = time.monotonic()
            hidden, x = _capture(model, ids, positions, parent.metadata["heads"])
            for device in range(3):
                torch.cuda.synchronize(device)
            bucket = bucket_times[str(next(limit for limit in (512, 1024, 2048, 4096) if len(ids) <= limit))]
            bucket["records"] += 1
            bucket["input_tokens"] += len(ids)
            bucket["forward_seconds"] += time.monotonic() - forward_started
            del hidden
            runtime.require(x.shape == (len(targets), 1024) and np.isfinite(x).all(), "Invalid selected-head matrix")
            pending["x"].append(x)
            for target in targets:
                role = target["role"]
                label = runtime.ROLES.index(role)
                runtime.require("label" not in target or int(target["label"]) == label, "Target role/label mismatch")
                group = dict(target.get("groups", {}))
                group.setdefault("profile", record["profile"])
                for key in ("body_length_bin", "body_quartile", "wrapper_type"):
                    if key in target:
                        group.setdefault(key, target[key])
                    elif key in record:
                        group.setdefault(key, record[key])
                pending["y"].append(label)
                runtime.require(bool(record.get("cluster_id")), "Source cluster identity is missing")
                pending["source"].append(record["cluster_id"])
                pending["family"].append(target["family_id"])
                pending["structure"].append(target.get("structure", record.get("structure", record["profile"] + ":" + record["template_id"])))
                pending["weight"].append(float(target.get("weight", 1)))
                pending["groups"].append(json.dumps(group, separators=(",", ":"), sort_keys=True))
            rows += len(targets)
            next_record = index + 1
            input_tokens += len(ids)
            processed += 1
            if rows >= 8192:
                runtime.require(max(monitor.peak) <= 40 * 1024, "Extraction exceeded the 40-GiB limit")
                commit()
                print(json.dumps({"stage": "extract", "split": args.split, "next_record": next_record,
                                  "processed_records": processed, "input_tokens": input_tokens,
                                  "seconds": time.monotonic() - t0,
                                  "input_tokens_per_second": input_tokens / max(time.monotonic() - t0, 1e-9)}), flush=True)
        commit()
    runtime.require(not monitor.errors, "GPU monitoring failed during extraction")
    manifest["progress"][args.split] = {"next_record": next_record, "complete": True,
                                         "rows": sum(e["rows"] for e in manifest["splits"][args.split]),
                                         "attempt_input_tokens": input_tokens,
                                         "attempt_seconds": time.monotonic() - t0,
                                         "model_load_seconds": model_load_seconds,
                                         "length_bucket_timings": bucket_times,
                                         "device_wide_peak_mib": monitor.peak}
    runtime.atomic_json(manifest_path, manifest)
    return manifest["progress"][args.split]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "extract"))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--parent-probe", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--budget-file", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--inputs")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", choices=("fit", "dev", "test"))
    parser.add_argument("--freeze")
    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.run_dir) / "model-forward.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "preflight":
            arm_budget(args.budget_file)
        else:
            runtime.require(all((args.inputs, args.output_dir, args.split)), "Extraction arguments missing")
        check_budget(args.budget_file)
        print(json.dumps(preflight(args) if args.action == "preflight" else extract(args)), flush=True)


if __name__ == "__main__":
    main()
