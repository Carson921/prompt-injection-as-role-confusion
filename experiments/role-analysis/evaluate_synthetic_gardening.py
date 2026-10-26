"""Post-training diagnosis of the authors' exact public gardening conversation.

This independent tool never trains, reads formal TEST, selects a candidate, or
changes a frozen experiment. ``prepare`` uses only tokenizer/config/template and
the three explicitly authorized public prompts. ``infer`` additionally requires
a frozen pilot and the completed, predeclared formal linear-seed17 NPZ export.
Each format is forwarded once; both probes apply their own FIT standardizer on
CPU. Altered delimiters are scored only against original semantic roles.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
FORMATS = ("proper_tags", "everything_in_user_tags", "basic_no_format")
SUPPORT = dict(zip(ROLES, (57, 0, 97, 179, 583, 0)))
SOURCE_SHA = "6018b848a7fd12e0bc8d0e62246b805cef68e26e29d43ba87723352d0b167ce3"
CONFIG_SHA = "3b9e3a055a764723ae1f0f32c5b0f98f458a4862f97b5bc9130387d22bb1cbff"
TEMPLATE_SHA = "bcbbcd100a56ad723921990706697a40df73870d00f93746f00759f6f87994c9"
FORMAT_VERSION = "synthetic-agent-gardening-v1"
# This utility is scoped to this one authorized campaign, never a fresh budget.
HARD_DEADLINE = datetime(2026, 10, 27, 13, 19, 17, tzinfo=timezone.utc).timestamp()
PUBLIC_SOURCE = {"author_commit": "0af14644db45120963896c18bf2c2cf82162fad4",
                 "paper_appendix": "https://arxiv.org/html/2603.12277v6",
                 "config_git_blob_sha1": "3e9bdc58d3008cc2b2ba8939bc4b5595eeb256b6",
                 "template_git_blob_sha1": "322c99cae3ce13c5096087f125ac9a66d87a2642"}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source_binding(inputs, config, template):
    paths = {"inputs": Path(inputs), "config": Path(config), "template": Path(template)}
    expected = {"inputs": SOURCE_SHA, "config": CONFIG_SHA, "template": TEMPLATE_SHA}
    for key, path in paths.items():
        require(sha(path) == expected[key], "Exact public author source changed: " + key)
    return {key: {"path": str(path.resolve()), "sha256": expected[key]} for key, path in paths.items()}


def _render(messages, key):
    """Reconstruct exact archived prompts and original semantic body intervals."""
    roles = {"system": "system", "user": "user", "assistant-cot": "cot", "assistant-final": "assistant"}
    headers = {"system": "system", "user": "user", "assistant-cot": "assistant<|channel|>analysis",
               "assistant-final": "assistant<|channel|>final"}
    require([r["role"] for r in messages] == ["system", "user", "assistant-cot", "assistant-final",
                                             "user", "assistant-cot", "assistant-final"], "Author message sequence changed")
    text = "<|startoftext|>"
    if key == "everything_in_user_tags":
        text += "<|start|>user<|message|>"
    intervals, spans = [], []
    for i, message in enumerate(messages):
        start = len(text)
        if key == "proper_tags":
            text += "<|start|>" + headers[message["role"]] + "<|message|>"
        elif i:
            text += "\n"
        body_start = len(text)
        text += message["content"]
        body_end = len(text)
        if key == "proper_tags":
            text += "<|return|>" if i == len(messages) - 1 else "<|end|>"
        intervals.append({"start": body_start, "end": body_end, "message_ix": i, "role": roles[message["role"]]})
        spans.append({"header_start": start, "body_start": body_start, "body_end": body_end,
                      "message_end": len(text), "role": roles[message["role"]], "supervised": True})
    if key == "everything_in_user_tags":
        text += "<|end|>"
    return text, intervals, spans


def strict_native_roles(tokenizer, ids, expected_sequence):
    """Independent ID-level closed Harmony parser; no substring role labeler."""
    words = ("<|startoftext|>", "<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>")
    markers = {s: tokenizer.encode(s, add_special_tokens=False) for s in words}
    require(all(len(v) == 1 for v in markers.values()), "Harmony markers must be single IDs")
    markers = {k: v[0] for k, v in markers.items()}
    require(ids[0] == markers[words[0]], "Author BOS missing")
    headers = {"system": "system", "user": "user", "assistant<|channel|>analysis": "cot",
               "assistant<|channel|>final": "assistant"}
    position, labels, sequence = 1, {}, []
    while position < len(ids):
        require(ids[position] == markers["<|start|>"], "Malformed native message start")
        start = position = position + 1
        while position < len(ids) and ids[position] != markers["<|message|>"]:
            require(ids[position] not in markers.values(), "Malformed native header")
            position += 1
        require(position < len(ids), "Unclosed native header")
        header = tokenizer.decode(ids[start:position], skip_special_tokens=False)
        require(header in headers, "Unexpected author native header")
        role = headers[header]
        sequence.append(role)
        position += 1
        first = position
        while position < len(ids) and ids[position] not in (markers["<|end|>"], markers["<|return|>"], markers["<|call|>"]):
            require(ids[position] not in markers.values(), "Unexpected body marker")
            labels[position] = role
            position += 1
        require(first < position < len(ids), "Empty or unclosed native body")
        require(ids[position] == markers["<|end|>"] or (role == "assistant" and ids[position] == markers["<|return|>"]),
                "Unexpected author closing delimiter")
        position += 1
    require(sequence == expected_sequence, "Native role sequence changed")
    return labels


def prepare_samples(tokenizer, inputs, config, template):
    """Pure tokenizer/CPU preparation, independent of all historical predictions."""
    import yaml
    binding = source_binding(inputs, config, template)
    source = read(inputs)
    require(len(source) == 3 and {r["prompt_key"] for r in source} == set(FORMATS), "Exactly three author formats required")
    messages = yaml.safe_load(Path(config).read_text())["gptoss-20b"]
    prepared = []
    for key in FORMATS:
        row = next(r for r in source if r["prompt_key"] == key)
        text, intervals, spans = _render(messages, key)
        require(text == row["prompt"], "Config bodies do not exactly reconstruct author prompt: " + key)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(v) for v in encoded["input_ids"]]
        offsets = [[int(a), int(b)] for a, b in encoded["offset_mapping"]]
        require(ids == tokenizer.encode(text, add_special_tokens=False) and len(ids) == len(offsets), "Tokenizer offsets/IDs disagree")
        content = []
        for position, (a, b) in enumerate(offsets):
            matches = [s for s in intervals if max(a, s["start"]) < min(b, s["end"])]
            require(len(matches) <= 1, "A token crosses two original message bodies")
            if matches:
                s = matches[0]
                content.append({"token_position": position, "token_id": ids[position], "message_ix": s["message_ix"],
                                "semantic_role": s["role"], "label": ROLES.index(s["role"]),
                                "formal_role": s["role"] if key == "proper_tags" else "user" if key == "everything_in_user_tags" else None,
                                "in_old_915_comparison": True})
        require(len(content) == 916 and Counter(r["semantic_role"] for r in content) == Counter({k: v for k, v in SUPPORT.items() if v}),
                "Full 916-token semantic support changed")
        require(content[-1]["token_id"] == 109 and content[-1]["semantic_role"] == "assistant", "Final emoji continuation missing")
        content[-1]["in_old_915_comparison"] = False
        if key != "basic_no_format":
            sequence = [s["role"] for s in intervals] if key == "proper_tags" else ["user"]
            parsed = strict_native_roles(tokenizer, ids, sequence)
            expected = {r["token_position"]: r["formal_role"] for r in content}
            require(parsed == expected, "Independent ID parser differs from character/body offsets")
        else:
            require([r["token_position"] for r in content] == list(range(1, len(ids))), "Untagged coverage excludes more than BOS")
        prepared.append({"prompt_key": key, "prompt_ix": row["prompt_ix"], "text": text,
                         "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "input_ids": ids,
                         "token_offsets": offsets, "content": content, "message_spans": spans,
                         "semantic_intervals": intervals})
    reference = prepared[0]["content"]
    fusions = {}
    def pieces(sample):
        return [sample["text"][slice(*sample["token_offsets"][r["token_position"]])] for r in sample["content"]]
    native_pieces = pieces(prepared[0])
    for sample in prepared[1:]:
        require([(r["message_ix"], r["semantic_role"]) for r in sample["content"]] ==
                [(r["message_ix"], r["semantic_role"]) for r in reference], "Semantic position order changed")
        text_pieces = pieces(sample)
        changed = [i for i, (a, b) in enumerate(zip(native_pieces, text_pieces)) if a != b]
        id_changes = [i for i, (a, b) in enumerate(zip(reference, sample["content"])) if a["token_id"] != b["token_id"]]
        require(changed == id_changes == [56, 101, 184, 514, 566, 662] and
                all(text_pieces[i] == native_pieces[i] + "\n" for i in changed), "Only the six archived boundary-LF BPE fusions are permitted")
        fusions[sample["prompt_key"]] = changed
    return prepared, {"format_version": FORMAT_VERSION, "source_binding": binding, "roles": list(ROLES),
                      "public_source": PUBLIC_SOURCE,
                      "formats": list(FORMATS), "tokens_per_format": 916, "semantic_support": SUPPORT,
                      "independent_native_parser_agrees": True, "complete_system_body": True,
                      "restored_emoji_continuation_id": 109, "no_historical_predictions_or_labels_read": True,
                      "altered_boundary_lf_fusions_body_row_indices": fusions,
                      "alignment_note": "Original message/semantic-role order agrees for 916 rows; six sentence-end BPE IDs change by absorbing separator LF in altered formats. Cross-format token IDs are not identical.",
                      "no_training_or_selection": True, "model_forward_run": False}


def feature_identity(root, split):
    """Check only FIT/DEV commit metadata, without loading arrays or formal TEST."""
    m = read(Path(root) / "manifest.json")
    require(split in ("fit", "dev"), "Diagnostic may inspect only FIT/DEV provenance")
    signature = m["signature"]
    progress = m["progress"][split]
    require(progress["complete"] is True, "Required feature extraction incomplete")
    entries = m["splits"][split]
    for entry in entries:
        commit = (Path(root) / entry["path"] / "complete.json").resolve()
        require(commit.is_relative_to(Path(root).resolve()), "Feature commit escapes root")
        require(sha(commit) == entry["sha256"], "Feature commit SHA mismatch")
        rec = read(commit)
        require(rec["signature"] == {**signature, "split": split, "input_sha256": m["split_sources"][split]}
                and rec["rows"] == entry["rows"], "Feature commit provenance mismatch")
    value = {"format_version": m["format_version"], "roles": m["roles"], "feature_dim": m["feature_dim"],
             "feature_position": m["feature_position"], "signature": signature, "data_hash": m["data_hash"],
             "split": split, "input_sha256": m["split_sources"][split], "shards": entries,
             "complete": progress.get("complete"), "records": progress.get("records"), "rows": progress.get("rows")}
    return hashlib.sha256(canonical(value).encode()).hexdigest(), m


def validate_run(run_dir, phase):
    """Validate a committed NPZ; a last.pt or partial first epoch never qualifies."""
    import synthetic_agent_classifier as classifier
    import synthetic_agent_gpu as gpu
    root = Path(run_dir)
    training = read(root / "training.json")
    config = read(root / "config.json")
    require(training["phase"] == phase and config["phase"] == phase, "Run phase differs")
    require(training["implementation_sha256"] == sha(classifier.__file__), "Classifier implementation changed")
    require(training["options"] == classifier._training_options(config), "Training options changed")
    require(not training["options"]["allow_cpu_smoke"] and training["options"]["min_epochs"] >= 10, "Engineering or short training forbidden")
    features = root / "features" / phase
    identities = {}
    for split in ("fit", "dev"):
        identities[split], manifest = feature_identity(features, split)
    require(manifest["config_hash"] == gpu.object_digest(config), "Feature/runtime config hash differs")
    require(manifest["signature"]["implementation_sha256"] == sha(gpu.__file__), "GPU implementation changed")
    for key, expected in (("model_binding", manifest["model_binding"]), ("heads", manifest["heads"]),
                          ("config_sha256", manifest["config_hash"]), ("capture", manifest["feature_position"]),
                          ("selection_sha256", manifest["selection_sha256"]), ("pilot_fit_sha256", manifest["pilot_fit_sha256"]),
                          ("phase", phase)):
        require(manifest["signature"].get(key) == expected, "Feature signature differs: " + key)
    require(identities == training["feature_split_hashes"], "Training feature identities changed")
    binding = {k: manifest[k] for k in ("roles", "feature_dim", "heads", "feature_position", "model_binding",
                                       "selection_sha256", "pilot_fit_sha256", "data_hash", "config_hash")}
    require(binding == training["binding"], "Training binding differs from feature manifest")
    expected_hash = hashlib.sha256(canonical({"fit_features": identities["fit"], "dev_features": identities["dev"],
                                              "options": training["options"], "implementation_sha256": sha(classifier.__file__),
                                              "candidates": classifier.CANDIDATES}).encode()).hexdigest()
    require(training["training_hash"] == expected_hash, "Training hash mismatch")
    head_path = root / "head-selection.json"
    head = read(head_path)
    require(head["signature"]["implementation_sha256"] == sha(gpu.__file__), "Head-discovery GPU implementation changed")
    require(sha(head_path) == binding["selection_sha256"] and head["heads"] == binding["heads"]
            and head["signature"]["model_binding"] == binding["model_binding"]
            and head["signature"]["pilot_fit_sha256"] == binding["pilot_fit_sha256"], "Selected heads/model changed")
    if phase == "pilot":
        frozen = read(root / "freeze.json")
        selection = read(root / "selection.json")
        require(frozen["training_sha256"] == sha(root / "training.json") and
                frozen["selection_sha256"] == sha(root / "selection.json") and
                frozen["selection_split"] == selection["selection_split"] == "dev", "Pilot freeze changed")
        candidate = frozen["candidate_id"]
        require(selection["candidate_id"] == candidate and selection["training_hash"] == expected_hash,
                "Pilot selection training differs")
        probe_path = root / "best.npz"
    else:
        # Fixed before the public example is evaluated. Never choose by its scores.
        candidate = "linear-seed17"
        probe_path = root / "candidates" / candidate / "best.npz"
    require(candidate in [r[0] for r in classifier.CANDIDATES], "Unknown candidate ID")
    candidate_root = root / "candidates" / candidate
    complete = read(candidate_root / "complete.json")
    history = read(candidate_root / "history.json")
    require(complete["candidate_id"] == candidate and complete["training_hash"] == expected_hash,
            "Candidate completion belongs to different training")
    require(10 <= int(complete["epochs"]) <= training["options"]["max_epochs"] and len(history) == int(complete["epochs"]) and
            [h["epoch"] for h in history] == list(range(1, int(complete["epochs"]) + 1)), "Candidate training not complete >=10 epochs")
    require(sha(candidate_root / "history.json") == complete["history_sha256"], "Candidate history changed")
    require(sha(candidate_root / "best.npz") == sha(probe_path) == complete["probe_sha256"], "Completed NPZ SHA mismatch")
    require(np.isfinite(complete["export_max_absolute_error"]) and complete["export_max_absolute_error"] <= 3e-5,
            "NPZ/PyTorch export check failed")
    probe = classifier.load_probe(probe_path, expected_binding=binding)
    require(probe.metadata["training_hash"] == expected_hash and probe.metadata["candidate_id"] == candidate and
            probe.metadata["best_epoch"] == complete["best_epoch"] and not probe.metadata["engineering_smoke_only"], "Probe metadata mismatch")
    require(1 <= complete["best_epoch"] <= complete["epochs"], "Invalid best epoch")
    if phase == "formal":
        require(probe.metadata["hidden"] == 0 and probe.metadata["seed"] == 17, "First formal candidate must be fixed linear-seed17")
    else:
        require(frozen["probe_sha256"] == sha(probe_path) and frozen["metadata"] == probe.metadata, "Frozen pilot NPZ differs")
    with np.load(root / "standardizer.npz", allow_pickle=False) as standardizer:
        require(str(standardizer["training_hash"]) == expected_hash and np.array_equal(standardizer["mean"], probe.mean)
                and np.array_equal(standardizer["scale"], probe.scale), "Probe does not use its own FIT standardizer")
    return probe, {"label": "frozen pilot" if phase == "pilot" else "first formal candidate: linear-seed17 (not final DEV selection)",
                   "candidate_id": candidate, "heads": probe.metadata["heads"],
                   "checkpoint_path": str(probe_path.resolve()), "checkpoint_sha256": sha(probe_path),
                   "training_sha256": sha(root / "training.json"), "training_hash": expected_hash,
                   "complete_sha256": sha(candidate_root / "complete.json"), "history_sha256": complete["history_sha256"],
                   "epochs": complete["epochs"], "best_epoch": complete["best_epoch"],
                   "export_max_absolute_error": complete["export_max_absolute_error"], "metadata": probe.metadata}


def validate_pair(pilot_dir, formal_dir, model_dir):
    import synthetic_agent_gpu as gpu
    probes, models = {}, {}
    for key, path, phase in (("pilot", pilot_dir, "pilot"), ("first_formal", formal_dir, "formal")):
        probes[key], models[key] = validate_run(path, phase)
    reference = probes["pilot"].metadata
    for key in ("roles", "feature_dim", "heads", "feature_position", "model_binding", "selection_sha256", "pilot_fit_sha256"):
        require(reference[key] == probes["first_formal"].metadata[key], "Shared forward binding differs: " + key)
    require(reference["model_binding"] == gpu.model_binding(model_dir), "Actual model/tokenizer differs from both NPZs")
    return probes, models


def metrics(sample, probabilities):
    p = np.asarray(probabilities)
    require(p.shape == (len(sample["content"]), 6) and np.isfinite(p).all() and (p >= 0).all()
            and np.allclose(p.sum(1), 1, atol=2e-5), "Invalid six-role probabilities")
    y = np.array([r["label"] for r in sample["content"]], dtype=np.int64)
    predicted = p.argmax(1)  # Six-class argmax, with no legacy system threshold.
    confusion = np.bincount(y * 6 + predicted, minlength=36).reshape(6, 6)
    per_role = {}
    f1s = []
    for i, role in enumerate(ROLES):
        support, count, tp = int(confusion[i].sum()), int(confusion[:, i].sum()), int(confusion[i, i])
        precision = tp / count if count else None
        recall = tp / support if support else None
        f1 = 2 * tp / (support + count) if support else None
        if f1 is not None:
            f1s.append(f1)
        per_role[role] = {"support": support, "predicted": count, "true_positive": tp,
                          "false_positive": count - tp, "false_negative": support - tp,
                          "precision": precision, "recall": recall, "f1": f1}
    native = sample["prompt_key"] == "proper_tags"
    return {"rows": len(y), "metric_kind": "formal_role" if native else "original_semantic_alignment",
            "accuracy" if native else "semantic_alignment_accuracy": float((y == predicted).mean()),
            "macro_f1_present": float(np.mean(f1s)), "per_role": per_role, "confusion_matrix": confusion.tolist()}


def validate_figures(index, output_dir):
    """Commit a report only after all seven comparison/format pairs exist."""
    out = Path(output_dir).resolve()
    require(index["format_version"] == "synthetic-gardening-plots-v1" and read(out / "index.json") == index,
            "Plot index differs from committed index.json")
    stems = {"gardening-native-comparison"} | {f"gardening-{model}-{key}" for model in ("pilot", "first_formal") for key in FORMATS}
    require(set(index["figures"]) == stems, "All seven required figure pairs must be present")
    verified = {}
    for stem, entry in index["figures"].items():
        if stem == "gardening-native-comparison":
            key, models = "proper_tags", ["pilot", "first_formal"]
        else:
            model = "first_formal" if stem.startswith("gardening-first_formal-") else "pilot"
            key, models = stem[len("gardening-" + model + "-"):], [model]
        require(entry["format"] == key and entry["models"] == models and entry["complete_context"] is True and
                entry["body_tokens_per_model"] == 916 and entry["probability_rows"] == 6, "Figure omitted full context/body or a model/role row")
        require(set(entry["files"]) == {"png", "pdf"}, "Figure pair is incomplete")
        for kind, record in entry["files"].items():
            path = (out / record["path"]).resolve()
            require(path.is_relative_to(out) and path.is_file() and path.suffix == "." + kind,
                    "Figure path is missing or escapes output directory")
            require(sha(path) == record["sha256"], "Figure SHA mismatch")
            with path.open("rb") as f:
                prefix = f.read(8)
            require(prefix.startswith(b"\x89PNG\r\n\x1a\n" if kind == "png" else b"%PDF-"), "Figure file format differs")
            verified[str(path.relative_to(out))] = record["sha256"]
    return verified


def _deadline(pilot_dir, formal_dir):
    values = []
    for root in (pilot_dir, formal_dir):
        budget = read(Path(root) / "config.json")["budget"]
        end = datetime.fromisoformat(budget["deadline_utc"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(budget["started_utc"].replace("Z", "+00:00"))
        hours = float(budget["authorized_hours"])
        require(end.tzinfo is not None and start.tzinfo is not None and 0 < hours <= 24
                and 0 < (end - start).total_seconds() <= hours * 3600,
                "Deadline must be explicit and within authorized 24-hour window")
        values.append(end.timestamp())
    deadline = min(*values, HARD_DEADLINE)
    require(deadline > time.time(), "Authorized experiment deadline has passed")
    return deadline


def gpu_snapshot(devices, min_free_gib=28):
    """Resource gate and context provenance; never stop or inspect other workloads' data."""
    wanted = [int(s) for s in devices.split(",")]
    require(wanted == [0, 1, 2, 3], "This 120B diagnostic requires physical GPUs 0,1,2,3")
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=15).stdout
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if int(parts[0]) in wanted:
            rows.append(dict(zip(("index", "uuid", "total_mib", "free_mib", "used_mib", "utilization_percent"),
                                 (int(parts[0]), parts[1], *[int(p) for p in parts[2:]]))))
    require(len(rows) == 4 and all(r["free_mib"] >= min_free_gib * 1024 for r in rows),
            "Resources unavailable: every GPU needs >=28 GiB free; leave existing training running and queue this diagnosis")
    raw = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=15).stdout
    processes = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, PermissionError):
            argv = []
        training = any(Path(v).name == "run_synthetic_agent_roles.py" for v in argv) and "train" in argv
        processes.append({"pid": pid, "gpu_uuid": parts[1], "process_name": Path(parts[2]).name,
                          "used_gpu_memory_mib": parts[3], "synthetic_probe_training": training})
    return {"diagnostic_pid": os.getpid(), "captured_utc": datetime.now(timezone.utc).isoformat(),
            "gpus": rows, "existing_compute_processes": processes,
            "concurrent_probe_training_detected": any(p["synthetic_probe_training"] for p in processes),
            "memory_interpretation": "global snapshots may include ongoing formal probe training; these are not standalone formal model peaks"}


def infer(args, samples, prepare_manifest):
    # All gates precede model import/load/forward and any output probabilities.
    probes, models = validate_pair(args.pilot_run_dir, args.formal_run_dir, args.model_dir)
    deadline = _deadline(args.pilot_run_dir, args.formal_run_dir)
    timer = threading.Timer(deadline - time.time(), lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    import torch
    import synthetic_agent_gpu as gpu
    out = Path(args.output_dir)
    result = {"format_version": FORMAT_VERSION, "roles": list(ROLES), "models": models,
              "formats": {}, "source_binding": prepare_manifest["source_binding"],
              "prepared_inputs_sha256": sha(out / "prepared-inputs.json"),
              "prepare_manifest_sha256": sha(out / "prepare-manifest.json"),
              "no_training_or_selection": True, "formal_test_read": False,
              "diagnostic_only": True, "effect_on_formal_dev_selection": "none",
              "forward_count": 0, "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
              "implementation_sha256": sha(__file__)}
    try:
        require(time.time() < deadline, "Deadline before model load")
        result["gpu_context_before_load"] = gpu_snapshot(args.devices)
        atomic_json(out / "inference-started.json", {"diagnostic_pid": os.getpid(), "deadline_utc": result["deadline_utc"],
                    "models": models, "gpu_context_before_load": result["gpu_context_before_load"]})
        model = gpu.load_model(args.model_dir, out / "backend", args.devices, args.cache_dir)
        for sample in samples:
            require(time.time() < deadline, "Deadline before forward")
            start = time.monotonic()
            positions = [r["token_position"] for r in sample["content"]]
            capture = gpu.HeadCapture(positions, selected=probes["pilot"].metadata["heads"])
            with torch.inference_mode(), capture.install(model):
                hidden = gpu._forward(model, sample["input_ids"])
                require(bool(torch.isfinite(hidden).all()), "Nonfinite model hidden state")
                del hidden
            x = capture.matrix()
            require(x.shape == (916, 1024), "Shared feature shape differs")
            row = {"metric_kind": "formal_role" if sample["prompt_key"] == "proper_tags" else "original_semantic_alignment",
                   "input_tokens": len(sample["input_ids"]), "body_rows": len(x), "models": {}}
            for key, probe in probes.items():
                p = probe.predict_proba(x)
                name = sample["prompt_key"] + "-" + key + "-probabilities.npy"
                temp = out / (name + ".tmp")
                with temp.open("wb") as f:
                    np.save(f, p, allow_pickle=False)
                os.replace(temp, out / name)
                row["models"][key] = {"probabilities_file": name, "probabilities_sha256": sha(out / name),
                                       "metrics": metrics(sample, p)}
            row["elapsed_seconds"] = time.monotonic() - start
            result["formats"][sample["prompt_key"]] = row
            result["forward_count"] += 1
            print(canonical({"stage": "gardening_forward", "format": sample["prompt_key"], "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        require(result["forward_count"] == 3, "Each author format must have exactly one shared forward")
        atomic_json(out / "results.json", result)
    finally:
        timer.cancel()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "check", "infer", "report"), default="prepare")
    parser.add_argument("--source-inputs", type=Path, default=HERE / "outputs/local-gptoss120b/tomato/inputs.json")
    parser.add_argument("--source-config", type=Path, default=HERE / "config/tomato.yaml")
    parser.add_argument("--source-template", type=Path, default=HERE.parents[1] / "utils/chat_templates/gptoss.j2")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pilot-run-dir", type=Path)
    parser.add_argument("--formal-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage in ("check", "infer"):
        require(args.pilot_run_dir is not None and args.formal_run_dir is not None, "Both run directories required")
    if args.stage == "check":
        _, models = validate_pair(args.pilot_run_dir, args.formal_run_dir, args.model_dir)
        print(canonical({"status": "ready", "models": models}), flush=True)
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    samples, manifest = prepare_samples(tokenizer, args.source_inputs, args.source_config, args.source_template)
    # Bind tokenizer bytes even when preparation deliberately precedes training.
    manifest["tokenizer_assets_sha256"] = {p.name: sha(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()
                                           and (p.name.startswith("tokenizer") or p.name == "special_tokens_map.json")}
    out = args.output_dir
    if args.stage == "prepare":
        for name, value in (("prepared-inputs.json", samples), ("prepare-manifest.json", manifest)):
            require(not (out / name).exists() or read(out / name) == value, "Existing preparation differs; use a new output directory")
            atomic_json(out / name, value)
        print(canonical({"status": "prepared", "formats": list(FORMATS), "body_rows_per_format": 916}), flush=True)
        return
    require(read(out / "prepared-inputs.json") == samples and read(out / "prepare-manifest.json") == manifest,
            "Prepared source/tokenizer changed")
    if args.stage == "infer":
        require(not (out / "results.json").exists(), "Committed diagnosis exists; report it or use a new output directory")
        infer(args, samples, manifest)
    else:
        results = read(out / "results.json")
        require(results["prepared_inputs_sha256"] == sha(out / "prepared-inputs.json") and
                results["prepare_manifest_sha256"] == sha(out / "prepare-manifest.json") and results["forward_count"] == 3,
                "Inference/preparation binding changed")
        require(set(results["formats"]) == set(FORMATS) and set(results["models"]) == {"pilot", "first_formal"},
                "Required three formats/two models missing")
        for key, row in results["formats"].items():
            require(set(row["models"]) == {"pilot", "first_formal"}, "Format missing a comparison model")
            sample = next(s for s in samples if s["prompt_key"] == key)
            for model in row["models"].values():
                path = out / model["probabilities_file"]
                require(path.resolve().is_relative_to(out.resolve()) and sha(path) == model["probabilities_sha256"], "Probability file changed")
                require(metrics(sample, np.load(path, allow_pickle=False)) == model["metrics"], "Metrics/probabilities differ")
        from synthetic_gardening_plots import plot_gardening
        results["artifact_dir"] = str(out.resolve())
        index = plot_gardening(samples, results, out)
        figures = validate_figures(index, out)
        atomic_json(out / "report-complete.json", {"format_version": FORMAT_VERSION, "diagnostic_only": True,
                    "results_sha256": sha(out / "results.json"), "index_sha256": sha(out / "index.json"),
                    "figures": figures, "required_figure_pairs": 7, "missing_figure_files": 0,
                    "implementation_sha256": sha(__file__)})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit("Gardening diagnosis refused: " + str(error)) from error
