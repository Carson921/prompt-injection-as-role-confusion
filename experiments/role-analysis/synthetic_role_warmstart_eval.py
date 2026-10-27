"""Post-freeze CPU readout of the sealed 57-task native feature observations.

The source archives are observations, not training data.  This module never
loads the language model or changes the archived features, masks, or outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import synthetic_agent_classifier as classifier

ROLES = tuple(classifier.ROLES)
SCOPES = ("actual_input", "actual_generated")
TAG_ROLE = {"system": 0, "developer": 1, "user": 2, "assistant_analysis": 3,
            "assistant_final": 4, "assistant_commentary": 4,
            "assistant_unspecified": 4, "tool": 5}
TAGS = tuple(TAG_ROLE)
FEATURE_KEYS = ("roles", "heads", "feature_dim", "feature_position", "model_binding")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0]) if rows else fieldnames
    require(bool(names), "CSV needs a nonempty schema")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


class SourceFiles:
    """Validate consumed bytes against the existing result sealing manifest."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        path = self.root / "result_manifest.json"
        manifest = read_json(path)
        require(manifest.get("status") == "complete", "Source result is not sealed complete")
        self.manifest_sha256 = sha256(path)
        self.entries = {}
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "Unsafe path in source manifest")
            require(relative.as_posix() not in self.entries, "Duplicate sealed source path")
            self.entries[relative.as_posix()] = entry
        self.consumed = {}

    def verify(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "Source artifact escapes source result")
        name = path.relative_to(self.root).as_posix()
        if name not in self.consumed:
            require(name in self.entries, f"Unsealed source artifact: {name}")
            expected = self.entries[name]
            digest = sha256(path)
            require(path.stat().st_size == expected["bytes"] and digest == expected["sha256"],
                    f"Sealed source bytes changed: {name}")
            self.consumed[name] = {"bytes": expected["bytes"], "sha256": digest}
        return path


def validate_freeze(probe_path, parent_path, freeze_path):
    freeze = read_json(freeze_path)
    probe_digest, parent_digest = sha256(probe_path), sha256(parent_path)
    require(freeze.get("probe_sha256") == probe_digest, "Frozen probe SHA mismatch")
    require(freeze.get("selection_split") == "dev", "Selection must use DEV only")
    require(isinstance(freeze.get("frozen_at_unix"), (int, float)) and
            0 < freeze["frozen_at_unix"] <= time.time() + 1,
            "Freeze has no valid past timestamp")
    if "parent_probe_sha256" in freeze:
        require(freeze["parent_probe_sha256"] == parent_digest, "Frozen parent SHA mismatch")
    probe, parent = classifier.Probe(probe_path), classifier.Probe(parent_path)
    for key in FEATURE_KEYS:
        require(probe.metadata[key] == parent.metadata[key], f"Cached feature binding changed: {key}")
    if probe_digest != parent_digest or freeze.get("format_version") == "synthetic-role-warm-freeze-v1":
        require(freeze.get("format_version") == "synthetic-role-warm-freeze-v1", "New probe needs a warm-training freeze")
        require(freeze.get("parent_probe_sha256") == parent_digest, "Warm freeze omits the parent identity")
        for key, name in (("training_sha256", "training.json"), ("selection_sha256", "selection.json")):
            require(freeze.get(key) == sha256(Path(freeze_path).parent / name), "Frozen training/selection bytes changed")
        for key in FEATURE_KEYS:
            require(freeze.get("feature_binding", {}).get(key) == parent.metadata[key],
                    "Frozen feature identity differs: " + key)
        require(probe.metadata.get("warm_start", {}).get("parent_probe_sha256") == parent_digest,
                "New checkpoint omits warm-start/standardizer parent lineage")
    return probe, parent, freeze


def load_masks(path):
    """Keep the existing first-position selections; never text-deduplicate."""
    rows = [row for row in csv_rows(path) if row["content_filter"] == "normal_content"]
    require(bool(rows), "No normal-content first-position masks")
    seen_positions, seen_frames = set(), set()
    groups = defaultdict(list)
    for row in rows:
        require(row["scope"] in SCOPES and row["tag"] in TAG_ROLE, "Unknown role/scope mask")
        indices = np.asarray(json.loads(row["token_indices"]), dtype=np.int64)
        require(indices.ndim == 1 and len(indices) == int(row["tokens"]), "Mask token count differs")
        require(len(indices) == len(set(indices.tolist())) and (indices >= 0).all(), "Invalid mask indices")
        require(len(indices) < 2 or bool((np.diff(indices) > 0).all()), "Mask indices are not ordered")
        require(not len(indices) or row["structural_tool_call"] == "False", "Tool-call JSON entered main mask")
        for index in indices:
            key = (row["scope"], row["call_id"], int(index))
            require(key not in seen_positions, "Overlapping first-position masks")
            seen_positions.add(key)
        stable = row.get("stable_frame_key") or (row["call_id"] + "/frame_" + row["frame_index"])
        frame_key = (row["task_key"], row["scope"], stable)
        require(frame_key not in seen_frames, "Repeated stable body frame in retained masks")
        seen_frames.add(frame_key)
        row = {**row, "indices": indices, "label": TAG_ROLE[row["tag"]]}
        groups[(row["call_id"], row["scope"])].append(row)
    return rows, groups


def validate_archive(arrays, request, scope, parent):
    x = arrays["head_features"]
    require(x.dtype == np.float16 and x.ndim == 2 and x.shape[1] == 1024,
            "Cached features must be raw FP16 Nx1024")
    n = len(x)
    require(arrays["roles"].tolist() == list(ROLES), "Cached role order differs")
    require(arrays["head_ids"].tolist() == parent.metadata["heads"], "Cached head order differs")
    require(request["head_ids"] == parent.metadata["heads"] and request["roles"] == list(ROLES)
            and request["feature_dim"] == 1024, "Request feature binding differs")
    probabilities = arrays["probabilities"]
    require(probabilities.dtype == np.float32 and probabilities.shape == (n, 6), "Bad archived probability shape")
    if scope == "actual_input":
        observed = np.ones(n, dtype=bool)
        require(n == request["input_tokens"], "Input feature count differs from actual input")
        require(np.array_equal(arrays["token_index"], np.arange(n)), "Input index shifted")
    else:
        observed = arrays["captured"]
        require(observed.dtype == bool and observed.shape == (n,), "Invalid generated capture mask")
        require(np.array_equal(arrays["generated_ids"], request["generated_ids"]), "Generated IDs changed")
        require(np.array_equal(arrays["generated_token_index"], np.arange(n)), "Generated index shifted")
        require(np.array_equal(arrays["full_sequence_token_index"], request["input_tokens"] + np.arange(n)),
                "Generated readout is not each emitted token's own state")
        missing = np.flatnonzero(~observed)
        require(all(i == n - 1 and int(arrays["generated_ids"][i]) in (200002, 200012) for i in missing),
                "Only unforwarded final stop delimiters may lack features")
        sources = arrays["capture_source"]
        require(sources.shape == (n,) and bool(np.isin(sources[observed],
                ["native_decode_forward", "terminal_completion_forward"]).all()), "Unexpected native readout source")
        require(bool((sources[~observed] == "not_forwarded_stop_delimiter").all()), "Missing stop source differs")
        terminal = np.flatnonzero(sources == "terminal_completion_forward")
        require(all(i == n - 1 and int(arrays["generated_ids"][i]) not in (200002, 200012)
                    for i in terminal), "Invalid terminal ordinary-token completion")
        require(np.isnan(x[~observed]).all() and np.isnan(probabilities[~observed]).all(),
                "Unobserved stop values must remain NaN")
    require(np.isfinite(x[observed]).all(), "Observed feature is nonfinite")
    p = probabilities[observed]
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and
            np.max(abs(p.sum(1) - 1), initial=0) <= 3e-5, "Invalid observed archived probabilities")
    require(arrays["token_char_offsets"].shape == (n, 2), "Missing token offset map")
    return observed


def sd_segments(tag, prompt, start, end, offsets):
    """Predeclared text-anchor partitions; boundary-crossing tokens use midpoint."""
    text = prompt[start:end]
    if tag == "developer":
        anchor = text.find("# Tools")
        cuts = [(0, "developer/instructions")]
        if anchor >= 0:
            cuts.append((anchor, "developer/tool_schema"))
    elif tag == "system":
        cuts = [(0, "system/identity_and_date")]
        for anchor, name in [("Reasoning:", "reasoning_setting"),
                             ("# Valid channels:", "channel_rules"),
                             ("Calls to these tools", "tool_routing")]:
            location = text.find(anchor)
            if location >= 0:
                cuts.append((location, "system/" + name))
    else:
        return np.full(len(offsets), "", dtype="U40")
    cuts.sort()
    midpoints = np.clip(offsets.mean(axis=1) - start, 0, max(0, len(text) - .5))
    locations = np.searchsorted([cut[0] for cut in cuts], midpoints, side="right") - 1
    return np.asarray([cuts[i][1] for i in locations], dtype="U40")


def metric_report(y, probabilities):
    if len(y):
        result = classifier.metrics(probabilities, y)
    else:
        result = classifier._metrics_from_confusion(np.zeros((6, 6), np.int64))
        result["accuracy"] = None
    p = probabilities.argmax(1) if len(y) else np.array([], np.int64)
    low = np.isin(y, (2, 5))
    result["user_tool_to_system_developer"] = {
        "denominator": int(low.sum()), "false_positives": int(np.isin(p[low], (0, 1)).sum()),
        "rate": float(np.isin(p[low], (0, 1)).mean()) if low.any() else None}
    return result


def metric_row(model, dimensions, metric):
    row = {"model": model, **dimensions, "tokens": metric["tokens"],
           "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
           "user_tool_to_system_developer": metric["user_tool_to_system_developer"]["rate"]}
    for role in ROLES:
        for key in ("support", "precision", "recall", "f1", "false_positives"):
            row[role + "_" + key] = metric["roles"][role][key]
    return row


def probability_tables(y, probabilities, scopes, tags, tasks):
    per_task, summary = [], []
    all_tasks = sorted(set(tasks.tolist()))
    for scope in SCOPES:
        for tag in TAGS:
            values, sums, tokens = [], np.zeros(6, np.float64), 0
            for task in all_tasks:
                mask = (scopes == scope) & (tags == tag) & (tasks == task)
                n = int(mask.sum())
                mean = probabilities[mask].astype(np.float64).mean(0) if n else None
                item = {"task_key": task, "scope": scope, "content_filter": "normal_content", "tag": tag, "tokens": n}
                item.update({"p_" + role: None if mean is None else float(mean[j]) for j, role in enumerate(ROLES)})
                per_task.append(item)
                if n:
                    values.append(mean)
                    sums += mean * n
                    tokens += n
            for aggregation in ("unique_task_equal_mean", "token_pooled_mean"):
                mean = (np.mean(values, axis=0) if aggregation == "unique_task_equal_mean" else sums / tokens) if tokens else None
                item = {"scope": scope, "content_filter": "normal_content", "tag": tag,
                        "aggregation": aggregation, "scheduled_unique_tasks": len(all_tasks),
                        "tasks_observed": len(values), "tasks_without_tag": len(all_tasks) - len(values), "tokens": tokens}
                item.update({"p_" + role: None if mean is None else float(mean[j]) for j, role in enumerate(ROLES)})
                summary.append(item)
    return per_task, summary


def compare_original_summary(summary, source_path, *, tolerance=1e-11):
    source = {(r["scope"], r["tag"]): r for r in csv_rows(source_path)
              if r["content_filter"] == "normal_content" and r["aggregation"] == "unique_task_equal_mean"}
    error = 0.0
    for row in summary:
        if row["aggregation"] != "unique_task_equal_mean":
            continue
        expected = source[(row["scope"], row["tag"])]
        for key in ("tokens", "scheduled_unique_tasks", "tasks_observed", "tasks_without_tag"):
            require(row[key] == int(expected[key]), f"Archived summary denominator differs: {key}")
        for role in ROLES:
            value = row["p_" + role]
            if value is None:
                require(expected["p_" + role] == "", "Absent group was zero-filled")
            else:
                error = max(error, abs(value - float(expected["p_" + role])))
    require(error <= tolerance, f"Archived first-position summary differs: {error}")
    return error


def make_plots(output, reports, groups, labels, probabilities, *, model_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    model_labels = model_labels or {"parent": "parent", "new": "new"}
    plot_paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, name in zip(axes, ("parent", "new")):
        matrix = np.asarray(reports[name]["overall"]["confusion"])
        denominators = matrix.sum(1, keepdims=True)
        rate = np.divide(matrix, denominators, out=np.zeros_like(matrix, dtype=float), where=denominators > 0)
        heat = axis.imshow(rate, vmin=0, vmax=1, cmap="Blues")
        axis.set(xticks=range(6), yticks=range(6), xticklabels=ROLES, yticklabels=ROLES,
                 xlabel="Predicted role", ylabel="Native enclosing role",
                 title=f"{model_labels[name]}: row-normalized confusion")
        axis.tick_params(axis="x", rotation=35)
        for i in range(6):
            for j in range(6):
                axis.text(j, i, f"{100*rate[i,j]:.1f}%", ha="center", va="center",
                          fontsize=8, color="white" if rate[i, j] > .55 else "black")
    fig.colorbar(heat, ax=axes, shrink=.8, label="Fraction of actual role")
    for ext in ("png", "pdf"):
        path = output / f"role-confusion-parent-vs-new.{ext}"
        fig.savefig(path, dpi=180)
        plot_paths.append(path.name)
    plt.close(fig)
    names = sorted(set(groups.tolist()) - {""})
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .48 * len(names) + 1.5)), layout="constrained")
    for axis, model in zip(axes, ("parent", "new")):
        matrix = np.asarray([probabilities[model][groups == name].mean(0) for name in names])
        heat = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set(xticks=range(6), yticks=range(len(names)), xticklabels=ROLES,
                 yticklabels=[f"{n} (n={int((groups == n).sum()):,})" for n in names],
                 title=model_labels[model] + ": S/D segment mean probabilities", xlabel="Readout role")
        axis.tick_params(axis="x", rotation=35)
        for i in range(len(names)):
            for j in range(6):
                axis.text(j, i, f"{100*matrix[i,j]:.1f}", ha="center", va="center", fontsize=8,
                          color="black" if matrix[i, j] > .55 else "white")
    fig.colorbar(heat, ax=axes, shrink=.8, label="Mean probability")
    for ext in ("png", "pdf"):
        path = output / f"system-developer-segments.{ext}"
        fig.savefig(path, dpi=180)
        plot_paths.append(path.name)
    plt.close(fig)
    return plot_paths


def evaluate(calls_root, analysis_root, probe_path, parent_path, output_dir, *, freeze_path=None,
             expected_tasks=57, plots=True, save_probabilities=True):
    calls_root, analysis_root = Path(calls_root).resolve(), Path(analysis_root).resolve()
    source_root, output = calls_root.parent, Path(output_dir).resolve()
    require(not output.is_relative_to(source_root), "New outputs must be outside the sealed source result")
    require(not output.exists() or not any(output.iterdir()), "Output directory must be new or empty")
    freeze_path = Path(freeze_path) if freeze_path else Path(probe_path).parent / "freeze.json"
    probe, parent, freeze = validate_freeze(probe_path, parent_path, freeze_path)
    probe_digest, parent_digest, freeze_digest = sha256(probe_path), sha256(parent_path), sha256(freeze_path)
    source = SourceFiles(source_root)
    provenance = read_json(source.verify(source_root / "probe_provenance.json"))
    require(provenance["sha256"] == parent_digest, "Parent does not match source experiment")
    for key in ("model_binding", "feature_dim"):
        require(provenance[key] == parent.metadata[key], f"Source provenance differs: {key}")
    require(provenance["selected_head_ids"] == parent.metadata["heads"], "Source provenance head order differs")
    mask_path = source.verify(analysis_root / "call_body_probabilities.csv")
    rows, by_call = load_masks(mask_path)
    source_metrics = read_json(source.verify(analysis_root / "metrics.json"))
    require(source_metrics.get("valid") and source_metrics.get("complete"), "Source analysis is not valid complete")
    task_keys = sorted({row["task_key"] for row in rows})
    require(len(task_keys) == expected_tasks == source_metrics["scheduled_unique_tasks"], "Task coverage differs")
    output.mkdir(parents=True, exist_ok=True)
    predictions, label_blocks = {"parent": [], "new": [], "archived_parent": []}, []
    metadata = {key: [] for key in ("scope", "tag", "task", "suite", "segment", "quartile")}
    templates = {"system": set(), "developer": set()}
    counters, capture_sources = Counter(), Counter()
    call_rows, per_body = [], []
    max_probability_error = max_body_error = 0.0
    parent_argmax_mismatches = 0
    selected_counts = Counter()
    seen_call_scopes = set()
    request_paths = sorted(calls_root.glob("*/*/*/call_*/request.json"))
    require(len(request_paths) == source_metrics["calls"], "Request inventory differs")
    for request_path in request_paths:
        request = read_json(source.verify(request_path))
        call_id = request_path.parent.relative_to(calls_root).as_posix()
        require(request["status"] == "complete" and request["metadata"]["call_id"] == call_id,
                "Call identity differs from artifact path")
        require(request["probe_sha256"] == provenance["sha256"], "Archived request has a different parent probe")
        counters["calls"] += 1
        for scope, filename in zip(SCOPES, ("input.npz", "generation_probe.npz")):
            artifact = source.verify(request_path.parent / filename)
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            observed = validate_archive(arrays, request, scope, parent)
            counters[scope + "/all_rows"] += len(observed)
            counters[scope + "/observed_rows"] += int(observed.sum())
            if scope == "actual_generated":
                counters["unforwarded_stop_delimiters"] += int((~observed).sum())
                capture_sources.update(arrays["capture_source"].tolist())
            observed_parent = parent.predict_proba(arrays["head_features"][observed])
            observed_new = probe.predict_proba(arrays["head_features"][observed])
            error = float(np.max(abs(observed_parent - arrays["probabilities"][observed]), initial=0))
            max_probability_error = max(max_probability_error, error)
            require(error <= 5e-4, f"Parent readout formula does not reproduce source: {call_id}/{scope}")
            parent_argmax_mismatches += int((observed_parent.argmax(1) != arrays["probabilities"][observed].argmax(1)).sum())
            model_p = {"archived_parent": arrays["probabilities"],
                       "parent": np.full((len(observed), 6), np.nan, np.float32),
                       "new": np.full((len(observed), 6), np.nan, np.float32)}
            model_p["parent"][observed], model_p["new"][observed] = observed_parent, observed_new
            if save_probabilities:
                destination = output / "probabilities" / call_id / (scope + ".npz")
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, probabilities=model_p["new"], captured=observed,
                                    token_index=np.arange(len(observed)), roles=np.asarray(ROLES),
                                    source_npz_sha256=np.array(source.consumed[artifact.relative_to(source_root).as_posix()]["sha256"]),
                                    probe_sha256=np.array(probe_digest))
            for row in by_call.get((call_id, scope), []):
                indices = row["indices"]
                seen_call_scopes.add((call_id, scope))
                require(len(indices) == 0 or indices[-1] < len(observed), "Mask index exceeds actual token rows")
                require(bool(observed[indices].all()), "Mask selects an unforwarded stop token")
                old_sums = arrays["probabilities"][indices].astype(np.float64).sum(0)
                max_body_error = max(max_body_error, float(np.max(abs(old_sums -
                    np.array([float(row["sum_" + role]) for role in ROLES])), initial=0)))
                require(max_body_error <= 1e-9, "Source mask probability sums do not match sealed body table")
                n = len(indices)
                selected_counts[scope + "/" + row["tag"]] += n
                y = np.full(n, row["label"], np.int64)
                label_blocks.append(y)
                for name, values in model_p.items():
                    predictions[name].append(values[indices])
                metadata["scope"].extend([scope] * n)
                metadata["tag"].extend([row["tag"]] * n)
                metadata["task"].extend([row["task_key"]] * n)
                metadata["suite"].extend([row["task_key"].split("/")[0]] * n)
                metadata["quartile"].extend((np.arange(n) * 4 // max(1, n)).tolist())
                offsets = arrays["token_char_offsets"][indices]
                start, end = int(row["body_char_start"]), int(row["body_char_end"])
                segments = sd_segments(row["tag"], request["rendered_prompt"], start, end, offsets)
                metadata["segment"].extend(segments.tolist())
                if row["tag"] in templates and scope == "actual_input":
                    templates[row["tag"]].add(hashlib.sha256(request["rendered_prompt"][start:end].encode()).hexdigest())
                body = {"task_key": row["task_key"], "call_id": call_id, "scope": scope,
                        "tag": row["tag"], "tokens": n, "stable_frame_key": row.get("stable_frame_key", "")}
                for name in ("parent", "new"):
                    values = model_p[name][indices]
                    metrics = metric_report(y, values)
                    body[name + "_accuracy"] = metrics["accuracy"]
                    body[name + "_mean_true_probability"] = float(values[:, row["label"]].mean()) if n else None
                per_body.append(body)
            call_rows.append({"call_id": call_id, "scope": scope, "rows": len(observed),
                              "captured_rows": int(observed.sum()), "parent_max_probability_error": error})
    require(set(by_call) == seen_call_scopes, "Some retained masks have no actual source archive")
    for key, expected in source_metrics["retained_normal_positions"].items():
        require(selected_counts[key] == expected, f"Retained normal-content denominator differs: {key}")
    require(counters["actual_input/all_rows"] == source_metrics["input_tokens"] and
            counters["actual_generated/all_rows"] == source_metrics["generated_tokens"] and
            counters["unforwarded_stop_delimiters"] == source_metrics["unforwarded_stop_delimiters"],
            "Full source feature inventory differs from sealed source analysis")
    labels = np.concatenate(label_blocks)
    predictions = {name: np.concatenate(blocks) for name, blocks in predictions.items()}
    metadata = {key: np.asarray(value) for key, value in metadata.items()}
    reports = {}
    for name, values in predictions.items():
        report = {"overall": metric_report(labels, values), "by_scope": {}, "by_suite": {}, "sd_segments": {},
                  "sd_body_quartiles": {}}
        for scope in SCOPES:
            m = metadata["scope"] == scope
            report["by_scope"][scope] = metric_report(labels[m], values[m])
        for suite in sorted(set(metadata["suite"].tolist())):
            m = metadata["suite"] == suite
            report["by_suite"][suite] = metric_report(labels[m], values[m])
        for segment in sorted(set(metadata["segment"].tolist()) - {""}):
            m = metadata["segment"] == segment
            report["sd_segments"][segment] = metric_report(labels[m], values[m])
        for role in ("system", "developer"):
            for quartile in range(4):
                m = (metadata["tag"] == role) & (metadata["quartile"] == quartile)
                report["sd_body_quartiles"][f"{role}/q{quartile + 1}"] = metric_report(labels[m], values[m])
        per_task, summary = probability_tables(labels, values, metadata["scope"], metadata["tag"], metadata["task"])
        write_csv(output / (name + "-per-task-probabilities.csv"), per_task)
        write_csv(output / (name + "-mean-probabilities.csv"), summary)
        if name == "archived_parent":
            summary_error = compare_original_summary(summary, source.verify(analysis_root / "summary.csv"))
        reports[name] = report
    per_task_metrics = []
    for task in task_keys:
        for scope in SCOPES:
            m = (metadata["task"] == task) & (metadata["scope"] == scope)
            for name in ("parent", "new"):
                metric = metric_report(labels[m], predictions[name][m])
                per_task_metrics.append(metric_row(name, {"task_key": task, "scope": scope}, metric))
    write_csv(output / "per-task-hard-metrics.csv", per_task_metrics)
    suite_rows, segment_rows = [], []
    for suite in sorted(set(metadata["suite"].tolist())):
        for scope in SCOPES:
            m = (metadata["suite"] == suite) & (metadata["scope"] == scope)
            for name in ("parent", "new"):
                metric = metric_report(labels[m], predictions[name][m])
                suite_rows.append(metric_row(name, {"suite": suite, "scope": scope}, metric))
    for segment in sorted(set(metadata["segment"].tolist()) - {""}):
        m = metadata["segment"] == segment
        for name in ("parent", "new"):
            row = metric_row(name, {"segment": segment}, metric_report(labels[m], predictions[name][m]))
            row.update({"p_" + role: float(predictions[name][m, i].mean()) for i, role in enumerate(ROLES)})
            segment_rows.append(row)
    write_csv(output / "per-suite-hard-metrics.csv", suite_rows)
    write_csv(output / "sd-segment-comparison.csv", segment_rows)
    write_csv(output / "per-body-comparison.csv", per_body)
    write_csv(output / "call-feature-audit.csv", call_rows)
    selected_path = output / "first-position-probabilities.npz"
    np.savez_compressed(selected_path, labels=labels, parent=predictions["parent"], new=predictions["new"],
                        archived_parent=predictions["archived_parent"], **metadata)
    proof = {"status": "passed", "source_result_manifest_sha256": source.manifest_sha256,
             "consumed_source_files": source.consumed, "counters": dict(counters),
             "generated_capture_sources": dict(capture_sources), "retained_body_rows": len(rows),
             "retained_normal_tokens": len(labels), "retained_normal_positions": dict(selected_counts),
             "no_text_deduplication": True, "input_generated_scopes_separate": True,
             "mask_policy": "Reuse sealed first-position normal-content indices; the original selector is not rerun.",
             "parent_probability_max_error": max_probability_error,
             "parent_argmax_mismatches_all_observed_rows": parent_argmax_mismatches,
             "archived_body_probability_sum_max_error": max_body_error,
             "archived_summary_max_error": summary_error,
             "unique_sd_template_counts": {key: len(value) for key, value in templates.items()},
             "sd_template_sha256": {key: sorted(value) for key, value in templates.items()},
             "first_position_npz_sha256": sha256(selected_path)}
    write_json(output / "cache-reuse-audit.json", proof)
    require(sha256(probe_path) == probe_digest and sha256(parent_path) == parent_digest and
            sha256(freeze_path) == freeze_digest, "Frozen probe or selection changed during evaluation")
    result = {"schema": "synthetic-role-warmstart-cache-evaluation-v1", "status": "complete",
              "probe_sha256": probe_digest, "parent_probe_sha256": parent_digest,
              "freeze_sha256": freeze_digest, "frozen_at_unix": freeze["frozen_at_unix"],
              "evaluated_at_unix": time.time(), "selection_split": "dev",
              "historical_regression_only": True, "new_language_model_forward": False,
              "native_cached_decode_preserved": True, "tasks": len(task_keys), "roles": list(ROLES),
              "template_counts": proof["unique_sd_template_counts"], "models": reports,
              "limitations": ["Observed 57-task regression set; not a new blind test.",
                              "Reuses the original four-A40 forward observations; does not validate a three-A40 deployment.",
                              "S/D prompt templates are repeated across tasks; task count is not independent template support.",
                              "No agent rollout, task utility, or attack success rate is remeasured."]}
    write_json(output / "metrics.json", result)
    plot_files = make_plots(output, reports, metadata["segment"], labels, predictions) if plots else []
    new, before = reports["new"]["overall"], reports["parent"]["overall"]
    lines = ["# AgentDojo 57 任务：冻结后缓存复评", "",
             "复用原始输入和真实 native decode 特征，不执行新的大模型前向，也不重跑 agent。",
             "本结果是已观察历史集上的回归测试，不能称为新的盲测或三卡部署验证。", "",
             f"正文评分位置 {len(labels):,}；system 模板 {len(templates['system'])} 种、developer 模板 {len(templates['developer'])} 种。",
             "任务数不能作为独立提示模板数。输入和生成分开统计；缺失停止标记保持排除。", "",
             "| 真实角色 | 正例 token | 父模型召回 | 新模型召回 |", "|---|---:|---:|---:|"]
    for role in ROLES:
        a, b = before["roles"][role], new["roles"][role]
        fmt = lambda value: "N/A" if value is None else f"{100*value:.4f}%"
        lines.append(f"| {role} | {b['support']:,} | {fmt(a['recall'])} | {fmt(b['recall'])} |")
    lines += ["", "均值概率另列，不能当作 argmax 准确率。完整 P/R/F1、混淆、分段及位置结果见 [metrics.json](metrics.json)。",
              "本轮未重新测量任务成功率或 ASR。", "", "校验与图表：", "",
              "- [缓存与掩码校验](cache-reuse-audit.json)", "- [逐任务硬分类指标](per-task-hard-metrics.csv)",
              "- [逐正文对比](per-body-comparison.csv)", "- [新模型任务等权／token加权概率](new-mean-probabilities.csv)"]
    lines.extend(["- [逐 suite 硬分类指标](per-suite-hard-metrics.csv)",
                  "- [S/D 分段对比](sd-segment-comparison.csv)"])
    lines.extend(f"- [{name}]({name})" for name in plot_files)
    (output / "README.zh-CN.md").write_text("\n".join(lines) + "\n")
    files = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output.rglob("*")) if path.is_file()]
    write_json(output / "result-manifest.json", {"status": "complete", "probe_sha256": result["probe_sha256"],
                                                "freeze_sha256": result["freeze_sha256"], "files": files})
    return {"status": "complete", "tasks": len(task_keys), "scored_tokens": len(labels),
            "parent_sha256": result["parent_probe_sha256"], "probe_sha256": result["probe_sha256"],
            "output_dir": str(output), "parent_feature_probability_max_error": max_probability_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls-root", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--parent-probe", required=True, type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.calls_root, args.analysis_root, args.probe, args.parent_probe,
                      args.output_dir, freeze_path=args.freeze)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
