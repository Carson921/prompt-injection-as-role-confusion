"""Independent byte, provenance, split, standardizer and NumPy forward checks.

This verifier imports neither the training runner nor its classifier/metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
TAG_ROLE = {"system": 0, "developer": 1, "user": 2, "assistant_analysis": 3,
            "assistant_final": 4, "assistant_commentary": 4, "assistant_unspecified": 4, "tool": 5}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y, p):
    cm = np.zeros((6, 6), np.int64)
    np.add.at(cm, (y, p.argmax(1)), 1)
    tp, support, predicted = cm.diagonal().astype(float), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(6), where=support + predicted > 0)
    return {"confusion": cm.tolist(), "accuracy": float(tp.sum() / len(y)), "macro_f1": float(f1[support > 0].mean()),
            "roles": {role: {"support": int(support[i]), "precision": float(precision[i]) if predicted[i] else None,
                             "recall": float(recall[i]) if support[i] else None, "f1": float(f1[i]) if support[i] else None}
                      for i, role in enumerate(ROLES)}}


def check_metrics(y, p, saved):
    actual = metrics(y, p)
    require(actual["confusion"] == saved["confusion"], "Confusion does not reproduce")
    for key in ("accuracy", "macro_f1"):
        require(abs(actual[key] - saved[key]) < 1e-12, "Metric does not reproduce: " + key)
    for role in ROLES:
        for key in ("support", "precision", "recall", "f1"):
            a, b = actual["roles"][role][key], saved["roles"][role][key]
            require(a == b or a is not None and b is not None and abs(a-b) < 1e-12, "Role metric differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    protocol, assignments = read(root / "protocol.json"), read(root / "split-assignment.json")
    manifest, audit = read(root / "features/manifest.json"), read(root / "data-audit.json")
    freeze, selection, training = (read(root / "run" / f"{name}.json") for name in ("freeze", "selection", "training"))
    require(sha(root / "features/manifest.json") == protocol["derived_feature_manifest_sha256"], "Manifest changed")
    require(sha(root / "split-assignment.json") == protocol["split_assignment_sha256"], "Assignment changed")
    for key, path in (("probe_sha256", "best.npz"), ("selection_sha256", "selection.json"), ("training_sha256", "training.json")):
        require(freeze[key] == sha(root / "run" / path), "Freeze changed: " + key)
    require(sha(root / "checkpoint.npz") == freeze["probe_sha256"], "Export differs")
    require(freeze["selection_split"] == "dev", "Non-DEV checkpoint selection")
    require(training["options"]["profile_weights"] == {"c4_dolma3": .5, "agentdojo": .5}, "Wrong domain sampling")
    require(not training["options"]["allow_cpu_smoke"], "Smoke run")
    require(training["phase"] == "c4-single-plus-agentdojo-native", "Wrong experiment")
    for record in selection["candidate_results"]:
        require(all(abs(record["sampled_profile_proportions"][d] - .5) < .005 for d in ("c4_dolma3", "agentdojo")), "Wrong domain draws")
    def selection_key(record):
        dev = record["dev"]
        return (not dev["meets_dev_floor"], -dev["structure_equal_macro_f1"], -dev["minimum_precision_recall"],
                dev["overall"]["nll"], record["candidate_id"])
    require(min(selection["candidate_results"], key=selection_key)["candidate_id"] == selection["candidate_id"], "DEV winner differs")
    gpu = read(root / "gpu-training.json")
    require(gpu["uuid"] == protocol["device_uuid"] and "2080 Ti" in gpu["name"], "Wrong GPU")
    with np.load(root / "checkpoint.npz", allow_pickle=False) as z:
        parameters = {key: z[key].copy() for key in z.files}
    metadata = json.loads(str(parameters["metadata"]))
    require(metadata["heads"] == protocol["heads"] == manifest["heads"], "Head order differs")
    mean, scale = parameters["mean"], parameters["scale"]
    def forward(x):
        h = (x.astype(np.float32) - mean) / scale
        h = h @ parameters["weight::0.weight"].T + parameters["weight::0.bias"]
        if metadata["hidden"]:
            h = .5 * h * (1 + erf(h / np.float32(np.sqrt(2))))
            h = h @ parameters["weight::3.weight"].T + parameters["weight::3.bias"]
        p = np.exp(h - h.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    article_root = Path(protocol["article_feature_root"])
    article_manifest = read(article_root / "manifest.json")
    require(sha(article_root / "manifest.json") == manifest["derivation"]["article_manifest_sha256"], "Article source changed")
    origin_offsets = {s: np.cumsum([0] + [e["rows"] for e in entries]) for s, entries in article_manifest["splits"].items()}
    native_root = Path(protocol["agentdojo_root"])
    require(sha(native_root / "result_manifest.json") == manifest["derivation"]["agentdojo_result_manifest_sha256"], "AgentDojo source changed")
    source_files = {entry["path"]: entry["sha256"] for entry in read(native_root / "result_manifest.json")["files"]}
    require(sha(native_root / "analysis/call_body_probabilities.csv") == source_files["analysis/call_body_probabilities.csv"],
            "Native supervision masks changed")
    coordinate_labels = {}
    with (native_root / "analysis/call_body_probabilities.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["content_filter"] != "normal_content":
                continue
            indices = json.loads(row["token_indices"])
            require(not indices or row["structural_tool_call"] == "False", "Tool call JSON in main mask")
            for index in indices:
                key = (row["call_id"], row["scope"], index)
                require(key not in coordinate_labels, "Repeated original mask position")
                coordinate_labels[key] = (TAG_ROLE[row["tag"]], "agentdojo:" + row["task_key"])
    require(len(coordinate_labels) == 155103, "Changed native mask denominator")
    checked_files, consumed_coordinates = set(), set()
    source_sets, family_sets, result = {}, {}, {}
    sums, squares, fit_count = np.zeros(1024), np.zeros(1024), 0
    total_by_domain = Counter()
    def checked(path, expected):
        key = str(path)
        if key not in checked_files:
            require(sha(path) == expected, "Source bytes changed: " + key)
            checked_files.add(key)

    for split in ("fit", "dev", "test"):
        published = np.load(root / f"run/evaluations/{split}-probabilities.npy", mmap_mode="r")
        saved = read(root / f"run/evaluations/{split}.json")
        require(saved["probe_sha256"] == freeze["probe_sha256"], "Evaluated wrong probe")
        if split == "test":
            require((root / "run/freeze.json").stat().st_mtime <= (root / "run/evaluations/test.json").stat().st_mtime, "TEST predates freeze")
        source_sets[split], family_sets[split], ys, structures, groups = set(), set(), [], [], []
        offset, max_error, disagreements = 0, 0., 0
        split_result = {}
        for entry in manifest["splits"][split]:
            folder = root / "features" / entry["path"]
            checked(folder / "complete.json", entry["sha256"])
            commit = read(folder / "complete.json")
            for name, digest in commit["files"].items():
                checked(folder / name, digest)
            arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in folder.glob("*.npy")}
            n = entry["rows"]
            x, y, source, family = (arrays[k] for k in ("x", "y", "source", "family"))
            require(x.shape == (n, 1024) and x.dtype == np.float16 and np.isfinite(x).all(), "Wrong feature geometry")
            require(len(np.unique(family)) == n and not (set(map(str, family)) & family_sets[split]), "Repeated family")
            require(np.array_equal(arrays["weight"], np.ones(n)), "Unexpected family weighting")
            domain = json.loads(str(arrays["groups"][0]))["domain"]
            require(all(assignments["domains"][domain][str(s)] == split for s in np.unique(source)), "Group assigned to wrong split")
            require(len(np.unique(source)) == audit["splits"][split][domain]["groups"], "Wrong group count")
            require(n == audit["splits"][split][domain]["rows"], "Wrong row count")
            total_by_domain[domain] += n
            if domain == "c4_dolma3":
                for old_split in np.unique(arrays["origin_partition"]):
                    positions = np.flatnonzero(arrays["origin_partition"] == old_split)
                    raw_rows = arrays["origin_row"][positions]
                    shard_ids = np.searchsorted(origin_offsets[str(old_split)][1:], raw_rows, side="right")
                    for shard_id in np.unique(shard_ids):
                        chosen = positions[shard_ids == shard_id]
                        raw_entry = article_manifest["splits"][str(old_split)][int(shard_id)]
                        raw_folder = article_root / raw_entry["path"]
                        checked(raw_folder / "complete.json", raw_entry["sha256"])
                        raw_commit = read(raw_folder / "complete.json")
                        local = arrays["origin_row"][chosen] - origin_offsets[str(old_split)][shard_id]
                        for key in ("x", "y", "source", "family", "structure", "groups"):
                            digest = raw_commit["files"][key + ".npy"]
                            checked(raw_folder / (key + ".npy"), digest["sha256"] if isinstance(digest, dict) else digest)
                            raw = np.load(raw_folder / (key + ".npy"), mmap_mode="r")[local]
                            if key in ("x", "y"):
                                require(np.array_equal(raw, arrays[key][chosen]), "Article raw vector/label differs")
                            elif key in ("source", "family"):
                                require(np.array_equal(np.asarray(["article:" + str(v) for v in raw]), arrays[key][chosen]), "Article provenance differs")
                            elif key == "structure":
                                require(np.char.startswith(raw, "single:").all(), "Non-single article leaked")
                            else:
                                require(all(json.loads(str(g))["profile"] == "single" and json.loads(str(g))["rounds"] == 0 for g in np.unique(raw)), "Wrong article profile")
            else:
                require(domain == "agentdojo", "Unknown domain")
                for call_id in np.unique(arrays["call_id"]):
                    for scope in ("actual_input", "actual_generated"):
                        positions = np.flatnonzero((arrays["call_id"] == call_id) & (arrays["scope"] == scope))
                        if not len(positions):
                            continue
                        filename = "input.npz" if scope == "actual_input" else "generation_probe.npz"
                        relative = f"calls/{call_id}/{filename}"
                        checked(native_root / relative, source_files[relative])
                        token_indices = arrays["token_index"][positions]
                        with np.load(native_root / relative, allow_pickle=False) as z:
                            require(np.array_equal(x[positions], z["head_features"][token_indices]), "Native raw vector differs")
                            require(z["head_ids"].tolist() == protocol["heads"] and z["roles"].tolist() == list(ROLES), "Native feature binding differs")
                            if scope == "actual_generated":
                                require(z["captured"][token_indices].all(), "Unforwarded stop entered data")
                        for pos, token_index in zip(positions, token_indices):
                            key = (str(call_id), scope, int(token_index))
                            require(key not in consumed_coordinates, "Native position used twice")
                            consumed_coordinates.add(key)
                            require(coordinate_labels[key] == (int(y[pos]), str(source[pos])), "Native target/task differs")
                            require(TAG_ROLE[str(arrays["tag"][pos])] == int(y[pos]), "Role/channel mapping differs")
            for start in range(0, n, 8192):
                stop = min(n, start + 8192)
                raw = x[start:stop]
                if split == "fit":
                    values = raw.astype(np.float64)
                    sums += values.sum(0)
                    squares += (values * values).sum(0)
                    fit_count += len(raw)
                independent = forward(raw)
                actual = published[offset + start:offset + stop]
                max_error = max(max_error, float(np.max(abs(independent - actual))))
                disagreements += int(np.count_nonzero(independent.argmax(1) != actual.argmax(1)))
            metric = saved["by_attribute"]["domain"][domain]
            check_metrics(y, published[offset:offset+n], metric)
            split_result[domain] = {"rows": n, "groups": len(np.unique(source)), "metrics_reproduced": True,
                                    "original_feature_rows_bit_exact": True}
            source_sets[split].update(map(str, source))
            family_sets[split].update(map(str, family))
            ys.append(np.array(y)); structures.append(np.array(arrays["structure"])); groups.extend(map(str, arrays["groups"]))
            offset += n
        require(offset == len(published), "Probability row count differs")
        y, structure = np.concatenate(ys), np.concatenate(structures)
        check_metrics(y, published, saved["overall"])
        for value, metric in saved["by_structure"].items():
            mask = structure == value
            check_metrics(y[mask], published[mask], metric)
        objects = [json.loads(g) for g in groups]
        for key, values in saved["by_attribute"].items():
            attributes = np.asarray([str(g[key]) for g in objects])
            for value, metric in values.items():
                mask = attributes == value
                check_metrics(y[mask], published[mask], metric)
        require(max_error < 5e-4 and disagreements == 0, "NumPy/GPU predictions disagree")
        result[split] = {"domains": split_result, "rows": len(y), "max_numpy_gpu_probability_error": max_error,
                         "argmax_disagreements": disagreements, "all_metrics_reproduced": True}
        print(json.dumps({"stage": "independent_validation", "split": split, **result[split]}), flush=True)
    require(total_by_domain == {"c4_dolma3": 546304, "agentdojo": 155103}, "Wrong total inventory")
    require(consumed_coordinates == set(coordinate_labels), "Native rows missing or extra")
    overlaps = {}
    for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test")):
        overlaps[a + "/" + b] = len(source_sets[a] & source_sets[b])
        require(not overlaps[a + "/" + b] and not family_sets[a] & family_sets[b], "Cross-split leakage")
    independent_mean = sums / fit_count
    independent_scale = np.sqrt(np.maximum(squares / fit_count - independent_mean ** 2, 0))
    independent_scale[independent_scale < 1e-6] = 1
    require(np.max(abs(independent_mean.astype(np.float32) - mean)) < 2e-6, "FIT mean differs")
    require(np.max(abs(independent_scale.astype(np.float32) - scale)) < 2e-6, "FIT scale differs")
    report = {"status": "passed", "splits": result, "cross_split_source_overlap": overlaps,
              "fit_standardizer_independently_recomputed": True, "native_labels_independently_reconstructed": True,
              "native_positions": len(consumed_coordinates), "source_files_verified": len(checked_files),
              "checkpoint_sha256": freeze["probe_sha256"], "dev_only_selection_reproduced": True}
    (root / "independent-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
