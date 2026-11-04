"""Independent checks of the saved single-message experiment and predictions.

Does not import the training runner, feature loader, probe consumer or metrics.
Uses raw NumPy arrays and an explicit NumPy/SciPy forward for the exported probe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def measures(y, p):
    predictions = p.argmax(1)
    matrix = np.zeros((6, 6), np.int64)
    np.add.at(matrix, (y, predictions), 1)
    tp = matrix.diagonal().astype(float)
    support, predicted = matrix.sum(1), matrix.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(6), where=(support + predicted) > 0)
    return {"confusion": matrix.tolist(), "accuracy": float(tp.sum() / len(y)),
            "macro_f1": float(f1[support > 0].mean()),
            "roles": {r: {"support": int(support[i]), "precision": float(precision[i]) if predicted[i] else None,
                          "recall": float(recall[i]) if support[i] else None,
                          "f1": float(f1[i]) if support[i] else None} for i, r in enumerate(ROLES)}}


def compare_metric(actual, expected):
    require(actual["confusion"] == expected["confusion"], "Confusion differs from saved predictions")
    for key in ("accuracy", "macro_f1"):
        require(abs(actual[key] - expected[key]) < 1e-12, "Metric differs: " + key)
    for role in ROLES:
        for key in ("support", "precision", "recall", "f1"):
            a, b = actual["roles"][role][key], expected["roles"][role][key]
            require((a is None and b is None) or (a is not None and b is not None and abs(a-b) < 1e-12),
                    "Role metric differs: " + role + "/" + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    protocol, audit = read(root / "protocol.json"), read(root / "data-audit.json")
    manifest = read(root / "features/manifest.json")
    freeze = read(root / "run/freeze.json")
    selection, training = read(root / "run/selection.json"), read(root / "run/training.json")
    require(sha(root / "features/manifest.json") == protocol["derived_feature_manifest_sha256"], "Derived manifest changed")
    require(freeze["probe_sha256"] == sha(root / "run/best.npz"), "Frozen weights changed")
    require(freeze["selection_sha256"] == sha(root / "run/selection.json"), "Frozen selection changed")
    require(freeze["training_sha256"] == sha(root / "run/training.json"), "Frozen training changed")
    require(training["options"]["profile_weights"] == {"single": 1., "one_round": 0., "multi_round": 0.}, "Mixed profile training")
    require(training["phase"] == "single-message", "Wrong experiment phase")
    gpu = read(root / "gpu-training.json")
    require(gpu["name"] == "NVIDIA GeForce RTX 2080 Ti" and gpu["uuid"] == protocol["device_uuid"], "Wrong GPU")
    with np.load(root / "run/best.npz", allow_pickle=False) as z:
        parameters = {key: z[key].copy() for key in z.files}
    metadata = json.loads(str(parameters["metadata"]))
    require(not metadata["engineering_smoke_only"] and metadata["heads"] == protocol["heads"], "Wrong checkpoint binding")
    mean, scale = parameters["mean"], parameters["scale"]

    def forward(x):
        h = (x.astype(np.float32) - mean) / scale
        h = h @ parameters["weight::0.weight"].T + parameters["weight::0.bias"]
        if metadata["hidden"]:
            h = .5 * h * (1 + erf(h / np.float32(np.sqrt(2))))
            h = h @ parameters["weight::3.weight"].T + parameters["weight::3.bias"]
        p = np.exp(h - h.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    result, source_sets = {}, {}
    fit_sums, fit_squared_sums, fit_rows = np.zeros(1024), np.zeros(1024), 0
    for split in ("fit", "dev", "test"):
        published = np.load(root / f"run/evaluations/{split}-probabilities.npy", mmap_mode="r")
        y_parts, source_parts, families = [], [], []
        cursor, max_error, disagreement = 0, 0., 0
        role_counts = Counter()
        for entry in manifest["splits"][split]:
            folder = root / "features" / entry["path"]
            commit = read(folder / "complete.json")
            require(sha(folder / "complete.json") == entry["sha256"], "Subset commit changed")
            arrays = {}
            for key in ("x", "y", "structure", "source", "family", "weight", "groups"):
                path = folder / (key + ".npy")
                require(sha(path) == commit["files"][path.name], "Subset array changed")
                arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            require(np.char.startswith(arrays["structure"], "single:").all(), "Mixed single-message subset")
            for value in np.unique(arrays["groups"]):
                group = json.loads(str(value))
                require(group["profile"] == "single" and group["rounds"] == 0, "Mixed input groups")
            original = Path(commit["original_shard"])
            original_commit = read(original / "complete.json")
            require(sha(original / "complete.json") == commit["original_complete_sha256"], "Original commit changed")
            offsets = np.load(folder / "original_row_indices.npy")
            name = original.name
            origin_manifest = read(Path(protocol["source_feature_root"]) / "manifest.json")
            start = 0
            for source_entry in origin_manifest["splits"][split]:
                if source_entry["path"] == name:
                    break
                start += source_entry["rows"]
            local = offsets - start
            ox = np.load(original / "x.npy", mmap_mode="r")
            require(np.array_equal(ox[local], arrays["x"]), "Derived feature vectors changed from original")
            require(np.all(arrays["weight"] == 1), "Unexpected single-message family weights")
            y_parts.append(np.asarray(arrays["y"]))
            source_parts.append(np.asarray(arrays["source"]))
            families.extend(map(str, arrays["family"]))
            role_counts.update(map(int, arrays["y"]))
            for start in range(0, len(arrays["y"]), 8192):
                x = arrays["x"][start:start+8192]
                independent = forward(x)
                reference = published[cursor:cursor+len(x)]
                max_error = max(max_error, float(np.max(abs(independent-reference))))
                disagreement += int((independent.argmax(1) != reference.argmax(1)).sum())
                cursor += len(x)
                if split == "fit":
                    xd = x.astype(np.float64)
                    fit_sums += xd.sum(0)
                    fit_squared_sums += (xd * xd).sum(0)
                    fit_rows += len(x)
        require(cursor == len(published) == audit["splits"][split]["rows"], "Prediction count differs")
        require(len(set(families)) == len(families), "Unexpected repeated single-message family")
        require(max_error < 5e-4 and disagreement == 0, "Independent CPU export predictions disagree")
        labels, sources = np.concatenate(y_parts), np.concatenate(source_parts)
        metric = measures(labels, published)
        compare_metric(metric, read(root / f"run/evaluations/{split}.json")["overall"])
        source_sets[split] = set(map(str, sources))
        result[split] = {"rows": cursor, "source_clusters": len(source_sets[split]),
                         "maximum_numpy_gpu_probability_difference": max_error, "argmax_disagreements": disagreement,
                         "metric_recomputation": "exact confusion and per-role metric agreement"}
    require(not any(source_sets[a] & source_sets[b] for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test"))), "Cross-split sources")
    expected_mean = (fit_sums / fit_rows).astype(np.float32)
    expected_scale = np.sqrt(np.maximum(fit_squared_sums / fit_rows - (fit_sums / fit_rows)**2, 0)).astype(np.float32)
    expected_scale[expected_scale < 1e-6] = 1
    require(np.max(abs(expected_mean - mean)) < 1e-6 and np.max(abs(expected_scale-scale)) < 1e-6,
            "Export standardizer does not match single-message FIT")
    for candidate in selection["candidate_results"]:
        require(set(candidate["sampled_rows_by_profile"]) == {"single"}, "Mixed sampled training rows")
        require(candidate["sampled_profile_proportions"] == {"single": 1.}, "Mixed sampler fractions")
    with np.load(root / "agentdojo/first-position-probabilities.npz", allow_pickle=False) as z:
        y, p, task = z["labels"], z["probabilities"], z["task"]
        require(len(y) == 155103 and len(np.unique(task)) == 57, "Wrong AgentDojo denominators")
        agent_metrics = read(root / "agentdojo/metrics.json")
        compare_metric(measures(y, p), agent_metrics["overall"])
        compare_metric(measures(y, z["archived_baseline"]), agent_metrics["archived_mixed_probe_baseline"])
    result.update(status="passed", probe_sha256=freeze["probe_sha256"],
                  single_fit_standardizer_independently_recomputed=True, all_candidate_draws_single_only=True,
                  agentdojo_rows=155103, agentdojo_tasks=57,
                  method="Independent raw-array checks and explicit NumPy/SciPy exported-model forward; no training/evaluation helper imports")
    destination = root / "independent-validation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
