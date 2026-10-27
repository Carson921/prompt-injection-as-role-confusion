"""Independent, read-only CPU acceptance of the fixed-head warm-start result.

No producer prediction, metric, cache-loader or bootstrap function is imported.
Torch CPU exact GELU is the independent readout implementation; producer
probabilities are also used to check its published metrics without concealing
floating-point argmax ties.  The only output is a small JSON audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
HEADS = (1166, 1069, 2296, 1161, 2093, 2088, 2295, 2045,
         2179, 2288, 1274, 897, 1305, 2091, 2094, 1930)
PARENT_SHA = "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125"
BINDING = ("roles", "heads", "feature_dim", "feature_position", "model_binding")
LEARNING_RATES = {"warm-all": 1e-4, "warm-output": 3e-4, "scratch": 1e-3}
TAG_ROLE = {"system": 0, "developer": 1, "user": 2, "assistant_analysis": 3,
            "assistant_final": 4, "assistant_commentary": 4, "assistant_unspecified": 4, "tool": 5}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def safe(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact path")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact path escapes its root")
    return path


class Evidence:
    def __init__(self):
        self.files = {}
        self.ratio_max_error = 0.
        self.nll_max_error = 0.
        self.scalar_checks = 0

    def file(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f"Artifact SHA differs: {path.name}")
        previous = self.files.get(str(path))
        require(previous is None or previous["sha256"] == digest, "Artifact changed during independent validation")
        self.files[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
        return digest

    def number(self, actual, expected, name, *, tolerance=1e-12, nll=False):
        self.scalar_checks += 1
        if actual is None or expected is None:
            require(actual is expected, "N/A differs: " + name)
            return
        require(np.isfinite(actual) and np.isfinite(expected), "Nonfinite metric: " + name)
        error = abs(float(actual) - float(expected))
        if nll:
            self.nll_max_error = max(self.nll_max_error, error)
        else:
            self.ratio_max_error = max(self.ratio_max_error, error)
        require(error <= tolerance, f"Metric differs: {name}, absolute error={error:.12g}")

    def summary(self):
        return dict(verified_files=len(self.files), verified_bytes=sum(x["bytes"] for x in self.files.values()),
                    verified_inventory_sha256=object_sha(self.files), scalar_checks=self.scalar_checks,
                    maximum_ratio_absolute_error=self.ratio_max_error, maximum_nll_absolute_error=self.nll_max_error)


class ExactTorchProbe:
    """Raw FP16 -> F32 parent scaling -> linear/GELU(erf)/linear/softmax."""
    def __init__(self, path, *, evidence=None):
        import torch
        self.path = Path(path)
        if evidence is not None:
            evidence.file(path)
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"].item()))
            self.mean, self.scale = archive["mean"].copy(), archive["scale"].copy()
            self.weights = {key[8:]: archive[key].copy() for key in archive.files if key.startswith("weight::")}
        m = self.metadata
        require(m.get("format_version") == "synthetic-agent-six-role-probe-v1" and tuple(m["roles"]) == ROLES,
                "Not the approved six-role probe")
        require(tuple(m["heads"]) == HEADS and m["feature_dim"] == 1024 and m["feature_position"] == "pre-o_proj",
                "Probe feature specification differs")
        require(m["hidden"] == 128 and m["activation"] == "gelu_exact", "Probe architecture differs")
        require(self.mean.dtype == self.scale.dtype == np.float32 and self.mean.shape == self.scale.shape == (1024,),
                "Invalid standardizer array shape/dtype")
        require(np.isfinite(self.mean).all() and np.isfinite(self.scale).all() and (self.scale > 0).all(),
                "Invalid inherited standardizer")
        expected = {"0.weight": (128, 1024), "0.bias": (128,), "3.weight": (6, 128), "3.bias": (6,)}
        require(set(self.weights) == set(expected), "Unexpected probe parameter set")
        for key, shape in expected.items():
            require(self.weights[key].dtype == np.float32 and self.weights[key].shape == shape
                    and np.isfinite(self.weights[key]).all(), "Invalid parameter: " + key)
        self.torch = torch
        self.tensors = {key: torch.from_numpy(value) for key, value in self.weights.items()}
        self.mean_t, self.scale_t = torch.from_numpy(self.mean), torch.from_numpy(self.scale)

    def predict(self, features, batch_size=8192):
        torch = self.torch
        x = np.asarray(features)
        require(x.ndim == 2 and x.shape[1] == 1024 and np.isfinite(x).all(), "Invalid raw probe features")
        result = np.empty((len(x), 6), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                stop = min(start + batch_size, len(x))
                z = (torch.from_numpy(np.array(x[start:stop], dtype=np.float32, copy=True)) - self.mean_t) / self.scale_t
                z = torch.nn.functional.linear(z, self.tensors["0.weight"], self.tensors["0.bias"])
                z = torch.nn.functional.gelu(z, approximate="none")
                z = torch.nn.functional.linear(z, self.tensors["3.weight"], self.tensors["3.bias"])
                result[start:stop] = torch.softmax(z, dim=1).numpy()
        return result


def validate_probabilities(p, rows):
    require(p.shape == (rows, 6) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            "Invalid probability matrix")
    require(np.max(np.abs(p.sum(1) - 1), initial=0) <= 3e-5, "Probabilities do not sum to one")


def probability_comparison(actual, expected, tolerance):
    validate_probabilities(actual, len(expected))
    validate_probabilities(expected, len(actual))
    maximum = float(np.max(np.abs(actual - expected), initial=0))
    require(maximum <= tolerance, f"Independent Torch probability error={maximum:.12g} exceeds tolerance")
    mismatch = actual.argmax(1) != expected.argmax(1)
    sorted_p = np.sort(expected[mismatch], axis=1)
    margin = float(np.max(sorted_p[:, -1] - sorted_p[:, -2], initial=0))
    require(margin <= 2 * tolerance + 1e-7, "Argmax disagreement is not explained by bounded numerical differences")
    return dict(rows=len(actual), maximum_probability_absolute_error=maximum, absolute_tolerance=tolerance,
                exact_argmax_match=not bool(mismatch.any()), argmax_disagreements=int(mismatch.sum()),
                largest_producer_top_two_margin_at_disagreement=margin)


class VerifiedBank:
    """Independent array/hash loader, not the producer FeatureBank/WarmBank."""
    def __init__(self, root, split, parent, evidence, *, source_only=False):
        self.root, self.split = Path(root).resolve(), split
        m = self.manifest = read_json(self.root / "manifest.json")
        self.manifest_sha256 = evidence.file(self.root / "manifest.json")
        kind = m.get("format_version")
        require(kind in ("synthetic-role-warm-features-v1", "synthetic-agent-features-v1"), "Unknown feature format")
        for key in BINDING:
            require(m[key] == parent.metadata[key], "Cache binding differs: " + key)
        if "progress" in m:
            require(m["progress"].get(split, {}).get("complete") is True, "Incomplete feature split")
        self.new = kind == "synthetic-role-warm-features-v1"
        if self.new:
            signature = m["inputs"][split]["signature"]
            require(m["parent_probe_sha256"] == sha(parent.path), "Cache parent differs")
            for key in ("heads", "model_binding", "parent_probe_sha256"):
                require(signature[key] == m[key], "Extraction signature differs: " + key)
            require(signature["input_sha256"] == m["inputs"][split]["sha256"], "Input signature differs")
        else:
            for key, expected in (("model_binding", m["model_binding"]), ("heads", m["heads"]),
                                  ("capture", m["feature_position"]), ("config_sha256", m["config_hash"]),
                                  ("selection_sha256", m["selection_sha256"]),
                                  ("pilot_fit_sha256", m["pilot_fit_sha256"])):
                require(m["signature"].get(key) == expected, "Historical signature differs: " + key)
            signature = {**m["signature"], "split": split, "input_sha256": m["split_sources"][split]}
        entries = m["splits"][split]
        require(entries, "Empty cache split")
        self.shards, self.offsets = [], [0]
        columns = {key: [] for key in (["source"] if source_only else ["y", "source", "family", "structure", "weight", "groups"])}
        seen = set()
        for entry in entries:
            directory = safe(self.root, entry["path"])
            require(directory not in seen, "Repeated feature shard")
            seen.add(directory)
            complete = read_json(directory / "complete.json")
            evidence.file(directory / "complete.json", entry["sha256"])
            require(complete["signature"] == signature and complete["rows"] == entry["rows"], "Committed shard identity differs")
            if self.new:
                require(complete["split"] == split, "Committed shard split differs")
            for name in (["source"] if source_only else ["x", *columns]):
                path = safe(directory, name + ".npy")
                expected = complete["files"][name + ".npy"]
                if isinstance(expected, dict):
                    expected = expected["sha256"]
                evidence.file(path, expected)
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if name == "x":
                    require(value.dtype == np.float16 and value.shape == (entry["rows"], 1024), "Invalid FP16 feature array")
                    self.shards.append(value)
                else:
                    require(value.shape == (entry["rows"],), "Invalid cache metadata shape")
                    columns[name].append(np.asarray(value).copy())
            self.offsets.append(self.offsets[-1] + entry["rows"])
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        for key, values in columns.items():
            setattr(self, key, np.concatenate(values))
        require((self.source.astype(str) != "").all(), "Empty source cluster")
        if not source_only:
            require(self.y.dtype == np.int64 and ((self.y >= 0) & (self.y < 6)).all(), "Invalid role labels")
            require(np.isfinite(self.weight).all() and (self.weight > 0).all(), "Invalid family weights")
        progress = m.get("progress", {}).get(split, {})
        if self.new:
            require(progress.get("rows") in (None, len(self)), "Completed cache row count differs")
            identity = {**{key: m[key] for key in BINDING}, "parent_probe_sha256": m["parent_probe_sha256"],
                        "split": split, "shards": entries, "input": m["inputs"][split]}
        else:
            identity = dict(format_version=m["format_version"], roles=m["roles"], feature_dim=m["feature_dim"],
                feature_position=m["feature_position"], signature=m["signature"], data_hash=m["data_hash"], split=split,
                input_sha256=m["split_sources"][split], shards=entries,
                complete=progress.get("complete"), records=progress.get("records"), rows=progress.get("rows"))
        self.split_identity_sha256 = object_sha(identity)

    def __len__(self):
        return int(self.offsets[-1])

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        require(indices.ndim == 1 and ((indices >= 0) & (indices < len(self))).all(), "Feature index out of bounds")
        result = np.empty((len(indices), 1024), dtype=np.float16)
        shard_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard in np.unique(shard_ids):
            selected = np.flatnonzero(shard_ids == shard)
            result[selected] = self.shards[shard][indices[selected] - self.offsets[shard]]
        require(np.isfinite(result).all(), "Nonfinite body feature")
        return result

    def predict(self, probe):
        out = np.empty((len(self), 6), dtype=np.float32)
        for start in range(0, len(self), 8192):
            stop = min(start + 8192, len(self))
            out[start:stop] = probe.predict(self.read(np.arange(start, stop)))
        return out


def metrics_from_matrix(matrix, nll=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    correct = np.diag(matrix)
    p = np.divide(correct, predicted, out=np.zeros(6, float), where=predicted > 0)
    r = np.divide(correct, support, out=np.zeros(6, float), where=support > 0)
    f = np.divide(2 * correct, support + predicted, out=np.zeros(6, float), where=(support + predicted) > 0)
    result = dict(tokens=int(matrix.sum()), accuracy=float(correct.sum() / max(matrix.sum(), 1)),
                  macro_f1=float(f[support > 0].mean()) if (support > 0).any() else None,
                  confusion=matrix.tolist(), roles={})
    if nll is not None:
        result["nll"] = float(nll)
    for i, role in enumerate(ROLES):
        result["roles"][role] = dict(support=int(support[i]), predicted=int(predicted[i]),
            precision=float(p[i]) if predicted[i] else None, recall=float(r[i]) if support[i] else None,
            f1=float(f[i]) if support[i] else None, false_positives=int(predicted[i] - correct[i]),
            false_positive_rate=float((predicted[i] - correct[i]) / (matrix.sum() - support[i]))
                if matrix.sum() > support[i] else None)
    return result


def metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    validate_probabilities(probabilities, len(y))
    predicted = probabilities.argmax(1)
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    selected = probabilities[np.arange(len(y)), y]
    # The published score uses FP32 log and FP32 reduction.  Reimplement that
    # numerical contract independently, and retain the F64 value as a diagnostic.
    losses32 = -np.log(np.clip(selected.astype(np.float32), np.float32(1e-30), np.float32(1)))
    losses64 = -np.log(np.clip(selected.astype(np.float64), 1e-30, 1))
    result = metrics_from_matrix(matrix, float(np.mean(losses32, dtype=np.float32)))
    result["nll_float64_diagnostic"] = float(losses64.mean())
    return result


def compare_metrics(actual, expected, evidence, name):
    require(actual["tokens"] == expected["tokens"] and actual["confusion"] == expected["confusion"],
            "Hard confusion/count differs: " + name)
    for key in ("accuracy", "macro_f1"):
        evidence.number(actual[key], expected[key], name + "/" + key)
    if "nll" in actual and "nll" in expected:
        evidence.number(actual["nll"], expected["nll"], name + "/nll", tolerance=1e-6, nll=True)
    for role in ROLES:
        for key, value in actual["roles"][role].items():
            evidence.number(value, expected["roles"][role][key], name + "/" + role + "/" + key)


def bootstrap(labels, probabilities, sources, replicates, seed):
    """Independent cluster sufficient statistics and 2000 whole-source resamples."""
    names = sorted(set(map(str, sources)))
    index = {name: i for i, name in enumerate(names)}
    rows = np.asarray([index[str(source)] for source in sources], dtype=np.int64)
    predicted = probabilities.argmax(1)
    matrices = np.zeros((len(names), 6, 6), dtype=np.int64)
    np.add.at(matrices, (rows, labels, predicted), 1)
    # Match the stored FP32-loss contract; bincount then accumulates those
    # already-rounded losses in F64, independently of the producer bootstrap.
    selected = probabilities[np.arange(len(labels)), labels].astype(np.float32)
    loss = -np.log(np.clip(selected, np.float32(1e-30), np.float32(1)))
    loss_sum = np.bincount(rows, weights=loss, minlength=len(names))
    totals = matrices.sum((1, 2))
    samples = {key: [] for key in ("accuracy", "macro_f1", "nll")}
    for role in ROLES:
        for key in ("precision", "recall", "f1"):
            samples[role + "." + key] = []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        draw = rng.integers(0, len(names), len(names))
        value = metrics_from_matrix(matrices[draw].sum(0))
        samples["accuracy"].append(value["accuracy"])
        samples["macro_f1"].append(value["macro_f1"])
        samples["nll"].append(float(loss_sum[draw].sum() / totals[draw].sum()))
        for role in ROLES:
            for key in ("precision", "recall", "f1"):
                if value["roles"][role][key] is not None:
                    samples[role + "." + key].append(value["roles"][role][key])
    intervals = {key: dict(lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)),
                          valid_replicates=len(values)) if values else None for key, values in samples.items()}
    return dict(unit="source_cluster", clusters=len(names), replicates=replicates, seed=seed,
                method="percentile", confidence=.95, intervals=intervals)


def compare_bootstrap(actual, expected, evidence):
    for key in ("unit", "clusters", "replicates", "seed", "method", "confidence"):
        require(actual[key] == expected[key], "Bootstrap protocol differs: " + key)
    require(set(actual["intervals"]) == set(expected["intervals"]), "Bootstrap interval inventory differs")
    for key, value in actual["intervals"].items():
        target = expected["intervals"][key]
        if value is None or target is None:
            require(value is target, "Bootstrap absent-class convention differs")
            continue
        require(value["valid_replicates"] == target["valid_replicates"], "Bootstrap denominator differs")
        for bound in ("lower", "upper"):
            evidence.number(value[bound], target[bound], "bootstrap/" + key + "/" + bound,
                            tolerance=1e-6 if key == "nll" else 1e-12, nll=key == "nll")


def check_low_permission(labels, p, expected, evidence, name):
    mask = np.isin(labels, [2, 5])
    errors = int(np.sum(mask & np.isin(p.argmax(1), [0, 1])))
    evidence.number(errors, expected["errors"], name + "/errors")
    evidence.number(int(mask.sum()), expected["denominator"], name + "/denominator")
    evidence.number(float(errors / mask.sum()) if mask.any() else None, expected["rate"], name + "/rate")


def compare_test_groups(bank, probabilities, producer, evidence):
    values = []
    require(set(producer["by_structure"]) == set(map(str, bank.structure)), "Structure inventory differs")
    for structure in sorted(set(map(str, bank.structure))):
        mask = bank.structure == structure
        value = metrics(bank.y[mask], probabilities[mask])
        compare_metrics(value, producer["by_structure"][structure], evidence, "test/structure/" + structure)
        values.append(value["macro_f1"])
    evidence.number(float(np.mean(values)), producer["structure_equal_macro_f1"], "test/structure_equal_macro_f1")
    all_roles = producer["overall"]["roles"]
    minimum = min(value or 0. for role in ROLES for value in (all_roles[role]["precision"], all_roles[role]["recall"]))
    evidence.number(minimum, producer["minimum_precision_recall"], "test/minimum_precision_recall")
    check_low_permission(bank.y, probabilities, producer["low_permission_to_sd"], evidence, "test/low_permission")
    objects = [json.loads(str(x)) for x in bank.groups]
    groups = {}
    recalls = []
    for key in ("body_length_bin", "body_quartile", "profile", "wrapper_type"):
        column = np.asarray([str(value.get(key, "N/A")) for value in objects])
        for value in sorted(set(column)):
            for role in (0, 1):
                mask = (column == value) & (bank.y == role)
                positives, sources = int(mask.sum()), len(set(map(str, bank.source[mask])))
                eligible = value != "N/A" and sources >= 16 and positives >= 128
                recall = float(np.mean(probabilities[mask].argmax(1) == role)) if positives else None
                groups[key, value, ROLES[role]] = dict(positives=positives, sources=sources, eligible=eligible, recall=recall)
                if eligible:
                    recalls.append(recall)
    require(len(groups) == len(producer["weak_groups"]), "Weak-group inventory differs")
    for group in producer["weak_groups"]:
        value = groups[group["attribute"], group["value"], group["role"]]
        for key in ("positives", "sources", "eligible"):
            require(value[key] == group[key], "Weak-group denominator differs")
        evidence.number(value["recall"], group["recall"], "test/weak_group")
    evidence.number(min(recalls) if recalls else None, producer["weakest_sd_recall"], "test/weakest_sd_recall")


def check_input_metadata(bank, inputs_path, evidence):
    evidence.file(inputs_path, bank.manifest["inputs"][bank.split]["sha256"])
    position = 0
    records, source_set = 0, set()
    with Path(inputs_path).open() as stream:
        for line in stream:
            record = json.loads(line)
            records += 1
            require(0 < len(record["input_ids"]) <= 4096, "Oversize/empty extraction input")
            source_set.add(record["cluster_id"])
            for target in record["targets"]:
                require(position < len(bank), "Input contains more targets than cache")
                require(int(bank.y[position]) == ROLES.index(target["role"]) == target["label"], "Cached role label shifted")
                require(str(bank.source[position]) == record["cluster_id"], "Cached source is not original cluster ID")
                require(str(bank.family[position]) == target["family_id"], "Cached family differs from actual input")
                expected_structure = target.get("structure", record.get("structure", record["profile"] + ":" + record["template_id"]))
                require(str(bank.structure[position]) == expected_structure, "Cached structure differs from actual input")
                require(float(bank.weight[position]) == float(target.get("weight", 1)), "Cached family weight differs")
                groups = json.loads(str(bank.groups[position]))
                for key in ("profile", "body_length_bin", "body_quartile", "wrapper_type"):
                    expected = target.get(key, record.get(key))
                    require(groups[key] == expected, "Cached diagnostic group differs: " + key)
                position += 1
    require(position == len(bank), "Cache contains targets missing from actual input")
    return dict(records=records, targets=position, source_clusters=len(source_set), input_sha256=sha(inputs_path))


def candidate_decision(dev, candidate):
    new, old = dev["new_dev"], dev["old_dev"]
    eligible = (new["minimum_precision_recall"] >= .9 and old["minimum_precision_recall"] >= .9
                and old["overall"]["macro_f1"] >= dev["parent_old_dev_macro_f1"] - .005
                and new["low_permission_to_sd"]["rate"] is not None and old["low_permission_to_sd"]["rate"] is not None
                and new["low_permission_to_sd"]["rate"] <= .01 and old["low_permission_to_sd"]["rate"] <= .01
                and new["weakest_sd_recall"] is not None)
    key = [not eligible, -(new["weakest_sd_recall"] or 0.), -new["structure_equal_macro_f1"],
           new["overall"]["nll"], candidate]
    return bool(eligible), key


def initialization_weights(candidate, parent):
    if candidate != "scratch":
        return parent.weights
    import torch
    # Producer creates these two CPU Linear layers before moving the net to CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        first, last = torch.nn.Linear(1024, 128), torch.nn.Linear(128, 6)
    return {"0.weight": first.weight.detach().numpy(), "0.bias": first.bias.detach().numpy(),
            "3.weight": last.weight.detach().numpy(), "3.bias": last.bias.detach().numpy()}


def check_candidates(root, parent, training, budget_identity, evidence):
    import torch
    records, results = [], {}
    options = training["options"]
    approved = dict(seed=17, batch_size=4096, epoch_rows=500000, min_epochs=10, max_epochs=20,
                    patience=4, weight_decay=.001, dropout=.1, gradient_clip=5., replay_rows=250000)
    for key, value in approved.items():
        require(options[key] == value, "Training option changed: " + key)
    batches = math.ceil(options["epoch_rows"] / options["batch_size"])
    for candidate in LEARNING_RATES:
        directory = root / "run/candidates" / candidate
        complete_path, history_path = directory / "complete.json", directory / "history.json"
        complete, history = read_json(complete_path), read_json(history_path)
        evidence.file(complete_path)
        evidence.file(history_path, complete["history_sha256"])
        evidence.file(directory / "best.npz", complete["probe_sha256"])
        require(complete["candidate_id"] == candidate and complete["training_hash"] == training["training_hash"], "Candidate identity differs")
        require(complete["parent_probe_sha256"] == sha(parent.path) and not complete["engineering_smoke_only"], "Candidate lineage differs")
        require(complete["gpu_budget"] == budget_identity, "Candidate budget differs")
        probe = ExactTorchProbe(directory / "best.npz", evidence=evidence)
        for key in BINDING:
            require(probe.metadata[key] == parent.metadata[key], "Candidate feature identity differs")
        require(array_sha(probe.mean) == array_sha(parent.mean) and array_sha(probe.scale) == array_sha(parent.scale),
                "Candidate standardizer bytes changed")
        lineage = probe.metadata["warm_start"]
        require(lineage["parent_probe_sha256"] == sha(parent.path) and lineage["parent_best_epoch"] == 30
                and lineage["optimizer_reset"] is True and lineage["epoch_reset"] is True
                and lineage["learning_rate"] == LEARNING_RATES[candidate]
                and lineage["gpu_budget"] == budget_identity
                and lineage["quality_exclusions"] == training["quality_exclusions"], "Warm-start settings differ")
        initial = initialization_weights(candidate, parent)
        proof_path = directory / "initialization-proof.json"
        proof = read_json(proof_path)
        evidence.file(proof_path)
        require(proof["candidate_id"] == candidate and proof["parent_probe_sha256"] == sha(parent.path)
                and proof["gpu_budget"] == budget_identity and proof["optimizer_initialized_empty"] is True
                and proof["epoch_reset_to"] == 0 and proof["standardizer_inherited"] is True
                and proof["inherited_weights"] == (candidate != "scratch"), "Initialization proof differs")
        require(proof["initialized_weights_sha256"] == {key: array_sha(value) for key, value in initial.items()},
                "Initialization parameter SHA differs")
        for key, array in (("mean", parent.mean), ("scale", parent.scale)):
            require(proof["standardizer_" + key + "_sha256"] == array_sha(array)
                    == complete["standardizer_" + key + "_sha256"], "Inherited standardizer bytes differ")
        expected_trainable = ["3.weight", "3.bias"] if candidate == "warm-output" else ["0.weight", "0.bias", "3.weight", "3.bias"]
        require(proof["trainable_parameters"] == expected_trainable, "Trainable modules differ")
        if candidate != "scratch":
            require(0 <= proof["initial_parent_probability_max_error"] <= 3e-5
                    and 0 <= complete["init_parent_probability_max_error"] <= 3e-5, "Warm prediction initialization failed")
        require(options["min_epochs"] <= len(history) == complete["epochs"] <= options["max_epochs"], "Epoch count differs")
        counts = dict(new_paired=0, new_context=0, old_replay=0)
        best_key, best_epoch = None, None
        for index, epoch in enumerate(history, 1):
            require(epoch["epoch"] == index and epoch["optimizer_step"] == index * batches, "Training update history shifted")
            require(set(epoch["sampled_rows"]) == set(counts) and sum(epoch["sampled_rows"].values()) == options["epoch_rows"],
                    "Actual sampled rows differ from protocol")
            for key in counts:
                require(isinstance(epoch["sampled_rows"][key], int) and epoch["sampled_rows"][key] >= 0, "Invalid sampler count")
                counts[key] += epoch["sampled_rows"][key]
            require(counts == epoch["cumulative_sampled_rows"], "Sampler cumulative totals differ")
            eligible, key = candidate_decision(epoch["dev"], candidate)
            require(eligible == epoch["dev"]["eligible"] and key == epoch["dev"]["selection_key"], "DEV decision formula differs")
            require(epoch["dev"].get("old_dev_full_used_for_selection") is False, "Full old DEV participated in selection")
            improved = best_key is None or tuple(key) < tuple(best_key)
            require(improved == epoch["improved"], "Best-epoch marker differs")
            if improved:
                best_key, best_epoch = key, index
        require(best_epoch == complete["best_epoch"] == probe.metadata["best_epoch"], "Best epoch differs")
        require(complete["dev"] == history[best_epoch - 1]["dev"], "Best DEV result differs")
        require(counts == complete["sampled_rows"] and complete["best_sampled_rows"] == history[best_epoch - 1]["cumulative_sampled_rows"],
                "Exported training-path sampler counts differ")
        checkpoint_path = directory / "last.pt"
        evidence.file(checkpoint_path)
        # These are this experiment's own checkpoint files, not third-party pickles.
        last = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = last["state"]
        require(last["candidate_id"] == candidate and last["training_hash"] == training["training_hash"]
                and last["parent_probe_sha256"] == sha(parent.path), "Last checkpoint lineage differs")
        require(state["epoch"] == complete["epochs"] and state["batch_cursor"] == 0
                and state["best_epoch"] == best_epoch and state["optimizer_step"] == complete["epochs"] * batches
                and state["cumulative_counts"] == counts and state["history"] == history, "Last checkpoint progress differs")
        for name in initial:
            require(array_sha(state["best_state"][name].numpy()) == array_sha(probe.weights[name]), "Export is not recorded best checkpoint")
        hidden_same = all(array_sha(probe.weights[key]) == array_sha(initial[key]) for key in ("0.weight", "0.bias"))
        require(hidden_same == complete["hidden_layer_unchanged"], "Hidden-freeze proof contradicts actual weights")
        if candidate == "warm-output":
            require(hidden_same and all(array_sha(last["model"][key].numpy()) == array_sha(parent.weights[key])
                                       for key in ("0.weight", "0.bias")), "Frozen hidden weights changed")
        optimizer = last["optimizer"]
        require(len(optimizer["param_groups"]) == 1 and optimizer["param_groups"][0]["lr"] == LEARNING_RATES[candidate], "Optimizer settings differ")
        require(len(optimizer["state"]) == len(expected_trainable), "Optimizer contains unexpected/inherited modules")
        require(all(int(value["step"].item()) == state["optimizer_step"] for value in optimizer["state"].values()), "Optimizer step did not restart at zero")
        results[candidate] = dict(epochs=complete["epochs"], best_epoch=best_epoch, optimizer_steps=state["optimizer_step"],
            initialized_parameter_hashes_matched=True, inherited_standardizer_bytes_matched=True,
            hidden_layer_unchanged=hidden_same, full_sampled_rows=counts,
            best_sampled_rows=complete["best_sampled_rows"], eligible=complete["dev"]["eligible"],
            old_dev_clean_tokens=complete["dev"]["old_dev"]["overall"]["tokens"],
            old_dev_full_diagnostic_tokens=complete["dev"]["old_dev_full_diagnostic"]["overall"]["tokens"])
        records.append(complete)
        del last
    return records, results


def validate_root(root, old_features_dir, output_path, *, expected_parent_sha=PARENT_SHA,
                  calls_root=None, analysis_root=None, cached_evaluation_dir=None, probability_tolerance=5e-4,
                  old_sources_path=None, threads=2):
    import torch
    torch.set_num_threads(threads)
    root, old_features_dir = Path(root).resolve(), Path(old_features_dir).resolve()
    evidence = Evidence()
    started = time.time()
    parent = ExactTorchProbe(root / "parent/best.npz", evidence=evidence)
    require(sha(parent.path) == expected_parent_sha and parent.metadata["best_epoch"] == 30, "Approved parent checkpoint differs")
    frozen_path = root / "run/freeze.json"
    frozen, selection, training = (read_json(root / "run" / name) for name in ("freeze.json", "selection.json", "training.json"))
    evidence.file(frozen_path)
    evidence.file(root / "run/selection.json", frozen["selection_sha256"])
    evidence.file(root / "run/training.json", frozen["training_sha256"])
    evidence.file(root / "run/best.npz", frozen["probe_sha256"])
    require(frozen["format_version"] == "synthetic-role-warm-freeze-v1" and frozen["selection_split"] == "dev"
            and frozen["parent_probe_sha256"] == expected_parent_sha, "Freeze protocol differs")
    require(training["training_hash"] == object_sha(training["identity"]) and not training["engineering_smoke_only"],
            "Training identity differs or is engineering-only")
    evidence.file(root / "code/synthetic_role_warmstart_train.py", training["identity"]["implementation_sha256"])
    budget = read_json(root / "budget.json")
    budget_sha = evidence.file(root / "budget.json")
    require(0 < budget["deadline_unix"] - budget["started_unix"] <= 21600.01, "GPU budget exceeds six hours")
    budget_identity = read_json(root / "run/training-budget.json")
    evidence.file(root / "run/training-budget.json", frozen["training_budget_sha256"])
    require(budget_identity == frozen["gpu_budget"] and budget_identity["budget_sha256"] == budget_sha
            and budget_identity["started_unix"] == budget["started_unix"]
            and budget_identity["deadline_unix"] == budget["deadline_unix"], "Budget binding differs")
    require(budget["started_unix"] <= frozen["frozen_at_unix"] <= budget["deadline_unix"], "Freeze is outside authorized GPU window")
    complete_gpu = read_json(root / "gpu-complete.json")
    evidence.file(root / "gpu-complete.json")
    require(complete_gpu["freeze_sha256"] == sha(frozen_path)
            and frozen["frozen_at_unix"] <= complete_gpu["completed_unix"] <= budget["deadline_unix"], "GPU completion proof differs")
    binding = read_json(root / "campaign-binding.json")
    evidence.file(root / "campaign-binding.json")
    for name, expected in binding["code"].items():
        evidence.file(safe(root / "code", name), expected)
    preflight = read_json(root / "run/preflight.json")
    evidence.file(root / "run/preflight.json")
    require(preflight["status"] == "passed" and preflight["devices"] == [0, 1, 2]
            and preflight["maximum_input_tokens"] == 4096 and preflight["forward_batch_size"] == 1
            and preflight["heads"] == list(HEADS) and preflight["parent_probe_sha256"] == expected_parent_sha
            and preflight["model_binding"] == parent.metadata["model_binding"], "Three-A40 preflight identity differs")
    require(preflight["runtime_sha256"] == sha(root / "code/synthetic_agent_gpu.py")
            and preflight["extractor_sha256"] == sha(root / "code/synthetic_role_warmstart_gpu.py"),
            "Preflight code identity differs")
    require(len(preflight["device_wide_peak_mib"]) == 3 and max(preflight["device_wide_peak_mib"]) <= 40 * 1024
            and not preflight["memory_monitor_errors"], "Preflight memory gate differs")
    require({entry["input_tokens"] for entry in preflight["checks"]} == {256, 2048, 4096}
            and all(entry["repetitions"] == 3 and entry["features_bitexact"] and entry["hidden_states_bitexact"]
                    for entry in preflight["checks"]), "Fixed-shape preflight coverage differs")
    backend = read_json(root / "run/model_backend.json")
    evidence.file(root / "run/model_backend.json")
    require(backend["model_binding"] == parent.metadata["model_binding"]
            and len(backend["gpus"]) == 3 and all("A40" in value for value in backend["gpus"])
            and backend["expert_backends"] == ["Mxfp4GptOssExperts"], "Model backend identity differs")
    device_index = lambda value: int(str(value).removeprefix("cuda:"))
    require(all(device_index(backend["device_map"][f"model.layers.{i}"]) == i // 12 for i in range(36))
            and device_index(backend["device_map"]["model.embed_tokens"]) == 0
            and device_index(backend["device_map"]["model.norm"]) == 2
            and device_index(backend["device_map"]["lm_head"]) == 2, "Three-card model partition differs")
    evidence.file(root / "data/data-manifest.json", binding["data_manifest_sha256"])
    evidence.file(root / "data/article-audit.json", binding["article_audit_sha256"])
    article_gate = read_json(root / "data/article-audit.json")
    require(article_gate["status"] == "passed" and article_gate["reviewed_articles"] == 50, "Article review gate differs")
    quality = read_json(root / "data/quality-excluded-clusters.json")
    quality_sha = evidence.file(root / "data/quality-excluded-clusters.json", training["quality_exclusions"]["ledger_sha256"])
    rejected = set(quality["excluded_clusters"])
    require(len(rejected) == len(quality["excluded_clusters"]) == 14, "Final source rejection ledger differs")
    require(quality["old_sources_sha256"] == training["quality_exclusions"]["old_sources_sha256"], "Original source ledger hash differs")
    if old_sources_path is not None:
        evidence.file(old_sources_path, quality["old_sources_sha256"])
    parent_meta = parent.metadata
    new_banks = {split: VerifiedBank(root / "features", split, parent, evidence) for split in ("fit", "dev", "test")}
    for bank in new_banks.values():
        signature = bank.manifest["inputs"][bank.split]["signature"]
        require(signature["runtime_sha256"] == preflight["runtime_sha256"]
                and signature["extractor_sha256"] == preflight["extractor_sha256"]
                and signature["layout"] == "3xa40-12layers-b1-no-cache", "Extraction implementation/layout differs from preflight")
    old_fit = VerifiedBank(old_features_dir, "fit", parent, evidence, source_only=True)
    old_dev = VerifiedBank(old_features_dir, "dev", parent, evidence, source_only=True)
    old_test = VerifiedBank(old_features_dir, "test", parent, evidence, source_only=True)
    require(training["identity"]["old_fit"] == old_fit.split_identity_sha256
            and training["identity"]["old_dev"] == old_dev.split_identity_sha256, "Old cache source identity differs")
    for split in ("fit", "dev"):
        require(training["identity"]["new_" + split] == new_banks[split].split_identity_sha256, "New training cache identity differs")
    source_sets = {split: set(map(str, bank.source)) for split, bank in new_banks.items()}
    require(all(not source_sets[a].intersection(source_sets[b]) for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test"))),
            "New source partitions overlap")
    require(not source_sets["fit"].intersection(map(str, old_dev.source))
            and not source_sets["dev"].intersection(map(str, old_fit.source)), "Old/new development source leakage")
    all_old = set(map(str, old_fit.source)) | set(map(str, old_dev.source)) | set(map(str, old_test.source))
    require(not source_sets["test"].intersection(all_old), "Fresh TEST overlaps historical sources")
    require(all(not values.intersection(rejected) for values in source_sets.values()), "New inputs contain a quality-rejected source")
    replay_path = root / "run/replay-indices.npy"
    evidence.file(replay_path, training["replay_sha256"])
    replay = np.load(replay_path, allow_pickle=False)
    require(replay.dtype == np.int64 and len(replay) == 250000 and len(np.unique(replay)) == len(replay)
            and ((replay >= 0) & (replay < len(old_fit))).all(), "Replay index protocol differs")
    require(not set(map(str, old_fit.source[replay])).intersection(rejected), "Rejected sources entered old FIT replay")
    clean_old_dev = np.flatnonzero(~np.isin(old_dev.source, list(rejected)))
    require(array_sha(clean_old_dev) == training["identity"]["old_dev_clean_indices_sha256"], "Clean old DEV view differs")
    evidence.file(root / "run/cache-import-proof.json")
    imported = read_json(root / "run/cache-import-proof.json")
    require(imported["replay_indices_sha256"] == training["replay_sha256"] and imported["rejected_sources_in_replay"] == 0
            and imported["old_dev_full_rows"] == len(old_dev) and imported["old_dev_clean_rows"] == len(clean_old_dev)
            and imported["quality_exclusions"] == training["quality_exclusions"], "Cache import proof differs")
    require(imported["old_fit_excluded_rows"] == int(np.isin(old_fit.source, list(rejected)).sum())
            and imported["old_dev_excluded_rows"] == len(old_dev) - len(clean_old_dev), "Quality exclusion row totals differ")
    denominators = training["old_dev_denominators"]
    require(denominators == dict(full_rows=len(old_dev), clean_rows=len(clean_old_dev),
        excluded_rows=len(old_dev) - len(clean_old_dev), full_sources=len(set(old_dev.source)),
        clean_sources=len(set(old_dev.source[clean_old_dev]))), "Old DEV clean/full denominator declaration differs")
    input_metadata = {}
    for split, bank in new_banks.items():
        possibilities = [root / "data" / f"inputs-{split}.jsonl"]
        if split == "fit":
            possibilities.append(root / "data/inputs-fit-conservative.jsonl")
        matches = [p for p in possibilities if p.is_file() and sha(p) == bank.manifest["inputs"][split]["sha256"]]
        require(len(matches) == 1, "Actual extraction input cannot be identified")
        input_metadata[split] = check_input_metadata(bank, matches[0], evidence)
    records, candidates = check_candidates(root, parent, training, budget_identity, evidence)
    for record in records:
        require(record["dev"]["old_dev"]["overall"]["tokens"] == len(clean_old_dev)
                and record["dev"]["old_dev_full_diagnostic"]["overall"]["tokens"] == len(old_dev), "Old DEV metric denominator differs")
    winner = min(records, key=lambda r: tuple(r["dev"]["selection_key"]))
    require(selection["candidate_id"] == frozen["candidate_id"] == winner["candidate_id"]
            and selection["candidate_results"] == records and selection["selection_split"] == "dev", "DEV-only model selection differs")
    require(sha(root / "run/best.npz") == winner["probe_sha256"], "Frozen model is not selected candidate")
    final = ExactTorchProbe(root / "run/best.npz", evidence=evidence)
    require(array_sha(final.mean) == array_sha(parent.mean) and array_sha(final.scale) == array_sha(parent.scale),
            "Final standardizer bytes changed")
    test_bank = new_banks["test"]
    manual = test_bank.predict(final)
    probability_path = root / "run/evaluations/test-probabilities.npy"
    evidence.file(probability_path)
    probabilities = np.load(probability_path, allow_pickle=False)
    require(probabilities.dtype == np.float32, "Producer TEST probabilities are not FP32")
    comparison = probability_comparison(manual, probabilities, probability_tolerance)
    producer_path = root / "run/evaluations/test.json"
    producer = read_json(producer_path)
    evidence.file(producer_path)
    require(producer["probe_sha256"] == frozen["probe_sha256"]
            and producer["feature_split_identity_sha256"] == test_bank.split_identity_sha256, "TEST result identity differs")
    independent = metrics(test_bank.y, probabilities)
    compare_metrics(independent, producer["overall"], evidence, "test/overall")
    compare_test_groups(test_bank, probabilities, producer, evidence)
    protocol = producer["confidence_intervals"]
    require(protocol["replicates"] == 2000 and protocol["confidence"] == .95 and protocol["unit"] == "source_cluster", "Bootstrap protocol differs")
    intervals = bootstrap(test_bank.y, probabilities, test_bank.source, protocol["replicates"], protocol["seed"])
    compare_bootstrap(intervals, protocol, evidence)
    report = dict(status="passed", format_version="independent-role-warmstart-result-validation-v1",
        checked_at_unix=time.time(), validator_sha256=sha(__file__), torch_cpu_version=torch.__version__,
        parent_probe_sha256=sha(parent.path), final_probe_sha256=sha(final.path), freeze_sha256=sha(frozen_path),
        budget=dict(started_unix=budget["started_unix"], deadline_unix=budget["deadline_unix"],
                    gpu_completed_unix=complete_gpu["completed_unix"], budget_sha256=budget_sha),
        preflight=dict(model_binding_matched_parent=True, three_a40_12_layers_each=True,
            longest_input_tokens=4096, device_wide_sampled_peak_mib=preflight["device_wide_peak_mib"],
            fixed_shape_repetitions=3, loaded_expert_backend="Mxfp4GptOssExperts"),
        source_isolation=dict(new_sources={s: len(v) for s, v in source_sets.items()},
            historical_sources=len(all_old), rejected_clusters=len(rejected), quality_ledger_sha256=quality_sha,
            old_dev_full_rows=len(old_dev), old_dev_clean_rows=len(clean_old_dev), replay_rows=len(replay)),
        input_metadata=input_metadata, candidates=candidates, selection_candidate=winner["candidate_id"],
        test=dict(probability_comparison=comparison, independently_recomputed_producer_metrics=independent,
            manual_torch_metrics=metrics(test_bank.y, manual), confidence_intervals=intervals),
        numeric_policy="Published NLL uses independent FP32 clip/log/mean; source bootstrap accumulates FP32 losses in FP64. FP64 point NLL is diagnostic only.",
        limitations=["No 120B forward or gradient update is replayed.",
            "Published metrics/CI are recomputed independently from stored producer probabilities after a full independent Torch readout comparison.",
            "Any bounded floating-point argmax disagreements are counted explicitly; they are not silently repaired.",
            "Per-epoch sampler counters are cross-checked with histories and terminal optimizer checkpoints, not a replay of all gradient draws."])
    if cached_evaluation_dir is not None:
        require(calls_root is not None and analysis_root is not None, "Cached evaluation needs both original source roots")
        report["cached_agentdojo"] = validate_cached(calls_root, analysis_root, cached_evaluation_dir,
                                                      parent, final, frozen_path, evidence, probability_tolerance)
    else:
        report["cached_agentdojo"] = {"status": "not_requested"}
    report["evidence"] = evidence.summary()
    report["elapsed_seconds"] = time.time() - started
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".tmp")
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pending.replace(path)
    return report


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def retained_masks(path):
    """Preserve empty N/A frames; only nonempty ordinary bodies are scored."""
    masks, positions = defaultdict(list), set()
    rows, empty_rows, structural_empty_rows = 0, 0, 0
    for row in _csv(path):
        if row["content_filter"] != "normal_content":
            continue
        require(row["scope"] in ("actual_input", "actual_generated") and row["tag"] in TAG_ROLE,
                "Unknown first-position scope or tag")
        indices = np.asarray(json.loads(row["token_indices"]), dtype=np.int64)
        require(indices.ndim == 1 and len(indices) == int(row["tokens"]) and (indices >= 0).all()
                and (len(indices) < 2 or (np.diff(indices) > 0).all()), "First-position mask differs")
        require(not len(indices) or row["structural_tool_call"] == "False", "Tool-call JSON entered normal body")
        for index in indices:
            key = row["call_id"], row["scope"], int(index)
            require(key not in positions, "Duplicate first-position token")
            positions.add(key)
        masks[row["call_id"], row["scope"]].append((row, indices))
        rows += 1
        empty_rows += int(not len(indices))
        structural_empty_rows += int(not len(indices) and row["structural_tool_call"] == "True")
    return masks, positions, dict(rows=rows, empty_rows=empty_rows, structural_empty_rows=structural_empty_rows)


def _cached_segments(tag, prompt, start, end, offsets):
    if tag not in ("system", "developer"):
        return np.full(len(offsets), "", dtype="U40")
    body = prompt[start:end]
    cuts = [(0, "developer/instructions" if tag == "developer" else "system/identity_and_date")]
    anchors = [("# Tools", "developer/tool_schema")] if tag == "developer" else [
        ("Reasoning:", "system/reasoning_setting"), ("# Valid channels:", "system/channel_rules"),
        ("Calls to these tools", "system/tool_routing")]
    for anchor, label in anchors:
        index = body.find(anchor)
        if index >= 0:
            cuts.append((index, label))
    cuts.sort()
    result = []
    for left, right in offsets:
        midpoint = max(0, min((float(left) + float(right)) / 2 - start, max(0, len(body) - .5)))
        result.append(next(label for position, label in reversed(cuts) if position <= midpoint))
    return np.asarray(result, dtype="U40")


def _cached_metric_compare(labels, p, expected, evidence, name):
    value = metrics(labels, p)
    compare_metrics(value, expected, evidence, name)
    low = np.isin(labels, [2, 5])
    wrong = int(np.sum(low & np.isin(p.argmax(1), [0, 1])))
    target = expected["user_tool_to_system_developer"]
    evidence.number(int(low.sum()), target["denominator"], name + "/low_denominator")
    evidence.number(wrong, target["false_positives"], name + "/low_false_positives")
    evidence.number(float(wrong / low.sum()) if low.any() else None, target["rate"], name + "/low_rate")
    return value


def _mean_tables(p, metadata, directory, model, evidence):
    scope, tag, task = (metadata[key] for key in ("scope", "tag", "task"))
    task_rows = _csv(directory / f"{model}-per-task-probabilities.csv")
    summary_rows = _csv(directory / f"{model}-mean-probabilities.csv")
    evidence.file(directory / f"{model}-per-task-probabilities.csv")
    evidence.file(directory / f"{model}-mean-probabilities.csv")
    table = {}
    all_tasks = sorted(set(task))
    for row in task_rows:
        mask = (scope == row["scope"]) & (tag == row["tag"]) & (task == row["task_key"])
        require(int(mask.sum()) == int(row["tokens"]), "Per-task probability denominator differs")
        mean = p[mask].astype(np.float64).mean(0) if mask.any() else None
        for i, role in enumerate(ROLES):
            expected = None if row["p_" + role] == "" else float(row["p_" + role])
            evidence.number(None if mean is None else float(mean[i]), expected, "cached/task_mean", tolerance=1e-11)
        table[row["scope"], row["tag"], row["task_key"]] = (int(mask.sum()), mean)
    require(len(table) == len(task_rows), "Duplicate task probability row")
    for row in summary_rows:
        values = [table[row["scope"], row["tag"], name] for name in all_tasks]
        present = [(n, value) for n, value in values if n]
        count = sum(n for n, _ in present)
        require(int(row["tokens"]) == count and int(row["tasks_observed"]) == len(present)
                and int(row["scheduled_unique_tasks"]) == len(all_tasks)
                and int(row["tasks_without_tag"]) == len(all_tasks) - len(present), "Mean probability support differs")
        if not present:
            mean = None
        elif row["aggregation"] == "unique_task_equal_mean":
            mean = np.mean([value for _, value in present], axis=0)
        else:
            require(row["aggregation"] == "token_pooled_mean", "Unknown mean aggregation")
            mean = sum(n * value for n, value in present) / count
        for i, role in enumerate(ROLES):
            expected = None if row["p_" + role] == "" else float(row["p_" + role])
            evidence.number(None if mean is None else float(mean[i]), expected, "cached/aggregate_mean", tolerance=1e-11)


def validate_cached(calls_root, analysis_root, directory, parent, final, frozen_path, evidence, tolerance):
    calls_root, analysis_root, directory = map(lambda x: Path(x).resolve(), (calls_root, analysis_root, directory))
    source_root = calls_root.parent
    source_manifest_path = source_root / "result_manifest.json"
    source_manifest = read_json(source_manifest_path)
    evidence.file(source_manifest_path)
    require(source_manifest["status"] == "complete", "Source cache is not sealed")
    source_files = {entry["path"]: entry for entry in source_manifest["files"]}
    require(len(source_files) == len(source_manifest["files"]), "Duplicate sealed source path")

    def source_file(path):
        path = Path(path).resolve()
        require(path.is_relative_to(source_root), "Source path escapes the sealed result")
        name = path.relative_to(source_root).as_posix()
        entry = source_files[name]
        require(path.stat().st_size == entry["bytes"], "Sealed source byte count differs")
        return evidence.file(path, entry["sha256"])

    source_file(source_root / "probe_provenance.json")
    provenance = read_json(source_root / "probe_provenance.json")
    require(provenance["sha256"] == sha(parent.path) and tuple(provenance["selected_head_ids"]) == HEADS
            and provenance["model_binding"] == parent.metadata["model_binding"], "Cached model/head provenance differs")
    source_file(analysis_root / "call_body_probabilities.csv")
    source_file(analysis_root / "metrics.json")
    source_metrics = read_json(analysis_root / "metrics.json")
    masks, mask_positions, mask_stats = retained_masks(analysis_root / "call_body_probabilities.csv")
    manifest_path = directory / "result-manifest.json"
    manifest = read_json(manifest_path)
    evidence.file(manifest_path)
    require(manifest["status"] == "complete" and manifest["probe_sha256"] == sha(final.path)
            and manifest["freeze_sha256"] == sha(frozen_path), "New cache evaluation seal differs")
    for entry in manifest["files"]:
        path = safe(directory, entry["path"])
        require(path.stat().st_size == entry["bytes"], "New evaluation artifact byte count differs")
        evidence.file(path, entry["sha256"])
    result = read_json(directory / "metrics.json")
    require(result["status"] == "complete" and result["probe_sha256"] == sha(final.path)
            and result["parent_probe_sha256"] == sha(parent.path) and result["freeze_sha256"] == sha(frozen_path)
            and result["selection_split"] == "dev" and result["historical_regression_only"] is True
            and result["new_language_model_forward"] is False, "Cached evaluation lineage/scope differs")
    saved = np.load(directory / "first-position-probabilities.npz", allow_pickle=False)
    metadata = {key: [] for key in ("scope", "tag", "task", "suite", "segment", "quartile")}
    label_blocks, new_blocks, archive_blocks = [], [], []
    comparison = {name: dict(rows=0, maximum_probability_absolute_error=0., argmax_disagreements=0,
                             largest_producer_top_two_margin_at_disagreement=0.) for name in ("new", "parent")}
    counters = dict(calls=0, input_rows=0, generated_rows=0, generated_observed_rows=0, unforwarded_stops=0)
    retained = defaultdict(int)
    templates = {"system": set(), "developer": set()}
    request_paths = sorted(calls_root.glob("*/*/*/call_*/request.json"))
    require(len(request_paths) == source_metrics["calls"] == 359, "359-call inventory differs")
    for request_path in request_paths:
        source_file(request_path)
        request = read_json(request_path)
        call_id = request_path.parent.relative_to(calls_root).as_posix()
        require(request["metadata"]["call_id"] == call_id and request["status"] == "complete"
                and request["probe_sha256"] == sha(parent.path), "Original call identity differs")
        counters["calls"] += 1
        for scope, name in (("actual_input", "input.npz"), ("actual_generated", "generation_probe.npz")):
            path = request_path.parent / name
            input_sha = source_file(path)
            with np.load(path, allow_pickle=False) as archive:
                x = archive["head_features"]
                source_probabilities = archive["probabilities"]
                require(source_probabilities.dtype == np.float32 and source_probabilities.shape == (len(x), 6),
                        "Original probability shape/dtype differs")
                require(x.dtype == np.float16 and x.ndim == 2 and x.shape[1] == 1024
                        and tuple(archive["head_ids"].tolist()) == HEADS and tuple(archive["roles"].tolist()) == ROLES,
                        "Native feature matrix identity differs")
                offsets = archive["token_char_offsets"]
                if scope == "actual_input":
                    observed = np.ones(len(x), dtype=bool)
                    require(len(x) == request["input_tokens"] and np.array_equal(archive["token_index"], np.arange(len(x))),
                            "Input rows shifted")
                    counters["input_rows"] += len(x)
                else:
                    observed = archive["captured"]
                    require(observed.dtype == bool and observed.shape == (len(x),)
                            and np.array_equal(archive["generated_ids"], request["generated_ids"])
                            and np.array_equal(archive["generated_token_index"], np.arange(len(x)))
                            and np.array_equal(archive["full_sequence_token_index"], request["input_tokens"] + np.arange(len(x))),
                            "Native generated token-state alignment differs")
                    require(np.isnan(x[~observed]).all() and np.isnan(source_probabilities[~observed]).all(),
                            "Missing generated states were filled")
                    missing = np.flatnonzero(~observed)
                    require(all(i == len(x) - 1 and int(archive["generated_ids"][i]) in (200002, 200012) for i in missing),
                            "An ordinary generated token has no own state")
                    require(np.isin(archive["capture_source"][observed], ["native_decode_forward", "terminal_completion_forward"]).all(),
                            "Generated features are not native own-token states")
                    require((archive["capture_source"][~observed] == "not_forwarded_stop_delimiter").all(),
                            "Missing generated-state provenance differs")
                    terminal = np.flatnonzero(archive["capture_source"] == "terminal_completion_forward")
                    require(all(i == len(x) - 1 and int(archive["generated_ids"][i]) not in (200002, 200012)
                                for i in terminal), "Invalid terminal ordinary-token completion")
                    counters["generated_rows"] += len(x)
                    counters["generated_observed_rows"] += int(observed.sum())
                    counters["unforwarded_stops"] += len(missing)
                require(offsets.shape == (len(x), 2) and np.isfinite(x[observed]).all(), "Invalid observed features/offsets")
            independent_parent, independent_new = parent.predict(x[observed]), final.predict(x[observed])
            new_path = directory / "probabilities" / call_id / (scope + ".npz")
            require(new_path.is_file(), "Full native-state probability artifact missing")
            with np.load(new_path, allow_pickle=False) as archive:
                actual = archive["probabilities"]
                require(actual.dtype == np.float32 and actual.shape == (len(x), 6) and np.array_equal(archive["captured"], observed)
                        and np.isnan(actual[~observed]).all() and np.array_equal(archive["token_index"], np.arange(len(x)))
                        and tuple(archive["roles"].tolist()) == ROLES
                        and str(archive["source_npz_sha256"].item()) == input_sha
                        and str(archive["probe_sha256"].item()) == sha(final.path), "New cached probabilities shifted or unbound")
            for model, independent, target in (("parent", independent_parent, source_probabilities[observed]),
                                                ("new", independent_new, actual[observed])):
                record = probability_comparison(independent, target, tolerance)
                comparison[model]["rows"] += record["rows"]
                comparison[model]["maximum_probability_absolute_error"] = max(comparison[model]["maximum_probability_absolute_error"],
                                                                               record["maximum_probability_absolute_error"])
                comparison[model]["argmax_disagreements"] += record["argmax_disagreements"]
                comparison[model]["largest_producer_top_two_margin_at_disagreement"] = max(
                    comparison[model]["largest_producer_top_two_margin_at_disagreement"], record["largest_producer_top_two_margin_at_disagreement"])
            for row, indices in masks.get((call_id, scope), []):
                require(not len(indices) or indices[-1] < len(x), "First-position index exceeds source")
                require(observed[indices].all(), "First-position mask includes an unforwarded stop")
                labels = np.full(len(indices), TAG_ROLE[row["tag"]], dtype=np.int64)
                label_blocks.append(labels)
                new_blocks.append(actual[indices])
                archive_blocks.append(source_probabilities[indices])
                retained[scope + "/" + row["tag"]] += len(indices)
                for key, value in (("scope", scope), ("tag", row["tag"]), ("task", row["task_key"]),
                                   ("suite", row["task_key"].split("/")[0])):
                    metadata[key].extend([value] * len(indices))
                metadata["quartile"].extend((np.arange(len(indices)) * 4 // max(len(indices), 1)).tolist())
                start, end = int(row["body_char_start"]), int(row["body_char_end"])
                metadata["segment"].extend(_cached_segments(row["tag"], request["rendered_prompt"], start, end, offsets[indices]))
                if row["tag"] in templates and scope == "actual_input":
                    templates[row["tag"]].add(hashlib.sha256(request["rendered_prompt"][start:end].encode()).hexdigest())
                sums = source_probabilities[indices].astype(np.float64).sum(0)
                for i, role in enumerate(ROLES):
                    evidence.number(float(sums[i]), float(row["sum_" + role]), "cached/original_mask_sum", tolerance=1e-9)
    labels = np.concatenate(label_blocks)
    values = {"new": np.concatenate(new_blocks), "archived_parent": np.concatenate(archive_blocks), "parent": saved["parent"]}
    require(np.array_equal(labels, saved["labels"]), "Retained ground-truth sequence differs")
    for key in metadata:
        metadata[key] = np.asarray(metadata[key])
        require(np.array_equal(metadata[key], saved[key]), "Retained scope/group sequence differs: " + key)
    for model in ("new", "archived_parent"):
        require(np.array_equal(values[model], saved[model]), "Retained probability sequence differs: " + model)
    # The parent selected values must agree with the independent CPU readout;
    # use a separate selected-row pass to keep source NaNs and scope order exact.
    selected_parent = []
    for request_path in request_paths:
        call_id = request_path.parent.relative_to(calls_root).as_posix()
        for scope, name in (("actual_input", "input.npz"), ("actual_generated", "generation_probe.npz")):
            rows = masks.get((call_id, scope), [])
            if rows:
                with np.load(request_path.parent / name, allow_pickle=False) as archive:
                    x = archive["head_features"]
                for _, indices in rows:
                    selected_parent.append(parent.predict(x[indices]))
    selected_parent = np.concatenate(selected_parent)
    parent_selected_comparison = probability_comparison(selected_parent, values["parent"], tolerance)
    require(len(labels) == 155103 == len(mask_positions) and mask_stats["rows"] == 1470,
            "First-position normal-body denominator differs")
    require(mask_stats["empty_rows"] == mask_stats["structural_empty_rows"] == 603,
            "Empty structural N/A-frame inventory differs")
    require(counters == dict(calls=359, input_rows=596379, generated_rows=79552,
                            generated_observed_rows=79194, unforwarded_stops=358), "Native cache inventory differs")
    require({key: value for key, value in retained.items() if value}
            == {key: value for key, value in source_metrics["retained_normal_positions"].items() if value},
            "Retained role denominators differ")
    require(len(set(metadata["task"])) == result["tasks"] == 57, "Unique native task count differs")
    reuse = read_json(directory / "cache-reuse-audit.json")
    require(reuse["source_result_manifest_sha256"] == sha(source_manifest_path)
            and reuse["first_position_npz_sha256"] == sha(directory / "first-position-probabilities.npz"),
            "Cached reuse audit source identity differs")
    require(result["template_counts"] == {key: len(value) for key, value in templates.items()}
            == {"system": 1, "developer": 3}, "Template support was overstated")
    for model, p in values.items():
        expected = result["models"][model]
        _cached_metric_compare(labels, p, expected["overall"], evidence, "cached/" + model + "/overall")
        for key, attribute in (("by_scope", "scope"), ("by_suite", "suite"), ("sd_segments", "segment")):
            keys = set(metadata[attribute]) - ({""} if attribute == "segment" else set())
            require(set(expected[key]) == keys, "Cached metric group inventory differs")
            for value in keys:
                mask = metadata[attribute] == value
                _cached_metric_compare(labels[mask], p[mask], expected[key][value], evidence, "cached/" + model + "/" + key)
        for role in ("system", "developer"):
            for quarter in range(4):
                mask = (metadata["tag"] == role) & (metadata["quartile"] == quarter)
                _cached_metric_compare(labels[mask], p[mask], expected["sd_body_quartiles"][f"{role}/q{quarter + 1}"],
                                       evidence, "cached/" + model + "/quarter")
        _mean_tables(p, metadata, directory, model, evidence)
    return dict(status="passed", calls=counters["calls"], source_result_manifest_sha256=sha(source_manifest_path),
        all_native_observed_feature_rows=counters["input_rows"] + counters["generated_observed_rows"],
        retained_normal_tokens=len(labels), retained_normal_positions=dict(retained), unforwarded_stop_delimiters=358,
        retained_frame_masks=mask_stats,
        unique_system_templates=1, unique_developer_templates=3, complete_torch_readout_comparisons=comparison,
        selected_parent_torch_comparison=parent_selected_comparison,
        metric_scopes_checked=["overall", "input/generated", "suite", "S/D segments", "S/D quarters", "task-equal and token-pooled probabilities"],
        new_language_model_forward=False, native_own_token_decode_preserved=True)


def self_test():
    """Temporary CPU fixtures only; never writes a claimed experiment audit."""
    import torch
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="role-warm-independent-fixture-") as temporary:
        root = Path(temporary)
        torch.manual_seed(17)
        first, last = torch.nn.Linear(1024, 128), torch.nn.Linear(128, 6)
        weights = {"0.weight": first.weight.detach().numpy(), "0.bias": first.bias.detach().numpy(),
                   "3.weight": last.weight.detach().numpy(), "3.bias": last.bias.detach().numpy()}
        mean, scale = np.zeros(1024, np.float32), np.ones(1024, np.float32)
        metadata = dict(format_version="synthetic-agent-six-role-probe-v1", roles=list(ROLES), heads=list(HEADS),
                        feature_dim=1024, feature_position="pre-o_proj", hidden=128, activation="gelu_exact",
                        model_binding={"model": "independent-temporary-fixture"})
        path = root / "fixture.npz"
        np.savez(path, metadata=np.array(json.dumps(metadata)), mean=mean, scale=scale,
                 **{"weight::" + k: v for k, v in weights.items()})
        probe = ExactTorchProbe(path)
        x = np.random.default_rng(23).normal(0, .2, (48, 1024)).astype(np.float16)
        p = probe.predict(x, batch_size=7)
        with torch.inference_mode():
            reference = torch.softmax(last(torch.nn.functional.gelu(first(torch.from_numpy(x.astype(np.float32))), approximate="none")), 1).numpy()
        probability_comparison(p, reference, 1e-6)
        labels = np.arange(48, dtype=np.int64) % 6
        sources = np.asarray(["cluster-" + str(i // 6) for i in range(48)])
        value = metrics(labels, p)
        require(sum(map(sum, value["confusion"])) == 48, "Fixture confusion mismatch")
        ci = bootstrap(labels, p, sources, 100, 20261026)
        require(ci["clusters"] == 8 and ci["intervals"]["accuracy"]["valid_replicates"] == 100
                and all(v is None or 1 <= v["valid_replicates"] <= 100 for v in ci["intervals"].values()), "Fixture bootstrap denominator mismatch")
        require(ci == bootstrap(labels, p, sources, 100, 20261026), "Fixture bootstrap is not reproducible")
        # Unequal cluster sizes distinguish whole-source from token bootstrap.
        unequal_y = np.tile(np.arange(6, dtype=np.int64), 11)
        unequal_p = np.full((66, 6), .02, dtype=np.float32)
        unequal_p[np.arange(66), unequal_y] = .9
        unequal_p[:6] = np.roll(unequal_p[:6], 1, axis=1)
        unequal_sources = np.asarray(["bad-small"] * 6 + ["good-large"] * 60)
        unequal_ci = bootstrap(unequal_y, unequal_p, unequal_sources, 100, 20261026)
        require(unequal_ci["intervals"]["accuracy"]["lower"] == 0
                and unequal_ci["intervals"]["accuracy"]["upper"] == 1,
                "Whole-source fixture was replaced by token resampling")
        known_p = np.zeros((6, 6), dtype=np.float32)
        known_p[:, 0] = 1
        known = metrics(np.arange(6), known_p)
        require(known["roles"]["system"]["precision"] == 1 / 6
                and known["roles"]["developer"]["precision"] is None
                and known["roles"]["developer"]["recall"] == 0
                and abs(known["macro_f1"] - 1 / 21) < 1e-12,
                "Manual absent-prediction/role metric convention differs")
        high_loss = np.zeros((798, 6), dtype=np.float32)
        high_loss[:, 0] = np.float32(math.exp(-27.2644781))
        high_loss[:, 1] = 1 - high_loss[:, 0]
        high_metric = metrics(np.zeros(798, np.int64), high_loss)
        expected32 = float((-np.log(np.clip(high_loss[:, 0], 1e-30, 1))).mean())
        expected64 = -math.log(float(high_loss[0, 0]))
        require(high_metric["nll"] == expected32
                and abs(high_metric["nll_float64_diagnostic"] - expected64) < 1e-12
                and abs(expected32 - expected64) > 1e-6,
                "High-loss FP32 publication contract was replaced by FP64 reduction")
        cache = root / "cache"
        cache.mkdir()
        signature = dict(input_sha256="fixture-input", heads=list(HEADS), model_binding=metadata["model_binding"],
                         parent_probe_sha256=sha(path), runtime_sha256="fixture-runtime", extractor_sha256="fixture-extractor")
        entries = []
        for i, (start, stop) in enumerate(((0, 23), (23, 48))):
            directory = cache / f"test-{i}"
            directory.mkdir()
            arrays = dict(x=x[start:stop], y=labels[start:stop], source=sources[start:stop],
                family=np.asarray([f"family-{j}" for j in range(start, stop)]),
                structure=np.full(stop - start, "paired:fixture"), weight=np.ones(stop - start, dtype=np.float32),
                groups=np.full(stop - start, '{"profile":"paired"}'))
            hashes = {}
            for key, array in arrays.items():
                np.save(directory / (key + ".npy"), array, allow_pickle=False)
                hashes[key + ".npy"] = sha(directory / (key + ".npy"))
            complete = dict(rows=stop - start, split="test", signature=signature, files=hashes)
            (directory / "complete.json").write_text(json.dumps(complete))
            entries.append(dict(path=directory.name, rows=stop - start, sha256=sha(directory / "complete.json")))
        manifest = {key: metadata[key] for key in BINDING}
        manifest.update(format_version="synthetic-role-warm-features-v1", parent_probe_sha256=sha(path),
            inputs={"test": {"signature": signature, "sha256": "fixture-input"}},
            splits={"test": entries}, progress={"test": {"complete": True, "rows": 48}})
        (cache / "manifest.json").write_text(json.dumps(manifest))
        bank = VerifiedBank(cache, "test", probe, Evidence())
        np.testing.assert_array_equal(bank.read([47, 0, 23, 24, 0]), x[[47, 0, 23, 24, 0]])
        probability_comparison(bank.predict(probe), p, 1e-6)
        manifest["progress"]["test"]["complete"] = False
        (cache / "manifest.json").write_text(json.dumps(manifest))
        try:
            VerifiedBank(cache, "test", probe, Evidence())
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete cache was accepted")
        manifest["progress"]["test"]["complete"] = True
        (cache / "manifest.json").write_text(json.dumps(manifest))
        with (cache / "test-0/x.npy").open("ab") as stream:
            stream.write(b"corrupt-array")
        try:
            VerifiedBank(cache, "test", probe, Evidence())
        except ValueError:
            pass
        else:
            raise AssertionError("Feature corruption was accepted")
        try:
            probability_comparison(p, np.roll(p, 1, axis=1), 1e-6)
        except ValueError:
            pass
        else:
            raise AssertionError("Column corruption was not detected")
        try:
            safe(root, "../outside")
        except ValueError:
            pass
        else:
            raise AssertionError("Path escape was not detected")
        mask_path = root / "mask-fixture.csv"
        mask_schema = ["content_filter", "scope", "tag", "token_indices", "tokens", "structural_tool_call", "call_id"]
        rows = [dict(zip(mask_schema, ["normal_content", "actual_generated", "assistant_commentary", "[]", 0, "True", "call-0"])),
                dict(zip(mask_schema, ["normal_content", "actual_input", "system", "[3, 4]", 2, "False", "call-0"]))]
        with mask_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=mask_schema)
            writer.writeheader()
            writer.writerows(rows)
        _, positions, mask_stats = retained_masks(mask_path)
        require(mask_stats == dict(rows=2, empty_rows=1, structural_empty_rows=1) and len(positions) == 2,
                "Empty structural N/A frame was dropped or scored")
        rows[0].update(token_indices="[0]", tokens=1)
        with mask_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=mask_schema)
            writer.writeheader()
            writer.writerows(rows)
        try:
            retained_masks(mask_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Structural tool-call tokens entered the ordinary-body mask")
        evidence = Evidence()
        evidence.file(path)
        with path.open("ab") as stream:
            stream.write(b"changed")
        try:
            evidence.file(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed artifact was not detected")
    return dict(status="self_tests_passed", temporary_fixtures_only=True,
                tests=["manual exact GELU readout", "partial readout batches", "six-role confusion",
                       "absent-class convention", "whole-source bootstrap unequal-size clusters",
                       "whole-source bootstrap reproducibility", "verified cross-shard array ordering",
                       "high-NLL FP32 publication contract", "empty structural N/A frames",
                       "nonempty structural-call mask rejection", "incomplete cache rejection",
                       "feature hash corruption", "role-column corruption", "path escape", "artifact mutation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--old-features-dir", type=Path)
    parser.add_argument("--old-sources", type=Path)
    parser.add_argument("--calls-root", type=Path)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--cached-evaluation-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-parent-sha256", default=PARENT_SHA)
    parser.add_argument("--probability-tolerance", type=float, default=5e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
    else:
        require(args.root is not None and args.output is not None, "A complete local mirror and audit output are required")
        old = args.old_features_dir or args.root / "old-features"
        report = validate_root(args.root, old, args.output, expected_parent_sha=args.expected_parent_sha256,
            calls_root=args.calls_root, analysis_root=args.analysis_root, cached_evaluation_dir=args.cached_evaluation_dir,
            probability_tolerance=args.probability_tolerance, old_sources_path=args.old_sources, threads=args.threads)
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "selection_candidate", "tests") if key in report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
