"""Six-class readout for frozen, synthetic-only GPT-OSS attention features.

The consumer deliberately rejects historical five-class exports.  Feature arrays
are memory mapped; only labels and sampling metadata are kept in RAM.  Selection
uses DEV only, and the TEST entry point requires an immutable freeze manifest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
FEATURE_FORMAT = "synthetic-agent-features-v1"
PROBE_FORMAT = "synthetic-agent-six-role-probe-v1"
CANDIDATES = (
    ("linear-seed17", 0, 17),
    ("mlp128-seed17", 128, 17),
    ("mlp128-seed29", 128, 29),
    ("mlp128-seed43", 128, 43),
    ("mlp256-seed17", 256, 17),
    ("mlp256-seed29", 256, 29),
    ("mlp256-seed43", 256, 43),
)
PROFILE_ORDER = ("single", "one_round", "multi_round")
DEFAULT_PROFILE_WEIGHTS = {"single": 0.2, "one_round": 0.3, "multi_round": 0.5}


def _profile_weights(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(PROFILE_ORDER):
        raise ValueError("Profile weights must name only single, one_round, multi_round")
    weights = {key: float(value.get(key, 0.0)) for key in PROFILE_ORDER}
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Profile weights must be finite, nonnegative, and have positive total mass")
    total = sum(weights.values())
    return {key: weights[key] / total for key in PROFILE_ORDER}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def _atomic_torch_save(path, value):
    import torch
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _safe_path(root, path):
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Feature path escapes cache: {path}")
    return resolved


class FeatureBank:
    """Memory-mapped verified shards; labels/IDs may be resident in CPU RAM.

    A shard entry is {path, rows, sha256}; ``sha256`` is the digest of the
    shard's complete.json, whose ``files`` field maps filenames to SHA256s.
    Explicit x/y/structure/source/family/weight paths are also accepted.
    Optional groups.npy holds JSON objects with diagnostic group attributes.
    """

    def __init__(self, feature_dir, split, *, expected_binding=None, verify=True):
        self.root = Path(feature_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if m.get("format_version") != FEATURE_FORMAT or tuple(m.get("roles", ())) != ROLES:
            raise ValueError("Only the new six-role feature format is accepted")
        if m.get("feature_position") != "pre-o_proj" or int(m.get("feature_dim", -1)) != 1024:
            raise ValueError("Expected ordered 16-head, 1024-dimensional pre-o_proj features")
        heads = m.get("heads", [])
        if len(heads) != 16 or len(set(heads)) != 16:
            raise ValueError("Feature manifest requires 16 distinct ordered heads")
        for key in ("model_binding", "data_hash", "config_hash"):
            if not m.get(key):
                raise ValueError(f"Missing cache binding: {key}")
        signature = m.get("signature")
        if not isinstance(signature, dict) or not signature:
            raise ValueError("Feature manifest has no extraction signature")
        for key, expected in (("model_binding", m["model_binding"]), ("heads", heads),
                              ("config_sha256", m["config_hash"]), ("capture", m["feature_position"])):
            if signature.get(key) != expected:
                raise ValueError(f"Top-level feature binding disagrees with extraction signature: {key}")
        for key in ("selection_sha256", "pilot_fit_sha256"):
            if not m.get(key) or signature.get(key) != m[key]:
                raise ValueError(f"Missing or inconsistent pilot head-discovery binding: {key}")
        input_sha = m.get("split_sources", {}).get(split)
        if not input_sha:
            raise ValueError(f"Feature manifest has no input hash for {split!r}")
        expected_shard_signature = {**signature, "split": split, "input_sha256": input_sha}
        if expected_binding is not None:
            for key in ("roles", "feature_dim", "heads", "feature_position", "model_binding",
                        "selection_sha256", "pilot_fit_sha256"):
                if m.get(key) != expected_binding.get(key):
                    raise ValueError(f"Feature/probe binding mismatch: {key}")
        self.split = split
        self.shards = []
        metadata = {k: [] for k in ("y", "structure", "source", "family", "weight", "groups")}
        offsets = [0]
        entries = m.get("splits", {}).get(split)
        if not entries:
            raise ValueError(f"No nonempty {split!r} split in feature manifest")
        if "progress" in m and not m["progress"].get(split, {}).get("complete"):
            raise ValueError(f"Feature extraction for {split!r} is not complete")
        for entry in entries:
            base = _safe_path(self.root, entry.get("path", "."))
            complete = base / "complete.json"
            if not complete.is_file():
                raise ValueError(f"Missing shard commit metadata: {base}")
            record = json.loads(complete.read_text())
            if record.get("signature") != expected_shard_signature or record.get("rows") != int(entry["rows"]):
                raise ValueError(f"Shard signature differs from manifest/model/heads/input: {base}")
            file_hashes = record.get("files", record.get("sha256", {}))
            if isinstance(file_hashes, list):
                file_hashes = {x["path"]: x["sha256"] for x in file_hashes}
            if not isinstance(file_hashes, dict) or not file_hashes:
                raise ValueError(f"Missing per-file hashes: {complete}")
            if verify:
                if not entry.get("sha256"):
                    raise ValueError(f"Missing shard commit hash: {base}")
                if _sha(complete) != entry["sha256"]:
                    raise ValueError(f"Invalid shard commit: {base}")
                for name, digest in file_hashes.items():
                    digest = digest.get("sha256") if isinstance(digest, dict) else digest
                    if _sha(_safe_path(base, name)) != digest:
                        raise ValueError(f"Invalid feature file: {base / name}")
            paths = {}
            for key in metadata.keys() | {"x"}:
                paths[key] = _safe_path(self.root, entry[key]) if key in entry else base / f"{key}.npy"
                if paths[key].is_file():
                    resolved_path = paths[key].resolve()
                    if not resolved_path.is_relative_to(base.resolve()) or str(resolved_path.relative_to(base.resolve())) not in file_hashes:
                        raise ValueError(f"Consumed array is not covered by its shard commit: {paths[key]}")
            x = np.load(paths["x"], mmap_mode="r", allow_pickle=False)
            n = len(x)
            if x.dtype != np.float16 or x.shape != (n, 1024) or n != int(entry["rows"]):
                raise ValueError(f"Incorrect FP16 feature shape: {base}")
            self.shards.append(x)
            offsets.append(offsets[-1] + n)
            for key in metadata:
                if paths[key].is_file():
                    v = np.load(paths[key], mmap_mode="r", allow_pickle=False)
                    if v.shape != (n,):
                        raise ValueError(f"Incorrect metadata shape: {paths[key]}")
                    metadata[key].append(np.array(v))
                elif key == "groups":
                    metadata[key].append(np.full(n, "{}"))
                elif key == "weight":
                    metadata[key].append(np.ones(n, dtype=np.float32))
                else:
                    raise ValueError(f"Required sampling metadata missing: {paths[key]}")
        self.offsets = np.array(offsets, dtype=np.int64)
        for key, pieces in metadata.items():
            setattr(self, key, np.concatenate(pieces))
        self.y = self.y.astype(np.int64)
        self.weight = self.weight.astype(np.float64)
        if np.any((self.y < 0) | (self.y >= len(ROLES))):
            raise ValueError("Unknown role ID in feature cache")
        if np.any(~np.isfinite(self.weight)) or np.any(self.weight <= 0):
            raise ValueError("Training weights must be finite and positive")
        if np.any(self.family.astype(str) == "") or np.any(self.source.astype(str) == ""):
            raise ValueError("Family and source/cluster IDs cannot be empty")
        # A snapshot family receives one unit of base mass regardless of how
        # many times a historical body token was extracted in that family.
        _, family_index = np.unique(self.family, return_inverse=True)
        totals = np.bincount(family_index, weights=self.weight)
        self.base_weight = self.weight / totals[family_index]
        self.manifest_hash = _sha(self.root / "manifest.json")
        # Extraction can legitimately update timing fields while replaying a
        # fully verified cache. Resume identity binds committed bytes and input
        # provenance, not such operational bookkeeping or the cache's location.
        progress = m.get("progress", {}).get(split, {})
        self.split_identity_hash = hashlib.sha256(_json({
            "format_version": m["format_version"], "roles": m["roles"],
            "feature_dim": m["feature_dim"], "feature_position": m["feature_position"],
            "signature": signature, "data_hash": m["data_hash"], "split": split,
            "input_sha256": input_sha, "shards": entries,
            "complete": progress.get("complete"), "records": progress.get("records"),
            "rows": progress.get("rows"),
        }).encode()).hexdigest()

    def __len__(self):
        return int(self.offsets[-1])

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Feature row outside cache")
        result = np.empty((len(indices), 1024), dtype=np.float32)
        shard_id = np.searchsorted(self.offsets[1:], indices, side="right")
        for s in np.unique(shard_id):
            locations = np.flatnonzero(shard_id == s)
            result[locations] = self.shards[s][indices[locations] - self.offsets[s]]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen-model features")
        return result

    def binding(self):
        return {k: self.manifest[k] for k in (
            "roles", "feature_dim", "heads", "feature_position", "model_binding",
            "selection_sha256", "pilot_fit_sha256", "data_hash", "config_hash")}


class BalancedSampler:
    """Profile quota, uniform structure, present role, then family base mass.

    No nonexistent structure-role cell is fabricated, and no class weight is
    added to cross entropy. ``profile_weights=None`` retains the generic flat
    structure sampler for explicitly labelled engineering fixtures only.
    """

    def __init__(self, bank, profile_weights=None):
        self.profile_weights = _profile_weights(profile_weights)
        self.cells = []
        self.structures = []
        self.report = []
        structure_values = np.unique(bank.structure)
        profile_values = []
        for structure in structure_values:
            profile, separator, template = str(structure).partition(":")
            if self.profile_weights is not None:
                if profile not in PROFILE_ORDER or not separator or not template:
                    raise ValueError(f"Scientific structure requires a known profile:template prefix: {structure}")
            profile_values.append(profile)
        self.profile_labels = tuple(key for key in PROFILE_ORDER if key in profile_values) + tuple(
            sorted(set(profile_values) - set(PROFILE_ORDER)))
        self.profile_index = {key: i for i, key in enumerate(self.profile_labels)}
        self.row_profile = np.empty(len(bank), dtype=np.int16)
        self.structure_profiles = []
        if self.profile_weights is not None:
            for profile, weight in self.profile_weights.items():
                if weight > 0 and profile not in profile_values:
                    raise ValueError(f"Positive-weight profile has no applicable structures: {profile}")
        self.profiles = tuple(key for key in PROFILE_ORDER if self.profile_weights is not None and self.profile_weights[key] > 0)
        self.profile_probabilities = np.asarray([self.profile_weights[key] for key in self.profiles], dtype=float)
        for structure in structure_values:
            cells = []
            smask = bank.structure == structure
            profile = str(structure).partition(":")[0]
            self.row_profile[smask] = self.profile_index[profile]
            self.structure_profiles.append(profile)
            for role in np.unique(bank.y[smask]):
                indices = np.flatnonzero(smask & (bank.y == role))
                weights = bank.base_weight[indices]
                probabilities = weights / weights.sum()
                cells.append((indices, probabilities))
                self.report.append({"profile": profile, "structure": str(structure), "role": ROLES[int(role)],
                                    "rows": len(indices), "family_mass": float(weights.sum())})
            self.structures.append(str(structure))
            self.cells.append(cells)
        self.profile_structures = {profile: np.flatnonzero(np.asarray(self.structure_profiles) == profile)
                                   for profile in self.profiles}
        self.protocol = {
            "mode": "profile_then_structure_then_present_role_then_family" if self.profile_weights is not None else "flat_structure_engineering",
            "profile_order": list(self.profiles if self.profile_weights is not None else self.profile_labels),
            "profile_weights": self.profile_weights,
            "distinct_structures_by_profile": {profile: self.structure_profiles.count(profile) for profile in self.profile_labels},
            "role_rule": "uniform among roles actually present in the selected structure",
            "row_rule": "family-normalized base weight within the selected structure/role cell",
        }

    def draw(self, rng, count):
        out = np.empty(count, dtype=np.int64)
        if self.profile_weights is None:
            structure_draws = rng.integers(len(self.cells), size=count)
        else:
            structure_draws = np.empty(count, dtype=np.int64)
            profile_draws = rng.choice(len(self.profiles), size=count, p=self.profile_probabilities)
            for p, profile in enumerate(self.profiles):
                locations = np.flatnonzero(profile_draws == p)
                structures = self.profile_structures[profile]
                structure_draws[locations] = structures[rng.integers(len(structures), size=len(locations))]
        for s, cells in enumerate(self.cells):
            selected = np.flatnonzero(structure_draws == s)
            role_draws = rng.integers(len(cells), size=len(selected))
            for r, (indices, probabilities) in enumerate(cells):
                destinations = selected[role_draws == r]
                out[destinations] = rng.choice(indices, size=len(destinations), p=probabilities)
        return out

    def count_profiles(self, indices):
        counts = np.bincount(self.row_profile[np.asarray(indices, dtype=np.int64)], minlength=len(self.profile_labels))
        return {profile: int(counts[i]) for i, profile in enumerate(self.profile_labels)}


def fit_standardizer(bank, chunk_size=8192):
    """Weighted online sufficient statistics, FIT only; no feature megacopy."""
    if bank.split != "fit":
        raise ValueError("Standardization may only be fitted on FIT")
    total = 0.0
    mean = np.zeros(1024, dtype=np.float64)
    m2 = np.zeros(1024, dtype=np.float64)
    for start in range(0, len(bank), chunk_size):
        stop = min(start + chunk_size, len(bank))
        x = bank.read(np.arange(start, stop)).astype(np.float64)
        w = bank.base_weight[start:stop]
        mass = float(w.sum())
        chunk_mean = np.sum(x * w[:, None], axis=0) / mass
        chunk_m2 = np.sum((x - chunk_mean) ** 2 * w[:, None], axis=0)
        delta = chunk_mean - mean
        combined = total + mass
        m2 += chunk_m2 + delta ** 2 * (total * mass / combined)
        mean += delta * (mass / combined)
        total = combined
    scale = np.sqrt(np.maximum(m2 / total, 0))
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def make_net(hidden, *, dropout=0.1):
    from torch import nn
    if hidden:
        return nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 6))
    return nn.Sequential(nn.Linear(1024, 6))


def _metrics_from_confusion(matrix, nll=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = np.divide(correct, predicted, out=np.zeros(6, dtype=float), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(6, dtype=float), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(6, dtype=float), where=(support + predicted) > 0)
    result = {"tokens": int(matrix.sum()), "accuracy": float(correct.sum() / max(matrix.sum(), 1)),
              "macro_f1": float(f1[support > 0].mean()) if (support > 0).any() else None,
              "confusion": matrix.tolist(), "roles": {}}
    if nll is not None:
        result["nll"] = float(nll)
    for i, role in enumerate(ROLES):
        result["roles"][role] = {
            "support": int(support[i]), "predicted": int(predicted[i]),
            "precision": float(precision[i]) if predicted[i] else None,
            "recall": float(recall[i]) if support[i] else None,
            "f1": float(f1[i]) if support[i] else None,
            "false_positives": int(predicted[i] - correct[i]),
            "false_positive_rate": float((predicted[i] - correct[i]) / (matrix.sum() - support[i]))
            if matrix.sum() > support[i] else None,
        }
    return result


def metrics(probabilities, labels):
    p, y = np.asarray(probabilities), np.asarray(labels, dtype=np.int64)
    if p.shape != (len(y), 6) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=2e-5):
        raise ValueError("Invalid six-role probabilities")
    pred = p.argmax(axis=1)
    confusion = np.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-30, 1)).mean()
    return _metrics_from_confusion(confusion, nll)


def grouped_metrics(bank, probabilities):
    result = {"overall": metrics(probabilities, bank.y), "by_structure": {}}
    for structure in np.unique(bank.structure):
        mask = bank.structure == structure
        result["by_structure"][str(structure)] = metrics(probabilities[mask], bank.y[mask])
    result["structure_equal_macro_f1"] = float(np.mean([v["macro_f1"] for v in result["by_structure"].values()]))
    values = []
    for role in ROLES:
        r = result["overall"]["roles"][role]
        values.extend([r["precision"] or 0.0, r["recall"] or 0.0])
    result["minimum_precision_recall"] = float(min(values))
    result["meets_dev_floor"] = bool(min(values) >= 0.9)
    group_objects = [json.loads(str(s)) for s in bank.groups]
    group_keys = sorted(set().union(*(obj.keys() for obj in group_objects)))
    if group_keys:
        result["by_attribute"] = {}
        for key in group_keys:
            groups = np.array([str(obj.get(key, "N/A")) for obj in group_objects])
            result["by_attribute"][key] = {}
            for value in np.unique(groups):
                mask = groups == value
                result["by_attribute"][key][str(value)] = metrics(probabilities[mask], bank.y[mask])
    return result


def selection_key(result, candidate_id):
    """Smallest tuple wins. Candidate ID is the fixed final tiebreaker."""
    return (not result["meets_dev_floor"], -result["structure_equal_macro_f1"],
            -result["minimum_precision_recall"], result["overall"]["nll"], candidate_id)


def bootstrap_clusters(bank, probabilities, *, replicates=2000, seed=20261026):
    """Cluster resampling of sufficient statistics, preserving all family rows."""
    sources, inv = np.unique(bank.source, return_inverse=True)
    count = len(sources)
    if count < 2:
        raise ValueError("Cluster bootstrap requires at least two held-out source clusters")
    pred = probabilities.argmax(axis=1)
    confusion = np.zeros((count, 6, 6), dtype=np.int64)
    np.add.at(confusion, (inv, bank.y, pred), 1)
    losses = -np.log(np.clip(probabilities[np.arange(len(bank)), bank.y], 1e-30, 1))
    loss_sum = np.bincount(inv, weights=losses, minlength=count)
    totals = confusion.sum(axis=(1, 2))
    rng = np.random.default_rng(seed)
    samples = {"accuracy": [], "macro_f1": [], "nll": []}
    for role in ROLES:
        for metric in ("precision", "recall", "f1"):
            samples[f"{role}.{metric}"] = []
    for _ in range(replicates):
        draw = rng.integers(count, size=count)
        m = _metrics_from_confusion(confusion[draw].sum(axis=0))
        samples["accuracy"].append(m["accuracy"])
        samples["macro_f1"].append(m["macro_f1"])
        samples["nll"].append(float(loss_sum[draw].sum() / totals[draw].sum()))
        for role in ROLES:
            for metric in ("precision", "recall", "f1"):
                value = m["roles"][role][metric]
                if value is not None:
                    samples[f"{role}.{metric}"].append(value)
    intervals = {}
    for key, values in samples.items():
        intervals[key] = {"lower": float(np.quantile(values, 0.025)),
                          "upper": float(np.quantile(values, 0.975)),
                          "valid_replicates": len(values)} if values else None
    return {"unit": "source_cluster", "clusters": count, "replicates": replicates,
            "seed": seed, "method": "percentile", "confidence": 0.95, "intervals": intervals}


def _torch_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["torch_cuda"]])


def _feature_batch(bank, indices, device):
    """Read identical raw FP32 rows from an optional per-run CUDA cache."""
    import torch
    if hasattr(bank, "device_features"):
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        return bank.device_features.index_select(0, index)
    return torch.from_numpy(bank.read(indices)).to(device)


def _cache_features_on_device(bank, device, chunk_size=8192):
    import torch
    if not str(device).startswith("cuda"):
        raise ValueError("Feature preloading requires a CUDA device")
    bank.device_features = torch.empty((len(bank), 1024), dtype=torch.float32, device=device)
    for start in range(0, len(bank), chunk_size):
        stop = min(start + chunk_size, len(bank))
        bank.device_features[start:stop] = torch.from_numpy(bank.read(np.arange(start, stop))).to(device)


def _predict(net, bank, mean, scale, *, device, batch_size=8192):
    import torch
    net.eval()
    result = np.empty((len(bank), 6), dtype=np.float32)
    mean_t = torch.as_tensor(mean, device=device)
    scale_t = torch.as_tensor(scale, device=device)
    with torch.inference_mode():
        for start in range(0, len(bank), batch_size):
            stop = min(start + batch_size, len(bank))
            x = _feature_batch(bank, np.arange(start, stop), device)
            result[start:stop] = net((x - mean_t) / scale_t).softmax(dim=1).cpu().numpy()
    return result


def _export(path, net, mean, scale, metadata):
    arrays = {f"weight::{name}": value.detach().cpu().numpy() for name, value in net.state_dict().items()}
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, mean=mean, scale=scale, metadata=np.array(_json(metadata)), **arrays)
    os.replace(temp, path)


class Probe:
    def __init__(self, path, *, expected_binding=None):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"]))
            m = self.metadata
            if m.get("format_version") != PROBE_FORMAT or tuple(m.get("roles", ())) != ROLES:
                raise ValueError("Not a six-role synthetic-agent probe; old five-role exports are incompatible")
            if m.get("feature_position") != "pre-o_proj" or m.get("feature_dim") != 1024:
                raise ValueError("Unsupported probe feature binding")
            if len(m.get("heads", [])) != 16 or len(set(m["heads"])) != 16 or not m.get("model_binding"):
                raise ValueError("Incomplete model/head binding")
            if expected_binding:
                for key, value in expected_binding.items():
                    if m.get(key) != value:
                        raise ValueError(f"Probe binding differs: {key}")
            self.mean = archive["mean"].astype(np.float32)
            self.scale = archive["scale"].astype(np.float32)
            self.weights = {key[8:]: archive[key].copy() for key in archive.files if key.startswith("weight::")}
        if self.mean.shape != (1024,) or self.scale.shape != (1024,) or np.any(self.scale <= 0):
            raise ValueError("Invalid standardizer")
        if not all(np.isfinite(x).all() for x in [self.mean, self.scale, *self.weights.values()]):
            raise ValueError("Nonfinite probe parameters")
        hidden = int(m["hidden"])
        expected = {"0.weight": (hidden or 6, 1024), "0.bias": (hidden or 6,)}
        if hidden:
            expected.update({"3.weight": (6, hidden), "3.bias": (6,)})
        if set(self.weights) != set(expected) or any(self.weights[k].shape != shape for k, shape in expected.items()):
            raise ValueError("Probe weight shapes differ from metadata")

    def predict_proba(self, x, batch_size=8192):
        """NumPy/SciPy CPU consumer, preserving ordered head concatenation."""
        from scipy.special import erf
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 1024:
            raise ValueError("Probe expects N×1024 ordered head features")
        result = np.empty((len(x), 6), dtype=np.float32)
        for start in range(0, len(x), batch_size):
            z = (np.asarray(x[start:start + batch_size], dtype=np.float32) - self.mean) / self.scale
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite input features")
            z = z @ self.weights["0.weight"].T + self.weights["0.bias"]
            if self.metadata["hidden"]:
                z = z * 0.5 * (1 + erf(z / np.float32(math.sqrt(2))))
                z = z @ self.weights["3.weight"].T + self.weights["3.bias"]
            z -= z.max(axis=1, keepdims=True)
            z = np.exp(z)
            result[start:start + len(z)] = z / z.sum(axis=1, keepdims=True)
        return result

    def torch(self, device="cuda:0"):
        import torch
        net = make_net(int(self.metadata["hidden"]), dropout=float(self.metadata["dropout"]))
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights.items()}, strict=True)
        return net.eval().to(device)


def load_probe(path, *, expected_binding=None):
    return Probe(path, expected_binding=expected_binding)


def _training_options(config):
    options = dict(config.get("classifier", {}))
    phase = config.get("phase", config.get("mode", "pilot"))
    defaults = {"device": "cuda:0", "batch_size": 4096, "learning_rate": 1e-3,
                "weight_decay": 1e-3, "dropout": 0.1, "gradient_clip": 5.0,
                "max_epochs": 15 if phase == "pilot" else 45, "min_epochs": 10,
                "patience": 8, "checkpoint_steps": 100, "epoch_rows": None,
                "allow_cpu_smoke": False, "bootstrap_replicates": 2000,
                "profile_weights": dict(DEFAULT_PROFILE_WEIGHTS)}
    defaults.update(options)
    defaults["profile_weights"] = _profile_weights(defaults["profile_weights"])
    if defaults["profile_weights"] is None and not defaults["allow_cpu_smoke"]:
        raise ValueError("Scientific training requires hierarchical profile weights; flat mode is engineering only")
    if phase == "pilot":
        defaults["max_epochs"] = min(defaults["max_epochs"], 15)
    if not 1 <= int(defaults["min_epochs"]) <= int(defaults["max_epochs"]) <= 45:
        raise ValueError("Invalid classifier epoch limits")
    if int(defaults["min_epochs"]) < 10 and not defaults["allow_cpu_smoke"]:
        raise ValueError("The experiment requires at least 10 epochs")
    if defaults["device"].startswith("cpu") and not defaults["allow_cpu_smoke"]:
        raise ValueError("CPU is only allowed for explicitly labelled engineering smoke tests")
    return defaults


def train_candidates(feature_dir, run_dir, config, resume=False):
    """Train all seven fixed candidates sequentially, selecting only on DEV."""
    import torch
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "freeze.json").exists():
        raise ValueError("This run is frozen; use a new run directory for retraining")
    options = _training_options(config)
    device = options["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU classifier training requires CUDA")
    # Avoid TF32-dependent candidate rankings or export discrepancies.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fit = FeatureBank(feature_dir, "fit")
    dev = FeatureBank(feature_dir, "dev", expected_binding=fit.binding())
    if np.intersect1d(fit.source, dev.source).size:
        raise ValueError("FIT/DEV source clusters overlap")
    if np.intersect1d(fit.family, dev.family).size:
        raise ValueError("FIT/DEV source families overlap")
    binding = fit.binding()
    implementation_sha256 = _sha(__file__)
    training_hash = hashlib.sha256(_json({"fit_features": fit.split_identity_hash,
                                          "dev_features": dev.split_identity_hash, "options": options,
                                          "implementation_sha256": implementation_sha256,
                                          "candidates": CANDIDATES}).encode()).hexdigest()
    previous = root / "training.json"
    if previous.exists():
        if not resume:
            raise ValueError("Run already contains training; specify resume or a fresh directory")
        if json.loads(previous.read_text())["training_hash"] != training_hash:
            raise ValueError("Cannot resume different data/config/model/head order")
    sampler = BalancedSampler(fit, profile_weights=options["profile_weights"])
    _atomic_json(previous, {"format_version": PROBE_FORMAT, "training_hash": training_hash,
                           "phase": config.get("phase", config.get("mode", "pilot")),
                           "implementation_sha256": implementation_sha256,
                           "feature_dir": str(Path(feature_dir).resolve()),
                           "feature_manifest_hash": fit.manifest_hash, "options": options,
                           "feature_split_hashes": {"fit": fit.split_identity_hash, "dev": dev.split_identity_hash},
                           "binding": binding, "fit_rows": len(fit), "dev_rows": len(dev),
                           "candidates": [x[0] for x in CANDIDATES], "sampling": sampler.report,
                           "sampler_protocol": sampler.protocol,
                           "selection_rule": ["all precision/recall >=0.9", "structure_equal_macro_f1",
                                              "minimum_precision_recall", "NLL", "candidate_id"]})
    standardizer = root / "standardizer.npz"
    if standardizer.exists() and resume:
        with np.load(standardizer, allow_pickle=False) as a:
            if str(a["training_hash"]) != training_hash:
                raise ValueError("Standardizer belongs to another run")
            mean, scale = a["mean"].copy(), a["scale"].copy()
    else:
        mean, scale = fit_standardizer(fit)
        with standardizer.with_suffix(".tmp").open("wb") as f:
            np.savez(f, mean=mean, scale=scale, training_hash=np.array(training_hash))
        os.replace(standardizer.with_suffix(".tmp"), standardizer)
    if mean.shape != (1024,) or scale.shape != (1024,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid FIT standardizer")
    mean_t = torch.as_tensor(mean, device=device)
    scale_t = torch.as_tensor(scale, device=device)
    if options.get("cache_features_on_gpu", False):
        _cache_features_on_device(fit, device)
        _cache_features_on_device(dev, device)
    validation = root / "resume-validation.json"
    if not validation.exists():
        validate_resume_and_export(root, device=device)
    elif json.loads(validation.read_text()).get("device") != device:
        raise ValueError("Recovery validation used a different training device")
    results = []
    for candidate_id, hidden, seed in CANDIDATES:
        candidate_dir = root / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        completed = candidate_dir / "complete.json"
        if resume and completed.exists():
            record = json.loads(completed.read_text())
            if record["training_hash"] != training_hash or _sha(candidate_dir / "best.npz") != record["probe_sha256"]:
                raise ValueError("Completed candidate is inconsistent")
            results.append(record)
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        net = make_net(hidden, dropout=options["dropout"]).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
        rng = np.random.default_rng(seed)
        history = []
        best_score = None
        best_state = None
        best_result = None
        best_epoch = 0
        stale = 0
        epoch = 0
        batch_cursor = 0
        optimizer_step = 0
        sampler_before = None
        profile_draw_counts = {profile: 0 for profile in sampler.profile_labels}
        epoch_profile_draw_counts = dict(profile_draw_counts)
        checkpoint_path = candidate_dir / "last.pt"
        start_time = time.monotonic()
        elapsed_before = 0.0
        if resume and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state["training_hash"] != training_hash or state["candidate_id"] != candidate_id:
                raise ValueError("Checkpoint binding mismatch")
            if state.get("standardizer_hash") != _sha(standardizer):
                raise ValueError("Checkpoint standardizer was modified")
            net.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            for value in optimizer.state.values():
                for key, item in value.items():
                    if torch.is_tensor(item):
                        value[key] = item.to(device)
            epoch, batch_cursor = state["epoch"], state["batch_cursor"]
            optimizer_step, stale = state["optimizer_step"], state["stale"]
            best_score, best_state = state["best_score"], state["best_state"]
            best_result, best_epoch = state["best_result"], state["best_epoch"]
            history, elapsed_before = state["history"], state["elapsed_seconds"]
            sampler_before = state["sampler_before"]
            rng.bit_generator.state = state["sampler_after"]
            profile_draw_counts = state["profile_draw_counts"]
            epoch_profile_draw_counts = state["epoch_profile_draw_counts"]
            _restore_rng(state["rng"])

        def checkpoint():
            _atomic_torch_save(checkpoint_path, {
                "training_hash": training_hash, "candidate_id": candidate_id,
                "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "optimizer": optimizer.state_dict(), "epoch": epoch,
                "batch_cursor": batch_cursor, "optimizer_step": optimizer_step,
                "sampler_before": sampler_before, "sampler_after": copy.deepcopy(rng.bit_generator.state),
                "rng": _torch_rng(), "best_score": best_score, "best_state": best_state,
                "best_result": best_result, "best_epoch": best_epoch, "stale": stale,
                "history": history, "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
                "standardizer_hash": _sha(standardizer),
                "profile_draw_counts": dict(profile_draw_counts),
                "epoch_profile_draw_counts": dict(epoch_profile_draw_counts),
            })

        count = int(options["epoch_rows"] or len(fit))
        batch_size = int(options["batch_size"])
        while epoch < int(options["max_epochs"]):
            if epoch >= options["min_epochs"] and stale >= options["patience"]:
                break
            if batch_cursor and sampler_before is not None:
                replay = np.random.default_rng()
                replay.bit_generator.state = sampler_before
                drawn = sampler.draw(replay, count)
            else:
                sampler_before = copy.deepcopy(rng.bit_generator.state)
                drawn = sampler.draw(rng, count)
            net.train()
            loss_sum, seen = 0.0, 0
            for batch in range(batch_cursor, math.ceil(count / batch_size)):
                indices = drawn[batch * batch_size:(batch + 1) * batch_size]
                x = _feature_batch(fit, indices, device)
                y = torch.from_numpy(fit.y[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = net((x - mean_t) / scale_t)
                loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite classifier loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), options["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer_step += 1
                batch_cursor = batch + 1
                for profile, rows in sampler.count_profiles(indices).items():
                    profile_draw_counts[profile] += rows
                    epoch_profile_draw_counts[profile] += rows
                loss_sum += float(loss.detach()) * len(indices)
                seen += len(indices)
                if options["checkpoint_steps"] and optimizer_step % int(options["checkpoint_steps"]) == 0:
                    checkpoint()
            p = _predict(net, dev, mean, scale, device=device)
            result = grouped_metrics(dev, p)
            score = selection_key(result, candidate_id)
            improved = best_score is None or score < tuple(best_score)
            if improved:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                best_result, best_epoch, stale = result, epoch + 1, 0
            else:
                stale += 1
            history.append({"epoch": epoch + 1, "train_loss_for_processed_batches": loss_sum / max(seen, 1),
                            "processed_rows_this_process": seen, "dev": result,
                            "improved": improved, "optimizer_step": optimizer_step,
                            "sampled_rows_by_profile": dict(epoch_profile_draw_counts),
                            "sampled_profile_proportions": {profile: rows / max(sum(epoch_profile_draw_counts.values()), 1)
                                                            for profile, rows in epoch_profile_draw_counts.items()},
                            "cumulative_sampled_rows_by_profile": dict(profile_draw_counts)})
            print(_json({"candidate": candidate_id, "epoch": epoch + 1,
                         "dev_macro_f1": result["overall"]["macro_f1"],
                         "dev_structure_macro_f1": result["structure_equal_macro_f1"],
                         "minimum_precision_recall": result["minimum_precision_recall"]}), flush=True)
            epoch += 1
            batch_cursor = 0
            sampler_before = None
            epoch_profile_draw_counts = {profile: 0 for profile in sampler.profile_labels}
            checkpoint()
            _atomic_json(candidate_dir / "history.json", history)
        if best_state is None:
            raise RuntimeError("No trained classifier state")
        _atomic_json(candidate_dir / "history.json", history)
        net.load_state_dict(best_state)
        net.eval()
        metadata = {**binding, "format_version": PROBE_FORMAT, "candidate_id": candidate_id,
                    "hidden": hidden, "seed": seed, "dropout": options["dropout"],
                    "activation": "gelu_exact", "standardizer": "FIT_family_weighted_population_std",
                    "training_hash": training_hash, "best_epoch": best_epoch,
                    "training_device": device, "engineering_smoke_only": bool(options["allow_cpu_smoke"])}
        export_path = candidate_dir / "best.npz"
        _export(export_path, net, mean, scale, metadata)
        probe = load_probe(export_path, expected_binding=binding)
        check_indices = np.arange(min(len(dev), 257))
        check_x = dev.read(check_indices)
        with torch.inference_mode():
            reference = net((torch.from_numpy(check_x).to(device) - mean_t) / scale_t).softmax(1).cpu().numpy()
        exported = probe.predict_proba(check_x)
        error = float(np.max(np.abs(reference - exported)))
        if error > 3e-5:
            raise ValueError(f"NPZ consumer disagrees with PyTorch by {error}")
        record = {"candidate_id": candidate_id, "training_hash": training_hash,
                  "best_epoch": best_epoch, "epochs": epoch, "dev": best_result,
                  "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
                  "probe_sha256": _sha(export_path), "export_max_absolute_error": error,
                  "history_sha256": _sha(candidate_dir / "history.json"),
                  "sampled_rows_by_profile": dict(profile_draw_counts),
                  "sampled_profile_proportions": {profile: rows / max(sum(profile_draw_counts.values()), 1)
                                                  for profile, rows in profile_draw_counts.items()},
                  "sampler_protocol": sampler.protocol}
        _atomic_json(completed, record)
        results.append(record)
        del net, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    winner = min(results, key=lambda x: selection_key(x["dev"], x["candidate_id"]))
    summary = {"training_hash": training_hash, "candidate_id": winner["candidate_id"],
               "selection_split": "dev", "candidate_results": results,
               "dev_floor_met": winner["dev"]["meets_dev_floor"],
               "selected_probe": f"candidates/{winner['candidate_id']}/best.npz"}
    _atomic_json(root / "selection.json", summary)
    return summary


def freeze(run_dir):
    """Commit the DEV decision before permitting any final TEST evaluation."""
    root = Path(run_dir)
    training_path = root / "training.json"
    training = json.loads(training_path.read_text())
    external_pointer = root / "external-protocol.json"
    if training.get("phase") == "formal" and not external_pointer.is_file():
        raise ValueError("Formal freezing requires the predeclared external protocol")
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    source = root / selection["selected_probe"]
    probe = load_probe(source)
    if probe.metadata.get("engineering_smoke_only"):
        raise ValueError("An engineering smoke test cannot become an experiment freeze")
    target = root / "best.npz"
    freeze_path = root / "freeze.json"
    if freeze_path.exists():
        record = json.loads(freeze_path.read_text())
        if record["selection_sha256"] != _sha(selection_path) or record["probe_sha256"] != _sha(target):
            raise ValueError("Frozen artifact was modified")
        if external_pointer.is_file() and record.get("external_protocol_pointer_sha256") != _sha(external_pointer):
            raise ValueError("External protocol changed after classifier freeze")
        return record
    temp = target.with_suffix(".tmp")
    temp.write_bytes(source.read_bytes())
    os.replace(temp, target)
    record = {"format_version": PROBE_FORMAT, "frozen_at_unix": time.time(),
              "selection_sha256": _sha(selection_path), "training_sha256": _sha(training_path),
              "probe_sha256": _sha(target), "candidate_id": selection["candidate_id"],
              "selection_split": "dev", "metadata": probe.metadata}
    if external_pointer.is_file():
        external = json.loads(external_pointer.read_text())
        if not external.get("protocol_sha256") or not external.get("history_index_sha256"):
            raise ValueError("Incomplete external-protocol binding")
        record.update(external_protocol_pointer_sha256=_sha(external_pointer),
                      external_protocol_sha256=external["protocol_sha256"],
                      external_history_index_sha256=external["history_index_sha256"])
    _atomic_json(freeze_path, record)
    return record


def evaluate(feature_dir, run_dir, split="test", *, bootstrap_replicates=2000, device="cuda:0", output_name=None):
    root = Path(run_dir)
    frozen = json.loads((root / "freeze.json").read_text())
    if frozen["probe_sha256"] != _sha(root / "best.npz") or frozen["selection_sha256"] != _sha(root / "selection.json"):
        raise ValueError("Frozen classifier/selection changed after freezing")
    probe = load_probe(root / "best.npz")
    bank = FeatureBank(feature_dir, split, expected_binding=probe.metadata)
    training = json.loads((root / "training.json").read_text())
    if split not in ("dev", "fit"):
        training_dir = training["feature_dir"]
        # Refuse a final test that reuses FIT/DEV source clusters.
        for previous_split in ("fit", "dev"):
            previous = FeatureBank(training_dir, previous_split)
            if np.intersect1d(bank.source, previous.source).size:
                raise ValueError(f"{split} source clusters overlap {previous_split}")
            del previous
    net = probe.torch(device)
    probabilities = _predict(net, bank, probe.mean, probe.scale, device=device)
    result = grouped_metrics(bank, probabilities)
    result.update({"split": split, "probe_sha256": frozen["probe_sha256"],
                   "feature_manifest_sha256": bank.manifest_hash,
                   "format": "native_harmony_body_supervision_only"})
    if split not in ("fit", "dev"):
        result["confidence_intervals"] = bootstrap_clusters(bank, probabilities, replicates=bootstrap_replicates)
    output = root / "evaluations"
    output.mkdir(exist_ok=True)
    name = str(output_name or split)
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise ValueError("Evaluation output name must be a simple identifier")
    _atomic_json(output / f"{name}.json", result)
    np.save(output / f"{name}-probabilities.npy", probabilities, allow_pickle=False)
    return result


def validate_resume_and_export(run_dir, *, device="cuda:0"):
    """One real dropout/AdamW update resumed from disk, plus CPU NPZ parity.

    This engineering check has independent random inputs and does not select
    heads/candidates or expose final TEST data.
    """
    import torch
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(93017)
    random.seed(93017)
    np.random.seed(93017)
    net = make_net(128).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.randn(31, 1024, device=device)
    y = torch.arange(31, device=device) % 6
    net.train()
    loss = torch.nn.functional.cross_entropy(net(x), y)
    loss.backward()
    optimizer.step()
    state = {"model": copy.deepcopy(net.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": _torch_rng()}
    checkpoint = root / "resume-validation.pt"
    _atomic_torch_save(checkpoint, state)

    def update(model, opt):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        return float(loss.detach())

    uninterrupted_loss = update(net, optimizer)
    reference = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    recovered = make_net(128).to(device)
    recovered.load_state_dict(loaded["model"])
    recovered_optimizer = torch.optim.AdamW(recovered.parameters(), lr=1e-3, weight_decay=1e-3)
    recovered_optimizer.load_state_dict(loaded["optimizer"])
    _restore_rng(loaded["rng"])
    resumed_loss = update(recovered, recovered_optimizer)
    error = max(float((reference[k] - v.detach().cpu()).abs().max()) for k, v in recovered.state_dict().items())
    if error > 1e-7 or abs(uninterrupted_loss - resumed_loss) > 1e-7:
        raise ValueError(f"Checkpoint recovery mismatch: weights={error}, losses={uninterrupted_loss}/{resumed_loss}")
    export = root / "resume-validation.npz"
    metadata = {"format_version": PROBE_FORMAT, "roles": list(ROLES), "hidden": 128,
                "heads": list(range(16)), "feature_position": "pre-o_proj", "feature_dim": 1024,
                "model_binding": {"fixture": "independent engineering validation"}, "dropout": .1,
                "engineering_smoke_only": True}
    mean = np.zeros(1024, dtype=np.float32)
    scale = np.ones(1024, dtype=np.float32)
    _export(export, recovered.eval(), mean, scale, metadata)
    p = load_probe(export).predict_proba(x.detach().cpu().numpy())
    with torch.inference_mode():
        expected = recovered(x).softmax(1).cpu().numpy()
    export_error = float(np.max(np.abs(p - expected)))
    if export_error > 3e-5:
        raise ValueError(f"Checkpoint export/consumer mismatch: {export_error}")
    record = {"device": device, "weight_max_absolute_error": error,
              "loss_absolute_error": abs(uninterrupted_loss - resumed_loss),
              "numpy_export_max_absolute_error": export_error,
              "checkpoint_sha256": _sha(checkpoint), "passed": True}
    _atomic_json(root / "resume-validation.json", record)
    checkpoint.unlink()
    export.unlink()
    return record
