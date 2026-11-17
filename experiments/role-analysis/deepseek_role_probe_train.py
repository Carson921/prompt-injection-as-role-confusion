"""Train the archived mixed-domain protocol on newly extracted DeepSeek features.

Only the readout is trained. Ordered head features and native role labels must
already be committed by the extractor. Dimensions and role names come from the
feature manifest; V4's capture point is the input to ``o_a_proj``.

Feature contract: manifest.json contains roles, feature_dim, head_dim, ordered
heads, model_binding, feature_position, data_hash, config_hash,
selection_sha256, pilot_fit_sha256, signature, split_sources, and splits.
Each split entry {path, rows, sha256} names a directory containing complete.json
(rows, signature, files={filename: sha256}) and x/y/domain/source/family.npy.
The shard signature is {**manifest.signature, split, input_sha256}. ``source``
identifies the article cluster or AgentDojo task; ``family`` identifies a unique
token position. Optional groups.npy contains per-row JSON diagnostic attributes.

CLI: train --features FEATURES --output OUTPUT [--config CONFIG] [--resume]
     evaluate --features FEATURES --output OUTPUT [--split test]
Training freezes OUTPUT/checkpoint.npz before its default, one-time TEST pass.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

FEATURE_FORMAT = "deepseek-role-features-v1"
PROBE_FORMAT = "deepseek-role-probe-v1"
CAPTURE = "attention-head-output-before-o_a_proj"
DOMAINS = ("c4_dolma3", "agentdojo")
SPLITS = ("fit", "dev", "test")
CANDIDATES = (("linear-seed17", 0, 17),
              ("mlp128-seed17", 128, 17), ("mlp128-seed29", 128, 29), ("mlp128-seed43", 128, 43),
              ("mlp256-seed17", 256, 17), ("mlp256-seed29", 256, 29), ("mlp256-seed43", 256, 43))
BINDING_KEYS = ("roles", "feature_dim", "head_dim", "heads", "feature_position", "model_binding",
                "selection_sha256", "pilot_fit_sha256", "data_hash", "config_hash")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/deepseek-v4-flash-0731-role-probe-20261115"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def safe_path(root, value):
    result = (Path(root) / value).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Feature path escapes its root: {value}")
    return result


class FeatureBank:
    """Hash-verified, memory-mapped features; no other split's arrays are opened."""

    def __init__(self, feature_dir, split, *, expected_binding=None):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.root, self.split = Path(feature_dir), split
        self.manifest = m = read_json(self.root / "manifest.json")
        if m.get("format_version") != FEATURE_FORMAT:
            raise ValueError("Unsupported DeepSeek feature format")
        roles = m.get("roles", [])
        if len(roles) < 2 or len(set(roles)) != len(roles) or not all(isinstance(r, str) and r for r in roles):
            raise ValueError("Role names must be distinct, nonempty strings")
        self.roles = tuple(roles)
        self.feature_dim = int(m.get("feature_dim", 0))
        heads = m.get("heads", [])
        if not heads or len({canonical(h) for h in heads}) != len(heads):
            raise ValueError("Ordered heads must be nonempty and distinct")
        if int(m.get("head_dim", 0)) * len(heads) != self.feature_dim or self.feature_dim <= 0:
            raise ValueError("feature_dim must equal ordered head count times head_dim")
        if m.get("feature_position") != CAPTURE:
            raise ValueError(f"Expected V4 capture point {CAPTURE}")
        for key in BINDING_KEYS:
            if not m.get(key):
                raise ValueError(f"Missing feature binding: {key}")
        if expected_binding:
            for key in BINDING_KEYS:
                if m[key] != expected_binding.get(key):
                    raise ValueError(f"Feature/probe binding mismatch: {key}")
        signature = m.get("signature", {})
        for key, value in (("model_binding", m["model_binding"]), ("heads", heads),
                           ("capture", CAPTURE), ("config_sha256", m["config_hash"]),
                           ("selection_sha256", m["selection_sha256"]), ("pilot_fit_sha256", m["pilot_fit_sha256"])):
            if signature.get(key) != value:
                raise ValueError(f"Feature signature differs from manifest: {key}")
        input_sha = m.get("split_sources", {}).get(split)
        if not input_sha:
            raise ValueError(f"Missing split input hash: {split}")
        expected_signature = {**signature, "split": split, "input_sha256": input_sha}
        entries = m.get("splits", {}).get(split)
        if not entries:
            raise ValueError(f"Missing nonempty feature split: {split}")
        if "progress" in m and not m["progress"].get(split, {}).get("complete"):
            raise ValueError(f"Feature split is incomplete: {split}")
        arrays = {key: [] for key in ("y", "domain", "source", "family", "groups")}
        self.shards, offsets = [], [0]
        for entry in entries:
            base = safe_path(self.root, entry["path"])
            complete = base / "complete.json"
            if not entry.get("sha256") or sha(complete) != entry["sha256"]:
                raise ValueError(f"Invalid shard commit hash: {base}")
            record = read_json(complete)
            n = int(entry["rows"])
            if n <= 0 or record.get("rows") != n or record.get("signature") != expected_signature:
                raise ValueError(f"Shard rows/signature differ from manifest: {base}")
            hashes = record.get("files", {})
            if not isinstance(hashes, dict) or not hashes:
                raise ValueError(f"Missing per-file shard hashes: {base}")
            for name, digest in hashes.items():
                if sha(safe_path(base, name)) != digest:
                    raise ValueError(f"Invalid feature file hash: {base / name}")
            for key in ("x", *arrays):
                path = base / f"{key}.npy"
                if not path.is_file() and key == "groups":
                    arrays[key].append(np.full(n, "{}"))
                    continue
                if path.name not in hashes:
                    raise ValueError(f"Array is not covered by shard commit: {path}")
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if key == "x":
                    if value.dtype != np.float16 or value.shape != (n, self.feature_dim):
                        raise ValueError(f"Incorrect FP16 feature shape: {path}")
                    self.shards.append(value)
                else:
                    if value.shape != (n,):
                        raise ValueError(f"Incorrect metadata shape: {path}")
                    if key == "y" and value.dtype.kind not in "iu":
                        raise ValueError("Role IDs must have an integer dtype")
                    if key != "y" and value.dtype.kind not in "US":
                        raise ValueError(f"String metadata must have a fixed-width string dtype: {path}")
                    arrays[key].append(np.asarray(value))
            offsets.append(offsets[-1] + n)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        for key, pieces in arrays.items():
            setattr(self, key, np.concatenate(pieces))
        self.y = self.y.astype(np.int64)
        if np.any((self.y < 0) | (self.y >= len(self.roles))):
            raise ValueError("Unknown role ID")
        if set(self.domain) != set(DOMAINS):
            raise ValueError("Every split must contain exactly c4_dolma3 and agentdojo")
        if any(np.any(getattr(self, key) == "") for key in ("source", "family")):
            raise ValueError("Cluster/source and unique position IDs cannot be empty")
        if len(np.unique(self.family)) != len(self):
            raise ValueError("Repeated unique position/family IDs are not allowed in the mixed protocol")
        self.group_objects = [json.loads(str(group)) for group in self.groups]
        if not all(isinstance(group, dict) for group in self.group_objects):
            raise ValueError("Diagnostic groups must be JSON objects")
        self.manifest_hash = sha(self.root / "manifest.json")
        self.split_identity_hash = object_sha({"binding": self.binding(), "signature": expected_signature, "shards": entries})
        self.device_features = None

    def __len__(self):
        return int(self.offsets[-1])

    def binding(self):
        return {key: self.manifest[key] for key in BINDING_KEYS}

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Feature row indices must be an in-range vector")
        result = np.empty((len(indices), self.feature_dim), dtype=np.float32)
        shard_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard in np.unique(shard_ids):
            locations = np.flatnonzero(shard_ids == shard)
            result[locations] = self.shards[shard][indices[locations] - self.offsets[shard]]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen-model features")
        return result


def assert_disjoint(first, second):
    for key in ("source", "family"):
        if np.intersect1d(getattr(first, key), getattr(second, key)).size:
            raise ValueError(f"{first.split}/{second.split} {key} clusters overlap")


class BalancedSampler:
    """Uniform domain, present role, unique position, with replacement."""

    def __init__(self, bank):
        self.bank = bank
        self.cells = {domain: [np.flatnonzero((bank.domain == domain) & (bank.y == role))
                              for role in np.unique(bank.y[bank.domain == domain])] for domain in DOMAINS}
        if any(not cells for cells in self.cells.values()):
            raise ValueError("Both domains must contain positions")
        self.protocol = {"domain_weights": {domain: .5 for domain in DOMAINS}, "replacement": True,
                         "role_rule": "uniform among roles present in the selected domain",
                         "position_rule": "uniform among unique positions in the selected domain/role",
                         "cells": {domain: [{"role": bank.roles[int(bank.y[indices[0]])], "positions": len(indices)}
                                            for indices in cells] for domain, cells in self.cells.items()}}

    def draw(self, rng, count):
        result = np.empty(count, dtype=np.int64)
        domains = rng.integers(len(DOMAINS), size=count)
        for d, domain in enumerate(DOMAINS):
            locations = np.flatnonzero(domains == d)
            cells = self.cells[domain]
            roles = rng.integers(len(cells), size=len(locations))
            for r, indices in enumerate(cells):
                target = locations[roles == r]
                result[target] = indices[rng.integers(len(indices), size=len(target))]
        return result


def fit_standardizer(bank, chunk_size=8192):
    """Unweighted population mean/std across unique FIT positions only."""
    if bank.split != "fit":
        raise ValueError("Standardization may only be fitted on FIT")
    mean = np.zeros(bank.feature_dim, dtype=np.float64)
    m2, total = np.zeros_like(mean), 0
    for start in range(0, len(bank), chunk_size):
        values = bank.read(np.arange(start, min(start + chunk_size, len(bank)))).astype(np.float64)
        mass = len(values)
        chunk_mean = values.mean(0)
        delta = chunk_mean - mean
        combined = total + mass
        m2 += np.sum((values - chunk_mean) ** 2, axis=0) + delta ** 2 * total * mass / combined
        mean += delta * mass / combined
        total = combined
    scale = np.sqrt(np.maximum(m2 / total, 0))
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def make_net(feature_dim, n_roles, hidden, *, dropout=.1):
    from torch import nn
    if hidden:
        return nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_roles))
    return nn.Sequential(nn.Linear(feature_dim, n_roles))


def confusion_metrics(matrix, roles, nll=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted, correct = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    precision = np.divide(correct, predicted, out=np.zeros(len(roles), dtype=float), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(len(roles), dtype=float), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(len(roles), dtype=float), where=support + predicted > 0)
    result = {"tokens": int(matrix.sum()), "accuracy": float(correct.sum() / max(matrix.sum(), 1)),
              "macro_f1": float(f1[support > 0].mean()) if np.any(support > 0) else None,
              "confusion": matrix.tolist(), "roles": {}}
    if nll is not None:
        result["nll"] = float(nll)
    for i, role in enumerate(roles):
        negatives = int(matrix.sum() - support[i])
        result["roles"][role] = {"support": int(support[i]), "predicted": int(predicted[i]),
                                 "precision": float(precision[i]) if predicted[i] else None,
                                 "recall": float(recall[i]) if support[i] else None,
                                 "f1": float(f1[i]) if support[i] else None,
                                 "false_positives": int(predicted[i] - correct[i]),
                                 "false_positive_rate": float((predicted[i] - correct[i]) / negatives) if negatives else None}
    return result


def metrics(probabilities, labels, roles):
    p, y = np.asarray(probabilities), np.asarray(labels, dtype=np.int64)
    n_roles = len(roles)
    if p.shape != (len(y), n_roles) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=2e-5):
        raise ValueError("Invalid role probabilities")
    confusion = np.bincount(y * n_roles + p.argmax(1), minlength=n_roles ** 2).reshape(n_roles, n_roles)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-30, 1)).mean()
    return confusion_metrics(confusion, roles, nll)


def grouped_metrics(bank, probabilities):
    result = {"overall": metrics(probabilities, bank.y, bank.roles), "by_domain": {}}
    for domain in DOMAINS:
        mask = bank.domain == domain
        result["by_domain"][domain] = metrics(probabilities[mask], bank.y[mask], bank.roles)
    result["domain_equal_macro_f1"] = float(np.mean([value["macro_f1"] for value in result["by_domain"].values()]))
    values = [item.get(metric) or 0.0 for item in result["overall"]["roles"].values() for metric in ("precision", "recall")]
    result["minimum_precision_recall"] = float(min(values))
    result["meets_dev_floor"] = bool(min(values) >= .9)
    keys = sorted(set().union(*(group.keys() for group in bank.group_objects)))
    result["by_attribute"] = {"domain": result["by_domain"]}
    for key in keys:
        if key == "domain":
            continue
        attributes = np.asarray([str(group.get(key, "N/A")) for group in bank.group_objects])
        result["by_attribute"][key] = {str(value): metrics(probabilities[attributes == value], bank.y[attributes == value], bank.roles)
                                          for value in np.unique(attributes)}
    return result


def selection_key(result, candidate_id):
    return (not result["meets_dev_floor"], -result["domain_equal_macro_f1"],
            -result["minimum_precision_recall"], result["overall"]["nll"], candidate_id)


def bootstrap_clusters(bank, probabilities, *, replicates=2000, seed=20261115, mask=None):
    mask = np.ones(len(bank), dtype=bool) if mask is None else mask
    sources, inv = np.unique(bank.source[mask], return_inverse=True)
    count, k = len(sources), len(bank.roles)
    if count < 2 or replicates < 1:
        return {"unit": "source_cluster", "clusters": count, "replicates": 0, "reason": "Insufficient clusters or disabled"}
    p, y = probabilities[mask], bank.y[mask]
    confusion = np.zeros((count, k, k), dtype=np.int64)
    np.add.at(confusion, (inv, y, p.argmax(1)), 1)
    loss_sum = np.bincount(inv, weights=-np.log(np.clip(p[np.arange(len(y)), y], 1e-30, 1)), minlength=count)
    totals = confusion.sum((1, 2))
    samples = {"accuracy": [], "macro_f1": [], "nll": []}
    for role in bank.roles:
        for metric in ("precision", "recall", "f1"):
            samples[f"{role}.{metric}"] = []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        draw = rng.integers(count, size=count)
        result = confusion_metrics(confusion[draw].sum(0), bank.roles)
        for key in ("accuracy", "macro_f1"):
            samples[key].append(result[key])
        samples["nll"].append(float(loss_sum[draw].sum() / totals[draw].sum()))
        for role in bank.roles:
            for metric in ("precision", "recall", "f1"):
                value = result["roles"][role][metric]
                if value is not None:
                    samples[f"{role}.{metric}"].append(value)
    return {"unit": "source_cluster", "clusters": count, "replicates": replicates, "seed": seed,
            "method": "percentile", "confidence": .95,
            "intervals": {key: {"lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975)),
                                "valid_replicates": len(values)} if values else None for key, values in samples.items()}}


def feature_batch(bank, indices, device):
    import torch
    if bank.device_features is not None:
        return bank.device_features[torch.as_tensor(indices, device=device)]
    return torch.from_numpy(bank.read(indices)).to(device)


def cache_on_device(bank, device, chunk_size=8192):
    import torch
    if not device.startswith("cuda"):
        return
    bank.device_features = torch.empty((len(bank), bank.feature_dim), dtype=torch.float32, device=device)
    for start in range(0, len(bank), chunk_size):
        end = min(start + chunk_size, len(bank))
        bank.device_features[start:end] = torch.from_numpy(bank.read(np.arange(start, end))).to(device)


def predict(net, bank, mean, scale, *, device, batch_size=8192):
    import torch
    net.eval()
    result = np.empty((len(bank), len(bank.roles)), dtype=np.float32)
    mean_t, scale_t = torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device)
    with torch.inference_mode():
        for start in range(0, len(bank), batch_size):
            end = min(start + batch_size, len(bank))
            x = feature_batch(bank, np.arange(start, end), device)
            result[start:end] = net((x - mean_t) / scale_t).softmax(1).cpu().numpy()
    return result


def export_probe(path, net, mean, scale, metadata):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    arrays = {f"weight::{key}": value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, mean=mean, scale=scale, metadata=np.asarray(canonical(metadata)), **arrays)
    os.replace(temporary, path)


class Probe:
    """Portable NPZ consumer; dimensions and role ordering are checkpoint-bound."""

    def __init__(self, path, *, expected_binding=None):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = m = json.loads(str(archive["metadata"]))
            if m.get("format_version") != PROBE_FORMAT or m.get("feature_position") != CAPTURE:
                raise ValueError("Unsupported DeepSeek probe format or capture point")
            if expected_binding:
                for key in BINDING_KEYS:
                    if m.get(key) != expected_binding.get(key):
                        raise ValueError(f"Probe binding mismatch: {key}")
            self.mean = archive["mean"].astype(np.float32)
            self.scale = archive["scale"].astype(np.float32)
            self.weights = {key[8:]: archive[key].copy() for key in archive.files if key.startswith("weight::")}
        self.feature_dim, self.roles = int(m["feature_dim"]), tuple(m["roles"])
        hidden, k = int(m["hidden"]), len(self.roles)
        expected = {"0.weight": (hidden or k, self.feature_dim), "0.bias": (hidden or k,)}
        if hidden:
            expected.update({"3.weight": (k, hidden), "3.bias": (k,)})
        if set(expected) != set(self.weights) or any(self.weights[key].shape != shape for key, shape in expected.items()):
            raise ValueError("Probe parameter shapes differ from metadata")
        if self.mean.shape != (self.feature_dim,) or self.scale.shape != (self.feature_dim,) or np.any(self.scale <= 0):
            raise ValueError("Invalid FIT standardizer")
        if not all(np.isfinite(value).all() for value in (self.mean, self.scale, *self.weights.values())):
            raise ValueError("Nonfinite probe parameters")

    def torch(self, device="cpu"):
        import torch
        net = make_net(self.feature_dim, len(self.roles), int(self.metadata["hidden"]), dropout=float(self.metadata["dropout"]))
        net.load_state_dict({key: torch.from_numpy(value) for key, value in self.weights.items()}, strict=True)
        return net.eval().to(device)

    def predict_proba(self, x, batch_size=8192):
        try:
            from scipy.special import erf
        except ImportError:
            # The training environment already has PyTorch. Its CPU erf keeps
            # exact GELU available without an additional SciPy dependency.
            import torch

            def erf(value):
                return torch.erf(torch.from_numpy(value)).numpy()
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"Expected N x {self.feature_dim} ordered head features")
        result = np.empty((len(x), len(self.roles)), dtype=np.float32)
        for start in range(0, len(x), batch_size):
            z = (np.asarray(x[start:start + batch_size], dtype=np.float32) - self.mean) / self.scale
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite input features")
            z = z @ self.weights["0.weight"].T + self.weights["0.bias"]
            if self.metadata["hidden"]:
                z = .5 * z * (1 + erf(z / np.float32(math.sqrt(2))))
                z = z @ self.weights["3.weight"].T + self.weights["3.bias"]
            z -= z.max(1, keepdims=True)
            z = np.exp(z)
            result[start:start + len(z)] = z / z.sum(1, keepdims=True)
        return result


def training_options(config):
    options = {"device": "cuda:0", "batch_size": 4096, "learning_rate": .001, "weight_decay": .001,
               "dropout": .1, "gradient_clip": 5., "max_epochs": 45, "min_epochs": 10, "patience": 8,
               "checkpoint_steps": 100, "epoch_rows": None, "bootstrap_replicates": 2000,
               "cache_features_on_gpu": True, "allow_cpu_smoke": False}
    supplied = config.get("classifier", {})
    unknown = set(supplied) - set(options) - {"profile_weights", "domain_weights"}
    if unknown:
        raise ValueError(f"Unknown classifier options: {sorted(unknown)}")
    options.update({key: value for key, value in supplied.items() if key in options})
    for key in ("profile_weights", "domain_weights"):
        if key in supplied and supplied[key] != {domain: .5 for domain in DOMAINS}:
            raise ValueError("The mixed protocol requires domain weights 0.5 / 0.5")
    if not 1 <= int(options["min_epochs"]) <= int(options["max_epochs"]) <= 45:
        raise ValueError("Invalid classifier epoch limits")
    if not options["allow_cpu_smoke"] and int(options["min_epochs"]) < 10:
        raise ValueError("Scientific training requires at least 10 epochs")
    if options["device"].startswith("cpu") and not options["allow_cpu_smoke"]:
        raise ValueError("CPU is allowed only for explicitly labelled engineering smoke tests")
    for key in ("batch_size", "patience"):
        if int(options[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if options["epoch_rows"] is not None and int(options["epoch_rows"]) < 1:
        raise ValueError("epoch_rows must be positive")
    if not 0 <= float(options["dropout"]) < 1:
        raise ValueError("Dropout must be in [0, 1)")
    return options


def save_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_candidates(feature_dir, run_dir, config=None, resume=False):
    """Train seven random initializations, reading FIT/DEV arrays only."""
    import torch
    config = config or {}
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "freeze.json").exists():
        raise ValueError("This run is frozen; retraining requires a new output directory")
    options = training_options(config)
    device = options["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA classifier training requires an available GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fit = FeatureBank(feature_dir, "fit")
    dev = FeatureBank(feature_dir, "dev", expected_binding=fit.binding())
    assert_disjoint(fit, dev)
    if set(fit.y) != set(range(len(fit.roles))):
        raise ValueError("FIT must contain every declared role")
    binding = fit.binding()
    training_hash = object_sha({"fit": fit.split_identity_hash, "dev": dev.split_identity_hash,
                                "options": options, "candidates": CANDIDATES, "implementation_sha256": sha(__file__)})
    training_path = root / "training.json"
    if training_path.exists():
        if not resume:
            raise ValueError("Training already exists; use --resume or a fresh directory")
        if read_json(training_path)["training_hash"] != training_hash:
            raise ValueError("Cannot resume changed data, configuration, code, or head order")
    sampler = BalancedSampler(fit)
    training = {"format_version": PROBE_FORMAT, "training_hash": training_hash, "binding": binding,
                "feature_dir": str(Path(feature_dir).resolve()), "feature_manifest_sha256": fit.manifest_hash,
                "split_hashes": {"fit": fit.split_identity_hash, "dev": dev.split_identity_hash},
                "fit_rows": len(fit), "dev_rows": len(dev), "options": options, "candidates": CANDIDATES,
                "sampler_protocol": sampler.protocol, "standardizer": "Unweighted FIT unique-position population mean/std",
                "selection_rule": ["all overall precision/recall >= 0.9", "domain-equal macro-F1",
                                   "minimum overall precision/recall", "overall NLL", "candidate ID"],
                "test_policy": "Freeze checkpoint before one TEST inference; no TEST selection or retuning",
                "implementation_sha256": sha(__file__), "engineering_smoke_only": bool(options["allow_cpu_smoke"])}
    atomic_json(training_path, training)
    standardizer = root / "standardizer.npz"
    if standardizer.exists() and resume:
        with np.load(standardizer, allow_pickle=False) as archive:
            if str(archive["training_hash"]) != training_hash:
                raise ValueError("Standardizer belongs to a different run")
            mean, scale = archive["mean"].copy(), archive["scale"].copy()
    else:
        mean, scale = fit_standardizer(fit)
        with standardizer.with_suffix(".tmp").open("wb") as stream:
            np.savez(stream, mean=mean, scale=scale, training_hash=np.asarray(training_hash))
        os.replace(standardizer.with_suffix(".tmp"), standardizer)
    if mean.shape != (fit.feature_dim,) or scale.shape != (fit.feature_dim,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid FIT standardizer")
    mean_t, scale_t = torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device)
    if options["cache_features_on_gpu"]:
        cache_on_device(fit, device)
        cache_on_device(dev, device)
    results = []
    for candidate_id, hidden, seed in CANDIDATES:
        folder = root / "candidates" / candidate_id
        folder.mkdir(parents=True, exist_ok=True)
        complete = folder / "complete.json"
        if complete.exists() and resume:
            record = read_json(complete)
            if record["training_hash"] != training_hash or record["probe_sha256"] != sha(folder / "best.npz"):
                raise ValueError("Completed candidate binding or bytes changed")
            results.append(record)
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        net = make_net(fit.feature_dim, len(fit.roles), hidden, dropout=options["dropout"]).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
        rng = np.random.default_rng(seed)
        state = {"epoch": 0, "batch_cursor": 0, "optimizer_step": 0, "drawn": None, "stale": 0,
                 "best_score": None, "best_state": None, "best_result": None, "best_epoch": 0,
                 "history": [], "loss_sum": 0., "seen": 0,
                 "domain_counts": {domain: 0 for domain in DOMAINS}, "elapsed_seconds": 0.}
        checkpoint_path = folder / "last.pt"
        if checkpoint_path.exists() and resume:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved["training_hash"] != training_hash or saved["standardizer_sha256"] != sha(standardizer):
                raise ValueError("Resume checkpoint identity mismatch")
            state = saved["state"]
            net.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            for value in optimizer.state.values():
                for key, item in value.items():
                    if torch.is_tensor(item):
                        value[key] = item.to(device)
            rng.bit_generator.state = saved["sampler_rng"]
            torch.set_rng_state(saved["torch_rng"])
            if device.startswith("cuda"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        elapsed_before, start_time = state["elapsed_seconds"], time.monotonic()

        def checkpoint():
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - start_time
            save_torch(checkpoint_path, {"training_hash": training_hash, "standardizer_sha256": sha(standardizer),
                       "state": state, "model": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                       "optimizer": optimizer.state_dict(), "sampler_rng": copy.deepcopy(rng.bit_generator.state),
                       "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None})

        epoch_rows = int(options["epoch_rows"] or len(fit))
        batch_size = int(options["batch_size"])
        while state["epoch"] < int(options["max_epochs"]):
            if state["epoch"] >= int(options["min_epochs"]) and state["stale"] >= int(options["patience"]):
                break
            if state["drawn"] is None:
                state["drawn"] = sampler.draw(rng, epoch_rows)
            net.train()
            for batch in range(state["batch_cursor"], math.ceil(epoch_rows / batch_size)):
                indices = state["drawn"][batch * batch_size:(batch + 1) * batch_size]
                x = feature_batch(fit, indices, device)
                y = torch.from_numpy(fit.y[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(net((x - mean_t) / scale_t), y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite classifier loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), options["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                state["optimizer_step"] += 1
                state["batch_cursor"] = batch + 1
                state["loss_sum"] += float(loss.detach()) * len(indices)
                state["seen"] += len(indices)
                for domain in DOMAINS:
                    state["domain_counts"][domain] += int(np.sum(fit.domain[indices] == domain))
                if options["checkpoint_steps"] and state["optimizer_step"] % int(options["checkpoint_steps"]) == 0:
                    checkpoint()
            result = grouped_metrics(dev, predict(net, dev, mean, scale, device=device))
            score = selection_key(result, candidate_id)
            improved = state["best_score"] is None or score < tuple(state["best_score"])
            if improved:
                state.update(best_score=score, best_state={key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
                             best_result=result, best_epoch=state["epoch"] + 1, stale=0)
            else:
                state["stale"] += 1
            state["history"].append({"epoch": state["epoch"] + 1, "train_loss": state["loss_sum"] / state["seen"],
                                     "dev": result, "improved": improved, "optimizer_step": state["optimizer_step"],
                                     "cumulative_draws_by_domain": dict(state["domain_counts"])})
            print(canonical({"candidate": candidate_id, "epoch": state["epoch"] + 1,
                             "dev_domain_equal_macro_f1": result["domain_equal_macro_f1"],
                             "dev_minimum_precision_recall": result["minimum_precision_recall"]}), flush=True)
            state.update(epoch=state["epoch"] + 1, batch_cursor=0, drawn=None, loss_sum=0., seen=0)
            checkpoint()
            atomic_json(folder / "history.json", state["history"])
        if state["best_state"] is None:
            raise RuntimeError("No trained classifier state")
        # A process can stop after committing the final epoch checkpoint but
        # before history.json. Rebuild history from the resumable state too.
        atomic_json(folder / "history.json", state["history"])
        net.load_state_dict(state["best_state"])
        metadata = {**binding, "format_version": PROBE_FORMAT, "candidate_id": candidate_id, "hidden": hidden,
                    "seed": seed, "dropout": options["dropout"], "activation": "gelu_exact",
                    "standardizer": "FIT_unique_position_population_std", "training_hash": training_hash,
                    "best_epoch": state["best_epoch"], "training_device": device,
                    "engineering_smoke_only": bool(options["allow_cpu_smoke"])}
        export_probe(folder / "best.npz", net, mean, scale, metadata)
        probe = Probe(folder / "best.npz", expected_binding=binding)
        check_x = dev.read(np.arange(min(len(dev), 257)))
        net.eval()
        with torch.inference_mode():
            reference = net((torch.from_numpy(check_x).to(device) - mean_t) / scale_t).softmax(1).cpu().numpy()
        error = float(np.max(np.abs(probe.predict_proba(check_x) - reference)))
        if error > 3e-5:
            raise ValueError(f"NPZ consumer disagrees with PyTorch by {error}")
        record = {"candidate_id": candidate_id, "training_hash": training_hash, "best_epoch": state["best_epoch"],
                  "epochs": state["epoch"], "dev": state["best_result"], "probe_sha256": sha(folder / "best.npz"),
                  "export_max_absolute_error": error, "elapsed_seconds": state["elapsed_seconds"],
                  "sampled_rows_by_domain": state["domain_counts"], "history_sha256": sha(folder / "history.json")}
        atomic_json(complete, record)
        results.append(record)
        del net, optimizer
    winner = min(results, key=lambda result: selection_key(result["dev"], result["candidate_id"]))
    selection = {"training_hash": training_hash, "candidate_id": winner["candidate_id"], "selection_split": "dev",
                 "candidate_results": results, "dev_floor_met": winner["dev"]["meets_dev_floor"],
                 "selected_probe": f"candidates/{winner['candidate_id']}/best.npz"}
    atomic_json(root / "selection.json", selection)
    return selection


def freeze(run_dir):
    """Commit DEV-only selection and publish the final top-level checkpoint."""
    root = Path(run_dir)
    if (root / "freeze.json").exists():
        return verify_freeze(root)
    training, selection = read_json(root / "training.json"), read_json(root / "selection.json")
    if selection["training_hash"] != training["training_hash"] or selection["selection_split"] != "dev":
        raise ValueError("Selection does not belong to this DEV-only training run")
    source = safe_path(root, selection["selected_probe"])
    probe = Probe(source, expected_binding=training["binding"])
    winner = next(result for result in selection["candidate_results"] if result["candidate_id"] == selection["candidate_id"])
    if sha(source) != winner["probe_sha256"] or probe.metadata["training_hash"] != training["training_hash"]:
        raise ValueError("Selected candidate bytes or training identity changed")
    temporary = root / "checkpoint.npz.tmp"
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, root / "checkpoint.npz")
    record = {"format_version": PROBE_FORMAT, "frozen_at_unix": time.time(), "selection_split": "dev",
              "candidate_id": selection["candidate_id"], "probe_sha256": sha(root / "checkpoint.npz"),
              "training_sha256": sha(root / "training.json"), "selection_sha256": sha(root / "selection.json"),
              "metadata": probe.metadata}
    atomic_json(root / "manifest.json", {"format_version": PROBE_FORMAT, "checkpoint": "checkpoint.npz", **record,
                                         "training": "training.json", "selection": "selection.json",
                                         "metrics": "metrics.json", "feature_manifest_sha256": training["feature_manifest_sha256"]})
    atomic_json(root / "metrics.json", {"selected_candidate": selection["candidate_id"], "selection_split": "dev",
                                        "selected_dev": winner["dev"], "checkpoint_sha256": record["probe_sha256"],
                                        "test": None})
    atomic_json(root / "freeze.json", record)
    return record


def verify_freeze(run_dir):
    root = Path(run_dir)
    record = read_json(root / "freeze.json")
    for key, name in (("probe_sha256", "checkpoint.npz"), ("training_sha256", "training.json"),
                      ("selection_sha256", "selection.json")):
        if record[key] != sha(root / name):
            raise ValueError(f"Frozen artifact changed: {name}")
    return record


def evaluate(feature_dir, run_dir, split="test", *, device=None, bootstrap_replicates=None):
    """Evaluate the frozen checkpoint; completed TEST results are reused verbatim."""
    root = Path(run_dir)
    frozen = verify_freeze(root)
    training = read_json(root / "training.json")
    output = root / "evaluations"
    output.mkdir(exist_ok=True)
    result_path = output / f"{split}.json"
    commit_path = output / f"{split}-complete.json"
    if split == "test" and commit_path.exists():
        commit = read_json(commit_path)
        if commit["probe_sha256"] != frozen["probe_sha256"] or commit["feature_manifest_sha256"] != sha(Path(feature_dir) / "manifest.json"):
            raise ValueError("Completed TEST evaluation belongs to another checkpoint/cache")
        for name, digest in commit["files"].items():
            if sha(output / name) != digest:
                raise ValueError("Completed TEST evaluation was modified")
        return read_json(result_path)
    if split not in SPLITS:
        raise ValueError("Unknown evaluation split")
    started = output / f"{split}-started.json"
    if split == "test" and started.exists():
        raise ValueError("A TEST pass was already started; refusing a second inference")
    probe = Probe(root / "checkpoint.npz", expected_binding=training["binding"])
    bank = FeatureBank(feature_dir, split, expected_binding=probe.metadata)
    # TEST may be extracted only after classifier freeze. Its new shard entries
    # legitimately change the shared manifest; FIT/DEV identities and the full
    # extraction/model/head binding must remain exactly as trained.
    for previous_split in ("fit", "dev"):
        previous = FeatureBank(feature_dir, previous_split, expected_binding=probe.metadata)
        if previous.split_identity_hash != training["split_hashes"][previous_split]:
            raise ValueError("Training split changed after freezing")
        if split == "test":
            assert_disjoint(previous, bank)
        del previous
    device = device or training["options"]["device"]
    net = probe.torch(device)
    if split == "test":
        # Exclusive creation prevents concurrent evaluators from exposing TEST twice.
        with started.open("x") as stream:
            stream.write(canonical({"started_unix": time.time(), "probe_sha256": frozen["probe_sha256"],
                                    "feature_manifest_sha256": bank.manifest_hash}) + "\n")
    probabilities = predict(net, bank, probe.mean, probe.scale, device=device)
    result = grouped_metrics(bank, probabilities)
    result.update(split=split, probe_sha256=frozen["probe_sha256"], feature_manifest_sha256=bank.manifest_hash,
                  training_feature_manifest_sha256=training["feature_manifest_sha256"],
                  label_policy="Native message/channel roles; no archived probe predictions")
    if split == "test":
        replicates = int(training["options"]["bootstrap_replicates"] if bootstrap_replicates is None else bootstrap_replicates)
        result["confidence_intervals"] = bootstrap_clusters(bank, probabilities, replicates=replicates)
        for domain in DOMAINS:
            result["by_domain"][domain]["confidence_intervals"] = bootstrap_clusters(
                bank, probabilities, replicates=replicates, mask=bank.domain == domain)
    probability_path = output / f"{split}-probabilities.npy"
    temporary = probability_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, probabilities, allow_pickle=False)
    os.replace(temporary, probability_path)
    atomic_json(result_path, result)
    if split == "test":
        atomic_json(commit_path, {"probe_sha256": frozen["probe_sha256"], "feature_manifest_sha256": bank.manifest_hash,
                                  "files": {result_path.name: sha(result_path), probability_path.name: sha(probability_path)},
                                  "completed_unix": time.time()})
        summary = read_json(root / "metrics.json")
        summary["test"] = result
        atomic_json(root / "metrics.json", summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--config", type=Path)
    train.add_argument("--device")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no-test", action="store_true", help="Freeze without exposing TEST features")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--features", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    evaluation.add_argument("--split", choices=SPLITS, default="test")
    evaluation.add_argument("--device")
    args = parser.parse_args()
    if args.command == "train":
        config = read_json(args.config) if args.config else {}
        if args.device:
            config.setdefault("classifier", {})["device"] = args.device
        train_candidates(args.features, args.output, config, resume=args.resume)
        freeze(args.output)
        if not args.no_test:
            evaluate(args.features, args.output, "test")
    else:
        result = evaluate(args.features, args.output, args.split, device=args.device)
        print(canonical({"split": args.split, "overall": result["overall"]}), flush=True)


if __name__ == "__main__":
    main()
