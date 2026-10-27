"""Fixed-feature six-role warm start, with article-only DEV selection.

Old cache bytes remain under their original signatures.  A replay view records
their indices and provenance; it never relabels an old cache as a new extraction.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

import synthetic_agent_classifier as base

ROLES = base.ROLES
FORMAT = "synthetic-role-warm-features-v1"
CANDIDATES = {"warm-all": 1e-4, "warm-output": 3e-4, "scratch": 1e-3}
BINDING_KEYS = ("roles", "heads", "feature_dim", "feature_position", "model_binding")
DEFAULTS = dict(seed=17, batch_size=4096, epoch_rows=500000, max_epochs=20,
                min_epochs=10, patience=4, weight_decay=.001, dropout=.1,
                gradient_clip=5., replay_rows=250000, checkpoint_steps=100,
                weak_group_min_sources=16, weak_group_min_positives=128)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return base._sha(path)


def canonical(value):
    return base._json(value)


def object_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    base._atomic_json(path, value)


def _safe(root, path):
    return base._safe_path(root, path)


def _binding_matches(actual, expected):
    for key in BINDING_KEYS:
        require(actual.get(key) == expected.get(key), f"Feature binding differs: {key}")


class BudgetGuard:
    """One immutable six-hour window, checked before CUDA and every batch."""

    def __init__(self, path):
        require(path is not None, "GPU training requires --budget-file")
        self.path = Path(path)
        self.sha256 = digest(self.path)
        budget = json.loads(self.path.read_text())
        self.started = float(budget["started_unix"])
        self.deadline = float(budget["deadline_unix"])
        require(0 < self.deadline - self.started <= 21600.01, "Invalid six-hour training budget")
        now = time.time()
        require(self.started <= now < self.deadline, "Six-hour GPU training budget expired or not active")
        self.monotonic_deadline = time.monotonic() + self.deadline - now
        self.identity = dict(budget_sha256=self.sha256, started_unix=self.started,
                             deadline_unix=self.deadline, authorized_hours=(self.deadline - self.started) / 3600)
        self.check()

    def check(self):
        require(time.time() < self.deadline and time.monotonic() < self.monotonic_deadline,
                "Six-hour GPU training budget expired")
        require(digest(self.path) == self.sha256, "Training budget changed after it was fixed")

    def bind(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".budget.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = root / "training-budget.json"
            if path.exists():
                require(json.loads(path.read_text()) == self.identity,
                        "New experiment cannot resume with a different GPU budget")
            else:
                atomic_json(path, self.identity)


class WarmBank:
    """Verified memory-mapped feature shards for the new article wrappers."""

    def __init__(self, root, split, *, expected_binding=None, verify=True):
        self.root, self.split = Path(root), split
        self.manifest = m = json.loads((self.root / "manifest.json").read_text())
        require(m.get("format_version") == FORMAT, "Unsupported warm feature format")
        require(tuple(m.get("roles", ())) == ROLES, "Six-role ordering differs")
        require(m.get("feature_dim") == 1024 and m.get("feature_position") == "pre-o_proj",
                "Warm caches require original 1024-dimensional pre-o_proj features")
        require(len(m.get("heads", ())) == 16 and len(set(m["heads"])) == 16,
                "Warm caches require 16 distinct ordered heads")
        require(m.get("model_binding") and m.get("parent_probe_sha256"), "Missing model/parent identity")
        if expected_binding is not None:
            _binding_matches(m, expected_binding)
        entries = m.get("splits", {}).get(split)
        require(isinstance(entries, list) and entries, f"No committed {split} features")
        if "progress" in m:
            require(m["progress"].get(split, {}).get("complete") is True,
                    f"Feature extraction for {split} is incomplete")
        input_record = m.get("inputs", {}).get(split, {})
        expected_signature = input_record.get("signature")
        require(isinstance(expected_signature, dict) and expected_signature,
                "Missing extraction input/runtime signature")
        for key in ("heads", "model_binding", "parent_probe_sha256"):
            require(expected_signature.get(key) == m[key], f"Extraction signature differs: {key}")
        require(expected_signature.get("input_sha256") == input_record.get("sha256")
                and bool(input_record.get("sha256")), "Input signature hash differs")
        self.shards, offsets = [], [0]
        columns = {key: [] for key in ("y", "source", "family", "structure", "weight", "groups")}
        used = set()
        for entry in entries:
            directory = _safe(self.root, entry["path"])
            require(directory not in used, "Feature shard listed more than once")
            used.add(directory)
            complete_path = directory / "complete.json"
            require(complete_path.is_file(), "Missing shard commit")
            if verify:
                require(entry.get("sha256") == digest(complete_path), "Invalid shard commit hash")
            complete = json.loads(complete_path.read_text())
            require(complete.get("split") == split, "Shard split differs")
            require(complete.get("signature") == expected_signature, "Shard extraction signature differs")
            rows = int(entry["rows"])
            require(rows > 0 and complete.get("rows") == rows, "Shard row count differs")
            hashes = complete.get("files")
            require(isinstance(hashes, dict), "Missing shard file hashes")
            for key in ("x", *columns):
                name = key + ".npy"
                require(name in hashes, f"Uncommitted feature array: {name}")
                path = _safe(directory, name)
                expected = hashes[name]
                if isinstance(expected, dict):
                    expected = expected.get("sha256")
                if verify:
                    require(expected == digest(path), f"Invalid feature file: {path}")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if key == "x":
                    require(array.shape == (rows, 1024) and array.dtype == np.float16,
                            "Warm feature array must be FP16 N x 1024")
                    self.shards.append(array)
                else:
                    require(array.shape == (rows,), f"Metadata shape differs: {name}")
                    columns[key].append(np.asarray(array).copy())
            offsets.append(offsets[-1] + rows)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        committed_rows = m.get("progress", {}).get(split, {}).get("rows")
        require(committed_rows is None or committed_rows == int(self.offsets[-1]),
                "Completed split row count differs")
        for key, arrays in columns.items():
            setattr(self, key, np.concatenate(arrays))
        self.y = self.y.astype(np.int64)
        self.weight = self.weight.astype(np.float64)
        require(np.all((self.y >= 0) & (self.y < 6)), "Unknown role ID")
        require(np.isfinite(self.weight).all() and (self.weight > 0).all(), "Invalid base weights")
        require((self.source.astype(str) != "").all() and (self.family.astype(str) != "").all(),
                "Empty source/family ID")
        self.group_objects = [json.loads(str(x)) for x in self.groups]
        require(all(isinstance(x, dict) for x in self.group_objects), "Groups must be JSON objects")
        self.base_weight = family_weights(self.family, self.weight)
        self.manifest_hash = digest(self.root / "manifest.json")
        self.split_identity_hash = object_digest({
            **{key: m[key] for key in BINDING_KEYS}, "parent_probe_sha256": m["parent_probe_sha256"],
            "split": split, "shards": entries,
            "input": m.get("inputs", {}).get(split),
        })

    def __len__(self):
        return int(self.offsets[-1])

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        require(indices.ndim == 1 and ((indices >= 0) & (indices < len(self))).all(),
                "Feature index outside cache")
        out = np.empty((len(indices), 1024), dtype=np.float32)
        shard_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            out[positions] = self.shards[shard_id][indices[positions] - self.offsets[shard_id]]
        require(np.isfinite(out).all(), "Nonfinite model features")
        return out

    def binding(self):
        return {key: self.manifest[key] for key in BINDING_KEYS}


def family_weights(families, weights):
    _, inverse = np.unique(families, return_inverse=True)
    total = np.bincount(inverse, weights=weights)
    return np.asarray(weights, dtype=np.float64) / total[inverse]


class BankView:
    def __init__(self, bank, indices):
        self.bank = bank
        self.indices = np.asarray(indices, dtype=np.int64)
        require(len(np.unique(self.indices)) == len(self.indices), "Replay view repeats row indices")
        for key in ("y", "source", "family", "structure", "weight", "groups"):
            setattr(self, key, np.asarray(getattr(bank, key))[self.indices])
        self.base_weight = family_weights(self.family, self.weight)
        self.split = bank.split
        self.group_objects = [json.loads(str(x)) for x in self.groups]

    def __len__(self):
        return len(self.indices)

    def read(self, indices):
        return self.bank.read(self.indices[np.asarray(indices, dtype=np.int64)])


def stratified_replay(bank, count=250000, seed=17, *, excluded_sources=()):
    """Uniform present structure-role quotas, sampled without replacement."""
    eligible = ~np.isin(bank.source, np.asarray(list(excluded_sources), dtype=str))
    count = min(int(count), int(eligible.sum()))
    require(count > 0, "Empty replay request")
    cells = [np.flatnonzero(eligible & (bank.structure == structure) & (bank.y == role))
             for structure in np.unique(bank.structure[eligible])
             for role in np.unique(bank.y[eligible & (bank.structure == structure)])]
    capacity = np.asarray([len(x) for x in cells], dtype=np.int64)
    quota = np.zeros(len(cells), dtype=np.int64)
    left = count
    while left:
        active = np.flatnonzero(quota < capacity)
        increment = max(1, left // len(active))
        for i in active:
            amount = min(increment, int(capacity[i] - quota[i]), left)
            quota[i] += amount
            left -= amount
            if not left:
                break
    rng = np.random.default_rng(seed)
    selected = []
    for indices, amount in zip(cells, quota):
        p = bank.base_weight[indices]
        selected.append(rng.choice(indices, size=int(amount), replace=False, p=p / p.sum()))
    return np.sort(np.concatenate(selected))


class HierarchicalSampler:
    def __init__(self, bank, *, profile_weights=None):
        self.bank, self.profile_weights = bank, profile_weights
        self.cells = {}
        for structure in np.unique(bank.structure):
            mask = bank.structure == structure
            self.cells[str(structure)] = []
            for role in np.unique(bank.y[mask]):
                indices = np.flatnonzero(mask & (bank.y == role))
                weights = bank.base_weight[indices]
                self.cells[str(structure)].append((indices, weights / weights.sum()))
        if profile_weights is None:
            self.profiles = {"all": list(self.cells)}
            self.profile_weights = {"all": 1.}
        else:
            self.profiles = {}
            for structure in self.cells:
                indices = np.flatnonzero(bank.structure == structure)
                profiles = {str(bank.group_objects[i].get("profile", structure.partition(":")[0]))
                            for i in indices}
                require(len(profiles) == 1, "One structure cannot have multiple sampling profiles")
                self.profiles.setdefault(profiles.pop(), []).append(structure)
            require(set(self.profiles) == set(profile_weights), "New cache must contain paired and context profiles")

    def draw(self, rng, count):
        out = np.empty(count, dtype=np.int64)
        names = sorted(self.profile_weights)
        probabilities = np.asarray([self.profile_weights[k] for k in names], dtype=float)
        profile_draws = rng.choice(len(names), count, p=probabilities / probabilities.sum())
        for p, profile in enumerate(names):
            locations = np.flatnonzero(profile_draws == p)
            structures = self.profiles[profile]
            draws = rng.integers(len(structures), size=len(locations))
            for s, structure in enumerate(structures):
                destinations = locations[draws == s]
                cells = self.cells[structure]
                role_draws = rng.integers(len(cells), size=len(destinations))
                for r, (indices, weights) in enumerate(cells):
                    positions = destinations[role_draws == r]
                    out[positions] = rng.choice(indices, size=len(positions), p=weights)
        return out


class MixedSampler:
    def __init__(self, new_bank, replay_bank):
        self.new, self.old = new_bank, replay_bank
        self.new_sampler = HierarchicalSampler(new_bank, profile_weights={"paired": .6, "context": .4})
        self.old_sampler = HierarchicalSampler(replay_bank)

    def draw(self, rng, count):
        is_new = rng.random(count) < .6
        rows = np.empty(count, dtype=np.int64)
        rows[is_new] = self.new_sampler.draw(rng, int(is_new.sum()))
        rows[~is_new] = self.old_sampler.draw(rng, int((~is_new).sum()))
        return is_new, rows

    def read(self, is_new, rows):
        x = np.empty((len(rows), 1024), dtype=np.float32)
        y = np.empty(len(rows), dtype=np.int64)
        for mask, bank in ((is_new, self.new), (~is_new, self.old)):
            x[mask] = bank.read(rows[mask])
            y[mask] = bank.y[rows[mask]]
        return x, y

    def counts(self, is_new, rows):
        profiles = [self.new.group_objects[i].get("profile") for i in rows[is_new]]
        return {"new_paired": profiles.count("paired"), "new_context": profiles.count("context"),
                "old_replay": int((~is_new).sum())}


def _summary(bank, probabilities, *, weak_min_sources=16, weak_min_positives=128, details=True):
    result = {"overall": base.metrics(probabilities, bank.y), "by_structure": {}}
    predicted = probabilities.argmax(1)
    low = np.isin(bank.y, [2, 5])
    result["low_permission_to_sd"] = {
        "denominator": int(low.sum()), "errors": int(np.sum(low & np.isin(predicted, [0, 1]))),
        "rate": float(np.mean(np.isin(predicted[low], [0, 1]))) if low.any() else None,
    }
    result["minimum_precision_recall"] = min(
        value or 0. for record in result["overall"]["roles"].values()
        for value in (record["precision"], record["recall"]))
    if details:
        for structure in np.unique(bank.structure):
            mask = bank.structure == structure
            result["by_structure"][str(structure)] = base.metrics(probabilities[mask], bank.y[mask])
        result["structure_equal_macro_f1"] = float(np.mean(
            [value["macro_f1"] for value in result["by_structure"].values()]))
        objects = getattr(bank, "group_objects", None)
        if objects is None:
            objects = [json.loads(str(value)) for value in bank.groups]
        result["weak_groups"] = []
        recalls = []
        for key in ("body_length_bin", "body_quartile", "profile", "wrapper_type"):
            values = np.asarray([str(obj.get(key, "N/A")) for obj in objects])
            for value in np.unique(values):
                for role in (0, 1):
                    mask = (values == value) & (bank.y == role)
                    positives = int(mask.sum())
                    sources = len(np.unique(bank.source[mask]))
                    eligible = value != "N/A" and sources >= weak_min_sources and positives >= weak_min_positives
                    recall = float(np.mean(predicted[mask] == role)) if positives else None
                    result["weak_groups"].append(dict(attribute=key, value=value, role=ROLES[role],
                        positives=positives, sources=sources, eligible=bool(eligible), recall=recall))
                    if eligible:
                        recalls.append(recall)
        result["weakest_sd_recall"] = min(recalls) if recalls else None
    return result


def dev_decision(new, old, parent_old_macro, candidate_id, *, old_full=None, parent_old_full_macro=None):
    no_weak = new.get("weakest_sd_recall") is None
    eligible = (new["minimum_precision_recall"] >= .9 and old["minimum_precision_recall"] >= .9
                and old["overall"]["macro_f1"] >= parent_old_macro - .005
                and new["low_permission_to_sd"]["rate"] is not None
                and old["low_permission_to_sd"]["rate"] is not None
                and new["low_permission_to_sd"]["rate"] <= .01
                and old["low_permission_to_sd"]["rate"] <= .01 and not no_weak)
    key = [not eligible, -(new.get("weakest_sd_recall") or 0.),
           -new["structure_equal_macro_f1"], new["overall"]["nll"], candidate_id]
    result = dict(eligible=bool(eligible), selection_key=key, new_dev=new, old_dev=old,
                  parent_old_dev_macro_f1=parent_old_macro, weak_groups_available=not no_weak,
                  old_dev_selection_denominator="quality-clean view; identical view for parent and candidate")
    if old_full is not None:
        result.update(old_dev_full_diagnostic=old_full, parent_old_dev_full_macro_f1=parent_old_full_macro,
                      old_dev_full_used_for_selection=False)
    return result


def initialize_network(parent, candidate, *, device="cpu", seed=17, dropout=.1):
    import torch
    require(candidate in CANDIDATES, "Unknown warm candidate")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = base.make_net(128, dropout=dropout)
    if candidate != "scratch":
        net.load_state_dict({key: torch.from_numpy(value) for key, value in parent.weights.items()}, strict=True)
    if candidate == "warm-output":
        for name, value in net.named_parameters():
            value.requires_grad_(name.startswith("3."))
    return net.to(device)


def _options(overrides=None):
    options = {**DEFAULTS, **(overrides or {})}
    require(1 <= options["min_epochs"] <= options["max_epochs"] <= 20, "Invalid epoch limits")
    require(options["epoch_rows"] > 0 and options["batch_size"] > 0 and options["replay_rows"] > 0,
            "Invalid row/batch counts")
    require(options["min_epochs"] >= 10 or options.get("engineering_smoke_only", False),
            "Scientific warm training requires at least 10 epochs")
    return options


def read_quality_exclusions(path, *, allow_cpu_smoke=False):
    if path is None:
        require(allow_cpu_smoke, "Scientific warm training requires --excluded-clusters quality ledger")
        return set(), dict(ledger_sha256=None, excluded_cluster_count=0, engineering_fixture=True)
    path = Path(path)
    ledger = json.loads(path.read_text())
    require(ledger.get("format_version") == "warmstart-source-quality-exclusions-v1",
            "Unsupported source quality exclusion ledger")
    clusters = ledger.get("excluded_clusters")
    require(isinstance(clusters, list) and all(isinstance(value, str) and value for value in clusters),
            "Quality ledger must contain nonempty cluster IDs")
    require(len(set(clusters)) == len(clusters), "Quality ledger repeats cluster IDs")
    for key in ("code_sha256", "old_sources_sha256", "supplementary_sources_sha256"):
        require(bool(ledger.get(key)), f"Quality ledger missing provenance: {key}")
    return set(clusters), dict(ledger_sha256=digest(path), excluded_cluster_count=len(clusters),
        old_sources_sha256=ledger["old_sources_sha256"],
        supplementary_sources_sha256=ledger["supplementary_sources_sha256"],
        quality_filter_code_sha256=ledger["code_sha256"], assessed_sources=ledger.get("assessed_sources"))


def prepare(features_dir, old_features_dir, parent_path, run_dir, *, options=None, allow_cpu_smoke=False,
            excluded_clusters=None):
    options = _options({**(options or {}), "engineering_smoke_only": bool(allow_cpu_smoke)})
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    parent = base.load_probe(parent_path)
    require(parent.metadata["hidden"] == 128, "Parent must be MLP128")
    require(not parent.metadata.get("engineering_smoke_only") or allow_cpu_smoke, "Engineering parent refused")
    parent_sha = digest(parent_path)
    new_fit = WarmBank(features_dir, "fit", expected_binding=parent.metadata)
    new_dev = WarmBank(features_dir, "dev", expected_binding=parent.metadata)
    require(new_fit.manifest["parent_probe_sha256"] == parent_sha == new_dev.manifest["parent_probe_sha256"],
            "New feature cache parent SHA differs")
    old_fit = base.FeatureBank(old_features_dir, "fit", expected_binding=parent.metadata)
    old_dev_full = base.FeatureBank(old_features_dir, "dev", expected_binding=parent.metadata)
    excluded, quality_identity = read_quality_exclusions(excluded_clusters, allow_cpu_smoke=allow_cpu_smoke)
    for bank in (new_fit, new_dev):
        require(not np.intersect1d(bank.source, list(excluded)).size, "New training/development cache contains a rejected source")
    old_dev_indices = np.flatnonzero(~np.isin(old_dev_full.source, list(excluded)))
    require(len(old_dev_indices) > 0, "Quality gate leaves no old DEV observations")
    old_dev = BankView(old_dev_full, old_dev_indices)
    old_fit_excluded_rows = int(np.isin(old_fit.source, list(excluded)).sum())
    require(allow_cpu_smoke or len(old_fit) - old_fit_excluded_rows >= options["replay_rows"],
            "Quality-clean old FIT cannot meet the fixed replay quota")
    for fit in (new_fit, old_fit):
        for dev in (new_dev, old_dev_full):
            require(not np.intersect1d(fit.source, dev.source).size, "FIT/DEV source overlap")
    identity = dict(parent_probe_sha256=parent_sha, new_fit=new_fit.split_identity_hash,
                    new_dev=new_dev.split_identity_hash, old_fit=old_fit.split_identity_hash,
                    old_dev=old_dev_full.split_identity_hash, quality_exclusions=quality_identity,
                    old_dev_clean_indices_sha256=hashlib.sha256(old_dev_indices.tobytes()).hexdigest(), options=options,
                    implementation_sha256=digest(__file__), allow_cpu_smoke=bool(allow_cpu_smoke))
    training_hash = object_digest(identity)
    training_path = root / "training.json"
    with (root / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if training_path.exists():
            previous = json.loads(training_path.read_text())
            require(previous["training_hash"] == training_hash, "New experiment data/config/parent changed")
            replay = np.load(root / "replay-indices.npy", allow_pickle=False)
            require(previous["replay_sha256"] == digest(root / "replay-indices.npy"), "Replay indices modified")
        else:
            require(not (root / "freeze.json").exists(), "Cannot prepare a frozen experiment")
            replay = stratified_replay(old_fit, options["replay_rows"], options["seed"], excluded_sources=excluded)
            path = root / "replay-indices.npy"
            temporary = root / "replay-indices.tmp"
            with temporary.open("wb") as stream:
                np.save(stream, replay, allow_pickle=False)
            os.replace(temporary, path)
            atomic_json(training_path, dict(training_hash=training_hash, identity=identity,
                parent_probe=str(Path(parent_path).resolve()), parent_probe_sha256=parent_sha,
                features_dir=str(Path(features_dir).resolve()), old_features_dir=str(Path(old_features_dir).resolve()),
                replay_sha256=digest(path), replay_rows=len(replay), selection_split="dev", options=options,
                created_at_unix=time.time(), standardizer_policy="unchanged_parent_FIT_mean_scale",
                feature_binding={key: parent.metadata[key] for key in BINDING_KEYS},
                quality_exclusions=quality_identity,
                old_dev_denominators=dict(full_rows=len(old_dev_full), clean_rows=len(old_dev),
                    excluded_rows=len(old_dev_full) - len(old_dev),
                    full_sources=len(np.unique(old_dev_full.source)), clean_sources=len(np.unique(old_dev.source))),
                fit_sources=sorted(set(map(str, new_fit.source)) | set(map(str, old_fit.source))),
                dev_sources=sorted(set(map(str, new_dev.source)) | set(map(str, old_dev_full.source))),
                sampling_protocol=dict(new=.6, old_replay=.4, new_paired=.6, new_context=.4,
                    structures="uniform within profile", roles="uniform among present roles", family="normalized base mass",
                    cross_entropy_class_weights=None), engineering_smoke_only=bool(allow_cpu_smoke)))
    replay_view = BankView(old_fit, replay)
    require(not np.intersect1d(replay_view.source, list(excluded)).size, "Replay indices include a rejected source")
    proof = dict(status="passed", parent_probe_sha256=parent_sha,
        old_manifest_sha256=old_fit.manifest_hash,
        old_fit_split_identity_sha256=old_fit.split_identity_hash,
        old_dev_split_identity_sha256=old_dev_full.split_identity_hash,
        new_fit_split_identity_sha256=new_fit.split_identity_hash,
        new_dev_split_identity_sha256=new_dev.split_identity_hash,
        replay_indices_sha256=digest(root / "replay-indices.npy"), replay_rows=len(replay),
        replay_indices_unique=bool(len(np.unique(replay)) == len(replay)),
        quality_exclusions=quality_identity, old_fit_excluded_rows=old_fit_excluded_rows,
        old_dev_full_rows=len(old_dev_full), old_dev_clean_rows=len(old_dev),
        old_dev_excluded_rows=len(old_dev_full) - len(old_dev),
        rejected_sources_in_replay=0,
        old_files_rewritten=False, old_hashes_relabelled=False,
        feature_binding={key: parent.metadata[key] for key in BINDING_KEYS})
    proof_path = root / "cache-import-proof.json"
    with (root / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if proof_path.exists():
            require(json.loads(proof_path.read_text()) == proof, "Cache import proof changed")
        else:
            atomic_json(proof_path, proof)
    return dict(parent=parent, new_fit=new_fit, new_dev=new_dev, old_fit=old_fit,
                old_dev=old_dev, old_dev_full=old_dev_full, replay=replay_view, training_hash=training_hash,
                training=json.loads(training_path.read_text()), options=options)


def _train_candidate(features_dir, old_features_dir, parent_path, run_dir, candidate, *, device="cuda:0",
                    resume=False, options=None, allow_cpu_smoke=False, stop_after_steps=None, budget_guard=None,
                    excluded_clusters=None):
    if budget_guard is not None:
        budget_guard.check()
    import torch
    require(device.startswith("cuda") or allow_cpu_smoke, "CPU training requires explicit engineering smoke mode")
    require(candidate in CANDIDATES, "Unknown candidate")
    root = Path(run_dir)
    require(not (root / "freeze.json").exists(), "Frozen experiment cannot train")
    prepared = prepare(features_dir, old_features_dir, parent_path, root, options=options,
                       allow_cpu_smoke=allow_cpu_smoke, excluded_clusters=excluded_clusters)
    parent, options = prepared["parent"], prepared["options"]
    candidate_dir = root / "candidates" / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    completed_path = candidate_dir / "complete.json"
    if completed_path.exists():
        record = json.loads(completed_path.read_text())
        require(resume and record["training_hash"] == prepared["training_hash"], "Candidate already completed")
        require(record["probe_sha256"] == digest(candidate_dir / "best.npz"), "Completed probe modified")
        return record
    checkpoint_path = candidate_dir / "last.pt"
    require(resume or not checkpoint_path.exists(), "Checkpoint exists; use explicit resume")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if budget_guard is not None:
        budget_guard.check()
    net = initialize_network(parent, candidate, device=device, seed=options["seed"], dropout=options["dropout"])
    initial_hidden = {key: value.detach().cpu().clone() for key, value in net.state_dict().items() if key.startswith("0.")}
    init_x = prepared["new_dev"].read(np.arange(min(257, len(prepared["new_dev"]))))
    mean, scale = parent.mean.copy(), parent.scale.copy()
    net.eval()
    with torch.inference_mode():
        actual = net((torch.from_numpy(init_x).to(device) - torch.as_tensor(mean, device=device)) /
                     torch.as_tensor(scale, device=device)).softmax(1).cpu().numpy()
    init_error = float(np.max(np.abs(actual - parent.predict_proba(init_x))))
    if candidate != "scratch":
        require(init_error <= 3e-5, "Warm initialization differs from parent predictions")
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),
                                 lr=CANDIDATES[candidate], weight_decay=options["weight_decay"])
    init_proof = dict(candidate_id=candidate, parent_probe_sha256=digest(parent_path),
        inherited_weights=candidate != "scratch", standardizer_inherited=True,
        standardizer_mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
        standardizer_scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),
        initialized_weights_sha256={key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                                   for key, value in net.state_dict().items()},
        trainable_parameters=[name for name, value in net.named_parameters() if value.requires_grad],
        optimizer_initialized_empty=not bool(optimizer.state), epoch_reset_to=0,
        initial_parent_probability_max_error=init_error if candidate != "scratch" else None)
    if budget_guard is not None:
        init_proof["gpu_budget"] = budget_guard.identity
    init_proof_path = candidate_dir / "initialization-proof.json"
    if init_proof_path.exists():
        require(json.loads(init_proof_path.read_text()) == init_proof, "Initialization proof differs")
    else:
        atomic_json(init_proof_path, init_proof)
    sampler = MixedSampler(prepared["new_fit"], prepared["replay"])
    rng = np.random.default_rng(options["seed"])
    parent_old_full_p = predict_probe(parent, prepared["old_dev_full"])
    parent_old_full_macro = base.metrics(parent_old_full_p, prepared["old_dev_full"].y)["macro_f1"]
    parent_old_macro = base.metrics(parent_old_full_p[prepared["old_dev"].indices], prepared["old_dev"].y)["macro_f1"]
    del parent_old_full_p
    state = dict(epoch=0, batch_cursor=0, optimizer_step=0, stale=0, best_score=None, best_state=None,
                 best_result=None, best_epoch=0, history=[], sampler_before=None, loss_sum=0., seen=0,
                 cumulative_counts=dict(new_paired=0, new_context=0, old_replay=0),
                 epoch_counts=dict(new_paired=0, new_context=0, old_replay=0), elapsed_seconds=0.)
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(saved["training_hash"] == prepared["training_hash"] and saved["candidate_id"] == candidate,
                "Warm checkpoint identity differs")
        require(saved["parent_probe_sha256"] == digest(parent_path), "Warm parent changed")
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        state = saved["state"]
        rng.bit_generator.state = saved["sampler_after"]
        base._restore_rng(saved["rng"])
    started = time.monotonic()
    elapsed_before = state["elapsed_seconds"]

    def save_checkpoint():
        state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        base._atomic_torch_save(checkpoint_path, dict(training_hash=prepared["training_hash"], candidate_id=candidate,
            parent_probe_sha256=digest(parent_path), model={k: v.detach().cpu() for k, v in net.state_dict().items()},
            optimizer=optimizer.state_dict(), state=copy.deepcopy(state), sampler_after=copy.deepcopy(rng.bit_generator.state),
            rng=base._torch_rng()))

    while state["epoch"] < options["max_epochs"]:
        if budget_guard is not None:
            budget_guard.check()
        if state["epoch"] >= options["min_epochs"] and state["stale"] >= options["patience"]:
            break
        if state["sampler_before"] is None:
            state["sampler_before"] = copy.deepcopy(rng.bit_generator.state)
            is_new, indices = sampler.draw(rng, options["epoch_rows"])
        else:
            replay_rng = np.random.default_rng()
            replay_rng.bit_generator.state = state["sampler_before"]
            is_new, indices = sampler.draw(replay_rng, options["epoch_rows"])
        net.train()
        for batch in range(state["batch_cursor"], math.ceil(len(indices) / options["batch_size"])):
            if budget_guard is not None:
                budget_guard.check()
            sl = slice(batch * options["batch_size"], (batch + 1) * options["batch_size"])
            x, y = sampler.read(is_new[sl], indices[sl])
            x_t = torch.from_numpy(x).to(device)
            y_t = torch.from_numpy(y).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net((x_t - torch.as_tensor(mean, device=device)) / torch.as_tensor(scale, device=device))
            loss = torch.nn.functional.cross_entropy(logits, y_t)
            require(bool(torch.isfinite(loss)), "Nonfinite classifier loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad),
                                          options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            state["optimizer_step"] += 1
            state["batch_cursor"] = batch + 1
            state["loss_sum"] += float(loss.detach()) * len(y)
            state["seen"] += len(y)
            for key, value in sampler.counts(is_new[sl], indices[sl]).items():
                state["cumulative_counts"][key] += value
                state["epoch_counts"][key] += value
            if options["checkpoint_steps"] and state["optimizer_step"] % options["checkpoint_steps"] == 0:
                save_checkpoint()
            if stop_after_steps is not None and state["optimizer_step"] >= stop_after_steps:
                save_checkpoint()
                return {"status": "suspended", "optimizer_step": state["optimizer_step"]}
        new_p = predict_network(net, prepared["new_dev"], mean, scale, device=device, budget_guard=budget_guard)
        old_full_p = predict_network(net, prepared["old_dev_full"], mean, scale, device=device, budget_guard=budget_guard)
        old_p = old_full_p[prepared["old_dev"].indices]
        new_result = _summary(prepared["new_dev"], new_p,
            weak_min_sources=options["weak_group_min_sources"], weak_min_positives=options["weak_group_min_positives"])
        old_result = _summary(prepared["old_dev"], old_p, details=False)
        old_full_result = _summary(prepared["old_dev_full"], old_full_p, details=False)
        result = dev_decision(new_result, old_result, parent_old_macro, candidate,
                              old_full=old_full_result, parent_old_full_macro=parent_old_full_macro)
        improved = state["best_score"] is None or tuple(result["selection_key"]) < tuple(state["best_score"])
        if improved:
            state.update(best_score=result["selection_key"], best_state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                         best_result=result, best_epoch=state["epoch"] + 1, stale=0)
        else:
            state["stale"] += 1
        state["history"].append(dict(epoch=state["epoch"] + 1,
            train_loss=state["loss_sum"] / max(state["seen"], 1), dev=result, improved=improved,
            optimizer_step=state["optimizer_step"], sampled_rows=dict(state["epoch_counts"]),
            cumulative_sampled_rows=dict(state["cumulative_counts"])))
        print(canonical(dict(candidate=candidate, epoch=state["epoch"] + 1,
            eligible=result["eligible"], weakest_sd_recall=new_result["weakest_sd_recall"],
            new_dev_macro_f1=new_result["overall"]["macro_f1"], old_dev_macro_f1=old_result["overall"]["macro_f1"])), flush=True)
        state.update(epoch=state["epoch"] + 1, batch_cursor=0, sampler_before=None, loss_sum=0., seen=0,
                     epoch_counts=dict(new_paired=0, new_context=0, old_replay=0))
        save_checkpoint()
        atomic_json(candidate_dir / "history.json", state["history"])
    require(state["best_state"] is not None, "No completed validation epoch")
    if budget_guard is not None:
        budget_guard.check()
    net.load_state_dict(state["best_state"])
    net.eval()
    hidden_unchanged = all(torch.equal(initial_hidden[key], net.state_dict()[key].detach().cpu()) for key in initial_hidden)
    if candidate == "warm-output":
        require(hidden_unchanged, "Frozen hidden layer changed")
    metadata = {**parent.metadata, "candidate_id": candidate, "seed": options["seed"], "best_epoch": state["best_epoch"],
                "dropout": options["dropout"], "training_hash": prepared["training_hash"],
                "data_hash": object_digest({key: prepared["training"]["identity"][key] for key in ("new_fit", "new_dev", "old_fit", "old_dev", "quality_exclusions")}),
                "config_hash": object_digest(options), "standardizer": "unchanged_parent_FIT_mean_scale",
                "engineering_smoke_only": bool(allow_cpu_smoke), "training_device": device,
                "warm_start": dict(parent_probe_sha256=digest(parent_path), parent_best_epoch=parent.metadata["best_epoch"],
                    initialize_weights=candidate != "scratch", optimizer_reset=True, epoch_reset=True,
                    quality_exclusions=prepared["training"]["quality_exclusions"],
                    trainable_modules="output_only" if candidate == "warm-output" else "all",
                    learning_rate=CANDIDATES[candidate])}
    if budget_guard is not None:
        metadata["warm_start"]["gpu_budget"] = budget_guard.identity
    base._export(candidate_dir / "best.npz", net, mean, scale, metadata)
    exported = base.load_probe(candidate_dir / "best.npz")
    with torch.inference_mode():
        reference = net((torch.from_numpy(init_x).to(device) - torch.as_tensor(mean, device=device)) /
                        torch.as_tensor(scale, device=device)).softmax(1).cpu().numpy()
    export_error = float(np.max(np.abs(reference - exported.predict_proba(init_x))))
    require(export_error <= 3e-5, "Exported CPU consumer differs from training")
    record = dict(candidate_id=candidate, training_hash=prepared["training_hash"],
        parent_probe_sha256=digest(parent_path), probe_sha256=digest(candidate_dir / "best.npz"),
        history_sha256=digest(candidate_dir / "history.json"), epochs=state["epoch"], best_epoch=state["best_epoch"],
        dev=state["best_result"], elapsed_seconds=state["elapsed_seconds"],
        sampled_rows=dict(state["cumulative_counts"]), best_sampled_rows=state["history"][state["best_epoch"] - 1]["cumulative_sampled_rows"],
        init_parent_probability_max_error=init_error if candidate != "scratch" else None,
        hidden_layer_unchanged=hidden_unchanged, export_max_absolute_error=export_error,
        standardizer_mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
        standardizer_scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),
        engineering_smoke_only=bool(allow_cpu_smoke))
    if budget_guard is not None:
        record["gpu_budget"] = budget_guard.identity
    atomic_json(completed_path, record)
    return record


def train_candidate(features_dir, old_features_dir, parent_path, run_dir, candidate, *, budget_file=None, **kwargs):
    require(candidate in CANDIDATES, "Unknown candidate")
    guard = None
    if kwargs.get("device", "cuda:0").startswith("cuda"):
        guard = BudgetGuard(budget_file)
        guard.bind(run_dir)
    directory = Path(run_dir) / "candidates" / candidate
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Candidate is already training") from error
        return _train_candidate(features_dir, old_features_dir, parent_path, run_dir, candidate,
                                budget_guard=guard, **kwargs)


def predict_network(net, bank, mean, scale, *, device, budget_guard=None, batch_size=8192):
    import torch
    if budget_guard is not None:
        budget_guard.check()
    net.eval()
    result = np.empty((len(bank), 6), dtype=np.float32)
    mean_t, scale_t = torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device)
    with torch.inference_mode():
        for start in range(0, len(bank), batch_size):
            if budget_guard is not None:
                budget_guard.check()
            stop = min(start + batch_size, len(bank))
            x = torch.from_numpy(bank.read(np.arange(start, stop))).to(device)
            result[start:stop] = net((x - mean_t) / scale_t).softmax(1).cpu().numpy()
    return result


def predict_probe(probe, bank, batch_size=8192):
    probabilities = np.empty((len(bank), 6), dtype=np.float32)
    for start in range(0, len(bank), batch_size):
        stop = min(start + batch_size, len(bank))
        probabilities[start:stop] = probe.predict_proba(bank.read(np.arange(start, stop)))
    return probabilities


def freeze(run_dir):
    root = Path(run_dir)
    training_path = root / "training.json"
    training = json.loads(training_path.read_text())
    require(not training.get("engineering_smoke_only"), "Engineering training cannot become a final freeze")
    records = []
    for candidate in CANDIDATES:
        path = root / "candidates" / candidate / "complete.json"
        record = json.loads(path.read_text())
        require(record["candidate_id"] == candidate and not record.get("engineering_smoke_only"),
                "Candidate identity/engineering status differs")
        require(record["training_hash"] == training["training_hash"], "Candidate training identity differs")
        require(record["probe_sha256"] == digest(path.parent / "best.npz"), "Candidate probe modified")
        require(record["history_sha256"] == digest(path.parent / "history.json"), "Candidate history modified")
        probe = base.load_probe(path.parent / "best.npz", expected_binding=training["feature_binding"])
        require(not probe.metadata.get("engineering_smoke_only")
                and probe.metadata["training_hash"] == training["training_hash"]
                and probe.metadata["warm_start"]["parent_probe_sha256"] == training["parent_probe_sha256"],
                "Exported candidate lineage differs")
        records.append(record)
    winner = min(records, key=lambda record: tuple(record["dev"]["selection_key"]))
    selection = dict(selection_split="dev", candidate_id=winner["candidate_id"],
        candidate_results=records, eligible=winner["dev"]["eligible"],
        selected_probe=f"candidates/{winner['candidate_id']}/best.npz")
    selection_path = root / "selection.json"
    target = root / "best.npz"
    frozen_path = root / "freeze.json"
    if frozen_path.exists():
        record = json.loads(frozen_path.read_text())
        require(record["training_sha256"] == digest(training_path)
                and record["selection_sha256"] == digest(selection_path)
                and record["probe_sha256"] == digest(target), "Frozen artifact changed")
        return record
    atomic_json(selection_path, selection)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes((root / selection["selected_probe"]).read_bytes())
    os.replace(temporary, target)
    record = dict(format_version="synthetic-role-warm-freeze-v1", selection_split="dev",
        frozen_at_unix=time.time(), candidate_id=winner["candidate_id"], eligible=winner["dev"]["eligible"],
        probe_sha256=digest(target), parent_probe_sha256=training["parent_probe_sha256"],
        selection_sha256=digest(selection_path), training_sha256=digest(training_path),
        feature_binding=training["feature_binding"], promotion="research_candidate_only")
    budget_path = root / "training-budget.json"
    if budget_path.exists():
        record["gpu_budget"] = json.loads(budget_path.read_text())
        record["training_budget_sha256"] = digest(budget_path)
    atomic_json(frozen_path, record)
    return record


def evaluate(features_dir, run_dir, split="test", *, bootstrap_replicates=2000):
    root = Path(run_dir)
    frozen = json.loads((root / "freeze.json").read_text())
    require(frozen.get("selection_split") == "dev" and frozen["probe_sha256"] == digest(root / "best.npz")
            and frozen["selection_sha256"] == digest(root / "selection.json")
            and frozen["training_sha256"] == digest(root / "training.json"), "Frozen experiment changed")
    probe = base.load_probe(root / "best.npz")
    bank = WarmBank(features_dir, split, expected_binding=probe.metadata)
    require(bank.manifest["parent_probe_sha256"] == frozen["parent_probe_sha256"], "TEST parent identity differs")
    training = json.loads((root / "training.json").read_text())
    if split not in ("fit", "dev"):
        previous_sources = set(training["fit_sources"]) | set(training["dev_sources"])
        require(not previous_sources.intersection(map(str, bank.source)), "TEST sources overlap FIT/DEV")
    probabilities = predict_probe(probe, bank)
    result = _summary(bank, probabilities)
    result.update(split=split, probe_sha256=frozen["probe_sha256"],
                  feature_split_identity_sha256=bank.split_identity_hash,
                  scoring="native article-body supervised positions")
    if split not in ("fit", "dev"):
        result["confidence_intervals"] = base.bootstrap_clusters(bank, probabilities, replicates=bootstrap_replicates)
    directory = root / "evaluations"
    directory.mkdir(exist_ok=True)
    atomic_json(directory / f"{split}.json", result)
    np.save(directory / f"{split}-probabilities.npy", probabilities, allow_pickle=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "freeze", "evaluate"))
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--old-features-dir", type=Path)
    parser.add_argument("--parent-probe", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", default="test", choices=("fit", "dev", "test"))
    parser.add_argument("--options-json", type=Path)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--excluded-clusters", type=Path)
    args = parser.parse_args()
    options = json.loads(args.options_json.read_text()) if args.options_json else None
    if args.action == "freeze":
        result = freeze(args.run_dir)
    elif args.action == "evaluate":
        result = evaluate(args.features_dir, args.run_dir, args.split)
    else:
        require(all((args.features_dir, args.old_features_dir, args.parent_probe)), "Feature and parent paths required")
        if args.action == "prepare":
            result = prepare(args.features_dir, args.old_features_dir, args.parent_probe, args.run_dir,
                             options=options, allow_cpu_smoke=args.allow_cpu_smoke,
                             excluded_clusters=args.excluded_clusters)["training"]
        else:
            require(args.candidate is not None, "Candidate required for training")
            result = train_candidate(args.features_dir, args.old_features_dir, args.parent_probe, args.run_dir,
                args.candidate, device=args.device, resume=args.resume, options=options,
                allow_cpu_smoke=args.allow_cpu_smoke, budget_file=args.budget_file,
                excluded_clusters=args.excluded_clusters)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
