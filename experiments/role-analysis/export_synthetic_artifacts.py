"""Build the small, complete frozen formal delivery using an explicit whitelist.

This CPU-only exporter never starts a stage or modifies a source artifact. Raw
histories/probabilities, feature arrays, checkpoints, article inputs, historical
conversations, source-provenance previews, and mask-audit JSONL stay remote.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import uuid

import numpy as np

VERSION = "synthetic-agent-small-delivery-v1"
ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]
SEGMENTS = ["UA", "UCA", "UCQ", "UQTCA", "UQTA"]
TEMPLATES = {"SD" + value for value in SEGMENTS}
VARIANTS = {"native", "all_user", "no_tags"}
NORMAL_CATEGORIES = {"single_" + symbol for symbol in "SDUCAT"} | {"one_round_SD" + value for value in SEGMENTS} | {
    "multi_repeated_segments", "multi_terminal_UCQ", "history_cot_retain", "history_cot_drop_completed_round_cot",
    "call_chain_pre_tool_call", "call_chain_tool_reply", "call_chain_output", "multi_K5", "multi_K6"}
RECORD_ID = re.compile(r"^(?:c4|dolma3)-[0-9a-f]{24}/(?:single|one_round|multi_round)/"
                       r"(?:output|pre_tool_call|tool_reply)/(?:retain|drop_completed_round_cot|drop_prior_call_cot|prior_call_cot_visible)$")
BODY_KEYS = {"text", "body", "content", "raw_text", "article_text", "sample_text", "excerpt", "audit_examples",
             "messages", "raw_messages", "input_ids", "state_text", "user_query"}
ROOT_FILES = ["best.npz", "standardizer.npz", "config.json", "training.json", "selection.json", "freeze.json",
              "head-selection.json", "model_backend.json", "preflight.json", "resume-validation.json",
              "data-formal-manifest.json", "pilot-inheritance.json", "external-protocol.json", "external-preparation.json",
              "REPORT.zh-CN.md", "report-status.json", "training-curves.png", "training-curves.pdf"]
CORE_CODE = ["run_synthetic_agent_roles.py", "run_synthetic_agent_campaign.py", "synthetic_agent_gpu.py",
             "synthetic_agent_classifier.py", "synthetic_agent_data.py", "synthetic_agent_external.py",
             "synthetic_agent_plots.py", "synthetic_agent_report.py", "synthetic_agent_position_diagnostics.py"]
CORE_CONFIG = ["synthetic-agent-120b.json", "synthetic-agent-120b-formal.json"]
OPTIONAL_CODE = ["reuse_synthetic_pilot.py", "watch_synthetic_deadline.py", "monitor_synthetic_gpu.py",
                 "select_synthetic_article_audit.py", "stage_synthetic_model.py"]
STAGES = {"prepare", "preflight", "scan", "extract", "train", "freeze", "evaluate", "plot", "report", "external-protocol", "external-prepare"}
RECEIPTS = {"pilot." + name for name in ["scan", "extract", "train", "freeze", "evaluate.dev", "report", "plot.dev"]} | {
    "formal." + name for name in ["prepare", "external-protocol", "extract", "train", "freeze", "evaluate.dev", "extract.test", "evaluate.test",
                                  "external-prepare", "extract.external", "evaluate.external", "plot.test", "dense", "report"]}
DENSE_NOTE = ("Dense deployment diagnostic only. Scores refer to original semantic roles; "
              "changed-format scores are not formal role accuracy.")


class NotReady(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise NotReady(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def body_free(value):
    if isinstance(value, dict):
        require(not BODY_KEYS.intersection(str(key).casefold() for key in value), "A purported metadata artifact contains a raw-body/input key")
        for item in value.values():
            body_free(item)
    elif isinstance(value, list):
        for item in value:
            body_free(item)


def bounded(root, relative):
    root = Path(root).resolve()
    original = root / relative
    result = original.resolve()
    require(result.is_relative_to(root) and not original.is_symlink(), "Whitelist artifact escapes its source directory or is a symlink")
    return result


def load_classifier(code_dir):
    path = Path(code_dir) / "synthetic_agent_classifier.py"
    spec = importlib.util.spec_from_file_location("_artifact_export_classifier", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric(value):
    return value is None or isinstance(value, (int, float, bool)) and (not isinstance(value, float) or math.isfinite(value))


def scalar_fields(mapping, exclude=()):
    result = {}
    for key, value in mapping.items():
        if key not in exclude and (value is None or isinstance(value, (int, float, bool))):
            require(numeric(value), "Nonfinite epoch/evaluation scalar cannot be silently omitted")
            result[key] = value
    return result


def count_tree(value):
    if isinstance(value, dict):
        require(all(isinstance(key, str) and len(key) <= 512 for key in value), "Invalid sampling-count keys")
        return {key: count_tree(item) for key, item in value.items()}
    require(numeric(value), "Sampling summaries must contain only numeric/boolean/null leaves")
    return value


def compact_history(path, candidate):
    """A derived scalar view, explicitly distinct from the untouched raw history."""
    raw = read(path)
    rows = raw if isinstance(raw, list) else raw.get("history", raw.get("epochs"))
    require(isinstance(rows, list) and rows, "A completed candidate has no epoch history")
    require([row["epoch"] for row in rows] == list(range(1, len(rows) + 1)), "Candidate history epochs are incomplete or reordered")
    require(len(rows) == candidate["epochs"], "Raw history length differs from completed candidate epochs")
    epochs = []
    for row in rows:
        scalars = scalar_fields(row, {"dev"})
        require(all(key in scalars for key in ["epoch", "optimizer_step", "improved"]), "History lacks step/epoch/improvement evidence")
        require(any("loss" in key for key in scalars), "History lacks a scalar training loss")
        dev = row["dev"]
        overall = scalar_fields(dev["overall"], {"roles"})
        require(all(key in overall for key in ["tokens", "accuracy", "macro_f1", "nll"]), "History lacks required DEV overall scalars")
        roles = {}
        for role in ROLES:
            values = scalar_fields(dev["overall"]["roles"][role])
            require(all(key in values for key in ["precision", "recall", "f1"]), "History lacks a per-class DEV P/R/F1")
            roles[role] = values
        dev_scalars = scalar_fields(dev, {"overall"})
        require(all(key in dev_scalars for key in ["structure_equal_macro_f1", "minimum_precision_recall", "meets_dev_floor"]),
                "History lacks the fixed DEV selection criteria")
        samples = {key: count_tree(value) for key, value in row.items() if "sampled" in key}
        require("sampled_rows_by_profile" in samples, "History lacks actual profile sampling counts")
        epochs.append({**scalars, "sampling": samples, "dev": {**dev_scalars, "overall": overall, "roles": roles}})
    return {"format_version": "synthetic-derived-epoch-scalars-v1", "derived_artifact": True,
            "not_the_raw_history": True, "candidate_id": candidate["candidate_id"],
            "source_history_path": str(Path(path).resolve()), "source_history_sha256": sha(path),
            "source_history_bytes": Path(path).stat().st_size, "raw_history_retained_remote": True,
            "best_epoch": candidate["best_epoch"], "epochs_run": candidate["epochs"], "epochs": epochs,
            "omitted": "Repeated full DEV by_structure/by_attribute/confusion trees; the untouched original history remains at the recorded remote path"}


def safe_probability_rows(value, count):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (count, 6) and np.isfinite(array).all() and np.all(array >= 0) and
            np.all(array <= 1) and np.allclose(array.sum(1), 1, atol=3e-5), "Invalid plot probability sidecar")


def int_positions(values, count, limit):
    require(isinstance(values, list) and len(values) == count and
            all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < limit for value in values) and
            len(set(values)) == count, "Invalid or duplicate plot positions")


def validate_sidecar(value, *, dense, probe_sha, entry, category=None):
    if dense:
        required = {"record_id", "variant", "formal_scoring", "note", "input_tokens", "body_tokens", "probabilities", "positions"}
        require(set(value) == required and value["formal_scoring"] is False and value["variant"] == entry["variant"] and
                value["note"] == DENSE_NOTE, "Unexpected dense sidecar fields or free text")
        count = value["body_tokens"]
        positions = value["positions"]
    else:
        required = {"record_id", "category", "input_tokens", "scored_positions", "original_roles", "probabilities", "token_offsets", "probe_sha256"}
        require(set(value) == required and value["category"] == category and value["probe_sha256"] == probe_sha,
                "Unexpected normal sidecar fields or classifier binding")
        count = entry["scored_tokens"]
        require(value["input_tokens"] == entry["input_tokens"], "Normal sidecar input count differs from its index")
        require(len(value["original_roles"]) == count and set(value["original_roles"]).issubset(ROLES), "Invalid original-role labels")
        offsets = value["token_offsets"]
        require(isinstance(offsets, list) and len(offsets) == value["input_tokens"] and
                all(isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in pair)
                    and 0 <= pair[0] <= pair[1] for pair in offsets), "Invalid token-offset-only sidecar")
        positions = value["scored_positions"]
    require(type(value["input_tokens"]) is int and 0 < value["input_tokens"] <= 4096 and
            type(count) is int and 0 < count <= value["input_tokens"], "Invalid plot input/body count")
    require(value["record_id"] == entry["record_id"] and RECORD_ID.fullmatch(value["record_id"]), "Plot record ID contains unexpected text or mismatches its index")
    int_positions(positions, count, value["input_tokens"])
    safe_probability_rows(value["probabilities"], count)


def validate_ci(evaluation):
    confidence = evaluation.get("confidence_intervals", {})
    require(confidence.get("replicates") == 2000 and confidence.get("unit") == "source_cluster" and
            confidence.get("confidence") == .95 and confidence.get("clusters", 0) >= 2,
            "Final TEST/external must retain the 2000 source-cluster 95% bootstrap")
    for key in ["accuracy", "macro_f1", "nll"] + [f"{role}.{metric}" for role in ROLES for metric in ["precision", "recall", "f1"]]:
        require(key in confidence["intervals"], "A final bootstrap interval is missing")
        value = confidence["intervals"][key]
        if value is not None:
            require(all(numeric(value.get(name)) and value.get(name) is not None for name in ["lower", "upper", "valid_replicates"]) and
                    isinstance(value["valid_replicates"], int) and 0 < value["valid_replicates"] <= 2000 and
                    0 <= value["lower"] <= value["upper"] and (key == "nll" or value["upper"] <= 1), "Invalid bootstrap interval")


def probe_archive(path, classifier, binding=None):
    probe = classifier.load_probe(path, expected_binding=binding)
    require(not probe.metadata.get("engineering_smoke_only"), "Engineering smoke probes cannot be exported as experiment results")
    expected = {"mean", "scale", "metadata", "weight::0.weight", "weight::0.bias"}
    if probe.metadata["hidden"]:
        expected |= {"weight::3.weight", "weight::3.bias"}
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected, "NPZ contains unknown arrays; never export hidden raw/debug data")
    return probe


class Exporter:
    def __init__(self, args):
        self.args, self.run, self.data, self.code = args, args.run_dir, args.data_dir, args.code_dir
        self.files, self.derived, self.retained = {}, {}, []
        self.classifier = None

    def add(self, relative, source, *, json_metadata=False):
        require(relative not in self.files and relative not in self.derived, "Duplicate delivery path")
        source = Path(source)
        require(source.is_file() and not source.is_symlink(), f"Required committed artifact missing: {source}")
        if source.suffix in {".png", ".pdf"}:
            with source.open("rb") as stream:
                header = stream.read(8)
            require(header.startswith(b"\x89PNG\r\n\x1a\n" if source.suffix == ".png" else b"%PDF-") and source.stat().st_size > 8,
                    "A required figure is empty or has an invalid file signature")
        if json_metadata:
            body_free(read(source))
        self.files[relative] = {"source_path": str(source.resolve()), "sha256": sha(source), "bytes": source.stat().st_size,
                                "kind": "verbatim"}

    def derive(self, relative, value, dependencies):
        require(relative not in self.files and relative not in self.derived, "Duplicate derived delivery path")
        payload = value if isinstance(value, bytes) else encoded(value)
        self.derived[relative] = payload
        self.files[relative] = {"kind": "derived", "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                                "source_dependencies": dependencies}

    def validate(self):
        root = self.run
        campaign = read(root.parent / "campaign" / "status.json")
        require(campaign.get("status") == "complete" and campaign.get("run_root") == str(root.parent.resolve()) and
                campaign.get("data_dir") == str(self.data.resolve()), "The entire formal campaign has not completed")
        require(read(root / "status.json").get("status") == "complete", "A formal stage is still incomplete")
        frozen, selection, training = read(root / "freeze.json"), read(root / "selection.json"), read(root / "training.json")
        require(frozen["probe_sha256"] == sha(root / "best.npz") and sha(root / "freeze.json") == campaign["formal_freeze_sha256"] and
                frozen["selection_sha256"] == sha(root / "selection.json") and frozen["training_sha256"] == sha(root / "training.json"),
                "Final freeze/selection/training/model SHA changed")
        require(training["phase"] == "formal" and frozen["selection_split"] == selection["selection_split"] == "dev",
                "Formal model must be selected on DEV only")
        require(training["implementation_sha256"] == sha(self.code / "synthetic_agent_classifier.py"), "Frozen classifier source SHA changed")
        classifier = self.classifier = load_classifier(self.code)
        require(training["options"] == classifier._training_options(read(root / "config.json")), "Recorded training options differ from frozen code/config")
        probe = probe_archive(root / "best.npz", classifier, training["binding"])
        require(probe.metadata == frozen["metadata"] and list(probe.metadata["roles"]) == ROLES and
                probe.metadata["training_hash"] == training["training_hash"], "Frozen NPZ metadata changed")
        heads = read(root / "head-selection.json")
        require(heads["heads"] == probe.metadata["heads"] and heads["signature"]["model_binding"] == probe.metadata["model_binding"] and
                heads["signature"]["implementation_sha256"] == sha(self.code / "synthetic_agent_gpu.py"), "Head-selection/model/runtime binding changed")
        require(probe.metadata["selection_sha256"] == sha(root / "head-selection.json") and
                probe.metadata["pilot_fit_sha256"] == heads["signature"]["pilot_fit_sha256"], "NPZ pilot head-discovery binding changed")
        require(read(root / "preflight.json")["status"] == "passed" and read(root / "resume-validation.json").get("passed") is True,
                "Preflight or recovery validation did not pass")
        with np.load(root / "standardizer.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"training_hash", "mean", "scale"} and str(archive["training_hash"]) == training["training_hash"] and
                    np.array_equal(archive["mean"], probe.mean) and np.array_equal(archive["scale"], probe.scale), "Exported FIT standardizer changed")
        results = []
        for candidate_id, hidden, seed in classifier.CANDIDATES:
            directory = root / "candidates" / candidate_id
            candidate = read(directory / "complete.json")
            require(candidate["candidate_id"] == candidate_id and candidate["training_hash"] == training["training_hash"] and
                    candidate["probe_sha256"] == sha(directory / "best.npz") and candidate["history_sha256"] == sha(directory / "history.json"),
                    "A completed candidate or raw history changed")
            require(training["options"]["min_epochs"] <= candidate["epochs"] <= training["options"]["max_epochs"] and
                    1 <= candidate["best_epoch"] <= candidate["epochs"] and candidate["elapsed_seconds"] > 0,
                    "Candidate epoch/timing evidence does not meet the formal protocol")
            candidate_probe = probe_archive(directory / "best.npz", classifier, training["binding"])
            require(candidate_probe.metadata["training_hash"] == training["training_hash"] and
                    candidate_probe.metadata["candidate_id"] == candidate_id and candidate_probe.metadata["hidden"] == hidden and
                    candidate_probe.metadata["seed"] == seed and np.array_equal(candidate_probe.mean, probe.mean) and
                    np.array_equal(candidate_probe.scale, probe.scale), "Candidate NPZ identity or FIT standardizer changed")
            summary = compact_history(directory / "history.json", candidate)
            self.derive(f"formal/candidates/{candidate_id}/curve-scalars.json", summary,
                        {str(directory / "history.json"): candidate["history_sha256"], str(directory / "complete.json"): sha(directory / "complete.json")})
            for name in ["complete.json", "best.npz"]:
                self.add(f"formal/candidates/{candidate_id}/{name}", directory / name, json_metadata=name.endswith(".json"))
            self.retained.append({"path": str(directory / "history.json"), "sha256": candidate["history_sha256"],
                                  "bytes": (directory / "history.json").stat().st_size, "reason": "Untouched full raw epoch history retained remotely; delivered curve-scalars is explicitly derived"})
            results.append(candidate)
        winner = min(results, key=lambda row: classifier.selection_key(row["dev"], row["candidate_id"]))
        require(selection["candidate_results"] == results and selection["candidate_id"] == winner["candidate_id"] and
                winner["probe_sha256"] == frozen["probe_sha256"], "The fixed seven-candidate DEV selection changed")
        require(not read(root / "report-status.json")["missing"], "Final report is incomplete")
        self.frozen = frozen
        self.protocol = {"freeze_sha256": sha(root / "freeze.json"), "probe_sha256": frozen["probe_sha256"],
                         "selection_sha256": frozen["selection_sha256"], "training_hash": training["training_hash"],
                         "roles": ROLES, "heads": probe.metadata["heads"], "model_binding": probe.metadata["model_binding"],
                         "campaign_status_sha256": sha(root.parent / "campaign" / "status.json")}
        for split in ["dev", "test", "external"]:
            self.evaluation(split)
        self.plot_files()

    def evaluation(self, split):
        root = self.run
        path = root / "evaluations" / f"{split}.json"
        value = read(path)
        require(value["probe_sha256"] == self.frozen["probe_sha256"] and value["split"] == ("test" if split == "external" else split),
                "Evaluation belongs to a different frozen model/split")
        if split != "dev":
            validate_ci(value)
        probabilities_path = root / "evaluations" / f"{split}-probabilities.npy"
        probabilities = np.load(probabilities_path, mmap_mode="r", allow_pickle=False)
        require(probabilities.shape == (value["overall"]["tokens"], 6) and np.isfinite(probabilities).all() and
                np.all(probabilities >= 0) and np.all(probabilities <= 1) and
                np.allclose(probabilities.sum(1), 1, atol=3e-5), "Evaluation probability file is incomplete")
        matrix = np.asarray(value["overall"]["confusion"], dtype=np.int64)
        require(matrix.shape == (6, 6) and matrix.sum() == len(probabilities) and
                np.array_equal(np.bincount(probabilities.argmax(1), minlength=6), matrix.sum(0)), "Evaluation confusion/probability totals disagree")
        positions_path = root / "evaluations" / f"{split}-positions.json"
        positions = read(positions_path)
        require(positions["probe_sha256"] == self.frozen["probe_sha256"] and positions["freeze_sha256"] == sha(root / "freeze.json") and
                positions["evaluation_sha256"] == sha(path) and positions["probabilities_sha256"] == sha(probabilities_path) and
                positions["selection_permitted"] is False, "Required position diagnostics do not bind the frozen evaluation")
        for source in [path, positions_path]:
            self.add("formal/evaluations/" + source.name, source, json_metadata=True)
        self.retained.append({"path": str(probabilities_path), "sha256": sha(probabilities_path),
                              "bytes": probabilities_path.stat().st_size, "reason": "Full per-token probabilities remain remote; metrics/CI/position summaries and selected plot sidecars are delivered"})
        for suffix in ["png", "pdf"]:
            self.add(f"formal/{split}-confusion.{suffix}", root / f"{split}-confusion.{suffix}")

    def plot_files(self):
        root, probe_sha = self.run, self.frozen["probe_sha256"]
        normal_root, dense_root = root / "plots" / "test", root / "plots" / "dense"
        normal = read(normal_root / "index.json")
        require(set(normal) == {"phase", "split", "probe_sha256", "scoring", "selection", "examples", "missing_coverage", "call_chain_note"},
                "Normal plot index has unknown fields")
        require(normal["scoring"] == "sampled article-body mask only; no interpolation across unsampled tokens" and
                normal["selection"] == "fixed structural categories; lexicographically first cluster/input ID, independent of predictions" and
                normal["call_chain_note"] == "Stages contain different target bodies; their overall scores do not estimate the causal effect of deleting CoT.",
                "Normal plot index contains unexpected free text")
        require(normal["phase"] == "formal" and normal["split"] == "test" and normal["probe_sha256"] == probe_sha and
                not normal["missing_coverage"] and set(normal["examples"]) == NORMAL_CATEGORIES,
                "All 20 required native tomato categories must be present")
        self.add("formal/plots/test/index.json", normal_root / "index.json", json_metadata=True)
        seen = set()
        for category, entry in normal["examples"].items():
            require(set(entry) == {"record_id", "png", "pdf", "scored_tokens", "input_tokens"}, "Normal plot index entry has unknown fields")
            stem = category + "-" + hashlib.sha256(entry["record_id"].encode()).hexdigest()[:10]
            for suffix in ["png", "pdf"]:
                require(entry[suffix] == stem + "." + suffix, "Normal figure filename is not bound to its indexed record/category")
                source = bounded(normal_root, entry[suffix])
                require(source.suffix == "." + suffix and source not in seen, "Duplicate or unexpected normal plot filename")
                seen.add(source)
                self.add("formal/plots/test/" + source.name, source)
            sidecar = bounded(normal_root, str(Path(entry["png"]).with_suffix(".json")))
            validate_sidecar(read(sidecar), dense=False, probe_sha=probe_sha, entry=entry, category=category)
            self.add("formal/plots/test/" + sidecar.name, sidecar, json_metadata=True)
        dense = read(dense_root / "index.json")
        require(set(dense) == {"formal_scoring", "probe_sha256", "test_inputs_sha256", "selection", "note", "examples"},
                "Dense plot index has unknown fields")
        require(dense["selection"] == "lexicographically first input ID in each of the five output templates; independent of scores" and
                dense["note"] == "Changed-format probabilities refer to original semantic roles and are not formal accuracy.",
                "Dense plot index contains unexpected free text")
        require(dense["formal_scoring"] is False and dense["probe_sha256"] == probe_sha and len(dense["examples"]) == 15,
                "All 15 fixed dense diagnostic figures are required")
        require(dense["test_inputs_sha256"] == read(self.run / "data-formal-manifest.json")["sha256"]["inputs-test.jsonl"],
                "Dense diagnostic inputs differ from the committed formal TEST")
        combinations = [(entry["template_id"], entry["variant"]) for entry in dense["examples"]]
        require(len(set(combinations)) == 15 and set(combinations) == {(t, v) for t in TEMPLATES for v in VARIANTS},
                "Dense diagnostics do not cover five templates × native/all_user/no_tags")
        self.add("formal/plots/dense/index.json", dense_root / "index.json", json_metadata=True)
        for entry in dense["examples"]:
            require(set(entry) == {"template_id", "record_id", "variant", "png", "pdf", "json"}, "Dense plot index entry has unknown fields")
            stem = entry["variant"] + "-" + hashlib.sha256(entry["record_id"].encode()).hexdigest()[:10]
            for suffix in ["png", "pdf", "json"]:
                require(entry[suffix] == stem + "." + suffix, "Dense figure filename is not bound to its indexed record/variant")
                source = bounded(dense_root, entry[suffix])
                require(source.suffix == "." + suffix and source not in seen, "Duplicate or unexpected dense plot filename")
                seen.add(source)
                if suffix == "json":
                    validate_sidecar(read(source), dense=True, probe_sha=probe_sha, entry=entry)
                self.add("formal/plots/dense/" + source.name, source, json_metadata=suffix == "json")

    def metadata(self):
        root = self.run
        for phase in ["pilot", "formal"]:
            folder = self.data / phase
            manifest = read(folder / "data-manifest.json")
            require(manifest["phase"] == phase and manifest["sha256"]["source-keys.json"] == sha(folder / "source-keys.json"), "Data source-key metadata changed")
            keys = read(folder / "source-keys.json")
            expected_counts = {"fit": 512, "dev": 128} if phase == "pilot" else {"fit": 4096, "dev": 512, "test": 1024}
            require(isinstance(keys, list) and Counter(row["split"] for row in keys) == expected_counts and
                    len({row["source_id"] for row in keys}) == len(keys), "Delivered source-key allocation has incomplete or duplicate source IDs")
            audit = read(folder / "article-audit.json")
            require(audit["status"] == "passed" and audit["data_manifest_sha256"] == sha(folder / "data-manifest.json") and
                    audit["sources_sha256"] == manifest["sha256"]["sources.jsonl"], "Article audit does not bind the delivered data manifest")
            # These retained body files are streamed into SHA only: never
            # parsed, decoded, copied, or printed by the exporter.
            inputs = {}
            for name in ["sources.jsonl", *[f"inputs-{split}.jsonl" for split in expected_counts]]:
                source = folder / name
                actual = sha(source)
                require(actual == manifest["sha256"][name], "Current retained source/input bytes differ from the data manifest")
                self.retained.append({"path": str(source), "sha256": actual, "bytes": source.stat().st_size,
                                      "reason": "Hash-only current-state source/input provenance; raw bytes were not parsed or copied"})
                if name.startswith("inputs-"):
                    inputs[name[len("inputs-"):-len(".jsonl")]] = actual
            for name in ["data-manifest.json", "source-keys.json", "article-audit.json"]:
                self.add(f"metadata/data/{phase}/{name}", folder / name, json_metadata=True)
            self.protocol.setdefault("data_manifest_sha256", {})[phase] = sha(folder / "data-manifest.json")
            self.protocol.setdefault("input_sha256", {})[phase] = inputs
        require(sha(root / "data-formal-manifest.json") == sha(self.data / "formal" / "data-manifest.json"), "Run/data formal manifest mirrors disagree")
        pointer = read(root / "external-protocol.json")
        external_root = self.data / "external"
        protocol, external = read(external_root / "protocol.json"), read(external_root / "manifest.json")
        require(pointer["protocol_sha256"] == sha(external_root / "protocol.json") == self.frozen["external_protocol_sha256"] and
                self.frozen["external_protocol_pointer_sha256"] == sha(root / "external-protocol.json") and
                protocol["declared_at_unix"] < self.frozen["frozen_at_unix"] < external["created_at_unix"] and
                external["probe_sha256"] == self.frozen["probe_sha256"] and external["protocol_sha256"] == pointer["protocol_sha256"] and
                external["freeze_sha256"] == sha(root / "freeze.json") and external["counts_by_source"] == {"oasst": 128, "toxicchat": 128},
                "External protocol/freeze/sample chronology changed")
        for name in ["protocol.json", "manifest.json"]:
            self.add("metadata/external/" + name, external_root / name, json_metadata=True)
        external_inputs = external_root / "inputs-test.jsonl"
        require(sha(external_inputs) == external["inputs_sha256"], "Current external input bytes changed after evaluation")
        self.protocol.setdefault("input_sha256", {})["external"] = {"test": sha(external_inputs)}
        self.retained.append({"path": str(external_inputs), "sha256": sha(external_inputs), "bytes": external_inputs.stat().st_size,
                              "reason": "Hash-only current-state external input provenance; real conversation bytes remain remote"})
        for name in ["history-index.json", "legacy-hashes.json"]:
            source = external_root / name
            require(source.is_file(), "Historical hash-only exclusion proof is missing")
            self.retained.append({"path": str(source), "sha256": sha(source), "bytes": source.stat().st_size,
                                  "reason": "Hash-only historical exclusion ledger stays remote; its protocol binding and audit counts are delivered"})
        require(protocol["history_index_sha256"] == sha(external_root / "history-index.json") == self.frozen["external_history_index_sha256"],
                "External historical exclusion index changed")
        legacy = read(external_root / "legacy-hashes.json")["binding"]
        require(legacy["protocol_sha256"] == sha(external_root / "protocol.json") and legacy["freeze_sha256"] == sha(root / "freeze.json") and
                legacy["probe_sha256"] == self.frozen["probe_sha256"] and legacy["history_index_sha256"] == protocol["history_index_sha256"] and
                legacy["implementation_sha256"] == sha(self.code / "synthetic_agent_external.py"), "External local hash augmentation binding changed")
        self.protocol["external"] = {"protocol_sha256": sha(external_root / "protocol.json"), "manifest_sha256": sha(external_root / "manifest.json"),
                                      "history_index_sha256": protocol["history_index_sha256"], "legacy_hashes_sha256": sha(external_root / "legacy-hashes.json")}
        for phase in ["formal", "external"]:
            feature_root = root / "features" / phase
            manifest = read(feature_root / "manifest.json")
            expected_binding = self.frozen["metadata"]
            for key in ["roles", "feature_dim", "feature_position", "heads", "model_binding", "selection_sha256", "pilot_fit_sha256"]:
                require(manifest.get(key) == expected_binding.get(key), "Feature provenance metadata differs from frozen probe: " + key)
            require(set(manifest["splits"]) == ({"fit", "dev", "test"} if phase == "formal" else {"test"}), "Final feature metadata is missing required splits")
            self.protocol.setdefault("feature_manifest_sha256", {})[phase] = sha(feature_root / "manifest.json")
            self.add(f"metadata/features/{phase}/manifest.json", feature_root / "manifest.json", json_metadata=True)
            for split, entries in manifest["splits"].items():
                require(manifest["progress"][split]["complete"], "A retained feature cache split is incomplete")
                require(manifest["split_sources"][split] == self.protocol["input_sha256"][phase][split],
                        "Feature metadata does not bind the current retained input JSONL")
                signature = {**manifest["signature"], "split": split, "input_sha256": manifest["split_sources"][split]}
                for entry in entries:
                    directory = bounded(feature_root, entry["path"])
                    commit = directory / "complete.json"
                    require(sha(commit) == entry["sha256"] and read(commit)["signature"] == signature, "Feature metadata commit changed")
                    self.add(f"metadata/features/{phase}/{entry['path']}/complete.json", commit, json_metadata=True)
            self.retained.append({"path": str(feature_root), "manifest_sha256": sha(feature_root / "manifest.json"),
                                  "reason": "All activation/feature arrays remain remote; exporter revalidates their manifest/commit metadata and delivers recorded array checksums, without rereading the multi-GiB array payloads"})
        migration = root.parent / "pilot" / "migration.json"
        if migration.is_file():
            self.add("metadata/pilot-migration.json", migration, json_metadata=True)

    def logs_and_code(self):
        root, campaign = self.run, self.run.parent / "campaign"
        for name in ROOT_FILES:
            self.add("formal/" + name, root / name, json_metadata=name.endswith(".json"))
        for stage in sorted(STAGES):
            source = root / (stage + "-stage.json")
            if not source.is_file():
                continue
            marker = read(source)
            require(marker["status"] == "complete" and marker["phase"] == "formal", "A final formal stage marker is incomplete")
            self.add("formal/" + source.name, source, json_metadata=True)
        for name in ["binding.json", "status.json", "formal-budget-accepted.json", "formal-budget-estimate.json", "driver.log"]:
            self.add("campaign/" + name, campaign / name, json_metadata=name.endswith(".json"))
        for name in sorted(RECEIPTS):
            source = campaign / "receipts" / (name + ".json")
            if not source.is_file():
                continue
            receipt = read(source)
            require(receipt.get("completed_unix") and receipt.get("proof_sha256") == object_sha(receipt["proof"]), "Campaign receipt is not a completed commit")
            self.add("campaign/receipts/" + source.name, source, json_metadata=True)
            if source.name.startswith("formal."):
                log = receipt.get("execution", {}).get("log")
                if log:
                    log_path = Path(log).resolve()
                    require(log_path.is_relative_to(campaign.resolve() / "logs") and log_path.name.startswith("formal.") and log_path.suffix == ".log",
                            "Completed formal receipt points outside the scoped stage log directory")
                    target = "campaign/logs/" + log_path.name
                    if target not in self.files:
                        self.add(target, log_path)
        for name in CORE_CODE:
            self.add("code/" + name, self.code / name)
        binding = read(campaign / "binding.json")
        for name, phase in zip(CORE_CONFIG, ["pilot", "formal"]):
            source = self.code / name
            config = read(source)
            effective = json.loads(canonical(config))
            effective["phase"] = phase
            effective.setdefault("classifier", {})["max_epochs"] = 15 if phase == "pilot" else 45
            require(binding["configs"][phase] == object_sha(effective), "Original source configuration differs from the completed campaign binding")
            if phase == "formal":
                require(effective == read(root / "config.json"), "Original and effective formal configurations disagree")
            self.add("code/" + name, source, json_metadata=True)
        for name in OPTIONAL_CODE:
            if (self.code / name).is_file():
                self.add("code/" + name, self.code / name)
        self.add("code/export_synthetic_artifacts.py", Path(__file__))
        require(self.args.requirements_file.is_file(), "Provide the pinned requirements file for the final delivery")
        self.add("code/requirements-synthetic-agent-4xa40.txt", self.args.requirements_file)
        if self.args.environment_file:
            self.add("metadata/training-environment-freeze.txt", self.args.environment_file)
        environment = {"export_process_python": sys.version, "export_process_platform": platform.platform(),
                       "export_process_distributions": sorted(f"{d.metadata.get('Name', 'unknown')}=={d.version}" for d in importlib.metadata.distributions()),
                       "training_runtime_reference": "formal/model_backend.json and the supplied pinned requirements; exporter environment is recorded separately"}
        self.derive("metadata/export-process-environment.json", environment, {})
        self.derive("examples/load_probe.py", LOADER_EXAMPLE.encode(), {"formal/best.npz": self.frozen["probe_sha256"]})
        self.derive("README.md", README.encode(), {"formal/freeze.json": sha(root / "freeze.json")})

    def build(self):
        self.validate()
        self.metadata()
        self.logs_and_code()
        return {"format_version": VERSION, "complete": True, "created_unix": time.time(),
                "exporter_sha256": sha(__file__), "formal_source_dir": str(self.run), "data_source_dir": str(self.data),
                "code_source_dir": str(self.code), "protocol_bindings": self.protocol, "files": self.files,
                "remote_retained": self.retained,
                "exclusions": ["full raw history.json", "*-probabilities.npy", "feature/activation arrays", "last.pt", "raw article/conversation inputs",
                               "source-provenance.json", "mask-audit.jsonl", "raw source-body files (requested full-context figures are included)",
                               "historical CSV bodies", "old preparation/debug logs"],
                "curve_policy": "curve-scalars.json is derived; original histories remain byte-identical remotely at their recorded SHA256",
                "gpu_monitoring": "The task owner collects the cross-version GPU CSV/summary separately; this exporter does not traverse old run roots"}


LOADER_EXAMPLE = '''"""Use raw ordered pre-o_proj features from the bound frozen model; no toy data."""
import argparse
from pathlib import Path
import sys
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "code"))
from synthetic_agent_classifier import load_probe, ROLES
p = argparse.ArgumentParser()
p.add_argument("features", type=Path, help="Actual raw N x 1024 selected-head features from the bound GPT-OSS model")
p.add_argument("--rows", type=int, default=8)
a = p.parse_args()
probe = load_probe(root / "formal" / "best.npz")
x = np.load(a.features, mmap_mode="r", allow_pickle=False)
probability = probe.predict_proba(x[:a.rows])  # FIT mean/scale are applied internally exactly once.
print("roles:", ROLES)
print("ordered heads:", probe.metadata["heads"])
print(probability)
'''

README = '''# Frozen six-role GPT-OSS-120B delivery

The final classifier is `formal/best.npz`. `formal/freeze.json` binds its exact SHA256,
six-role order, model/tokenizer, ordered sixteen heads, feature position, and FIT
standardizer. The frozen 120B backbone was not trained. Original articles were
placed in native message/channel wrappers; the C label is an operational analysis
channel label and does not assert genuine reasoning semantics.

Run `sha256sum -c SHA256SUMS` from this directory to verify the package. The
`bundle-manifest.json` records original remote paths, exact file hashes, protocol
bindings, and remote-retained large artifacts. Activation arrays, original epoch
histories, raw synthetic/external inputs, and full probability arrays remain on
the run host. Historical fitting and user-query CSV files remain on the original
local host; those CSV files were not uploaded. Their de-duplication contribution
on the run host consists of bound hashes and public source IDs. The separately
authorized public gardening diagnostic is archived outside this package.

The seven `curve-scalars.json` files are **derived summaries, not raw histories**.
They preserve every epoch's scalar losses/DEV criteria, class P/R/F1, sampling
counts, optimizer step, and improvement flag, and record each untouched raw
history's path/SHA. Seven candidate `complete.json` and `best.npz` files are also
included; final model selection used DEV only. TEST/external results and 2000
source-cluster confidence intervals appear under `formal/evaluations/`.

`formal/plots/test/` contains twenty native full-context masked tomato examples.
`formal/plots/dense/` contains fifteen fixed dense diagnostics: five templates by
native/all-user/no-tags. Changed-format scores refer to original semantic roles
and are not formal role accuracy. Numeric plot sidecars contain only positions,
offsets, role IDs, and probabilities, not raw source text. Figures intentionally
show the requested full visible context.

For CPU consumption, install NumPy/SciPy, then run:

```bash
python examples/load_probe.py /path/to/actual-selected-head-features.npy
```

Input columns must be the sixteen `metadata.heads` in fixed order, each 64
dimensions before o_proj, from the exact frozen model/tokenizer. A matching shape
alone does not establish feature provenance. Do not reorder heads or pre-apply
standardization. The full stage implementation and pinned requirements are in
`code/`; reproducing extraction needs the official model weights and four A40s.
The archived budget/deadline records are historical authorization limits, not a
fresh authorization to restart compute. GPU monitoring CSV/summary is delivered
separately by the task owner.
'''


def export(args):
    require(all(not args.output_dir.is_relative_to(source) and not source.is_relative_to(args.output_dir)
                for source in [args.run_dir, args.data_dir, args.code_dir]),
            "Delivery must be an independent directory, outside the source run/data/code trees")
    exporter = Exporter(args)
    manifest = exporter.build()
    destination = args.output_dir
    if destination.exists():
        existing_path = destination / "bundle-manifest.json"
        require(existing_path.is_file() and (destination / "EXPORT_COMPLETE.json").is_file(), "Destination exists without a completed export; do not overwrite it")
        existing = read(existing_path)
        completion = read(destination / "EXPORT_COMPLETE.json")
        require(existing.get("complete") is True and completion.get("status") == "complete" and
                completion.get("bundle_manifest_sha256") == sha(existing_path) and
                completion.get("sha256sums_sha256") == sha(destination / "SHA256SUMS"), "Existing delivery completion/SHA record is invalid")
        require(existing["protocol_bindings"] == manifest["protocol_bindings"] and existing["files"] == manifest["files"], "Existing delivery belongs to different source artifacts")
        actual_names = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
        require(actual_names == set(existing["files"]) | {"bundle-manifest.json", "SHA256SUMS", "EXPORT_COMPLETE.json"},
                "Existing delivery contains files outside its explicit whitelist")
        for name, entry in existing["files"].items():
            require(sha(bounded(destination, name)) == entry["sha256"], "An already exported delivery artifact changed")
        return {"status": "already_exported", "output_dir": str(destination), "files": len(existing["files"])}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".export-" + uuid.uuid4().hex + ".tmp")
    temporary.mkdir()
    try:
        for name, entry in manifest["files"].items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry["kind"] == "verbatim":
                source = Path(entry["source_path"])
                require(sha(source) == entry["sha256"], "A source artifact changed before copying")
                shutil.copyfile(source, target)  # FUSE-safe: do not clone mtime/mode/xattrs.
            else:
                target.write_bytes(exporter.derived[name])
            require(sha(target) == entry["sha256"], "Exported bytes differ from their source/derived checksum")
        for entry in manifest["files"].values():
            if entry["kind"] == "verbatim":
                require(sha(entry["source_path"]) == entry["sha256"], "A source artifact changed during export")
        for retained in manifest["remote_retained"]:
            if "sha256" in retained:
                require(sha(retained["path"]) == retained["sha256"], "A retained source proof changed during export")
        (temporary / "bundle-manifest.json").write_bytes(encoded(manifest))
        lines = [f"{entry['sha256']}  {name}" for name, entry in sorted(manifest["files"].items())]
        lines.append(f"{sha(temporary / 'bundle-manifest.json')}  bundle-manifest.json")
        (temporary / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        completion = {"format_version": VERSION, "status": "complete", "files": len(manifest["files"]),
                      "bundle_manifest_sha256": sha(temporary / "bundle-manifest.json"),
                      "sha256sums_sha256": sha(temporary / "SHA256SUMS"), "probe_sha256": exporter.frozen["probe_sha256"]}
        (temporary / "EXPORT_COMPLETE.json").write_bytes(encoded(completion))
        temporary.replace(destination)
        return {**completion, "status": "exported", "output_dir": str(destination), "files": len(manifest["files"]),
                "bytes": sum(entry["bytes"] for entry in manifest["files"].values())}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/role-retraining-20261026/delivery-v7"))
    parser.add_argument("--requirements-file", type=Path)
    parser.add_argument("--environment-file", type=Path)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.requirements_file = args.requirements_file or args.code_dir / "requirements-synthetic-agent-4xa40.txt"
    return args


def main():
    try:
        print(json.dumps(export(parse_args()), indent=2), flush=True)
    except (NotReady, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
