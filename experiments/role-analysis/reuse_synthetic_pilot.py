"""Reuse a completed frozen pilot after a documented article-filter revision.

This is cache reuse, never retraining. FIT/DEV input bytes and every source's
identity, full raw/normalized text hashes, split, and ordinal must be identical.
Only committed pilot artifacts are copied; model weights, source articles,
head-scan checkpoints, and campaign receipts remain in their original locations.
No GPU, old-stage process, or training function is started by this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata
import uuid
from urllib.parse import urlsplit, urlunsplit

import numpy as np

VERSION = "synthetic-pilot-byte-identical-reuse-v1"
DEFAULT_REVISION = "Documented source-filter/URL-deduplication revision: strictly identical pilot inputs reuse the completed frozen pilot cache; no retraining is claimed"
ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]
CANDIDATES = ["linear-seed17", "mlp128-seed17", "mlp128-seed29", "mlp128-seed43",
              "mlp256-seed17", "mlp256-seed29", "mlp256-seed43"]
ROOT_FILES = ["config.json", "preflight.json", "model_backend.json", "head-selection.json",
              "training.json", "selection.json", "freeze.json", "best.npz", "standardizer.npz",
              "resume-validation.json", "scan-stage.json", "extract-stage.json", "train-stage.json",
              "freeze-stage.json", "evaluate-stage.json"]
PREPARATION_FILES = {"config.json", "data-pilot-manifest.json", "prepare-stage.json", "status.json"}
FEATURE_FILES = {"x.npy", "y.npy", "structure.npy", "source.npy", "family.npy", "weight.npy", "groups.npy"}


class Refused(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Refused(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalized_url(value):
    parts = urlsplit(value.strip())
    require(bool(parts.netloc), "An available normalized source URL has no host")
    return urlunsplit(("https", parts.netloc.lower().removeprefix("www."),
                       re.sub("/+", "/", parts.path).rstrip("/"), "", ""))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact points outside its committed parent directory")
    return path


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pilot_data(data_dir, code_dir):
    folder = Path(data_dir) / "pilot"
    manifest_path = folder / "data-manifest.json"
    require(manifest_path.is_file(), f"Pilot data is not committed: {manifest_path}")
    manifest = read(manifest_path)
    require(manifest["phase"] == "pilot" and manifest["roles"] == ROLES and not manifest.get("test_opened"),
            "Only the article-only pilot FIT/DEV cache may be reused")
    require(set(manifest["statistics"]) == {"fit", "dev"}, "Pilot cache contains unexpected splits")
    builder_sha = sha(Path(code_dir) / "synthetic_agent_data.py")
    require(manifest["configuration"]["code_sha256"] == builder_sha, "Pilot cache builder SHA differs from its declared code directory")
    for name, expected in manifest["sha256"].items():
        require(sha(safe_path(folder, name)) == expected, f"Prepared pilot file changed: {name}")
    for split, count in [("fit", 512), ("dev", 128)]:
        require(manifest["statistics"][split]["sources"] == count, "Pilot source quota differs from the planned 512/128")
    sources, urls = {}, {}
    actual_splits, url_counts = Counter(), Counter()
    for row in jsonl(folder / "sources.jsonl"):
        require(row["split"] in {"fit", "dev"}, "A pilot source belongs to TEST")
        sid = row["source_id"]
        require(sid not in sources, "Duplicate pilot source_id")
        raw_hash = text_sha(row["text"])
        normalized_hash = text_sha(normalized_text(row["text"]))
        require(row["normalized_sha256"] == normalized_hash, "A stored normalized article hash is inconsistent")
        sources[sid] = {"source_id": sid, "cluster_id": row["cluster_id"], "split": row["split"],
                        "source": row["source"], "source_group": row.get("source_group"),
                        "source_ordinal": row["source_ordinal"], "raw_text_sha256": raw_hash,
                        "normalized_text_sha256": normalized_hash, "normalized_sha256": row["normalized_sha256"]}
        actual_splits[row["split"]] += 1
        url = row.get("normalized_url")
        key = f"{row['source']}/{row['split']}"
        if url:
            require(isinstance(url, str) and normalized_url(url) == url, "Stored normalized source URL is not canonical")
            require(url not in urls, "Duplicate available normalized URL within/across pilot source splits")
            urls[url] = sid
            url_counts[key + "/available"] += 1
        else:
            url_counts[key + "/unavailable"] += 1
    require(len(sources) == 640, "The complete pilot source set must contain 640 articles")
    require(dict(actual_splits) == {"fit": 512, "dev": 128}, "Actual pilot source rows do not have the planned 512/128 split counts")
    return {"manifest": manifest, "manifest_sha256": sha(manifest_path),
            "sources_sha256": sha(folder / "sources.jsonl"), "builder_sha256": builder_sha,
            "input_sha256": {split: sha(folder / f"inputs-{split}.jsonl") for split in ["fit", "dev"]},
            "sources": sources, "source_equivalence_sha256": object_sha([sources[key] for key in sorted(sources)]),
            "url_integrity": {"available_urls": len(urls), "unavailable_urls": len(sources) - len(urls),
                              "all_available_urls_unique_across_splits": True,
                              "available_url_set_sha256": object_sha(sorted(urls)), "counts": dict(sorted(url_counts.items()))}}


def input_equivalence(old, new):
    require(old["input_sha256"] == new["input_sha256"], "Old/new pilot FIT/DEV input JSONL bytes differ")
    require(set(old["sources"]) == set(new["sources"]), "Old/new pilot source_id sets differ")
    differences = [sid for sid in old["sources"] if old["sources"][sid] != new["sources"][sid]]
    require(not differences, "Pilot source raw/normalized text, cluster, split, source group, or ordinal differs")
    require(old["source_equivalence_sha256"] == new["source_equivalence_sha256"], "Source equivalence digest differs")
    old_tokenizer = old["manifest"]["configuration"].get("tokenizer_sha256")
    new_tokenizer = new["manifest"]["configuration"].get("tokenizer_sha256")
    require(old_tokenizer and old_tokenizer == new_tokenizer, "Old/new pilot tokenizer backend hashes differ")
    return {"fit_dev_input_bytes_identical": True, "input_sha256": old["input_sha256"],
            "source_count": len(old["sources"]), "source_equivalence_sha256": old["source_equivalence_sha256"],
            "tokenizer_sha256": old_tokenizer,
            "compared_source_fields": list(next(iter(old["sources"].values()))),
            "parent_data_manifest_sha256": old["manifest_sha256"], "child_data_manifest_sha256": new["manifest_sha256"],
            "parent_sources_sha256": old["sources_sha256"], "child_sources_sha256": new["sources_sha256"],
            "parent_builder_sha256": old["builder_sha256"], "child_builder_sha256": new["builder_sha256"],
            "parent_url_integrity": old["url_integrity"], "child_url_integrity": new["url_integrity"],
            "url_metadata_byte_identity_required": False}


def readiness(run_dir, include_plots=False):
    root = Path(run_dir)
    status_path = root / "status.json"
    if not status_path.exists():
        return ["missing committed old pilot status.json"]
    status = read(status_path)
    if status.get("status") == "running":
        # Do not open mutable feature manifests, checkpoints, or model exports.
        return ["old pilot stage is still running: " + str(status.get("stage"))]
    if status.get("status") != "complete":
        return ["old pilot status is not complete: " + str(status.get("status"))]
    names = ROOT_FILES + ["features/pilot/manifest.json", "evaluations/dev.json", "evaluations/dev-probabilities.npy"]
    for candidate in CANDIDATES:
        names += [f"candidates/{candidate}/{name}" for name in ["complete.json", "best.npz", "history.json"]]
    if include_plots:
        names.append("plots/dev/index.json")
    return ["missing committed pilot artifact: " + name for name in names if not (root / name).is_file()]


def current_runtime(args, gpu):
    from transformers import AutoTokenizer
    binding = gpu.model_binding(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    require(hasattr(tokenizer, "backend_tokenizer"), "The actual model tokenizer has no stable backend serialization")
    return binding, text_sha(tokenizer.backend_tokenizer.to_str())


def carry_audit(old_audit_path, old, new, equivalence):
    audit = read(old_audit_path)
    require(audit.get("status") == "passed", "The original article audit did not pass")
    require(audit["data_manifest_sha256"] == old["manifest_sha256"] and audit["sources_sha256"] == old["sources_sha256"],
            "The original article audit is not bound to the verified parent source cache")
    require(audit["builder_sha256"] == old["builder_sha256"], "The original audit builder SHA changed")
    sample_ids = audit.get("sample_ids", [])
    reviews = audit.get("sample_reviews", [])
    require(sample_ids and len(set(sample_ids)) == len(sample_ids) == audit.get("full_articles_read") and
            set(sample_ids).issubset(old["sources"]), "Original sampled article review scope is incomplete")
    require({row["source_id"] for row in reviews} == set(sample_ids), "Original review records do not cover the declared sample IDs")
    for row in reviews:
        source = old["sources"][row["source_id"]]
        require(row["split"] == source["split"] and row["source"] == source["source"], "An inherited review source/split changed")
    return {"status": "passed", "scope": "pilot FIT/DEV only; no new formal TEST article reviewed",
            "auditor": {"type": "carried_forward_with_byte_identity", "original_auditor": audit.get("auditor"),
                        "new_article_reviews_performed": False},
            "data_manifest_sha256": new["manifest_sha256"], "sources_sha256": new["sources_sha256"],
            "builder_sha256": new["builder_sha256"], "parent_audit_sha256": sha(old_audit_path),
            "parent_data_manifest_sha256": old["manifest_sha256"], "parent_sources_sha256": old["sources_sha256"],
            "parent_builder_sha256": old["builder_sha256"], "equivalence": equivalence,
            "full_articles_previously_read": audit["full_articles_read"], "sample_ids": sample_ids,
            "inherited_sample_reviews": reviews, "selection_rule": audit.get("selection_rule"),
            "limitations": audit.get("limitations", []) + [
                "This carries the prior sampled audit; it is not a fresh review of all 640 articles.",
                "A source-filter revision changed the cache builder, while every pilot source and actual FIT/DEV input remained identical."]}


def validated_old_pilot(args, old, new, equivalence):
    root = args.old_run_dir
    require(read(root / "status.json").get("status") == "complete", "The parent pilot became active before model validation")
    status_sha = sha(root / "status.json")
    old_config = read(root / "config.json")
    new_config = read(args.new_config)
    require(new_config.get("phase", "pilot") == "pilot", "New runtime configuration is not a pilot configuration")
    new_config["phase"] = "pilot"
    new_config.setdefault("classifier", {})["max_epochs"] = 15
    require(old_config == new_config, "Old/new pilot runtime configuration differs")
    code_hashes = {}
    for name in ["synthetic_agent_gpu.py", "synthetic_agent_classifier.py"]:
        parent_sha, child_sha = sha(args.old_code_dir / name), sha(args.new_code_dir / name)
        require(parent_sha == child_sha, f"Frozen runtime implementation changed: {name}")
        code_hashes[name] = parent_sha
    classifier = module_from_file(args.old_code_dir / "synthetic_agent_classifier.py", "_reuse_parent_classifier")
    gpu = module_from_file(args.old_code_dir / "synthetic_agent_gpu.py", "_reuse_parent_gpu")
    actual_binding, actual_tokenizer = current_runtime(args, gpu)
    require(actual_tokenizer == equivalence["tokenizer_sha256"], "Current model tokenizer differs from the verified pilot tokenizer")
    preflight, backend = read(root / "preflight.json"), read(root / "model_backend.json")
    require(preflight["status"] == "passed" and preflight["maximum_input_tokens"] == 4096 and
            preflight["model_binding"] == backend["model_binding"] == actual_binding, "Model/preflight/tokenizer binding changed")
    require(backend["expert_backends"] == ["Mxfp4GptOssExperts"], "Parent model backend was not native MXFP4")
    selected = read(root / "head-selection.json")
    signature = selected["signature"]
    require(selected["roles"] == ROLES and len(selected["heads"]) == len(set(selected["heads"])) == 16,
            "Invalid ordered six-role head selection")
    require(signature["model_binding"] == actual_binding and signature["pilot_fit_sha256"] == old["input_sha256"]["fit"] and
            signature["implementation_sha256"] == code_hashes["synthetic_agent_gpu.py"] and
            signature["config_sha256"] == gpu.object_digest(old_config), "Head-selection signature changed")
    require(selected.get("no_dev_test_or_historical_probe_used") is True, "Head-discovery data scope is not recorded")
    if (root / "head-scan" / "progress.json").exists():
        scan_state = read(root / "head-scan" / "progress.json")
        require(scan_state.get("phase") == "complete" and scan_state["signature"] == signature and scan_state["heads"] == selected["heads"],
                "Parent all-head scan did not commit the selected heads")
    if (root / "head-scan" / "plan.json").exists():
        plan = read(root / "head-scan" / "plan.json")
        require(plan["signature"] == signature and gpu.object_digest({key: value for key, value in plan.items() if key != "signature"}) == signature["scan_plan_sha256"],
                "Head-scan plan no longer matches its selection signature")
    feature_dir = root / "features" / "pilot"
    feature_manifest = read(feature_dir / "manifest.json")
    require(set(feature_manifest["splits"]) == {"fit", "dev"}, "The parent pilot feature cache contains an unexpected split")
    require(feature_manifest["signature"]["implementation_sha256"] == code_hashes["synthetic_agent_gpu.py"], "Feature implementation SHA changed")
    banks = {}
    for split in ["fit", "dev"]:
        bank = classifier.FeatureBank(feature_dir, split)
        require(bank.manifest["model_binding"] == actual_binding and bank.manifest["heads"] == selected["heads"] and
                bank.manifest["selection_sha256"] == sha(root / "head-selection.json") and
                bank.manifest["pilot_fit_sha256"] == old["input_sha256"]["fit"] and
                bank.manifest["config_hash"] == gpu.object_digest(old_config) and
                bank.manifest["data_hash"] == gpu.object_digest(old["input_sha256"]), "Feature binding changed")
        require(bank.manifest["split_sources"][split] == old["input_sha256"][split], "Feature inputs differ from the byte-identical pilot inputs")
        clusters = {row["cluster_id"] for row in old["sources"].values() if row["split"] == split}
        require(set(bank.source.astype(str)) == clusters, "Feature bank source clusters do not match the full source allocation")
        cursor = 0
        for record in jsonl(args.old_data_dir / "pilot" / f"inputs-{split}.jsonl"):
            targets = record["targets"]
            stop = cursor + len(targets)
            require(np.array_equal(bank.y[cursor:stop], [row["label"] for row in targets]), "Feature/body target labels are misaligned")
            require(np.all(bank.source[cursor:stop].astype(str) == str(record["cluster_id"])), "Feature/body source clusters are misaligned")
            require(np.array_equal(bank.family[cursor:stop].astype(str), [str(row["family_id"]) for row in targets]), "Feature/body families are misaligned")
            cursor = stop
        require(cursor == len(bank), "Feature row count differs from the input body mask")
        for start in range(0, len(bank), 8192):
            bank.read(np.arange(start, min(start + 8192, len(bank))))  # checks all FP16 feature values.
        banks[split] = bank
    require(not np.intersect1d(banks["fit"].source, banks["dev"].source).size and
            not np.intersect1d(banks["fit"].family, banks["dev"].family).size, "FIT/DEV leakage in the completed parent feature banks")
    training = read(root / "training.json")
    options = classifier._training_options(old_config)
    require(training["phase"] == "pilot" and training["implementation_sha256"] == code_hashes["synthetic_agent_classifier.py"] and
            training["options"] == options, "Training runtime/configuration changed")
    split_hashes = {split: bank.split_identity_hash for split, bank in banks.items()}
    expected_hash = object_sha({"fit_features": split_hashes["fit"], "dev_features": split_hashes["dev"],
        "options": options, "implementation_sha256": code_hashes["synthetic_agent_classifier.py"], "candidates": classifier.CANDIDATES})
    require(training["feature_split_hashes"] == split_hashes and training["training_hash"] == expected_hash,
            "Training does not bind the complete committed feature shards")
    require(training["binding"] == banks["fit"].binding(), "Training feature/model/head binding changed")
    recovery = read(root / "resume-validation.json")
    require(recovery.get("passed") is True and recovery["device"] == options["device"] and
            recovery["weight_max_absolute_error"] <= 1e-7 and recovery["loss_absolute_error"] <= 1e-7 and
            recovery["numpy_export_max_absolute_error"] <= 3e-5, "Parent recovery/export engineering validation failed")
    with np.load(root / "standardizer.npz", allow_pickle=False) as archive:
        require(str(archive["training_hash"]) == expected_hash, "FIT standardizer belongs to different training")
        mean, scale = archive["mean"].copy(), archive["scale"].copy()
    candidates = []
    for candidate_id, hidden, seed in classifier.CANDIDATES:
        directory = root / "candidates" / candidate_id
        record = read(directory / "complete.json")
        require(record["training_hash"] == expected_hash and record["probe_sha256"] == sha(directory / "best.npz") and
                record["history_sha256"] == sha(directory / "history.json"), "A completed candidate changed")
        require(record["epochs"] >= 10 and record["epochs"] <= 15 and record["elapsed_seconds"] > 0 and
                record["export_max_absolute_error"] <= 3e-5, "Candidate timing/epochs/export evidence is invalid")
        probe = classifier.load_probe(directory / "best.npz", expected_binding=training["binding"])
        require(probe.metadata["candidate_id"] == candidate_id and probe.metadata["hidden"] == hidden and
                probe.metadata["seed"] == seed and probe.metadata["training_hash"] == expected_hash and
                not probe.metadata.get("engineering_smoke_only"), "A candidate export does not match its committed metadata")
        require(np.array_equal(probe.mean, mean) and np.array_equal(probe.scale, scale), "Candidate export standardizer changed")
        candidates.append(record)
    selection = read(root / "selection.json")
    winner = min(candidates, key=lambda row: classifier.selection_key(row["dev"], row["candidate_id"]))
    require(selection["candidate_results"] == candidates and selection["candidate_id"] == winner["candidate_id"] and
            selection["training_hash"] == expected_hash and selection["selection_split"] == "dev", "The fixed DEV selection differs from the seven candidate commits")
    frozen = read(root / "freeze.json")
    require(frozen["probe_sha256"] == sha(root / "best.npz") == winner["probe_sha256"] and
            frozen["selection_sha256"] == sha(root / "selection.json") and
            frozen["training_sha256"] == sha(root / "training.json") and frozen["selection_split"] == "dev",
            "Frozen pilot artifact binding changed")
    probe = classifier.load_probe(root / "best.npz", expected_binding=training["binding"])
    require(probe.metadata == frozen["metadata"], "Frozen NPZ metadata differs from the freeze manifest")
    dev = banks["dev"]
    probabilities_path = root / "evaluations" / "dev-probabilities.npy"
    probabilities = np.load(probabilities_path, mmap_mode="r", allow_pickle=False)
    require(probabilities.shape == (len(dev), 6) and np.isfinite(probabilities).all() and
            np.allclose(probabilities.sum(1), 1, atol=3e-5), "Committed DEV probabilities are incomplete")
    evaluation = read(root / "evaluations" / "dev.json")
    require(evaluation["split"] == "dev" and evaluation["probe_sha256"] == frozen["probe_sha256"] and
            evaluation["feature_manifest_sha256"] in {sha(feature_dir / "manifest.json"), training.get("feature_manifest_hash")},
            "DEV evaluation provenance changed")
    recomputed = classifier.grouped_metrics(dev, probabilities)
    require(all(evaluation.get(key) == value for key, value in recomputed.items()), "DEV metrics disagree with the committed probabilities/body labels")
    maximum_error = 0.
    changed_argmax = 0
    for start in range(0, len(dev), 8192):
        stop = min(start + 8192, len(dev))
        exported = probe.predict_proba(dev.read(np.arange(start, stop)))
        reference = np.asarray(probabilities[start:stop])
        maximum_error = max(maximum_error, float(np.abs(exported - reference).max()))
        changed_argmax += int(np.count_nonzero(exported.argmax(1) != reference.argmax(1)))
    require(maximum_error <= 3e-5, "Frozen CPU NPZ consumer disagrees with the actual saved GPU DEV probabilities")
    config_marker_hash = hashlib.sha256(json.dumps(old_config, sort_keys=True).encode()).hexdigest()
    for stage in ["scan", "extract", "train", "freeze", "evaluate"]:
        marker = read(root / f"{stage}-stage.json")
        require(marker["status"] == "complete" and marker["stage"] == stage and marker["phase"] == "pilot" and
                marker["config_hash"] == config_marker_hash, "A required completed stage marker changed")
    require(read(root / "status.json")["status"] == "complete" and sha(root / "status.json") == status_sha,
            "The old pilot resumed writing during migration validation")
    return {"runtime_config_sha256": object_sha(old_config), "runtime_source_sha256": code_hashes,
            "model_binding": actual_binding, "heads": selected["heads"], "feature_split_identity": split_hashes,
            "training_hash": expected_hash, "probe_sha256": frozen["probe_sha256"],
            "head_selection_sha256": sha(root / "head-selection.json"), "freeze_sha256": sha(root / "freeze.json"),
            "dev_probability_sha256": sha(probabilities_path), "dev_export_max_absolute_error": maximum_error,
            "dev_cpu_gpu_argmax_differences_within_probability_tolerance": changed_argmax,
            "parent_status_sha256": status_sha,
            "actual_parent_candidate_timings": {row["candidate_id"]: {"epochs": row["epochs"], "elapsed_seconds": row["elapsed_seconds"]} for row in candidates},
            "actual_parent_extraction_seconds": read(root / "extract-stage.json")["seconds"]}


def copy_plan(args):
    root = args.old_run_dir
    names = set(ROOT_FILES)
    for optional in ["preflight-stage.json", "head-scan/plan.json", "head-scan/progress.json"]:
        if (root / optional).is_file():
            names.add(optional)
    for candidate in CANDIDATES:
        for name in ["best.npz", "complete.json", "history.json"]:
            names.add(f"candidates/{candidate}/{name}")
    feature_root = root / "features" / "pilot"
    manifest = read(feature_root / "manifest.json")
    names.add("features/pilot/manifest.json")
    if (feature_root / "selected-hook-validation.json").exists():
        names.add("features/pilot/selected-hook-validation.json")
    for entries in manifest["splits"].values():
        for entry in entries:
            directory = safe_path(feature_root, entry["path"])
            commit = read(directory / "complete.json")
            require(set(commit["files"]).issubset(FEATURE_FILES) and FEATURE_FILES - {"groups.npy"} <= set(commit["files"]),
                    "Feature commit contains unknown files or lacks required arrays; never copy raw/debug data")
            for name in ["complete.json", *commit["files"]]:
                source = safe_path(directory, name)
                names.add(str(source.relative_to(root)))
    # The completed DEV evaluation only; never copy another final-test scope.
    names.update(["evaluations/dev.json", "evaluations/dev-probabilities.npy"])
    if args.include_plots:
        plot_root = root / "plots" / "dev"
        index = read(plot_root / "index.json")
        require(not index["missing_coverage"], "Requested parent plots do not cover all pilot structures")
        require(index["probe_sha256"] == read(root / "freeze.json")["probe_sha256"], "Requested parent plots use a different probe")
        names.add("plots/dev/index.json")
        for entry in index["examples"].values():
            for extension in ["png", "pdf"]:
                path = safe_path(plot_root, entry[extension])
                names.add(str(path.relative_to(root)))
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    names.add(str(sidecar.relative_to(root)))
        if (root / "plot-stage.json").exists():
            names.add("plot-stage.json")
    result = {}
    for name in sorted(names):
        path = safe_path(root, name)
        require(path.is_file() and not (root / name).is_symlink(), "Only committed regular artifact files may be copied")
        result[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def transactional_copy(args, files, new, migration, carried_audit):
    destination = args.new_run_dir
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".reuse-" + uuid.uuid4().hex + ".tmp")
    backup = None
    committed = False
    require(not destination.exists() or destination.is_dir(), "New run path is not a directory")
    if destination.exists():
        allowed = {path.name for path in destination.iterdir()}
        require(allowed.issubset(PREPARATION_FILES), "New run already contains training/extraction artifacts; do not overwrite them")
        require(all(path.is_file() and not path.is_symlink() for path in destination.iterdir()), "Unexpected new-run preparation artifact")
        if (destination / "status.json").exists():
            require(read(destination / "status.json").get("status") != "running", "The new pilot preparation is still writing")
        if (destination / "config.json").exists():
            require(read(destination / "config.json") == read(args.old_run_dir / "config.json"), "New pilot's existing runtime config differs")
        if (destination / "data-pilot-manifest.json").exists():
            require(sha(destination / "data-pilot-manifest.json") == new["manifest_sha256"], "Existing new-data manifest mirror differs")
    audit_path = args.new_data_dir / "pilot" / "article-audit.json"
    audit_was_present = audit_path.exists()
    if audit_path.exists():
        require(read(audit_path) == carried_audit, "New pilot audit already has different provenance; do not overwrite it")
    stage.mkdir()
    try:
        for name, expected in files.items():
            source, target = safe_path(args.old_run_dir, name), stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            require(sha(source) == expected["sha256"], "A parent artifact changed before its copy")
            # FUSE may permit byte writes while rejecting utime/chmod in
            # copy2/copystat. Scientific identity binds content hashes, not
            # filesystem metadata; genuine content-copy errors still propagate.
            shutil.copyfile(source, target)
            require(sha(target) == expected["sha256"], "A copied artifact is not byte-identical")
        if destination.exists():
            for name in ["prepare-stage.json", "status.json"]:
                if (destination / name).exists():
                    shutil.copyfile(destination / name, stage / name)
        # This is the sole canonical data mirror that intentionally changes.
        shutil.copyfile(args.new_data_dir / "pilot" / "data-manifest.json", stage / "data-pilot-manifest.json")
        require(sha(stage / "data-pilot-manifest.json") == new["manifest_sha256"], "New pilot data mirror differs")
        if not (stage / "status.json").exists():
            atomic_json(stage / "status.json", {"stage": "pilot_cache_reuse", "phase": "pilot", "status": "complete",
                                                "new_training_performed": False, "parent_run_dir": str(args.old_run_dir)})
        require(sha(args.old_run_dir / "status.json") == migration["runtime_equivalence"]["parent_status_sha256"] and
                read(args.old_run_dir / "status.json")["status"] == "complete", "Old pilot changed state during the copy")
        for name, expected in files.items():
            require(sha(args.old_run_dir / name) == expected["sha256"], "A parent artifact changed during the copy")
        for data_dir, expected_manifest, expected_sources in [
            (args.old_data_dir, migration["input_equivalence"]["parent_data_manifest_sha256"], migration["input_equivalence"]["parent_sources_sha256"]),
            (args.new_data_dir, new["manifest_sha256"], new["sources_sha256"])]:
            require(sha(data_dir / "pilot" / "data-manifest.json") == expected_manifest and
                    sha(data_dir / "pilot" / "sources.jsonl") == expected_sources,
                    "A verified pilot source cache changed during the copy")
            for split, expected in migration["input_equivalence"]["input_sha256"].items():
                require(sha(data_dir / "pilot" / f"inputs-{split}.jsonl") == expected,
                        "A verified pilot input JSONL changed during the copy")
        for code_dir in [args.old_code_dir, args.new_code_dir]:
            for name, expected in migration["runtime_equivalence"]["runtime_source_sha256"].items():
                require(sha(code_dir / name) == expected, "Frozen runtime code changed during the copy")
        migration["child_files"] = {name: {"sha256": sha(stage / name), "bytes": (stage / name).stat().st_size} for name in files}
        require(migration["child_files"] == migration["parent_files"], "Some immutable pilot artifacts changed in migration")
        migration["child_data_pilot_manifest_sha256"] = sha(stage / "data-pilot-manifest.json")
        migration["child_article_audit_sha256"] = text_sha(json.dumps(carried_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_json(stage / "migration.json", migration)
        if destination.exists():
            backup = destination.with_name(destination.name + ".pre-reuse-" + uuid.uuid4().hex + ".tmp")
            destination.replace(backup)
        stage.replace(destination)
        committed = True
        atomic_json(audit_path, carried_audit)
        require(sha(audit_path) == migration["child_article_audit_sha256"], "Carried audit bytes differ from their migration binding")
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        if committed and destination.exists():
            shutil.rmtree(destination)
        if backup and backup.exists() and not destination.exists():
            backup.replace(destination)
        if not audit_was_present and audit_path.exists() and "child_article_audit_sha256" in migration and sha(audit_path) == migration["child_article_audit_sha256"]:
            audit_path.unlink()
        raise
    # The child and its carried audit are fully committed now. Cleanup failure
    # must never delete the valid child and restore a half-deleted backup.
    if backup:
        try:
            shutil.rmtree(backup)
        except OSError as error:
            warning = {"backup_path": str(backup), "error": str(error), "committed_child_preserved": True}
            migration["backup_cleanup_warning"] = warning
            try:
                atomic_json(destination / "migration.json", migration)
            except OSError as record_error:
                warning["warning_record_error"] = str(record_error)
            return warning
    return None


def reuse(args):
    require(args.old_run_dir != args.new_run_dir and not args.new_run_dir.is_relative_to(args.old_run_dir) and
            not args.old_run_dir.is_relative_to(args.new_run_dir), "Parent and child run directories must be distinct independent directories")
    old = pilot_data(args.old_data_dir, args.old_code_dir)
    new = pilot_data(args.new_data_dir, args.new_code_dir)
    equivalence = input_equivalence(old, new)
    carried = carry_audit(args.old_audit_proof, old, new, equivalence)
    report = {"format_version": VERSION, "dry_run": args.dry_run, "input_equivalence": equivalence,
              "new_training_performed": False, "parent_run_dir": str(args.old_run_dir), "child_run_dir": str(args.new_run_dir)}
    pending = readiness(args.old_run_dir, args.include_plots)
    if pending:
        return {**report, "status": "inputs_equivalent_waiting_completion", "refusal_reasons": pending}
    runtime = validated_old_pilot(args, old, new, equivalence)
    files = copy_plan(args)
    report.update(status="ready_to_reuse", runtime_equivalence=runtime, committed_files=len(files),
                  committed_bytes=sum(row["bytes"] for row in files.values()), refusal_reasons=[])
    if args.dry_run:
        return report
    if (args.new_run_dir / "migration.json").exists():
        existing = read(args.new_run_dir / "migration.json")
        require(existing["input_equivalence"] == equivalence and existing["runtime_equivalence"] == runtime and
                existing["parent_files"] == files, "An existing migration belongs to different parents or inputs")
        for name, expected in files.items():
            require(sha(safe_path(args.new_run_dir, name)) == expected["sha256"], "An already migrated child artifact changed")
        require(sha(args.new_run_dir / "data-pilot-manifest.json") == new["manifest_sha256"], "Migrated new-data mirror changed")
        audit_path = args.new_data_dir / "pilot" / "article-audit.json"
        if not audit_path.exists():
            atomic_json(audit_path, carried)
        require(read(audit_path) == carried, "Migrated article audit changed")
        return {**report, "status": "already_reused", "migration_path": str(args.new_run_dir / "migration.json")}
    migration = {"format_version": VERSION, "created_unix": time.time(), "implementation_sha256": sha(__file__),
                 "protocol_revision": getattr(args, "protocol_revision", DEFAULT_REVISION),
                 "new_training_performed": False, "new_feature_extraction_performed": False,
                 "parent_run_dir": str(args.old_run_dir), "child_run_dir": str(args.new_run_dir),
                 "parent_data_dir": str(args.old_data_dir), "child_data_dir": str(args.new_data_dir),
                 "parent_code_dir": str(args.old_code_dir), "child_code_dir": str(args.new_code_dir),
                 "parent_audit_path": str(args.old_audit_proof), "parent_audit_sha256": sha(args.old_audit_proof),
                 "input_equivalence": equivalence, "runtime_equivalence": runtime, "parent_files": files,
                 "unchanged_artifact_policy": "All copied feature/candidate/NPZ/training/freeze/evaluation/stage-marker bytes remain unchanged, including original actual benchmark timings",
                 "filesystem_metadata_policy": "Content-only copies with SHA256 verification; FUSE timestamps/modes/xattrs are not cloned. Original measured timings inside committed JSON remain byte-identical.",
                 "absolute_parent_paths_preserved": "Immutable metadata may retain original feature/cache paths; the parent run remains intact for provenance",
                 "excluded": ["source articles", "model weights", "head-scan checkpoints", "last.pt", "campaign receipts", "old data-pilot-manifest mirror", "old reports"]}
    if args.old_audit_report:
        migration["parent_audit_report_sha256"] = sha(args.old_audit_report)
    warning = transactional_copy(args, files, new, migration, carried)
    return {**report, "status": "reused", "migration_path": str(args.new_run_dir / "migration.json"), "backup_cleanup_warning": warning}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["old-run-dir", "new-run-dir", "old-data-dir", "new-data-dir", "old-code-dir", "new-code-dir", "model-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--new-config", type=Path)
    parser.add_argument("--old-audit-proof", type=Path)
    parser.add_argument("--old-audit-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-plots", action="store_true")
    parser.add_argument("--protocol-revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.old_audit_proof = args.old_audit_proof or args.old_data_dir / "pilot" / "article-audit.json"
    args.new_config = args.new_config or args.new_code_dir / "synthetic-agent-120b.json"
    return args


def main():
    args = parse_args()
    try:
        if args.dry_run:
            result = reuse(args)
        else:
            args.new_run_dir.parent.mkdir(parents=True, exist_ok=True)
            lock_path = args.new_run_dir.with_name(args.new_run_dir.name + ".reuse.lock")
            with lock_path.open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise Refused("Another migration process owns this child run")
                result = reuse(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 3 if result["status"] == "inputs_equivalent_waiting_completion" else 0
    except (Refused, OSError, ValueError, KeyError) as error:
        print(json.dumps({"format_version": VERSION, "status": "refused", "dry_run": args.dry_run,
                          "new_training_performed": False, "refusal_reasons": [str(error)]}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
