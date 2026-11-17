"""CPU-only, independent validation and Chinese report for a DeepSeek role probe.

This module never imports the trainer, loads a backbone, invokes CUDA, or runs a
new TEST experiment. It recomputes statistics and metrics from committed files.
Missing completion artifacts produce status=running, never status=passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROLES = ("system", "user", "cot", "assistant", "tool")
DOMAINS = ("c4_dolma3", "agentdojo")
SPLITS = ("fit", "dev", "test")
CAPTURE = "attention-head-output-before-o_a_proj"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/deepseek-v4-flash-0731-role-probe-20261115"
REQUIRED = ("config.json", "data/manifest.json", "data/data-audit.json", "head-selection.json",
            "head-scan/plan.json", "head-scan/inner-confusion.npz", "head-scan/lda.npz",
            "model_backend.json", "preflight.json", "load-integrity-audit.json", "training.json", "selection.json",
            "freeze.json", "checkpoint.npz", "standardizer.npz", "features/manifest.json", "test-release.json",
            "evaluations/test-started.json", "evaluations/test-complete.json", "evaluations/test.json",
            "evaluations/test-probabilities.npy", "runtime-final.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, value):
    result = (Path(root) / value).resolve()
    require(result.is_relative_to(Path(root).resolve()), f"Path escapes artifact root: {value}")
    return result


def prefix_sha(path, count):
    """Verify the load auditor's log prefix without hashing later training logs."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while count:
            block = stream.read(min(count, 8 << 20))
            require(block, "A bound loading-log prefix was truncated")
            digest.update(block)
            count -= len(block)
    return digest.hexdigest()


def performance_summary(root):
    """Read report summaries only; never open or reproduce thousands of targets."""
    root = Path(root)
    applied_path = root / "performance-cache-applied.json"
    if not applied_path.exists():
        return None
    applied = read(applied_path)
    plan = applied.get("plan", {})
    sources = [{key: source.get(key) for key in ("projection", "source_tokens", "measured_candidates", "winner_ms", "sha256")}
               for source in plan.get("sources", {}).values()]
    result = {"status": applied.get("status"), "application": applied.get("application"),
              "planned_counts": plan.get("counts"), "unique_input_lengths": plan.get("unique_input_lengths"),
              "protected_tokens": plan.get("protected_tokens"), "measured_sources": sources,
              "target_lengths_individually_autotuned": False,
              "scope": "Reuse of original native kernel configurations; no new target-length autotuning timings are asserted"}
    validation_path = root / "performance-cache-gpu-validation.json"
    if validation_path.exists():
        validated = read(validation_path)
        gpu = validated.get("validation", {})
        checks = gpu.get("checks", [])
        result["gpu_validation"] = {"report_status": validated.get("status"), "status": gpu.get("status"),
             "scope": gpu.get("scope"), "cases": len(checks), "tokens": sorted({item["tokens"] for item in checks}),
             "bitexact_and_finite": bool(checks) and all(item.get("bitexact") is True and item.get("finite") is True for item in checks),
             "all_native_cache_hits": bool(checks) and all(item.get("native_disk_cache_hit") is True for item in checks),
             "search_called": any(item.get("search_called") is not False for item in checks),
             "peak_extra_allocated_bytes": gpu.get("peak_extra_allocated_bytes"),
             "allocator_limit_bytes": gpu.get("allocator_limit_bytes")}
    return result


def checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        return {"metadata": json.loads(str(archive["metadata"])), "mean": archive["mean"].copy(),
                "scale": archive["scale"].copy(),
                "weights": {name[8:]: archive[name].copy() for name in archive.files if name.startswith("weight::")}}


def feature_split(root, manifest, split):
    """Independent reader: verify every consumed array against shard commits."""
    entries = manifest["splits"][split]
    require(entries, f"Empty feature split: {split}")
    require(manifest.get("progress", {}).get(split, {}).get("complete") is True, f"Incomplete feature split: {split}")
    signature = {**manifest["signature"], "split": split, "input_sha256": manifest["split_sources"][split]}
    result = {"x": [], "y": [], "domain": [], "source": [], "family": []}
    for entry in entries:
        folder = safe_path(root, entry["path"])
        require(sha(folder / "complete.json") == entry["sha256"], f"Changed shard commit: {folder}")
        complete = read(folder / "complete.json")
        require(complete["signature"] == signature and complete["rows"] == entry["rows"], f"Wrong shard signature: {folder}")
        for name, digest in complete["files"].items():
            require(sha(safe_path(folder, name)) == digest, f"Changed feature array: {folder / name}")
        for name in result:
            require(name + ".npy" in complete["files"], f"Uncommitted consumed array: {name}")
            array = np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            if name == "x":
                require(array.dtype == np.float16 and array.shape == (entry["rows"], manifest["feature_dim"]), "Wrong feature shape/dtype")
                result[name].append(array)
            else:
                require(array.shape == (entry["rows"],), "Wrong sampling metadata shape")
                result[name].append(np.array(array))
    for key in ("y", "domain", "source", "family"):
        result[key] = np.concatenate(result[key])
    require(result["y"].dtype.kind in "iu", "Noninteger native role labels")
    require(np.all((result["y"] >= 0) & (result["y"] < len(manifest["roles"]))), "Invalid native role label")
    require(set(result["domain"]) == set(DOMAINS), "Unexpected feature domains")
    require(len(np.unique(result["family"])) == len(result["y"]), "Duplicate unique positions")
    require(sum(len(array) for array in result["x"]) == len(result["y"]), "Feature/label row count mismatch")
    return result


def population_standardizer(shards, dim, chunk_size=4096):
    """Independent two-pass F64 NumPy calculation, not the trainer's Welford code."""
    total = sum(len(array) for array in shards)
    require(total > 0, "No FIT positions")
    first = np.zeros(dim, np.float64)
    for array in shards:
        for start in range(0, len(array), chunk_size):
            values = np.asarray(array[start:start + chunk_size], dtype=np.float64)
            require(np.isfinite(values).all(), "Nonfinite FIT features")
            first += values.sum(axis=0)
    mean = first / total
    second = np.zeros(dim, np.float64)
    for array in shards:
        for start in range(0, len(array), chunk_size):
            delta = np.asarray(array[start:start + chunk_size], dtype=np.float64) - mean
            second += np.einsum("nd,nd->d", delta, delta)
    scale = np.sqrt(second / total)
    scale[scale < 1e-6] = 1
    return mean, scale


def recompute_metrics(probabilities, labels, roles):
    """Independent confusion/accuracy/F1/NLL implementation with integer counts."""
    p, y = np.asarray(probabilities), np.asarray(labels, dtype=np.int64)
    k = len(roles)
    require(p.shape == (len(y), k) and len(y) > 0, "TEST probability shape mismatch")
    require(np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(1), 1, atol=2e-5), "Invalid saved TEST probabilities")
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (y, p.argmax(1)), 1)
    support, predicted, diagonal = confusion.sum(1), confusion.sum(0), np.diag(confusion)
    f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros(k, dtype=np.float64), where=(support + predicted) > 0)
    result = {"tokens": len(y), "accuracy": float(diagonal.sum() / len(y)), "confusion": confusion.tolist(),
              "macro_f1": float(f1[support > 0].mean()),
              "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-30, 1)).mean()), "roles": {}}
    for index, role in enumerate(roles):
        result["roles"][role] = {"support": int(support[index]), "predicted": int(predicted[index]),
                                 "precision": float(diagonal[index] / predicted[index]) if predicted[index] else None,
                                 "recall": float(diagonal[index] / support[index]) if support[index] else None,
                                 "f1": float(f1[index]) if support[index] else None}
    return result


def compare_metrics(actual, saved):
    for key in ("tokens", "confusion"):
        require(actual[key] == saved[key], f"Recomputed TEST {key} differs from saved metrics")
    for key in ("accuracy", "macro_f1", "nll"):
        require(np.isclose(actual[key], saved[key], rtol=2e-7, atol=2e-8), f"Recomputed TEST {key} differs")
    for role, values in actual["roles"].items():
        for key, value in values.items():
            expected = saved["roles"][role][key]
            require(value == expected if value is None or expected is None or key in ("support", "predicted")
                    else bool(np.isclose(value, expected, rtol=2e-7, atol=2e-8)), f"Recomputed role metric differs: {role}/{key}")


def verify_model_audit(audit):
    for key in ("parameters_frozen", "all_parameters_on_cuda", "native_packed_fp4_fp8"):
        require(audit.get(key) is True, f"Missing native all-GPU audit guarantee: {key}")
    require(set(audit["parameter_bytes_by_device_dtype"]) == {f"cuda:{i}" for i in range(4)}, "Model audit does not cover exactly four GPUs")
    require(all(sum(values.values()) > 0 for values in audit["parameter_bytes_by_device_dtype"].values()), "An audited GPU has no model weights")
    require(audit["dtype_bytes"].get("torch.int8", 0) > 0 and audit["dtype_bytes"].get("torch.float8_e4m3fn", 0) > 0,
            "Audit lacks packed FP4 / FP8 weight storage")
    quantization = audit["quantization_config"]
    require(quantization.get("quant_method") == "fp8" and quantization.get("scale_fmt") == "ue8m0", "Native FP8 configuration changed")
    require(quantization.get("expert_dtype", "fp4") == "fp4" and quantization.get("dequantize", False) is False,
            "Audit reports expanded/dequantized experts")
    require(quantization.get("weight_block_size") == [128, 128], "Native FP8 block size changed")


def raw_target_metadata(path, roles):
    labels, domains, sources, families = [], [], [], []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            source = record.get("source_cluster", record.get("cluster_id", record.get("source_id")))
            for target in record["targets"]:
                value = target["label"]
                label = roles.index(value) if isinstance(value, str) else int(value)
                require(target.get("role", roles[label]) == roles[label], "Raw target role/index mismatch")
                labels.append(label)
                domains.append(record["domain"])
                sources.append(str(source))
                families.append(str(target.get("family", target.get("family_id", f"{record['id']}:{target['token_ix']}"))))
    return {"y": np.asarray(labels, np.int64), "domain": np.asarray(domains),
            "source": np.asarray(sources), "family": np.asarray(families)}


def sample_features(shards, count=257):
    remaining, pieces = count, []
    for shard in shards:
        if not remaining:
            break
        part = np.asarray(shard[:remaining], dtype=np.float32)
        pieces.append(part)
        remaining -= len(part)
    return np.concatenate(pieces)


def independent_probabilities(probe, values):
    z = (values.astype(np.float32) - probe["mean"]) / probe["scale"]
    weights = probe["weights"]
    z = z @ weights["0.weight"].T + weights["0.bias"]
    if probe["metadata"]["hidden"]:
        # Pure CPU scalar erf is sufficient for the small independent sample.
        erf = np.vectorize(math.erf, otypes=[np.float32])(z / np.float32(math.sqrt(2)))
        z = .5 * z * (1 + erf)
        z = z @ weights["3.weight"].T + weights["3.bias"]
    z -= z.max(1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(1, keepdims=True)


def validate_head_selection(root):
    """Recompute FIT-inner support and every head score without reading TEST."""
    root = Path(root)
    selected, plan = read(root / "head-selection.json"), read(root / "head-scan/plan.json")
    roles = tuple(selected["roles"])
    require(roles == ROLES, "Head-selection role order differs from the five native roles")
    n_heads, classes, width = 43 * 64, len(roles), 512
    with np.load(root / "head-scan/inner-confusion.npz", allow_pickle=False) as archive:
        confusion = archive["confusion"]
    require(confusion.shape == (n_heads, classes, classes) and confusion.dtype.kind in "iu" and
            np.all(confusion >= 0), "Expected nonnegative integer 2752 x 5 x 5 inner confusion")
    with np.load(root / "head-scan/lda.npz", allow_pickle=False) as archive:
        weight, bias = archive["weight"], archive["bias"]
    require(weight.shape == (n_heads, classes, width) and bias.shape == (n_heads, classes), "Wrong all-head LDA coefficient shapes")
    require(weight.dtype == np.float64 and bias.dtype == np.float64 and
            np.isfinite(weight).all() and np.isfinite(bias).all(), "LDA coefficients must be finite float64")
    support = np.zeros(classes, dtype=np.int64)
    by_cell = {}
    inner = plan["records"]["inner"]
    require(inner, "FIT-inner position plan is empty")
    seen = set()
    with (root / "data/inputs-fit.jsonl").open() as stream:
        record_index = 0
        for line in stream:
            if not line.strip():
                continue
            key = str(record_index)
            record_index += 1
            if key not in inner:
                continue
            seen.add(key)
            record = json.loads(line)
            require(record["split"] == "fit" and record["domain"] in DOMAINS, "A head-scan record is not native FIT data")
            indices = inner[key]
            require(len(indices) == len(set(indices)) and
                    all(isinstance(index, int) and 0 <= index < len(record["targets"]) for index in indices), "Invalid or duplicate FIT-inner target index")
            for index in indices:
                target = record["targets"][index]
                label = roles.index(target["label"]) if isinstance(target["label"], str) else int(target["label"])
                require(0 <= label < classes and target.get("role", roles[label]) == roles[label], "FIT-inner native role/index mismatch")
                support[label] += 1
                cell = f"inner/{record['domain']}/{roles[label]}"
                by_cell[cell] = by_cell.get(cell, 0) + 1
    require(seen == set(inner), "FIT-inner plan refers to a nonexistent record")
    require(np.all(support > 0), "A native role has no true FIT-inner support")
    require(selected["counts"] == plan["counts"], "Head-selection counts differ from the fixed scan plan")
    planned_cells = {key: count for key, count in plan["counts"].items() if key.startswith("inner/")}
    require(by_cell == planned_cells, "FIT-inner planned class support differs from raw native targets")
    require(np.array_equal(confusion.sum(axis=2), np.broadcast_to(support, (n_heads, classes))),
            "Inner confusion has the wrong true-label axis or class support")
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    predicted = confusion.sum(axis=1)
    true_support = confusion.sum(axis=2)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal, dtype=np.float64), where=predicted > 0)
    recall = diagonal / true_support
    # Algebraically equivalent to 2*p*r/(p+r), with independent integer-count
    # sufficient statistics. The audit tolerates its last-bit differences.
    f1 = np.divide(2 * diagonal, true_support + predicted, out=np.zeros_like(diagonal, dtype=np.float64),
                   where=(true_support + predicted) > 0)
    macro = f1.mean(axis=1)
    weakest = np.minimum(precision, recall).min(axis=1)
    ranking = selected["ranking"]
    require(len(ranking) == n_heads, "Head ranking does not contain all 2752 heads")
    ids = [item["head"] for item in ranking]
    require(set(ids) == set(range(n_heads)) and len(set(ids)) == n_heads, "Head ranking contains missing/duplicate IDs")
    macro_error = weakest_error = 0.
    for item in ranking:
        head = item["head"]
        require(item["layer"] == head // 64 and item["within_layer"] == head % 64, "Global head ID and layer coordinates disagree")
        require(np.isfinite(item["macro_f1"]) and np.isfinite(item["weakest_precision_recall"]), "Nonfinite saved head score")
        macro_error = max(macro_error, abs(float(macro[head]) - item["macro_f1"]))
        weakest_error = max(weakest_error, abs(float(weakest[head]) - item["weakest_precision_recall"]))
    require(macro_error <= 1e-12 and weakest_error <= 1e-12, "Saved all-head scores differ from independently recomputed confusion metrics")
    declared_order = [item["head"] for item in sorted(ranking, key=lambda item:
                      (-item["macro_f1"], -item["weakest_precision_recall"], item["head"]))]
    require(ids == declared_order, "Saved ranking is not sorted by its verified macro-F1/weakest/head-ID rule")
    require(selected["heads"] == ids[:16], "Selected sixteen heads do not match the verified ranking prefix")
    return {"heads_verified": n_heads, "selected_heads": selected["heads"], "head_dim": width,
            "lda_weight_shape": list(weight.shape), "lda_bias_shape": list(bias.shape),
            "inner_positions": int(support.sum()), "true_support_by_role": dict(zip(roles, map(int, support))),
            "max_absolute_macro_f1_error": macro_error, "max_absolute_weakest_precision_recall_error": weakest_error,
            "score_tolerance": 1e-12, "ordering": "saved verified scores: descending macro-F1, descending weakest precision/recall, ascending head ID",
            "inputs": {name: sha(root / name) for name in ("head-scan/inner-confusion.npz", "head-scan/lda.npz", "head-selection.json")}}


def audit_run(output):
    """Return a complete audit only after every final artifact has been committed."""
    root = Path(output)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    result = {"format_version": "deepseek-role-independent-validation-v1", "status": "running",
              "complete": not missing, "checked_unix": time.time(), "missing_artifacts": missing,
              "pending_artifacts": [],
              "validator_sha256": sha(__file__),
              "checks": [], "artifact_sha256": {}, "metrics": None, "limitations": [
                  "五分类标签为 system/user/cot/assistant/tool；developer 按官方编码归并到 user。",
                  "与 GPT-OSS 六分类指标比较时，需考虑标签空间和分词差异。",
                  "AgentDojo 使用已有 GPT-OSS 轨迹做跨模型 teacher-forced replay；本结果不衡量新的 DeepSeek agent rollout 或攻击成功率。",
                  "全 GPU 审计覆盖用于探针的完整 43 层主干；Transformers 不加载检查点中可选的 MTP speculative-decoding 辅助模块。",
                  "GPU/混合精度审计依据已保存的运行时快照；本工具不重新加载或执行基座模型。"]}
    for name in REQUIRED:
        if (root / name).is_file():
            result["artifact_sha256"][name] = sha(root / name)
    load_audit = None
    if (root / "load-integrity-audit.json").is_file():
        load_audit = read(root / "load-integrity-audit.json")
        inventory = load_audit.get("index_inventory") or {}
        result["load_integrity"] = {"status": load_audit.get("status"), "expected_loaded_keys": inventory.get("expected_loaded_keys"),
                                    "fusion_equation": inventory.get("fusion_equation"),
                                    "evidence_scope": load_audit.get("evidence_scope")}
        if load_audit.get("status") == "running":
            result["pending_artifacts"].append("load-integrity-audit.json: running")
    result["performance_cache"] = performance_summary(root)
    performance = result["performance_cache"]
    if performance:
        for name in ("performance-cache-applied.json", "performance-cache-gpu-validation.json"):
            if (root / name).is_file():
                result["artifact_sha256"][name] = sha(root / name)
            elif name not in missing:
                missing.append(name)
        if performance["status"] == "dry_run":
            result["pending_artifacts"].append("performance-cache-applied.json: dry_run")
        if performance.get("gpu_validation", {}).get("report_status") == "dry_run":
            result["pending_artifacts"].append("performance-cache-gpu-validation.json: dry_run")
    result["complete"] = not (missing or result["pending_artifacts"])
    if not result["complete"]:
        return result

    def check(name, action):
        try:
            detail = action()
            result["checks"].append({"name": name, "ok": True, "detail": detail})
        except Exception as error:
            result["checks"].append({"name": name, "ok": False, "error": f"{type(error).__name__}: {error}"})

    config, data = read(root / "config.json"), read(root / "data/manifest.json")
    data_audit, heads = read(root / "data/data-audit.json"), read(root / "head-selection.json")
    backend, preflight = read(root / "model_backend.json"), read(root / "preflight.json")
    training, selection, frozen = read(root / "training.json"), read(root / "selection.json"), read(root / "freeze.json")
    feature_manifest, test = read(root / "features/manifest.json"), read(root / "evaluations/test.json")
    final, plan = read(root / "runtime-final.json"), read(root / "head-scan/plan.json")
    probe = checkpoint(root / "checkpoint.npz")
    probabilities = np.load(root / "evaluations/test-probabilities.npy", mmap_mode="r", allow_pickle=False)
    splits = {}

    def frozen_check():
        for key, name in (("probe_sha256", "checkpoint.npz"), ("training_sha256", "training.json"), ("selection_sha256", "selection.json")):
            require(frozen[key] == sha(root / name), f"Frozen file changed: {name}")
        require(frozen["selection_split"] == selection["selection_split"] == "dev", "Selection was not DEV-only")
        require(frozen["candidate_id"] == selection["candidate_id"] == probe["metadata"]["candidate_id"], "Selected candidate mismatch")
        require(training["training_hash"] == selection["training_hash"] == probe["metadata"]["training_hash"], "Training identity mismatch")
        require(probe["metadata"].get("engineering_smoke_only") is False, "Engineering smoke checkpoint cannot receive an experiment pass")
        expected_candidates = {"linear-seed17", "mlp128-seed17", "mlp128-seed29", "mlp128-seed43",
                               "mlp256-seed17", "mlp256-seed29", "mlp256-seed43"}
        require(len(selection["candidate_results"]) == 7 and
                {item["candidate_id"] for item in selection["candidate_results"]} == expected_candidates,
                "Expected the seven declared random-init candidates")
        started, completed, release = (read(root / name) for name in
                                       ("evaluations/test-started.json", "evaluations/test-complete.json", "test-release.json"))
        digest = frozen["probe_sha256"]
        require(all(record.get(key) == digest for record, key in
                    ((started, "probe_sha256"), (completed, "probe_sha256"), (release, "checkpoint_sha256"),
                     (test, "probe_sha256"), (final, "checkpoint_sha256"))), "TEST artifacts refer to another checkpoint")
        require(frozen["frozen_at_unix"] <= release["released_unix"] <= started["started_unix"] <= completed["completed_unix"],
                "TEST release/evaluation preceded classifier freeze")
        require(completed["completed_unix"] <= final["finished_unix"], "Final runtime audit predates TEST completion")
        for name, digest in completed["files"].items():
            require(sha(safe_path(root / "evaluations", name)) == digest, "Changed committed TEST output")
        require(completed["feature_manifest_sha256"] == sha(root / "features/manifest.json"), "TEST cache identity changed")
        return {"checkpoint_sha256": frozen["probe_sha256"], "candidate": frozen["candidate_id"], "test_after_freeze": True}

    check("checkpoint_freeze_and_test_commit", frozen_check)

    def schema_check():
        require(tuple(data["roles"]) == tuple(feature_manifest["roles"]) == tuple(probe["metadata"]["roles"]) == ROLES, "Expected five native DeepSeek roles")
        require(data["protocol"]["source_role_mapping"]["developer"] == "user", "Developer/user merge is not declared")
        require("teacher-forced" in data["protocol"]["agentdojo_adaptation"] and "cross-model" in data["protocol"]["agentdojo_adaptation"],
                "Cross-model teacher-forced replay policy is not declared")
        require(data_audit["status"] == "passed" and not any(data_audit["cross_split_cluster_overlap"].values()), "Data audit reports cluster leakage")
        require(feature_manifest["feature_dim"] == heads["feature_dim"] == probe["metadata"]["feature_dim"] == 8192, "Expected 8192-dimensional feature schema")
        require(feature_manifest["head_dim"] == heads["head_dim"] == 512 and len(heads["heads"]) == 16 and len(set(heads["heads"])) == 16,
                "Expected sixteen distinct 512-dimensional heads")
        require(all(isinstance(head, int) and 0 <= head < 43 * 64 for head in heads["heads"]), "Invalid global V4 Flash head ID")
        require(feature_manifest["heads"] == heads["heads"] == probe["metadata"]["heads"], "Ordered head binding changed")
        require(feature_manifest["feature_position"] == heads["feature_position"] == probe["metadata"]["feature_position"] == CAPTURE,
                "Wrong V4 head capture point")
        require(feature_manifest["selection_sha256"] == sha(root / "head-selection.json"), "Head-selection hash differs")
        require(feature_manifest["config_hash"] == object_sha(config), "Extraction configuration hash differs")
        for key in ("roles", "feature_dim", "head_dim", "heads", "feature_position", "model_binding",
                    "selection_sha256", "pilot_fit_sha256", "data_hash", "config_hash"):
            require(feature_manifest[key] == training["binding"][key] == probe["metadata"][key], f"Model/data/head binding changed: {key}")
        require(heads["signature"]["model_binding"] == feature_manifest["model_binding"], "Head scan model binding differs")
        require(heads["signature"]["config_sha256"] == object_sha(config), "Head scan configuration differs")
        for name, digest in data["files_sha256"].items():
            require(sha(safe_path(root / "data", name)) == digest, f"Changed prepared native data: {name}")
        return {"roles": list(ROLES), "head_count": 16, "head_dim": 512, "feature_dim": 8192, "developer_maps_to": "user"}

    check("native_data_and_feature_schema", schema_check)

    def model_check():
        for snapshot in (backend, preflight["backbone_audit"], final["backbone_audit"]):
            verify_model_audit(snapshot)
        placements = {str(value) for value in backend["device_map"].values()}
        require(not placements & {"cpu", "disk", "meta"}, "Model device map includes offload")
        require(placements <= {"0", "1", "2", "3", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}, "Unexpected model placement")
        require(preflight["status"] == "passed" and preflight["use_cache"] is False and preflight["lm_head_skipped"] is True,
                "Backbone preflight was not completed with the declared extraction path")
        require(preflight["checks"] and all(item["all_layers_captured"] == 43 and item["hook_output_max_abs_difference"] <= 1e-5
                                            for item in preflight["checks"]), "Capture preflight failed")
        require(feature_manifest["model_binding"].get("offload") is False, "Feature binding does not forbid offload")
        require((backend["layers"], backend["heads_per_layer"], backend["head_dim"]) == (43, 64, 512), "Wrong V4 Flash model geometry")
        return {"gpu_devices": 4, "native_packed_fp4_experts": True, "native_fp8_dense": True,
                "frozen_backbone": True, "audit_source": "saved runtime snapshots"}

    check("native_precision_all_gpu_and_capture_preflight", model_check)
    check("independent_fit_inner_head_ranking_and_lda", lambda: validate_head_selection(root))

    def load_check():
        require(load_audit["format_version"] == "deepseek-load-integrity-audit-v1" and load_audit["status"] == "passed",
                "A completed passing load-integrity audit is required")
        require(load_audit["checks"] and all(item["status"] == "passed" for item in load_audit["checks"]), "A load-integrity check did not pass")
        require(Path(load_audit["run_dir"]).resolve() == root.resolve(), "Load audit belongs to another run")
        inventory = load_audit["index_inventory"]
        require(inventory["valid"] is True and inventory["expected_loaded_keys"] == 1736 and
                inventory["geometry"] == [43, 256, 64, 512], "Invalid official expert-fusion inventory")
        evidence = load_audit["log_evidence"]
        require(evidence["weights_progress_complete"] is True and evidence["loaded_event_seen"] is True and
                not evidence["loading_diagnostics"] and not evidence["loading_errors"], "Load completion is not supported by clean logs")
        require(evidence["latest_progress"]["loaded"] == evidence["latest_progress"]["total"] == 1736, "Incomplete fused-key load")
        inputs = load_audit["inputs"]
        for label, name in (("run_config", "config.json"), ("model_backend", "model_backend.json"),
                            ("preflight", "preflight.json"), ("weight_verification", "weight-verification.json")):
            binding = inputs[label]
            require(binding["exists"] is True and Path(binding["path"]).resolve() == (root / name).resolve(), f"Wrong load-audit input: {label}")
            require(binding["sha256"] == sha(root / name) and binding["bytes"] == (root / name).stat().st_size, f"Changed load-audit input: {label}")
        for label in ("official_config", "official_index", "backend_source", "audit_source"):
            binding = inputs[label]
            path = Path(binding["path"])
            require(binding["exists"] is True and binding["bytes"] == path.stat().st_size and binding["sha256"] == sha(path), f"Changed audited model/source input: {label}")
        require(backend["config_sha256"] == inputs["official_config"]["sha256"] and
                backend["index_sha256"] == inputs["official_index"]["sha256"], "Load audit and model metadata differ")
        log = inputs["run_log"]
        require(log["exists"] is True and Path(log["path"]).resolve() == (root / "run.log").resolve() and
                log["bytes"] > 0 and prefix_sha(log["path"], log["bytes"]) == log["sha256"], "Loading-log prefix differs from its audit")
        return {"load_integrity_sha256": sha(root / "load-integrity-audit.json"), "fused_loaded_keys": 1736,
                "fusion_equation": inventory.get("fusion_equation"), "log_prefix_bytes": log["bytes"],
                "log_scope": "SHA-bound prefix at audit time; later training log content is excluded"}

    check("completed_load_integrity_and_bound_evidence", load_check)

    if performance:
        def performance_check():
            applied = read(root / "performance-cache-applied.json")
            validated = read(root / "performance-cache-gpu-validation.json")
            require(applied.get("status") == "applied" and validated.get("status") == "validation_passed" and
                    validated.get("validation", {}).get("status") == "passed", "Native kernel-cache reuse was not applied and GPU-validated")
            for item in (applied, validated):
                require(item["format_version"] == "native-fp4-performance-cache-reuse-v1", "Unsupported native-cache provenance format")
                require(item["invariants"] and all(value is True for value in item["invariants"].values()), "A native-cache invariant is not preserved")
            require(applied["native_identity"] == validated["native_identity"], "Applied and validated native kernel identities differ")
            require(applied["prepared_inputs"] == validated["prepared_inputs"], "Applied and validated input identities differ")
            require(applied["tool_source_sha256"] == validated["tool_source_sha256"], "Cache-reuse tool changed between validation and application")
            gpu = validated["validation"]
            checks = gpu["checks"]
            require(checks and all(item.get("finite") is True and item.get("bitexact") is True and
                    item.get("native_disk_cache_hit") is True and item.get("search_called") is False and
                    item.get("maximum_absolute_difference") == 0 and item["S"] == 6 * item["tokens"] for item in checks),
                    "GPU validation does not establish native fixed-configuration/cache-dispatch equivalence")
            verified_winners = {item["source_winner_sha256"] for item in checks}
            sources = applied["plan"]["sources"].values()
            require(all(source["sha256"] in verified_winners and source["measured_candidates"] > 1 and
                        math.isfinite(source["winner_ms"]) and source["winner_ms"] > 0 for source in sources),
                    "A reused configuration lacks a measured source winner and GPU validation")
            require(gpu["peak_extra_allocated_bytes"] <= gpu["allocator_limit_bytes"], "Native-cache GPU validation exceeded its declared memory limit")
            require(applied.get("application"), "Native-cache application counts are absent")
            require(8309 in applied["plan"]["protected_tokens"], "The longest preflight input is not protected from cache fabrication")
            return performance

        check("native_kernel_configuration_reuse_provenance", performance_check)

    def feature_check():
        for split in SPLITS:
            splits[split] = feature_split(root / "features", feature_manifest, split)
            input_path = root / "data" / f"inputs-{split}.jsonl"
            require(feature_manifest["split_sources"][split] == sha(input_path), f"Wrong native input hash for {split}")
            raw = raw_target_metadata(input_path, list(ROLES))
            for key in raw:
                require(np.array_equal(raw[key], splits[split][key]), f"Native targets differ from cached {split}/{key}")
        for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test")):
            for key in ("source", "family"):
                require(not np.intersect1d(splits[a][key], splits[b][key]).size, f"Cross-split {key} leakage: {a}/{b}")
        require(heads["signature"]["pilot_fit_sha256"] == sha(root / "data/inputs-fit.jsonl"), "Head scan did not bind FIT inputs")
        require(heads["no_dev_test_or_historical_probe_used"] is True, "Head scan is not declared FIT-only")
        statistics, inner = set(plan["clusters"]["statistics"]), set(plan["clusters"]["inner"])
        require(statistics and inner and not statistics & inner, "Internal head-scan source partitions overlap or are empty")
        require((statistics | inner) <= set(splits["fit"]["source"]), "Head discovery contains a non-FIT source")
        return {"verified_rows": {split: len(values["y"]) for split, values in splits.items()},
                "cross_split_source_overlap": 0, "head_scan_fit_only": True}

    check("committed_features_native_labels_and_source_isolation", feature_check)

    def standardizer_check():
        require("fit" in splits, "Verified FIT cache is unavailable")
        mean, scale = population_standardizer(splits["fit"]["x"], feature_manifest["feature_dim"])
        mean_error = float(np.max(np.abs(mean - probe["mean"])))
        scale_error = float(np.max(np.abs(scale - probe["scale"])))
        require(np.allclose(mean, probe["mean"], rtol=2e-6, atol=2e-6), "Checkpoint standardizer mean is not FIT-only population mean")
        require(np.allclose(scale, probe["scale"], rtol=2e-6, atol=2e-6), "Checkpoint standardizer scale is not FIT-only population std")
        with np.load(root / "standardizer.npz", allow_pickle=False) as saved:
            require(np.array_equal(saved["mean"], probe["mean"]) and np.array_equal(saved["scale"], probe["scale"]), "Scaler sidecar and NPZ checkpoint differ")
        return {"method": "independent two-pass NumPy float64 population mean/std over unique FIT positions",
                "fit_rows": len(splits["fit"]["y"]), "max_absolute_mean_error": mean_error, "max_absolute_scale_error": scale_error}

    check("independent_fit_standardizer", standardizer_check)

    def metric_check():
        require("test" in splits, "Verified TEST labels are unavailable")
        values = splits["test"]
        actual = recompute_metrics(probabilities, values["y"], ROLES)
        compare_metrics(actual, test["overall"])
        domains = {}
        for domain in DOMAINS:
            mask = values["domain"] == domain
            domains[domain] = recompute_metrics(probabilities[mask], values["y"][mask], ROLES)
            compare_metrics(domains[domain], test["by_domain"][domain])
        result["metrics"] = {"overall": actual, "by_domain": domains}
        sample = sample_features(values["x"])
        predicted = independent_probabilities(probe, sample)
        error = float(np.max(np.abs(predicted - probabilities[:len(sample)])))
        require(error <= 3e-5, "Saved TEST probabilities differ from independent CPU checkpoint readout")
        return {"test_rows": len(values["y"]), "overall_accuracy": actual["accuracy"], "overall_macro_f1": actual["macro_f1"],
                "checkpoint_sample_rows": len(sample), "checkpoint_probability_max_absolute_error": error}

    check("independent_test_metrics_and_checkpoint_sample", metric_check)
    result["status"] = "passed" if all(check["ok"] for check in result["checks"]) else "failed"
    return result


def optional_plots(root, validation):
    if validation["status"] != "passed":
        return {"status": "skipped", "reason": "Final independent validation is incomplete or failed"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "skipped", "reason": "matplotlib is unavailable"}
    paths = []
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    matrix = np.asarray(validation["metrics"]["overall"]["confusion"])
    normalized = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(1, keepdims=True) > 0)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(xticks=np.arange(5), yticks=np.arange(5), xticklabels=ROLES, yticklabels=ROLES,
             xlabel="Predicted native role", ylabel="Native role", title="Frozen TEST confusion (row normalized)")
    for row in range(5):
        for column in range(5):
            axis.text(column, row, f"{matrix[row, column]:,}\n{normalized[row, column]:.1%}", ha="center", va="center",
                      color="white" if normalized[row, column] > .5 else "black", fontsize=8)
    figure.colorbar(image, ax=axis, fraction=.045)
    figure.tight_layout()
    figure.savefig(root / "test-confusion.png", dpi=180)
    plt.close(figure)
    paths.append("test-confusion.png")
    histories = sorted((root / "candidates").glob("*/history.json"))
    if histories:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for path in histories:
            history = read(path)
            epochs = [item["epoch"] for item in history]
            axes[0].plot(epochs, [item["train_loss"] for item in history], label=path.parent.name)
            axes[1].plot(epochs, [item["dev"]["domain_equal_macro_f1"] for item in history], label=path.parent.name)
        axes[0].set(xlabel="Epoch", ylabel="Cross entropy", title="FIT sampled training loss")
        axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="DEV domain-equal macro F1")
        axes[1].legend(fontsize=7, loc="best")
        figure.tight_layout()
        figure.savefig(root / "training-curves.png", dpi=180)
        plt.close(figure)
        paths.append("training-curves.png")
    return {"status": "created", "files": {name: sha(root / name) for name in paths}}


def report_markdown(root, validation):
    status = {"running": "运行中，最终结果尚未完成", "failed": "独立验证发现问题", "passed": "训练完成，独立验证通过"}[validation["status"]]
    configuration_label = "已验证配置" if validation["status"] == "passed" else "目标配置（尚未通过最终验证）"
    lines = ["# DeepSeek V4 Flash 角色探针", "", f"状态：**{status}**。", "",
             f"{configuration_label}：`deepseek-ai/DeepSeek-V4-Flash-0731`；冻结基座，原生 packed FP4 专家与 FP8 dense 权重置于 4 张 H100。", "",
             "探针读取逆 RoPE 后、`o_a_proj` 前的 16 个有序注意力头，每头 512 维，合计 8192 维。", ""]
    if validation["status"] == "running":
        lines += ["最终审计尚未执行；目前没有可发布的最终 TEST 结论。", "", "待完成产物：", ""]
        lines += [f"- `{name}`" for name in validation["missing_artifacts"]]
        lines += [f"- `{name}`" for name in validation.get("pending_artifacts", [])]
    elif validation["status"] == "failed":
        lines += ["以下检查需要修复；本报告不将当前产物判为验证通过。", ""]
        lines += [f"- `{item['name']}`：{item.get('error', 'unknown')}" for item in validation["checks"] if not item["ok"]]
    else:
        frozen, selection = read(root / "freeze.json"), read(root / "selection.json")
        winner = next(item for item in selection["candidate_results"] if item["candidate_id"] == selection["candidate_id"])
        lines += [f"选中候选：`{selection['candidate_id']}`；最佳 epoch：{winner['best_epoch']}。", "",
                  f"检查点：`checkpoint.npz`；SHA256：`{frozen['probe_sha256']}`。", "",
                  "标准化器仅使用 FIT 的唯一位置；训练按 domain 0.5/0.5，再按实际存在角色及唯一位置均匀有放回采样。", "",
                  "7 个随机初始化 linear/MLP 候选仅用 DEV 选模及早停。冻结后解封 TEST，保存的一次 TEST 输出经独立 NumPy 重算。", "",
                  "| TEST 范围 | 位置数 | Accuracy | Macro F1 |", "|---|---:|---:|---:|"]
        for name, metric in [("合并", validation["metrics"]["overall"]), *validation["metrics"]["by_domain"].items()]:
            lines.append(f"| {name} | {metric['tokens']:,} | {metric['accuracy']:.4%} | {metric['macro_f1']:.6f} |")
        lines += ["", "| 原生角色 | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
        for role, metric in validation["metrics"]["overall"]["roles"].items():
            values = [f"{metric[key]:.6f}" if metric[key] is not None else "N/A" for key in ("precision", "recall", "f1")]
            lines.append(f"| {role} | {' | '.join(values)} |")
        lines += ["", "独立审计已核验冻结文件哈希、原生标签与特征行顺序、跨 split source/family 隔离、FIT 内选头来源、两遍 F64 标准化统计、TEST 指标及小样本 CPU 预测一致性。", ""]
        head_audit = next(item["detail"] for item in validation["checks"] if item["name"] == "independent_fit_inner_head_ranking_and_lda")
        lines += [f"选头验收覆盖全部 {head_audit['heads_verified']} 个头：从 FIT-inner 原生目标重算的各类支持为 "
                  f"`{head_audit['true_support_by_role']}`；全部排名指标误差不超过 `1e-12`，排序及前 16 个头一致。",
                  f"有序头编号：`{head_audit['selected_heads']}`。", ""]
    load = validation.get("load_integrity")
    if load:
        lines += ["", f"加载完整性：`load-integrity-audit.json` 状态为 `{load['status']}`。"]
        if load.get("fusion_equation"):
            lines += [f"官方逐专家权重到加载目标张量的数量关系：`{load['fusion_equation']}`。"]
        lines += ["该审计核验完整加载与 GPU preflight，日志哈希覆盖审计时的已有前缀。", ""]
    performance = validation.get("performance_cache")
    if performance:
        lines += ["", f"原生 kernel 性能缓存状态：`{performance.get('status')}`。", ""]
        for source in performance.get("measured_sources", []):
            lines.append(f"- `{source.get('projection')}`：复用 {source.get('source_tokens')} token 参考输入的已测原生配置；"
                         f"参考搜索记录 {source.get('measured_candidates')} 个候选，原 winner 时间为 {source.get('winner_ms')} ms。")
        gpu = performance.get("gpu_validation")
        if gpu:
            lines += ["", f"独立 GPU 对照状态：`{gpu.get('report_status')}` / `{gpu.get('status')}`；"
                      f"已对照 token 长度 {gpu.get('tokens')}，共 {gpu.get('cases')} 组原 kernel 固定配置与原生 disk-cache dispatch。",
                      f"记录的 finite/bitexact 全通过：`{gpu.get('bitexact_and_finite')}`；验证时调用搜索：`{gpu.get('search_called')}`。"]
        lines += ["", "其他真实长度复用参考配置，保留真实 `S=6×input_tokens` 及原 kernel 的网格、路由和数值格式。"
                  "这不表示每个目标长度都重新做过 autotune，也不为复用条目声称新的性能测量时间。",
                  f"保护的原始输入长度：`{performance.get('protected_tokens')}`；应用计数：`{performance.get('application')}`。", ""]
    lines += ["", "适用范围与限制：", "", *[f"- {text}" for text in validation["limitations"]], "",
              "完整检查结果见 `independent-validation.json`；原始 TEST 指标及 source-cluster bootstrap 区间见 `evaluations/test.json`。", ""]
    plots = validation.get("plots", {})
    if plots.get("status") == "created":
        for name in plots["files"]:
            lines += [f"![{name}]({name})", ""]
    return "\n".join(lines)


def generate_report(output, *, plots=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    try:
        validation = audit_run(root)
    except Exception as error:
        validation = {"format_version": "deepseek-role-independent-validation-v1", "status": "failed", "complete": False,
                      "checked_unix": time.time(), "missing_artifacts": [], "checks": [
                          {"name": "artifact_loading", "ok": False, "error": f"{type(error).__name__}: {error}"}],
                      "artifact_sha256": {}, "metrics": None, "limitations": ["产物读取失败，未形成有效最终审计。"]}
    if plots:
        try:
            validation["plots"] = optional_plots(root, validation)
        except Exception as error:
            validation["plots"] = {"status": "skipped", "reason": f"{type(error).__name__}: {error}"}
    write(root / "independent-validation.json", validation)
    temporary = root / "REPORT.zh-CN.md.tmp"
    temporary.write_text(report_markdown(root, validation))
    os.replace(temporary, root / "REPORT.zh-CN.md")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = generate_report(args.output, plots=args.plots)
    print(json.dumps({"status": result["status"], "missing_artifacts": result["missing_artifacts"],
                      "checks": [{"name": check["name"], "ok": check["ok"]} for check in result["checks"]]}), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
