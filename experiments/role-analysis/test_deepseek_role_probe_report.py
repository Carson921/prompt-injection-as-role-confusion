"""Synthetic artifact tests of the independent CPU audit; no model is loaded."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import deepseek_role_probe_report as r


def fixture(root):
    """Construct a self-consistent, explicitly artificial completed artifact set."""
    config = {"seed": 17, "fixture": True}
    r.write(root / "config.json", config)
    data_root, feature_root = root / "data", root / "features"
    data_root.mkdir()
    feature_root.mkdir()
    roles, sources, labels, families = list(r.ROLES), {}, {}, {}
    for split in r.SPLITS:
        records = []
        for domain in r.DOMAINS:
            for group in range(2):
                cluster = f"{split}:{domain}:source-{group}"
                records.append({"id": cluster + ":record", "source_cluster": cluster, "source_id": cluster,
                                "split": split, "domain": domain, "input_ids": list(range(6)),
                                "targets": [{"token_ix": label, "label": label, "role": role,
                                             "family_id": f"{cluster}:position-{label}"} for label, role in enumerate(roles)]})
        (data_root / f"inputs-{split}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        sources[split] = [record["source_cluster"] for record in records for _ in roles]
        labels[split] = np.tile(np.arange(5), len(records))
        families[split] = [target["family_id"] for record in records for target in record["targets"]]
    r.write(data_root / "data-audit.json", {"status": "passed", "cross_split_cluster_overlap": {"fit/dev": 0, "fit/test": 0, "dev/test": 0}})
    r.write(data_root / "manifest.json", {"roles": roles, "protocol": {
        "source_role_mapping": {"developer": "user"}, "agentdojo_adaptation": "cross-model teacher-forced replay"},
        "files_sha256": {name: r.sha(data_root / name) for name in
                          [*(f"inputs-{split}.jsonl" for split in r.SPLITS), "data-audit.json"]}})
    split_sources = {split: r.sha(data_root / f"inputs-{split}.jsonl") for split in r.SPLITS}
    model_binding = {"model_id": "synthetic-report-test-fixture", "offload": False}
    confusion = np.zeros((2752, 5, 5), np.int64)
    confusion[:, :, 0] = 1  # Five true classes, deliberately unequal predicted support.
    inner_counts = {f"inner/agentdojo/{role}": 1 for role in roles}
    heads = {"heads": list(range(16)), "head_dim": 512, "feature_dim": 8192, "feature_position": r.CAPTURE,
             "roles": roles, "counts": inner_counts,
             "ranking": [{"head": head, "layer": head // 64, "within_layer": head % 64,
                          "macro_f1": 1 / 15, "weakest_precision_recall": 0.} for head in range(2752)],
             "signature": {"model_binding": model_binding, "pilot_fit_sha256": split_sources["fit"],
                           "config_sha256": r.object_sha(config)}, "no_dev_test_or_historical_probe_used": True}
    r.write(root / "head-selection.json", heads)
    r.write(root / "head-scan/plan.json", {"clusters": {"statistics": [sources["fit"][0]], "inner": [sources["fit"][-1]]},
                                            "counts": inner_counts, "records": {"inner": {"3": [0, 1, 2, 3, 4]}}})
    np.savez_compressed(root / "head-scan/inner-confusion.npz", confusion=confusion)
    np.savez_compressed(root / "head-scan/lda.npz", weight=np.zeros((2752, 5, 512), np.float64), bias=np.zeros((2752, 5), np.float64))
    manifest = {"format_version": "deepseek-role-features-v1", "roles": roles, "head_dim": 512, "feature_dim": 8192,
                "heads": list(range(16)), "feature_position": r.CAPTURE, "model_binding": model_binding,
                "selection_sha256": r.sha(root / "head-selection.json"), "pilot_fit_sha256": split_sources["fit"],
                "data_hash": r.object_sha(split_sources), "config_hash": r.object_sha(config), "split_sources": split_sources,
                "signature": {"model_binding": model_binding, "capture": r.CAPTURE, "heads": list(range(16))},
                "progress": {}, "splits": {}}
    rng = np.random.default_rng(2)
    fit_x = None
    for split in r.SPLITS:
        folder = feature_root / split
        folder.mkdir()
        x = rng.normal(size=(20, 8192)).astype(np.float16)
        if split == "fit":
            fit_x = x
        arrays = {"x": x, "y": labels[split], "source": np.asarray(sources[split]), "family": np.asarray(families[split]),
                  "domain": np.repeat(r.DOMAINS, 10)}
        hashes = {}
        for name, values in arrays.items():
            np.save(folder / (name + ".npy"), values, allow_pickle=False)
            hashes[name + ".npy"] = r.sha(folder / (name + ".npy"))
        r.write(folder / "complete.json", {"rows": 20, "signature": {**manifest["signature"], "split": split,
                           "input_sha256": split_sources[split]}, "files": hashes})
        manifest["splits"][split] = [{"path": split, "rows": 20, "sha256": r.sha(folder / "complete.json")}]
        manifest["progress"][split] = {"complete": True}
    r.write(feature_root / "manifest.json", manifest)
    binding = {key: manifest[key] for key in ("roles", "feature_dim", "head_dim", "heads", "feature_position",
                                             "model_binding", "selection_sha256", "pilot_fit_sha256", "data_hash", "config_hash")}
    mean = fit_x.astype(np.float64).mean(0).astype(np.float32)
    scale = fit_x.astype(np.float64).std(0).astype(np.float32)
    scale[scale < 1e-6] = 1
    metadata = {**binding, "candidate_id": "linear-seed17", "training_hash": "fixture-training-hash", "hidden": 0,
                "engineering_smoke_only": False}
    np.savez(root / "checkpoint.npz", mean=mean, scale=scale, metadata=np.asarray(json.dumps(metadata)),
             **{"weight::0.weight": np.zeros((5, 8192), np.float32), "weight::0.bias": np.zeros(5, np.float32)})
    np.savez(root / "standardizer.npz", mean=mean, scale=scale)
    r.write(root / "training.json", {"binding": binding, "training_hash": "fixture-training-hash"})
    candidates = ["linear-seed17", "mlp128-seed17", "mlp128-seed29", "mlp128-seed43", "mlp256-seed17", "mlp256-seed29", "mlp256-seed43"]
    r.write(root / "selection.json", {"selection_split": "dev", "candidate_id": "linear-seed17", "training_hash": "fixture-training-hash",
               "candidate_results": [{"candidate_id": candidate, "best_epoch": 10} for candidate in candidates]})
    digest = r.sha(root / "checkpoint.npz")
    r.write(root / "freeze.json", {"selection_split": "dev", "candidate_id": "linear-seed17", "probe_sha256": digest,
               "training_sha256": r.sha(root / "training.json"), "selection_sha256": r.sha(root / "selection.json"), "frozen_at_unix": 100.})
    audit = {"parameters_frozen": True, "all_parameters_on_cuda": True, "native_packed_fp4_fp8": True,
             "parameter_bytes_by_device_dtype": {f"cuda:{i}": {"torch.int8": 100, "torch.float8_e4m3fn": 10} for i in range(4)},
             "dtype_bytes": {"torch.int8": 400, "torch.float8_e4m3fn": 40}, "quantization_config": {
                 "quant_method": "fp8", "scale_fmt": "ue8m0", "expert_dtype": "fp4", "dequantize": False, "weight_block_size": [128, 128]}}
    r.write(root / "model_backend.json", {**audit, "device_map": {str(i): i for i in range(4)}, "layers": 43, "heads_per_layer": 64, "head_dim": 512})
    r.write(root / "preflight.json", {"backbone_audit": audit, "status": "passed", "use_cache": False, "lm_head_skipped": True,
              "checks": [{"all_layers_captured": 43, "hook_output_max_abs_difference": 0.}]})
    r.write(root / "runtime-final.json", {"backbone_audit": audit, "checkpoint_sha256": digest, "finished_unix": 400.})
    r.write(root / "test-release.json", {"checkpoint_sha256": digest, "released_unix": 150.})
    r.write(root / "evaluations/test-started.json", {"probe_sha256": digest, "started_unix": 200.})
    p = np.full((20, 5), .2, np.float32)
    np.save(root / "evaluations/test-probabilities.npy", p, allow_pickle=False)
    test = {"overall": r.recompute_metrics(p, labels["test"], roles), "by_domain": {}, "probe_sha256": digest}
    for domain, indices in zip(r.DOMAINS, (slice(0, 10), slice(10, 20))):
        test["by_domain"][domain] = r.recompute_metrics(p[indices], labels["test"][indices], roles)
    r.write(root / "evaluations/test.json", test)
    r.write(root / "evaluations/test-complete.json", {"probe_sha256": digest, "feature_manifest_sha256": r.sha(feature_root / "manifest.json"),
                  "completed_unix": 300., "files": {name: r.sha(root / "evaluations" / name) for name in ("test.json", "test-probabilities.npy")}})
    # Synthetic but internally bound load-audit evidence; never real model data.
    r.write(root / "model/config.json", {"fixture": True})
    r.write(root / "model/model.safetensors.index.json", {"fixture": True})
    backend = r.read(root / "model_backend.json")
    backend.update(config_sha256=r.sha(root / "model/config.json"), index_sha256=r.sha(root / "model/model.safetensors.index.json"))
    r.write(root / "model_backend.json", backend)
    r.write(root / "weight-verification.json", {"status": "passed", "fixture": True})
    (root / "run.log").write_text("synthetic fixture: complete loading and preflight\n")
    (root / "backend-fixture.py").write_text("# synthetic fixture source, never imported\n")
    (root / "load-audit-fixture.py").write_text("# synthetic load-audit source, never imported\n")
    paths = {"run_config": root / "config.json", "model_backend": root / "model_backend.json", "preflight": root / "preflight.json",
             "weight_verification": root / "weight-verification.json", "official_config": root / "model/config.json",
             "official_index": root / "model/model.safetensors.index.json", "backend_source": root / "backend-fixture.py",
             "audit_source": root / "load-audit-fixture.py", "run_log": root / "run.log"}
    r.write(root / "load-integrity-audit.json", {"format_version": "deepseek-load-integrity-audit-v1", "status": "passed",
        "run_dir": str(root), "checks": [{"name": "synthetic-fixture-evidence", "status": "passed"}],
        "inputs": {name: {"exists": True, "path": str(path), "bytes": path.stat().st_size, "sha256": r.sha(path)} for name, path in paths.items()},
        "index_inventory": {"valid": True, "expected_loaded_keys": 1736, "geometry": [43, 256, 64, 512],
                            "fusion_equation": "67612 - 66048 + 172 = 1736"},
        "log_evidence": {"weights_progress_complete": True, "loaded_event_seen": True, "loading_diagnostics": [],
                         "loading_errors": [], "latest_progress": {"loaded": 1736, "total": 1736}},
        "evidence_scope": "Synthetic unit-test fixture only"})
    with (root / "run.log").open("a") as stream:
        stream.write("synthetic fixture: later training log is allowed\n")


def performance_fixture(root):
    sources = {projection: {"projection": projection, "source_tokens": 8, "measured_candidates": 62, "winner_ms": .1,
                            "sha256": projection + "-fixture-sha"} for projection in ("gate_up", "down")}
    base = {"format_version": "native-fp4-performance-cache-reuse-v1", "tool_source_sha256": "fixture-tool-sha",
            "native_identity": {"fixture": True}, "prepared_inputs": [{"fixture": True}],
            "invariants": {"original_kernel_source": True, "true_S_and_grid_unchanged": True},
            "plan": {"sources": sources, "counts": {"create": 2022, "protected": 2}, "protected_tokens": [8309],
                     "unique_input_lengths": 1012, "targets": [{"fixture": "large target list need not enter the report"}]}}
    r.write(root / "performance-cache-applied.json", {**base, "status": "applied", "application": {"created": 2022, "skipped_protected": 2}})
    checks = [{"projection": projection, "tokens": tokens, "S": 6 * tokens, "finite": True, "bitexact": True,
               "native_disk_cache_hit": True, "search_called": False, "maximum_absolute_difference": 0.,
               "source_winner_sha256": source["sha256"]} for projection, source in sources.items() for tokens in (8, 257, 1029, 8309)]
    r.write(root / "performance-cache-gpu-validation.json", {**base, "status": "validation_passed", "validation": {
        "status": "passed", "checks": checks, "scope": "Original native fixed Config versus true-S native disk-cache dispatch; no target-length full autotune",
        "peak_extra_allocated_bytes": 1273370112, "allocator_limit_bytes": 2 * 2**30}})


class IndependentReport(unittest.TestCase):
    def test_incomplete_run_never_claims_passed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            result = r.generate_report(root, plots=True)
            self.assertEqual(result["status"], "running")
            self.assertFalse(result["complete"])
            self.assertTrue(result["missing_artifacts"])
            self.assertEqual(result["checks"], [])
            report = (root / "REPORT.zh-CN.md").read_text()
            self.assertIn("运行中", report)
            self.assertIn("尚未通过最终验证", report)
            self.assertNotIn("独立验证通过", report)

    def test_completed_fixture_checks_and_optional_plot(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            result = r.generate_report(root, plots=True)
            self.assertEqual(result["status"], "passed", result["checks"])
            self.assertEqual(len(result["checks"]), 8)
            self.assertEqual(result["metrics"]["overall"]["accuracy"], .2)
            self.assertEqual(result["plots"]["status"], "created")
            self.assertTrue((root / "test-confusion.png").is_file())
            report = (root / "REPORT.zh-CN.md").read_text()
            self.assertIn("developer", report)
            self.assertIn("teacher-forced", report)
            self.assertIn("五分类", report)

    def test_load_audit_is_required_and_pending_is_never_passed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            (root / "load-integrity-audit.json").unlink()
            missing = r.audit_run(root)
            self.assertEqual(missing["status"], "running")
            self.assertIn("load-integrity-audit.json", missing["missing_artifacts"])
            fixture_path = root / "load-integrity-audit.json"
            r.write(fixture_path, {"status": "running"})
            pending = r.audit_run(root)
            self.assertEqual(pending["status"], "running")
            self.assertIn("load-integrity-audit.json: running", pending["pending_artifacts"])

    def test_load_bound_evidence_change_is_detected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            preflight = r.read(root / "preflight.json")
            preflight["changed"] = True
            r.write(root / "preflight.json", preflight)
            result = r.audit_run(root)
            check = next(item for item in result["checks"] if item["name"] == "completed_load_integrity_and_bound_evidence")
            self.assertFalse(check["ok"])
            self.assertIn("preflight", check["error"])

    def test_performance_reuse_is_disclosed_without_claiming_new_autotune(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            performance_fixture(root)
            result = r.generate_report(root)
            self.assertEqual(result["status"], "passed", result["checks"])
            summary = result["performance_cache"]
            self.assertFalse(summary["target_lengths_individually_autotuned"])
            self.assertEqual(summary["gpu_validation"]["cases"], 8)
            self.assertEqual(summary["gpu_validation"]["tokens"], [8, 257, 1029, 8309])
            self.assertNotIn("targets", summary)
            report = (root / "REPORT.zh-CN.md").read_text()
            self.assertIn("不表示每个目标长度都重新做过 autotune", report)
            self.assertIn("62 个候选", report)

    def test_invalid_performance_validation_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            performance_fixture(root)
            path = root / "performance-cache-gpu-validation.json"
            gpu = r.read(path)
            gpu["validation"]["checks"][0]["bitexact"] = False
            r.write(path, gpu)
            result = r.audit_run(root)
            check = next(item for item in result["checks"] if item["name"] == "native_kernel_configuration_reuse_provenance")
            self.assertFalse(check["ok"])
            self.assertEqual(result["status"], "failed")

    def test_checkpoint_tampering_is_detected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            with (root / "checkpoint.npz").open("ab") as stream:
                stream.write(b"changed")
            result = r.audit_run(root)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"][0]["ok"])

    def test_scaler_is_independently_recomputed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            with np.load(root / "checkpoint.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["mean"] = arrays["mean"] + 1
            np.savez(root / "checkpoint.npz", **arrays)
            result = r.audit_run(root)
            scaler = next(item for item in result["checks"] if item["name"] == "independent_fit_standardizer")
            self.assertFalse(scaler["ok"])
            self.assertIn("FIT-only population mean", scaler["error"])

    def test_metrics_are_independently_recomputed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            test_path = root / "evaluations/test.json"
            test = r.read(test_path)
            test["overall"]["accuracy"] = .99
            r.write(test_path, test)
            result = r.audit_run(root)
            metric = next(item for item in result["checks"] if item["name"] == "independent_test_metrics_and_checkpoint_sample")
            self.assertFalse(metric["ok"])
            self.assertIn("accuracy differs", metric["error"])

    def test_cpu_or_expanded_model_audit_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            backend = r.read(root / "model_backend.json")
            backend["all_parameters_on_cuda"] = False
            r.write(root / "model_backend.json", backend)
            result = r.audit_run(root)
            model = next(item for item in result["checks"] if item["name"] == "native_precision_all_gpu_and_capture_preflight")
            self.assertFalse(model["ok"])

    def test_overlap_in_head_scan_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            plan = r.read(root / "head-scan/plan.json")
            plan["clusters"]["inner"] = plan["clusters"]["statistics"]
            r.write(root / "head-scan/plan.json", plan)
            result = r.audit_run(root)
            isolation = next(item for item in result["checks"] if item["name"] == "committed_features_native_labels_and_source_isolation")
            self.assertFalse(isolation["ok"])

    def test_head_metric_last_bit_rounding_is_tolerated(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            selected = r.read(root / "head-selection.json")
            for item in selected["ranking"]:
                item["macro_f1"] += 1e-16
            r.write(root / "head-selection.json", selected)
            check = r.validate_head_selection(root)
            self.assertEqual(check["heads_verified"], 2752)
            self.assertLess(check["max_absolute_macro_f1_error"], 1e-12)
            self.assertEqual(check["true_support_by_role"], dict.fromkeys(r.ROLES, 1))

    def test_transposed_confusion_axis_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "head-scan/inner-confusion.npz"
            with np.load(path, allow_pickle=False) as archive:
                confusion = archive["confusion"]
            np.savez_compressed(path, confusion=confusion.transpose(0, 2, 1))
            with self.assertRaisesRegex(ValueError, "true-label axis or class support"):
                r.validate_head_selection(root)

    def test_all_head_scores_sort_order_and_top_sixteen_are_verified(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "head-selection.json"
            original = r.read(path)
            changed = json.loads(json.dumps(original))
            changed["ranking"][2740]["macro_f1"] += 1e-6
            r.write(path, changed)
            with self.assertRaisesRegex(ValueError, "independently recomputed"):
                r.validate_head_selection(root)
            changed = json.loads(json.dumps(original))
            changed["ranking"][-2:] = reversed(changed["ranking"][-2:])
            r.write(path, changed)
            with self.assertRaisesRegex(ValueError, "not sorted"):
                r.validate_head_selection(root)
            changed = json.loads(json.dumps(original))
            changed["heads"] = list(reversed(changed["heads"]))
            r.write(path, changed)
            with self.assertRaisesRegex(ValueError, "ranking prefix"):
                r.validate_head_selection(root)

    def test_invalid_lda_coefficients_are_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "head-scan/lda.npz"
            np.savez_compressed(path, weight=np.zeros((2752, 5, 511), np.float64), bias=np.zeros((2752, 5), np.float64))
            with self.assertRaisesRegex(ValueError, "LDA coefficient shapes"):
                r.validate_head_selection(root)


if __name__ == "__main__":
    unittest.main()
