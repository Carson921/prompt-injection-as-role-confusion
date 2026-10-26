"""CPU tests for source equality, no live reads, and migration rollback.

Model-readout validation is mocked only in transaction tests; the copied bytes
are independent fixtures and never experiment classifiers.
"""
import argparse
import copy
import errno
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import reuse_synthetic_pilot as reuse


class ReuseChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pilot-reuse-check-")
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(old_data_dir=self.root / "old-data", new_data_dir=self.root / "new-data",
            old_code_dir=self.root / "old-code", new_code_dir=self.root / "new-code",
            old_run_dir=self.root / "old-run", new_run_dir=self.root / "new-run",
            model_dir=self.root / "fixture-model", old_audit_report=None, dry_run=False, include_plots=False)
        for code, version in [(self.args.old_code_dir, "v5"), (self.args.new_code_dir, "v6")]:
            code.mkdir()
            (code / "synthetic_agent_data.py").write_text("fixture source filter " + version)
            for name in ["synthetic_agent_gpu.py", "synthetic_agent_classifier.py"]:
                (code / name).write_text("frozen independent fixture runtime " + name)
        rows = []
        for split, count in [("fit", 512), ("dev", 128)]:
            for index in range(count):
                text = f"Article {split} {index}. Original complete source body."
                normalized = reuse.text_sha(reuse.normalized_text(text))
                rows.append({"source_id": f"{split}-{index}", "cluster_id": normalized, "split": split,
                    "source": "c4" if index % 2 == 0 else "dolma3", "source_group": None if index % 2 == 0 else "fixture-topic",
                    "source_ordinal": index // 2, "text": text, "normalized_sha256": normalized, "text_sha256": normalized})
        self.rows = rows
        for data, code in [(self.args.old_data_dir, self.args.old_code_dir), (self.args.new_data_dir, self.args.new_code_dir)]:
            folder = data / "pilot"
            folder.mkdir(parents=True)
            (folder / "sources.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            for split in ["fit", "dev"]:
                (folder / f"inputs-{split}.jsonl").write_text(json.dumps({"engineering_fixture_only": split}) + "\n")
            manifest = {"phase": "pilot", "roles": reuse.ROLES, "test_opened": False,
                "configuration": {"code_sha256": reuse.sha(code / "synthetic_agent_data.py"), "tokenizer_sha256": "fixture-backend"},
                "statistics": {"fit": {"sources": 512}, "dev": {"sources": 128}},
                "sha256": {name: reuse.sha(folder / name) for name in ["sources.jsonl", "inputs-fit.jsonl", "inputs-dev.jsonl"]}}
            reuse.atomic_json(folder / "data-manifest.json", manifest)
        old = reuse.pilot_data(self.args.old_data_dir, self.args.old_code_dir)
        self.args.old_audit_proof = self.args.old_data_dir / "pilot" / "article-audit.json"
        audit = {"status": "passed", "auditor": {"type": "independent_fixture"},
            "data_manifest_sha256": old["manifest_sha256"], "sources_sha256": old["sources_sha256"],
            "builder_sha256": old["builder_sha256"], "full_articles_read": 49,
            "sample_ids": [row["source_id"] for row in rows[:49]],
            "sample_reviews": [{"source_id": row["source_id"], "source": row["source"], "split": row["split"]} for row in rows[:49]],
            "selection_rule": "independent unit-test fixture", "limitations": ["Not experiment data"]}
        reuse.atomic_json(self.args.old_audit_proof, audit)
        self.args.old_run_dir.mkdir()
        self.config = {"phase": "pilot", "classifier": {"max_epochs": 15}}
        self.args.new_config = self.args.new_code_dir / "synthetic-agent-120b.json"
        reuse.atomic_json(self.args.new_config, self.config)
        reuse.atomic_json(self.args.old_run_dir / "config.json", self.config)
        reuse.atomic_json(self.args.old_run_dir / "status.json", {"stage": "evaluate", "phase": "pilot", "status": "complete"})
        for name, content in [("best.npz", b"independent fixture NPZ bytes"),
                              ("training.json", b'{"actual_elapsed_seconds":123.456}\n'),
                              ("freeze.json", b'{"unchanged_fixture":true}\n'),
                              ("features/pilot/manifest.json", b'{"immutable_fixture":true}\n')]:
            path = self.args.old_run_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.files = {name: {"sha256": reuse.sha(self.args.old_run_dir / name),
                             "bytes": (self.args.old_run_dir / name).stat().st_size}
                      for name in ["config.json", "best.npz", "training.json", "freeze.json", "features/pilot/manifest.json"]}
        self.runtime = {"parent_status_sha256": reuse.sha(self.args.old_run_dir / "status.json"),
                        "runtime_source_sha256": {name: reuse.sha(self.args.old_code_dir / name)
                            for name in ["synthetic_agent_gpu.py", "synthetic_agent_classifier.py"]},
                        "actual_parent_candidate_timings": {"fixture": {"elapsed_seconds": 123.456}}}

    def tearDown(self):
        self.temporary.cleanup()

    def update_new_hashes(self):
        folder = self.args.new_data_dir / "pilot"
        path = folder / "data-manifest.json"
        value = reuse.read(path)
        for name in value["sha256"]:
            value["sha256"][name] = reuse.sha(folder / name)
        reuse.atomic_json(path, value)

    def test_running_parent_dry_run_checks_inputs_but_not_models(self):
        self.args.dry_run = True
        reuse.atomic_json(self.args.old_run_dir / "status.json", {"stage": "extract", "status": "running"})
        before = reuse.sha(self.args.old_audit_proof)
        with patch.object(reuse, "validated_old_pilot", side_effect=AssertionError("Must not open live model artifacts")) as check:
            report = reuse.reuse(self.args)
        self.assertEqual(report["status"], "inputs_equivalent_waiting_completion")
        self.assertTrue(report["input_equivalence"]["fit_dev_input_bytes_identical"])
        self.assertFalse(self.args.new_run_dir.exists())
        self.assertFalse((self.args.new_data_dir / "pilot" / "article-audit.json").exists())
        self.assertEqual(before, reuse.sha(self.args.old_audit_proof))
        check.assert_not_called()

    def test_raw_text_change_rejected_even_if_normalization_and_inputs_match(self):
        old = reuse.pilot_data(self.args.old_data_dir, self.args.old_code_dir)
        rows = copy.deepcopy(self.rows)
        rows[0]["text"] = rows[0]["text"].upper()  # identical NFKC-casefold text, different original bytes.
        (self.args.new_data_dir / "pilot" / "sources.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.update_new_hashes()
        new = reuse.pilot_data(self.args.new_data_dir, self.args.new_code_dir)
        self.assertEqual(old["input_sha256"], new["input_sha256"])
        with self.assertRaisesRegex(reuse.Refused, "raw/normalized text"):
            reuse.input_equivalence(old, new)

    def test_ordinal_and_input_byte_changes_are_rejected(self):
        old = reuse.pilot_data(self.args.old_data_dir, self.args.old_code_dir)
        new = copy.deepcopy(old)
        new["sources"]["fit-0"]["source_ordinal"] += 1
        with self.assertRaisesRegex(reuse.Refused, "ordinal"):
            reuse.input_equivalence(old, new)
        path = self.args.new_data_dir / "pilot" / "inputs-dev.jsonl"
        with path.open("a") as stream:
            stream.write("\n")
        self.update_new_hashes()
        new = reuse.pilot_data(self.args.new_data_dir, self.args.new_code_dir)
        with self.assertRaisesRegex(reuse.Refused, "JSONL bytes"):
            reuse.input_equivalence(old, new)

    def test_copy_preserves_immutable_bytes_timings_and_carries_audit_honestly(self):
        self.args.new_run_dir.mkdir()
        # Semantically identical preparation config may use different whitespace.
        (self.args.new_run_dir / "config.json").write_text(json.dumps(self.config))
        shutil.copy2(self.args.new_data_dir / "pilot" / "data-manifest.json", self.args.new_run_dir / "data-pilot-manifest.json")
        source_before = {name: reuse.sha(self.args.old_run_dir / name) for name in self.files}
        with patch.object(reuse, "readiness", return_value=[]), patch.object(reuse, "validated_old_pilot", return_value=self.runtime), patch.object(reuse, "copy_plan", return_value=self.files):
            result = reuse.reuse(self.args)
            self.assertEqual(result["status"], "reused")
            self.assertEqual(reuse.reuse(self.args)["status"], "already_reused")
        for name, expected in source_before.items():
            self.assertEqual(reuse.sha(self.args.old_run_dir / name), expected)
            self.assertEqual(reuse.sha(self.args.new_run_dir / name), expected)
        migration = reuse.read(self.args.new_run_dir / "migration.json")
        self.assertEqual(migration["parent_files"], migration["child_files"])
        self.assertFalse(migration["new_training_performed"])
        self.assertEqual(migration["runtime_equivalence"]["actual_parent_candidate_timings"]["fixture"]["elapsed_seconds"], 123.456)
        carried = reuse.read(self.args.new_data_dir / "pilot" / "article-audit.json")
        self.assertEqual(carried["auditor"]["type"], "carried_forward_with_byte_identity")
        self.assertFalse(carried["auditor"]["new_article_reviews_performed"])
        self.assertEqual(carried["full_articles_previously_read"], 49)
        self.assertNotIn("full_articles_read", carried)
        self.assertEqual(carried["data_manifest_sha256"], reuse.sha(self.args.new_data_dir / "pilot" / "data-manifest.json"))
        self.assertNotEqual(carried["builder_sha256"], carried["parent_builder_sha256"])

    def test_postcommit_failure_rolls_back_new_preparation_directory(self):
        self.args.new_run_dir.mkdir()
        (self.args.new_run_dir / "config.json").write_text(json.dumps(self.config))
        original = {"config.json": (self.args.new_run_dir / "config.json").read_bytes()}
        actual_atomic = reuse.atomic_json
        audit_path = self.args.new_data_dir / "pilot" / "article-audit.json"

        def fail_audit(path, value):
            if Path(path) == audit_path:
                raise OSError("independent simulated audit commit failure")
            return actual_atomic(path, value)

        with patch.object(reuse, "readiness", return_value=[]), patch.object(reuse, "validated_old_pilot", return_value=self.runtime), patch.object(reuse, "copy_plan", return_value=self.files), patch.object(reuse, "atomic_json", side_effect=fail_audit):
            with self.assertRaisesRegex(OSError, "simulated audit commit"):
                reuse.reuse(self.args)
        self.assertEqual({p.name: p.read_bytes() for p in self.args.new_run_dir.iterdir()}, original)
        self.assertFalse(audit_path.exists())
        self.assertFalse(list(self.root.glob("new-run.*.tmp")))
        for name, expected in self.files.items():
            self.assertEqual(reuse.sha(self.args.old_run_dir / name), expected["sha256"])

    def test_fuse_metadata_eperm_is_avoided_by_content_only_copy(self):
        self.args.new_run_dir.mkdir()
        (self.args.new_run_dir / "config.json").write_text(json.dumps(self.config))
        reuse.atomic_json(self.args.new_run_dir / "status.json", {"stage": "prepare", "status": "complete"})
        shutil.copyfile(self.args.new_data_dir / "pilot" / "data-manifest.json", self.args.new_run_dir / "data-pilot-manifest.json")
        with patch.object(reuse, "readiness", return_value=[]), patch.object(reuse, "validated_old_pilot", return_value=self.runtime), patch.object(reuse, "copy_plan", return_value=self.files), patch.object(reuse.shutil, "copystat", side_effect=PermissionError(errno.EPERM, "FUSE refuses filesystem metadata")) as metadata:
            result = reuse.reuse(self.args)
        self.assertEqual(result["status"], "reused")
        metadata.assert_not_called()
        for name, expected in self.files.items():
            self.assertEqual(reuse.sha(self.args.new_run_dir / name), expected["sha256"])
            self.assertEqual(reuse.sha(self.args.old_run_dir / name), expected["sha256"])
        self.assertEqual(reuse.sha(self.args.new_run_dir / "data-pilot-manifest.json"), reuse.sha(self.args.new_data_dir / "pilot" / "data-manifest.json"))

    def test_content_copy_eperm_is_not_masked(self):
        self.args.new_run_dir.mkdir()
        (self.args.new_run_dir / "config.json").write_text(json.dumps(self.config))
        before = (self.args.new_run_dir / "config.json").read_bytes()
        actual_copyfile = shutil.copyfile

        def content_write_denied(source, destination, *args, **kwargs):
            if Path(source).name == "best.npz":
                raise PermissionError(errno.EPERM, "FUSE content write really denied")
            return actual_copyfile(source, destination, *args, **kwargs)

        with patch.object(reuse, "readiness", return_value=[]), patch.object(reuse, "validated_old_pilot", return_value=self.runtime), patch.object(reuse, "copy_plan", return_value=self.files), patch.object(reuse.shutil, "copyfile", side_effect=content_write_denied):
            with self.assertRaisesRegex(PermissionError, "content write really denied"):
                reuse.reuse(self.args)
        self.assertEqual((self.args.new_run_dir / "config.json").read_bytes(), before)
        self.assertFalse((self.args.new_run_dir / "best.npz").exists())
        self.assertFalse((self.args.new_data_dir / "pilot" / "article-audit.json").exists())
        self.assertFalse(list(self.root.glob("new-run.reuse-*.tmp")))

    def test_parent_audit_binding_cannot_be_carried_to_other_sources(self):
        old = reuse.pilot_data(self.args.old_data_dir, self.args.old_code_dir)
        new = reuse.pilot_data(self.args.new_data_dir, self.args.new_code_dir)
        value = reuse.read(self.args.old_audit_proof)
        value["sources_sha256"] = "0" * 64
        reuse.atomic_json(self.args.old_audit_proof, value)
        with self.assertRaisesRegex(reuse.Refused, "not bound"):
            reuse.carry_audit(self.args.old_audit_proof, old, new, reuse.input_equivalence(old, new))

    def test_url_enrichment_is_allowed_but_duplicate_urls_are_rejected(self):
        old = reuse.pilot_data(self.args.old_data_dir, self.args.old_code_dir)
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["normalized_url"] = "https://example.test/article/" + row["source_id"]
        path = self.args.new_data_dir / "pilot" / "sources.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.update_new_hashes()
        new = reuse.pilot_data(self.args.new_data_dir, self.args.new_code_dir)
        proof = reuse.input_equivalence(old, new)
        self.assertEqual(proof["child_url_integrity"]["available_urls"], 640)
        self.assertFalse(proof["url_metadata_byte_identity_required"])
        self.assertNotEqual(old["sources_sha256"], new["sources_sha256"])
        rows[-1]["normalized_url"] = rows[0]["normalized_url"]  # DEV/FIT collision.
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.update_new_hashes()
        with self.assertRaisesRegex(reuse.Refused, "Duplicate available normalized URL"):
            reuse.pilot_data(self.args.new_data_dir, self.args.new_code_dir)

    def test_actual_source_split_counts_cannot_hide_behind_manifest_counts(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "dev"
        (self.args.new_data_dir / "pilot" / "sources.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.update_new_hashes()
        with self.assertRaisesRegex(reuse.Refused, "Actual pilot source rows"):
            reuse.pilot_data(self.args.new_data_dir, self.args.new_code_dir)

    def test_cleanup_failure_preserves_fully_committed_child(self):
        self.args.new_run_dir.mkdir()
        (self.args.new_run_dir / "config.json").write_text(json.dumps(self.config))
        actual_rmtree = shutil.rmtree

        def partially_fail_backup(path, *args, **kwargs):
            if ".pre-reuse-" in Path(path).name:
                (Path(path) / "config.json").unlink()
                raise OSError("independent simulated partial backup deletion")
            return actual_rmtree(path, *args, **kwargs)

        with patch.object(reuse, "readiness", return_value=[]), patch.object(reuse, "validated_old_pilot", return_value=self.runtime), patch.object(reuse, "copy_plan", return_value=self.files), patch.object(reuse.shutil, "rmtree", side_effect=partially_fail_backup):
            result = reuse.reuse(self.args)
        self.assertEqual(result["status"], "reused")
        self.assertTrue(result["backup_cleanup_warning"]["committed_child_preserved"])
        self.assertEqual(reuse.sha(self.args.new_run_dir / "best.npz"), self.files["best.npz"]["sha256"])
        self.assertEqual(reuse.read(self.args.new_data_dir / "pilot" / "article-audit.json")["status"], "passed")
        self.assertTrue(reuse.read(self.args.new_run_dir / "migration.json")["backup_cleanup_warning"]["committed_child_preserved"])

    def test_copy_plan_excludes_unknown_dev_debug_files_and_feature_payloads(self):
        root = self.args.old_run_dir
        for name in reuse.ROOT_FILES:
            path = root / name
            if not path.exists():
                path.write_bytes(b"independent selection-only fixture")
        for candidate in reuse.CANDIDATES:
            directory = root / "candidates" / candidate
            directory.mkdir(parents=True)
            for name in ["best.npz", "history.json", "complete.json"]:
                (directory / name).write_bytes(b"independent selection-only fixture")
        (root / "evaluations").mkdir()
        for name in ["dev.json", "dev-probabilities.npy", "dev-debug-inputs.json"]:
            (root / "evaluations" / name).write_bytes(b"independent selection-only fixture")
        feature = root / "features" / "pilot"
        shard = feature / "fit-000000"
        shard.mkdir()
        for name in reuse.FEATURE_FILES:
            (shard / name).write_bytes(b"independent selection-only fixture")
        commit = {"files": {name: reuse.sha(shard / name) for name in reuse.FEATURE_FILES}}
        reuse.atomic_json(shard / "complete.json", commit)
        reuse.atomic_json(feature / "manifest.json", {"splits": {"fit": [{"path": shard.name}]}})
        plan = reuse.copy_plan(self.args)
        self.assertNotIn("evaluations/dev-debug-inputs.json", plan)
        (shard / "raw-source.json").write_text("independent raw-body export boundary fixture")
        commit["files"]["raw-source.json"] = reuse.sha(shard / "raw-source.json")
        reuse.atomic_json(shard / "complete.json", commit)
        with self.assertRaisesRegex(reuse.Refused, "unknown files"):
            reuse.copy_plan(self.args)

    def test_complete_featurebank_npz_and_dev_validation_then_real_fixture_copy(self):
        """A complete independent numerical fixture; no training/GPU process."""
        source_directory = Path(reuse.__file__).parent
        for directory in [self.args.old_code_dir, self.args.new_code_dir]:
            for name in ["synthetic_agent_gpu.py", "synthetic_agent_classifier.py"]:
                shutil.copy2(source_directory / name, directory / name)
        classifier = reuse.module_from_file(self.args.old_code_dir / "synthetic_agent_classifier.py", "_reuse_fixture_classifier")
        gpu = reuse.module_from_file(self.args.old_code_dir / "synthetic_agent_gpu.py", "_reuse_fixture_gpu")
        target_rows = {}
        for split in ["fit", "dev"]:
            records = []
            documents = [row for row in self.rows if row["split"] == split]
            for index, source in enumerate(documents):
                records.append({"source_id": source["source_id"], "cluster_id": source["cluster_id"], "split": split,
                    "input_ids": [1, index % 6 + 20, 2], "targets": [{"token_ix": 1, "label": index % 6,
                        "role": reuse.ROLES[index % 6], "family_id": source["source_id"] + ":body-0"}]})
            target_rows[split] = records
            payload = "\n".join(json.dumps(row) for row in records) + "\n"
            for directory in [self.args.old_data_dir, self.args.new_data_dir]:
                (directory / "pilot" / f"inputs-{split}.jsonl").write_text(payload)
                path = directory / "pilot" / "data-manifest.json"
                manifest = reuse.read(path)
                manifest["sha256"][f"inputs-{split}.jsonl"] = reuse.sha(directory / "pilot" / f"inputs-{split}.jsonl")
                reuse.atomic_json(path, manifest)
        old = reuse.pilot_data(self.args.old_data_dir, self.args.old_code_dir)
        audit = reuse.read(self.args.old_audit_proof)
        audit["data_manifest_sha256"] = old["manifest_sha256"]
        reuse.atomic_json(self.args.old_audit_proof, audit)
        root = self.args.old_run_dir
        model_binding = {"independent_engineering_fixture": True}
        selected = {"roles": reuse.ROLES, "heads": list(range(16)), "no_dev_test_or_historical_probe_used": True,
            "signature": {"model_binding": model_binding, "pilot_fit_sha256": old["input_sha256"]["fit"],
                          "implementation_sha256": reuse.sha(self.args.old_code_dir / "synthetic_agent_gpu.py"),
                          "config_sha256": gpu.object_digest(self.config)}}
        reuse.atomic_json(root / "head-selection.json", selected)
        reuse.atomic_json(root / "preflight.json", {"status": "passed", "maximum_input_tokens": 4096, "model_binding": model_binding})
        reuse.atomic_json(root / "model_backend.json", {"model_binding": model_binding, "expert_backends": ["Mxfp4GptOssExperts"]})
        feature_root = root / "features" / "pilot"
        signature = {"phase": "pilot", "capture": "pre-o_proj", "heads": list(range(16)), "model_binding": model_binding,
            "selection_sha256": reuse.sha(root / "head-selection.json"), "pilot_fit_sha256": old["input_sha256"]["fit"],
            "implementation_sha256": reuse.sha(self.args.old_code_dir / "synthetic_agent_gpu.py"),
            "config_sha256": gpu.object_digest(self.config)}
        feature_manifest = {"format_version": classifier.FEATURE_FORMAT, "roles": reuse.ROLES, "feature_dim": 1024,
            "feature_position": "pre-o_proj", "heads": list(range(16)), "model_binding": model_binding,
            "selection_sha256": signature["selection_sha256"], "pilot_fit_sha256": signature["pilot_fit_sha256"],
            "signature": signature, "config_hash": signature["config_sha256"], "data_hash": gpu.object_digest(old["input_sha256"]),
            "splits": {}, "split_sources": old["input_sha256"], "progress": {}}
        for split, records in target_rows.items():
            shard = feature_root / (split + "-000000")
            shard.mkdir()
            n = len(records)
            arrays = {"x": np.zeros((n, 1024), np.float16), "y": np.array([r["targets"][0]["label"] for r in records], np.int64),
                "source": np.array([r["cluster_id"] for r in records]),
                "family": np.array([r["targets"][0]["family_id"] for r in records]),
                "structure": np.array(["fixture:" + r["targets"][0]["role"] for r in records]), "weight": np.ones(n, np.float32)}
            for name, value in arrays.items():
                np.save(shard / (name + ".npy"), value, allow_pickle=False)
            reuse.atomic_json(shard / "complete.json", {"rows": n, "record_start": 0, "record_end": n,
                "signature": {**signature, "split": split, "input_sha256": old["input_sha256"][split]},
                "files": {name + ".npy": reuse.sha(shard / (name + ".npy")) for name in arrays}})
            feature_manifest["splits"][split] = [{"path": shard.name, "rows": n, "sha256": reuse.sha(shard / "complete.json")}]
            feature_manifest["progress"][split] = {"complete": True, "rows": n, "records": n}
        reuse.atomic_json(feature_root / "manifest.json", feature_manifest)
        fit, dev = classifier.FeatureBank(feature_root, "fit"), classifier.FeatureBank(feature_root, "dev")
        options = classifier._training_options(self.config)
        split_hashes = {"fit": fit.split_identity_hash, "dev": dev.split_identity_hash}
        implementation = reuse.sha(self.args.old_code_dir / "synthetic_agent_classifier.py")
        training_hash = reuse.object_sha({"fit_features": split_hashes["fit"], "dev_features": split_hashes["dev"],
            "options": options, "implementation_sha256": implementation, "candidates": classifier.CANDIDATES})
        binding = fit.binding()
        training = {"phase": "pilot", "implementation_sha256": implementation, "options": options,
            "feature_split_hashes": split_hashes, "training_hash": training_hash, "binding": binding,
            "feature_manifest_hash": reuse.sha(feature_root / "manifest.json")}
        reuse.atomic_json(root / "training.json", training)
        mean, scale = np.zeros(1024, np.float32), np.ones(1024, np.float32)
        np.savez(root / "standardizer.npz", training_hash=np.array(training_hash), mean=mean, scale=scale)
        reuse.atomic_json(root / "resume-validation.json", {"passed": True, "device": "cuda:0", "weight_max_absolute_error": 0,
            "loss_absolute_error": 0, "numpy_export_max_absolute_error": 0, "independent_fixture_only": True})
        probabilities = np.full((len(dev), 6), np.float32(1 / 6))
        metrics = classifier.grouped_metrics(dev, probabilities)
        complete = []
        for candidate_id, hidden, seed in classifier.CANDIDATES:
            directory = root / "candidates" / candidate_id
            directory.mkdir(parents=True)
            metadata = {**binding, "format_version": classifier.PROBE_FORMAT, "candidate_id": candidate_id,
                "hidden": hidden, "seed": seed, "dropout": .1, "training_hash": training_hash,
                "engineering_smoke_only": False, "independent_fixture_only": True}
            weights = {"weight::0.weight": np.zeros((hidden or 6, 1024), np.float32), "weight::0.bias": np.zeros(hidden or 6, np.float32)}
            if hidden:
                weights.update({"weight::3.weight": np.zeros((6, hidden), np.float32), "weight::3.bias": np.zeros(6, np.float32)})
            np.savez(directory / "best.npz", metadata=np.array(json.dumps(metadata)), mean=mean, scale=scale, **weights)
            reuse.atomic_json(directory / "history.json", [{"epoch": i + 1, "dev": metrics} for i in range(15)])
            record = {"candidate_id": candidate_id, "training_hash": training_hash,
                "probe_sha256": reuse.sha(directory / "best.npz"), "history_sha256": reuse.sha(directory / "history.json"),
                "epochs": 15, "elapsed_seconds": 10.5, "export_max_absolute_error": 0, "dev": metrics}
            reuse.atomic_json(directory / "complete.json", record)
            complete.append(record)
        winner = min(complete, key=lambda row: classifier.selection_key(row["dev"], row["candidate_id"]))
        reuse.atomic_json(root / "selection.json", {"candidate_results": complete, "candidate_id": winner["candidate_id"],
            "training_hash": training_hash, "selection_split": "dev"})
        shutil.copy2(root / "candidates" / winner["candidate_id"] / "best.npz", root / "best.npz")
        probe = classifier.load_probe(root / "best.npz")
        reuse.atomic_json(root / "freeze.json", {"probe_sha256": reuse.sha(root / "best.npz"),
            "selection_sha256": reuse.sha(root / "selection.json"), "training_sha256": reuse.sha(root / "training.json"),
            "selection_split": "dev", "metadata": probe.metadata})
        (root / "evaluations").mkdir()
        np.save(root / "evaluations" / "dev-probabilities.npy", probabilities, allow_pickle=False)
        reuse.atomic_json(root / "evaluations" / "dev.json", {**metrics, "split": "dev",
            "probe_sha256": reuse.sha(root / "best.npz"), "feature_manifest_sha256": reuse.sha(feature_root / "manifest.json")})
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        for stage in ["scan", "extract", "train", "freeze", "evaluate"]:
            reuse.atomic_json(root / f"{stage}-stage.json", {"stage": stage, "phase": "pilot", "status": "complete",
                "config_hash": config_hash, "seconds": 123.456})
        self.assertEqual(reuse.readiness(root), [])
        self.args.dry_run = True
        with patch.object(reuse, "current_runtime", return_value=(model_binding, "fixture-backend")):
            report = reuse.reuse(self.args)
            self.assertEqual(report["status"], "ready_to_reuse")
            self.assertEqual(report["runtime_equivalence"]["dev_export_max_absolute_error"], 0)
            self.assertFalse(self.args.new_run_dir.exists())
            self.args.dry_run = False
            self.assertEqual(reuse.reuse(self.args)["status"], "reused")
        self.assertEqual(reuse.sha(root / "best.npz"), reuse.sha(self.args.new_run_dir / "best.npz"))
        self.assertFalse((self.args.new_run_dir / "head-scan" / "statistics-final.npz").exists())


if __name__ == "__main__":
    unittest.main()
