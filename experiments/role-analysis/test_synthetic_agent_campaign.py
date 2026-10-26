"""CPU-only campaign safety/recovery checks; no real stage process is launched."""
import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run_synthetic_agent_campaign as driver


class CampaignChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="six-role-campaign-check-")
        self.root = Path(self.temporary.name)
        self.campaign = driver.Campaign.__new__(driver.Campaign)
        self.campaign.work = self.root / "campaign"
        self.campaign.work.mkdir()
        self.campaign.runs = {phase: self.root / phase for phase in ["pilot", "formal"]}
        for path in self.campaign.runs.values():
            path.mkdir()
        self.campaign.binding = {"fixture": True}
        self.campaign.remaining = lambda: 60
        self.campaign.publish = Mock()
        self.campaign.configs = {"pilot": {}, "formal": {}}
        self.campaign.args = argparse.Namespace(code_dir=self.root / "code", data_dir=self.root / "data",
                                                model_dir=self.root / "model", dense_diagnostics=False,
                                                devices="0,1,2,3")
        self.campaign.args.code_dir.mkdir()
        self.campaign.args.data_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_frozen_training_is_adopted_but_never_retrained(self):
        c = self.campaign
        driver.atomic_json(c.runs["pilot"] / "freeze.json", {"fixture": True})
        c._execute = Mock(side_effect=AssertionError("A real process must not start"))
        receipt = c.step("pilot", "train", lambda: {"fixed_probe": "abc"})
        self.assertTrue(receipt["execution"]["adopted_verified_atomic_artifacts"])
        self.assertEqual(receipt, c.step("pilot", "train", lambda: {"fixed_probe": "abc"}))
        with self.assertRaisesRegex(driver.CampaignStop, "Completed stage changed"):
            c.step("pilot", "train", lambda: {"fixed_probe": "changed"})
        (c.work / "receipts" / "pilot.train.json").unlink()
        with self.assertRaisesRegex(driver.CampaignStop, "already frozen"):
            c.step("pilot", "train", lambda: None)
        c._execute.assert_not_called()

    def test_failed_stage_cannot_create_success_receipt(self):
        c = self.campaign
        c._command = Mock(return_value=["fixture-no-execution"])
        c._execute = Mock(side_effect=driver.CampaignStop("failed", "fixture failure"))
        with self.assertRaisesRegex(driver.CampaignStop, "fixture failure"):
            c.step("pilot", "scan", lambda: None)
        self.assertFalse((c.work / "receipts" / "pilot.scan.json").exists())

    def test_deadline_kills_child_process_group(self):
        c = self.campaign
        c.remaining = lambda: 2
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired(["fixture"], 2), -9]
        process.poll.return_value = -9
        with patch.object(driver.subprocess, "Popen", return_value=process) as start, patch.object(driver.os, "killpg") as kill:
            with self.assertRaises(driver.CampaignStop) as error:
                c._execute("pilot.fixture", ["not-executed"])
        self.assertEqual(error.exception.status, "deadline_exceeded")
        self.assertEqual(process.wait.call_args_list[0].kwargs, {"timeout": 2})
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(123456, signal.SIGKILL)

    def test_slow_status_or_process_start_does_not_extend_deadline(self):
        c = self.campaign
        c.remaining = Mock(side_effect=[10, -1])
        with patch.object(driver.subprocess, "Popen") as start:
            with self.assertRaises(driver.CampaignStop) as error:
                c._execute("pilot.status-delay", ["not-executed"])
        self.assertEqual(error.exception.status, "deadline_exceeded")
        start.assert_not_called()
        c.remaining = Mock(side_effect=[10, 8, 5])
        process = Mock(pid=123456)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(driver.subprocess, "Popen", return_value=process):
            c._execute("pilot.spawn-delay", ["not-executed"])
        process.wait.assert_called_once_with(timeout=5)

    def test_article_audit_and_external_wait_gates(self):
        c = self.campaign
        folder = c.args.data_dir / "pilot"
        folder.mkdir()
        (folder / "data-manifest.json").write_text("{}")
        (folder / "sources.jsonl").write_text("fixture\n")
        with self.assertRaises(driver.CampaignStop) as error:
            c.audit_gate("pilot")
        self.assertEqual(error.exception.status, "waiting_article_audit")
        path = folder / "article-audit.json"
        driver.atomic_json(path, {"status": "passed", "data_manifest_sha256": driver.sha(folder / "data-manifest.json"),
                                  "sources_sha256": driver.sha(folder / "sources.jsonl")})
        self.assertTrue(c.audit_gate("pilot"))
        (folder / "sources.jsonl").write_text("changed\n")
        with self.assertRaisesRegex(driver.CampaignStop, "source articles"):
            c.audit_gate("pilot")
        for method, status in [(c.external_protocol_gate, "waiting_external_protocol"), (c.legacy_hash_gate, "waiting_legacy_hashes")]:
            with self.assertRaises(driver.CampaignStop) as error:
                method()
            self.assertEqual(error.exception.status, status)

    def test_venv_python_symlink_is_preserved(self):
        executable = self.root / "venv" / "bin" / "python"
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
        args = driver.parse_args(["--python", str(executable), "--code-dir", str(self.root),
            "--model-dir", str(self.root), "--data-dir", str(self.root), "--run-root", str(self.root),
            "--cache-dir", str(self.root), "--pilot-config", str(self.root / "pilot.json"),
            "--formal-config", str(self.root / "formal.json")])
        self.assertEqual(args.python, executable)
        self.assertNotEqual(args.python, executable.resolve())

    def test_immutable_fit_identity_survives_legal_test_append(self):
        c = self.campaign
        code = c.args.code_dir / "synthetic_agent_gpu.py"
        code.write_text("independent engineering fixture")
        for phase in ["pilot", "formal"]:
            (c.args.data_dir / phase).mkdir()
            (c.args.data_dir / phase / "inputs-fit.jsonl").write_text(phase + " fixture")
        selection = c.runs["formal"] / "head-selection.json"
        selection.write_text("fixture selection")
        from synthetic_agent_gpu import object_digest
        signature = {"implementation_sha256": driver.sha(code), "model_binding": {"fixture": True},
                     "heads": list(range(16)), "selection_sha256": driver.sha(selection),
                     "pilot_fit_sha256": driver.sha(c.args.data_dir / "pilot" / "inputs-fit.jsonl"),
                     "config_sha256": object_digest({}), "phase": "formal", "capture": "pre-o_proj"}
        feature_root = c.runs["formal"] / "features" / "formal"
        shard = feature_root / "fit-000000"
        shard.mkdir(parents=True)
        arrays = {"x": np.zeros((1, 1024), np.float16), "y": np.zeros(1, np.int64),
                  "structure": np.array(["single:S"]), "source": np.array(["s1"]),
                  "family": np.array(["s1:0"]), "weight": np.ones(1, np.float32)}
        for name, array in arrays.items():
            np.save(shard / (name + ".npy"), array, allow_pickle=False)
        input_sha = driver.sha(c.args.data_dir / "formal" / "inputs-fit.jsonl")
        commit = {"signature": {**signature, "split": "fit", "input_sha256": input_sha}, "rows": 1,
                  "files": {name + ".npy": driver.sha(shard / (name + ".npy")) for name in arrays}}
        driver.atomic_json(shard / "complete.json", commit)
        manifest = {"format_version": "synthetic-agent-features-v1", "roles": driver.ROLES,
                    "feature_dim": 1024, "feature_position": "pre-o_proj", "model_binding": {"fixture": True},
                    "config_hash": object_digest({}), "selection_sha256": driver.sha(selection),
                    "pilot_fit_sha256": signature["pilot_fit_sha256"], "signature": signature,
                    "data_hash": object_digest({"fit": input_sha}), "split_sources": {"fit": input_sha},
                    "splits": {"fit": [{"path": "fit-000000", "rows": 1, "sha256": driver.sha(shard / "complete.json")}]},
                    "progress": {"fit": {"complete": True, "records": 1, "rows": 1}}}
        path = feature_root / "manifest.json"
        driver.atomic_json(path, manifest)
        with patch("synthetic_agent_gpu.model_binding", return_value={"fixture": True}):
            first = c.feature_manifest("formal", "fit")
            manifest["last_extraction_seconds"] = 0.001
            manifest["splits"]["test"] = [{"unrelated": "new final test shard"}]
            manifest["split_sources"]["test"] = "new final test hash"
            manifest["progress"]["test"] = {"complete": True, "rows": 5}
            driver.atomic_json(path, manifest)
            self.assertEqual(first, c.feature_manifest("formal", "fit"))
            with (shard / "x.npy").open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(driver.CampaignStop, "bytes changed"):
                c.feature_manifest("formal", "fit")

    def test_budget_refusal_preserves_full_quotas_and_inputs(self):
        c = self.campaign
        (c.work / "logs").mkdir()
        for phase in ["pilot", "formal"]:
            folder = c.args.data_dir / phase
            folder.mkdir()
            stats = {split: {"rows": count * 100, "sources": count} for split, count in driver.QUOTAS[phase].items()}
            driver.atomic_json(folder / "data-manifest.json", {"input_tokens": {split: count * 1000 for split, count in driver.QUOTAS[phase].items()}, "statistics": stats})
        driver.atomic_json(c.runs["pilot"] / "extract-stage.json", {"seconds": 640})
        driver.atomic_json(c.runs["pilot"] / "features" / "pilot" / "manifest.json", {
            "progress": {"fit": {"newly_processed_input_tokens": 512000}, "dev": {"newly_processed_input_tokens": 128000}}})
        candidates = [{"candidate_id": str(i), "epochs": 15, "elapsed_seconds": 10} for i in range(7)]
        driver.atomic_json(c.runs["pilot"] / "selection.json", {"candidate_results": candidates})
        before = driver.sha(c.args.data_dir / "formal" / "data-manifest.json")
        with self.assertRaises(driver.CampaignStop) as error:
            c.estimate_budget()
        self.assertEqual(error.exception.status, "budget_insufficient")
        estimate = driver.read(c.work / "formal-budget-estimate.json")
        self.assertFalse(estimate["fits"])
        self.assertFalse(estimate["quota_reduction_permitted"])
        self.assertEqual(before, driver.sha(c.args.data_dir / "formal" / "data-manifest.json"))


if __name__ == "__main__":
    unittest.main()
