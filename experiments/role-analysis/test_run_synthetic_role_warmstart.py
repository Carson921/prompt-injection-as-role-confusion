import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import run_synthetic_role_warmstart as driver
import watch_synthetic_role_warmstart as watchdog


class ResourceSchedulingTests(unittest.TestCase):
    def test_expired_budget_prevents_subprocess_launch_even_with_old_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = object.__new__(driver.Campaign)
            campaign.receipts = Path(directory)
            (campaign.receipts / "preflight.json").write_text(json.dumps({"returncode": 0, "argv": ["old"]}))
            with patch.object(campaign, "remaining", side_effect=TimeoutError("expired")), patch.object(driver.subprocess, "run") as run:
                with self.assertRaises(TimeoutError):
                    campaign.command("preflight", ["old"])
                run.assert_not_called()

    def test_successful_receipt_cannot_hide_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = object.__new__(driver.Campaign)
            campaign.receipts = Path(directory)
            (campaign.receipts / "preflight.json").write_text(json.dumps({"returncode": 0, "argv": ["old"],
                "artifact_proof": {"preflight_sha256": "previous"}}))
            with patch.object(campaign, "remaining", return_value=100), patch.object(campaign, "artifact_proof", return_value={"preflight_sha256": "changed"}), patch.object(driver.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    campaign.command("preflight", ["old"])
                run.assert_not_called()

    def test_scale_selection_cannot_be_resumed_without_bound_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scale-selection.json").write_text(json.dumps({"chosen": {"fit_sources": 512}}))
            campaign = object.__new__(driver.Campaign)
            campaign.root = root
            with self.assertRaises((ValueError, FileNotFoundError)):
                campaign.choose_scale()

    def test_timing_prediction_includes_load_io_and_margin(self):
        measured = {str(limit): {"records": 1, "input_tokens": limit, "forward_seconds": limit / 100}
                    for limit in (512, 1024, 2048, 4096)}
        total = sum(x["forward_seconds"] for x in measured.values())
        pilot = {"length_bucket_timings": measured, "attempt_seconds": total + 20 + 10, "model_load_seconds": 20}
        counts = {key: {"records": 2, "input_tokens": value["input_tokens"] * 2} for key, value in measured.items()}
        self.assertAlmostEqual(driver.predict_seconds(counts, pilot), (2 * total + 20 + 20) * 1.25)

    def test_size_choice_never_uses_model_scores_and_preserves_test_budget(self):
        import json
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "pilot-features").mkdir()
            now = time.time()
            (root / "budget.json").write_text(json.dumps({"started_unix": now - 1800, "deadline_unix": now + 19800}))
            measured = {str(limit): {"records": 1, "input_tokens": limit, "forward_seconds": limit / 100}
                        for limit in (512, 1024, 2048, 4096)}
            (root / "pilot-features" / "manifest.json").write_text(json.dumps({"progress": {"fit": {
                "length_bucket_timings": measured, "attempt_seconds": 76.8, "model_load_seconds": 0}}}))
            for name, records in (("inputs-fit.jsonl", 240), ("inputs-fit-conservative.jsonl", 80),
                                  ("inputs-dev.jsonl", 32), ("inputs-test.jsonl", 32)):
                (root / "data" / name).write_text((json.dumps({"input_ids": [1] * 4096}) + "\n") * records)
            campaign = object.__new__(driver.Campaign)
            campaign.root, campaign.data, campaign.budget = root, root / "data", root / "budget.json"
            chosen = campaign.choose_scale()
            self.assertEqual(chosen["chosen"]["fit_sources"], 256)
            self.assertFalse(chosen["estimates"][0]["fits_windows"])
            self.assertIn("test_seconds_with_25pct_margin", chosen["chosen"])

    def test_watchdog_matches_only_script_and_exact_run_root(self):
        # Helpers never treat incidental shell strings as permission to signal a PID.
        self.assertIsNone(watchdog.option(["python", "-c", "--run-dir"], "--run-dir"))
        self.assertEqual(watchdog.option(["python", "train.py", "--run-dir", "/root/one/run"], "--run-dir"), "/root/one/run")
        with patch.object(watchdog, "targets", return_value=[{"pid": 71, "start_ticks": 2}]):
            self.assertFalse(watchdog.same_process(Path("/root/one"), {"pid": 71, "start_ticks": 1}))


if __name__ == "__main__":
    unittest.main()
