"""Targeted CPU tests; synthetic metadata only, never invoke tool main."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("resources", Path(__file__).with_name("summarize_synthetic_resources.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class Fixture:
    def __init__(self, root):
        root = Path(root)
        self.old, self.new = root / "runs-v5", root / "runs-v7"
        self.scan = 1793025434.2479506
        self.first = 1793025687.522992
        self.train = 1793031409.914779
        self.train_seconds = 406.8144552707672
        self.now = self.train + 1000
        self.csv = self.old / "campaign/gpu-all-stages.csv"
        self.claim, self.report, self.observation = [root / name for name in ("dispatch-claim.json", "report-complete.json", "observation.json")]
        self.csv.parent.mkdir(parents=True)
        times = [(self.first, 1000), (self.first + 5, 1200), (self.first + 10, 1100),
                 (self.scan + 1080, 1199), (self.train + 1, 2000), (self.train + 200, 2400),
                 (self.train + self.train_seconds - 1, 2300)]
        with self.csv.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unix_time", "gpu", "memory_used_mib", "utilization_percent"])
            for t, memory in times:
                for gpu in range(4):
                    writer.writerow([t + gpu * .00001, gpu, memory + gpu, 75])
        dump(self.csv.with_name("gpu-all-stages-summary.json"), {"sample_period_seconds": 5, "device_wide_peak_mib": [99999] * 4})
        self.markers = {"pilot.scan": {"stage": "scan", "phase": "pilot", "status": "complete", "started_unix": self.scan, "seconds": 1081.1371684074402},
                        "pilot.train": {"stage": "train", "phase": "pilot", "status": "complete", "started_unix": self.train, "seconds": self.train_seconds}}
        for name, marker in self.markers.items():
            self.receipt(self.old, name, marker)
            self.receipt(self.new, name, marker, adopted=True)
        dump(self.new / "pilot/migration.json", {"parent_run_dir": str(self.old / "pilot"), "new_training_performed": False})
        preflight = {"started_unix": self.scan - 3000, "seconds": 51.36, "device_wide_peak_mib": [23170, 21990, 21990, 23090],
                     "pytorch_peak": [{"device": g, "allocated_bytes": 22000000000, "reserved_bytes": 23000000000} for g in range(4)],
                     "memory_note": "PyTorch counter begins after weight loading"}
        for path in [self.old / "pilot/preflight.json", self.new / "pilot/preflight.json", self.new / "formal/preflight.json"]:
            dump(path, preflight)
        dump(self.new / "formal/pilot-inheritance.json", {"preflight_sha256": r.identity(self.old / "pilot/preflight.json")["sha256"]})
        dump(self.new / "formal/status.json", {"phase": "formal", "stage": "extract", "status": "running", "started_unix": self.train + 500})

    def receipt(self, root, name, marker, adopted=False):
        dump(root / "campaign/receipts" / (name + ".json"), {"name": name, "stage_marker_snapshot": marker,
             "execution": {"seconds": 99999, "started_unix": 1, "adopted_verified_atomic_artifacts": adopted},
             "completed_unix": 9999999999})

    def build(self, **kwargs):
        return r.build_report(self.csv, self.new, self.claim, self.report, observed_unix=self.now, **kwargs)


class ResourcesTest(unittest.TestCase):
    def test_original_receipts_not_adoption_walltime_and_253_second_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            out = f.build()
            scan = next(s for s in out["stages"] if s["name"] == "pilot.scan")
            self.assertEqual(scan["start_unix"], f.scan)
            self.assertAlmostEqual(scan["seconds"], 1081.1371684074402)
            self.assertTrue(scan["reused_without_new_execution"])
            self.assertEqual(scan["original_version"], "runs-v5")
            self.assertEqual(len(scan["receipt_chain"]), 2)
            self.assertEqual(scan["source_receipt_sha256"], r.identity(f.old / "campaign/receipts/pilot.scan.json")["sha256"])
            gpu = scan["gpu"]["0"]
            self.assertAlmostEqual(gpu["leading_unobserved_seconds"], 253.27504134178162, places=5)
            self.assertTrue(gpu["partial_coverage"])
            self.assertEqual(gpu["sampled_memory_peak_mib"], 1200)
            self.assertEqual(gpu["peak_first_unix"], f.first + 5)
            self.assertEqual(gpu["sample_count"], 4)
            self.assertTrue(gpu["long_internal_gaps"])
            self.assertEqual(out["csv"]["sha256"], hashlib.sha256(f.csv.read_bytes()).hexdigest())
            self.assertEqual(out["preflight"]["original_version"], "runs-v5")
            self.assertEqual(out["preflight"]["device_wide_peak_mib"][0], 23170)

    def test_pending_and_dense_missing_timing_never_invent_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            f.receipt(f.new, "formal.dense", {"stage": "dense", "phase": "formal", "kind": "verified_functional_commit"})
            out = f.build()
            for name in ["formal.extract", "formal.train", "formal.extract.test", "formal.extract.external"]:
                stage = next(s for s in out["stages"] if s["name"] == name)
                self.assertEqual(stage["status"], "pending")
                self.assertNotIn("gpu", stage)
            test = next(s for s in out["stages"] if s["name"] == "formal.extract.test")
            self.assertNotIn("current_stage_status", test)
            dense = next(s for s in out["stages"] if s["name"] == "formal.dense")
            self.assertEqual(dense["status"], "complete_timing_unavailable")
            self.assertNotIn("seconds", dense)
            text = r.markdown(out)
            self.assertIn("pending", text)
            self.assertIn("权重加载后", text)

    def test_gardening_bound_uses_hash_bound_root_observation_not_dispatch_return_or_elapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            dump(f.claim, {"claimed_unix": f.train + 2, "dispatched_unix": f.train + 50})
            dump(f.report, {"no_prediction_arrays_here": True})
            dump(f.observation, {"observed_unix": f.train + 300,
                 "report_complete_sha256": r.identity(f.report)["sha256"], "actual_infer_processes_absent": True})
            out = f.build(completion_observation=f.observation)
            bound = out["gardening_possible_concurrency"]
            self.assertEqual(bound["start_unix"], f.train + 2)
            self.assertEqual(bound["end_unix"], f.train + 300)
            self.assertEqual(bound["end_source"], "root_report_hash_bound_completion_observation")
            train = next(s for s in out["stages"] if s["name"] == "pilot.train")
            self.assertEqual(train["possible_gardening_overlap_seconds"], 298)
            self.assertTrue(train["gpu"]["0"]["memory_peak_may_include_gardening"])
            self.assertEqual(train["gpu"]["0"]["sampled_memory_peak_mib"], 2400)
            for field in ["report_complete_sha256", "actual_infer_processes_absent"]:
                bad = json.loads(f.observation.read_text())
                bad[field] = "0" * 64 if field.endswith("sha256") else False
                dump(f.observation, bad)
                with self.assertRaises(ValueError):
                    f.build(completion_observation=f.observation)

    def test_open_gardening_bound_and_requested_observation_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            dump(f.claim, {"claimed_unix": f.train + 2})
            out = f.build(completion_observation=f.observation)
            self.assertEqual(out["gardening_possible_concurrency"]["status"], "awaiting_root_completion_observation")
            self.assertEqual(out["gardening_possible_concurrency"]["end_unix"], f.now)

    def test_partial_csv_tail_is_hashed_but_not_invented_as_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            with f.csv.open("ab") as stream:
                stream.write(b"1793039999,0,999")
            samples, source = r.snapshot_csv(f.csv)
            self.assertEqual(source["discarded_partial_tail_bytes"], len(b"1793039999,0,999"))
            self.assertEqual(source["sample_count"], 28)
            self.assertEqual(source["sha256"], hashlib.sha256(f.csv.read_bytes()).hexdigest())
            self.assertFalse(source["identity_is_final"])
            with self.assertRaisesRegex(ValueError, "incomplete final row"):
                r.snapshot_csv(f.csv, monitor_stopped=True)
            with f.csv.open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(ValueError):
                r.snapshot_csv(f.csv)

    def test_final_identity_requires_explicit_stopped_monitor_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            provisional = f.build()
            final = f.build(monitor_stopped=True)
            self.assertEqual(provisional["report_status"], "provisional_prefix_metadata_only")
            self.assertFalse(provisional["csv"]["identity_is_final"])
            self.assertEqual(final["report_status"], "final_monitor_stopped_by_root")
            self.assertTrue(final["csv"]["identity_is_final"])
            self.assertEqual(final["csv"]["sha256"], hashlib.sha256(f.csv.read_bytes()).hexdigest())

    def test_fail_closed_preflight_drift_and_no_prediction_or_input_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            original = Path.read_bytes
            def guarded(path):
                self.assertNotIn(path.suffix, (".npz", ".npy", ".jsonl"))
                return original(path)
            with patch.object(Path, "read_bytes", guarded):
                f.build()
            copy_path = f.new / "formal/preflight.json"
            copy_path.write_bytes(copy_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "preflight bytes"):
                f.build()


if __name__ == "__main__":
    unittest.main()
