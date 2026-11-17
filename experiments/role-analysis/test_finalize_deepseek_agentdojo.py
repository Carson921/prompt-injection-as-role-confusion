"""CPU-only guards; no production HTTP requests, bridge launches, or CUDA."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import finalize_deepseek_agentdojo as controller


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root, self.source = self.base / "run", self.base / "source"
        self.root.mkdir(); self.source.mkdir()
        for group, names in (("core", controller.CORE), ("validation", controller.VALIDATION)):
            records = {}
            for name in names:
                original = self.source / name
                original.write_text("# frozen fixture " + name)
                snapshot = self.root / "implementation-source" / group / name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(original.read_bytes())
                records[name] = {"original_path": str(original), "sha256": controller.sha(original)}
            controller.write(self.root / "implementation-source" / (group + "-manifest.json"), {"files": records})
        self.cases = []
        for condition in ("plain", "plain_delimiter"):
            for suite, count in (("banking", 144), ("slack", 105), ("travel", 140)):
                for index in range(count):
                    self.cases.append({"case_id": f"{condition}/{suite}/case_{index}", "condition": condition, "suite": suite})
        controller.write(self.root / "schedule.json", self.cases)
        controller.write(self.root / "protocol.json", {"schedule_sha256": controller.object_sha(self.cases)})
        server = {"fixture": "native server"}
        controller.write(self.root / "server_identity.json", server)
        self.identity = {"mode": "full", "initial_episodes": 778, "conditions": ["plain", "plain_delimiter"],
            "output_root": str(self.root), "server_identity_path": str(self.root / "server_identity.json"),
            "schedule_sha256": controller.object_sha(self.cases), "server_identity": server,
            "server_identity_sha256": controller.sha(self.root / "server_identity.json"),
            "protocol_manifest_sha256": controller.sha(self.root / "protocol.json"),
            "server_url": "http://127.0.0.1:8894/generate"}
        controller.write(self.root / "run_identity.json", self.identity)
        controller.write(self.root / "server_launch.json", {"pid": os.getpid(), "port": 8894,
            "started_unix": time.time(), "source_sha256": controller.sha(self.source / "run_deepseek_agentdojo_server.py")})

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self):
        args = argparse.Namespace(output=self.root, source_dir=self.source, port=8894,
            native_python=Path(sys.executable), report_python=Path(sys.executable), native_repo=self.base,
            resume_bridge=True, poll_seconds=.001, ready_timeout=.01, completion_timeout=.001,
            shutdown_timeout=.01, command_timeout=1)
        return controller.prepare_plan(args)

    def commits(self, *, count=778, errors=0):
        for index, case in enumerate(self.cases[:count]):
            failed = index < errors
            controller.write(self.root / "cases" / (case["case_id"] + ".json"), {**case,
                "status": "error" if failed else "complete", "security": None if failed else False,
                "utility": None if failed else True, "error": "Original native parser error" if failed else None,
                "spec_sha256": controller.object_sha(self.identity)})
        conditions = {}
        for condition in self.identity["conditions"]:
            selected = [i for i,c in enumerate(self.cases) if c["condition"] == condition]
            condition_errors = sum(i < errors for i in selected)
            conditions[condition] = {"expected_cases": 389, "terminal_cases": 389, "missing_cases": 0,
                "scored_cases": 389 - condition_errors, "error_cases": condition_errors,
                "asr": None if condition_errors else 0, "utility": None if condition_errors else 1}
        controller.write(self.root / "summary.json", {"execution_complete": True, "scheduled_cases": 778,
            "spec_sha256": controller.object_sha(self.identity), "status": "complete_with_initial_errors" if errors else "complete",
            "data_fully_scored": errors == 0, "conditions": conditions})

    def health(self, **updates):
        return {"stage": "ready", "identity_sha256": self.identity["server_identity_sha256"],
                "queued_or_running": 0, "failed_calls": 3, "completed_calls": 10, "fatal_error": None, **updates}

    def test_13_sources_and_snapshots_are_required_and_frozen(self):
        plan = self.plan()
        controller.verify_plan(plan, controller.object_sha(plan))
        self.assertEqual(len(controller.CORE | controller.VALIDATION), 13)
        (self.source / "deepseek_agentdojo_report.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            controller.verify_plan(plan, controller.object_sha(plan))

    def test_plan_hash_and_pid_reuse_guard(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "plan object SHA"):
            controller.verify_plan(plan, "0" * 64)
        with patch.object(controller, "process_identity", return_value={"pid": os.getpid(), "start_ticks": 1}):
            with self.assertRaisesRegex(ValueError, "PID was reused"):
                controller.verify_server_instance(plan)

    def test_partial_is_pending_and_extra_case_is_fatal(self):
        self.commits(count=777)
        self.assertFalse(controller.terminal_snapshot(self.root)["ready"])
        controller.write(self.root / "cases" / "extra.json", {})
        with self.assertRaisesRegex(ValueError, "Unexpected terminal"):
            controller.terminal_snapshot(self.root)

    def test_native_errors_remain_null_and_full_denominator_unavailable(self):
        self.commits(errors=1)
        snapshot = controller.terminal_snapshot(self.root)
        self.assertTrue(snapshot["ready"])
        self.assertEqual(snapshot["native_error_cases"], 1)
        path = self.root / "cases" / (self.cases[0]["case_id"] + ".json")
        record = controller.read(path); record["security"] = False
        controller.write(path, record)
        with self.assertRaisesRegex(ValueError, "original null"):
            controller.terminal_snapshot(self.root)

    def test_health_native_errors_allowed_but_queue_fatal_identity_blocked(self):
        expected = self.identity["server_identity_sha256"]
        controller.check_health(self.health(), expected, ready=True, idle=True)
        for bad in ({"fatal_error": {"type": "CUDAError"}}, {"queued_or_running": 1}, {"identity_sha256": "changed"}):
            with self.assertRaises(ValueError):
                controller.check_health(self.health(**bad), expected, ready=True, idle=True)

    def test_resume_cannot_launch_before_ready_or_over_existing_lock(self):
        instance = controller.Controller(self.plan(), "fixture")
        instance.folder.mkdir()
        with patch.object(instance, "guard"), patch.object(controller, "http_json", return_value=self.health(stage="loading")), \
                patch.object(controller.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "Ready server"):
                instance.resume_bridge()
            launch.assert_not_called()
        lock = controller.exclusive(self.root / ".bridge.lock")
        try:
            with patch.object(instance, "guard"), patch.object(controller, "http_json", return_value=self.health()), \
                    patch.object(controller.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "already active"):
                    instance.resume_bridge()
                launch.assert_not_called()
        finally:
            lock.close()

    def test_partial_never_reaches_shutdown_or_report(self):
        plan = self.plan(); plan["resume_bridge"] = False
        instance = controller.Controller(plan, "fixture")
        instance.folder.mkdir()
        with patch.object(instance, "guard"), patch.object(instance, "wait_ready"), \
                patch.object(controller, "http_json", return_value=self.health()), \
                patch.object(instance, "command") as command, patch.object(instance, "shutdown") as shutdown:
            self.assertEqual(instance.run(), 1)
            command.assert_not_called(); shutdown.assert_not_called()
        state = controller.read(instance.folder / "status.json")
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["task_complete"])

    def test_transient_health_timeout_keeps_same_live_execution(self):
        plan = self.plan(); plan.update(resume_bridge=False, completion_timeout=1)
        instance = controller.Controller(plan, "fixture")
        locked = unittest.mock.Mock()
        snapshot = {"ready": True, "terminal_cases": 778, "native_error_cases": 0}
        with patch.object(instance, "guard"), patch.object(instance, "stage") as stage, \
                patch.object(controller, "http_json", side_effect=[TimeoutError("temporary observation"), self.health()]), \
                patch.object(controller, "terminal_snapshot", return_value=snapshot), \
                patch.object(controller, "exclusive", return_value=locked), patch.object(controller.time, "sleep"), \
                patch.object(controller.subprocess, "Popen") as launch:
            self.assertEqual(instance.wait_terminal(), (locked, snapshot))
            launch.assert_not_called()
            self.assertIn("TimeoutError", stage.call_args.kwargs["health_observation_error"])

    def test_fresh_nonzero_bridge_exit_race_is_rejected(self):
        plan = self.plan(); plan["completion_timeout"] = 1
        instance = controller.Controller(plan, "fixture"); instance.folder.mkdir()
        path = instance.folder / "bridge-process.json"; path.write_text("{}")
        running = {"plan_sha256": "fixture", "exit_code": None, "pid": os.getpid(),
                   "process_identity": controller.process_identity(os.getpid())}
        failed = {**running, "exit_code": 1}
        with patch.object(instance, "guard"), patch.object(controller, "http_json", return_value=self.health()), \
                patch.object(controller, "terminal_snapshot", return_value={"ready": True}), \
                patch.object(controller, "read", side_effect=[running, failed]), \
                patch.object(controller, "exclusive", return_value=unittest.mock.Mock()):
            with self.assertRaisesRegex(ValueError, "final exit record is unsuccessful"):
                instance.wait_terminal()

    def test_stale_exit_files_cannot_certify_new_shutdown(self):
        instance = controller.Controller(self.plan(), "fixture"); instance.folder.mkdir()
        backend = {"native_backbone_audit": {"all_parameters_on_cuda": True}}
        controller.write(self.root / "generation-backend.json", backend)
        expected = self.identity["server_identity_sha256"]
        controller.write(self.root / "server_exit.json", {"identity_sha256": expected, "exit_code": 0, "finished_unix": 1})
        controller.write(self.root / "server_runtime_final.json", {"identity_sha256": expected, "finished_unix": 1})
        controller.write(self.root / "generation-final.json", {"finished_at_utc": "2000-01-01T00:00:00+00:00"})
        with patch.object(instance, "guard"), patch.object(controller, "process_identity", return_value=None), \
                patch.object(controller, "http_json", side_effect=[self.health(), {"status": "closing"}]):
            with self.assertRaisesRegex(ValueError, "Stale shutdown"):
                instance.shutdown()

    def test_success_stops_at_pending_visual_review_without_task_completion(self):
        plan = self.plan(); plan["resume_bridge"] = False
        instance = controller.Controller(plan, "fixture"); instance.folder.mkdir()
        snapshot = {"ready": True, "terminal_cases": 778, "native_error_cases": 0, "cases_sha256": {}}
        events = []

        def command(label, argv):
            events.append(label)
            if label == "native_audit":
                controller.write(self.root / "native-evaluation-audit.json", {"status": "passed", "complete": True,
                    "scheduled_cases": 778, "audited_cases": 778, "model_generation_performed": False, "cases_sha256": {}})
            else:
                plots = {}
                for name in ("asr-comparison.png", "first-payload-role-probabilities.png"):
                    (self.root / name).write_bytes(b"fixture plot")
                    plots[name] = controller.sha(self.root / name)
                controller.write(self.root / "independent-validation.json", {"status": "passed", "complete": True,
                    "plots": {"status": "created", "files_sha256": plots}})

        locked = unittest.mock.Mock()
        with patch.object(instance, "guard"), patch.object(instance, "wait_ready"), \
                patch.object(instance, "wait_terminal", return_value=(locked, snapshot)), \
                patch.object(controller, "terminal_snapshot", return_value=snapshot), \
                patch.object(instance, "command", side_effect=command), \
                patch.object(instance, "shutdown", side_effect=lambda: events.append("shutdown")):
            self.assertEqual(instance.run(), 0)
        self.assertEqual(events, ["native_audit", "shutdown", "report"])
        state = controller.read(instance.folder / "status.json")
        self.assertEqual(state["status"], "pending_visual_review")
        self.assertFalse(state["task_complete"]); self.assertFalse(state["complete"])
        self.assertTrue(state["visual_review_required"])
        locked.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
