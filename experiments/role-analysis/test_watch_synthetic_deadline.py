"""CPU-only fake-/proc and injected-clock tests; no real signal is sent."""
from dataclasses import replace
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import watch_synthetic_deadline as watch

EXE = "/opt/independent-fixture/python3.11"
UID = 1000


def driver(pid=200, worker=False):
    argv = ["/opt/venv/bin/python", "-u", watch.CODE + "/run_synthetic_agent_campaign.py",
            "--data-dir", watch.DATA, "--run-root", watch.RUNS]
    if worker:
        argv += ["--worker", "dense"]
    return watch.Process(pid, 12345 + pid, UID, EXE, tuple(argv))


def stage(pid=100, kind="extract"):
    return watch.Process(pid, 12345 + pid, UID, EXE, ("/opt/venv/bin/python", "-u",
        watch.CODE + "/run_synthetic_agent_roles.py", kind, "--data-dir", watch.DATA,
        "--run-dir", watch.RUNS + "/formal"))


class Reader:
    def __init__(self, processes):
        self.processes = {p.pid: p for p in processes}

    def scan(self):
        return list(self.processes.values())

    def read(self, pid):
        return self.processes.get(pid)


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class WatchdogChecks(unittest.TestCase):
    def setUp(self):
        self.scope = watch.Scope(EXE, UID)

    def test_exact_driver_forward_training_and_dense_match(self):
        self.assertEqual(self.scope.match(driver()).kind, "driver")
        for name in watch.GPU_STAGES:
            self.assertEqual(self.scope.match(stage(kind=name)).stage, name)
        dense = self.scope.match(driver(worker=True))
        self.assertEqual((dense.kind, dense.stage), ("stage", "dense"))
        equals = replace(driver(), argv=("python", watch.CODE + "/run_synthetic_agent_campaign.py",
            "--data-dir=" + watch.DATA, "--run-root=" + watch.RUNS))
        self.assertIsNotNone(self.scope.match(equals))

    def test_monitor_shell_strings_other_tasks_and_cpu_roles_are_not_targets(self):
        original = stage()
        processes = [
            replace(original, executable="/bin/bash", argv=("bash", "-c", " ".join(original.argv))),
            replace(original, argv=("python", "-c", "literal argument", *original.argv[2:])),
            replace(original, argv=("python", "-m", "monitor", *original.argv[2:])),
            replace(original, argv=("python", "/root/monitor.py", *original.argv[3:])),
            replace(original, argv=tuple(watch.DATA + "-other" if value == watch.DATA else value for value in original.argv)),
            replace(original, argv=tuple(watch.RUNS + "-other/formal" if value == watch.RUNS + "/formal" else value for value in original.argv)),
            replace(original, argv=tuple(value.replace("code-v7/", "code-v5/") for value in original.argv)),
            replace(original, uid=UID + 1), replace(original, executable="/opt/other/python3.11"),
            replace(original, argv=original.argv + ("--data-dir", watch.DATA)),
            stage(kind="plot"), stage(kind="prepare"), stage(kind="report"),
            replace(driver(), argv=driver().argv + ("--prepare-formal-only",)),
            replace(driver(), argv=driver().argv + ("--formal-run-dir", "/unrelated/formal")),
        ]
        for process in processes:
            with self.subTest(argv=process.argv):
                self.assertIsNone(self.scope.match(process))

    def test_deadline_no_early_signals_driver_first_and_fork_race_sweeps(self):
        clock = Clock(9.)
        unrelated = replace(stage(900), argv=("python", "/root/monitor.py"))
        reader = Reader([stage(100), driver(200), unrelated])
        events = []

        def kill(target, scope, source):
            events.append((clock(), target.process.pid, target.kind))
            source.processes.pop(target.process.pid, None)
            if target.process.pid == 200:
                source.processes[300] = driver(300)
                source.processes[400] = stage(400, "train")
            elif target.process.pid == 300:
                source.processes[500] = driver(500, worker=True)
            return {"pid": target.process.pid, "result": "fake_SIGKILL"}

        result = watch.enforce(self.scope, reader, 10., clock=clock, sleep=clock.sleep, killer=kill,
                               poll=5., sweep_seconds=1., sweep_interval=.25)
        self.assertEqual(result["status"], "deadline_enforced")
        self.assertTrue(all(at >= 10 for at, _, _ in events))
        self.assertEqual([pid for _, pid, _ in events[:2]], [200, 300])
        self.assertEqual({pid for _, pid, _ in events}, {100, 200, 300, 400, 500})
        self.assertEqual(set(reader.processes), {900})
        self.assertGreaterEqual(result["sweep_passes"], 2)

    def test_normal_completion_exits_without_killing(self):
        clock = Clock(5.)
        kill = Mock(side_effect=AssertionError("No signal before completion"))
        result = watch.enforce(self.scope, Reader([]), 10., clock=clock, sleep=clock.sleep,
                               completed=lambda: True, killer=kill)
        self.assertEqual(result, {"status": "campaign_completed", "kills": []})
        self.assertEqual(clock(), 5.)
        kill.assert_not_called()

    def test_pid_reuse_or_changed_scope_is_never_signalled(self):
        original = stage()
        for replacement in [replace(original, start_ticks=original.start_ticks + 1),
                            replace(original, argv=("python", "/root/monitor.py"))]:
            reader = Reader([replacement])
            with patch.object(watch.os, "kill") as kill, patch.object(watch.os, "pidfd_open", create=True) as open_fd:
                result = watch.kill_exact(self.scope.match(original), self.scope, reader)
            self.assertEqual(result["result"], "gone_or_identity_changed")
            kill.assert_not_called()
            open_fd.assert_not_called()

    def test_pidfd_targets_one_exact_pid_not_a_process_group(self):
        process = stage()
        reader = Reader([process])
        with patch.object(watch.os, "pidfd_open", return_value=987, create=True) as open_fd, patch.object(watch.signal, "pidfd_send_signal", create=True) as send, patch.object(watch.os, "close") as close, patch.object(watch.os, "kill") as kill:
            result = watch.kill_exact(self.scope.match(process), self.scope, reader)
        self.assertEqual(result["result"], "SIGKILL_sent")
        open_fd.assert_called_once_with(process.pid)
        send.assert_called_once_with(987, signal.SIGKILL)
        close.assert_called_once_with(987)
        kill.assert_not_called()

    def test_fake_proc_parser_handles_comm_parentheses_and_nul_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "12345"
            folder.mkdir()
            fields = ["0"] * 20
            fields[0], fields[19] = "S", "98765"
            (folder / "stat").write_text("12345 (python worker (nested)) " + " ".join(fields))
            (folder / "status").write_text(f"Uid:\t{UID}\t{UID}\t{UID}\t{UID}\n")
            (folder / "exe").symlink_to(EXE)
            (folder / "cmdline").write_bytes(b"\0".join(value.encode() for value in stage().argv) + b"\0")
            reader = watch.ProcFS(temporary)
            process = reader.read(12345)
            self.assertEqual((process.pid, process.start_ticks, process.uid), (12345, 98765, UID))
            self.assertEqual(self.scope.match(process).stage, "extract")
            self.assertEqual(len(reader.scan()), 1)

    def test_config_minimum_and_24_hour_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "pilot.json", root / "formal.json"]
            for path, deadline in zip(paths, [watch.HARD_CAP, "2026-10-27T13:00:00Z"]):
                path.write_text(json.dumps({"budget": {"started_utc": "2026-10-26T13:19:17Z",
                    "deadline_utc": deadline, "authorized_hours": 24}}))
            deadline, proofs = watch.authorized_deadline(paths)
            self.assertEqual(deadline, watch.utc("2026-10-27T13:00:00Z"))
            self.assertEqual(len(proofs), 2)
            paths[0].write_text(json.dumps({"budget": {"started_utc": "2026-10-26T13:19:17Z",
                "deadline_utc": "2026-10-28T13:19:17Z", "authorized_hours": 48}}))
            with self.assertRaisesRegex(ValueError, "authorized 24-hour"):
                watch.authorized_deadline(paths)


if __name__ == "__main__":
    unittest.main()
