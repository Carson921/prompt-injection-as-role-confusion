"""CPU smoke: synthetic files, fake tmux and clock; never launch a process."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

spec = importlib.util.spec_from_file_location("gardening_waiter", Path(__file__).with_name("wait_synthetic_gardening_once.py"))
w = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = w
spec.loader.exec_module(w)


class Fixture:
    def __init__(self, root, ready=True):
        root = Path(root)
        self.scope = w.Scope(root / "runs", root / "out", root / "launcher.sh", root / "state", root)
        self.scope.launcher.write_text("#!/bin/bash\nexit 0\n")
        self.calls = []
        self.now = w.HARD_DEADLINE - 3600
        for phase in ("pilot", "formal"):
            p = self.scope.runs / phase / "config.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"phase": phase, "budget": {"started_utc": "2026-10-26T13:19:17Z",
                 "deadline_utc": "2026-10-27T13:19:17Z", "authorized_hours": 24}}))
        if ready:
            c = self.scope.runs / "formal/candidates/linear-seed17"
            c.mkdir(parents=True)
            (c / "complete.json").write_text("{}")
            (c / "best.npz").write_bytes(b"synthetic-not-valid-real-npz-check-is-authoritative")

    def run(self, argv, **kwargs):
        self.calls.append(argv)
        return types.SimpleNamespace(returncode=1 if argv[1] == "has-session" else 0, stderr="")

    def waiter(self):
        return w.Waiter(self.scope, clock=lambda: self.now, run=self.run)


class Smoke(unittest.TestCase):
    def test_waits_without_npz_and_ignores_lastpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, ready=False)
            c = f.scope.runs / "formal/candidates/linear-seed17"
            c.mkdir(parents=True)
            (c / "last.pt").write_bytes(b"not-an-export")
            self.assertEqual(f.waiter().step(), "waiting_first_formal_candidate")
            self.assertFalse(any(a[1] == "new-session" for a in f.calls))

    def test_ready_dispatches_fixed_session_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            self.assertEqual(f.waiter().step(), "dispatched")
            self.assertEqual(f.waiter().step(), "already_claimed")
            calls = [a for a in f.calls if a[1] == "new-session"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][4], w.SESSION)
            self.assertIn(str(f.scope.launcher), calls[0][-1])
            self.assertTrue((f.scope.state / "dispatch-result.json").is_file())

    def test_existing_inference_report_or_session_never_dispatches(self):
        for name in ("inference-started.json", "report-complete.json", "session"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp)
                if name == "session":
                    f.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stderr="")
                else:
                    f.scope.output.mkdir()
                    (f.scope.output / name).write_text("{}")
                self.assertTrue(f.waiter().step().startswith("already_"))
                self.assertFalse((f.scope.state / "dispatch-claim.json").exists())

    def test_expired_window_and_invalid_authorized_hours_do_not_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            f.now = w.HARD_DEADLINE
            self.assertEqual(f.waiter().step(), "deadline_exceeded")
            self.assertEqual(f.calls, [])
            p = f.scope.runs / "formal/config.json"
            c = json.loads(p.read_text())
            c["budget"]["authorized_hours"] = 25
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                f.waiter()

    def test_deadline_rechecked_after_durable_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            waiter = f.waiter()
            old_open = w.os.open
            def advance(*args, **kwargs):
                fd = old_open(*args, **kwargs)
                f.now = w.HARD_DEADLINE
                return fd
            from unittest.mock import patch
            with patch.object(w.os, "open", side_effect=advance):
                self.assertEqual(waiter.step(), "deadline_exceeded_after_claim")
            self.assertFalse(any(a[1] == "new-session" for a in f.calls))

    def test_code_or_config_change_refuses_without_dispatch(self):
        for kind in ("launcher", "config"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp)
                waiter = f.waiter()
                p = f.scope.launcher if kind == "launcher" else f.scope.runs / "formal/config.json"
                p.write_bytes(p.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    waiter.step()
                self.assertFalse(any(a[1] == "new-session" for a in f.calls))

    def test_failed_dispatch_retains_claim_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            def fail(argv, **kwargs):
                f.calls.append(argv)
                return types.SimpleNamespace(returncode=1, stderr="synthetic tmux failure")
            f.run = fail
            with self.assertRaisesRegex(RuntimeError, "claim retained"):
                f.waiter().step()
            self.assertEqual(f.waiter().step(), "already_claimed")
            self.assertEqual(sum(a[1] == "new-session" for a in f.calls), 1)

    def test_concurrent_waiters_have_one_atomic_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            waiters = [f.waiter(), f.waiter()]
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                states = list(pool.map(lambda a: a.step(), waiters))
            self.assertIn("dispatched", states)
            self.assertEqual(sum(a[1] == "new-session" for a in f.calls), 1)


if __name__ == "__main__":
    unittest.main()
