#!/usr/bin/env python3
"""One-shot CPU waiter for this campaign's already bounded gardening launcher.

No model, torch, checkpoint deserialization, process killing, or core edits.
Existing launcher --stage check remains authoritative for checkpoint validity.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

TASK = Path("/root/role-retraining-20261026")
RUNS = Path("/workspace/role-retraining-20261026/runs-v7")
SESSION = "role-gardening-first-20261026"
HARD_DEADLINE = datetime(2026, 10, 27, 13, 19, 17, tzinfo=timezone.utc).timestamp()


@dataclass(frozen=True)
class Scope:
    runs: Path = RUNS
    output: Path = Path("/workspace/role-retraining-20261026/diagnostics/gardening-pilot-vs-first-formal")
    launcher: Path = TASK / "launch-gardening-first.sh"
    state: Path = TASK / "gardening-first-trigger"
    working_dir: Path = TASK


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deadline(scope):
    ends, bindings = [HARD_DEADLINE], {}
    for phase in ("pilot", "formal"):
        path = scope.runs / phase / "config.json"
        raw = path.read_bytes()
        config = json.loads(raw)
        if config["phase"] != phase:
            raise ValueError("Run phase differs")
        budget = config["budget"]
        start = datetime.fromisoformat(budget["started_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(budget["deadline_utc"].replace("Z", "+00:00"))
        hours = float(budget["authorized_hours"])
        if not (start.tzinfo and end.tzinfo and 0 < hours <= 24 and 0 < (end - start).total_seconds() <= hours * 3600):
            raise ValueError("Invalid authorized experiment budget")
        ends.append(end.timestamp())
        bindings[phase] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return min(ends), bindings


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class Waiter:
    def __init__(self, scope=Scope(), *, clock=time.time, run=subprocess.run):
        self.scope, self.clock, self.run = scope, clock, run
        self.end, self.configs = deadline(scope)
        self.launcher_sha = digest(scope.launcher)

    def existing(self):
        for name in ("report-complete.json", "inference-started.json"):
            if (self.scope.output / name).exists():
                return "already_" + name.removesuffix(".json").replace("-", "_")
        if (self.scope.state / "dispatch-claim.json").exists():
            return "already_claimed"
        if self.clock() >= self.end:
            return "deadline_exceeded"
        result = self.run(["tmux", "has-session", "-t", "=" + SESSION], capture_output=True,
                          text=True, timeout=min(5, max(.01, self.end - self.clock())))
        if result.returncode == 0:
            return "already_tmux_session"
        if result.returncode != 1:
            raise RuntimeError("tmux has-session failed: " + result.stderr[:300])
        return None

    def step(self):
        # A missing/incomplete NPZ never causes a CUDA import or subprocess launch.
        state = self.existing()
        if state:
            return state
        candidate = self.scope.runs / "formal/candidates/linear-seed17"
        files = [candidate / name for name in ("complete.json", "best.npz")]
        if not all(p.is_file() and p.stat().st_size > 0 for p in files):
            return "waiting_first_formal_candidate"
        # Recheck all immutable authorization immediately before claiming dispatch.
        end, configs = deadline(self.scope)
        if configs != self.configs or digest(self.scope.launcher) != self.launcher_sha:
            raise ValueError("Run configuration or approved launcher changed while waiting")
        self.end = min(self.end, end)
        state = self.existing()
        if state:
            return state
        self.scope.state.mkdir(parents=True, exist_ok=True)
        claim = {"waiter_pid": os.getpid(), "session": SESSION, "deadline_unix": self.end,
                 "claimed_unix": self.clock(), "launcher": str(self.scope.launcher), "launcher_sha256": self.launcher_sha,
                 "configs": configs, "candidate_files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
                 "validity_gate": "Existing launcher CPU --stage check; never last.pt or global formal freeze"}
        try:
            fd = os.open(self.scope.state / "dispatch-claim.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return "already_claimed"
        with os.fdopen(fd, "w") as stream:
            json.dump(claim, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.scope.state, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        # Keep the claim after every failure: automatic retries could start a
        # second teacher after a transient tmux/SSH failure hid a successful launch.
        if self.clock() >= self.end:
            return "deadline_exceeded_after_claim"
        for name in ("report-complete.json", "inference-started.json"):
            if (self.scope.output / name).exists():
                return "already_" + name.removesuffix(".json").replace("-", "_")
        command = "exec " + shlex.join(["/bin/bash", str(self.scope.launcher)])
        if self.clock() >= self.end:
            return "deadline_exceeded_after_claim"
        result = self.run(["tmux", "new-session", "-d", "-s", SESSION, "-c", str(self.scope.working_dir), command],
                          capture_output=True, text=True, timeout=min(5, max(.01, self.end - self.clock())))
        if result.returncode:
            raise RuntimeError("tmux dispatch failed; claim retained, no automatic retry: " + result.stderr[:300])
        atomic_json(self.scope.state / "dispatch-result.json", {**claim, "status": "dispatched", "dispatched_unix": self.clock()})
        return "dispatched"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not 10 <= args.poll_seconds <= 60:
        parser.error("poll interval must be 10..60 seconds")
    waiter = Waiter()
    previous = None
    while True:
        try:
            status = waiter.step()
            if status != previous:
                value = {"status": status, "waiter_pid": os.getpid(), "deadline_unix": waiter.end,
                         "session": SESSION, "scope": {"runs": str(waiter.scope.runs), "output": str(waiter.scope.output),
                                                       "launcher": str(waiter.scope.launcher)}}
                atomic_json(waiter.scope.state / "status.json", value)
                print(json.dumps(value, sort_keys=True), flush=True)
                previous = status
            if status != "waiting_first_formal_candidate":
                return 20 if status.startswith("deadline_exceeded") else 0
            time.sleep(min(args.poll_seconds, max(.01, waiter.end - time.time())))
        except Exception as error:
            # No deletion of the claim, no restart of a session, and no signals.
            value = {"status": "failed_no_retry", "error": str(error), "waiter_pid": os.getpid(), "session": SESSION}
            atomic_json(waiter.scope.state / "status.json", value)
            print(json.dumps(value, sort_keys=True), flush=True)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
