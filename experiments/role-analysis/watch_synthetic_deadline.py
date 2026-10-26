"""Independent, PID-scoped hard deadline for this one code-v7 campaign.

No GPU library is imported. No process is signalled before the authorized UTC
deadline. Matching requires the actual Python executable, UID, exact executed
script, and parsed data/run arguments; shell strings are never searched.
Start independently with setsid. State/log files must be on local /root or /tmp,
not the network-mounted data/run tree. Campaign-status reads run in a daemon
thread so a stalled FUSE read cannot delay deadline enforcement.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import posixpath
import signal
import threading
import time

CODE = "/root/role-retraining-20261026/code-v7"
DATA = "/workspace/role-retraining-20261026/data-v7"
RUNS = "/workspace/role-retraining-20261026/runs-v7"
PYTHON = "/root/role-retraining-venv/bin/python"
HARD_CAP = "2026-10-27T13:19:17Z"
GPU_STAGES = {"preflight", "scan", "extract", "train", "evaluate"}


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("The authorized deadline needs an explicit timezone")
    return parsed.timestamp()


def authorized_deadline(paths):
    deadlines, proofs = [], {}
    for path in paths:
        raw = Path(path).read_bytes()
        budget = json.loads(raw)["budget"]
        start, deadline = utc(budget["started_utc"]), utc(budget["deadline_utc"])
        hours = float(budget["authorized_hours"])
        if not (0 < hours <= 24 and start < deadline <= start + hours * 3600 and deadline <= utc(HARD_CAP)):
            raise ValueError("Configured execution exceeds the authorized 24-hour window or this run's hard UTC cap")
        deadlines.append(deadline)
        proofs[str(path)] = hashlib.sha256(raw).hexdigest()
    if not deadlines:
        raise ValueError("Both pilot/formal deadline configurations are required")
    return min(deadlines), proofs


@dataclass(frozen=True)
class Process:
    pid: int
    start_ticks: int
    uid: int
    executable: str
    argv: tuple[str, ...]
    state: str = "S"


@dataclass(frozen=True)
class Target:
    process: Process
    kind: str
    stage: str
    script: str


def option(args, name):
    values = []
    for index, arg in enumerate(args):
        if arg == name:
            if index + 1 == len(args) or args[index + 1].startswith("--"):
                return None
            values.append(args[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg[len(name) + 1:])
    return values[0] if len(values) == 1 else None


def executed_script(argv):
    """Recognize Python's script slot; reject -c, -m, and unknown options."""
    index = 1
    simple = {"-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "-q", "-b", "-bb"}
    while index < len(argv):
        arg = argv[index]
        if arg in {"-c", "-m"}:
            return None, ()
        if arg == "--":
            index += 1
            break
        if arg in simple or (arg.startswith(("-W", "-X")) and len(arg) > 2):
            index += 1
        elif arg in {"-W", "-X"}:
            index += 2
        elif arg.startswith("-"):
            return None, ()
        else:
            break
    return (argv[index], argv[index + 1:]) if index < len(argv) else (None, ())


@dataclass(frozen=True)
class Scope:
    executable: str
    uid: int
    code: str = CODE
    data: str = DATA
    runs: str = RUNS

    def match(self, process):
        if process.pid <= 1 or process.pid == os.getpid() or process.state == "Z" or process.uid != self.uid:
            return None
        if process.executable != self.executable:
            return None
        script, args = executed_script(process.argv)
        role_script = self.code + "/run_synthetic_agent_roles.py"
        campaign_script = self.code + "/run_synthetic_agent_campaign.py"
        if script not in {role_script, campaign_script} or option(args, "--data-dir") != self.data:
            return None
        if script == role_script:
            if not args or args[0] not in GPU_STAGES or option(args, "--run-dir") not in {self.runs + "/pilot", self.runs + "/formal"}:
                return None
            return Target(process, "stage", args[0], script)
        if option(args, "--run-root") != self.runs or "--prepare-formal-only" in args:
            return None
        for flag, expected in [("--pilot-run-dir", self.runs + "/pilot"), ("--formal-run-dir", self.runs + "/formal")]:
            if any(value == flag or value.startswith(flag + "=") for value in args) and option(args, flag) != expected:
                return None
        if "--worker" in args or any(arg.startswith("--worker=") for arg in args):
            return Target(process, "stage", "dense", script) if option(args, "--worker") == "dense" else None
        return Target(process, "driver", "campaign", script)


class ProcFS:
    def __init__(self, root="/proc"):
        self.root = Path(root)

    def read(self, pid):
        folder = self.root / str(pid)
        try:
            stat = (folder / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            uid = next(line.split()[1] for line in (folder / "status").read_text().splitlines() if line.startswith("Uid:"))
            argv = tuple(value.decode("utf-8", "surrogateescape") for value in (folder / "cmdline").read_bytes().split(b"\0") if value)
            executable = os.readlink(folder / "exe").removesuffix(" (deleted)")
            return Process(int(pid), int(fields[19]), int(uid), posixpath.normpath(executable), argv, fields[0])
        except (OSError, ValueError, IndexError, StopIteration):
            return None

    def scan(self):
        try:
            pids = [int(path.name) for path in self.root.iterdir() if path.name.isdigit()]
        except OSError:
            return []
        return [process for pid in pids if (process := self.read(pid)) is not None]


def scoped_targets(scope, reader):
    return sorted((target for process in reader.scan() if (target := scope.match(process)) is not None), key=lambda t: t.process.pid)


def kill_exact(target, scope, reader):
    """Revalidate identity and prefer Linux pidfds to prevent PID-reuse races."""
    original = target.process
    current = reader.read(original.pid)
    matched = scope.match(current) if current is not None else None
    if matched is None or current.start_ticks != original.start_ticks:
        return {"pid": original.pid, "result": "gone_or_identity_changed"}
    fd = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                fd = os.pidfd_open(original.pid)
            except OSError as error:
                if error.errno not in {errno.ENOSYS, errno.EINVAL}:
                    raise
        latest = reader.read(original.pid)
        if latest is None or latest.start_ticks != original.start_ticks or scope.match(latest) is None:
            return {"pid": original.pid, "result": "gone_or_identity_changed"}
        if fd is None:
            os.kill(original.pid, signal.SIGKILL)
        else:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        return {"pid": original.pid, "start_ticks": original.start_ticks, "kind": matched.kind,
                "stage": matched.stage, "script": matched.script, "result": "SIGKILL_sent"}
    except ProcessLookupError:
        return {"pid": original.pid, "result": "already_exited"}
    except OSError as error:
        return {"pid": original.pid, "result": "signal_failed", "error": str(error)}
    finally:
        if fd is not None:
            os.close(fd)


def campaign_complete(path, scope):
    try:
        value = json.loads(Path(path).read_text())
        return value.get("status") == "complete" and value.get("data_dir") == scope.data and value.get("run_root") == scope.runs
    except (OSError, ValueError):
        return False


def completion_reader(path, scope, poll):
    """Only this daemon may touch the FUSE status file during monitoring."""
    completed = threading.Event()

    def read_status():
        while not completed.is_set():
            if campaign_complete(path, scope):
                completed.set()
                return
            time.sleep(poll)

    threading.Thread(target=read_status, daemon=True).start()
    return completed.is_set


def enforce(scope, reader, deadline, *, clock=time.time, sleep=time.sleep,
            completed=lambda: False, killer=kill_exact, poll=5., sweep_seconds=5., sweep_interval=.25):
    while True:
        now = clock()
        if now >= deadline:
            break
        if completed() and not scoped_targets(scope, reader):
            return {"status": "campaign_completed", "kills": []}
        sleep(min(poll, max(0, deadline - clock())))
    actions, passes = [], 0
    end = clock() + sweep_seconds
    while True:
        # Kill every currently visible launch-capable driver before GPU stages.
        for target in scoped_targets(scope, reader):
            if target.kind == "driver":
                actions.append({"at_unix": clock(), **killer(target, scope, reader)})
        # A driver could have forked a stage immediately before it was killed.
        fresh = scoped_targets(scope, reader)
        for target in [t for t in fresh if t.kind == "driver"] + [t for t in fresh if t.kind == "stage"]:
            actions.append({"at_unix": clock(), **killer(target, scope, reader)})
        passes += 1
        if clock() >= end and passes >= 2:
            break
        sleep(sweep_interval)
    remaining = scoped_targets(scope, reader)
    return {"status": "deadline_enforced" if not remaining else "deadline_enforcement_pending",
            "kills": actions, "sweep_passes": passes,
            "remaining_pids": [target.process.pid for target in remaining]}


def atomic_state(path, state):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config", type=Path, default=Path(CODE) / "synthetic-agent-120b.json")
    parser.add_argument("--formal-config", type=Path, default=Path(CODE) / "synthetic-agent-120b-formal.json")
    parser.add_argument("--python", type=Path, default=Path(PYTHON))
    parser.add_argument("--state-file", type=Path, default=Path(f"/tmp/role-deadline-watchdog-{os.getpid()}.json"))
    parser.add_argument("--poll-seconds", type=float, default=5.)
    args = parser.parse_args()
    if not (0 < args.poll_seconds <= 60):
        raise ValueError("Watchdog poll interval must be in (0,60] seconds")
    state_path = args.state_file.resolve()
    if not any(state_path.is_relative_to(Path(root)) for root in ["/root", "/tmp"]):
        raise ValueError("Watchdog state must be on local /root or /tmp, outside the FUSE run/data tree")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    deadline, config_proofs = authorized_deadline([args.pilot_config, args.formal_config])
    scope = Scope(posixpath.normpath(os.path.realpath(args.python)), os.getuid())
    state = {"pid": os.getpid(), "status": "watching", "started_unix": time.time(),
             "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(), "deadline_unix": deadline,
             "config_sha256": config_proofs, "state_file": str(state_path),
             "scope": {"executable": scope.executable, "uid": scope.uid, "data_dir": DATA, "run_root": RUNS,
                       "script_whitelist": [CODE + "/run_synthetic_agent_campaign.py", CODE + "/run_synthetic_agent_roles.py"],
                       "gpu_stages": sorted(GPU_STAGES), "dense_worker": True,
                       "excluded": "other executables/scripts/tasks, CPU-only roles, prepare-formal-only, monitors, shell command strings"}}
    atomic_state(state_path, state)
    print(json.dumps({"watchdog_pid": os.getpid(), "deadline_utc": state["deadline_utc"], "state_file": str(state_path)}), flush=True)
    completed = completion_reader(Path(RUNS) / "campaign" / "status.json", scope, args.poll_seconds)
    result = enforce(scope, ProcFS(), deadline, completed=completed, poll=args.poll_seconds)
    # No filesystem/status/log writes occur on the deadline-to-kill path.
    state.update(result, finished_unix=time.time())
    atomic_state(state_path, state)
    return 0 if result["status"] != "deadline_enforcement_pending" else 2


if __name__ == "__main__":
    raise SystemExit(main())
