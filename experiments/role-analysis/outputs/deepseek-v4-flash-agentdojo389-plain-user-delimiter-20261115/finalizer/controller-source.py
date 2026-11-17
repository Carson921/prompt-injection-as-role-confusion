"""Guarded, one-shot native AgentDojo resume/finalization controller.

Prepare and review a plan before --run. No model loads, score edits, episode
retries, goal updates, or visual-review claims occur here. A successful run ends
at pending_visual_review. The bridge worker is detached and records its exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request


CORE = {
    "run_deepseek_agentdojo_server.py", "deepseek_agentdojo_model.py",
    "deepseek_agentdojo_protocol.py", "deepseek_agentdojo_bridge.py",
    "deepseek_agentdojo_preflight.py", "deepseek_agentdojo_generate_diagnostics.py",
    "deepseek_agentdojo_cache_audit.py", "deepseek_role_probe_model.py",
    "deepseek_role_probe_train.py", "deepseek_role_probe_data.py",
    "prepare_deepseek_performance_cache.py",
}
VALIDATION = {"deepseek_agentdojo_native_audit.py", "deepseek_agentdojo_report.py"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    """Linux PID + start ticks excludes reused PIDs and zombie processes."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def exclusive(path):
    stream = Path(path).open("a")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def verify_sources(root, source_dir):
    """Verify both original sources and frozen copies, exactly 11 + 2 files."""
    root, source_dir = Path(root), Path(source_dir).resolve()
    bindings = {}
    for group, expected in (("core", CORE), ("validation", VALIDATION)):
        manifest_path = root / "implementation-source" / (group + "-manifest.json")
        manifest = read(manifest_path)
        require(set(manifest["files"]) == expected, "Frozen " + group + " manifest file set differs")
        bindings[str(manifest_path.resolve())] = sha(manifest_path)
        for name, record in manifest["files"].items():
            original = Path(record["original_path"]).resolve()
            snapshot = manifest_path.parent / group / name
            require(original == source_dir / name and sha(original) == sha(snapshot) == record["sha256"],
                    "Frozen source changed: " + name)
            bindings[str(original)] = record["sha256"]
            bindings[str(snapshot.resolve())] = record["sha256"]
    return bindings


def validate_schedule(root):
    root = Path(root).resolve()
    identity, cases, protocol = (read(root / name) for name in ("run_identity.json", "schedule.json", "protocol.json"))
    require(identity["mode"] == "full" and identity["initial_episodes"] == len(cases) == 778,
            "Only the full exact 778-episode run may be finalized")
    require(Path(identity["output_root"]).resolve() == root and
            Path(identity["server_identity_path"]).resolve() == root / "server_identity.json",
            "Run or server identity crosses output scope")
    require(identity["schedule_sha256"] == protocol["schedule_sha256"] == object_sha(cases), "Schedule binding changed")
    require(len({case["case_id"] for case in cases}) == 778, "Duplicated scheduled cases")
    require(set(identity["conditions"]) == {"plain", "plain_delimiter"}, "Changed condition set")
    for condition in identity["conditions"]:
        selected = [case for case in cases if case["condition"] == condition]
        require(len(selected) == 389 and {suite: sum(c["suite"] == suite for c in selected)
                for suite in ("banking", "slack", "travel")} == {"banking": 144, "slack": 105, "travel": 140},
                "Changed native paired schedule")
    require(sha(root / "server_identity.json") == identity["server_identity_sha256"] and
            read(root / "server_identity.json") == identity["server_identity"], "Frozen server byte identity changed")
    require(sha(root / "protocol.json") == identity["protocol_manifest_sha256"], "Protocol manifest changed")
    return identity, cases


def terminal_snapshot(root):
    """Partial is pending; malformed/extra/error-score substitutions are fatal."""
    root = Path(root).resolve()
    identity, cases = validate_schedule(root)
    expected = {case["case_id"]: case for case in cases}
    paths = {str(path.relative_to(root / "cases"))[:-5]: path for path in (root / "cases").rglob("*.json")}
    require(set(paths) <= set(expected), "Unexpected terminal case IDs")
    hashes, errors = {}, 0
    for key, path in paths.items():
        require(path.resolve().is_relative_to(root / "cases"), "Terminal case symlink crosses scope")
        record = read(path)
        require(all(record.get(k) == v for k, v in expected[key].items()) and
                record["spec_sha256"] == object_sha(identity), "Terminal case identity changed: " + key)
        require(record["status"] in {"complete", "error"}, "Nonterminal case commit: " + key)
        if record["status"] == "complete":
            require(type(record["security"]) is bool and type(record["utility"]) is bool,
                    "Invalid native boolean scores: " + key)
        else:
            errors += 1
            require(record["security"] is None and record["utility"] is None and bool(record.get("error")),
                    "Native errors must retain original null scores: " + key)
        hashes[key] = sha(path)
    result = {"ready": False, "terminal_cases": len(paths), "native_error_cases": errors, "cases_sha256": hashes}
    if len(paths) != 778 or not (root / "summary.json").exists():
        return result
    summary = read(root / "summary.json")
    require(summary["execution_complete"] is True and summary["scheduled_cases"] == 778 and
            summary["spec_sha256"] == object_sha(identity), "Full summary identity or completion differs")
    require(summary["status"] == ("complete_with_initial_errors" if errors else "complete") and
            summary["data_fully_scored"] is (errors == 0), "Summary overwrote or hid initial native errors")
    for condition in identity["conditions"]:
        rows = [read(paths[c["case_id"]]) for c in cases if c["condition"] == condition]
        item = summary["conditions"][condition]
        count = sum(r["status"] == "complete" for r in rows)
        require(item["expected_cases"] == item["terminal_cases"] == 389 and item["missing_cases"] == 0 and
                item["scored_cases"] == count and item["error_cases"] == 389 - count,
                "Condition terminal/scored denominator differs")
        if count != 389:
            require(item["asr"] is None and item["utility"] is None, "Incomplete native scores acquired full-denominator rates")
    result.update(ready=True, summary_sha256=sha(root / "summary.json"))
    return result


def http_json(port, path, payload=None, *, timeout=15):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=None if payload is None else canonical(payload).encode(),
        headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_health(health, identity_sha256, *, ready=False, idle=False):
    require(not health.get("fatal_error") and health.get("stage") != "failed", "Native model server has a fatal failure")
    if ready:
        require(health.get("stage") == "ready" and health.get("identity_sha256") == identity_sha256,
                "Ready server identity differs")
    elif health.get("stage") not in {"starting", "loading", "preflight"}:
        require(health.get("stage") == "ready" and health.get("identity_sha256") == identity_sha256,
                "Unexpected server stage or identity")
    if idle:
        require(health.get("queued_or_running") == 0, "Server still has queued/running requests")
    # failed_calls also counts immutable native model errors; it is not a fatal gate.
    return health


def verify_server_instance(plan):
    root = Path(plan["output"])
    require(sha(root / "server_launch.json") == plan["server_launch_sha256"], "Server launch instance changed")
    require(process_identity(plan["server_instance"]["pid"]) == plan["server_instance"], "Bound server exited or PID was reused")


def prepare_plan(args):
    root, source = args.output.resolve(), args.source_dir.resolve()
    identity, _ = validate_schedule(root)
    require(identity["server_url"] == f"http://127.0.0.1:{args.port}/generate", "Requested port differs from frozen bridge URL")
    launch = read(root / "server_launch.json")
    instance = process_identity(launch["pid"])
    require(instance is not None and launch["port"] == args.port, "Current server launch is not alive on the explicit port")
    require(launch["source_sha256"] == sha(source / "run_deepseek_agentdojo_server.py"), "Server launch source changed")
    for executable in (args.native_python, args.report_python):
        require(executable.is_file() and os.access(executable, os.X_OK), "Missing explicit runtime: " + str(executable))
    require(args.native_repo.is_dir(), "Native repository is absent")
    bindings = verify_sources(root, source)
    for executable in (args.native_python, args.report_python):
        bindings[str(executable.absolute())] = sha(executable)
    for name in ("run_identity.json", "schedule.json", "protocol.json", "server_identity.json"):
        bindings[str(root / name)] = sha(root / name)
    plan = {"format": "deepseek-agentdojo-finalizer-plan-v1", "created_unix": time.time(),
        "output": str(root), "source_dir": str(source), "port": args.port,
        "native_python": str(args.native_python.absolute()), "report_python": str(args.report_python.absolute()),
        "native_repo": str(args.native_repo.resolve()), "resume_bridge": args.resume_bridge,
        "server_identity_sha256": identity["server_identity_sha256"], "server_instance": instance,
        "server_launch_sha256": sha(root / "server_launch.json"), "server_started_unix": launch["started_unix"],
        "source_bindings": bindings, "controller_path": str(Path(__file__).resolve()), "controller_sha256": sha(__file__),
        "poll_seconds": args.poll_seconds, "ready_timeout": args.ready_timeout,
        "completion_timeout": args.completion_timeout, "shutdown_timeout": args.shutdown_timeout,
        "command_timeout": args.command_timeout, "final_status": "pending_visual_review"}
    plan["bridge_command"] = [plan["native_python"], "-u", str(source / "deepseek_agentdojo_bridge.py"),
        "--output", str(root), "--repo", plan["native_repo"], "--mode", "full", "--workers", "8", "--resume",
        "--server-url", identity["server_url"], "--server-identity", str(root / "server_identity.json")]
    return plan


def verify_plan(plan, expected_sha):
    require(object_sha(plan) == expected_sha, "Prepared plan object SHA differs")
    require(sha(__file__) == plan["controller_sha256"] and str(Path(__file__).resolve()) == plan["controller_path"],
            "Controller source changed after plan review")
    require(verify_sources(Path(plan["output"]), Path(plan["source_dir"])).items() <= plan["source_bindings"].items(),
            "Frozen source manifest differs from reviewed plan")
    for name, digest in plan["source_bindings"].items():
        require(sha(name) == digest, "Reviewed source/run identity changed: " + name)
    validate_schedule(plan["output"])


def cpu_environment():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1"}


def bridge_worker(plan, expected_sha):
    """Detached worker records the real bridge child's exit without restarting it."""
    verify_plan(plan, expected_sha)
    folder = Path(plan["output"]) / "finalizer"
    destination = folder / "bridge-process.json"
    require(not destination.exists(), "Bridge worker is one-shot; refusing to replace an earlier process record")
    child = subprocess.Popen(plan["bridge_command"], stdin=subprocess.DEVNULL, env=cpu_environment())
    record = {"command": plan["bridge_command"], "worker_pid": os.getpid(),
              "worker_process_identity": process_identity(os.getpid()), "pid": child.pid,
              "process_identity": process_identity(child.pid), "started_unix": time.time(), "exit_code": None,
              "plan_sha256": expected_sha, "bridge_source_sha256": sha(Path(plan["source_dir"]) / "deepseek_agentdojo_bridge.py")}
    write(destination, record)
    code = child.wait()
    write(destination, {**record, "exit_code": code, "finished_unix": time.time()})
    return code


class Controller:
    def __init__(self, plan, plan_sha):
        self.plan, self.plan_sha = plan, plan_sha
        self.root = Path(plan["output"])
        self.folder = self.root / "finalizer"
        self.state = {"format": "deepseek-agentdojo-finalizer-status-v1", "status": "starting",
            "complete": False, "task_complete": False, "visual_review_required": True, "plan_sha256": plan_sha,
            "controller_sha256": sha(__file__), "started_unix": time.time(), "pid": os.getpid()}

    def stage(self, name, **detail):
        self.state.update(status=name, updated_unix=time.time(), **detail)
        write(self.folder / "status.json", self.state)

    def guard(self, *, server_alive=True):
        verify_plan(self.plan, self.plan_sha)
        if server_alive:
            verify_server_instance(self.plan)

    def wait_ready(self):
        deadline = time.monotonic() + self.plan["ready_timeout"]
        while True:
            self.guard()
            try:
                health = http_json(self.plan["port"], "/health")
            except (OSError, TimeoutError):
                health = None
            if health is not None:
                check_health(health, self.plan["server_identity_sha256"])
                if health["stage"] == "ready":
                    return health
            require(time.monotonic() < deadline, "Timed out waiting for the bound live server to become ready")
            self.stage("waiting_server_ready", health=health)
            time.sleep(self.plan["poll_seconds"])

    def resume_bridge(self):
        self.guard()
        check_health(http_json(self.plan["port"], "/health"), self.plan["server_identity_sha256"], ready=True, idle=True)
        bridge_lock = exclusive(self.root / ".bridge.lock")
        require(bridge_lock is not None, "A bridge is already active; refusing to launch a duplicate")
        bridge_lock.close()
        require(not (self.folder / "bridge-process.json").exists(), "No automatic bridge retry/restart is permitted")
        with (self.folder / "bridge-resume.log").open("xb") as log:
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--bridge-worker",
                str(self.folder / "plan.json"), "--plan-sha256", self.plan_sha], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=cpu_environment())
        write(self.folder / "bridge-worker-launch.json", {"pid": worker.pid, "started_unix": time.time(),
            "process_identity": process_identity(worker.pid), "plan_sha256": self.plan_sha})

    def wait_terminal(self):
        deadline = time.monotonic() + self.plan["completion_timeout"]
        worker_exit_grace = None
        while True:
            self.guard()
            observation_error = None
            try:
                health = check_health(http_json(self.plan["port"], "/health"), self.plan["server_identity_sha256"], ready=True)
            except (OSError, TimeoutError) as error:
                # Continue observing this exact live PID/launch; never relaunch a
                # model or bridge because a health observation timed out.
                health, observation_error = None, f"{type(error).__name__}: {error}"
            record_path = self.folder / "bridge-process.json"
            if self.plan["resume_bridge"] and record_path.exists():
                record = read(record_path)
                require(record["plan_sha256"] == self.plan_sha, "Bridge worker belongs to another plan")
                if record["exit_code"] is not None:
                    require(record["exit_code"] == 0, "Bridge exited unsuccessfully; no automatic rerun")
                elif process_identity(record["pid"]) != record["process_identity"]:
                    # Child exit and its worker's atomic exit record can cross this read.
                    latest = read(record_path)
                    require(latest["exit_code"] == 0 or (latest["exit_code"] is None and
                            record["worker_process_identity"] is not None and
                            process_identity(record["worker_pid"]) == record["worker_process_identity"]),
                            "Bridge vanished unsuccessfully or without an exit record")
                    if latest["exit_code"] is None:
                        worker_exit_grace = worker_exit_grace or time.monotonic() + 30
                        require(time.monotonic() < worker_exit_grace, "Bridge worker did not commit its exit record within 30 seconds")
            elif self.plan["resume_bridge"]:
                worker = read(self.folder / "bridge-worker-launch.json")
                require(worker["process_identity"] is not None and
                        process_identity(worker["pid"]) == worker["process_identity"],
                        "Detached bridge worker failed before writing its process record; inspect bridge-resume.log")
                require(time.time() - worker["started_unix"] < 30, "Detached bridge worker did not start its child within 30 seconds")
            snapshot = terminal_snapshot(self.root)
            if snapshot["ready"] and health is not None:
                bridge_lock = exclusive(self.root / ".bridge.lock")
                if bridge_lock is not None:
                    if self.plan["resume_bridge"]:
                        record = read(record_path)
                        if record["exit_code"] is None:
                            bridge_lock.close()
                        else:
                            require(record["plan_sha256"] == self.plan_sha and record["exit_code"] == 0,
                                    "Bridge final exit record is unsuccessful or belongs to another plan")
                            return bridge_lock, snapshot
                    else:
                        return bridge_lock, snapshot
            require(time.monotonic() < deadline, "Timed out waiting for all 778 immutable native terminal cases")
            self.stage("waiting_full_native_completion", terminal_cases=snapshot["terminal_cases"],
                       native_error_cases=snapshot["native_error_cases"], health=health,
                       health_observation_error=observation_error)
            time.sleep(min(1, self.plan["poll_seconds"]) if worker_exit_grace else self.plan["poll_seconds"])

    def command(self, label, command):
        self.guard(server_alive=label == "native_audit")
        self.stage("running_" + label)
        start = time.time()
        with (self.folder / (label + ".log")).open("xb") as log:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=cpu_environment(), timeout=self.plan["command_timeout"], check=False)
        write(self.folder / (label + "-process.json"), {"command": command, "started_unix": start,
              "finished_unix": time.time(), "exit_code": result.returncode, "plan_sha256": self.plan_sha})
        require(result.returncode == 0, label + " failed; inspect its durable log; no automatic retry")
        self.guard(server_alive=label == "native_audit")

    def shutdown(self):
        self.guard()
        deadline = time.monotonic() + self.plan["ready_timeout"]
        while True:
            self.guard()
            try:
                health = check_health(http_json(self.plan["port"], "/health"), self.plan["server_identity_sha256"], ready=True, idle=True)
                break
            except (OSError, TimeoutError) as error:
                require(time.monotonic() < deadline, "Timed out observing the bound live server before owned shutdown")
                self.stage("waiting_shutdown_health", health_observation_error=f"{type(error).__name__}: {error}")
                time.sleep(self.plan["poll_seconds"])
        requested = time.time()
        self.stage("requesting_owned_shutdown", shutdown_requested_unix=requested, shutdown_health=health)
        answer = http_json(self.plan["port"], "/shutdown", {"identity_sha256": self.plan["server_identity_sha256"]})
        require(answer == {"status": "closing"}, "Owned shutdown was not acknowledged")
        deadline = time.monotonic() + self.plan["shutdown_timeout"]
        while process_identity(self.plan["server_instance"]["pid"]) is not None:
            require(time.monotonic() < deadline, "Bound server PID did not exit after owned shutdown")
            time.sleep(min(1, self.plan["poll_seconds"]))
        self.guard(server_alive=False)
        exit_record, runtime, final, backend = (read(self.root / name) for name in
            ("server_exit.json", "server_runtime_final.json", "generation-final.json", "generation-backend.json"))
        expected = self.plan["server_identity_sha256"]
        require(exit_record["exit_code"] == 0 and exit_record["identity_sha256"] == runtime["identity_sha256"] == expected,
                "Server did not cleanly exit with the frozen byte identity")
        require(exit_record["finished_unix"] >= requested and runtime["finished_unix"] >= requested and
                datetime.fromisoformat(final["finished_at_utc"]).timestamp() >= requested,
                "Stale shutdown/runtime/final files cannot certify this server instance")
        require(final["identity_sha256"] == object_sha(backend) and final["engineering_test_only"] is False and
                runtime["backbone_audit"] == final["backbone_audit"] == backend["native_backbone_audit"],
                "Final native frozen all-CUDA backbone identity changed")
        require(runtime["requests_completed"] == health["completed_calls"] and runtime["requests_failed"] == health["failed_calls"],
                "Final server counters differ from the idle owned shutdown")
        write(self.folder / "owned-shutdown.json", {"requested_unix": requested, "health": health,
            "server_instance": self.plan["server_instance"], "artifacts_sha256": {name: sha(self.root / name)
                for name in ("server_exit.json", "server_runtime_final.json", "generation-final.json")}})

    def run(self):
        bridge_lock = None
        try:
            self.guard()
            self.wait_ready()
            if self.plan["resume_bridge"]:
                require(not terminal_snapshot(self.root)["ready"], "A complete run must not resume native episodes")
                self.resume_bridge()
            bridge_lock, snapshot = self.wait_terminal()
            self.guard()
            write(self.folder / "terminal-snapshot.json", snapshot)
            source = Path(self.plan["source_dir"])
            self.command("native_audit", [self.plan["native_python"], str(source / "deepseek_agentdojo_native_audit.py"),
                "--output", str(self.root), "--repo", self.plan["native_repo"]])
            audit = read(self.root / "native-evaluation-audit.json")
            require(audit["status"] == "passed" and audit["complete"] is True and
                    audit["scheduled_cases"] == audit["audited_cases"] == 778 and audit["model_generation_performed"] is False and
                    audit["cases_sha256"] == snapshot["cases_sha256"], "Independent full native audit did not pass the exact terminal snapshot")
            require(terminal_snapshot(self.root) == snapshot, "Native case/summary commits changed during the audit")
            self.shutdown()
            self.command("report", [self.plan["report_python"], str(source / "deepseek_agentdojo_report.py"),
                "--output", str(self.root), "--plots"])
            validation = read(self.root / "independent-validation.json")
            require(validation["status"] == "passed" and validation["complete"] is True and
                    validation["plots"]["status"] == "created", "Full independent report/plots have not passed")
            require(set(validation["plots"]["files_sha256"]) ==
                    {"asr-comparison.png", "first-payload-role-probabilities.png"}, "Expected report plots are absent")
            for name, digest in validation["plots"]["files_sha256"].items():
                path = (self.root / name).resolve()
                require(path.is_relative_to(self.root) and sha(path) == digest, "Report plot binding differs")
            self.guard(server_alive=False)
            self.stage("pending_visual_review", complete=False, task_complete=False, automation_finished=True,
                terminal_cases=778, validation_sha256=sha(self.root / "independent-validation.json"),
                native_audit_sha256=sha(self.root / "native-evaluation-audit.json"), plots=validation["plots"])
            return 0
        except BaseException as error:
            self.stage("failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                       stopped_unix=time.time(), automatic_retry=False)
            return 1
        finally:
            if bridge_lock is not None:
                bridge_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--bridge-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--report-python", type=Path)
    parser.add_argument("--native-repo", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--resume-bridge", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    parser.add_argument("--completion-timeout", type=float, default=259200)
    parser.add_argument("--shutdown-timeout", type=float, default=300)
    parser.add_argument("--command-timeout", type=float, default=14400)
    args = parser.parse_args(argv)
    if args.bridge_worker:
        require(args.plan_sha256, "Detached worker requires the reviewed plan SHA")
        return bridge_worker(read(args.bridge_worker), args.plan_sha256)
    require(args.output is not None, "Explicit --output is required")
    folder = args.output.resolve() / "finalizer"
    if args.prepare_only:
        require(all(x is not None for x in (args.native_python, args.report_python, args.native_repo, args.port)),
                "Preparing requires explicit native/report runtimes, native repo and port")
        require(0 < args.port < 65536 and 0 < args.poll_seconds <= 60 and
                min(args.ready_timeout, args.completion_timeout, args.shutdown_timeout, args.command_timeout) > 0,
                "Invalid explicit timeouts/port")
        require(not (folder / "plan.json").exists(), "Reviewed plan is immutable; refusing to replace it")
        plan = prepare_plan(args)
        write(folder / "plan.json", plan)
        (folder / "controller-source.py").write_bytes(Path(__file__).read_bytes())
        print(canonical({"status": "prepared_not_started", "plan_sha256": object_sha(plan), "controller_sha256": sha(__file__)}))
        return 0
    require(args.plan_sha256, "--check-only/--run require the reviewed --plan-sha256")
    plan = read(folder / "plan.json")
    require(Path(plan["output"]) == args.output.resolve(), "Plan output differs")
    verify_plan(plan, args.plan_sha256)
    if args.check_only:
        verify_server_instance(plan)
        print(canonical({"status": "verified_not_started", "plan_sha256": args.plan_sha256,
                         "terminal_cases": terminal_snapshot(plan["output"])["terminal_cases"]}))
        return 0
    controller_lock = exclusive(folder / ".controller.lock")
    require(controller_lock is not None, "Another finalizer controller owns this plan")
    require(not (folder / "status.json").exists(), "One-shot controller has a durable prior status; root review is required")
    try:
        return Controller(plan, args.plan_sha256).run()
    finally:
        controller_lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Even failure before Controller construction leaves a durable record.
        # Read-only check-only and prepare-only never alter execution status.
        if "--run" in sys.argv and "--output" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output") + 1]).resolve() / "finalizer"
            write(target / f"launch-failure-{time.time_ns()}.json", {
                "status": "failed", "task_complete": False, "automatic_retry": False,
                "controller_sha256": sha(__file__), "at_unix": time.time(),
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})
        raise
