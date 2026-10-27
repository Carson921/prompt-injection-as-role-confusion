"""Independent six-hour deadline and telemetry for this exact experiment root."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def option(argv, key):
    return argv[argv.index(key) + 1] if key in argv and argv.index(key) + 1 < len(argv) else None


def targets(root):
    allowed = {str(root / "code" / name) for name in (
        "synthetic_role_warmstart_gpu.py", "synthetic_role_warmstart_train.py")}
    found = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            argv = [s.decode() for s in (item / "cmdline").read_bytes().split(b"\0") if s]
            # Recognize only the executed Python script slot, not incidental arguments.
            if len(argv) < 2 or "python" not in Path(argv[0]).name:
                continue
            index = 1
            while index < len(argv) and argv[index] in ("-u", "-B", "-E", "-s"):
                index += 1
            if index >= len(argv) or argv[index] not in allowed:
                continue
            scoped = (option(argv, "--budget-file") == str(root / "budget.json") or
                      option(argv, "--run-dir") == str(root / "run"))
            if not scoped:
                continue
            stat = (item / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            found.append({"pid": int(item.name), "start_ticks": int(stat[19]),
                          "script": argv[index], "state": stat[0]})
        except (OSError, ValueError, UnicodeError):
            continue
    return found


def same_process(root, old):
    return any(p["pid"] == old["pid"] and p["start_ticks"] == old["start_ticks"] for p in targets(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proof = {"pid": os.getpid(), "root": str(root), "started_unix": started,
             "status": "waiting_for_first_gpu_load", "kills": []}
    state = root / "watchdog.json"
    write_json(state, proof)
    header = ("unix", "gpu", "uuid", "memory_used_mib", "memory_total_mib", "utilization_pct")
    budget = None
    with (root / "gpu-resources.csv").open("a", buffering=1) as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(header)
        while True:
            if budget is None and (root / "budget.json").is_file():
                try:
                    budget = json.loads((root / "budget.json").read_text())
                except (OSError, ValueError) as error:
                    proof["last_budget_read_error"] = repr(error)
                    write_json(state, proof)
                    time.sleep(.1)
                    continue
                if not 0 < budget["deadline_unix"] - budget["started_unix"] <= 21600.01:
                    raise ValueError("Budget is not a six-hour authorization")
                if budget.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
                    raise ValueError("Budget belongs to another host boot")
                if not 0 < budget["deadline_monotonic"] - budget["started_monotonic"] <= 21600.01:
                    raise ValueError("Invalid monotonic six-hour authorization")
                proof.update(status="monitoring", budget=budget)
            # Deadline is read once from local storage; a stalled data mount cannot extend it.
            seconds_left = (min(budget["deadline_unix"] - time.time(), budget["deadline_monotonic"] - time.monotonic())
                            if budget is not None else None)
            if seconds_left is not None and seconds_left <= 0:
                proof["status"] = "deadline_reached"
                victims = targets(root)
                for victim in victims:
                    if same_process(root, victim):
                        try:
                            os.kill(victim["pid"], signal.SIGKILL)
                            proof["kills"].append({**victim, "signal": "KILL", "unix": time.time()})
                        except ProcessLookupError:
                            pass
                proof.update(finished_unix=time.time(), remaining_targets=targets(root))
                write_json(state, proof)
                return
            if (root / "gpu-complete.json").is_file() and not targets(root):
                proof.update(status="gpu_completed", finished_unix=time.time(), remaining_targets=[])
                write_json(state, proof)
                return
            # Do not let a telemetry subprocess overrun the hard deadline.
            if seconds_left is not None and seconds_left < 4:
                time.sleep(max(0, seconds_left))
                continue
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True, timeout=3)
                now = time.time()
                for line in output.strip().splitlines():
                    writer.writerow([now, *[part.strip() for part in line.split(",")]])
            except (OSError, subprocess.SubprocessError) as error:
                proof["last_telemetry_error"] = repr(error)
            proof["observed_unix"] = time.time()
            proof["active_targets"] = targets(root)
            write_json(state, proof)
            delay = 5 if budget is None else min(5, max(0, min(budget["deadline_unix"] - time.time(), budget["deadline_monotonic"] - time.monotonic())))
            time.sleep(delay)


if __name__ == "__main__":
    main()
