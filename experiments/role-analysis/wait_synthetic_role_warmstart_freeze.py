"""One-shot local checkpoint delivery and cached AgentDojo evaluation.

This companion never starts remote GPU work and never changes the sealed calls.
It waits for the already-running campaign's DEV freeze, copies small artifacts,
and launches the CPU consumer against the original native feature observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def remote_snapshot(alias, root):
    code = f"""from pathlib import Path
import hashlib,json,time
r=Path({str(root)!r})
s=json.loads((r/'campaign-status.json').read_text())
l=json.loads((r/'campaign-launch.json').read_text())
p=Path('/proc/'+str(l['pid'])+'/stat')
a=p.read_text().rsplit(')',1)[1].split() if p.exists() else []
f=r/'run/freeze.json'
d={{'status':s,'observed_unix':time.time(),'driver_pid':l['pid'],
 'driver_live':bool(a and a[0]!='Z' and int(a[19])==l['start_ticks']),
 'freeze_exists':f.is_file()}}
if f.is_file():
 d['freeze']=json.loads(f.read_text())
 d['freeze_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
 d['probe_sha256']=hashlib.sha256((r/'run/best.npz').read_bytes()).hexdigest()
print(json.dumps(d))
"""
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias,
        "python", "-c", shlex.quote(code)], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-alias", default="runpod")
    parser.add_argument("--remote-root", default="/root/role-warmstart-20261027-6h")
    parser.add_argument("--private-root", required=True, type=Path)
    parser.add_argument("--calls-root", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--parent-probe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cpu-python", required=True)
    parser.add_argument("--budget-file", required=True, type=Path)
    parser.add_argument("--once", action="store_true", help="Only poll and record current state; do not evaluate")
    args = parser.parse_args()
    if args.ssh_alias != "runpod" or args.remote_root != "/root/role-warmstart-20261027-6h":
        raise ValueError("This one-shot delivery is bound to the authorized runpod experiment")
    args.private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(args.private_root, 0o700)
    status_path = args.private_root / "delivery-status.json"
    budget = json.loads(args.budget_file.read_text())
    if not 0 < budget["deadline_unix"] - budget["started_unix"] <= 21600.01:
        raise ValueError("Invalid source six-hour budget")
    last_stage = None
    while True:
        try:
            observed = remote_snapshot(args.ssh_alias, args.remote_root)
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            write(status_path, {"status": "observation_retry", "observed_unix": time.time(), "error": repr(error)})
            if args.once or time.time() > budget["deadline_unix"] + 600:
                raise
            time.sleep(30)
            continue
        stage = observed["status"].get("stage", observed["status"]["status"])
        write(status_path, {"status": "waiting_for_dev_freeze", "remote": observed})
        if stage != last_stage:
            print(json.dumps({"stage": stage, "driver_live": observed["driver_live"], "observed_unix": observed["observed_unix"]}), flush=True)
            last_stage = stage
        if args.once:
            return
        if observed["freeze_exists"]:
            frozen = observed["freeze"]
            if frozen.get("selection_split") != "dev" or frozen["probe_sha256"] != observed["probe_sha256"]:
                raise ValueError("Remote freeze is incomplete or the selected probe changed")
            break
        if not observed["driver_live"] or observed["status"]["status"] == "stopped":
            raise RuntimeError("Campaign stopped without a DEV-frozen checkpoint; inspect the existing driver")
        if time.time() > budget["deadline_unix"] + 600:
            raise TimeoutError("No checkpoint delivered within the original GPU window")
        time.sleep(30)
    snapshot = args.private_root / "checkpoint-snapshot" / "run"
    snapshot.mkdir(parents=True, exist_ok=True)
    source = f"{args.ssh_alias}:{args.remote_root}/run/"
    subprocess.run(["rsync", "-az", "--partial", "--exclude=*.pt", "--exclude=*.npy", source, str(snapshot) + "/"], check=True)
    if sha256(snapshot / "best.npz") != observed["probe_sha256"] or sha256(snapshot / "freeze.json") != observed["freeze_sha256"]:
        raise ValueError("Checkpoint copy differs from the frozen remote bytes")
    write(status_path, {"status": "checkpoint_copied_starting_cached_evaluation", "remote": observed,
                        "local_checkpoint": str(snapshot / "best.npz")})
    evaluator = Path(__file__).with_name("synthetic_role_warmstart_eval.py")
    argv = [args.cpu_python, str(evaluator), "--calls-root", str(args.calls_root), "--analysis-root", str(args.analysis_root),
            "--probe", str(snapshot / "best.npz"), "--parent-probe", str(args.parent_probe),
            "--freeze", str(snapshot / "freeze.json"), "--output-dir", str(args.output_dir)]
    manifest = args.output_dir / "result-manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("status") != "complete" or previous["probe_sha256"] != observed["probe_sha256"]:
            raise ValueError("Existing evaluation is not for this frozen checkpoint")
    else:
        with (args.private_root / "cached-evaluation.log").open("a") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = json.loads(manifest.read_text())
    if result.get("status") != "complete" or result["probe_sha256"] != observed["probe_sha256"]:
        raise ValueError("Cached evaluation did not complete for this checkpoint")
    for item in result["files"]:
        path = args.output_dir / item["path"]
        if not path.resolve().is_relative_to(args.output_dir.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError("Cached evaluation artifact changed: " + item["path"])
    write(status_path, {"status": "cached_evaluation_complete", "completed_unix": time.time(),
                        "local_checkpoint": str(snapshot / "best.npz"), "probe_sha256": observed["probe_sha256"],
                        "result_manifest_sha256": sha256(manifest), "evaluation_dir": str(args.output_dir)})
    print(json.dumps({"status": "cached_evaluation_complete", "probe_sha256": observed["probe_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
