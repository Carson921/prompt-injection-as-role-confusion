"""Run the approved, fixed-head, six-hour three-A40 experiment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def length_counts(path):
    result = {str(limit): {"records": 0, "input_tokens": 0} for limit in (512, 1024, 2048, 4096)}
    with Path(path).open() as stream:
        for line in stream:
            n = len(json.loads(line)["input_ids"])
            if not 0 < n <= 4096:
                raise ValueError("Invalid complete input length")
            bucket = result[str(next(limit for limit in (512, 1024, 2048, 4096) if n <= limit))]
            bucket["records"] += 1
            bucket["input_tokens"] += n
    return result


def predict_seconds(counts, pilot, loads=1):
    measured = pilot["length_bucket_timings"]
    total_tokens = sum(b["input_tokens"] for b in measured.values())
    total_forward = sum(b["forward_seconds"] for b in measured.values())
    if total_tokens <= 0 or total_forward <= 0:
        raise ValueError("Pilot has no completed timing observations")
    seconds = 0.0
    for key, value in counts.items():
        bucket = measured[key]
        per_token = bucket["forward_seconds"] / bucket["input_tokens"] if bucket["input_tokens"] else total_forward / total_tokens
        seconds += value["input_tokens"] * per_token
    # Include measured parsing, file commits and model loading, rather than just kernel time.
    extra = max(0, pilot["attempt_seconds"] - pilot["model_load_seconds"] - total_forward)
    seconds += extra / total_tokens * sum(b["input_tokens"] for b in counts.values())
    seconds += loads * pilot["model_load_seconds"]
    return seconds * 1.25


class Campaign:
    def __init__(self, root, python):
        self.root, self.python = Path(root).resolve(), python
        self.code = self.root / "code"
        self.data = self.root / "data"
        self.run = self.root / "run"
        self.features = self.root / "features"
        self.parent = self.root / "parent" / "best.npz"
        self.budget = self.root / "budget.json"
        self.logs = self.root / "logs"
        self.receipts = self.root / "phase-receipts"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.run.mkdir(parents=True, exist_ok=True)
        code_hashes = {p.name: digest(p) for p in sorted(self.code.glob("*.py"))}
        binding = {"code": code_hashes, "data_manifest_sha256": digest(self.data / "data-manifest.json"),
                   "article_audit_sha256": digest(self.data / "article-audit.json"),
                   "parent_probe_sha256": digest(self.parent), "root": str(self.root)}
        old = self.root / "campaign-binding.json"
        if old.exists() and json.loads(old.read_text()) != binding:
            raise ValueError("Code/data/parent changed during campaign; inspect before resuming")
        audit = json.loads((self.data / "article-audit.json").read_text())
        if (audit.get("status") != "passed" or
                audit.get("data_manifest_sha256") != binding["data_manifest_sha256"]):
            raise ValueError("The actual article review gate has not passed")
        manifest = json.loads((self.data / "data-manifest.json").read_text())
        if manifest.get("development_only") or manifest.get("quotas") != {"fit": 512, "dev": 128, "test": 128}:
            raise ValueError("Both formal scales and the full independent TEST must be prepared before GPU loading")
        selection = json.loads((self.data / "article-audit-sample/selection.json").read_text())
        payload_sha = digest(self.data / "article-audit-sample/articles.jsonl")
        if (len(selection.get("samples", [])) != 50 or selection.get("payload_sha256") != payload_sha or
                audit.get("sample_payload_sha256") != payload_sha or audit.get("reviewed_articles") != 50):
            raise ValueError("Article review is not bound to the actual fixed50 payload")
        if manifest.get("code_sha256") != binding["code"].get("synthetic_role_warmstart_data.py"):
            raise ValueError("Prepared data came from another builder")
        for name, expected in manifest["sha256"].items():
            if digest(self.data / name) != expected:
                raise ValueError("Prepared data changed: " + name)
        for split in ("fit", "dev", "test"):
            counts = length_counts(self.data / f"inputs-{split}.jsonl")
            if sum(item["records"] for item in counts.values()) != manifest["statistics"][split]["records"]:
                raise ValueError("Prepared split is missing records: " + split)
        write_json(old, binding)

    def remaining(self):
        if not self.budget.exists():
            return 21600
        value = json.loads(self.budget.read_text())
        if not 0 < value["deadline_unix"] - value["started_unix"] <= 21600.01:
            raise ValueError("Invalid six-hour budget")
        if value.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
            raise ValueError("GPU authorization belongs to another host boot")
        result = min(value["deadline_unix"] - time.time(), value["deadline_monotonic"] - time.monotonic())
        if result <= 0:
            raise TimeoutError("Six-hour execution window expired")
        return result

    def status(self, stage, **extra):
        write_json(self.root / "campaign-status.json", {"status": "running", "stage": stage,
                   "observed_unix": time.time(), **extra})

    def completed(self, name):
        path = self.receipts / (name + ".json")
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        if value.get("returncode") != 0:
            return False
        if value.get("artifact_proof") != self.artifact_proof(name):
            raise ValueError("Completed stage artifacts changed: " + name)
        return True

    def watchdog(self):
        state = self.root / "watchdog.json"
        if state.exists():
            old = json.loads(state.read_text())
            try:
                argv = Path(f"/proc/{old['pid']}/cmdline").read_bytes().split(b"\0")
                if str(self.code / "watch_synthetic_role_warmstart.py").encode() in argv:
                    return
            except OSError:
                pass
        with (self.logs / "watchdog.log").open("a") as stream:
            process = subprocess.Popen([self.python, "-u", str(self.code / "watch_synthetic_role_warmstart.py"),
                "--root", str(self.root)], stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Independent deadline supervisor exited before model loading")
            if state.exists() and json.loads(state.read_text()).get("pid") == process.pid:
                return
            time.sleep(.1)
        raise RuntimeError("Independent deadline supervisor did not initialize")

    def command(self, name, argv, *, env=None):
        self.remaining()
        receipt = self.receipts / (name + ".json")
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value.get("returncode") == 0 and value["argv"] == argv:
                if value.get("artifact_proof") != self.artifact_proof(name):
                    raise ValueError("Completed stage artifacts changed: " + name)
                return
        self.status(name)
        t0 = time.time()
        with (self.logs / (name + ".log")).open("a") as stream:
            completed = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                       timeout=self.remaining(), check=False)
        result = {"argv": argv, "started_unix": t0, "finished_unix": time.time(), "returncode": completed.returncode}
        if completed.returncode == 0:
            result["artifact_proof"] = self.artifact_proof(name)
        write_json(receipt, result)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed with return code {completed.returncode}; inspect its log")

    def artifact_proof(self, name):
        from synthetic_agent_classifier import Probe
        from synthetic_role_warmstart_gpu import validate_test_freeze
        from synthetic_role_warmstart_train import WarmBank

        if name == "preflight":
            path = self.run / "preflight.json"
            value = json.loads(path.read_text())
            parent = Probe(self.parent)
            if (value.get("status") != "passed" or value.get("devices") != [0, 1, 2] or
                    value.get("parent_probe_sha256") != digest(self.parent) or value.get("heads") != parent.metadata["heads"] or
                    value.get("runtime_sha256") != digest(self.code / "synthetic_agent_gpu.py") or
                    value.get("extractor_sha256") != digest(self.code / "synthetic_role_warmstart_gpu.py") or
                    value.get("model_binding") != parent.metadata["model_binding"]):
                raise ValueError("Preflight proof is invalid")
            return {"preflight_sha256": digest(path)}
        if name in ("pilot", "extract-fit", "extract-dev", "extract-test"):
            split = "fit" if name in ("pilot", "extract-fit") else name.split("-")[-1]
            directory = self.root / "pilot-features" if name == "pilot" else self.features
            bank = WarmBank(directory, split, expected_binding=Probe(self.parent).metadata)
            return {"feature_split_identity_sha256": bank.split_identity_hash, "rows": len(bank)}
        if name == "prepare-training":
            value = json.loads((self.run / "training.json").read_text())
            if value.get("parent_probe_sha256") != digest(self.parent) or value.get("engineering_smoke_only"):
                raise ValueError("Training preparation proof is invalid")
            return {"training_sha256": digest(self.run / "training.json")}
        if name == "freeze":
            validate_test_freeze(self.run / "freeze.json", Probe(self.parent))
            return {"freeze_sha256": digest(self.run / "freeze.json")}
        if name == "evaluate-test":
            validate_test_freeze(self.run / "freeze.json", Probe(self.parent))
            path = self.run / "evaluations/test.json"
            value = json.loads(path.read_text())
            if value.get("probe_sha256") != digest(self.run / "best.npz"):
                raise ValueError("Evaluation belongs to another checkpoint")
            return {"evaluation_sha256": digest(path),
                    "probabilities_sha256": digest(self.run / "evaluations/test-probabilities.npy")}
        raise ValueError("No artifact validator for stage " + name)

    def gpu_args(self, action):
        return [self.python, "-u", str(self.code / "synthetic_role_warmstart_gpu.py"), action,
                "--model-dir", "/root/role-retraining-model", "--parent-probe", str(self.parent),
                "--run-dir", str(self.run), "--budget-file", str(self.budget),
                "--cache-dir", str(self.root / "cache")]

    def extract(self, name, inputs, split, output):
        argv = self.gpu_args("extract") + ["--inputs", str(inputs), "--split", split, "--output-dir", str(output)]
        if split == "test":
            argv += ["--freeze", str(self.run / "freeze.json")]
        self.command(name, argv)

    def train_args(self, action):
        return [self.python, "-u", str(self.code / "synthetic_role_warmstart_train.py"), action,
                "--run-dir", str(self.run), "--features-dir", str(self.features),
                "--old-features-dir", str(self.root / "old-features"), "--parent-probe", str(self.parent),
                "--excluded-clusters", str(self.data / "quality-excluded-clusters.json")]

    def choose_scale(self):
        path = self.root / "scale-selection.json"
        if path.exists():
            result = json.loads(path.read_text())
            if (result.get("pilot_manifest_sha256") != digest(self.root / "pilot-features/manifest.json") or
                    result.get("budget_sha256") != digest(self.budget) or not result.get("chosen")):
                raise ValueError("Fixed scale selection lost its pilot/budget binding")
            chosen = result["chosen"]
            if chosen not in result["estimates"] or chosen["fit_sources"] not in (256, 512):
                raise ValueError("Fixed scale selection changed")
            pending = sum(chosen[key] for name, key in (("extract-fit", "fit_seconds_with_25pct_margin"),
                           ("extract-dev", "dev_seconds_with_25pct_margin")) if not self.completed(name))
            if not (self.run / "freeze.json").exists():
                pending += 1800
            if not (self.root / "gpu-complete.json").exists():
                pending += chosen["test_seconds_with_25pct_margin"]
            if pending + 4500 > self.remaining():
                raise TimeoutError("Selected scale cannot resume while preserving TEST and backup time")
            return result
        pilot = json.loads((self.root / "pilot-features" / "manifest.json").read_text())["progress"]["fit"]
        budget = json.loads(self.budget.read_text())
        dev = length_counts(self.data / "inputs-dev.jsonl")
        test = length_counts(self.data / "inputs-test.jsonl")
        estimates = []
        for quota, name in ((512, "inputs-fit.jsonl"), (256, "inputs-fit-conservative.jsonl")):
            fit = length_counts(self.data / name)
            fit_seconds, dev_seconds = predict_seconds(fit, pilot), predict_seconds(dev, pilot)
            predicted = fit_seconds + dev_seconds
            test_seconds = predict_seconds(test, pilot)
            fits = (time.time() + predicted <= budget["started_unix"] + 3 * 3600 and
                    time.time() + predicted + 1800 + test_seconds <= budget["started_unix"] + 4.75 * 3600)
            estimates.append({"fit_sources": quota, "file": name, "fit_dev_seconds_with_25pct_margin": predicted,
                              "fit_seconds_with_25pct_margin": fit_seconds, "dev_seconds_with_25pct_margin": dev_seconds,
                              "test_seconds_with_25pct_margin": test_seconds, "fits_windows": fits})
        chosen = next((e for e in estimates if e["fits_windows"]), None)
        result = {"selection_basis": "fixed64-source timing and memory only; no class scores", "estimates": estimates,
                  "chosen": chosen, "pilot_manifest_sha256": digest(self.root / "pilot-features" / "manifest.json"),
                  "budget_sha256": digest(self.budget),
                  "selected_unix": time.time()}
        write_json(path, result)
        if chosen is None:
            raise RuntimeError("Neither predeclared scale fits six hours; preserve preflight/pilot without formal training")
        return result

    def parallel_train(self):
        processes = []
        try:
            prepared = json.loads((self.run / "training.json").read_text())
            for name in ("warm-all", "warm-output", "scratch"):
                path = self.run / "candidates" / name / "complete.json"
                if path.exists():
                    value = json.loads(path.read_text())
                    if (value.get("candidate_id") != name or value.get("training_hash") != prepared["training_hash"] or
                            value.get("probe_sha256") != digest(path.parent / "best.npz") or
                            value.get("history_sha256") != digest(path.parent / "history.json") or
                            value.get("parent_probe_sha256") != digest(self.parent)):
                        raise ValueError("Completed candidate changed: " + name)
            for gpu, name in enumerate(("warm-all", "warm-output", "scratch")):
                complete = self.run / "candidates" / name / "complete.json"
                if complete.exists():
                    continue
                self.remaining()
                argv = self.train_args("train") + ["--candidate", name, "--device", "cuda:0", "--resume",
                                                   "--budget-file", str(self.budget)]
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                               "OPENBLAS_NUM_THREADS": "2"}
                stream = (self.logs / f"train-{name}.log").open("a")
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                processes.append((name, process, stream))
            while any(p.poll() is None for _, p, _ in processes):
                self.remaining()
                failures = [(name, p.returncode) for name, p, _ in processes if p.poll() is not None and p.returncode != 0]
                if failures:
                    raise RuntimeError("Candidate training failed: " + repr(failures))
                time.sleep(2)
            for name, process, _ in processes:
                if process.returncode:
                    raise RuntimeError(f"Candidate {name} failed: {process.returncode}")
        finally:
            for _, process, stream in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                stream.close()

    def run_campaign(self):
        self.watchdog()
        self.command("preflight", self.gpu_args("preflight"))
        self.extract("pilot", self.data / "inputs-pilot.jsonl", "fit", self.root / "pilot-features")
        scale = self.choose_scale()["chosen"]
        self.extract("extract-fit", self.data / scale["file"], "fit", self.features)
        self.extract("extract-dev", self.data / "inputs-dev.jsonl", "dev", self.features)
        self.command("prepare-training", self.train_args("prepare"))
        self.status("train-three-candidates")
        self.parallel_train()
        self.command("freeze", self.train_args("freeze"))
        self.extract("extract-test", self.data / "inputs-test.jsonl", "test", self.features)
        write_json(self.root / "gpu-complete.json", {"completed_unix": time.time(), "freeze_sha256": digest(self.run / "freeze.json")})
        self.command("evaluate-test", self.train_args("evaluate") + ["--split", "test"])
        write_json(self.root / "campaign-status.json", {"status": "gpu_and_article_test_complete",
            "finished_unix": time.time(), "remaining": "Local cached AgentDojo evaluation, independent review, report and verified backup"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(root, args.python)
        try:
            campaign.run_campaign()
        except Exception as error:
            write_json(root / "campaign-status.json", {"status": "stopped", "error": repr(error), "observed_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
