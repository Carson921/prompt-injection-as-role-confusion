"""Resume the fixed pilot/formal campaign without changing scientific quotas.

This driver launches the existing stage CLI in child processes, with one log
per attempt. It never continues after a failed stage, audit gate, or budget
gate. Run it on the four-A40 host; this module does not submit remote jobs.
``--prepare-formal-only`` is a separate CPU-only preparation mode.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

QUOTAS = {"pilot": {"fit": 512, "dev": 128},
          "formal": {"fit": 4096, "dev": 512, "test": 1024}}
ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]


class CampaignStop(RuntimeError):
    def __init__(self, status, message, code=1):
        super().__init__(message)
        self.status, self.code = status, code


def require(value, message):
    if not value:
        raise CampaignStop("integrity_error", message, 23)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def files_proof(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def bounded_path(root, name):
    result = (Path(root) / name).resolve()
    require(result.is_relative_to(Path(root).resolve()), "Artifact path escapes its owned run directory")
    return result


def effective_config(path, phase):
    config = read(path)
    config["phase"] = phase
    config.setdefault("classifier", {})["max_epochs"] = 15 if phase == "pilot" else 45
    return config


def _parse_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Budget deadlines must have an explicit timezone")
    return result.timestamp()


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = args.run_root.resolve()
        self.work = self.root / "campaign"
        self.work.mkdir(parents=True, exist_ok=True)
        self.runs = {"pilot": (args.pilot_run_dir or self.root / "pilot").resolve(),
                     "formal": (args.formal_run_dir or self.root / "formal").resolve()}
        self.config_paths = {"pilot": args.pilot_config.resolve(), "formal": args.formal_config.resolve()}
        self.configs = {phase: effective_config(path, phase) for phase, path in self.config_paths.items()}
        deadlines = []
        for config in self.configs.values():
            budget = config.get("budget", {})
            require(all(key in budget for key in ["started_utc", "deadline_utc", "authorized_hours"]),
                    "Both configurations must record the authorized start, deadline, and hours")
            hours = float(budget["authorized_hours"])
            started, deadline = _parse_time(budget["started_utc"]), _parse_time(budget["deadline_utc"])
            require(0 < hours <= 24 and started < deadline <= started + hours * 3600,
                    "A configured deadline exceeds its authorized window or the 24-hour maximum")
            deadlines.append(deadline)
        self.deadline = min(deadlines)
        self.stage = None
        self.status = {"format_version": "synthetic-agent-campaign-v1", "status": "starting",
                       "deadline_utc": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
                       "data_dir": str(args.data_dir.resolve()), "run_root": str(self.root)}
        self.binding = {"model_dir": str(args.model_dir.resolve()), "data_dir": str(args.data_dir.resolve()),
                        "run_dirs": {k: str(v) for k, v in self.runs.items()},
                        "configs": {k: identity(v) for k, v in self.configs.items()},
                        "deadline_unix": self.deadline, "devices": args.devices}
        binding_path = self.work / "binding.json"
        with (self.work / "setup.lock").open("a+") as setup_lock:
            fcntl.flock(setup_lock, fcntl.LOCK_EX)
            if binding_path.exists():
                require(read(binding_path) == self.binding, "Campaign paths/configurations changed; do not mix runs")
            else:
                atomic_json(binding_path, self.binding)
        for phase, run in self.runs.items():
            run.mkdir(parents=True, exist_ok=True)
            if (run / "config.json").exists():
                require(read(run / "config.json") == self.configs[phase], f"{phase} run configuration changed")
        sys.path.insert(0, str(args.code_dir.resolve()))

    def remaining(self):
        return self.deadline - time.time()

    def publish(self, **fields):
        self.status.update(fields, updated_unix=time.time(), remaining_seconds=max(0, self.remaining()))
        filename = "formal-prepare-status.json" if self.args.prepare_formal_only else "status.json"
        atomic_json(self.work / filename, self.status)

    def audit_gate(self, phase):
        folder = self.args.data_dir / phase
        path = folder / "article-audit.json"
        if not path.exists():
            raise CampaignStop("waiting_article_audit", f"Missing scientific audit gate: {path}", 20)
        audit = read(path)
        if audit.get("status") != "passed":
            raise CampaignStop("waiting_article_audit", f"Article audit has not passed: {path}", 20)
        require(audit.get("data_manifest_sha256") == sha(folder / "data-manifest.json"),
                f"{phase} audit does not bind the current data manifest")
        require(audit.get("sources_sha256") == sha(folder / "sources.jsonl"),
                f"{phase} audit does not bind the current source articles")
        return files_proof([path])

    def prepared(self, phase):
        folder = self.args.data_dir / phase
        path = folder / "data-manifest.json"
        if not path.exists():
            return None
        manifest = read(path)
        require(manifest.get("phase") == phase and manifest.get("roles") == ROLES, "Wrong prepared data format")
        require(manifest["configuration"]["code_sha256"] == sha(self.args.code_dir / "synthetic_agent_data.py"),
                "Prepared articles belong to a different data implementation")
        for split, expected in QUOTAS[phase].items():
            require(manifest["statistics"][split]["sources"] == expected,
                    f"{phase}/{split} source quota changed; the campaign does not shrink quotas")
        require(set(manifest["statistics"]) == set(QUOTAS[phase]), "Unexpected prepared data splits")
        # Reuse the builder's cache-only validation branch so its complete
        # config/tokenizer/backend/local-source fingerprint cannot drift from
        # this driver's adoption logic. A manifest already exists here, so the
        # builder returns before source collection or input generation.
        from transformers import AutoTokenizer
        from synthetic_agent_data import prepare_data
        if not hasattr(self, "_bound_tokenizer"):
            self._bound_tokenizer = AutoTokenizer.from_pretrained(self.args.model_dir, local_files_only=True)
        checked = prepare_data(self.configs[phase], self._bound_tokenizer, self.args.data_dir,
                               None, phase=phase, resume=True)
        require(checked == manifest, "Prepared data changed during cache adoption")
        # prepare_data's resume path returns early, so publish the small report
        # mirror here as well when CPU preparation ran in another process.
        mirror = self.runs[phase] / f"data-{phase}-manifest.json"
        if not mirror.exists():
            shutil.copyfile(path, mirror)
        require(sha(mirror) == sha(path), "Run data-manifest mirror does not match prepared data")
        return {"data_manifest_sha256": sha(path), "source_sha256": manifest["sha256"]["sources.jsonl"],
                "configuration_sha256": manifest["config_sha256"]}

    def preflight(self, phase):
        from synthetic_agent_gpu import model_binding
        path = self.runs[phase] / "preflight.json"
        if not path.exists():
            return None
        value = read(path)
        require(value.get("status") == "passed" and value.get("maximum_input_tokens") == 4096,
                "The four-A40 longest-input preflight has not passed")
        require(value["model_binding"] == model_binding(self.args.model_dir), "Preflight backbone identity changed")
        require(value.get("full_head_scan_workspace", {}).get("statistics_finite") is True,
                "The actual all-head statistic workspace was not preflighted")
        return files_proof([path])

    def selected(self, phase):
        from synthetic_agent_gpu import model_binding
        path = self.runs[phase] / "head-selection.json"
        if not path.exists():
            return None
        value = read(path)
        require(value["roles"] == ROLES and len(value["heads"]) == len(set(value["heads"])) == 16,
                "Invalid six-class ordered head selection")
        signature = value["signature"]
        require(signature["pilot_fit_sha256"] == sha(self.args.data_dir / "pilot" / "inputs-fit.jsonl"),
                "Selected heads belong to different pilot FIT inputs")
        require(signature["model_binding"] == model_binding(self.args.model_dir), "Selected-head backbone changed")
        require(signature["implementation_sha256"] == sha(self.args.code_dir / "synthetic_agent_gpu.py"),
                "Selected heads belong to a different GPU implementation")
        require(value.get("no_dev_test_or_historical_probe_used") is True, "Head-discovery scope is not explicit")
        if phase == "formal":
            require(sha(path) == sha(self.runs["pilot"] / "head-selection.json"),
                    "Formal heads must inherit the exact pilot selection")
        return files_proof([path])

    def feature_manifest(self, phase, split="dev", external=False):
        import numpy as np
        name = "external" if external else phase
        root = self.runs[phase] / "features" / name
        path = root / "manifest.json"
        if not path.exists():
            return None
        manifest = read(path)
        if not manifest.get("progress", {}).get(split, {}).get("complete"):
            return None
        data_path = self.args.data_dir / name / f"inputs-{split}.jsonl"
        require(manifest.get("format_version") == "synthetic-agent-features-v1" and manifest.get("roles") == ROLES and
                manifest.get("feature_dim") == 1024 and manifest.get("feature_position") == "pre-o_proj",
                "Feature format is not the new six-role 16-head format")
        require(manifest["split_sources"][split] == sha(data_path), "Feature inputs changed")
        require(manifest["selection_sha256"] == sha(self.runs[phase] / "head-selection.json") and
                manifest["pilot_fit_sha256"] == sha(self.args.data_dir / "pilot" / "inputs-fit.jsonl"),
                "Feature head-discovery binding changed")
        from synthetic_agent_gpu import model_binding, object_digest
        require(manifest["model_binding"] == model_binding(self.args.model_dir), "Feature backbone changed")
        require(manifest["config_hash"] == object_digest(self.configs[phase]), "Feature configuration changed")
        primary = {part: sha(self.args.data_dir / name / f"inputs-{part}.jsonl") for part in ["fit", "dev"]
                   if (self.args.data_dir / name / f"inputs-{part}.jsonl").exists()}
        require(manifest["data_hash"] == object_digest(primary), "Primary feature-data binding changed")
        require(manifest["signature"]["implementation_sha256"] == sha(self.args.code_dir / "synthetic_agent_gpu.py"),
                "Feature implementation changed")
        signature = {**manifest["signature"], "split": split, "input_sha256": sha(data_path)}
        entries = manifest["splits"][split]
        require(entries, "Completed feature split is empty")
        for entry in entries:
            directory = bounded_path(root, entry["path"])
            complete = directory / "complete.json"
            require(sha(complete) == entry["sha256"], "Feature shard commit changed")
            commit = read(complete)
            require(commit["signature"] == signature and commit["rows"] == entry["rows"], "Feature shard identity changed")
            for filename, expected in commit["files"].items():
                require(sha(bounded_path(directory, filename)) == expected, "Feature shard bytes changed")
            require({"x.npy", "y.npy", "structure.npy", "source.npy", "family.npy", "weight.npy"}.issubset(commit["files"]),
                    "A completed feature shard is missing required arrays")
            x = np.load(directory / "x.npy", mmap_mode="r", allow_pickle=False)
            require(x.dtype == np.float16 and x.shape == (entry["rows"], 1024), "Feature shard FP16 shape changed")
            for field in ["y", "structure", "source", "family", "weight"]:
                require(np.load(directory / f"{field}.npy", mmap_mode="r", allow_pickle=False).shape == (entry["rows"],),
                        f"Feature shard row metadata shape changed: {field}")
        progress = manifest["progress"][split]
        require(progress["rows"] == sum(e["rows"] for e in entries), "Feature row count does not match committed shards")
        # Match the consumer's immutable split identity, which survives adding
        # TEST to the same manifest after FIT/DEV are already frozen.
        split_identity = identity({"format_version": manifest["format_version"], "roles": manifest["roles"],
            "feature_dim": manifest["feature_dim"], "feature_position": manifest["feature_position"],
            "signature": manifest["signature"], "data_hash": manifest["data_hash"], "split": split,
            "input_sha256": manifest["split_sources"][split], "shards": entries,
            "complete": progress.get("complete"), "records": progress.get("records"), "rows": progress.get("rows")})
        return {"split_identity": split_identity, "rows": sum(e["rows"] for e in entries),
                "input_sha256": manifest["split_sources"][split], "entries": entries}

    def features(self, phase, splits=("fit", "dev"), external=False):
        result = {split: self.feature_manifest(phase, split, external) for split in splits}
        return result if all(result.values()) else None

    def trained(self, phase):
        from synthetic_agent_classifier import CANDIDATES, _training_options, load_probe, selection_key
        root = self.runs[phase]
        if not (root / "selection.json").exists():
            return None
        training = read(root / "training.json")
        require(training["phase"] == phase and training["implementation_sha256"] == sha(self.args.code_dir / "synthetic_agent_classifier.py"),
                "Training implementation or phase changed")
        require(training["options"] == _training_options(self.configs[phase]), "Training options changed")
        feature_proofs = {split: self.feature_manifest(phase, split) for split in ["fit", "dev"]}
        require(all(feature_proofs.values()), "Completed training has incomplete feature inputs")
        split_hashes = {split: proof["split_identity"] for split, proof in feature_proofs.items()}
        require(training["feature_split_hashes"] == split_hashes,
                "Training belongs to different committed FIT/DEV feature shards")
        expected_training_hash = identity({"fit_features": split_hashes["fit"], "dev_features": split_hashes["dev"],
            "options": training["options"], "implementation_sha256": training["implementation_sha256"], "candidates": CANDIDATES})
        require(training["training_hash"] == expected_training_hash, "Training hash does not match the current immutable feature identities")
        manifest = read(root / "features" / phase / "manifest.json")
        for key, value in training["binding"].items():
            require(manifest.get(key) == value, f"Training feature binding changed: {key}")
        results, paths = [], [root / "training.json", root / "selection.json", root / "standardizer.npz", root / "resume-validation.json"]
        require(read(root / "resume-validation.json").get("passed") is True, "GPU recovery/export validation did not pass")
        for candidate_id, _, _ in CANDIDATES:
            directory = root / "candidates" / candidate_id
            path = directory / "complete.json"
            if not path.exists():
                return None
            candidate = read(path)
            require(candidate["training_hash"] == training["training_hash"], "Candidate training identity changed")
            require(candidate["probe_sha256"] == sha(directory / "best.npz") and
                    candidate["history_sha256"] == sha(directory / "history.json"), "Candidate artifacts changed")
            probe = load_probe(directory / "best.npz", expected_binding=training["binding"])
            require(not probe.metadata.get("engineering_smoke_only"), "A CPU smoke model cannot enter the campaign")
            results.append(candidate)
            paths += [path, directory / "best.npz", directory / "history.json"]
        selection = read(root / "selection.json")
        winner = min(results, key=lambda row: selection_key(row["dev"], row["candidate_id"]))
        require(selection["candidate_results"] == results and selection["candidate_id"] == winner["candidate_id"] and
                selection["training_hash"] == training["training_hash"], "DEV selection differs from the fixed seven-candidate rule")
        return files_proof(paths)

    def frozen(self, phase):
        root = self.runs[phase]
        path = root / "freeze.json"
        if not path.exists():
            return None
        value = read(path)
        require(value["probe_sha256"] == sha(root / "best.npz") and
                value["selection_sha256"] == sha(root / "selection.json") and
                value["training_sha256"] == sha(root / "training.json"), "Frozen classifier artifacts changed")
        require(value["selection_split"] == "dev" and value["metadata"]["roles"] == ROLES,
                "Freeze is not a six-role DEV-only decision")
        if phase == "formal":
            pointer = root / "external-protocol.json"
            require(value["external_protocol_pointer_sha256"] == sha(pointer), "External protocol pointer changed after freeze")
            require(value["external_protocol_sha256"] == sha(self.args.data_dir / "external" / "protocol.json"),
                    "External protocol changed after freeze")
        return files_proof([path, root / "best.npz"])

    def evaluation(self, phase, split):
        import numpy as np
        root = self.runs[phase]
        path = root / "evaluations" / f"{split}.json"
        probability_path = path.with_name(f"{split}-probabilities.npy")
        if not path.exists() or not probability_path.exists():
            return None
        value = read(path)
        require(value["probe_sha256"] == read(root / "freeze.json")["probe_sha256"], "Evaluation uses a different frozen probe")
        external = split == "external"
        feature_split = "test" if external else split
        feature_root = root / "features" / ("external" if external else phase)
        manifest = read(feature_root / "manifest.json")
        expected_rows = sum(e["rows"] for e in manifest["splits"][feature_split])
        probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
        require(probability.shape == (expected_rows, 6) and np.isfinite(probability).all() and
                np.allclose(probability.sum(1), 1, atol=3e-5), "Evaluation probability output is incomplete")
        labels = np.concatenate([np.load(bounded_path(feature_root, e["path"]) / "y.npy", mmap_mode="r", allow_pickle=False)
                                 for e in manifest["splits"][feature_split]])
        confusion = np.zeros((6, 6), dtype=np.int64)
        np.add.at(confusion, (labels, probability.argmax(1)), 1)
        require(value["overall"]["confusion"] == confusion.tolist(), "Evaluation metrics and probability rows disagree")
        if split != "dev":
            require(value.get("confidence_intervals", {}).get("replicates") == 2000 and
                    value["confidence_intervals"]["unit"] == "source_cluster", "Final cluster confidence intervals are missing")
        return files_proof([path, probability_path])

    def external_protocol(self):
        root = self.args.data_dir / "external"
        pointer = self.runs["formal"] / "external-protocol.json"
        if not pointer.exists() or not (root / "protocol.json").exists():
            return None
        value = read(pointer)
        require(value["protocol_sha256"] == sha(root / "protocol.json") and
                value["history_index_sha256"] == sha(root / "history-index.json") and
                value["new_external_bodies_read"] is False, "External protocol binding changed")
        require(read(root / "protocol.json")["implementation_sha256"] == sha(self.args.code_dir / "synthetic_agent_external.py"),
                "External protocol implementation changed")
        return files_proof([pointer, root / "protocol.json", root / "history-index.json"])

    def external_protocol_gate(self):
        root = self.args.data_dir / "external"
        required = [root / "protocol.json", root / "history-index.json"]
        if not all(path.is_file() for path in required):
            raise CampaignStop("waiting_external_protocol",
                "Copy the locally predeclared metadata-only external protocol and history-index.json before continuing; historical CSV bodies must stay local", 25)
        protocol = read(root / "protocol.json")
        require(protocol["history_index_sha256"] == sha(root / "history-index.json"), "External exclusion index changed")

    def legacy_hash_gate(self):
        path = self.args.data_dir / "external" / "legacy-hashes.json"
        if not path.is_file():
            raise CampaignStop("waiting_legacy_hashes",
                "After classifier freeze, compute legacy-hashes.json locally and copy only that hash ledger; do not transfer historical CSV bodies", 26)
        from synthetic_agent_external import _load_legacy_hashes
        protocol_path = self.args.data_dir / "external" / "protocol.json"
        _load_legacy_hashes(path, read(protocol_path), protocol_path, self.runs["formal"],
                            read(self.runs["formal"] / "freeze.json"))
        return files_proof([path])

    def external_prepared(self):
        root = self.args.data_dir / "external"
        path = root / "manifest.json"
        if not path.exists():
            return None
        value = read(path)
        require(value["inputs_sha256"] == sha(root / "inputs-test.jsonl") and
                value["probe_sha256"] == read(self.runs["formal"] / "freeze.json")["probe_sha256"] and
                value["protocol_sha256"] == sha(root / "protocol.json"), "External inputs changed")
        require(value["counts_by_source"] == {"oasst": 128, "toxicchat": 128}, "External quotas changed")
        return files_proof([path, root / "inputs-test.jsonl"])

    def plotted(self, phase, split):
        root = self.runs[phase] / "plots" / split
        path = root / "index.json"
        if not path.exists():
            return None
        value = read(path)
        require(value["probe_sha256"] == read(self.runs[phase] / "freeze.json")["probe_sha256"], "Plots use a different frozen probe")
        require(not value["missing_coverage"], "Required tomato structural coverage is missing")
        paths = [path]
        for entry in value["examples"].values():
            paths += [bounded_path(root, entry["png"]), bounded_path(root, entry["pdf"])]
        return files_proof(paths)

    def reported(self, phase):
        root = self.runs[phase]
        if not (root / "report-status.json").exists() or not (root / "REPORT.zh-CN.md").exists():
            return None
        if read(root / "report-status.json")["missing"]:
            return None
        return files_proof([root / "report-status.json", root / "REPORT.zh-CN.md",
                            root / "training-curves.png", root / "training-curves.pdf"])

    def _command(self, phase, stage, split):
        result = [str(self.args.python), "-u", str(self.args.code_dir / "run_synthetic_agent_roles.py"), stage,
                  "--config", str(self.config_paths[phase]), "--phase", phase,
                  "--model-dir", str(self.args.model_dir), "--data-dir", str(self.args.data_dir),
                  "--run-dir", str(self.runs[phase]), "--cache-dir", str(self.args.cache_dir),
                  "--devices", self.args.devices, "--resume"]
        if split:
            result += ["--split", split]
        return result

    def _execute(self, name, command):
        remaining = self.remaining()
        if remaining <= 0:
            raise CampaignStop("deadline_exceeded", "The authorized 24-hour execution window expired", 22)
        folder = self.work / "logs"
        folder.mkdir(exist_ok=True)
        attempt = len(list(folder.glob(name + ".*.log"))) + 1
        log_path = folder / f"{name}.{attempt:03d}.log"
        print(canonical({"campaign_stage": name, "status": "running", "log": str(log_path)}), flush=True)
        self.publish(status="running", stage=name, log=str(log_path))
        started = time.time()
        with log_path.open("w") as log:
            # Status writes on a network volume and process startup consume
            # budget too. Never give a child the stale pre-write remainder.
            if self.remaining() <= 0:
                raise CampaignStop("deadline_exceeded", "The budget expired before child-process launch", 22)
            process = subprocess.Popen(command, cwd=self.args.code_dir, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1",
                                                                   "CUDA_VISIBLE_DEVICES": self.args.devices})
            try:
                returncode = process.wait(timeout=max(0, self.remaining()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise CampaignStop("deadline_exceeded", f"Stopped {name} at the authorized deadline; inspect {log_path}", 22)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                log.flush()
                os.fsync(log.fileno())
        if returncode:
            raise CampaignStop("failed", f"{name} exited {returncode}; inspect {log_path}")
        return {"started_unix": started, "seconds": time.time() - started,
                "log": str(log_path), "command": command}

    def step(self, phase, stage, validator, split=None, command=None):
        name = ".".join(x for x in [phase, stage, split] if x)
        self.stage = name
        if self.remaining() <= 0:
            raise CampaignStop("deadline_exceeded", "The authorized execution window expired", 22)
        receipt_path = self.work / "receipts" / f"{name}.json"
        proof = validator()
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt["binding"] == self.binding and proof is not None and
                    receipt["proof_sha256"] == identity(proof), f"Completed stage changed: {name}")
            return receipt
        marker_path = self.runs[phase] / f"{stage}-stage.json"
        execution = None
        if proof is None:
            if stage == "train" and (self.runs[phase] / "freeze.json").exists():
                raise CampaignStop("integrity_error", "Refusing to retrain an already frozen run", 23)
            execution = self._execute(name, command or self._command(phase, stage, split))
            proof = validator()
            require(proof is not None, f"{name} exited successfully without complete functional artifacts")
            if command is None:
                marker = read(marker_path)
                require(marker["status"] == "complete" and marker["stage"] == stage and marker["phase"] == phase,
                        "CLI success marker does not match the completed stage")
                expected_hash = hashlib.sha256(json.dumps(self.configs[phase], sort_keys=True).encode()).hexdigest()
                require(marker["config_hash"] == expected_hash, "CLI marker configuration mismatch")
        else:
            marker = read(marker_path) if marker_path.exists() else None
            # Immutable atomic commits (manifest, seven candidate commits, or
            # freeze) may come from an independently completed preparation job.
            # Their verified functional proof is a stronger adoption criterion
            # than the generic stage marker, which later split runs overwrite.
            execution = {"adopted_verified_atomic_artifacts": True, "seconds": 0}
        marker = read(marker_path) if marker_path.exists() else {"kind": "verified_functional_commit", "stage": stage, "phase": phase}
        receipt = {"name": name, "binding": self.binding, "proof": proof,
                   "proof_sha256": identity(proof), "stage_marker_snapshot": marker,
                   "execution": execution, "completed_unix": time.time()}
        atomic_json(receipt_path, receipt)
        print(canonical({"campaign_stage": name, "status": "complete", "adopted": execution.get("adopted_verified_atomic_artifacts", False)}), flush=True)
        return receipt

    def inherit_pilot(self):
        self.preflight("pilot")
        self.selected("pilot")
        paths = ["preflight.json", "head-selection.json"]
        for name in paths:
            source, target = self.runs["pilot"] / name, self.runs["formal"] / name
            if target.exists():
                require(sha(target) == sha(source), f"Formal {name} differs from its pilot source")
            else:
                temporary = target.with_name(target.name + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
        provenance = {"pilot_run": str(self.runs["pilot"]), "pilot_fit_sha256": sha(self.args.data_dir / "pilot" / "inputs-fit.jsonl"),
                      "head_selection_sha256": sha(self.runs["pilot"] / "head-selection.json"),
                      "preflight_sha256": sha(self.runs["pilot"] / "preflight.json"), "formal_head_rescan": False}
        path = self.runs["formal"] / "pilot-inheritance.json"
        if path.exists():
            require(read(path) == provenance, "Formal pilot-inheritance provenance changed")
        else:
            atomic_json(path, provenance)
        self.preflight("formal")
        self.selected("formal")

    def prepare_formal(self):
        with (self.work / "formal-prepare.lock").open("a+") as lock:
            announced = False
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self.remaining() <= 0:
                        raise CampaignStop("deadline_exceeded", "The budget expired while waiting for parallel formal preparation", 22)
                    if not announced:
                        self.publish(status="waiting_parallel_preparation", stage="formal.prepare")
                        print(canonical({"campaign_stage": "formal.prepare", "status": "waiting_parallel_preparation"}), flush=True)
                        announced = True
                    time.sleep(min(1, self.remaining()))
            self.step("formal", "prepare", lambda: self.prepared("formal"))

    def estimate_budget(self):
        pilot = read(self.args.data_dir / "pilot" / "data-manifest.json")
        formal = read(self.args.data_dir / "formal" / "data-manifest.json")
        rates = []
        terminal_measurements = []
        for path in (self.work / "logs").glob("pilot.extract*.log"):
            last_by_split = {}
            for line in path.open(errors="replace"):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("stage") == "extract" and event.get("phase") == "pilot" and event.get("input_tokens_per_second", 0) > 0:
                    last_by_split[event["split"]] = event
            for split, event in last_by_split.items():
                if event.get("seconds", 0) > 0 and event.get("input_tokens", 0) > 0:
                    terminal_measurements.append({"log": str(path), "split": split,
                        "input_tokens": int(event["input_tokens"]), "seconds": float(event["seconds"])})
        if terminal_measurements:
            rates.append(sum(item["input_tokens"] for item in terminal_measurements) /
                         sum(item["seconds"] for item in terminal_measurements))
        marker = self.runs["pilot"] / "extract-stage.json"
        feature = read(self.runs["pilot"] / "features" / "pilot" / "manifest.json")
        total_pilot_tokens = sum(pilot["input_tokens"].values())
        processed = sum(feature["progress"][split].get("newly_processed_input_tokens", 0) for split in ["fit", "dev"])
        if marker.exists() and processed >= total_pilot_tokens and read(marker).get("seconds", 0) > 0:
            rates.append(total_pilot_tokens / read(marker)["seconds"])
        require(rates and min(rates) > 0, "No measured pilot extraction throughput is available for the formal budget gate")
        rate = min(rates)
        selection = read(self.runs["pilot"] / "selection.json")
        candidates = selection["candidate_results"]
        require(len(candidates) == 7 and all(row["epochs"] >= 10 and row["elapsed_seconds"] > 0 for row in candidates),
                "All seven measured pilot candidate times are required")
        pilot_rows = sum(pilot["statistics"][split]["rows"] for split in ["fit", "dev"])
        formal_rows = sum(formal["statistics"][split]["rows"] for split in ["fit", "dev"])
        train_seconds = sum(row["elapsed_seconds"] / row["epochs"] * 45 for row in candidates) * formal_rows / pilot_rows
        train_marker = self.runs["pilot"] / "train-stage.json"
        pilot_overhead = max(0, read(train_marker).get("seconds", 0) - sum(row["elapsed_seconds"] for row in candidates)) if train_marker.exists() else 0
        train_seconds += pilot_overhead * formal_rows / pilot_rows
        formal_tokens = sum(formal["input_tokens"].values())
        external_tokens_upper_bound = 256 * 4096
        forward_seconds = (formal_tokens + external_tokens_upper_bound) / rate
        options = self.configs["formal"].get("campaign", {})
        reserve_external = float(options.get("external_preparation_reserve_seconds", 1800))
        reserve_tests_report = float(options.get("test_reporting_reserve_seconds", 900))
        reserve_dense = float(options.get("dense_reserve_seconds", 900)) if self.args.dense_diagnostics else 0
        require(min(reserve_external, reserve_tests_report, reserve_dense) >= 0, "Budget reserves cannot be negative")
        prediction = 1.3 * (forward_seconds + train_seconds) + reserve_external + reserve_tests_report + reserve_dense
        formal_features_path = self.runs["formal"] / "features" / "formal" / "manifest.json"
        existing = read(formal_features_path) if formal_features_path.exists() else {}
        remaining_tokens = sum(count for split, count in formal["input_tokens"].items()
                               if not existing.get("progress", {}).get(split, {}).get("complete"))
        external_features_path = self.runs["formal"] / "features" / "external" / "manifest.json"
        external_features = read(external_features_path) if external_features_path.exists() else {}
        if not external_features.get("progress", {}).get("test", {}).get("complete"):
            remaining_tokens += external_tokens_upper_bound
        unfinished = [row for row in candidates if not
                      (self.runs["formal"] / "candidates" / row["candidate_id"] / "complete.json").exists()]
        remaining_train = sum(row["elapsed_seconds"] / row["epochs"] * 45 for row in unfinished) * formal_rows / pilot_rows
        if unfinished:
            remaining_train += pilot_overhead * formal_rows / pilot_rows
        remaining_external_reserve = reserve_external if not (self.args.data_dir / "external" / "manifest.json").exists() else 0
        report_complete = (self.runs["formal"] / "report-status.json").exists() and not read(self.runs["formal"] / "report-status.json")["missing"]
        plots_complete = (self.runs["formal"] / "plots" / "test" / "index.json").exists()
        evals_complete = all((self.runs["formal"] / "evaluations" / f"{split}.json").exists() for split in ["dev", "test", "external"])
        remaining_reporting_reserve = 0 if report_complete and plots_complete and evals_complete else reserve_tests_report
        remaining_dense_reserve = 0 if (self.runs["formal"] / "plots" / "dense" / "index.json").exists() else reserve_dense
        remaining_prediction = 1.3 * (remaining_tokens / rate + remaining_train) + remaining_external_reserve + remaining_reporting_reserve + remaining_dense_reserve
        estimate = {"created_unix": time.time(), "pilot_extraction_input_tokens_per_second": rate,
                    "throughput_rule": "slower of aggregate terminal cumulative progress throughput and complete-invocation wall-clock throughput; early warmup snapshots are not separate forecasts",
                    "terminal_progress_measurements": terminal_measurements,
                    "pilot_tokens": total_pilot_tokens, "formal_input_tokens_by_split": formal["input_tokens"],
                    "formal_input_tokens": formal_tokens, "external_input_token_upper_bound": external_tokens_upper_bound,
                    "pilot_candidate_seconds": {r["candidate_id"]: {"seconds": r["elapsed_seconds"], "epochs": r["epochs"]} for r in candidates},
                    "formal_training_row_scale": formal_rows / pilot_rows, "formal_max_epochs": 45,
                    "pilot_training_overhead_seconds": pilot_overhead,
                    "predicted_forward_seconds": forward_seconds, "predicted_training_seconds": train_seconds,
                    "safety_factor": 1.3, "external_preparation_reserve_seconds": reserve_external,
                    "test_reporting_reserve_seconds": reserve_tests_report, "dense_reserve_seconds": reserve_dense,
                    "predicted_total_seconds": prediction, "predicted_remaining_seconds": remaining_prediction,
                    "remaining_forward_input_tokens": remaining_tokens, "remaining_training_candidates": [row["candidate_id"] for row in unfinished],
                    "remaining_seconds": self.remaining(),
                    "formal_manifest_sha256": sha(self.args.data_dir / "formal" / "data-manifest.json"),
                    "fits": remaining_prediction == 0 or remaining_prediction <= self.remaining(), "quota_reduction_permitted": False}
        atomic_json(self.work / "formal-budget-estimate.json", estimate)
        if not estimate["fits"]:
            raise CampaignStop("budget_insufficient", "Conservative full-quota formal prediction exceeds the remaining authorized time", 21)
        return estimate

    def run(self):
        if self.remaining() <= 0:
            raise CampaignStop("deadline_exceeded", "The authorized execution window expired", 22)
        if self.args.prepare_formal_only:
            self.prepare_formal()
            self.publish(status="formal_prepared", stage=None,
                         audit_gate=str(self.args.data_dir / "formal" / "article-audit.json"))
            return
        require(self.prepared("pilot") is not None, "Prepare pilot data before launching the GPU campaign")
        self.audit_gate("pilot")
        require(self.preflight("pilot") is not None, "Run the longest-input GPU preflight before launching the campaign")
        self.step("pilot", "scan", lambda: self.selected("pilot"))
        self.step("pilot", "extract", lambda: self.features("pilot"))
        self.step("pilot", "train", lambda: self.trained("pilot"))
        self.step("pilot", "freeze", lambda: self.frozen("pilot"))
        self.step("pilot", "evaluate", lambda: self.evaluation("pilot", "dev"), "dev")
        self.step("pilot", "report", lambda: self.reported("pilot"))
        self.step("pilot", "plot", lambda: self.plotted("pilot", "dev"), "dev")
        self.prepare_formal()
        self.audit_gate("formal")
        self.inherit_pilot()
        # This gate reads article manifests and pilot measurements, never final
        # TEST predictions. The full quotas remain fixed when it refuses work.
        gate = self.work / "formal-budget-accepted.json"
        if gate.exists():
            accepted = read(gate)
            require(accepted["formal_manifest_sha256"] == sha(self.args.data_dir / "formal" / "data-manifest.json"),
                    "Accepted formal budget belongs to different prepared data")
            self.estimate_budget()
        else:
            atomic_json(gate, self.estimate_budget())
        self.external_protocol_gate()
        self.step("formal", "external-protocol", self.external_protocol)
        self.step("formal", "extract", lambda: self.features("formal"))
        self.step("formal", "train", lambda: self.trained("formal"))
        self.step("formal", "freeze", lambda: self.frozen("formal"))
        self.step("formal", "evaluate", lambda: self.evaluation("formal", "dev"), "dev")
        self.step("formal", "extract", lambda: self.features("formal", ("test",)), "test")
        self.step("formal", "evaluate", lambda: self.evaluation("formal", "test"), "test")
        self.legacy_hash_gate()
        self.step("formal", "external-prepare", self.external_prepared)
        self.step("formal", "extract", lambda: self.features("formal", ("test",), external=True), "external")
        self.step("formal", "evaluate", lambda: self.evaluation("formal", "external"), "external")
        self.step("formal", "plot", lambda: self.plotted("formal", "test"), "test")
        if self.args.dense_diagnostics:
            command = self.worker_command()
            self.step("formal", "dense", self.dense_completed, command=command)
        self.step("formal", "report", lambda: self.reported("formal"))
        self.publish(status="complete", stage=None, formal_freeze_sha256=sha(self.runs["formal"] / "freeze.json"))

    def worker_command(self):
        result = [str(self.args.python), "-u", str(Path(__file__).resolve()), "--worker", "dense"]
        for flag in ["python", "code_dir", "model_dir", "data_dir", "run_root", "cache_dir", "pilot_config", "formal_config", "devices"]:
            result += ["--" + flag.replace("_", "-"), str(getattr(self.args, flag))]
        for flag in ["pilot_run_dir", "formal_run_dir"]:
            if getattr(self.args, flag):
                result += ["--" + flag.replace("_", "-"), str(getattr(self.args, flag))]
        return result

    def dense_completed(self):
        root = self.runs["formal"] / "plots" / "dense"
        path = root / "index.json"
        if not path.exists():
            return None
        value = read(path)
        require(value["probe_sha256"] == read(self.runs["formal"] / "freeze.json")["probe_sha256"] and
                value["test_inputs_sha256"] == sha(self.args.data_dir / "formal" / "inputs-test.jsonl"),
                "Dense diagnostic binding changed")
        require(len(value["examples"]) == 15 and value["formal_scoring"] is False, "Dense diagnostic coverage is incomplete")
        paths = [path]
        for entry in value["examples"]:
            paths += [bounded_path(root, entry[key]) for key in ["png", "pdf", "json"]]
        return files_proof(paths)


def run_dense_diagnostics(campaign):
    """Optional independent worker: five fixed templates × three formats."""
    from transformers import AutoTokenizer
    import torch
    from synthetic_agent_classifier import load_probe
    from synthetic_agent_data import iter_records
    from synthetic_agent_gpu import load_model
    from synthetic_agent_plots import dense_diagnostic
    require(campaign.frozen("formal") is not None and campaign.evaluation("formal", "test") is not None,
            "Dense diagnostics require a frozen classifier and completed formal TEST")
    root = campaign.runs["formal"]
    output = root / "plots" / "dense"
    output.mkdir(parents=True, exist_ok=True)
    wanted = {"SD" + x for x in ["UA", "UCA", "UCQ", "UQTCA", "UQTA"]}
    chosen = {}
    for record in iter_records(campaign.args.data_dir, "formal", "test"):
        key = record["template_id"]
        if record["profile"] == "one_round" and record["snapshot_stage"] == "output" and key in wanted:
            if key not in chosen or record["id"] < chosen[key]["id"]:
                chosen[key] = record
    require(set(chosen) == wanted, "The fixed five diagnostic templates are unavailable")
    probe = load_probe(root / "best.npz")
    tokenizer = AutoTokenizer.from_pretrained(campaign.args.model_dir, local_files_only=True)
    model = load_model(campaign.args.model_dir, root, campaign.args.devices, campaign.args.cache_dir)
    index = {"formal_scoring": False, "probe_sha256": sha(root / "best.npz"),
             "test_inputs_sha256": sha(campaign.args.data_dir / "formal" / "inputs-test.jsonl"),
             "selection": "lexicographically first input ID in each of the five output templates; independent of scores",
             "note": "Changed-format probabilities refer to original semantic roles and are not formal accuracy.", "examples": []}
    try:
        for template, record in sorted(chosen.items()):
            for variant in ["native", "all_user", "no_tags"]:
                if campaign.remaining() <= 0:
                    raise CampaignStop("deadline_exceeded", "Dense diagnostic reached the authorized deadline", 22)
                dense_diagnostic(record, model, probe, tokenizer, output, variant)
                stem = variant + "-" + hashlib.sha256(record["id"].encode()).hexdigest()[:10]
                index["examples"].append({"template_id": template, "record_id": record["id"], "variant": variant,
                                          "png": stem + ".png", "pdf": stem + ".pdf", "json": stem + ".json"})
        atomic_json(output / "index.json", index)
    finally:
        del model
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["python", "code-dir", "model-dir", "data-dir", "run-root", "cache-dir", "pilot-config", "formal-config"]:
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--pilot-run-dir", type=Path)
    parser.add_argument("--formal-run-dir", type=Path)
    parser.add_argument("--prepare-formal-only", action="store_true")
    parser.add_argument("--dense-diagnostics", action="store_true")
    parser.add_argument("--worker", choices=["dense"], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # A venv's bin/python is often a symlink. Resolving that symlink would
    # silently discard the virtual environment and its pinned dependencies.
    args.python = Path(os.path.abspath(args.python))
    for name in ["code_dir", "model_dir", "data_dir", "run_root", "cache_dir", "pilot_config", "formal_config"]:
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    campaign = None

    def interrupted(signum, frame):
        raise CampaignStop("interrupted", f"Campaign received signal {signum}", 130)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        campaign = Campaign(args)
        if args.worker:
            run_dense_diagnostics(campaign)
            return 0
        # Formal CPU preparation has a separate lock and can run alongside the
        # pilot GPU campaign without launching a competing model process.
        lock_name = "formal-prepare-driver.lock" if args.prepare_formal_only else "campaign.lock"
        with (campaign.work / lock_name).open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise CampaignStop("already_running", f"Another process owns {lock_name}", 24)
            campaign.run()
        return 0
    except BaseException as error:
        state = error.status if isinstance(error, CampaignStop) else "failed"
        if campaign is not None and state == "already_running":
            atomic_json(campaign.work / "last-rejected-start.json", {"status": state, "error": str(error), "updated_unix": time.time()})
        elif campaign is not None:
            campaign.publish(status=state, stage=campaign.stage, error=str(error))
        else:
            filename = "formal-prepare-status.json" if args.prepare_formal_only else "status.json"
            atomic_json(args.run_root / "campaign" / filename, {"status": state, "error": str(error), "updated_unix": time.time()})
        print(canonical({"campaign_status": state, "error": str(error)}), flush=True)
        return error.code if isinstance(error, CampaignStop) else 1


if __name__ == "__main__":
    raise SystemExit(main())
