"""CPU-only delivery tests with independent formal-shaped fixtures.

The fixture uses real six-role NPZ readers, FeatureBank, training option helpers,
and Campaign.publish. It does not run training, a backbone, a real export, or
read any experiment article, TEST input, or remote file. All outputs are /tmp.
"""
from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import export_synthetic_artifacts as exporter
import run_synthetic_agent_campaign as campaign
import synthetic_agent_classifier as classifier

HERE = Path(__file__).resolve().parent
PRIVATE = "PRIVATE_FIXTURE_DO_NOT_EXPORT_928513"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def image(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nCPU-only-independent-figure" if path.suffix == ".png"
                     else b"%PDF-1.4\n% CPU-only-independent-figure\n%%EOF\n")


def interval_tree(overall):
    keys = ["accuracy", "macro_f1", "nll"] + [f"{role}.{metric}" for role in classifier.ROLES
                                              for metric in ("precision", "recall", "f1")]
    intervals = {key: {"lower": 0.0, "upper": 2.0 if key == "nll" else 1.0,
                       "valid_replicates": 2000} for key in keys}
    for role in classifier.ROLES:
        for metric in ("precision", "recall", "f1"):
            if overall["roles"][role][metric] is None:
                intervals[f"{role}.{metric}"] = None
    return {"replicates": 2000, "unit": "source_cluster", "confidence": .95,
            "clusters": 12, "method": "percentile", "seed": 20261026,
            "intervals": intervals}


class DeliveryFixture:
    """A complete, deliberately non-scientific source tree for exporter tests."""

    def __init__(self, parent):
        self.parent = Path(parent)
        self.code, self.data, self.runs = [self.parent / name for name in ("code", "data", "runs")]
        self.run = self.runs / "formal"
        self.run.mkdir(parents=True)
        self.code.mkdir()
        for name in exporter.CORE_CODE:
            shutil.copyfile(HERE / name, self.code / name)
        self.requirements = self.code / "requirements-synthetic-agent-4xa40.txt"
        self.requirements.write_text("numpy==2.2.6\nscipy==1.17.1\n")
        self.configs = {}
        for phase, filename in zip(("pilot", "formal"), exporter.CORE_CONFIG):
            config = {"experiment": "independent-export-unit-fixture", "phase": phase,
                "roles": list(classifier.ROLES), "seed": 20261026,
                "budget": {"started_utc": "2026-10-26T13:19:17Z", "deadline_utc": "2026-10-27T13:19:17Z",
                           "authorized_hours": 24},
                "classifier": {"max_epochs": 15 if phase == "pilot" else 45, "min_epochs": 10}}
            self.configs[phase] = config
            write_json(self.code / filename, config)
        write_json(self.run / "config.json", self.configs["formal"])
        self.source_keys = {}
        for phase, quotas in (("pilot", {"fit": 512, "dev": 128}),
                              ("formal", {"fit": 4096, "dev": 512, "test": 1024})):
            folder = self.data / phase
            folder.mkdir(parents=True)
            keys = []
            for split, count in quotas.items():
                for i in range(count):
                    sid = f"c4-{dict(fit=0,dev=1000000,test=2000000)[split] + i:024x}"
                    keys.append({"source_id": sid, "cluster_id": digest(sid), "split": split,
                                 "normalized_sha256": digest(sid), "source_ordinal": i,
                                 "normalized_url": f"https://public-fixture.invalid/{split}/{i}"})
                (folder / f"inputs-{split}.jsonl").write_text(json.dumps({"input_ids": [1, 2], "text": PRIVATE}) + "\n")
            self.source_keys[phase] = keys
            write_json(folder / "source-keys.json", keys)
            (folder / "sources.jsonl").write_text(json.dumps({"text": PRIVATE}) + "\n")
            write_json(folder / "source-provenance.json", {"audit_examples": [{"text": PRIVATE}]})
            (folder / "mask-audit-fit.jsonl").write_text(json.dumps({"body": PRIVATE}) + "\n")
            hashes = {name: exporter.sha(folder / name) for name in ["source-keys.json", "sources.jsonl", "source-provenance.json"]}
            hashes.update({f"inputs-{split}.jsonl": exporter.sha(folder / f"inputs-{split}.jsonl") for split in quotas})
            manifest = {"format_version": "synthetic-agent-roles-v1", "phase": phase,
                        "roles": list(classifier.ROLES), "sha256": hashes,
                        "statistics": {split: {"sources": count} for split, count in quotas.items()}}
            write_json(folder / "data-manifest.json", manifest)
            write_json(folder / "article-audit.json", {"status": "passed",
                "data_manifest_sha256": exporter.sha(folder / "data-manifest.json"),
                "sources_sha256": hashes["sources.jsonl"], "scope": "independent CPU fixture metadata only"})
        shutil.copyfile(self.data / "formal/data-manifest.json", self.run / "data-formal-manifest.json")
        self.model_binding = {"repository": "independent-fixture/no-model", "revision": "0" * 40,
                              "assets_sha256": {"tokenizer.json": digest("independent tokenizer fixture")}}
        heads = {"roles": list(classifier.ROLES), "heads": list(range(16)),
                 "signature": {"model_binding": self.model_binding,
                               "pilot_fit_sha256": exporter.sha(self.data / "pilot/inputs-fit.jsonl"),
                               "implementation_sha256": exporter.sha(self.code / "synthetic_agent_gpu.py")},
                 "no_dev_test_or_historical_probe_used": True}
        write_json(self.run / "head-selection.json", heads)
        self.binding = {"roles": list(classifier.ROLES), "feature_dim": 1024, "feature_position": "pre-o_proj",
                        "heads": list(range(16)), "model_binding": self.model_binding,
                        "selection_sha256": exporter.sha(self.run / "head-selection.json"),
                        "pilot_fit_sha256": heads["signature"]["pilot_fit_sha256"],
                        "data_hash": digest("independent feature inputs"), "config_hash": exporter.object_sha(self.configs["formal"])}
        self.banks = {}
        self._features("formal", {"fit": 24, "dev": 12, "test": 12})
        self._features("external", {"test": 12})
        options = classifier._training_options(self.configs["formal"])
        self.training_hash = exporter.object_sha({
            "fit_features": self.banks[("formal", "fit")].split_identity_hash,
            "dev_features": self.banks[("formal", "dev")].split_identity_hash,
            "options": options, "implementation_sha256": exporter.sha(self.code / "synthetic_agent_classifier.py"),
            "candidates": classifier.CANDIDATES})
        training = {"phase": "formal", "implementation_sha256": exporter.sha(self.code / "synthetic_agent_classifier.py"),
                    "options": options, "training_hash": self.training_hash, "binding": self.binding,
                    "feature_split_hashes": {split: self.banks[("formal", split)].split_identity_hash for split in ("fit", "dev")}}
        write_json(self.run / "training.json", training)
        mean, scale = np.zeros(1024, dtype=np.float32), np.ones(1024, dtype=np.float32)
        np.savez(self.run / "standardizer.npz", mean=mean, scale=scale, training_hash=np.array(self.training_hash))
        dev_bank = self.banks[("formal", "dev")]
        dev = classifier.grouped_metrics(dev_bank, self._probabilities(len(dev_bank)))
        candidates = []
        for candidate_id, hidden, seed in classifier.CANDIDATES:
            folder = self.run / "candidates" / candidate_id
            folder.mkdir(parents=True)
            metadata = {**self.binding, "format_version": classifier.PROBE_FORMAT, "hidden": hidden,
                        "dropout": .1, "seed": seed, "candidate_id": candidate_id,
                        "training_hash": self.training_hash, "best_epoch": 1,
                        "activation": "gelu_exact", "training_device": "cuda:0", "engineering_smoke_only": False}
            arrays = {"mean": mean, "scale": scale, "metadata": np.array(json.dumps(metadata)),
                      "weight::0.weight": np.zeros((hidden or 6, 1024), dtype=np.float32),
                      "weight::0.bias": np.zeros(hidden or 6, dtype=np.float32)}
            if hidden:
                arrays.update({"weight::3.weight": np.zeros((6, hidden), dtype=np.float32),
                               "weight::3.bias": np.zeros(6, dtype=np.float32)})
            np.savez_compressed(folder / "best.npz", **arrays)
            history = []
            for epoch in range(1, 11):
                history.append({"epoch": epoch, "optimizer_step": epoch, "improved": epoch == 1,
                    "train_loss_for_processed_batches": 2.0 / epoch,
                    "processed_rows_this_process": 24, "sampled_rows_by_profile": {"single": 5, "one_round": 7, "multi_round": 12},
                    "sampled_profile_proportions": {"single": 5 / 24, "one_round": 7 / 24, "multi_round": .5},
                    "cumulative_sampled_rows_by_profile": {"single": 5 * epoch, "one_round": 7 * epoch, "multi_round": 12 * epoch},
                    "dev": copy.deepcopy(dev), "debug_payload": {"text": PRIVATE}})
            write_json(folder / "history.json", history)
            (folder / "last.pt").write_text(PRIVATE)
            candidate = {"candidate_id": candidate_id, "training_hash": self.training_hash, "best_epoch": 1,
                         "epochs": 10, "elapsed_seconds": 12.345, "dev": dev,
                         "probe_sha256": exporter.sha(folder / "best.npz"),
                         "history_sha256": exporter.sha(folder / "history.json"), "export_max_absolute_error": 0.0,
                         "sampled_rows_by_profile": {"single": 50, "one_round": 70, "multi_round": 120}}
            write_json(folder / "complete.json", candidate)
            candidates.append(candidate)
        winner = min(candidates, key=lambda item: classifier.selection_key(item["dev"], item["candidate_id"]))
        selection = {"selection_split": "dev", "training_hash": self.training_hash,
                     "candidate_id": winner["candidate_id"], "candidate_results": candidates}
        write_json(self.run / "selection.json", selection)
        shutil.copyfile(self.run / "candidates" / winner["candidate_id"] / "best.npz", self.run / "best.npz")
        self.probe = classifier.load_probe(self.run / "best.npz")
        external_root = self.data / "external"
        write_json(external_root / "history-index.json", {"body_sha256": [digest("historical hash fixture")]})
        protocol = {"declared_at_unix": 1, "history_index_sha256": exporter.sha(external_root / "history-index.json"),
                    "implementation_sha256": exporter.sha(self.code / "synthetic_agent_external.py"), "legacy_csv": []}
        write_json(external_root / "protocol.json", protocol)
        write_json(self.run / "external-protocol.json", {"protocol_sha256": exporter.sha(external_root / "protocol.json"),
                    "history_index_sha256": protocol["history_index_sha256"], "new_external_bodies_read": False})
        frozen = {"probe_sha256": exporter.sha(self.run / "best.npz"),
                  "selection_sha256": exporter.sha(self.run / "selection.json"),
                  "training_sha256": exporter.sha(self.run / "training.json"), "selection_split": "dev",
                  "metadata": self.probe.metadata, "frozen_at_unix": 2,
                  "external_protocol_sha256": exporter.sha(external_root / "protocol.json"),
                  "external_protocol_pointer_sha256": exporter.sha(self.run / "external-protocol.json"),
                  "external_history_index_sha256": protocol["history_index_sha256"]}
        write_json(self.run / "freeze.json", frozen)
        external = {"created_at_unix": 3, "probe_sha256": frozen["probe_sha256"],
                    "freeze_sha256": exporter.sha(self.run / "freeze.json"),
                    "protocol_sha256": frozen["external_protocol_sha256"], "counts_by_source": {"oasst": 128, "toxicchat": 128},
                    "inputs": "inputs-test.jsonl", "inputs_sha256": exporter.sha(external_root / "inputs-test.jsonl")}
        write_json(external_root / "manifest.json", external)
        write_json(self.run / "external-preparation.json", external)
        write_json(external_root / "legacy-hashes.json", {"binding": {
            "protocol_sha256": frozen["external_protocol_sha256"], "freeze_sha256": exporter.sha(self.run / "freeze.json"),
            "probe_sha256": frozen["probe_sha256"], "history_index_sha256": protocol["history_index_sha256"],
            "implementation_sha256": exporter.sha(self.code / "synthetic_agent_external.py")}})
        for split in ("dev", "test", "external"):
            bank = self.banks[("external" if split == "external" else "formal", "test" if split == "external" else split)]
            path = self.run / "evaluations" / f"{split}.json"
            path.parent.mkdir(exist_ok=True)
            probabilities = self._probabilities(len(bank))
            np.save(path.with_name(f"{split}-probabilities.npy"), probabilities, allow_pickle=False)
            evaluation = {"split": "test" if split == "external" else split, "probe_sha256": frozen["probe_sha256"],
                          "overall": classifier.metrics(probabilities, bank.y)}
            if split != "dev":
                evaluation["confidence_intervals"] = interval_tree(evaluation["overall"])
            write_json(path, evaluation)
            write_json(path.with_name(f"{split}-positions.json"), {
                "probe_sha256": frozen["probe_sha256"], "freeze_sha256": exporter.sha(self.run / "freeze.json"),
                "evaluation_sha256": exporter.sha(path), "probabilities_sha256": exporter.sha(path.with_name(f"{split}-probabilities.npy")),
                "selection_permitted": False})
            for extension in ("png", "pdf"):
                image(self.run / f"{split}-confusion.{extension}")
        self._plots(frozen["probe_sha256"])
        for name, value in {
            "preflight.json": {"status": "passed", "maximum_input_tokens": 4096, "model_binding": self.model_binding},
            "model_backend.json": {"model_binding": self.model_binding, "expert_backends": ["Mxfp4GptOssExperts"]},
            "resume-validation.json": {"passed": True, "device": "cuda:0", "weight_max_absolute_error": 0.0,
                                       "loss_absolute_error": 0.0, "numpy_export_max_absolute_error": 0.0},
            "pilot-inheritance.json": {"formal_head_rescan": False},
            "report-status.json": {"phase": "formal", "missing": []},
            "status.json": {"stage": "report", "phase": "formal", "status": "complete"},
        }.items():
            write_json(self.run / name, value)
        (self.run / "REPORT.zh-CN.md").write_text("# Independent CPU export fixture\n")
        for extension in ("png", "pdf"):
            image(self.run / f"training-curves.{extension}")
        # Use the actual producer, not an invented dict: this guards the
        # freeze-file-SHA versus probe-SHA distinction in the completion schema.
        args = campaign.parse_args(["--python", sys.executable, "--code-dir", str(self.code),
            "--model-dir", str(self.parent / "fixture-model"), "--data-dir", str(self.data),
            "--run-root", str(self.runs), "--cache-dir", str(self.parent / "fixture-cache"),
            "--pilot-config", str(self.code / exporter.CORE_CONFIG[0]),
            "--formal-config", str(self.code / exporter.CORE_CONFIG[1])])
        producer = campaign.Campaign(args)
        producer.publish(status="complete", stage=None, formal_freeze_sha256=exporter.sha(self.run / "freeze.json"))
        self.producer = producer
        self.campaign_root = self.runs / "campaign"
        for name in ("formal-budget-accepted.json", "formal-budget-estimate.json"):
            write_json(self.campaign_root / name, {"fits": True, "quota_reduction_permitted": False})
        (self.campaign_root / "driver.log").write_text("completed independent CPU delivery fixture\n")
        for name in exporter.RECEIPTS:
            proof = {"independent_fixture": True}
            execution = {"adopted_verified_atomic_artifacts": True}
            if name.startswith("formal."):
                log = self.campaign_root / "logs" / f"{name}.001.log"
                log.parent.mkdir(exist_ok=True)
                log.write_text("completed stage in independent CPU fixture\n")
                execution = {"log": str(log), "seconds": 1.0}
            write_json(self.campaign_root / "receipts" / f"{name}.json", {
                "name": name, "binding": producer.binding, "proof": proof,
                "proof_sha256": exporter.object_sha(proof), "completed_unix": 4, "execution": execution})
        (self.campaign_root / "logs/formal.debug.log").write_text(PRIVATE)
        write_json(self.run / "evaluations/debug-inputs.json", {"text": PRIVATE})
        self.args = argparse.Namespace(run_dir=self.run, data_dir=self.data, code_dir=self.code,
            output_dir=self.parent / "delivery", requirements_file=self.requirements, environment_file=None)

    @staticmethod
    def _probabilities(count):
        return np.full((count, 6), 1 / 6, dtype=np.float32)

    def _features(self, phase, splits):
        root = self.run / "features" / phase
        root.mkdir(parents=True)
        signature = {"model_binding": self.model_binding, "heads": self.binding["heads"],
            "selection_sha256": self.binding["selection_sha256"], "pilot_fit_sha256": self.binding["pilot_fit_sha256"],
            "config_sha256": self.binding["config_hash"], "capture": "pre-o_proj", "phase": phase,
            "implementation_sha256": exporter.sha(self.code / "synthetic_agent_gpu.py")}
        manifest = {"format_version": classifier.FEATURE_FORMAT, **self.binding,
                    "signature": signature, "splits": {}, "split_sources": {}, "progress": {}}
        for split, count in splits.items():
            input_path = self.data / phase / f"inputs-{split}.jsonl"
            if not input_path.exists():
                input_path.parent.mkdir(parents=True, exist_ok=True)
                input_path.write_text(json.dumps({"input_ids": [1], "text": PRIVATE}) + "\n")
            input_sha = exporter.sha(input_path)
            folder = root / f"{split}-000000"
            folder.mkdir()
            y = np.where(np.arange(count) % 2 == 0, 2, 4).astype(np.int64) if phase == "external" else np.arange(count, dtype=np.int64) % 6
            arrays = {"x": np.zeros((count, 1024), dtype=np.float16), "y": y,
                "source": np.array([f"{phase}-{split}-cluster-{i}" for i in range(count)]),
                "family": np.array([f"{phase}-{split}-family-{i}" for i in range(count)]),
                "structure": np.array([f"{classifier.PROFILE_ORDER[i % 3]}:fixture" for i in range(count)]),
                "weight": np.ones(count, dtype=np.float32), "groups": np.array(["{}"] * count)}
            files = {}
            for key, value in arrays.items():
                np.save(folder / f"{key}.npy", value, allow_pickle=False)
                files[key + ".npy"] = exporter.sha(folder / f"{key}.npy")
            write_json(folder / "complete.json", {"rows": count, "files": files,
                "signature": {**signature, "split": split, "input_sha256": input_sha}})
            manifest["splits"][split] = [{"path": folder.name, "rows": count, "sha256": exporter.sha(folder / "complete.json")}]
            manifest["split_sources"][split] = input_sha
            manifest["progress"][split] = {"complete": True, "rows": count, "records": count}
        write_json(root / "manifest.json", manifest)
        for split in splits:
            self.banks[(phase, split)] = classifier.FeatureBank(root, split)

    def _plots(self, probe_sha):
        normal_root = self.run / "plots/test"
        normal = {"phase": "formal", "split": "test", "probe_sha256": probe_sha,
            "scoring": "sampled article-body mask only; no interpolation across unsampled tokens",
            "selection": "fixed structural categories; lexicographically first cluster/input ID, independent of predictions",
            "examples": {}, "missing_coverage": [],
            "call_chain_note": "Stages contain different target bodies; their overall scores do not estimate the causal effect of deleting CoT."}
        for i, category in enumerate(sorted(exporter.NORMAL_CATEGORIES)):
            profile, stage, policy = "multi_round", "output", "retain"
            if category.startswith("single_"):
                profile = "single"
            elif category.startswith("one_round_"):
                profile = "one_round"
            elif category == "history_cot_drop_completed_round_cot":
                policy = "drop_completed_round_cot"
            elif category == "call_chain_pre_tool_call":
                stage, policy = "pre_tool_call", "prior_call_cot_visible"
            elif category == "call_chain_tool_reply":
                stage, policy = "tool_reply", "drop_prior_call_cot"
            elif category == "call_chain_output":
                policy = "drop_prior_call_cot"
            record_id = f"c4-{i + 1:024x}/{profile}/{stage}/{policy}"
            stem = category + "-" + hashlib.sha256(record_id.encode()).hexdigest()[:10]
            entry = {"record_id": record_id, "png": stem + ".png", "pdf": stem + ".pdf",
                     "scored_tokens": 6, "input_tokens": 16}
            normal["examples"][category] = entry
            for extension in ("png", "pdf"):
                image(normal_root / f"{stem}.{extension}")
            write_json(normal_root / f"{stem}.json", {"record_id": record_id, "category": category,
                "input_tokens": 16, "scored_positions": list(range(6)), "original_roles": list(classifier.ROLES),
                "probabilities": self._probabilities(6).tolist(), "token_offsets": [[j, j + 1] for j in range(16)],
                "probe_sha256": probe_sha})
        write_json(normal_root / "index.json", normal)
        dense_root = self.run / "plots/dense"
        dense = {"formal_scoring": False, "probe_sha256": probe_sha,
                 "test_inputs_sha256": exporter.sha(self.data / "formal/inputs-test.jsonl"),
                 "selection": "lexicographically first input ID in each of the five output templates; independent of scores",
                 "note": "Changed-format probabilities refer to original semantic roles and are not formal accuracy.", "examples": []}
        for i, template in enumerate(sorted(exporter.TEMPLATES)):
            record_id = f"dolma3-{i + 100:024x}/one_round/output/retain"
            for variant in sorted(exporter.VARIANTS):
                stem = variant + "-" + hashlib.sha256(record_id.encode()).hexdigest()[:10]
                dense["examples"].append({"template_id": template, "record_id": record_id, "variant": variant,
                                         "png": stem + ".png", "pdf": stem + ".pdf", "json": stem + ".json"})
                for extension in ("png", "pdf"):
                    image(dense_root / f"{stem}.{extension}")
                write_json(dense_root / f"{stem}.json", {"record_id": record_id, "variant": variant,
                    "formal_scoring": False, "note": exporter.DENSE_NOTE, "input_tokens": 16, "body_tokens": 6,
                    "positions": list(range(6)), "probabilities": self._probabilities(6).tolist()})
        write_json(dense_root / "index.json", dense)


class ExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-delivery-cpu-check-")
        self.addCleanup(self.temp.cleanup)
        self.fixture = DeliveryFixture(Path(self.temp.name))

    def test_complete_delivery_checksums_idempotence_and_exclusions(self):
        f = self.fixture
        # The real producer's field is the freeze FILE hash, never probe hash.
        status = exporter.read(f.campaign_root / "status.json")
        self.assertEqual(status["formal_freeze_sha256"], exporter.sha(f.run / "freeze.json"))
        self.assertNotEqual(status["formal_freeze_sha256"], exporter.sha(f.run / "best.npz"))
        first = exporter.export(f.args)
        self.assertEqual(first["status"], "exported")
        manifest = exporter.read(f.args.output_dir / "bundle-manifest.json")
        self.assertTrue(manifest["complete"])
        for line in (f.args.output_dir / "SHA256SUMS").read_text().splitlines():
            checksum, relative = line.split("  ", 1)
            self.assertEqual(exporter.sha(f.args.output_dir / relative), checksum)
        delivered = {str(path.relative_to(f.args.output_dir)) for path in f.args.output_dir.rglob("*") if path.is_file()}
        self.assertFalse(any(path.endswith((".npy", ".pt", ".jsonl", "/history.json")) for path in delivered))
        self.assertFalse(any("source-provenance" in path or "debug" in path or "legacy-hashes.json" in path for path in delivered))
        for path in f.args.output_dir.rglob("*.json"):
            self.assertNotIn(PRIVATE, path.read_text())
        probe = classifier.load_probe(f.args.output_dir / "formal/best.npz")
        p = probe.predict_proba(np.zeros((3, 1024), dtype=np.float32))
        np.testing.assert_allclose(p.sum(1), 1)
        self.assertEqual(p.shape, (3, 6))
        before = exporter.sha(f.args.output_dir / "bundle-manifest.json")
        second = exporter.export(f.args)
        self.assertEqual(second["status"], "already_exported")
        self.assertEqual(before, exporter.sha(f.args.output_dir / "bundle-manifest.json"))

    def test_compact_curves_preserve_each_epoch_scalars_roles_sampling_and_raw_sha(self):
        f = self.fixture
        exporter.export(f.args)
        for candidate_id, _, _ in classifier.CANDIDATES:
            source = f.run / "candidates" / candidate_id / "history.json"
            raw = exporter.read(source)
            compact = exporter.read(f.args.output_dir / "formal/candidates" / candidate_id / "curve-scalars.json")
            self.assertTrue(compact["derived_artifact"])
            self.assertTrue(compact["not_the_raw_history"])
            self.assertEqual(compact["source_history_sha256"], exporter.sha(source))
            self.assertEqual(compact["source_history_path"], str(source.resolve()))
            self.assertEqual(len(compact["epochs"]), 10)
            for old, new in zip(raw, compact["epochs"]):
                for key in ("epoch", "optimizer_step", "improved", "train_loss_for_processed_batches"):
                    self.assertEqual(new[key], old[key])
                for role in classifier.ROLES:
                    self.assertEqual(new["dev"]["roles"][role], old["dev"]["overall"]["roles"][role])
                for key in ("sampled_rows_by_profile", "sampled_profile_proportions", "cumulative_sampled_rows_by_profile"):
                    self.assertEqual(new["sampling"][key], old[key])
                self.assertNotIn("by_structure", new["dev"])
                self.assertNotIn("debug_payload", new)

    def test_incomplete_campaign_and_probe_instead_of_freeze_hash_are_rejected(self):
        f = self.fixture
        f.producer.publish(status="running", stage="formal.train")
        with self.assertRaisesRegex(exporter.NotReady, "entire formal campaign"):
            exporter.export(f.args)
        f.producer.publish(status="complete", formal_freeze_sha256=exporter.sha(f.run / "best.npz"))
        with self.assertRaisesRegex(exporter.NotReady, "SHA changed"):
            exporter.export(f.args)
        self.assertFalse(f.args.output_dir.exists())

    def test_missing_final_ci_prevents_any_completed_directory(self):
        f = self.fixture
        path = f.run / "evaluations/test.json"
        value = exporter.read(path)
        del value["confidence_intervals"]
        write_json(path, value)
        with self.assertRaisesRegex(exporter.NotReady, "2000 source-cluster"):
            exporter.export(f.args)
        self.assertFalse(f.args.output_dir.exists())

    def test_missing_dense_coverage_is_rejected(self):
        f = self.fixture
        path = f.run / "plots/dense/index.json"
        value = exporter.read(path)
        value["examples"].pop()
        write_json(path, value)
        with self.assertRaisesRegex(exporter.NotReady, "15 fixed dense"):
            exporter.export(f.args)
        self.assertFalse(f.args.output_dir.exists())

    def test_malicious_raw_text_in_plot_sidecar_is_rejected(self):
        f = self.fixture
        index = exporter.read(f.run / "plots/test/index.json")
        entry = next(iter(index["examples"].values()))
        path = (f.run / "plots/test" / entry["png"]).with_suffix(".json")
        sidecar = exporter.read(path)
        sidecar["text"] = PRIVATE
        write_json(path, sidecar)
        with self.assertRaisesRegex(exporter.NotReady, "sidecar fields"):
            exporter.export(f.args)
        self.assertFalse(f.args.output_dir.exists())

    def test_export_uses_copyfile_without_fuse_stat_cloning(self):
        f = self.fixture
        real_copy = shutil.copyfile
        unsupported = PermissionError(errno.EPERM, "FUSE does not support metadata cloning")
        with patch.object(exporter.shutil, "copy2", side_effect=unsupported), \
             patch.object(exporter.shutil, "copystat", side_effect=unsupported), \
             patch.object(exporter.shutil, "copy", side_effect=unsupported), \
             patch.object(exporter.shutil, "copyfile", wraps=real_copy) as calls:
            self.assertEqual(exporter.export(f.args)["status"], "exported")
            self.assertGreater(calls.call_count, 50)

    def test_real_content_copy_failure_leaves_no_completed_or_partial_delivery(self):
        f = self.fixture
        real_copy = shutil.copyfile

        def fail(source, target):
            if Path(source).name == "best.npz":
                raise OSError(errno.EIO, "independent fixture content-copy failure")
            return real_copy(source, target)

        with patch.object(exporter.shutil, "copyfile", side_effect=fail):
            with self.assertRaisesRegex(OSError, "content-copy failure"):
                exporter.export(f.args)
        self.assertFalse(f.args.output_dir.exists())
        self.assertEqual(list(f.parent.glob("delivery.export-*.tmp")), [])

    def test_retained_raw_sources_and_inputs_must_match_actual_current_bytes(self):
        f = self.fixture
        # Each case changes only the live retained bytes, never the signed
        # manifest/audit. Full raw contents must remain excluded from delivery.
        for relative in ("pilot/sources.jsonl", "formal/sources.jsonl",
                         "formal/inputs-fit.jsonl", "formal/inputs-dev.jsonl",
                         "formal/inputs-test.jsonl", "external/inputs-test.jsonl"):
            with self.subTest(relative=relative):
                path = f.data / relative
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b"\nchanged-independent-fixture-bytes\xff")
                    with self.assertRaisesRegex(exporter.NotReady, "bytes differ|input bytes changed"):
                        exporter.export(f.args)
                    self.assertFalse(f.args.output_dir.exists())
                finally:
                    path.write_bytes(before)

    def test_feature_input_binding_cannot_relabel_current_raw_input(self):
        f = self.fixture
        path = f.run / "features/formal/manifest.json"
        manifest = exporter.read(path)
        # Byte provenance must match the actual retained file even when an
        # unrelated, syntactically valid SHA is placed in the feature metadata.
        manifest["split_sources"]["test"] = digest("unrelated retained inputs")
        write_json(path, manifest)
        with self.assertRaisesRegex(exporter.NotReady, "current retained input JSONL"):
            exporter.export(f.args)
        self.assertFalse(f.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
