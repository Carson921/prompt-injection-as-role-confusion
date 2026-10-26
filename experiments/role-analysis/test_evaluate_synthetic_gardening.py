"""CPU-only public-source/alignment and committed-checkpoint safety tests."""
from __future__ import annotations

import copy
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_synthetic_gardening as g
import synthetic_agent_classifier as classifier
import synthetic_agent_gpu as gpu


def dump(path, value):
    g.atomic_json(path, value)


class CheckpointFixture:
    """Real NPZ consumer/schema, no torch, model, article cache, or TEST input."""
    def __init__(self, root):
        self.root = Path(root)
        self.model_binding = {"repository": "fixture/no-model", "revision": "0" * 40,
                              "identity_sha256": "1" * 64}
        self.head = {"heads": list(range(16)), "signature": {"model_binding": self.model_binding,
                   "pilot_fit_sha256": "2" * 64, "implementation_sha256": g.sha(gpu.__file__)}}
        for phase in ("pilot", "formal"):
            self.make_run(phase)

    def make_run(self, phase):
        root = self.root / phase
        root.mkdir()
        config = {"phase": phase, "classifier": {"min_epochs": 10, "max_epochs": 15 if phase == "pilot" else 45},
                  "budget": {"started_utc": "2026-10-26T13:19:17Z", "deadline_utc": "2026-10-27T13:19:17Z", "authorized_hours": 24}}
        dump(root / "config.json", config)
        dump(root / "head-selection.json", self.head)
        options = classifier._training_options(config)
        signature = {"model_binding": self.model_binding, "heads": list(range(16)), "capture": "pre-o_proj",
                     "config_sha256": gpu.object_digest(config), "selection_sha256": g.sha(root / "head-selection.json"),
                     "pilot_fit_sha256": "2" * 64, "phase": phase, "implementation_sha256": g.sha(gpu.__file__)}
        m = {"format_version": classifier.FEATURE_FORMAT, "roles": list(g.ROLES), "feature_dim": 1024,
             "feature_position": "pre-o_proj", "heads": list(range(16)), "model_binding": self.model_binding,
             "selection_sha256": signature["selection_sha256"], "pilot_fit_sha256": "2" * 64,
             "data_hash": "4" * 64, "config_hash": signature["config_sha256"], "signature": signature,
             "split_sources": {"fit": "5" * 64, "dev": "6" * 64}, "splits": {}, "progress": {}}
        feature = root / "features" / phase
        for split in ("fit", "dev"):
            path = feature / (split + "-000")
            dump(path / "complete.json", {"signature": {**signature, "split": split,
                 "input_sha256": m["split_sources"][split]}, "rows": 1, "files": {}})
            m["splits"][split] = [{"path": path.name, "rows": 1, "sha256": g.sha(path / "complete.json")}]
            m["progress"][split] = {"complete": True, "records": 1, "rows": 1}
        dump(feature / "manifest.json", m)
        hashes = {split: g.feature_identity(feature, split)[0] for split in ("fit", "dev")}
        binding = {k: m[k] for k in ("roles", "feature_dim", "heads", "feature_position", "model_binding",
                                     "selection_sha256", "pilot_fit_sha256", "data_hash", "config_hash")}
        training_hash = hashlib.sha256(g.canonical({"fit_features": hashes["fit"], "dev_features": hashes["dev"],
            "options": options, "implementation_sha256": g.sha(classifier.__file__), "candidates": classifier.CANDIDATES}).encode()).hexdigest()
        training = {"phase": phase, "options": options, "binding": binding, "feature_split_hashes": hashes,
                    "implementation_sha256": g.sha(classifier.__file__), "training_hash": training_hash}
        dump(root / "training.json", training)
        mean = np.full(1024, 1 if phase == "pilot" else 2, dtype=np.float32)
        scale = np.full(1024, 2 if phase == "pilot" else 3, dtype=np.float32)
        np.savez(root / "standardizer.npz", mean=mean, scale=scale, training_hash=np.array(training_hash))
        candidate = "linear-seed17"
        candidate_dir = root / "candidates" / candidate
        candidate_dir.mkdir(parents=True)
        metadata = {**binding, "format_version": classifier.PROBE_FORMAT, "candidate_id": candidate, "hidden": 0,
                    "seed": 17, "dropout": .1, "activation": "gelu_exact", "standardizer": "FIT_family_weighted_population_std",
                    "training_hash": training_hash, "best_epoch": 8, "training_device": "cuda:0", "engineering_smoke_only": False}
        np.savez(candidate_dir / "best.npz", mean=mean, scale=scale, metadata=np.array(g.canonical(metadata)),
                 **{"weight::0.weight": np.arange(6144, dtype=np.float32).reshape(6, 1024) / 100000,
                    "weight::0.bias": np.zeros(6, dtype=np.float32)})
        dump(candidate_dir / "history.json", [{"epoch": i} for i in range(1, 11)])
        dump(candidate_dir / "complete.json", {"candidate_id": candidate, "training_hash": training_hash,
             "epochs": 10, "best_epoch": 8, "probe_sha256": g.sha(candidate_dir / "best.npz"),
             "history_sha256": g.sha(candidate_dir / "history.json"), "export_max_absolute_error": 1e-6})
        if phase == "pilot":
            (root / "best.npz").write_bytes((candidate_dir / "best.npz").read_bytes())
            dump(root / "selection.json", {"selection_split": "dev", "candidate_id": candidate, "training_hash": training_hash,
                                          "selected_probe": "candidates/linear-seed17/best.npz"})
            dump(root / "freeze.json", {"selection_split": "dev", "candidate_id": candidate, "metadata": metadata,
                 "training_sha256": g.sha(root / "training.json"), "selection_sha256": g.sha(root / "selection.json"),
                 "probe_sha256": g.sha(root / "best.npz")})


class GardeningTest(unittest.TestCase):
    def test_exact_author_source_all_916_with_native_independent_parser(self):
        tokenizer_dir = Path("/tmp/role-retraining-tokenizer")
        if not (tokenizer_dir / "tokenizer.json").is_file():
            self.skipTest("Optional cached official tokenizer absent; use --stage prepare on deployment")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        samples, manifest = g.prepare_samples(tokenizer, g.HERE / "outputs/local-gptoss120b/tomato/inputs.json",
                    g.HERE / "config/tomato.yaml", g.HERE.parents[1] / "utils/chat_templates/gptoss.j2")
        self.assertEqual([len(s["input_ids"]) for s in samples], [953, 921, 917])
        self.assertEqual(manifest["semantic_support"], g.SUPPORT)
        for s in samples:
            self.assertEqual(len(s["content"]), 916)
            self.assertEqual(s["content"][-1]["token_id"], 109)
            self.assertFalse(s["content"][-1]["in_old_915_comparison"])
            self.assertEqual(sum(r["semantic_role"] == "system" for r in s["content"]), 57)
        self.assertTrue(manifest["independent_native_parser_agrees"])
        self.assertEqual(manifest["altered_boundary_lf_fusions_body_row_indices"],
                         {k: [56, 101, 184, 514, 566, 662] for k in g.FORMATS[1:]})

    def test_source_mutation_fails_before_any_tokenizer_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "inputs.json"
            p.write_bytes((g.HERE / "outputs/local-gptoss120b/tomato/inputs.json").read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "source changed"):
                g.prepare_samples(None, p, g.HERE / "config/tomato.yaml", g.HERE.parents[1] / "utils/chat_templates/gptoss.j2")

    def test_six_argmax_absent_roles_na_and_semantic_not_accuracy(self):
        sample = {"prompt_key": "proper_tags", "content": [{"label": 0}, {"label": 4}]}
        p = np.zeros((2, 6), dtype=np.float32)
        p[0, 1], p[1, 4] = 1, 1  # Explicit false positive for absent developer.
        m = g.metrics(sample, p)
        self.assertEqual(m["accuracy"], .5)
        self.assertEqual(m["per_role"]["developer"]["false_positive"], 1)
        self.assertIsNone(m["per_role"]["developer"]["recall"])
        self.assertIsNone(m["per_role"]["tool"]["f1"])
        sample["prompt_key"] = "everything_in_user_tags"
        m = g.metrics(sample, p)
        self.assertNotIn("accuracy", m)
        self.assertEqual(m["semantic_alignment_accuracy"], .5)
        self.assertEqual(m["metric_kind"], "original_semantic_alignment")

    def test_complete_both_and_separate_cpu_standardizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            with patch("synthetic_agent_gpu.model_binding", return_value=f.model_binding):
                probes, models = g.validate_pair(f.root / "pilot", f.root / "formal", "/no-model-loaded")
            self.assertEqual(models["first_formal"]["candidate_id"], "linear-seed17")
            self.assertNotEqual(models["pilot"]["checkpoint_sha256"], models["first_formal"]["checkpoint_sha256"])
            x = np.ones((3, 1024), dtype=np.float16)
            a, b = [probes[k].predict_proba(x) for k in ("pilot", "first_formal")]
            self.assertFalse(np.allclose(a, b))
            np.testing.assert_allclose(a.sum(1), 1, atol=2e-6)
            np.testing.assert_allclose(b.sum(1), 1, atol=2e-6)
            self.assertFalse((f.root / "formal/freeze.json").exists())  # First candidate, not final selection.

    def test_partial_lastpt_and_hash_tampering_refused(self):
        for field in ("missing", "epochs", "training", "history", "npz", "standardizer", "heads", "export"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                f = CheckpointFixture(tmp)
                root = f.root / "formal"
                c = root / "candidates/linear-seed17"
                if field == "missing":
                    (c / "complete.json").unlink()
                    (c / "last.pt").write_bytes(b"partial checkpoint must not qualify")
                elif field in ("epochs", "export"):
                    v = g.read(c / "complete.json")
                    v["epochs" if field == "epochs" else "export_max_absolute_error"] = 9 if field == "epochs" else .01
                    dump(c / "complete.json", v)
                elif field in ("training", "heads"):
                    p = root / ("training.json" if field == "training" else "head-selection.json")
                    v = g.read(p)
                    if field == "training":
                        v["training_hash"] = "0" * 64
                    else:
                        v["heads"] = list(reversed(v["heads"]))
                    dump(p, v)
                elif field == "history":
                    dump(c / "history.json", [{"epoch": 1}])
                elif field == "npz":
                    (c / "best.npz").write_bytes((c / "best.npz").read_bytes() + b"tampered")
                else:
                    np.savez(root / "standardizer.npz", mean=np.zeros(1024), scale=np.ones(1024), training_hash="wrong")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    g.validate_run(root, "formal")

    def test_changed_feature_commit_refused_without_opening_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            p = f.root / "formal/features/formal/fit-000/complete.json"
            p.write_text(p.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                g.validate_run(f.root / "formal", "formal")

    def test_actual_config_and_gpu_code_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            path = f.root / "formal/config.json"
            v = g.read(path)
            v["seed"] = 928513
            dump(path, v)
            with self.assertRaisesRegex(ValueError, "config hash"):
                g.validate_run(f.root / "formal", "formal")
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            wrong_code = f.root / "wrong_gpu.py"
            wrong_code.write_text("# changed GPU backend, must not be used\n")
            with patch.object(gpu, "__file__", str(wrong_code)), self.assertRaisesRegex(ValueError, "GPU implementation"):
                g.validate_run(f.root / "formal", "formal")

    def test_shared_binding_mismatch_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            with patch("synthetic_agent_gpu.model_binding", return_value={"wrong": True}), self.assertRaisesRegex(ValueError, "Actual model"):
                g.validate_pair(f.root / "pilot", f.root / "formal", "/no-model-loaded")

    def test_deadline_uses_min_absolute_and_rejects_expired_or_over24h(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            start = datetime(2026, 10, 26, 13, 19, 17, tzinfo=timezone.utc).timestamp()
            with patch.object(g.time, "time", return_value=start + 20):
                self.assertEqual(g._deadline(f.root / "pilot", f.root / "formal"), start + 86400)
            with patch.object(g.time, "time", return_value=start + 86400):
                with self.assertRaisesRegex(ValueError, "has passed"):
                    g._deadline(f.root / "pilot", f.root / "formal")
            v = g.read(f.root / "formal/config.json")
            v["budget"]["deadline_utc"] = "2026-10-28T13:19:17Z"
            dump(f.root / "formal/config.json", v)
            with self.assertRaisesRegex(ValueError, "24-hour"):
                g._deadline(f.root / "pilot", f.root / "formal")

    def test_gpu_snapshot_refuses_busy_without_any_process_mutation(self):
        lines = "\n".join(f"{i}, GPU-{i}, 46068, {free}, 10000, 30" for i, free in enumerate([30000, 30000, 1000, 30000]))
        with patch.object(g.subprocess, "run", return_value=types.SimpleNamespace(stdout=lines)) as run:
            with self.assertRaisesRegex(ValueError, "Resources unavailable"):
                g.gpu_snapshot("0,1,2,3")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], "nvidia-smi")

    def test_exactly_three_shared_forwards_and_six_cpu_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            probes = {key: g.validate_run(f.root / phase, phase)[0] for key, phase in (("pilot", "pilot"), ("first_formal", "formal"))}
            out = f.root / "diagnosis"
            out.mkdir()
            dump(out / "prepared-inputs.json", [])
            dump(out / "prepare-manifest.json", {})
            args = types.SimpleNamespace(pilot_run_dir=f.root / "pilot", formal_run_dir=f.root / "formal", model_dir="/never-loaded",
                                         output_dir=out, devices="0,1,2,3", cache_dir=None)
            samples = [{"prompt_key": key, "input_ids": list(range(920)), "content":
                        [{"token_position": i, "label": i % 6} for i in range(916)]} for key in g.FORMATS]
            class Capture:
                def __init__(self, positions, selected):
                    self.positions = positions
                def install(self, model):
                    return nullcontext()
                def matrix(self):
                    return np.ones((len(self.positions), 1024), dtype=np.float16)
            fake_torch = types.SimpleNamespace(inference_mode=nullcontext, isfinite=np.isfinite)
            with patch.object(g, "validate_pair", return_value=(probes, {"pilot": {}, "first_formal": {}})), \
                 patch.object(g, "_deadline", return_value=g.time.time() + 60), \
                 patch.object(g, "gpu_snapshot", return_value={"diagnostic_pid": 123}), \
                 patch("synthetic_agent_gpu.load_model", return_value=object()) as load, \
                 patch("synthetic_agent_gpu.HeadCapture", Capture), \
                 patch("synthetic_agent_gpu._forward", return_value=np.zeros((1, 920, 2))) as forward, \
                 patch.dict("sys.modules", {"torch": fake_torch}):
                result = g.infer(args, samples, {"source_binding": {}})
            self.assertEqual(load.call_count, 1)
            self.assertEqual(forward.call_count, 3)
            self.assertEqual(result["forward_count"], 3)
            self.assertEqual(len(list(out.glob("*-probabilities.npy"))), 6)
            self.assertNotEqual(result["formats"]["proper_tags"]["models"]["pilot"]["probabilities_sha256"],
                                result["formats"]["proper_tags"]["models"]["first_formal"]["probabilities_sha256"])

    def test_incomplete_formal_gate_precedes_any_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CheckpointFixture(tmp)
            (f.root / "formal/candidates/linear-seed17/complete.json").unlink()
            args = types.SimpleNamespace(pilot_run_dir=f.root / "pilot", formal_run_dir=f.root / "formal", model_dir="/never-loaded")
            with patch("synthetic_agent_gpu.load_model") as load:
                with self.assertRaises(FileNotFoundError):
                    g.infer(args, [], {})
                load.assert_not_called()

    def test_report_requires_all_seven_pairs_and_exact_figure_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            figures = {}
            stems = ["gardening-native-comparison"] + [f"gardening-{model}-{key}" for model in ("pilot", "first_formal") for key in g.FORMATS]
            for stem in stems:
                files = {}
                for kind, data in (("png", b"\x89PNG\r\n\x1a\nfixture"), ("pdf", b"%PDF-1.4\nfixture")):
                    p = out / (stem + "." + kind)
                    p.write_bytes(data)
                    files[kind] = {"path": p.name, "sha256": g.sha(p)}
                if stem == "gardening-native-comparison":
                    key, models = "proper_tags", ["pilot", "first_formal"]
                else:
                    model = "first_formal" if "first_formal" in stem else "pilot"
                    key, models = stem[len("gardening-" + model + "-"):], [model]
                figures[stem] = {"files": files, "format": key, "models": models, "complete_context": True,
                                 "body_tokens_per_model": 916, "probability_rows": 6}
            index = {"format_version": "synthetic-gardening-plots-v1", "figures": figures}
            dump(out / "index.json", index)
            self.assertEqual(len(g.validate_figures(index, out)), 14)
            bad = copy.deepcopy(index)
            bad["figures"].pop(stems[-1])
            dump(out / "index.json", bad)
            with self.assertRaisesRegex(ValueError, "seven"):
                g.validate_figures(bad, out)
            dump(out / "index.json", index)
            p.write_bytes(p.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                g.validate_figures(index, out)


if __name__ == "__main__":
    unittest.main()
