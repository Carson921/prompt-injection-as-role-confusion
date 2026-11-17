"""CPU checks for dynamic head dimensions, scientific isolation, and recovery."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import deepseek_role_probe_train as c


def write_features(root, *, dim=8, roles=("system", "user", "assistant"), n_per_cell=4):
    root.mkdir(exist_ok=True)
    manifest = {"format_version": c.FEATURE_FORMAT, "roles": list(roles), "feature_dim": dim,
                "head_dim": dim // 2, "heads": [[1, 2], [2, 3]], "feature_position": c.CAPTURE,
                "model_binding": {"model_id": "engineering-fixture", "revision": "fixture"},
                "data_hash": "fixture-data", "config_hash": "fixture-config",
                "selection_sha256": "fixture-selection", "pilot_fit_sha256": "fixture-pilot"}
    manifest["signature"] = {"model_binding": manifest["model_binding"], "heads": manifest["heads"],
                              "capture": c.CAPTURE, "config_sha256": manifest["config_hash"],
                              "selection_sha256": manifest["selection_sha256"], "pilot_fit_sha256": manifest["pilot_fit_sha256"]}
    manifest["split_sources"] = {split: f"fixture-{split}" for split in c.SPLITS}
    manifest["splits"] = {}
    rng = np.random.default_rng(17)
    for split in c.SPLITS:
        folder = root / split
        folder.mkdir()
        y = np.tile(np.repeat(np.arange(len(roles)), n_per_cell), len(c.DOMAINS))
        n = len(y)
        domain = np.repeat(c.DOMAINS, len(y) // 2)
        x = rng.normal(size=(n, dim)).astype(np.float16)
        x[:, 0] += y * 3
        arrays = {"x": x, "y": y, "domain": domain,
                  "source": np.asarray([f"{split}:{d}:cluster-{i % 2}" for i, d in enumerate(domain)]),
                  "family": np.asarray([f"{split}:position-{i}" for i in range(n)]),
                  "groups": np.asarray([json.dumps({"domain": str(d), "corpus": "fixture"}) for d in domain])}
        files = {}
        for key, value in arrays.items():
            path = folder / f"{key}.npy"
            np.save(path, value, allow_pickle=False)
            files[path.name] = c.sha(path)
        c.atomic_json(folder / "complete.json", {"rows": n, "files": files,
                       "signature": {**manifest["signature"], "split": split, "input_sha256": manifest["split_sources"][split]}})
        manifest["splits"][split] = [{"path": split, "rows": n, "sha256": c.sha(folder / "complete.json")}]
    c.atomic_json(root / "manifest.json", manifest)
    return manifest


def recommit(root, split, name, value):
    path = root / split / name
    np.save(path, value, allow_pickle=False)
    complete = c.read_json(root / split / "complete.json")
    complete["files"][name] = c.sha(path)
    c.atomic_json(root / split / "complete.json", complete)
    manifest = c.read_json(root / "manifest.json")
    manifest["splits"][split][0]["sha256"] = c.sha(root / split / "complete.json")
    c.atomic_json(root / "manifest.json", manifest)


SMOKE = {"classifier": {"device": "cpu", "allow_cpu_smoke": True, "batch_size": 12,
                         "min_epochs": 1, "max_epochs": 1, "patience": 1, "checkpoint_steps": 1,
                         "bootstrap_replicates": 7, "cache_features_on_gpu": False}}


class ScientificProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_v4_dimension_and_dynamic_roles(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            manifest = write_features(root, dim=8192, roles=("system", "user"), n_per_cell=1)
            manifest["heads"] = [[1, head] for head in range(16)]
            manifest["head_dim"] = 512
            manifest["signature"]["heads"] = manifest["heads"]
            for split in c.SPLITS:
                path = root / split / "complete.json"
                complete = c.read_json(path)
                complete["signature"]["heads"] = manifest["heads"]
                c.atomic_json(path, complete)
                manifest["splits"][split][0]["sha256"] = c.sha(path)
            c.atomic_json(root / "manifest.json", manifest)
            bank = c.FeatureBank(root, "fit")
            self.assertEqual(bank.read([1, 0]).shape, (2, 8192))
            self.assertEqual(bank.roles, ("system", "user"))

    def test_standardizer_uses_fit_population_only(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            write_features(root)
            bank = c.FeatureBank(root, "fit")
            values = bank.read(np.arange(len(bank)))
            mean, scale = c.fit_standardizer(bank, chunk_size=5)
            np.testing.assert_allclose(mean, values.astype(np.float64).mean(0), atol=2e-7)
            np.testing.assert_allclose(scale, values.astype(np.float64).std(0), atol=2e-7)
            with self.assertRaisesRegex(ValueError, "only be fitted on FIT"):
                c.fit_standardizer(c.FeatureBank(root, "dev"))

    def test_sampler_equal_domains_roles_unique_positions(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            write_features(root)
            bank = c.FeatureBank(root, "fit")
            sampler = c.BalancedSampler(bank)
            draws = sampler.draw(np.random.default_rng(473), 120000)
            for domain in c.DOMAINS:
                selected = bank.domain[draws] == domain
                self.assertAlmostEqual(selected.mean(), .5, delta=.006)
                for role in range(len(bank.roles)):
                    cell = selected & (bank.y[draws] == role)
                    self.assertAlmostEqual(cell.mean(), 1 / 6, delta=.005)
                    counts = np.bincount(draws[cell], minlength=len(bank))
                    counts = counts[(bank.domain == domain) & (bank.y == role)]
                    self.assertLess(counts.max() - counts.min(), 700)
            self.assertLess(len(np.unique(draws)), len(draws))

    def test_duplicate_positions_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            write_features(root)
            path = root / "fit" / "x.npy"
            values = np.load(path)
            values[0, 0] += 1
            np.save(path, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "hash"):
                c.FeatureBank(root, "fit")
            recommit(root, "fit", "x.npy", values)
            families = np.load(root / "fit" / "family.npy")
            families[1] = families[0]
            recommit(root, "fit", "family.npy", families)
            with self.assertRaisesRegex(ValueError, "Repeated unique"):
                c.FeatureBank(root, "fit")

    def test_source_leakage_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            write_features(root)
            sources = np.load(root / "dev" / "source.npy")
            sources[0] = np.load(root / "fit" / "source.npy")[0]
            recommit(root, "dev", "source.npy", sources)
            with self.assertRaisesRegex(ValueError, "overlap"):
                c.assert_disjoint(c.FeatureBank(root, "fit"), c.FeatureBank(root, "dev"))

    def test_seven_candidates_and_selection_floor(self):
        self.assertEqual(len(c.CANDIDATES), 7)
        self.assertEqual([candidate[1] for candidate in c.CANDIDATES], [0, 128, 128, 128, 256, 256, 256])
        good = {"meets_dev_floor": True, "domain_equal_macro_f1": .91, "minimum_precision_recall": .9, "overall": {"nll": .2}}
        bad = {"meets_dev_floor": False, "domain_equal_macro_f1": .99, "minimum_precision_recall": .89, "overall": {"nll": .1}}
        self.assertLess(c.selection_key(good, "z"), c.selection_key(bad, "a"))
        with self.assertRaisesRegex(ValueError, "0.5"):
            c.training_options({"classifier": {"domain_weights": {c.DOMAINS[0]: .9, c.DOMAINS[1]: .1}}})

    def test_full_cpu_pipeline_freezes_before_test_and_reuses_result(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            features, run = Path(temporary) / "features", Path(temporary) / "run"
            write_features(features)
            original = c.FeatureBank
            opened = []

            def bank(*args, **kwargs):
                opened.append(args[1])
                return original(*args, **kwargs)

            with mock.patch.object(c, "FeatureBank", side_effect=bank), contextlib.redirect_stdout(io.StringIO()):
                selection = c.train_candidates(features, run, SMOKE)
            self.assertEqual(set(opened), {"fit", "dev"})
            self.assertEqual(len(selection["candidate_results"]), 7)
            with self.assertRaises(FileNotFoundError):
                c.evaluate(features, run)
            frozen = c.freeze(run)
            self.assertEqual(c.sha(run / "checkpoint.npz"), frozen["probe_sha256"])
            result = c.evaluate(features, run)
            self.assertEqual(result["split"], "test")
            self.assertEqual(result["confidence_intervals"]["replicates"], 7)
            self.assertEqual(c.read_json(run / "metrics.json")["test"], result)
            with mock.patch.object(c, "predict", side_effect=AssertionError("TEST rerun")):
                self.assertEqual(c.evaluate(features, run), result)
            with self.assertRaisesRegex(ValueError, "frozen"):
                c.train_candidates(features, run, SMOKE, resume=True)
            # Every exported candidate has a working dynamic CPU consumer.
            for candidate_id, _, _ in c.CANDIDATES:
                probe = c.Probe(run / "candidates" / candidate_id / "best.npz")
                probabilities = probe.predict_proba(original(features, "dev").read([0, 1]))
                self.assertEqual(probabilities.shape, (2, 3))
                np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6)
            (run / "checkpoint.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                c.evaluate(features, run)

    def test_mid_batch_resume_reproduces_all_candidate_weights(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            features = Path(temporary) / "features"
            fresh, resumed = Path(temporary) / "fresh", Path(temporary) / "resumed"
            write_features(features)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = c.train_candidates(features, fresh, SMOKE)
            original_save = c.save_torch
            calls = 0

            def interrupt(path, value):
                nonlocal calls
                original_save(path, value)
                calls += 1
                if calls == 1:
                    raise RuntimeError("simulated interruption after first batch")

            with mock.patch.object(c, "save_torch", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                c.train_candidates(features, resumed, SMOKE)
            with contextlib.redirect_stdout(io.StringIO()):
                actual = c.train_candidates(features, resumed, SMOKE, resume=True)
            self.assertEqual(expected["candidate_id"], actual["candidate_id"])
            for candidate_id, _, _ in c.CANDIDATES:
                first = c.Probe(fresh / "candidates" / candidate_id / "best.npz")
                second = c.Probe(resumed / "candidates" / candidate_id / "best.npz")
                for name in first.weights:
                    np.testing.assert_array_equal(first.weights[name], second.weights[name])
                self.assertEqual(c.read_json(fresh / "candidates" / candidate_id / "history.json"),
                                 c.read_json(resumed / "candidates" / candidate_id / "history.json"))

    def test_test_inference_cannot_restart_after_interruption(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            features, run = Path(temporary) / "features", Path(temporary) / "run"
            write_features(features)
            with contextlib.redirect_stdout(io.StringIO()):
                c.train_candidates(features, run, SMOKE)
            c.freeze(run)
            with mock.patch.object(c, "predict", side_effect=RuntimeError("interrupted TEST")):
                with self.assertRaisesRegex(RuntimeError, "interrupted TEST"):
                    c.evaluate(features, run)
            with self.assertRaisesRegex(ValueError, "already started"):
                c.evaluate(features, run)

    def test_test_features_can_be_appended_after_freeze(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            features, run = Path(temporary) / "features", Path(temporary) / "run"
            complete_manifest = write_features(features)
            pending = json.loads(json.dumps(complete_manifest))
            pending["splits"].pop("test")
            c.atomic_json(features / "manifest.json", pending)
            with contextlib.redirect_stdout(io.StringIO()):
                c.train_candidates(features, run, SMOKE)
            c.freeze(run)
            original_hash = c.sha(features / "manifest.json")
            c.atomic_json(features / "manifest.json", complete_manifest)
            self.assertNotEqual(original_hash, c.sha(features / "manifest.json"))
            result = c.evaluate(features, run)
            self.assertEqual(result["training_feature_manifest_sha256"], original_hash)
            self.assertEqual(result["feature_manifest_sha256"], c.sha(features / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
