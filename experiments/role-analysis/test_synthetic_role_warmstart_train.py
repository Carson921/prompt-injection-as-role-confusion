"""CPU semantic tests: cache identity, warm initialization, and exact resume."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import synthetic_agent_classifier as base
import synthetic_role_warmstart_train as warm

try:
    import torch
except ImportError:
    torch = None


def arrays(prefix, *, sources=8):
    rng = np.random.default_rng(37)
    labels, structures, groups, source_ids = [], [], [], []
    for source in range(sources):
        for profile in ("paired", "context"):
            for role in range(6):
                labels.append(role)
                structures.append(f"{profile}:six-role")
                source_ids.append(f"{prefix}-{source}")
                groups.append(json.dumps(dict(profile=profile, body_length_bin=str(128 * (source % 4 + 1)),
                    body_quartile=source % 4, wrapper_type="plain" if profile == "paired" else "tools-before")))
    y = np.asarray(labels, dtype=np.int64)
    x = rng.normal(0, .1, (len(y), 1024)).astype(np.float16)
    x[np.arange(len(y)), y] += 2
    return dict(x=x, y=y, source=np.asarray(source_ids),
                family=np.asarray([f"{prefix}-family-{i}" for i in range(len(y))]),
                structure=np.asarray(structures), weight=np.ones(len(y), dtype=np.float32),
                groups=np.asarray(groups))


def write_features(root, parent_metadata, parent_sha, *, old=False):
    root.mkdir()
    manifest = {key: parent_metadata[key] for key in warm.BINDING_KEYS}
    manifest.update(format_version=base.FEATURE_FORMAT if old else warm.FORMAT,
                    parent_probe_sha256=parent_sha, splits={}, inputs={})
    if old:
        manifest.update(data_hash="fixture-data", config_hash="fixture-config",
                        selection_sha256="fixture-heads", pilot_fit_sha256="fixture-pilot")
        signature = dict(model_binding=manifest["model_binding"], heads=manifest["heads"],
            config_sha256=manifest["config_hash"], capture="pre-o_proj",
            selection_sha256=manifest["selection_sha256"], pilot_fit_sha256=manifest["pilot_fit_sha256"])
        manifest.update(signature=signature, split_sources={})
    for split in ("fit", "dev", "test"):
        data = arrays(("old-" if old else "new-") + split)
        folder = root / f"{split}-0000"
        folder.mkdir()
        hashes = {}
        for key, value in data.items():
            np.save(folder / f"{key}.npy", value, allow_pickle=False)
            hashes[key + ".npy"] = warm.digest(folder / f"{key}.npy")
        complete = dict(rows=len(data["y"]), split=split, files=hashes)
        if old:
            manifest["split_sources"][split] = f"fixture-input-{split}"
            complete["signature"] = {**signature, "split": split, "input_sha256": f"fixture-input-{split}"}
        else:
            complete["signature"] = dict(input_sha256=f"fixture-input-{split}", heads=manifest["heads"],
                model_binding=manifest["model_binding"], parent_probe_sha256=parent_sha,
                extractor_sha256="fixture-extractor", runtime_sha256="fixture-runtime")
        warm.atomic_json(folder / "complete.json", complete)
        manifest["splits"][split] = [dict(path=folder.name, rows=len(data["y"]),
                                         sha256=warm.digest(folder / "complete.json"))]
        manifest["inputs"][split] = {"sha256": f"fixture-input-{split}", "signature": complete["signature"]}
    warm.atomic_json(root / "manifest.json", manifest)


def fixture(root):
    if torch is None:
        raise unittest.SkipTest("PyTorch not installed")
    torch.set_num_threads(1)
    torch.manual_seed(9)
    net = base.make_net(128, dropout=.1)
    metadata = dict(format_version=base.PROBE_FORMAT, roles=list(base.ROLES), hidden=128,
        heads=list(range(16)), feature_dim=1024, feature_position="pre-o_proj", best_epoch=30,
        candidate_id="parent-fixture", dropout=.1, model_binding={"model": "test-fixture"},
        selection_sha256="fixture-heads", pilot_fit_sha256="fixture-pilot", engineering_smoke_only=True)
    parent = root / "parent.npz"
    mean = np.linspace(-.1, .1, 1024, dtype=np.float32)
    scale = np.linspace(.8, 1.2, 1024, dtype=np.float32)
    base._export(parent, net, mean, scale, metadata)
    write_features(root / "new", metadata, warm.digest(parent))
    write_features(root / "old", metadata, warm.digest(parent), old=True)
    return parent


class CacheAndSamplerTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_cache_order_integrity_and_split_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root)
            bank = warm.WarmBank(root / "new", "fit")
            x = np.load(root / "new/fit-0000/x.npy")
            np.testing.assert_array_equal(bank.read([7, 1, 7]), x[[7, 1, 7]].astype(np.float32))
            identity = bank.split_identity_hash
            manifest_path = root / "new/manifest.json"
            m = json.loads(manifest_path.read_text())
            m["operational_timestamp"] = 100
            m["splits"]["test"][0]["sha256"] = "not-used-by-fit"
            warm.atomic_json(manifest_path, m)
            self.assertEqual(identity, warm.WarmBank(root / "new", "fit").split_identity_hash)
            with (root / "new/fit-0000/x.npy").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Invalid feature file"):
                warm.WarmBank(root / "new", "fit")

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_paths_and_parent_binding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root)
            probe = base.load_probe(parent)
            wrong = {**probe.metadata, "heads": list(reversed(probe.metadata["heads"]))}
            with self.assertRaisesRegex(ValueError, "Feature binding differs: heads"):
                warm.WarmBank(root / "new", "fit", expected_binding=wrong)
            m = json.loads((root / "new/manifest.json").read_text())
            m["splits"]["fit"][0]["path"] = "../../escape"
            warm.atomic_json(root / "new/manifest.json", m)
            with self.assertRaisesRegex(ValueError, "escapes cache"):
                warm.WarmBank(root / "new", "fit")

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_incomplete_split_and_committed_wrong_signature_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "new/manifest.json"
            m = json.loads(path.read_text())
            m["progress"] = {"fit": {"complete": False}}
            warm.atomic_json(path, m)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                warm.WarmBank(root / "new", "fit")
            m["progress"]["fit"] = {"complete": True}
            folder = root / "new/fit-0000"
            complete = json.loads((folder / "complete.json").read_text())
            complete["signature"]["runtime_sha256"] = "different-runtime"
            warm.atomic_json(folder / "complete.json", complete)
            m["splits"]["fit"][0]["sha256"] = warm.digest(folder / "complete.json")
            warm.atomic_json(path, m)
            with self.assertRaisesRegex(ValueError, "Shard extraction signature"):
                warm.WarmBank(root / "new", "fit")

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_replay_and_hierarchical_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            new = warm.WarmBank(root / "new", "fit")
            old = base.FeatureBank(root / "old", "fit")
            indices = warm.stratified_replay(old, 60)
            np.testing.assert_array_equal(indices, warm.stratified_replay(old, 60))
            self.assertEqual(len(indices), len(set(map(int, indices))))
            sampler = warm.MixedSampler(new, warm.BankView(old, indices))
            is_new, rows = sampler.draw(np.random.default_rng(173), 100000)
            counts = sampler.counts(is_new, rows)
            self.assertAlmostEqual(counts["new_paired"] / len(rows), .36, delta=.008)
            self.assertAlmostEqual(counts["new_context"] / len(rows), .24, delta=.008)
            self.assertAlmostEqual(counts["old_replay"] / len(rows), .4, delta=.008)
            x, labels = sampler.read(is_new[:10], rows[:10])
            self.assertEqual(x.shape, (10, 1024))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(((labels >= 0) & (labels < 6)).all())


@unittest.skipIf(torch is None, "PyTorch not installed")
class TrainingSemantics(unittest.TestCase):
    def options(self):
        return dict(epoch_rows=96, batch_size=16, replay_rows=60, max_epochs=2,
                    min_epochs=2, patience=4, checkpoint_steps=1,
                    weak_group_min_sources=1, weak_group_min_positives=1)

    def train(self, root, parent, run, candidate="warm-all", **kwargs):
        return warm.train_candidate(root / "new", root / "old", parent, root / run, candidate,
                                    device="cpu", options=self.options(), allow_cpu_smoke=True, **kwargs)

    def test_initialization_and_output_freezing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root)
            record = self.train(root, parent, "output", candidate="warm-output")
            self.assertTrue(record["hidden_layer_unchanged"])
            self.assertLess(record["init_parent_probability_max_error"], 3e-5)
            original = base.load_probe(parent)
            trained = base.load_probe(root / "output/candidates/warm-output/best.npz")
            np.testing.assert_array_equal(original.mean, trained.mean)
            np.testing.assert_array_equal(original.scale, trained.scale)
            for key in ("0.weight", "0.bias"):
                np.testing.assert_array_equal(original.weights[key], trained.weights[key])
            self.assertFalse(np.array_equal(original.weights["3.weight"], trained.weights["3.weight"]))
            self.assertEqual(sum(record["sampled_rows"].values()), 192)
            with self.assertRaisesRegex(ValueError, "Engineering training"):
                warm.freeze(root / "output")

    def test_exact_resume_in_middle_of_dropout_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root)
            direct = self.train(root, parent, "direct")
            suspended = self.train(root, parent, "resumed", stop_after_steps=3)
            self.assertEqual(suspended, dict(status="suspended", optimizer_step=3))
            resumed = self.train(root, parent, "resumed", resume=True)
            first = base.load_probe(root / "direct/candidates/warm-all/best.npz")
            second = base.load_probe(root / "resumed/candidates/warm-all/best.npz")
            for key in first.weights:
                np.testing.assert_array_equal(first.weights[key], second.weights[key])
            self.assertEqual(direct["sampled_rows"], resumed["sampled_rows"])
            self.assertEqual(direct["best_epoch"], resumed["best_epoch"])
            a = json.loads((root / "direct/candidates/warm-all/history.json").read_text())
            b = json.loads((root / "resumed/candidates/warm-all/history.json").read_text())
            self.assertEqual(a, b)

    def test_resume_refuses_changed_new_data_and_no_implicit_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root)
            self.train(root, parent, "run", stop_after_steps=1)
            with self.assertRaisesRegex(ValueError, "explicit resume"):
                self.train(root, parent, "run")
            options = self.options()
            options["epoch_rows"] = 192
            with self.assertRaisesRegex(ValueError, "data/config/parent changed"):
                warm.train_candidate(root / "new", root / "old", parent, root / "run", "warm-all",
                    device="cpu", options=options, allow_cpu_smoke=True, resume=True)

    def test_quality_exclusions_clean_replay_and_matched_dev_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root)
            ledger_path = root / "quality-excluded-clusters.json"
            ledger = dict(format_version="warmstart-source-quality-exclusions-v1",
                code_sha256="fixture-quality-code", old_sources_sha256="fixture-old-sources",
                supplementary_sources_sha256="fixture-supplementary", assessed_sources=24,
                excluded_clusters=["old-fit-0", "old-dev-0"])
            warm.atomic_json(ledger_path, ledger)
            old_manifest_before = warm.digest(root / "old/manifest.json")
            prepared = warm.prepare(root / "new", root / "old", parent, root / "clean",
                options=self.options(), allow_cpu_smoke=True, excluded_clusters=ledger_path)
            self.assertNotIn("old-fit-0", prepared["replay"].source)
            self.assertNotIn("old-dev-0", prepared["old_dev"].source)
            self.assertIn("old-dev-0", prepared["old_dev_full"].source)
            self.assertEqual(len(prepared["old_dev_full"]) - len(prepared["old_dev"]), 12)
            record = self.train(root, parent, "clean", excluded_clusters=ledger_path)
            self.assertEqual(record["dev"]["old_dev"]["overall"]["tokens"], 84)
            self.assertEqual(record["dev"]["old_dev_full_diagnostic"]["overall"]["tokens"], 96)
            self.assertFalse(record["dev"]["old_dev_full_used_for_selection"])
            self.assertEqual(warm.digest(root / "old/manifest.json"), old_manifest_before)
            proof = json.loads((root / "clean/cache-import-proof.json").read_text())
            self.assertEqual(proof["old_fit_excluded_rows"], 12)
            self.assertEqual(proof["old_dev_excluded_rows"], 12)
            self.assertEqual(proof["rejected_sources_in_replay"], 0)
            self.assertEqual(proof["quality_exclusions"]["ledger_sha256"], warm.digest(ledger_path))
            ledger["excluded_clusters"].append("old-dev-1")
            warm.atomic_json(ledger_path, ledger)
            with self.assertRaisesRegex(ValueError, "data/config/parent changed"):
                self.train(root, parent, "clean", excluded_clusters=ledger_path, resume=True)
            ledger["excluded_clusters"].append("new-fit-0")
            warm.atomic_json(ledger_path, ledger)
            with self.assertRaisesRegex(ValueError, "contains a rejected source"):
                warm.prepare(root / "new", root / "old", parent, root / "bad-new",
                    options=self.options(), allow_cpu_smoke=True, excluded_clusters=ledger_path)

    def test_freeze_test_isolation_and_immutable_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_path = fixture(root)
            run = root / "final-fixture"
            prepared = warm.prepare(root / "new", root / "old", parent_path, run,
                                    options=self.options(), allow_cpu_smoke=True)
            # Construct protocol fixtures, not a claimed scientific CPU result.
            training = prepared["training"]
            training["engineering_smoke_only"] = False
            warm.atomic_json(run / "training.json", training)
            parent = prepared["parent"]
            net = parent.torch("cpu")
            for candidate in warm.CANDIDATES:
                directory = run / "candidates" / candidate
                directory.mkdir(parents=True)
                metadata = {**parent.metadata, "engineering_smoke_only": False,
                    "candidate_id": candidate, "training_hash": prepared["training_hash"],
                    "warm_start": dict(parent_probe_sha256=warm.digest(parent_path))}
                base._export(directory / "best.npz", net, parent.mean, parent.scale, metadata)
                warm.atomic_json(directory / "history.json", [])
                warm.atomic_json(directory / "complete.json", dict(candidate_id=candidate,
                    engineering_smoke_only=False, training_hash=prepared["training_hash"],
                    probe_sha256=warm.digest(directory / "best.npz"),
                    history_sha256=warm.digest(directory / "history.json"),
                    dev=dict(eligible=True, selection_key=[False, -.96 if candidate == "warm-all" else -.90,
                                                         -.99, .01, candidate])))
            with self.assertRaises(FileNotFoundError):
                warm.evaluate(root / "new", run, bootstrap_replicates=10)
            frozen = warm.freeze(run)
            self.assertEqual(frozen["candidate_id"], "warm-all")
            self.assertEqual(frozen["selection_split"], "dev")
            self.assertEqual(frozen, warm.freeze(run))
            result = warm.evaluate(root / "new", run, bootstrap_replicates=10)
            self.assertEqual(result["confidence_intervals"]["clusters"], 8)
            # A validly committed cache still cannot reuse a fitting source.
            folder = root / "new/test-0000"
            source = np.load(folder / "source.npy")
            source[0] = "new-fit-0"
            np.save(folder / "source.npy", source, allow_pickle=False)
            complete = json.loads((folder / "complete.json").read_text())
            complete["files"]["source.npy"] = warm.digest(folder / "source.npy")
            warm.atomic_json(folder / "complete.json", complete)
            m = json.loads((root / "new/manifest.json").read_text())
            m["splits"]["test"][0]["sha256"] = warm.digest(folder / "complete.json")
            warm.atomic_json(root / "new/manifest.json", m)
            with self.assertRaisesRegex(ValueError, "TEST sources overlap"):
                warm.evaluate(root / "new", run, bootstrap_replicates=10)
            with (run / "best.npz").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Frozen experiment changed"):
                warm.evaluate(root / "new", run, bootstrap_replicates=10)


class DecisionTests(unittest.TestCase):
    def result(self):
        return dict(minimum_precision_recall=.95, overall=dict(macro_f1=.99, nll=.02),
                    low_permission_to_sd=dict(rate=.001), weakest_sd_recall=.94,
                    structure_equal_macro_f1=.98)

    def test_selection_checks_both_domains_and_no_weak_group_fabrication(self):
        new, old = self.result(), self.result()
        self.assertTrue(warm.dev_decision(new, old, .99, "warm-all")["eligible"])
        old["overall"]["macro_f1"] = .98
        self.assertFalse(warm.dev_decision(new, old, .99, "warm-all")["eligible"])
        old = self.result()
        new["weakest_sd_recall"] = None
        self.assertFalse(warm.dev_decision(new, old, .99, "warm-all")["eligible"])
        new = self.result()
        new["low_permission_to_sd"]["rate"] = .02
        self.assertFalse(warm.dev_decision(new, old, .99, "warm-all")["eligible"])


class BudgetTests(unittest.TestCase):
    def test_expired_budget_rejected_before_network_or_cache_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "budget.json"
            now = time.time()
            warm.atomic_json(path, dict(started_unix=now - 30, deadline_unix=now - 1))
            with patch.object(warm, "initialize_network") as initialized:
                with self.assertRaisesRegex(ValueError, "budget expired"):
                    warm.train_candidate(root / "absent-new", root / "absent-old", root / "absent-parent",
                                         root / "run", "warm-all", device="cuda:0", budget_file=path)
                initialized.assert_not_called()
            with self.assertRaisesRegex(ValueError, "requires --budget-file"):
                warm.train_candidate(root / "absent-new", root / "absent-old", root / "absent-parent",
                                     root / "run", "warm-all", device="cuda:0")

    def test_budget_fixed_for_process_and_new_run_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "budget.json"
            now = time.time()
            warm.atomic_json(path, dict(started_unix=now - 1, deadline_unix=now + 60))
            guard = warm.BudgetGuard(path)
            guard.bind(root / "run")
            guard.check()
            with patch.object(warm.time, "time", return_value=now + 61):
                with self.assertRaisesRegex(ValueError, "budget expired"):
                    guard.check()
            warm.atomic_json(path, dict(started_unix=now - 1, deadline_unix=now + 120))
            with self.assertRaisesRegex(ValueError, "budget changed"):
                guard.check()
            extended = warm.BudgetGuard(path)
            with self.assertRaisesRegex(ValueError, "different GPU budget"):
                extended.bind(root / "run")


if __name__ == "__main__":
    unittest.main()
