"""Independent CPU checks of data isolation, balancing, and selection semantics."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import synthetic_agent_classifier as c


def _write_bank(root, x, y, *, structure=None, family=None, source=None, weight=None):
    n = len(y)
    shard = root / "fit-000000"
    shard.mkdir()
    arrays = {
        "x": np.asarray(x, dtype=np.float16), "y": np.asarray(y, dtype=np.int64),
        "structure": np.asarray(structure if structure is not None else ["single"] * n),
        "family": np.asarray(family if family is not None else [f"family-{i}" for i in range(n)]),
        "source": np.asarray(source if source is not None else [f"source-{i}" for i in range(n)]),
        "weight": np.asarray(weight if weight is not None else np.ones(n), dtype=np.float32),
    }
    files = {}
    for key, value in arrays.items():
        path = shard / f"{key}.npy"
        np.save(path, value, allow_pickle=False)
        files[path.name] = c._sha(path)
    manifest = {"format_version": c.FEATURE_FORMAT, "roles": list(c.ROLES), "feature_dim": 1024,
                "model_binding": {"model": "test-fixture", "tokenizer": "test-fixture"},
                "heads": list(range(16)), "feature_position": "pre-o_proj",
                "data_hash": "data-fixture", "config_hash": "config-fixture",
                "selection_sha256": "heads-fixture", "pilot_fit_sha256": "pilot-fit-fixture"}
    signature = {"model_binding": manifest["model_binding"], "heads": manifest["heads"],
                 "config_sha256": manifest["config_hash"], "phase": "pilot", "capture": "pre-o_proj",
                 "selection_sha256": "heads-fixture", "pilot_fit_sha256": "pilot-fit-fixture",
                 "implementation_sha256": "code-fixture"}
    manifest.update(signature=signature, split_sources={"fit": "fit-input-fixture"})
    c._atomic_json(shard / "complete.json", {"files": files, "rows": n,
                                             "signature": {**signature, "split": "fit", "input_sha256": "fit-input-fixture"}})
    manifest["splits"] = {"fit": [{"path": shard.name, "rows": n, "sha256": c._sha(shard / "complete.json")} ]}
    c._atomic_json(root / "manifest.json", manifest)


def _hierarchical_bank(root, *, include_multi=True, include_unknown=False):
    labels, structures, families = [], [], []

    def add(structure, roles, names=None):
        start = len(labels)
        labels.extend(roles)
        structures.extend([structure] * len(roles))
        families.extend(names if names is not None else [f"family-{i}" for i in range(start, start + len(roles))])

    add("single:S", [0] * 5, ["duplicate"] * 4 + ["unique"])
    add("single:U", [2, 2])
    add("one_round:SDUA", [0] * 10 + [1, 2, 4])
    if include_multi:
        for i in range(20):
            add(f"multi_round:fixture-{i:02d}", list(range(6)))
    if include_unknown:
        add("unknown:fixture", [0])
    _write_bank(root, np.zeros((len(labels), 1024)), labels, structure=structures,
                family=families, source=["source-" + family for family in families])
    return c.FeatureBank(root, "fit")


class ScientificSemantics(unittest.TestCase):
    def test_scientific_training_defaults_to_hierarchical_profile_weights(self):
        options = c._training_options({})
        self.assertEqual(options["profile_weights"], {"single": .2, "one_round": .3, "multi_round": .5})
        self.assertEqual(list(options["profile_weights"]), list(c.PROFILE_ORDER))
        with self.assertRaisesRegex(ValueError, "flat mode is engineering"):
            c._training_options({"classifier": {"profile_weights": None}})
        engineering = c._training_options({"classifier": {"allow_cpu_smoke": True, "device": "cpu",
            "min_epochs": 1, "max_epochs": 1, "profile_weights": None}})
        self.assertIsNone(engineering["profile_weights"])

    def test_hierarchy_preserves_profile_structure_role_and_family_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = _hierarchical_bank(Path(tmp))
            sampler = c.BalancedSampler(bank, profile_weights=c.DEFAULT_PROFILE_WEIGHTS)
            draw = sampler.draw(np.random.default_rng(173), 100000)
            counts = sampler.count_profiles(draw)
            self.assertEqual(sum(counts.values()), 100000)
            for profile, expected in c.DEFAULT_PROFILE_WEIGHTS.items():
                self.assertAlmostEqual(counts[profile] / len(draw), expected, delta=.004)
            sampled_structures = bank.structure[draw]
            # Twenty multi templates share 50%, rather than multiplying it.
            for i in range(20):
                self.assertAlmostEqual(np.mean(sampled_structures == f"multi_round:fixture-{i:02d}"), .025, delta=.003)
            # The two single templates split 20%, regardless of row counts.
            for template in ("single:S", "single:U"):
                self.assertAlmostEqual(np.mean(sampled_structures == template), .1, delta=.004)
            one_round = draw[sampled_structures == "one_round:SDUA"]
            for role in (0, 1, 2, 4):
                self.assertAlmostEqual(np.mean(bank.y[one_round] == role), .25, delta=.015)
            # Four context copies of one family receive the same cell mass as
            # the other, unduplicated family; no class weighting is applied.
            s_draw = draw[sampled_structures == "single:S"]
            ratio = np.sum(bank.family[s_draw] == "duplicate") / np.sum(bank.family[s_draw] == "unique")
            self.assertAlmostEqual(ratio, 1, delta=.08)

    def test_profile_key_order_global_rng_and_resume_replay_are_irrelevant(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = _hierarchical_bank(Path(tmp))
            first = c.BalancedSampler(bank, profile_weights={"single": .2, "one_round": .3, "multi_round": .5})
            reordered = c.BalancedSampler(bank, profile_weights={"multi_round": .5, "one_round": .3, "single": .2})
            self.assertEqual(first.protocol, reordered.protocol)
            rng = np.random.default_rng(29)
            state = copy.deepcopy(rng.bit_generator.state)
            expected = first.draw(rng, 1003)
            np.random.seed(123)
            replay = np.random.default_rng()
            replay.bit_generator.state = state
            actual = reordered.draw(replay, 1003)
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(first.draw(rng, 97), reordered.draw(replay, 97))
            # Checkpoint batch_cursor skips the exact already-counted prefix.
            full = first.count_profiles(expected)
            prefix, suffix = first.count_profiles(expected[:384]), first.count_profiles(actual[384:])
            self.assertEqual(full, {key: prefix[key] + suffix[key] for key in full})

    def test_hierarchy_rejects_missing_and_unknown_profiles_without_fabrication(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = _hierarchical_bank(Path(tmp), include_multi=False)
            with self.assertRaisesRegex(ValueError, "no applicable structures: multi_round"):
                c.BalancedSampler(bank, profile_weights=c.DEFAULT_PROFILE_WEIGHTS)
        with tempfile.TemporaryDirectory() as tmp:
            bank = _hierarchical_bank(Path(tmp), include_unknown=True)
            with self.assertRaisesRegex(ValueError, "known profile:template"):
                c.BalancedSampler(bank, profile_weights=c.DEFAULT_PROFILE_WEIGHTS)
            with self.assertRaisesRegex(ValueError, "only single, one_round, multi_round"):
                c.BalancedSampler(bank, profile_weights={**c.DEFAULT_PROFILE_WEIGHTS, "unknown": 0})

    def test_family_mass_and_weighted_fit_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.zeros((3, 1024), dtype=np.float16)
            x[:, 0] = [0, 0, 10]
            _write_bank(root, x, [0, 0, 1], family=["same", "same", "other"])
            bank = c.FeatureBank(root, "fit")
            np.testing.assert_allclose(bank.base_weight, [.5, .5, 1])
            mean, scale = c.fit_standardizer(bank, chunk_size=1)
            self.assertAlmostEqual(float(mean[0]), 5)
            self.assertAlmostEqual(float(scale[0]), 5)
            self.assertEqual(float(scale[1]), 1)
            np.testing.assert_array_equal(bank.read([2, 0, 2]), x[[2, 0, 2]].astype(np.float32))

    def test_no_nonexistent_cells_and_no_duplicate_family_advantage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Structure a has only S,D; b only U. Duplicate S token family
            # must not obtain more sampling mass than another S family.
            _write_bank(root, np.zeros((8, 1024)), [0, 0, 0, 0, 0, 1, 1, 2],
                        structure=["a"] * 7 + ["b"],
                        family=["dup"] * 4 + ["unique", "d1", "d2", "u1"])
            bank = c.FeatureBank(root, "fit")
            sampler = c.BalancedSampler(bank)
            draw = sampler.draw(np.random.default_rng(1), 120000)
            self.assertAlmostEqual(np.mean(bank.structure[draw] == "a"), .5, delta=.01)
            self.assertAlmostEqual(np.mean(bank.y[draw] == 0), .25, delta=.01)
            self.assertAlmostEqual(np.mean(bank.y[draw] == 1), .25, delta=.01)
            self.assertAlmostEqual(np.mean(bank.y[draw] == 2), .5, delta=.01)
            duplicated = np.sum(draw < 4)
            unique = np.sum(draw == 4)
            self.assertAlmostEqual(duplicated / unique, 1, delta=.05)

    def test_shard_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bank(root, np.zeros((1, 1024)), [0])
            with (root / "fit-000000" / "y.npy").open("ab") as f:
                f.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Invalid feature file"):
                c.FeatureBank(root, "fit")

    def test_valid_other_bank_shard_rejected_even_with_recomputed_commit_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bank(root, np.zeros((1, 1024)), [0])
            complete = root / "fit-000000/complete.json"
            record = json.loads(complete.read_text())
            record["signature"]["heads"] = list(reversed(range(16)))
            c._atomic_json(complete, record)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            # Every byte checksum remains internally valid, yet head order is
            # from another extraction and must not be consumed.
            manifest["splits"]["fit"][0]["sha256"] = c._sha(complete)
            c._atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "Shard signature"):
                c.FeatureBank(root, "fit")

    def test_top_level_heads_cannot_relabel_an_otherwise_valid_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bank(root, np.zeros((1, 1024)), [0])
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["heads"] = list(reversed(range(16)))
            c._atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "extraction signature: heads"):
                c.FeatureBank(root, "fit")

    def test_resume_identity_ignores_timing_but_binds_committed_feature_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir, second_dir = root / "first", root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            _write_bank(first_dir, np.zeros((1, 1024)), [0])
            x = np.zeros((1, 1024))
            x[0, 0] = 1
            _write_bank(second_dir, x, [0])
            first = c.FeatureBank(first_dir, "fit")
            path = first_dir / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(last_extraction_seconds=123.4, feature_root="/a/relocated/cache")
            c._atomic_json(path, manifest)
            replayed = c.FeatureBank(first_dir, "fit")
            changed = c.FeatureBank(second_dir, "fit")
            self.assertNotEqual(first.manifest_hash, replayed.manifest_hash)
            self.assertEqual(first.split_identity_hash, replayed.split_identity_hash)
            self.assertNotEqual(first.split_identity_hash, changed.split_identity_hash)

    def test_floor_precedes_macro_f1_and_nll(self):
        low = {"meets_dev_floor": False, "structure_equal_macro_f1": .999,
               "minimum_precision_recall": .899, "overall": {"nll": .001}}
        qualified = {"meets_dev_floor": True, "structure_equal_macro_f1": .92,
                     "minimum_precision_recall": .9, "overall": {"nll": .1}}
        self.assertLess(c.selection_key(qualified, "b"), c.selection_key(low, "a"))

    def test_absent_role_recall_is_na_but_false_positives_remain(self):
        p = np.full((2, 6), .01)
        p[0, 0] = .95
        p[1, 5] = .95
        result = c.metrics(p, np.array([0, 0]))
        self.assertIsNone(result["roles"]["tool"]["recall"])
        self.assertIsNone(result["roles"]["tool"]["f1"])
        self.assertEqual(result["roles"]["tool"]["false_positives"], 1)
        self.assertEqual(result["roles"]["system"]["recall"], .5)

    def test_sampler_replay_does_not_depend_on_global_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bank(root, np.zeros((12, 1024)), np.arange(12) % 6)
            sampler = c.BalancedSampler(c.FeatureBank(root, "fit"))
            rng = np.random.default_rng(17)
            before = rng.bit_generator.state
            first = sampler.draw(rng, 121)
            np.random.seed(99)
            replay = np.random.default_rng()
            replay.bit_generator.state = before
            np.testing.assert_array_equal(first, sampler.draw(replay, 121))

    def test_old_five_class_export_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.npz"
            np.savez(path, metadata=np.array(json.dumps({"roles": ["system", "user", "cot", "assistant", "tool"]})))
            with self.assertRaisesRegex(ValueError, "five-role"):
                c.load_probe(path)

    def test_formal_freeze_requires_external_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c._atomic_json(root / "training.json", {"phase": "formal"})
            with self.assertRaisesRegex(ValueError, "external protocol"):
                c.freeze(root)

    def test_linear_numpy_consumer_order_and_head_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.npz"
            weights = np.zeros((6, 1024), dtype=np.float32)
            weights[:, :6] = np.eye(6)
            meta = {"format_version": c.PROBE_FORMAT, "roles": list(c.ROLES),
                    "heads": list(range(16)), "model_binding": {"model": "test"},
                    "feature_position": "pre-o_proj", "feature_dim": 1024, "hidden": 0}
            np.savez(path, metadata=np.array(json.dumps(meta)), mean=np.zeros(1024, dtype=np.float32),
                     scale=np.ones(1024, dtype=np.float32),
                     **{"weight::0.weight": weights, "weight::0.bias": np.zeros(6, dtype=np.float32)})
            probe = c.load_probe(path)
            x = np.zeros((6, 1024), dtype=np.float32)
            x[:, :6] = 10 * np.eye(6)
            np.testing.assert_array_equal(probe.predict_proba(x).argmax(1), np.arange(6))
            with self.assertRaisesRegex(ValueError, "heads"):
                c.load_probe(path, expected_binding={"heads": list(reversed(range(16)))})


if __name__ == "__main__":
    unittest.main()
