"""No Torch/GPU imports: native performance-cache planning and safe publication."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("performance_cache", HERE / "prepare_deepseek_performance_cache.py")
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class Config:
    def __init__(self, value):
        self.value = value
    def all_kwargs(self):
        return self.value


class PerformanceCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.identity = {"triton_key": "triton-v", "backend_hash": "cuda90",
                         "jit_cache_key": "kernel-src", "invalidating_environment_repr": "[]",
                         "config_grid_repr": ["first-config", "second-config"]}
        self.best = {"COMPUTE_MODE": "dot_scaled", "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
                     "num_warps": 2, "num_ctas": 1, "num_stages": 3}
        self.alternative = {**self.best, "num_stages": 4}
        self.configs = [Config(self.best), Config(self.alternative)]
        for projection in cache.PROJECTIONS:
            path = cache.native_cache_path(self.root, self.identity, cache.tuning_key(projection, 8))
            path.parent.mkdir()
            path.write_text(json.dumps({"best": self.best, "timings": [[self.best, 0.1], [self.alternative, 0.2]]}))

    def plan(self):
        return cache.make_plan(self.identity, self.configs, self.root, [8, 259, 8309], [8], {8309})

    def test_hash_is_exact_native_concatenation(self):
        key = cache.tuning_key("gate_up", 259)
        expected = hashlib.sha256(("triton-v-cuda90-kernel-src-[]-" + str(tuple(key))
                                   + "-first-config-second-config").encode()).hexdigest()
        self.assertEqual(expected, cache.native_cache_hash(self.identity, key))

    def test_runtime_source_backend_grid_env_and_dtype_change_identity(self):
        key = cache.tuning_key("down", 259)
        original = cache.native_cache_hash(self.identity, key)
        for field in ("triton_key", "backend_hash", "jit_cache_key", "invalidating_environment_repr"):
            other = {**self.identity, field: self.identity[field] + "changed"}
            self.assertNotEqual(original, cache.native_cache_hash(other, key), field)
        other = {**self.identity, "config_grid_repr": list(reversed(self.identity["config_grid_repr"]))}
        self.assertNotEqual(original, cache.native_cache_hash(other, key))
        other_key = [*key[:3], "torch.float16", *key[4:]]
        self.assertNotEqual(original, cache.native_cache_hash(self.identity, other_key))

    def test_all_targets_keep_exact_S_and_protect_current_search(self):
        plan = self.plan()
        self.assertEqual({"existing": 2, "create": 2, "protected": 2}, plan["counts"])
        for target in plan["targets"]:
            self.assertEqual(6 * target["tokens"], target["target_key"][2])
        self.assertEqual(2, len(plan["sources"]))

    def test_dry_run_does_not_create_target_directories(self):
        plan = self.plan()
        for target in plan["targets"]:
            if target["action"] != "existing":
                self.assertFalse(Path(target["path"]).parent.exists())

    def test_apply_never_overwrites_existing_or_protected_entries(self):
        plan = self.plan()
        existing = {p: p.read_bytes() for p in self.root.glob("*/*")}
        result = cache.apply_plan(plan)
        self.assertEqual({"skipped_existing": 2, "created": 2, "skipped_protected": 2}, result)
        for path, contents in existing.items():
            self.assertEqual(contents, path.read_bytes())
        for target in plan["targets"]:
            path = Path(target["path"])
            if target["action"] == "create":
                self.assertEqual({"best": self.best, "timings": []}, json.loads(path.read_text()))
            if target["action"] == "protected":
                self.assertFalse(path.exists())

    def test_no_clobber_handles_race_after_plan(self):
        plan = self.plan()
        target = next(t for t in plan["targets"] if t["action"] == "create")
        path = Path(target["path"])
        path.parent.mkdir()
        path.write_text("independent native search completed")
        result = cache.apply_plan(plan)
        self.assertEqual(1, result["skipped_race_existing"])
        self.assertEqual("independent native search completed", path.read_text())

    def test_source_change_after_planning_is_rejected(self):
        plan = self.plan()
        for source in plan["sources"].values():
            Path(source["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            cache.apply_plan(plan)

    def test_unmeasured_reused_entry_cannot_become_a_measured_source(self):
        path = self.root / "unmeasured.json"
        path.write_text(json.dumps({"best": self.best, "timings": []}))
        with self.assertRaisesRegex(ValueError, "not a measured"):
            cache.measured_winner(path, [], self.configs)

    def test_missing_or_incompatible_runtime_identity_fails_closed(self):
        changed = {**self.identity, "backend_hash": "cuda100"}
        with self.assertRaisesRegex(ValueError, "No measured native winner"):
            cache.make_plan(changed, self.configs, self.root, [259], [8], {8309})

    def test_invalid_winner_outside_official_grid_is_rejected(self):
        path = self.root / "bad.json"
        bad = {**self.best, "BLOCK_SIZE_N": 999}
        path.write_text(json.dumps({"best": bad, "timings": [[bad, 0.1], [self.best, 0.2]]}))
        with self.assertRaisesRegex(ValueError, "not a measured"):
            cache.measured_winner(path, [], self.configs)

    def test_lengths_are_counted_without_forwarding_input_tokens(self):
        data = self.root / "data"
        data.mkdir()
        (data / "inputs-fit.jsonl").write_text("\n".join(json.dumps({"input_ids": list(range(n))}) for n in (3, 8, 8)))
        (data / "inputs-test.jsonl").write_text(json.dumps({"input_ids": [1, 2, 3, 4]}))
        lengths, bindings, statistics = cache.input_lengths(data)
        self.assertEqual([3, 4, 8], lengths)
        self.assertEqual(3, statistics["inputs-fit"]["records"])
        self.assertEqual(2, statistics["inputs-fit"]["unique_lengths"])
        self.assertTrue(all(len(row["sha256"]) == 64 for row in bindings))

    def test_published_json_is_complete_and_second_publication_cannot_replace_it(self):
        path = self.root / "publication" / "cache.json"
        self.assertTrue(cache.create_absent_cache(path, {"best": self.best, "timings": []}))
        self.assertFalse(cache.create_absent_cache(path, {"bad": True}))
        self.assertEqual(self.best, json.loads(path.read_text())["best"])
        self.assertFalse(list(path.parent.glob("cache-reuse-*")))


if __name__ == "__main__":
    unittest.main()
