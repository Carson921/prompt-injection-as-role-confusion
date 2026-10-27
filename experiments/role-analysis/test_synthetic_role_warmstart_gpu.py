import tempfile
import unittest
from pathlib import Path
import numpy as np

import synthetic_agent_gpu as runtime
import synthetic_role_warmstart_gpu as warm


class LayoutAndBudgetTests(unittest.TestCase):
    def test_gpu_shard_commit_is_accepted_by_training_loader(self):
        from synthetic_role_warmstart_train import WarmBank
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signature = {"input_sha256": "a" * 64, "heads": list(range(16)),
                         "model_binding": {"fixture": "fixed"}, "parent_probe_sha256": "b" * 64,
                         "runtime_sha256": "c" * 64, "extractor_sha256": "d" * 64}
            x = np.arange(2048, dtype=np.float16).reshape(2, 1024)
            arrays = {"x": x, "y": np.array([0, 1], dtype=np.int64), "source": np.array(["one", "two"]),
                      "family": np.array(["first", "second"]), "structure": np.array(["paired:S", "paired:D"]),
                      "weight": np.ones(2, np.float32), "groups": np.array(['{"profile":"paired"}'] * 2)}
            entry = warm._write_shard(root, "fit", 0, arrays, signature, 1)
            runtime.atomic_json(root / "manifest.json", {"format_version": warm.FORMAT, "roles": runtime.ROLES,
                "heads": signature["heads"], "feature_dim": 1024, "feature_position": "pre-o_proj",
                "model_binding": signature["model_binding"], "parent_probe_sha256": signature["parent_probe_sha256"],
                "splits": {"fit": [entry]}, "inputs": {"fit": {"sha256": signature["input_sha256"], "signature": signature}},
                "progress": {"fit": {"complete": True, "rows": 2}}})
            bank = WarmBank(root, "fit")
            np.testing.assert_array_equal(bank.read(np.array([1, 0])), x[[1, 0]])
            (root / entry["path"] / "x.npy").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                WarmBank(root, "fit")

    def test_three_card_mapping_and_legacy_mapping(self):
        for count, width in ((3, 12), (4, 9)):
            mapping = runtime.a40_device_map(count)
            self.assertEqual([mapping[f"model.layers.{i}"] for i in range(36)],
                             [i // width for i in range(36)])
            self.assertEqual(mapping["lm_head"], count - 1)
            self.assertEqual(mapping["model.embed_tokens"], 0)

    def test_unsupported_layout_is_rejected(self):
        with self.assertRaises(RuntimeError):
            runtime.a40_device_map(2)

    def test_budget_is_not_rearmed_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            first = warm.arm_budget(path)
            self.assertEqual(first, warm.arm_budget(path))
            self.assertEqual(first["deadline_unix"] - first["started_unix"], 21600)
            first["deadline_unix"] = first["started_unix"] - 1
            runtime.atomic_json(path, first)
            with self.assertRaises(RuntimeError):
                warm.arm_budget(path)


if __name__ == "__main__":
    unittest.main()
