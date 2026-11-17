"""CPU integration checks of the DeepSeek runner's scientific boundaries."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import deepseek_role_probe_train as trainer
import run_deepseek_role_probe as runner


ROLES = ["system", "user"]
CONFIG = {"scan_statistics_sources_per_stratum": 2, "scan_inner_sources_per_stratum": 2,
          "scan_rows_per_domain_role": 8, "scan_checkpoint_records": 2,
          "extraction_shard_rows": 4}
SMOKE = {"classifier": {"device": "cpu", "allow_cpu_smoke": True, "batch_size": 8,
                        "min_epochs": 1, "max_epochs": 1, "patience": 1,
                        "checkpoint_steps": 1, "bootstrap_replicates": 2,
                        "cache_features_on_gpu": False}}


def input_record(split, domain, index):
    cluster = f"{split}:{domain}:source-{index}"
    return {"id": f"{cluster}:record", "split": split, "domain": domain,
            "source_id": cluster, "source_cluster": cluster, "input_ids": [1, 2, 3],
            "targets": [{"token_ix": role, "label": role, "role": name,
                         "family_id": f"{cluster}:position-{role}"} for role, name in enumerate(ROLES)]}


def write_inputs(root, count=8):
    root.mkdir()
    for split in runner.SPLITS:
        with (root / f"inputs-{split}.jsonl").open("w") as stream:
            for domain in trainer.DOMAINS:
                for index in range(count):
                    stream.write(json.dumps(input_record(split, domain, index)) + "\n")


class RunnerBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_head_scan_plan_only_reads_fit_and_separates_source_clusters(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            path = root / "inputs-fit.jsonl"
            records = [input_record("fit", domain, index) for domain in trainer.DOMAINS for index in range(80)]
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            plan = runner.make_scan_plan(path, ROLES, CONFIG)
            self.assertFalse(set(plan["clusters"]["statistics"]) & set(plan["clusters"]["inner"]))
            self.assertTrue(plan["no_dev_or_test"])
            for partition in ("statistics", "inner"):
                for row in plan["records"][partition]:
                    record = records[int(row)]
                    self.assertEqual(record["split"], "fit")
                    self.assertIn(record["source_cluster"], plan["clusters"][partition])
                for domain in trainer.DOMAINS:
                    for role in ROLES:
                        self.assertGreater(plan["counts"][f"{partition}/{domain}/{role}"], 0)

    def test_full_covariance_keeps_float64_and_restores_without_new_updates(self):
        import torch
        with mock.patch.multiple(runner.backend, LAYERS=2, HEADS=3, HEAD_DIM=4):
            values = torch.arange(8 * 3 * 4, dtype=torch.float32).reshape(8, 3, 4) / 7
            labels = np.asarray([0, 1] * 4)
            moments = runner.OnlineMoments(2)
            for layer in range(2):
                moments.update(layer, values + layer, labels)
            arrays = moments.export()
            self.assertEqual(arrays["xx"].dtype, np.float64)
            self.assertEqual(arrays["sum"].shape, (2, 3, 2, 4))
            expected = np.einsum("nhd,nhe->hde", values.numpy().astype(np.float64), values.numpy().astype(np.float64))
            np.testing.assert_allclose(arrays["xx"][0], expected, rtol=1e-13)
            # Simulates a stop after the final periodic statistics checkpoint.
            restored = runner.OnlineMoments(2, arrays).export()
            for name in arrays:
                np.testing.assert_array_equal(restored[name], arrays[name])
            original_tensor, original_eye = torch.as_tensor, torch.eye

            def cpu_tensor(value, **kwargs):
                kwargs["device"] = "cpu"
                return original_tensor(value, **kwargs)

            def cpu_eye(*args, **kwargs):
                kwargs["device"] = "cpu"
                return original_eye(*args, **kwargs)

            # Exercise the runner's real full-covariance solve without CUDA.
            with mock.patch.object(torch, "as_tensor", side_effect=cpu_tensor), mock.patch.object(torch, "eye", side_effect=cpu_eye):
                weight, bias = runner.fit_lda(arrays, 2)
            self.assertEqual(weight.shape, (6, 2, 4))
            self.assertEqual(weight.dtype, np.float64)
            self.assertEqual(bias.dtype, np.float64)
            self.assertTrue(np.isfinite(weight).all())

    def test_sixteen_head_schema_and_sealed_test_extraction(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary)
            data = output / "data"
            write_inputs(data)
            selection = {"heads": list(range(16))}
            runner.write(output / "head-selection.json", selection)
            binding = {"model_id": "test-fixture"}

            def selected(model, input_ids, positions, heads):
                return np.tile(np.arange(len(heads) * 3, dtype=np.float16), (len(positions), 1))

            with mock.patch.multiple(runner.backend, HEAD_DIM=3), mock.patch.object(runner.backend, "extract_selected", side_effect=selected):
                runner.extract(object(), data, output, CONFIG, ROLES, binding, selection, ("fit", "dev"))
                fit = trainer.FeatureBank(output / "features", "fit")
                dev = trainer.FeatureBank(output / "features", "dev", expected_binding=fit.binding())
                self.assertEqual(fit.feature_dim, 48)
                self.assertEqual(len(fit.manifest["heads"]), 16)
                trainer.assert_disjoint(fit, dev)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    runner.extract(object(), data, output, CONFIG, ROLES, binding, selection, ("test",))
                # A similarly named ordinary file must not release TEST.
                (output / "checkpoint.npz").write_bytes(b"not a frozen probe")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    runner.extract(object(), data, output, CONFIG, ROLES, binding, selection, ("test",))

    def test_late_test_extraction_matches_frozen_trainer_interface(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary)
            data = output / "data"
            write_inputs(data, count=2)
            selection = {"heads": list(range(16))}
            runner.write(output / "head-selection.json", selection)
            binding = {"model_id": "test-fixture"}

            def selected(model, input_ids, positions, heads):
                return np.tile(np.arange(len(heads) * 3, dtype=np.float16), (len(positions), 1))

            with mock.patch.multiple(runner.backend, HEAD_DIM=3), mock.patch.object(runner.backend, "extract_selected", side_effect=selected):
                runner.extract(object(), data, output, CONFIG, ROLES, binding, selection, ("fit", "dev"))
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.train_candidates(output / "features", output, SMOKE)
                frozen = trainer.freeze(output)
                runner.extract(object(), data, output, CONFIG, ROLES, binding, selection, ("test",))
                result = trainer.evaluate(output / "features", output, "test")
                self.assertEqual(result["probe_sha256"], frozen["probe_sha256"])
                with mock.patch.object(trainer, "predict", side_effect=AssertionError("Second TEST forward")):
                    self.assertEqual(trainer.evaluate(output / "features", output, "test"), result)

    def test_main_resume_with_frozen_probe_never_retrains(self):
        import torch
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary)
            data = output / "data"
            data.mkdir()
            runner.write(data / "manifest.json", {"roles": ROLES})
            runner.write(output / "freeze.json", {"fixture": True})
            (output / "checkpoint.npz").write_bytes(b"fixture immutable checkpoint")
            model = object()
            with mock.patch.object(sys, "argv", ["runner", "--output", str(output), "--model", str(output / "model"), "--skip-weight-hashes"]), \
                 mock.patch.object(runner.backend, "load_model", return_value=model), \
                 mock.patch.object(runner.backend, "audit_model", return_value={}), \
                 mock.patch.object(runner.backend, "memory_report", return_value=[]), \
                 mock.patch.object(runner, "preflight"), \
                 mock.patch.object(runner, "model_binding", return_value={"model_id": "fixture"}), \
                 mock.patch.object(runner, "select_heads", return_value={"heads": list(range(16))}), \
                 mock.patch.object(runner, "extract") as extract, \
                 mock.patch.object(trainer, "verify_freeze") as verify, \
                 mock.patch.object(trainer, "train_candidates") as train, \
                 mock.patch.object(trainer, "freeze") as freeze, \
                 mock.patch.object(trainer, "evaluate") as evaluate, \
                 mock.patch.object(torch.cuda, "empty_cache"):
                runner.main()
            verify.assert_called_once_with(output)
            train.assert_not_called()
            freeze.assert_not_called()
            evaluate.assert_called_once_with(output / "features", output, "test")
            self.assertEqual([call.args[-1] for call in extract.call_args_list], [("fit", "dev"), ("test",)])


if __name__ == "__main__":
    unittest.main()
