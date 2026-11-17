"""CPU-only fixtures for the load-integrity evidence auditor."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("audit_deepseek_load", HERE / "audit_deepseek_load.py")
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LoadAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Match the official checkpoint's complete key geometry without reading
        # weights or importing Torch. One fake shard suffices for manifest tests.
        keys = {key: "weights.safetensors" for key in audit_module.TOP_KEYS}
        for layer in range(43):
            for expert in range(256):
                for projection in ("w1", "w2", "w3"):
                    for kind in ("weight", "scale"):
                        keys[f"layers.{layer}.ffn.experts.{expert}.{projection}.{kind}"] = "weights.safetensors"
            nonexpert = 29 if layer < 2 else (40 if layer % 2 == 0 else 33)
            keys.update({f"layers.{layer}.nonexpert.{i}": "weights.safetensors" for i in range(nonexpert)})
        keys.update({f"mtp.fixture.{i}": "weights.safetensors" for i in range(4705)})
        cls.index = {"weight_map": keys}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.model = self.root / "model"
        self.run.mkdir()
        self.model.mkdir()
        self.config = {"num_hidden_layers": 43, "n_routed_experts": 256,
                       "num_attention_heads": 64, "head_dim": 512,
                       "hidden_size": 4096, "moe_intermediate_size": 2048}
        config_sha = write_json(self.model / "config.json", self.config)
        index_sha = write_json(self.model / "model.safetensors.index.json", self.index)
        write_json(self.run / "config.json", {"model": str(self.model)})
        expert_bytes = 43 * 256 * 3 * 4096 * 2048 // 2
        counts = {"torch.int8": expert_bytes, "torch.float8_e8m0fnu": expert_bytes // 16,
                  "torch.float8_e4m3fn": 4096, "torch.bfloat16": 4096}
        self.backend = {
            "model_path": str(self.model), "config_sha256": config_sha, "index_sha256": index_sha,
            "backend_source_sha256": hashlib.sha256((HERE / "deepseek_role_probe_model.py").read_bytes()).hexdigest(),
            "parameters_frozen": True, "all_parameters_on_cuda": True, "native_packed_fp4_fp8": True,
            "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4", "scale_fmt": "ue8m0",
                                    "dequantize": False, "weight_block_size": [128, 128]},
            "expert_backends": ["FP8Experts"], "experts_implementation": "batched_mm",
            "attention_implementation": "deepseek_role_probe_tiled_eager", "dtype_bytes": counts,
            "parameter_bytes_total": sum(counts.values()),
            "parameter_bytes_by_device_dtype": {f"cuda:{i}": {k: v // 4 for k, v in counts.items()} for i in range(4)},
            "layers": 43, "heads_per_layer": 64, "head_dim": 512,
            "capture": "attention-head-output-before-o_a_proj", "capture_after_inverse_rope": True,
            "attention_query_tile_size": 256, "device_map": {f"model.layers.{i}": i % 4 for i in range(43)},
        }
        self.forwards = [{"tokens": n, "all_layers_captured": 43, "hook_output_bitexact": True,
                          "hook_output_max_abs_difference": 0.0, "seconds": 1.0} for n in (11, 8309)]
        self.preflight = {"status": "passed", "checks": deepcopy(self.forwards),
                          "backbone_audit": deepcopy(self.backend), "maximum_input_tokens": 8309,
                          "batch_size": 1, "use_cache": False, "lm_head_skipped": True}
        self.verification = {"status": "passed", "safetensors_bytes": 100,
                             "files": [{"path": "config.json", "sha256": config_sha},
                                       {"path": "model.safetensors.index.json", "sha256": index_sha},
                                       {"path": "weights.safetensors", "size": 100,
                                        "official_lfs_verified": True, "sha256": "a" * 64}]}
        self.log_start = json.dumps({"stage": "loading_deepseek_v4"}) + "\n"
        self.log_loaded = self.log_start + "Loading weights: 100%|████| 1736/1736 [10:00<00:00]\n" + json.dumps(
            {"stage": "deepseek_v4_loaded", "parameter_bytes": self.backend["parameter_bytes_total"]}) + "\n"
        self.log_complete = self.log_loaded + "".join(json.dumps({"stage": "preflight_forward", **item}) + "\n" for item in self.forwards)
        self.write_evidence()

    def write_evidence(self):
        write_json(self.run / "model_backend.json", self.backend)
        write_json(self.run / "preflight.json", self.preflight)
        write_json(self.run / "weight-verification.json", self.verification)
        (self.run / "run.log").write_text(self.log_complete)

    def result(self):
        return audit_module.audit(self.run)

    def assert_state(self, expected, check=None):
        result = self.result()
        failures = [item["name"] for item in result["checks"] if item["status"] != "passed"]
        self.assertEqual(expected, result["status"], failures)
        if check is not None:
            self.assertIn(check, failures)
        return result

    def test_complete_checkpoint_and_preflight_pass(self):
        result = self.assert_state("passed")
        inventory = result["index_inventory"]
        self.assertEqual((72317, 4705, 67612, 66048, 1736), tuple(inventory[k] for k in (
            "checkpoint_keys", "ignored_mtp_keys", "backbone_checkpoint_keys",
            "routed_expert_checkpoint_keys", "expected_loaded_keys")))
        self.assertEqual(hashlib.sha256(self.log_complete.encode()).hexdigest(), result["inputs"]["run_log"]["sha256"])
        self.assertEqual(len(self.log_complete.encode()), result["inputs"]["run_log"]["bytes"])

    def test_partial_progress_without_completion_is_running(self):
        (self.run / "run.log").write_text(self.log_start + "Loading weights: 25%|▎| 434/1736\n")
        (self.run / "model_backend.json").unlink()
        (self.run / "preflight.json").unlink()
        self.assert_state("running", "weights_progress_complete")

    def test_progress_complete_without_loaded_event_remains_running(self):
        (self.run / "run.log").write_text(self.log_start + "Loading weights: 100%|█| 1736/1736\n")
        self.assert_state("running", "loaded_event_present")

    def test_preflight_artifact_requires_current_logged_completion(self):
        (self.run / "run.log").write_text(self.log_loaded)
        self.assert_state("running", "preflight_completion_in_log")

    def test_restarted_load_cannot_reuse_old_completion(self):
        (self.run / "run.log").write_text(self.log_complete + self.log_start + "Loading weights: 1%|▏| 17/1736\n")
        self.assert_state("running", "loaded_event_present")

    def test_report_of_randomly_initialized_missing_key_fails(self):
        with (self.run / "run.log").open("a") as stream:
            stream.write("DeepseekV4ForCausalLM LOAD REPORT\nmodel.layers.0.input_layernorm.weight | MISSING\n")
        self.assert_state("failed", "load_diagnostics_absent")

    def test_missing_official_expert_key_fails_even_if_total_is_maintained(self):
        changed = deepcopy(self.index)
        del changed["weight_map"]["layers.7.ffn.experts.13.w2.weight"]
        changed["weight_map"]["layers.7.extra.weight"] = "weights.safetensors"
        write_json(self.model / "model.safetensors.index.json", changed)
        self.assert_state("failed", "official_backbone_key_inventory")

    def test_wrong_checkpoint_sha_fails(self):
        self.backend["config_sha256"] = "b" * 64
        self.write_evidence()
        self.assert_state("failed", "backend_config_binding")

    def test_offload_or_dequantization_fails(self):
        self.backend["device_map"]["model.layers.42"] = "cpu"
        self.backend["quantization_config"]["dequantize"] = True
        self.write_evidence()
        self.assert_state("failed", "four_gpu_device_map")

    def test_native_expert_byte_count_must_match_full_geometry(self):
        self.backend["dtype_bytes"]["torch.int8"] -= 512
        self.write_evidence()
        self.assert_state("failed", "backend_packed_expert_byte_count")

    def test_unverified_shard_fails(self):
        self.verification["files"][-1]["official_lfs_verified"] = False
        self.write_evidence()
        self.assert_state("failed", "all_indexed_shards_official_sha_verified")

    def test_missing_verification_is_running(self):
        (self.run / "weight-verification.json").unlink()
        self.assert_state("running", "weight_verification_present")

    def test_short_preflight_fails(self):
        self.preflight["maximum_input_tokens"] = 11
        self.write_evidence()
        self.assert_state("failed", "preflight_long_input")

    def test_changed_preflight_artifact_fails_logged_agreement(self):
        self.preflight["checks"][1]["seconds"] += 1
        self.write_evidence()
        self.assert_state("failed", "preflight_log_artifact_agreement")

    def test_later_classifier_failure_does_not_invalidate_loaded_weights(self):
        with (self.run / "run.log").open("a") as stream:
            stream.write("Traceback (most recent call last):\nRuntimeError: later classifier\n")
        self.assert_state("passed")

    def test_snapshot_remains_bound_after_append(self):
        path = self.run / "run.log"
        contents, binding = audit_module.snapshot(path)
        with path.open("a") as stream:
            stream.write("new scan stage\n")
        self.assertEqual(hashlib.sha256(contents).hexdigest(), binding["sha256"])
        self.assertEqual(len(contents), binding["bytes"])
        self.assertGreater(path.stat().st_size, binding["bytes"])

    def test_atomic_result_can_be_repeated(self):
        path = self.run / "load-integrity-audit.json"
        audit_module.write_result(path, self.result())
        audit_module.write_result(path, self.result())
        self.assertEqual("passed", json.loads(path.read_text())["status"])
        self.assertEqual([], list(self.run.glob("*.tmp-*")))


if __name__ == "__main__":
    unittest.main()
