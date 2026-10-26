"""CPU fixtures for body-token denominators and post-freeze row alignment."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import synthetic_agent_classifier as c
import synthetic_agent_data as data
import synthetic_agent_position_diagnostics as positions


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class MixedBoundaryTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        i = 0
        while i < len(text):
            size = 3 if text.startswith("X\n\n# Tools", i) else 1
            ids.append(1000000 if size == 3 else ord(text[i]))
            offsets.append((i, i + size))
            i += size
        return {"input_ids": ids, "offset_mapping": offsets}


def _record(tokenizer=None, *, mixed=False):
    tokenizer = tokenizer or CharacterTokenizer()
    messages = []
    for i, symbol in enumerate("SDUA"):
        message = {"symbol": symbol, "body": "abcdefgh", "event_id": f"event-{i}",
                   "message_id": f"event-{i}", "round_ix": 0 if i < 2 else 1}
        if mixed and symbol == "D":
            message.update(body="abcdX", tool_definition=True)
        messages.append(message)
    return data.encode_record({"id": "fixture", "source_id": "source", "cluster_id": "cluster",
        "split": "test", "profile": "one_round", "template_id": "SDUA",
        "round_segments": ["UA"], "snapshot_stage": "output", "history_policy": "keep",
        "messages": messages}, tokenizer)


class ArrayBank:
    def __init__(self, records):
        self.y = np.array([target["label"] for record in records for target in record["targets"]])
        self.family = np.array([target["family_id"] for record in records for target in record["targets"]])
        self.source = np.array([record["cluster_id"] for record in records for target in record["targets"]])
        self.structure = np.array([target["structure"] for record in records for target in record["targets"]])
        self.groups = np.array(["{}"] * len(self.y))

    def __len__(self):
        return len(self.y)


def _complete_fixture(root, split="test"):
    tokenizer = CharacterTokenizer()
    record = _record(tokenizer)
    record["split"] = split
    run = root / "run"
    run.mkdir()
    data_dir = root / "data"
    inputs = data_dir / "formal" / f"inputs-{split}.jsonl"
    inputs.parent.mkdir(parents=True)
    inputs.write_text(json.dumps(record) + "\n")
    bank = run / "features/formal"
    shard = bank / f"{split}-000000"
    shard.mkdir(parents=True)
    array_bank = ArrayBank([record])
    count = len(array_bank)
    binding = {"roles": list(c.ROLES), "feature_dim": 1024, "feature_position": "pre-o_proj",
               "heads": list(range(16)), "model_binding": {"fixture": True},
               "selection_sha256": "selection-fixture", "pilot_fit_sha256": "pilot-fixture",
               "data_hash": "data-fixture", "config_hash": "config-fixture"}
    signature = {"model_binding": binding["model_binding"], "heads": binding["heads"],
                 "selection_sha256": binding["selection_sha256"], "pilot_fit_sha256": binding["pilot_fit_sha256"],
                 "config_sha256": binding["config_hash"], "capture": "pre-o_proj", "phase": "formal",
                 "implementation_sha256": "code-fixture"}
    arrays = {"x": np.zeros((count, 1024), dtype=np.float16), "y": array_bank.y,
              "structure": array_bank.structure, "source": array_bank.source, "family": array_bank.family,
              "groups": array_bank.groups, "weight": np.ones(count, dtype=np.float32)}
    hashes = {}
    for key, value in arrays.items():
        path = shard / f"{key}.npy"
        np.save(path, value, allow_pickle=False)
        hashes[path.name] = c._sha(path)
    input_sha = c._sha(inputs)
    c._atomic_json(shard / "complete.json", {"rows": count, "files": hashes,
        "signature": {**signature, "split": split, "input_sha256": input_sha}})
    manifest = {"format_version": c.FEATURE_FORMAT, **binding, "signature": signature,
                "split_sources": {split: input_sha},
                "splits": {split: [{"path": shard.name, "rows": count, "sha256": c._sha(shard / "complete.json")}]},
                "progress": {split: {"complete": True, "rows": count, "records": 1}}}
    c._atomic_json(bank / "manifest.json", manifest)
    metadata = {**binding, "format_version": c.PROBE_FORMAT, "hidden": 0, "dropout": .1,
                "engineering_smoke_only": False}
    np.savez(run / "best.npz", metadata=np.array(json.dumps(metadata)),
             mean=np.zeros(1024, dtype=np.float32), scale=np.ones(1024, dtype=np.float32),
             **{"weight::0.weight": np.zeros((6, 1024), dtype=np.float32), "weight::0.bias": np.zeros(6, dtype=np.float32)})
    c._atomic_json(run / "selection.json", {})
    c._atomic_json(run / "config.json", {})
    feature_bank = c.FeatureBank(bank, split)
    c._atomic_json(run / "training.json", {"phase": "formal", "feature_split_hashes": {split: feature_bank.split_identity_hash}})
    c._atomic_json(run / "freeze.json", {"probe_sha256": c._sha(run / "best.npz"),
        "selection_sha256": c._sha(run / "selection.json"), "training_sha256": c._sha(run / "training.json"), "metadata": metadata})
    probabilities = np.full((count, 6), 1 / 6, dtype=np.float32)
    (run / "evaluations").mkdir()
    np.save(run / "evaluations" / f"{split}-probabilities.npy", probabilities, allow_pickle=False)
    c._atomic_json(run / "evaluations" / f"{split}.json", {
        "probe_sha256": c._sha(run / "best.npz"), "feature_manifest_sha256": c._sha(bank / "manifest.json"),
        "overall": c.metrics(probabilities, array_bank.y)})
    return run, data_dir, bank, tokenizer


class PositionSemantics(unittest.TestCase):
    def test_sparse_masks_use_full_pure_body_denominator(self):
        tokenizer = CharacterTokenizer()
        record = _record(tokenizer)
        record["targets"] = [target for target in record["targets"] if target["body_position"] in (0, 7)]
        mapped, audit = positions._target_positions(record, tokenizer)
        self.assertEqual(len(mapped), 8)
        self.assertEqual({row["body_tokens"] for row in mapped}, {8})
        self.assertEqual([row["body_quartile"] for row in mapped], [0, 3] * 4)
        self.assertEqual(audit["pure_body_tokens"], 32)
        self.assertTrue(all(row["first_25pct"] != row["last_25pct"] for row in mapped))

    def test_body_relative_and_complete_input_positions_differ(self):
        record = _record()
        mapped, audit = positions._target_positions(record, CharacterTokenizer())
        self.assertTrue(any(row["body_quartile"] != row["input_quartile"] for row in mapped))

    def test_mixed_structure_boundary_token_is_not_in_body_denominator(self):
        tokenizer = MixedBoundaryTokenizer()
        record = _record(tokenizer, mixed=True)
        mapped, audit = positions._target_positions(record, tokenizer)
        developer = [row for row in mapped if row["label"] == 1]
        self.assertEqual(len(developer), 4)
        self.assertEqual({row["body_tokens"] for row in developer}, {4})
        self.assertEqual(audit["excluded_mixed_boundary_tokens"], 1)

    def test_reencoding_and_body_ordinal_mismatches_rejected(self):
        record = _record()
        changed = copy.deepcopy(record)
        changed["input_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "input_ids"):
            positions._target_positions(changed, CharacterTokenizer())
        changed = copy.deepcopy(record)
        changed["targets"][0]["body_position"] += 1
        with self.assertRaisesRegex(ValueError, "body_position"):
            positions._target_positions(changed, CharacterTokenizer())

    def test_feature_family_and_label_row_alignment_checked(self):
        record = _record()
        bank = ArrayBank([record])
        bank.family[[0, 1]] = bank.family[[1, 0]]
        with self.assertRaisesRegex(ValueError, "family order"):
            positions._mapped_arrays([record], bank, CharacterTokenizer(), ("profile",))
        bank = ArrayBank([record])
        bank.y[0] = 5
        with self.assertRaisesRegex(ValueError, "labels differ"):
            positions._mapped_arrays([record], bank, CharacterTokenizer(), ("profile",))

    def test_empty_cells_have_na_recall_and_preserve_false_positives(self):
        empty = positions._empty_metrics()
        self.assertIsNone(empty["accuracy"])
        self.assertIsNone(empty["roles"]["tool"]["recall"])
        p = np.array([[0, 0, 0, 0, 0, 1]], dtype=np.float32)
        observed = positions._metrics(p, np.array([2]), np.array([True]))
        self.assertIsNone(observed["roles"]["tool"]["recall"])
        self.assertEqual(observed["roles"]["tool"]["false_positives"], 1)

    def test_missing_evaluation_stops_before_opening_test_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c._atomic_json(root / "freeze.json", {})
            with mock.patch.object(positions, "_load_tokenizer", side_effect=AssertionError("tokenizer opened")):
                with self.assertRaisesRegex(ValueError, "existing completed evaluation"):
                    positions.evaluate_positions(root, root / "unseen-data")

    def test_complete_api_writes_six_role_position_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, data_dir, bank, tokenizer = _complete_fixture(Path(tmp))
            with mock.patch.object(positions, "_load_tokenizer", return_value=(tokenizer, {"unit_test_fixture": True})):
                result = positions.evaluate_positions(run, data_dir)
            self.assertEqual(result["overall"]["tokens"], 32)
            self.assertFalse(result["selection_permitted"])
            self.assertEqual(result["by_body_quartile"]["q4"]["tokens"], 8)
            self.assertEqual(result["by_body_edge"]["last_25pct"]["tokens"], 8)
            self.assertIsNone(result["overall"]["roles"]["cot"]["recall"])
            self.assertIsNone(result["by_body_length"]["1024+"]["overall"]["accuracy"])
            self.assertTrue((run / "evaluations/test-positions.json").exists())

    def test_dev_manifest_append_can_use_frozen_stable_split_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, data_dir, bank, tokenizer = _complete_fixture(Path(tmp), split="dev")
            manifest = c._sha(bank / "manifest.json")
            value = json.loads((bank / "manifest.json").read_text())
            value["last_extraction_seconds"] = 123.4
            value["splits"]["test"] = []
            c._atomic_json(bank / "manifest.json", value)
            with mock.patch.object(positions, "_load_tokenizer", return_value=(tokenizer, {"unit_test_fixture": True})):
                result = positions.evaluate_positions(run, data_dir, split="dev")
            self.assertFalse(result["binding_validation"]["evaluation_manifest_exact"])
            self.assertTrue(result["binding_validation"]["frozen_dev_training_split"])


if __name__ == "__main__":
    unittest.main()
