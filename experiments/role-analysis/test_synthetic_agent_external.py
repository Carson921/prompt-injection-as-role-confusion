"""Tests for the external-test freeze gate and authentic-body-only masks."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import synthetic_agent_external as external


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ExternalBoundaries(unittest.TestCase):
    def _fixture(self, root):
        history = root / "history"
        history.mkdir()
        ledger = {"records": [{"dataset": "oasst", "native_id": "old-tree",
                               "messages": [{"source_message_id": "old-message", "normalized_raw_sha256": "a" * 64}]}]}
        (history / "ledger.json").write_text(json.dumps(ledger))
        # Byte hashing is allowed pre-freeze; decoding a body would fail.
        (history / "legacy.csv").write_bytes(b"assistant\n\xff\n")
        config = {"external": {"history_root": str(history), "history_metadata": ["ledger.json"],
                                "legacy_csv": ["legacy.csv"]}}
        return config, history

    def _frozen_hash_fixture(self, root):
        config, history = self._fixture(root)
        with (history / "legacy.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["dataset", "user_query", "assistant", "cot", "state_text"])
            writer.writeheader()
            writer.writerow({"dataset": "oasst", "user_query": "LOCAL PRIVATE QUESTION ONE",
                             "assistant": "LOCAL PRIVATE RESPONSE ONE", "cot": "LOCAL PRIVATE ANALYSIS"})
            writer.writerow({"dataset": "toxicchat", "user_query": "LOCAL PRIVATE QUESTION TWO",
                             "assistant": "LOCAL PRIVATE RESPONSE TWO",
                             "state_text": "<|start|>assistant<|channel|>final<|message|>LOCAL PRIVATE NATIVE SEGMENT<|return|>"})
        data_dir, run_dir = root / "data", root / "run"
        protocol = external.prepare_external_protocol(config, data_dir, run_dir)
        (run_dir / "best.npz").write_bytes(b"frozen-readout-engineering-fixture")
        (run_dir / "selection.json").write_text("{}")
        frozen = {"probe_sha256": external._file_sha(run_dir / "best.npz"),
                  "selection_sha256": external._file_sha(run_dir / "selection.json"),
                  "external_protocol_pointer_sha256": external._file_sha(run_dir / "external-protocol.json"),
                  "external_protocol_sha256": external._file_sha(data_dir / "external/protocol.json"),
                  "external_history_index_sha256": protocol["history_index_sha256"],
                  "frozen_at_unix": protocol["declared_at_unix"] + 1}
        external._atomic_json(run_dir / "freeze.json", frozen)
        return config, history, data_dir, run_dir, protocol, frozen

    def test_protocol_only_does_not_decode_old_or_new_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history = self._fixture(root)
            with mock.patch.object(external, "_snapshot", side_effect=AssertionError("source opened")):
                protocol = external.prepare_external_protocol(config, root / "data", root / "run")
            self.assertEqual(protocol["scored_roles"], ["user", "assistant"])
            self.assertEqual(protocol["count_per_source"], 128)
            index = json.loads((root / "data/external/history-index.json").read_text())
            self.assertEqual(index["native_ids"], [["oasst", "old-tree"]])

    def test_missing_freeze_stops_before_snapshot_or_legacy_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history = self._fixture(root)
            external.prepare_external_protocol(config, root / "data", root / "run")
            with mock.patch.object(external, "_snapshot", side_effect=AssertionError("source opened")):
                with self.assertRaisesRegex(RuntimeError, "freeze.json"):
                    external.prepare_external(config, CharacterTokenizer(), root / "data", root / "run")

    def test_added_prefixes_are_context_and_separate_diagnostic_targets(self):
        candidate = {"dataset": "oasst", "native_id": "fresh",
                     "messages": [{"symbol": "U", "body": "Question body", "source_message_id": "u"},
                                  {"symbol": "A", "body": "Answer body", "source_message_id": "a"}]}
        protocol = {"prefixes": external.PREFIXES, "prefix_provenance": "added context",
                    "source_pins": external.PINS, "max_input_tokens": 4096}
        record = external._encode(candidate, protocol, CharacterTokenizer())
        self.assertEqual({target["role"] for target in record["targets"]}, {"user", "assistant"})
        self.assertEqual({target["role"] for target in record["prefix_diagnostic_targets"]}, {"system", "developer"})
        self.assertEqual(record["text"].count("Question body"), 1)
        self.assertEqual(record["text"].count("Answer body"), 1)
        self.assertEqual(len(record["targets"]), len("Question bodyAnswer body"))
        self.assertTrue(record["text"].endswith("<|return|>"))
        self.assertEqual(record["external_scoring_scope"], "authentic_UA_only")

    def test_history_index_uses_whitelisted_hashes_and_native_ids(self):
        hashes, ids, messages = set(), set(), set()
        external._collect_history({"dataset": "toxicchat0124", "native_id": "conv",
                                   "raw_sha256": "c" * 64, "accuracy": .99,
                                   "text": "must not be used", "source_snapshot_sha256": "b" * 64},
                                  hashes, ids, messages)
        self.assertEqual(hashes, {"c" * 64})
        self.assertEqual(ids, {("toxicchat", "conv")})

    def test_normalized_body_keys_collapse_compatibility_case_and_space(self):
        first = external._body_keys("Ｆｏｏ\n BAR")
        second = external._body_keys("foo bar")
        self.assertTrue(first & second)

    def test_protocol_seed_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history = self._fixture(root)
            external.prepare_external_protocol(config, root / "data", root / "run")
            changed = copy.deepcopy(config)
            changed["external"]["seed"] = 1
            with self.assertRaisesRegex(ValueError, "seed"):
                external.prepare_external_protocol(changed, root / "data", root / "run")

    def test_copied_protocol_can_bind_a_new_run_before_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history = self._fixture(root)
            protocol = external.prepare_external_protocol(config, root / "data", root / "run-a")
            external.prepare_external_protocol(config, root / "data", root / "run-b")
            pointer = json.loads((root / "run-b/external-protocol.json").read_text())
            self.assertEqual(pointer["protocol_sha256"], external._file_sha(root / "data/external/protocol.json"))

    def test_external_protocol_is_verified_against_model_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "best.npz").write_bytes(b"frozen-model-fixture")
            (root / "selection.json").write_text("{}")
            pointer = {"protocol_sha256": "a" * 64, "history_index_sha256": "b" * 64}
            external._atomic_json(root / "external-protocol.json", pointer)
            freeze = {"probe_sha256": external._file_sha(root / "best.npz"),
                      "selection_sha256": external._file_sha(root / "selection.json"),
                      "external_protocol_pointer_sha256": external._file_sha(root / "external-protocol.json"),
                      "external_protocol_sha256": "a" * 64, "external_history_index_sha256": "b" * 64,
                      "frozen_at_unix": 2}
            external._atomic_json(root / "freeze.json", freeze)
            external._check_freeze(root, {"declared_at_unix": 1, "history_index_sha256": "b" * 64}, "a" * 64)
            (root / "external-protocol.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "pointer"):
                external._check_freeze(root, {"declared_at_unix": 1, "history_index_sha256": "b" * 64}, "a" * 64)

    def test_local_export_contains_only_hashes_and_remote_never_needs_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history, data_dir, run_dir, protocol, frozen = self._frozen_hash_fixture(root)
            result = external.export_legacy_hashes(config, data_dir, run_dir, tokenizer=CharacterTokenizer())
            path = Path(result["path"])
            self.assertFalse(result["historical_text_exported"])
            serialized = path.read_text()
            self.assertNotIn("LOCAL PRIVATE", serialized)
            self.assertNotIn("local private", serialized)
            self.assertNotIn("token_ids", serialized)
            (history / "legacy.csv").unlink()
            hashes, users, audit = external._load_legacy_hashes(path, protocol,
                data_dir / "external/protocol.json", run_dir, frozen)
            self.assertIn(external._sha("LOCAL PRIVATE NATIVE SEGMENT"), hashes)
            self.assertIn(external._sha(external._normalized("LOCAL PRIVATE QUESTION ONE")), users["oasst"])
            self.assertTrue(audit["legacy_192_token_cap_hashes_added"])
            self.assertFalse(audit["historical_csv_read_on_run_host"])

    def test_legacy_export_rejects_changed_csv_and_missing_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history, data_dir, run_dir, protocol, frozen = self._frozen_hash_fixture(root)
            with (history / "legacy.csv").open("a") as stream:
                stream.write("changed bytes\n")
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                external.export_legacy_hashes(config, data_dir, run_dir)
            (run_dir / "freeze.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "freeze.json"):
                external.export_legacy_hashes(config, data_dir, run_dir)

    def test_hash_augmentation_cannot_carry_text_in_hash_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history, data_dir, run_dir, protocol, frozen = self._frozen_hash_fixture(root)
            result = external.export_legacy_hashes(config, data_dir, run_dir)
            path = Path(result["path"])
            record = json.loads(path.read_text())
            record["body_sha256"].append("a text fragment must be rejected")
            external._atomic_json(path, record)
            with self.assertRaisesRegex(ValueError, "irreversible SHA256"):
                external._load_legacy_hashes(path, protocol, data_dir / "external/protocol.json", run_dir, frozen)

    def test_remote_prepare_requires_hash_artifact_before_public_body_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, history, data_dir, run_dir, protocol, frozen = self._frozen_hash_fixture(root)
            # The configured historical path may remain local and inaccessible.
            config["external"]["history_root"] = "/an/inaccessible/local/history/path"
            with mock.patch.object(external, "_extend_legacy_hashes", side_effect=AssertionError("CSV fallback")):
                with mock.patch.object(external, "_snapshot", side_effect=AssertionError("public body opened")):
                    with self.assertRaisesRegex(FileNotFoundError, "Hash-only legacy"):
                        external.prepare_external(config, CharacterTokenizer(), data_dir, run_dir)


if __name__ == "__main__":
    unittest.main()
