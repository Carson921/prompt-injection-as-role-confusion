"""CPU-only checks for deterministic sampling and the TEST export boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import select_synthetic_article_audit as audit


class AuditSelectionChecks(unittest.TestCase):
    def fixture(self, root):
        seed, rows, pilot = "independent-fixture-seed", [], []
        for corpus in ["c4", "dolma3"]:
            for split in ["fit", "dev"]:
                identifiers = [f"{corpus}-{split}-{index}" for index in range(12)]
                ordered = sorted(identifiers, key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())
                pilot.append({"source_id": ordered[0], "source": corpus, "split": split})
                for source_id in identifiers:
                    row = {"source_id": source_id, "source": corpus, "split": split,
                           "text": "FULL FIT/DEV ARTICLE " + source_id + ". " + "All body characters must remain intact. " * 12}
                    if corpus == "dolma3":
                        row["source_group"] = "rare" if source_id == ordered[-1] else "common"
                    rows.append(row)
        rows.append({"source_id": "reserved-test-identity", "source": "c4", "split": "test",
                     "text": "TEST_BODY_CANARY_DO_NOT_EXPORT_6ee3afec"})
        for phase in ["pilot", "formal"]:
            (root / phase).mkdir()
        sources = root / "formal" / "sources.jsonl"
        sources.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        manifest = {"phase": "formal", "statistics": {"fit": {"sources": 24}, "dev": {"sources": 24}, "test": {"sources": 1}},
                    "sha256": {"sources.jsonl": audit.sha(sources)}}
        (root / "formal" / "data-manifest.json").write_text(json.dumps(manifest))
        (root / "pilot" / "source-keys.json").write_text(json.dumps(pilot))
        (root / "pilot" / "data-manifest.json").write_text(json.dumps({"sha256": {"source-keys.json": audit.sha(root / "pilot" / "source-keys.json")}}))
        return seed, rows, {row["source_id"] for row in pilot}

    def test_order_independence_pilot_exclusion_and_group_supplements(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed, rows, excluded = self.fixture(Path(temporary))
            chosen, statistics = audit.select_rows(rows, seed, 2, excluded)
            reverse, reverse_statistics = audit.select_rows(reversed(rows), seed, 2, excluded)
            self.assertEqual(chosen, reverse)
            self.assertEqual(statistics, reverse_statistics)
            self.assertEqual(len(chosen), 10)  # 4×2 primary + rare Dolma3 group in both splits.
            self.assertEqual(len(statistics["supplement_source_ids"]), 2)
            self.assertTrue(all(row["split"] in {"fit", "dev"} and row["source_id"] not in excluded for row in chosen))
            for value in statistics["strata"].values():
                self.assertEqual(value["eligible"], 11)
                self.assertEqual(value["excluded_pilot"], 1)
                self.assertEqual(value["primary_selected"], 2)
            self.assertTrue(all(value["selected"] > 0 for value in statistics["dolma3_group_split"].values()))

    def test_export_resume_and_test_body_canary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed, rows, _ = self.fixture(root)
            destination = root / "formal" / "audit-sample"
            proof = audit.select_audit(root, "formal", destination, seed, 2, True)
            self.assertEqual(proof, audit.select_audit(root, "formal", destination, seed, 2, True))
            self.assertEqual(set(path.name for path in destination.iterdir()), {"sample.jsonl", "proof.json", "data-manifest.json"})
            for path in destination.iterdir():
                self.assertNotIn(b"TEST_BODY_CANARY_DO_NOT_EXPORT_6ee3afec", path.read_bytes())
            original = {row["source_id"]: row for row in rows}
            for row in audit.source_rows(destination / "sample.jsonl"):
                self.assertEqual(row["text"], original[row["source_id"]]["text"])
            self.assertEqual(audit.sha(destination / "data-manifest.json"), audit.sha(root / "formal" / "data-manifest.json"))
            with self.assertRaisesRegex(ValueError, "different inputs/seed/code"):
                audit.select_audit(root, "formal", destination, seed + "changed", 2, True)

    def test_no_quota_shrink_or_body_bearing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed, rows, excluded = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
                audit.select_rows(rows, seed, 12, excluded)
            manifest_path = root / "formal" / "data-manifest.json"
            manifest = audit.read(manifest_path)
            manifest["embedded_preview"] = {"text": "TEST preview must never be copied"}
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "possible article-body"):
                audit.select_audit(root, "formal", root / "audit", seed, 2, True)


if __name__ == "__main__":
    unittest.main()
