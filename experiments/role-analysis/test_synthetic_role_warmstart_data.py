"""Semantic CPU checks for article-only warm-start preparation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import synthetic_role_warmstart_data as warm
import synthetic_agent_data as base


class CharacterTokenizer:
    """Engineering tokenizer with exact offsets and native control spans."""
    name_or_path = "engineering-character-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def document(index=0, split="fit", source="c4", length=2048):
    sid = f"{source}-{split}-{index}"
    body = (f"The passage {index} describes the movement of materials through a natural process. " * 64)[:length]
    fingerprint = hashlib.sha256(sid.encode()).hexdigest()
    return {"source_id": sid, "source": source, "split": split, "text": body,
            "text_sha256": base.sha(body), "cluster_id": fingerprint,
            "normalized_sha256": fingerprint, "normalized_url": f"https://example.org/{sid}",
            "source_url_available": True, "source_url_field": "row.url", "source_url_missing_reason": None,
            "simhash64": fingerprint[:16], "paired_length": 256}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_paired_has_same_body_and_identical_scored_positions(self):
        records = warm.make_records(document(), self.tokenizer, 0)
        paired = records[:6]
        self.assertEqual({r["template_id"] for r in paired}, set("SDUCAT"))
        self.assertEqual({len(r["targets"]) for r in paired}, {96})
        positions = [{(t["body_char_start"], t["body_char_end"]) for t in r["targets"]} for r in paired]
        self.assertTrue(all(p == positions[0] for p in positions))
        for r in paired:
            self.assertEqual([sum(t["body_quartile"] == q for t in r["targets"]) for q in range(4)], [24] * 4)
        self.assertTrue(warm.audit_records(records, self.tokenizer)["valid"])

    def test_tool_structure_and_call_json_are_never_supervised(self):
        for ordinal in range(21):
            records = warm.make_records(document(ordinal), self.tokenizer, ordinal)
            context = records[-1]
            self.assertTrue(warm.audit_records(records, self.tokenizer)["valid"])
            d = next(s for s in context["message_spans"] if s["symbol"] == "D")
            article = context["text"][d["body_start"]:d["body_end"]]
            self.assertNotIn("namespace functions", article)
            for target in context["targets"]:
                position = target["token_ix"]
                left, right = position, position + 1
                spans = [s for s in context["message_spans"] if s["body_start"] <= left < right <= s["body_end"]]
                self.assertEqual(len(spans), 1)
                self.assertNotEqual(spans[0]["symbol"], "Q")
                self.assertEqual(target["role"], spans[0]["role"])

    def test_short_article_cannot_claim_long_bin(self):
        with self.assertRaisesRegex(ValueError, "fewer than 1024"):
            warm.make_records(dict(document(length=300), paired_length=1024), self.tokenizer)

    def test_no_message_middle_truncation(self):
        messages = [{"symbol": "D", "event_id": "d", "body": "x" * 5000}]
        with self.assertRaises(OverflowError):
            warm.encode({"source_id": "s", "profile": "paired", "template_id": "D",
                         "paired_length": 1024, "wrapper_type": "plain"}, self.tokenizer, messages)

    def test_only_complete_balanced_length_qualified_source_cells(self):
        old = [document(i, split, source) for split in ("fit", "dev") for source in ("c4", "dolma3") for i in range(16)]
        fresh = [document(i + 100, "test", source) for source in ("c4", "dolma3") for i in range(8)]
        selected = warm.select_documents(old, fresh, self.tokenizer, fit_count=32, dev_count=16, test_count=16)
        fit = [d for d in selected if d["split"] == "fit"]
        self.assertEqual(len(fit), 32)
        self.assertEqual({(d["source"], d["paired_length"]) for d in fit[:16]}, {(s, n) for s in ("c4", "dolma3") for n in warm.LENGTHS})
        self.assertEqual(len({d["source_id"] for d in selected}), len(selected))
        self.assertEqual({d["origin"] for d in selected if d["split"] == "test"}, {"fresh_pinned_stream"})

    def test_old_test_is_not_recycled_as_new_test(self):
        old = [document(i, split, source) for split in ("fit", "dev", "test") for source in ("c4", "dolma3") for i in range(16)]
        with self.assertRaisesRegex(ValueError, "test/c4"):
            warm.select_documents(old, [], self.tokenizer, fit_count=16, dev_count=16, test_count=16)

    def test_historical_near_duplicates_are_excluded_without_reading_history_body(self):
        old = document()
        old.update(normalized_sha256=base.sha(base.normalize_text(old["text"])), simhash64=f"{base.simhash(old['text']):016x}")
        dedup = warm._seed_deduplicator([old])
        _, reason = dedup.add(old["text"], old["normalized_url"])
        self.assertEqual(reason, "duplicate_normalized_text")

    def test_source_provenance_audit_rejects_modified_body(self):
        doc = document()
        records = warm.make_records(doc, self.tokenizer)
        self.assertTrue(warm.audit_records(records, self.tokenizer, [doc])["original_source_slices_verified"])
        records[0]["message_spans"][0]["article_provenance"]["body_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source provenance"):
            warm.audit_records(records, self.tokenizer, [doc])

    def test_wrapper_is_not_a_deterministic_system_length_alias(self):
        assignments = {}
        for ordinal in range(0, 48, 2):
            context = warm.make_records(document(ordinal), self.tokenizer, ordinal)[-1]
            assignments.setdefault(context["wrapper_type"], set()).add(context["context_lengths"]["system"])
        self.assertEqual(set(assignments), set(warm.WRAPPERS))
        self.assertTrue(all(len(lengths) > 1 for lengths in assignments.values()))

    def test_quality_gate_rejects_page_lists_but_not_coherent_topics(self):
        event = "The Gallery is hosting an exhibition. Call 021-35373582 for more information.\n" * 3
        gallery = "Door seal was posted January 2. The photograph has viewed by 4091 visitor.\n" * 3
        cards = "Coil profile. The conductor provides... otherToroidal Inductors. It has a shapeThree-Phase Transformers... Featured Industries"
        self.assertEqual(warm.warm_article_rejection(event), "repeated_event_listing_phone_notices")
        self.assertEqual(warm.warm_article_rejection(gallery), "repeated_posted_view_count_gallery")
        self.assertEqual(warm.warm_article_rejection(cards), "truncated_related_cards_and_merged_titles")
        self.assertIsNone(warm.warm_article_rejection(event.splitlines()[0]))
        self.assertIsNone(warm.warm_article_rejection("The company designs and manufactures electric coils. Its production plant is located beside the river."))
        listing = "Five bedrooms and a covered deck. Listing provided courtesy of Realty. Buyer to verify all information. Multiple Listing Service. Properties may or may not be listed by the agent."
        self.assertEqual(warm.warm_article_rejection(listing), "property_listing_with_repeated_mls_disclaimers")
        self.assertIsNone(warm.warm_article_rejection("The study examined data from the multiple listing service to investigate regional housing prices."))
        inventory = "Rental property: 3 bedrooms, 2 bathrooms, 2 living rooms and 1 car garage. The kitchen and bedroom have a covered deck. Tenants pay utilities and the landlord owns the building."
        self.assertEqual(warm.warm_article_rejection(inventory), "property_transaction_room_inventory")
        news = "A report compared rental property regulations and tenant rights. " + "The economic study described population changes and lending conditions. " * 12 + inventory
        self.assertIsNone(warm.warm_article_rejection(news))

    def test_manual_semantic_reject_applies_to_matching_body_across_splits(self):
        first = document(1)
        alias = dict(first, source_id="other-source-id", cluster_id="other-cluster", split="test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "old.jsonl").write_text(json.dumps(first) + "\n")
            (root / "extra.jsonl").write_text(json.dumps(alias) + "\n")
            (root / "manual.json").write_text(json.dumps({"rejected_sources": [{"source_id": first["source_id"],
                "text_sha256": first["text_sha256"], "decision": "uncertain_reject", "reason": "article qualification uncertain"}]}))
            result = warm.quality_exclusions([first], [alias], root / "old.jsonl", root / "extra.jsonl", root / "manual.json")
        self.assertEqual(set(result["excluded_clusters"]), {first["cluster_id"], alias["cluster_id"]})


if __name__ == "__main__":
    unittest.main()
