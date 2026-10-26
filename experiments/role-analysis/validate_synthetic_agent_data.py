"""Independent CPU checks of article partitions, Harmony masks and confounds.

Uses a deterministic character tokenizer for syntax/property tests, not for any
experiment. Production prepare also checks the actual model tokenizer offsets.
"""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import re
import tempfile
import unittest
from itertools import islice
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import synthetic_agent_data as data


class CharacterTokenizer:
    name_or_path = "cpu-test-character-tokenizer"
    token_pattern = r"<\|[^<>]+\|>|."
    def __init__(self):
        self.vocab = {}
        self.reverse = {}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        for match in re.finditer(self.token_pattern, text, re.S):
            token = match.group()
            if token not in self.vocab:
                index = len(self.vocab)
                self.vocab[token] = index
                self.reverse[index] = token
            ids.append(self.vocab[token])
            offsets.append(match.span())
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.reverse[index] for index in ids)


def articles(source, count=350):
    rng = random.Random(source)
    vocabulary = ["material" + "".join(chr(97 + (index // 26**position) % 26) for position in [2, 1, 0]) for index in range(4000)]
    for index in range(count):
        sentences = []
        for _ in range(14):
            words = rng.sample(vocabulary, 7)
            sentences.append(f"The {words[0]} and {words[1]} of the {words[2]} are {words[3]} with {words[4]} in a {words[5]} system during the {words[6]} process.")
        yield {"text": " ".join(sentences),
               "url": f"https://{source}.example.org/article/{index}?tracking=1"}


class SyntheticDataChecks(unittest.TestCase):
    def test_v7_explicit_source_urls_and_archive_rejection(self):
        page = "https://content.example.org/article/topic"
        for row, field in [({"url": page}, "row.url"),
                           ({"metadata": {"url": page}}, "metadata.url"),
                           ({"source_url": page}, "row.source_url"),
                           ({"metadata": {"source_url": page}}, "metadata.source_url"),
                           ({"original_url": page}, "row.original_url"),
                           ({"metadata": {"original_url": page}}, "metadata.original_url"),
                           ({"warc_url": page}, "row.warc_url"),
                           ({"metadata": {"warc_url": page}}, "metadata.warc_url"),
                           ({"metadata": json.dumps({"warc_url": page})}, "metadata.warc_url")]:
            with self.subTest(field=field, metadata_type=type(row.get("metadata")).__name__):
                result = data.source_url_info(row)
                self.assertTrue(result["available"])
                self.assertEqual(result["field"], field)
                self.assertEqual(result["url"], page)
                self.assertEqual(result["value_sha256"], data.sha(page))
        self.assertEqual(data.source_url_info({"url": page, "metadata": {"url": "https://other.example.org/article"}})["field"], "row.url")
        archives = ["https://data.commoncrawl.org/crawl-data/CC-MAIN/a", "https://commoncrawl.s3.amazonaws.com/a",
                    "https://commoncrawl.s3.us-east-1.amazonaws.com/a", "https://s3.amazonaws.com/commoncrawl/a",
                    "https://archive.example.org/a.warc.gz", "https://archive.example.org/a.warc.wet.gz",
                    "https://archive.example.org/a.warc.wat.gz", "https://archive.example.org/a.wet",
                    "https://archive.example.org/a.wat.zst", "https://archive.example.org/a.arc.gz",
                    "https://archive.example.org/a%2Ewarc%2Ewet.gz", "file:///archive/a.warc"]
        for archive in archives:
            with self.subTest(archive=archive):
                result = data.source_url_info({"metadata": {"warc_url": archive}})
                self.assertFalse(result["available"])
                self.assertIsNone(result["url"])
                self.assertEqual(result["missing_reason"], "no_usable_target_url")
        self.assertTrue(data.source_url_info({"metadata": {"warc_url": archives[0], "source_url": page}})["available"])
        self.assertEqual(data.source_url_info({"metadata": "invalid JSON"})["metadata_parse"], "invalid_json")
        self.assertFalse(data.source_url_info({})["available"])

    def test_v7_cross_corpus_url_dedup_and_partition_validation(self):
        first, second = next(articles("url-first")), next(articles("url-second"))
        first["url"] = "http://shared.example.org/article?tracking=one"
        second.pop("url")
        second["metadata"] = json.dumps({"warc_url": "https://www.shared.example.org/article?tracking=two#fragment"})
        first_url, second_url = data.source_url_info(first), data.source_url_info(second)
        self.assertNotEqual(data.sha(data.normalize_text(first["text"])), data.sha(data.normalize_text(second["text"])))
        seen = data.Deduplicator()
        representative, reason = seen.add(first["text"], first_url["url"])
        self.assertIsNone(reason)
        duplicate, reason = seen.add(second["text"], second_url["url"])
        self.assertIsNone(duplicate)
        self.assertEqual(reason, "duplicate_normalized_url")

        other = next(articles("url-third"))
        other_url = data.source_url_info(other)
        other_info, reason = seen.add(other["text"], other_url["url"])
        self.assertIsNone(reason)
        docs = []
        for index, (info, url) in enumerate([(representative, first_url), (other_info, other_url)]):
            docs.append({**info, "source_id": str(index), "source": ["c4", "dolma3"][index],
                         "split": ["fit", "dev"][index], "source_url_available": True,
                         "source_url_field": url["field"], "source_url_missing_reason": None})
        self.assertTrue(data.validate_source_partitions(docs)["available_canonical_url_disjoint"])
        for key in ["normalized_url", "normalized_sha256", "simhash64"]:
            bad = copy.deepcopy(docs)
            bad[1][key] = bad[0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                data.validate_source_partitions(bad)

    def test_transient_network_retries_resume_without_repeating_rows(self):
        class Http504(OSError):
            response = SimpleNamespace(status_code=504)
        skipped = []
        class Dataset:
            def __init__(self, rows, fail=False):
                self.rows, self.fail = rows, fail
            def __iter__(self):
                for index, row in enumerate(self.rows):
                    if self.fail and index == 1:
                        raise ConnectionError("temporary transport disconnect")
                    yield row
            def skip(self, count):
                skipped.append(count)
                return Dataset(self.rows[count:])
        calls = []
        def loader():
            calls.append(1)
            if len(calls) == 1:
                raise Http504("temporary gateway error")
            return Dataset(["row-0", "row-1", "row-2"], fail=len(calls) == 2)
        sleeps = []
        with patch.object(data.time, "sleep", side_effect=sleeps.append):
            result = list(data.stable_dataset_rows(loader, "mock-fixed-revision"))
        self.assertEqual(result, ["row-0", "row-1", "row-2"])
        self.assertEqual(skipped, [1])
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [5, 5])

    def test_source_retry_is_bounded_and_does_not_swallow_configuration_errors(self):
        class Http504(OSError):
            response = SimpleNamespace(status_code=504)
        calls, sleeps = [], []
        def outage():
            calls.append(1)
            raise Http504("temporary gateway error")
        with patch.object(data.time, "sleep", side_effect=sleeps.append), self.assertRaises(Http504):
            data.source_retry(outage, "mock-outage")
        self.assertEqual(len(calls), 5)
        self.assertEqual(sleeps, [5, 10, 20, 30])
        calls.clear()
        def bad_config():
            calls.append(1)
            raise ValueError("invalid configuration")
        with patch.object(data.time, "sleep") as sleep, self.assertRaises(ValueError):
            data.source_retry(bad_config, "mock-config")
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_balanced_assignment_breaks_six_topic_group_role_period(self):
        groups = [f"topic-{index}" for index in range(6)]
        for corpus in ["c4", "dolma3"]:
            mappings = []
            for block in range(10):
                mapping = tuple(data.balanced_block_choice("SDUCAT", block*6 + index,
                                20261026, "single-role-v1", corpus) for index, _group in enumerate(groups))
                self.assertEqual(set(mapping), set("SDUCAT"))
                mappings.append(mapping)
            self.assertGreater(len(set(mappings)), 1)
            repeated = [data.balanced_block_choice("SDUCAT", ordinal, 20261026, "single-role-v1", corpus)
                        for ordinal in range(60)]
            self.assertEqual(repeated, [value for mapping in mappings for value in mapping])
            for block in range(10):
                self.assertEqual(set(data.balanced_block_choice(data.SEGMENTS, block*5 + index,
                                     20261026, "single-template-v1", corpus) for index in range(5)), set(data.SEGMENTS))

    def test_explicit_hf_file_groups_round_robin_and_prefix_stability(self):
        rows = {"a/*.parquet": [{"text": f"a{index}"} for index in range(2)],
                "b/*.parquet": [{"text": f"b{index}"} for index in range(5)],
                "c/*.parquet": [{"text": "c0"}]}
        calls = []
        def load_dataset(**kwargs):
            calls.append(kwargs)
            return iter(rows[kwargs["data_files"]])
        datasets = ModuleType("datasets")
        datasets.load_dataset = load_dataset
        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub.HfApi = lambda: SimpleNamespace(dataset_info=lambda *args, **kwargs:
            SimpleNamespace(id="test/official-corpus", sha="a" * 40))
        spec = {"dataset": "test/official-corpus", "revision": "a" * 40, "split": "train",
                "data_file_groups": [{"name": group, "data_files": group + "/*.parquet"} for group in ["a", "b", "c"]]}
        with patch.dict("sys.modules", {"datasets": datasets, "huggingface_hub": huggingface_hub}):
            pilot_meta, formal_meta = {}, {}
            pilot = list(islice(data._source_stream("dolma3", spec, Path("."), pilot_meta), 5))
            formal = list(data._source_stream("dolma3", spec, Path("."), formal_meta))
        self.assertEqual([row["text"] for row in formal], ["a0", "b0", "c0", "a1", "b1", "b2", "b3", "b4"])
        self.assertEqual(pilot, formal[:5])
        self.assertEqual(pilot_meta, formal_meta)
        self.assertEqual([row["_synthetic_data_file_group"] for row in formal], ["a", "b", "c", "a", "b", "b", "b", "b"])
        self.assertEqual(formal_meta["dolma3"]["data_file_groups"], spec["data_file_groups"])
        self.assertTrue(all(call["revision"] == "a" * 40 and call["streaming"] for call in calls))
        self.assertEqual(data.normalized_file_groups({"c": "c/*", "a": "a/*", "b": "b/*"}),
                         [{"name": group, "data_files": group + "/*"} for group in ["a", "b", "c"]])
        with self.assertRaises(ValueError):
            data.normalized_file_groups([])
        with self.assertRaises(ValueError):
            data.normalized_file_groups([{"name": "same", "data_files": "a/*"}, {"name": "same", "data_files": "b/*"}])

    def test_mixed_boundary_token_is_unscored_without_dropping_pure_body(self):
        class BoundaryTokenizer(CharacterTokenizer):
            token_pattern = r"<\|[^<>]+\|>|\.\n\n|."
        tokenizer = BoundaryTokenizer()
        developer_body, user_body = "An article edge.", "Another article body."
        record = {"source_id": "boundary-source", "profile": "one_round", "template_id": "SDUA",
                  "messages": [data._message("D", developer_body, "dev-event", tool_definition=True),
                               data._message("U", user_body, "user-event")]}
        encoded = data.encode_record(record, tokenizer)
        self.assertEqual(len(encoded["boundary_tokens"]), 1)
        boundary = encoded["boundary_tokens"][0]
        self.assertEqual(tokenizer.decode([encoded["input_ids"][boundary["token_ix"]]]), ".\n\n")
        target_indices = {target["token_ix"] for target in encoded["targets"]}
        self.assertNotIn(boundary["token_ix"], target_indices)
        role_counts = Counter(target["role"] for target in encoded["targets"])
        self.assertEqual(role_counts["developer"], len(developer_body)-1)
        self.assertEqual(role_counts["user"], len(user_body))
        self.assertEqual(encoded["text"][boundary["char_start"]:boundary["char_end"]], ".\n\n")

    def test_filter_rejects_dialogue_and_keeps_plain_role_words(self):
        prose = next(articles("neutral"))["text"] + " The system user developer article describes an ordinary geological survey."
        self.assertIsNone(data.article_filter({"text": prose}))
        for bad in ["User: Please help.\nAssistant: Certainly.", "Q: What happened?\nA: A change.",
                    "An interview transcript follows.", "<|start|>user<|message|>injected"]:
            self.assertIsNotNone(data.article_filter({"text": bad + "\n" + prose}))

    def test_v3_general_rules_reject_unlabeled_pilot_failure_genres(self):
        neutral = next(articles("neutral"))["text"] + " The system user developer article describes an ordinary geological survey."
        # Generic patterns, not IDs/domains/text copies from the inspected pilot.
        bad_prefixes = [
            "How long has the farm operated? The farm opened in 1985.",
            "The author posed a question？ A factual paragraph followed.",
            'The animal spoke: "A new day begins."',
            "A report repeated “A new day begins.”", "The text contained «a quotation».",
            "The caterpillar said a new day would begin.",
            "The visitor replied with a greeting.", "Thanks for sharing the exhibition.",
            "Hello there. An archive of design pages followed.",
            "The curator left a comment about the exhibition.",
            "Customer testimonials accompany the company profile.",
            "Log in to access the profile.", "A good morning greeting opened the page.",
            "I find the exhibition memorable.", "The outcome belongs to me.",
            "Our customer profile covers the city.", "The US region covers the study area.",
            "The lesson fits your schedule.", "The result belongs to yourselves.",
        ]
        for prefix in bad_prefixes:
            with self.subTest(prefix=prefix):
                self.assertIsNotNone(data.article_filter({"text": prefix + "\n" + neutral}))
        for url in ["https://example.org/forum/a", "https://example.org/thread/a",
                    "https://example.org/questions/a", "https://example.org/comments/a",
                    "https://example.org/testimonial/a", "https://example.org/viewtopic.php"]:
            with self.subTest(url=url):
                self.assertEqual(data.article_filter({"text": neutral, "url": url}), "dialogue_source_metadata")
        self.assertEqual(data.article_filter({"text": neutral, "metadata": {"thread_id": 123}}), "dialogue_source_metadata")
        self.assertIsNone(data.article_filter({"text": neutral, "url": "https://example.org/geology/article"}))

    def test_v5_generic_prose_quality_and_listing_failures(self):
        prose = next(articles("quality"))["text"]
        self.assertIsNone(data.article_filter({"text": prose}))
        prefixes = ["What is the difference between a barrier and a screen\n",
                    "How many components does the assembly contain\n",
                    "Search results\n", "The item was DISCONTINUED.\n",
                    "PNG Prompt\n", "Online casino software\n",
                    "Wash hand basin, double glazed windows, coved ceiling, radiator.\n",
                    "brake hose, brake pedal, brake reservoir, brake fluid, valve cap, valve nut, fluid tank, brake housing, wheel unit, manual lever, brake pipe, pedal position, mounting bolt\n"]
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                self.assertTrue(data.article_filter({"text": prefix + prose}).startswith("v5_"))
        scores = "\n".join(f"Athlete Alpha {i} 12 8 7 2 1 Team Beta {i} 14 3 4 6 0" for i in range(40))
        self.assertTrue(data.prose_rejection(scores).startswith("v5_prose_"))
        self.assertIsNotNone(data.article_filter({"text": scores}))
        repeated = "\n".join("The mineral and the material are contained in the geological system." for _ in range(40))
        self.assertTrue(data.article_filter({"text": repeated}).startswith("v5_prose_repeated_"))
        tutorial = ("Bring the water to the boil and add the vegetables to the pan.\n"
                    "Remove the cover from the cylinder and fill the reservoir to the top level.\n" + prose)
        self.assertIsNone(data.article_filter({"text": tutorial}))

    def test_tokenizer_binding_requires_local_production_bytes(self):
        tokenizer = CharacterTokenizer()
        self.assertEqual(data.tokenizer_file_binding(tokenizer)["mode"], "engineering_only")
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "fixed-backend")
        with self.assertRaises(ValueError):
            data.tokenizer_file_binding(tokenizer)

    def test_v6_download_splice_requires_both_semantic_class_and_repeat_thresholds(self):
        two_classes = "PDF files and free download labels describe an archive."
        three_classes = two_classes + " The archive also offers read online access."
        self.assertFalse(data.download_search_splice(two_classes, .20))
        self.assertFalse(data.download_search_splice(three_classes, .099999))
        self.assertTrue(data.download_search_splice(three_classes, .10))
        self.assertTrue(data.download_search_splice(three_classes, .20))
        self.assertFalse(data.download_search_splice("A PDF is a document format.", .20))

        base = next(articles("v6-prose"))["text"]
        self.assertIsNone(data.article_filter({"text": base + " The PDF contains the published article and its bibliographic record."}))
        self.assertIsNone(data.article_filter({"text": base + " " + three_classes}))
        repeated = ". ".join(base.split(". ")[:3])
        candidate = base + " " + repeated + " " + three_classes
        ratio = data.prose_quality_metrics(candidate)["repeated_fivegram_fraction"]
        self.assertGreaterEqual(ratio, .10)
        self.assertLess(ratio, .25)
        self.assertEqual(data.article_filter({"text": candidate}), "v6_download_search_splice")

    def test_harmony_and_targets(self):
        tokenizer = CharacterTokenizer()
        doc = next(articles("c4"))
        document = dict(source_id="test-source", source="c4", split="fit", source_ordinal=3,
                        cluster_id=data.sha(doc["text"]), text=doc["text"])
        all_records = []
        # Exercise every original template, all independent roles, repeated
        # multi-round segments and pre-tool/post-tool snapshot stages.
        for ordinal in range(60):
            document["source_id"] = f"test-source-{ordinal}"
            document["source_ordinal"] = ordinal
            all_records.extend(data.make_source_records(document, tokenizer, {"seed": 49}))
        self.assertEqual({row["template_id"] for row in all_records if row["profile"] == "one_round"},
                         {"SD" + segment for segment in data.SEGMENTS})
        self.assertTrue(any(len(set(row["round_segments"])) < len(row["round_segments"]) for row in all_records if row["profile"] == "multi_round"))
        self.assertTrue(any(row["snapshot_stage"] == "pre_tool_call" for row in all_records))
        self.assertTrue(any(row["history_policy"] == "drop_completed_round_cot" for row in all_records))
        for record in all_records:
            article_ids = tokenizer.encode(document["text"])
            for span in record["message_spans"]:
                if span["symbol"] == "Q":
                    continue
                fragment = record["article_fragments"][span["event_id"]]
                start, count = fragment["article_token_start"], fragment["article_token_length"]
                self.assertEqual(record["text"][span["body_start"]:span["body_end"]],
                                 tokenizer.decode(article_ids[start:start+count]))
            self.assertLessEqual(len(record["input_ids"]), 4096)
            for target in record["targets"]:
                self.assertEqual(target["role"], data.ROLES[target["label"]])
            if record["history_policy"] == "drop_completed_round_cot":
                self.assertFalse(any(span["symbol"] == "C" and span["round_ix"] < record["rounds"] for span in record["message_spans"]))

    def test_source_and_test_length_assignments_are_not_corpus_confounded(self):
        tokenizer = CharacterTokenizer()
        raw = next(articles("c4"))
        roles, lengths, one_round_templates = defaultdict(set), defaultdict(set), defaultdict(Counter)
        for source in ["c4", "dolma3"]:
            for ordinal in range(12):
                document = dict(source_id=f"{source}-{ordinal}", source=source, split="test",
                                source_ordinal=ordinal, cluster_id=str(ordinal), text=raw["text"])
                records = data.make_source_records(document, tokenizer, {"seed": 91})
                roles[source].update(target["role"] for row in records if row["profile"] == "single" for target in row["targets"])
                lengths[source].update(row["rounds"] for row in records if row["profile"] == "multi_round")
                one_round_templates[source].update(row["template_id"] for row in records if row["profile"] == "one_round" and row["snapshot_stage"] == "output")
        for source in roles:
            self.assertEqual(roles[source], set(data.ROLES))
            self.assertTrue({5, 6}.issubset(lengths[source]))
            self.assertTrue(any(value < 5 for value in lengths[source]))
            values = one_round_templates[source].values()
            self.assertLessEqual(max(values) - min(values), 1)

    def test_pilot_is_subset_of_formal_even_with_cross_corpus_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c4, dolma = list(articles("c4")), list(articles("dolma3"))
            # Cross-source copies at different source positions challenge the
            # interleaved representative ordering as quotas grow.
            for index in range(10, 40, 3):
                dolma.insert(index, c4[index-7])
            url_duplicate = next(articles("metadata-url-copy"))
            url_duplicate.pop("url")
            url_duplicate["metadata"] = json.dumps({"warc_url": c4[2]["url"]})
            dolma.insert(9, url_duplicate)
            for source, rows in [("c4", c4), ("dolma3", dolma)]:
                data.write_jsonl(root / f"{source}.jsonl", rows)
            config = {"seed": 77,
                "sources": {source: {"path": f"{source}.jsonl"} for source in ["c4", "dolma3"]},
                "document_quotas": {"pilot": {"fit": 12, "dev": 6},
                                    "formal": {"fit": 24, "dev": 12, "test": 12}},
                "row_caps": {"pilot": {"fit": 2400, "dev": 1200},
                             "formal": {"fit": 4800, "dev": 2400, "test": 2400}}}
            tokenizer = CharacterTokenizer()
            model_dir = root / "tokenizer"
            model_dir.mkdir()
            for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                (model_dir / name).write_text("{}")
            tokenizer.name_or_path = str(model_dir)
            tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "engineering-only-mock-backend")
            pilot = data.prepare_data(config, tokenizer, root, phase="pilot")
            self.assertGreaterEqual(pilot["filter_counts"]["dolma3"].get("duplicate_normalized_url", 0), 1)
            self.assertEqual(pilot["source_url_counts"]["available"], 18)
            self.assertTrue(pilot["source_partition_validation"]["available_canonical_url_disjoint"])
            self.assertFalse((root / "pilot/inputs-test.jsonl").exists())
            formal = data.prepare_data(config, tokenizer, root, phase="formal")
            old = list(data.read_jsonl(root / "pilot/sources.jsonl"))
            new = list(data.read_jsonl(root / "formal/sources.jsonl"))
            for split in ["fit", "dev"]:
                self.assertTrue({doc["source_id"] for doc in old if doc["split"] == split}.issubset(
                                {doc["source_id"] for doc in new if doc["split"] == split}))
            all_records = [record for split in ["fit", "dev", "test"] for record in data.iter_records(root, "formal", split)]
            audit = data.validate_records(all_records)
            self.assertTrue(audit["source_disjoint"])
            self.assertTrue(audit["family_weight_normalization"])
            self.assertEqual(data.prepare_data(config, tokenizer, root, phase="pilot", resume=True)["config_sha256"], pilot["config_sha256"])
            with self.assertRaises(ValueError):
                data.prepare_data({**config, "seed": 78}, tokenizer, root, phase="pilot", resume=True)
            for key, value in [("min_article_words", 181), ("max_source_candidates", 1000001)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    data.prepare_data({**config, key: value}, tokenizer, root, phase="pilot", resume=True)
            (model_dir / "tokenizer_config.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                data.prepare_data(config, tokenizer, root, phase="pilot", resume=True)
            self.assertLessEqual(pilot["statistics"]["fit"]["rows"], 2400)
            self.assertLessEqual(formal["statistics"]["fit"]["rows"], 4800)


if __name__ == "__main__":
    unittest.main()
