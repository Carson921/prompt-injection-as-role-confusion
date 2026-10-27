"""CPU checks for post-freeze feature reuse and statistical denominators."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

import synthetic_agent_classifier as classifier
import synthetic_role_warmstart_eval as evaluation


def make_probe(path, *, new=False, parent_sha256=None):
    metadata = {"format_version": classifier.PROBE_FORMAT, "roles": list(evaluation.ROLES),
                "feature_dim": 1024, "feature_position": "pre-o_proj", "heads": list(range(16)),
                "model_binding": {"fixture_model": "frozen", "tokenizer": "unchanged"},
                "hidden": 2, "dropout": .1}
    if parent_sha256:
        metadata["warm_start"] = {"parent_probe_sha256": parent_sha256}
    first = np.zeros((2, 1024), np.float32)
    first[0, 0], first[1, 1] = 1, 1
    last = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [.5, .5], [-.5, -.5]], np.float32)
    bias = np.array([.2 if new else 0, .1 if new else 0, 0, 0, 0, 0], np.float32)
    np.savez(path, mean=np.zeros(1024, np.float32), scale=np.ones(1024, np.float32),
             metadata=np.array(json.dumps(metadata)), **{
                 "weight::0.weight": first, "weight::0.bias": np.zeros(2, np.float32),
                 "weight::3.weight": last, "weight::3.bias": bias})
    return classifier.Probe(path)


def fixture(root):
    source = root / "source"
    analysis, calls = source / "analysis", source / "calls"
    analysis.mkdir(parents=True)
    parent_path, new_path = root / "parent.npz", root / "new.npz"
    parent = make_probe(parent_path)
    new = make_probe(new_path, new=True, parent_sha256=evaluation.sha256(parent_path))
    evaluation.write_json(root / "training.json", {"fixture": "article-only training"})
    evaluation.write_json(root / "selection.json", {"selection_split": "dev", "fixture": "candidate selection"})
    freeze = root / "freeze.json"
    evaluation.write_json(freeze, {"format_version": "synthetic-role-warm-freeze-v1",
        "probe_sha256": evaluation.sha256(new_path),
        "parent_probe_sha256": evaluation.sha256(parent_path), "selection_split": "dev",
        "training_sha256": evaluation.sha256(root / "training.json"),
        "selection_sha256": evaluation.sha256(root / "selection.json"),
        "feature_binding": {key: parent.metadata[key] for key in evaluation.FEATURE_KEYS},
        "frozen_at_unix": time.time() - 1})
    evaluation.write_json(source / "probe_provenance.json", {
        "sha256": evaluation.sha256(parent_path), "model_binding": parent.metadata["model_binding"],
        "feature_dim": 1024, "selected_head_ids": list(range(16))})
    masks, p_blocks, ys = [], [], []
    scopes, tags, tasks = [], [], []
    system = "You are a model.\n\nReasoning: medium\n\n# Valid channels: analysis.\nCalls to these tools use commentary."
    developer = "# Instructions\nRules.\n\n# Tools\nnamespace functions { type lookup = () => any; }"
    prompt = system + developer + "User asks."
    for task_index in range(2):
        task = "suite/task_" + str(task_index)
        call = "no_payload/" + task + "/call_0000"
        directory = calls / call
        directory.mkdir(parents=True)
        request = {"status": "complete", "metadata": {"call_id": call},
                   "input_tokens": 7, "generated_ids": [11, 12, 13, 14, 200012],
                   "head_ids": list(range(16)), "roles": list(evaluation.ROLES), "feature_dim": 1024,
                   "probe_sha256": evaluation.sha256(parent_path), "rendered_prompt": prompt}
        evaluation.write_json(directory / "request.json", request)
        rng = np.random.default_rng(task_index)
        x = np.zeros((7, 1024), np.float16)
        x[:, :2] = rng.normal(size=(7, 2))
        offsets = np.array([[0, 3], [17, 20], [len(system), len(system) + 3],
                            [len(system) + 16, len(system) + 20],
                            [len(system) + 25, len(system) + 31],
                            [len(system) + 32, len(system) + 35],
                            [len(system) + len(developer), len(prompt)]])
        ip = parent.predict_proba(x)
        np.savez(directory / "input.npz", head_features=x, roles=np.array(evaluation.ROLES),
                 head_ids=np.arange(16), probabilities=ip, input_ids=np.arange(7),
                 token_index=np.arange(7), token_char_offsets=offsets)
        gx = np.zeros((5, 1024), np.float16)
        gx[:4, :2] = rng.normal(size=(4, 2))
        gx[-1] = np.nan
        gp = np.full((5, 6), np.nan, np.float32)
        gp[:4] = parent.predict_proba(gx[:4])
        np.savez(directory / "generation_probe.npz", head_features=gx, roles=np.array(evaluation.ROLES),
                 head_ids=np.arange(16), probabilities=gp, captured=np.array([True] * 4 + [False]),
                 generated_ids=request["generated_ids"], generated_token_index=np.arange(5),
                 full_sequence_token_index=7 + np.arange(5), token_char_offsets=np.array([[i, i+1] for i in range(5)]),
                 capture_source=np.array(["native_decode_forward"] * 4 + ["not_forwarded_stop_delimiter"]))
        for scope, tag, indices, start, end in [
            ("actual_input", "system", [0, 1], 0, len(system)),
            ("actual_input", "developer", [2, 3, 4, 5], len(system), len(system) + len(developer)),
            ("actual_input", "user", [6], len(system) + len(developer), len(prompt)),
            ("actual_generated", "assistant_analysis", [0, 1], len(prompt), len(prompt) + 2),
            ("actual_generated", "assistant_final", [2, 3], len(prompt) + 2, len(prompt) + 4)]:
            index = np.asarray(indices)
            values = ip[index] if scope == "actual_input" else gp[index]
            row = {"task_key": task, "call_id": call, "scope": scope, "content_filter": "normal_content",
                   "tag": tag, "structural_tool_call": "False", "frame_index": str(len(masks)),
                   "stable_frame_key": "initial/" + tag if scope == "actual_input" else "call_0/" + tag,
                   "body_char_start": start, "body_char_end": end, "tokens": len(index),
                   "token_indices": json.dumps(indices)}
            row.update({"sum_" + role: float(values[:, i].astype(np.float64).sum())
                        for i, role in enumerate(evaluation.ROLES)})
            masks.append(row)
            p_blocks.append(values)
            ys.extend([evaluation.TAG_ROLE[tag]] * len(index))
            scopes.extend([scope] * len(index)); tags.extend([tag] * len(index)); tasks.extend([task] * len(index))
    evaluation.write_csv(analysis / "call_body_probabilities.csv", masks)
    per_task, summary = evaluation.probability_tables(np.array(ys), np.concatenate(p_blocks),
        np.array(scopes), np.array(tags), np.array(tasks))
    evaluation.write_csv(analysis / "summary.csv", [r for r in summary if r["aggregation"] == "unique_task_equal_mean"])
    counts = {}
    for row in masks:
        key = row["scope"] + "/" + row["tag"]
        counts[key] = counts.get(key, 0) + row["tokens"]
    evaluation.write_json(analysis / "metrics.json", {"valid": True, "complete": True,
        "scheduled_unique_tasks": 2, "calls": 2, "input_tokens": 14, "generated_tokens": 10,
        "unforwarded_stop_delimiters": 2, "retained_normal_positions": counts})
    files = [{"path": p.relative_to(source).as_posix(), "bytes": p.stat().st_size,
              "sha256": evaluation.sha256(p)} for p in sorted(source.rglob("*")) if p.is_file()]
    evaluation.write_json(source / "result_manifest.json", {"status": "complete", "files": files})
    return dict(calls_root=calls, analysis_root=analysis, probe_path=new_path,
                parent_path=parent_path, freeze_path=freeze, expected_tasks=2)


class CachedEvaluation(unittest.TestCase):
    def test_complete_reuse_preserves_native_decode_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            result = evaluation.evaluate(**args, output_dir=root / "out", plots=False)
            self.assertEqual(result["tasks"], 2)
            self.assertEqual(result["scored_tokens"], 22)
            proof = evaluation.read_json(root / "out/cache-reuse-audit.json")
            self.assertEqual(proof["counters"]["unforwarded_stop_delimiters"], 2)
            self.assertEqual(proof["generated_capture_sources"]["native_decode_forward"], 8)
            self.assertEqual(proof["unique_sd_template_counts"], {"system": 1, "developer": 1})
            self.assertEqual(proof["parent_probability_max_error"], 0)
            self.assertLess(proof["archived_summary_max_error"], 1e-14)
            archive = root / "out/probabilities/no_payload/suite/task_0/call_0000/actual_generated.npz"
            with np.load(archive, allow_pickle=False) as arrays:
                self.assertTrue(np.isnan(arrays["probabilities"][-1]).all())
                self.assertFalse(arrays["captured"][-1])
            metrics = evaluation.read_json(root / "out/metrics.json")
            self.assertTrue(metrics["native_cached_decode_preserved"])
            self.assertFalse(metrics["new_language_model_forward"])
            self.assertIsNone(metrics["models"]["new"]["overall"]["roles"]["tool"]["recall"])

    def test_requires_frozen_selected_bytes_and_matching_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            freeze = evaluation.read_json(args["freeze_path"])
            freeze["selection_split"] = "test"
            evaluation.write_json(args["freeze_path"], freeze)
            with self.assertRaisesRegex(ValueError, "DEV only"):
                evaluation.evaluate(**args, output_dir=root / "out", plots=False)
            freeze["selection_split"] = "dev"
            freeze["probe_sha256"] = "changed"
            evaluation.write_json(args["freeze_path"], freeze)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                evaluation.evaluate(**args, output_dir=root / "out", plots=False)

    def test_changed_selection_cannot_be_used_with_same_probe_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            evaluation.write_json(root / "selection.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "selection bytes changed"):
                evaluation.evaluate(**args, output_dir=root / "out", plots=False)

    def test_rejects_changed_source_and_overlapping_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            path = args["analysis_root"] / "call_body_probabilities.csv"
            rows = evaluation.csv_rows(path)
            evaluation.write_csv(path, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                evaluation.load_masks(path)
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                evaluation.evaluate(**args, output_dir=root / "out", plots=False)

    def test_probability_mean_is_task_equal_not_token_equal(self):
        p = np.zeros((4, 6), np.float32)
        p[0, 0], p[1:, 1] = 1, 1
        _, means = evaluation.probability_tables(np.zeros(4, np.int64), p,
            np.array(["actual_input"] * 4), np.array(["system"] * 4),
            np.array(["task_one", "task_two", "task_two", "task_two"]))
        lookup = {r["aggregation"]: r for r in means if r["scope"] == "actual_input" and r["tag"] == "system"}
        self.assertEqual(lookup["unique_task_equal_mean"]["p_system"], .5)
        self.assertEqual(lookup["token_pooled_mean"]["p_system"], .25)
        self.assertEqual(lookup["unique_task_equal_mean"]["tasks_observed"], 2)

    def test_segment_partition_uses_offsets_not_probe_outcome(self):
        text = "identity\nReasoning: medium\n# Valid channels: analysis\nCalls to these tools use commentary"
        offsets = np.array([[0, 2], [12, 15], [35, 38], [60, 64]])
        segments = evaluation.sd_segments("system", text, 0, len(text), offsets)
        self.assertEqual(segments.tolist(), ["system/identity_and_date", "system/reasoning_setting",
                                            "system/channel_rules", "system/tool_routing"])
        p = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32)
        report = evaluation.metric_report(np.array([2, 5, 2]), p)
        self.assertEqual(report["user_tool_to_system_developer"]["false_positives"], 2)
        self.assertAlmostEqual(report["user_tool_to_system_developer"]["rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
