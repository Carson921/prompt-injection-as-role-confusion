"""Independent CPU fixtures; no real checkpoint, feature bank, or TEST input."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

import evaluate_synthetic_gardening as diagnostic
import synthetic_gardening_plots as plots


MESSAGE_ROLES = ("system", "user", "cot", "assistant", "user", "cot", "assistant")
MESSAGE_ROWS = (57, 45, 83, 330, 52, 96, 253)
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def fixture(overlap=False):
    """Invented engineering characters with the public example's token counts."""
    samples = []
    for key in plots.FORMATS:
        text, ids, offsets, rows = "", [], [], []

        def append_token(piece, token_id):
            nonlocal text
            start = len(text)
            text += piece
            ids.append(token_id)
            offsets.append([start, len(text)])

        append_token("<BOS>", 5000)
        if key == "everything_in_user_tags":
            append_token("<user>", 5001)
        for message_ix, (role, count) in enumerate(zip(MESSAGE_ROLES, MESSAGE_ROWS)):
            if key == "proper_tags":
                append_token("\n<" + role + ">", 6000 + message_ix)
            elif message_ix:
                append_token("\n", 7000 + message_ix)
            for body_ix in range(count):
                final = message_ix == 6 and body_ix == count - 1
                piece = "" if final else "🌱" if message_ix == 6 and body_ix == count - 2 else chr(97 + message_ix)
                append_token(piece, 109 if final else 10000 + len(rows))
                if final and overlap:
                    offsets[-1] = list(offsets[-2])
                rows.append({"token_position": len(ids) - 1, "token_id": ids[-1], "message_ix": message_ix,
                             "semantic_role": role, "label": plots.ROLES.index(role),
                             "formal_role": role if key == "proper_tags" else "user" if key == "everything_in_user_tags" else None,
                             "in_old_915_comparison": not final})
            if key == "proper_tags":
                append_token("</message>", 8000 + message_ix)
        if key == "everything_in_user_tags":
            append_token("</user>", 9000)
        samples.append({"prompt_key": key, "text": text, "input_ids": ids, "token_offsets": offsets, "content": rows})
    models = {model: {"label": "CPU engineering fixture " + model,
                       "candidate_id": "mlp256-seed29" if model == "pilot" else "linear-seed17",
                       "heads": list(range(16)), "checkpoint_sha256": hashlib.sha256(model.encode()).hexdigest()}
              for model in plots.MODELS}
    results = {"roles": list(plots.ROLES), "models": models, "formats": {}, "engineering_only": True}
    for sample in samples:
        section = {"metric_kind": "formal_role" if sample["prompt_key"] == "proper_tags" else "original_semantic_alignment", "models": {}}
        for model in plots.MODELS:
            p = np.full((916, 6), .01, np.float32)
            p[np.arange(916), [r["label"] for r in sample["content"]]] = .95
            wrong = slice(0, 3) if model == "pilot" else slice(914, 916)
            p[wrong] = .002
            p[wrong, plots.ROLES.index("developer" if model == "pilot" else "tool")] = .99
            section["models"][model] = {"probabilities": p.tolist(), "metrics": diagnostic.metrics(sample, p)}
        results["formats"][sample["prompt_key"]] = section
    return samples, results


def save_fixture(root, samples, results):
    """Use production probability filenames and actual byte SHA256, still fake data."""
    results = copy.deepcopy(results)
    for key, section in results["formats"].items():
        for model, entry in section["models"].items():
            path = root / f"{key}-{model}-probabilities.npy"
            np.save(path, np.array(entry.pop("probabilities"), dtype=np.float32), allow_pickle=False)
            entry.update(probabilities_file=path.name, probabilities_sha256=plots._sha_file(path))
    (root / "prepared-inputs.json").write_text(json.dumps(samples, ensure_ascii=False))
    (root / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    return root / "prepared-inputs.json", root / "results.json"


class AlignmentTests(unittest.TestCase):
    def test_full_empty_or_overlapping_emoji_token_is_retained(self):
        for overlap in (False, True):
            samples, _ = fixture(overlap)
            values = plots._prepared(samples)
            for sample in values.values():
                token_roles, char_roles = plots._styles(sample)
                last = sample["content"][-1]
                self.assertEqual(token_roles[last["token_position"]], plots.ROLES.index("assistant"))
                self.assertEqual(int((token_roles >= 0).sum()), 916)
                self.assertEqual(len(char_roles), len(sample["text"]))
                self.assertTrue(sample["text"].endswith("🌱") or sample["text"].endswith("</message>") or sample["text"].endswith("</user>"))
                if not overlap:
                    self.assertEqual(*sample["token_offsets"][last["token_position"]])

    def test_styles_keep_structure_gray_and_full_original_text(self):
        samples, _ = fixture()
        sample = samples[0]
        token_roles, char_roles = plots._styles(sample)
        self.assertEqual(token_roles[0], -1)
        self.assertTrue(np.all(char_roles[:5] == -1))
        # Wrapping does not lose non-newline original characters or the emoji.
        lines = plots._lines(sample["text"], width=17)
        reconstructed = "".join(char for line in lines for _, char in line)
        self.assertEqual(reconstructed, sample["text"].replace("\n", ""))
        self.assertIn("🌱", reconstructed)

    def test_rejects_old_915_mask_and_changed_body_ids(self):
        samples, _ = fixture()
        samples[0]["content"].pop()
        with self.assertRaisesRegex(ValueError, "916"):
            plots._prepared(samples)
        samples, _ = fixture()
        row = samples[0]["content"][7]
        samples[0]["input_ids"][row["token_position"]] += 1
        with self.assertRaisesRegex(ValueError, "token ID"):
            plots._prepared(samples)

    def test_same_original_message_identity_across_three_formats(self):
        samples, _ = fixture()
        samples[2]["content"][0]["message_ix"] = 99
        with self.assertRaisesRegex(ValueError, "identity differs"):
            plots._prepared(samples)
        samples, _ = fixture()
        samples[1]["content"][0]["formal_role"] = "system"
        with self.assertRaisesRegex(ValueError, "conflated"):
            plots._prepared(samples)

    def test_wrapper_specific_token_ids_do_not_imply_cross_format_id_equality(self):
        samples, _ = fixture()
        for i, sample in enumerate(samples):
            row = sample["content"][20]
            row["token_id"] += i * 100000
            sample["input_ids"][row["token_position"]] = row["token_id"]
        self.assertEqual(set(plots._prepared(samples)), set(plots.FORMATS))

    def test_complete_prompt_sha_and_model_metric_kind_cannot_drift(self):
        samples, results = fixture()
        samples[0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt SHA256"):
            plots._prepared(samples)
        samples, results = fixture()
        results["formats"]["basic_no_format"]["models"]["pilot"]["metrics"]["metric_kind"] = "formal_role"
        with self.assertRaisesRegex(ValueError, "Per-model metric kind"):
            plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))

    def test_no_support_recall_na_and_false_positives_recomputed(self):
        samples, results = fixture()
        _, loaded = plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))
        dev = loaded["proper_tags"]["pilot"]["metrics"]["per_role"]["developer"]
        self.assertIsNone(dev["recall"])
        self.assertIsNone(dev["f1"])
        self.assertEqual(dev["false_positive"], 3)
        tool = loaded["proper_tags"]["first_formal"]["metrics"]["per_role"]["tool"]
        self.assertIsNone(tool["recall"])
        self.assertEqual(tool["false_positive"], 2)
        results["formats"]["proper_tags"]["models"]["pilot"]["metrics"]["per_role"]["developer"]["recall"] = 0
        with self.assertRaisesRegex(ValueError, "developer recall"):
            plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))

    def test_fixed_checkpoint_and_sixteen_shared_heads_required(self):
        for mutation in ("roles", "heads", "different_heads", "candidate"):
            samples, results = fixture()
            if mutation == "roles":
                results["roles"].pop()
            elif mutation == "heads":
                results["models"]["pilot"]["heads"] = list(range(10))
            elif mutation == "different_heads":
                results["models"]["pilot"]["heads"] = list(range(1, 17))
            else:
                results["models"]["first_formal"]["candidate_id"] = "mlp128-seed17"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))

    def test_changed_format_alignment_cannot_be_called_formal_accuracy(self):
        samples, results = fixture()
        results["formats"]["basic_no_format"]["metric_kind"] = "formal_role"
        with self.assertRaisesRegex(ValueError, "formal role accuracy"):
            plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))
        samples, results = fixture()
        results["formats"]["everything_in_user_tags"]["models"]["pilot"]["metrics"]["accuracy"] = 1.0
        with self.assertRaisesRegex(ValueError, "not formal accuracy"):
            plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))

    def test_file_inputs_and_probability_sha_fail_closed(self):
        samples, results = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared_path, result_path = save_fixture(root, samples, results)
            _, loaded = plots._loaded_results(result_path, plots._prepared(prepared_path), root)
            self.assertEqual(loaded["proper_tags"]["pilot"]["probabilities"].shape, (916, 6))
            path = root / "proper_tags-pilot-probabilities.npy"
            with path.open("ab") as out:
                out.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                plots._loaded_results(result_path, plots._prepared(prepared_path), root)

    def test_inline_only_marked_engineering_and_invalid_probability_rejected(self):
        samples, results = fixture()
        results["engineering_only"] = False
        with self.assertRaisesRegex(ValueError, "engineering fixtures"):
            plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))
        samples, results = fixture()
        results["formats"]["proper_tags"]["models"]["pilot"]["probabilities"][-1][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "probability rows"):
            plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))


@unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib needed only for CPU figure rendering")
class FigureTests(unittest.TestCase):
    def test_six_rows_each_have_all_916_strokes_including_final_continuation(self):
        import matplotlib.pyplot as plt
        samples, results = fixture()
        sample = plots._prepared(samples)["proper_tags"]
        _, loaded = plots._loaded_results(results, plots._prepared(samples), Path("/tmp"))
        fig, axes = plots._figure(sample, loaded["proper_tags"], results["models"], title="CPU engineering fixture")
        try:
            self.assertEqual(set(axes), set(plots.MODELS))
            for model in plots.MODELS:
                self.assertEqual(len(axes[model]), 6)
                for axis in axes[model]:
                    segments = axis.collections[0].get_segments()
                    self.assertEqual(len(segments), 916)
                    self.assertEqual(float(segments[-1][0][0]), sample["content"][-1]["token_position"])
                    # All helper Line2D items are vertical message separators, no thresholds.
                    self.assertTrue(all(len(set(line.get_xdata())) == 1 for line in axis.lines))
                self.assertIn("R N/A | FP", axes[model][1].get_ylabel())
                self.assertIn("R N/A | FP", axes[model][5].get_ylabel())
            note = fig._supxlabel.get_text()
            self.assertIn("U+1F331", note)
            self.assertIn("IDs differ across formats", note)
            self.assertTrue(all(len(line) <= 300 for line in note.splitlines()))
        finally:
            plt.close(fig)

    def test_complete_seven_pairs_index_all_context_and_sha(self):
        samples, results = fixture(overlap=True)
        with tempfile.TemporaryDirectory() as temp, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph .*missing from font")
            root = Path(temp)
            prepared_path, result_path = save_fixture(root, samples, results)
            index = plots.plot_gardening(prepared_path, result_path, root / "plots")
            self.assertEqual(index, json.loads((root / "plots/index.json").read_text()))
            stems = {"gardening-native-comparison"} | {f"gardening-{m}-{f}" for m in plots.MODELS for f in plots.FORMATS}
            self.assertEqual(set(index["figures"]), stems)
            self.assertTrue(index["engineering_only"])
            self.assertFalse(index["selection_permitted"])
            for stem, entry in index["figures"].items():
                self.assertEqual(entry["body_tokens_per_model"], 916)
                self.assertEqual(entry["probability_rows"], 6)
                self.assertTrue(entry["complete_context"])
                for suffix, record in entry["files"].items():
                    path = root / "plots" / record["path"]
                    self.assertEqual(record["path"], f"{stem}.{suffix}")
                    self.assertEqual(record["sha256"], plots._sha_file(path))
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG" if suffix == "png" else b"%PDF"))
            for sample in samples:
                entry = index["formats"][sample["prompt_key"]]
                self.assertEqual(entry["body_positions"], [r["token_position"] for r in sample["content"]])
                self.assertEqual(entry["text_sha256"], hashlib.sha256(sample["text"].encode()).hexdigest())
                self.assertEqual(entry["restored_emoji_continuation"]["token_id"], 109)
                self.assertTrue(entry["restored_emoji_continuation"]["retained"])
            self.assertEqual(len(list((root / "plots").glob("*.png"))), 7)
            self.assertEqual(len(list((root / "plots").glob("*.pdf"))), 7)


if __name__ == "__main__":
    unittest.main()
