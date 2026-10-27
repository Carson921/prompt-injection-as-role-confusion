"""Independent CPU readout of a supplementary, hash-sealed gardening cache.

This imports the independent Torch implementation, never the producer Probe.
No cached features, historical measurements, or frozen run metadata are edited.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[4]
ANALYSIS = REPO / "experiments/role-analysis"
OLD = REPO.parent / "runpod-backups/role-retraining-20261026-20261027"
FINAL = REPO.parent / "runpod-backups/role-warmstart-20261027-6h-final/task"
ARCHIVE = OLD / "workspace-task/diagnostics/gardening-pilot-vs-first-formal"
PARENT = ANALYSIS / "outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz"
PARENT_SHA = "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125"
NEW_SHA = "45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df"
FORMATS = ("proper_tags", "everything_in_user_tags", "basic_no_format")
SUPPORT = {"system": 57, "developer": 0, "user": 97, "cot": 179, "assistant": 583, "tool": 0}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_formal_roles(sample, specials):
    """Read message boundaries from exact IDs, and headers from saved offsets."""
    ids, text, offsets = sample["input_ids"], sample["text"], sample["token_offsets"]
    start, separator = specials["<|start|>"], specials["<|message|>"]
    closers = {specials[key] for key in ("<|end|>", "<|return|>", "<|call|>")}
    assert ids[0] == specials["<|startoftext|>"]
    if sample["prompt_key"] == "basic_no_format":
        assert start not in ids and separator not in ids
        return None
    roles, sequence, i = {}, [], 1
    headers = {"system": "system", "user": "user", "assistant<|channel|>analysis": "cot",
               "assistant<|channel|>final": "assistant"}
    while i < len(ids):
        assert ids[i] == start
        header_start = offsets[i][1]
        i += 1
        while ids[i] != separator:
            assert ids[i] not in closers | {start}
            i += 1
        header = text[header_start:offsets[i][0]]
        role = headers[header]
        sequence.append(role)
        i += 1
        while i < len(ids) and ids[i] not in closers:
            assert ids[i] not in {start, separator}
            roles[i] = role
            i += 1
        assert i < len(ids)
        i += 1
    expected = ["system", "user", "cot", "assistant", "user", "cot", "assistant"]
    assert sequence == (expected if sample["prompt_key"] == "proper_tags" else ["user"])
    return roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--producer-output", type=Path)
    parser.add_argument("--producer-private", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("new-cache-independent-validation.json"))
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(2)
    module_path = ANALYSIS / "validate_synthetic_role_warmstart_results.py"
    assert sha(module_path) == "016f14739001ed823621ff2d257bc764a6e225c81f82e13b297b6b59a744182c"
    spec = importlib.util.spec_from_file_location("independent_warm_verifier", module_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    evidence = verifier.Evidence()
    evidence.file(module_path)
    evidence.file(Path(__file__))
    root = args.cache_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    evidence.file(manifest_path)
    assert manifest["format_version"] == "warmstart-gardening-raw-cache-v1"
    assert manifest["status"] == "complete" and manifest["forward_count"] == 3
    assert manifest["budget_not_rearmed"] and manifest["selection_unchanged"]
    assert manifest["fixed_original_conversation_forward"] and not manifest["new_response_generation"]
    assert not manifest["new_agent_rollout"] and manifest["devices"] == [0, 1, 2]
    budget, frozen = read(FINAL / "budget.json"), read(FINAL / "run/freeze.json")
    evidence.file(FINAL / "budget.json", manifest["original_budget_sha256"])
    evidence.file(FINAL / "run/freeze.json", manifest["freeze_sha256"])
    assert manifest["original_budget_sha256"] == frozen["gpu_budget"]["budget_sha256"]
    assert manifest["original_budget_start_unix"] == budget["started_unix"]
    assert manifest["original_budget_deadline_unix"] == budget["deadline_unix"]
    assert budget["started_unix"] <= manifest["started_unix"] <= manifest["finished_unix"]
    assert manifest["finished_unix"] <= manifest["local_deadline_unix"] <= budget["deadline_unix"]
    assert manifest["local_deadline_unix"] - manifest["started_unix"] <= 600.01
    assert not manifest["memory_monitor_errors"]
    parent = verifier.ExactTorchProbe(PARENT, evidence=evidence)
    new = verifier.ExactTorchProbe(FINAL / "run/best.npz", evidence=evidence)
    assert sha(parent.path) == PARENT_SHA == manifest["parent_probe_sha256"]
    assert sha(new.path) == NEW_SHA == manifest["new_probe_sha256"] == frozen["probe_sha256"]
    for probe in (parent, new):
        for key in ("roles", "heads", "feature_dim", "feature_position", "model_binding"):
            assert manifest[key] == probe.metadata[key]
    for key, name in (("runtime_sha256", "synthetic_agent_gpu.py"),
                      ("source_parser_sha256", "evaluate_synthetic_gardening.py"),
                      ("extractor_sha256", "extract_synthetic_warmstart_gardening.py")):
        evidence.file(ANALYSIS / name, manifest[key])
    samples = read(root / "prepared-inputs.json")
    evidence.file(root / "prepared-inputs.json", manifest["prepared_inputs_sha256"])
    evidence.file(root / "prepare-manifest.json", manifest["prepare_manifest_sha256"])
    archival = read(ARCHIVE / "prepared-inputs.json")
    evidence.file(ARCHIVE / "prepared-inputs.json")
    assert samples == archival, "Supplementary inputs differ from the archived complete original inputs"
    tokenizer_path = OLD / "model-metadata/tokenizer.json"
    evidence.file(tokenizer_path, parent.metadata["model_binding"]["assets_sha256"]["tokenizer.json"])
    specials = {row["content"]: row["id"] for row in read(tokenizer_path)["added_tokens"]}
    assert set(manifest["formats"]) == set(FORMATS)
    producer, producer_files = None, {}
    if args.producer_output is not None or args.producer_private is not None:
        assert args.producer_output is not None and args.producer_private is not None
        seal = read(args.producer_output / "result-manifest.json")
        evidence.file(args.producer_output / "result-manifest.json")
        assert seal["status"] == "complete" and seal["parent_probe_sha256"] == PARENT_SHA
        assert seal["new_probe_sha256"] == NEW_SHA and seal["source_cache_manifest_sha256"] == sha(manifest_path)
        evidence.file(ANALYSIS / "evaluate_synthetic_warmstart_gardening.py", seal["implementation_sha256"])
        for row in seal["files"]:
            path = (args.producer_output / row["path"]).resolve()
            assert path.is_relative_to(args.producer_output.resolve()) and path.stat().st_size == row["bytes"]
            evidence.file(path, row["sha256"])
        producer = read(args.producer_output / "metrics.json")
        evidence.file(args.producer_output / "metrics.json")
        readout = read(args.producer_private / "readout-manifest.json")
        evidence.file(args.producer_private / "readout-manifest.json")
        assert readout["status"] == "complete" and len(readout["files"]) == 6
        assert readout["source_cache_manifest_sha256"] == sha(manifest_path)
        assert readout["probe_sha256"] == NEW_SHA and readout["parent_probe_sha256"] == PARENT_SHA
        for row in readout["files"]:
            pair = row["format"], row["model"]
            assert pair not in producer_files
            path = Path(row["path"])
            assert path.resolve().is_relative_to(args.producer_private.resolve())
            assert path.stat().st_size == row["bytes"]
            evidence.file(path, row["sha256"])
            producer_files[pair] = path
        assert set(producer_files) == {(f, m) for f in FORMATS for m in ("parent_final", "new_candidate")}
    formats = {}
    for sample in samples:
        key = sample["prompt_key"]
        entry = manifest["formats"][key]
        path = (root / entry["file"]).resolve()
        assert path.is_relative_to(root)
        evidence.file(path, entry["sha256"])
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"head_features", "positions", "input_ids", "token_ids", "head_ids", "roles"}
            x = arrays["head_features"].copy()
            assert x.dtype == np.float16 and x.shape == (916, 1024) and np.isfinite(x).all()
            targets = sample["content"]
            for name, expected in (("positions", [row["token_position"] for row in targets]),
                ("token_ids", [row["token_id"] for row in targets]), ("input_ids", sample["input_ids"]),
                ("head_ids", manifest["heads"])):
                assert arrays[name].dtype == np.int64 and np.array_equal(arrays[name], expected)
            assert arrays["roles"].tolist() == manifest["roles"]
        assert entry["input_tokens"] == len(sample["input_ids"]) and entry["body_rows"] == 916
        assert entry["prompt_sha256"] == hashlib.sha256(sample["text"].encode()).hexdigest()
        assert targets[-1]["token_id"] == 109 and targets[-1]["semantic_role"] == "assistant"
        semantic = np.asarray([row["label"] for row in targets], dtype=np.int64)
        assert {role: int((semantic == index).sum()) for index, role in enumerate(manifest["roles"])} == SUPPORT
        parsed = independent_formal_roles(sample, specials)
        if parsed is None:
            assert all(row["formal_role"] is None for row in targets)
            formal = None
        else:
            assert parsed == {row["token_position"]: row["formal_role"] for row in targets}
            formal = np.asarray([manifest["roles"].index(parsed[row["token_position"]]) for row in targets])
        models = {}
        for name, probe in (("parent_final", parent), ("new_scratch_epoch11", new)):
            p = probe.predict(x, batch_size=127)
            assert p.shape == (916, 6) and p.dtype == np.float32 and np.isfinite(p).all()
            semantic_metrics = verifier.metrics(semantic, p)
            models[name] = dict(semantic_alignment_metrics=semantic_metrics,
                formal_role_metrics=verifier.metrics(formal, p) if formal is not None else None,
                predicted_role_counts={role: int((p.argmax(1) == index).sum()) for index, role in enumerate(manifest["roles"])},
                mean_probabilities=p.astype(np.float64).mean(0).tolist())
            if producer is not None:
                published_name = "parent_final" if name == "parent_final" else "new_candidate"
                stored = np.load(producer_files[key, published_name], allow_pickle=False)
                comparison = verifier.probability_comparison(p, stored, 5e-4)
                expected = producer["formats"][key]["models"][published_name]
                assert expected["rows"] == 916 and expected["confusion_matrix"] == semantic_metrics["confusion"]
                score_key = "accuracy" if key == "proper_tags" else "semantic_alignment_accuracy"
                other_key = "semantic_alignment_accuracy" if key == "proper_tags" else "accuracy"
                assert expected[other_key] is None
                assert expected["metric_kind"] == ("formal_role" if key == "proper_tags" else "original_semantic_alignment")
                evidence.number(semantic_metrics["accuracy"], expected[score_key], key + "/" + name + "/accuracy")
                evidence.number(semantic_metrics["macro_f1"], expected["macro_f1_present"], key + "/" + name + "/macro_f1")
                evidence.number(models[name]["formal_role_metrics"]["accuracy"] if key == "everything_in_user_tags" else None,
                    expected["formal_all_user_accuracy"], key + "/" + name + "/formal_all_user_accuracy")
                for index, role in enumerate(manifest["roles"]):
                    actual_role = semantic_metrics["roles"][role]
                    expected_role = expected["per_role"][role]
                    for field in ("support", "predicted", "precision", "recall", "f1"):
                        evidence.number(actual_role[field], expected_role[field], key + "/" + name + "/" + role + "/" + field)
                    tp = semantic_metrics["confusion"][index][index]
                    for field, value in (("true_positive", tp), ("false_negative", actual_role["support"] - tp),
                                         ("false_positive", actual_role["predicted"] - tp)):
                        evidence.number(value, expected_role[field], key + "/" + name + "/" + role + "/" + field)
                models[name]["producer_probability_comparison"] = comparison
        formats[key] = dict(input_tokens=len(sample["input_ids"]), body_tokens=916,
            formal_role_scoring="native closed message bodies" if key == "proper_tags" else
                "all 916 positions have formal user role" if key == "everything_in_user_tags" else "undefined: no role delimiters",
            semantic_scoring="original-message alignment diagnostic; not instruction-following accuracy", models=models)
    report = dict(format_version="warmstart-gardening-independent-raw-cache-readout-v1", status="passed",
        scope="Supplementary three original-conversation forwards after sealed main run; no retraining or reselection",
        parent_probe_sha256=PARENT_SHA, new_probe_sha256=NEW_SHA, new_probe_eligible=frozen["eligible"],
        cache_manifest_sha256=sha(manifest_path), exact_torch_gelu=True, torch_version=torch.__version__,
        formats=formats, evidence=evidence.summary(), elapsed_seconds=time.time()-started,
        producer_results_independently_compared=producer is not None,
        limitations=["Historically used single gardening example; not a new blind test or evidence of agent behavior.",
            "Original conversation contains no developer or tool positive examples; their recalls remain N/A.",
            "Altered-format original-role alignment is reported separately from formal message-role classification.",
            "This scope does not rewrite the earlier main-run GPU completion time or its sealed results."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", output=str(args.output.resolve()), sha256=sha(args.output),
                          elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()
