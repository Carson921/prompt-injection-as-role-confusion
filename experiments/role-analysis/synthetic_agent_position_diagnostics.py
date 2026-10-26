"""Post-freeze body-relative and absolute-input position diagnostics.

This module only reuses an existing evaluation. It never selects a classifier,
opens an unevaluated TEST, loads the backbone, or reads source-article files.
The full native input is retokenized with the bound local official tokenizer;
denominators count all pure body tokens, not merely sampled supervision rows.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

import synthetic_agent_classifier as classifier
import synthetic_agent_data as data

FORMAT_VERSION = "synthetic-agent-position-diagnostics-v1"
ROLES = tuple(classifier.ROLES)
ATTRIBUTE_KEYS = {"profile", "template_id", "history_policy", "snapshot_stage", "rounds"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _body_layout(record, tokenizer):
    """Independently recount pure body-token ordinals from character offsets."""
    text = record["text"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    _require(list(ids) == record["input_ids"], "Official tokenizer re-encoding differs from recorded input_ids")
    _require(len(ids) == len(offsets) == record["input_tokens"], "Input/token-offset length mismatch")
    spans, parsed = record["message_spans"], data.parse_native(text)
    _require(len(spans) == len(parsed), "Message spans disagree with independently parsed Harmony")
    bodies, events = [], set()
    for span, native in zip(spans, parsed):
        for key in ("role", "symbol", "header_start", "body_start", "message_end"):
            _require(span[key] == native[key], f"Independent native message mismatch: {key}")
        expected_end = span["body_end"]
        if span.get("tool_definition"):
            expected_end += len(data.TOOL_DEFINITION)
            _require(text[span["body_end"]:expected_end] == data.TOOL_DEFINITION,
                     "Developer structural tool definition differs from the renderer")
        _require(expected_end == native["body_end"], "Pure body and structural suffix boundaries disagree")
        _require(bool(span["supervised"]) == (span["symbol"] != "Q"), "Unexpected formal body supervision flag")
        if not span["supervised"]:
            continue
        _require(span["role"] in ROLES and 0 <= span["body_start"] <= span["body_end"] <= len(text),
                 "Invalid six-role pure-body span")
        _require(span["event_id"] not in events, "An event appears twice in one visible snapshot")
        events.add(span["event_id"])
        bodies.append(span)
    starts = [span["body_start"] for span in bodies]
    _require(starts == sorted(starts), "Body spans are not in input order")
    _require(all(left["body_end"] <= right["body_start"] for left, right in zip(bodies, bodies[1:])),
             "Body spans overlap")
    pure, counts, mixed = {}, Counter(), []
    for token_ix, (left, right) in enumerate(offsets):
        _require(0 <= left <= right <= len(text), "Tokenizer returned an invalid character offset")
        if right == left:
            continue
        previous = bisect_right(starts, left) - 1
        candidates = [bodies[j] for j in (previous, previous + 1) if 0 <= j < len(bodies)]
        overlapping = [span for span in candidates if left < span["body_end"] and right > span["body_start"]]
        _require(len(overlapping) <= 1, "A token crosses multiple formal body spans")
        if not overlapping:
            continue
        span = overlapping[0]
        if left < span["body_start"] or right > span["body_end"]:
            mixed.append(token_ix)
            continue
        event = span["event_id"]
        pure[token_ix] = {"event_id": event, "message_id": span["message_id"],
                          "role": span["role"], "body_position": counts[event]}
        counts[event] += 1
    if "boundary_tokens" in record:
        _require(sorted(mixed) == sorted(int(row["token_ix"]) for row in record["boundary_tokens"]),
                 "Mixed article/structure exclusions differ from the original supervision audit")
    return pure, dict(counts), mixed


def _target_positions(record, tokenizer):
    pure, lengths, mixed = _body_layout(record, tokenizer)
    result, seen = [], set()
    for target in record["targets"]:
        token_ix = int(target["token_ix"])
        _require(token_ix not in seen, "A supervised token is repeated in one input")
        seen.add(token_ix)
        _require(token_ix in pure, "A supervised target is structural or crosses a body boundary")
        observed = pure[token_ix]
        for key in ("event_id", "message_id", "role", "body_position"):
            _require(target[key] == observed[key], f"Recorded target differs from pure-body token ordinal: {key}")
        label = ROLES.index(target["label"]) if isinstance(target["label"], str) else int(target["label"])
        _require(0 <= label < 6 and ROLES[label] == observed["role"], "Target role/label mismatch")
        _require(target["source_id"] == record["source_id"], "Target/source correspondence changed")
        _require(target["family_id"] == f"{observed['event_id']}:{observed['body_position']}",
                 "Target event-token family identity changed")
        length = lengths[observed["event_id"]]
        position = int(observed["body_position"])
        edge_count = (length + 3) // 4
        result.append({"label": label, "body_position": position, "body_tokens": length,
                       "body_quartile": min(3, 4 * position // length),
                       "first_25pct": position < edge_count,
                       "last_25pct": position >= length - edge_count,
                       "input_quartile": min(3, 4 * token_ix // len(record["input_ids"]))})
    return result, {"pure_body_tokens": sum(lengths.values()), "visible_body_messages": len(lengths),
                    "excluded_mixed_boundary_tokens": len(mixed)}


def _empty_metrics():
    result = classifier._metrics_from_confusion(np.zeros((6, 6), dtype=np.int64))
    result.update(accuracy=None, macro_f1=None, nll=None)
    return result


def _metrics(probabilities, labels, mask):
    return classifier.metrics(probabilities[mask], labels[mask]) if np.any(mask) else _empty_metrics()


def _position_metrics(probabilities, labels, arrays, mask):
    return {
        "overall": _metrics(probabilities, labels, mask),
        "by_body_quartile": {f"q{i + 1}": _metrics(probabilities, labels, mask & (arrays["body_quartile"] == i)) for i in range(4)},
        "by_body_edge": {key: _metrics(probabilities, labels, mask & arrays[key]) for key in ("first_25pct", "last_25pct")},
        "by_absolute_input_quartile": {f"q{i + 1}": _metrics(probabilities, labels, mask & (arrays["input_quartile"] == i)) for i in range(4)},
    }


def _length_bucket(length):
    if length < 128:
        return "0-127"
    if length < 256:
        return "128-255"
    if length < 512:
        return "256-511"
    if length < 1024:
        return "512-1023"
    return "1024+"


def _load_tokenizer(run_dir, metadata):
    """Local official assets only; this never loads model weights or a GPU."""
    root = Path(run_dir)
    backend_path = root / "model_backend.json"
    config = _read(root / "config.json") if (root / "config.json").exists() else {}
    backend = _read(backend_path) if backend_path.exists() else {}
    if backend:
        _require(backend.get("model_binding") == metadata["model_binding"], "Model backend differs from frozen tokenizer/model binding")
    path = backend.get("load_path", config.get("model_dir"))
    _require(path is not None, "Cannot locate the bound official tokenizer: model_backend.json.load_path is required")
    path = Path(path)
    assets = metadata["model_binding"].get("assets_sha256", {})
    _require("tokenizer.json" in assets and "tokenizer_config.json" in assets,
             "Frozen model metadata lacks official tokenizer asset hashes")
    checked = {}
    for name, expected in assets.items():
        # model.safetensors.index.json and model config are small, useful binding
        # checks too. Large weight files are not opened by this diagnostic.
        candidate = path / name
        _require(candidate.is_file() and classifier._sha(candidate) == expected,
                 f"Bound official tokenizer/config asset differs: {name}")
        checked[name] = expected
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    _require(getattr(tokenizer, "is_fast", False), "Exact body offsets require the official fast tokenizer")
    return tokenizer, {"local_path": str(path.resolve()), "assets_sha256": checked,
                       "chat_template_used": False, "backbone_loaded": False}


def _receipts(run_dir, run_phase, evaluation_name, feature_split, external):
    root = Path(run_dir)
    for parent in (root, root.parent):
        folder = parent / "campaign" / "receipts"
        if folder.is_dir():
            suffix = ".external" if external else (".test" if feature_split == "test" else "")
            feature = folder / f"{run_phase}.extract{suffix}.json"
            evaluation = folder / f"{run_phase}.evaluate.{evaluation_name}.json"
            return (feature if feature.exists() else None, evaluation if evaluation.exists() else None)
    return None, None


def _verify_evaluation_binding(root, bank, evaluation, evaluation_path, probability_path, frozen,
                                feature_split, run_phase, name, external):
    _require(evaluation["probe_sha256"] == frozen["probe_sha256"], "Evaluation uses a different frozen probe")
    feature_receipt, evaluation_receipt = _receipts(root, run_phase, name, feature_split, external)
    proof = {"evaluation_manifest_exact": evaluation.get("feature_manifest_sha256") == bank.manifest_hash,
             "stable_split_receipt": False, "frozen_dev_training_split": False,
             "evaluation_receipt_verified": False}
    if feature_receipt:
        value = _read(feature_receipt)
        _require(value["proof_sha256"] == hashlib.sha256(classifier._json(value["proof"]).encode()).hexdigest(),
                 "Feature receipt proof checksum changed")
        _require(value["proof"][feature_split]["split_identity"] == bank.split_identity_hash,
                 "Committed feature split differs from the campaign receipt")
        proof["stable_split_receipt"] = True
    if feature_split == "dev" and not external:
        training_path = root / "training.json"
        _require(frozen.get("training_sha256") == classifier._sha(training_path), "Training metadata changed after freeze")
        training = _read(training_path)
        _require(training["feature_split_hashes"]["dev"] == bank.split_identity_hash,
                 "DEV features differ from the frozen training cache")
        proof["frozen_dev_training_split"] = True
    _require(any(proof[key] for key in ("evaluation_manifest_exact", "stable_split_receipt", "frozen_dev_training_split")),
             "Evaluation manifest changed without a retained immutable split proof; re-evaluate the frozen model")
    if evaluation_receipt:
        value = _read(evaluation_receipt)
        _require(value["proof_sha256"] == hashlib.sha256(classifier._json(value["proof"]).encode()).hexdigest(),
                 "Evaluation receipt proof checksum changed")
        for path in (evaluation_path, probability_path):
            _require(value["proof"].get(str(path.resolve())) == classifier._sha(path),
                     "Evaluation/probability bytes differ from the campaign receipt")
        proof["evaluation_receipt_verified"] = True
    return proof


def _mapped_arrays(records, bank, tokenizer, group_by):
    count = len(bank)
    arrays = {key: np.empty(count, dtype=np.int32) for key in ("body_position", "body_tokens", "body_quartile", "input_quartile")}
    arrays.update({key: np.empty(count, dtype=bool) for key in ("first_25pct", "last_25pct")})
    attributes = {key: np.empty(count, dtype=object) for key in group_by}
    diagnostics = Counter()
    group_cache = {}
    cursor = 0
    for record in records:
        positions, audit = _target_positions(record, tokenizer)
        stop = cursor + len(positions)
        _require(stop <= count, "Input supervision has more rows than the evaluated feature split")
        labels = np.array([position["label"] for position in positions], dtype=np.int64)
        _require(np.array_equal(bank.y[cursor:stop], labels), "Probability/feature labels differ from input target order")
        expected_source = str(record.get("cluster_id", record["source_id"]))
        _require(all(str(value) == expected_source for value in bank.source[cursor:stop]),
                 "Feature source-cluster order differs from input records")
        _require([str(value) for value in bank.family[cursor:stop]] == [str(row["family_id"]) for row in record["targets"]],
                 "Feature event-token family order differs from input targets")
        expected_structure = [str(target.get("structure", record.get("profile", "") + ":" + record.get("template_id", "")))
                              for target in record["targets"]]
        _require([str(value) for value in bank.structure[cursor:stop]] == expected_structure,
                 "Feature structure order differs from input targets")
        for row_ix, position in enumerate(positions):
            ix = cursor + row_ix
            for key in arrays:
                arrays[key][ix] = position[key]
            encoded_group = str(bank.groups[ix])
            if encoded_group not in group_cache:
                group_cache[encoded_group] = json.loads(encoded_group)
            group = group_cache[encoded_group]
            expected = {"profile": record.get("profile"), "template_id": record.get("template_id"),
                        "rounds": len(record.get("round_segments", [])), "history_policy": record.get("history_policy"),
                        "snapshot_stage": record.get("snapshot_stage"), "position_bin": position["input_quartile"]}
            for key, value in group.items():
                if key in expected:
                    _require(value == expected[key], f"Feature snapshot/position group order differs: {key}")
            for key in group_by:
                attributes[key][ix] = str(expected.get(key, "N/A"))
        cursor = stop
        diagnostics["records_reencoded"] += 1
        diagnostics.update(audit)
    _require(cursor == count, "Evaluated feature split has rows not represented in the input target stream")
    return arrays, attributes, dict(diagnostics)


def evaluate_positions(run_dir, data_dir, phase="formal", split="test", feature_dir=None, output_name=None):
    """Create an additional positional report from an existing frozen evaluation."""
    root = Path(run_dir)
    _require((root / "freeze.json").is_file(), "Position diagnostics require freeze.json")
    external = split == "external" or phase == "external" or output_name == "external"
    data_phase = "external" if external else phase
    feature_split = "test" if external else split
    name = str(output_name or ("external" if external else split))
    _require(name and all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name),
             "Output name must be a simple evaluation identifier")
    _require(feature_split in {"dev", "test"}, "Position diagnostics reuse DEV or TEST evaluations only")
    evaluation_path = root / "evaluations" / f"{name}.json"
    probability_path = root / "evaluations" / f"{name}-probabilities.npy"
    # This gate intentionally precedes opening any input JSONL or tokenizer.
    _require(evaluation_path.is_file() and probability_path.is_file(),
             "An existing completed evaluation and probability file are required before opening its inputs")
    frozen = _read(root / "freeze.json")
    _require(frozen["probe_sha256"] == classifier._sha(root / "best.npz") and
             frozen["selection_sha256"] == classifier._sha(root / "selection.json"), "Frozen classifier/selection bytes changed")
    _require(frozen.get("training_sha256") == classifier._sha(root / "training.json"), "Training metadata changed after freeze")
    probe = classifier.load_probe(root / "best.npz")
    _require(not probe.metadata.get("engineering_smoke_only"), "Engineering smoke probes have no scientific position report")
    _require(frozen["metadata"] == probe.metadata, "Frozen probe metadata differs from its exported NPZ")
    run_phase = _read(root / "training.json").get("phase", phase) if (root / "training.json").exists() else phase
    feature_root = Path(feature_dir) if feature_dir is not None else root / "features" / data_phase
    bank = classifier.FeatureBank(feature_root, feature_split, expected_binding=probe.metadata)
    evaluation = _read(evaluation_path)
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    _require(probabilities.shape == (len(bank), 6), "Evaluation probability row count differs from committed feature rows")
    binding = _verify_evaluation_binding(root, bank, evaluation, evaluation_path, probability_path, frozen,
                                         feature_split, run_phase, name, external)
    overall = classifier.metrics(probabilities, bank.y)
    _require(overall["confusion"] == evaluation["overall"]["confusion"] and overall["tokens"] == evaluation["overall"]["tokens"],
             "Existing evaluation metrics disagree with stored probability/label row order")
    inputs = Path(data_dir) / data_phase / f"inputs-{feature_split}.jsonl"
    _require(classifier._sha(inputs) == bank.manifest["split_sources"][feature_split], "Input records differ from the evaluated feature input hash")
    tokenizer, tokenizer_proof = _load_tokenizer(root, probe.metadata)
    config = _read(root / "config.json") if (root / "config.json").exists() else {}
    group_by = tuple(config.get("position_diagnostics", {}).get("group_by", ("profile", "history_policy")))
    _require(len(set(group_by)) == len(group_by) and set(group_by).issubset(ATTRIBUTE_KEYS), "Unsupported positional diagnostic attribute")
    with inputs.open() as stream:
        records = (json.loads(line) for line in stream if line.strip())
        arrays, attributes, audit = _mapped_arrays(records, bank, tokenizer, group_by)
    valid = np.ones(len(bank), dtype=bool)
    result = _position_metrics(probabilities, bank.y, arrays, valid)
    result.update({
        "format_version": FORMAT_VERSION, "phase": data_phase, "split": feature_split,
        "output_name": name, "selection_permitted": False,
        "probe_sha256": frozen["probe_sha256"], "freeze_sha256": classifier._sha(root / "freeze.json"),
        "evaluation_sha256": classifier._sha(evaluation_path), "probabilities_sha256": classifier._sha(probability_path),
        "input_sha256": bank.manifest["split_sources"][feature_split], "feature_split_identity": bank.split_identity_hash,
        "binding_validation": binding, "tokenizer": tokenizer_proof, "row_mapping_audit": audit,
        "definitions": {
            "body_denominator": "All nonzero-offset tokens wholly contained in that visible message's article/body span; excludes headers, delimiters, Q JSON, D tool definitions, and mixed body/structure tokens, including unsampled pure body tokens.",
            "body_quartile": "q=1+min(3,floor(4*zero_based_body_position/pure_body_token_count)); empty cells are N/A.",
            "first_last_25pct": "First and last ceil(pure_body_token_count/4) token ranks, respectively; the two edge sets may overlap for messages shorter than four tokens.",
            "absolute_input_quartile": "q=1+min(3,floor(4*token_ix/complete_input_token_count)), including all unscored structural context in the denominator.",
            "scores": "Only existing supervised targets are scored; absent true roles have N/A recall/F1, while false positives are reported for all six outputs.",
        },
        "confidence_intervals": {"per_cell_bootstrap": False,
            "overall_evaluation_has_bootstrap": bool(evaluation.get("confidence_intervals")),
            "reference": f"{name}.json",
            "note": "No per-cell bootstrap is added; retain the original final-test 2000 source-cluster overall bootstrap when available."},
        "by_attribute": {}, "by_body_length": {},
    })
    for key, values in attributes.items():
        result["by_attribute"][key] = {value: _position_metrics(probabilities, bank.y, arrays, values == value)
                                         for value in sorted(set(values))}
    length_values = np.array([_length_bucket(length) for length in arrays["body_tokens"]])
    for bucket in ("0-127", "128-255", "256-511", "512-1023", "1024+"):
        result["by_body_length"][bucket] = _position_metrics(probabilities, bank.y, arrays, length_values == bucket)
    classifier._atomic_json(root / "evaluations" / f"{name}-positions.json", result)
    return result
