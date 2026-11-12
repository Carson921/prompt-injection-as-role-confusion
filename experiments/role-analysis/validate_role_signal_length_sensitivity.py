"""Independent audit of the post-hoc single-ending-token removal diagnostic.

Imports only the independent NumPy/math.erf validator, never an inference or
analysis implementation. Primary artifacts are read-only. This checks artifact
and numerical-readout consistency; it does not prove global causal invariance.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from validate_role_signal_factorial import (
    DEFAULT_PROBE, PROBE_SHA256, ROLES, compare_fields, jsonl, numpy_logits,
    read, require, sha, softmax,
)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_digest(path, expected, label):
    require(sha(path) == expected, "Hash mismatch: " + label)


def validate_paired_comparison(root, original_rows, alternate_rows):
    """Explicit scalar within-unit contrasts and direct paired resampling.

    Deliberately avoids the analyzer's tensor axis reductions, bootstrap count
    matrix and original-minus-alternate bootstrap-array implementation.
    """
    provenance = read(root / "length-sensitivity-manifest.json")
    settings = provenance["bootstrap"]
    require(settings["replicates"] == 2000 and settings["seed"] == 20261112 and settings["unit"] == "semantic pair_id",
            "Unexpected paired sensitivity bootstrap specification")
    source_files = {
        "input-summary.csv": root / "input-summary.csv",
        "numerical-length-sensitivity/input-summary.csv": root / "numerical-length-sensitivity/input-summary.csv",
        "effects.csv": root / "effects.csv", "interactions.csv": root / "interactions.csv",
        "primary_script": Path(__file__).with_name("analyze_role_signal_factorial.py"),
        "length_sensitivity_script": Path(__file__).with_name("analyze_role_signal_length_sensitivity.py"),
    }
    require(set(provenance["source_hashes"]) == set(source_files), "Unknown/incomplete comparison source provenance")
    for name, path in source_files.items():
        check_digest(path, provenance["source_hashes"][name], "paired comparison " + name)
    require(provenance["delta_definition"] == "alternate minus original", "Unexpected change orientation")
    formal = [row for row in original_rows if row["split"] == "formal"]
    pair_ids = sorted({row["pair_id"] for row in formal})
    require(len(pair_ids) == 50, "Incorrect paired comparison cluster count")
    metrics = ["mean_margin", *("p_" + role for role in ROLES), "probability_contrast"]
    names = ["text_styled_minus_destyled", "wrapper_cot_minus_user", "position_256_minus_32", "position_768_minus_32",
             "text_x_wrapper_at_32", "text_x_wrapper_at_256", "text_x_wrapper_at_768",
             "text_x_wrapper_256_minus_32", "text_x_wrapper_768_minus_32"]
    groups = ["A", "B", "pooled"]
    cells = [{(row["pair_id"], row["filler_id"], row["text_style"], row["wrapper"], int(row["position"])): row for row in rows}
             for rows in (formal, alternate_rows)]
    require(all(len(values) == 1200 for values in cells), "Duplicate/missing formal condition cells")
    draws = np.random.default_rng(20261112).integers(0, 50, (2000, 50))

    def scalar_contrasts(cell, pair, filler, metric):
        def value(text, wrapper, position):
            row = cell[pair, filler, text, wrapper, position]
            return float(row[metric]) if metric != "probability_contrast" else float(row["p_cot"]) - float(row["p_user"])
        text = sum(value("styled", w, p) - value("destyled", w, p) for w in ("cot", "user") for p in (32, 256, 768)) / 6
        wrapper = sum(value(t, "cot", p) - value(t, "user", p) for t in ("styled", "destyled") for p in (32, 256, 768)) / 6
        positions = [sum(value(t, w, p) - value(t, w, 32) for t in ("styled", "destyled") for w in ("cot", "user")) / 4
                     for p in (256, 768)]
        interactions = [value("styled", "cot", p) - value("destyled", "cot", p) - value("styled", "user", p) + value("destyled", "user", p)
                        for p in (32, 256, 768)]
        return [text, wrapper, *positions, *interactions, interactions[1] - interactions[0], interactions[2] - interactions[0]]

    combined, table_counts = {}, {}
    for filename, relevant in (("length-sensitivity-effects.csv", names[:4]), ("length-sensitivity-interactions.csv", names[4:])):
        rows = csv_rows(root / filename)
        indexed = {(row["filler_id"], row["contrast"], row["metric"]): row for row in rows}
        require(len(rows) == len(indexed) and set(indexed) == set(itertools.product(groups, relevant, metrics)),
                "Incomplete sensitivity contrast table: " + filename)
        combined.update(indexed)
        table_counts[filename] = len(rows)
    maximum = 0.0
    for metric in metrics:
        matrices = []
        for cell in cells:
            by_filler = {filler: np.asarray([scalar_contrasts(cell, pair, filler, metric) for pair in pair_ids]) for filler in ("A", "B")}
            by_filler["pooled"] = (by_filler["A"] + by_filler["B"]) / 2
            matrices.append(by_filler)
        for group in groups:
            variants = {"original": matrices[0][group], "alternate": matrices[1][group],
                        "delta": matrices[1][group] - matrices[0][group]}
            for variant, values in variants.items():
                point = values.mean(axis=0)
                bounds = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
                for index, name in enumerate(names):
                    row = combined[group, name, metric]
                    require(int(row["n_pairs"]) == 50 and int(row["n_inputs_per_run"]) == 1200,
                            "Incorrect paired contrast independent-unit/input denominator")
                    for column, expected in ((variant + "_estimate", point[index]), (variant + "_ci_lower", bounds[0, index]),
                                             (variant + "_ci_upper", bounds[1, index])):
                        error = abs(float(row[column]) - expected)
                        maximum = max(maximum, error)
                        require(error < 1e-8 + abs(expected) * 1e-9,
                                "Independent paired change/interval differs: " + str((group, name, metric, column)))
                    if variant == "delta":
                        require(abs(float(row["delta"]) - point[index]) < 1e-8, "Sensitivity delta alias differs")
    return {"status": "passed", "all_original_alternate_and_paired_delta_estimates_and_intervals_recomputed": True,
            "replicates": 2000, "seed": 20261112, "unit": "semantic pair_id", "table_rows": table_counts,
            "maximum_scalar_contrast_or_interval_difference": maximum,
            "comparison_manifest_sha256": sha(root / "length-sensitivity-manifest.json"),
            "comparison_effects_sha256": sha(root / "length-sensitivity-effects.csv"),
            "comparison_interactions_sha256": sha(root / "length-sensitivity-interactions.csv")}


def validate(args):
    root = args.output.resolve()
    alternate = root / "numerical-length-sensitivity"
    probe_path = args.probe.resolve()
    check_digest(probe_path, PROBE_SHA256, "frozen checkpoint")
    protocol = read(alternate / "protocol.json")
    manifest = read(alternate / "run-manifest.json")
    require(manifest["status"] == "complete" and manifest["inputs"] == 1200, "Sensitivity inference incomplete")
    check_digest(root / "protocol.json", protocol["primary_protocol_sha256"], "primary protocol")
    check_digest(root / "inputs.jsonl", protocol["primary_inputs_sha256"], "primary inputs")
    check_digest(alternate / "inputs.jsonl", protocol["altered_inputs_sha256"], "sensitivity input freeze")
    check_digest(alternate / "inputs.jsonl", manifest["inputs_sha256"], "sensitivity inputs at completion")
    check_digest(alternate / "input-summary.csv", manifest["summary_sha256"], "sensitivity summary")
    runner = Path(__file__).with_name("run_role_signal_length_sensitivity.py")
    analysis = Path(__file__).with_name("analyze_role_signal_length_sensitivity.py")
    check_digest(runner, protocol["code_sha256"], "sensitivity runner freeze")
    check_digest(runner, manifest["runner_sha256"], "sensitivity runner at completion")
    check_digest(analysis, protocol["analysis_code_sha256"], "sensitivity analysis freeze")
    require(protocol["probe_sha256"] == manifest["probe_sha256"] == PROBE_SHA256, "Sensitivity probe identity changed")
    require(protocol["primary_unchanged"] is True and protocol["formal_inputs"] == 1200, "Incorrect diagnostic scope")
    primary_manifest = read(root / "run-manifest.json")
    check_digest(root / "input-summary.csv", primary_manifest["summary_sha256"], "unchanged primary completed summary")
    primary = [row for row in jsonl(root / "inputs.jsonl") if row["split"] == "formal"]
    altered = jsonl(alternate / "inputs.jsonl")
    require(len(primary) == len(altered) == 1200, "Expected all 1200 formal inputs")
    require(len({row["input_id"] for row in altered}) == 1200, "Repeated alternate input IDs")
    require(len({row["pair_id"] for row in altered}) == 50, "Incorrect independent semantic-unit count")
    for before, after in zip(primary, altered):
        key = before["input_id"]
        require(before["input_ids"][-1] == 200007 and before["body_end"] == len(before["input_ids"]) - 1,
                "Primary suffix is not the single native ending: " + key)
        require(set(before) == set(after), "Sensitivity input metadata schema changed: " + key)
        require(after["input_ids"] == before["input_ids"][:-1], "Changed preceding/body token IDs: " + key)
        require(after["body_end"] == len(after["input_ids"]), "Sensitivity input does not end at body end: " + key)
        require(after["rendered_prompt"] == before["rendered_prompt"][:-len("<|end|>")],
                "Sensitivity rendered prompt changed beyond ending removal: " + key)
        require(all(before[name] == after[name] for name in before if name not in {"input_ids", "rendered_prompt"}),
                "Changed factorial metadata or body coordinates: " + key)
        require(after["input_ids"][after["body_start"]:after["body_end"]] == after["body_token_ids"],
                "Sensitivity body ownership differs: " + key)
    summaries = csv_rows(alternate / "input-summary.csv")
    saved = {row["input_id"]: row for row in summaries}
    require(len(saved) == len(summaries) == 1200 and set(saved) == {row["input_id"] for row in altered},
            "Sensitivity summary inventory differs")
    original_summary_rows = csv_rows(root / "input-summary.csv")
    primary_summaries = {row["input_id"]: row for row in original_summary_rows if row["split"] == "formal"}
    require(set(primary_summaries) == set(saved), "Primary/sensitivity summary matching differs")
    with np.load(probe_path, allow_pickle=False) as archive:
        parameters = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(parameters["metadata"]))
    require(tuple(metadata["roles"]) == ROLES, "Probe role order changed")
    manifest_files = manifest["readout_sha256"]
    require(len(manifest_files) == 1200, "Readout hash inventory incomplete")
    require(set(manifest_files) == {row["input_id"] + ".npz" for row in altered}, "Readout hash IDs differ")
    require(len(list((alternate / "readouts").glob("*.npz"))) == 1200, "Stale or missing sensitivity archive")
    errors, feature_checks = {}, []
    max_softmax, max_logits, max_probability, body_tokens = 0.0, 0.0, 0.0, 0
    input_deltas = []
    for index, row in enumerate(altered):
        key = row["input_id"]
        path = alternate / "readouts" / (key + ".npz")
        check_digest(path, manifest_files[path.name], "sensitivity readout " + key)
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"positions", "logits", "probabilities", "roles"} | ({"features"} if index % 120 == 0 else set()),
                    "Unexpected readout schema or feature spot-check schedule: " + key)
            positions, logits, probabilities = (archive[name] for name in ("positions", "logits", "probabilities"))
            require(tuple(archive["roles"].tolist()) == ROLES, "Sensitivity role order changed: " + key)
            require(np.array_equal(positions, np.arange(row["body_start"], row["body_end"])), "Sensitivity body positions changed: " + key)
            require(logits.shape == probabilities.shape == (len(positions), 6), "Sensitivity token denominator differs: " + key)
            require(np.isfinite(logits).all() and np.isfinite(probabilities).all(), "Nonfinite sensitivity readout: " + key)
            require((probabilities >= 0).all() and (probabilities <= 1).all(), "Invalid sensitivity probabilities: " + key)
            error = float(np.abs(softmax(logits) - probabilities).max(initial=0))
            max_softmax = max(max_softmax, error)
            require(error < 2e-6, "Sensitivity softmax differs: " + key)
            if "features" in archive.files:
                independent = numpy_logits(archive["features"], parameters, metadata)
                require(independent.shape == logits.shape, "Independent feature/body shape mismatch: " + key)
                delta = float(np.abs(independent - logits).max(initial=0))
                probability_delta = float(np.abs(softmax(independent) - probabilities).max(initial=0))
                require(np.allclose(independent, logits, atol=3e-4, rtol=3e-5) and probability_delta < 1e-4,
                        "Independent sensitivity probe differs: " + key)
                max_logits, max_probability = max(max_logits, delta), max(max_probability, probability_delta)
                feature_checks.append({"input_id": key, "archive_sha256": manifest_files[path.name],
                                       "max_logit_error": delta, "max_probability_error": probability_delta})
        summary = saved[key]
        for name in ("pair_id", "split", "filler_id", "text_style", "wrapper", "position", "body_tokens"):
            require(summary[name] == str(row[name]), "Sensitivity condition/denominator summary differs: " + key + "/" + name)
            require(primary_summaries[key][name] == summary[name], "Sensitivity pairing metadata differs: " + key + "/" + name)
        margin = (logits[:, 3].astype(np.float64) - logits[:, 2]).mean()
        compare_fields(summary, ("mean_margin",), margin, errors, atol=1e-5)
        for role_id, role in enumerate(ROLES):
            compare_fields(summary, ("p_" + role,), probabilities[:, role_id].mean(dtype=np.float64), errors)
            compare_fields(summary, ("z_" + role,), logits[:, role_id].mean(dtype=np.float64), errors, atol=1e-5)
        body_tokens += len(positions)
        input_deltas.append(float(summary["mean_margin"]) - float(primary_summaries[key]["mean_margin"]))
    require(len(feature_checks) == 10, "Expected ten independent saved-feature checks")
    comparison = validate_paired_comparison(root, original_summary_rows, summaries)
    return {
        "status": "passed", "scope": "Artifact, input derivation and readout consistency; does not establish global causal or numerical invariance",
        "method": "Independent raw-array audit and existing independent NumPy/math.erf probe; no inference/analysis implementation imports",
        "probe_sha256": PROBE_SHA256, "validator_sha256": sha(Path(__file__)),
        "independent_numpy_helper_sha256": sha(Path(__file__).with_name("validate_role_signal_factorial.py")),
        "primary_protocol_sha256": sha(root / "protocol.json"), "primary_inputs_sha256": sha(root / "inputs.jsonl"),
        "primary_summary_sha256": sha(root / "input-summary.csv"),
        "sensitivity_protocol_sha256": sha(alternate / "protocol.json"), "sensitivity_manifest_sha256": sha(alternate / "run-manifest.json"),
        "sensitivity_inputs_sha256": sha(alternate / "inputs.jsonl"), "sensitivity_summary_sha256": sha(alternate / "input-summary.csv"),
        "formal_inputs": 1200, "independent_semantic_pairs": 50, "conditions_per_pair": 24, "scored_body_tokens": body_tokens,
        "all_inputs_exactly_primary_minus_final_id200007": True, "all_other_metadata_and_body_token_ids_preserved": True,
        "all1200_readout_hashes_verified": True, "all_body_positions_and_six_role_orders_preserved": True,
        "maximum_softmax_difference": max_softmax, "maximum_summary_differences": errors,
        "independent_probe_spot_checks": feature_checks,
        "maximum_independent_probe_logit_difference": max_logits,
        "maximum_independent_probe_probability_difference": max_probability,
        "input_mean_margin_change": {"mean": float(np.mean(input_deltas)), "max_absolute": float(np.max(np.abs(input_deltas))),
                                     "minimum": float(np.min(input_deltas)), "maximum": float(np.max(input_deltas))},
        "paired_effect_comparison_validation": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Primary experiment directory")
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    args = parser.parse_args()
    destination = args.output.resolve() / "validation-length-sensitivity.json"
    try:
        result = validate(args)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "validator_sha256": sha(Path(__file__))}
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
