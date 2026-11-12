"""Independently audit the frozen role-signal factorial inputs and readouts.

Uses only Python and NumPy; imports neither the experiment runner nor its probe,
statistics, or plotting implementation. Body coordinates are zero-based and end
exclusive. The exported probe is evaluated explicitly with math.erf on a fixed,
spread-out sample of ten archives, while every saved softmax and summary is
checked. This verifies artifact consistency, not semantic equivalence of rewrites.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
POSITIONS = (32, 256, 768)
PROBE_SHA256 = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"
DEFAULT_PROBE = Path(__file__).resolve().parent / "outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz"
# All published GPT-OSS Harmony special IDs, including non-message controls.
SPECIAL_ID_FLOOR = 199998
# Independently tokenized with the tokenizer assets bound inside the exact probe.
NATIVE_HEADERS = {
    "cot": [200006, 173781, 200005, 35644, 200008],
    "user": [200006, 1428, 200008],
}
SYSTEM_PREFIX = [200006, 17360, 200008, 3575, 553, 261, 10297, 29186, 13, 200007]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def numpy_logits(features, parameters, metadata):
    """Explicit exported-network computation, independent of Torch/SciPy."""
    features = np.asarray(features)
    require(features.ndim == 2 and features.shape[1] == 1024, "Invalid feature geometry")
    require(features.dtype == np.float16, "Persisted features must retain the frozen FP16 cast")
    require(np.isfinite(features).all(), "Nonfinite persisted features")
    # Float64 accumulation and scalar libm erf provide an independent numerical
    # implementation. Small differences from the inference float32 BLAS are OK.
    hidden = (features.astype(np.float64) - parameters["mean"]) / parameters["scale"]
    hidden = hidden @ parameters["weight::0.weight"].astype(np.float64).T + parameters["weight::0.bias"]
    if metadata["hidden"]:
        error_function = np.frompyfunc(math.erf, 1, 1)
        hidden *= 0.5 * (1.0 + error_function(hidden / math.sqrt(2.0)).astype(np.float64))
        hidden = hidden @ parameters["weight::3.weight"].astype(np.float64).T + parameters["weight::3.bias"]
    return hidden


def field(row, names, required=True):
    found = [(name, row[name]) for name in names if name in row and row[name] != ""]
    if required:
        require(found, "Missing summary field; expected one of " + ", ".join(names))
    return found


def compare_fields(row, names, expected, errors, required=True, atol=1e-6):
    for name, value in field(row, names, required):
        error = abs(float(value) - float(expected))
        errors[name] = max(errors.get(name, 0.0), error)
        require(np.isfinite(float(value)) and error <= atol + 1e-7 * abs(expected),
                f"Summary differs for {row.get('input_id')}/{name}: {value} versus {expected}")


def leaves(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, (*prefix, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, (*prefix, str(index)))
    else:
        yield prefix, value


def hash_checks(root, protocol, probe_path):
    """Verify explicit hashes whose file targets are unambiguous in this schema."""
    paths = {
        "probe_sha256": probe_path, "checkpoint_sha256": probe_path,
        "inputs_sha256": root / "inputs.jsonl", "pairs_sha256": root / "pairs.jsonl",
        "input_summary_sha256": root / "input-summary.csv",
        "summary_sha256": root / "input-summary.csv",
        "effects_sha256": root / "effects.csv", "interactions_sha256": root / "interactions.csv",
        "bootstrap_summary_sha256": root / "bootstrap-summary.json",
        "analysis_script_sha256": Path(__file__).with_name("analyze_role_signal_factorial.py"),
        "analysis_code_sha256": Path(__file__).with_name("analyze_role_signal_factorial.py"),
        "runner_sha256": Path(__file__).with_name("run_role_signal_factorial.py"),
    }
    checked = []
    for record_name, record in protocol.items():
        for keys, digest in leaves(record):
            if not keys or not isinstance(digest, str):
                continue
            label = keys[-1]
            target = paths.get(label)
            # Filename -> digest maps, including analysis manifest input hashes.
            if target is None and label in {"inputs.jsonl", "pairs.jsonl", "input-summary.csv", "effects.csv", "interactions.csv"}:
                target = root / label
            if target is not None:
                require(target.is_file(), "Hash target absent: " + str(target))
                require(sha(target) == digest, f"Hash mismatch at {record_name}/{'.'.join(keys)}")
                checked.append({"record": record_name, "key": ".".join(keys), "path": str(target), "sha256": digest})
    require(any(item["path"] == str(probe_path) for item in checked), "Protocol must bind the exact checkpoint hash")
    return checked


def validate_statistics(root, summaries, bootstrap):
    """Recompute signed contrasts as explicit within-pair sums, then resample.

    This operates on scalar dictionary cells, independently of the analyzer's
    tensor reshaping, axis reductions and bootstrap multiplicity matrices.
    """
    settings = bootstrap["bootstrap"]
    require(int(settings["replicates"]) == 2000 and settings["unit"] == "semantic pair_id",
            "Bootstrap must preserve complete semantic units")
    formal = [row for row in summaries if row["split"] == "formal"]
    pair_ids = sorted({row["pair_id"] for row in formal})
    fillers = sorted({row["filler_id"] for row in formal})
    require(fillers == ["A", "B"], "Expected independently replicated contexts A/B")
    cells = {(row["pair_id"], row["filler_id"], row["text_style"], row["wrapper"], int(row["position"])): row
             for row in formal}
    require({row["text_style"] for row in formal} == {"styled", "destyled"} and
            {row["wrapper"] for row in formal} == {"cot", "user"}, "Unexpected contrast orientation")
    metrics = ["mean_margin", *("p_" + role for role in ROLES), "probability_contrast"]
    contrast_names = {
        "effects.csv": ["text_styled_minus_destyled", "wrapper_cot_minus_user",
                        "position_256_minus_32", "position_768_minus_32"],
        "interactions.csv": ["text_x_wrapper_at_32", "text_x_wrapper_at_256", "text_x_wrapper_at_768",
                             "text_x_wrapper_256_minus_32", "text_x_wrapper_768_minus_32"],
    }
    samples = np.random.default_rng(int(settings["seed"])).integers(0, len(pair_ids), (2000, len(pair_ids)))

    def contrasts(pair_id, filler_id, metric):
        def value(text, wrapper, position):
            row = cells[pair_id, filler_id, text, wrapper, position]
            return float(row[metric]) if metric != "probability_contrast" else float(row["p_cot"]) - float(row["p_user"])
        text = sum(value("styled", w, p) - value("destyled", w, p) for w in ("cot", "user") for p in POSITIONS) / 6
        wrapper = sum(value(t, "cot", p) - value(t, "user", p) for t in ("styled", "destyled") for p in POSITIONS) / 6
        position = [sum(value(t, w, p) - value(t, w, 32) for t in ("styled", "destyled") for w in ("cot", "user")) / 4
                    for p in (256, 768)]
        interactions = [value("styled", "cot", p) - value("destyled", "cot", p) - value("styled", "user", p) + value("destyled", "user", p)
                        for p in POSITIONS]
        return [text, wrapper, *position], [*interactions, interactions[1] - interactions[0], interactions[2] - interactions[0]]

    maximum, counts = 0.0, {}
    for table_name, names in contrast_names.items():
        with (root / table_name).open(newline="", encoding="utf-8-sig") as stream:
            table = list(csv.DictReader(stream))
        indexed = {(row["filler_id"], row["contrast"], row["metric"]): row for row in table}
        expected = set(itertools.product(["A", "B", "pooled"], names, metrics))
        require(len(table) == len(indexed) and set(indexed) == expected, "Incomplete effect/interaction table: " + table_name)
        which = 0 if table_name == "effects.csv" else 1
        for metric in metrics:
            vectors = {f: np.asarray([contrasts(pair, f, metric)[which] for pair in pair_ids]) for f in fillers}
            vectors["pooled"] = (vectors["A"] + vectors["B"]) / 2
            for filler_id, matrix in vectors.items():
                resampled = matrix[samples].mean(axis=1)
                bounds = np.quantile(resampled, [0.025, 0.975], axis=0)
                for index, contrast in enumerate(names):
                    row = indexed[filler_id, contrast, metric]
                    require(int(row["n_pairs"]) == len(pair_ids), "Incorrect contrast independent sample count")
                    for key, actual in (("estimate", matrix[:, index].mean()), ("ci_lower", bounds[0, index]), ("ci_upper", bounds[1, index])):
                        error = abs(float(row[key]) - actual)
                        maximum = max(maximum, error)
                        require(error < 1e-8 + abs(actual) * 1e-9, "Independent paired contrast/bootstrap differs: " + table_name + "/" + str((filler_id, contrast, metric, key)))
        counts[table_name] = len(table)
    return {"replicates": 2000, "seed": int(settings["seed"]), "unit": "semantic pair_id",
            "all_main_effect_and_interaction_estimates_and_intervals_recomputed": True,
            "table_rows": counts, "maximum_scalar_contrast_or_interval_difference": maximum}


def validate(args):
    root, probe_path = args.output.resolve(), args.probe.resolve()
    require(sha(probe_path) == PROBE_SHA256, "Requested frozen checkpoint bytes differ")
    with np.load(probe_path, allow_pickle=False) as archive:
        parameters = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(parameters["metadata"]))
    require(tuple(metadata["roles"]) == ROLES, "Checkpoint role order differs")
    require(metadata["feature_position"] == "pre-o_proj" and metadata["feature_dim"] == 1024,
            "Checkpoint feature interface differs")
    require(metadata["activation"] == "gelu_exact", "Checkpoint activation differs")
    require(parameters["mean"].shape == parameters["scale"].shape == (1024,), "Invalid standardizer geometry")
    require((parameters["scale"] > 0).all(), "Nonpositive standardizer scale")
    inputs, pairs = jsonl(root / "inputs.jsonl"), jsonl(root / "pairs.jsonl")
    require(inputs, "No experimental inputs")
    input_ids = [row["input_id"] for row in inputs]
    require(len(set(input_ids)) == len(input_ids), "Repeated input_id")
    pair_by_id = {row["pair_id"]: row for row in pairs}
    require(len(pair_by_id) == len(pairs), "Repeated pair_id")
    require(set(row["split"] for row in inputs) <= {"pilot", "formal"}, "Unexpected main input split")
    by_pair, split_pairs = defaultdict(list), defaultdict(set)
    text_tokens, header_offsets, boundary_exclusions = {}, defaultdict(set), Counter()
    for row in inputs:
        key = row["input_id"]
        require(str(key) == Path(str(key)).name and key not in {".", ".."}, "Unsafe archive name")
        pair_id, split = row["pair_id"], row["split"]
        require(pair_id in pair_by_id, "Input has no saved semantic pair: " + key)
        require(pair_by_id[pair_id]["split"] == split, "Pair/input split mismatch: " + key)
        split_pairs[split].add(pair_id)
        by_pair[pair_id].append(row)
        ids = np.asarray(row["input_ids"], dtype=np.int64)
        body = np.asarray(row["body_token_ids"], dtype=np.int64)
        start, end = int(row["body_start"]), int(row["body_end"])
        require(int(row["position"]) in POSITIONS and start == int(row["position"]), "Wrong zero-based body position: " + key)
        require(ids.ndim == body.ndim == 1 and len(body) > 0 and 0 <= start < end <= len(ids), "Invalid body region: " + key)
        require(end - start == len(body) and np.array_equal(ids[start:end], body), "Body IDs/region disagree: " + key)
        require((body >= 0).all() and (body < SPECIAL_ID_FLOOR).all(), "Special token occurs in scored body: " + key)
        header_start, header_length = int(row["header_start"]), int(row["header_length"])
        require(0 <= header_start < start and header_length > 0 and header_start + header_length <= start,
                "Header overlaps scored body: " + key)
        require(row["wrapper"] in NATIVE_HEADERS, "Unexpected factorial wrapper: " + key)
        native_header = NATIVE_HEADERS[row["wrapper"]]
        require(header_length == len(native_header) and ids[header_start:start].tolist() == native_header,
                "Scored role header differs from bound native Harmony IDs: " + key)
        require(ids[:len(SYSTEM_PREFIX)].tolist() == SYSTEM_PREFIX and ids[header_start - 1] == 200007 and ids[-1] == 200007,
                "System prefix or message boundary differs: " + key)
        require(ids[len(SYSTEM_PREFIX):len(SYSTEM_PREFIX) + 3].tolist() == NATIVE_HEADERS["user"],
                "Neutral prefix message lacks its native user header: " + key)
        expected_body = pair_by_id[pair_id][row["text_style"]]
        if "body_sha256" in row:
            require(row["body_sha256"] == hashlib.sha256(expected_body.encode()).hexdigest(),
                    "Body text hash differs from saved semantic pair: " + key)
        if "rendered_prompt" in row:
            require(row["rendered_prompt"].endswith(expected_body + "<|end|>"),
                    "Rendered prompt body differs from saved semantic pair: " + key)
        header_offsets[row["wrapper"]].add(start - header_start)
        text_key = pair_id, row["text_style"]
        if text_key in text_tokens:
            require(text_tokens[text_key] == body.tolist(), "Text changed across wrapper/position/filler: " + key)
        text_tokens[text_key] = body.tolist()
        excluded = row["excluded_boundary_tokens"]
        if isinstance(excluded, list):
            require(all(isinstance(i, int) and 0 <= i < len(ids) and not start <= i < end for i in excluded),
                    "Excluded boundary token enters main body: " + key)
            excluded = len(excluded)
        require(isinstance(excluded, int) and excluded >= 0, "Invalid boundary exclusion count: " + key)
        boundary_exclusions[split] += excluded
    require(not (split_pairs["pilot"] & split_pairs["formal"]), "Pilot/formal semantic pair overlap")
    require(len(split_pairs["formal"]) == args.formal_pairs, "Wrong number of formal semantic pairs")
    require(len(split_pairs["pilot"]) == args.pilot_pairs, "Wrong number of independent pilot pairs")
    require(all(len(values) == 1 for values in header_offsets.values()), "Header-to-body distance changed within wrapper")
    factor_levels = {}
    for split in ("pilot", "formal"):
        rows = [row for row in inputs if row["split"] == split]
        levels = {name: sorted({row[name] for row in rows}) for name in ("filler_id", "text_style", "wrapper")}
        require(len(levels["text_style"]) == len(levels["wrapper"]) == 2, "Expected two text/wrapper levels: " + split)
        require(len(levels["filler_id"]) == 2 if split == "formal" else len(levels["filler_id"]) in (1, 2),
                "Expected two formal fillers and one/two pilot fillers")
        expected = set(itertools.product(levels["filler_id"], levels["text_style"], levels["wrapper"], POSITIONS))
        for pair_id in split_pairs[split]:
            actual = [(row["filler_id"], row["text_style"], row["wrapper"], int(row["position"])) for row in by_pair[pair_id]]
            require(len(actual) == len(expected) and set(actual) == expected, "Incomplete/repeated factorial pair: " + pair_id)
        factor_levels[split] = {**levels, "position": list(POSITIONS)}
    require(all(factor_levels["pilot"][name] == factor_levels["formal"][name] for name in ("text_style", "wrapper", "position")),
            "Pilot and formal text/wrapper/position levels differ")
    require(set(factor_levels["pilot"]["filler_id"]) <= set(factor_levels["formal"]["filler_id"]), "Pilot fillers differ from formal")
    # Detect exact text duplication between semantic units, especially cross split.
    owners = defaultdict(set)
    for (pair_id, style), tokens in text_tokens.items():
        owners[tuple(tokens)].add(pair_id)
    require(all(len(values) == 1 for values in owners.values()), "Exact body duplicated across distinct semantic units")
    with (root / "input-summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        summaries = list(csv.DictReader(stream))
    summary_by_id = {row["input_id"]: row for row in summaries}
    require(len(summary_by_id) == len(summaries) == len(inputs) and set(summary_by_id) == set(input_ids),
            "Summary input inventory differs")
    spot_indices = set(np.linspace(0, len(inputs) - 1, min(args.feature_spot_checks, len(inputs)), dtype=int).tolist())
    max_softmax, max_probe_logits, max_probe_probability, max_summary = 0.0, 0.0, 0.0, {}
    feature_files, scored_tokens = [], 0
    for index, row in enumerate(inputs):
        key = row["input_id"]
        path = root / "readouts" / (key + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            positions, logits, probabilities = (archive[name] for name in ("positions", "logits", "probabilities"))
            require(tuple(archive["roles"].tolist()) == ROLES, "Archive role order differs: " + key)
            expected_positions = np.arange(row["body_start"], row["body_end"])
            require(np.array_equal(positions, expected_positions), "Readout positions differ from main body: " + key)
            shape = (len(positions), len(ROLES))
            require(logits.shape == probabilities.shape == shape, "Readout shape differs: " + key)
            require(np.isfinite(logits).all() and np.isfinite(probabilities).all(), "Nonfinite readout: " + key)
            require((probabilities >= 0).all() and (probabilities <= 1).all(), "Invalid probability range: " + key)
            error = float(np.abs(softmax(logits) - probabilities).max(initial=0))
            max_softmax = max(max_softmax, error)
            require(error < 2e-6, "Saved probabilities differ from token softmax: " + key)
            if index in spot_indices:
                require("features" in archive.files, "Missing features for independent probe spot check: " + key)
                independent = numpy_logits(archive["features"], parameters, metadata)
                require(independent.shape == shape, "Feature/token denominator mismatch: " + key)
                delta = float(np.abs(independent - logits).max(initial=0))
                probability_delta = float(np.abs(softmax(independent) - probabilities).max(initial=0))
                max_probe_logits, max_probe_probability = max(max_probe_logits, delta), max(max_probe_probability, probability_delta)
                require(np.allclose(independent, logits, atol=3e-4, rtol=3e-5) and probability_delta < 1e-4,
                        "Independent exact-GELU probe differs: " + key)
                feature_files.append({"input_id": key, "archive_sha256": sha(path), "max_logit_error": delta,
                                      "max_probability_error": probability_delta})
        summary = summary_by_id[key]
        for name in ("pair_id", "split", "filler_id", "text_style", "wrapper"):
            if name in summary:
                require(summary[name] == str(row[name]), "Summary condition differs: " + key + "/" + name)
        compare_fields(summary, ("position", "body_start"), row["body_start"], max_summary, atol=0)
        compare_fields(summary, ("body_tokens", "n_body_tokens", "body_token_count", "tokens"), len(positions), max_summary, atol=0)
        mean_logits, mean_probabilities = logits.mean(0, dtype=np.float64), probabilities.mean(0, dtype=np.float64)
        margin = (logits[:, 3].astype(np.float64) - logits[:, 2]).mean()
        compare_fields(summary, ("margin", "mean_margin", "margin_cot_minus_user", "mean_margin_cot_user"), margin, max_summary, atol=1e-5)
        for role_id, role in enumerate(ROLES):
            compare_fields(summary, ("p_" + role, "prob_" + role, "mean_probability_" + role, "mean_p_" + role),
                           mean_probabilities[role_id], max_summary)
            compare_fields(summary, ("z_" + role, "logit_" + role, "mean_logit_" + role),
                           mean_logits[role_id], max_summary, required=False, atol=1e-5)
        scored_tokens += len(positions)
    require(len(list((root / "readouts").glob("*.npz"))) == len(inputs), "Unexpected stale/main readout archives")
    records = {"protocol": read(root / "protocol.json")}
    for name in ("run-manifest.json", "analysis-manifest.json", "bootstrap-summary.json"):
        if (root / name).is_file():
            records[name] = read(root / name)
    checked_hashes = hash_checks(root, records, probe_path)
    bootstrap_check = None
    if (root / "effects.csv").is_file():
        require("bootstrap-summary.json" in records, "Effects lack bootstrap provenance")
        bootstrap = records["bootstrap-summary.json"]
        all_leaves = list(leaves(bootstrap))
        counts = [value for keys, value in all_leaves if keys[-1] in {"replicates", "bootstrap_replicates", "n_bootstrap", "n_resamples"}]
        require(counts and all(int(value) == 2000 for value in counts), "Expected frozen 2000-replicate bootstrap")
        require(any("seed" in keys[-1].lower() for keys, _ in all_leaves), "Bootstrap seed absent")
        serialized = json.dumps(bootstrap).lower()
        require(any(word in serialized for word in ("pair", "semantic", "unit", "cluster")), "Bootstrap resampling unit absent")
        bootstrap_check = validate_statistics(root, summaries, bootstrap)
    return {
        "status": "passed", "method": "Independent raw-array audit and explicit NumPy/math.erf probe; no runner imports",
        "scope": "Artifact consistency; semantic pair equivalence requires separate content review",
        "probe_sha256": sha(probe_path), "validator_sha256": sha(Path(__file__)),
        "inputs_sha256": sha(root / "inputs.jsonl"), "pairs_sha256": sha(root / "pairs.jsonl"),
        "input_summary_sha256": sha(root / "input-summary.csv"), "inputs": len(inputs), "scored_body_tokens": scored_tokens,
        "split_pair_counts": {split: len(values) for split, values in split_pairs.items()},
        "conditions_per_pair": 24, "factor_levels": factor_levels,
        "zero_based_body_positions": list(POSITIONS), "all_body_ids_exact": True, "all_body_tokens_ordinary": True,
        "all_role_headers_match_bound_native_Harmony_token_ids": True,
        "pilot_formal_pairs_disjoint": True, "same_body_ids_across_interventions": True,
        "header_to_body_offsets": {str(key): sorted(values) for key, values in header_offsets.items()},
        "excluded_boundary_tokens": dict(boundary_exclusions),
        "maximum_softmax_difference": max_softmax, "maximum_summary_differences": max_summary,
        "independent_probe_spot_checks": feature_files, "maximum_independent_probe_logit_difference": max_probe_logits,
        "maximum_independent_probe_probability_difference": max_probe_probability,
        "checked_provenance_hashes": checked_hashes, "bootstrap_provenance": bootstrap_check,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--formal-pairs", type=int, default=50)
    parser.add_argument("--pilot-pairs", type=int, default=10)
    parser.add_argument("--feature-spot-checks", type=int, default=10)
    args = parser.parse_args()
    require(args.formal_pairs > 0 and args.pilot_pairs > 0 and args.feature_spot_checks > 0, "Counts must be positive")
    destination = args.output.resolve() / "validation-independent.json"
    try:
        result = validate(args)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "validator_sha256": sha(Path(__file__))}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
