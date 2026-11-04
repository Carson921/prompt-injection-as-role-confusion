"""Independently verify every rank, 20-bin ASR, and paired-bootstrap interval.

Does not import the producer or plotting code. Uses lexsort, raw native episode
outcomes, and a separate per-bin reconstruction from the saved bootstrap draws.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
PARTS = {"template": "payload_without_goal", "goal": "goal"}
LATEST_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    root = parser.parse_args().output.resolve()
    metadata = read(root / "metadata.json")
    predictions = Path(metadata["predictions_root"])
    source = Path(metadata["original_root"])
    require(metadata["probe_sha256"] == read(predictions / "rescore_manifest.json")["probe_sha256"] == LATEST_SHA, "Wrong probe")
    for name, digest in metadata["source_prediction_hashes"].items():
        require(sha(predictions / name) == digest, "Latest probability source changed")
    require(sha(source / "result_manifest.json") == metadata["source_result_manifest_sha256"], "Native source seal changed")
    native_inventory = {r["path"]: r["sha256"] for r in read(source / "result_manifest.json")["files"]}
    native = {}
    for case in csv_read(root / "all_778_cases.csv"):
        key = "cases/" + case["case_id"] + ".json"
        path = source / key
        require(sha(path) == native_inventory[key], "Native outcome bytes changed")
        native[case["case_id"]] = read(path)
        expected = native[case["case_id"]]["security"]
        actual = None if case["attack_success"] == "" else case["attack_success"] == "True"
        require(expected is actual, "Native attack score differs")
    require(len(native) == 778 and sum(r["security"] is None for r in native.values()) == 3, "Wrong native coverage")
    pair_ids = sorted({key.split("/", 1)[1] for key in native})
    with np.load(root / "bootstrap_pair_weights.npz", allow_pickle=False) as z:
        saved_pairs = z["pair_ids"].tolist()
        weights = z["weights"]
        seed = int(z["seed"])
    require(saved_pairs == pair_ids and len(pair_ids) == 389 and weights.shape == (10000, 389), "Wrong paired bootstrap")
    require(np.all(weights.sum(1) == 389), "Bootstrap does not draw 389 pairs")
    regenerated = np.random.default_rng(seed).multinomial(389, np.full(389, 1/389), size=10000)
    require(np.array_equal(weights, regenerated), "Bootstrap seed replay differs")
    pair_index = {key: i for i, key in enumerate(pair_ids)}
    original = csv_read(predictions / "per_case_probabilities.csv")
    assignments = csv_read(root / "ranked_cases.csv")
    bins = csv_read(root / "bins.csv")
    require(len(assignments) == 9060 and len(bins) == 240, "Wrong output size")
    expected_sizes = [38] * 15 + [37] * 5
    expected_group = np.concatenate([np.full(n, i+1) for i, n in enumerate(expected_sizes)])
    checked, max_ci_error, tie_statistics = 0, 0., {}
    for part, csv_part in PARTS.items():
        rows = sorted([r for r in original if r["part"] == csv_part], key=lambda r: r["case_id"])
        identifiers = np.asarray([r["case_id"] for r in rows])
        require(len(rows) == len(set(identifiers)) == 755, "Wrong observed probability cohort")
        for case in csv_read(root / "all_778_cases.csv"):
            require((case["case_id"] in set(identifiers)) == (case["probe_available"] == "True"), "Missing probability scope differs")
            if case["probe_available"] != "True":
                require(all(case[part + "_p_" + role] == "" for role in ROLES), "Missing probability was imputed")
        for role in ROLES:
            probabilities = np.asarray([float(r["p_" + role]) for r in rows])
            permutation = np.lexsort((identifiers, probabilities))
            expected_ids = identifiers[permutation]
            expected_p = probabilities[permutation]
            emitted = [r for r in assignments if r["part"] == part and r["role"] == role]
            emitted = sorted(emitted, key=lambda r: int(r["rank"]))
            require([r["case_id"] for r in emitted] == expected_ids.tolist(), "Probability/case-ID rank order differs")
            require([int(r["rank"]) for r in emitted] == list(range(1, 756)), "Rank positions differ")
            require(np.array_equal(np.asarray([int(r["bin"]) for r in emitted]), expected_group), "20 groups are not equal-count")
            require(np.array_equal(np.asarray([float(r["probe_probability"]) for r in emitted]), expected_p), "Wrong probability column")
            role_bins = sorted([r for r in bins if r["part"] == part and r["role"] == role], key=lambda r: int(r["bin"]))
            require(len(role_bins) == 20, "Not 20 bins per role/part")
            successes_total = scored_total = 0
            for group_number, result in enumerate(role_bins, 1):
                indices = np.flatnonzero(expected_group == group_number)
                ids = expected_ids[indices].tolist()
                p = expected_p[indices]
                scored = [key for key in ids if native[key]["security"] is not None]
                success = [key for key in scored if native[key]["security"] is True]
                require(int(result["ranked_cases"]) == len(ids) == expected_sizes[group_number-1], "Wrong bin size")
                require(int(result["scored_cases"]) == len(scored) and int(result["unscored_cases"]) == len(ids)-len(scored), "Wrong ASR denominator")
                require(int(result["attack_successes"]) == len(success) and int(result["attack_failures"]) == len(scored)-len(success), "Wrong bin outcomes")
                require(float(result["asr"]) == len(success)/len(scored), "ASR does not reproduce")
                require(float(result["probability_min"]) == p.min() and float(result["probability_max"]) == p.max(), "Probability range differs")
                require(abs(float(result["probability_mean"])-float(np.mean(p))) < 1e-15, "Probability mean differs")
                require(int(result["plain_cases"]) == sum(key.startswith("plain/") for key in ids), "Wrong plain composition")
                require(int(result["remember_cases"]) == sum(key.startswith("prerequisite/") for key in ids), "Wrong Remember composition")
                require(float(result["percentile_lower"]) == 100 * indices[0] / 755 and float(result["percentile_upper"]) == 100 * (indices[-1]+1) / 755, "Wrong percentile boundaries")
                # Independently sum cluster multiplicities for each scored row;
                # the same pair index appears twice when both conditions occur.
                denominator = weights[:, [pair_index[key.split("/",1)[1]] for key in scored]].astype(np.float64).sum(1)
                numerator = np.zeros(10000)
                for key in success:
                    numerator += weights[:, pair_index[key.split("/",1)[1]]]
                valid = denominator > 0
                low, high = np.percentile(numerator[valid]/denominator[valid], [2.5,97.5])
                error = max(abs(float(result["ci95_low"])-low), abs(float(result["ci95_high"])-high))
                require(error < 1e-12 and int(result["valid_bootstrap_replicates"]) == int(valid.sum()), "Bootstrap interval differs")
                max_ci_error = max(max_ci_error, error)
                successes_total += len(success)
                scored_total += len(scored)
                checked += 1
            require(scored_total == 752 and successes_total == 231, "Role aggregate outcomes differ")
            tie_statistics[part + "/" + role] = {"unique_probabilities": len(set(expected_p)), "largest_tie": max(Counter(expected_p).values())}
    validation = {"status": "passed", "probe_sha256": LATEST_SHA, "source_outcomes_checked": len(native),
                  "probability_sorts_checked": 12, "bin_asrs_checked": checked, "rank_assignments_checked": len(assignments),
                  "bootstrap_replicates": 10000, "seed_replay_exact": True,
                  "maximum_confidence_interval_error": max_ci_error,
                  "missing_probabilities_not_imputed": True, "null_outcomes_excluded_only_from_asr": True,
                  "tie_statistics": tie_statistics, "validator_sha256": sha(Path(__file__))}
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
