"""Independently recompute every published role-intervention mean and interval.

Reads scalar input-summary cells and the persisted task-bootstrap draw indices;
does not import the analyzer or reshape its condition tensor. Every point and
paired pointwise 95% percentile interval in effects.csv, swap-effects.csv, and
full-role-probabilities.csv is checked for the exact FIT/all task cohorts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
SEED, RESAMPLES = 20261112, 2000


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def audit_cohort(stage, all_rows, cohort):
    folder = stage / "analysis" / cohort
    rows = [row for row in all_rows if cohort != "fit" or row["source_split"] == "fit"]
    traces = sorted({row["trace_id"] for row in rows})
    n = len(traces)
    require(n == {"pilot": 1, "fit": 40, "all": 57}[cohort], "Incorrect scientific cohort support")
    require(len(rows) == 48 * n, "Wrong number of body-level input summary cells")
    bootstrap = read(folder / "bootstrap.json")
    require(bootstrap["trace_ids"] == traces and bootstrap["seed"] == SEED,
            "Bootstrap uses a different task list/order/seed")
    require(bootstrap["trace_equal_weight_after_body_token_mean"] is True
            and bootstrap["same_draws_for_all_conditions_roles_and_contrasts"] is True,
            "Bootstrap is not trace-equal and paired")
    require(bootstrap["role_support"] == dict.fromkeys(ROLES, n), "Incorrect per-role task support")
    with np.load(folder / "bootstrap-draws.npz", allow_pickle=False) as archive:
        draws, saved_traces = archive["draw_indices"], archive["trace_ids"].tolist()
    require(saved_traces == traces and sha(folder / "bootstrap-draws.npz") == bootstrap["draw_file_sha256"],
            "Bootstrap draw archive/order changed")
    expected_draws = np.random.default_rng(SEED).integers(0, n, (RESAMPLES, n)) if n > 1 else np.zeros((1, 1), dtype=int)
    require(np.array_equal(draws, expected_draws), "Saved draws do not resample whole tasks with the frozen seed")
    require(bootstrap["resamples"] == (RESAMPLES if n > 1 else 0), "Wrong bootstrap replicate count")
    if n > 1:
        require(bootstrap["confidence_level"] == .95 and bootstrap["intervals"] == "pointwise, not simultaneous",
                "Wrong advertised interval coverage")
    else:
        require(bootstrap["confidence_level"] is None, "Single-trace pilot should have no confidence interval")

    scalar, original_rows, by_condition = {}, {}, {}
    for row in rows:
        trace, role = row["trace_id"], row["source_role"]
        operation = row["operation"]
        condition = "swap_" + row["destination_role"] if operation == "swap" else operation
        require((trace, role, condition) not in by_condition, "Repeated body-level condition")
        by_condition[trace, role, condition] = row
        for predicted in ROLES:
            scalar[trace, role, condition, predicted] = float(row["p_" + predicted])
            if operation == "baseline":
                scalar[trace, role, "swap_" + role, predicted] = float(row["p_" + predicted])
        if operation == "baseline":
            original_rows[trace, role] = row
    conditions = ("baseline", "strip", "move", *("swap_" + role for role in ROLES))
    require(len(scalar) == n * 6 * 9 * 6, "Scalar probability dictionary omits a role/condition/prediction")
    cells = {}
    for role in ROLES:
        for condition in conditions:
            for predicted in ROLES:
                vector = np.array([scalar[trace, role, condition, predicted] for trace in traces], dtype=np.float64)
                reference = np.array([scalar[trace, role, "baseline", predicted] for trace in traces], dtype=np.float64)
                samples = vector[draws].mean(axis=1)
                paired_samples = (vector - reference)[draws].mean(axis=1)
                bounds = np.quantile(samples, [.025, .975]) if n > 1 else [np.nan, np.nan]
                paired_bounds = np.quantile(paired_samples, [.025, .975]) if n > 1 else [np.nan, np.nan]
                cells[role, condition, predicted] = dict(mean=float(vector.mean()), low=float(bounds[0]), high=float(bounds[1]),
                    delta=100 * float((vector - reference).mean()),
                    delta_low=100 * float(paired_bounds[0]), delta_high=100 * float(paired_bounds[1]))

    max_error, checks = 0.0, 0

    def compare(row, expected):
        nonlocal max_error, checks
        for name, value in expected.items():
            require(name in row, "Missing published scalar: " + name)
            actual = float(row[name]) if row[name] not in ("", None) else np.nan
            if not np.isfinite(value):
                require(np.isnan(actual), "Pilot falsely reports a finite confidence bound: " + name)
            else:
                error = abs(actual - value)
                max_error = max(max_error, error)
                require(np.isfinite(actual) and error <= 1e-9 + 1e-11 * abs(value),
                        f"Independent scalar/bootstrap differs: {cohort}/{name}: {actual} versus {value}")
            checks += 1

    full = table(folder / "full-role-probabilities.csv")
    indexed = {(row["source_role"], row["condition"], row["predicted_role"]): row for row in full}
    require(len(full) == len(indexed) == 324 and set(indexed) == set(cells), "Full probability publication grid differs")
    for key, row in indexed.items():
        value = cells[key]
        require(row["cohort"] == cohort, "Full probability cohort label differs")
        compare(row, dict(n_traces=n, mean_p=value["mean"], ci_lower=value["low"], ci_upper=value["high"],
                          delta_pp=value["delta"], delta_ci_lower_pp=value["delta_low"], delta_ci_upper_pp=value["delta_high"]))

    effects = table(folder / "effects.csv")
    indexed = {(row["source_role"], row["operation"]): row for row in effects}
    require(len(effects) == len(indexed) == 18 and set(indexed) == {(role, operation) for role in ROLES for operation in ("baseline", "strip", "move")},
            "Effect publication grid differs")
    for (role, operation), row in indexed.items():
        value, baseline = cells[role, operation, role], cells[role, "baseline", role]
        body_rows = [by_condition[trace, role, operation] for trace in traces]
        mean = lambda name: float(np.mean([float(body[name]) for body in body_rows]))
        original_fraction = np.mean([float(body["original_body_start"]) / float(original_rows[body["trace_id"], role]["total_input_tokens"]) for body in body_rows])
        new_fraction = np.mean([float(body["new_body_start"]) / float(body["total_input_tokens"]) for body in body_rows])
        require(row["cohort"] == cohort, "Effect cohort label differs")
        compare(row, dict(n_traces=n, n_cohort_traces=n,
            baseline_p=baseline["mean"], baseline_ci_lower=baseline["low"], baseline_ci_upper=baseline["high"],
            mean_p=value["mean"], ci_lower=value["low"], ci_upper=value["high"], delta_pp=value["delta"],
            delta_ci_lower_pp=value["delta_low"], delta_ci_upper_pp=value["delta_high"],
            mean_body_tokens=mean("body_tokens"), mean_original_body_start=mean("original_body_start"),
            mean_new_body_start=mean("new_body_start"), mean_original_start_fraction=original_fraction,
            mean_new_start_fraction=new_fraction, mean_total_input_tokens=mean("total_input_tokens")))

    swaps = table(folder / "swap-effects.csv")
    indexed = {(row["source_role"], row["destination_role"]): row for row in swaps}
    require(len(swaps) == len(indexed) == 36 and set(indexed) == {(source, destination) for source in ROLES for destination in ROLES},
            "Directed replacement publication grid differs")
    for (role, destination), row in indexed.items():
        original = cells[role, "swap_" + destination, role]
        attracted = cells[role, "swap_" + destination, destination]
        require(row["cohort"] == cohort and row["operation"] == ("baseline" if role == destination else "swap"),
                "Replacement diagonal or cohort mislabeled")
        compare(row, dict(n_traces=n, n_cohort_traces=n,
            original_role_p=original["mean"], original_role_ci_lower=original["low"], original_role_ci_upper=original["high"],
            original_role_delta_pp=original["delta"], original_role_delta_ci_lower_pp=original["delta_low"], original_role_delta_ci_upper_pp=original["delta_high"],
            destination_role_p=attracted["mean"], destination_role_ci_lower=attracted["low"], destination_role_ci_upper=attracted["high"],
            destination_role_delta_pp=attracted["delta"], destination_role_delta_ci_lower_pp=attracted["delta_low"], destination_role_delta_ci_upper_pp=attracted["delta_high"],
            baseline_original_role_p=cells[role, "baseline", role]["mean"], baseline_destination_role_p=cells[role, "baseline", destination]["mean"]))

    # The human report consumes this JSON; ensure it carries the audited tables.
    published = read(folder / "report-data.json")
    require(published["trace_ids"] == traces and published["n_traces"] == n, "Report JSON task support differs")
    for name, expected_rows in (("effects", effects), ("swaps", swaps), ("full_role_probabilities", full)):
        require(len(published[name]) == len(expected_rows), "Report JSON publication inventory differs")
        for report_row, csv_row in zip(published[name], expected_rows):
            require(set(report_row) == set(csv_row), "Report JSON fields differ from its audited table")
            for field, value in report_row.items():
                if value is None or isinstance(value, (int, float)):
                    compare(csv_row, {field: np.nan if value is None else value})
                else:
                    require(csv_row[field] == value, "Report JSON categorical field differs: " + field)
    return dict(status="passed", cohort=cohort, traces=n, bootstrap_resamples=RESAMPLES if n > 1 else 0,
                mean_weighting="equal trace means after body-token mean", bootstrap_unit="whole trace",
                paired_pointwise_95_percent_intervals_all_recomputed=(n > 1), maximum_numeric_difference=max_error,
                scalar_checks=checks, table_rows={"effects": 18, "swaps": 36, "full_probabilities": 324},
                files_sha256={name: sha(folder / name) for name in ("effects.csv", "swap-effects.csv", "full-role-probabilities.csv", "bootstrap-draws.npz", "bootstrap.json", "report-data.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "all"), required=True)
    args = parser.parse_args()
    stage = args.output.resolve() / args.stage
    summary = stage / "input-summary.csv"
    source_validation = read(stage / "analysis/validation.json")
    require(source_validation["input_summary_sha256"] == sha(summary), "Analyzer source summary changed")
    rows = table(summary)
    cohorts = ("pilot",) if args.stage == "pilot" else ("fit", "all")
    result = dict(status="passed", stage=args.stage, input_summary_sha256=sha(summary),
                  auditor_sha256=sha(__file__), analyzer_not_imported=True,
                  cohorts=[audit_cohort(stage, rows, cohort) for cohort in cohorts])
    destination = stage / "analysis/independent-statistics-validation.json"
    atomic_json(destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
