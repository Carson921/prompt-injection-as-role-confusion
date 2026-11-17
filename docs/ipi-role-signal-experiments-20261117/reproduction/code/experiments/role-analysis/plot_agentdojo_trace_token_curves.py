#!/usr/bin/env python3
"""Supplemental aligned body-token probability curves; no model inference.

Usage:
    python plot_agentdojo_trace_token_curves.py --output ROOT --stage pilot
    python plot_agentdojo_trace_token_curves.py --output ROOT --stage all

The pilot shows every target token, aligned by original body-token identity.
All-stage curves use 20 equal relative-position bins: mean tokens within each
trace/bin first, then average traces equally. Empty bins remain missing. The
same 2,000 trace-cluster bootstrap draws are shared across all conditions,
roles, bins, and paired contrasts within each cohort (seed 20261112).
Outputs are supplemental and do not modify the primary analysis files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
ROLE_LABELS = ("System", "Developer", "User", "CoT", "Assistant", "Tool")
OPERATIONS = ("baseline", "strip", "move")
LABELS = {"baseline": "Original", "strip": "Tags removed", "move": "Block moved"}
COLORS = {"baseline": "#57616E", "strip": "#2474B5", "move": "#C77726"}
STYLES = {"baseline": "--", "strip": "-", "move": "-"}
N_BINS, N_BOOTSTRAP, SEED = 20, 2000, 20261112


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")


def load_curves(root: Path, stage: str) -> tuple[dict[tuple[str, str, str], dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Check token identity/alignment against immutable input maps and summaries."""
    summary_path = root / stage / "input-summary.csv"
    inputs_path = root / stage / "inputs.jsonl"
    summary = pd.read_csv(summary_path, dtype={"trace_id": str, "input_id": str}, keep_default_na=False)
    summary = summary[summary.operation.isin(OPERATIONS)].copy()
    key_columns = ["trace_id", "source_role", "operation"]
    if summary.duplicated(key_columns).any():
        raise ValueError("Repeated main-condition readout in input summary")
    n_traces = summary.trace_id.nunique()
    splits = summary[["trace_id", "source_split"]].drop_duplicates()
    if splits.trace_id.duplicated().any():
        raise ValueError("A trace belongs to multiple source splits")
    if stage == "pilot" and n_traces != 1:
        raise ValueError("Pilot token curves require exactly one trace")
    if stage == "all" and (n_traces != 57 or (splits.source_split == "fit").sum() != 40):
        raise ValueError("All token curves require 57 traces including 40 fit traces")
    expected = summary.set_index(key_columns)
    expected_keys = set(expected.index)
    curves: dict[tuple[str, str, str], dict[str, Any]] = {}
    readout_hashes = {}
    max_mean_error = 0.0
    with inputs_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            operation = record["operation"]
            if operation not in OPERATIONS:
                continue
            input_id, trace_id = record["input_id"], record["trace_id"]
            path = root / "readouts" / f"{input_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                positions = data["positions"].copy()
                original_positions = data["original_positions"].copy()
                token_ids = data["token_ids"].copy()
                probabilities = data["probabilities"].astype(float)
                classes = data["roles"].tolist()
            if classes != list(ROLES):
                raise ValueError(f"{input_id}: unexpected probe role order {classes}")
            if not np.array_equal(positions, record["target_positions"]):
                raise ValueError(f"{input_id}: readout positions differ from input map")
            if not np.array_equal(original_positions, record["target_original_positions"]):
                raise ValueError(f"{input_id}: original-position readout differs from input map")
            expected_ids = np.asarray(record["input_ids"], dtype=np.int64)[positions]
            if not np.array_equal(token_ids, expected_ids):
                raise ValueError(f"{input_id}: readout token IDs differ from input")
            if probabilities.shape != (len(positions), 6) or not np.isfinite(probabilities).all():
                raise ValueError(f"{input_id}: invalid probability matrix")
            if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=2e-5, rtol=0):
                raise ValueError(f"{input_id}: invalid six-class probabilities")
            readout_hashes[input_id] = sha256(path)
            for role, role_positions in record["role_target_positions"].items():
                if role not in ROLES:
                    raise ValueError(f"{input_id}: unknown target role {role}")
                key = (trace_id, role, operation)
                if key not in expected_keys or key in curves:
                    raise ValueError(f"Unexpected or duplicated readout {key}")
                expected_summary = expected.loc[key]
                if expected_summary.input_id != input_id:
                    raise ValueError(f"{key}: input_id differs from summary")
                indices = np.flatnonzero(np.isin(positions, role_positions))
                # Body order must be identical before and after an intervention.
                if not np.array_equal(positions[indices], role_positions):
                    raise ValueError(f"{key}: incomplete or reordered target-body positions")
                expected_original = record["original_role_target_positions"][role]
                if not np.array_equal(original_positions[indices], expected_original):
                    raise ValueError(f"{key}: body-token alignment differs from original map")
                if len(indices) != expected_summary.body_tokens or len(indices) == 0:
                    raise ValueError(f"{key}: incorrect or empty body-token denominator")
                if len(indices) > 1 and (np.diff(original_positions[indices]) != 1).any():
                    raise ValueError(f"{key}: target body is not contiguous in the original trace")
                probability = probabilities[indices, ROLES.index(role)]
                error = abs(float(probability.mean()) - float(expected_summary[f"p_{role}"]))
                max_mean_error = max(max_mean_error, error)
                if error > 3e-6:
                    raise ValueError(f"{key}: token curve does not recover the primary probability mean ({error})")
                curves[key] = {
                    "input_id": input_id, "trace_id": trace_id, "source_split": expected_summary.source_split,
                    "source_role": role, "operation": operation, "body_tokens": len(indices),
                    "positions": positions[indices], "original_positions": original_positions[indices],
                    "token_ids": token_ids[indices], "probability": probability,
                }
    if set(curves) != expected_keys:
        raise ValueError(f"Missing expected readouts: {sorted(expected_keys - set(curves))[:10]}")
    for trace_id, role in sorted({key[:2] for key in curves}):
        baseline = curves[(trace_id, role, "baseline")]
        for operation in ("strip", "move"):
            current = curves[(trace_id, role, operation)]
            if not np.array_equal(current["original_positions"], baseline["original_positions"]) or not np.array_equal(current["token_ids"], baseline["token_ids"]):
                raise ValueError(f"{trace_id}/{role}: {operation} does not preserve body-token identity")
    validation = {
        "stage": stage, "n_traces": n_traces, "n_role_condition_curves": len(curves),
        "inputs_sha256": sha256(inputs_path), "input_summary_sha256": sha256(summary_path),
        "readout_sha256": readout_hashes,
        "token_ids_and_original_positions_identical_across_conditions": True,
        "baseline_union_mask_split_using_role_target_positions": True,
        "all_curve_means_match_primary_summary": True, "max_primary_mean_absolute_error": max_mean_error,
        "supplemental_only": True, "new_model_inference": False,
    }
    return curves, splits, validation


def make_figure(title: str, subtitle: str, footer: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), sharey=True)
    fig.subplots_adjust(left=.065, right=.98, top=.79, bottom=.16, wspace=.20, hspace=.53)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=.97)
    fig.text(.5, .915, subtitle, ha="center", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=COLORS[op], linestyle=STYLES[op], lw=2, label=LABELS[op]) for op in OPERATIONS],
               loc="upper center", bbox_to_anchor=(.5, .881), ncol=3, frameon=False, fontsize=11)
    fig.text(.5, .055, footer, ha="center", va="center", fontsize=10, linespacing=1.6, color="#4C5968")
    for ax in axes.flat:
        ax.set_ylim(-.025, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color="#E3E7EC", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Probe P(original role)")
    return fig, axes


def save_figure(fig: plt.Figure, destination: Path) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(destination / f"role-token-curves.{extension}", dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def plot_pilot(curves: dict[tuple[str, str, str], dict[str, Any]], destination: Path, validation: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    trace_id = next(iter(curves))[0]
    footer = (
        "The same body tokens are aligned by their original token order; wrappers are excluded. No smoothing or confidence intervals.\n"
        "Only the first block of each role is changed. Moving a block changes its causal context as well as its position."
    )
    fig, axes = make_figure("Where within a body does the original-role probability change?",
                            f"Single native benign trace: {trace_id}  |  Every body token shown", footer)
    rows = []
    for role, role_label, ax in zip(ROLES, ROLE_LABELS, axes.flat):
        key = (trace_id, role, "baseline")
        if key not in curves:
            ax.set_title(f"{role_label} content · absent from this trace")
            ax.text(.5, .5, "No observations", ha="center", transform=ax.transAxes)
            ax.set_xlabel("Original body token index")
            continue
        n = curves[key]["body_tokens"]
        means = []
        for operation in OPERATIONS:
            current = curves[(trace_id, role, operation)]
            x = np.arange(1, n + 1)
            ax.plot(x, current["probability"], color=COLORS[operation], linestyle=STYLES[operation],
                    lw=1.6 if operation == "baseline" else 1.3, alpha=.95, zorder=4 if operation == "baseline" else 3)
            means.append(float(current["probability"].mean()))
            for i in range(n):
                rows.append({"trace_id": trace_id, "source_role": role, "operation": operation,
                             "body_token_index": i + 1, "original_position": int(current["original_positions"][i]),
                             "actual_position": int(current["positions"][i]), "token_id": int(current["token_ids"][i]),
                             "p_original_role": float(current["probability"][i])})
        ax.set_xlim(.5, max(1.5, n + .5))
        ax.set_xlabel("Original body token index")
        ax.set_title(f"{role_label} · {n:,} body tokens", fontsize=12, pad=29)
        ax.text(.5, 1.045, f"Means: original {means[0]:.1%}  |  removed {means[1]:.1%}  |  moved {means[2]:.1%}",
                ha="center", transform=ax.transAxes, fontsize=9.3, color="#3B4857")
    pd.DataFrame(rows).to_csv(destination / "token-curves.csv", index=False)
    write_json(destination / "token-curve-validation.json", validation)
    save_figure(fig, destination)


def relative_bin_means(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin original token centers; empty bins remain NA, with no interpolation."""
    n = len(probability)
    index = np.minimum(((np.arange(n) + .5) / n * N_BINS).astype(int), N_BINS - 1)
    count = np.bincount(index, minlength=N_BINS)
    total = np.bincount(index, weights=probability, minlength=N_BINS)
    means = np.divide(total, count, out=np.full(N_BINS, np.nan), where=count > 0)
    return means, count


def mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(values)
    numerator = np.where(finite, values, 0).sum(axis=axis)
    denominator = finite.sum(axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def plot_all(curves: dict[tuple[str, str, str], dict[str, Any]], trace_ids: list[str], destination: Path, cohort: str, validation: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cells = np.full((len(trace_ids), 6, 3, N_BINS), np.nan)
    token_counts = np.zeros((len(trace_ids), 6, N_BINS), dtype=int)
    per_trace_rows = []
    for ti, trace_id in enumerate(trace_ids):
        for ri, role in enumerate(ROLES):
            if (trace_id, role, "baseline") not in curves:
                continue
            for oi, operation in enumerate(OPERATIONS):
                current = curves[(trace_id, role, operation)]
                binned, count = relative_bin_means(current["probability"])
                cells[ti, ri, oi] = binned
                if oi == 0:
                    token_counts[ti, ri] = count
                elif not np.array_equal(token_counts[ti, ri], count):
                    raise ValueError(f"{trace_id}/{role}: bin denominator changed after {operation}")
                for bi in range(N_BINS):
                    per_trace_rows.append({"trace_id": trace_id, "source_role": role, "operation": operation,
                                           "body_tokens": current["body_tokens"], "bin_index": bi + 1,
                                           "bin_center_fraction": (bi + .5) / N_BINS, "bin_tokens": int(count[bi]),
                                           "p_original_role": binned[bi]})
    draws = np.random.default_rng(SEED).integers(0, len(trace_ids), size=(N_BOOTSTRAP, len(trace_ids)))
    counts = np.stack([np.bincount(draw, minlength=len(trace_ids)) for draw in draws])
    flat = cells.reshape(len(trace_ids), -1)
    finite = np.isfinite(flat)
    numerator, denominator = counts @ np.where(finite, flat, 0), counts @ finite.astype(float)
    boot = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0).reshape(N_BOOTSTRAP, 6, 3, N_BINS)
    point = mean(cells)
    delta_point = point - point[:, :1]
    delta_boot = boot - boot[:, :, :1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanquantile(boot, [.025, .975], axis=0)
        delta_low, delta_high = np.nanquantile(delta_boot, [.025, .975], axis=0)
    bin_support = np.isfinite(cells[:, :, 0]).sum(axis=0)
    role_support = np.any(token_counts > 0, axis=2).sum(axis=0)
    rows = []
    for ri, role in enumerate(ROLES):
        for oi, operation in enumerate(OPERATIONS):
            for bi in range(N_BINS):
                rows.append({"cohort": cohort, "source_role": role, "operation": operation, "bin_index": bi + 1,
                             "bin_center_fraction": (bi + .5) / N_BINS, "n_traces": int(bin_support[ri, bi]),
                             "n_role_traces": int(role_support[ri]), "n_cohort_traces": len(trace_ids),
                             "mean_p": point[ri, oi, bi], "ci_lower": low[ri, oi, bi], "ci_upper": high[ri, oi, bi],
                             "delta_pp": 100 * delta_point[ri, oi, bi],
                             "delta_ci_lower_pp": 100 * delta_low[ri, oi, bi], "delta_ci_upper_pp": 100 * delta_high[ri, oi, bi],
                             "bootstrap_resamples_without_support": int(np.isnan(boot[:, ri, oi, bi]).sum())})
    pd.DataFrame(rows).to_csv(destination / "token-curves.csv", index=False)
    pd.DataFrame(per_trace_rows).to_csv(destination / "token-curves-per-trace-bins.csv", index=False)
    np.savez_compressed(destination / "token-curve-bootstrap.npz", draw_indices=draws, trace_ids=np.asarray(trace_ids))
    write_json(destination / "token-curve-bootstrap.json", {
        "cohort": cohort, "trace_ids": trace_ids, "resamples": N_BOOTSTRAP, "seed": SEED,
        "method": "paired trace-cluster percentile bootstrap", "confidence_level": .95,
        "pointwise_not_simultaneous": True, "shared_draws_across_roles_conditions_bins_contrasts": True,
        "binning": "20 equal relative-position bins; assign each token by its normalized body-token center; no interpolation",
        "weighting": "Within each trace/bin average tokens, then average supported traces equally",
        "empty_bins": "Missing, never zero; n_traces and unsupported-resample counts saved for every bin",
        "not_primary_body_mean": "A mean over plotted bins is not the primary full-body-token mean",
        "draw_file_sha256": sha256(destination / "token-curve-bootstrap.npz"),
    })
    write_json(destination / "token-curve-validation.json", validation)
    label = "Fit traces · primary cohort" if cohort == "fit" else "All traces · secondary cohort"
    footer = (
        "Shading: pointwise 95% paired trace-bootstrap intervals; 2,000 resamples. Empty bins remain missing.\n"
        "Tokens are averaged within each trace/bin before traces are weighted equally. Curves describe relative body position, not absolute trace position."
    )
    fig, axes = make_figure("Original-role probabilities across the target body",
                            f"{label} · n = {len(trace_ids)}  |  20 equal relative-position bins per body", footer)
    x = (np.arange(N_BINS) + .5) / N_BINS
    for ri, (role, role_label, ax) in enumerate(zip(ROLES, ROLE_LABELS, axes.flat)):
        for oi, operation in enumerate(OPERATIONS):
            ax.fill_between(x, low[ri, oi], high[ri, oi], color=COLORS[operation], alpha=.14, linewidth=0)
            ax.plot(x, point[ri, oi], color=COLORS[operation], linestyle=STYLES[operation], lw=1.9,
                    marker="o", markersize=2.5, zorder=4 if operation == "baseline" else 3)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Relative position within the original body")
        ax.set_title(f"{role_label} · {role_support[ri]} traces with this role", fontsize=12, pad=29)
        ax.text(.5, 1.045, f"Trace support per bin: {bin_support[ri].min()}–{bin_support[ri].max()}",
                ha="center", transform=ax.transAxes, fontsize=10, color="#3B4857")
        if role_support[ri] == 0:
            ax.text(.5, .5, "No observations", ha="center", transform=ax.transAxes)
    save_figure(fig, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "all"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    curves, splits, validation = load_curves(args.output, args.stage)
    analysis_root = args.output / args.stage / "analysis"
    if args.stage == "pilot":
        plot_pilot(curves, analysis_root / "pilot", validation)
    else:
        for cohort in ("fit", "all"):
            traces = sorted(splits.loc[splits.source_split == "fit", "trace_id"] if cohort == "fit" else splits.trace_id)
            plot_all(curves, traces, analysis_root / cohort, cohort, validation)
    print(json.dumps({"stage": args.stage, "n_traces": validation["n_traces"], "curves": validation["n_role_condition_curves"],
                      "max_primary_mean_absolute_error": validation["max_primary_mean_absolute_error"],
                      "output": str(analysis_root.resolve())}))


if __name__ == "__main__":
    main()
