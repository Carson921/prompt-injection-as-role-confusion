#!/usr/bin/env python3
"""Analyze first-occurrence role interventions in native benign AgentDojo traces.

Read ``OUTPUT/STAGE/input-summary.csv`` without loading a model. Each row is the
mean six-class probe probability over the *unchanged target body tokens*. A trace
has one baseline input (one readout per role present), then strip, five alternative
wrappers, and move interventions per present role. Roles absent from a trace are
missing observations, never zero probabilities. Trace means receive equal weight.

Usage::

    python analyze_agentdojo_trace_interventions.py --output OUTPUT --stage pilot
    python analyze_agentdojo_trace_interventions.py --output OUTPUT --stage all

Pilot requires one trace and gives descriptive results without confidence
intervals. The all stage requires 57 traces, of which 40 are fit traces, and emits
fit-only primary results and all-trace secondary results. All confidence intervals
are pointwise 95% paired trace-cluster percentile bootstrap intervals (2,000
resamples, seed 20261112). The same trace resampling is used for every condition,
role, and contrast within a cohort. No significance tests or causal contribution
percentages are computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
ROLE_LABELS = ("System", "Developer", "User", "CoT", "Assistant", "Tool")
P_COLUMNS = tuple(f"p_{role}" for role in ROLES)
OPERATIONS = ("baseline", "strip", "swap", "move")
CONDITIONS = ("baseline", "strip", "move", *(f"swap_{role}" for role in ROLES))
SEED = 20261112
N_BOOTSTRAP = 2000
COLORS = {"baseline": "#5A626D", "strip": "#2474B5", "move": "#C77726"}
MARKERS = {"baseline": "o", "strip": "s", "move": "^"}
OP_LABELS = {"baseline": "Original", "strip": "Tags removed", "move": "Block moved"}
LIMITATION = (
    "First occurrence of each role only. Wrapper edits may shift token positions; "
    "moving a block changes its causal context.\n"
    "These are intervention effects on probe probabilities, not a causal percentage decomposition."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")


def read_and_validate(path: Path, stage: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the complete per-trace design and frozen body denominator."""
    frame = pd.read_csv(path, dtype={"input_id": str, "trace_id": str}, keep_default_na=False)
    required = {
        "input_id", "trace_id", "source_split", "operation", "source_role", "destination_role",
        "body_tokens", "original_body_start", "new_body_start", "total_input_tokens", *P_COLUMNS,
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing input-summary columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input summary is empty")
    for column in ("input_id", "trace_id", "source_split", "operation", "source_role"):
        if (frame[column].astype(str).str.len() == 0).any():
            raise ValueError(f"Empty {column}")
    if not set(frame.source_role).issubset(ROLES):
        raise ValueError("source_role must use the six canonical role names")
    if not set(frame.operation).issubset(OPERATIONS):
        raise ValueError(f"operation must be one of {OPERATIONS}")
    if not set(frame.source_split).issubset({"fit", "dev", "test"}):
        raise ValueError("source_split must be fit, dev, or test")
    numeric = ("body_tokens", "original_body_start", "new_body_start", "total_input_tokens", *P_COLUMNS)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"Nonfinite values in {column}")
    for column in numeric[:4]:
        if (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain integer token counts/indices")
    if (frame.body_tokens <= 0).any() or (frame.total_input_tokens <= 0).any():
        raise ValueError("body_tokens and total_input_tokens must be positive")
    if (frame[["original_body_start", "new_body_start"]] < 0).any().any():
        raise ValueError("Body start indices must be nonnegative")
    if (frame.new_body_start + frame.body_tokens > frame.total_input_tokens).any():
        raise ValueError("Target body extends beyond its input")
    probabilities = frame[list(P_COLUMNS)].to_numpy(dtype=float)
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
    if sum_error > 2e-5:
        raise ValueError(f"Six-class probabilities do not sum to one: max error {sum_error}")

    trace_splits = frame.groupby("trace_id").source_split.nunique()
    if (trace_splits != 1).any():
        raise ValueError("A trace appears in multiple source splits")
    trace_ids = sorted(frame.trace_id.unique())
    split_counts = frame[["trace_id", "source_split"]].drop_duplicates().source_split.value_counts().to_dict()
    if stage == "pilot" and len(trace_ids) != 1:
        raise ValueError(f"Pilot must have exactly one trace; found {len(trace_ids)}")
    if stage == "all" and (len(trace_ids) != 57 or split_counts.get("fit", 0) != 40):
        raise ValueError(f"All stage requires 57 traces including 40 fit traces; found {split_counts}")

    coverage = []
    for trace_id, group in frame.groupby("trace_id", sort=True):
        baseline = group[group.operation == "baseline"]
        if baseline.empty or baseline.source_role.duplicated().any():
            raise ValueError(f"{trace_id}: baseline must have one row per present role")
        present = set(baseline.source_role)
        if set(group.source_role) != present:
            raise ValueError(f"{trace_id}: intervention role missing from baseline")
        if baseline.input_id.nunique() != 1:
            raise ValueError(f"{trace_id}: baseline rows must share one original input_id")
        if baseline.total_input_tokens.nunique() != 1:
            raise ValueError(f"{trace_id}: inconsistent original trace lengths")
        for role in ROLES:
            if role not in present:
                continue
            rows = group[group.source_role == role]
            counts = rows.operation.value_counts().to_dict()
            if counts != {"swap": 5, "baseline": 1, "strip": 1, "move": 1}:
                raise ValueError(f"{trace_id}/{role}: incomplete conditions: {counts}")
            swaps = rows[rows.operation == "swap"]
            if set(swaps.destination_role) != set(ROLES) - {role} or swaps.destination_role.duplicated().any():
                raise ValueError(f"{trace_id}/{role}: require all five other wrappers exactly once")
            for operation in ("baseline", "strip", "move"):
                destination = rows.loc[rows.operation == operation, "destination_role"].iloc[0]
                allowed = {"", role} | ({"none", "unwrapped"} if operation == "strip" else set())
                if destination not in allowed:
                    raise ValueError(f"{trace_id}/{role}: invalid destination for {operation}: {destination}")
            if rows.body_tokens.nunique() != 1:
                raise ValueError(f"{trace_id}/{role}: target-body denominator changed across conditions")
            if rows.original_body_start.nunique() != 1:
                raise ValueError(f"{trace_id}/{role}: original body start changed across conditions")
            original = rows[rows.operation == "baseline"].iloc[0]
            if original.original_body_start != original.new_body_start:
                raise ValueError(f"{trace_id}/{role}: baseline body start mismatch")
            if original.original_body_start + original.body_tokens > original.total_input_tokens:
                raise ValueError(f"{trace_id}/{role}: original body beyond baseline input")
            moved = rows[rows.operation == "move"].iloc[0]
            if moved.original_body_start == moved.new_body_start:
                raise ValueError(f"{trace_id}/{role}: move did not change body position")
        if len(group) != 8 * len(present):
            raise ValueError(f"{trace_id}: unexpected readout rows")
        if group.input_id.nunique() != 1 + 7 * len(present):
            raise ValueError(f"{trace_id}: unexpected unique inputs; interventions must be separate inputs")
        coverage.append({"trace_id": trace_id, "source_split": group.source_split.iloc[0],
                         "roles_present": [role for role in ROLES if role in present],
                         "summary_rows": len(group), "unique_inputs": group.input_id.nunique()})
    input_trace_counts = frame.groupby("input_id").trace_id.nunique()
    if (input_trace_counts != 1).any():
        raise ValueError("An input_id is reused across traces")
    for input_id, rows in frame.groupby("input_id", sort=False):
        if len(rows) > 1 and (rows.operation != "baseline").any():
            raise ValueError(f"{input_id}: only baseline inputs may have multiple target-body readouts")
    validation = {
        "stage": stage, "input_summary": str(path.resolve()), "input_summary_sha256": sha256(path),
        "n_traces": len(trace_ids), "split_trace_counts": split_counts,
        "summary_rows": len(frame), "unique_inputs": int(frame.input_id.nunique()),
        "role_order": list(ROLES), "full_condition_coverage": True,
        "body_token_denominator_invariant_within_target": True,
        "body_token_identity_validation": "The runner/manifest must validate identity; counts alone cannot prove identity.",
        "max_probability_sum_error": sum_error, "coverage": coverage,
    }
    return frame, validation


def nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    numerator = np.where(np.isfinite(values), values, 0).sum(axis=axis)
    denominator = np.isfinite(values).sum(axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)


def bounds(values: np.ndarray, bootstrap: bool) -> tuple[np.ndarray, np.ndarray]:
    if not bootstrap:
        missing = np.full(values.shape[1:], np.nan)
        return missing, missing.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanquantile(values, [0.025, 0.975], axis=0)
    return low, high


def cells_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    trace_ids = sorted(frame.trace_id.unique())
    trace_index = {trace_id: i for i, trace_id in enumerate(trace_ids)}
    cells = np.full((len(trace_ids), 6, len(CONDITIONS), 6), np.nan)
    for row in frame.itertuples(index=False):
        ti, ri = trace_index[row.trace_id], ROLES.index(row.source_role)
        condition = f"swap_{row.destination_role}" if row.operation == "swap" else row.operation
        ci = CONDITIONS.index(condition)
        cells[ti, ri, ci, :] = [getattr(row, column) for column in P_COLUMNS]
        if row.operation == "baseline":
            cells[ti, ri, CONDITIONS.index(f"swap_{row.source_role}"), :] = cells[ti, ri, ci, :]
    return cells, trace_ids


def summarize(frame: pd.DataFrame, destination: Path, cohort: str, validation: dict[str, Any]) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    cells, trace_ids = cells_from_frame(frame)
    n_traces = len(trace_ids)
    do_bootstrap = n_traces > 1
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_traces, size=(N_BOOTSTRAP, n_traces)) if do_bootstrap else np.zeros((1, 1), dtype=int)
    counts = np.stack([np.bincount(draw, minlength=n_traces) for draw in draws])
    values = cells.reshape(n_traces, -1)
    finite = np.isfinite(values)
    numerator = counts @ np.where(finite, values, 0)
    denominator = counts @ finite.astype(float)
    boot = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    boot = boot.reshape(len(draws), *cells.shape[1:])
    point = nanmean(cells)
    low, high = bounds(boot, do_bootstrap)
    # Subtract within each paired resample, preserving the source-role support.
    delta_point = point - point[:, :1, :]
    delta_boot = boot - boot[:, :, :1, :]
    delta_low, delta_high = bounds(delta_boot, do_bootstrap)
    support = np.isfinite(cells[:, :, 0, 0]).sum(axis=0)
    effects, swaps, full = [], [], []
    for ri, role in enumerate(ROLES):
        for ci, condition in enumerate(CONDITIONS):
            for pi, predicted in enumerate(ROLES):
                full.append({"cohort": cohort, "source_role": role, "condition": condition,
                             "predicted_role": predicted, "n_traces": int(support[ri]),
                             "mean_p": point[ri, ci, pi], "ci_lower": low[ri, ci, pi], "ci_upper": high[ri, ci, pi],
                             "delta_pp": 100 * delta_point[ri, ci, pi],
                             "delta_ci_lower_pp": 100 * delta_low[ri, ci, pi],
                             "delta_ci_upper_pp": 100 * delta_high[ri, ci, pi]})
        for operation in ("baseline", "strip", "move"):
            ci = CONDITIONS.index(operation)
            rows = frame[(frame.source_role == role) & (frame.operation == operation)]
            original_rows = frame[(frame.source_role == role) & (frame.operation == "baseline")]
            original_totals = original_rows.set_index("trace_id").total_input_tokens
            original_fraction = rows.apply(lambda row: row.original_body_start / original_totals.loc[row.trace_id], axis=1) if len(rows) else pd.Series(dtype=float)
            new_fraction = rows.new_body_start / rows.total_input_tokens
            effects.append({
                "cohort": cohort, "operation": operation, "source_role": role,
                "n_traces": int(support[ri]), "n_cohort_traces": n_traces,
                "baseline_p": point[ri, 0, ri], "baseline_ci_lower": low[ri, 0, ri], "baseline_ci_upper": high[ri, 0, ri],
                "mean_p": point[ri, ci, ri], "ci_lower": low[ri, ci, ri], "ci_upper": high[ri, ci, ri],
                "delta_pp": 100 * delta_point[ri, ci, ri],
                "delta_ci_lower_pp": 100 * delta_low[ri, ci, ri], "delta_ci_upper_pp": 100 * delta_high[ri, ci, ri],
                "mean_body_tokens": rows.body_tokens.mean(),
                "mean_original_body_start": rows.original_body_start.mean(), "mean_new_body_start": rows.new_body_start.mean(),
                "mean_original_start_fraction": original_fraction.mean(), "mean_new_start_fraction": new_fraction.mean(),
                "mean_total_input_tokens": rows.total_input_tokens.mean(),
            })
        for wi, wrapper in enumerate(ROLES):
            ci = CONDITIONS.index(f"swap_{wrapper}")
            swaps.append({
                "cohort": cohort, "source_role": role, "destination_role": wrapper,
                "operation": "baseline" if role == wrapper else "swap", "n_traces": int(support[ri]),
                "n_cohort_traces": n_traces,
                "original_role_p": point[ri, ci, ri], "original_role_ci_lower": low[ri, ci, ri], "original_role_ci_upper": high[ri, ci, ri],
                "original_role_delta_pp": 100 * delta_point[ri, ci, ri],
                "original_role_delta_ci_lower_pp": 100 * delta_low[ri, ci, ri],
                "original_role_delta_ci_upper_pp": 100 * delta_high[ri, ci, ri],
                "destination_role_p": point[ri, ci, wi], "destination_role_ci_lower": low[ri, ci, wi], "destination_role_ci_upper": high[ri, ci, wi],
                "destination_role_delta_pp": 100 * delta_point[ri, ci, wi],
                "destination_role_delta_ci_lower_pp": 100 * delta_low[ri, ci, wi],
                "destination_role_delta_ci_upper_pp": 100 * delta_high[ri, ci, wi],
                "baseline_original_role_p": point[ri, 0, ri], "baseline_destination_role_p": point[ri, 0, wi],
            })
    effects_frame, swap_frame, full_frame = pd.DataFrame(effects), pd.DataFrame(swaps), pd.DataFrame(full)
    for name, table in (("effects.csv", effects_frame), ("swap-effects.csv", swap_frame), ("full-role-probabilities.csv", full_frame)):
        table.to_csv(destination / name, index=False)
    np.savez_compressed(destination / "bootstrap-draws.npz", draw_indices=draws, trace_ids=np.array(trace_ids))
    bootstrap = {
        "method": "paired trace-cluster percentile bootstrap" if do_bootstrap else "single-trace descriptive analysis; no confidence intervals",
        "seed": SEED, "resamples": N_BOOTSTRAP if do_bootstrap else 0,
        "confidence_level": 0.95 if do_bootstrap else None, "intervals": "pointwise, not simultaneous" if do_bootstrap else None,
        "trace_ids": trace_ids, "same_draws_for_all_conditions_roles_and_contrasts": True,
        "trace_equal_weight_after_body_token_mean": True,
        "missing_roles": "Excluded from the role-specific denominator; never assigned zero probability.",
        "role_support": dict(zip(ROLES, support.tolist())),
        "resamples_without_role_support": {role: int(np.isnan(boot[:, ri, 0, 0]).sum()) for ri, role in enumerate(ROLES)},
        "draw_file": "bootstrap-draws.npz", "draw_file_sha256": sha256(destination / "bootstrap-draws.npz"),
    }
    write_json(destination / "bootstrap.json", bootstrap)
    report = {
        "cohort": cohort, "n_traces": n_traces, "trace_ids": trace_ids, "role_order": list(ROLES),
        "outcome": "Mean token-level P(original role) over frozen target-body tokens, then equal mean across traces",
        "source_validation": validation, "bootstrap": bootstrap,
        "effects": effects, "swaps": swaps, "full_role_probabilities": full,
        "limitations": [
            "Only the first occurrence of each role in a trace is intervened on.",
            "Removing or replacing a full role wrapper may also shift the body's absolute token position.",
            "Moving an intact body and wrapper changes causal context and message order, not only position encoding.",
            "Probe probability changes describe this checkpoint and trace set; they are not causal contribution percentages.",
            "Fit traces were used to train this probe; all-trace secondary means also include dev/test traces.",
            "Averaged probabilities do not establish whether every token or trace changed in the same way.",
        ],
    }
    write_json(destination / "report-data.json", report)
    plot_main(effects_frame, swap_frame, destination, cohort, n_traces, do_bootstrap)
    plot_destination(swap_frame, destination, cohort, n_traces, do_bootstrap)
    plot_full(full_frame, destination, cohort, n_traces, do_bootstrap)
    return report


def cohort_label(cohort: str, n: int) -> str:
    return {"pilot": "Single trace · descriptive pilot", "fit": f"Fit traces · primary analysis · n = {n}",
            "all": f"All traces · secondary analysis · n = {n}"}[cohort]


def uncertainty_label(bootstrap: bool) -> str:
    return "Pointwise 95% paired trace-bootstrap intervals; 2,000 resamples" if bootstrap else "Single trace; no confidence intervals"


def save_figure(fig: plt.Figure, destination: Path, name: str) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(destination / f"{name}.{extension}", dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, supports: list[int], delta: np.ndarray | None = None) -> Any:
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ECEEF1")
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="equal")
    for ri in range(6):
        for wi in range(6):
            value = matrix[ri, wi]
            label = "NA" if not np.isfinite(value) else f"{100 * value:.1f}%"
            if delta is not None and np.isfinite(value):
                label += "\noriginal" if ri == wi else f"\n{delta[ri, wi]:+.1f} pp"
            ax.text(wi, ri, label, ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(value) and value > 0.56 else "#172536")
            if ri == wi:
                ax.add_patch(Rectangle((wi - .49, ri - .49), .98, .98, fill=False, lw=1.5, edgecolor="#D88B25"))
    ax.set(xticks=np.arange(6), yticks=np.arange(6), title=title,
           xticklabels=ROLE_LABELS, yticklabels=[f"{label} (n={n})" for label, n in zip(ROLE_LABELS, supports)])
    ax.tick_params(axis="x", labelrotation=40, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Wrapper role after intervention")
    ax.set_ylabel("Original content role")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def swap_matrix(table: pd.DataFrame, column: str) -> np.ndarray:
    return table.pivot(index="source_role", columns="destination_role", values=column).reindex(index=ROLES, columns=ROLES).to_numpy()


def plot_main(effects: pd.DataFrame, swaps: pd.DataFrame, destination: Path, cohort: str, n: int, bootstrap: bool) -> None:
    fig = plt.figure(figsize=(21, 8.0))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.43, 1.05), left=.055, right=.98, top=.80, bottom=.24, wspace=.43)
    ax1, ax2, ax3 = [fig.add_subplot(grid[i]) for i in range(3)]
    support = effects[effects.operation == "baseline"].set_index("source_role").reindex(ROLES).n_traces.astype(int).tolist()
    y = np.arange(6)
    for operation, offset in (("strip", -.15), ("move", .15)):
        table = effects[effects.operation == operation].set_index("source_role").reindex(ROLES)
        ax1.scatter(table.delta_pp, y + offset, color=COLORS[operation], marker=MARKERS[operation], s=45, zorder=5)
        if bootstrap:
            ax1.hlines(y + offset, table.delta_ci_lower_pp, table.delta_ci_upper_pp, color=COLORS[operation], lw=1.5, zorder=4)
    limits = effects[effects.operation != "baseline"][["delta_pp", "delta_ci_lower_pp", "delta_ci_upper_pp"]].to_numpy()
    finite_limits = limits[np.isfinite(limits)]
    min_delta = min(-5, float(finite_limits.min()) - 5) if len(finite_limits) else -100
    max_delta = max(5, float(finite_limits.max()) + 5) if len(finite_limits) else 100
    ax1.axvline(0, color="#606872", lw=1, linestyle="--", zorder=1)
    ax1.set(yticks=y, yticklabels=[f"{role} (n={count})" for role, count in zip(ROLE_LABELS, support)],
            ylim=(5.55, -.55), xlim=(min_delta, max_delta), xlabel="Change in P(original role), percentage points",
            title="A · Paired changes from the original trace")
    ax1.grid(axis="x", color="#E1E5EB", lw=.7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(handles=[Line2D([], [], marker=MARKERS[op], color=COLORS[op], linestyle="none", label=OP_LABELS[op])
                        for op in ("strip", "move")], loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False, ncol=1, fontsize=9)
    matrix = swap_matrix(swaps, "original_role_p")
    heatmap(ax2, matrix, "B · Replace the role wrapper\nP(original role); diagonal = original", support)
    moved = effects[effects.operation == "move"].set_index("source_role").reindex(ROLES)
    for ri, row in enumerate(moved.itertuples()):
        if not np.isfinite(row.mean_p):
            ax3.text(.5, ri, "No observations", ha="center", va="center", fontsize=9, color="#777777")
            continue
        start, end = row.mean_original_start_fraction, row.mean_new_start_fraction
        color = "#A95042" if row.delta_pp < 0 else "#2474B5"
        ax3.annotate("", xy=(end, ri), xytext=(start, ri), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.4, "mutation_scale": 15})
        ax3.scatter(start, ri, s=40, facecolors="white", edgecolors="#2F3B48", zorder=5)
        ax3.scatter(end, ri, s=48, marker=">" if end > start else "<", color=color, edgecolors="#2F3B48", linewidths=.5, zorder=6)
        ax3.text(.5, ri - .22, f"ΔP = {row.delta_pp:+.1f} pp", ha="center", va="center", fontsize=9, color="#253447")
    ax3.set(yticks=y, yticklabels=ROLE_LABELS, ylim=(5.55, -.55), xlim=(-.04, 1.04),
            xticks=[0, .25, .5, .75, 1], xlabel="Mean body start / full trace length", title="C · Move to the farthest trace boundary")
    ax3.xaxis.set_major_formatter(PercentFormatter(1))
    ax3.grid(axis="x", color="#E1E5EB", lw=.7)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(handles=[Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#2F3B48", linestyle="none", label="Original position"),
                        Line2D([], [], marker=">", color="#5A6673", linestyle="none", label="Moved position")],
               loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False, fontsize=9)
    fig.suptitle("Role signals in native benign AgentDojo traces", fontsize=20, fontweight="bold", y=.97)
    fig.text(.5, .905, f"{cohort_label(cohort, n)}  |  Body-token means; equal weight per trace", ha="center", fontsize=12)
    fig.text(.5, .865, uncertainty_label(bootstrap), ha="center", fontsize=10, color="#4B5663")
    fig.text(.5, .065, LIMITATION, ha="center", va="center", fontsize=10, linespacing=1.6, color="#4B5663")
    save_figure(fig, destination, "role-interventions-main")


def plot_destination(swaps: pd.DataFrame, destination: Path, cohort: str, n: int, bootstrap: bool) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 9.2))
    fig.subplots_adjust(left=.18, right=.86, top=.77, bottom=.19)
    support = swaps[swaps.source_role == swaps.destination_role].set_index("source_role").reindex(ROLES).n_traces.astype(int).tolist()
    im = heatmap(ax, swap_matrix(swaps, "destination_role_p"), "Probability of the new wrapper's role", support,
                 swap_matrix(swaps, "destination_role_delta_pp"))
    colorbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("P(wrapper role)")
    fig.suptitle("How strongly does a replacement wrapper attract the probe?", fontsize=17, fontweight="bold", y=.96)
    fig.text(.5, .905, cohort_label(cohort, n), ha="center", fontsize=12)
    fig.text(.5, .861, "Each cell: P(new wrapper role); change vs. that same probability in the original trace", ha="center", fontsize=10)
    fig.text(.5, .825, "Diagonal: original wrapper. Orange outline marks the diagonal. Missing support is NA.", ha="center", fontsize=10, color="#4B5663")
    fig.text(.5, .065, "pp = percentage points. " + uncertainty_label(bootstrap) + ".\nWrapper changes may also shift body positions; these are paired intervention effects.",
             ha="center", fontsize=9.5, linespacing=1.6, color="#4B5663")
    save_figure(fig, destination, "wrapper-destination-attraction")


def plot_full(full: pd.DataFrame, destination: Path, cohort: str, n: int, bootstrap: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 13))
    fig.subplots_adjust(left=.095, right=.91, top=.83, bottom=.22, wspace=.48, hspace=.49)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ECEEF1")
    im = None
    for ri, ax in enumerate(axes.flat):
        role = ROLES[ri]
        conditions = ["baseline", "strip", "move", *(f"swap_{wrapper}" for wrapper in ROLES if wrapper != role)]
        table = full[full.source_role == role]
        matrix = table.pivot(index="condition", columns="predicted_role", values="mean_p").reindex(index=conditions, columns=ROLES).to_numpy()
        im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for row in range(8):
            for col in range(6):
                value = matrix[row, col]
                ax.text(col, row, "NA" if not np.isfinite(value) else f"{100 * value:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if np.isfinite(value) and value > .56 else "#172536")
        labels = ["Original", "Tags removed", "Block moved", *(f"Wrapper → {ROLE_LABELS[ROLES.index(wrapper)]}" for wrapper in ROLES if wrapper != role)]
        count = int(table.n_traces.iloc[0])
        ax.set(xticks=np.arange(6), xticklabels=ROLE_LABELS, yticks=np.arange(8), yticklabels=labels,
               title=f"{ROLE_LABELS[ri]} content (n={count})", xlabel="Probe role")
        ax.tick_params(axis="x", labelrotation=40, length=0, labelsize=9)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.8)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes((.935, .25, .014, .46))
    bar = fig.colorbar(im, cax=cax)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    bar.set_label("Mean role probability")
    fig.suptitle("Full six-role probe probabilities after each intervention", fontsize=20, fontweight="bold", y=.96)
    fig.text(.5, .91, cohort_label(cohort, n) + "  |  Cell values are percentages", ha="center", fontsize=12)
    fig.text(.5, .87, "Each target body has eight readouts: original, strip, move, and five alternative wrappers.", ha="center", fontsize=11)
    fig.text(.5, .065, uncertainty_label(bootstrap) + ". Full readouts and paired changes: full-role-probabilities.csv.\n" + LIMITATION,
             ha="center", fontsize=9.5, linespacing=1.6, color="#4B5663")
    save_figure(fig, destination, "full-six-role-probabilities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "all"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    stage_root = args.output / args.stage
    frame, validation = read_and_validate(stage_root / "input-summary.csv", args.stage)
    analysis_root = stage_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_json(analysis_root / "validation.json", validation)
    cohorts = {"pilot": frame} if args.stage == "pilot" else {"fit": frame[frame.source_split == "fit"], "all": frame}
    for cohort, subset in cohorts.items():
        report = summarize(subset, analysis_root / cohort, cohort, validation)
        print(json.dumps({"cohort": cohort, "n_traces": report["n_traces"], "role_support": report["bootstrap"]["role_support"],
                          "output": str((analysis_root / cohort).resolve())}))


if __name__ == "__main__":
    main()
