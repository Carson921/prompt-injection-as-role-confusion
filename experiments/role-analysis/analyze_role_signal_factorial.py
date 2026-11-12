#!/usr/bin/env python3
"""Analyze the preregistered paired Text x Wrapper x Position experiment.

The only observations used here are formal input-level readouts. Each of the
50 semantic pairs must have all 24 combinations, including both neutral-context
replicates. Each input and each semantic pair receive equal weight, regardless
of body length. Filler A and B are first analyzed separately; the pooled estimate
averages A and B within pair. The same semantic-pair bootstrap indices are used
for all conditions, fillers, metrics, contrasts, and plots.

The main outcome is mean token-level z_CoT - z_user supplied as mean_margin;
this script never reconstructs logits from averaged probabilities. Intervals
are pointwise paired cluster percentile bootstrap 95% intervals, not simultaneous
intervals. The effects are signed contrasts rather than contribution percentages.

Usage:
    python analyze_role_signal_factorial.py --output OUTPUT_DIRECTORY

Requires numpy, pandas, matplotlib. No model loading or network access occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FILLERS = ("A", "B")
GROUPS = ("A", "B", "pooled")
TEXTS = ("styled", "destyled")
WRAPPERS = ("cot", "user")
POSITIONS = (32, 256, 768)
PROBABILITIES = ("p_system", "p_developer", "p_user", "p_cot", "p_assistant", "p_tool")
METRICS = ("mean_margin", *PROBABILITIES, "probability_contrast")
EFFECTS = ("text_styled_minus_destyled", "wrapper_cot_minus_user", "position_256_minus_32", "position_768_minus_32")
EFFECT_LABELS = (
    "Text: styled − destyled",
    "Wrapper: analysis − user",
    "Position: 256 − 32",
    "Position: 768 − 32",
)
INTERACTIONS = (
    "text_x_wrapper_at_32", "text_x_wrapper_at_256", "text_x_wrapper_at_768",
    "text_x_wrapper_256_minus_32", "text_x_wrapper_768_minus_32",
)
INTERACTION_LABELS = (
    "Text × Wrapper at 32", "Text × Wrapper at 256", "Text × Wrapper at 768",
    "Interaction: 256 − 32", "Interaction: 768 − 32",
)
CONDITION_LABELS = (
    "Styled · analysis wrapper", "Styled · user wrapper",
    "Destyled · analysis wrapper", "Destyled · user wrapper",
)
GROUP_LABELS = {"A": "Context A", "B": "Context B", "pooled": "A/B equal average"}
GROUP_COLORS = {"A": "#8A9BB0", "B": "#D4A565", "pooled": "#253B63"}
CONDITION_COLORS = ("#256D9C", "#C56D36", "#256D9C", "#C56D36")
MODEL_TITLE = "GPT-OSS-120B · frozen six-role probe"
CI_LABEL = "Pointwise 95% paired cluster bootstrap CI; 50 semantic pairs; 2,000 resamples"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_and_validate(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reject incomplete/unbalanced formal data rather than silently dropping it."""
    frame = pd.read_csv(path, dtype={"input_id": str, "pair_id": str})
    required = {
        "input_id", "pair_id", "split", "filler_id", "text_style", "wrapper", "position",
        "mean_margin", "body_tokens", *PROBABILITIES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input-summary columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Input summary contains missing required values")
    if frame["input_id"].duplicated().any():
        raise ValueError("input_id must be unique across pilot and formal records")
    if not set(frame["split"]).issubset({"pilot", "formal"}):
        raise ValueError("split must contain only pilot or formal")
    pilot_pairs = set(frame.loc[frame["split"] == "pilot", "pair_id"])
    formal = frame.loc[frame["split"] == "formal"].copy()
    formal_pairs = sorted(formal["pair_id"].unique())
    if pilot_pairs.intersection(formal_pairs):
        raise ValueError("Semantic pair IDs overlap pilot and formal splits")
    if len(formal_pairs) != 50:
        raise ValueError(f"Expected exactly 50 frozen formal pairs; found {len(formal_pairs)}")
    if len(formal) != 1200:
        raise ValueError(f"Expected 1,200 formal inputs (50 × 24); found {len(formal)}")
    for column, allowed in (("filler_id", FILLERS), ("text_style", TEXTS), ("wrapper", WRAPPERS)):
        if set(formal[column]) != set(allowed):
            raise ValueError(f"Formal {column} must cover exactly {allowed}")
    for column in ("position", "body_tokens", "mean_margin", *PROBABILITIES):
        formal[column] = pd.to_numeric(formal[column], errors="raise")
        if not np.isfinite(formal[column]).all():
            raise ValueError(f"Nonfinite formal values in {column}")
    if set(formal["position"]) != set(POSITIONS):
        raise ValueError(f"Formal positions must be exactly {POSITIONS}")
    if (formal["body_tokens"] <= 0).any() or (formal["body_tokens"] % 1 != 0).any():
        raise ValueError("body_tokens must be positive integers")
    probability_values = formal[list(PROBABILITIES)].to_numpy()
    if ((probability_values < 0) | (probability_values > 1)).any():
        raise ValueError("Role probabilities must lie in [0, 1]")
    probability_sum_error = np.abs(probability_values.sum(axis=1) - 1)
    if probability_sum_error.max() > 1e-5:
        raise ValueError("Six-class softmax probabilities must sum to 1 within 1e-5")
    index_columns = ["pair_id", "filler_id", "text_style", "wrapper", "position"]
    if formal.duplicated(index_columns).any():
        raise ValueError("Duplicate formal factorial condition within a semantic pair")
    expected_index = pd.MultiIndex.from_product(
        [formal_pairs, FILLERS, TEXTS, WRAPPERS, POSITIONS], names=index_columns
    )
    actual_index = pd.MultiIndex.from_frame(formal[index_columns])
    absent = expected_index.difference(actual_index)
    extra = actual_index.difference(expected_index)
    if len(absent) or len(extra):
        raise ValueError(f"Incomplete formal factorial: {len(absent)} missing, {len(extra)} extra cells")
    # The same body must be reused under all wrappers, positions, and contexts.
    if (formal.groupby(["pair_id", "text_style"])["body_tokens"].nunique() != 1).any():
        raise ValueError("Body token count varies across wrapper/position/filler for the same text")
    formal["probability_contrast"] = formal["p_cot"] - formal["p_user"]
    formal = formal.set_index(index_columns).reindex(expected_index).reset_index()
    validation = {
        "source_rows": len(frame), "excluded_pilot_rows": int((frame["split"] == "pilot").sum()),
        "formal_inputs": len(formal), "formal_semantic_pairs": len(formal_pairs),
        "conditions_per_pair": 24, "full_factorial": True,
        "max_probability_sum_error": float(probability_sum_error.max()),
        "body_lengths_invariant_across_conditions": True,
        "body_tokens_min": int(formal["body_tokens"].min()),
        "body_tokens_max": int(formal["body_tokens"].max()),
        "pair_ids": formal_pairs,
    }
    return formal, validation


def estimands(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map [..., text, wrapper, position, metric] to signed paired contrasts."""
    text = (cells[..., 0, :, :, :] - cells[..., 1, :, :, :]).mean(axis=(-3, -2))
    wrapper = (cells[..., :, 0, :, :] - cells[..., :, 1, :, :]).mean(axis=(-3, -2))
    pos256 = (cells[..., :, :, 1, :] - cells[..., :, :, 0, :]).mean(axis=(-3, -2))
    pos768 = (cells[..., :, :, 2, :] - cells[..., :, :, 0, :]).mean(axis=(-3, -2))
    effects = np.stack((text, wrapper, pos256, pos768), axis=-2)
    interaction = cells[..., 0, 0, :, :] - cells[..., 1, 0, :, :] - cells[..., 0, 1, :, :] + cells[..., 1, 1, :, :]
    interactions = np.stack(
        (interaction[..., 0, :], interaction[..., 1, :], interaction[..., 2, :],
         interaction[..., 1, :] - interaction[..., 0, :], interaction[..., 2, :] - interaction[..., 0, :]),
        axis=-2,
    )
    return effects, interactions


def interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return bounds[0], bounds[1]


def contrast_frame(point: np.ndarray, bootstrap: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    low, high = interval(bootstrap)
    rows = []
    for g, group in enumerate(GROUPS):
        for e, name in enumerate(names):
            for m, metric in enumerate(METRICS):
                probability = metric != "mean_margin"
                rows.append({
                    "filler_id": group, "contrast": name, "metric": metric,
                    "estimate": float(point[g, e, m]), "ci_lower": float(low[g, e, m]),
                    "ci_upper": float(high[g, e, m]), "unit": "probability" if probability else "logit_margin",
                    "estimate_pp": float(100 * point[g, e, m]) if probability else np.nan,
                    "ci_lower_pp": float(100 * low[g, e, m]) if probability else np.nan,
                    "ci_upper_pp": float(100 * high[g, e, m]) if probability else np.nan,
                    "n_pairs": 50, "ci_type": "pointwise paired cluster percentile bootstrap 95%",
                })
    return pd.DataFrame(rows)


def raw_means_frame(point: np.ndarray, bootstrap: np.ndarray) -> pd.DataFrame:
    low, high = interval(bootstrap)
    rows = []
    for g, group in enumerate(GROUPS):
        for t, text in enumerate(TEXTS):
            for w, wrapper in enumerate(WRAPPERS):
                for p, position in enumerate(POSITIONS):
                    for m, metric in enumerate(METRICS):
                        cell = (g, t, w, p, m)
                        rows.append({
                            "filler_id": group, "text_style": text, "wrapper": wrapper,
                            "position": position, "metric": metric, "mean": float(point[cell]),
                            "ci_lower": float(low[cell]), "ci_upper": float(high[cell]), "n_pairs": 50,
                        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forest(ax: plt.Axes, table: pd.DataFrame, metric: str, names: tuple[str, ...],
           labels: tuple[str, ...], legend: bool = False) -> None:
    scale = 1 if metric == "mean_margin" else 100
    offsets = (-0.18, 0.0, 0.18)
    for group, offset in zip(GROUPS, offsets):
        selected = table[(table["metric"] == metric) & (table["filler_id"] == group)].set_index("contrast").loc[list(names)]
        estimate = selected["estimate"].to_numpy() * scale
        low = selected["ci_lower"].to_numpy() * scale
        high = selected["ci_upper"].to_numpy() * scale
        y = np.arange(len(names)) + offset
        # Draw endpoints directly: percentile intervals need not contain the estimate.
        ax.hlines(y, low, high, color=GROUP_COLORS[group], lw=1.7)
        ax.scatter(estimate, y, color=GROUP_COLORS[group], s=24 if group == "pooled" else 19,
                   marker="D" if group == "pooled" else "o", label=GROUP_LABELS[group], zorder=4)
    ax.axvline(0, color="#777777", lw=0.8, linestyle=":")
    ax.set_yticks(np.arange(len(names)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Effect on CoT − user logit margin" if metric == "mean_margin" else "Effect (percentage points)")
    ax.grid(axis="x", alpha=0.2)
    if legend:
        ax.legend(loc="best", frameon=False, fontsize=8)


def position_curves(ax: plt.Axes, cells: np.ndarray, bounds: tuple[np.ndarray, np.ndarray],
                    group: int, metric: str, legend: bool = False) -> None:
    m = METRICS.index(metric)
    scale = 1 if metric == "mean_margin" else 100
    low, high = bounds
    for condition, (t, w) in enumerate((t, w) for t in range(2) for w in range(2)):
        color = CONDITION_COLORS[condition]
        style = "-" if t == 0 else "--"
        y = cells[group, t, w, :, m] * scale
        ax.plot(POSITIONS, y, color=color, linestyle=style, marker="o", markersize=4,
                linewidth=1.8, label=CONDITION_LABELS[condition])
        ax.fill_between(POSITIONS, low[group, t, w, :, m] * scale,
                        high[group, t, w, :, m] * scale, color=color, alpha=0.085)
    ax.axhline(0, color="#777777", lw=0.8, linestyle=":")
    ax.set_xticks(POSITIONS)
    ax.set_xlabel("Body first-token position (zero-based)")
    ax.set_ylabel("Mean CoT − user logit margin" if metric == "mean_margin" else "P(CoT) − P(user), percentage points")
    ax.grid(alpha=0.15)
    if legend:
        ax.legend(loc="best", frameon=False, fontsize=7)


def make_figures(output: Path, cells: np.ndarray, bootstrap_cells: np.ndarray,
                 effects: pd.DataFrame, interactions: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    bounds = interval(bootstrap_cells)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.7), gridspec_kw={"width_ratios": (1.05, 1.2, 1.2)})
    matrix = cells[2, ..., METRICS.index("probability_contrast")].reshape(4, 3) * 100
    im = axes[0].imshow(matrix, cmap="RdBu", vmin=-100, vmax=100, aspect="auto")
    axes[0].set_xticks(range(3), POSITIONS)
    axes[0].set_yticks(range(4), CONDITION_LABELS)
    axes[0].set_xlabel("Body first-token position (zero-based)")
    axes[0].set_title("a  Role probability contrast · A/B average", loc="left", pad=12)
    for row in range(4):
        for column in range(3):
            value = matrix[row, column]
            axes[0].text(column, row, f"{value:+.1f}", ha="center", va="center",
                         color="white" if abs(value) > 55 else "#202020", fontsize=9)
    fig.colorbar(im, ax=axes[0], pad=0.03, shrink=0.8).set_label("P(CoT) − P(user), percentage points")
    forest(axes[1], effects, "mean_margin", EFFECTS, EFFECT_LABELS, legend=True)
    axes[1].set_title("b  Paired main effects · 95% CI", loc="left", pad=12)
    position_curves(axes[2], cells, bounds, 2, "mean_margin", legend=True)
    axes[2].set_title("c  Position curves · A/B average · 95% CI", loc="left", pad=12)
    fig.suptitle(MODEL_TITLE, x=0.5, y=1.015, fontsize=13)
    fig.text(0.5, -0.045, CI_LABEL + ". Context position includes the neutral-prefix-length intervention.", ha="center", fontsize=8)
    fig.tight_layout(w_pad=2.1)
    save_figure(fig, figure_dir / "role-signal-main")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.5), sharey=True)
    for g, group in enumerate(GROUPS):
        # 12 ordered conditions x six-role probabilities, equal semantic-pair weights.
        values = cells[g, ..., 1:7].reshape(12, 6)
        im = axes[g].imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes[g].set_xticks(range(6), ("System", "Developer", "User", "CoT", "Assistant", "Tool"), rotation=45, ha="right")
        axes[g].set_title(GROUP_LABELS[group])
        for row in range(12):
            for column in range(6):
                value = values[row, column]
                axes[g].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                             color="#222222" if value > 0.6 else "white")
    labels = [f"{CONDITION_LABELS[2*t+w]} · {position}" for t in range(2) for w in range(2) for position in POSITIONS]
    axes[0].set_yticks(range(12), labels)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.02, label="Mean role probability")
    fig.suptitle(MODEL_TITLE + "\nAll role probabilities by condition", y=1.01, fontsize=12)
    fig.subplots_adjust(left=0.25, bottom=0.13, top=0.89, right=0.9, wspace=0.12)
    save_figure(fig, figure_dir / "role-signal-probabilities")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), sharey=True)
    for ax, metric, label in zip(axes, ("p_cot", "p_user", "probability_contrast"), ("P(CoT)", "P(user)", "P(CoT) − P(user)")):
        forest(ax, effects, metric, EFFECTS, EFFECT_LABELS, legend=(metric == "p_cot"))
        ax.set_title(label)
    fig.suptitle(MODEL_TITLE + " · Probability main effects", fontsize=12)
    fig.text(0.5, -0.025, CI_LABEL, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, figure_dir / "role-signal-probability-effects")

    fig, axes = plt.subplots(2, 2, figsize=(12.7, 8.1))
    for ax, metric, label in zip(axes.flat, ("mean_margin", "probability_contrast", "p_cot", "p_user"),
                                ("CoT − user logit margin", "P(CoT) − P(user)", "P(CoT)", "P(user)")):
        forest(ax, interactions, metric, INTERACTIONS, INTERACTION_LABELS, legend=(metric == "mean_margin"))
        ax.axhline(2.5, color="#CCCCCC", lw=0.7)
        ax.set_title(label)
    fig.suptitle(MODEL_TITLE + " · Text × Wrapper interactions", fontsize=12)
    fig.text(0.5, -0.025, "Interaction = (styled − destyled) under analysis − (styled − destyled) under user. " + CI_LABEL,
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)
    save_figure(fig, figure_dir / "role-signal-interactions")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.1), sharex=True)
    for g, group in enumerate(GROUPS):
        for row, metric in enumerate(("mean_margin", "probability_contrast")):
            position_curves(axes[row, g], cells, bounds, g, metric)
            axes[row, g].set_title(GROUP_LABELS[group])
    handles = [Line2D([0], [0], color=CONDITION_COLORS[i], linestyle="-" if i < 2 else "--", marker="o", label=label)
               for i, label in enumerate(CONDITION_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(MODEL_TITLE + " · Position curves across neutral contexts", fontsize=12)
    fig.text(0.5, -0.055, CI_LABEL, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_figure(fig, figure_dir / "role-signal-position-by-context")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Existing experiment directory containing input-summary.csv")
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "input-summary.csv"
    formal, validation = read_and_validate(source)
    # Reindexing above establishes this exact axis order, independent of CSV order.
    by_filler = formal[list(METRICS)].to_numpy(dtype=np.float64).reshape(50, 2, 2, 2, 3, len(METRICS))
    pair_cells = np.stack((by_filler[:, 0], by_filler[:, 1], by_filler.mean(axis=1)), axis=1)
    cells = pair_cells.mean(axis=0)
    rng = np.random.default_rng(20261112)
    sample_indices = rng.integers(0, 50, size=(2000, 50))
    counts = np.stack([np.bincount(indices, minlength=50) for indices in sample_indices]) / 50.0
    bootstrap_cells = (counts @ pair_cells.reshape(50, -1)).reshape(2000, *cells.shape)
    point_effects, point_interactions = estimands(cells)
    boot_effects, boot_interactions = estimands(bootstrap_cells)
    effects = contrast_frame(point_effects, boot_effects, EFFECTS)
    interactions = contrast_frame(point_interactions, boot_interactions, INTERACTIONS)
    means = raw_means_frame(cells, bootstrap_cells)
    effects.to_csv(output / "effects.csv", index=False, float_format="%.12g")
    interactions.to_csv(output / "interactions.csv", index=False, float_format="%.12g")
    means.to_csv(output / "condition-means.csv", index=False, float_format="%.12g")
    summary = {
        "analysis": "paired Text x Wrapper x Position, dual neutral-context replication",
        "model": "GPT-OSS-120B", "probe": "frozen six-role probe, supplied checkpoint",
        "input_summary_sha256": sha256(source), "analysis_script_sha256": sha256(Path(__file__)),
        "validation": validation, "bootstrap": {
            "replicates": 2000, "seed": 20261112, "unit": "semantic pair_id",
            "method": "paired cluster percentile", "confidence_level": 0.95,
            "intervals": "pointwise, not simultaneous; no significance tests or multiple-testing claims",
            "shared_resamples": "all conditions, fillers, outcomes, main effects and interactions",
        },
        "weighting": {
            "input": "equal semantic-pair weights after averaging within each input's predefined body tokens",
            "pooled": "equal mean of filler A and B within each pair; not 100 independent pairs",
            "main_effects": "equal average over all levels of the other factors",
        },
        "factor_levels": {"text_style": TEXTS, "wrapper": WRAPPERS, "position": POSITIONS, "filler_id": FILLERS},
        "primary_outcome": "mean token-level z_CoT - z_user, read directly from mean_margin",
        "probability_outcomes": list(PROBABILITIES) + ["P(CoT)-P(user)"],
        "contrast_definitions": {
            "text": "styled minus destyled, averaged over wrappers and positions",
            "wrapper": "assistant analysis minus user, averaged over texts and positions",
            "position": "256 minus 32 and 768 minus 32, averaged over texts and wrappers",
            "text_x_wrapper": "(styled - destyled) under analysis - (styled - destyled) under user, at each position",
            "interaction_changes": "interaction at 256 or 768 minus interaction at 32",
        },
        "interpretation_limits": [
            "Frozen probe readouts, not behavioral attack success or instruction priority.",
            "Text effect applies to the implemented paired rewrites.",
            "Wrapper effect changes the complete native role header and associated context adjustment, not delimiter tokens in isolation.",
            "Position changes neutral-prefix length/content; it is not a pure positional-encoding intervention.",
            "No layer selection, checkpoint selection, formal-sample selection or normalized contribution percentages.",
        ],
        "outputs": {"effects": "effects.csv", "interactions": "interactions.csv", "raw_condition_means": "condition-means.csv"},
    }
    (output / "bootstrap-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    make_figures(output, cells, bootstrap_cells, effects, interactions)
    print(json.dumps({"output": str(output), "formal_pairs": 50, "formal_inputs": 1200,
                      "bootstrap_replicates": 2000, "main_plot": str(output / "figures/role-signal-main.pdf")}, indent=2))


if __name__ == "__main__":
    main()
