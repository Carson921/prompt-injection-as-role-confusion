#!/usr/bin/env python3
"""Paired numerical sensitivity to omitting the single final end token.

The frozen primary experiment is unchanged. Compare its input-summary.csv with
numerical-length-sensitivity/input-summary.csv, which must contain exactly the
same 1,200 formal inputs and body-token denominators. The alternate forward pass
omits the final post-body ending token. Factors, texts, body regions and probe
remain fixed. This comparison diagnoses numerical sensitivity to sequence length
in the native forward implementation; it does not estimate a causal influence
of a future suffix on earlier autoregressive token states.

Use the same 2,000 semantic-pair bootstrap resamples (seed 20261112) for original,
alternate, and alternate-minus-original effects. Keep all conditions and fillers
within every resampled semantic unit. CIs are pointwise percentile intervals.
No observation, model, checkpoint, or analysis choice is selected by its result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_role_signal_factorial as primary


SEED = 20261112
REPLICATES = 2000
ALTERNATE_RELATIVE = Path("numerical-length-sensitivity/input-summary.csv")


def paired_cells(original: pd.DataFrame, alternate: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Assert a complete input pairing before aggregating equal-weight means."""
    identity = ["input_id", "pair_id", "filler_id", "text_style", "wrapper", "position", "body_tokens"]
    if not original[identity].equals(alternate[identity]):
        different = [column for column in identity if not original[column].equals(alternate[column])]
        raise ValueError(f"Original/alternate formal input identity or denominator mismatch: {different}")
    arrays = []
    for frame in (original, alternate):
        by_filler = frame[list(primary.METRICS)].to_numpy(dtype=np.float64).reshape(50, 2, 2, 2, 3, len(primary.METRICS))
        arrays.append(np.stack((by_filler[:, 0], by_filler[:, 1], by_filler.mean(axis=1)), axis=1))
    return arrays[0], arrays[1]


def paired_bootstrap(original: np.ndarray, alternate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 50, size=(REPLICATES, 50))
    counts = np.stack([np.bincount(draw, minlength=50) for draw in draws]) / 50.0
    result = []
    for cells in (original, alternate):
        result.append((counts @ cells.reshape(50, -1)).reshape(REPLICATES, *cells.shape[1:]))
    return result[0], result[1]


def sensitivity_table(original: np.ndarray, alternate: np.ndarray,
                      original_bootstrap: np.ndarray, alternate_bootstrap: np.ndarray,
                      names: tuple[str, ...]) -> pd.DataFrame:
    original_low, original_high = primary.interval(original_bootstrap)
    alternate_low, alternate_high = primary.interval(alternate_bootstrap)
    delta = alternate - original
    delta_low, delta_high = primary.interval(alternate_bootstrap - original_bootstrap)
    rows = []
    for g, group in enumerate(primary.GROUPS):
        for e, name in enumerate(names):
            for m, metric in enumerate(primary.METRICS):
                index = (g, e, m)
                values = {
                    "original_estimate": float(original[index]),
                    "original_ci_lower": float(original_low[index]),
                    "original_ci_upper": float(original_high[index]),
                    "alternate_estimate": float(alternate[index]),
                    "alternate_ci_lower": float(alternate_low[index]),
                    "alternate_ci_upper": float(alternate_high[index]),
                    "delta": float(delta[index]), "delta_estimate": float(delta[index]),
                    "delta_ci_lower": float(delta_low[index]),
                    "delta_ci_upper": float(delta_high[index]),
                }
                row: dict[str, Any] = {
                    "filler_id": group, "contrast": name, "metric": metric, **values,
                    "unit": "logit_margin" if metric == "mean_margin" else "probability",
                    "n_pairs": 50, "n_inputs_per_run": 1200,
                    "delta_definition": "alternate minus original",
                    "ci_type": "pointwise paired semantic-pair cluster percentile bootstrap 95%",
                }
                for column, value in values.items():
                    row[column + "_pp"] = value * 100 if metric != "mean_margin" else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def verify_original_statistics(original_table: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    keys = ["filler_id", "contrast", "metric"]
    joined = original_table.merge(sensitivity, on=keys, validate="one_to_one")
    if len(joined) != len(original_table) or len(joined) != len(sensitivity):
        raise ValueError("Frozen primary result and sensitivity estimand sets differ")
    for frozen, recomputed in (("estimate", "original_estimate"), ("ci_lower", "original_ci_lower"),
                                ("ci_upper", "original_ci_upper")):
        if not np.allclose(joined[frozen], joined[recomputed], rtol=1e-9, atol=1e-10):
            raise ValueError(f"Recomputed primary {frozen} differs from the frozen primary result")


def forest(ax: plt.Axes, table: pd.DataFrame, names: tuple[str, ...], labels: tuple[str, ...],
           delta: bool, legend: bool = False) -> None:
    rows = table[(table["filler_id"] == "pooled") & (table["metric"] == "mean_margin")].set_index("contrast").loc[list(names)]
    variants = (("delta", 0.0, "#793C64", "D", "Paired change"),) if delta else (
        ("original", -0.12, "#253B63", "o", "Original: native final end token"),
        ("alternate", 0.12, "#CA8139", "D", "Alternate: final end token omitted"),
    )
    for prefix, offset, color, marker, label in variants:
        estimate_column = "delta" if prefix == "delta" else prefix + "_estimate"
        y = np.arange(len(names)) + offset
        ax.hlines(y, rows[prefix + "_ci_lower"], rows[prefix + "_ci_upper"], color=color, lw=1.8)
        ax.scatter(rows[estimate_column], y, color=color, marker=marker, s=25, label=label, zorder=3)
    ax.axvline(0, color="#777777", linestyle=":", lw=0.8)
    ax.set_yticks(np.arange(len(names)), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Change in effect on CoT − user logit margin" if delta else "Effect on CoT − user logit margin")
    if len(names) == 5:
        ax.axhline(2.5, color="#CCCCCC", lw=0.7)
    if legend:
        ax.legend(loc="best", frameon=False, fontsize=8)


def make_figure(output: Path, effects: pd.DataFrame, interactions: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.5))
    for row, table, names, labels in (
        (0, effects, primary.EFFECTS, primary.EFFECT_LABELS),
        (1, interactions, primary.INTERACTIONS, primary.INTERACTION_LABELS),
    ):
        forest(axes[row, 0], table, names, labels, delta=False, legend=(row == 0))
        forest(axes[row, 1], table, names, labels, delta=True)
    axes[0, 0].set_title("Main effects · original and alternate 95% CIs")
    axes[0, 1].set_title("Main effects · paired alternate − original 95% CIs")
    axes[1, 0].set_title("Interactions · original and alternate 95% CIs")
    axes[1, 1].set_title("Interactions · paired alternate − original 95% CIs")
    fig.suptitle(primary.MODEL_TITLE + " · Numerical sequence-length sensitivity · A/B equal average", fontsize=12)
    fig.text(0.5, -0.025, "All 1,200 formal inputs in each run. Pointwise 95% paired cluster bootstrap CIs; "
             "50 semantic pairs; 2,000 shared resamples. Only the post-body final end token is omitted in the alternate run.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.1)
    primary.save_figure(fig, output / "figures/role-signal-length-sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Main experiment directory")
    args = parser.parse_args()
    output = args.output.resolve()
    original_path = output / "input-summary.csv"
    alternate_path = output / ALTERNATE_RELATIVE
    original, original_validation = primary.read_and_validate(original_path)
    alternate, alternate_validation = primary.read_and_validate(alternate_path)
    original_pairs, alternate_pairs = paired_cells(original, alternate)
    original_bootstrap, alternate_bootstrap = paired_bootstrap(original_pairs, alternate_pairs)
    original_effects, original_interactions = primary.estimands(original_pairs.mean(axis=0))
    alternate_effects, alternate_interactions = primary.estimands(alternate_pairs.mean(axis=0))
    original_boot_effects, original_boot_interactions = primary.estimands(original_bootstrap)
    alternate_boot_effects, alternate_boot_interactions = primary.estimands(alternate_bootstrap)
    effects = sensitivity_table(original_effects, alternate_effects, original_boot_effects,
                                alternate_boot_effects, primary.EFFECTS)
    interactions = sensitivity_table(original_interactions, alternate_interactions, original_boot_interactions,
                                     alternate_boot_interactions, primary.INTERACTIONS)
    verify_original_statistics(pd.read_csv(output / "effects.csv"), effects)
    verify_original_statistics(pd.read_csv(output / "interactions.csv"), interactions)
    make_figure(output, effects, interactions)
    effects.to_csv(output / "length-sensitivity-effects.csv", index=False, float_format="%.12g")
    interactions.to_csv(output / "length-sensitivity-interactions.csv", index=False, float_format="%.12g")
    input_changes = alternate[list(primary.METRICS)].to_numpy() - original[list(primary.METRICS)].to_numpy()
    pooled_effects = effects[(effects["filler_id"] == "pooled") & (effects["metric"] == "mean_margin")]
    pooled_interactions = interactions[(interactions["filler_id"] == "pooled") & (interactions["metric"] == "mean_margin")]
    summary = {
        "analysis": "post-run numerical sensitivity to one-token post-body ending omission",
        "primary_analysis": "unchanged frozen 50-pair factorial analysis; no result-dependent data/model selection",
        "alternate_input_summary": str(ALTERNATE_RELATIVE),
        "original_validation": original_validation, "alternate_validation": alternate_validation,
        "paired_identity_and_body_denominators_equal": True, "original_estimates_and_CIs_reproduced": True,
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "unit": "semantic pair_id",
                      "confidence_level": 0.95, "intervals": "pointwise paired percentile, not simultaneous",
                      "shared_resamples": "original/alternate/difference; all conditions, fillers and metrics"},
        "delta_definition": "alternate minus original",
        "weighting": "equal semantic-pair weights; pooled averages fillers A/B within pair; factors equally averaged",
        "input_level_absolute_change": {
            metric: {"mean": float(np.abs(input_changes[:, m]).mean()),
                     "max": float(np.abs(input_changes[:, m]).max())}
            for m, metric in enumerate(primary.METRICS)
        },
        "pooled_margin_effects": pooled_effects[["contrast", "original_estimate", "alternate_estimate", "delta", "delta_ci_lower", "delta_ci_upper"]].to_dict(orient="records"),
        "pooled_margin_interactions": pooled_interactions[["contrast", "original_estimate", "alternate_estimate", "delta", "delta_ci_lower", "delta_ci_upper"]].to_dict(orient="records"),
        "interpretation": [
            "This is a numerical sequence-length sensitivity comparison under the fixed native forward implementation.",
            "A causal autoregressive model's earlier body states cannot depend on a later end token; differing body readouts reflect numerical/implementation effects rather than future-token conditioning.",
            "All formal inputs are retained. Original and alternate effects, their separate CIs, and paired changes with CIs are reported without a preset robustness conclusion.",
            "This sensitivity is separate from the frozen primary analysis; it does not replace the primary model, checkpoint, data or effect estimates.",
        ],
        "source_hashes": {
            "input-summary.csv": primary.sha256(original_path),
            str(ALTERNATE_RELATIVE): primary.sha256(alternate_path),
            "effects.csv": primary.sha256(output / "effects.csv"),
            "interactions.csv": primary.sha256(output / "interactions.csv"),
            "primary_script": primary.sha256(Path(primary.__file__)),
            "length_sensitivity_script": primary.sha256(Path(__file__)),
        },
    }
    (output / "length-sensitivity-manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "formal_pairs": 50, "inputs_per_run": 1200,
                      "original_statistics_reproduced": True, "paired_delta_bootstrap": True}, indent=2))


if __name__ == "__main__":
    main()
