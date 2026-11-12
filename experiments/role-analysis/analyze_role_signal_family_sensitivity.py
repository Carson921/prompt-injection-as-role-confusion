#!/usr/bin/env python3
"""Prespecified family-cluster uncertainty sensitivity, separate from primary code.

Run after analyze_role_signal_factorial.py, with the same --output directory.
Read pair_id/family assignments from its frozen pairs.jsonl. Draw 20 problem
families with replacement, preserving each family's complete semantic pairs and
all conditions/fillers. Divide the sampled family-sum numerator by the sampled
semantic-pair count: the point estimand remains the equally weighted 50-pair
mean, even though families contain two or three pairs and bootstrap draw sizes
vary. Never give a family equal point-estimate weight in place of a pair.

The frozen 50-pair primary analysis remains unchanged. These are pointwise,
percentile 95% bootstrap intervals, not simultaneous intervals or significance
tests. Shared text-style scaffolds across all families remain a limitation that
family resampling cannot remove.
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


def family_map(path: Path, pair_ids: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    formal = [record for record in records if record.get("split") == "formal"]
    mapping: dict[str, str] = {}
    for record in formal:
        pair_id = str(record["pair_id"])
        if pair_id in mapping:
            raise ValueError(f"Duplicate formal pair in pairs.jsonl: {pair_id}")
        family = record.get("family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"Missing problem-family assignment for {pair_id}")
        mapping[pair_id] = family
    if set(mapping) != set(pair_ids):
        raise ValueError("Formal pair IDs in pairs.jsonl and input-summary.csv differ")
    sizes = pd.Series(mapping).value_counts().sort_index()
    if len(sizes) != 20 or not set(sizes).issubset({2, 3}):
        raise ValueError("Expected 20 frozen formal families, each containing two or three semantic pairs")
    return mapping, {"n_families": 20, "family_sizes": {str(k): int(v) for k, v in sizes.items()},
                     "pair_to_family": mapping}


def family_resamples(pair_cells: np.ndarray, pair_families: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-pair bootstrap cell means and varying semantic-pair counts."""
    families = sorted(set(pair_families))
    family_indices = np.asarray([families.index(family) for family in pair_families])
    sums = np.stack([pair_cells[family_indices == f].sum(axis=0) for f in range(len(families))])
    sizes = np.bincount(family_indices, minlength=len(families))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(families), size=(REPLICATES, len(families)))
    multiplicities = np.stack([np.bincount(draw, minlength=len(families)) for draw in draws])
    sampled_pair_counts = multiplicities @ sizes
    numerator = multiplicities @ sums.reshape(len(families), -1)
    means = numerator / sampled_pair_counts[:, None]
    return means.reshape(REPLICATES, *pair_cells.shape[1:]), sampled_pair_counts


def comparison_forest(ax: plt.Axes, pair_table: pd.DataFrame, family_table: pd.DataFrame,
                      names: tuple[str, ...], labels: tuple[str, ...], metric: str,
                      legend: bool = False) -> None:
    scale = 1 if metric == "mean_margin" else 100
    for table, offset, color, marker, label in (
        (pair_table, -0.12, "#92A2B7", "o", "Primary: 50 semantic-pair clusters"),
        (family_table, 0.12, "#5B3A78", "D", "Sensitivity: 20 problem-family clusters"),
    ):
        rows = table[(table["filler_id"] == "pooled") & (table["metric"] == metric)].set_index("contrast").loc[list(names)]
        y = np.arange(len(names)) + offset
        ax.hlines(y, rows["ci_lower"] * scale, rows["ci_upper"] * scale, color=color, lw=1.8)
        ax.scatter(rows["estimate"] * scale, y, color=color, marker=marker, s=25, label=label, zorder=3)
    ax.axvline(0, linestyle=":", color="#777777", lw=0.8)
    ax.set_yticks(np.arange(len(names)), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Effect on CoT − user logit margin" if metric == "mean_margin" else "Effect on P(CoT) − P(user), percentage points")
    if len(names) == 5:
        ax.axhline(2.5, color="#CCCCCC", lw=0.7)
    if legend:
        ax.legend(loc="best", fontsize=8, frameon=False)


def make_figure(output: Path, effects: pd.DataFrame, interactions: pd.DataFrame) -> None:
    pair_effects = pd.read_csv(output / "effects.csv")
    pair_interactions = pd.read_csv(output / "interactions.csv")
    # Both methods estimate the same equally weighted population of semantic pairs.
    for pair, family in ((pair_effects, effects), (pair_interactions, interactions)):
        keys = ["filler_id", "contrast", "metric"]
        joined = pair.merge(family, on=keys, suffixes=("_pair", "_family"), validate="one_to_one")
        if len(joined) != len(pair) or len(joined) != len(family):
            raise ValueError("Primary/sensitivity contrast tables do not contain the same estimands")
        if not np.allclose(joined["estimate_pair"], joined["estimate_family"], rtol=1e-9, atol=1e-10):
            raise ValueError("Family sensitivity changed point estimates instead of only the uncertainty model")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for column, metric in enumerate(("mean_margin", "probability_contrast")):
        comparison_forest(axes[0, column], pair_effects, effects, primary.EFFECTS,
                          primary.EFFECT_LABELS, metric, legend=(column == 0))
        comparison_forest(axes[1, column], pair_interactions, interactions, primary.INTERACTIONS,
                          primary.INTERACTION_LABELS, metric)
        label = "CoT − user logit margin" if metric == "mean_margin" else "P(CoT) − P(user)"
        axes[0, column].set_title("Main effects · " + label)
        axes[1, column].set_title("Text × Wrapper interactions · " + label)
    fig.suptitle(primary.MODEL_TITLE + " · Family-cluster sensitivity · A/B equal average", fontsize=12)
    fig.text(0.5, -0.025, "Pointwise 95% paired cluster bootstrap CIs; 2,000 resamples. "
             "Semantic pairs remain equally weighted; common style scaffolds remain shared across all families.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.1)
    primary.save_figure(fig, output / "figures/role-signal-family-sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "input-summary.csv"
    formal, validation = primary.read_and_validate(source)
    pair_ids = validation["pair_ids"]
    mapping, family_validation = family_map(output / "pairs.jsonl", pair_ids)
    by_filler = formal[list(primary.METRICS)].to_numpy(dtype=np.float64).reshape(50, 2, 2, 2, 3, len(primary.METRICS))
    pair_cells = np.stack((by_filler[:, 0], by_filler[:, 1], by_filler.mean(axis=1)), axis=1)
    cells = pair_cells.mean(axis=0)
    bootstrap_cells, counts = family_resamples(pair_cells, [mapping[pair_id] for pair_id in pair_ids])
    effects_point, interactions_point = primary.estimands(cells)
    effects_bootstrap, interactions_bootstrap = primary.estimands(bootstrap_cells)
    effects = primary.contrast_frame(effects_point, effects_bootstrap, primary.EFFECTS)
    interactions = primary.contrast_frame(interactions_point, interactions_bootstrap, primary.INTERACTIONS)
    for frame in (effects, interactions):
        frame["n_families"] = 20
        frame["cluster_unit"] = "problem family"
        frame["ci_type"] = "pointwise family-cluster percentile bootstrap 95%; equal semantic-pair weights"
    # Check fixed primary estimands and produce the comparison before writing result tables.
    make_figure(output, effects, interactions)
    effects.to_csv(output / "effects-family-sensitivity.csv", index=False, float_format="%.12g")
    interactions.to_csv(output / "interactions-family-sensitivity.csv", index=False, float_format="%.12g")
    summary = {
        "analysis": "prespecified sensitivity of uncertainty estimates to problem-family clustering",
        "primary_analysis": "unchanged: 50 semantic-pair clusters; see bootstrap-summary.json",
        "formal_pairs": 50, "formal_inputs": 1200, **family_validation,
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "unit": "problem family",
                      "clusters_per_draw": 20, "confidence_level": 0.95,
                      "intervals": "pointwise percentile, not simultaneous",
                      "sampled_semantic_pair_count_min": int(counts.min()),
                      "sampled_semantic_pair_count_max": int(counts.max()),
                      "sampled_semantic_pair_count_mean": float(counts.mean()),
                      "shared_resamples": "all semantic pairs in each family, all conditions, both fillers, all metrics"},
        "estimand": "equal semantic-pair weights; pooled outcome first averages fillers A/B within pair",
        "bootstrap_ratio": "sum of all sampled semantic-pair cell values / number of sampled semantic pairs",
        "limitations": [
            "20 controlled problem families are the sensitivity clusters; 50 pair IDs are not 50 independent topics.",
            "Styled and destyled texts use controlled common style scaffolds across all families.",
            "Family clustering allows dependence within families; common scaffolds shared across families remain outside this bootstrap uncertainty model.",
            "Neither bootstrap is a guarantee of generalization to naturally generated CoT or unrestricted rewrite distributions.",
            "No family selection, sample selection, checkpoint selection, or outcome-dependent analysis choice.",
        ],
        "source_hashes": {"input-summary.csv": primary.sha256(source),
                          "pairs.jsonl": primary.sha256(output / "pairs.jsonl"),
                          "primary_script": primary.sha256(Path(primary.__file__)),
                          "family_sensitivity_script": primary.sha256(Path(__file__))},
    }
    (output / "bootstrap-family-sensitivity.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "family_clusters": 20, "semantic_pairs": 50,
                      "bootstrap_replicates": REPLICATES, "point_estimates_equal_primary": True}, indent=2))


if __name__ == "__main__":
    main()
