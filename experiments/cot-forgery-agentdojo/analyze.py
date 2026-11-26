#!/usr/bin/env python3
"""Analyze the AgentDojo CoT-forgery transfer experiment without loading a model.

All inferential summaries use traces as the sampling unit. Each segment is first
averaged across its body tokens, then payload variants are averaged within a
trace, then traces receive equal weight. Native reasoning is a descriptive
reference; it differs in content, role wrapper, and causal context. This is a
transfer experiment with the supplied six-class fixed MLP16head probe, not an
exact reproduction of the original layerwise five-class probes.
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
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
ROLE_LABELS = ("System", "Developer", "User", "CoT", "Assistant", "Tool")
P_COLUMNS = [f"p_{role}" for role in ROLES]
ROLE_COLORS = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#6B6B6B")
SEED = 20261125
N_BOOTSTRAP = 2000
GRID = np.linspace(0, 1, 51)
METRICS = {
    "benign_tool": ("no_injection", "original_tool"),
    "direct_base": ("base-injection", "direct_instruction"),
    "direct_forged": ("cot-forgery-injection", "direct_instruction"),
    "forged_reasoning": ("cot-forgery-injection", "forged_cot"),
    "native_reasoning": ("native_cot", "native_cot"),
    "payload_base": ("base-injection", "payload_all"),
    "payload_forged": ("cot-forgery-injection", "payload_all"),
}
METRIC_ORDER = ("benign_tool", "direct_base", "direct_forged", "forged_reasoning", "native_reasoning")
METRIC_LABELS = {
    "benign_tool": "Benign\ntool body",
    "direct_base": "Direct\ninstruction",
    "direct_forged": "Instruction in\nforgery payload",
    "forged_reasoning": "Forged\nreasoning",
    "native_reasoning": "Native\nreasoning",
}
SHORT_LABELS = {
    "benign_tool": "Benign tool",
    "direct_base": "Direct instruction",
    "direct_forged": "Instruction in forgery payload",
    "forged_reasoning": "Forged reasoning",
    "native_reasoning": "Native reasoning",
    "payload_base": "Base payload",
    "payload_forged": "Forgery payload",
}
COLORS = {
    "benign_tool": "#777777", "direct_base": "#0072B2",
    "direct_forged": "#D55E00", "forged_reasoning": "#E69F00",
    "native_reasoning": "#009E73",
}
CONTRASTS = {
    "forgery_instruction_minus_base_instruction": ("direct_forged", "direct_base"),
    "forgery_payload_minus_base_payload": ("payload_forged", "payload_base"),
    "forged_reasoning_minus_direct": ("forged_reasoning", "direct_base"),
    "forged_reasoning_minus_benign_tool": ("forged_reasoning", "benign_tool"),
    "forged_reasoning_minus_native_reasoning": ("forged_reasoning", "native_reasoning"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def segment_positions(raw: Any) -> np.ndarray:
    # Accept explicit lists and the documented {positions: [...]} extension.
    if isinstance(raw, dict):
        raw = raw["positions"]
    values = np.asarray(raw, dtype=np.int64)
    if values.ndim != 1 or len(values) != len(np.unique(values)):
        raise ValueError("Segment positions must be a unique, one-dimensional integer list")
    if len(values) and np.any(np.diff(values) <= 0):
        raise ValueError("Segment positions must be in ascending order")
    return values


def read_inputs(output: Path, allow_incomplete: bool = False) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    inputs_path = output / "inputs.jsonl"
    inputs = load_jsonl(inputs_path)
    if not inputs:
        raise ValueError("inputs.jsonl is empty")
    ids = [row["input_id"] for row in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate input_id")
    rows, integrity = [], []
    max_sum_error = max_softmax_error = 0.0
    for item in inputs:
        input_id = item["input_id"]
        if Path(input_id).name != input_id:
            raise ValueError(f"Invalid input_id {input_id!r}")
        ids_array = np.asarray(item["input_ids"], dtype=np.int64)
        segments = {name: segment_positions(pos) for name, pos in item["segments"].items()}
        union = np.array(sorted({int(pos) for positions in segments.values() for pos in positions}), dtype=np.int64)
        if not len(union) or union[0] < 0 or union[-1] >= len(ids_array):
            raise ValueError(f"{input_id}: no readout positions or out-of-bounds segment")
        readout_path = output / "readouts" / f"{input_id}.npz"
        with np.load(readout_path, allow_pickle=False) as readout:
            positions = readout["positions"].astype(np.int64)
            token_ids = readout["token_ids"].astype(np.int64)
            probabilities = readout["probabilities"].astype(np.float64)
            logits = readout["logits"].astype(np.float64)
            if "roles" in readout and tuple(readout["roles"].tolist()) != ROLES:
                raise ValueError(f"{input_id}: stored role order differs from the six canonical roles")
            if not np.array_equal(positions, union):
                raise ValueError(f"{input_id}: positions differ from the exact union of segments")
            if not np.array_equal(token_ids, ids_array[positions]):
                raise ValueError(f"{input_id}: token IDs do not match prepared input")
            if probabilities.shape != (len(positions), 6) or logits.shape != probabilities.shape:
                raise ValueError(f"{input_id}: unexpected probe output shape")
            if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
                raise ValueError(f"{input_id}: non-finite probe output")
            if (probabilities < 0).any() or (probabilities > 1).any():
                raise ValueError(f"{input_id}: probability outside [0, 1]")
            sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
            normalized = np.exp(logits - logits.max(axis=1, keepdims=True))
            normalized /= normalized.sum(axis=1, keepdims=True)
            softmax_error = float(np.max(np.abs(normalized - probabilities)))
            max_sum_error = max(max_sum_error, sum_error)
            max_softmax_error = max(max_softmax_error, softmax_error)
            if sum_error > 2e-5 or softmax_error > 2e-4:
                raise ValueError(f"{input_id}: invalid normalization ({sum_error=}, {softmax_error=})")
            if "head_features" in readout:
                features = readout["head_features"]
                if features.shape != (len(positions), 1024) or not np.isfinite(features).all():
                    raise ValueError(f"{input_id}: invalid MLP16head features")
        item["_positions"] = positions
        item["_probabilities"] = probabilities
        item["_segments"] = segments
        digest = sha256(readout_path)
        record_path = output / "records" / f"{input_id}.json"
        record_sha_verified = False
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("status") != "complete":
                raise ValueError(f"{input_id}: readout record is not complete")
            raw_input = {key: value for key, value in item.items() if not key.startswith("_")}
            object_digest = hashlib.sha256(json.dumps(raw_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if record.get("signature", {}).get("input_sha256") != object_digest:
                raise ValueError(f"{input_id}: record input signature mismatch")
            claimed = next((record[key] for key in ("readout_sha256", "readout_sha", "readout_file_sha256") if key in record), None)
            if claimed is not None:
                if claimed != digest:
                    raise ValueError(f"{input_id}: readout SHA mismatch against record")
                record_sha_verified = True
        integrity.append({"input_id": input_id, "readout_sha256": digest, "readout_tokens": len(positions),
                          "record_sha_verified": record_sha_verified})
        for name, seg_positions in segments.items():
            if not len(seg_positions):
                continue
            index = np.searchsorted(positions, seg_positions)
            segment_probs = probabilities[index]
            row = {"input_id": input_id, "trace_id": item["trace_id"],
                   "source_split": item["source_split"], "source_suite": item.get("source_suite", item.get("suite", "unknown")),
                   "condition": item["condition"], "payload_type": item.get("payload_type") or "none",
                   "segment": name, "body_tokens": len(seg_positions), "body_start": int(seg_positions[0]),
                   "body_end_exclusive": int(seg_positions[-1]) + 1, "input_tokens": len(ids_array)}
            row.update(dict(zip(P_COLUMNS, segment_probs.mean(axis=0))))
            rows.append(row)
    frame = pd.DataFrame(rows)
    trace_meta = frame[["trace_id", "source_split", "source_suite"]].drop_duplicates()
    if trace_meta.trace_id.duplicated().any():
        raise ValueError("A trace has inconsistent split or suite metadata")
    if not set(trace_meta.source_split).issubset({"fit", "dev", "test"}):
        raise ValueError("source_split must be fit, dev, or test")
    counts = trace_meta.source_split.value_counts().to_dict()
    if not allow_incomplete and (len(trace_meta) != 57 or counts != {"fit": 40, "test": 9, "dev": 8}):
        raise ValueError(f"Expected 57 traces (fit40/dev8/test9); found {counts}. --allow-incomplete is for explicit pilot checks.")
    for trace_id in trace_meta.trace_id:
        variants = [item for item in inputs if item["trace_id"] == trace_id]
        inventory = {condition: [item.get("payload_type") for item in variants if item["condition"] == condition]
                     for condition in ("no_injection", "base-injection", "cot-forgery-injection", "native_cot")}
        if inventory["no_injection"] != [None] or inventory["native_cot"] != [None]:
            raise ValueError(f"{trace_id}: expected exactly one benign and one native reference input")
        if sorted(inventory["base-injection"]) != sorted(inventory["cot-forgery-injection"]):
            raise ValueError(f"{trace_id}: base and forged payload types must be paired")
        if len(set(inventory["base-injection"])) != len(inventory["base-injection"]):
            raise ValueError(f"{trace_id}: duplicate payload type")
        if not allow_incomplete and set(inventory["base-injection"]) != {"direct", "simple", "debugging", "backup", "documentation"}:
            raise ValueError(f"{trace_id}: expected all five frozen original payload pairs")
    validation = {"inputs_sha256": sha256(inputs_path), "input_count": len(inputs), "trace_count": len(trace_meta),
                  "trace_counts_by_split": counts, "max_probability_sum_error": max_sum_error,
                  "max_probability_softmax_error": max_softmax_error, "readouts": integrity}
    return inputs, frame, validation


def build_task_means(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for metric, (condition, segment) in METRICS.items():
        part = frame[(frame.condition == condition) & (frame.segment == segment)].copy()
        if part.empty:
            raise ValueError(f"Missing metric {metric}: {condition}/{segment}")
        # Stage 1 is already body-token averaging. Stage 2 enforces equal payload-type weight.
        groups = ["trace_id", "source_split", "source_suite", "payload_type"]
        type_means = part.groupby(groups, as_index=False)[P_COLUMNS].mean()
        task_means = type_means.groupby(groups[:-1], as_index=False)[P_COLUMNS].mean()
        task_means["payload_variants"] = type_means.groupby(groups[:-1]).size().to_numpy()
        task_means["metric"] = metric
        parts.append(task_means)
    tasks = pd.concat(parts, ignore_index=True)
    for trace_id, group in tasks.groupby("trace_id"):
        if set(group.metric) != set(METRICS):
            raise ValueError(f"{trace_id}: incomplete metric coverage")
    return tasks


def boot_indices(n: int, n_bootstrap: int = N_BOOTSTRAP) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(n_bootstrap, n))


def cohort_names(tasks: pd.DataFrame) -> dict[str, list[str]]:
    meta = tasks[["trace_id", "source_split"]].drop_duplicates().sort_values("trace_id")
    cohorts = {"test": meta.loc[meta.source_split == "test", "trace_id"].tolist(),
               "all": meta.trace_id.tolist(), "fit": meta.loc[meta.source_split == "fit", "trace_id"].tolist(),
               "dev": meta.loc[meta.source_split == "dev", "trace_id"].tolist()}
    return {name: ids for name, ids in cohorts.items() if ids}


def summarize(tasks: pd.DataFrame, cohorts: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, contrast_rows = [], []
    for cohort, ids in cohorts.items():
        sub = tasks[tasks.trace_id.isin(ids)]
        indexes = boot_indices(len(ids))
        for metric in METRICS:
            values = sub[sub.metric == metric].set_index("trace_id").loc[ids, P_COLUMNS].to_numpy()
            boot = values[indexes].mean(axis=1)
            for i, role in enumerate(ROLES):
                lower, upper = np.quantile(boot[:, i], [0.025, 0.975])
                rows.append({"cohort": cohort, "metric": metric, "role": role, "n_traces": len(ids),
                             "mean": values[:, i].mean(), "ci_low": lower, "ci_high": upper,
                             "task_sd": values[:, i].std(ddof=1) if len(ids) > 1 else np.nan})
        pivot = sub.pivot(index="trace_id", columns="metric", values="p_cot").loc[ids]
        for name, (positive, negative) in CONTRASTS.items():
            delta = (pivot[positive] - pivot[negative]).to_numpy()
            boot = delta[indexes].mean(axis=1)
            lower, upper = np.quantile(boot, [0.025, 0.975])
            contrast_rows.append({"cohort": cohort, "contrast": name, "positive_metric": positive,
                                  "negative_metric": negative, "n_traces": len(ids), "mean_delta": delta.mean(),
                                  "ci_low": lower, "ci_high": upper, "traces_positive": int((delta > 0).sum()),
                                  "traces_negative": int((delta < 0).sum()), "traces_zero": int((delta == 0).sum())})
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows)


def make_curves(inputs: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    grouped: dict[tuple[str, str, str], list[np.ndarray]] = {}
    metadata = {item["trace_id"]: item for item in inputs}
    for item in inputs:
        for metric, (condition, segment) in METRICS.items():
            if item["condition"] != condition:
                continue
            positions = item["_segments"].get(segment, np.array([], dtype=np.int64))
            if not len(positions):
                continue
            probs = item["_probabilities"][np.searchsorted(item["_positions"], positions), 3]
            curve = np.interp(GRID, np.linspace(0, 1, len(probs)), probs)
            key = (item["trace_id"], metric, item.get("payload_type") or "none")
            grouped.setdefault(key, []).append(curve)
    by_task: dict[tuple[str, str], list[np.ndarray]] = {}
    for (trace_id, metric, _), curves in grouped.items():
        by_task.setdefault((trace_id, metric), []).append(np.mean(curves, axis=0))
    result, rows = {}, []
    for (trace_id, metric), curves in sorted(by_task.items()):
        values = np.mean(curves, axis=0)
        result[(trace_id, metric)] = values
        for position, value in zip(GRID, values):
            rows.append({"trace_id": trace_id, "source_split": metadata[trace_id]["source_split"],
                         "source_suite": metadata[trace_id].get("source_suite", metadata[trace_id].get("suite", "unknown")),
                         "metric": metric, "relative_body_position": position, "p_cot": value})
    return result, pd.DataFrame(rows)


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 220, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cohort_label(cohort: str, n: int) -> str:
    return f"Held-out test traces (n = {n})" if cohort == "test" else f"All traces (n = {n}; includes probe-fit traces)"


def plot_cotness(tasks: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame,
                 cohort: str, ids: list[str], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    indexes = np.arange(len(METRIC_ORDER))
    subset = summary[(summary.cohort == cohort) & (summary.role == "cot")].set_index("metric")
    for i, metric in enumerate(METRIC_ORDER):
        raw = tasks[(tasks.metric == metric) & tasks.trace_id.isin(ids)].sort_values("trace_id").p_cot.to_numpy()
        jitter = np.random.default_rng(SEED + i).uniform(-0.10, 0.10, len(raw))
        ax.scatter(i + jitter, raw, color=COLORS[metric], alpha=0.40, s=16, linewidths=0, zorder=2)
        row = subset.loc[metric]
        ax.errorbar(i, row["mean"], yerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                    color=COLORS[metric], marker="D", markersize=6, capsize=4, lw=1.6, zorder=3)
    ax.set_xticks(indexes, [METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_ylim(-0.025, 1.025)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Mean probe CoT probability")
    ax.set_title("A  Role-confusion readout", loc="left")
    ax.grid(axis="y", alpha=0.18)
    ax = axes[1]
    pivot = tasks[tasks.trace_id.isin(ids)].pivot(index="trace_id", columns="metric", values="p_cot").loc[ids]
    delta = (pivot.payload_forged - pivot.payload_base).to_numpy()
    order = np.argsort(delta)
    for i, index in enumerate(order):
        ax.plot([0, 1], [pivot.payload_base.iloc[index], pivot.payload_forged.iloc[index]],
                color="#929292", alpha=0.55, linewidth=0.8, zorder=1)
    ax.scatter(np.zeros(len(ids)), pivot.payload_base, color=COLORS["direct_base"], s=20, zorder=2)
    ax.scatter(np.ones(len(ids)), pivot.payload_forged, color=COLORS["direct_forged"], s=20, zorder=2)
    ax.set_xticks([0, 1], ["Base\npayload", "Full forgery\npayload"])
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.025, 1.025)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("B  Paired full-payload readout", loc="left")
    ax.grid(axis="y", alpha=0.18)
    contrast = contrasts[(contrasts.cohort == cohort) & (contrasts.contrast == "forgery_payload_minus_base_payload")].iloc[0]
    ax.text(0.02, 0.98, f"Mean Δ = {100 * contrast.mean_delta:+.1f} pp\n95% CI [{100 * contrast.ci_low:+.1f}, {100 * contrast.ci_high:+.1f}]",
            transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    fig.suptitle(f"CoT forgery in AgentDojo tool outputs · {cohort_label(cohort, len(ids))}", y=1.04)
    fig.text(0.01, -0.03, "Fixed 16-head, six-role MLP probe. Points: trace means; diamonds/bars: mean and paired trace-bootstrap 95% CI.\n"
             "Payload types receive equal weight within each trace. Full-payload contrast changes text composition; native reasoning is a reference.", fontsize=8, color="#444444")
    fig.tight_layout()
    save_figure(fig, figures, f"{cohort}_cotness_summary")


def plot_roles(summary: pd.DataFrame, cohort: str, ids: list[str], figures: Path) -> None:
    subset = summary[summary.cohort == cohort]
    values = subset.pivot(index="metric", columns="role", values="mean").loc[list(METRIC_ORDER), list(ROLES)].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 3.65))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > 0.55 else "#222222")
    ax.set_xticks(np.arange(len(ROLES)), ROLE_LABELS)
    ax.set_yticks(np.arange(len(METRIC_ORDER)), [SHORT_LABELS[m] for m in METRIC_ORDER])
    ax.tick_params(length=0)
    ax.set_title(f"Six-role probability profile · {cohort_label(cohort, len(ids))}", loc="left", pad=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("Mean probe probability")
    fig.text(0.01, -0.015, "Body tokens → payload-type means → equal-weight trace means. Rows sum to 1. Fixed 16-head, six-role MLP probe.", fontsize=8, color="#444444")
    fig.tight_layout()
    save_figure(fig, figures, f"{cohort}_role_profile")


def plot_relative(curves: dict[tuple[str, str], np.ndarray], cohort: str, ids: list[str], figures: Path) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharey=True)
    selections = (("direct_base", "direct_forged"), ("benign_tool", "forged_reasoning", "native_reasoning"))
    rows = []
    indexes = boot_indices(len(ids))
    for ax, metrics in zip(axes, selections):
        for metric in metrics:
            values = np.asarray([curves[(trace_id, metric)] for trace_id in ids])
            boot = values[indexes].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975], axis=0)
            mean = values.mean(axis=0)
            ax.plot(GRID, mean, color=COLORS[metric], label=SHORT_LABELS[metric], linewidth=1.8)
            ax.fill_between(GRID, low, high, color=COLORS[metric], alpha=0.15, linewidth=0)
            for x, y, lower, upper in zip(GRID, mean, low, high):
                rows.append({"cohort": cohort, "metric": metric, "n_traces": len(ids),
                             "relative_body_position": x, "mean": y, "ci_low": lower, "ci_high": upper})
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel("Relative position within body (0 = first, 1 = last token)")
        ax.grid(alpha=0.18)
        ax.legend(loc="best", frameon=False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Probe CoT probability")
    axes[0].set_title("A  Instruction-body profiles", loc="left")
    axes[1].set_title("B  Reasoning and tool-body references", loc="left")
    fig.suptitle(f"Token-position readout · {cohort_label(cohort, len(ids))}", y=1.04)
    fig.text(0.01, -0.025, "Each body is interpolated to 51 relative positions, then payload types and traces receive equal weight.\n"
             "Bands: pointwise 95% paired trace-bootstrap intervals; these curves do not imply token alignment across different texts.", fontsize=8, color="#444444")
    fig.tight_layout()
    save_figure(fig, figures, f"{cohort}_relative_position")
    return rows


def plot_task_pairs(tasks: pd.DataFrame, cohort: str, ids: list[str], figures: Path) -> None:
    pivot = tasks[tasks.trace_id.isin(ids)].pivot(index="trace_id", columns="metric", values="p_cot")
    delta = (pivot.payload_forged - pivot.payload_base).sort_values()
    fig, ax = plt.subplots(figsize=(8.6, max(3.4, 0.23 * len(ids) + 1.5)))
    meta = tasks.set_index("trace_id")
    for y, trace_id in enumerate(delta.index):
        a, b = pivot.loc[trace_id, ["payload_base", "payload_forged"]]
        ax.plot([a, b], [y, y], color="#B0B0B0", linewidth=1.5)
        ax.scatter(a, y, color=COLORS["direct_base"], s=27, zorder=2)
        ax.scatter(b, y, color=COLORS["direct_forged"], s=27, zorder=2)
    ax.set_yticks(np.arange(len(ids)), delta.index)
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Mean full-payload CoT probability")
    ax.set_title(f"Per-trace paired readouts · {cohort_label(cohort, len(ids))}", loc="left", pad=12)
    ax.scatter([], [], color=COLORS["direct_base"], label="Base payload")
    ax.scatter([], [], color=COLORS["direct_forged"], label="Full forgery payload")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.invert_yaxis()
    fig.text(0.01, -0.01, "Sorted by the within-trace shift. Forgery payload types are averaged equally within each trace.", fontsize=8, color="#444444")
    fig.tight_layout()
    save_figure(fig, figures, f"{cohort}_paired_tasks")


def plot_trajectory(inputs: list[dict[str, Any]], tasks: pd.DataFrame, ids: list[str], figures: Path) -> dict[str, Any]:
    pivot = tasks[tasks.trace_id.isin(ids)].pivot(index="trace_id", columns="metric", values="p_cot")
    delta = (pivot.payload_forged - pivot.payload_base).sort_values(kind="stable")
    trace_id = delta.index[(len(delta) - 1) // 2]
    matching = [item for item in inputs if item["trace_id"] == trace_id]
    forged = sorted((item for item in matching if item["condition"] == "cot-forgery-injection"), key=lambda x: str(x.get("payload_type")))[0]
    base = next(item for item in matching if item["condition"] == "base-injection" and item.get("payload_type") == forged.get("payload_type"))
    native = next(item for item in matching if item["condition"] == "native_cot")
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 6.4), sharey=True)
    sample_rows = []
    for ax, item, segment, title in zip(axes, (base, forged, native), ("payload_all", "payload_all", "native_cot"),
                                       ("A  Direct injection payload", "B  Forged reasoning + instruction", "C  Native reasoning reference")):
        positions = item["_segments"][segment]
        probs = item["_probabilities"][np.searchsorted(item["_positions"], positions)]
        x = np.arange(len(positions))
        for i, role in enumerate(ROLES):
            ax.plot(x, probs[:, i], label=ROLE_LABELS[i], color=ROLE_COLORS[i],
                    linewidth=1.25 if role in {"cot", "tool"} else 0.65, alpha=1 if role in {"cot", "tool"} else 0.65)
        for name, color in (("forged_cot", COLORS["forged_reasoning"]), ("direct_instruction", COLORS["direct_base"])):
            selected = item["_segments"].get(name, np.array([], dtype=np.int64))
            selected = selected[np.isin(selected, positions)]
            if len(selected):
                relative = np.searchsorted(positions, selected)
                ax.axvspan(relative[0] - 0.5, relative[-1] + 0.5, color=color, alpha=0.08)
                label = "Forged reasoning" if name == "forged_cot" else "Attack instruction"
                ax.text((relative[0] + relative[-1]) / 2, 0.98, label, ha="center", va="top", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.5, max(len(positions) - 0.5, 0.5))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylabel("Probability")
        ax.grid(axis="y", alpha=0.15)
        for relative, absolute, probability in zip(x, positions, probs):
            sample_rows.append({"input_id": item["input_id"], "trace_id": trace_id, "condition": item["condition"],
                                "payload_type": item.get("payload_type"), "segment": segment,
                                "relative_token_index": int(relative), "absolute_token_index": int(absolute),
                                **dict(zip(P_COLUMNS, probability))})
    axes[-1].set_xlabel("Token index within the displayed body")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.4), ncol=6, frameon=False)
    fig.suptitle(f"Example token-level role readout · {trace_id}", y=1.025)
    fig.text(0.01, -0.015, f"Deterministic selection: held-out trace at lower median paired payload shift; alphabetically first forgery type ({forged.get('payload_type')}).\n"
             "Curves show raw per-token probabilities. Each panel has its own token axis; native reasoning has different text and causal context.", fontsize=8, color="#444444")
    fig.tight_layout(h_pad=1.2)
    save_figure(fig, figures, "test_example_token_trajectory")
    return {"selection_rule": "lower-median task-level paired full-payload shift; alphabetically first forged payload type",
            "trace_id": trace_id, "forged_payload_type": forged.get("payload_type"),
            "paired_delta": delta.loc[trace_id], "input_ids": [base["input_id"], forged["input_id"], native["input_id"]],
            "token_rows": sample_rows}


def suite_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    # Small suite subgroups are descriptive, with no subgroup significance claims.
    result = tasks.groupby(["source_split", "source_suite", "metric"], as_index=False).agg(
        n_traces=("trace_id", "nunique"), **{column: (column, "mean") for column in P_COLUMNS})
    return result


def summarize_payload_types(frame: pd.DataFrame, cohorts: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for cohort, ids in cohorts.items():
        indexes = boot_indices(len(ids))
        for metric in ("direct_base", "direct_forged", "forged_reasoning", "payload_base", "payload_forged"):
            condition, segment = METRICS[metric]
            sub = frame[(frame.condition == condition) & (frame.segment == segment) & frame.trace_id.isin(ids)]
            for payload_type, group in sub.groupby("payload_type"):
                values = group.groupby("trace_id")[P_COLUMNS].mean().loc[ids].to_numpy()
                boot = values[indexes].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975], axis=0)
                for i, role in enumerate(ROLES):
                    rows.append({"cohort": cohort, "payload_type": payload_type, "metric": metric, "role": role,
                                 "n_traces": len(ids), "mean": values[:, i].mean(), "ci_low": low[i], "ci_high": high[i]})
    return pd.DataFrame(rows)


def plot_payload_types(frame: pd.DataFrame, summary: pd.DataFrame, cohort: str, ids: list[str], figures: Path) -> None:
    sub = summary[(summary.cohort == cohort) & (summary.role == "cot")]
    types = [name for name in ("direct", "simple", "debugging", "backup", "documentation") if name in set(sub.payload_type)]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharey=True)
    for ax, metrics in zip(axes, (("payload_base", "payload_forged"), ("direct_base", "direct_forged", "forged_reasoning"))):
        offsets = np.linspace(-0.15, 0.15, len(metrics))
        for offset, metric in zip(offsets, metrics):
            values = sub[sub.metric == metric].set_index("payload_type").loc[types]
            color = COLORS.get(metric, COLORS["direct_base"] if metric == "payload_base" else COLORS["direct_forged"])
            ax.errorbar(np.arange(len(types)) + offset, values["mean"],
                        yerr=np.vstack([values["mean"]-values.ci_low, values.ci_high-values["mean"]]),
                        color=color, marker="o", linewidth=0, elinewidth=1.4, capsize=3,
                        label=SHORT_LABELS[metric], markersize=4)
        ax.set_xticks(np.arange(len(types)), [name.title() for name in types], rotation=20, ha="right")
        ax.set_xlim(-0.5, len(types)-0.5)
        ax.set_ylim(-0.025, 1.025)
        ax.grid(axis="y", alpha=0.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Mean probe CoT probability")
    axes[0].set_title("A  Full payloads", loc="left")
    axes[1].set_title("B  Instruction and reasoning bodies", loc="left")
    fig.suptitle(f"Five original payload types · {cohort_label(cohort, len(ids))}", y=1.04)
    fig.text(0.01, -0.03, "Points/bars: equal-weight trace means and paired trace-bootstrap 95% intervals for each frozen payload type.\n"
             "Original base/forgery pairs can differ in instruction text or tokenization. These are descriptive readouts, not pure-CoT causal effects.", fontsize=8, color="#444444")
    fig.tight_layout()
    save_figure(fig, figures, f"{cohort}_payload_types")


def causal_prefix_diagnostics(inputs: list[dict[str, Any]], cohorts: dict[str, list[str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Measure numeric drift when a future suffix changes the prefill shape.

    Prefix-invariant probabilities need not be bitwise identical under the
    native MXFP4 kernels. Such differences are implementation drift, not an
    influence of future tokens on a causal model.
    """
    rows = []
    for trace_id in sorted({item["trace_id"] for item in inputs}):
        group = [item for item in inputs if item["trace_id"] == trace_id]
        baseline = next(item for item in group if item["condition"] == "no_injection")
        lookup = {(item["condition"], item.get("payload_type")): item for item in group}
        pairs = []
        for item in group:
            if item["condition"] in {"base-injection", "cot-forgery-injection"}:
                pairs.append((baseline, item, "original_tool", f"benign_vs_{item['condition']}"))
            if item["condition"] == "cot-forgery-injection":
                paired = lookup[("base-injection", item.get("payload_type"))]
                pairs.append((paired, item, "original_tool", "base_vs_forgery"))
                pairs.append((paired, item, "direct_instruction", "base_vs_forgery"))
        for a, b, segment, comparison in pairs:
            pa, pb = a["_segments"][segment], b["_segments"][segment]
            tokens_a = np.asarray(a["input_ids"], dtype=np.int64)[pa]
            tokens_b = np.asarray(b["input_ids"], dtype=np.int64)[pb]
            identical_tokens = np.array_equal(tokens_a, tokens_b)
            identical_prefix = identical_tokens and np.array_equal(pa, pb) and a["input_ids"][:int(pa[-1])+1] == b["input_ids"][:int(pb[-1])+1]
            absolute_delta = mean_signed_cot_delta = mean_abs_cot_delta = maximum = np.nan
            if identical_tokens:
                probs_a = a["_probabilities"][np.searchsorted(a["_positions"], pa)]
                probs_b = b["_probabilities"][np.searchsorted(b["_positions"], pb)]
                delta = probs_b - probs_a
                absolute_delta = float(np.abs(delta).mean())
                maximum = float(np.abs(delta).max())
                mean_abs_cot_delta = float(np.abs(delta[:, 3]).mean())
                mean_signed_cot_delta = float(delta[:, 3].mean())
            rows.append({"trace_id": trace_id, "source_split": a["source_split"], "source_suite": a.get("source_suite", a.get("suite", "unknown")),
                         "comparison": comparison, "segment": segment, "payload_type": b.get("payload_type"),
                         "input_a": a["input_id"], "input_b": b["input_id"], "identical_segment_token_ids": identical_tokens,
                         "body_tokens_a": len(pa), "body_tokens_b": len(pb),
                         "compared_tokens": len(pa) if identical_tokens else 0,
                         "identical_causal_prefix": identical_prefix, "max_abs_probability_delta": maximum,
                         "mean_abs_probability_delta": absolute_delta, "mean_abs_cot_delta": mean_abs_cot_delta,
                         "mean_signed_cot_delta": mean_signed_cot_delta})
            if segment == "direct_instruction" and not identical_prefix:
                ids_a, ids_b = np.asarray(a["input_ids"]), np.asarray(b["input_ids"])
                size = min(len(ids_a), len(ids_b))
                differences = np.flatnonzero(ids_a[:size] != ids_b[:size])
                common_end = int(differences[0]) if len(differences) else size
                shared = pa[(pa < common_end) & np.isin(pa, pb)]
                if len(shared):
                    va = a["_probabilities"][np.searchsorted(a["_positions"], shared)]
                    vb = b["_probabilities"][np.searchsorted(b["_positions"], shared)]
                    drift = vb - va
                    rows.append({"trace_id": trace_id, "source_split": a["source_split"],
                                 "source_suite": a.get("source_suite", a.get("suite", "unknown")),
                                 "comparison": comparison, "segment": "direct_instruction_common_prefix", "payload_type": b.get("payload_type"),
                                 "input_a": a["input_id"], "input_b": b["input_id"],
                                 "identical_segment_token_ids": True, "identical_causal_prefix": True,
                                 "body_tokens_a": len(pa), "body_tokens_b": len(pb), "compared_tokens": len(shared),
                                 "max_abs_probability_delta": float(np.abs(drift).max()),
                                 "mean_abs_probability_delta": float(np.abs(drift).mean()),
                                 "mean_abs_cot_delta": float(np.abs(drift[:, 3]).mean()),
                                 "mean_signed_cot_delta": float(drift[:, 3].mean())})
    frame = pd.DataFrame(rows)
    summaries = []
    for cohort, ids in cohorts.items():
        subset = frame[frame.trace_id.isin(ids)]
        for (comparison, segment), group in subset.groupby(["comparison", "segment"]):
            eligible = group[group.identical_causal_prefix]
            task_equal = eligible.groupby("trace_id")[["mean_abs_probability_delta", "mean_abs_cot_delta", "mean_signed_cot_delta"]].mean()
            summaries.append({"cohort": cohort, "comparison": comparison, "segment": segment,
                              "pairs": len(group), "identical_causal_prefix_pairs": len(eligible),
                              "changed_segment_token_pairs": int((~group.identical_segment_token_ids).sum()),
                              "max_abs_probability_delta": float(eligible.max_abs_probability_delta.max()) if len(eligible) else None,
                              "task_equal_mean_abs_probability_delta": float(task_equal.mean_abs_probability_delta.mean()) if len(task_equal) else None,
                              "task_equal_mean_abs_cot_delta": float(task_equal.mean_abs_cot_delta.mean()) if len(task_equal) else None,
                              "task_equal_mean_signed_cot_delta": float(task_equal.mean_signed_cot_delta.mean()) if len(task_equal) else None,
                              "interpretation": "Identical-prefix drift measures native prefill-shape numerical sensitivity, not a future-token effect."})
    return frame, summaries


def analyze_behavior(output: Path, figures: Path, tables: Path, expected_test_ids: list[str] | None = None) -> dict[str, Any] | None:
    path = output / "behavior.jsonl"
    if not path.exists():
        return None
    records = load_jsonl(path)
    if not records:
        return None
    rows = []
    for item in records:
        success = item.get("exact_final_success", item.get("marker_in_final", item.get("marker_success", item.get("success"))))
        if success is None:
            raise ValueError("Behavior rows require exact_final_success or marker_in_final")
        if "marker" in item and "final_text" in item:
            exact = item["final_text"].strip() == item["marker"]
            if bool(success) != exact or bool(item["contains_final_success"]) != (item["marker"] in item["final_text"]):
                raise ValueError(f"{item['trace_id']}/{item['condition']}: final-answer marker scoring mismatch")
        rows.append({"trace_id": item["trace_id"], "condition": item["condition"],
                     "source_split": item.get("source_split", "test"), "marker_in_final": bool(success),
                     "contains_final_success": bool(item.get("contains_final_success", success)),
                     "contains_anywhere": bool(item.get("contains_anywhere", success)),
                     "stop_reason": item.get("stop_reason", "unknown"),
                     "truncated": bool(item.get("length_censored", item.get("truncated", item.get("is_truncated", False)))),
                     "final_present": bool(item.get("final_present", item.get("has_final", bool(item.get("final_text", ""))))),
                     "generated_tokens": item.get("generated_tokens", item.get("output_tokens", np.nan))})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["trace_id", "condition"]).any():
        raise ValueError("Duplicate behavior trace/condition rows")
    if expected_test_ids is not None:
        if set(frame.trace_id) != set(expected_test_ids) or set(frame.source_split) != {"test"}:
            raise ValueError("Controlled continuations must cover all nine held-out test traces only")
        if any(set(group.condition) != {"control", "direct", "forged"} for _, group in frame.groupby("trace_id")):
            raise ValueError("Each held-out trace must have control, direct, and forged continuations")
    frame.to_csv(tables / "controlled-continuation-traces.csv", index=False)
    condition_order = [condition for condition in ("control", "direct", "forged") if condition in set(frame.condition)]
    summary_rows = []
    for condition in condition_order:
        group = frame[frame.condition == condition]
        ids = sorted(group.trace_id)
        values = group.set_index("trace_id").loc[ids, "marker_in_final"].astype(float).to_numpy()
        # Wilson intervals remain informative at 0/n or n/n; the paired
        # condition contrasts below retain trace-bootstrap intervals.
        z = 1.959963984540054
        n = len(values)
        rate = values.mean()
        center = (rate + z*z/(2*n)) / (1 + z*z/n)
        radius = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
        lower, upper = max(0.0, center-radius), min(1.0, center+radius)
        summary_rows.append({"condition": condition, "n_traces": len(group), "marker_successes": int(group.marker_in_final.sum()),
                             "marker_rate": values.mean(), "ci_low": lower, "ci_high": upper, "rate_interval": "Wilson 95%",
                             "contains_final_successes": int(group.contains_final_success.sum()),
                             "contains_anywhere": int(group.contains_anywhere.sum()),
                             "truncated": int(group.truncated.sum()), "final_absent": int((~group.final_present).sum()),
                             "tool_call_stops": int((group.stop_reason == "tool_call").sum())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "controlled-continuation-summary.csv", index=False)
    pivot = frame.pivot(index="trace_id", columns="condition", values="marker_in_final")
    contrast_rows = []
    for positive, negative in (("forged", "direct"), ("direct", "control"), ("forged", "control")):
        if positive not in pivot or negative not in pivot:
            continue
        pairs = pivot[[positive, negative]].dropna()
        delta = pairs[positive].astype(float).to_numpy() - pairs[negative].astype(float).to_numpy()
        low, high = np.quantile(delta[boot_indices(len(delta))].mean(axis=1), [0.025, 0.975])
        contrast_rows.append({"contrast": f"{positive}_minus_{negative}", "n_pairs": len(delta),
                              "mean_delta": delta.mean(), "ci_low": low, "ci_high": high})
    pd.DataFrame(contrast_rows).to_csv(tables / "controlled-continuation-paired-contrasts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), gridspec_kw={"width_ratios": [1, 1.45]})
    colors = {"control": "#777777", "direct": "#0072B2", "forged": "#D55E00"}
    for i, condition in enumerate(condition_order):
        row = summary[summary.condition == condition].iloc[0]
        axes[0].bar(i, row.marker_rate, color=colors.get(condition, "#777777"), alpha=0.8, width=0.6)
        axes[0].errorbar(i, row.marker_rate, yerr=[[row.marker_rate-row.ci_low], [row.ci_high-row.marker_rate]], color="#333333", capsize=4)
        axes[0].text(i, min(row.ci_high + 0.035, 1.10), f"{int(row.marker_successes)}/{int(row.n_traces)}", ha="center")
    axes[0].set_xticks(np.arange(len(condition_order)), [x.title() for x in condition_order])
    axes[0].set_ylim(0, 1.16)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Exact marker in completed final answer")
    axes[0].set_title("A  Controlled marker outcome", loc="left")
    values = pivot.reindex(columns=condition_order).astype(float)
    axes[1].imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1].set_yticks(np.arange(len(values)), values.index)
    axes[1].set_xticks(np.arange(len(condition_order)), [x.title() for x in condition_order])
    axes[1].tick_params(length=0)
    axes[1].set_title("B  Paired per-trace marker outcomes", loc="left")
    for y in range(len(values)):
        for x in range(len(condition_order)):
            value = values.iloc[y, x]
            record = frame[(frame.trace_id == values.index[y]) & (frame.condition == condition_order[x])].iloc[0]
            label = "yes" if value == 1 else "cens." if record.truncated else "tool" if record.stop_reason == "tool_call" else "no" if value == 0 else "missing"
            axes[1].text(x, y, label, ha="center", va="center", color="white" if value == 1 else "#333333", fontsize=8)
    fig.suptitle("Supplement: controlled continuation (held-out test traces)", y=1.04)
    fig.text(0.01, -0.04, "Success: exact marker only, in a completed ordinary final answer. Tool = tool-call stop; cens. = length cap (kept in denominator).\n"
             "Bars: Wilson 95% intervals. Paired differences use a trace bootstrap in the CSV. This endpoint is not AgentDojo attack success rate.", fontsize=8, color="#444444")
    fig.tight_layout()
    save_figure(fig, figures, "test_controlled_continuation")
    return {"file_sha256": sha256(path), "record_count": len(records), "summary": summary.to_dict(orient="records"),
            "paired_contrasts": contrast_rows, "endpoint": "exact marker and nothing else in a completed final answer; not AgentDojo ASR"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow explicit pilot/synthetic checks without the full 57-trace design")
    args = parser.parse_args()
    output = args.output.resolve()
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    inputs, frame, validation = read_inputs(output, args.allow_incomplete)
    tasks = build_task_means(frame)
    cohorts = cohort_names(tasks)
    summary, contrasts = summarize(tasks, cohorts)
    payload_summary = summarize_payload_types(frame, cohorts)
    curves, curve_tasks = make_curves(inputs)
    prefix_diagnostics, prefix_summary = causal_prefix_diagnostics(inputs, cohorts)
    prefix_diagnostics.to_csv(tables / "causal-prefix-numerical-diagnostics.csv", index=False)
    pd.DataFrame(prefix_summary).to_csv(tables / "causal-prefix-numerical-summary.csv", index=False)
    frame.to_csv(tables / "segment-means.csv", index=False)
    tasks.to_csv(tables / "task-means.csv", index=False)
    summary.to_csv(tables / "role-summary-by-split.csv", index=False)
    contrasts.to_csv(tables / "paired-cotness-contrasts-by-split.csv", index=False)
    payload_summary.to_csv(tables / "payload-type-summary-by-split.csv", index=False)
    curve_tasks.to_csv(tables / "relative-position-task-means.csv", index=False)
    suite_summary(tasks).to_csv(tables / "suite-descriptive-means.csv", index=False)
    pivot = tasks.pivot(index=["trace_id", "source_split", "source_suite"], columns="metric", values="p_cot").reset_index()
    for name, (positive, negative) in CONTRASTS.items():
        pivot[name] = pivot[positive] - pivot[negative]
    pivot.to_csv(tables / "paired-task-cotness.csv", index=False)
    style()
    curve_summary = []
    for cohort in ("test", "all"):
        if cohort not in cohorts:
            continue
        ids = cohorts[cohort]
        plot_cotness(tasks, summary, contrasts, cohort, ids, figures)
        plot_roles(summary, cohort, ids, figures)
        curve_summary.extend(plot_relative(curves, cohort, ids, figures))
        plot_task_pairs(tasks, cohort, ids, figures)
        plot_payload_types(frame, payload_summary, cohort, ids, figures)
    pd.DataFrame(curve_summary).to_csv(tables / "relative-position-summary.csv", index=False)
    example = None
    if "test" in cohorts:
        example = plot_trajectory(inputs, tasks, cohorts["test"], figures)
        pd.DataFrame(example.pop("token_rows")).to_csv(tables / "example-token-trajectory.csv", index=False)
    behavior = analyze_behavior(output, figures, tables, None if args.allow_incomplete else cohorts.get("test"))
    write_json(output / "analysis-validation.json", validation)
    result = {"analysis_version": 1, "analysis_script_sha256": sha256(Path(__file__)),
              "role_order": ROLES, "probe": "fixed six-class MLP over 16 selected attention heads, supplied by user",
              "primary_cohort": "test", "secondary_cohort": "all (includes probe-fit traces)",
              "cohort_counts": {name: len(ids) for name, ids in cohorts.items()},
              "aggregation": "body tokens -> equal payload-type means within trace -> equal trace means",
              "bootstrap": {"unit": "trace", "n_resamples": N_BOOTSTRAP, "seed": SEED,
                            "interval": "pointwise 95% paired percentile; identical trace draws within each cohort"},
              "role_summary": summary.to_dict(orient="records"), "paired_cotness_contrasts": contrasts.to_dict(orient="records"),
              "causal_prefix_numerical_diagnostics": prefix_summary,
              "example": example, "controlled_continuation": behavior,
              "limitations": ["Transfer reproduction: model, six-role probe, and benign AgentDojo contexts differ from the original five-class layerwise probe setting.",
                              "Native reasoning differs in text, wrapper, and causal context; its difference is descriptive, not a matched causal contrast.",
                              "All57 includes traces from probe fit/dev splits. TEST9 is held out from this MLP probe's fitting and dev-based selection, but historical head discovery predates this split; the full feature-selection pipeline is not claimed independent of TEST9.",
                              "Relative-position curves align body fractions, not matching semantic tokens across texts.",
                              "The original five payload pairs append reasoning after the instruction. Instruction bodies differ through wording, formatting, or boundary tokenization; their full-body shifts are descriptive, not causal evidence of reasoning adoption.",
                              "Identical-prefix numerical drift can arise from native MXFP4 prefill-shape changes and is quantified separately.",
                              "Probe CoT probability is a role readout, not a calibrated probability of behavioral attack success."]}
    write_json(output / "analysis-summary.json", result)
    print(json.dumps({"output": str(output), "inputs": len(inputs), "cohorts": result["cohort_counts"],
                      "figures_png": len(list(figures.glob("*.png"))), "behavior_records": behavior["record_count"] if behavior else 0}))


if __name__ == "__main__":
    main()
