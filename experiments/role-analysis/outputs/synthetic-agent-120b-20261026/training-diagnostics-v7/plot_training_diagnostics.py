"""Independent DEV-only visualization of exported epoch scalars; CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

CANDIDATES = ("linear-seed17", "mlp128-seed17", "mlp128-seed29", "mlp128-seed43",
              "mlp256-seed17", "mlp256-seed29", "mlp256-seed43")
WINNER = "mlp128-seed17"
COLORS = ("#666666", "#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00", "#56B4E9")
STYLES = ("--", "-", "--", ":", "-.", "--", ":")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only rendering requires an empty CUDA mask")
    source = read(args.source_manifest)
    for entry in source["inputs"]:
        path = (args.source_root / entry["relative_path"]).resolve()
        require(path.is_relative_to(args.source_root.resolve()), "Source path escapes the archive")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Exported source bytes changed")
    selection = read(args.source_root / "formal/selection.json")
    frozen = read(args.source_root / "formal/freeze.json")
    require(selection["candidate_id"] == frozen["candidate_id"] == WINNER, "Frozen DEV winner differs")
    require(frozen["selection_sha256"] == sha(args.source_root / "formal/selection.json"), "Frozen selection SHA differs")
    data, summaries = {}, {}
    for candidate in CANDIDATES:
        base = args.source_root / "formal/candidates" / candidate
        curve, complete = read(base / "curve-scalars.json"), read(base / "complete.json")
        require(curve["format_version"] == "synthetic-derived-epoch-scalars-v1" and
                curve["derived_artifact"] is True and curve["not_the_raw_history"] is True and
                curve["candidate_id"] == complete["candidate_id"] == candidate, "Derived scalar identity differs")
        require(curve["source_history_sha256"] == complete["history_sha256"] and
                curve["raw_history_retained_remote"] is True, "Original history provenance differs")
        epochs = curve["epochs"]
        require(curve["epochs_run"] == complete["epochs"] == len(epochs) and
                [e["epoch"] for e in epochs] == list(range(1, len(epochs) + 1)), "Epoch sequence is incomplete")
        best = int(curve["best_epoch"])
        require(best == complete["best_epoch"] and 1 <= best <= len(epochs), "Recorded best epoch differs")
        losses = [float(e["train_loss_for_processed_batches"]) for e in epochs]
        scores = [100 * float(e["dev"]["structure_equal_macro_f1"]) for e in epochs]
        require(all(math.isfinite(v) and v >= 0 for v in losses) and
                all(math.isfinite(v) and 0 <= v <= 100 for v in scores), "Nonfinite or invalid scalar")
        require(abs(scores[best - 1] / 100 - complete["dev"]["structure_equal_macro_f1"]) <= 1e-12,
                "Best-epoch DEV scalar differs from candidate commit")
        data[candidate] = (list(range(1, len(epochs) + 1)), losses, scores, best)
        summaries[candidate] = {"epochs_plotted": len(epochs), "first_epoch": 1, "last_epoch": len(epochs),
            "recorded_best_epoch": best, "best_epoch_recorded_loss": losses[best - 1],
            "best_epoch_dev_structure_equal_macro_f1_percent": scores[best - 1],
            "source_history_path": curve["source_history_path"], "source_history_sha256": curve["source_history_sha256"],
            "source_history_bytes": curve["source_history_bytes"], "raw_history_read_or_modified": False}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 7.5))
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.32, wspace=.22)
    handles, labels = [], []
    for candidate, color, style in zip(CANDIDATES, COLORS, STYLES):
        x, loss, score, best = data[candidate]
        selected = candidate == WINNER
        width, zorder = (3.0, 8) if selected else (1.55, 3)
        for axis, y in zip(axes, (loss, score)):
            line, = axis.plot(x, y, color=color, linestyle=style, linewidth=width,
                              alpha=1 if selected else .86, zorder=zorder)
            axis.scatter([best], [y[best - 1]], marker="*", s=190 if selected else 90,
                         color=color, edgecolor="#111111", linewidth=.55, zorder=10 if selected else 6)
        handles.append(line)
        labels.append(f"{candidate}{' [frozen winner]' if selected else ''} | best {best}, stop {x[-1]}")
    xmax = max(v[0][-1] for v in data.values())
    lossmax = max(max(v[1]) for v in data.values())
    axes[0].set_ylim(-.006, lossmax * 1.07)
    axes[0].set_yticks([0, .025, .05, .075, .10, .125, .15, .175])
    axes[0].set_ylabel("Recorded train_loss_for_processed_batches\n(cross-entropy; processed batches only)")
    axes[0].set_title("A. Recorded training loss", loc="left", fontweight="bold")
    values = [v for item in data.values() for v in item[2]]
    lower = math.floor((min(values) - .035) * 100) / 100
    upper = math.ceil((max(values) + .035) * 100) / 100
    axes[1].set_ylim(lower, upper)
    axes[1].yaxis.set_major_locator(MultipleLocator(.1))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].set_ylabel("DEV structure-equal macro-F1 (%)")
    axes[1].set_title(f"B. DEV selection metric — ZOOMED Y-AXIS ({lower:.2f}–{upper:.2f}%)",
                      loc="left", fontweight="bold", fontsize=10.5)
    winner = summaries[WINNER]
    for axis in axes:
        axis.set_xlim(.4, xmax + .8)
        axis.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, xmax])
        axis.set_xlabel("Recorded epoch (all epochs shown; runs stop where recorded)")
        axis.grid(True, linewidth=.5, alpha=.23)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.suptitle("Formal training diagnostics | seven predeclared candidates", y=.96, fontsize=16, fontweight="bold")
    fig.text(.5, .895, "Fixed DEV-selected winner: mlp128-seed17   |   Stars = recorded best_epoch   |   No TEST/external metrics used",
             ha="center", fontsize=10.5)
    fig.text(.5, .855, f"Frozen winner export: recorded epoch {winner['recorded_best_epoch']}; "
             f"DEV structure-equal macro-F1 = {winner['best_epoch_dev_structure_equal_macro_f1_percent']:.5f}%",
             ha="center", fontsize=10, color=COLORS[1])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .135), ncol=4,
               fontsize=8.5, frameon=False, columnspacing=1.4, handlelength=3)
    note = ("Loss is the recorded mean over batches processed in that process; it is not an empirical risk over unique FIT examples. "
            "Derived epoch scalars preserve original raw-history SHA provenance. The DEV ordinate is deliberately zoomed; selection is unchanged.")
    fig.text(.5, .045, textwrap.fill(note, 155), ha="center", va="bottom", fontsize=9, linespacing=1.45)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for extension in ("png", "pdf"):
        path = args.output_dir / ("training-loss-and-dev-zoom." + extension)
        fig.savefig(path, dpi=220, metadata={"Creator": "Independent scalar-only training diagnostic"})
        files[extension] = {"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
    plt.close(fig)
    for entry in source["inputs"]:
        require(sha(args.source_root / entry["relative_path"]) == entry["sha256"], "Source changed during plotting")
    provenance = {"format_version": "synthetic-training-diagnostics-v1", "supplement_only": True,
        "fixed_candidate_order": list(CANDIDATES), "frozen_dev_winner": WINNER, "candidate_summaries": summaries,
        "metric": "DEV structure_equal_macro_f1 multiplied by 100", "dev_y_axis": {"zoomed": True, "limits_percent": [lower, upper]},
        "loss_field": "train_loss_for_processed_batches", "loss_semantics": note,
        "all_recorded_epochs_plotted": True, "best_markers_from_committed_metadata": True,
        "test_or_external_metrics_read": False, "raw_history_read_or_modified": False, "original_final_bundle_modified": False,
        "source_manifest_sha256": sha(args.source_manifest), "script_sha256": sha(__file__),
        "source_inputs": source["inputs"], "original_training_plots_retained": source["original_training_plots_retained"],
        "render_environment": {"matplotlib": matplotlib.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "files": files}
    (args.output_dir / "figure-manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"rendered": list(files), "candidate_count": len(CANDIDATES), "total_epoch_points": sum(v["epochs_plotted"] for v in summaries.values()),
                      "dev_limits_percent": [lower, upper], "script_sha256": provenance["script_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
