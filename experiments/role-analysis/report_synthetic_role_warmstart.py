"""Build the final Chinese report from frozen warm-start experiment artifacts.

No language model, raw article body, raw feature array, or private hash ledger
is copied to the report.  The self-test writes only under a temporary directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

import synthetic_agent_classifier as classifier
from synthetic_role_warmstart_eval import validate_freeze

ROLES = tuple(classifier.ROLES)
CANDIDATES = ("warm-all", "warm-output", "scratch")
FEATURE_KEYS = ("roles", "heads", "feature_dim", "feature_position", "model_binding")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path, sources=None):
    path = Path(path)
    if sources is not None:
        sources[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return json.loads(path.read_text())


def read_csv(path, sources=None):
    path = Path(path)
    if sources is not None:
        sources[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def find_metadata(root, name, *, required=True):
    root = Path(root)
    candidates = [root / name, root / "data" / name, root / "run" / name]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        require(not required, "Missing resource/data metadata: " + name)
        return None
    require(len({sha256(p) for p in matches}) == 1, "Conflicting metadata copies: " + name)
    return matches[0]


def pct(value, digits=4):
    return "N/A" if value is None else f"{100 * value:.{digits}f}%"


def number(value):
    return "N/A" if value is None else f"{int(value):,}"


def local_time(value):
    if value is None:
        return "N/A"
    return datetime.fromtimestamp(value, ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(map(clean, headers)) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(map(clean, row)) + " |" for row in rows]]


def check_candidate(run, name, training, sources):
    root = run / "candidates" / name
    complete = read_json(root / "complete.json", sources)
    history = read_json(root / "history.json", sources)
    require(complete["candidate_id"] == name and complete["training_hash"] == training["training_hash"],
            "Candidate training identity differs: " + name)
    require(complete["parent_probe_sha256"] == training["parent_probe_sha256"], "Candidate parent differs")
    require(complete["history_sha256"] == sha256(root / "history.json"), "Candidate history changed")
    require(not complete.get("engineering_smoke_only"), "Engineering candidate cannot become a final report")
    require(isinstance(history, list) and [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
            "Candidate history has missing/repeated epochs")
    require(complete["epochs"] == len(history) and 1 <= complete["best_epoch"] <= len(history),
            "Candidate epoch denominator differs")
    require(history[complete["best_epoch"] - 1]["dev"] == complete["dev"], "Exported best epoch differs from history")
    if (root / "best.npz").is_file():
        require(complete["probe_sha256"] == sha256(root / "best.npz"), "Candidate checkpoint changed")
        sources[str((root / "best.npz").resolve())] = {"bytes": (root / "best.npz").stat().st_size,
                                                       "sha256": complete["probe_sha256"]}
    initialization = root / "initialization-proof.json"
    if initialization.is_file():
        proof = read_json(initialization, sources)
        require(proof["optimizer_initialized_empty"] and proof["epoch_reset_to"] == 0 and proof["standardizer_inherited"],
                "Candidate did not reset optimizer/epoch while preserving standardization")
        require(proof["inherited_weights"] == (name != "scratch"), "Candidate initialization policy differs")
    require(complete["standardizer_mean_sha256"] == training["report_parent_mean_sha256"] and
            complete["standardizer_scale_sha256"] == training["report_parent_scale_sha256"],
            "Candidate standardizer differs from parent")
    if name == "warm-output":
        require(complete["hidden_layer_unchanged"], "Output-only candidate changed its hidden layer")
    return {"complete": complete, "history": history}


def resource_summary(metadata, run, freeze, sources):
    budget_path = find_metadata(metadata, "budget.json")
    budget = read_json(budget_path, sources)
    duration = budget["deadline_unix"] - budget["started_unix"]
    require(0 < duration <= 21600.01, "Resource budget exceeds six hours")
    if freeze.get("gpu_budget"):
        require(freeze["gpu_budget"]["started_unix"] == budget["started_unix"] and
                freeze["gpu_budget"]["deadline_unix"] == budget["deadline_unix"], "Training and campaign budgets differ")
    preflight_path = run / "preflight.json"
    if not preflight_path.is_file():
        preflight_path = find_metadata(metadata, "preflight.json")
    preflight = read_json(preflight_path, sources)
    require(preflight.get("status") == "passed" and preflight.get("devices") == [0, 1, 2], "Three-card preflight did not pass")
    complete = read_json(find_metadata(metadata, "gpu-complete.json"), sources)
    require(complete["freeze_sha256"] == sha256(run / "freeze.json"), "GPU completion belongs to another freeze")
    stop = complete["completed_unix"]
    require(stop >= budget["started_unix"], "GPU completion predates budget")
    watchdog_path = find_metadata(metadata, "watchdog.json", required=False)
    watchdog = read_json(watchdog_path, sources) if watchdog_path else None
    scale = read_json(find_metadata(metadata, "scale-selection.json"), sources)
    require(scale["budget_sha256"] == sha256(budget_path) and scale.get("chosen"), "Scale decision has another budget")
    telemetry_path = find_metadata(metadata, "gpu-resources.csv", required=False)
    cards = {}
    if telemetry_path:
        rows = read_csv(telemetry_path, sources)
        for gpu in range(3):
            observed = [row for row in rows if int(row["gpu"]) == gpu and
                        budget["started_unix"] <= float(row["unix"]) <= stop]
            if observed:
                used = [float(row["memory_used_mib"]) for row in observed]
                utilization = [float(row["utilization_pct"]) for row in observed]
                cards[str(gpu)] = {"samples": len(observed), "sampled_peak_memory_mib": max(used),
                    "sample_mean_utilization_pct": float(np.mean(utilization)),
                    "uuids": sorted({row["uuid"] for row in observed}),
                    "first_sample_unix": min(float(row["unix"]) for row in observed),
                    "last_sample_unix": max(float(row["unix"]) for row in observed)}
    return {"budget": budget, "gpu_complete": complete, "preflight": preflight,
            "scale": scale, "watchdog": watchdog, "cards": cards,
            "gpu_window_seconds": stop - budget["started_unix"], "within_budget": stop <= budget["deadline_unix"],
            "telemetry_scope": "Device-wide discrete samples include all CUDA contexts; sampled peaks are lower bounds, not continuous maxima."}


def quality_summary(metadata, training, output, sources):
    ledger_path = find_metadata(metadata, "quality-excluded-clusters.json")
    ledger = read_json(ledger_path, sources)
    expected = training["quality_exclusions"]
    require(sha256(ledger_path) == expected["ledger_sha256"], "Final quality ledger differs from training")
    count = len(set(ledger["excluded_clusters"]))
    require(count == len(ledger["excluded_clusters"]) == expected["excluded_cluster_count"], "Quality count differs")
    origins = Counter(row.get("decision_origin", "unspecified") for row in ledger.get("reasons", []))
    audit_path = find_metadata(metadata, "article-audit.json")
    audit = read_json(audit_path, sources)
    manifest_path = find_metadata(metadata, "data-manifest.json")
    data_manifest = read_json(manifest_path, sources)
    require(audit.get("status") == "passed" and audit.get("data_manifest_sha256") == sha256(manifest_path),
            "Article audit does not bind final data")
    require(audit.get("reviewed_articles") == 50, "Final fixed50 review is incomplete")
    frame_path = find_metadata(metadata, "article-audit-sample/selection.json", required=False)
    sampling_frame = read_json(frame_path, sources).get("sampling_frame") if frame_path else None
    original_rejections = output / "data-preparation/manual-rejections-v2.json"
    initial_review_count = None
    if original_rejections.is_file():
        initial = read_json(original_rejections, sources)
        if isinstance(initial, dict):
            rejected = initial.get("rejected_sources", initial.get("rejections"))
            if isinstance(rejected, list):
                initial_review_count = len(rejected)
    return {"excluded_cluster_count": count, "assessed_sources": ledger.get("assessed_sources"),
            "counts_by_reason": ledger.get("counts_by_reason", {}),
            "counts_by_source_split": ledger.get("counts_by_source_split", {}),
            "decision_origin_counts": dict(origins), "initial_review_rejections": initial_review_count,
            "quality_ledger_sha256": sha256(ledger_path),
            "final_reviewed_articles": audit["reviewed_articles"], "audit_sha256": sha256(audit_path),
            "audit_reviewer_type": "LLM agent; not human gold", "final_data_manifest_sha256": sha256(manifest_path),
            "sampling_frame": sampling_frame,
            "quotas": data_manifest["quotas"], "statistics": data_manifest["statistics"],
            "limitations": "The inherited parent weights and FIT mean/scale retain their historical noisy-source provenance; new filtering does not undo prior training."}


def read_cached_evaluation(root, freeze, parent_sha, sources):
    root = Path(root)
    manifest = read_json(root / "result-manifest.json", sources)
    require(manifest["status"] == "complete" and manifest["probe_sha256"] == freeze["probe_sha256"], "Cached evaluation is not complete for this probe")
    entries = {row["path"]: row for row in manifest["files"]}
    names = ("metrics.json", "cache-reuse-audit.json", "parent-mean-probabilities.csv",
             "new-mean-probabilities.csv", "sd-segment-comparison.csv")
    for name in names:
        path = root / name
        require(entries[name]["sha256"] == sha256(path), "Cached evaluation changed: " + name)
    metrics = read_json(root / "metrics.json", sources)
    audit = read_json(root / "cache-reuse-audit.json", sources)
    require(metrics["status"] == "complete" and metrics["probe_sha256"] == freeze["probe_sha256"] and
            metrics["parent_probe_sha256"] == parent_sha and metrics["historical_regression_only"] and
            not metrics["new_language_model_forward"] and metrics["native_cached_decode_preserved"],
            "Cached evaluation provenance differs")
    require(metrics["frozen_at_unix"] == freeze["frozen_at_unix"] and
            metrics["evaluated_at_unix"] >= freeze["frozen_at_unix"], "Cached evaluation predates freeze")
    require(audit["status"] == "passed", "Cached feature/mask audit failed")
    probabilities = {}
    for name in ("parent", "new"):
        rows = read_csv(root / (name + "-mean-probabilities.csv"), sources)
        probabilities[name] = [row for row in rows if row["aggregation"] == "unique_task_equal_mean" and int(row["tokens"]) > 0]
    segments = read_csv(root / "sd-segment-comparison.csv", sources)
    return {"metrics": metrics, "audit": audit, "probability_rows": probabilities, "segment_rows": segments}


def plot_training(candidates, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    directory = output / "training-diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    colors = {"warm-all": "#0072B2", "warm-output": "#D55E00", "scratch": "#777777"}
    curve_data = {}
    for name in CANDIDATES:
        complete, history = candidates[name]["complete"], candidates[name]["history"]
        epochs = [row["epoch"] for row in history]
        best = complete["best_epoch"]
        values = {"train_loss": [row["train_loss"] for row in history],
                  "new_dev_weakest_sd_recall": [row["dev"]["new_dev"].get("weakest_sd_recall") for row in history],
                  "new_dev_structure_macro_f1": [row["dev"]["new_dev"]["structure_equal_macro_f1"] for row in history],
                  "old_dev_clean_macro_f1": [row["dev"]["old_dev"]["overall"]["macro_f1"] for row in history],
                  "old_dev_full_macro_f1": [row["dev"].get("old_dev_full_diagnostic", {}).get("overall", {}).get("macro_f1") for row in history]}
        curve_data[name] = {"epochs": epochs, "best_epoch": best, "actual_epochs": len(history), **values}
        label = f"{name}{' [selected]' if name == selected else ''} (best={best}, ran={len(history)})"
        width = 2.8 if name == selected else 1.6
        for axis, key, percent in [(axes[0, 0], "train_loss", False),
                                   (axes[0, 1], "new_dev_weakest_sd_recall", True),
                                   (axes[1, 0], "new_dev_structure_macro_f1", True),
                                   (axes[1, 1], "old_dev_clean_macro_f1", True)]:
            y = np.asarray([np.nan if x is None else x for x in values[key]], dtype=float) * (100 if percent else 1)
            axis.plot(epochs, y, color=colors[name], linewidth=width, label=label,
                      zorder=5 if name == selected else 2)
            axis.axvline(best, color=colors[name], alpha=.2, linewidth=1, linestyle=":")
            if np.isfinite(y[best - 1]):
                axis.scatter([best], [y[best - 1]], color=colors[name], edgecolors="black", linewidths=.6,
                             s=45, zorder=8 if name == selected else 4)
        full = np.asarray([np.nan if x is None else x for x in values["old_dev_full_macro_f1"]]) * 100
        axes[1, 1].plot(epochs, full, color=colors[name], linewidth=1, linestyle="--", alpha=.55)
    titles = ["Sampled training cross-entropy", "New DEV weakest S/D recall (zoom)",
              "New DEV structure-equal macro-F1 (zoom)", "Old DEV macro-F1: clean solid / full dashed (zoom)"]
    for index, axis in enumerate(axes.flat):
        axis.set_title(titles[index])
        axis.set_xlabel("Additional training epoch")
        axis.set_ylabel("Cross-entropy" if index == 0 else "Percent")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        axis.grid(alpha=.25)
        if index == 0:
            axis.set_ylim(bottom=0)
        else:
            bounds = [line.get_ydata() for line in axis.lines if len(line.get_ydata()) != 2 or line.get_linestyle() != ":"]
            finite = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in bounds]) if bounds else np.array([])
            if finite.size:
                pad = max(.1, .08 * (finite.max() - finite.min()))
                axis.set_ylim(max(0, finite.min() - pad), min(100.05, finite.max() + pad))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=1, fontsize=9)
    figure.suptitle("Fixed 16-head MLP128 readouts; best markers use recorded DEV selection", fontsize=14)
    files = []
    for extension in ("png", "pdf"):
        path = directory / ("training-loss-and-dev." + extension)
        figure.savefig(path, dpi=180)
        files.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    plt.close(figure)
    write_json(directory / "curve-data.json", {"selected_candidate": selected, "curves": curve_data,
        "training_loss_denominator": "Actual sampled training batches in the epoch; not a full-population loss.",
        "best_epoch_policy": "Recorded lexicographic DEV selection; not a curve maximum/minimum.",
        "old_dev_full": "Diagnostic only, not used for selection."})
    write_json(directory / "plot-manifest.json", {"files": files, "curve_data_sha256": sha256(directory / "curve-data.json")})
    return curve_data, files


def build_report(run_dir, cached_eval_dir, metadata_dir, output_dir, parent_probe, *, fixture=False):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if fixture:
        require(output.is_relative_to(Path(tempfile.gettempdir()).resolve()) and
                any(part.startswith("warmstart-report-fixture-") for part in output.parts),
                "Fixture reports may only be written under the dedicated temporary fixture directory")
    sources = {}
    training = read_json(run / "training.json", sources)
    selection = read_json(run / "selection.json", sources)
    freeze = read_json(run / "freeze.json", sources)
    probe, parent, _ = validate_freeze(run / "best.npz", parent_probe, run / "freeze.json")
    for path in (run / "best.npz", Path(parent_probe)):
        sources[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    require(not training.get("engineering_smoke_only") and not probe.metadata.get("engineering_smoke_only"),
            "Engineering run cannot become a final report")
    require(freeze["training_sha256"] == sha256(run / "training.json") and
            freeze["selection_sha256"] == sha256(run / "selection.json"), "Frozen selection changed")
    require(selection["selection_split"] == "dev" and freeze["candidate_id"] == selection["candidate_id"], "Selection identity differs")
    for key in FEATURE_KEYS:
        require(training["feature_binding"][key] == parent.metadata[key], "Training feature identity differs")
    require(training["parent_probe_sha256"] == sha256(parent_probe), "Training parent identity differs")
    require(np.array_equal(probe.mean, parent.mean) and np.array_equal(probe.scale, parent.scale), "Selected standardizer changed")
    require(probe.metadata["warm_start"].get("optimizer_reset") is True and
            probe.metadata["warm_start"].get("epoch_reset") is True,
            "Selected checkpoint does not declare optimizer/epoch reset")
    training = {**training, "report_parent_mean_sha256": hashlib.sha256(parent.mean.tobytes()).hexdigest(),
                "report_parent_scale_sha256": hashlib.sha256(parent.scale.tobytes()).hexdigest()}
    candidates = {name: check_candidate(run, name, training, sources) for name in CANDIDATES}
    selected = freeze["candidate_id"]
    require(selected in candidates, "Unknown selected candidate")
    require(candidates[selected]["complete"]["probe_sha256"] == freeze["probe_sha256"], "Selected checkpoint differs")
    require(probe.metadata.get("candidate_id") == selected and
            probe.metadata.get("best_epoch") == candidates[selected]["complete"]["best_epoch"],
            "Selected checkpoint metadata and recorded best-epoch marker differ")
    require(selection["candidate_results"] == [candidates[name]["complete"] for name in CANDIDATES], "Frozen candidate records changed")
    test = read_json(run / "evaluations/test.json", sources)
    require(test["split"] == "test" and test["probe_sha256"] == freeze["probe_sha256"], "Article TEST belongs to another freeze")
    ci = test["confidence_intervals"]
    require(ci["unit"] == "source_cluster" and ci["confidence"] == .95 and ci["clusters"] >= 2, "Expected 95% source-cluster interval")
    macro_ci = ci["intervals"]["macro_f1"]
    require(0 <= macro_ci["lower"] <= macro_ci["upper"] <= 1, "Invalid macro-F1 confidence interval")
    cached = read_cached_evaluation(cached_eval_dir, freeze, training["parent_probe_sha256"], sources)
    resources = resource_summary(metadata_dir, run, freeze, sources)
    quality = quality_summary(metadata_dir, training, output, sources)
    output.mkdir(parents=True, exist_ok=True)
    curves, plot_files = plot_training(candidates, selected, output)
    reports = cached["metrics"]["models"]
    chosen = candidates[selected]["complete"]
    summary = {"schema": "six-hour-warmstart-final-report-v1", "fixture_only": fixture,
        "selected_candidate": selected, "best_epoch": chosen["best_epoch"], "actual_epochs": chosen["epochs"],
        "eligible": chosen["dev"]["eligible"], "probe_sha256": freeze["probe_sha256"],
        "parent_probe_sha256": training["parent_probe_sha256"], "article_test": test,
        "agentdojo": cached["metrics"], "quality": quality, "resources": resources,
        "old_dev_denominators": training["old_dev_denominators"],
        "standardizer": {"mean_unchanged": True, "scale_unchanged": True,
            "mean_sha256": training["report_parent_mean_sha256"], "scale_sha256": training["report_parent_scale_sha256"]},
        "candidates": {name: value["complete"] for name, value in candidates.items()}}
    # Never serialize the ledger's excluded IDs or the training source/family lists.
    write_json(output / "report-summary.json", summary)
    lines = ["# GPT-OSS-120B 六角色旁路分类器：六小时优化结果", ""]
    if fixture:
        lines += ["**临时合成 fixture，仅用于报告代码检查；不是实验结果。**", ""]
    lines += [f"最终候选为 **{selected}**，最佳新增 epoch **{chosen['best_epoch']}**，实际训练 **{chosen['epochs']}** 轮；仅使用文章 DEV 选定。",
        f"DEV 门槛：**{'通过' if chosen['dev']['eligible'] else '未通过'}**。冻结 SHA：`{freeze['probe_sha256']}`。",
        "本轮只更新旁路分类器。120B 本体和原57任务的行为没有改变；缓存读出变化不代表 AgentDojo 任务成功率或安全性提升。", "",
        "## AgentDojo 历史缓存回归", "",
        f"复用原始输入和生成时的原生特征，评分 {number(cached['audit']['retained_normal_tokens'])} 个首次普通正文 token；没有重新运行120B或agent。",
        f"这 {cached['metrics']['tasks']} 个任务只有 **{cached['metrics']['template_counts']['system']} 种 system、{cached['metrics']['template_counts']['developer']} 种 developer 提示正文**。它们是已观察回归集，不是新的真实提示盲测，也不验证新三卡部署稳定性。", "",
        "硬分类结果如下，输入和生成各取其真实保存位置；单位为%。", ""]
    hard = []
    for role in ROLES:
        before, after = reports["parent"]["overall"]["roles"][role], reports["new"]["overall"]["roles"][role]
        require(before["support"] == after["support"], "Parent/new role supports differ")
        hard.append([role, number(after["support"]), pct(before["precision"]), pct(before["recall"]), pct(before["f1"]),
                     pct(after["precision"]), pct(after["recall"]), pct(after["f1"])])
    lines += table(["真实角色", "正例token", "父P", "父R", "父F1", "新P", "新R", "新F1"], hard)
    lines += ["", "下面另列任务内先对token求均值、再对有支持任务等权平均的角色概率；它们不是准确率，也不是上一表的召回率。", ""]
    probability_rows = []
    for model in ("parent", "new"):
        for row in cached["probability_rows"][model]:
            probability_rows.append([model, row["scope"], row["tag"], row["tasks_observed"], number(row["tokens"]),
                *[pct(float(row["p_" + role])) for role in ROLES]])
    lines += table(["模型", "来源", "正文标签", "支持任务", "token", *ROLES], probability_rows)
    lines += ["", "### S/D 分段与低权限误报", "",
              "分段使用固定字符锚点，跨界token按中点归属；各分段合计覆盖全部S/D正文。分段召回按token统计。", ""]
    segments = []
    for segment in sorted(reports["parent"]["sd_segments"]):
        role = segment.split("/")[0]
        a, b = reports["parent"]["sd_segments"][segment], reports["new"]["sd_segments"][segment]
        segments.append([segment, number(a["tokens"]), pct(a["roles"][role]["recall"]), pct(b["roles"][role]["recall"])])
    lines += table(["分段", "token", "父召回", "新召回"], segments)
    low = []
    for model in ("parent", "new"):
        item = reports[model]["overall"]["user_tool_to_system_developer"]
        low.append([model, number(item["denominator"]), number(item["false_positives"]), pct(item["rate"])])
    lines += [""] + table(["模型", "真实U/T token", "误判S/D", "误报率"], low)
    lines += ["", "完整输入／生成分表、suite与任务分表、正文四分位及分段概率见缓存复评目录；无该类正例时recall/F1保留N/A。", "",
              "## 文章 DEV 选择与全新 TEST", ""]
    denominator = training["old_dev_denominators"]
    lines += table(["旧DEV口径", "token行", "源簇", "是否用于选择"], [
        ["quality-clean", number(denominator["clean_rows"]), number(denominator["clean_sources"]), "是，父／新模型同一分母"],
        ["full", number(denominator["full_rows"]), number(denominator["full_sources"]), "否，仅诊断"],
        ["被排除", number(denominator["excluded_rows"]), number(denominator["full_sources"] - denominator["clean_sources"]), "不回填或算作识别成功"]])
    candidate_rows = []
    for name in CANDIDATES:
        c = candidates[name]["complete"]; dev = c["dev"]
        candidate_rows.append([name, c["epochs"], c["best_epoch"], "是" if dev["eligible"] else "否",
            pct(dev["new_dev"].get("weakest_sd_recall")), pct(dev["new_dev"]["structure_equal_macro_f1"]),
            pct(dev["old_dev"]["overall"]["macro_f1"]), pct(dev.get("old_dev_full_diagnostic", {}).get("overall", {}).get("macro_f1"))])
    lines += [""] + table(["候选", "实际轮数", "best epoch", "eligible", "新DEV最弱S/D R", "新DEV结构macro-F1", "旧DEV clean macro-F1", "旧DEV full macro-F1"], candidate_rows)
    lines += ["", f"父模型旧DEV clean macro-F1为 {pct(chosen['dev']['parent_old_dev_macro_f1'])}，full为 {pct(chosen['dev'].get('parent_old_dev_full_macro_f1'))}。",
        "候选选择优先满足两组DEV六类P/R≥90%、旧clean DEV macro-F1下降≤0.5个百分点、两组U/T→S/D误报≤1%；再比较新增DEV最弱S/D组、结构macro-F1、NLL和固定ID。best epoch不一定是单条曲线的最大值。",
        f"三个候选使用配置中的固定seed={number(training['options'].get('seed'))}，不构成多随机种子稳定性证明。未通过DEV门槛的导出只作为研究候选，不自动替换旧默认模型。", "",
        f"全新文章TEST：**accuracy {pct(test['overall']['accuracy'])}，六类macro-F1 {pct(test['overall']['macro_f1'])}**；",
        f"macro-F1 的95%源簇bootstrap区间 **[{pct(macro_ci['lower'])}, {pct(macro_ci['upper'])}]**，{ci['clusters']}个源簇、{ci['replicates']}次重采样，seed={ci['seed']}。",
        "该区间属于文章TEST，不能移用于57任务或单一S/D提示模板。", "",
        "## 继承参数、数据质量与限制", "",
        f"父checkpoint SHA：`{training['parent_probe_sha256']}`。固定原16头、1024维pre-o_proj FP16特征；父FIT mean/scale逐元素保留，未使用AgentDojo重新标准化。",
        "warm-all继承全部MLP权重后更新全网；warm-output只更新最后六类层；scratch只随机初始化MLP权重，仍沿用原16头和旧FIT mean/scale，不是整个pipeline从零训练。AdamW、epoch与早停状态重新开始，不拼接父实验最后一轮优化器。",
        f"每轮实际抽样上限／配额为 {number(training['options']['epoch_rows'])}，新增／旧FIT回放目标为60%／40%，新增内paired／context为60%／40%。实际累计计数及截至best epoch的路径均保留在候选记录。", ""]
    manual_count = quality["initial_review_rejections"]
    if manual_count is not None:
        lines += [f"最初V1固定样本审阅有 **{manual_count}篇拒绝**，原失败审计保留；此后重新生成文章集合并复审。"]
    lines += [f"最终质量规则／拒绝清单从 {number(quality['assessed_sources'])} 个可检查来源中排除 **{quality['excluded_cluster_count']}个源簇**，该数取自最终ledger，不是固定常数；拒绝簇也排除于旧FIT回放和旧DEV clean视图。",
        f"最终固定 **{quality['final_reviewed_articles']}篇** FIT/DEV审阅为LLM代理审阅，**不是人类gold**；同正文SHA的历史全文审阅可继承，不保证全语料绝无对话或噪声。",
        "审阅抽样框为嵌套保守档256篇FIT中的30篇，以及128篇DEV中的20篇；不是对标准512篇FIT均匀抽样，不能据此保证额外半数FIT的语义质量。",
        "保留文章中的文学风格、营销号召、多语页尾、OCR噪声、事实／地点混乱等质量注记。非对话通过不表示事实可靠、作者为人类或文章来源代表整个C4/Dolma3。",
        "warm父权重已见过被本轮新质量规则拒绝的非对话页面，父标准化器也保留历史身份；本轮从新增数据及回放排除这些来源不能抹除旧权重影响。质量不合格不等同于已识别到真实对话。", "",
        "## 三卡资源、时间与产物", "",
        f"GPU预算起点：{local_time(resources['budget']['started_unix'])}；六小时截止：{local_time(resources['budget']['deadline_unix'])}。",
        f"GPU阶段完成：{local_time(resources['gpu_complete']['completed_unix'])}，作业窗口 {resources['gpu_window_seconds']/3600:.3f}小时，**{'在预算内' if resources['within_budget'] else '超过预算，需调查'}**。授权上限18 GPU小时不等于kernel实际工作量或账单金额。",
        f"仅根据固定pilot计时选择 FIT={resources['scale']['chosen']['fit_sources']} 档位，DEV和全新TEST配额不随回归成绩变化。三卡共同驻留同一冻结120B，完成提取后释放本体，每卡并行一个旁路候选。", ""]
    card_rows = [[gpu, number(item["samples"]), f"{item['sampled_peak_memory_mib']:.0f}",
                  f"{item['sample_mean_utilization_pct']:.2f}%"] for gpu, item in sorted(resources["cards"].items())]
    if card_rows:
        lines += table(["GPU", "样本数", "采样峰值MiB", "采样平均利用率"], card_rows)
    else:
        lines += ["连续遥测未提供，不能报告整轮GPU峰值或平均利用率。"]
    lines += ["", "设备全局离散采样包含其他CUDA上下文；峰值为已观察下界，缺口不插值，不作为本进程连续真实峰值。",
        "", "- [训练loss和DEV曲线PNG](training-diagnostics/training-loss-and-dev.png)",
        "- [训练loss和DEV曲线PDF](training-diagnostics/training-loss-and-dev.pdf)",
        "- [曲线数据与best标记](training-diagnostics/curve-data.json)",
        "- [结构化报告摘要](report-summary.json)", "- [报告来源SHA清单](report-source-manifest.json)",
        "", "训练loss是实际抽样batch均值；DEV图明确使用放大纵轴。圆点和竖线标记记录的最佳epoch；旧DEV full虚线只作诊断。"]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    write_json(output / "report-source-manifest.json", {"generated_at_unix": time.time(),
        "script_sha256": sha256(__file__), "sources": sources,
        "coverage": "Frozen metadata, candidate histories, article TEST metrics/CI, cached native readouts, quality counts and resource samples. No new language-model forward or independent metric recomputation is claimed."})
    generated = [output / "REPORT.zh-CN.md", output / "report-summary.json", output / "report-source-manifest.json",
                 *sorted((output / "training-diagnostics").glob("*"))]
    write_json(output / "report-validation.json", {"status": "passed", "fixture_only": fixture,
        "probe_sha256": freeze["probe_sha256"], "freeze_sha256": sha256(run / "freeze.json"),
        "candidate_count": len(candidates), "actual_epoch_counts": {name: len(value["history"]) for name, value in candidates.items()},
        "best_epoch_markers": {name: value["complete"]["best_epoch"] for name, value in candidates.items()},
        "mean_scale_unchanged": True, "quality_count_from_final_ledger": quality["excluded_cluster_count"],
        "original_behavior_unchanged": True, "new_agentdojo_rollout": False,
        "actual_image_visual_review": "Not performed by the script; must be recorded separately after viewing PNGs.",
        "files": [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in generated if path.is_file()]})
    return {"status": "complete", "fixture_only": fixture, "selected_candidate": selected,
            "output_dir": str(output), "report": str(output / "REPORT.zh-CN.md"), "plots": plot_files}


def self_test():
    """Exercise frozen-source checks and plots without publishing toy results."""
    import shutil

    root = Path(tempfile.mkdtemp(prefix="warmstart-report-fixture-"))
    run, cached, metadata, output = [root / name for name in ("run", "cached", "metadata", "output")]
    for path in (run / "evaluations", cached, metadata / "data", output / "data-preparation"):
        path.mkdir(parents=True)
    base_metadata = {"format_version": classifier.PROBE_FORMAT, "roles": list(ROLES), "heads": list(range(16)),
                     "feature_dim": 1024, "feature_position": "pre-o_proj", "model_binding": {"fixture": "frozen"},
                     "hidden": 128, "dropout": .1, "best_epoch": 30, "engineering_smoke_only": False}
    arrays = {"mean": np.zeros(1024, np.float32), "scale": np.ones(1024, np.float32),
              "weight::0.weight": np.zeros((128, 1024), np.float32), "weight::0.bias": np.zeros(128, np.float32),
              "weight::3.weight": np.zeros((6, 128), np.float32), "weight::3.bias": np.zeros(6, np.float32)}
    parent = root / "parent.npz"
    np.savez(parent, metadata=np.array(json.dumps(base_metadata)), **arrays)
    parent_digest = sha256(parent)
    binding = {key: base_metadata[key] for key in FEATURE_KEYS}
    mean_sha = hashlib.sha256(arrays["mean"].tobytes()).hexdigest()
    scale_sha = hashlib.sha256(arrays["scale"].tobytes()).hexdigest()
    ledger = {"excluded_clusters": ["fixture-a", "fixture-b", "fixture-c", "fixture-d"],
              "assessed_sources": 100, "reasons": [{"decision_origin": "manual_full_article_review"}] * 2,
              "counts_by_reason": {"fixture_quality": 4}, "counts_by_source_split": {"c4/fit": 4}}
    ledger_path = metadata / "data/quality-excluded-clusters.json"
    write_json(ledger_path, ledger)
    data = {"quotas": {"fit": 512, "dev": 128, "test": 128},
            "statistics": {"fit": {"sources": 512}, "dev": {"sources": 128}, "test": {"sources": 128}}}
    write_json(metadata / "data/data-manifest.json", data)
    write_json(metadata / "data/article-audit.json", {"status": "passed", "reviewed_articles": 50,
        "data_manifest_sha256": sha256(metadata / "data/data-manifest.json")})
    write_json(output / "data-preparation/manual-rejections-v2.json", {"rejected_sources": [{}, {}]})
    training = {"training_hash": "fixture-training", "parent_probe_sha256": parent_digest,
                "feature_binding": binding, "engineering_smoke_only": False,
                "options": {"epoch_rows": 500000, "seed": 17},
                "quality_exclusions": {"ledger_sha256": sha256(ledger_path), "excluded_cluster_count": 4},
                "old_dev_denominators": {"full_rows": 100, "clean_rows": 90, "excluded_rows": 10,
                                         "full_sources": 10, "clean_sources": 9}}
    write_json(run / "training.json", training)

    def metric(correct=95):
        matrix = np.eye(6, dtype=np.int64) * correct
        for i in range(6):
            matrix[i, (i + 1) % 6] += 100 - correct
        return classifier._metrics_from_confusion(matrix, nll=.1)

    def dev(epoch, candidate):
        # Epoch2 wins on the first selection criterion even though macro rises at epoch3.
        weakest = [.8, .91, .85][epoch - 1] - (0 if candidate == "warm-all" else .02)
        return {"eligible": True, "selection_key": [False, -weakest, -.94, .1, candidate],
                "new_dev": {"overall": metric(), "weakest_sd_recall": weakest,
                            "structure_equal_macro_f1": [.92, .94, .96][epoch - 1]},
                "old_dev": {"overall": metric(96)}, "old_dev_full_diagnostic": {"overall": metric(94)},
                "parent_old_dev_macro_f1": .96, "parent_old_dev_full_macro_f1": .94,
                "old_dev_full_used_for_selection": False}
    records = []
    for name in CANDIDATES:
        directory = run / "candidates" / name
        directory.mkdir(parents=True)
        candidate_metadata = {**base_metadata, "candidate_id": name, "best_epoch": 2,
                              "warm_start": {"parent_probe_sha256": parent_digest, "optimizer_reset": True,
                                             "epoch_reset": True, "initialize_weights": name != "scratch"}}
        np.savez(directory / "best.npz", metadata=np.array(json.dumps(candidate_metadata)), **arrays)
        history = [{"epoch": epoch, "train_loss": (.3 if name == "scratch" else .1) / epoch,
                    "dev": dev(epoch, name)} for epoch in range(1, 4)]
        write_json(directory / "history.json", history)
        record = {"candidate_id": name, "training_hash": "fixture-training", "parent_probe_sha256": parent_digest,
                  "history_sha256": sha256(directory / "history.json"), "probe_sha256": sha256(directory / "best.npz"),
                  "epochs": 3, "best_epoch": 2, "dev": history[1]["dev"],
                  "engineering_smoke_only": False, "standardizer_mean_sha256": mean_sha,
                  "standardizer_scale_sha256": scale_sha, "hidden_layer_unchanged": name == "warm-output"}
        write_json(directory / "complete.json", record)
        write_json(directory / "initialization-proof.json", {"optimizer_initialized_empty": True,
            "epoch_reset_to": 0, "standardizer_inherited": True, "inherited_weights": name != "scratch"})
        records.append(record)
    shutil.copyfile(run / "candidates/warm-all/best.npz", run / "best.npz")
    selection = {"selection_split": "dev", "candidate_id": "warm-all", "candidate_results": records}
    write_json(run / "selection.json", selection)
    now = time.time()
    freeze = {"format_version": "synthetic-role-warm-freeze-v1", "selection_split": "dev", "candidate_id": "warm-all",
              "frozen_at_unix": now - 100, "probe_sha256": sha256(run / "best.npz"),
              "parent_probe_sha256": parent_digest, "feature_binding": binding,
              "training_sha256": sha256(run / "training.json"), "selection_sha256": sha256(run / "selection.json")}
    write_json(run / "freeze.json", freeze)
    test = {"overall": metric(), "split": "test", "probe_sha256": freeze["probe_sha256"],
            "confidence_intervals": {"unit": "source_cluster", "confidence": .95, "clusters": 128,
                "replicates": 2000, "seed": 17, "intervals": {"macro_f1": {"lower": .93, "upper": .97}}}}
    write_json(run / "evaluations/test.json", test)
    for path in (run / "preflight.json", metadata / "preflight.json"):
        write_json(path, {"status": "passed", "devices": [0, 1, 2]})
    write_json(metadata / "budget.json", {"started_unix": now - 3000, "deadline_unix": now - 3000 + 21600})
    write_json(metadata / "gpu-complete.json", {"completed_unix": now - 10, "freeze_sha256": sha256(run / "freeze.json")})
    write_json(metadata / "scale-selection.json", {"chosen": {"fit_sources": 256},
        "budget_sha256": sha256(metadata / "budget.json")})
    write_json(metadata / "watchdog.json", {"status": "gpu_completed", "kills": []})
    with (metadata / "gpu-resources.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unix", "gpu", "uuid", "memory_used_mib", "memory_total_mib", "utilization_pct"])
        for gpu in range(3):
            for timestamp in (now - 2000, now - 20):
                writer.writerow([timestamp, gpu, f"fixture-gpu-{gpu}", 21000 + gpu, 46000, 25])
    segment_sizes = {"system/identity_and_date": 50, "system/reasoning_setting": 10,
                     "system/channel_rules": 20, "system/tool_routing": 20,
                     "developer/instructions": 30, "developer/tool_schema": 70}
    models, segment_rows = {}, []
    for model in ("parent", "new"):
        matrix = np.zeros((6, 6), np.int64)
        for i in range(6):
            correct = ([60, 10, 95, 95, 95, 95] if model == "parent" else [80, 40, 96, 96, 96, 96])[i]
            matrix[i, i] = correct
            matrix[i, (i + 1) % 6] += 100 - correct
        overall = classifier._metrics_from_confusion(matrix)
        overall["user_tool_to_system_developer"] = {"denominator": 200, "false_positives": 5 if model == "parent" else 4,
                                                   "rate": .025 if model == "parent" else .02}
        segments = {}
        for segment, size in segment_sizes.items():
            role = segment.split("/")[0]; label = ROLES.index(role)
            values = np.zeros((6, 6), np.int64); correct = int(size * (.6 if model == "parent" else .8))
            values[label, label], values[label, (label + 1) % 6] = correct, size - correct
            segments[segment] = classifier._metrics_from_confusion(values)
            segment_rows.append({"segment": segment, "model": model, "tokens": size})
        models[model] = {"overall": overall, "sd_segments": segments}
        rows = []
        for role, tag, scope in [("system", "system", "actual_input"), ("developer", "developer", "actual_input"),
                ("user", "user", "actual_input"), ("tool", "tool", "actual_input"),
                ("cot", "assistant_analysis", "actual_generated"), ("assistant", "assistant_final", "actual_generated")]:
            row = {"scope": scope, "tag": tag, "aggregation": "unique_task_equal_mean", "tokens": 100,
                   "tasks_observed": 57}
            true_prob = .7 if model == "parent" else .8
            row.update({"p_" + r: true_prob if r == role else (1 - true_prob) / 5 for r in ROLES})
            rows.append(row)
        with (cached / (model + "-mean-probabilities.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cached_metrics = {"status": "complete", "probe_sha256": freeze["probe_sha256"], "parent_probe_sha256": parent_digest,
        "historical_regression_only": True, "new_language_model_forward": False, "native_cached_decode_preserved": True,
        "frozen_at_unix": freeze["frozen_at_unix"], "evaluated_at_unix": now, "tasks": 57,
        "template_counts": {"system": 1, "developer": 3}, "models": models}
    write_json(cached / "metrics.json", cached_metrics)
    write_json(cached / "cache-reuse-audit.json", {"status": "passed", "retained_normal_tokens": 600})
    with (cached / "sd-segment-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(segment_rows[0])); writer.writeheader(); writer.writerows(segment_rows)
    write_json(cached / "result-manifest.json", {"status": "complete", "probe_sha256": freeze["probe_sha256"],
        "files": [{"path": p.name, "sha256": sha256(p)} for p in sorted(cached.iterdir()) if p.is_file()]})
    result = build_report(run, cached, metadata, output, parent, fixture=True)
    summary = read_json(output / "report-summary.json")
    curve = read_json(output / "training-diagnostics/curve-data.json")
    require(summary["quality"]["excluded_cluster_count"] == 4 and summary["quality"]["initial_review_rejections"] == 2,
            "Report hardcoded quality counts")
    require(curve["curves"]["warm-all"]["best_epoch"] == 2 and
            np.argmax(curve["curves"]["warm-all"]["new_dev_structure_macro_f1"]) + 1 == 3,
            "Renderer replaced recorded best epoch by curve maximum")
    require(summary["old_dev_denominators"]["clean_rows"] == 90 and
            summary["old_dev_denominators"]["full_rows"] == 100, "Clean/full denominator collapsed")
    content = (output / "REPORT.zh-CN.md").read_text()
    require("70.0000%" in content and "60.0000%" in content and "它们不是准确率" in content,
            "Mean probability and hard recall were not kept separate")
    original = (run / "selection.json").read_bytes()
    write_json(run / "selection.json", {"tampered": True})
    try:
        try:
            build_report(run, cached, metadata, output, parent, fixture=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Report accepted modified frozen selection")
    finally:
        (run / "selection.json").write_bytes(original)
    return {"status": "passed", "fixture_only": True, "checks": ["freeze/source hashes", "three candidates and history",
            "recorded best epoch", "clean/full denominators", "mean probability vs recall", "dynamic quality counts",
            "rendered PNG/PDF", "reject modified selection"], "output_dir": str(output), "plots": result["plots"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cached-eval-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parent-probe", type=Path, default=Path(__file__).resolve().parent /
                        "outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        require(all((args.run_dir, args.cached_eval_dir, args.metadata_dir, args.output_dir)), "Report input directories are required")
        result = build_report(args.run_dir, args.cached_eval_dir, args.metadata_dir, args.output_dir, args.parent_probe)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
