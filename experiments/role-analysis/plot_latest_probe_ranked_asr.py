"""Pool plain/Remember and plot six role-specific ASRs in equal-count groups.

Probabilities come only from the latest mixed-data probe. Attack outcomes come
from the original sealed episodes. No probe or language model is run here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/latest-probe-ranked20-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = BASE / "outputs/agentdojo389-payload-mixed-probe-20261103"
LATEST_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"
ROLE_ORDER = ("system", "assistant", "user", "tool", "cot", "developer")
LABELS = {"system": "Systemness", "assistant": "Assistantness", "user": "Userness",
          "tool": "Toolness", "cot": "CoTness", "developer": "Developerness"}
COLORS = {"system": "#829ab6", "assistant": "#00b88c", "user": "#00afd5",
          "tool": "#b254ef", "cot": "#ffb000", "developer": "#d76351"}
PARTS = {"template": "payload_without_goal", "goal": "goal"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_score(value):
    require(value in ("True", "False", ""), "Unexpected native outcome")
    return None if value == "" else value == "True"


def data(args):
    predictions, output = args.predictions.resolve(), args.output.resolve()
    require(not output.exists(), "Use a new output directory")
    summary = read(predictions / "summary.json")
    rescore = read(predictions / "rescore_manifest.json")
    require(summary["probe_sha256"] == rescore["probe_sha256"] == LATEST_SHA, "Not the latest probe probabilities")
    require(read(predictions / "validation.json")["status"] == "passed", "Predictions not validated")
    require(sha(summary["probe"]) == LATEST_SHA, "Latest checkpoint changed")
    source = Path(rescore["source_run_root"])
    seal = read(source / "result_manifest.json")
    require(seal["status"] == "complete_with_initial_errors", "Unexpected original source state")
    inventory = {r["path"]: r for r in seal["files"]}
    consumed = {}
    def bound(relative):
        path = source / relative
        digest = sha(path)
        require(digest == inventory[relative]["sha256"], "Source bytes changed: " + relative)
        consumed[relative] = digest
        return path
    schedule = read(bound("schedule.json"))
    first = {r["case_id"]: r for r in read(bound("analysis/first_occurrences.json"))}
    cases = {}
    for spec in schedule:
        case = read(bound("cases/" + spec["case_id"] + ".json"))
        require(case["case_id"] == spec["case_id"] and case["condition"] == spec["condition"], "Native case identity differs")
        require(case["security"] is None or type(case["security"]) is bool, "Invalid native attack outcome")
        cases[case["case_id"]] = case
    require(len(schedule) == len(cases) == 778 and len(first) == 755, "Wrong original coverage")
    pair_ids = sorted({case_id.split("/", 1)[1] for case_id in cases})
    require(len(pair_ids) == 389, "Wrong paired case inventory")
    pair_index = {key: i for i, key in enumerate(pair_ids)}
    probabilities = read_csv(predictions / "per_case_probabilities.csv")
    source_checksums = {}
    for line in (predictions / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        source_checksums[relative] = digest
    for name in ("per_case_probabilities.csv", "summary.json", "rescore_manifest.json", "validation.json"):
        require(sha(predictions / name) == source_checksums[name], "Prediction artifact changed: " + name)
    by_part, all_cases = {}, []
    for part, stored_name in PARTS.items():
        cohort = []
        for row in probabilities:
            if row["part"] != stored_name:
                continue
            case = cases[row["case_id"]]
            require(parse_score(row["attack_success"]) == case["security"], "Native ASR outcome differs from export")
            require(row["condition"] == case["condition"] and row["case_id"] in first, "Wrong probability episode")
            require(int(row["tokens"]) == len(first[row["case_id"]]["parts"][stored_name]), "Wrong segment denominator")
            p = {role: float(row["p_" + role]) for role in ROLE_ORDER}
            require(all(np.isfinite(v) and 0 <= v <= 1 for v in p.values()) and abs(sum(p.values()) - 1) < 2e-6,
                    "Invalid role probability")
            cohort.append({"case_id": row["case_id"], "condition": row["condition"], "suite": row["suite"],
                           "score": case["security"], "p": p, "tokens": int(row["tokens"]),
                           "pair_index": pair_index[row["case_id"].split("/", 1)[1]]})
        require(len(cohort) == len({r["case_id"] for r in cohort}) == 755 and {r["case_id"] for r in cohort} == set(first), "Wrong ranked cohort")
        require(sum(r["score"] is not None for r in cohort) == 752 and sum(r["score"] is True for r in cohort) == 231, "Wrong scored cohort")
        require(Counter(r["condition"] for r in cohort) == {"plain": 375, "prerequisite": 380}, "Wrong condition counts")
        by_part[part] = cohort
    by_id = {part: {r["case_id"]: r for r in rows} for part, rows in by_part.items()}
    for case_id, case in sorted(cases.items()):
        row = {"case_id": case_id, "pair_id": case_id.split("/", 1)[1], "condition": case["condition"],
               "suite": case["suite"], "attack_success": case["security"], "episode_status": case["status"],
               "probe_available": case_id in first, "used_in_asr": case_id in first and case["security"] is not None}
        for part in PARTS:
            for role in ROLE_ORDER:
                row[part + "_p_" + role] = by_id[part][case_id]["p"][role] if case_id in first else None
        all_cases.append(row)
    weights = np.random.default_rng(args.seed).multinomial(389, np.full(389, 1 / 389), size=args.bootstrap).astype(np.uint16)
    bins, assignments, ties = [], [], {}
    partitions = np.array_split(np.arange(755), args.bins)
    for part, rows in by_part.items():
        ties[part] = {}
        for role in ROLE_ORDER:
            ranked = sorted(rows, key=lambda r: (r["p"][role], r["case_id"]))
            frequencies = Counter(r["p"][role] for r in ranked)
            ties[part][role] = {"unique_probabilities": len(frequencies), "largest_tie": max(frequencies.values()),
                                "exact_probability_one": frequencies.get(1., 0)}
            offset = 0
            for bin_id, indices in enumerate(partitions, 1):
                group = [ranked[int(i)] for i in indices]
                scored = [r for r in group if r["score"] is not None]
                success = [r for r in scored if r["score"] is True]
                probabilities_ = [r["p"][role] for r in group]
                denominator = weights[:, [r["pair_index"] for r in scored]].sum(1)
                numerator = weights[:, [r["pair_index"] for r in success]].sum(1) if success else np.zeros(args.bootstrap)
                valid = denominator > 0
                low, high = np.quantile(numerator[valid] / denominator[valid], [.025, .975])
                bins.append({"part": part, "role": role, "bin": bin_id, "rank_first": offset + 1,
                             "rank_last": offset + len(group), "percentile_lower": 100 * offset / 755,
                             "percentile_upper": 100 * (offset + len(group)) / 755,
                             "percentile_midpoint": 100 * (offset + len(group) / 2) / 755,
                             "probability_min": min(probabilities_), "probability_max": max(probabilities_),
                             "probability_mean": float(np.mean(probabilities_)), "ranked_cases": len(group),
                             "scored_cases": len(scored), "unscored_cases": len(group) - len(scored),
                             "attack_successes": len(success), "attack_failures": len(scored) - len(success),
                             "asr": len(success) / len(scored), "plain_cases": sum(r["condition"] == "plain" for r in group),
                             "remember_cases": sum(r["condition"] == "prerequisite" for r in group),
                             "ci95_low": float(low), "ci95_high": float(high), "valid_bootstrap_replicates": int(valid.sum())})
                for rank, row in enumerate(group, offset + 1):
                    assignments.append({"part": part, "role": role, "bin": bin_id, "rank": rank,
                                        "case_id": row["case_id"], "pair_id": row["case_id"].split("/", 1)[1],
                                        "condition": row["condition"], "suite": row["suite"],
                                        "probe_probability": row["p"][role], "attack_success": row["score"],
                                        "scored": row["score"] is not None})
                offset += len(group)
    metadata = {"schema": f"latest-probe-pooled-rank{args.bins}-asr-v1", "created_unix": time.time(),
                "predictions_root": str(predictions), "original_root": str(source), "probe_sha256": LATEST_SHA,
                "probe_candidate": summary["candidate"], "checkpoint_epoch": summary["checkpoint_epoch"],
                "source_prediction_hashes": {name: sha(predictions / name) for name in ("per_case_probabilities.csv", "summary.json", "rescore_manifest.json", "validation.json")},
                "source_result_manifest_sha256": sha(source / "result_manifest.json"), "source_files_sha256": consumed,
                "roles": list(ROLE_ORDER), "parts": PARTS, "scheduled_episodes": 778, "ranked_episodes": 755,
                "scored_episodes": 752, "unscored_ranked_episodes": 3, "missing_probe_episodes": 23,
                "attack_successes": 231, "ranked_scored_asr": 231 / 752, "bins_per_role": args.bins,
                "bin_sizes": [len(g) for g in partitions], "tie_statistics": ties,
                "overview_probability_annotations": args.bins <= 20,
                "sorting": "Ascending per-episode token-mean probability, then canonical case_id; each part/role independently",
                "null_policy": "Keep null outcomes in ranks; exclude them from ASR numerator and denominator. Missing probabilities cannot be ranked",
                "bootstrap": {"replicates": args.bootstrap, "seed": args.seed, "unit": "389 base suite/user_task/injection_task pairs",
                              "conditions_sampled_together": True, "fixed_rank_membership": True, "confidence": .95,
                              "method": "Percentile; zero-denominator draws omitted"},
                "model_forward_performed": False, "script_sha256": sha(Path(__file__))}
    output.mkdir(parents=True)
    csv_write(output / "all_778_cases.csv", all_cases)
    csv_write(output / "ranked_cases.csv", assignments)
    csv_write(output / "bins.csv", bins)
    write(output / "bins.json", bins)
    write(output / "metadata.json", metadata)
    np.savez_compressed(output / "bootstrap_pair_weights.npz", pair_ids=np.asarray(pair_ids), weights=weights, seed=np.asarray(args.seed))
    code = output / "code"
    code.mkdir()
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    print(json.dumps({"stage": "ranked_and_binned", "bin_rows": len(bins), "ranked_rows": len(assignments), "baseline_asr": 231/752}), flush=True)


def range_labels(low, high):
    for precision in range(3, 14):
        result = (f"{100 * low:.{precision}g}%", f"{100 * high:.{precision}g}%")
        if result[0] != result[1] or low == high:
            return result
    return result


def draw(ax, rows, role, baseline, small=False):
    color = COLORS[role]
    dense = len(rows) > 20
    x = np.asarray([r["percentile_midpoint"] / 100 for r in rows])
    y = np.asarray([r["asr"] for r in rows])
    ax.set_axisbelow(True)
    ax.grid(color="#dce1e6", linewidth=.8)
    ax.axhline(baseline, color="#737b85", linestyle="--", linewidth=1.1)
    ax.fill_between(x, [r["ci95_low"] for r in rows], [r["ci95_high"] for r in rows], color=color, alpha=.20, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1.9 if small else 2.4, marker="o", markersize=3.7 if small else 5, zorder=3)
    for i, row in enumerate([] if small and dense else rows):
        low, high = range_labels(row["probability_min"], row["probability_max"])
        label = f"[{low},\n{high}]"
        if row["unscored_cases"]:
            label += f"\nscored {row['scored_cases']}/{row['ranked_cases']}"
        if y[i] < .17:
            offset = 25 if i % 2 == 0 else 56
        elif y[i] > .82:
            offset = -30 if i % 2 == 0 else -59
        else:
            offset = 29 if i % 2 == 0 else -45
        if small:
            offset *= .75
        ax.annotate(label, (x[i], y[i]), xytext=(0, offset), textcoords="offset points", ha="center",
                    va="bottom" if offset > 0 else "top", fontsize=6.6 if small else (6.8 if dense else 9.1), linespacing=1.08,
                    bbox={"boxstyle": "round,pad=.20", "facecolor": "white", "edgecolor": color, "linewidth": .6, "alpha": .96},
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": .65, "shrinkA": 2, "shrinkB": 4},
                    zorder=5, annotation_clip=False)
    ax.set(xlim=(0, 1), ylim=(-.10, 1.13), xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(labelsize=10 if small else 12)


def legend(role, baseline):
    color = COLORS[role]
    return [Line2D([0], [0], color=color, lw=2.2, marker="o", label="Observed bin ASR (scored cases)"),
            Line2D([0], [0], color=color, alpha=.22, lw=9, label="95% paired-bootstrap interval"),
            Line2D([0], [0], color="#737b85", linestyle="--", label=f"Ranked scored-case ASR: {baseline:.2%}")]


def plots(args):
    output = args.output.resolve()
    metadata, bins = read(output / "metadata.json"), read(output / "bins.json")
    baseline = metadata["ranked_scored_asr"]
    group_count = metadata["bins_per_role"]
    size_min, size_max = min(metadata["bin_sizes"]), max(metadata["bin_sizes"])
    dense = group_count > 20
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12, "axes.linewidth": 1.,
                         "savefig.dpi": 180, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for part in PARTS:
        subject = "payload template (excluding goal)" if part == "template" else "attacker goal"
        target = output / part
        target.mkdir(exist_ok=True)
        part_bins = [r for r in bins if r["part"] == part]
        csv_write(target / "bins.csv", part_bins)
        by_role = {role: [r for r in part_bins if r["role"] == role] for role in ROLE_ORDER}
        for role in ROLE_ORDER:
            fig, ax = plt.subplots(figsize=(34, 9) if dense else (24, 8.5))
            fig.subplots_adjust(left=.06, right=.98, bottom=.25, top=.80)
            draw(ax, by_role[role], role, baseline)
            ax.set_xlabel(f"{LABELS[role]} of {subject}, as percentile of ranked cases", fontsize=16, labelpad=12)
            ax.set_ylabel("Attack success rate", fontsize=16, labelpad=12)
            fig.suptitle(f"{LABELS[role]}: attack success vs. role-probe probability", fontsize=24, y=.97)
            fig.text(.5, .90, f"plain + Remember pooled | First complete payload: {subject}", ha="center", fontsize=15)
            fig.text(.5, .85, f"{group_count} equal-count rank groups | {size_min}-{size_max} cases per group | Labels: probability [min, max] (%)", ha="center", fontsize=12)
            fig.legend(handles=legend(role, baseline), loc="lower left", bbox_to_anchor=(.06, .105), ncol=3, frameon=False, fontsize=11)
            fig.text(.06, .075, "Latest mixed-data probe: MLP256, seed 43, epoch 12 | SHA256 fe602209cab6", fontsize=10.5, color="#424951")
            fig.text(.06, .045, "755 ranked; 752 scored; 231 successes. 23 missing probabilities; 3 native null scores kept in ranks, excluded from ASR.", fontsize=10.5, color="#424951")
            note = "Ascending episode token-mean probability; ties by case_id. X is rank percentile. Bootstrap resamples 389 paired task/injection cases."
            if part == "template" and role == "tool":
                note = "120 cases have P(tool)=1 exactly; tied ranks use case_id. ASR changes inside this plateau do not reflect different tool probabilities."
            fig.text(.06, .015, note, fontsize=10.5, color="#535c66")
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(target / f"{role}_asr.{suffix}", bbox_inches="tight")
            plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(24, 17) if dense else (30, 22))
        fig.subplots_adjust(left=.065, right=.98, bottom=.15 if dense else .14, top=.87, hspace=.37 if dense else .48, wspace=.20)
        for ax, role in zip(axes.flat, ROLE_ORDER):
            draw(ax, by_role[role], role, baseline, small=True)
            ax.set_title(f"{LABELS[role]}: attack success vs. role-probe probability", fontsize=16 if dense else 18, pad=29)
            subtitle = f"{group_count} rank groups | probability ranges in individual figures / CSV" if dense else f"{group_count} rank groups | labels: probability [min, max] (%)"
            ax.text(.5, 1.035, subtitle, transform=ax.transAxes, ha="center", fontsize=9 if dense else 10)
            ax.set_xlabel("Percentile of cases sorted by role-probe probability", fontsize=12, labelpad=8)
            ax.set_ylabel("Attack success rate", fontsize=12)
        fig.suptitle(f"ASR by role-probe rank: {subject}", fontsize=28, y=.985)
        fig.text(.5, .94, f"plain + Remember | {group_count} equal-count groups per role | Latest mixed-data probe", ha="center", fontsize=18 if dense else 21)
        fig.legend(handles=legend("cot", baseline), loc="lower center", bbox_to_anchor=(.5, .084), ncol=3, frameon=False, fontsize=13)
        fig.text(.065, .065, f"755 ranked ({size_min}-{size_max}/group); 752 scored; 231 attack successes. 23 cases without probability cannot be ranked; 3 null outcomes excluded from ASR.", fontsize=11 if dense else 13)
        fig.text(.065, .043, "Probe: MLP256, seed 43, epoch 12. Shading: 95% paired bootstrap (10,000 draws); both conditions of each base case sampled together; rank bins fixed.", fontsize=10 if dense else 12)
        note = "Ascending episode-level token-mean probability; ties by case_id. X axis is rank percentile; individual figure labels show probability ranges (%)."
        if part == "template":
            note += " 120 cases have P(tool)=1 exactly."
        fig.text(.065, .021, note, fontsize=9.5 if dense else 11.5, color="#535c66")
        for suffix in ("png", "pdf"):
            fig.savefig(target / f"overview.{suffix}", bbox_inches="tight")
        plt.close(fig)
        print(json.dumps({"stage": "plotted", "part": part, "individual_figures": 6}), flush=True)
    quotient, remainder = divmod(755, group_count)
    size_description = f"前 {remainder} 组各 {quotient + 1} 个，后 {group_count - remainder} 组各 {quotient} 个" if remainder else f"每组 {quotient} 个"
    lines = [f"# 最新 probe：plain＋Remember 合并后按六类概率排序的 {group_count} 组 ASR", "",
             "使用最新混合训练 probe `mlp256-seed43`、第 12 轮 checkpoint，SHA256 `" + LATEST_SHA + "`。", "",
             "分别对 template（除 goal 外的 payload）和 goal，使用每个 case 内 token 概率的平均值；plain 与 Remember 合并，每个 role 单独升序排序，同概率按 case_id 排序。", "",
             f"755 个有概率的 case 均分成 {group_count} 组：{size_description}。3 个无评分 case 保留排序位置，不计入 ASR 分母；23 个未暴露 payload 的 case 无概率，不能排序。分组 ASR＝组内明确攻击成功数／组内有评分数。", "",
             "虚线为有概率、有评分 cohort 的 ASR：231／752＝30.7181%。ASR 沿用原始攻击轨迹结果，本次只是按新 probe 概率重排。", "",
             "阴影为 10,000 次配对 bootstrap 的 95% 百分位区间（seed=20261028）。在 389 个基础 suite／user_task／injection_task 组合上重采样，两种 payload 成对取样，每次保持已有排序和分组不变。", "",
             "template 的 tool 概率有 120 个 case 恰为 1，按 case_id 确定并列顺序。这些并列 case 跨越的组间 ASR 差异不代表 tool 概率不同；各组 plain／Remember 构成保留在 bins.csv。", "",
             "- template/、goal/：各 6 张单图（PNG／PDF／SVG），以及六图总览 overview.png／overview.pdf。",
             f"- bins.csv：{12 * group_count} 组的概率范围、人数、成功数、ASR、置信区间及条件构成。",
             "- ranked_cases.csv：9,060 行排序明细。",
             "- all_778_cases.csv：全部 case 的概率可用性和原生评分，缺失不填零。",
             "- bootstrap_pair_weights.npz：准确复现置信区间的配对重采样权重。", "",
             "| Role | Template | Goal |", "|---|---|---|"]
    for role in ROLE_ORDER:
        lines.append(f"| {role} | [PNG](template/{role}_asr.png) | [PNG](goal/{role}_asr.png) |")
    if dense:
        lines += ["", "六图总览省略密集的概率范围标注，12 张单图保留全部 50 组的概率范围。"]
    (output / "README.zh-CN.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "data", "plots"), default="all")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bins", type=int, choices=(20, 50), default=20)
    args = parser.parse_args()
    require(args.bootstrap == 10000, "Figures declare 10,000 paired bootstrap draws")
    if args.stage in ("all", "data"):
        data(args)
    if args.stage in ("all", "plots"):
        plots(args)


if __name__ == "__main__":
    main()
