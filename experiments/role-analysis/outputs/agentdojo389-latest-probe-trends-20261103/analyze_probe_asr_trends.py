"""Exploratory unbinned role-probability/attack-outcome associations.

Primary analysis: one binomial-logit GEE per part/role, adjusting for payload
condition and suite. Independent working correlation, user-task cluster-robust
covariance, average ranks for ties, and BH correction across 12 primary slopes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
PARTS = {"template": "payload_without_goal", "goal": "goal"}
LATEST_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(y, x, groups):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = sm.GEE(y, x, groups=groups, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Independence()).fit(maxiter=200, ctol=1e-10, cov_type="robust")
    require(result.converged and np.isfinite(result.params).all(), "GEE failed to converge")
    index = list(x.columns).index("rank10")
    coefficient, se, p = (float(v[index]) for v in (result.params, result.bse, result.pvalues))
    lower, upper = np.asarray(result.conf_int())[index]
    # Independence-working GEE coefficients must agree with binomial GLM MLE.
    reference = sm.GLM(y, x, family=sm.families.Binomial()).fit(tol=1e-11, maxiter=200)
    parameter_error = float(np.max(abs(np.asarray(reference.params) - np.asarray(result.params))))
    require(parameter_error < 1e-5, "Independent GLM coefficient check failed")
    # Independently reconstruct the cluster sandwich from score contributions.
    design, mu = np.asarray(x), np.asarray(result.fittedvalues)
    bread = np.linalg.inv(design.T @ ((mu * (1-mu))[:, None] * design))
    _, inverse = np.unique(groups, return_inverse=True)
    cluster_scores = np.zeros((int(inverse.max())+1, design.shape[1]))
    np.add.at(cluster_scores, inverse, design * (y-mu)[:, None])
    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread
    covariance_error = float(np.max(abs(covariance - np.asarray(result.cov_params()))))
    require(np.allclose(covariance, result.cov_params(), atol=1e-7, rtol=1e-5), "Independent cluster covariance check failed")
    return {"coefficient_per_rank_decile": coefficient, "robust_se": se,
            "odds_ratio_per_10_percentile_points": float(np.exp(coefficient)),
            "or_ci95_low": float(np.exp(lower)), "or_ci95_high": float(np.exp(upper)),
            "p_two_sided": p, "converged": bool(result.converged),
            "maximum_glm_parameter_error": parameter_error, "maximum_covariance_error": covariance_error,
            "warnings": [str(w.message) for w in recorded]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.predictions.resolve(), args.output.resolve()
    require(not output.exists(), "Use a fresh output directory")
    require(read(source / "summary.json")["probe_sha256"] == LATEST_SHA, "Wrong probe probabilities")
    require(read(source / "validation.json")["status"] == "passed", "Predictions not validated")
    source_hashes = dict(line.split("  ", 1)[::-1] for line in (source / "SHA256SUMS").read_text().splitlines())
    for name in ("per_case_probabilities.csv", "summary.json", "validation.json"):
        require(sha(source / name) == source_hashes[name], "Probability artifact changed")
    data = pd.read_csv(source / "per_case_probabilities.csv", keep_default_na=False)
    primary, exploratory, descriptions = [], [], []
    native_counts = None
    for part, stored in PARTS.items():
        all_part = data[data.part == stored].copy()
        require(len(all_part) == 755, "Wrong payload coverage")
        scored = all_part[all_part.attack_success.astype(str) != ""].copy()
        require(len(scored) == 752, "Wrong scored cohort")
        scored["y"] = (scored.attack_success.astype(str) == "True").astype(int)
        groups = np.asarray(["/".join(key.split("/")[1:3]) for key in scored.case_id])
        require(len(np.unique(groups)) == 56 and scored.y.sum() == 231, "Wrong outcome/user-task clusters")
        y = scored.y.to_numpy()
        native_counts = {c: {"cases": int((scored.condition == c).sum()),
                             "successes": int(scored.loc[scored.condition == c, "y"].sum())}
                         for c in ("plain", "prerequisite")}
        for role in ROLES:
            probabilities = scored["p_" + role].to_numpy(float)
            # Identical probabilities receive identical ranks. No case-ID jitter.
            rank10 = (rankdata(probabilities, method="average") - .5) / len(scored) * 10
            x = pd.DataFrame({"intercept": np.ones(len(scored)), "rank10": rank10,
                              "remember": (scored.condition == "prerequisite").to_numpy(float),
                              "suite_slack": (scored.suite == "slack").to_numpy(float),
                              "suite_travel": (scored.suite == "travel").to_numpy(float)})
            adjusted = fit(y, x, groups)
            unadjusted = fit(y, x[["intercept", "rank10"]], groups)
            primary.append({"part": part, "role": role, **adjusted})
            exploratory.append({"part": part, "role": role, **unadjusted})
            for condition in ("pooled", "plain", "prerequisite"):
                mask = np.ones(len(scored), bool) if condition == "pooled" else (scored.condition == condition).to_numpy()
                p, outcomes = probabilities[mask], y[mask]
                rho = float(spearmanr(p, outcomes).statistic)
                ranks = rankdata(p, method="average")
                positives, negatives = int(outcomes.sum()), int((1-outcomes).sum())
                auc = float((ranks[outcomes == 1].sum() - positives * (positives+1) / 2) / (positives*negatives))
                descriptions.append({"part": part, "role": role, "condition": condition, "cases": len(p),
                                     "successes": positives, "spearman_rho": rho, "descriptive_auc": auc})
            print(json.dumps({"part": part, "role": role,
                              "or": adjusted["odds_ratio_per_10_percentile_points"], "p": adjusted["p_two_sided"]}), flush=True)
    reject, q, _, _ = multipletests([r["p_two_sided"] for r in primary], method="fdr_bh")
    for row, significant, adjusted_p in zip(primary, reject, q):
        row["q_bh_12_tests"] = float(adjusted_p)
        row["positive_after_bh_05"] = bool(significant and row["coefficient_per_rank_decile"] > 0)
        row["negative_after_bh_05"] = bool(significant and row["coefficient_per_rank_decile"] < 0)
    output.mkdir(parents=True)
    csv_write(output / "adjusted_gee.csv", primary)
    csv_write(output / "unadjusted_gee.csv", exploratory)
    csv_write(output / "descriptive_correlations.csv", descriptions)
    protocol = {"probe_sha256": LATEST_SHA, "cases": 752, "user_task_clusters": 56, "native_condition_counts": native_counts,
                "primary_model": "logit(P(attack_success)) = intercept + rank10 + Remember + suite_slack + suite_travel",
                "predictor": "Average rank among the scored cohort, scaled so one unit is 10 percentile points; ties receive the same rank",
                "inference": "GEE, independent working correlation, user-task cluster-robust sandwich covariance",
                "cluster": "suite/user_task; all injection tasks and both payload conditions retained together",
                "multiplicity": "Two-sided Wald p-values, BH over exactly 12 primary part/role slopes",
                "secondary": "Unadjusted GEE, pooled/stratified descriptive Spearman and AUC; no significance claim from naive Spearman p-values",
                "scope": "Exploratory post-hoc within observed historical payload cohort; association, not causal evidence or strict monotonicity proof",
                "probabilities_root": str(source), "input_csv_sha256": sha(source / "per_case_probabilities.csv"),
                "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                            "pandas": pd.__version__, "statsmodels": statsmodels.__version__},
                "script_sha256": sha(Path(__file__))}
    result = {"protocol": protocol, "primary_results": primary, "unadjusted_results": exploratory,
              "descriptive_results": descriptions,
              "validation": {"status": "passed", "all_models_converged": True,
                             "glm_parameters_independently_checked": True, "cluster_sandwich_independently_reconstructed": True,
                             "max_parameter_error": max(r["maximum_glm_parameter_error"] for r in primary+exploratory),
                             "max_covariance_error": max(r["maximum_covariance_error"] for r in primary+exploratory)}}
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 最新 probe 概率与攻击成功的探索性趋势分析", "",
             "主分析不进行分箱。对 12 个 template／goal × role 分别做 binomial-logit GEE，控制 payload 类型和 suite；以 suite／user_task 的 56 个 cluster 估计稳健标准误。", "",
             "概率转换为平均秩百分位，相同概率取相同秩；每增加 10 个百分位的成功 odds ratio（OR）如下。OR 不等于 ASR 倍数，且正斜率不证明整条实际关系严格单调。", "",
             "| 部分 | role | OR | 95% CI | BH q（12 项） |", "|---|---|---:|---|---:|"]
    for row in primary:
        lines.append(f"| {row['part']} | {row['role']} | {row['odds_ratio_per_10_percentile_points']:.3f} | [{row['or_ci95_low']:.3f}, {row['or_ci95_high']:.3f}] | {row['q_bh_12_tests']:.4g} |")
    lines += ["", "plain：12／374；Remember：219／378。直接混合的概率—ASR 关联可能受 payload 类型构成影响，主分析显式控制该变量。", "",
              "同一个 user task 下的多个 injection_task 及两种 payload 都在同一 cluster，比只配对同一 injection_task 的推断单位更粗。23 个无 payload 特征、3 个无评分 case 未进入回归。", "",
              "所有 24 个 GEE 模型均收敛，并以独立 GLM 拟合检验参数、手工重建 user-task cluster sandwich 协方差。", "",
              "结果为已经观察过这些图后的探索性分析。控制 payload 和 suite 不能排除其他混杂；6 个概率相加为 1，每次只解释其中一个 role，结果不代表单独干预该概率的因果效果。", "",
              "`adjusted_gee.csv` 为主结果；`unadjusted_gee.csv` 为未控制类型／suite 的对照；`descriptive_correlations.csv` 保存分类型的 Spearman 与 AUC，未使用默认 Spearman p 值做推断。"]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    print(json.dumps({"validation": result["validation"], "positive": [(r["part"],r["role"]) for r in primary if r["positive_after_bh_05"]]}), flush=True)


if __name__ == "__main__":
    main()
