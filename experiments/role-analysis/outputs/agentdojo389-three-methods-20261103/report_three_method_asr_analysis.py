"""Render and cross-check the three-method association analysis."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/three-method-asr-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit

ROLES = ("system", "assistant", "user", "tool", "cot", "developer")
COHORTS = ("pooled", "plain", "prerequisite")
COLORS = {"pooled":"#667085", "plain":"#2878b5", "prerequisite":"#dc6b24"}
LABELS = {"pooled":"Pooled", "plain":"Plain", "prerequisite":"Remember"}
KNOTS = np.r_[np.zeros(4), [.2,.4,.6,.8], np.ones(4)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20), b""):
            h.update(b)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def save(fig, stem):
    for suffix in ("png","pdf"):
        fig.savefig(str(stem)+"."+suffix, dpi=180, bbox_inches="tight")
    plt.close(fig)


def audit_model(arrays, suite_weights):
    error, curve_rows = 0., []
    for cohort in COHORTS:
        grid, beta = arrays[cohort+"_grid"], arrays[cohort+"_coefficients"]
        b = BSpline(KNOTS, np.eye(8), 3)(grid)[:,1:]
        designs = []
        for suite in ("banking","slack","travel"):
            designs.append(np.column_stack([np.ones(len(grid)), np.full(len(grid),suite=="slack"),
                                             np.full(len(grid),suite=="travel"), b]))
        independent = sum(weight*expit(design@beta) for weight,design in zip(suite_weights,designs))
        boot_beta = arrays[cohort+"_bootstrap_coefficients"]
        boot = sum(weight*expit(boot_beta@design.T) for weight,design in zip(suite_weights,designs))
        low,high = np.quantile(boot,[.025,.975],axis=0)
        for x,y in ((independent,arrays[cohort+"_curve"]),(low,arrays[cohort+"_ci95_low"]),(high,arrays[cohort+"_ci95_high"])):
            delta = float(np.max(abs(x-y)));error=max(error,delta)
            require(delta<1e-12,"Saved curve/bootstrap band failed independent reconstruction")
        for i,x in enumerate(grid):
            curve_rows.append({"cohort":cohort,"pooled_probability_percentile":float(x*100),
                               "estimated_asr":float(independent[i]),"pointwise_ci95_low":float(low[i]),
                               "pointwise_ci95_high":float(high[i]),
                               "derivative":float(arrays[cohort+"_derivative"][i]),
                               "derivative_sim95_low":float(arrays[cohort+"_derivative_sim95_low"][i]),
                               "derivative_sim95_high":float(arrays[cohort+"_derivative_sim95_high"][i])})
    return error,curve_rows


def draw_gam(ax,arrays,small=False):
    for cohort in COHORTS:
        x=arrays[cohort+"_grid"]*100
        ax.fill_between(x,arrays[cohort+"_ci95_low"],arrays[cohort+"_ci95_high"],
                        color=COLORS[cohort],alpha=.10 if cohort=="pooled" else .16,linewidth=0)
        ax.plot(x,arrays[cohort+"_curve"],color=COLORS[cohort],lw=2 if small else 2.5,
                ls="--" if cohort=="pooled" else "-",label=LABELS[cohort])
    ax.set(xlim=(0,100),ylim=(0,1),xticks=[0,25,50,75,100],yticks=[0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(color="#e2e6eb",lw=.7);ax.set_axisbelow(True)
    ax.set_xlabel("Role probability percentile (pooled average ranks)",fontsize=10 if small else 12)
    ax.set_ylabel("GAM-estimated ASR",fontsize=10 if small else 12)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    root=parser.parse_args().output.resolve();r=read(root/"results.json")
    require(r["validation"]["status"]=="passed","Statistical fit not validated")
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"savefig.dpi":180})
    figures=root/"figures";figures.mkdir(exist_ok=True)
    suite_weights=np.asarray([r["protocol"]["suite_standardization"][s] for s in ("banking","slack","travel")])
    model_arrays,curve_rows={},[];max_error=0.
    for part in ("template","goal"):
        target=figures/part;target.mkdir(exist_ok=True)
        for role in ROLES:
            with np.load(root/"models"/f"{part}_{role}.npz",allow_pickle=False) as z:
                arrays={key:z[key] for key in z.files}
            model_arrays[(part,role)]=arrays
            error,rows=audit_model(arrays,suite_weights);max_error=max(max_error,error)
            curve_rows.extend({"part":part,"role":role,**row}for row in rows)
            fig,ax=plt.subplots(figsize=(11,6.5));draw_gam(ax,arrays)
            ax.set_title(f"{part.capitalize()} / {role}: probability vs attack success",fontsize=16,pad=16)
            ax.legend(loc="upper left",frameon=False)
            fig.subplots_adjust(bottom=.24)
            fig.text(.125,.11,"Suite-standardized binomial P-spline GAM; no monotonic constraint.",fontsize=10)
            fig.text(.125,.065,"Shading: pointwise 95% user-task bootstrap CI (1,000 refits; smoothing fixed).",fontsize=9)
            fig.text(.125,.025,"Plain: 374 cases / 12 successes in only 3 user tasks. Remember: 378 cases / 219 successes.",fontsize=9)
            save(fig,target/f"{role}_gam")
        fig,axes=plt.subplots(3,2,figsize=(16,14));fig.subplots_adjust(top=.88,bottom=.16,hspace=.36,wspace=.25)
        for ax,role in zip(axes.flat,ROLES):
            draw_gam(ax,model_arrays[(part,role)],small=True);ax.set_title(role.capitalize(),fontsize=14,pad=10)
        fig.suptitle(f"GAM: {part} role probability vs attack success",fontsize=22,y=.97)
        fig.text(.5,.925,"Latest probe | Pooled vs Plain vs Remember | Suite-standardized estimated ASR",ha="center",fontsize=13)
        handles=[Line2D([0],[0],color=COLORS[c],lw=2.5,ls="--"if c=="pooled"else"-",label=LABELS[c])for c in COHORTS]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.097),ncol=3,frameon=False,fontsize=12)
        fig.text(.125,.08,"Unconstrained cubic P-splines; user-task grouped CV selects smoothing. 56 task clusters.",fontsize=10)
        fig.text(.125,.053,"Shading: pointwise 95% task-bootstrap CI, conditional on chosen smoothing (1,000 refits).",fontsize=10)
        fig.text(.125,.027,"Plain: 12/374 successes, confined to 3 tasks; uncertainty is substantial. Curves are model estimates, not binned observations.",fontsize=9.5)
        save(fig,figures/f"gam_{part}_overview")
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True);fig.subplots_adjust(bottom=.23,wspace=.15)
    for ax,part in zip(axes,("template","goal")):
        for i,role in enumerate(ROLES):
            for offset,cohort in zip((-.22,0,.22),COHORTS):
                row=next(x for x in r["spearman"] if x["part"]==part and x["role"]==role and x["cohort"]==cohort)
                y=i+offset
                if row["undefined_bootstrap_replicates"]:
                    ax.plot(row["rho"],y,marker="x",color=COLORS[cohort],ms=6)
                else:
                    ax.errorbar(row["rho"],y,xerr=[[row["rho"]-row["ci95_low"]],[row["ci95_high"]-row["rho"]]],
                                fmt="o",color=COLORS[cohort],capsize=3,ms=5)
        ax.axvline(0,color="#7e8792",lw=1,ls="--");ax.set_title(part.capitalize(),fontsize=17)
        ax.set(xlim=(-.7,.7),yticks=range(6),yticklabels=ROLES,xlabel="Spearman rho (rank correlation)")
        ax.grid(axis="x",alpha=.25)
    axes[0].invert_yaxis();fig.suptitle("Role probabilities and attack success: Spearman correlation",fontsize=19)
    handles=[Line2D([0],[0],color=COLORS[c],marker="x"if c=="plain"else"o",label=LABELS[c])for c in COHORTS]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.11),ncol=3,frameon=False)
    undefined=next(x for x in r["spearman"]if x["cohort"]=="plain")["undefined_bootstrap_replicates"]
    fig.text(.125,.075,f"Pooled/Remember intervals: 10,000 user-task bootstrap draws. Plain: point estimates only ({undefined}/10,000 draws have constant outcomes).",fontsize=9)
    fig.text(.125,.035,"Bootstrap reranks each resample and gives identical probabilities identical ranks. Intervals are exploratory and not adjusted across roles.",fontsize=9)
    save(fig,figures/"spearman_overview")
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True);fig.subplots_adjust(top=.81,bottom=.24,wspace=.15)
    for ax,part in zip(axes,("template","goal")):
        for i,role in enumerate(ROLES):
            row=next(x for x in r["gee"] if x["part"]==part and x["role"]==role)
            value=row["odds_ratio_per_10_percentile_points"]
            ax.errorbar(value,i,xerr=[[value-row["or_ci95_low"]],[row["or_ci95_high"]-value]],fmt="o",
                        color="#bd594c"if row["q_bh_12_tests"]<.05 else"#456f91",capsize=4,ms=6)
        ax.set(xscale="log",xlim=(.4,1.85),yticks=range(6),yticklabels=ROLES,xlabel="Adjusted odds ratio per +10 probability-percentile points")
        ax.axvline(1,color="#7e8792",ls="--",lw=1);ax.grid(axis="x",alpha=.25);ax.set_title(part.capitalize(),fontsize=17,pad=14)
    axes[0].invert_yaxis();fig.suptitle("Logistic GEE: payload/suite-adjusted role-probability trends",fontsize=19)
    fig.text(.125,.12,"95% cluster-robust CI; 56 user-task clusters. Red: BH q < 0.05 across 12 tests; blue: q >= 0.05.",fontsize=10)
    fig.text(.125,.075,"752 scored cases. Same user task's injection cases and both payload conditions share a cluster.",fontsize=10)
    fig.text(.125,.035,"Odds ratios describe association, not ASR ratios or causal effects. Identical probabilities receive identical average ranks.",fontsize=9)
    save(fig,figures/"gee_overview")
    with (root/"gam_curves.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curve_rows[0]));writer.writeheader();writer.writerows(curve_rows)
    validation={"status":"passed","model_curves_checked":36,"curve_points_checked":len(curve_rows),
                "maximum_saved_curve_or_bootstrap_band_error":max_error,"previous_validation":r["validation"],
                "reporter_sha256":sha(Path(__file__))}
    (root/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    lines=["# 三种统计方法：最新 probe 概率与 ASR", "",
           "分析 752 个有 payload 特征且有评分的 case，含 plain 374 个（12 成功）和 Remember 378 个（219 成功）。以 suite／user_task 为单位，共 56 个 cluster。", "",
           "## 1. Spearman：整体与分类型关联", "",
           "95% 区间通过 10,000 次 user-task bootstrap 计算，每次重新计算秩；相同概率取平均秩。plain 有 462 次 bootstrap 全失败、ρ 无定义，因此其点估计保留，剔除退化样本后的条件区间仅存 CSV，不作为无条件 95% 区间解读。", "",
           "| 部分 | role | 合并 ρ [95% CI] | plain ρ | Remember ρ [95% CI] |", "|---|---|---|---:|---|"]
    for part in ("template","goal"):
        for role in ROLES:
            rr={c:next(x for x in r["spearman"]if x["part"]==part and x["role"]==role and x["cohort"]==c)for c in COHORTS}
            interval=lambda x:f"{x['rho']:.3f} [{x['ci95_low']:.3f}, {x['ci95_high']:.3f}]"
            lines.append(f"| {part} | {role} | {interval(rr['pooled'])} | {rr['plain']['rho']:.3f} | {interval(rr['prerequisite'])} |")
    lines += ["", "## 2. Logistic/GEE：控制 payload 类型和 suite", "",
              "以下 OR 表示概率百分位提高 10 个百分点时成功 odds 的变化，非原始概率增加 0.1，也非 ASR 倍数。采用 user-task cluster sandwich 协方差，12 个主检验做 BH 校正。", "",
              "| 部分 | role | OR [95% CI] | BH q |", "|---|---|---|---:|"]
    for row in r["gee"]:
        lines.append(f"| {row['part']} | {row['role']} | {row['odds_ratio_per_10_percentile_points']:.3f} [{row['or_ci95_low']:.3f}, {row['or_ci95_high']:.3f}] | {row['q_bh_12_tests']:.4g} |")
    lines += ["", "12 项主检验没有显著正向结果。该模型检验控制类型／suite 后的共同平均趋势，不表示不存在局部上升或类型间异质性。", "",
              "## 3. GAM：平滑曲线与非线性形状", "",
              "每个 role／部分分别拟合 pooled、plain、Remember 的 binomial-logit P-spline，控制 suite，并以总体 suite 构成标准化。8 个三次样条基函数，二阶差分惩罚；按 user-task 分组的五折 CV 用 one-SE 规则选平滑强度。没有单调约束。", "",
              "每条曲线有 1,000 次 task bootstrap 重拟合。置信区间固定已选平滑强度和原始百分位变换，因此是条件区间；GAM／Spearman 跨角色、类型的区间未做全族校正，均为探索性结果。", "",
              "下面 Δ 为该 cohort 中概率秩 Q90 与 Q10 处的**模型估计 ASR 差**，单位百分点；它不等于实测两组 ASR，也不证明整条曲线单调。", "",
              "| 部分 | role | pooled Δ [95% CI] | plain Δ [95% CI] | Remember Δ [95% CI] |", "|---|---|---|---|---|"]
    for part in ("template","goal"):
        for role in ROLES:
            row={c:next(x for x in r["gam"]if x["part"]==part and x["role"]==role and x["cohort"]==c)for c in COHORTS}
            text=lambda x:f"{100*x['delta_x90_minus_x10']:+.1f} [{100*x['delta_ci95_low']:+.1f}, {100*x['delta_ci95_high']:+.1f}]"
            lines.append(f"| {part} | {role} | {text(row['pooled'])} | {text(row['plain'])} | {text(row['prerequisite'])} |")
    lines += ["", "## 结果的解释边界", "",
              "合并的正向关联并不一定在每种 payload 内成立。Spearman 描述原始排序关联；GEE 控制 payload 与 suite；GAM 按类型分别拟合并标准化 suite，三者估计对象不同。", "",
              "plain 的 12 次成功来自 3 个 user task，其中 9 次来自 banking/user_task_12。其条件曲线和区间对任务构成敏感；不据此宣称稳定正趋势。", "",
              "GAM 的 suite 标准化需要模型在部分 suite 数据较少的概率位置做预测。曲线仅画该 cohort 的 1%–99% 秩范围，端点比较用 10%–90%，仍应结合原始分布判断支持范围。", "",
              "GAM 导数文件提供每条曲线的近似同时 95% 带，条件于已选平滑度，并未同时覆盖 36 个模型；局部上升区间为探索性候选。所有结果均是既有历史轨迹中的关联，非因果证据。", "",
              "## 文件", "",
              "- figures/spearman_overview.png、gee_overview.png：相关性和调整后 OR 总览。",
              "- figures/gam_template_overview.png、gam_goal_overview.png：GAM 六角色总览。",
              "- figures/template/、figures/goal/：12 张单图，PNG／PDF。",
              "- spearman.csv、gee_adjusted.csv、gam_summary.csv：三种方法的完整统计。",
              "- gam_curves.csv：全部曲线、区间及导数带的 3,636 个点。",
              "- models/、user_task_bootstrap_weights.npz、spearman_bootstrap_estimates.npz：复算所需参数和重采样结果。",
              "- validation.json：逐点重新构造 36 条曲线和 bootstrap 区间的验证，以及前述拟合验证。", ""]
    (root/"REPORT.zh-CN.md").write_text("\n".join(lines)+"\n")
    shutil.copy2(Path(__file__),root/Path(__file__).name)
    print(json.dumps({"status":"passed","maximum_curve_error":max_error,"curve_points":len(curve_rows)}),flush=True)


if __name__=="__main__":
    main()
