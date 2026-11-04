# 最新 probe：plain＋Remember 合并后按六类概率排序的 20 组 ASR

使用最新混合训练 probe `mlp256-seed43`、第 12 轮 checkpoint，SHA256 `fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33`。

分别对 template（除 goal 外的 payload）和 goal，使用每个 case 内 token 概率的平均值；plain 与 Remember 合并，每个 role 单独升序排序，同概率按 case_id 排序。

755 个有概率的 case 均分成 20 组：前 15 组各 38 个，后 5 组各 37 个。3 个无评分 case 保留排序位置，不计入 ASR 分母；23 个未暴露 payload 的 case 无概率，不能排序。分组 ASR＝组内明确攻击成功数／组内有评分数。

虚线为有概率、有评分 cohort 的 ASR：231／752＝30.7181%。ASR 沿用原始攻击轨迹结果，本次只是按新 probe 概率重排。

阴影为 10,000 次配对 bootstrap 的 95% 百分位区间（seed=20261028）。在 389 个基础 suite／user_task／injection_task 组合上重采样，两种 payload 成对取样，每次保持已有排序和分组不变。

template 的 tool 概率有 120 个 case 恰为 1，按 case_id 确定并列顺序。这些并列 case 跨越的组间 ASR 差异不代表 tool 概率不同；各组 plain／Remember 构成保留在 bins.csv。

- template/、goal/：各 6 张单图（PNG／PDF／SVG），以及六图总览 overview.png／overview.pdf。
- bins.csv：240 组的概率范围、人数、成功数、ASR、置信区间及条件构成。
- ranked_cases.csv：9,060 行排序明细。
- all_778_cases.csv：全部 case 的概率可用性和原生评分，缺失不填零。
- bootstrap_pair_weights.npz：准确复现置信区间的配对重采样权重。

| Role | Template | Goal |
|---|---|---|
| system | [PNG](template/system_asr.png) | [PNG](goal/system_asr.png) |
| assistant | [PNG](template/assistant_asr.png) | [PNG](goal/assistant_asr.png) |
| user | [PNG](template/user_asr.png) | [PNG](goal/user_asr.png) |
| tool | [PNG](template/tool_asr.png) | [PNG](goal/tool_asr.png) |
| cot | [PNG](template/cot_asr.png) | [PNG](goal/cot_asr.png) |
| developer | [PNG](template/developer_asr.png) | [PNG](goal/developer_asr.png) |

独立验证全部通过：核验 778 个原始 case，使用独立 lexsort 重建 12 套排序、9,060 个分组归属和 240 组 ASR；重新生成配对 bootstrap 权重并复算全部置信区间，差异为 0。详见 `validation.json`。

[Template 六图总览](template/overview.png) · [Goal 六图总览](goal/overview.png)
