# 最新 probe 概率与攻击成功的探索性趋势分析

主分析不进行分箱。对 12 个 template／goal × role 分别做 binomial-logit GEE，控制 payload 类型和 suite；以 suite／user_task 的 56 个 cluster 估计稳健标准误。

概率转换为平均秩百分位，相同概率取相同秩；每增加 10 个百分位的成功 odds ratio（OR）如下。OR 不等于 ASR 倍数，且正斜率不证明整条实际关系严格单调。

| 部分 | role | OR | 95% CI | BH q（12 项） |
|---|---|---:|---|---:|
| template | system | 0.827 | [0.710, 0.964] | 0.03606 |
| template | developer | 0.748 | [0.613, 0.912] | 0.01657 |
| template | user | 1.280 | [0.964, 1.700] | 0.1315 |
| template | cot | 0.730 | [0.591, 0.902] | 0.01657 |
| template | assistant | 1.020 | [0.786, 1.323] | 0.8846 |
| template | tool | 0.767 | [0.546, 1.076] | 0.1658 |
| goal | system | 0.782 | [0.655, 0.934] | 0.02032 |
| goal | developer | 0.588 | [0.502, 0.688] | 4.136e-10 |
| goal | user | 1.131 | [0.986, 1.298] | 0.1315 |
| goal | cot | 1.021 | [0.894, 1.167] | 0.8254 |
| goal | assistant | 1.181 | [0.992, 1.407] | 0.1228 |
| goal | tool | 0.971 | [0.858, 1.100] | 0.7783 |

plain：12／374；Remember：219／378。直接混合的概率—ASR 关联可能受 payload 类型构成影响，主分析显式控制该变量。

同一个 user task 下的多个 injection_task 及两种 payload 都在同一 cluster，比只配对同一 injection_task 的推断单位更粗。23 个无 payload 特征、3 个无评分 case 未进入回归。

所有 24 个 GEE 模型均收敛，并以独立 GLM 拟合检验参数、手工重建 user-task cluster sandwich 协方差。

结果为已经观察过这些图后的探索性分析。控制 payload 和 suite 不能排除其他混杂；6 个概率相加为 1，每次只解释其中一个 role，结果不代表单独干预该概率的因果效果。

`adjusted_gee.csv` 为主结果；`unadjusted_gee.csv` 为未控制类型／suite 的对照；`descriptive_correlations.csv` 保存分类型的 Spearman 与 AUC，未使用默认 Spearman p 值做推断。
