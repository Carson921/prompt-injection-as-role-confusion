# 三种统计方法：最新 probe 概率与 ASR

分析 752 个有 payload 特征且有评分的 case，含 plain 374 个（12 成功）和 Remember 378 个（219 成功）。以 suite／user_task 为单位，共 56 个 cluster。

## 1. Spearman：整体与分类型关联

95% 区间通过 10,000 次 user-task bootstrap 计算，每次重新计算秩；相同概率取平均秩。plain 有 462 次 bootstrap 全失败、ρ 无定义，因此其点估计保留，剔除退化样本后的条件区间仅存 CSV，不作为无条件 95% 区间解读。

| 部分 | role | 合并 ρ [95% CI] | plain ρ | Remember ρ [95% CI] |
|---|---|---|---:|---|
| template | system | -0.111 [-0.235, 0.044] | -0.040 | -0.306 [-0.468, -0.093] |
| template | assistant | -0.029 [-0.161, 0.119] | 0.177 | -0.178 [-0.339, 0.006] |
| template | user | 0.158 [0.018, 0.305] | 0.247 | -0.093 [-0.272, 0.086] |
| template | tool | -0.142 [-0.295, -0.000] | -0.232 | 0.122 [-0.057, 0.295] |
| template | cot | 0.067 [-0.053, 0.212] | 0.069 | -0.286 [-0.423, -0.120] |
| template | developer | -0.030 [-0.167, 0.140] | 0.085 | -0.345 [-0.480, -0.167] |
| goal | system | -0.038 [-0.160, 0.116] | -0.052 | -0.310 [-0.460, -0.116] |
| goal | assistant | 0.228 [0.105, 0.348] | 0.255 | -0.055 [-0.232, 0.131] |
| goal | user | 0.230 [0.101, 0.352] | 0.217 | -0.035 [-0.215, 0.141] |
| goal | tool | -0.193 [-0.321, -0.069] | -0.177 | 0.189 [0.014, 0.351] |
| goal | cot | 0.278 [0.168, 0.389] | 0.173 | -0.131 [-0.291, 0.039] |
| goal | developer | -0.082 [-0.194, 0.058] | -0.023 | -0.469 [-0.576, -0.333] |

## 2. Logistic/GEE：控制 payload 类型和 suite

以下 OR 表示概率百分位提高 10 个百分点时成功 odds 的变化，非原始概率增加 0.1，也非 ASR 倍数。采用 user-task cluster sandwich 协方差，12 个主检验做 BH 校正。

| 部分 | role | OR [95% CI] | BH q |
|---|---|---|---:|
| template | system | 0.827 [0.710, 0.964] | 0.03606 |
| template | developer | 0.748 [0.613, 0.912] | 0.01657 |
| template | user | 1.280 [0.964, 1.700] | 0.1315 |
| template | cot | 0.730 [0.591, 0.902] | 0.01657 |
| template | assistant | 1.020 [0.786, 1.323] | 0.8846 |
| template | tool | 0.767 [0.546, 1.076] | 0.1658 |
| goal | system | 0.782 [0.655, 0.934] | 0.02032 |
| goal | developer | 0.588 [0.502, 0.688] | 4.136e-10 |
| goal | user | 1.131 [0.986, 1.298] | 0.1315 |
| goal | cot | 1.021 [0.894, 1.167] | 0.8254 |
| goal | assistant | 1.181 [0.992, 1.407] | 0.1228 |
| goal | tool | 0.971 [0.858, 1.100] | 0.7783 |

12 项主检验没有显著正向结果。该模型检验控制类型／suite 后的共同平均趋势，不表示不存在局部上升或类型间异质性。

## 3. GAM：平滑曲线与非线性形状

每个 role／部分分别拟合 pooled、plain、Remember 的 binomial-logit P-spline，控制 suite，并以总体 suite 构成标准化。8 个三次样条基函数，二阶差分惩罚；按 user-task 分组的五折 CV 用 one-SE 规则选平滑强度。没有单调约束。

每条曲线有 1,000 次 task bootstrap 重拟合。置信区间固定已选平滑强度和原始百分位变换，因此是条件区间；GAM／Spearman 跨角色、类型的区间未做全族校正，均为探索性结果。

下面 Δ 为该 cohort 中概率秩 Q90 与 Q10 处的**模型估计 ASR 差**，单位百分点；它不等于实测两组 ASR，也不证明整条曲线单调。

| 部分 | role | pooled Δ [95% CI] | plain Δ [95% CI] | Remember Δ [95% CI] |
|---|---|---|---|---|
| template | system | -17.7 [-26.2, +4.4] | -8.2 [-35.7, -0.0] | -27.4 [-49.4, +10.7] |
| template | assistant | +8.8 [-16.6, +37.0] | -0.2 [-23.4, +10.7] | -9.3 [-44.0, +27.4] |
| template | user | +55.5 [+39.2, +72.5] | +4.3 [-21.6, +14.7] | +22.4 [-7.1, +45.4] |
| template | tool | -61.1 [-76.4, -47.4] | -3.9 [-14.4, +18.5] | -20.7 [-43.7, +11.1] |
| template | cot | +39.3 [+23.2, +56.2] | -19.0 [-33.5, -0.0] | -42.5 [-66.9, -4.1] |
| template | developer | -1.9 [-15.4, +22.6] | -12.0 [-35.0, -0.0] | -47.6 [-68.4, -16.2] |
| goal | system | -4.1 [-18.2, +27.8] | -11.4 [-35.4, -0.0] | -31.0 [-53.6, +5.6] |
| goal | assistant | +49.7 [+36.1, +63.5] | +7.4 [-0.0, +19.9] | +17.1 [-5.8, +37.0] |
| goal | user | +45.6 [+31.3, +60.0] | +6.3 [-0.2, +18.2] | +17.1 [-6.2, +35.5] |
| goal | tool | -48.1 [-62.1, -36.1] | -3.0 [-11.3, +2.8] | -2.6 [-21.9, +20.1] |
| goal | cot | +43.7 [+31.8, +58.0] | +3.8 [-2.1, +15.0] | -3.4 [-23.3, +15.7] |
| goal | developer | -11.8 [-24.8, +7.8] | -14.7 [-35.1, -0.0] | -68.4 [-83.4, -45.9] |

## 结果的解释边界

合并的正向关联并不一定在每种 payload 内成立。Spearman 描述原始排序关联；GEE 控制 payload 与 suite；GAM 按类型分别拟合并标准化 suite，三者估计对象不同。

plain 的 12 次成功来自 3 个 user task，其中 9 次来自 banking/user_task_12。其条件曲线和区间对任务构成敏感；不据此宣称稳定正趋势。

GAM 的 suite 标准化需要模型在部分 suite 数据较少的概率位置做预测。曲线仅画该 cohort 的 1%–99% 秩范围，端点比较用 10%–90%，仍应结合原始分布判断支持范围。

GAM 导数文件提供每条曲线的近似同时 95% 带，条件于已选平滑度，并未同时覆盖 36 个模型；局部上升区间为探索性候选。所有结果均是既有历史轨迹中的关联，非因果证据。

## 文件

- figures/spearman_overview.png、gee_overview.png：相关性和调整后 OR 总览。
- figures/gam_template_overview.png、gam_goal_overview.png：GAM 六角色总览。
- figures/template/、figures/goal/：12 张单图，PNG／PDF。
- spearman.csv、gee_adjusted.csv、gam_summary.csv：三种方法的完整统计。
- gam_curves.csv：全部曲线、区间及导数带的 3,636 个点。
- models/、user_task_bootstrap_weights.npz、spearman_bootstrap_estimates.npz：复算所需参数和重采样结果。
- validation.json：逐点重新构造 36 条曲线和 bootstrap 区间的验证，以及前述拟合验证。


方法资料：[Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)、[GEE](https://www.statsmodels.org/stable/gee.html)、[GAM](https://www.statsmodels.org/stable/gam.html)。
