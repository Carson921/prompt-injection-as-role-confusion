# 轻量图表与指标副本

本次冻结的是 `scratch` 第 11 轮研究候选。三个候选均未通过全部文章 DEV 门槛，不替换默认模型；developer 召回提高，同时 system 和 tool 召回下降。


本目录只包含报告、指标与图片。逐调用／逐位置概率数组和完整规范包保留在本机仓库外：

`/home/wentian/Github/SPAR/runpod-backups/role-warmstart-20261027-6h/cached-evaluation`

规范包 result-manifest SHA256：`d8c1b836a30870526bfb1c35a4671ec239514bf03bf1ce609e685115d887cf31`。本副本不会修改规范包或原始 AgentDojo 缓存。

---

# AgentDojo 57 任务：冻结后缓存复评

复用原始输入和真实 native decode 特征，不执行新的大模型前向，也不重跑 agent。
本结果是已观察历史集上的回归测试，不能称为新的盲测或三卡部署验证。

正文评分位置 155,103；system 模板 1 种、developer 模板 3 种。
任务数不能作为独立提示模板数。输入和生成分开统计；缺失停止标记保持排除。

| 真实角色 | 正例 token | 父模型召回 | 新模型召回 |
|---|---:|---:|---:|
| system | 4,047 | 60.0445% | 55.0284% |
| developer | 61,085 | 8.1624% | 94.5977% |
| user | 2,969 | 99.9663% | 99.7642% |
| cot | 59,285 | 99.9966% | 99.9983% |
| assistant | 7,907 | 99.9874% | 99.9874% |
| tool | 19,810 | 98.6219% | 96.1484% |

均值概率另列，不能当作 argmax 准确率。完整 P/R/F1、混淆、分段及位置结果见 [metrics.json](metrics.json)。
本轮未重新测量任务成功率或 ASR。

校验与图表：

- [缓存与掩码校验](cache-reuse-audit.json)
- [逐任务硬分类指标](per-task-hard-metrics.csv)
- [逐正文对比](per-body-comparison.csv)
- [新模型任务等权／token加权概率](new-mean-probabilities.csv)
- [逐 suite 硬分类指标](per-suite-hard-metrics.csv)
- [S/D 分段对比](sd-segment-comparison.csv)
- [role-confusion-parent-vs-new.png](role-confusion-parent-vs-new.png)
- [role-confusion-parent-vs-new.pdf](role-confusion-parent-vs-new.pdf)
- [system-developer-segments.png](system-developer-segments.png)
- [system-developer-segments.pdf](system-developer-segments.pdf)
