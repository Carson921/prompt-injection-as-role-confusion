# 三张 A40、六小时预算：实验交付

本轮改善了历史 AgentDojo 缓存上的 developer 识别，但 system 和 tool 召回下降。三个候选均未通过全部文章 DEV 门槛；按预先固定的排序导出 `scratch` 的第 11 轮 checkpoint，实际训练 15 轮，**仅作为研究候选，不替换原默认模型**。

## 主要结果

以下是原 57 个任务、155,103 个首次普通正文 token 的硬分类召回率。直接复用原 `calls/` 中输入和原生生成时的特征，没有重新提取、重新运行 agent 或根据这些成绩追加训练。

| 角色 | 父 checkpoint | 本轮候选 | 变化 |
|---|---:|---:|---:|
| system | 60.04% | 55.03% | −5.02 个百分点 |
| developer | 8.16% | 94.60% | +86.44 个百分点 |
| tool | 98.62% | 96.15% | −2.47 个百分点 |

这 57 个任务只有 1 种 system、3 种 developer 提示正文，属于已观察回归集。该结果衡量旁路分类器，不能证明新提示泛化、AgentDojo 成功率或安全性提升；120B 本体及原 agent 行为均未改变。

全新文章 TEST 共 128 个源簇、133,631 个监督位置：accuracy **96.21%**，六类 macro-F1 **96.24%**；macro-F1 的 95% 源簇 bootstrap 区间为 **[95.45%, 96.93%]**。system 精确率 **88.41%**、U/T→S/D 误报 **3.39%**，仍未达到 90% / 1% 目标；该区间不能套用到 57 个历史任务。

## 继承参数与实际用时

保留了父 checkpoint 的原 16 个有序 pre-o_proj 头、1024 维特征接口及 FIT mean/scale。并行比较了继承全部 MLP 权重、只更新输出层，以及随机初始化 MLP 三种方案；AdamW、epoch、早停和随机状态重新开始。被选中的 `scratch` 只随机初始化 MLP 权重，仍使用旧头和标准化器；这轮结果未证明继承旧 MLP 权重更有优势。

三张 A40 的 GPU 作业窗口为 **53.73 分钟**，从 2026-10-27 13:18:23 EDT 到 14:12:07 EDT；六小时是授权上限。GPU 提取和训练已结束，后续复算、绘图与备份在 CPU 完成。原模型、特征缓存及历史结果保留原样。

## 文件与验收

- [完整结果报告](REPORT.zh-CN.md)，[结构化全部指标](report-summary.json)
- [实验 checkpoint](checkpoints/best.npz)，[冻结与谱系](checkpoints/freeze.json)，[三候选选择记录](checkpoints/selection.json)
- [训练曲线](training-diagnostics/training-loss-and-dev.png)，[六角色混淆矩阵](cached-evaluation/role-confusion-parent-vs-new.png)，[S/D 分段图](cached-evaluation/system-developer-segments.png)
- [完整独立复算](independent-validation/final.json)，[备份验收](backup-verification/backup-summary.json)，[最终交付审计](completion-audit.json)

checkpoint SHA256：`45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df`。独立 CPU 验收覆盖文章 TEST、三候选初始化与训练记录、2,000 次源簇重采样，以及全部 675,573 行历史缓存读出；预测 argmax 差异为 0。

完整私有备份位于 `/home/wentian/Github/SPAR/runpod-backups/role-warmstart-20261027-6h-final/`。新数据、特征、训练恢复状态、日志和代码保存在 `task/`；已验证的旧特征备份及原 AgentDojo 缓存作为明确的本地依赖保留。基础模型权重和运行环境按固定版本与校验哈希恢复，未重复拷贝大模型权重。

仓库仅保存小型 checkpoint、指标、图表和审计记录。逐 token 概率与原始载荷保存在仓库外；缓存复评目录的 [display-manifest](cached-evaluation/display-manifest.json) 区分展示副本和完整私有产物。文章固定 50 篇审阅为 LLM 代理审阅，不是人类 gold；抽样范围、质量排除和历史影响详见完整报告。
