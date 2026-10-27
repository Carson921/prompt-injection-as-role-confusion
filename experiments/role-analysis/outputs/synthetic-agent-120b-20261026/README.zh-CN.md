# GPT-OSS-120B 六角色重训：最终结果

本轮按[重训计划](../../RETRAINING-PLAN-120B.zh-CN.md)在 `runpod` 的 4×A40 上完成。120B MXFP4 本体冻结，只训练旁路分类器。最终模型为 **`mlp128-seed17`，最佳 epoch 30，实际运行 38 epochs**；七候选及最佳 epoch 均按预先固定的 DEV 规则选择。

计算与报告于 **2026-10-27 04:23:06 UTC** 完成，距本轮预算起点约 **15 小时 4 分钟**，早于 24 小时截止。GPU 计算进程已结束，截止监督器正常完成，资源监控已停止。RunPod 实例仍在线。

## 结果与使用范围

| 评估 | 评分位置 | 源文章／对话簇 | Accuracy | Macro-F1 | Macro-F1 的 95% CI |
|---|---:|---:|---:|---:|---:|
| 合成 TEST，六类 | 498,688 | 1,024 | 98.8035% | **98.8538%** | 98.6896%–99.0173% |
| 外部原始 U/A 正文 | 70,866 | 256 | 99.3805% | **99.2878%** | 98.4820%–99.9173% |

合成 TEST 达到预定汇总目标：macro-F1≥95%，六类 precision、recall 均≥90%。最低 precision 为 system **96.2955%**，最低 recall 为 developer **97.3359%**。目标按点估计判断；表中置信区间另列。

外部测试为去重后的 OASST、ToxicChat 各 128 条，只评分原始 U/A；后加 S/D 是固定上下文。外部 macro-F1 只平均 U/A，不能解读为六类真实对话均已验证。S/D/C/T 无正例，recall、F1 为 N/A，误报分别为 **109／183／15／115**，仍计入混淆矩阵。

- [完整自动报告](final-v7/formal/REPORT.zh-CN.md)与 [TEST 指标及分项](final-v7/formal/evaluations/test.json)、[外部指标](final-v7/formal/evaluations/external.json)。
- [TEST 独立重算](formal-test-independent-validation.json)与[外部独立重算](formal-external-independent-validation.json)：从提交的标签、概率和分组元数据重算全部指标及 2,000 次整簇 bootstrap。比例指标差异为 0；float64/float32 NLL 差异在声明的 `1e-6` 容差内。该复核未重新运行大模型。

## 主要薄弱项与位置口径

汇总达标不表示每个分组均达到同一阈值：

| 条件 | 评分位置 | Macro-F1 |
|---|---:|---:|
| 独立单消息 | 99,328 | **94.6658%** |
| 单轮 | 149,504 | 99.7277% |
| 多轮 | 249,856 | 99.9206% |
| 多轮中的 K=5 | 62,464 | 99.9472% |
| 多轮中的 K=6 | 62,464 | 99.9262% |

独立消息的 system precision 为 **81.7814%**（S 正例 16,587），developer recall 为 **85.7169%**（D 正例 16,684）。其中 D→S 1,963 次、T→S 987 次，是 system 误报的重要来源。这些分项没有用于测试后选模或调参。

每条消息**正文最后 25%**共有 126,105 个评分位置，macro-F1 **98.2789%**；独立消息的正文尾部为 24,987 个位置、**92.4165%**。另一口径——**整个输入的绝对末四分位**——为 120,136 个位置、**94.3572%**。两种分母和取整规则不同，不能混称“尾部”。位置、profile 和角色组成也相互关联，不能据此直接归因于上下文长度。

正文位置与角色支持量详见 [TEST 位置诊断](final-v7/formal/evaluations/test-positions.json)及[外部位置诊断](final-v7/formal/evaluations/external-positions.json)。这些位置分项没有单独 bootstrap CI。工具前后快照与保留／清理组的目标正文、组成不同，分数差不表示删除 CoT 的因果效果；pre-tool-call 的 100%只涉及 C 的 5,576 个位置。

## Checkpoint 与图片

[直接打开全部正式 PNG/PDF 图表](FIGURES.zh-CN.md)。

- [最终六类 best.npz](final-v7/formal/best.npz)、[冻结清单](final-v7/formal/freeze.json)、[加载与预测代码](final-v7/code/synthetic_agent_classifier.py)、[消费端示例及包说明](final-v7/README.md)。NPZ SHA256：`22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125`。
- 六类顺序固定为 `system, developer, user, cot, assistant, tool`。特征是同一模型/tokenizer 的有序 16 头 `pre-o_proj`，每头 64 维，共 1024 维；消费端自动应用自己的 FIT 标准化器。仅有正确矩阵形状不足以证明特征来源匹配。
- [20 类原生图清单](final-v7/formal/plots/test/index.json)：独立六类、五种单轮、重复多轮、末尾 UCQ、CoT 清理、工具阶段及 K5/K6，全部 PNG/PDF 齐全。展示完整实际输入；灰色为结构，浅黄色为未采样文章正文，彩色概率只对应实际评分位置。
- [15 个 dense 诊断清单](final-v7/formal/plots/dense/index.json)：五种模板×原生／统一 user／去标签。改写格式的分数指原语义对齐，不能称为改写后的形式角色 accuracy。
- [原始训练曲线](final-v7/formal/training-curves.png)保留在严格导出包中；另有 [训练 loss 与放大 DEV 曲线](training-diagnostics-v7/README.zh-CN.md)，保留固定最佳 epoch 标记。
- 用户要求的[小试与首正式 checkpoint gardening 对比](gardening-pilot-vs-first-formal/README.zh-CN.md)已交付，共 **7 对 PNG/PDF**。该首正式模型为固定的 `linear-seed17`，与本节最终选定的 MLP 分开归档；其历史样例成绩不用于正式选择。
- [小试 checkpoint](pilot-checkpoint-v7/README.zh-CN.md)和[首正式 checkpoint](first-formal-checkpoint-v7/README.zh-CN.md)均保留。首正式候选在用户提出 8 小时目标后约 7 小时 30 分完成。

[图表审阅记录](plot-render-review-v7/README.zh-CN.md)记录了哈希范围和实际查看的图片；数值清单检查覆盖全部 35 对，逐张视觉检查的范围另外列出。

## 数据、采样和审计

正式文章配额 **4096 FIT／512 DEV／1024 TEST**，C4 与 Dolma3 各半。最终 V7 的 [formal 清单](final-v7/metadata/data/formal/data-manifest.json)和 [pilot 清单](final-v7/metadata/data/pilot/data-manifest.json)均已回传；正式 5,632 个 URL 全部可用，规范化正文、URL、源簇及近重复分区检查通过。

正式固定 50 篇 FIT/DEV 样本审阅为 **49 篇相同正文 SHA 的既有审阅＋1 篇新审阅**；审阅者为 LLM 代理，非人类 gold，结论不覆盖所有文章。严格过滤使数据偏向第三人称说明文章；Dolma3 六个主题组并非全库随机样本。详见[审计说明](../../synthetic-agent-data-audit.md)及包内[最终审计决定](final-v7/metadata/data/formal/article-audit.json)。

按单消息／单轮／多轮顺序，以下三种分母分别记录：

| 口径 | 单消息 | 单轮 | 多轮 |
|---|---:|---:|---:|
| FIT 缓存行 | 19.9179% | 29.9795% | 50.1027% |
| family 归一基础权重质量 | 24.2203% | 35.4363% | 40.3435% |
| 最终模型截至最佳 epoch30 的实际梯度抽样 | 20.0022% | 29.9941% | 50.0037% |

最佳 checkpoint 经历 59,842,560 次实际抽样；38 轮全程计数另列，末 8 轮不属于导出权重的训练路径。[真实采样证据](formal-sampling-summary.json)保留来源 SHA。固定 family 归一规则和 20/30/50 分层 sampler 不叠加类别损失权重。

[全输入 mask 审计](formal-input-mask-audit.json)重新分词 25,415 条实际输入，核验全部 2,742,784 个监督位置，结构及 target 违规均为零。单消息、合法未回复 Q 单独处理；3 条缺少保存兄弟快照的清理记录明示无法配对。该审计复用部分冻结解析器，不是人类语义 gold，也不单独证明历史 forward；实际提取回执与 `use_cache=False` 实现另行绑定。

C 是 analysis channel 的操作性标签，不表示文章片段具有真实推理语义；本轮 probe 的文章限制也不描述预训练 120B 的历史数据。

## 资源与复现

[最终资源附录](gpu-resources/resource-appendix.md)、[原始 CSV](gpu-resources/gpu-all-stages.csv)、[采样汇总](gpu-resources/gpu-all-stages-summary.json)及[正常完成／监控停止证明](gpu-resources/completion-and-monitor-stop-v7.json)已回传并核对最终 SHA。设备值为离散采样下界，包含其他 CUDA contexts；PyTorch 峰值计数从权重加载后重置。扫描起点前约 253.275 秒缺少该连续 CSV；dense receipt 没有完整绝对时间窗，资源表如实留空。正式训练窗口与 gardening 有保守并发范围，不能把其峰值当作纯分类器显存。

历史拟合、用户查询 CSV 文件仍保留在原本地目录；服务器只接收绑定的去重 hash／公开 ID，完整历史摘要清单不进公开 Git。单独授权的公开 gardening 输入按其诊断协议归档。大型权重、原始 epoch histories、全量正文输入 JSONL、特征和完整概率数组留在运行机；小型导出包明确排除这些载荷。

运行、固定依赖、同参数恢复及授权窗口限制见[运行手册](../../SYNTHETIC-AGENT-RUNBOOK.zh-CN.md)和包内 `code/`。官方模型 15 分片均按固定 revision 校验；原挂载读取失败后采用独立执行副本，没有完成与原挂载的全量字节对照。冻结运行核心源码快照为 `f790b10b118cab13d8ea153526340c137230f601`；后续验收／交付辅助脚本各自附 SHA。

严格包包含 412 项清单载荷及 3 个控制文件，逐项 `SHA256SUMS` 核验通过；补充验收、资源、gardening 与训练图在包外单列，避免改动已绑定报告。逐项完成证据见[完成审计](COMPLETION-AUDIT.zh-CN.md)。
