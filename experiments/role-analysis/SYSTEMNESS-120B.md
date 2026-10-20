**GPT-OSS-120B MXFP4 systemness 实验入口**

此页记录已冻结并完成独立测试的 48 候选轮次。语言模型权重保持冻结，使用四张 A40 提取 MXFP4 模型的激活，只训练角色读出。训练与评估正常保留 system，五类输出顺序为 system、user、cot、assistant、tool。

最佳读出为 1200 篇训练原文、单消息加多轮上下文上的 MLP-64。输入拼接第 16/20/24 层归一化前残差，每层 2880 维，共 8640 维；层编号从 0 开始。阈值和模型只用验证集选择，测试前固定。

独立测试单元为 400 篇新 C4 原文和 144 段已有生成对话。C4 单消息与构造多轮共用同一批 400 篇原文；64 篇长正文是其子集。新 system 措辞是同 144 段对话的配对变体，含 32 种未参与训练或验证的提示，不能重复计为新增独立对话。

| 评估 | System precision | System recall | 二分类 balanced accuracy |
|---|---:|---:|---:|
| C4 单消息 | 80.63% | 71.05% | 83.39% |
| C4 构造多轮 | 98.17% | 95.06% | 97.40% |
| 自然对话 | 88.60% | 99.42% | 99.48% |
| C4 长正文 | 73.97% | 61.58% | 78.08% |
| 新 system 措辞对话 | 97.22% | 98.67% | 99.23% |

上表是验证阈值下的 system 二分类指标。下表为五角色 argmax 准确率，基线在完全相同的新测试输入上重新评估。

| 评估 | 上轮三层探针 | 本轮固定读出 |
|---|---:|---:|
| C4 单消息 | 65.40% | 84.31% |
| C4 构造多轮 | 51.93% | 97.37% |
| 自然对话 | 58.23% | 81.14% |
| C4 长正文 | 71.93% | 78.48% |
| 新 system 措辞对话 | 56.70% | 81.45% |

单消息和长正文的 system 识别仍偏弱，自然对话的 system precision 也未达到 90%。全部放入 user 标签的自然对话中，结构角色准确率为 14.19%，system 误报率为 1.19%；新措辞配对变体的对应值为 15.66% 和 2.17%。格式泛化和其他角色混淆仍未解决。

C4 多轮输入按固定角色协议用原文切片构造；自然对话来自仓库已有 GPT-OSS-20B 生成文本，消息正文每条最多保留 192 token。它们用于输入给 120B 提取激活，并非本轮新生成的 120B 回答。C4 按原文、自然对话按 system 措辞模板聚类计算配对 bootstrap，相关 token 和重复提示不视为独立样本。

结果与复现：

- [完整报告及复现顺序](outputs/local-gptoss120b/systemness-expansion/report.txt)
- [完整测试指标](outputs/local-gptoss120b/systemness-expansion/test-metrics.csv)
- [配对改进区间](outputs/local-gptoss120b/systemness-expansion/paired-test-comparisons.csv)
- [固定模型及阈值选择记录](outputs/local-gptoss120b/systemness-expansion/selection.json)
- [固定读出模型](outputs/local-gptoss120b/systemness-expansion/models/selected.npz)
- [12 段对话、三种格式的 36 组五面板图](outputs/local-gptoss120b/systemness-expansion/plots/gallery-index.md)
- [结果包](outputs/local-gptoss120b/systemness-expansion/gptoss-120b-systemness-results.tar.gz) 与 [SHA256](outputs/local-gptoss120b/systemness-expansion/gptoss-120b-systemness-results.tar.gz.sha256)
- [依赖](requirements-systemness.txt)

模型可通过 `systemness_probes.load_model(path, device="cpu")` 读取；传入按所述顺序拼接的 8640 维特征，先使用返回的 mean/scale 标准化，再运行模型并对 log 概率取 exp。system 阈值读取 selection.json，五类 argmax 与二分类阈值是不同指标。

后续试验单独保存，原始 72 个单层探针与旧 tomato 图仍保留。


**后续改进试验：验证结果**

注意力/MLP 分支试验对四种特征使用相同的 200 篇训练原文及完整原验证集。直接注意力输出在单消息上的 system F1 为 75.47%，相同规模残差对照为 66.27%；长正文为 70.71% 对 64.50%。注意力特征达到试验前固定的晋级标准，正在扩大到全部 1200 篇训练原文。这里是验证收益，不能当作新增独立测试收益。

- [分支试验报告](outputs/local-gptoss120b/systemness-component-pilot/report.txt)
- [分支完整验证指标](outputs/local-gptoss120b/systemness-component-pilot/validation-metrics.csv)
- [预设晋级标准](outputs/local-gptoss120b/systemness-component-pilot/promotion-plan.json)及[实际决定](outputs/local-gptoss120b/systemness-component-pilot/promotion-decision.json)
- [四卡分支试验入口](systemness_components.py)及[全量注意力提取入口](systemness_attention.py)

另一组试验固定原残差特征、1200 篇训练原文与 MLP-64，比较 ERM、按角色及采样位置均衡的 CE、Focal Loss、Group DRO。后面三种目标在本轮配置下都未通过晋级标准，继续保留原选定探针。[完整报告](outputs/local-gptoss120b/systemness-objective-pilot/report.txt)、[验证指标](outputs/local-gptoss120b/systemness-objective-pilot/validation-metrics.csv)、[试验入口](systemness_objectives.py)。方法依据为 [Group DRO](https://arxiv.org/abs/1911.08731) 与 [Focal Loss](https://arxiv.org/abs/1708.02002)，本实验没有复现论文中的其他任务。

后续仍可尝试：在现有同正文多角色数据上增加显式配对损失、增加独立长正文和困难负例、细分误差后补充真实 system 提示与上下文。新增模型应先仅用验证集确定，再增加至少 400 篇与既有 fit/validation/test 按正文、完整原文和 URL 去重的新原文作确认。旧测试指标保留为已报告基准，不重新包装为独立确认。训练组别、真实角色和标签不作为推断输入。
