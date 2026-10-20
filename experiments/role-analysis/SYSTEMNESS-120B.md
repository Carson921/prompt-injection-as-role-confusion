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

注意力/MLP 分支试验对四种特征使用相同的 200 篇训练原文及完整原验证集。直接注意力输出在单消息上的 system F1 为 75.47%，相同规模残差对照为 66.27%；长正文为 70.71% 对 64.50%。注意力特征达到试验前固定的晋级标准，已扩大到全部 1200 篇训练原文。这里是验证收益，不能当作新增独立测试收益。

- [分支试验报告](outputs/local-gptoss120b/systemness-component-pilot/report.txt)
- [分支完整验证指标](outputs/local-gptoss120b/systemness-component-pilot/validation-metrics.csv)
- [预设晋级标准](outputs/local-gptoss120b/systemness-component-pilot/promotion-plan.json)及[实际决定](outputs/local-gptoss120b/systemness-component-pilot/promotion-decision.json)
- [四卡分支试验入口](systemness_components.py)及[全量注意力提取入口](systemness_attention.py)

另一组试验固定原残差特征、1200 篇训练原文与 MLP-64，比较 ERM、按角色及采样位置均衡的 CE、Focal Loss、Group DRO。后面三种目标在本轮配置下都未通过晋级标准，继续保留原选定探针。[完整报告](outputs/local-gptoss120b/systemness-objective-pilot/report.txt)、[验证指标](outputs/local-gptoss120b/systemness-objective-pilot/validation-metrics.csv)、[试验入口](systemness_objectives.py)。方法依据为 [Group DRO](https://arxiv.org/abs/1911.08731) 与 [Focal Loss](https://arxiv.org/abs/1708.02002)，本实验没有复现论文中的其他任务。

后续仍可尝试：在现有同正文多角色数据上增加显式配对损失、增加独立长正文和困难负例、细分误差后补充真实 system 提示与上下文。新增模型应先仅用验证集确定，再增加至少 400 篇与既有 fit/validation/test 按正文、完整原文和 URL 去重的新原文作确认。旧测试指标保留为已报告基准，不重新包装为独立确认。训练组别、真实角色和标签不作为推断输入。


模型配置核对发现：当前 attention 特征的 16/20/24 层均为 128-token 窗口的局部注意力层，而相邻的 17/21/25 层是全局注意力层。200 篇原文试验中，长正文超过 512 token 后的验证 system recall 两种特征均为 34.02%，整体 F1 提高尚未解决远距离位置。已据此完成相邻全局注意力层对照；输出投影前的头级特征（4096 维，而投影后为 2880 维）仍待检验。残差仍能传递更早的角色信息，局部窗口不能单独解释全部误差。见[实际配置核对](outputs/local-gptoss120b/systemness-component-pilot/architecture-audit.json)和[验证位置分解](outputs/local-gptoss120b/systemness-component-pilot/validation-system-by-position.csv)。


1200 篇原文的局部 attention 轮次已完成四个读出的训练与验证。固定模型为 MLP-64，全局 system 阈值为 0.5554444444444444；相同验证集的平均 system F1 从原模型的 85.44% 提高至 89.00%。本项仍只有验证证据，独立确认尚未执行，后续将与完整长文候选共同进入464篇确认。[报告](outputs/local-gptoss120b/systemness-attention/report.txt)、[冻结选择](outputs/local-gptoss120b/systemness-attention/selection.json)、[完整指标](outputs/local-gptoss120b/systemness-attention/validation-metrics.csv)、[原400篇确认入口](systemness_attention_confirm.py)、[新464篇确认入口](systemness_long_confirm.py)。全局注意力匹配对照入口为[systemness_global_attention.py](systemness_global_attention.py)。
全量局部 attention 的自然对话验证五分类准确率由 80.77% 降至 78.66%，CoT recall 由 59.52% 变为 54.17%；systemness 的改善不能等同于全部角色改善。


全局邻层对照现已完成，未通过晋级标准：与同200篇原文的局部 attention 线性对照相比，平均验证 system F1 下降4.99个百分点。[完整负结果](outputs/local-gptoss120b/systemness-global-attention-pilot/report.txt)。该试验同时改变注意力类型和层深度，不能单独归因为窗口大小。

长度审计发现，旧200篇“长正文”训练原文只有26篇达到1024 token，原50篇长验证只有10篇；512 token之后的system训练样本只有739个。原1200篇fit中共有132篇可达到1024，原200篇validation中共有31篇：[长度审计](outputs/local-gptoss120b/systemness-component-pilot/length-audit.json)、[完整长文分组清单](outputs/local-gptoss120b/systemness-component-pilot/full-length-cohort.csv)。

新一轮保持local16/20/24特征，使用全部132篇完整长训练原文，每角色均匀采256个位置；31篇完整长验证原文每角色128位置。815个输入、188800个采样位置均已通过独立native角色解析和正文SHA核验。训练会替换对应旧long-single稀疏行；验证保留原四组及其匹配指标，另加long-full组，其中10篇与原长验证重叠，不重复计为新原文。[数据计划](outputs/local-gptoss120b/systemness-long-full/plan.json)、[准备完成记录](outputs/local-gptoss120b/systemness-long-full/data-complete.json)。

[读出入口](systemness_long_probes.py)预设五个候选：原数据MLP-64、完整长文MLP-64/128和线性读出，以及原数据重采样MLP-64。主对照重采样原413290行至每轮578090实例，与完整长文候选均为142次更新；原数据未重采样读出保留作次对照。它们共用原fit标准化参数。独立早停仍可能产生不同最佳更新数，且此控制没有匹配长文权重，不能称为严格的最终总步数或全部训练分布匹配。当前尚无本轮读出训练结果。

已提前声明后续独立确认：400篇新一般原文，加64篇另行去重、真实达到1024-token的完整长原文，共464篇独立来源；原400篇中的64个up-to1024长变体仍是配对子集。[确认协议](outputs/local-gptoss120b/systemness-long-full/confirmation-plan.json)。模型、阈值先用验证集冻结，再准备和执行该确认。当前这组独立确认尚未运行。

本轮初始化在删除已声明的两个可重建特征缓存后，因即时可用空间检查失败而退出，尚未进入新长文提取。原文、行表、已发布模型和分组完整保留：[失败记录](outputs/local-gptoss120b/systemness-long-full/initialization-failure.json)。[显式恢复入口](systemness_recover_local_bank.py)按历史两个阶段重算原特征，只有pilot与完整前缀的float16 SHA都与旧值完全相同，才允许原提取程序继续；恢复不会删除其他缓存。

后续候选方法仍是待验证假设。优先考虑同正文同位置、不同正式角色之间的配对排序损失，在训练中压低内容语义的干扰，推理仍只读取单个token特征；以及在o_proj之前选择或稀疏组合注意力头。前者可参考[监督对比学习](https://arxiv.org/abs/2004.11362)的监督结构，后者可参考[注意力头探针](https://arxiv.org/abs/2306.03341)的分析方法，这些论文并不证明本任务会有提升。另可增加命令语气与正式角色相反的困难样本、更多自然对话system模板，以及轮数和工具循环不同的正常native输入。所有system标签正常保留，错误挖掘只使用fit数据，方法和阈值仍在验证集决定。
