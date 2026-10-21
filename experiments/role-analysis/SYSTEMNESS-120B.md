**GPT-OSS-120B MXFP4 systemness 实验入口**

此页记录已冻结并完成独立测试的48候选轮次，以及后续验证和独立确认。语言模型权重保持冻结，使用四张 A40 提取 MXFP4 模型的激活，只训练角色读出。训练与评估正常保留 system，五类输出顺序为 system、user、cot、assistant、tool。

初始48候选轮次的最佳读出为 1200 篇训练原文、单消息加多轮上下文上的 MLP-64。输入拼接第 16/20/24 层归一化前残差，每层 2880 维，共 8640 维；层编号从 0 开始。阈值和模型只用验证集选择，测试前固定。后续完整长文读出及新增464篇确认记录在下半页。

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


**后续改进试验与独立确认**

注意力/MLP 分支试验对四种特征使用相同的 200 篇训练原文及完整原验证集。直接注意力输出在单消息上的 system F1 为 75.47%，相同规模残差对照为 66.27%；长正文为 70.71% 对 64.50%。注意力特征达到试验前固定的晋级标准，已扩大到全部 1200 篇训练原文。这里是验证收益，不能当作新增独立测试收益。

- [分支试验报告](outputs/local-gptoss120b/systemness-component-pilot/report.txt)
- [分支完整验证指标](outputs/local-gptoss120b/systemness-component-pilot/validation-metrics.csv)
- [预设晋级标准](outputs/local-gptoss120b/systemness-component-pilot/promotion-plan.json)及[实际决定](outputs/local-gptoss120b/systemness-component-pilot/promotion-decision.json)
- [四卡分支试验入口](systemness_components.py)及[全量注意力提取入口](systemness_attention.py)

另一组试验固定原残差特征、1200 篇训练原文与 MLP-64，比较 ERM、按角色及采样位置均衡的 CE、Focal Loss、Group DRO。后面三种目标在本轮配置下都未通过晋级标准，继续保留原选定探针。[完整报告](outputs/local-gptoss120b/systemness-objective-pilot/report.txt)、[验证指标](outputs/local-gptoss120b/systemness-objective-pilot/validation-metrics.csv)、[试验入口](systemness_objectives.py)。方法依据为 [Group DRO](https://arxiv.org/abs/1911.08731) 与 [Focal Loss](https://arxiv.org/abs/1708.02002)，本实验没有复现论文中的其他任务。

后续配对损失及完整长正文试验见下文；尚可增加独立长正文和困难负例、细分误差后补充真实 system 提示与上下文。新增模型应先仅用验证集确定，再增加至少 400 篇与既有 fit/validation/test 按正文、完整原文和 URL 去重的新原文作确认。旧测试指标保留为已报告基准，不重新包装为独立确认。训练组别、真实角色和标签不作为推断输入。


模型配置核对发现：当前 attention 特征的 16/20/24 层均为 128-token 窗口的局部注意力层，而相邻的 17/21/25 层是全局注意力层。200 篇原文试验中，长正文超过 512 token 后的验证 system recall 两种特征均为 34.02%，整体 F1 提高尚未解决远距离位置。已据此完成相邻全局注意力层对照；输出投影前的头级特征（4096 维，而投影后为 2880 维）仍待检验。残差仍能传递更早的角色信息，局部窗口不能单独解释全部误差。见[实际配置核对](outputs/local-gptoss120b/systemness-component-pilot/architecture-audit.json)和[验证位置分解](outputs/local-gptoss120b/systemness-component-pilot/validation-system-by-position.csv)。


1200 篇原文的局部 attention 轮次已完成四个读出的训练与验证。固定模型为 MLP-64，全局 system 阈值为 0.5554444444444444；相同验证集的平均 system F1 从原模型的 85.44% 提高至 89.00%。该验证收益现已与完整长文候选共同完成464篇独立确认，见下文。[验证报告](outputs/local-gptoss120b/systemness-attention/report.txt)、[冻结选择](outputs/local-gptoss120b/systemness-attention/selection.json)、[完整验证指标](outputs/local-gptoss120b/systemness-attention/validation-metrics.csv)、[原400篇确认入口](systemness_attention_confirm.py)、[464篇确认入口](systemness_long_confirm.py)。全局注意力匹配对照入口为[systemness_global_attention.py](systemness_global_attention.py)。
全量局部 attention 的自然对话验证五分类准确率由 80.77% 降至 78.66%，CoT recall 由 59.52% 变为 54.17%；systemness 的改善不能等同于全部角色改善。


全局邻层对照现已完成，未通过晋级标准：与同200篇原文的局部 attention 线性对照相比，平均验证 system F1 下降4.99个百分点。[完整负结果](outputs/local-gptoss120b/systemness-global-attention-pilot/report.txt)。该试验同时改变注意力类型和层深度，不能单独归因为窗口大小。

长度审计发现，旧200篇“长正文”训练原文只有26篇达到1024 token，原50篇长验证只有10篇；512 token之后的system训练样本只有739个。原1200篇fit中共有132篇可达到1024，原200篇validation中共有31篇：[长度审计](outputs/local-gptoss120b/systemness-component-pilot/length-audit.json)、[完整长文分组清单](outputs/local-gptoss120b/systemness-component-pilot/full-length-cohort.csv)。

新一轮保持local16/20/24特征，使用全部132篇完整长训练原文，每角色均匀采256个位置；31篇完整长验证原文每角色128位置。815个输入、188800个采样位置均已通过独立native角色解析和正文SHA核验。训练会替换对应旧long-single稀疏行；验证保留原四组及其匹配指标，另加long-full组，其中10篇与原长验证重叠，不重复计为新原文。[数据计划](outputs/local-gptoss120b/systemness-long-full/plan.json)、[准备完成记录](outputs/local-gptoss120b/systemness-long-full/data-complete.json)。

[读出入口](systemness_long_probes.py)的五候选均已完成：原数据MLP-64、完整长文MLP-64/128和线性读出，以及原数据重采样MLP-64。主对照重采样原413290行至每轮578090实例，与完整长文候选均为142次更新；原数据未重采样读出保留作次对照。它们共用原fit标准化参数。独立早停仍可能产生不同最佳更新数，且此控制没有匹配长文权重，不能称为严格的最终总步数或全部训练分布匹配。[选定MLP-128](outputs/local-gptoss120b/systemness-long-full/selection.json)相对重采样控制的完整长文system F1为67.21%对58.37%，尾部60.41%对48.42%；同结构MLP-64也分别达到67.33%和60.64%。本轮五组平均F1含新增完整长文组，不能直接与旧四组均值比较。单消息和完整长文仍未达到工作目标。

独立确认已按提前声明的协议完成：400篇新一般原文，加64篇另行去重、真实达到1024-token的完整长原文，共464篇独立来源；原400篇中的64个up-to1024长变体仍是配对子集。[确认协议](outputs/local-gptoss120b/systemness-long-full/confirmation-plan.json)。模型、阈值先用验证集冻结，再准备和执行确认，全部962889个五角色内容位置参与指标；工具调用动作不属于这五类评分对象。

本轮初始化在删除已声明的两个可重建特征缓存后，因即时可用空间检查失败而退出，当时尚未进入新长文提取。原文、行表、已发布模型和分组完整保留：[失败记录](outputs/local-gptoss120b/systemness-long-full/initialization-failure.json)。[显式恢复入口](systemness_recover_local_bank.py)已正常退出，完整566482行原特征重建完成；pilot SHA与完整前缀SHA均与旧值逐bit一致：[恢复审计](outputs/local-gptoss120b/systemness-long-full/cache-recovery.json)、[pilot阶段](outputs/local-gptoss120b/systemness-long-full/recovery-stages/pilot.json)、[剩余阶段](outputs/local-gptoss120b/systemness-long-full/recovery-stages/remaining.json)。新增188800位置也已提取完成，全755282行通过有限值、非空行和文件SHA核验：[提取完成记录](outputs/local-gptoss120b/systemness-long-full/extraction-complete.json)。

同正文同位置、不同正式角色之间的配对排序损失已经按下述三候选完成；另一个待验证方向是在o_proj之前选择或稀疏组合注意力头。前者可参考[监督对比学习](https://arxiv.org/abs/2004.11362)的监督结构，后者可参考[注意力头探针](https://arxiv.org/abs/2306.03341)的分析方法，这些论文并不证明本任务会有提升。另可增加命令语气与正式角色相反的困难样本、更多自然对话system模板，以及轮数和工具循环不同的正常native输入。所有system标签正常保留，错误挖掘只使用fit数据，方法和阈值仍在验证集决定。

配对排序[三候选](systemness_paired_probes.py)已全部完成。132篇fit正文形成33792组五角色同位置配对，每步增加128组辅助前向，仅计算排序损失；原CE仍为每轮578090实例和142次更新。lambda=0、0.1、0.5共用初始化、CE顺序和隔离的辅助dropout RNG。CPU检查已验证lambda0与纯CE梯度一致、排序梯度方向及fit配对索引；实际lambda0的全部173032行验证概率与原dense-MLP64逐项完全一致。lambda0.1及0.5均未晋级，五组平均F1相对控制分别下降0.02和0.39个百分点：[完整负结果](outputs/local-gptoss120b/systemness-paired-pilot/selection.json)。辅助梯度只作用完整长文，此试验不能单独区分配对结构与普通长文重加权的影响。

在任何新确认准备前，464篇协议升级到v2，并[声明最终选择规则](outputs/local-gptoss120b/systemness-final-readout/plan.json)。[最终选择入口](systemness_final_select.py)先独立重算两试验所选模型的同一验证行概率指标；配对候选须先通过自身匹配控制的晋级，再超过长文候选平均F1，完整长文和尾部F1均不下降，原四组各自system F1回退最多2个百分点，并满足accuracy/FPR限制。否则保留长文候选。确认仍比较原残差、已发布attention与最终读出三个冻结模型；确认准备锁出现后禁止重选。

最终选择已正常完成，保留[dense-MLP128](outputs/local-gptoss120b/systemness-final-readout/selection.json)，system阈值0.6965757575757576；[选择报告](outputs/local-gptoss120b/systemness-final-readout/report.txt)保留所有跨试验门控和未采用配对候选的理由。[八候选验证汇总](outputs/local-gptoss120b/systemness-final-readout/validation-report.txt)区分开发验证与以下独立确认。

464篇独立确认的最终读出指标：

| 组别 | 独立原文 | System precision | System recall | System F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| 单消息 | 400 | 87.48% | 79.25% | 83.16% | 88.21% |
| 构造多轮 | 同400 | 99.16% | 97.56% | 98.36% | 98.72% |
| up-to1024长变体 | 原400中的64 | 80.32% | 71.80% | 75.82% | 83.70% |
| 真正1024正文 | 另64 | 73.72% | 59.53% | 65.87% | 77.11% |
| 正文512–1023位置 | 同另64 | 67.62% | 54.64% | 60.45% | 74.05% |

与同一新数据上的已发布attention控制相比，完整长文system F1提高6.44个百分点，2000次原文配对bootstrap的95%区间为[4.56,8.25]；尾部提高8.42点，区间[5.52,11.32]。单消息F1差为−0.06点，区间[−0.45,0.33]，不能称为单消息进一步改善。除构造多轮外，其余组仍未达到system P/R/BAcc各90%的工作目标，继续保留失败事实。[完整确认报告](outputs/local-gptoss120b/systemness-long-full/confirmation/report.txt)、[全部指标](outputs/local-gptoss120b/systemness-long-full/confirmation/metrics.csv)、[配对区间](outputs/local-gptoss120b/systemness-long-full/confirmation/paired-comparisons.csv)、[12个固定样本的五面板三模型图集](outputs/local-gptoss120b/systemness-long-full/confirmation/plots/gallery-index.md)。图集另含真正1024正文的分位置召回曲线，未做平滑。

原144段自然对话及32种措辞配对变体的[冻结模型复核](systemness_natural_recheck.py)已完成，不计作新增独立测试，也不用于重新选模型或阈值。720个旧缓存输入含334464个输入token；保留324798个旧语义位置，补齐972个没有语义归属但形式上属于user的内容位置，共325770个评分/投影位置。formal指标包含新增位置，语义投影和图仍只使用原位置，同时单列旧分母指标。

| 最终dense-MLP128 | System precision | System recall | System F1 | 五角色准确率 | CoT recall |
|---|---:|---:|---:|---:|---:|
| 正常native标签 | 92.39% | 100.00% | 96.05% | 76.34% | 49.61% |
| 未见system措辞，正常native标签 | 97.73% | 99.91% | 98.81% | 76.42% | 48.05% |

全部放入user标签时，完整formal分母分别为64437和66618，user准确率仅16.77%和17.55%，system误报率为1.60%和2.80%。这说明systemness较好不能代表整体结构角色稳健；正常native对话中CoT/tool混淆仍明显。原无标签变体只报告语义投影，不定义正式角色准确率。配对bootstrap以8个原system模板或32个未见模板为重采样单位，各2000次，不把token当独立样本。旧语义子集的all-user准确率单列为16.44%/17.29%，不可与完整formal分母混用。

[完整复核报告](outputs/local-gptoss120b/systemness-natural-recheck/report.txt)、[全部指标](outputs/local-gptoss120b/systemness-natural-recheck/metrics.csv)、[旧分母对照](outputs/local-gptoss120b/systemness-natural-recheck/legacy-all-user-metrics.csv)、[模板配对区间](outputs/local-gptoss120b/systemness-natural-recheck/paired-comparisons.csv)、[12段固定对话×三种格式的36张三模型图](outputs/local-gptoss120b/systemness-natural-recheck/plots/gallery-index.md)。

复核是同一冻结模型、输入及阈值的重新推理，并非旧概率逐项复现。独立[旧新概率审计](outputs/local-gptoss120b/systemness-natural-recheck/legacy-probability-audit.json)确认原324798行及720个逐token输入完全相同，但残差控制在原64294个native位置上有48个system阈值判断、581个argmax改变，概率最大绝对差0.9899，平均差0.00346；旧存档概率仍可精确复算旧报告。批次成员、padding及读出分块布局均改变，尚未通过额外GPU复刻确定差异原因，因此不称为逐bit或可忽略的舍入差异。三个模型在本次复核中共享同次前向的对应特征，旧报告原样保留。最终读出相对残差控制的native五角色准确率下降4.75/5.08个百分点，CoT recall下降11.60/12.29点，不能只看system指标。

下一项[注意力头筛选实验](systemness_head_scan.py)扫描全部36层×64个query head，在o_proj之前读取每头64维向量；语言模型仍为冻结MXFP4。原132篇完整长fit原文按document_id%5划成101篇统计训练和31篇头筛选，两组都属于fit；每头用固定收缩LDA，在31篇上按原文平均的全位置/512+尾部AUROC等权选8头，再用全部132篇fit的统计量重拟合。8头的五类概率等权平均，只读取当前token的特征，不把位置、标签或配对作为推断输入。全层特征在线累计统计，不保存全层原始激活。

方法与数据已[提前冻结](outputs/local-gptoss120b/systemness-head-scan/plan.json)，数学、padding、hook不改激活、退化head和portable聚合均通过[CPU检查](outputs/local-gptoss120b/systemness-head-scan/cpu-preflight.json)。随后只在原173032行、五种native验证组上校准一个全局阈值和评估；头筛选只用31篇fit，2304选8仍可能过拟合。此诊断尚不能视为提升，也不替换当前模型，不读取已报告464篇或旧测试来决定方法。若得到新候选，仍需另取去重的新原文确认。
