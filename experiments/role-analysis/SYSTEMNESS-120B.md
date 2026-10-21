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

方法与数据已[提前冻结](outputs/local-gptoss120b/systemness-head-scan/plan.json)，数学、padding、hook不改激活、退化head和portable聚合均通过[CPU检查](outputs/local-gptoss120b/systemness-head-scan/cpu-preflight.json)。扫描及原173032行、五种native组的验证均已正常完成。8头来自7/15/31层（分别5/1/2个），均为full_attention；这是fit内筛选结果，不能归因为全局注意力的单独因果效果。全局system阈值为0.41935353535353537。

| 原验证组system F1 | dense-MLP128 | 8头概率平均LDA |
|---|---:|---:|
| 单消息 | 84.01% | 79.86% |
| 构造多轮 | 98.21% | 83.42% |
| 自然对话 | 95.66% | 93.69% |
| 原up-to1024长变体 | 78.71% | 84.05% |
| 真正1024正文 | 67.21% | 76.83% |
| 512–1023尾部 | 60.41% | 75.33% |

完整长文的P/R为79.12%/74.67%，尾部为76.83%/73.89%；五组平均F1从84.76%降至83.57%，原四组五分类平均准确率从87.74%降至70.49%，自然CoT recall从50.36%降至11.24%。因此保留长文收益和其他组回退，不替换当前主模型。五profile的FPR均≤5%，尾部FPR为5.57%，不在该约束内。此LDA只使用132篇长fit，而dense模型使用混合578090行，特征、分类器和训练分布均不同，不能把差值单独归因于选头。

[验证报告](outputs/local-gptoss120b/systemness-head-scan/head-validation-report.txt)、[逐组指标](outputs/local-gptoss120b/systemness-head-scan/head-validation-comparison.csv)、[2304头排名](outputs/local-gptoss120b/systemness-head-scan/discovery-head-metrics.csv)、[排名热图](outputs/local-gptoss120b/systemness-head-scan/plots/discovery-head-score-heatmap.png)、[P/R/F1比较图](outputs/local-gptoss120b/systemness-head-scan/plots/validation-systemness-comparison.png)。[独立实际结果审计](outputs/local-gptoss120b/systemness-head-scan/independent-results-audit.json)从保存概率重算所有头的AUROC/排名及全部验证指标，合并132篇重拟合的权重和bias逐bit一致。头筛选只用31篇fit，2304选8仍可能过拟合；当前只是复用开发验证，不读取已报告464篇或旧测试来决定方法。新候选仍需另取去重的新原文确认。

在本轮头验证开始前，下一组联合读出的[条件协议](outputs/local-gptoss120b/systemness-head-scan/readout-preparation/conditional-protocol.json)及源码也已固定：以原755282行提取选定8头的512维F16特征，约0.720GiB；在相同578090个densefit位置上比较8头概率平均LDA、联合线性、MLP64及MLP128，验证仍为173032行。训练与验证保持正常system标签。联合模型使用fit-only新scaler，LDA保留原始头向量；分类器、缩放与旧模型的特征均有变化，不称为纯因果消融。晋级须平均system F1严格提高、完整长文及尾部F1不降，并通过逐组FPR、原四组F1及五分类准确率限制；否则只保留诊断结果。该协议发布时尚未启动新的特征提取或读出训练，当前模型不变。[特征入口](systemness_head_extract.py)、[四候选入口](systemness_head_probes.py)。

已按[正式提取计划](outputs/local-gptoss120b/systemness-head-features/plan.json)完成四卡特征提取，原进程正常退出；[完成记录](outputs/local-gptoss120b/systemness-head-features/extraction-complete.json)核验755282×512个F16特征全部有限且无空行，行表SHA与旧8640维库完全相同。实际hook不改hidden states，portable概率最大差8.11e-15；BF16→F16往返最大差2.99e-8，后者不代表所有BF16值均可精确保存。四候选读出也已全部正常完成，仍未读取新测试或改变主模型。

后续[条件确认协议](outputs/local-gptoss120b/systemness-head-confirmation-preparation/conditional-protocol.json)及[独立入口](systemness_head_confirm.py)在新读出训练开始前固定，只有完成验证并通过晋级约束的候选才能准备新测试。先冻结残差、已发布attention、dense128和所选head四模型及阈值，再从相同固定C4快照另取400篇一般原文和64篇真实1024正文，排除包括最近464篇确认在内的全部已知来源。报告四模型逐组及完整尾部指标、2000次来源配对区间、12个固定来源的5×4角色概率图及分位置曲线。34项CPU检查和独立审查已通过，包括完整合成尾部、图集和缓存/来源/模型签名篡改拒绝。该批四卡native MXFP4确认与独立结果审计现均正常退出；模型、阈值和头顺序未调整，完整结果及未达标项见下文。

新增464篇确认共有3040个native输入、989490个输入token和946202个五角色内容评分位置；这个分母与前一批962889不同。[数据完成记录](outputs/local-gptoss120b/systemness-head-confirmation/data-complete.json)、[预读冻结模型](outputs/local-gptoss120b/systemness-head-confirmation/frozen-readouts.json)已落盘。[根agent独立准备审计](outputs/local-gptoss120b/systemness-head-confirmation/preparation-audit/audit.json)正常退出：重新编码全部输入、独立按native边界分段核对内容标签，并逐行重放全部464个C4来源。已知2364个旧来源行/URL及4898个指纹均排除；另64篇完整长文覆盖327680行，其中512–1023尾部为163840行。原249篇没有可恢复的URL/行身份，且未做语义近重复或域隔离，保留这些限制。该审计没有打开任何测试概率。

混合头验证的[单独报告入口](systemness_head_probes_report.py)已正常完成，只读冻结的dense128与四候选概率，复算五profile及尾部指标、记录所有未晋级原因，并生成12个预定C4验证来源的五角色×五模型tomato图以及F1和分位置曲线，共14对PNG/PDF。每个单消息角色版本为原有64个评分采样位置，不能声称全正文覆盖；这些仍是开发验证，不算新增独立测试。[报告规格与15项CPU检查](outputs/local-gptoss120b/systemness-head-probes-report-preparation/report-spec.json)及7项独立CPU附审已固定。[实际验证报告](outputs/local-gptoss120b/systemness-head-probes-report/report.txt)、[多样本图集](outputs/local-gptoss120b/systemness-head-probes-report/plots/gallery-index.md)、[分位置召回曲线](outputs/local-gptoss120b/systemness-head-probes-report/plots/validation-long-full-by-position.png)。

预定规则选中joint-MLP128，system阈值0.620969696969697，模型SHA为27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266。两个MLP通过全部开发晋级约束；联合线性虽平均F1为95.24%，但自然组F1回退3.34个百分点、多轮五类准确率回退1.07点而未晋级。混合raw pooled LDA平均F1仅81.98%，负结果完整保留。MLP128的最佳checkpoint为第4轮/568更新，实际训练12轮/1704更新；MLP64为最佳第13轮/1846更新、实际21轮/2982更新，不能称为最终总步数严格匹配。[冻结选择](outputs/local-gptoss120b/systemness-head-probes/selection.json)。

| 开发验证组 | 原dense128 system F1 | 联合MLP128 precision | recall | F1 | 五类准确率 |
|---|---:|---:|---:|---:|---:|
| 单消息 | 84.01% | 98.69% | 96.93% | 97.80% | 98.83% |
| 构造多轮 | 98.21% | 99.68% | 99.07% | 99.37% | 99.26% |
| 自然对话 | 95.66% | 96.22% | 100.00% | 98.07% | 90.23% |
| up-to1024变体 | 78.71% | 95.72% | 96.34% | 96.03% | 97.12% |
| 真正1024正文 | 67.21% | 97.08% | 93.12% | 95.06% | 96.92% |
| 512–1023尾部 | 60.41% | 96.06% | 90.83% | 93.37% | 95.45% |

五profile平均system F1为97.27%对84.76%（+12.51个百分点），完整长文及尾部F1分别提高27.85和32.96点；原四profile平均五类准确率为96.36%对87.74%，自然CoT recall为79.57%对50.36%。五profile及尾部的system P/R/BAcc在开发验证均超过90%，但这尚不证明独立测试达标。变化同时涉及选头特征、维度、scaler、分类器和提取batch，不能单独归因某一种注意力或训练因素。

[独立实际审计](outputs/local-gptoss120b/systemness-head-probes-report/independent-audit/actual-audit.json)核对173032行、31篇长验证中的9920行尾部、阈值网格、所有晋级约束及MLP预算，指标差为0，24行CSV最大差1.12e-16。审计器v1在无tool支持的自然组recall差值处发生None相减，修正为N/A后，27项CPU检查和实际审计均正常退出；生产源、模型、阈值和概率均未改，[修订记录与失败证据](outputs/local-gptoss120b/systemness-head-probes-report/independent-audit/auditor-revision.json)保留。已通过开发晋级，因此未运行仅在未晋级时允许的融合备用试验；另一批464篇去重原文确认已按既定协议完成，长文召回仍未达到90%的工作目标，见下文。[发布证据清单](outputs/local-gptoss120b/systemness-head-probes-report/publication-complete.json)。

新增自然对话的数据来源也已完成CPU准备和独立去重审查：[来源与模板提案](outputs/local-gptoss120b/systemness-natural-sources/candidates/bb1bfe8c11fbaae2/source-proposal.json)、[可重放入口](systemness_natural_sources.py)。固定OpenAssistant/oasst1与ToxicChat的官方revision及快照SHA，从2296个合格来源中按预定SHA顺序提出448个根对话，训练/验证/测试分别256/64/128，每组两来源均衡。排除全部200个旧请求，包括16次失败，以及旧OASST树和与旧ToxicChat文本对应的108个原生ID；全树用户文本、原文/规范化/标签清理后的正文和192-token前缀、URL均参与去重。ToxicChat的0124/1123训练用户文本和ID完全相同，更换该配置不能视为新样本。旧加载器未固定revision，不能追认其版本。

本次另固定80条正常system正文，训练/验证/测试为32/16/32条，与旧44条及跨组的精确、规范化、前缀匹配均无碰撞。测试32条措辞各配4个来源，正类区间须按32个措辞刺激聚类，不能把128段对话或token当作独立system刺激。措辞共享部分指令主题，不声称语义独立。原始公开快照只存忽略缓存；发布来源ID、指纹、措辞、分配和审计。此阶段**没有生成新对话或评估新测试**，生成路径、预算和失败规则仍需单独固定；OASST最多保留两条真实用户轮，ToxicChat仅一条，真实analysis/final须分开记录，不伪造截断后的final。[来源排除证据](outputs/local-gptoss120b/systemness-natural-sources/candidates/bb1bfe8c11fbaae2/source-exclusion-proof.json)、[模板审计](outputs/local-gptoss120b/systemness-natural-sources/candidates/bb1bfe8c11fbaae2/template-audit.json)、[生成提案及边界](outputs/local-gptoss120b/systemness-natural-sources/candidates/bb1bfe8c11fbaae2/generation-proposal.json)。

混合头联合MLP已通过开发晋级。后续优先增加真实长system指令，并只在fit内挖掘角色冲突困难样本；当前按system信号选择的8头也可与按CoT/tool等类别分别选头的新读出比较。逐头归一化/白化和更强正则化可检验对激活尺度的依赖，MLP64/128的验证选定集成可检验方差。这些均须另行提前固定候选和选择规则，目前没有完成的收益证据。若使用前序token特征训练因果卷积或小GRU，或者采用[序列标注的CRF方法](https://arxiv.org/abs/1508.01991)，应单列为上下文读出，与当前逐token读出分别报告，不能把额外上下文带来的提升说成单个token表征更可解码。真实长system数据应在所有正式角色间匹配正文语气，训练中仍正常保留system标签。角色可解码性也不等同于模型遵守指令层级的行为。

[仅开发验证的互补诊断](outputs/local-gptoss120b/systemness-fusion-preparation/validation-complementarity.json)使用两模型各自冻结的阈值，在相同173032行上重现旧指标。选头LDA在完整长文/尾部分别修复1455/870个dense128的system二分类错误，同时把891/443个原本正确的位置变错；多轮组只修复137个，却引入2873个错误。可见长文互补性，但不能据此证明修正网络能识别这些情况。诊断中的oracle使用真实标签逐token在这两组冻结判断中择优，仅是这一选择操作的上界，不是任意新分类器的上界或可部署成绩。system logit修正只能保留非system条件分布，五类argmax和CoT recall仍可能改变，不能修复CoT/tool之间的相对排序。[诊断范围与未执行的方法提案](outputs/local-gptoss120b/systemness-fusion-preparation/diagnostic-scope.json)。

另一个可节省生成成本的数据方案，是保留新公开来源中的真实user/assistant文本，再用固定正常system正文构建native输入。OASST保留公开非合成对话，ToxicChat回复来自Vicuna演示且精确checkpoint未知。该变体只有system/user/assistant，CoT/tool指标必须为N/A；新system并非这些源回复当时的生成条件，不能据此评估遵循新指令的行为。新开发数据尚未用于训练，后续独立测试准备记录见下。

该公开对话路线现已完成[独立开发数据入口](systemness_natural_transcripts.py)及[冻结协议](outputs/local-gptoss120b/systemness-natural-transcripts/protocol.json)：先发布源码/CPU审查，再落盘协议，随后只材料化256个fit与64个validation来源的回复。固定SHA路径不回换第一条assistant以追求更长轮次；共有82段使用两条真实用户轮，其余一轮。全部320段准备成功，80436个内容位置（system10000、user16350、assistant54086）均与独立native解析一致；逐消息记录原文、清理及192-token cap指纹。发现1个fit回复与旧生成内容碰撞，明确禁止其进入训练，因此可用fit为255段；64段验证均无旧内容碰撞。新dev内部及跨划分assistant精确/规范化/前缀重复cluster为0，system措辞依赖仍为32个fit和16个validation组件。所有固定ID均保留，失败/污染无替换；CoT/tool均无实际样本并记N/A。[数据完成记录](outputs/local-gptoss120b/systemness-natural-transcripts/data-complete.json)、[重复审计](outputs/local-gptoss120b/systemness-natural-transcripts/duplicate-audit.json)、[完整来源与内容重放](outputs/local-gptoss120b/systemness-natural-transcripts/root-replay-audit.json)。本批开发数据尚未提取模型特征或用于训练，不属于已完成的准确率提升；另128个held-out根对话的测试准备见下。

如果当前四头候选未晋级且尚未创建其正式确认队列，才允许执行[有界system修正入口](systemness_head_fusion.py)。两个预定候选为head512→linear和head512→MLP32，delta=4*tanh(g/4)，末层零初始化；dense128及其原8640维scaler冻结。两候选共享完整开发集的5维base raw logits缓存和已验证的fit-only head scaler，训练仍为578090实例、142更新/轮及45轮上限。除原dense128外，增加完整173032行delta0重算对照，记录概率/判断变化及原阈值、相同helper重选阈值的指标。晋级须对原对照与重算对照同时通过全部约束，平均F1严格超过两者；不能把重算漂移当作学习收益。[实现前设计](outputs/local-gptoss120b/systemness-fusion-preparation/conditional-design-v1.json)、[源码冻结协议及67项CPU检查和36项独立附审](outputs/local-gptoss120b/systemness-fusion-preparation/conditional-protocol.json)。尚未启动融合训练或验证实际数值对齐。

新增公开自然对话的[128根测试准备入口](systemness_natural_test.py)与[四模型评估入口](systemness_natural_test_confirm.py)现已完成源码冻结；[条件协议与CPU证据](outputs/local-gptoss120b/systemness-natural-test-preparation/conditional-protocol.json)、[实际预读协议](outputs/local-gptoss120b/systemness-natural-test/protocol.json)及[模型/阈值声明](outputs/local-gptoss120b/systemness-natural-test/frozen-readouts.json)均先于测试回复材料化。固定64个OASST和64个ToxicChat来源、32种system措辞，所用head128模型和阈值与上述C4确认完全相同，源码不依赖其测试结果。准备所有来源并保留失败ledger，分别报告all-prepared及剔除旧/dev回复直接重复后的clean-test；重复内容和共享system措辞按连通组件做2000次模型配对bootstrap，CoT/tool支持为0时为N/A。12个图集来源按固定ID等距选定，失败不替换，另加汇总图，共13对PNG/PDF。66项runner CPU检查及17+9项独立附审通过，包括全部128根失败时仍保留ledger、零GPU推理和占位图；少于2个依赖组件或无定义比率时不虚构置信区间。本批测试回复已按冻结规则准备并完成下述审计；在464篇原文确认进程正常退出后，其四卡评估及实际结果审计也已正常退出，完成结果见下。预读完整性验证会读取既有开发验证概率，不读取任何C4或自然测试概率；[准备审计范围修订](outputs/local-gptoss120b/systemness-head-confirmation/preparation-audit/revision.json)也对这一点作了明确区分。

自然128根正式准备已正常退出：128段全部成功、0质量失败，35776个输入token、460条消息、33476个S/U/A内容位置（system4228、user6638、assistant22610）；all-prepared和clean-test均为128段，32个system措辞/32个依赖组件，没有发现与旧生成、320根开发回复或测试内部的精确/规范化/前缀重复。[完成记录](outputs/local-gptoss120b/systemness-natural-test/data-complete.json)、[去重与组件审计](outputs/local-gptoss120b/systemness-natural-test/duplicate-audit.json)。根agent的[完整来源/缓存重放](outputs/local-gptoss120b/systemness-natural-test/root-source-replay.json)及[实际独立准备审计](outputs/local-gptoss120b/systemness-natural-test/preparation-audit/audit.json)均正常退出，核对全部内容位置的native标签、token/字符边界、消息上限与指纹、冻结模型/阈值、all/clean计数、组件和12个固定图集ID，绑定文件核验前后SHA未变。独审不重复提取原始回复的碰撞键，其范围在报告中明确，来源重放由根agent另行完成。[发布清单](outputs/local-gptoss120b/systemness-natural-test/preparation-publication.json)。本批正式报告与实际结果审计现均完成，成绩及适用范围见下。

464篇独立确认的[结果审计器](outputs/local-gptoss120b/systemness-head-confirmation-preparation/result-auditor/audit.py)也已在读取测试结果前固定。[合成CPU预检与独立附审](outputs/local-gptoss120b/systemness-head-confirmation-preparation/result-auditor/preparation.json)核对20行四模型指标、5504行原文计数、396组配对比较、264组区间及13对PNG/PDF的规格和文本数字。统计公式独立实现；实际入口必须先通过终态、模型/输入/概率SHA、backend/hook/portable链，再读取概率，不能对未完成缓存报告成绩。该正式确认与结果审计现均正常退出，完整结果及未达标项如下。

自然128根的[结果审计器](outputs/local-gptoss120b/systemness-natural-test-preparation/result-auditor/audit.py)也已在其正式推理前固定。[21项合成预检与8项独立附审](outputs/local-gptoss120b/systemness-natural-test-preparation/result-auditor/preparation.json)覆盖all/clean及两个来源的24行指标、每组9个统计量的2000次组件配对bootstrap、13对图和128个来源状态。冻结生产指标的macro F1仅平均有真值支持的类别；审计明确区别于sklearn默认truth/prediction联合类别平均，CoT/tool的无支持recall仍为N/A。正式结果审计现已正常退出，独立复算全部24行主指标、216组区间和324组配对比较，主指标最大差为0。

新一轮464篇C4独立确认已正常退出并通过[实际独立结果审计](outputs/local-gptoss120b/systemness-head-confirmation/independent-audit/actual-audit.json)：20行主指标、5504行原文计数、396组配对比较、264组区间、51个核心完成文件的SHA链均通过。模型仍为验证集选出的joint-MLP128，阈值仍为0.620969696969697，没有使用这批测试调参。

| 独立确认组 | System precision | recall | F1 | Balanced accuracy | 五类准确率 |
|---|---:|---:|---:|---:|---:|
| 单消息（400原文） | 98.46% | 96.42% | 97.43% | 98.02% | 98.78% |
| 构造多轮（同400） | 99.66% | 99.11% | 99.38% | 99.53% | 99.28% |
| up-to1024（原400中64） | 97.43% | 94.77% | 96.08% | 97.07% | 98.00% |
| 真正1024正文（另64） | 96.72% | 87.19% | 91.71% | 93.23% | 94.93% |
| 512–1023尾部（同另64） | 96.28% | 83.93% | 89.68% | 91.56% | 93.37% |

相对同次共享前向的冻结dense128，完整长文system F1提高28.18个百分点，2000次来源配对bootstrap的95%区间为[24.52,32.13]；尾部提高32.44点，区间[28.08,37.12]。三组较短native输入通过P/R/BAcc各90%的工作目标，但完整长文和尾部的recall仅87.19%与83.93%，因此本轮**未达成全部工作目标**，不替换旧冻结primary，也不把开发验证收益当作最终达标。下一轮只在开发数据中探索目标对齐的阈值/小模型集成，并评估增加独立长训练来源；任何新的模型和阈值都需要另一批新来源确认。

[完整报告](outputs/local-gptoss120b/systemness-head-confirmation/report.txt)、[逐组指标](outputs/local-gptoss120b/systemness-head-confirmation/metrics.csv)、[原文配对区间](outputs/local-gptoss120b/systemness-head-confirmation/paired-comparisons.csv)、[12个固定新来源五角色四模型图集](outputs/local-gptoss120b/systemness-head-confirmation/plots/gallery-index.md)、[1024正文位置曲线](outputs/local-gptoss120b/systemness-head-confirmation/plots/long-full-system-by-position.png)、[发布证据](outputs/local-gptoss120b/systemness-head-confirmation/publication-complete.json)。已实际查看4张不同PNG（含位置图）；曲线为不等宽离散位置bin的无平滑点估计，区间另见表。


128根公开自然对话的[正式报告](outputs/local-gptoss120b/systemness-natural-test-confirmation/report.txt)及[实际独立结果审计](outputs/local-gptoss120b/systemness-natural-test-confirmation/independent-audit/actual-audit.json)已完成。下表为固定head128及原阈值在all-prepared的结果，clean-test因零排除而完全相同；每个数据源64段，共32个system措辞依赖组件。

| 新自然对话组 | System precision | recall | F1 | Balanced accuracy | 五类准确率 |
|---|---:|---:|---:|---:|---:|
| 全部128段 | 99.50% | 99.65% | 99.57% | 99.79% | 97.88% |
| OASST（64） | 99.34% | 99.72% | 99.53% | 99.82% | 97.39% |
| ToxicChat（64） | 99.67% | 99.57% | 99.62% | 99.76% | 98.58% |

相对同次前向的冻结dense128，五类准确率提高7.24个百分点，2000次依赖组件配对bootstrap的95%区间为[5.13,9.52]；system F1仅提高0.071点，区间[−0.324,0.446]跨0，不能声称system F1显著提升或所有来源/指标均改善。全部33476内容位置参与评估，只存在S/U/A真值，CoT/tool支持为0、recall记N/A。公开回复不由120B产生，也不在新分配system措辞下生成；本试验评价形式角色读出，不能据此推断遵循system的行为能力。较短自然对话通过system工作目标，但不改变上方C4长文/尾部recall未达标的事实。

[完整24行指标](outputs/local-gptoss120b/systemness-natural-test-confirmation/metrics.csv)、[组件配对区间](outputs/local-gptoss120b/systemness-natural-test-confirmation/paired-comparisons.csv)、[12个固定来源的五行四模型图集](outputs/local-gptoss120b/systemness-natural-test-confirmation/plots/gallery-index.md)、[汇总图](outputs/local-gptoss120b/systemness-natural-test-confirmation/plots/natural-test-summary.png)、[完成发布清单](outputs/local-gptoss120b/systemness-natural-test-confirmation/publication-complete.json)。已实际查看OA/TC各例及汇总，13对PNG/PDF齐全，未因低分替换来源；汇总为无置信带的点估计，区间另列于表。


[开发验证集的集成与目标对齐校准](outputs/local-gptoss120b/systemness-head-calibration-v2/report.json)已正常完成：固定64/128两模型、5个混合权重、2种阈值目标共10项，均保留[60行分组指标](outputs/local-gptoss120b/systemness-head-calibration-v2/metrics.csv)。混合未超过通过双重保护条件的纯128模型。maximin目标将6组的18项P/R/BAcc最低值从90.83%提高到92.84%，开发尾部recall由90.83%到92.84%，但平均五组system F1从97.27%降至96.93%，自然组precision也下降；新的0.4495959595959596仅是开发诊断阈值，没有替换正式0.620969696969697，更未在已见测试上挑选。

首版新maximin分支发现NumPy2的F32概率/F64分位阈值边界计数差异，已[原样归档并明确已知问题](outputs/local-gptoss120b/systemness-head-calibration-diagnostic-v1/known-issue-and-completion.json)；单独v2在真实概率访问前[冻结源码和协议](outputs/local-gptoss120b/systemness-head-calibration-v2-preparation/root-approval.json)，仅修复新maximin分支。旧mean-F1控制及既有正式模型/阈值均未改动。实际v2对所有混合权重、每个网格阈值直接计数，44460项P/R/BAcc统计差为0；原5条mean-F1记录与首版相同。[完成及复核记录](outputs/local-gptoss120b/systemness-head-calibration-v2/completion-and-review.json)。这些结果不能当作新独立确认；下一轮拟增加256个独立长正文fit来源及96个validation来源，并以等量旧长文重复采样作训练预算对照。


下一轮[长正文扩展设计](outputs/local-gptoss120b/systemness-long-augmentation-preparation/prospective-experiment-design.json)已在新候选正文读取前声明。固定扫描未使用的4111–12302行，选256个fit及96个validation真1024-token来源；不足即停，不扩大窗口。新数据与旧578090行组合训练MLP128，另设增加等量旧132篇长文重复采样的匹配控制；两者每轮905770行、222次更新，使用相同旧fit-only scaler及初始权重，实际早停步数另报。六条模型/阈值记录完整保留，只有预定new-maximin候选可晋级，须同时胜过matched与纯重新校准控制并通过逐组退步限制。新增96篇属于开发验证，之后仍须全新来源确认。该声明发布时处于实现/CPU检查阶段；后续来源准备及实际独审已完成，四卡特征提取正在进行，新模型训练成绩尚未产生。


扩展长正文的正式CPU准备已正常退出：固定8192行中752条达到1024-token最低长度，精确正文/前缀去重后751条，按冻结规则取352个新来源并独立划分fit256/validation96。历史2828个已知C4来源已恢复完整/192/1024前缀指纹，并排除原249两份库存；原249缺完整URL/行号的限制仍保留。[准备完成](outputs/local-gptoss120b/systemness-long-augmentation/data-complete.json)、[扫描](outputs/local-gptoss120b/systemness-long-augmentation/scan-audit.json)、[排除审计](outputs/local-gptoss120b/systemness-long-augmentation/exclusion-audit.json)。共1760个native输入、1814032输入tokens，独立解析1802240个完整内容位置后取389120个特征位置，五类严格等量、system标签正常保留。新数据尚未证明准确率收益。

[独立新bank提取入口](systemness_long_augmentation_extract.py)及[元数据提取计划](outputs/local-gptoss120b/systemness-long-augmentation/extraction-plan.json)在GPU启动前固定；[CPU合成及独立附审](outputs/local-gptoss120b/systemness-long-augmentation-preparation/extractor/preparation.json)验证完整行覆盖、native位置、所选头顺序、禁写hook和断点恢复，不代替真实四卡运行。新bank为389120×512 F16（380MiB），旧bank保留；仍使用四张A40上冻结的原MXFP4底座。


新352篇来源的[实际独立重放审计](outputs/local-gptoss120b/systemness-long-augmentation/preparation-audit/actual-source-replay.json)已正常退出：从固定C4 archive恢复2828个历史来源与352个新来源及原249双库存的完整/192/1024指纹，未发现行号、URL或正文/前缀碰撞；用独立native解析核对全部1760输入、1802240正文位置和389120抽样位置。独审未重新扫描未选中的窗口候选排序，原249的行号/URL限制不变；不读取任何分类器概率。[来源准备发布清单](outputs/local-gptoss120b/systemness-long-augmentation/source-preparation-publication.json)。后续四卡提取不修改这批已完成源材料。


四卡新bank提取进程已启动，原MXFP4模型加载完成，禁写采集hook的最终hidden states逐bit一致检查通过；[启动记录](outputs/local-gptoss120b/systemness-long-augmentation-preparation/gpu-extraction-launch.json)仅记录运行状态，不是完成或accuracy证明。[匹配训练入口](systemness_long_augmentation_probes.py)及[训练协议](outputs/local-gptoss120b/systemness-long-augmentation-probes/protocol.json)已在任何新验证概率访问前冻结，[合成/独立CPU审查](outputs/local-gptoss120b/systemness-long-augmentation-preparation/trainer/preparation.json)通过；只有同一提取进程正常退出并验证完整bank后才能启动训练。正式报告仍保留六条模型/阈值记录，new-maximin若不满足预声明保护条件则不晋级，不回退挑选其他条目冒充通过。


下一批确认的[前瞻设计](outputs/local-gptoss120b/systemness-long-augmentation-confirmation-preparation/root-prospective-design.json)拟使用528个全新C4来源（400一般＋另128个真正1024正文）及128个全新公开自然对话。新模型、matched和旧head共3个实际网络；旧head两阈值共享同一概率，仅system判断不同。C4图库固定12个一般来源、另4个完整system长文来源和位置曲线，自然图库固定6OA＋6TC及汇总，分别17/13对图；均先于概率固定，不挑高分样本。旧128段自然对话不再作为本轮新独立证明。模型须先完成开发晋级与实际独立审计才能声明测试快照；当前未选择任何新增测试ID、材料化新回复或启动测试推理。


匹配长文试验的[独立实际结果审计器](outputs/local-gptoss120b/systemness-long-augmentation-preparation/result-auditor/audit.py)已在真实训练概率产生前固定，[CPU合成及独立附审](outputs/local-gptoss120b/systemness-long-augmentation-preparation/result-auditor/preparation.json)通过。其实际入口须先核对训练终态和源码/模型/概率/抽样顺序链，再独立复算六条记录×八组的48行指标及六行仅描述的pool、全部阈值网格与保护条件；当前仅是审计准备，不能替代尚未完成的训练结果。


未来测试的[静态闭合模型快照入口](systemness_aug_confirm_snapshot.py)及[CPU/独立附审](outputs/local-gptoss120b/systemness-long-augmentation-confirmation-preparation/closed-snapshot-helper/preparation.json)已冻结。它在开发正式完成且实际独审通过、新模型按预定规则晋级后才允许真实声明；逐项绑定已审计模型路径/SHA、四条阈值、固定依赖和checkpoint元数据，不通过扫描后来新增的测试目录重建旧输入。原始特征bank记录已审计指纹，确认时不重新宣称其逐字核验；当前未声明任何真实测试模型快照。
