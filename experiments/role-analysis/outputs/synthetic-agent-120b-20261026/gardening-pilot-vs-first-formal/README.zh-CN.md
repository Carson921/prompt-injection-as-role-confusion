# Gardening：小试与首正式候选

使用[论文 Appendix E](https://arxiv.org/html/2603.12277v6)的同一公开对话和三种原文格式。小试为 `mlp256-seed29`（最佳第15轮）；首正式候选为 `linear-seed17`（18轮完成、最佳第10轮）。每种格式仅做一次冻结120B forward，两个分类器共享16头特征，各自使用自己的 FIT 标准化器。

| 格式与指标 | 小试 | 首正式候选 |
|---|---:|---:|
| 原生 Harmony：正文角色准确率 | 94.3231% | 97.2707% |
| 全部放入 user：原语义对齐率 | 93.2314% | 90.2838% |
| 去标签：原语义对齐率 | 97.2707% | 96.5066% |

原生格式的 system 召回从 8.8% 到 56.1%，仍是主要误差来源；其余三个有真实正文的角色召回均为100%。本例支持为 S57/D0/U97/C179/A583/T0，D/T 召回为 N/A，误报保留在图与 `results.json` 中。此为历史样例诊断；正式 TEST 与七候选最终选择按主实验协议继续。

- [原生双模型 PNG](gardening-native-comparison.png) · [PDF](gardening-native-comparison.pdf)
- 小试单列：[原生](gardening-pilot-proper_tags.png) · [全部 user](gardening-pilot-everything_in_user_tags.png) · [去标签](gardening-pilot-basic_no_format.png)
- 首正式单列：[原生](gardening-first_formal-proper_tags.png) · [全部 user](gardening-first_formal-everything_in_user_tags.png) · [去标签](gardening-first_formal-basic_no_format.png)

共7对 PNG/PDF。所有图显示完整上下文与六行概率，916个正文 token 全部保留，结构/分隔符位置为灰色；无阈值或平滑。末尾🌱的字体显示为替代字形，原始 Unicode、末尾续接 token109及其概率均完整保留，脚注已说明。改写格式存在6处句末与换行融合的 BPE；只有同一格式的双模型共享完全相同的输入 ID。

[原始交付清单](delivery-manifest.json)包含31个源产物的字节数及SHA；本机逐项核验通过，并核对14个图片文件、3次forward与两个NPZ SHA。原机完成观察和推理退出证明见 [completion-observation.json](completion-observation.json)。完整指标与概率SHA见 [results.json](results.json)，公开来源协议见 [诊断协议](../gardening-diagnostic-protocol.json)。
