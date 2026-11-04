# 纯文章 FIT 与 TEST：是否完美分离？

**答案：没有完美分离，但在训练域接近可分。** 本页使用已冻结的scratch epoch11研究候选，不训练、不更换checkpoint，不修改此前实验或默认模型。

纯文章TEST由128篇未参与本轮FIT/DEV的源文章构成，C4与Dolma3各64篇；它们被包装为六角色原生消息及合成多轮结构。未加入AgentDojo、gardening、OASST、ToxicChat等真实对话样本。CoT标签的监督正文也是文章，不是实际对话中的模型推理轨迹。
这里核对的是来源与构造协议，不是新增的逐篇人类语义审阅；不能据此保证每一篇网页完全没有引用过对话。

下表均为当前scratch epoch11。FIT每个保留位置或固定回放索引只计一次，使用确定性推理（关闭dropout）。这不是按60/40分层抽样的优化器事件分布，不能将此Accuracy直接等同于训练日志中的loss。

| 数据 | 保留位置数 | Accuracy | 六类macro-F1 | system召回 | 错分位置 |
|---|---:|---:|---:|---:|---:|
| 新增文章 FIT | 535,328 | 99.5259% | 99.5416% | 99.7136% | 2,538 |
| 旧 FIT 固定回放子集 | 250,000 | 99.9676% | 99.9681% | 99.8663% | 81 |
| 全新文章 TEST | 133,631 | 96.2075% | 96.2366% | 96.9525% | 5,068 |

新增FIT中，system错分266/92,864，其中20个流向CoT；文章TEST中，system错分708/23,232，其中117个流向CoT（S→CoT约0.5036%）。这些结果与AgentDojo历史真实任务的S召回55.03%属于不同数据域，不能混用。

文章TEST当前checkpoint的混淆矩阵与此前封存test.json完全一致。此次补充了旧最终checkpoint在同一批新文章TEST上的对照，并首次补算两类FIT的完整混淆矩阵。

独立Torch CPU exact-GELU对全部918,959个位置、两个模型逐行核对，分类argmax差异为0；全部手工累积的混淆矩阵、Accuracy、macro-F1及逐类召回与NumPy读出一致。全程未使用GPU。

- [文章TEST：父模型与当前模型PNG](article-test-parent-vs-new.png) · [PDF](article-test-parent-vs-new.pdf)
- [训练FIT：当前模型PNG](article-fit-current.png) · [PDF](article-fit-current.pdf)
- [完整指标与来源哈希](metrics.json) · [独立数值核验](independent-numeric-validation.json)

原始文章和特征仍在既有私有备份中，本目录不包含原文输入、特征数组或逐token概率。
