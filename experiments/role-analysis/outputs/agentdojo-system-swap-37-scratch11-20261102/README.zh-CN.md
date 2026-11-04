# System顺序调换：37任务的冻结checkpoint读出

使用用户指定的scratch epoch11，checkpoint SHA256 `45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df`。该模型仍是DEV未达全部门槛的研究候选，本轮没有训练、选模或替换默认模型。

原始包：`/home/wentian/agentdojo-system-swap-system-probe-banking-slack-20261102.tar.gz`。银行16任务、Slack21任务，共211调用、14,981个system正文位置。包内22项SHA256SUMS全部验证，未执行压缩包中的Python代码。

## 主表：每任务首个调用

与此前首次system口径对齐，每任务取首个调用，共37调用、2,627个位置。以下是分段内逐token softmax概率的均值，单位%；它们不是argmax召回或校准后的真实性概率。`≈0`表示小于0.0001%。

| 分段 | token | system | developer | user | CoT | assistant | tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel_rules | 703 | 72.3753 | 11.2017 | 0.0007 | 16.2078 | 0.0020 | 0.2124 |
| tool_routing | 518 | 15.3078 | 80.5638 | 0.0018 | 4.1260 | 0.0006 | ≈0 |
| identity_and_date | 1258 | 11.7144 | 88.1684 | 0.0465 | 0.0699 | 0.0004 | 0.0004 |
| reasoning_setting | 148 | 22.1841 | 75.1513 | 0.0387 | 2.6260 | ≈0 | ≈0 |

## 全部211调用

所有输入均已经过该checkpoint；以下表另保留重复system文本，长轨迹任务会贡献更多调用。它不作为37个任务等权的主口径。

| 分段 | token | system | developer | user | CoT | assistant | tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel_rules | 4009 | 72.3452 | 10.9385 | 0.0007 | 16.5032 | 0.0020 | 0.2103 |
| tool_routing | 2954 | 15.1904 | 80.6572 | 0.0017 | 4.1501 | 0.0006 | ≈0 |
| identity_and_date | 7174 | 12.1304 | 87.7466 | 0.0483 | 0.0739 | 0.0004 | 0.0004 |
| reasoning_setting | 844 | 22.4127 | 75.2515 | 0.0371 | 2.2988 | ≈0 | ≈0 |

## 相同37任务的历史原顺序参考

历史参考过滤掉travel，仅保留相同banking/slack任务，因而会与此前57任务表略有差异。新旧均使用同一45c34ed8 checkpoint。此表列平均P(system)，不是召回。

| 分段 | 原顺序P(system) | 调换后P(system) | 调换后主要类别及平均概率 |
|---|---:|---:|---|
| channel_rules | 8.6442% | 72.3753% | system: 72.3753% |
| tool_routing | 0.0768% | 15.3078% | developer: 80.5638% |
| identity_and_date | 99.8809% | 11.7144% | developer: 88.1684% |
| reasoning_setting | 79.2634% | 22.1841% | developer: 75.1513% |

## 对齐及解释边界

四段均仍处于同一条真实system消息中。按用户包原样使用AB→BA文本，未插入换行或修复`functions'.You`连接处。开头旋转来的两个换行计入channel_rules，因此每调用四段为19/14/34/4个token；原顺序的reasoning_setting末尾有换行，分段token数不完全相同。完整全文retokenize后的71个body token、ID、原输入位置及全部211调用的行映射均已独立核对，没有跨分段BPE token。

原始导出是BF16状态精确转成的FP32，直接使用checkpoint原FIT mean/scale，没有重新标准化或打乱头。原16头、pre-o_proj、模型权重及tokenizer绑定与指定checkpoint一致。额外把输入按旧缓存方式转FP16再回FP32，概率及argmax均未变化。

与历史参考相比，新运行日期从2026-10-21变为2026-11-02，硬件由4×A40变为H100，另有旋转边界的tokenization/空白差异。该对比显示预测对这次上下文变动高度敏感，但不能单凭它把全部变化归因于位置。37任务共用一种system模板，不是37种独立system提示。

## 校验与文件

全14,981行由独立Torch CPU exact-GELU重新读出，最大概率绝对误差为1.79e-06, argmax差异0。原始BF16 safetensors与X.npy逐值完全一致；首次调用与全部调用的六类均值、硬分类计数均独立复核通过。全程未新跑120B或使用GPU。

- [六类均值及原顺序参考CSV](segment-probabilities.csv)
- [逐任务首次调用CSV](per-task-initial-probabilities.csv)
- [完整结构化指标](metrics.json)
- [新旧顺序概率图PNG](system-swap-probabilities.png) · [PDF](system-swap-probabilities.pdf)
- [独立数值核验](independent-numeric-validation.json) · [独立位置/分段核验](independent-alignment-validation.json)

原始压缩包、解包文件、逐位置概率及分段索引保持在仓库外的 `/home/wentian/Github/SPAR/runpod-backups/agentdojo-system-swap-system-probe-banking-slack-20261102`。
