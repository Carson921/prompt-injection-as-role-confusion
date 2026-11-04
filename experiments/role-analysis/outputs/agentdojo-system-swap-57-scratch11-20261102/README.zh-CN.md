# 完整57任务：调换system顺序后的六类预测概率

固定用户选中的scratch epoch11研究候选，checkpoint SHA256 `45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df`。全部计算复用包内pre-o_proj原16头状态，使用原FIT mean/scale，无训练、重新选模或新大模型前向。

数据来自用户提供的`agentdojo-gpt-oss-120b-system-swapped-57-20261102.tar.gz`，覆盖banking16、Slack21、travel20，共57任务、386调用、27,406个system正文位置。整包2,139项SHA声明均已核验，另对控制SHA文件本身计算了哈希。未执行包内Python代码；仅解包system_probe和顶层元数据，其他完整对话状态只作流式校验。

## 主表：每个任务首次调用

为与此前首次system评估一致，每任务只取首次调用，共57调用、4,047个位置。所有数值是逐token softmax概率均值，单位%；不是argmax召回。`≈0`表示小于0.0001%。

| 分段 | token | system | developer | user | CoT | assistant | tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel_rules | 1083 | 72.0303 | 12.1820 | 0.0008 | 15.5261 | 0.0023 | 0.2585 |
| tool_routing | 798 | 13.8288 | 81.2505 | 0.0019 | 4.9181 | 0.0007 | ≈0 |
| identity_and_date | 1938 | 11.4298 | 88.4478 | 0.0479 | 0.0737 | 0.0004 | 0.0004 |
| reasoning_setting | 228 | 21.3592 | 75.9291 | 0.0394 | 2.6723 | ≈0 | ≈0 |

## 全部386调用

全部27,406个位置均已通过checkpoint，以下保留重复system输入，长轨迹任务会有更高调用权重。

| 分段 | token | system | developer | user | CoT | assistant | tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel_rules | 7334 | 71.7482 | 12.8276 | 0.0009 | 15.1267 | 0.0025 | 0.2941 |
| tool_routing | 5404 | 12.8941 | 81.6946 | 0.0019 | 5.4086 | 0.0007 | ≈0 |
| identity_and_date | 13124 | 11.1391 | 88.7365 | 0.0483 | 0.0753 | 0.0004 | 0.0004 |
| reasoning_setting | 1544 | 20.4740 | 77.3013 | 0.0384 | 2.1864 | ≈0 | ≈0 |

## 原顺序参考与调换后比较

历史参考也是相同57个任务和同一个45c34ed8 checkpoint。下表只列平均P(system)，不等同于最终分类比例。

| 分段 | 原顺序P(system) | 调换后P(system) | 调换后的最高平均类别 |
|---|---:|---:|---|
| channel_rules | 8.6403% | 72.0303% | system: 72.0303% |
| tool_routing | 0.0760% | 13.8288% | developer: 81.2505% |
| identity_and_date | 99.8801% | 11.4298% | developer: 88.4478% |
| reasoning_setting | 79.3079% | 21.3592% | developer: 75.9291% |

四段仍属于同一条system消息。顺序为channel_rules→tool_routing→identity_and_date→reasoning_setting。按原始AB→BA保留空白，开头两个换行计入channel_rules，没有修补`functions'.You`的连接。每调用对应19/14/34/4个token，共71个；独立retokenize核对全部386调用的tokenID、原输入位置和分段，未发现跨分段BPE token。

原始BF16 safetensors与导出的FP32 X逐值一致，按原始X直接读出；额外按旧缓存FP16精度重算，最大概率差为9.31e-10,argmax没有变化。

前一版banking+Slack37任务的211调用、14,981行与完整版对应原始特征逐值一致；分类概率差最多9.83e-07,argmax差异0。完整版新增20个travel任务，不改写旧37任务产物。

历史原顺序参考与本次运行的日期/硬件不同（2026-10-21、4A40对2026-11-02、H100），且AB→BA改变分段边界的空白分布。因此这里是上下文变动后的诊断对比，不能单独作为纯位置效应的因果证明。57任务共用一种system模板，也不能视为57种独立system提示。新模型仍为DEV未达全部门槛的研究候选，没有替换默认模型。

独立Torch CPU exact-GELU重算全部27,406行，最大概率差1.94e-06,argmax差异0。首次调用和全部调用的六类均值及硬分类计数均独立复核通过。

- [完整分段概率CSV：首次调用/全部调用/历史参考](segment-probabilities.csv)
- [全部system正文的整体均值CSV](overall-system-probabilities.csv)
- [逐任务首次调用CSV](per-task-initial-probabilities.csv)
- [完整指标JSON](metrics.json)
- [原顺序/调换后PNG](system-swap-probabilities.png) · [PDF](system-swap-probabilities.pdf)
- [独立数值核验](independent-numeric-validation.json) · [独立对齐核验](independent-alignment-validation.json) · [前37任务一致性](previous37-overlap-validation.json)

原始system特征、逐位置概率、整包SHA流式清单及复现副本保存在仓库外的`/home/wentian/Github/SPAR/runpod-backups/agentdojo-gpt-oss-120b-system-swapped-57-20261102`。完整1.4GB原始压缩包仍保留在用户提供的位置。
