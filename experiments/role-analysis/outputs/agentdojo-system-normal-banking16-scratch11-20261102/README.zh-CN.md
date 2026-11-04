# 同日同H100对照：banking16正常system与调换system

使用同一scratch epoch11固定checkpoint，SHA256 `45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df`；研究候选状态不变，无训练、重选或GPU前向。

用户提供的正常顺序包覆盖banking16、75调用、5,325个system正文位置。其完整462项SHA声明已验证；raw BF16 states和X.npy逐值相等。配对组从已计算的完整调换57任务中，严格筛选相同banking16，而非使用57任务总体均值。

## 主比较：同16任务各首次调用

两组各16调用、1,136个system正文位置。下表是逐token softmax均值，单位%；不是argmax召回。≈0表示小于0.0001%。

| 分段 | 顺序 | token | system | developer | user | CoT | assistant | tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| identity_and_date | 正常 | 544 | 99.8980 | 0.0209 | 0.0384 | 0.0428 | ≈0 | ≈0 |
| identity_and_date | 调换 | 544 | 11.5774 | 88.3005 | 0.0489 | 0.0725 | 0.0003 | 0.0004 |
| reasoning_setting | 正常 | 80 | 79.4536 | 0.2587 | 0.0489 | 20.2387 | ≈0 | ≈0 |
| reasoning_setting | 调换 | 64 | 22.6571 | 74.9641 | 0.0406 | 2.3383 | ≈0 | ≈0 |
| channel_rules | 正常 | 288 | 7.0813 | 0.7645 | 0.0460 | 92.1081 | ≈0 | ≈0 |
| channel_rules | 调换 | 304 | 72.4192 | 10.7015 | 0.0007 | 16.6695 | 0.0020 | 0.2070 |
| tool_routing | 正常 | 224 | 0.1293 | 14.3091 | 0.0002 | 85.5614 | ≈0 | ≈0 |
| tool_routing | 调换 | 224 | 14.5002 | 81.4745 | 0.0018 | 4.0229 | 0.0006 | ≈0 |

## 配对核验

两组16份原始首轮请求JSON的SHA256逐一完全一致；原始system文本也完全一致（日期2026-11-02）。模型权重与tokenizer绑定、原16头、H100后端配置、生成参数均一致。采集源码差异经只读对比/AST检查：任务范围、元数据标记、normal render和相应校验有变化；公共数值捕获函数及前向/解码核心未变。没有执行压缩包中的代码。

因此这个配对比较消除了此前历史参考的日期与硬件差异。它支持probe对system内部顺序及前缀上下文敏感，但AB→BA还同时改变了位置、前置内容和空白边界，仍不能单独归因于纯绝对位置。16任务也只使用一种system模板，不能外推为16种独立提示模板。

正常顺序是identity/date→reasoning→channel rules→tool routing；调换为channel rules→tool routing→identity/date→reasoning。分段按原始字符锚点和token中点确定，未改动任何输入字符。正常每调用四段按上述正常顺序为34/5/18/14token；调换按其顺序为19/14/34/4，差异来自原样保留且随AB→BA移动的换行。

## 其余统计与校验

正常组全部75调用及调换banking组全部78调用也有单独统计。后续轨迹的调用数量和上下文不同，不把它们当作一一配对的首次输入。

正常组全5,325行独立Torch CPU重算最大概率差1.07e-06,argmax差异0；四段tokenID/位置均独立核对通过。配对筛选另以16任务的独立逐任务CSV加权核对，最大均值差8.33e-17,通过。

- [配对六类概率CSV：首次及全部调用](matched-banking16-comparison.csv)
- [配对对照图PNG](banking16-normal-vs-swapped.png) · [PDF](banking16-normal-vs-swapped.pdf)
- [完整配对指标与控制条件JSON](matched-banking16-comparison.json)
- [正常组自身完整指标](metrics.json) · [正常组自身分段CSV](segment-probabilities.csv)
- [配对控制验证](paired-control-validation.json) · [独立筛选验证](independent-paired-selection-validation.json)
- [独立数值验证](independent-numeric-validation.json) · [独立对齐验证](independent-alignment-validation.json)

`system-normal-probabilities.png/pdf`仅为正常banking16与更早历史正常banking16的辅助核对图；主配对比较请使用banking16-normal-vs-swapped图。

原始特征与逐位置概率保存在仓库外`/home/wentian/Github/SPAR/runpod-backups/agentdojo-gpt-oss-120b-system-normal-banking16-20261102`；原始194MB压缩包保留在用户提供的位置。完整调换57任务的源文件和结果未改写。
