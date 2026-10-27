# 父模型缓存复评校验

此目录只验证原父 checkpoint 的计算和绘图；左右两侧均为相同父模型，不是优化模型成绩。

覆盖718份原始特征档案和155,103个首次普通正文位置；生成保留原生 decode 状态，358个未前向停止标记排除。

没有复制原始特征、输入正文或历史对话。source-integrity-audit.json保留已消费源文件的校验哈希及分母；validation.json记录脚本SHA和硬分类结果。

图表为相同模型双边渲染校验，最终优化模型另行生成对比。

独立 Torch CPU 验收见 [independent-verifier-parent-only.json](independent-verifier-parent-only.json)：逐行覆盖675,573个已观察状态，28692个指标标量及另8568个任务／suite／S-D硬分类表格标量通过。1470个正文框包含603个空结构N/A框，保持155,103个评分位置。该验收仍使用父模型与父模型，不包含新训练候选成绩。
