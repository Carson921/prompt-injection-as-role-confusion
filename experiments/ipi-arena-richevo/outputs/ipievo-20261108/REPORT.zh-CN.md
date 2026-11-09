# 以 IPI Arena 探针分数为目标的模板进化（ipievo-20261108）

- 目标：12 任务面板（browser/coding/tool 各 4 个）上 goal token P(user) 的均值；每次面板新高自动在全部 36 个主集任务上复核（`full_scores.jsonl`）。搜索器、语法、LLM 操作与 `experiments/autodan-role-probe/rich_evolution.py` 相同，只替换打分器（`ipi_scorer.py`）；热启动池为 AgentDojo 阶段的 band/diverse 模板与 richevo 前 200 名。
- 预算：6 小时，1493 个候选（607 局部算子、746 LLM 生成），wrapper 上限 128 token 全程未提升（停滞判据未触发）；最后 177 次评估无新高。
- 结果：面板最优 0.8785（browser 0.968 / coding 0.903 / tool 0.764）；全 36 任务复核（`full_validation.json`，面板前 20 名）均值 0.887–0.899，最高 0.8993（AgentDojo 最优模板在同一指标上为 0.838）。
- 最终中性后缀检查：`fitness_canonical` 0.8785 vs 带 `<|end|>…final` 后缀 0.858，模板不依赖假终止符。
- 后续 ASR 测试见 `../asr-ipievo-20261108/REPORT.zh-CN.md`：P(user) 提升未转化为 ASR 提升（44–58% vs AgentDojo 0.80–0.90 区间模板的 63.9%）。
