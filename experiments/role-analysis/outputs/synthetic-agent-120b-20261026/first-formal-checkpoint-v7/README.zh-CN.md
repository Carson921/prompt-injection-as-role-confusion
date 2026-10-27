# 首个正式六角色分类器 checkpoint

固定首候选 `linear-seed17` 已完成 18 epochs，按固定 DEV 规则保存最佳第 10 轮。该包用于首候选比较；整轮最终模型由七候选 DEV 规则产生。

| DEV 指标 | 值 |
|---|---:|
| 汇总 macro-F1 | 98.484161% |
| 结构等权 macro-F1 | 99.577424% |
| 最弱类别 min(precision, recall) | 96.282971% |
| NPZ / PyTorch 最大绝对误差 | 1.84774399e-06 |

`best.npz` SHA256：`175cc3621d014d0381c90b15a20c549fc5500f6165f48b8de2f4c0614ba3623a`。其完成标记已在 2026-10-27 00:50:08 UTC 的一次性派发 claim 前就绪，落在用户提问后的八小时窗口内。

输入为冻结 GPT-OSS-120B、按 NPZ 元数据中 16 头顺序拼接的 `pre-o_proj` 原始 1024 维特征。六类顺序为 `system, developer, user, cot, assistant, tool`。[消费端](../../../synthetic_agent_classifier.py)的 `load_probe` 自动应用该模型自己的 FIT 标准化器；不要重排头或重复标准化。模型、tokenizer、MXFP4 后端和格式绑定都在 NPZ 元数据中。

原始八文件哈希与远端来源见 [source-artifacts.json](source-artifacts.json)，CPU 加载与独立绑定验收见 [validation.json](validation.json)。逐 epoch 原始 history、冻结 120B 和大型特征继续保留在远端；完整七候选的小包将在整轮完成后导出。

[Gardening 真实对比结果](../gardening-pilot-vs-first-formal/README.zh-CN.md) · [运行与恢复说明](../../../SYNTHETIC-AGENT-RUNBOOK.zh-CN.md)
