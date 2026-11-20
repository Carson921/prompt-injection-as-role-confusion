# dsevo4-20261119：断点续跑说明

背景：DeepSeek 角色探针按 developer→system、原 system 丢弃重新训练（新探针
`experiments/role-analysis/outputs/deepseek-v4-flash-0731-role-probe-devsys-20261119`，checkpoint sha `66c100b8…`，
`ds_common.PROBE_RUN / PROBE_SHA256` 已指向它）。本目录是用新探针重做的模板优化，AgentDojo 与 IPI Arena 分开优化。
GPU 作业在 2026-11-19 15:06 结束，14:56 由 `stop_and_save.sh` 停止两个优化和探针服务器。

## 已完成（都在磁盘上，不需要重做）
- 新探针训练与测试（test 准确率 0.9931）。
- 新探针下的参考值：`refs_joint.json`（旧联合面板 AD6+IPI6，8 个条件）、`agentdojo/refs.json`、`ipi/refs.json`
  （各 12 个上下文，6 个条件），汇总见 `refs.log`。
- 两个优化的部分进度：`{agentdojo,ipi}/scores.jsonl`（代理评分）、`exact_scores.jsonl`（精确复核）、`bank.json`、
  `llm_seeds.jsonl`；停止时的状态在 `stop_state.json`（已用时间、剩余时间、当前最好模板）。

## 续跑
四张 H100 空闲时执行（约 20 分钟加载模型，然后两个优化按剩余时间继续，已评分候选不会重算）：

```bash
bash experiments/deepseek-richevo/outputs/dsevo4-20261119/resume.sh
```

tmux socket：`/scratch/wz55191/tmux-richevo.sock`（会话 dsprobe、dsevo4ag、dsevo4ip）。

## 续跑之后的步骤
1. 精确复核：`ds_finalize4.py outputs/dsevo4-20261119/agentdojo 10`，IPI 同理（`8`）；输出 `exact_top.json`。
2. 选模板：每个基准取精确 P(user) 最高的一个，再取不带 `</tool_result>` 前置文本的最好一个。
3. 补测参考条件（`ds_refs4.py`，其余条件：ctrl_leadin、opt2_top2、opt2_assistant、original、original_delimiter、
   goal_only；`DS_IPI_PRIMARY=1` 可测全部 36 个 IPI 行为），以及交换实验 `ds_diag_ipi_user.py`。
4. 停掉探针服务器，用 `asr2.sh OUT_DIR` 测 AgentDojo 与 IPI Arena 的 ASR（OUT_DIR/conditions.json）。
5. 写中文报告（新旧探针对比、P(user) 与 ASR 的关系、新模板 ASR）。
