# Qwen3.5-397B-A17B-GPTQ-Int4：角色探针 + 模板优化（进度笔记）

用户请求（2026-11-19 23:40）：把 Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 下载到 /scratch/wz55191/Hugginface，在该模型上重复 DeepSeek 的实验
（角色探针训练，developer→system、GPT-OSS system 丢弃；baseline 与真实 delimiter 参考；分别针对 AgentDojo / IPI Arena 的模板优化；ASR；中文报告）。
用户还要求：usage limit 达到后 reset 自动继续（会话内 cron 4a7972d6 每小时 :17 触发）。

## 关键事实
- 模型目录 /scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4（220 GiB，94 个分片）；下载日志 同目录名 + .download.log；tmux 会话 qwendl（socket /scratch/wz55191/tmux-richevo.sock）。
- 结构：60 层，每 4 层一个 full_attention（15 层 × 32 头 × 256 维），其余为 linear_attention（Gated DeltaNet）；512 专家 top-10；只有专家权重做了 GPTQ 4bit。
- vLLM 服务：`--quantization moe_wna16`（README）；现有 venv chatinject-table1 是 vLLM 0.29。
- transformers 的 Qwen3_5MoeExperts 用融合 3D 参数，gptqmodel 不能直接替换 → 需要自写原生加载（打包 int4 专家，前向时按需解包）。
- 聊天模板：system 在最前（tools 说明并入 system）；tool 结果渲染为 `<|im_start|>user\n<tool_response>\n…\n</tool_response><|im_end|>`；思考 `<think>\n…\n</think>\n\n`。

## 步骤状态
1. [进行中] 下载模型
2. [ ] vLLM 服务冒烟测试（serve_vllm.sh）
3. [ ] 原生加载 + 取 full-attention 头输出（o_proj 之前）
4. [ ] 角色探针数据/训练（复用 role-analysis 的流程）
5. [ ] baseline ASR（original / original_delimiter）与 delimiter 参考 P(user)
6. [ ] 分别针对 AgentDojo / IPI 的模板优化
7. [ ] ASR、报告 REPORT.zh-CN.md、memory、（用户要求时）提交

## 2026-11-20 00:00 更新
- 已写：role-analysis/qwen35_role_probe_model.py（原生加载，PackedExperts，HeadCapture）、qwen35_role_probe_data.py（Qwen 模板渲染）、run_qwen35_role_probe.py（批量提取）。
- 探针数据已生成：role-analysis/outputs/qwen35-397b-gptq-int4-role-probe-20261120/data（5051 条，渲染已人工核对）。
- 下一步：下载完成后 → 原生加载冒烟（生成一句话验证解包正确）→ `run_qwen35_role_probe.py --stage all`（tmux，venv deepseek-v4-role-probe）。

## 2026-11-20 00:10 更新（移植状态）
- qwen35-richevo/ 是 deepseek-richevo 脚本的 Qwen 移植（文件名仍叫 ds_*）：
  - 已改好并测试：ds_common.py（Codec=官方 jinja 模板 + XML tool_call 解析）、ds_agentdojo.py、ds_ipi.py（prepare 已对 3 个 baseline 条件跑通）、ds_contexts.py、ds_serving.py、serve_vllm.sh、asr2.sh、ds_grammar.py（Qwen 近似 delimiter 词表）、ds_evolution.py / ds_evolution2.py / ds_evolution4.py（已 import 测试）、ds_scorer.py（Prober 用 qwen35_role_probe_model）、ds_refs4.py（Qwen 条件）。
  - 未改：ds_finalize4.py / ds_select4.py（no_pretext 规则要按 Qwen 改）、ds_probe_ops.py（应可直接用，未在 GPU 上验证）、报告脚本（待写）。
- baseline 条件文件：qwen35-richevo/outputs/baseline-20261120/conditions.json（goal_only / original / original_delimiter）。
- 计划顺序：下载完 → `bash role-analysis/outputs/qwen35-.../run.sh`（tmux 会话 qwprobe；preflight 里有生成自检，必须输出含 "Paris"）→ 探针训练完后 `bash asr2.sh outputs/baseline-20261120`（baseline + 真实 delimiter 参考条件，产出 AgentDojo 记录用于构建打分上下文）→ 探针服务器 + refs + 两个单独目标的进化 → ASR → 报告。
- 注意：`</tool_response>`、`<tool_response>`、`<think>` 在 Qwen 里都是真实 token，优化模板里禁止，只能用近似拼写。

## 2026-11-20 01:30
- 原生加载器已验证（生成自检输出 "The capital of France is Paris."；加载约 31 分钟；单次前向 8–20 秒）。
- 发现：批量与单条前向的后层激活只近似一致（cos 0.97–0.998），即使两行完全相同、无 padding 也如此 → 是 BF16 舍入 + top-k 路由离散翻转造成的数值噪声，不是 padding 问题；preflight 改为 cos ≥ 0.9 并记录。单条重复前向是逐位确定的。
- 探针流程已在 tmux qwprobe 重新启动（01:23）。日志 role-analysis/outputs/qwen35-.../run.log，结束标志 run.status（run.status.attempt1 是第一次失败的记录）。
- 已写未测：ds_probe_ops.py 的 `chunks`（多上下文 lockstep，一次前向里 MoE 只解包一次）和 ds_fast_scorer.score_cases；探针服务器起来后先用单个 `chunk` 对比验证。
- role-analysis/qwen35_dev_shell.py：常驻加载模型、执行目录里 .py 的调试壳（调试用）。

## 2026-11-20 03:10
- 探针训练完成（03:08）：选中 mlp128-seed43，checkpoint sha 3449f989...，dev 0.9991，test 0.9992（文章 0.9990，AgentDojo 0.9999）。16 个头主要在解码层 3 和 7。ds_common.PROBE_SHA256 自动读取 freeze.json。
- 已启动 baseline/参考 ASR：tmux qwasr，`asr2.sh outputs/baseline-20261120`，条件 original / original_delimiter / delim_close_user / delim_close_assistant_user / near_simple / goal_only。日志 outputs/baseline-20261120/{asr2.log,vllm.log,agentdojo_run.log,ipi_run.log}。
- 之后：用 baseline 的 original 条件记录构建打分上下文（deepseek-richevo/ds_build_contexts.py 需移植）→ 探针服务器（ds_probe_server.py，venv deepseek-v4-role-probe）→ 验证 chunks op → refs → 进化。

## 2026-11-20 05:30
- vLLM 可用（需 `--quantization moe_wna16`；加载 30–35 分钟；日志里大量 "memory allocation failed" 告警可忽略）。很慢：开 prefix caching（`--enable-prefix-caching --mamba-cache-mode align`）+ 96 并发后生成约 220–245 tok/s，AgentDojo 约 6.3 局/分钟（每个条件约 1 小时）。
- baseline ASR（tmux qwasr）03:50 重启后在跑，预计 10:30 左右结束（6 个条件，goal_only 在最后，可视时间在 AgentDojo 上砍掉）。
- 打分上下文面板已建好：outputs/qwevo-20261120/agentdojo/dataset.json（12 个上下文，replay/locate 已校验）和 outputs/qwevo-20261120/ipi/dataset.json（与 DeepSeek 相同的 12 个 IPI 行为）。
- ASR 结束后：起探针服务器 `ds_probe_server.py --output outputs/qwevo-20261120/probe_server`（tmux qwserver，venv deepseek-v4-role-probe，环境变量同 role-analysis run.sh）→ 验证 chunks vs chunk → ds_refs4.py 两个面板 → 写 facts.txt + run.sh（参照 deepseek-richevo/outputs/dsevo4-20261119/*/run.sh，脚本 ds_evolution4.py）→ 进化 → ds_finalize4 / ds_select4 → asr2.sh。

## 2026-11-20 08:52 —— 用户要求中断并保存进度（所有后台任务已停，GPU 空闲，自动续跑 cron 已删除）
已完成：
1. 模型下载并校验（94 分片，大小全部一致）。
2. 角色探针训练完成：role-analysis/outputs/qwen35-397b-gptq-int4-role-probe-20261120（mlp128-seed43，sha 3449f989...，test 0.9992）。
3. IPI Arena baseline/参考 ASR 全部完成（36 主行为）：original 2.8%，original_delimiter 2.8%，goal_only 5.6%，delim_close_user 11.1%，delim_close_assistant_user 25.0%，near_simple 22.2%。
4. AgentDojo baseline/参考 ASR 部分完成（outputs/baseline-20261120/agentdojo/cases，可断点续跑，已完成的局不会重跑）：
   original 1/389 (0.3%)；original_delimiter 16/389 (4.1%)；delim_close_user 178/389 (45.8%)；
   delim_close_assistant_user 251/372（未完，差 17 局）；near_simple 17/67（未完）；goal_only 0（未开始，可不跑）。
5. 打分上下文面板：outputs/qwevo-20261120/{agentdojo,ipi}/dataset.json。
未做（按顺序）：
a. 续跑 AgentDojo 剩余：`QW_AD_WORKERS=72 bash asr2.sh outputs/baseline-20261120`（tmux；vLLM 加载约 35 分钟；IPI 部分已完成会跳过）。不想跑 goal_only 的话先把它从 outputs/baseline-20261120/conditions.json 里去掉（注意 ipi 目录已有 goal_only 结果，保留即可）。
b. 探针服务器 + 验证 `chunks` op（未在 GPU 上测过）+ ds_refs4.py 参考测量。
c. 写 facts.txt / run.sh，两个单独目标的进化（ds_evolution4.py），ds_finalize4 / ds_select4。
d. 选出模板的 ASR、中文报告（报告脚本待写）、更新 memory；提交仅在用户要求时做（experiments/qwen35-richevo 和 role-analysis 的 qwen35_* 文件目前都未提交）。

## 2026-11-23 12:00 续跑
- 探针服务器 + lockstep `chunks` 已验证（12 个上下文一次 8 秒，与逐个 chunk、精确前向一致）。
- 重要发现：第一版探针（...-role-probe-20261120，头在解码层 3/7）在 Qwen 自己的对话上不成立：保留在历史里的 <think> 之后所有 token 一直判为 cot；工具结果之后的真实第二个 user turn 判为 tool（P(user) 0.06）；真实 delimiter 条件 P(user)≈0。原因：复用的 GPT-OSS 数据每条文章只有一个角色、AgentDojo 历史里没有保留的 reasoning、也没有第二个 user turn，探针学成了"粘住上一个状态"。
- 修正：qwen35_role_probe_data.py 新增 v2 数据（QWEN_MULTI=1 默认）：把文章片段拼成多角色对话（1173 条，含 reasoning→回答→工具结果→再一轮、工具结果后的 user turn），其余数据不变。新 run 目录 role-analysis/outputs/qwen35-397b-gptq-int4-role-probe-multiturn-20261123，head 扫描样本加大（QWEN_SCAN_STAT=48 / INNER=24）。tmux qwprobe 12:00 启动。
- 训练完后：把 qwen35-richevo/ds_common.py 的 PROBE_RUN 指向新目录 → 起服务器 → 先用 ds_diag_roles.py 和 ds_refs4.py 的 true_user_turn 验证（真实 user turn 应接近 1、原对话结构应标对），通过后才继续优化。

## 2026-11-23 15:40 — v2 探针仍不合格，改训 v3（agent-loop 数据）
- v2 探针（multiturn-20261123，mlp256-seed43，sha f4411a3a…，dev 0.9961 / test 0.9978）验证不过：AgentDojo 面板 true_user_turn P(user)=0.54（banking 0.48 / slack 0.27 / travel 0.87）；保留推理时工具结果 token 在 tool/cot 间跳；真实 delimiter 后 goal 判为 cot。v1 的 refs.json / verify_ad 已移到 outputs/v1probe-stale/；现有 outputs/qwevo-20261120/agentdojo/refs.json 是 v2 探针的数（也作废）。
- 诊断（ds_diag_dump.py + ds_diag_layers.py，数据在 /scratch/wz55191/qwen-diag，日志 layers.log）：逐层线性探针在真实对话上，没有任何一层能把"工具结果之后的真实 user turn"判对（最好 ~0.5），所以问题在训练数据而不是 head 选择。
- 原因：GPT-OSS 的 AgentDojo 调用历史里不带推理（Qwen 模板会保留最后一个 user 之后所有推理），也没有工具结果后的 user turn。
- v3 数据（qwen35_role_probe_data.py，QWEN_AGENT=1，FORMAT v3-agent-loop）：在 v2 基础上，用同样 359 个 benign AgentDojo 调用加 301 条 kept-reasoning（历史里恢复各轮推理）+ 301 条 followup（最后一个工具结果后，可选短 assistant 回复，再接一个 user turn，可选再接推理+回答）。划分沿用任务划分。回复词表里故意不含验证用的 "Done."。
- v3 训练：tmux qwprobe，目录 role-analysis/outputs/qwen35-397b-gptq-int4-role-probe-agentloop-20261123（run.sh all）。训练完：ds_common.py PROBE_RUN 改指向该目录 → 起服务器 → 验证（true_user_turn 两个面板，IPI 面板是训练中没有的域，作为独立检验；ds_diag_roles.py）。

## 2026-11-24 15:35 — 第一轮结果 + 第二轮（chain2）
- v3 探针（agentloop-20261123，sha 5d804b0f…）验证通过：true_user_turn AD 0.999 / IPI 0.974。之后一切用它。
- 第一轮进化（chain.sh）完成：AD 最高 0.163；IPI 最高 0.779。选入 outputs/optimized-20261123/conditions.json：opt_ad_top / opt_ad_short / opt_ipi_top。
- ASR（outputs/optimized-20261123，完整）：AgentDojo opt_ad_top 17/389=4.4%，opt_ad_short 22/389=5.7%，opt_ipi_top 70/389=18.0%；IPI 8.3% / 11.1% / 19.4%。
- baseline 续跑在 08:41 随 SLURM 作业结束而中断（1923/2334；剩 near_simple + goal_only 共 411 例）。新作业 50531686 自 14:11 起，时限 24h。
- chain2（tmux qwchain2，日志 outputs/qwevo-20261120/chain2.log）：起探针服务器 → AD 第二轮进化（agentdojo2/，用第一轮 IPI+AD 模板 warm start，2.5h）→ 精确重测 → 选 opt_ad2_top/short 到 outputs/optimized2-20261124 → 面板参考 → vLLM → baseline 续跑 → 第二轮模板 ASR。
- 之后：写 REPORT.zh-CN.md（还没有报告脚本），更新 memory。未提交。

## 2026-11-24 23:20 — 全部完成
- chain2 于 23:05 结束（各步 exit 0），GPU 已空，tmux 无会话。
- AD 第二轮：992 候选，最高精确 P(user) 0.704；选 opt_ad2_top（58 tok）/ opt_ad2_short（48 tok，0.625）→ outputs/optimized2-20261124。
- ASR：AgentDojo opt_ad2_top 85/389=21.9%，opt_ad2_short 140/389=36.0%；IPI 13.9% / 16.7%。baseline 补完：delim_close_assistant_user 252/389=64.8%，near_simple 109/389=28.0%，goal_only 28/389=7.2%。
- 报告：REPORT.zh-CN.md（手写，无报告脚本）。未提交。
