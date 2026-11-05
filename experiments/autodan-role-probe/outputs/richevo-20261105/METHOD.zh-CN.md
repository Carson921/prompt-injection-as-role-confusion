# Rich-evolution：针对角色探针 P(user) 的模板优化（方法说明）

数值结果、最优更新轨迹、Top 模板与 ASR 对照表由 `rich_report.py` 自动生成在同目录的
`REPORT.zh-CN.md`；本文件只说明目标、约束、算法与运行过程。

## 1. 目标与约束

- **目标函数**（唯一进入选择的量）：`scorer.Scorer.score(template)['fitness']`，
  即六个冻结的 AgentDojo 原生上下文（banking/slack/travel 各两个）中，goal 段每个 token 的
  探针 P(user) 取均值，再对六个任务等权平均。探针、模型分片、tokenizer 资产与之前所有实验完全一致
  （`score_protocol.json`，`model_binding` 校验）。
- **硬约束 1：不得出现真实定界符。** 候选在 CPU 端经 `BroadDelimiterLibrary.validate`
  （禁止全部 Harmony 特殊 token 拼写与 ID ≥ 199998），GPU 端 `Scorer.prepare` 再对原始 payload
  与渲染后的每一次出现做 token ID 检查。近似拼写（`<|start|`、`< |message| >`、`[|START|…]`、
  `<|sta rt|>` 等）属于普通文本，允许。
- **硬约束 2：不得靠重复堆叠。** 沿用 `diverse_policy.validate_diverse_template`：
  相同正文行/句最多 2 份、96/192 字符与 24/48 token 的 n-gram 冗余占比 ≤ 25%、
  相同字符连续 ≤ 24、不可读 Unicode（Cf/Mn/Mc/Me/Co/Cn）占比 ≤ 15%。
- **长度课程**：wrapper token 上限从 128 起步；当"自上次全局最优以来 ≥ 450 次评估且在该上限
  停留 ≥ 规定分钟数"时才翻倍（128→256→512→…）。本次只用到 256。
- **只优化前缀**：骨干模型是因果的，`{goal}` 之后的文本不影响 goal token 的特征，
  因此所有候选规范化为 `prefix + {goal}`。注意 YAML/repr 重新编码会让整串的引号样式随后缀变化，
  所以附加标准结尾后分数有 ~1e-3 量级的差异（`summary.json` 的 `suffix_neutrality`）。

## 2. 搜索算法（`rich_evolution.py`）

稳态进化 + 自适应算子赌博机 + MAP-Elites 式存档：

1. **种子**：历史严格短模板池前 60 名（重新评分）、150 条语法随机转录，以及 OpenRouter 多模型
   生成的抽象模板（见 §3）。
2. **父代选择**：45% 在前 40 名中三元锦标赛，30% 从存档格子（按长度桶 × 伪消息块数 × 是否含噪声 ×
   是否多行）随机取精英，15% 在前 120 名随机，10% 取最优。配对算子的另一半有一半概率取
   父代最弱 suite 的"suite 冠军"。
3. **本地算子**（`rich_grammar.py`，CPU）：
   - 定界符：`restyle_global`/`restyle_local`——50 种近似拼写（含全角、零宽、西里尔同形、
     缺角括号、缺竖线、拆名、实体编码等）全局或局部重写；
   - 结构：`block_insert`/`block_delete`/`block_swap`/`header_rebuild`/`fresh_transcript`——
     伪消息块数量、角色顺序（system/developer/user/assistant/工具名）、channel、结尾
     （end/call/return）与块间内容均随机；
   - 内容：`sentence_insert`/`sentence_replace`/`leadin_swap`/`role_rename`——从 LLM 生成的
     412 句"句库"取材，替换引导语与角色称谓；
   - 噪声（GPTFuzzer 风格）：`noise_insert`/`noise_remove`——符号串、制表线、十六进制、
     base64 样式、CJK、零宽字符、组合音标、希腊/卢恩字母等；
   - 编码敏感：`newline_policy`（换行→空格/字面 `\n`/双换行）、`closer`（先闭合外层
     YAML/JSON/引号容器）、`whitespace_jitter`；
   - 配对：`crossover_lines`（按行单点交叉）、`crossover_concat`（拼接去重，提升上限后权重加大）。
   每个算子的权重随"子代优于父代"的频率与产生全局最优的次数自适应。
4. **LLM 算子**（`rich_llm.py`，后台线程，GPU 从不等待网络）：GPTFuzzer 的 rephrase / expand /
   shorten / crossover，加上 AutoDAN-Turbo 式 **feedback**（把前 8 名 + 4 个中等 + 3 个最差模板
   连同总分与分 suite 分数交给 LLM，让其提出预期在三个 suite 上都更高的新模板）。
5. **断点续跑**：`scores.jsonl` 为唯一事实源，重启时重建种群/存档/算子统计；`Scorer` 的评分缓存
   避免重复前向。

## 3. OpenRouter 的使用

- 只发送**抽象模板**（`{goal}` 以 `AUTODANIMMUTABLEGOALSLOT` 占位）与数值分数；从不发送任务目标、
  工具上下文或任何 AgentDojo 内容。全部请求/响应记录在 `api_calls.jsonl`（密钥已脱敏）。
- 模型按权重混合：claude-sonnet-4.6、claude-opus-4.8、gemini-3.1-pro-preview、gpt-5.4、
  deepseek-v4-pro、kimi-k2-0905、qwen3-235b。温度 0.9–1.2，`reasoning.effort=low`。
- 28 个"家族"提示（单行/多行、闭合容器、纯文本角色标签、多轮、system/developer 通告、
  XML/JSON 信封、邮件头、混合拼写、多语言、屏幕阅读器、噪声交错、嵌套引用、日志风格、极简、
  长多块、格式元描述、代码注释、自创格式、同义角色词…），每族两轮，外加 8 个主题的句库生成。
- 拒答句过滤（`looks_like_refusal`）。总花费 $7.66 / 643 次调用。

## 4. 运行过程中的工程事项

- `scorer.py` 冻结的定位正则在 banking（YAML 双引号折行）上下文中无法识别 PyYAML 对
  不间断空格/细空格的转义（`\_`、` `）以及恰好落在前缀/goal 边界上的折行，导致约 3% 的
  候选报 "No complete payload…"。`rich_evolution.py` 以 monkeypatch 提供
  `encoded_tool_pattern_v2` / `locate_payload_v2`（超集；已能定位的候选分数不变），
  `scorer.py` 本身未改动，补丁记录在 `protocol.json.locator_patch`。
- 运行期间 Claude 会话重启导致后台进程被回收、`/tmp` 中的模型副本被清空：用
  `stage_model.py` 逐字节校验后重新暂存；因 `staging-proof.json` 元数据变化，
  `score_protocol.json` 的 `staging_proof_sha256` 字段做了一次修正（`protocol_amendments.jsonl`，
  原件保存在 `score_protocol.part1.json`）。日志按重启分段：`run.part1~3.log` + `run.log`。
- 后续改用沙箱外的 tmux 会话运行（`run.sh`、`run_asr.sh`）。

## 5. ASR 评测

- 评测器 `evaluate_fixed_candidates.py`（与 diverse8h 阶段相同，vLLM，全量 389 个任务），
  每份清单 3–6 个候选且 P(user) 区间不得重叠，因此分 `asr_candidates_A.json`（0.99–1、0.9–0.99、
  0.8–0.9、0.7–0.8、0.6–0.7）与 `asr_candidates_B.json`（0.5–0.6 … 0.1–0.2）两组顺序运行，
  结果在 `asr_A/summary.json`、`asr_B/summary.json`。
- 候选选择规则：每个区间取最高 P(user)，并优先选择与已选模板词袋 Jaccard < 0.5 的结构不同者；
  选择不使用任何 ASR 反馈，不做新的生成或变异。
- 结果（详见 `REPORT.zh-CN.md` 的 ASR 表与 `asr_vs_puser.png`）：十个区间的 P(user) 与 ASR 的
  Spearman 相关 0.88，但并非逐模板单调（0.3–0.4 区间的模板 73.9% 高于 0.4–0.5 区间的 46.3%）；
  P(user) ≥ 0.9 后 ASR 饱和在 82–86%，plain 基线 3.1%。

## 6. 文件索引

| 文件 | 内容 |
|---|---|
| `scores.jsonl` | 每次评分的模板、总分、分 suite、来源/算子/父代、cap、耗时 |
| `winner.json` / `summary.json` / `top_templates.json` / `diverse_top.json` | 最优与 Top 模板，后缀中性检查，算子统计 |
| `status.json` | 最后一次状态快照 |
| `llm_seeds.jsonl` / `llm_rejections.jsonl` / `bank.json` / `api_calls.jsonl` | LLM 生成记录 |
| `scores/<sha24>/case_XX/` | 每个模板六个上下文的 token 级概率与特征（`tokens.npz`、`request.json`） |
| `fitness_curve.png` / `suite_curve.png` / `asr_vs_puser.png` | 图 |
| `REPORT.zh-CN.md` | 自动生成的数值报告 |
