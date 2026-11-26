# CoT Forgery：AgentDojo benign traces 上的角色探针复现

实验日期：2026-11-25。此目录将[原仓库 CoT Forgery 实验](https://github.com/role-confusion/prompt-injection-as-role-confusion)的工具注入与角色概率分析迁移到现有 AgentDojo benign traces，并增加无害的受控续写。本文说明方法和运行命令；推理进度、结果与最终结论另见运行目录中的报告。

完成结果见 [REPORT.zh-CN.md](outputs/gptoss120b-20261125/REPORT.zh-CN.md)。仓库保留复现代码、固定输入、图表、CSV、续写及审计结果；原始 head 特征、逐输入记录和重复依赖/来源快照保留在完整本地运行目录。`delivery-manifest.json` 与 `SHA256SUMS` 描述完整本地包，不能用于核验仅有 Git 文件的检出；已提交结果的独立清单与校验命令见 [PUBLICATION.zh-CN.md](outputs/gptoss120b-20261125/PUBLICATION.zh-CN.md)。

## 输入与实验条件

使用用户指定的 [checkpoint.npz](../role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz)，SHA256 为 `fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33`。它是六分类 MLP：按 checkpoint 中的固定顺序拼接 **16 个 attention head 的 pre-o_proj 输出**，得到 1,024 维特征，经保存的 mean/scale 标准化后输入 `1024 → 256 → 6` 网络，使用 exact GELU 和 softmax。类别顺序为 `system, developer, user, cot, assistant, tool`；CoTness 指 `P(cot)`。这不是线性探针，也不是单独读取第 16 层 residual stream；模型、head 顺序、特征位置和标准化均由 checkpoint 决定。

固定来源是 57 个无注入 AgentDojo 任务，沿用 probe 来源中的 FIT 40 / DEV 8 / TEST 9 划分。工具条件直接读取第一份真实工具暴露请求的原始 `input.npz`，保留所有已有 token、工具正文、外层 Harmony 标签及末尾未闭合的 assistant 开头。原生 CoT 参考保留首次真实生成的精确 token 前缀，不重新生成或改写正文。

[prepare.py](prepare.py) 读取原始 [injections.yaml](../cot-forgery-agent-evals/prompts/injections.yaml)，为每个任务建立以下输入：

| 条件 | 每任务数量 | 内容 |
| --- | ---: | --- |
| `no_injection` | 1 | 原始工具暴露请求 |
| `base-injection` | 5 | 原 YAML 五种直接注入文本 |
| `cot-forgery-injection` | 5 | 原 YAML 五种指令加伪造推理文本 |
| `native_cot` | 1 | 原生首次 CoT 的参考前缀 |

共 684 个主实验输入。注入位置为第一段工具正文的原生 END 前，追加 `\n\n` 和完整 YAML 文本；仅新增文本被 tokenization。保存 `original_tool`、`direct_instruction`、`forged_cot`、`payload_all`、`entire_tool` 等 token 位置掩码，并记录来源、split、正文和文件哈希。全部原始恶意 payload 只用于 teacher-forced probe 评分；不会据此生成后续工具行为或执行工具。

原 agent 实验将 payload 插入 Wikipedia HTML，进行 ReAct 交互和攻击结果分类；本实验改用 AgentDojo 原生工具返回，probe 架构也不同，因此是角色混淆机制的迁移复现。原 YAML 中部分 forgery 的第一段还增加了 lucky coin、green shirt 等限定语，主实验两条件的差异不能完全归因于第二段伪造推理。原仓库的 agent ASR 绘图把成功与未成功的攻击尝试合并计数；这里不会由 probe 概率推断对应 ASR。

## 无害受控续写

[prepare_behavior.py](prepare_behavior.py) 仅使用 TEST 9，每任务建立 `control`、`direct`、`forged` 三条件，共 27 个额外输入。目标是最终回答一个任务特定的无害验证字符串。`direct` 和 `forged` 使用**完全相同的直接指令**；`forged` 仅追加第一人称推理段，外层工具标签保持不变。

合计 711 个 probe 输入，并对补充实验的 27 个输入各进行一次 greedy 续写：固定 seed `20261125`，最多 512 个新 token，只观察首次响应，不执行任何生成的工具调用。主要观察事件是一个以原生 RETURN 结束的普通 final 消息，其全文精确等于目标字符串。另记录 final 中包含目标、任意生成文本中包含目标、工具调用停止和长度截断。

该补充衡量首次响应的受控指令重定向，不是完整 AgentDojo 安全 ASR，也不是原始外传攻击成功率。达到 512 token 上限的样本表示长度删失；在固定预算内属于“未观察到成功”，不能解释成无条件失败。逐例停止原因和生成内容应与汇总比例一起阅读。

## 统计与解释范围

主要分析使用 TEST 9，以任务为单位：先对 token 求区域均值，再在每个任务内平均五组 payload，最后对任务做配对比较和 bootstrap。不能把同一任务的五组文本或大量 token 当作独立样本。ALL 57 仅作描述性分析，因为 FIT、DEV 参与过 probe 拟合或选择。TEST 9 相对于当前 probe 的拟合划分留出，但 head 的历史发现可能早于此划分，不能声称整个特征发现流程均独立于 TEST 9。

CoTness 表示探针训练类别中的相似程度；它本身不能证明模型采纳了伪造政策、改变了真实信任边界或完成了攻击。无害续写为该概率读数补充一个有限预算的行为观察。

## 复现命令

以下命令从仓库根目录执行，依赖已安装的 `.venv`、本地原生 MXFP4 模型、冻结的来源文件及四张可用 H100。每个 worker 使用两张卡；两个 worker 按输入序号交错分片。选择新的输出目录；主输入准备步骤拒绝覆盖已有 `inputs.jsonl`。本次运行的结果位置是 [outputs/gptoss120b-20261125](outputs/gptoss120b-20261125)，以下使用独立复算目录。

```bash
REPO=/home/wz55191/Github/prompt-injection-as-role-confusion
cd "$REPO"
PYTHON="$REPO/.venv/bin/python"
SCRIPTS="$REPO/experiments/cot-forgery-agentdojo"
OUT="$SCRIPTS/outputs/reproduce-gptoss120b-20261125"
MODEL=/tmp/wz55191-agentdojo-first-tool-model
PROBE="$REPO/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz"
SOURCE=/scratch/wz55191/agentdojo-trace-source-20261112/full57_no_payload_first_occurrence_probe_v7_20261027
TRACES="$REPO/experiments/role-analysis/outputs/agentdojo-trace-interventions-20261112/sources/traces.jsonl"

"$PYTHON" "$SCRIPTS/prepare.py" --output "$OUT" --tokenizer "$MODEL" --probe "$PROBE" --source-root "$SOURCE" --traces "$TRACES"
"$PYTHON" "$SCRIPTS/prepare_behavior.py" --output "$OUT" --model "$MODEL"
"$PYTHON" "$SCRIPTS/validate.py" --output "$OUT" --probe "$PROBE" --model "$MODEL" --source-root "$SOURCE" --traces "$TRACES" --inputs-only

CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" "$SCRIPTS/run.py" --output "$OUT" --model "$MODEL" --probe "$PROBE" --worker 0 --workers 2 --mode both --max-new-tokens 512 > "$OUT/worker0.log" 2>&1 &
WORKER0_PID=$!
CUDA_VISIBLE_DEVICES=2,3 "$PYTHON" "$SCRIPTS/run.py" --output "$OUT" --model "$MODEL" --probe "$PROBE" --worker 1 --workers 2 --mode both --max-new-tokens 512 > "$OUT/worker1.log" 2>&1 &
WORKER1_PID=$!
wait "$WORKER0_PID"
wait "$WORKER1_PID"
```

两个 worker 均成功结束后，合并其行为 JSON 数组，再分析与独立校验：

```bash
"$PYTHON" - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
out = Path(sys.argv[1])
rows = []
for worker in (0, 1):
    rows.extend(json.loads((out / f'behavior-worker{worker}.json').read_text()))
assert len(rows) == 27 and len({row['input_id'] for row in rows}) == 27
rows.sort(key=lambda row: row['input_id'])
(out / 'behavior.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
PY
"$PYTHON" "$SCRIPTS/analyze.py" --output "$OUT"
"$PYTHON" "$SCRIPTS/validate.py" --output "$OUT" --probe "$PROBE" --model "$MODEL" --source-root "$SOURCE" --traces "$TRACES"
```

准备与来源记录位于 `protocol.json`、`source_inventory.json`、`behavior-protocol.json`；逐输入读数位于 `readouts/`、`records/`，续写位于 `continuations/`。分析生成的图表、数据表和独立校验报告均保存在同一输出目录。成功完成的判据应结合 worker 清单、全部预期读数、续写记录及独立校验结果，而非只看单张图。
