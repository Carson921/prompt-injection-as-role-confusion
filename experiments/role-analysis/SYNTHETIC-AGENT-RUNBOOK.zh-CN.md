# 六角色 GPT-OSS-120B 实验运行与恢复手册

本手册对应本轮最终 V7：`code-v7 / data-v7 / runs-v7`，通过 `ssh runpod` 操作同机 4×A40。实验定义见[重训计划](RETRAINING-PLAN-120B.zh-CN.md)，数据范围及审计限制见[审计说明](synthetic-agent-data-audit.md)，当前进度见[执行状态](outputs/synthetic-agent-120b-20261026/execution-status.json)。

本轮授权窗口为 **2026-10-26 13:19:17 UTC 至 2026-10-27 13:19:17 UTC**，即纽约时间 **10 月 27 日 09:19:17 截止**。以下恢复命令仅适用于该窗口剩余时间；静态文档不产生未来 GPU 运行授权。过期后保留产物并只读归档，新一轮须另行确认资源、时间和独立目录。不要修改 deadline 延长本轮。

## 1. 远端路径与整轮恢复

在远端定义本轮路径。`task_py` 保留 venv 的入口路径，不将其解析为系统 Python。依赖以[实际环境记录](outputs/synthetic-agent-120b-20261026/environment-freeze.txt)及[本轮 A40 固定依赖](requirements-synthetic-agent-4xa40.txt)为准。

```bash
ssh runpod
task_root=/root/role-retraining-20261026
task_code=$task_root/code-v7
task_py=/root/role-retraining-venv/bin/python
task_model=/root/role-retraining-model
task_cache=/root/role-retraining-cache
task_data=/workspace/role-retraining-20261026/data-v7
task_runs=/workspace/role-retraining-20261026/runs-v7

task_campaign=("$task_py" -u "$task_code/run_synthetic_agent_campaign.py"
  --python "$task_py" --code-dir "$task_code" --model-dir "$task_model"
  --data-dir "$task_data" --run-root "$task_runs" --cache-dir "$task_cache"
  --pilot-config "$task_code/synthetic-agent-120b.json"
  --formal-config "$task_code/synthetic-agent-120b-formal.json"
  --devices 0,1,2,3 --dense-diagnostics)
```

实际启动脚本为 `/root/role-retraining-20261026/launch-campaign-v7.sh`，使用上述相同参数，输出至 `runs-v7/campaign/driver.log`。先查看现有 tmux、`campaign/status.json` 及日志；仅在先前执行器已退出时启动或恢复：

```bash
tmux list-sessions
cat "$task_runs/campaign/status.json"
# launcher 会重新打开 driver.log；保留上次汇总日志，分阶段 attempt 日志仍独立保存。
if [ -f "$task_runs/campaign/driver.log" ]; then
  cp "$task_runs/campaign/driver.log" "$task_runs/campaign/driver.before-resume.$(date -u +%Y%m%dT%H%M%SZ).log"
fi
tmux new-session -d -s role-retraining-v7 'bash /root/role-retraining-20261026/launch-campaign-v7.sh'
```

恢复使用相同配置、路径、设备和开关。执行器校验 `campaign/binding.json`、各阶段 receipt 和实际文件 SHA，跳过完整阶段；失败阶段依赖已有原子分片或 `last.pt` 恢复。不要删除 receipt 强制通过。整轮锁避免重复 GPU 作业，子进程按剩余授权秒数运行，到截止时终止整个进程组。

本轮另以 `setsid -f` 启动独立监督器 [watch_synthetic_deadline.py](watch_synthetic_deadline.py)，状态位于 `/root/role-retraining-20261026/watchdog-v7.json`，[启动证明](outputs/synthetic-agent-120b-20261026/deadline-watchdog-start.json)已回传。它只匹配同 UID、实际 Python executable、本轮两个精确脚本及 V7 数据/结果参数；截止前不发信号，到期先停止调度器，再停止已有 forward、训练或 dense worker，正常完成后自行退出。状态读取与本地截止计时分离，持久卷读取阻塞不会阻挡截止执行。恢复时先确认该监督器仍存活。

独立 CPU 正式数据准备可与 pilot GPU 阶段并行，仍受相同截止及专用准备锁约束：

```bash
"${task_campaign[@]}" --prepare-formal-only
```

`launch-campaign-v7.sh` 不转发额外参数，不能通过给它追加 `--prepare-formal-only` 切换模式。

## 2. 单阶段入口与门禁

通常让 campaign 编排并校验、跳过已完成阶段；以下函数仅供尚未完成阶段的同参数诊断或恢复，不能与正在运行的 GPU 阶段并发执行。需要重新预检时使用独立诊断目录，避免覆盖已被 receipt、继承清单或冻结记录绑定的产物。

```bash
task_stage() {
  local task_phase=$1 task_action=$2
  shift 2
  local task_config="$task_code/synthetic-agent-120b.json"
  if [ "$task_phase" = formal ]; then task_config="$task_code/synthetic-agent-120b-formal.json"; fi
  local task_seconds_left
  task_seconds_left=$(( $(date -u -d '2026-10-27 13:19:17 UTC' +%s) - $(date -u +%s) ))
  if [ "$task_seconds_left" -le 0 ]; then printf '%s\n' '本轮授权窗口已截止。' >&2; return 1; fi
  timeout --signal=KILL "${task_seconds_left}s" "$task_py" -u "$task_code/run_synthetic_agent_roles.py" "$task_action" \
    --config "$task_config" --phase "$task_phase" --model-dir "$task_model" \
    --data-dir "$task_data" --run-dir "$task_runs/$task_phase" \
    --cache-dir "$task_cache" --devices 0,1,2,3 --resume "$@"
}
```

| 顺序 | 入口 | 约束与产物 |
|---|---|---|
| 数据准备 | `task_stage formal prepare` | 固定 4096 FIT / 512 DEV / 1024 TEST 源文章；小试为 512/128，正式包含相同 pilot 子集 |
| 预检、选头 | `task_stage pilot preflight`；`task_stage pilot scan` | 4096-token、全头统计 workspace、hook 无扰动和原生 MXFP4；16 头仅由 pilot FIT 内部划分选择 |
| FIT/DEV 提取、训练 | `task_stage formal extract`；`task_stage formal train` | 默认提取 FIT+DEV；正式继承 pilot 的确切选头/preflight，campaign 写入 `pilot-inheritance.json` |
| 外部协议、冻结 | `task_stage formal external-protocol`；`task_stage formal freeze` | 先放置本地预先声明的纯 ID/hash 协议；冻结绑定其 SHA，不能事后补协议追认 |
| 已冻结评估 | `task_stage formal evaluate --split dev`；`extract --split test` 后 `evaluate --split test` | TEST 只做最终评分，不用于调参、选头、标准化、早停或挑模型 |
| 真实外部测试 | `external-prepare`；`extract --split external`；`evaluate --split external` | 必须先完成第 4 节本地哈希导出；对应输入位于 `data-v7/external/inputs-test.jsonl` |
| 图表、报告 | `task_stage formal plot --split test`；`task_stage formal report` | campaign 另生成五种模板×三种格式的 dense 诊断；报告引用原始指标与正文位置诊断 |

表中省略的 action 也通过同一 `task_stage formal ...` 调用。单阶段 CLI 本身只检查入口时间，因此手动示例另外按剩余窗口加 KILL timeout；正常整轮继续使用 campaign 的绝对截止监管。数据准备生成正式 TEST 不等于允许查看 TEST 预测；直到模型和全部选择参数冻结才执行测试。

门禁状态应按其原因处理：

- `waiting_article_audit`：`data-v7/{pilot,formal}/article-audit.json` 必须由真实审阅形成 `passed`，并绑定当前数据清单、源文件及 builder。通过的是所记录的固定分层样本，不保证全部网页无对话；不能仅修改状态字段。
- `waiting_external_protocol`：缺冻结前的 `external/protocol.json`、`history-index.json`，从本地复制纯哈希/公开 ID 文件，历史 CSV 留在本地。
- `waiting_legacy_hashes`：正式模型已冻结后，按第 4 节产生并复制 `legacy-hashes.json`，再以相同参数恢复 campaign。
- `budget_insufficient` / `deadline_exceeded`：记录实际完成范围；不缩减正式文章配额、修改 TEST 或延长当前 deadline 绕过门禁。

## 3. 采样、配置与缓存身份

正文缓存目标份额为单消息/单轮/多轮 20%/30%/50%。实际梯度采样也先按这三个 profile 的固定概率分层，再在层内均匀选择完整结构、存在的角色，最后按事件正文 token 的 family 归一权重选行；不叠加类别损失权重。`training.json` 记录有效 options 和协议，candidate 的 `history.json`、`complete.json` 记录实际抽样计数。累计计数涵盖所有已运行 epochs，导出参数取该候选的最佳 epoch。

七个固定候选为线性 seed17，以及 MLP128/MLP256 各 seed17/29/43；正式最多 45 epochs、至少 10、patience 8，batch 4096、AdamW lr/wd 均为 `1e-3`、dropout `0.1`、clip 5。候选仅按 DEV 的“六类 P/R≥90% 合格优先→结构等权 macro-F1→最弱 P/R→NLL→固定 ID”选择。Pilot 最多 15 epochs；其 DEV 成绩不作为正式 TEST 成绩。

配置、模型/tokenizer、GPU/classifier 实现、选头顺序、输入或已提交特征分片不同均不能混用。分片为 `features/<phase>/<split>-*/{x,y,structure,source,family,weight,groups}.npy` 与 `complete.json`；只有已校验提交才可跳过。训练 `last.pt` 保存模型、optimizer、RNG、sampler 前后状态、batch cursor、早停与抽样计数。计时变化和追加 TEST 不改变原 FIT/DEV 的稳定 split identity。不能手改 SHA 让不匹配的缓存通过。

## 4. 正式冻结后的本地去重哈希导出

本节在本机执行，历史原文始终留在本机。只需要从远端取正式 `freeze.json / best.npz / selection.json` 和冻结前已声明的两份协议文件；保持原字节，不改 JSON 路径或格式。该步骤不需要训练 checkpoint、特征库或 120B 权重。

```bash
task_bundle=/tmp/role-retraining-final-external
mkdir -p "$task_bundle/formal" "$task_bundle/data/external"
rsync -az --include='freeze.json' --include='best.npz' --include='selection.json' --exclude='*' \
  runpod:/workspace/role-retraining-20261026/runs-v7/formal/ "$task_bundle/formal/"
rsync -az --include='protocol.json' --include='history-index.json' --exclude='*' \
  runpod:/workspace/role-retraining-20261026/data-v7/external/ "$task_bundle/data/external/"

PYTHONPATH=/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis python - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer
from synthetic_agent_external import export_legacy_hashes

bundle = Path('/tmp/role-retraining-final-external')
tokenizer = AutoTokenizer.from_pretrained('/tmp/role-retraining-tokenizer', local_files_only=True)
proof = export_legacy_hashes({}, bundle/'data', bundle/'formal',
    history_root_override='/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis',
    tokenizer=tokenizer)
print(proof['sha256'])
PY
rsync -az "$task_bundle/data/external/legacy-hashes.json" \
  runpod:/workspace/role-retraining-20261026/data-v7/external/
```

本机 `/tmp/role-retraining-tokenizer` 使用与服务器一致的官方 tokenizer 小文件（`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`config.json`）。本地导出强校验三份 CSV 的原字节 SHA 与协议一致：`data/conversations/gptoss-20b.csv`、`data/conversations/gptoss-20b-raw.csv`、`outputs/local-gptoss20b/user_queries.csv`，均相对 `history_root_override`。它们不向服务器上传。

导出还校验 protocol/index、冻结 probe/selection、非 engineering 模型、声明早于冻结及 exporter 实现 SHA；局部路径 pointer 无需相同。`legacy-hashes.json` 只含不可逆摘要及计数/布尔审计，绑定完整 `freeze.json` SHA。服务器严格验证该产物后才下载新的 OASST/ToxicChat 各 128 条，不能回退读取历史 CSV。外部正式指标仅评分原始 U/A，后加 S/D 事实段落仅作上下文；缺 C/T 的召回为 N/A，误报仍报告。

## 5. 六类消费端与回传

在远端已配置环境中可直接消费已有 FP16 特征，无需加载 120B：

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/role-retraining-20261026/code-v7')
from synthetic_agent_classifier import FeatureBank, ROLES, load_probe

run = Path('/workspace/role-retraining-20261026/runs-v7/formal')
probe = load_probe(run/'best.npz')
bank = FeatureBank(run/'features/formal', 'dev', expected_binding=probe.metadata)
x = bank.read(np.arange(min(8, len(bank))))  # 原始 N×1024
p = probe.predict_proba(x)                   # 自动应用 FIT 标准化，输出 N×6
assert tuple(probe.metadata['roles']) == ROLES
assert p.shape == (len(x), 6)
print(ROLES, probe.metadata['heads'], p)
```

六类顺序为 `system, developer, user, cot, assistant, tool`；特征必须按导出的 16 头顺序拼接 `pre-o_proj` 的 16×64 维，并匹配 `model_binding`。形状正确不足以证明任意向量匹配模型；不要重排头或二次标准化。旧五类文件由该消费端拒绝。

持久数据、特征、`last.pt` 和日志保留在远端。阶段日志见 `runs-v7/campaign/logs/`，状态和 receipt 见 `campaign/status.json`、`campaign/receipts/`；大型 head-scan、模型及 source-cache 不纳入 Git。小型结果同步至仓库 `experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/`，例如在本机执行：

```bash
task_out=/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7
mkdir -p "$task_out/formal/evaluations" "$task_out/formal/plots"
rsync -az --include='best.npz' --include='freeze.json' --include='selection.json' --include='training.json' \
  --include='REPORT.zh-CN.md' --include='training-curves.png' --include='training-curves.pdf' --exclude='*' \
  runpod:/workspace/role-retraining-20261026/runs-v7/formal/ "$task_out/formal/"
rsync -az --include='dev.json' --include='test.json' --include='external.json' \
  --include='dev-positions.json' --include='test-positions.json' --include='external-positions.json' --exclude='*' \
  runpod:/workspace/role-retraining-20261026/runs-v7/formal/evaluations/ "$task_out/formal/evaluations/"
rsync -az --include='*/' --include='*.png' --include='*.pdf' --include='index.json' --exclude='*' \
  runpod:/workspace/role-retraining-20261026/runs-v7/formal/plots/ "$task_out/formal/plots/"
```

上述白名单不复制正文输入 JSONL、特征/概率大数组、调试正文 sidecar 或 checkpoint。再按小型清单取配置、预检、环境、迁移和哈希证明，逐文件 SHA 核对后提交；不将尚缺的测试或图表写成已完成。

整轮完成后，推荐使用已验证的完整小包导出器，代替逐项手动取文件。它检查真实 campaign 完成、冻结字节、7 候选、TEST/外部 2000 次 source-cluster CI、20 项原生图和 15 项 dense 图、位置诊断及报告；未完成时拒绝导出。以下在本机执行：

```bash
ssh runpod '/root/role-retraining-venv/bin/python /root/role-retraining-20261026/code-v7/export_synthetic_artifacts.py --run-dir /workspace/role-retraining-20261026/runs-v7/formal --data-dir /workspace/role-retraining-20261026/data-v7 --code-dir /root/role-retraining-20261026/code-v7 --output-dir /root/role-retraining-20261026/delivery-v7 --requirements-file /root/role-retraining-20261026/code-v7/requirements-synthetic-agent-4xa40.txt --environment-file /workspace/role-retraining-20261026/environment-freeze.txt'
rsync -a runpod:/root/role-retraining-20261026/delivery-v7/ "$task_out/"
cd "$task_out"
sha256sum -c SHA256SUMS
```

导出器不修改任何训练源产物。体积较大的逐 epoch `history.json` 原字节留在远端；包内 `curve-scalars.json` 明确标为派生视图，保留每个 epoch 的 loss、DEV 排序量、六类 P/R/F1、实际采样计数和原 history SHA。包内另含 7 候选 NPZ、完整指标、图表及数值 sidecar、源码和两份配置。原始文章、对话、输入 JSONL、特征数组、完整逐 token 概率与恢复 `last.pt` 不复制。GPU 全阶段 CSV/summary 另行回传，因其最初位于旧运行根目录。

V5 pilot→V7 的复用不是重训：640 个源的 raw/normalized 正文 hash、cluster/split/group/ordinal 及 FIT/DEV 输入字节完全相同，GPU/classifier/model/tokenizer 身份、七候选与保存 DEV 概率均验证；290 个必要产物保持原字节和真实计时。URL 补齐的覆盖率、新旧集合 hash 单列，来源文件 SHA 可以不同，复用依赖上述正文/源元组和实际输入的等同性；原 49 篇样本审阅诚实标为沿用。详见 V7 pilot 的 `migration.json` 与仓库[迁移证明](outputs/synthetic-agent-120b-20261026/pilot-migration.json)。正式 V7 仍使用新 4096/512/1024 配额及其审计，不能把小试复用当作正式实验完成。

## 6. 首正式 checkpoint 的 gardening 图片

用户补充要求的[诊断协议](outputs/synthetic-agent-120b-20261026/gardening-diagnostic-protocol.json)只授权这一篇作者公开 gardening 对话。三种原文格式、YAML 与原生模板已逐字节固定并与作者公开提交核对；没有转移其他历史 CSV 或查询原文。输入位于远端 `/root/role-retraining-20261026/public-gardening-v7/`，CPU 准备已经完成，953/921/917 个输入 token 各包含完整 916 个正文位置。

执行门禁是 `formal/candidates/linear-seed17/complete.json` 与有效 `best.npz`：至少完成 10 epochs，匹配正式 FIT/DEV 提交、标准化器、GPU/分类器源码和训练哈希。恢复用 `last.pt` 不能替代该 NPZ。小试使用已经冻结的 `pilot/best.npz`；第一正式候选不等于七候选最终选定模型。

门禁通过后，在独立 tmux 中执行已准备的启动脚本：

```bash
ssh runpod 'tmux new-session -d -s role-gardening-first-20261026 bash /root/role-retraining-20261026/launch-gardening-first.sh'
```

脚本按 check→infer→report 顺序运行，整个进程组使用 `min(900 秒, 本轮授权剩余秒数)` 的 KILL timeout；推理进程另有绝对截止自保护。每卡至少 28 GiB 可用才加载 120B；不停止或暂停主训练。三种输入各 forward 一次，两个 checkpoint 共享其 16 头特征，在 CPU 各自应用自己的 FIT 标准化器。正式 TEST、主候选排序及任何参数均不因该诊断改变。

产物位于 `/workspace/role-retraining-20261026/diagnostics/gardening-pilot-vs-first-formal/`：一组原生双模型对比，加每模型三种格式的单列图，共 7 对 PNG/PDF。`report-complete.json` 只有在 14 个文件、完整正文、六行概率和真实 SHA 均核对后写入。回传至仓库对应 `outputs/synthetic-agent-120b-20261026/gardening-pilot-vs-first-formal/` 并在聊天中展示真实图片；工程布局图不作为模型结果交付。

该例的真实标签为 S57/D0/U97/C179/A583/T0，D/T 召回为 N/A、误报仍报告；不沿用旧五类阈值。末尾 emoji 的第 916 个续接 token 109 保留。改写格式各有 6 处句末与换行融合的 BPE，三种格式的 token ID 与坐标不能混同；只有同格式的双模型比较共享完全相同的输入。统一 user/去标签图采用原语义对齐指标，不是形式角色 accuracy。本例属于历史诊断，原完整实验仍继续到最终冻结、TEST、外部测试与报告。
