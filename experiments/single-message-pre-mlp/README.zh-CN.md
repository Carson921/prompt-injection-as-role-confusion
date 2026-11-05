# GPT-OSS-120B 六角色单消息全层 pre-MLP 采集

本次请求使用 **C4 和 Dolma3 各 1000 条**，共 2000 篇正文。每篇分别包装成
`system, developer, user, assistant, tool_response, think` 六种单消息，得到 12000 条输入。
模型仅做前向计算，不生成文本，不训练 probe。

此前采集于 **2026-11-01** 完成，历史核验为 500/500 分片、6010750 个真实 tokens、
36 层 BF16，逻辑形状 `[6010750, 36, 2880]`，约 1.25 TB。
**2026-11-02 恢复状态：原实验目录已删除，代码已从操作记录恢复；原激活文件尚未找回。**
恢复记录位于 `/scratch/wz55191/recovery-single-message-pre-mlp-20261102/recovery.json`。
只有输出目录中新生成的 `complete.json` 才表示重新采集完成。

沿用 `experiments/role-analysis/02-train-role-probes.ipynb` 的 seed=123、streaming shuffle
buffer=50000、正文最多 1024 tokens 和 `utils/role_templates.py` 的 Harmony 单消息模板。
与历史实验相比，本次扩大样本量、加入 developer，并保留用户指定的 label 名称：
`think` 对应原来的 `cot`（assistant analysis），`tool_response` 对应原来的 `tool`。
tool 模板仍为历史的 `functions. to=assistant`、commentary channel，不加入前置工具调用。
所有消息以历史模板的 `<|end|>` 结尾，无额外 system 前缀或 generation prompt。

采集点为每一层 `post_attention_layernorm` 输出，即实际送入 MLP 的归一化 hidden state；
层号为 **0–35**，hidden size 为 **2880**。权重保持原生 MXFP4，attention 使用 eager，
激活计算及保存均为 **BF16**，保留原值。与历史 FP16 文件相比，本次直接保存 BF16。
批次默认右侧 padding 到本次数据的最长消息，避免可变形状反复触发 Triton 编译；padding
不写入激活文件。`--dynamic-padding` 可恢复逐批按最大长度 padding。执行参数记录在 backend
文件及各分片 receipt；batch/形状改变可产生通常的 BF16 数值舍入差异。

准备阶段将数据仓库版本解析为完整 commit，保存原文、抽样参数、URL/ID（来源有提供时）、
每条消息和精确 token IDs。`source_row` 指 shuffle 后在各语料中的抽样序号。
同一正文的六种消息通过 `sample_id` 配对；以后划分训练测试集时应按 `sample_id` 分组。

## 执行与恢复

在仓库根目录使用独立 `.venv`。本地模型默认位于
`/scratch/wz55191/models/gpt-oss-120b`，可以用 `--model` 修改。
安装依赖可使用 `uv pip install --python .venv/bin/python -r experiments/single-message-pre-mlp/requirements.txt`。
完整的两 GPU 准备、采集及核验可直接运行
`bash experiments/single-message-pre-mlp/run_collection.sh`，日志保存在输出目录。
若共享存储的 mmap 读取较慢，可先将同一份权重复制到节点本地磁盘，再设置
`TASK_MODEL=/节点本地模型路径` 运行上述脚本；配置、tokenizer 及权重文件大小必须匹配原 manifest。
可使用 `stage_model.py 原模型目录 本地模型目录` 复制所需文件并核对源文件与副本 SHA-256。

```bash
.venv/bin/python experiments/single-message-pre-mlp/collect.py prepare
```

准备完成后可在两张空闲 GPU 各启动一个独立进程，按固定 shard ID 分工。以下两条命令
应在不同终端执行（每个进程各加载一份模型）：

```bash
.venv/bin/python -u experiments/single-message-pre-mlp/collect.py extract --device 0 --workers 2 --worker-index 0
.venv/bin/python -u experiments/single-message-pre-mlp/collect.py extract --device 1 --workers 2 --worker-index 1
```

也支持单 GPU，使用 `extract --device 0`。现有完整分片会检查签名和形状并跳过，未完成的
临时分片会重算；不会将其他数据配置的缓存混入当前结果。不要同时运行重叠的 worker 分工。
`--batch-size` 可调整显存占用，默认 4，不影响分片或消息顺序。
完整脚本通过 `TASK_BATCH_SIZE=12` 等环境变量设置 batch；具体大小须与 GPU 显存匹配。
无需加载模型即可查看进度：`python experiments/single-message-pre-mlp/progress.py`。

```bash
.venv/bin/python experiments/single-message-pre-mlp/collect.py verify
.venv/bin/python -m pytest -q experiments/single-message-pre-mlp/test_collect.py
```

## 保存格式

默认输出：`experiments/single-message-pre-mlp/outputs/gpt-oss-120b-c4-1000-dolma3-1000/`。

- `manifest.json`：冻结的配置、数据/模型身份、角色映射、全局 token 数及分片计划。
- `raw_samples.json`、`sources.json`：原文及数据来源。
- `messages.parquet`：12000 条单消息的索引、prompt、input_ids、正文区间、全局激活区间。
- `shards/shard-NNNNN.safetensors`：每片 24 条消息，末片可更少；含每层
  `pre_mlp.layer_00` 至 `pre_mlp.layer_35`，各自形状为 `[该片真实 token 数, 2880]`。
  同文件保存 `input_ids`、`label_ids`、`message_ids`、`message_offsets`。
- `receipts/`、`progress-worker-*.json`：已提交分片及进度。
- `validation-worker-*.json`：全部 36 层 hook 与 MLP 输入逐值一致、加 hook 前后 logits 逐值一致。
- `backend-worker-*.json`：实际 GPU、依赖版本、权重与激活精度、kernel commit。
- `verification.json`：落盘后的索引、形状和逐层抽查；仅全部分片存在并通过核验后才生成 `complete.json`。

核验使用 `verify_saved.py` 的有界 `pread` 并行读取，避免共享文件系统随机 mmap 缺页耗时。
所有 token IDs、标签、消息偏移、36 层形状及 BF16 dtype 都会核验；逐层读取前 16 个真实 token
检查有限值。采集阶段则已对全部保存向量检查有限值。进度写入 `verification-progress.json`。

保存所有真实 token，**包括 header 与结束符，不含 padding**。`label_ids` 仅给正文赋六类
标签，header/结束符为 -1。`messages.parquet` 的 `[content_start, content_end)` 为消息内正文区间；
`[activation_start, activation_end)` 为跨分片拼接后该消息的全局 token 区间。
整体逻辑形状为 `[全部真实 token 数, 36, 2880]`，物理上按消息分片、按层保存以便部分读取。

仅读取某条消息的某一层正文，不必加载整个分片：

```bash
.venv/bin/python experiments/single-message-pre-mlp/read_activations.py \
  experiments/single-message-pre-mlp/outputs/gpt-oss-120b-c4-1000-dolma3-1000 0 18
```

Python 中可调用 `read_activations.read_message(output, message_id=0, layer=18)`，
返回消息元信息、token IDs 和 `[正文 token 数, 2880]` BF16 tensor。
