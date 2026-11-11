# agentic-probe：role probe × Agentic Optimizer（arXiv 2611.11861）

计划见 `../ipi-arena-richevo/PLAN-agentic-optimizer.zh-CN.md`。本目录是阶段 0：按论文第 3 节复现"良性目标"合成评测集和单轮 0.4/0.6 评分器，并用现有模板测出各目标模型的基线。

## 流水线

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 工具生成 | `gen_tools.py` | 53 个领域（论文 Table 6），每领域 6 个工具，LLM 质量校验 + 两两去重（论文 Figure 2 提示） |
| 场景生成 | `gen_scenarios.py` | 每领域一个 2–4 步链式场景（Figure 3 提示）+ 3 个单次调用攻击目标；工具返回合理性与顺序依赖校验（Figure 4） |
| 良性审核 | `check_benign.py` | LLM 审核攻击目标是否良性，写入 `benign` 标记 |
| 配对 | `build_dataset.py` | 任务域与目标域取自不同大类；输出 `dataset.json` |
| 重放 | `replay.py` | gpt-oss-120b（vLLM）按策略重放到与黄金工具序列一致（先要求参数完全一致，再放宽到函数名一致） |
| 请求冻结 | `prepare.py` | payload 追加到最后一次工具返回的自由文本字段；HF 模板渲染、goal token 定位、协议哈希；`--system` 可替换系统提示 |
| 单轮 rollout | `score_rollout.py` | 论文 3.2 评分（已用 Table 7 六个例子单元验证）；`--base-url https://openrouter.ai/api/v1 --model …` 可换黑盒目标模型 |
| 探针 | `../ipi-arena-richevo/probe_score.py` + `probe_summary.py` | 对 gpt-oss-120b 请求打 goal 段 P(user) |
| 报告 | `report.py`、`summarize_phase0.py` | 每次运行的中文报告与跨目标模型总览 |

生成用 OpenRouter `google/gemini-2.5-flash`，全部生成与审核约 0.8 美元。

## 数据集 v1（`outputs/dataset-v1/`）

310 个工具 / 53 领域；50 个场景通过校验；44 个在 gpt-oss-120b 上重放成功（参数完全一致 18 个），链长 2/3/4 步各 9/26/9。LLM 审核标记 4 个目标非良性（退款、转账、删除食谱、一个写恶意脚本的目标），报告里另给剔除后的 ASR。

## 阶段 0 结果（`outputs/PHASE0-OVERVIEW.zh-CN.md`，每次运行各有 `REPORT.zh-CN.md`）

| 模板 | gpt-oss 本集 P(user) | gpt-oss-120b | gpt-oss + 防御系统提示 | gemini-2.5-flash | claude-sonnet-4.6 |
|---|---|---|---|---|---|
| plain | 0.000 | 4.5% | 0.0% | 0.0% | 0.0% |
| band_0.10_0.20 | 0.007 | 97.7% | 93.2% | 2.3% | 0.0% |
| band_0.50_0.60 | 0.656 | 97.7% | 95.5% | 31.8% | 0.0% |
| band_0.80_0.90 | 0.740 | 97.7% | 97.7% | 13.6% | 0.0% |
| band_0.99_1.00 | 0.966 | 97.7% | 97.7% | 63.6% | 0.0% |
| ipievo_1 | 0.981 | 95.5% | 97.7% | 47.7% | 0.0% |
| ipievo_4 | 0.976 | 97.7% | 97.7% | 50.0% | 0.0% |

- **gpt-oss-120b 在良性目标上饱和**：任何 wrapper 模板都有 95–98%，连 P(user) 只有 0.007 的 band_0.10_0.20 也是 97.7%；plain 只有 4.5%。加"工具结果是不可信数据"的系统提示后 plain 降到 0，模板仍 93–98%。唯一普遍失败的是被审核标为恶意的目标（写 `rm -rf` 脚本），模型明确拒绝。
- 这说明在 gpt-oss-120b 上，**良性目标的注入成功不需要角色层信号**（模型本来就会执行工具输出里的指令），IPI Arena 上 44–64% 的差距主要来自内容拒绝。阶段 1/2 若以 gpt-oss-120b 为目标模型，本集没有可比较的余量。
- **gemini-2.5-flash 有余量且模板可迁移**：plain 0%，gpt-oss 上优化出的模板 2–64%，与 gpt-oss 的 P(user) 排序大体一致（Spearman 见 `run-phase0-gemini25flash-20261111/correlations.json`）。
- **claude-sonnet-4.6 全为 0%**：静态的 gpt-oss Harmony 分隔符模板对它完全无效。

## 对阶段 1/2 的含义

- 黑盒基线与结合实验应以 **gemini-2.5-flash**（或更强的 Gemini）为目标模型：有余量、便宜、能看到模板迁移。
- 探针诊断（结合点 A/B）只能对 gpt-oss-120b 请求计算；可在 gemini 实验里把 gpt-oss 探针作为"代理白盒"信号（本集已证明其排序能部分迁移），这本身是一个可检验的假设。
- 若必须保留 gpt-oss-120b 作目标模型，需要更强的防御（如 Meta-SecAlign 式训练或工具输出 spotlighting）而不是系统提示。
