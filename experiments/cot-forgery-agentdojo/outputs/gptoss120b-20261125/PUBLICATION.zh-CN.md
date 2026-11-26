# 仓库发布范围

此目录发布 2026-11-25 CoT Forgery 的 AgentDojo 适配复现结果：684 个主实验固定输入、27 个无害续写输入与生成记录、12 组 PNG/PDF/SVG 图、15 份 CSV、协议、运行清单及独立审计。实现和完整复算命令位于 [实验 README](../../README.zh-CN.md)。

`readouts/`（711 份原始 attention-head 特征和 probe 读数）、`records/`（逐输入提交记录）、`reproduction/`（重复代码、模型/tokenizer 资产和原生调用来源快照）及运行日志保留在本地，不纳入 Git。重新执行模型评分需要 README 指定的 checkpoint、模型、来源归档和冻结依赖；全部输入 token 已保存在提交的 JSONL 中。模型权重仍为外部依赖。

`delivery-manifest.json` 与 `SHA256SUMS` 是完整本地实验包的既有封存记录，保留其原始内容和完整文件哈希。它们包含以上本地文件，不能用来宣称 Git 检出包含完整的原始激活包。`independent-validation.json` 和 `independent-statistics-validation.json` 记录的是封存前对完整本地数据完成的审计；其中九份核心 CSV 独立重算，六份补充表仅作哈希绑定。

`repository-publication-manifest.json` 单独列出本目录发布到 Git 的文件和哈希；`REPOSITORY_SHA256SUMS` 只覆盖这些已发布结果（不含该清单本身和自身）。在本目录核验已发布内容：

```bash
sha256sum --quiet -c REPOSITORY_SHA256SUMS
```

报告中的 `readouts/`、`records/`、`reproduction/` 说明均指完整本地运行目录。已发布的图表和表格可直接读取；重新从特征生成全部图表与执行完整审计需先按 README 恢复或重新计算本地文件。
