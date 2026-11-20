# DeepSeek V4 Flash 角色探针

状态：**运行中，最终结果尚未完成**。

目标配置（尚未通过最终验证）：`deepseek-ai/DeepSeek-V4-Flash-0731`；冻结基座，原生 packed FP4 专家与 FP8 dense 权重置于 4 张 H100。

探针读取逆 RoPE 后、`o_a_proj` 前的 16 个有序注意力头，每头 512 维，合计 8192 维。

最终审计尚未执行；目前没有可发布的最终 TEST 结论。

待完成产物：

- `performance-cache-gpu-validation.json`

加载完整性：`load-integrity-audit.json` 状态为 `passed`。
官方逐专家权重到加载目标张量的数量关系：`67612 - 66048 + 172 = 1736`。
该审计核验完整加载与 GPU preflight，日志哈希覆盖审计时的已有前缀。


原生 kernel 性能缓存状态：`applied`。

- `down`：复用 8 token 参考输入的已测原生配置；参考搜索记录 62 个候选，原 winner 时间为 0.16140800714492798 ms。
- `gate_up`：复用 8 token 参考输入的已测原生配置；参考搜索记录 62 个候选，原 winner 时间为 0.3145599961280823 ms。

其他真实长度复用参考配置，保留真实 `S=6×input_tokens` 及原 kernel 的网格、路由和数值格式。这不表示每个目标长度都重新做过 autotune，也不为复用条目声称新的性能测量时间。
保护的原始输入长度：`[]`；应用计数：`{'skipped_existing': 1966}`。


适用范围与限制：

- 五分类标签为 system/user/cot/assistant/tool；GPT-OSS 的 developer 映射为 system（渲染在原生 system 位置），GPT-OSS 的 system 丢弃。
- 与 GPT-OSS 六分类指标比较时，需考虑标签空间和分词差异。
- AgentDojo 使用已有 GPT-OSS 轨迹做跨模型 teacher-forced replay；本结果不衡量新的 DeepSeek agent rollout 或攻击成功率。
- 全 GPU 审计覆盖用于探针的完整 43 层主干；Transformers 不加载检查点中可选的 MTP speculative-decoding 辅助模块。
- GPU/混合精度审计依据已保存的运行时快照；本工具不重新加载或执行基座模型。

完整检查结果见 `independent-validation.json`；原始 TEST 指标及 source-cluster bootstrap 区间见 `evaluations/test.json`。
