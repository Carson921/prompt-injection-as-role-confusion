# 正式图表审阅范围

[原始审阅记录](final-review.json)与[逐文件元数据／SHA](metadata-review.json)保留审阅时原字节；[笔记](review-notes.md)记录视觉范围。20 类原生图与 15 个 dense 组合全部核对实际 PNG/PDF、提交哈希和数值 sidecar，共 35 对。

代理实际查看四张 PNG：独立 D、删除已完成轮次 CoT、多轮 K6，以及 SDUQTCA 的统一 user 版本。该 K6 图为 2362 个输入 token，并非图集最长。根线程另外实际查看最长的 `history_cot_retain` 图（3313 token）、训练曲线和 TEST 混淆矩阵；完整开头／结尾、六行概率、灰色结构与浅黄色未采样正文可见，未发现明显剪裁。

原始 JSON 中 native 条目的 `semantic_caption_correct=false` 表示**不适用**：该布尔字段当时直接由 `section == "dense"` 填入，原生图使用正确的形式角色评分说明，不使用改写格式的语义说明。该字段不是 caption 错误判定；代理已逐图确认。dense 统一 user 图的原语义／非形式 accuracy 说明实际可见。

哈希和位置检查覆盖全部文件；视觉检查只覆盖上述样本，不声称逐张完成视觉审阅。完整图表保持[严格导出包](../final-v7/README.md)中的原字节。
