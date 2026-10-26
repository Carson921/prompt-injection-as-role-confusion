# GPU 资源临时元数据（不是最终原始 CSV 身份）

观察时间：2026-10-26T22:06:57.892035Z

峰值均为设备全局离散采样值，包含其他 CUDA contexts；不能当作连续真实峰值或纯模型占用。

CSV：`/workspace/role-retraining-20261026/runs-v5/campaign/gpu-all-stages.csv`；SHA256：`d6559a1cecc308265731a7dd2aeb1a7ca7ebc1292daa330d380ad9d0e21ff041`；字节：544428；范围：provisional exact prefix only; NOT final raw CSV identity。

监测首末样本：2026-10-26T14:41:27.522992Z → 2026-10-26T22:06:53.417363Z。

Gardening 可能并发：— → —；not_recorded（保守范围，含 CPU check/report）。

| 阶段 | 原版本 | UTC 起止 | 秒 | GPU0/1/2/3 采样峰值 MiB | 覆盖/并发说明 |
|---|---|---|---:|---|---|
| pilot.scan | runs-v5 | 2026-10-26T14:37:14.247951Z → 2026-10-26T14:55:15.385119Z | 1081.137 | 19198.0/18060.0/18060.0/19180.0 | GPU0: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; GPU1: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; GPU2: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; GPU3: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; 复用原执行，未重复计时 |
| pilot.extract | runs-v5 | 2026-10-26T14:55:18.872248Z → 2026-10-26T16:02:03.475191Z | 4004.603 | 19473.0/18349.0/18349.0/19469.0 | GPU0: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; GPU1: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; GPU2: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; GPU3: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; 复用原执行，未重复计时 |
| pilot.train | runs-v5 | 2026-10-26T16:16:49.914779Z → 2026-10-26T16:23:36.729234Z | 406.814 | 509.0/3.0/3.0/3.0 | GPU0: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; GPU1: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; GPU2: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; GPU3: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; 复用原执行，未重复计时 |
| formal.extract | — | pending | — | — | receipt_missing |
| formal.train | — | pending | — | — | receipt_missing |
| formal.extract.test | — | pending | — | — | receipt_missing |
| formal.extract.external | — | pending | — | — | receipt_missing |
| formal.dense | — | pending | — | — | receipt_missing |

各卡首末样本、内部采样缺口、峰值发生时间及 receipt/migration 原始路径和 SHA 见配套 JSON。

独立 preflight：设备全局峰值 MiB=[23170, 21990, 21990, 23090]；PyTorch allocated/reserved 原始字节见 JSON。计数从权重加载后开始，继承文件不表示重跑。

- Sampled global device peaks are lower bounds on unobserved continuous maxima; include every CUDA context.
- Prefix/tail gaps and gaps over twice the nominal sampling period are explicit; no interpolation or missing-peak imputation.
- FIT+DEV extraction is one combined CLI window; individual candidates have no absolute timing windows and receive no per-candidate peaks.
- Garden overlap is a conservative claim-to-report interval including CPU check/report, never exact inference time or pure-model memory.
- Wall time is stage-marker elapsed time, not GPU kernel time or utilization-weighted GPU hours; unfinished stages remain pending.
