# GPU 资源附录

观察时间：2026-10-27T04:28:25.907647Z

峰值均为设备全局离散采样值，包含其他 CUDA contexts；不能当作连续真实峰值或纯模型占用。

CSV：`/workspace/role-retraining-20261026/runs-v5/campaign/gpu-all-stages.csv`；SHA256：`b7ab1d7a617435fc46a86eb6c3dd81cc01c3b681d137ccd01a2e14780342d97d`；字节：1028154；范围：entire stopped-monitor CSV。

监测首末样本：2026-10-26T14:41:27.522992Z → 2026-10-27T04:27:40.908077Z。

Gardening 可能并发：2026-10-27T00:50:08.452816Z → 2026-10-27T00:54:42.268333Z；complete_conservative_bound（保守范围，含 CPU check/report）。

| 阶段 | 原版本 | UTC 起止 | 秒 | GPU0/1/2/3 采样峰值 MiB | 覆盖/并发说明 |
|---|---|---|---:|---|---|
| pilot.scan | runs-v5 | 2026-10-26T14:37:14.247951Z → 2026-10-26T14:55:15.385119Z | 1081.137 | 19198.0/18060.0/18060.0/19180.0 | GPU0: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; GPU1: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; GPU2: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; GPU3: n=146, 前缺=253.275s, 后缺=3.957s, partial=True; 复用原执行，未重复计时 |
| pilot.extract | runs-v5 | 2026-10-26T14:55:18.872248Z → 2026-10-26T16:02:03.475191Z | 4004.603 | 19473.0/18349.0/18349.0/19469.0 | GPU0: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; GPU1: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; GPU2: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; GPU3: n=699, 前缺=3.983s, 后缺=2.088s, partial=False; 复用原执行，未重复计时 |
| pilot.train | runs-v5 | 2026-10-26T16:16:49.914779Z → 2026-10-26T16:23:36.729234Z | 406.814 | 509.0/3.0/3.0/3.0 | GPU0: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; GPU1: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; GPU2: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; GPU3: n=69, 前缺=2.108s, 后缺=2.674s, partial=False; 复用原执行，未重复计时 |
| formal.extract | runs-v7 | 2026-10-26T16:42:25.253604Z → 2026-10-27T00:33:51.794032Z | 28286.540 | 20280.0/19138.0/19138.0/20238.0 | GPU0: n=5060, 前缺=3.299s, 后缺=0.358s, partial=False; GPU1: n=5060, 前缺=3.299s, 后缺=0.358s, partial=False; GPU2: n=5060, 前缺=3.299s, 后缺=0.358s, partial=False; GPU3: n=5060, 前缺=3.299s, 后缺=0.358s, partial=False |
| formal.train | runs-v7 | 2026-10-27T00:40:01.497296Z → 2026-10-27T01:20:04.324427Z | 2402.827 | 17360.0/15763.0/15763.0/16863.0 | GPU0: n=430, 前缺=0.958s, 后缺=5.061s, partial=False; GPU1: n=430, 前缺=0.958s, 后缺=5.061s, partial=False; GPU2: n=430, 前缺=0.958s, 后缺=5.061s, partial=False; GPU3: n=430, 前缺=0.958s, 后缺=5.061s, partial=False; 可能叠加 gardening 273.816s（含 CPU check/report） |
| formal.extract.test | runs-v7 | 2026-10-27T01:27:05.344301Z → 2026-10-27T03:59:35.737415Z | 9150.393 | 22940.0/21760.0/21758.0/22611.0 | GPU0: n=1534, 前缺=4.244s, 后缺=2.722s, partial=False; GPU1: n=1534, 前缺=4.244s, 后缺=2.722s, partial=False; GPU2: n=1534, 前缺=4.244s, 后缺=2.722s, partial=False; GPU3: n=1534, 前缺=4.244s, 后缺=2.722s, partial=False |
| formal.extract.external | runs-v7 | 2026-10-27T04:09:22.213276Z → 2026-10-27T04:11:38.727347Z | 136.514 | 19449.0/18327.0/18327.0/19427.0 | GPU0: n=24, 前缺=3.837s, 后缺=5.22s, partial=False; GPU1: n=24, 前缺=3.837s, 后缺=5.22s, partial=False; GPU2: n=24, 前缺=3.837s, 后缺=5.22s, partial=False; GPU3: n=24, 前缺=3.837s, 后缺=5.22s, partial=False |
| formal.dense | — | complete_timing_unavailable | — | — | Committed receipt lacks a complete absolute stage-marker window; no execution/candidate timing substitution. |

各卡首末样本、内部采样缺口、峰值发生时间及 receipt/migration 原始路径和 SHA 见配套 JSON。

独立 preflight：设备全局峰值 MiB=[23170, 21990, 21990, 23090]；PyTorch allocated/reserved 原始字节见 JSON。计数从权重加载后开始，继承文件不表示重跑。

- Sampled global device peaks are lower bounds on unobserved continuous maxima; include every CUDA context.
- Prefix/tail gaps and gaps over twice the nominal sampling period are explicit; no interpolation or missing-peak imputation.
- FIT+DEV extraction is one combined CLI window; individual candidates have no absolute timing windows and receive no per-candidate peaks.
- Garden overlap is a conservative claim-to-report interval including CPU check/report, never exact inference time or pure-model memory.
- Wall time is stage-marker elapsed time, not GPU kernel time or utilization-weighted GPU hours; unfinished stages remain pending.
