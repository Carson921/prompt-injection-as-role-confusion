# 正式图表入口

全部图来自固定DEV选择的 `mlp128-seed17`。正常格式只在保存的正文mask位置显示正式概率；灰色为结构，浅黄为未采样正文。每张图均保留完整实际输入。

## 20类原生图

| 类别 | 输入token | 评分位置 | PNG | PDF |
|---|---:|---:|---|---|
| `single_S` | 545 | 97 | [PNG](final-v7/formal/plots/test/single_S-34b1e7958f.png) | [PDF](final-v7/formal/plots/test/single_S-34b1e7958f.pdf) |
| `single_D` | 445 | 97 | [PNG](final-v7/formal/plots/test/single_D-14e9c035e2.png) | [PDF](final-v7/formal/plots/test/single_D-14e9c035e2.pdf) |
| `single_U` | 191 | 97 | [PNG](final-v7/formal/plots/test/single_U-05e0f08e71.png) | [PDF](final-v7/formal/plots/test/single_U-05e0f08e71.pdf) |
| `single_C` | 336 | 97 | [PNG](final-v7/formal/plots/test/single_C-1167fd6508.png) | [PDF](final-v7/formal/plots/test/single_C-1167fd6508.pdf) |
| `single_A` | 284 | 97 | [PNG](final-v7/formal/plots/test/single_A-5e69593c55.png) | [PDF](final-v7/formal/plots/test/single_A-5e69593c55.pdf) |
| `single_T` | 302 | 97 | [PNG](final-v7/formal/plots/test/single_T-3a5fbf45fe.png) | [PDF](final-v7/formal/plots/test/single_T-3a5fbf45fe.pdf) |
| `one_round_SDUA` | 756 | 146 | [PNG](final-v7/formal/plots/test/one_round_SDUA-ca3d3cb5aa.png) | [PDF](final-v7/formal/plots/test/one_round_SDUA-ca3d3cb5aa.pdf) |
| `one_round_SDUCA` | 1072 | 146 | [PNG](final-v7/formal/plots/test/one_round_SDUCA-2092a41ee0.png) | [PDF](final-v7/formal/plots/test/one_round_SDUCA-2092a41ee0.pdf) |
| `one_round_SDUCQ` | 1097 | 146 | [PNG](final-v7/formal/plots/test/one_round_SDUCQ-eeba0ab13c.png) | [PDF](final-v7/formal/plots/test/one_round_SDUCQ-eeba0ab13c.pdf) |
| `one_round_SDUQTCA` | 1510 | 120 | [PNG](final-v7/formal/plots/test/one_round_SDUQTCA-c9352e0b03.png) | [PDF](final-v7/formal/plots/test/one_round_SDUQTCA-c9352e0b03.pdf) |
| `one_round_SDUQTA` | 1459 | 146 | [PNG](final-v7/formal/plots/test/one_round_SDUQTA-c515cf8e7d.png) | [PDF](final-v7/formal/plots/test/one_round_SDUQTA-c515cf8e7d.pdf) |
| `multi_repeated_segments` | 1361 | 109 | [PNG](final-v7/formal/plots/test/multi_repeated_segments-ce47c1f62a.png) | [PDF](final-v7/formal/plots/test/multi_repeated_segments-ce47c1f62a.pdf) |
| `multi_terminal_UCQ` | 2362 | 118 | [PNG](final-v7/formal/plots/test/multi_terminal_UCQ-811256bd16.png) | [PDF](final-v7/formal/plots/test/multi_terminal_UCQ-811256bd16.pdf) |
| `history_cot_retain` | 3313 | 129 | [PNG](final-v7/formal/plots/test/history_cot_retain-fc6cf4b6fb.png) | [PDF](final-v7/formal/plots/test/history_cot_retain-fc6cf4b6fb.pdf) |
| `history_cot_drop_completed_round_cot` | 2798 | 115 | [PNG](final-v7/formal/plots/test/history_cot_drop_completed_round_cot-2d0197d25f.png) | [PDF](final-v7/formal/plots/test/history_cot_drop_completed_round_cot-2d0197d25f.pdf) |
| `call_chain_pre_tool_call` | 1919 | 10 | [PNG](final-v7/formal/plots/test/call_chain_pre_tool_call-9659985802.png) | [PDF](final-v7/formal/plots/test/call_chain_pre_tool_call-9659985802.pdf) |
| `call_chain_tool_reply` | 1847 | 20 | [PNG](final-v7/formal/plots/test/call_chain_tool_reply-444e8f6ef7.png) | [PDF](final-v7/formal/plots/test/call_chain_tool_reply-444e8f6ef7.pdf) |
| `call_chain_output` | 2066 | 112 | [PNG](final-v7/formal/plots/test/call_chain_output-3cb473f344.png) | [PDF](final-v7/formal/plots/test/call_chain_output-3cb473f344.pdf) |
| `multi_K5` | 2017 | 117 | [PNG](final-v7/formal/plots/test/multi_K5-1ce3baff89.png) | [PDF](final-v7/formal/plots/test/multi_K5-1ce3baff89.pdf) |
| `multi_K6` | 2362 | 118 | [PNG](final-v7/formal/plots/test/multi_K6-811256bd16.png) | [PDF](final-v7/formal/plots/test/multi_K6-811256bd16.pdf) |

## 五模板×三格式的15个dense诊断

统一user和去标签版本的分数表示原语义对齐，不是改写后的形式角色accuracy。选择按固定ID完成，不按模型分数挑图。

| 模板 | 格式 | PNG | PDF |
|---|---|---|---|
| `SDUA` | `native` | [PNG](final-v7/formal/plots/dense/native-277789079f.png) | [PDF](final-v7/formal/plots/dense/native-277789079f.pdf) |
| `SDUA` | `all_user` | [PNG](final-v7/formal/plots/dense/all_user-277789079f.png) | [PDF](final-v7/formal/plots/dense/all_user-277789079f.pdf) |
| `SDUA` | `no_tags` | [PNG](final-v7/formal/plots/dense/no_tags-277789079f.png) | [PDF](final-v7/formal/plots/dense/no_tags-277789079f.pdf) |
| `SDUCA` | `native` | [PNG](final-v7/formal/plots/dense/native-349af14743.png) | [PDF](final-v7/formal/plots/dense/native-349af14743.pdf) |
| `SDUCA` | `all_user` | [PNG](final-v7/formal/plots/dense/all_user-349af14743.png) | [PDF](final-v7/formal/plots/dense/all_user-349af14743.pdf) |
| `SDUCA` | `no_tags` | [PNG](final-v7/formal/plots/dense/no_tags-349af14743.png) | [PDF](final-v7/formal/plots/dense/no_tags-349af14743.pdf) |
| `SDUCQ` | `native` | [PNG](final-v7/formal/plots/dense/native-d442423d73.png) | [PDF](final-v7/formal/plots/dense/native-d442423d73.pdf) |
| `SDUCQ` | `all_user` | [PNG](final-v7/formal/plots/dense/all_user-d442423d73.png) | [PDF](final-v7/formal/plots/dense/all_user-d442423d73.pdf) |
| `SDUCQ` | `no_tags` | [PNG](final-v7/formal/plots/dense/no_tags-d442423d73.png) | [PDF](final-v7/formal/plots/dense/no_tags-d442423d73.pdf) |
| `SDUQTCA` | `native` | [PNG](final-v7/formal/plots/dense/native-cf386705a3.png) | [PDF](final-v7/formal/plots/dense/native-cf386705a3.pdf) |
| `SDUQTCA` | `all_user` | [PNG](final-v7/formal/plots/dense/all_user-cf386705a3.png) | [PDF](final-v7/formal/plots/dense/all_user-cf386705a3.pdf) |
| `SDUQTCA` | `no_tags` | [PNG](final-v7/formal/plots/dense/no_tags-cf386705a3.png) | [PDF](final-v7/formal/plots/dense/no_tags-cf386705a3.pdf) |
| `SDUQTA` | `native` | [PNG](final-v7/formal/plots/dense/native-93dbe778ce.png) | [PDF](final-v7/formal/plots/dense/native-93dbe778ce.pdf) |
| `SDUQTA` | `all_user` | [PNG](final-v7/formal/plots/dense/all_user-93dbe778ce.png) | [PDF](final-v7/formal/plots/dense/all_user-93dbe778ce.pdf) |
| `SDUQTA` | `no_tags` | [PNG](final-v7/formal/plots/dense/no_tags-93dbe778ce.png) | [PDF](final-v7/formal/plots/dense/no_tags-93dbe778ce.pdf) |

## 训练和整体混淆矩阵

- [训练loss与放大DEV曲线 PNG](training-diagnostics-v7/training-loss-and-dev-zoom.png) · [PDF](training-diagnostics-v7/training-loss-and-dev-zoom.pdf) · [来源与轴缩放说明](training-diagnostics-v7/README.zh-CN.md)。
- 原始全纵轴曲线：[PNG](final-v7/formal/training-curves.png) · [PDF](final-v7/formal/training-curves.pdf)。
- dev混淆矩阵：[PNG](final-v7/formal/dev-confusion.png) · [PDF](final-v7/formal/dev-confusion.pdf)。
- test混淆矩阵：[PNG](final-v7/formal/test-confusion.png) · [PDF](final-v7/formal/test-confusion.pdf)。
- external混淆矩阵：[PNG](final-v7/formal/external-confusion.png) · [PDF](final-v7/formal/external-confusion.pdf)。

## 用户要求的gardening对比

[小试与首正式linear checkpoint的7对图](gardening-pilot-vs-first-formal/README.zh-CN.md)单独归档，包括原生双模型对比与每模型三种格式。首正式linear样例的结果不代表最终MLP，历史诊断不替代正式TEST。

[返回结果与范围说明](README.zh-CN.md)。
