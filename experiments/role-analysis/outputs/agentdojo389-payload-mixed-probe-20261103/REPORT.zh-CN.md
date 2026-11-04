# 最新混合训练 probe：AgentDojo 双 payload 已存特征预测

使用 `mlp256-seed43`，第 12 轮 checkpoint，SHA256 `fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33`。

本次按用户后续要求先完成已有 payload 特征。所有已存首次完整 payload token 都重新预测；没有重新运行 120B 或修改原始轨迹。

| 条件 | 原 episode 数 | 有首次 payload 特征 | 未暴露 payload |
|---|---:|---:|---:|
| plain | 389 | 375 | 14 |
| prerequisite | 389 | 380 | 9 |

主结果为 token 合并平均：各部分六类概率逐 token 求和后除以该部分总 token 数。表内概率采用 0–1 刻度，较小值以科学计数法显示。另在 CSV/JSON 中保留先在 episode 内平均、再对 episode 等权平均的结果。

| 条件 | 部分 | token 数 | system | developer | user | cot | assistant | tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| plain | template | 5,598 | 1.325e-09 | 3.027e-06 | 1.102e-05 | 1.113e-06 | 2.843e-07 | 0.99998456 |
| plain | goal | 13,509 | 1.616e-08 | 4.348e-05 | 3.753e-05 | 1.177e-05 | 7.358e-07 | 0.99990647 |
| prerequisite | template | 6,458 | 3.338e-10 | 7.273e-07 | 1.778e-05 | 4.594e-06 | 1.240e-07 | 0.99997677 |
| prerequisite | goal | 13,611 | 8.099e-08 | 0.00077819 | 0.00028282 | 0.00080276 | 1.216e-06 | 0.99813494 |
| combined | template | 12,056 | 7.939e-10 | 1.795e-06 | 1.465e-05 | 2.977e-06 | 1.984e-07 | 0.99998039 |
| combined | goal | 27,120 | 4.870e-08 | 0.00041222 | 0.00016064 | 0.00040875 | 9.767e-07 | 0.99901737 |

template 指 payload 中除 goal 外的全部 token；与边界相交的 token 归入 goal。没有排除词或重新对片段分词。未暴露 payload 的 23 个 episode 不填零，原生异常但特征有效的 episode 保留。

payload 所在 tool response 的其他 token 没有保存 hidden states，当前没有最新 probe 的对应概率。本报告不混入旧 probe 的历史概率。

独立验证通过：全部 755 个 episode 的分段和均值逐项复算；另在本地 2080 Ti 上从 BF16 位模式独立重建 18 个分层样本的特征并用 PyTorch 前向，最大概率差 8.34e-07，argmax 差异为 0。

- `summary.json` / `mean_probabilities.csv`：汇总及两种平均口径。
- `token_probabilities.csv.gz`：39,176 个 payload token 的六类概率、原始位置、文本及所属部分。
- `probabilities/`：逐 episode 的全部 payload token 预测 NPZ 和来源元数据。
- `per_episode_segment_probabilities.csv`：每个 episode 的分段平均。
- `validation.json`：封存来源核验、分段和独立 GPU 验证。

重评分源码的 7 项单元测试全部通过，见 `rescore-tests.log`。
