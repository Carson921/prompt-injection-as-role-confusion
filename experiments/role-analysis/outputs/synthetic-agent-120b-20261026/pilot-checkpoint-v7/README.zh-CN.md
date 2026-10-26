# 已冻结小试 checkpoint

这是六角色旁路分类器的小试交付，120B 本体保持冻结。正式全配额训练与最终 TEST 继续独立运行，此文件不是正式最优模型。

候选：`mlp256-seed29`；最佳 epoch：15。FIT/DEV 为 512/128 篇文章，仅按 DEV 选择。DEV macro-F1：97.749048%；六类最低 precision/recall：94.647525%。

| 类别 | DEV precision | DEV recall | DEV F1 |
|---|---:|---:|---:|
| system | 94.8822% | 97.5864% | 96.2153% |
| developer | 97.4626% | 94.6475% | 96.0344% |
| user | 97.4468% | 98.6839% | 98.0615% |
| cot | 99.7098% | 98.8673% | 99.2867% |
| assistant | 99.6546% | 99.9215% | 99.7879% |
| tool | 97.6885% | 96.5352% | 97.1084% |

`best.npz` SHA256：`7ad3846bc5f381db5dae7f9868a71feb3668d1278c8a3482c7ddcc288932651b`。`freeze.json` 对 best/selection/training 的原字节哈希已在本地复核，六类加载器已成功读取。

模型包含 FIT 标准化器与固定的 16 头顺序，输入为相同冻结 120B 的 N×1024 pre-o_proj 特征。六类顺序为 system、developer、user、cot、assistant、tool。按[运行手册](../../../SYNTHETIC-AGENT-RUNBOOK.zh-CN.md)的消费示例加载；小试远端 run 为 `/workspace/role-retraining-20261026/runs-v7/pilot`，特征目录为 `features/pilot`，使用 `dev` split。

V5→V7 的模型输入字节完全相同，完整继承核验见[迁移证明](../pilot-migration.json)。此交付保留原冻结文件，不把小试 DEV 成绩当作正式 TEST 成绩。
