# Gardening：旧最终模型与本轮研究候选

本轮新候选为scratch，第11轮，**DEV门槛未通过**；本页不替换旧默认模型。

旧最终模型为MLP128 parent22cb…，新候选为45c…；它们不是此前的pilot或首正式linear-seed17，也不是更早的五类十头probe。

三种原文格式各执行一次冻结120B前向，两分类器共享同一格式的原始16头特征，再由CPU各自读出。没有重新生成对话、训练或重新选择候选。

| 格式／指标 | 旧最终parent | 新scratch候选 |
|---|---:|---:|
| Native Harmony: formal body-role accuracy | 95.4148% | 93.9956% |
| All in user: original-semantic alignment only | 92.3581% | 92.2489% |
| No tags: original-semantic alignment only | 97.0524% | 96.3974% |

每种格式支持为S57/D0/U97/C179/A583/T0，共916个正文token，含末尾emoji续接109。D/T召回与F1无定义，误报仍计入。改写格式只统计原语义对齐，不能称为原生角色准确率；all-user的实际user角色准确率另列metrics.json。

图中灰色为未评分结构，颜色表示原消息语义；各格式坐标独立、存在六处换行BPE融合。同一格式的两个模型才共享完全相同的input IDs。

这是已使用公开例子的冻结后诊断，不是新的独立盲测，也不代表指令遵循、任务成功率或安全性提升。

- [gardening-native-parent-vs-new.png](gardening-native-parent-vs-new.png)
- [gardening-everything_in_user_tags-parent-vs-new.png](gardening-everything_in_user_tags-parent-vs-new.png)
- [gardening-basic_no_format-parent-vs-new.png](gardening-basic_no_format-parent-vs-new.png)
- [gardening-three-format-summary.png](gardening-three-format-summary.png)
- [六类指标与三格式分表](metrics.json)
- [逐角色P/R/F1](per-role-metrics.csv)
- [缓存、坐标及图SHA校验](cache-readout-audit.json)
