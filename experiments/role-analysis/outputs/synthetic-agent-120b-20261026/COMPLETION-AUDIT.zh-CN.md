# 本轮计划逐项完成审计

对象为[原计划](../../RETRAINING-PLAN-120B.zh-CN.md)的完整 V7 正式实验及用户补充要求。以下证据来自实际运行、提交文件、独立聚合复核和实际图片；单项工程测试不代替全实验。模型效果目标按计划的汇总口径判断，分项弱点另见[最终说明](README.zh-CN.md)。

| 要求 | 实际证据与结论 |
|---|---|
| 使用 `runpod`、同机 4×A40 | [backend](model_backend.json)记录四卡、36层每卡9层和执行副本。SSH 连接均使用指定别名。 |
| 官方固定模型/tokenizer/内核；原生 MXFP4；骨干冻结 | [权重校验](weight-verification.json)覆盖15分片、65,248,893,184 bytes；backend记录官方 revision、tokenizer SHA、`Mxfp4GptOssExperts`。绑定实现要求全参数 CUDA、`dequantize=False`、无骨干梯度。原挂载读取失败，未完成原挂载全量字节对照，已如实记录。 |
| 4096输入、全头 workspace、hook 无扰动、显存 | [预检](preflight.json)：batch1、4096、全部2304头、82,577,088-byte统计 workspace finite；短／4096输入的全头和工程16头 hook bit-exact，logits finite。真实选中16头另有379／302-token检查；不冒称其单独做过4096复检。设备和PyTorch峰值计数范围另见资源附录。 |
| 六类及原生 S/D、C/A channel、Q/T 对应 | [完整核心源码](final-v7/code/synthetic_agent_data.py)、冻结角色顺序和[全输入审计](formal-input-mask-audit.json)共同核验25,415条实际输入；独立S/D及带recipient的Q、工具结果均按各自原生头解析。 |
| 原始五模板、SD一次、多轮有放回、UCQ仅末轮 | [V7正式清单](final-v7/metadata/data/formal/data-manifest.json)保留结构计数；全输入审计核对实际可见符号、S/D次数、轮数、工具阶段和合法单消息／未回复Q例外，违规为零。K5/6只出现在TEST，重复片段有实际覆盖。 |
| CoT 整条清理、重新分词和重新 forward | 全输入审计实际独立重分词全部版本、核对删除事件及可配对兄弟记录；4,119对历史轮次清理、2,198对先前调用清理通过，3条缺少保存兄弟记录明确 unavailable。[实际特征 manifest](final-v7/metadata/features/formal/manifest.json)及提取 receipts 证明各快照完成，绑定 GPU 实现使用 `use_cache=False`；CPU审计本身不单独证明历史forward。 |
| 纯文章正文 mask；排除头、边界、Q和D工具定义 | 实际2,742,784个 targets 全部核验，header/delimiter、Q JSON、D工具定义、混合BPE及角色标签违规均0。正式 FIT/DEV/TEST 排除跨边界token为1898/247/512。原图以灰结构、浅黄未采样正文区分评分分母。 |
| C4+Dolma3、完整配额、源／URL／近重复分区 | V7清单固定4096/512/1024，二者各半；5632个URL全部可用，四项分区验证均true，近重复定义为规范化5词shingle SimHash半径3。保留六主题采样与第三人称过滤偏差说明。 |
| 固定文章样本审阅 | [正式决定](final-v7/metadata/data/formal/article-audit.json)绑定当前清单/source/builder，50篇=49篇相同正文SHA继承+1篇新审。审阅为LLM代理、非人类gold，未用TEST预测挑样本；原失败决定保留。 |
| Pilot 子集与缓存复用真实身份 | [迁移证明](final-v7/metadata/pilot-migration.json)覆盖640个源、FIT/DEV输入、模型/源码/特征及290个原字节产物；实际DEV概率验证通过。正式继承同一选头，不重训小试、不将零秒adopt当作原成本。最终V7 pilot清单已另行交付。 |
| FIT-only 全头扫描与有序16头 | [head-selection](final-v7/formal/head-selection.json)含全部0…2303唯一头的rank；使用pilot FIT内128簇（统计98、内部评估30，交集0），未使用DEV/TEST/旧自然对话probe。formal继承标记`formal_head_rescan=false`。 |
| `pre-o_proj`、16×64、FP16分片、FIT标准化 | 实际分片提交与模型binding固定1024维、同一头顺序和FP16；标准化器由FIT估计。最终 [NPZ](final-v7/formal/best.npz)、[standardizer](final-v7/formal/standardizer.npz)由导出器逐值核对；消费端检查六类、模型及来源绑定。 |
| 20/30/50实际梯度份额、family权重、无额外class weights | [采样证据](formal-sampling-summary.json)分别保留缓存、family基础权重和七候选实际抽样。胜者截至epoch30抽样59,842,560次，20.0022/29.9941/50.0037%；末8轮单列。family总和规则来自固定prepare检查，不将新mask审计夸为重新全量family求和。 |
| 七候选GPU训练、超参数、早停和恢复 | [training](final-v7/formal/training.json)、七候选complete/curve-scalars及[最终分层工程验证](classifier-hierarchical-engineering-validation.json)：min10/max45、patience8、固定LR/WD/dropout/batch/clip；实际轮数合格。独立GPU dropout/AdamW恢复权重、loss、采样计数一致，NPZ导出误差在记录容差内。最佳epoch6/8与实际至少运行10轮不矛盾。 |
| 固定DEV规则与测试前冻结 | [selection](final-v7/formal/selection.json)、[freeze](final-v7/formal/freeze.json)和[冻结门禁证明](formal-freeze-and-external-hash-gate.json)：先六类P/R合格，再结构等权F1等规则选`mlp128-seed17`；冻结01:26:43UTC，TEST提取随后开始。未按TEST或gardening重选。 |
| 合成TEST、完整指标、分项及2000次簇CI | [TEST](final-v7/formal/evaluations/test.json)498,688行，全部六类P/R/F1、6×6矩阵、accuracy、macro-F1、NLL与结构/轮数/阶段/清理/位置分项齐全；[独立复核](formal-test-independent-validation.json)从实际labels/probabilities重算1024簇、2000次bootstrap及712结构，比例误差0。预定汇总目标通过，单消息S/D等低项如实单列。 |
| 正文尾部与输入位置分开 | [TEST positions](final-v7/formal/evaluations/test-positions.json)和[external positions](final-v7/formal/evaluations/external-positions.json)绑定冻结、评估及概率SHA；正文四分位、末25%、长度桶和上下文策略齐全。正文尾部与绝对输入quartile不同；无单独位置CI，不作因果推断。 |
| 冻结后128+128真实外部测试与本地历史去重 | [external manifest](final-v7/metadata/external/manifest.json)和[协议](final-v7/metadata/external/protocol.json)绑定固定源版本、先于冻结的协议及冻结后hash-only门禁。CPU并行准备实际18.988秒；模型和协议不变。历史拟合/查询CSV文件未上传，完整历史hash清单不进入Git。 |
| 外部仅原始UA、缺类N/A及FP | [外部结果](final-v7/formal/evaluations/external.json)与[独立复核](formal-external-independent-validation.json)覆盖70,866行、256簇、2000次bootstrap。只有U/A真值；S/D/C/T召回和F1为N/A、FP109/183/15/115保留；不套用DEV六类门槛当作外部失败。 |
| 原生20类、dense五模板×三格式，PNG/PDF单列新probe | [原生索引](final-v7/formal/plots/test/index.json)20/20且missing为空；[dense索引](final-v7/formal/plots/dense/index.json)15/15，明确原语义而非改写后的形式accuracy。35对实际文件、SHA及sidecar均核验；[视觉记录](plot-render-review-v7/README.zh-CN.md)明确逐图抽查范围，根另查看最长3313-token原图。 |
| 用户要求的小试与首正式gardening图片、8小时checkpoint | [首正式交付](first-formal-checkpoint-v7/README.zh-CN.md)为linear17真实18轮完成、最佳10轮，在约7.5小时内；[gardening](gardening-pilot-vs-first-formal/README.zh-CN.md)固定公开原文、三种格式、共享三次forward、各自FIT标准化，共7对PNG/PDF，已展示实际图片。无D/T正例的召回N/A并保留FP。 |
| 日志、训练曲线、固定环境、恢复及消费代码 | 严格包包含14份日志、七候选全epoch派生scalars、PNG/PDF曲线、env与requirements、运行实现和消费示例；完整raw history及`last.pt`留运行机，派生文件附其原SHA。[运行手册](../../SYNTHETIC-AGENT-RUNBOOK.zh-CN.md)保留同参数恢复与历史授权窗口边界。补充loss/放大DEV图在独立目录保留来源。 |
| 24小时上限、进程终态及最终GPU资源 | [收尾证明](gpu-resources/completion-and-monitor-stop-v7.json)：计算/报告约15.064小时完成，watchdog正常`campaign_completed`、无kill；主调度Z终态，GPU计算列表为空，监控实际退出、CSV稳定。附录与CSV最终整文件SHA一致，原扫描前253.275秒缺口、离散峰值、gardening并发及dense缺绝对窗口均明示。 |
| 小型产物回传、SHA、加载与版本归档 | [导出包](final-v7/README.md)412项载荷及3控制文件，实际逐项SHA通过，最终NPZ SHA与冻结/两套评估一致；本地交付消费端六类/1024维加载与概率形状检查通过。原文输入/完整概率/特征/大权重不在该包。提交和push在最终归档步骤核对远端分支HEAD。 |

这些证据验证的是本轮有限数据、协议和实际运行。样本审阅、源码断言、工程恢复、全输入聚合、模型评分、数值复算和视觉抽查各有不同范围，未将任一窄检查替代其他要求。
