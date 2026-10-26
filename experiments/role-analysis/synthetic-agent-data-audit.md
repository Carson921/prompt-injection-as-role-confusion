# 六角色合成数据的科学审核

日期：2026-10-26。审核范围：数据构造代码、独立 CPU 属性校验、旧 pilot、v4/v5 pilot，以及 formal-v5/v7 的非 pilot FIT/DEV 样本。**最终 formal-v7 抽样审计已通过：49 篇按正文 SHA 透明继承既有通过审阅，新增一篇全文复核通过。v7 pilot 完整重放后的模型来源身份及 FIT/DEV 输入字节与 v5 相同。** formal-v5 的真实 SEO 拼接失败保留记录，不能改写为通过。本文没有读取真实正式 TEST，也没有使用分类器结果修改数据规则。逐篇语义审阅由 LLM 代理完成，不是人类标注或人类 gold-standard 审核；每项决定仅适用于其绑定文件。

本次本地状态核对时，小试七个真实候选已完成训练与仅用 DEV 的选择，候选 `mlp256-seed29` 的小试 DEV macro-F1 为 0.9774904759，已完成 18 项小试绘图。来源、输入、完整产物及概率一致性迁移校验均已通过，290 个不可变文件保持字节一致，见[迁移证明](outputs/synthetic-agent-120b-20261026/pilot-migration.json)。全 DEV 的 CPU 导出概率最大绝对差约为 1.13e-6，预测类别差异为零。此处的小试 DEV 数值不是正式模型成绩或 TEST 指标；正式调度已启动，分类器仍待后续训练与冻结。运行状态以[执行状态记录](outputs/synthetic-agent-120b-20261026/execution-status.json)为准。

## 审核对象与证据

- 数据构造：[synthetic_agent_data.py](synthetic_agent_data.py)。当前 v7 SHA256：`cefc02ef9b49ec37fb334d2be98ec75f6b473a66e922191f12e5b460c2a2e07d`。冻结 v5 SHA256 为 `8d28164339956e6f7e0143eaec1e1314a827212acda7592b6921918feecaa412`，冻结 v6 为 `8798c83001031f9fe59d163b404f33dfc33cf3125ac7b6313b7b6324711e8790`。v7 在通用联合过滤基础上修复实际来源 URL schema，没有改变记录格式、渲染、mask、角色/模板、RNG 或 `prose_metrics` 字段。旧代码和审计产物保留。
- 独立 CPU 校验：[validate_synthetic_agent_data.py](validate_synthetic_agent_data.py)。当前 SHA256：`993dddb51cef3761602673393a3154af79953d2078175cb6289792a112399b28`。
- 旧 pilot 文章抽样的[去正文元数据版](outputs/synthetic-agent-120b-20261026/pilot-source-provenance.metadata.json)保留来源、配额、48 个 source ID、全文及抽样片段 SHA、截断状态和已有代理审阅。原始载荷 SHA256 为 `772e0f6647c1897016e6e9e30489d24e763da452555c5f3668470882949f2937`，已按原字节移存到本机 `/tmp/role-retraining-original-audit-evidence/pilot-source-provenance.raw.json`，不随 Git 发布；已绑定远端证据文件未修改。去正文版 SHA256 为 `79220095d88f913489d84fa49f61e9d12555450210b8cf3bb5f5e7f898c7a7f3`。这是旧过滤器的失败审计证据，不能当作最终数据通过证明。
- C4 固定 revision：`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`，`en/validation`。
- Dolma3 固定 revision：`3a8349c2f7946cdc56f8ccf22c555672be0b3208`，`train`。本次实际解析得到的仓库仍是 `allenai/dolma3_mix-150B-1025`，以运行记录为准。

执行 `python experiments/role-analysis/validate_synthetic_agent_data.py`，当前 v7 十五项 CPU 校验全部通过，最后一轮用时 8.422 秒；`py_compile` 通过。v5 冻结前十二项、v6 冻结前十三项校验也均已通过。校验使用简化字符 tokenizer、专门模拟跨边界 BPE 的 tokenizer，以及 mock 的官方分组流。它不替代实际 GPT-OSS tokenizer 的偏移复核或远端 4096-token forward 验证，不能写成 A40 显存或模型正确性的证明。

## 文章审计：旧来源集不合格

旧过滤器各读取 C4、Dolma3 的 1576 个候选，各保留 320 篇：FIT 256、DEV 64。LLM 代理逐篇阅读了 `audit_examples` 的 48 篇，即每个语料 × FIT/DEV 各 12 篇。这是每组最早入选的样例，**不是随机抽样**，不能据此估计总体污染率。`dolma3-c959bea49c0358a908a5f38c` 在样例中只保存前 12000 字符，因此不能据此批准其完整文章。

发现十篇具有清楚的问答、访谈、评论线程或拟人对话结构；另有多篇混合归档、客户证言、论坛回复痕迹、引用和垃圾拼接。来源名为 C4 或 Dolma3 并不保证正文没有对话。错误不能通过只删除已见 source ID 来修复。

下表的“非对话”只判断已读文本的体裁，不认证事实质量；“保守弃”表示未证实为完整对话，但按本次严格边界应舍弃。v3 会额外排除大量原本属于非对话的文章。

| 组 | source_id | 代理语义判断与依据 |
| --- | --- | --- |
| C4 FIT | `c4-061b9a4ce6aad4e41325dd2c` | 第一人称软件随笔，含多次修辞问句；v3 保守弃。 |
| C4 FIT | `c4-286399b04c32b08774966c32` | 人物传记，非对话；v3 因叙述中的人称词弃。 |
| C4 FIT | `c4-70957400fb2c6eff89a3b55c` | **明确排除**：开头毛虫与蝴蝶之间的两句拟人对话。 |
| C4 FIT | `c4-7b635f4b253e2cc5fe90470f` | 拉伸服务说明，非对话广告正文；v3 因人称词弃。 |
| C4 FIT | `c4-ee3571255a1ddc1047e00506` | **明确排除**：新闻正文后混入多段读者评论。 |
| C4 FIT | `c4-f4d6f1f3713a97e1f20d25fb` | 机场建筑报道，已读正文非对话；旧样例中 v3 仍保留。 |
| C4 FIT | `c4-35b70e0da818846c4d691492` | **明确排除**：旅游博客评论、称呼、问句与多次回复交替。 |
| C4 FIT | `c4-f9cc06fa13f775085eaa1863` | 解剖书推广介绍，非对话；v3 因问号、人称词、引号弃。 |
| C4 FIT | `c4-f7b977cdf996571c3487d585` | 大量重复的地方服务 SEO 推广；保守弃。 |
| C4 FIT | `c4-ad7dc0297656bf73cba82938` | 咖啡馆自述介绍，非对话；v3 因人称词弃。 |
| C4 FIT | `c4-cbd2bd797c7d1da849689152` | **明确排除**：连续问题及对应解答段落。 |
| C4 FIT | `c4-d1b1469eea8672af82a25614` | **明确排除**：逐段向农场主提问，再由其第一人称回答的访谈。 |
| Dolma3 FIT | `dolma3-aa333781a744bcb9d9ffff09` | 画廊展览介绍，主体为英文非对话文章；旧样例中 v3 仍保留。 |
| Dolma3 FIT | `dolma3-932e61f81fce38fb84ae5db6` | **明确排除**：艺术馆介绍中嵌有多段访客留言。 |
| Dolma3 FIT | `dolma3-c113bed85d7563b266cc3d9d` | SEO 拼接，含针对另一用户名及观点的回复痕迹；保守弃。 |
| Dolma3 FIT | `dolma3-b3eba0ef092fc21180fe8197` | 公司介绍中含署名的第一人称长引语；保守弃。 |
| Dolma3 FIT | `dolma3-c470d0544dc5d14a4f64868b` | 多篇博客、活动和广播通知的归档拼接；保守弃。 |
| Dolma3 FIT | `dolma3-dd57ab06d7eb35f1b71961d6` | 重复 SEO 模板推广，混有问题形式的内容；保守弃。 |
| Dolma3 FIT | `dolma3-1afe15df7a34aecfc0559e69` | **明确排除**：项目介绍包含连续 FAQ。 |
| Dolma3 FIT | `dolma3-9080388e6c58f864316e5881` | **明确排除**：页面介绍采用连续自问自答和问候语。 |
| Dolma3 FIT | `dolma3-502577305fff46d0aa782ec1` | 捐赠商品介绍，非对话；v3 因人称词弃。 |
| Dolma3 FIT | `dolma3-b82a211d8f1e145b7cd7f702` | 严重垃圾改写及无关链接，文章质量不确定；保守弃。 |
| Dolma3 FIT | `dolma3-17224e29960b092061210981` | 人物传记，非对话；v3 因引号弃。 |
| Dolma3 FIT | `dolma3-69ffe76706cdb59356f9c19c` | 学校周记，非对话；v3 因第一人称弃。 |
| C4 DEV | `c4-35c5ed93486e059cba9466d0` | 企业案例介绍，已读正文非对话；旧样例中 v3 仍保留。 |
| C4 DEV | `c4-32392fa7f59fd55d09e03d06` | **明确排除**：WineHQ 论坛多个用户的真实提问、建议及回复。 |
| C4 DEV | `c4-d0d39a4399b3e87fc7c59dba` | 重复的产品 SEO 推广及不完整引语；保守弃。 |
| C4 DEV | `c4-5d93e3d5a1acf319e3520119` | 书籍商品页包含引文、序言及目录，文章边界混合；保守弃。 |
| C4 DEV | `c4-b1793e398693cc440506d089` | 大量重复的地方服务 SEO 推广；保守弃。 |
| C4 DEV | `c4-7a4252fe18e98a2bbb0ff7a9` | 组织声明，主要为引用的集体陈述；v3 保守弃。 |
| C4 DEV | `c4-b4c2a9234b8da5b92ffe5ee4` | 医院工作人员日记，已读文本非对话，但含修辞问句及第一人称；v3 弃。 |
| C4 DEV | `c4-243f1b0f8d6894d4aea196bc` | 产品技术介绍，非对话；v3 因术语引号弃。 |
| C4 DEV | `c4-8b41c57e66ee50897a3ced94` | 多个食谱摘要的拼接，文章边界不明确；保守弃。 |
| C4 DEV | `c4-be7c112ada44cdd96ec0bb9b` | **明确排除**：个人博客末尾含读者评论。 |
| C4 DEV | `c4-921b5b858c94fc183e6e52c3` | 医疗机构服务介绍，非对话；v3 因人称词弃。 |
| C4 DEV | `c4-bc775e00ed08a70927dfbb44` | 数字货币推广文章，未见对话结构；v3 因问号、引号、人称词弃。 |
| Dolma3 DEV | `dolma3-b5da9e374b3f55a89fd6b953` | App 下载页，含提示词示例、问答式推广及评论框 boilerplate；保守弃。 |
| Dolma3 DEV | `dolma3-0b43d7ca83121aa7c0bd0886` | 讣告文章，含直接引语及追踪代码残片；v3 保守弃。 |
| Dolma3 DEV | `dolma3-9fbf2186647142084d102719` | 建筑报道中含设计师的多段直接引语；v3 保守弃。 |
| Dolma3 DEV | `dolma3-c959bea49c0358a908a5f38c` | 已见部分为教师履历，样例达到 12000 字符截断上限，不能批准全文。 |
| Dolma3 DEV | `dolma3-5b3726faafb10a6cc2a79400` | 艺术介绍中转引创作者的第一人称陈述；保守弃。 |
| Dolma3 DEV | `dolma3-fc8cfc8ebfac7053e4d97bfa` | 艺术家履历，非对话；v3 因书名、作品名引号弃。 |
| Dolma3 DEV | `dolma3-eab6f7f62ec0a4cb7b91eee4` | 学术报告摘要与履历，未见对话结构；v3 因问号、人称词弃。 |
| Dolma3 DEV | `dolma3-1f47eba21f4a52d421bf3e7f` | 公司介绍后含五位客户署名的证言/评价；保守排除这种多作者回应材料。 |
| Dolma3 DEV | `dolma3-8ab704589d8822f7eeb738a2` | 博客归档拼接，含问候、多个问句和直接对读者说话；保守弃。 |
| Dolma3 DEV | `dolma3-7effb8350162e08d808d8ecd` | LiveJournal 个人简介，属于社交个人页；保守弃。 |
| Dolma3 DEV | `dolma3-d32ece298c366871a380021a` | 报道中包含大量人物直接引语，引号已被清洗成重复逗号；v3 仍应靠人称/话语归属词排除。 |
| Dolma3 DEV | `dolma3-d3ae06908894bfcf863de581` | 作品说明及艺术家简介，非对话；v3 因作品名引号弃。 |

### v3 通用修正与再审计条件

在任何选头或分类器拟合之前，固定以下规则：

1. 任意 `?`、`？`，或 `"`、`“`、`”`、`«`、`»` 出现即舍弃。
2. 第一、二人称人称代词整词出现即舍弃，忽略大小写；包括 `I/me/my/mine/myself/we/us/our/ours/ourselves/you/your/yours/yourself/yourselves`。因而 `US` 也会被过度排除，本轮接受此损失。
3. 舍弃话语归属、回复、问候、评论、论坛、证言、评论框和登录等通用线索；不仅检查正文，也检查 URL 与其他元数据中的相关来源线索。
4. 继续拒绝保留控制串、说话人标签、访谈、剧本、聊天、可疑排版与非英文/非散文内容；普通 `system`、`developer`、`user` 单词本身仍可保留。
5. **运行前门槛要求重新读取固定来源进行完整过滤与划分，不能只删旧 640 篇中的失败 ID，也不能未经实际输入等价验证就复用缓存。** 后续 v5/v7 已按固定种子完成抽样语义审计，v7 完整重放和输入等价证据见下。

将 v3 应用于旧 48 个审计样例的可见正文和 URL，只有三篇仍保留，三篇均为已读非对话材料。将其应用于旧 640 篇完整缓存文章的正文和规范化 URL，C4 最多保留 25 篇、Dolma3 最多保留 18 篇，共 43/640，即 6.72%。该复查没有旧候选的全部原始元数据，因此是“正文和已存 URL”层面的存活上界，不是 v3 新来源筛选的验收结果或过滤精度估计。

这些规则会显著改变文章风格与主题分布，倾向第三人称陈述、机构介绍和说明文。正式结论须说明这种筛选偏差，不能声称代表全部 C4/Dolma3 或全部非对话写作。早期来源未通过时没有用于扫描；最终通过链见 v5/v7 各项决定。

### v4 固定官方主题分组

v3 读取过程中，Dolma3 在 FIT 63、DEV 10 时提前耗尽。调查表明，固定旧 revision 的默认 README config 仅列出 `data/common_crawl-art_and_design-0016/*`，不能把默认 config 读出的这个子集当作完整 mix。此问题发生于选头/拟合之前，没有通过减少 Dolma3 配额或放宽非对话过滤来凑数。

v4 增加显式 `data_file_groups`：每组以同一个固定官方 revision、明确的文件路径或 glob 调用 `load_dataset`；每轮从每个活跃组读取一篇，只有组耗尽才退出。轮转不依赖正文、角色或评分。provenance 保存规范化组名、文件选择和 round-robin 策略，选中的文章与快照保存 `source_group`。C4/Dolma3 源文章数仍按 50%/50% 配额。

本轮配置实际使用 `art_and_design/history_and_geography/education_and_jobs/science_math_and_technology/industrial/transportation` 六个官方主题的固定 0019 分片；准确路径以冻结 config 和运行 provenance 为准。不同长度的三组 mock 流验证了公平轮转、耗尽处理及 pilot/formal 的确定性前缀一致性。该修正使用了新缓存目录，没有把 v3 默认子集来源清单当作完整来源。

运行前审查还发现六组 round-robin 与独立角色的六周期轮换可能产生关联。因此最终 v4 不用固定角色轮换：每个语料内部每连续六篇接受文章组成一块，对 `SDUCAT` 做独立、固定种子的随机排列；单轮模板同样每五篇一块，对五个片段做随机排列。角色与模板分别使用 `single-role-v1`、`single-template-v1` RNG namespace。

精确种子为 `int(SHA256(f'{seed}:{namespace}:{source}:{block}')[:16], 16)`，再用 `random.Random(seed_integer).shuffle` 生成当前块排列，`source_ordinal % block_size` 取对应位置。它不消费正文抽样和多轮结构的 RNG。完整块保持类别均衡，块间打乱主题映射。新增 CPU 测试构造六个等长、无过滤的主题流，检查每块六角色全覆盖，十块间映射不固定并可逐项复现。实际仍应报告 `corpus × source_group × independent_role` 计数，不能把有限样本的偶然相关性当作已经不存在。

## v4 随机分层审计与 v5 固定门槛

v4 pilot 的完整来源集为 640 篇，绑定 `sources.jsonl` SHA256 `d13bf5bfeae57759c17797f422d6fef2189f04a4cdfc9c912787a1ec1c1b9bb8`，`data-manifest.json` SHA256 `39bceaee822a2a146cf75d615a36a9cfa1616229e3c07309a808d14902b9e83b`。其 FIT 有 249344 行、2292 个快照、2644469 input tokens；DEV 有 124710 行、558 个快照、629612 input tokens。这些是被拒绝的数据版本的工程计数，不是训练成绩。

对每个语料 × FIT/DEV，按 `SHA256('audit-v4-1:'+source_id)` 取最小的 12 篇，共 48 篇，逐篇阅读全文；另对原抽样缺少的 `history_and_geography × DEV` 补最小 hash 的 `dolma3-ea8ed769fa97bd8f5836a1af`，共 49 篇。六个 Dolma3 主题 × FIT/DEV 均有样本。所有 Dolma3 FIT 主题均覆盖六个独立角色；部分小 DEV 主题缺类，未补造不存在的样本。

代理审阅未发现明确的交替对话、评论、访谈或脚本；但文章边界仍不合格。固定边界为：允许连贯的非对话说明文本，包括教程、食谱、技术分类介绍、新闻、摘要；排除成绩表、商品/房产设备索引、搜索结果、SEO/关键词拼接和疑似问答。事实真伪、题材和一般文风不作为追加拒绝标准。

| v4 违规或不确定来源 | 通用问题 | v5 拒绝原因 |
| --- | --- | --- |
| `c4-df0ae08797cefc707e326149` | 成绩和球员计分列表 | 英文功能词率不足 |
| `c4-96566364ddd13d7d3d3423ea` | 多个辅具产品数据库条目 | 目录/停售 boilerplate |
| `c4-f7fbe0522961712b7e46ab92` | RAM 型号、购物关键词和外链拼接 | 目录/购物 boilerplate |
| `c4-cfafe3452c2fd3388e9f73b7` | 重复图片编号搜索结果 | 图片编号/检索 boilerplate |
| `c4-7c1292cb8f8059d6edb17b84` | 房间和设备名词索引占主要部分 | 多类房产设备索引线索共现 |
| `dolma3-d927ae7b4681510d2eab39ff` | 无问号的问题标题及解答 | 行首 WH 疑问结构 |
| `dolma3-670e886c35c971b754891ce4` | 栏杆文章中插入博彩软件锚文本 | 博彩/SEO 不确定线索 |
| `dolma3-f59c2c6cdb63ba25d6c6d8f6` | AI 图片提示词图库和用途页面 | 提示词/图库 boilerplate |
| `dolma3-f611f1d239c86664d911c8f1` | 连贯刹车教程末尾拼接关键词块 | 低功能词密度的逗号关键词串 |

v4 `audit-decision.json` 标记 `failed`，绑定上述两个 SHA，并明确 `auditor.type=LLM_agent`、`human_gold_annotation=false`。不能据此启动扫描。此决定没有改写旧来源或旧清单。

v5 在扫描/训练前固定以下门槛，所有阈值写入 manifest，代码 SHA 冻结：

- 英文功能词率至少 0.15，使用代码中的固定词表；字母词/数字词比至少 3。
- 按句末标点或换行切分；候选 prose 片段至少 8 个字母词、功能词率至少 0.15、字母词/数字词比至少 3。至少三个片段，覆盖至少 60% 字母词。这是可审计的 prose 代理指标，不是语法正确性的证明。
- 重复五词组比例不超过 0.25；数字归一化后的重复行比例不超过 0.30。
- 拒绝通用购物/停售/检索编号/图库提示词等 boilerplate、多个房产设备索引线索共现、明显逗号关键词串，以及博彩/SEO 不确定线索。
- 拒绝无问号的行首 WH 疑问结构；保留 `How to` 教程和普通的 `When notification...` 等条件陈述形式。v3 的问号、引号、人称和对话线索规则继续生效。

将 v5 应用于旧 v4 640 篇的完整正文与已存规范化 URL，保留 581/640，即 90.78%：C4 为 277/320，Dolma3 为 304/320。49 篇审阅样本中的上述九项均由通用规则拒绝，其余四十篇均保留，包括技术分类介绍、食谱、教程和摘要。原始上游元数据未全部保留，因此这些是基于现有缓存的复查计数，不是新 v5 来源集的验收。没有设置 source ID 或站点黑名单。后续仅在发现真实违反已固定边界的情况下修复，不因主题、事实质量或文风继续扩展标准。

### 缓存绑定与网络恢复

数据指纹包含 `min_article_words`、`max_source_candidates`、seed、来源定义、划分与行数配额、长度上限、代码 SHA 和 tokenizer 绑定。实际 tokenizer 必须在本地有可校验的 `tokenizer.json`；另绑定 `config.json/tokenizer_config.json/special_tokens_map.json` 的 SHA 或明确缺失状态，并保留 backend tokenizer 序列化 SHA。同一路径下改变 tokenizer 配置字节会拒绝恢复。无文件的简化 tokenizer fallback 明记 `engineering_only`，仅用于 CPU 工程测试。

HF dataset info 和各组初始化最多尝试五次，固定退避 5/10/20/30 秒。迭代遇已识别的暂时网络错误时，每组最多重启四次，重建同一固定、未打乱的流，再用 `dataset.skip(emitted)` 跳过已交付行。单测模拟初始化 504 及交付一行后的断连，确认恢复后逐行无重复、无遗漏；另确认持续错误有界退出，配置 `ValueError` 不重试。该机制不吞掉文章解析或过滤错误。

### v5 pilot 的绑定通过决定

[pilot-data-audit.json](outputs/synthetic-agent-120b-20261026/pilot-data-audit.json) 保存完整审计决定、49 个来源 ID、逐篇语义理由、角色/模板/主题计数和局限。绑定如下：

- `data-manifest.json`：`0068a8919f87c4e8d7059106f6246634a11ee0f1a0e52703130581522b94a694`。
- `sources.jsonl`：`9dbc26b4bf8a9bad6be74cb9f39a04ce50efdfc8267665683feccdc2ebb9b41b`。
- 原 48 篇抽样文件：`9700229675d2f864f9f42610ff298c0057f70048480a4183ea1956d0b00abe1b`。
- Builder：`8d28164339956e6f7e0143eaec1e1314a827212acda7592b6921918feecaa412`。

固定抽样规则改用 `SHA256('audit-v5-1:'+source_id)`，每个语料 × FIT/DEV 取最小的 12 篇，再为缺少的 Dolma3 `history_and_geography × DEV` 补最小 hash 样本 `dolma3-8f1c0785c54a01a364c6e01a`。49 篇均逐篇阅读全文，未使用截断摘要代替。它们符合已定的连贯非对话文本边界。租赁说明、课程介绍、球员评论、技术分类和新闻摘要只要由连贯描述句构成，均按原边界保留，没有因题材、事实真假或一般文风追加标准。

来源配额实测为两个语料各 FIT 256、DEV 64。四个语料 × 划分均覆盖六类独立消息与五个单轮模板；六个 Dolma3 主题 × FIT/DEV 均有审阅样本，没有主题固定为单一独立角色。有限样本存在缺类：`education_and_jobs × FIT` 没有独立 C，若干小 DEV 主题也缺个别独立角色。此处没有补造数据；这也不等于完整多消息 profile 中缺失 C。

此决定为有限的 LLM 代理样本审计，不能证明所有 640 篇乃至原数据集绝无对话，也不证明分类器准确率或任何 TEST 成绩。正式数据扩展的来源绑定、URL/正文分离和固定样本语义审计已在 v7 完成，结果见下；最终 TEST 未用于本次过滤、选头或模型选择。

### formal-v5 非 pilot 来源审计未通过

用户明确授权回传并审计这 50 篇公开文章后，代理只读取选中的 formal FIT/DEV 正文和选样 proof，没有读取正式 TEST 正文或预测。主样本按 `SHA256('audit-formal-v5-1:'+source_id)`，每个语料 × FIT/DEV 取 12 篇，排除所有 pilot source ID；再为未覆盖的 Dolma3 主题 × DEV 补两篇，共 50 篇。代理逐篇阅读全文，复核全文哈希、入选 rank、pilot 排除和样本分层。

[formal-v5-data-audit.json](outputs/synthetic-agent-120b-20261026/formal-v5-data-audit.json) 标记 `failed`，绑定：

- 正式 `data-manifest.json`：`a7707e1fadbd31fa6053caad6e9cbdfea77b442b066efb8b72a1f0efc4cf6c55`。
- 正式 `sources.jsonl`：`3bfb59184f8f55e83f0c16ccdfd034d941c46f8547a86bb9e20bf78a6bcaac78`。
- 50 篇 `sample.jsonl`：`5b2d892bf43bf38aa9c256f239a484d60b56ea68315b92684695c55c2bef4284`。
- Builder：冻结 v5 SHA256 `8d28164339956e6f7e0143eaec1e1314a827212acda7592b6921918feecaa412`。

失败来源是新 FIT 文章 `c4-8e4c82e18102414120b0060d`：阿拉伯故事介绍中混入 Windows/PDF 故障词、torrent 下载、泰米尔食谱 PDF 和手机 App 下载检索片段，大段不完整内容重复。它违反早已固定的 SEO/关键词拼接排除边界，并非因新题材、事实真假或文风而被额外拒绝。其功能词率 0.367、prose 覆盖率 0.783、重复五词组比例 0.214，解释了固定代理指标为何仍会漏过这种带有足够散文填充的垃圾页面。

其余 49 篇符合已定边界；没有发现明确对话或真实问答线程。但一项真实失败已足以让这次正式来源质量审计不通过。50 篇含分层与补样，不能将 1/50 直接当作整个正式来源集的无偏污染率。

本次审计没有更改已冻结 pilot 数据、已进行的扫描、正式源文件或 TEST。任何后续正式数据处理必须记录科学协议/数据版本变更，不得只删除此 source ID、重抽直到通过、按测试成绩删样本，或把这次失败报告为通过。

### v6 下载/检索拼接过滤修订协议

formal-v5 FIT 中的明确失败触发以下公开记录的过滤修订；修订发生在任何 TEST 正文审阅或 TEST GPU 预测之前。不是新题材、事实正确性或文风要求，也不是 source ID、书名或站点黑名单。v5 正式集失败决定和 SHA 均保留。

五个通用下载语义类分别是：torrent/magnet；PDF/ebook/epub/mobi 等文档格式；free download；download apps/games/music/movies/software 等媒体/App 下载；read online。**至少三个不同类共现，并且重复五词组比例 ≥0.10，才联合判为下载/检索拼接。** 单个 PDF 标记，或者三个语义类但重复率未达门槛，均不会由这个新 predicate 拒绝。

[v6-download-splice-amendment-check.json](outputs/synthetic-agent-120b-20261026/v6-download-splice-amendment-check.json) 记录独立 `/tmp` 实验：旧 v5 pilot 640 篇中 632 篇无这些标记、8 篇仅一个语义类，影响为零；即使只要求两个类，pilot 也无命中。原 formal 50 篇中仅已报告的失败来源命中：四个语义类、重复率 0.213559；其余 49 篇不命中。新单测额外验证类数与重复率两项门槛、正常 PDF 说明文本保留，以及完整文章过滤入口的实际触发。

修订版必须部署到独立 `code-v6/data-v6`，保留远端 `code-v5` 和运行中的旧 pilot 工程产物。v6 pilot 必须重新走完整原始来源读取、过滤、去重和 FIT/DEV 渲染，不能复制旧 640 篇充当新流程。**旧选中文章全部不命中，并不能证明重跑数据不变**：未入选候选仍参与重复代表竞争。继承 v5 选头/特征前，必须核对实际重跑的来源成员归属和两个输入文件的逐字节 SHA256 全部一致，并记录跨版本等价证据；若不一致，则须重新选择头、训练相应小试，不能把两版混作同一来源。

正式集按原配额从原始固定来源完整重筛，仍使用抽样 seed `audit-formal-v5-1`，没有更换 seed 或仅删除已见失败 ID。v6 是中间修订版本；最终执行协议使用下述 v7 绑定及其实际重放/通过证据，不能将本段修订意图本身当作验证结果。

### v7 实际 URL schema 修复

在固定 Dolma3 revision、`art_and_design-0019` 原始流上，只读核对最初三个 raw 记录的元数据 schema。没有输出正文或 URL 明文，没有读取派生 TEST 正文，也没有运行 GPU 操作。实际 `metadata` 为 dict；文章目标 URL 候选位于 `metadata.warc_url`，而旧提取逻辑只查 `row.url/metadata.url`，因此漏用了存在的字段。

观察到的 URL 为普通 HTTPS 形状，非 CommonCrawl 档案 host、非占位 host、非空路径且无 WARC/WET 标记。三个记录的 URL、ID 与元数据 text-hash 的哈希均相同，因此这三行带有重复指示，不能据此断言已选数据实际发生 URL 跨划分泄漏。schema 漏洞和具体数据是否受影响应分别核实。

v7 增加显式 `source_url_info()`，支持 row 或 metadata 中的 `url/source_url/original_url/warc_url`，metadata 可为 dict 或 JSON 对象字符串。只接受可解析的 HTTP(S) 目标页 URL；排除 CommonCrawl/S3 档案位置和 `.warc/.warc.wet/.warc.wat/.wet/.wat/.arc` 等档案路径，包括常见压缩及 URL 转义形式。不会把存储多条记录的 WARC 文件 URL 当作某篇文章的 canonical URL。

每个新 source 记录是否有可用 URL、采用的字段、原值哈希、metadata 解析状态及不采用候选的原因。URL 信息只进入来源清单和 manifest，不增加输入 record 字段。新增独立检查验证保留代表的 canonical URL、规范化全文、SimHash 半径三都不重复，包括跨语料和跨划分情形。单测以不同正文的 C4 及 JSON-metadata Dolma3 指向同一规范化 URL，验证后者被去重；人为制造 URL、正文或 SimHash 跨划分冲突均会拒绝。

manifest 明记 URL 版本、可用/不可用及字段计数、独立检查结果和无 URL 限制：只对可用 canonical URL 声称 URL 分离，缺失或仅有档案地址的文章仍依赖全文及 SimHash 去重。不能把记录了“URL 缺失”写成保证未知 URL 绝不重合。

这次是实际 schema 修复，不是根据模型指标修改数据。协议要求在独立 `code-v7/data-v7` 全量重放来源选择和去重。来源 JSON 增加字段或补齐 URL 本身会改变元数据 SHA；只有 raw text、source ID、split、ordinal 等语义身份及 FIT/DEV 输入字节 SHA 均相同时，才允许有据继承旧 GPU 结果。任何实际源或输入改变都需重选头、重训相应小试。已冻结 v5/v6 远端代码和运行中小试保留，实际等价结果见下，不能仅凭这项协议假定相同。

### v7 完整重放的实际等价结果

主代理在远端对 v7 进行了完整原始来源选择/去重和输入重建，核对 640 篇模型来源身份映射与 v5 完全相同；不仅是对旧 640 篇应用 predicate 得到零命中。两个实际输入文件的字节 SHA 均与 v5 相同：

- FIT `inputs-fit.jsonl`：`cbeb667a883979c8bf27b9072bee78207893b0443a5b92527a42e75fd1a9390f`。
- DEV `inputs-dev.jsonl`：`b5a1e0aa3799a38eb4318c90ee5ae2f5a344041b4564d2c8e7bc9f2ec4925d3c`。

因此 URL 元数据修复没有改变这些 pilot 模型输入、目标位置或标签，可以按显式跨版本等价记录继承已运行的扫描/特征。v7 pilot 有 640 个可用 canonical URL、零不可用，其中 C4 `row.url` 320、Dolma3 `metadata.warc_url` 320；source ID、规范化正文、canonical URL 和 SimHash 半径三独立分离检查均通过。新来源元数据 SHA 发生变化是预期行为，不把它误说成旧来源 JSON 字节相同。

### 最终 formal-v7 的固定样本通过决定

[formal-v7-data-audit.json](outputs/synthetic-agent-120b-20261026/formal-v7-data-audit.json) 标记 `passed`，绑定：

- 正式 `data-manifest.json`：`6de19b0062d1ea6face9ae877b8f1dd0062dd03700dd792ebb0c9ed2e850cc6b`。
- 正式 `sources.jsonl`：`7b5b1f98bedba4b190d2187cfe0b8cb5cebd76495f7e04bc0b2a97074070329c`。
- 远端完整新样本文件：`4f178017e1e0c8a32f5b06cd494e37fe41e4fbcb0fb43be5253ffa1fa9dbd77b`。
- Builder：冻结 v7 SHA256 `cefc02ef9b49ec37fb334d2be98ec75f6b473a66e922191f12e5b460c2a2e07d`。

抽样仍使用 `audit-formal-v5-1`，四个语料 × FIT/DEV 各 12 篇，另为缺少的 Dolma3 主题 × DEV 补两篇，总数仍为 50。原失败 source 由公开记录的通用规则排除，不是改 seed 或删除一个 ID 后伪称旧版本通过。

新 proof 中 49 篇与此前逐篇通过的正文 SHA 完全相同，代理逐一核对后继承这 49 项已有语义判断，不再传输或重新声称阅读了 49 篇新正文。新增且本轮实际阅读全文的是 `c4-3cc0bcfe9e7837cd0bdcb3f3`，正文 SHA256 为 `3cc0bcfe9e7837cd0bdcb3f348b81210af658bde30e917ad795c1ebe13668412`：它由三个连贯段落介绍区域楼梯升降机供应商及功能，符合非对话商业/技术说明边界，不是型号/设备索引、检索垃圾或问答。没有追加事实真假或题材标准。

复用只依据正文身份；新 proof 的 source、split、group 才是当前审计元数据。旧本地 cohort 中的 `source_ordinal` 等字段没有用于新角色/模板计数。决定记录旧 formal-v5 `failed` parent 的文件哈希及绑定，只继承其 49 项单篇通过结果，不继承或改写其整体失败状态。

主代理还在远端核对正式 5632 个来源：5632 个 URL 可用、零缺失，C4 `row.url` 和 Dolma3 `metadata.warc_url` 各 2816 个；四项来源分离检查均通过。这个全量结构检查和有限的 LLM 语义样本审计是两种不同证据，均不能当作分类器准确率或正式 TEST 成绩。正式 TEST 正文和预测没有进入本次审计。

## 角色、工具与清理事件

固定类别顺序为 `system, developer, user, cot, assistant, tool`。S 与 D 各自使用原生消息头，未调用会重新映射角色的默认 chat template。C 的含义为 analysis channel 中的文章正文，而非真实推理。

| 构造项 | 已核对行为 | 解释范围 |
| --- | --- | --- |
| 独立消息 | 每篇文章分配一个角色；每语料六篇一块独立排列，完整块覆盖六类。 | 不是每篇文章都做六类完全配对。 |
| 单轮原始模板 | 仅 `SDUA/SDUCA/SDUCQ/SDUQTCA/SDUQTA` 五种。 | 快照仍保留其原始 `template_id`，另记实际阶段。 |
| 多轮 | 前 K−1 段从四个完成段独立有放回抽样，最后从五段抽样；UCQ 只在最后。 | K 配额均衡不是正文 token 数均衡。 |
| Q | 固定 recipient `functions.lookup`，JSON 包含字符串 `document_id/query`，与 D 中的 schema 一致，以 `<\|call\|>` 结束。 | 确定性模拟，不实际执行工具。 |
| T | 使用匹配的工具名和 `call_id`；正文来自当前文章的独立片段。 | 只保证结构、工具与来源对应，不把模拟工具返回当作真实检索结果。 |
| 完成轮次 C 清理 | 整条删除历史 C，重新渲染、分词、生成新 input IDs；当前轮 C 不删。 | 有保留/清理版本可比较相同事件正文。 |
| 前一次调用 C 清理 | 对适用的末尾工具轨迹保存 C_before→Q 快照、无 C_before 的 Q/T 快照和新输出。 | 不是增加第六种原始模板。 |

前次调用清理的来源记录由 `call_chain_id`、稳定事件 ID、`logical_event_ids`、`removed_event_ids` 和文章片段坐标关联。原始 `SDUCQ` 没有被自动续成完整中间轮次。分类器只读取实际可见输入；不能复用删除 C 前的后续 hidden states 或 KV cache。

当前正式数据没有额外提供保留工具前 C 的完整 `SDUCQT(C)A` 对照。pre-tool 与 tool-reply 阶段的新增目标不同，因此**不能**拿它们的总体准确率差异声称删除 C 的因果效应。跨已完成轮次清理则保留相同正文事件家族，适合在匹配目标上做上下文诊断。工具前 C 保留的完整对照若以后增加，只能作为另行标注的部署诊断。

## 正文 mask 与边界修复

构造器生成字符跨度；`parse_native()` 另外从完整字符串识别消息、角色、channel、recipient、工具 JSON 和停止标记，并逐项复核跨度。先识别带 recipient 的 Q，避免把 Q 的参数当作普通 assistant 正文。随后通过实际 tokenizer 的 offset mapping，只有完整落在文章正文中的 token 才进入六类目标。

头部的普通文字角色名、channel 名、控制串、停止标记、Q JSON 和 D 工具定义保留在输入中，但没有训练标签。D 的文章正文与工具定义分开记录，不能把整个 D 消息一律评分。

代码审查发现，D 正文末尾标点可能与工具定义开头换行组成同一个 BPE token。旧版本会抛错。修复后保留原始输入，**将整个跨正文/结构边界的 token 排除监督**，写入 `boundary_tokens`，报告各划分的 `excluded_boundary_tokens`。没有给混合 token 强行贴 D 标签，也没有删除相邻纯正文 token。

专门的 CPU 反例将 `.` 与其后的两个换行合成一个 token：校验该 token 不在 targets 中，其余 D 和 U 的纯正文字符 token 全部仍在。实际 prepare 也以绑定的真实 tokenizer 生成并复核偏移；绘图重建 offset 时，仍须逐项断言 input IDs 与冻结缓存相同。

## 划分、去重与 pilot/formal 子集

先做文章过滤及全局去重，再按固定文章簇键的哈希分为 FIT/DEV/TEST。划分阈值固定为 8/11、1/11、2/11；每个语料另取各划分最早达到配额的来源。所有正文片段、模板和清理版本随源文章归属一个划分。

去重同时检查规范化全文、规范化 URL，以及规范化 5-word shingles 的 64-bit SimHash 汉明距离 ≤3。四个 16-bit 分带对该半径的候选检索有确定性覆盖。重复候选不保留为额外监督文章，最终保留代表两两不落在此相似阈值内。**这不是通用改写检测，也不宣称构建了所有网页关系的单链接连通分量。** `cluster_id` 是保留代表的规范化文本 SHA256。

为保证小试是正式划分的子集，两个语料始终按相同顺序交替读取；一个源达到小试配额后，仍继续参与去重，直到另一源也完成。否则 pilot/formal 的停止位置不同可能改变跨源重复的获胜代表。独立测试特意在两个流的不同位置加入跨源重复，验证小试 FIT/DEV 的 `(source_id, split)` 均为正式对应集合的子集。正式 prepare 另做实际子集断言，并拒绝两个阶段不同的来源文件哈希或数据集 revision。

pilot 不生成 `inputs-test.jsonl`；经过过滤的候选若哈希落入 TEST，只记录跳过原因和去重指纹，不保留其正文作小试拟合或评分。CPU 校验中的合成单测 TEST 仅验证程序属性，与正式语料 TEST 无关。

## 配额与来源混淆检查

已修复两项运行前发现的混淆：早期把交错后的偶/奇编号用于独立角色时，会让 C4 只出现 S/U/A、Dolma3 只出现 D/C/T；同样编号用于 TEST 长度时，会让某个语料全为长组合。当前使用每个语料内部自己的 `source_ordinal`，角色与单轮模板按上述均衡块随机排列分配，长度按原有规则分配，两个语料分别有六类和长短组合。

以下为默认配额下的算法分配，并已用 v5 冻结源清单中的 `source_ordinal` 复核一致。每源只计原始分配一次，不把清理和调用快照扩增计入“原模板均衡”。

| 默认 pilot FIT，每个语料各 256 篇，seed=20261026 | C4 | Dolma3 |
| --- | --- | --- |
| 独立 S/D/U/C/A/T | 43/42/42/43/43/43 篇 | 43/43/43/42/42/43 篇 |
| 单轮 SDUA/SDUCA/SDUCQ/SDUQTCA/SDUQTA | 51/51/52/51/51 篇 | 51/51/51/52/51 篇 |
| 多轮 K=2/3/4 | 86/85/85 篇 | 86/85/85 篇 |

FIT 默认最多保存 250000/2000000 行，DEV/TEST 独立计数。20%/30%/50% 是保存行的目标预算；文章长度不足、某角色不存在、家族包含多个上下文等原因可以使实际份额偏离。manifest 同时报 raw 保存份额和家族权重后的份额，不能把两者混用。

同一正文事件位置跨上下文共用 `family_id`，采样时整家族保留适用上下文，家族权重合计为一。pre-tool 快照只额外监督新增 C_before，tool-reply 快照只监督 T，避免所有旧 S/D/U 随调用阶段重复大量计权。下游分类器仍须核对采样确实消费这些权重，且不得再叠加类别损失权重。

v5 pilot 清单实测如下。raw 是缓存行比例，weighted 是同一事件位置家族归一化后的质量比例；分类器的结构/适用角色平衡采样分布还需另报。

| 划分 | 行数 / 快照 | input tokens | 边界排除 token | raw 单消息/单轮/多轮 | weighted 单消息/单轮/多轮 |
| --- | --- | --- | --- | --- | --- |
| FIT | 249344 / 2304 | 2667660 | 245 | 19.92% / 29.98% / 50.10% | 24.18% / 35.40% / 40.42% |
| DEV | 124724 / 552 | 621798 | 63 | 19.98% / 29.97% / 50.06% | 23.73% / 34.84% / 41.43% |

来源与工程检查的通过仅批准绑定的 v5 pilot 进入后续扫描，不等于批准正式来源扩展、分类器选择或任何正式 TEST 结论。
