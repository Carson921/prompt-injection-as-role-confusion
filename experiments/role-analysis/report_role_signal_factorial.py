"""Write an auditable report from the completed frozen factorial experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    protocol=json.loads((out/'protocol.json').read_text())
    run=json.loads((out/'run-manifest.json').read_text())
    validation=json.loads((out/'validation-independent.json').read_text())
    assert run['status']=='complete' and validation['status']=='passed'
    effects=pd.read_csv(out/'effects.csv')
    inter=pd.read_csv(out/'interactions.csv')
    family=pd.read_csv(out/'effects-family-sensitivity.csv')
    length=pd.read_csv(out/'length-sensitivity-effects.csv')
    length_validation=json.loads((out/'validation-length-sensitivity.json').read_text())
    length_run=json.loads((out/'numerical-length-sensitivity/run-manifest.json').read_text())
    assert length_validation['status']=='passed'
    length_main=length[(length.filler_id=='pooled')&(length.metric=='mean_margin')]
    checks=json.loads((out/'normal-checks/summary.json').read_text())['checks']
    summary=pd.read_csv(out/'input-summary.csv')
    pooled=effects[(effects.filler_id=='pooled')&(effects.metric=='mean_margin')]
    labels={'text_styled_minus_destyled':'文本风格 styled 减 destyled',
            'wrapper_cot_minus_user':'角色包装 analysis 减 user',
            'position_256_minus_32':'上下文位置 256 减 32',
            'position_768_minus_32':'上下文位置 768 减 32'}

    def table(frame,names=None):
        text='| 对比 | 估计 | 95% 区间 |\n| --- | ---: | --- |\n'
        for _,r in frame.iterrows():
            name=names.get(r['contrast'],r['contrast']) if names else r['contrast']
            text+=f'| {name} | {r.estimate:+.3f} | [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}] |\n'
        return text

    normal='| 角色 | 正确角色平均概率 | token argmax 比例 |\n| --- | ---: | ---: |\n'
    for role in protocol['roles']:
        rows=[r for r in checks if r['role']==role]
        normal+=f'| {role} | {sum(r["native_role_mean_probability"] for r in rows)/2:.4f} | {sum(r["token_accuracy"] for r in rows)/2:.4f} |\n'
    ilabels={'text_x_wrapper_at_32':'Text × Wrapper 在 32',
             'text_x_wrapper_at_256':'Text × Wrapper 在 256',
             'text_x_wrapper_at_768':'Text × Wrapper 在 768',
             'text_x_wrapper_256_minus_32':'交互差值 256 减 32',
             'text_x_wrapper_768_minus_32':'交互差值 768 减 32'}
    e=pooled.set_index('contrast')
    length_table='| 对比 | 原主估计 | 无结束符估计 | 改变 | 改变的 95% 区间 |\n| --- | ---: | ---: | ---: | --- |\n'
    for _,r in length_main.iterrows():
        length_table+=f'| {labels[r["contrast"]]} | {r.original_estimate:+.3f} | {r.alternate_estimate:+.3f} | {r.delta_estimate:+.3f} | [{r.delta_ci_lower:+.3f}, {r.delta_ci_upper:+.3f}] |\n'
    directions_match=bool((length_main.original_estimate*length_main.alternate_estimate>0).all())
    code=out/'code';code.mkdir(exist_ok=True)
    files=['run_role_signal_factorial.py','prepare_role_signal_pairs.py','analyze_role_signal_factorial.py',
           'analyze_role_signal_family_sensitivity.py','validate_role_signal_factorial.py',
           'export_role_signal_readouts.py','report_role_signal_factorial.py','synthetic_agent_classifier.py',
           'check_role_signal_numerics.py','run_role_signal_length_sensitivity.py',
           'analyze_role_signal_length_sensitivity.py','validate_role_signal_length_sensitivity.py',
           'polish_role_signal_figures.py']
    for name in files:shutil.copy2(HERE/name,code/name)
    for name in ['role_signal_pairs-audit.json']:
        shutil.copy2(HERE/name,out/name)
    plan=Path('/home/wz55191/ROLE-SIGNAL-EXPERIMENT-PLAN.zh-CN.md')
    shutil.copy2(plan,out/'source-experiment-plan.zh-CN.md')
    dependencies=HERE.parents[1]/'experiments'
    runtime_files={'scorer.py':dependencies/'autodan-role-probe/scorer.py',
                   'collect.py':dependencies/'agentdojo-first-tool/collect.py',
                   'attention.py':dependencies/'agentdojo-first-tool/attention.py',
                   'spans.py':dependencies/'agentdojo-first-tool/spans.py'}
    runtime_dir=code/'runtime-reference';runtime_dir.mkdir(exist_ok=True)
    for name,path in runtime_files.items():shutil.copy2(path,runtime_dir/name)
    (out/'source-manifest.json').write_text(json.dumps(dict(
        plan_source=str(plan),plan_sha256=sha(plan),
        code={str(p.relative_to(out)):sha(p) for p in code.rglob('*.py')},
        probe_requested=protocol['probe_requested'],probe_used=protocol['probe_local'],probe_sha256=protocol['probe_sha256'],
        runtime_reference='Copies preserve provenance; execute repository scripts in their original relative layout.'),indent=2,ensure_ascii=False)+'\n')

    caption=(
      'Text, role packaging, and context position in a frozen role readout. '
      'Fifty benign controlled reasoning units each have a first-person deliberative version and an impersonal version '
      'with identical factual premises, derivations, and conclusions. Each version is crossed with native assistant-analysis '
      'and user headers and body-start token positions 32, 256, and 768, under two distinct neutral-prefix '
      'templates (1,200 formal inputs). GPT-OSS-120B and the supplied six-role 16-head MLP probe are frozen. '
      '(a) Equal-pair, equal-context mean P(CoT) minus P(user), in percentage points. '
      '(b) Signed paired main effects on the mean token-level CoT-minus-user logit margin, separately for each neutral context '
      'and their equal average. (c) Position curves for every text/header combination. '
      'Intervals are pointwise 95% percentile intervals from 2,000 semantic-unit cluster bootstrap resamples, preserving '
      'all conditions and both contexts within a unit. Wrapper effects include header-length compensation in the preceding '
      'neutral message; position effects include changed visible-prefix length and content. These are probe-readout effects, '
      'not normalized contribution percentages or behavioral attack-success effects. Family-cluster uncertainty sensitivity '
      'and signed interactions are reported separately. Native-runtime sequence-length sensitivity is assessed '
      'by removing the unscored final ending token from all formal inputs in an additional diagnostic run.\n')
    (out/'figure-caption.en.txt').write_text(caption)
    reply=(
      'We added a paired factorial experiment crossing text style, native role packaging, and context position, '
      'with 50 semantic units and two neutral-context replications (1,200 formal inputs). '
      'GPT-OSS-120B and a fixed six-role, 16-head MLP role probe were frozen. '
      f'On the mean CoT-minus-user logit margin, the paired text effect was {e.loc["text_styled_minus_destyled","estimate"]:+.2f}, '
      f'the role-packaging effect {e.loc["wrapper_cot_minus_user","estimate"]:+.2f}, '
      f'and the position contrasts were {e.loc["position_256_minus_32","estimate"]:+.2f} (256 vs. 32) and '
      f'{e.loc["position_768_minus_32","estimate"]:+.2f} (768 vs. 32). '
      'The figure shows condition means, signed effects with paired cluster-bootstrap confidence intervals, and all '
      'text-by-wrapper position curves; supplementary plots report interactions, family-cluster sensitivity, and '
      'a full-set numerical sequence-length sensitivity check. '
      'We describe the delimiter intervention as complete role packaging and the position intervention as changed '
      'visible context, rather than attributing either effect exclusively to special delimiter tokens or position encoding.\n')
    (out/'reviewer-response.en.txt').write_text(reply)
    report=f'''# 文本风格 角色包装与上下文位置对角色读出的影响

2026 年 11 月 12 日完成。主实验为 CoT 与 User 的配对交叉设计，使用指定 checkpoint 的同源本地文件，完成 **1200 条正式输入、120 条独立 pilot 和 12 条新中性正文包装检查**；另补跑 **1200 条正式输入的数值敏感性检查**。24 条临时输入只做构造检查。在本设计中，完整角色包装产生最大的平均角色读出变化，受控思考式表达相对客观推理表达的变化为负。下表单位为正文平均 `z_cot − z_user` 的变化；正值表示相对向 CoT 移动，负值表示相对向 User 移动。两套中性前文先在语义单元内等权平均。

{table(pooled,labels)}
![主实验三 panel 图](figures/role-signal-main.png)

## 实验设置与 checkpoint

原请求路径 `{protocol['probe_requested']}` 在本机不存在。使用同一实验归档的本地文件 `{protocol['probe_local']}`，SHA256 为 `{protocol['probe_sha256']}`，与归档 `SHA256SUMS`、`artifact-validation.json` 以及既有 scorer 常量一致。

该 checkpoint 绑定 GPT-OSS-120B 原生 MXFP4 backbone、16 个固定顺序的 pre-o_proj 注意力头和 1024 维特征，经 MLP256 与 exact GELU 输出 `system, developer, user, cot, assistant, tool` 六类 logits。特征转换保持 BF16 → FP16 → FP32。模型和 probe 均冻结，不训练或按结果更换 checkpoint。计划默认的 20B 第 12 层五角色设置因此改为指定 checkpoint 的原有接口；没有将它错误应用于单独层，也没有混合历史五角色概率。

正式 50 个语义单元分别具有第一人称思考式版本和客观推理版本。所有前提、推导和结论保持一致，仅替换六处表达过渡。正文长度 {int(summary.body_tokens.min())}–{int(summary.body_tokens.max())} tokens。每个单元覆盖两种风格、两种原生包装、三个位置和两套中性前文，共 24 条输入。文本属于新编写的受控推理表达，不能称为自然生成的模型 CoT。62 个来源单元的配对语义和算术已独立审核，见 `pair-review-independent.json` 及 `role_signal_pairs-audit.json`。

固定短 system 前缀为 `You are a helpful assistant.`；显式使用原生 Harmony 特殊 token 构造最小消息格式，避免 chat template 自动长前缀超过最早位置。CoT header 为 assistant/analysis，长度 5 tokens；user header 长度 3 tokens。两者都用 `<|end|>` 结束。前置完整 user 消息包含中性描述；用可复现的句子前缀与自然结句达到精确位置，补偿两种 header 的 2-token 长度差。每条实际前文、正文范围、header 起点和完整原始 IDs 均保存在 `inputs.jsonl`。正文跨边界混合 token 排除数为 0。

10 个 pilot 与 50 个正式来源互不重合。正式样本数、输入和主分析在正式读出前冻结，pilot 不用于选择 probe 或检验显著性。条件按固定 seed 在每个阶段随机执行。所有输入单条完整前向，无 padding 或截断；硬件为两张 H100 80GB，使用原始 eager attention 的 query tiling。主实验前向流程耗时 {run['seconds']:.1f} 秒，完整长度敏感性补跑耗时 {length_run['seconds']:.1f} 秒，详见 `runtime.json`、`run-manifest.json` 与 `numerical-length-sensitivity/run-manifest.json`。

## 主效应 交互和不确定性

每条输入先在预定义正文 token 内平均原始 logit 差，再对语义单元等权平均。Text 和 Wrapper 效应在其余因素水平间等权平均；Position 报告 256 对 32 和 768 对 32。原始六类概率全部保留，辅助图的概率变化以百分点表示。没有从平均概率反推 token logits，也没有计算总和为 100% 的贡献比例。

按 50 个完整语义单元进行 2000 次配对 cluster bootstrap，seed 为 20261112。每次同时保留该单元全部条件和 A/B 两套前文，因此 A/B 不作为 100 个独立样本。区间是逐项 95% percentile 区间，不作同时覆盖或多重检验显著性声明。`effects.csv` 和 `interactions.csv` 分别保留 A、B 和等权汇总。

Text × Wrapper 定义为 `(styled − destyled) 在 analysis 包装下的效应 − (styled − destyled) 在 user 包装下的效应`。对应 logit-margin 结果为：

{table(inter[(inter.filler_id=='pooled')&(inter.metric=='mean_margin')],ilabels)}

正式单元共享 20 种推理家族和相同的风格过渡模板。补充分析重抽 20 个完整家族，按抽中语义单元数归一化，保留原有的等语义单元点估计；这改变不确定性的假设。其主效应区间如下，家族间仍共享的风格模板限制了总体外推。

{table(family[(family.filler_id=='pooled')&(family.metric=='mean_margin')],labels)}

## 测量检查

每类用两个新中性正文做原生格式检查，下表对两个正文等权平均。此诊断仅 2 个正文，不是新的泛化基准。Developer 与 Tool 在这些最小上下文中表现较弱，应保留这一测量限制；主实验仅比较 CoT 和 User。

{normal}

独立验证已通过：逐条检查完整 12 或 24 条件、精确正文 token IDs 与三个位置、原生 header、逐 token softmax、全部输入汇总和全部效应及 bootstrap 区间；另对 10 个固定抽样输入使用独立 NumPy 与 math.erf 实现重算 probe。独立 probe 最大 logit 差 {validation['maximum_independent_probe_logit_difference']:.3g}，最大概率差 {validation['maximum_independent_probe_probability_difference']:.3g}。正文总读出 token 数为 {validation['scored_body_tokens']:,}。单条重放和移除正文后结束符的数值诊断见 `forward-validation.json`。

## 运行后端的输入长度敏感性

三个事先固定的输入重复前向得到完全相同的特征和读出。把正文后的结束 token 等长替换成 return 特殊 token 或普通 token，三条输入的正文特征、logits 和概率也全部逐位一致。删除这一 token 改变输入长度后，正文特征出现 3.7%–9.7% 的相对 L2 差异；正文平均 margin 改变约 +0.043、+0.591 和 +0.466，其中 CoT 平均概率最大改变约 −2.75 个百分点。原生 MXFP4 与 BF16 计算因输入形状改变而产生的数值变化不能按近零误差处理；等长替换检查仅支持这三条输入没有未来内容泄漏，不能作为全局证明。

据此新增事后数值敏感性实验：对全部 1200 条正式输入仅移除正文后的结束 token，保留全部正文及其前文 IDs、位置和条件。原主实验数据和冻结分析均保持不变。替代读出与主读出采用同一批语义单元和同一组 2000 次配对 bootstrap，直接对效应差值计算区间。

{length_table}
四项主效应方向{'全部一致' if directions_match else '存在变化'}。完整四项主效应、五项交互、A/B 前文和全部概率指标的原估计、替代估计与差值区间见 `length-sensitivity-effects.csv` 和 `length-sensitivity-interactions.csv`。这项检查量化了该长度扰动的影响；它不能消除风格版本本身多 13 tokens 的混杂，也不能证明其他运行后端或序列长度下的数值一致性。

## 图与复现文件

主图为 `figures/role-signal-main.pdf`，PNG 便于预览，SVG 便于编辑。补充图包括所有六类概率、概率效应、交互、两套前文的完整位置曲线，以及家族 bootstrap 和输入长度敏感性。所有图均有 PDF、PNG 和 SVG。可直接使用的英文图注为 `figure-caption.en.txt`，review 回复草稿为 `reviewer-response.en.txt`。

原始输入见 `inputs.jsonl`；逐 token logits、概率和特征见 `readouts/`；可重新汇总的长表见 `token-readouts.parquet`；逐输入均值见 `input-summary.csv`。`protocol.json` 保存冻结身份和口径，`validation-independent.json` 保存独立复核，`export-manifest.json` 与 `SHA256SUMS` 保存来源和产物哈希。代码快照见 `code/`，复现命令见 `reproduce.sh`。

## 结论范围

这些效应仅描述指定冻结 probe 在该受控语义集合和模板设计上的角色读出。角色包装同时改变角色文字、channel、特殊 token 和为匹配位置所作的中性前文微调；长度补偿可替换前文的结尾句，并非仅在相同前文后添加两个 token。user 条件包含相邻的中性 user 消息和目标 user 消息。上下文位置同时改变可见前文长度与内容。因此不能称为特殊 delimiter token 或纯位置编码的独立贡献。正文样式只覆盖六处受控表达变化，styled 版本固定长 13 tokens，不能将该效应解释为完全独立于长度的纯文风贡献。

本次完成计划推荐的 CoT/User 双前文主实验。User/Tool、System/User 以及 delimiter 结构 token 的独立消融属于后续扩展；本次不据此推断 ASR、指令优先级或其他角色对比。指定 checkpoint 是跨层联合读出，没有与各层分别匹配的同一 checkpoint probe，因此不生成会混淆测量接口的逐层贡献图。
'''
    (out/'README.zh-CN.md').write_text(report)
    print(json.dumps(dict(report=str(out/'README.zh-CN.md'),caption=str(out/'figure-caption.en.txt')),ensure_ascii=False))


if __name__=='__main__':main()
