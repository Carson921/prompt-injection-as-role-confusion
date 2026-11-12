"""Report genuine-trace role interventions, with explicit pilot and final stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_agentdojo_trace_interventions as runtime

HERE=Path(__file__).resolve().parent
NAMES={'system':'System','developer':'Developer','user':'User','cot':'CoT','assistant':'Assistant','tool':'Tool'}


def read(path):return json.loads(Path(path).read_text())


def probability_table(path):
    d=pd.read_csv(path)
    rows=[]
    text='| Role | Replayed baseline P | Tags removed P | Change in pp | Farthest moved P | Change in pp |\n| --- | ---: | ---: | --- | ---: | --- |\n'
    for role in runtime.ROLES:
        group=d[d.source_role==role].set_index('operation')
        original,strip,move=(group.loc[op] for op in ('baseline','strip','move'))
        def change(row):
            value=f'{row.delta_pp:+.2f}'
            if pd.notna(row.delta_ci_lower_pp):value+=f' [{row.delta_ci_lower_pp:+.2f}, {row.delta_ci_upper_pp:+.2f}]'
            return value
        text+=f'| {NAMES[role]} | {100*original.mean_p:.2f}% | {100*strip.mean_p:.2f}% | {change(strip)} | {100*move.mean_p:.2f}% | {change(move)} |\n'
        rows.append(dict(role=role,baseline_p=float(original.mean_p),strip_p=float(strip.mean_p),move_p=float(move.mean_p),
                         strip_delta_pp=float(strip.delta_pp),move_delta_pp=float(move.delta_pp)))
    return text,rows


def summarize_changes(rows):
    strip=sorted(rows,key=lambda r:r['strip_delta_pp'])[:2]
    move=sorted(rows,key=lambda r:r['move_delta_pp'])[:2]
    return ('去标签后原角色概率下降最多的两类是 '+ '、'.join(f'{NAMES[r["role"]]}（{r["strip_delta_pp"]:+.2f} pp）' for r in strip)
            +'；整体移到最远边界后下降最多的是 '+ '、'.join(f'{NAMES[r["role"]]}（{r["move_delta_pp"]:+.2f} pp）' for r in move)+'。')


def role_dependent_findings(rows, swaps):
    by_role={r['role']:r for r in rows}
    developer,cot,system,user,assistant,tool=(by_role[r] for r in ('developer','cot','system','user','assistant','tool'))
    replaced=swaps.set_index(['source_role','destination_role'])
    ua,ta=(replaced.loc[(r,'assistant')] for r in ('user','tool'))
    return (
        f'去标签的影响因角色而异：保留原正文与前文上下文时，Developer 与 CoT 的原角色概率仍分别为 '
        f'{100*developer["strip_p"]:.2f}% 和 {100*cot["strip_p"]:.2f}%；User 为 {100*user["strip_p"]:.2f}%'
        f'（{user["strip_delta_pp"]:+.2f} pp）。Tool 则变化 {tool["strip_delta_pp"]:+.2f} pp，'
        '说明正确包装并非对所有正文都表现为统一的概率增益。\n\n'
        f'保留完整正文与正确包装、整块移动时，System 从 {100*system["baseline_p"]:.2f}% 变为 '
        f'{100*system["move_p"]:.2f}%；Developer 与 User 分别变为 {100*developer["move_p"]:.2f}% 和 '
        f'{100*user["move_p"]:.2f}%。CoT 与 Assistant 则仍为 {100*cot["move_p"]:.2f}% 和 '
        f'{100*assistant["move_p"]:.2f}%。因此，完整正确包装本身不足以保证位置与上下文变化后的原角色读出保持稳定。\n\n'
        f'更换为 Assistant 包装时，User 正文的 P(User) 为 {100*ua.original_role_p:.2f}%，'
        f'P(Assistant) 为 {100*ua.destination_role_p:.2f}%；Tool 正文的 P(Tool) 为 '
        f'{100*ta.original_role_p:.2f}%，P(Assistant) 为 {100*ta.destination_role_p:.2f}%。'
        '正文 token 不变时，包装可以强烈改变这些角色的读出；这种敏感度并不在六类正文之间一致。'
    )


def english_findings(rows, swaps, n, pilot):
    scope=f'the single pilot trace' if pilot else f'{n} FIT training traces'
    strip='; '.join(f'{NAMES[r["role"]]} {r["strip_delta_pp"]:+.2f}' for r in rows)
    move='; '.join(f'{NAMES[r["role"]]} {r["move_delta_pp"]:+.2f}' for r in rows)
    off=swaps[swaps.operation=='swap']
    worst=off.loc[off.original_role_delta_pp.idxmin()]
    return (f'On {scope}, removing the full observed wrapper changed mean P(original role) by '
        f'{strip} percentage points. Moving the correctly wrapped block to the farthest trace boundary '
        f'changed the same probabilities by {move} percentage points. '
        f'Among the 30 directed wrapper replacements, the largest observed reduction in P(original role) '
        f'was for {NAMES[worst.source_role]} content with the {NAMES[worst.destination_role]} wrapper: '
        f'{100*worst.baseline_original_role_p:.2f}% to {100*worst.original_role_p:.2f}% '
        f'({worst.original_role_delta_pp:+.2f} percentage points). P(the replacement role) changed '
        f'from {100*worst.baseline_destination_role_p:.2f}% to {100*worst.destination_role_p:.2f}%.\n\n'
        'Each target is the first ordinary body of its role, and its token IDs are unchanged. '
        'Token probabilities are averaged within body, then traces are equally weighted. '
        +('The pilot is descriptive; it has no confidence interval. ' if pilot else
          'The tables and plots include pointwise 95% paired trace-cluster bootstrap intervals; shared template families are not bootstrap clusters. ')+
        'These are effects on a frozen six-role probe in retrospective teacher-forced transcripts that retain historical CoT. '
        'Residual probability after stripping reflects body, position, and context jointly. Wrapper changes can shift body positions, '
        'and movement changes preceding context and message order as well as position. '
        'The results do not identify additive text/delimiter/position contribution percentages or behavioral attack success.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=['pilot','all'],required=True)
    args=parser.parse_args();out=args.output.resolve();stage=out/args.stage
    run=read(stage/'run-manifest.json')
    validation=read(stage/'independent-validation.json')
    statistics=read(stage/'analysis/independent-statistics-validation.json')
    exported=read(stage/'export-manifest.json')
    export_validation=read(stage/'export-independent-validation.json')
    assert run['status']=='complete' and validation['status']=='passed'
    assert statistics['status']=='passed' and exported['status']=='complete'
    assert export_validation['status']=='passed' and export_validation['all_saved_token_rows_compared']
    assert export_validation['run_manifest_sha256']==runtime.digest(stage/'run-manifest.json')
    assert export_validation['export_manifest_sha256']==runtime.digest(stage/'export-manifest.json')
    assert export_validation['saved_logits_bit_exact'] and export_validation['saved_probabilities_bit_exact']
    assert export_validation['token_rows']==exported['token_rows']
    assert validation['inputs_sha256']==run['inputs_sha256']==runtime.digest(stage/'inputs.jsonl')
    assert validation['protocol_sha256']==run['protocol_sha256']==runtime.digest(stage/'protocol.json')
    assert statistics['input_summary_sha256']==run['summary_sha256']==runtime.digest(stage/'input-summary.csv')
    assert exported['source_hashes'][f'{args.stage}/run-manifest.json']==runtime.digest(stage/'run-manifest.json')
    assert validation['totals']['saved_body_token_rows']==exported['token_rows']
    assert run['inputs']==exported['inputs']==validation['inputs']['inputs']
    assert run['hiddenstate_bytes']==exported['hiddenstate_bytes']==validation['totals']['hiddenstate_bytes']
    assert validation['hiddenstate_full_payload_sha256_all_verified']
    assert validation['all_probe_rows_independently_recomputed']
    source=read(out/'sources/source-manifest.json')
    protocol=read(stage/'protocol.json')
    analysis=stage/'analysis'/('pilot' if args.stage=='pilot' else 'fit')
    table,rows=probability_table(analysis/'effects.csv')
    sw=pd.read_csv(analysis/'swap-effects.csv')
    matrix='| Original content role / replacement wrapper | '+' | '.join(NAMES[r] for r in runtime.ROLES)+' |\n'
    matrix+='| --- | '+' | '.join(['---:']*6)+' |\n'
    for role in runtime.ROLES:
        part=sw[sw.source_role==role].set_index('destination_role')
        matrix+='| '+NAMES[role]+' | '+' | '.join(f'{100*part.loc[dest].original_role_p:.1f}%' for dest in runtime.ROLES)+' |\n'
    pilot=read(out/'pilot/protocol.json')['pilot_trace_id']
    n=len(protocol['selected_traces'])
    findings=summarize_changes(rows)
    title='单条 benign AgentDojo trace 的角色干预' if args.stage=='pilot' else 'Benign AgentDojo traces 的角色包装与位置干预'
    stagescope=(f'先在训练集中的 `{pilot}` 上完成 43 个条件，随后向用户反馈单条结果。该 trace 有 939 个原始 token，六种角色各只干预第一次出现的普通正文。单条结果为描述性观察，没有置信区间。'
        if args.stage=='pilot' else
        '在单条试运行结果反馈之后，完成全部 57 条原始 benign 任务的 2451 个条件。主分析对 checkpoint 实际 FIT 的 40 条训练 trace 等权平均；另对完整 57 条（FIT 40、DEV 8、TEST 9）单独汇总。每条 trace 只选每种角色第一次出现的普通正文。')
    uncertainty=('表中 P 均为目标正文对其原角色的平均概率；pp 为百分点。' if args.stage=='pilot' else
        '表中 P 均为目标正文对其原角色的平均概率；pp 为百分点，方括号内是 95% 配对 trace-cluster percentile bootstrap 区间。每条正文先对 token 求均值，再对 trace 等权平均，2000 次重采样、seed 20261112；同一 trace 的全部干预始终保留在同一重采样单元内。区间为逐项区间，不作多重检验显著性声明。重采样未按共享 system/developer/schema 模板家族聚类，因此不能把区间解释为模板独立的不确定性评估。')
    other=''
    if args.stage=='all':
        all_table,all_rows=probability_table(stage/'analysis/all/effects.csv')
        all_swaps=pd.read_csv(stage/'analysis/all/swap-effects.csv')
        other=f'''## 全部 57 条 trace 的平均

{all_table}
![All 57 traces]({stage/'analysis/all/role-interventions-main.png'})

{role_dependent_findings(all_rows,all_swaps)}

完整数据的均值与 FIT 主分析分开报告，不把包含训练来源的结果当作未见任务泛化表现。[全 57 条主图 PDF]({stage/'analysis/all/role-interventions-main.pdf'}) · [逐 token 曲线]({stage/'analysis/all/role-token-curves.pdf'})。
'''
    report=f'''# {title}

{stagescope}

[英文主图 PDF]({analysis/'role-interventions-main.pdf'}) · [新包装角色概率]({analysis/'wrapper-destination-attraction.pdf'}) · [完整六类分布]({analysis/'full-six-role-probabilities.pdf'}) · [逐 token 曲线]({analysis/'role-token-curves.pdf'})

{findings} 这些差值反映具体干预下的冻结 probe 读出变化。

{role_dependent_findings(rows,sw)}

## {'单条结果' if args.stage=='pilot' else '40 条训练 trace 的主结果'}

{uncertainty}

{table}
![Role interventions]({analysis}/role-interventions-main.png)

主图 C 先用每个输入的完整 token 长度归一化正文起点，再对 trace 求平均。箭头连接平均相对位置，不代表每条 trace 都朝同一方向移动。

## 更换角色包装

下表行是正文原角色，列是替换后的包装角色，数值为 **P 原角色**。对角线是原始正确包装；每个非对角单元对应一次独立替换，不进行累计改写。`wrapper-destination-attraction` 图另外显示 **P 新包装角色**，完整六类分布保存在 CSV 与逐 token readouts 中。

{matrix}

逐 token 变化见 `role-token-curves` 图：单条试运行按原始正文 token 对齐，全量按正文相对位置分成 20 个区间，先在每条 trace 内取均值，再对有该区间 token 的 trace 等权平均。短正文可能没有覆盖所有区间；CSV 逐格记录实际支持数。

{other}
## 来源与运行轨迹

使用用户上传的 `/home/wz55191/full57_no_payload_first_occurrence_probe_v7_20261027.tar.gz`。压缩包 SHA256 为 `{source['source_verification']['archive_sha256']}`。1673 个原始封存文件与源清单逐个核验一致，checkpoint 实际消费的 1080 个来源文件也匹配。数据共 57 个唯一 benign 任务、359 次模型调用和 58 次 native pass，没有注入 payload。

“完整 running trace”由第一份原生提示、每次原始生成 token（含 CoT 和工具调用）以及后来首次可见的原生工具返回按时间拼接。保存所有 79552 个原始生成 token，且每个只出现一次；不重复加入后续请求中重渲染过的工具调用。342 个首次角色正文与训练位置逐 token 对应；原训练保留的全部 155103 个普通正文位置均能在轨迹中追溯。原始 API 会在后续请求里删除早期 CoT，所以这里是保留完整思考过程的回顾性轨迹重放，不能把新 baseline 当作历史实时调用概率。

DEV 的 `banking/user_task_9` 包含两次真实 pass，重置边界明确记录。FIT 的 `travel/user_task_19` 有一个真正因 1024-token 上限而截断的终端 assistant 正文；保留可见正文及缺失结束标签的状态，不补造 token，也不删除这个样本。替换或移动均按该包装原样操作，这一例在来源清单和输入中有标记。

## 三类干预定义

1. 脱标签：只移除目标正文现有的完整起始 header 和结束标记，正文原始 token IDs 不变。
2. 换标签：复制同一条 trace 中另一角色首次正文的完整原生包装，保留原正文；每个原角色覆盖其余五类。assistant 的结构化工具调用 JSON 保留在上下文中，不当作普通 Assistant 目标。CoT 对应 assistant analysis channel，Assistant 对应普通 final/commentary 正文。
3. 最远移动：把目标完整包装与正文作为整体移到 trace 两端中 token 位移最大的边界，距离相同则移到开头；其他内容的相对顺序不变。独立检查与所有剩余消息边界比较，确认距离最大。

每条 trace 共 1 baseline、6 次脱标签、30 次换标签、6 次移动。干预分别从原始 trace 出发；从不把一个干预结果当作下一个的输入。所有输入完整前向，没有截断、mask padding、文本生成或真实工具执行。

## Checkpoint 与数值接口

统一使用指定 checkpoint 的本地同源副本，SHA256 `{protocol['probe_sha256']}`。它绑定原生 MXFP4 GPT-OSS-120B、固定顺序的 16 个 pre-o_proj 注意力头和 1024 维特征，六类顺序为 System、Developer、User、CoT、Assistant、Tool。BF16 原生特征按原接口转换为 FP16，再以 FP32 标准化并经冻结的 MLP256 exact GELU 输出 logits 和 softmax。模型与 probe 均为 eval 状态，没有重新训练或选择 checkpoint。

新 baseline 和全部干预都用同一个冻结 probe、同一运行后端重算，未使用源归档旧 probe 的概率作基准。原生 MXFP4/BF16 在输入形状变化时可能存在数值差异；本实验报告该固定实现上的实际读出。

## Hidden states 与可复现文件

本阶段涉及 {run['inputs']} 个输入，hidden states 总量 {run['hiddenstate_bytes']/2**30:.2f} GiB。每个输入保存目标正文 token 的 embedding、全部 36 层 post-block residual states 和 final norm 输出，使用原生 BF16；另存有序 16-head pre-o_proj BF16 特征、实际 probe 输入 FP16 特征及 token 位置。baseline 保存六种首次正文的并集，单个干预保存被干预正文。它们不是全 trace 每个上下文 token 的激活缓存。

原始文件位于 `{out}/hiddenstates/`。`{args.stage}/hiddenstates-index.parquet` 提供路径、SHA256、字节数和条件索引；正文 mask 和完整输入 IDs 在 `{args.stage}/inputs.jsonl`，逐 token logits/概率在 `{args.stage}/token-readouts.parquet` 与共享 `readouts/` 中。原始训练 trace 与映射位于 `sources/`。

[Hidden states 索引]({stage/'hiddenstates-index.parquet'}) · [逐 token 读出]({stage/'token-readouts.parquet'}) · [读取示例]({stage/'hiddenstates-loading-example.txt'}) · [复现说明]({out/'REPRODUCE.txt'})

```python
from safetensors.torch import load_file
states = load_file("/absolute/path/to/hiddenstates/INPUT_ID.safetensors")
layer12 = states["residual_layer_12"]  # [scored_body_tokens, 2880], torch.bfloat16
positions = states["positions"]      # exact token indices in the modified trace
features = states["probe_features_fp16"]  # [scored_body_tokens, 1024]
```

被动 hiddenstate hooks 已在 pilot baseline 上与不带这些 hooks 的前向逐位比较，16-head 特征完全一致。独立审计核验每个输入的 token 来源、标签改写、最大距离移动、所有 hiddenstate 文件哈希、38 组浮点状态的形状/dtype/有限性，以及实际 BF16→FP16 读出特征；还用独立 NumPy 与 math.erf 重算所有已保存 token 的 probe 概率，并复核输入均值。统计表由另一份不导入分析器的实现独立重算。Parquet 中全部 {exported['token_rows']:,} 行的坐标、token ID、logits 和概率也与原始 NPZ 逐位比较一致。记录见 `{args.stage}/independent-validation.json`、`{args.stage}/analysis/independent-statistics-validation.json` 和 `{args.stage}/export-independent-validation.json`。

## 结论范围

脱标签或换标签也可能改变正文的绝对起始位置；最远移动同时改变可见前文和位置。因此这些是包装干预与上下文位置干预的效应。去标签后仍保留高 P，只能说明剩余正文、位置和上下文共同足以支持该 probe 读出，不能单独计算“文本贡献百分比”。不同角色使用不同来源正文和长度，角色间差值也不等于纯因素贡献分解。

训练 trace 是用户指定的分析对象，且多个任务共享系统提示、开发者提示和工具 schema；这里描述固定来源上的模型内部读出，不据此推断 ASR、指令优先级、未见模板泛化或其他模型。
'''
    target=out/('PILOT.zh-CN.md' if args.stage=='pilot' else 'README.zh-CN.md')
    target.write_text(report)
    caption=(f'Role signals in benign AgentDojo running traces ({"single trace" if args.stage=="pilot" else "40 FIT training traces; all 57 reported separately"}). '
        'Each role is tested only at its first ordinary body; structural assistant tool calls remain context. '
        'Original body token IDs are preserved. The interventions independently remove the observed wrapper, replace it '
        'with each of the five other native wrappers from the same trace, or move the complete correctly wrapped block '
        'to its farthest trace boundary. Panel A shows paired changes in the mean token-level probability of the original '
        'role, in percentage points; panel B shows original-role probability under each replacement wrapper '
        '(diagonal: original baseline); panel C shows mean normalized original and moved body-start positions and associated probability changes. '
        'Each start is normalized by its complete input length before averaging; the mean arrow need not represent every trace\'s movement direction. '
        'Body tokens are averaged within input, then traces are equally weighted. '
        +('The pilot is descriptive and has no confidence intervals. ' if args.stage=='pilot' else
          'Intervals are pointwise 95% paired trace-cluster percentile bootstrap intervals from 2,000 resamples; shared template families are not bootstrap clusters. ')+
        'The backbone and supplied six-role probe are frozen. Full historical CoT is preserved in a retrospective '
        'teacher-forced transcript, rather than silently dropped by the later-request chat template. Wrapper edits may '
        'shift body positions; movement changes the visible context. These are intervention effects on probe readouts, '
        'not normalized causal contributions or behavioral attack-success rates.\n')
    (stage/'figure-caption.en.txt').write_text(caption)
    conclusions=stage/'conclusions.en.txt'
    conclusions.write_text(english_findings(rows,sw,n if args.stage=='pilot' else 40,args.stage=='pilot'))
    writeup=dict(stage=args.stage,traces=n,main_table_rows=rows,source_sha256=runtime.digest(out/'sources/traces.jsonl'),
        input_summary_sha256=runtime.digest(stage/'input-summary.csv'),report_sha256=runtime.digest(target),
        report=str(target.relative_to(out)),caption=f'{args.stage}/figure-caption.en.txt',
        conclusions=f'{args.stage}/conclusions.en.txt',conclusions_sha256=runtime.digest(conclusions),
        independent_validation_sha256=runtime.digest(stage/'independent-validation.json'),
        independent_statistics_validation_sha256=runtime.digest(stage/'analysis/independent-statistics-validation.json'),
        export_manifest_sha256=runtime.digest(stage/'export-manifest.json'),
        export_independent_validation_sha256=runtime.digest(stage/'export-independent-validation.json'))
    runtime.write(stage/'report-manifest.json',writeup)
    print(json.dumps(dict(report=str(target),stage=args.stage),ensure_ascii=False))


if __name__=='__main__':main()
