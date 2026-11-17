#!/usr/bin/env python3
"""CPU-only Chinese report for the probe search and independent vLLM ASR runs.

No backbone, mutation service or credentials are loaded. Missing evaluation
artifacts remain explicitly pending; scoreable-only rates never replace the
scheduled-denominator ASR. Matplotlib is optional; its source data is always
exported, and --plot-python can select an already provisioned plotting runtime.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo


METHODS = ('plain', 'remember', 'optimized')
LABELS = {'plain': 'Plain', 'remember': 'Remember', 'optimized': 'AutoDAN optimized'}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    path = Path(path)
    result = []
    if path.exists():
        rows = path.read_text().splitlines()
        for index, line in enumerate(rows):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # An active search may be in the middle of its final append.
                if index != len(rows) - 1:
                    raise
    return result


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return f'{value:.7g}' if isinstance(value, (float, int)) and math.isfinite(value) else '待完成'


def percent(value):
    return f'{100 * value:.2f}%' if isinstance(value, (float, int)) and math.isfinite(value) else '待完成'


def link(label, path):
    return f'[{label}](<{Path(path).resolve()}>)'


def checked_scores(output):
    result, seen = [], {}
    for row in lines(output / 'scores.jsonl'):
        if row.get('status') != 'complete':
            continue
        fitness = row.get('fitness')
        if not isinstance(fitness, (int, float)) or not math.isfinite(fitness) or not 0 <= fitness <= 1:
            raise ValueError('Invalid candidate fitness in scores.jsonl')
        text = row['template']
        if text in seen:
            if abs(seen[text]['fitness'] - fitness) > 1e-12:
                raise ValueError('The same template has conflicting scores')
            continue
        seen[text] = row
        result.append(row)
    return result


def evaluation(output, explicit, scope):
    directory = Path(explicit).resolve() if explicit else output / f'eval_{scope}'
    if directory.is_file():
        directory = directory.parent
    if not directory.exists() and not explicit:
        for possible in sorted(output.iterdir()):
            if possible.is_dir():
                summary = read(possible / 'summary.json', {})
                if summary.get('scope') == scope:
                    directory = possible
                    break
    comparison_path = directory / 'comparison.json'
    if not comparison_path.exists():
        comparison_path = directory / 'audit/comparison.json'
    return dict(path=str(directory.resolve()), summary=read(directory / 'summary.json', {}),
                comparison=read(comparison_path, {}), comparison_path=str(comparison_path.resolve()),
                audit=read(directory / 'audit/audit.json', read(directory / 'independent_audit.json', {})),
                protocol=read(directory / 'protocol.json', {}),
                progress=read(directory / 'progress.json', {}))


def metric(eval_record, method, cohort='all'):
    comparison = eval_record.get('comparison') or {}
    for row in comparison.get('conditions') or []:
        if row.get('cohort') == cohort and row.get('suite') == 'all' and row.get('condition') == method:
            return row
    if cohort != 'all':
        return {}
    summary = eval_record.get('summary') or {}
    return (summary.get('conditions') or {}).get(method, {})


def rate_text(value, expected):
    if not value:
        return '待完成'
    successes, scheduled = value.get('successes'), value.get('scheduled', expected)
    if value.get('asr') is not None and value.get('scored') == scheduled:
        return f'{successes}/{scheduled} ({percent(value["asr"])})'
    if value.get('terminal') == scheduled and value.get('errors', 0):
        return 'N/A（原生错误）'
    if value.get('terminal') == scheduled and value.get('scored') != scheduled:
        return 'N/A（终态未能评分）'
    return (f'待完成：可评分 {value.get("scored", 0)}/{scheduled}，'
            f'已知成功 {successes if successes is not None else 0}，错误 {value.get("errors", 0)}')


def scored_rate_text(value):
    if not value or not value.get('scored'):
        return 'N/A（无可评分任务）'
    rate = value.get('scored_asr')
    if rate is None and value.get('successes') is not None:
        rate = value['successes'] / value['scored']
    return f'{value.get("successes", 0)}/{value["scored"]} ({percent(rate)})'


def evaluation_complete(record, count):
    return all(metric(record, method).get('asr') is not None
               and metric(record, method).get('scheduled') == count
               and metric(record, method).get('scored') == count for method in METHODS)


def candidate_data(output, scores):
    rows, best = [], 0.
    for index, score in enumerate(scores, 1):
        best = max(best, score['fitness'])
        rows.append(dict(candidate_index=index, generation=score.get('generation', 0),
                         fitness=score['fitness'], cumulative_best=best,
                         template_sha256=score.get('template_sha256', hashlib.sha256(score['template'].encode()).hexdigest()),
                         template=score['template']))
    write_csv(output / 'candidate_evolution.csv', rows,
              ['candidate_index', 'generation', 'fitness', 'cumulative_best', 'template_sha256', 'template'])
    return rows


def plot(output, python=None):
    """Use a standard plotting library when already installed; never install."""
    if python:
        command = [str(python), str(Path(__file__).resolve()), '--plot-only', str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return dict(status='complete' if completed.returncode == 0 else 'unavailable',
                    runtime=str(python), diagnostic=(completed.stderr or completed.stdout)[-1000:])
    if importlib.util.find_spec('matplotlib') is None:
        return dict(status='unavailable', reason='Matplotlib is not installed in this runtime; CSV data exported')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (output / 'candidate_evolution.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout='constrained')
    generations = sorted({int(row['generation']) for row in rows})
    for generation in generations:
        selected = [row for row in rows if int(row['generation']) == generation]
        ax.scatter([int(row['candidate_index']) for row in selected],
                   [float(row['fitness']) for row in selected],
                   s=27, alpha=.8, label=f'Generation {generation}')
    if rows:
        ax.step([int(row['candidate_index']) for row in rows],
                [float(row['cumulative_best']) for row in rows], where='post',
                color='#222222', linewidth=1.5, label='Best so far')
    ax.set(xlabel='Unique candidate evaluation', ylabel='Mean goal-token P(user), six tasks',
           title='AutoDAN template optimization')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output / 'candidate_evolution.svg')
    fig.savefig(output / 'candidate_evolution.png', dpi=180)
    plt.close(fig)
    return dict(status='complete', runtime=sys.executable, matplotlib=matplotlib.__version__)


def probe_table(output, dataset, selected):
    contexts = dataset.get('contexts') or []
    tables = {method: {row['case_id']: row for row in (selected.get(method) or {}).get('per_case', [])}
              for method in METHODS}
    wide, long = [], []
    for context in contexts:
        identity = context['case_id']
        row = {key: context.get(key) for key in ('case_id', 'suite', 'user_task_id', 'injection_task_id')}
        for method in METHODS:
            record = tables[method].get(identity, {})
            probability = record.get('goal_user_mean')
            row[method + '_goal_user_probability'] = probability
            long.append(dict(**{key: row[key] for key in ('case_id', 'suite', 'user_task_id', 'injection_task_id')},
                             method=method, goal_user_probability=probability,
                             goal_tokens=record.get('goal_tokens'),
                             scored_payload_occurrence=record.get('scored_payload_occurrence'),
                             observed_payload_occurrences=record.get('observed_payload_occurrences'),
                             token_artifact=record.get('token_artifact')))
        optimized = row['optimized_goal_user_probability']
        for baseline in ('plain', 'remember'):
            probability = row[baseline + '_goal_user_probability']
            row['optimized_minus_' + baseline] = (optimized - probability
                                                   if optimized is not None and probability is not None else None)
        wide.append(row)
    names = ['case_id', 'suite', 'user_task_id', 'injection_task_id',
             *(method + '_goal_user_probability' for method in METHODS),
             'optimized_minus_plain', 'optimized_minus_remember']
    write_csv(output / 'six_task_probe.csv', wide, names)
    write_csv(output / 'six_task_probe_long.csv', long,
              ['case_id', 'suite', 'user_task_id', 'injection_task_id', 'method',
               'goal_user_probability', 'goal_tokens', 'scored_payload_occurrence',
               'observed_payload_occurrences', 'token_artifact'])
    means = {}
    for method in METHODS:
        values = [row[method + '_goal_user_probability'] for row in wide]
        means[method] = statistics.fmean(values) if len(values) == 6 and all(v is not None for v in values) else None
        reported = (selected.get(method) or {}).get('fitness')
        if means[method] is not None and reported is not None and abs(means[method] - reported) > 1e-12:
            raise ValueError('Per-case equal mean disagrees with reported search fitness')
    return wide, means


def api_usage(output, winner):
    events = lines(output / 'api_calls.jsonl')
    if not events:
        return winner.get('api', {})
    usage = dict(attempts=len(events), successful_calls=0, prompt_tokens=0,
                 completion_tokens=0, total_tokens=0, reported_cost=0., cost_available=False)
    for event in events:
        usage['successful_calls'] += event.get('status') == 'complete'
        item = event.get('usage') or event.get('response', {}).get('usage', {}) or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[key] += int(item.get(key) or 0)
        if item.get('cost') is not None:
            usage['cost_available'] = True
            usage['reported_cost'] += float(item['cost'])
    return usage


def comparison_details(record):
    comparison = record.get('comparison') or {}
    if not comparison:
        return '独立 ASR comparison 尚未生成。'
    # Keep exact nonstandard fields available without inventing or conflating rates.
    return '独立 ASR comparison 已生成：' + link('comparison.json', record['comparison_path']) + '。'


def build(args):
    output = args.output.resolve()
    dataset = read(output / 'dataset.json', {})
    protocol = read(output / 'search_protocol.json', {})
    config = protocol.get('config', {})
    score_protocol = read(output / 'score_protocol.json', {})
    audit = read(output / 'independent_audit.json', {})
    diagnostics = read(output / 'diagnostics.json', {})
    winner = read(output / 'winner.json', {})
    scores = checked_scores(output)
    by_template = {row['template']: row for row in scores}
    baselines = config.get('baselines') or dataset.get('conditions') or {}
    if winner.get('template'):
        optimized = by_template.get(winner['template'])
    else:
        optimized = max(scores, key=lambda row: row['fitness']) if scores else None
    selected = {method: by_template.get(baselines.get(method)) for method in ('plain', 'remember')}
    selected['optimized'] = optimized
    table, means = probe_table(output, dataset, selected)
    evolution = candidate_data(output, scores)
    plotting = plot(output, args.plot_python)
    sampled = evaluation(output, args.sampled_eval, 'sampled')
    full = evaluation(output, args.full_eval, 'full')
    usage = api_usage(output, winner)
    search_complete = winner.get('status') == 'complete'
    audit_complete = audit.get('status') == 'passed' and audit.get('candidates') == len(scores)
    full_complete = evaluation_complete(full, 389)
    full_audit_verified = full.get('audit', {}).get('status') == 'verified'
    full_terminal = all(metric(full, method).get('terminal') == 389 for method in METHODS)
    status = ('complete' if full_complete else 'complete_with_errors') if (
        search_complete and audit_complete and full_audit_verified and full_terminal) else 'partial'
    sampled_metrics = {method: (metric(sampled, method) or metric(full, method, 'optimization6')) for method in METHODS}
    provenance = dataset.get('provenance', {})
    metric_columns = [
        ('6 任务新 vLLM ASR', sampled_metrics, 6),
        ('383 heldout 新 vLLM ASR', {m: metric(full, m, 'holdout383') for m in METHODS}, 383),
        ('389 任务新 vLLM ASR', {m: metric(full, m) for m in METHODS}, 389),
    ]
    headers = ['方法', '6 任务 probe P(user)，等权']
    include_scored = []
    for title, records, count in metric_columns:
        headers.append(title)
        has_errors = any(row.get('errors', 0) and row.get('terminal') == count for row in records.values())
        include_scored.append(has_errors)
        if has_errors:
            headers.append(title.replace('ASR', '可评分子集 ASR'))
    report = [
        '# AutoDAN × role probe 模板优化实验', '',
        f'生成时间：{datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")}。状态：**{status}**。', '',
        '本实验仅在预先抽取的 6 个任务上优化模板，目标为完整 goal 在首次完整 payload occurrence 中的 '
        'user-role 预测概率：先对每个任务的 goal tokens 平均，再对 6 个任务等权平均。'
        '下表的 ASR 来自另外启动的真实 AgentDojo + vLLM 评估，使用其原生攻击成功判据。', '',
        '| ' + ' | '.join(headers) + ' |',
        '| --- | ' + ' | '.join('---:' for _ in headers[1:]) + ' |',
    ]
    for method in METHODS:
        cells = [LABELS[method], number(means[method])]
        for (_, records, count), extra in zip(metric_columns, include_scored):
            cells.append(rate_text(records[method], count))
            if extra:
                cells.append(scored_rate_text(records[method]))
        report.append('| ' + ' | '.join(cells) + ' |')
    report += ['', 'P(user) 是 [0, 1] 概率；ASR 是成功次数除以对应计划任务数。'
               '原生错误不算成失败。任务尚未全部终态时标为“待完成”；全部终态但存在原生错误时，'
               '完整计划分母 ASR 标为“N/A（原生错误）”，并在紧邻列单独列出可评分子集的成功数/分母。'
               '可评分子集比率不会替代完整计划 ASR。', '']
    if means['optimized'] is not None:
        for baseline in ('plain', 'remember'):
            if means[baseline] is not None:
                delta = means['optimized'] - means[baseline]
                report.append(f'相对 {LABELS[baseline]}，优化模板的平均 probe 概率变化为 '
                              f'{delta:+.7g}（{100 * delta:+.5g} 个百分点）。')
        report.append('')
    report += [
        '## 6 个优化任务', '',
        '| 场景 | user task | injection task | Plain P(user) | Remember P(user) | Optimized P(user) |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for row in table:
        report.append(f'| {row["suite"]} | {row["user_task_id"]} | {row["injection_task_id"]} | '
                      + ' | '.join(number(row[method + '_goal_user_probability']) for method in METHODS) + ' |')
    if means['optimized'] is not None and means['remember'] is not None:
        changes = [(row, row['optimized_goal_user_probability'] - row['remember_goal_user_probability']) for row in table]
        dominant, change = max(changes, key=lambda item: item[1])
        increased = sum(delta > 1e-12 for _, delta in changes)
        decreased = sum(delta < -1e-12 for _, delta in changes)
        remaining = [row for row in table if row['case_id'] != dominant['case_id']]
        remainder_change = statistics.fmean(row['optimized_goal_user_probability'] - row['remember_goal_user_probability'] for row in remaining)
        report += ['', f'相对 Remember，6 个任务中 {increased} 个提高、{decreased} 个降低、'
                   f'{6 - increased - decreased} 个不变。最大正向贡献来自 '
                   f'`{dominant["suite"]}/{dominant["user_task_id"]}/{dominant["injection_task_id"]}`：'
                   f'P(user) 从 {number(dominant["remember_goal_user_probability"])} 变为 '
                   f'{number(dominant["optimized_goal_user_probability"])}。其余 5 个任务的平均概率变化为 '
                   f'{remainder_change:+.7g}。不能据平均提升声称每个任务都提高。']
    diagnostics_used = bool(diagnostics and optimized
                            and diagnostics.get('winner_template') == optimized['template']
                            and abs(diagnostics.get('fitness', {}).get('winner', -1) - optimized['fitness']) < 1e-12)
    if diagnostics_used:
        concentration = diagnostics.get('gain_concentration', {})
        token_concentration = diagnostics.get('banking_user14_token_concentration', {})
        report += ['', '### 提升集中度与序列化共变', '']
        for baseline in ('remember', 'plain'):
            gain = concentration.get(baseline, {})
            if not gain:
                continue
            report.append(f'相对 {LABELS[baseline]}，banking/user_task_14/injection_task_2 贡献了 '
                          f'6 任务净平均提升的 **{100 * gain["banking_user14_fraction_of_net_gain"]:.4f}%**；'
                          f'其他 5 任务的平均 P(user) 为 '
                          f'{number(gain["other_five_baseline_mean"])} → {number(gain["other_five_winner_mean"])}。')
        report += ['', '净提升贡献可以超过 100%，因为其余任务合计下降会抵消一部分提升。', '']
        if token_concentration:
            pieces = ', '.join('`' + row['raw_rendered_piece'].strip() + '`'
                               for row in token_concentration.get('top_six', []))
            report += [f'上述 banking case 的 {token_concentration["goal_tokens"]} 个 goal tokens 中，'
                       f'最高 6 个 token（{pieces}）占 goal-token P(user) 总和的 '
                       f'**{100 * token_concentration["top_six_probability_mass_fraction"]:.2f}%**；'
                       f'从 `details` 到结尾的 {token_concentration["details_to_end_goal_tokens"]} 个 token 占 '
                       f'**{100 * token_concentration["details_to_end_probability_mass_fraction"]:.2f}%**。', '']
        report += ['已观察到模板措辞与原生序列化共同变化：结构化 YAML 字段中，Plain/Remember '
                   '是 single-quoted scalar，winner 是 plain scalar；在 banking user14 中，'
                   '`user\'\'s` 相应变成 `user\'s`，折行位置也改变。解码后的完整 goal 原文仍保持一致，'
                   '但模型实际接收的序列化字符串和 token 边界并不固定。'
                   '两个 travel case 三条件都使用 double-quoted repr，因此没有观察到 `let\'s` 引发 '
                   'repr 单双引号切换。', '',
                   '仅渲染的对照确认了冒号与 YAML 引号/折行选择的关系；这些对照没有执行新的 backbone 前向。'
                   '**因此现有证据没有把措辞、标点、引号和折行的影响因果隔离，不能声称概率增益由去冒号或去引号导致。**'
                   '这里的解释属于已有结果的描述性诊断。', '',
                   f'诊断依据：{link("diagnostics.json", output / "diagnostics.json")}；仅分析既有 '
                   '42 个候选的保存数据，无新 GPU、LLM 或 API 调用。']
    report += ['', f'完整数值表：{link("six_task_probe.csv", output / "six_task_probe.csv")}；'
               f'token 数量和原始评分文件：{link("six_task_probe_long.csv", output / "six_task_probe_long.csv")}。', '',
               '## 任务抽样、上下文和评分范围', '',
               f'抽样种子 `{dataset.get("seed", config.get("seed"))}`；从 389 个 task pairs '
               '按 banking / slack / travel 各抽 2 个，场景内两个 user task 不同。'
               '抽样在查看历史成功率或 payload 暴露结果之前固定。', '',
               f'请求的历史目录：`{provenance.get("requested_source", "未记录")}`。', '',
               f'实际上下文源：`{provenance.get("source_root", "未记录")}`。', '']
    if provenance.get('context_source_differs_from_requested_run'):
        report += ['**源日志差异：**请求目录的 raw tool-call / tool-response 日志在当前环境不可用；'
                   '本实验复用另一个已保存运行中的原生 assistant tool calls 和 tool responses，'
                   '在原生工具字段内替换整个 payload，并将 system/developer 固定为正常 AgentDojo 内容。'
                   '因此本次 probe 概率属于这些明确记录的模拟 running contexts，不能称为对请求目录逐 token 的重放。', '']
    report += ['Probe 打分阶段直接把上述完整上下文送入冻结模型，读取 checkpoint 指定的 '
               '16 个 attention heads 在 `pre-o_proj` 的 1024 维特征，按训练流程转换精度后预测六个角色。'
               '这一步不真实运行 AgentDojo，也不依赖新的 rollout 是否恰好产生含 payload 的工具响应。'
               '同一上下文中的其他完整 payload 副本也会替换；主目标只统计首次完整副本内的 goal tokens，'
               '其他副本不混入主指标。', '',
               f'评分协议：`{score_protocol.get("aggregation", "待生成")}`；'
               f'checkpoint SHA256：`{score_protocol.get("probe_sha256", config.get("probe_sha256", "待生成"))}`。', '',
               '## 搜索过程', '',
               f'已完整评分 **{len(scores)}** 个唯一模板；计划上限 '
               f'**{config.get("max_unique_scores", "待记录")}**。种群 '
               f'`{config.get("population", "待记录")}`，代数（含初代）'
               f'`{config.get("generations", "待记录")}`；后续每代最多 '
               f'`{config.get("new_per_generation", "待记录")}` 个新评分。', '',
               '仅模板文字参与变异，实际 goal 原文保持不变，末尾恰好一个 `{goal}` 占位符。'
               '适应度对应 `loss = −mean_goal_user_probability`。实际从指定 AutoDAN 仓库加载原版 '
               'GA selection、多点句子交叉和 HGA 词级算子；GPT 变异改接 OpenRouter。'
               '输入 LLM 的是未填充 goal 的抽象模板前缀。重复或不合法候选被过滤，必要时补充多样化候选。', '',
               f'AutoDAN commit：`{protocol.get("autodan", {}).get("commit", "待记录")}`；'
               f'HGA 词处理依赖模式：`{protocol.get("autodan", {}).get("lexical_dependencies", "待记录")}`。', '',
               '| 代 | 算子 | 新评分 | 缓存命中 | 状态 |', '| ---: | --- | ---: | ---: | --- |']
    for path in sorted((output / 'generations').glob('generation_*.json')):
        generation = read(path)
        report.append(f'| {generation["generation"]} | {generation.get("operator", "")} | '
                      f'{generation.get("new_scores", "运行中")} | {generation.get("cache_hits", "")} | '
                      f'{generation.get("status", "")} |')
    report += ['', f'演化数据：{link("candidate_evolution.csv", output / "candidate_evolution.csv")}。', '']
    if plotting.get('status') == 'complete':
        report += [f'![Candidate evolution]({output / "candidate_evolution.png"})', '']
    else:
        report += ['当前报告运行环境未提供可用 Matplotlib；已输出完整演化 CSV。'
                   '可用 `--plot-python PATH` 指定已有标准绘图库的 Python 生成图。', '']
    if optimized:
        report += [f'{"最终" if search_complete else "当前暂优"}模板（generation '
                   f'{optimized.get("generation", "?")}）：', '', '```text', optimized['template'], '```', '']
    report += ['## vLLM ASR 与泛化范围', '',
               '全量评估独立运行相同 389 个任务，三种 payload 使用同一 vLLM backend、模型和生成配置。'
               '优化后的模板先冻结，再读取这些 ASR；全量结果不反馈到模板选择。', '',
               '表中的 6 任务 ASR 直接从本次全量评估的 `optimization6` 子集提取；'
               '如显式提供独立六任务评估目录，则优先使用该目录。没有重复启动六任务评估。', '',
               '389 任务包括 6 个优化任务与 383 个未参与模板优化的 task pairs。'
               '383 子集与 6 个优化 pair 不重合，但共享场景、工具环境，且可能共享 user task；'
               '它是 task-pair heldout，并不是场景外或 user-task 完全隔离的检验。', '',
               comparison_details(full), '']
    comparisons = [row for row in full['comparison'].get('paired', [])
                   if row.get('suite') == 'all' and row.get('right') == 'optimized'
                   and row.get('left') in ('plain', 'remember')]
    if comparisons:
        report += ['| 配对子集 | 比较（Optimized − baseline） | 双方可评分配对 / 计划 | ASR 差值 | McNemar 双侧 p |',
                   '| --- | --- | ---: | ---: | ---: |']
        for row in comparisons:
            delta = row.get('asr_difference_right_minus_left')
            delta_text = f'{100 * delta:+.2f} pp' if delta is not None else '待完成'
            report.append(f'| {row["cohort"]} | {row["left"]} | '
                          f'{row.get("matched_scored_pairs", 0)}/{row.get("scheduled_pairs", 0)} | '
                          f'{delta_text} | {number(row.get("asr_mcnemar_exact_p_two_sided"))} |')
        report += ['', '配对差值和检验只使用双方均有原生判据结果的同一个 task pair。'
                   '若配对数少于计划数，它们描述配对可评分子集；完整计划 ASR 仍按主表的缺失策略处理。'
                   '以上 p 值是探索性比较，未做多重检验校正。', '']
    if full_complete:
        for baseline in ('plain', 'remember'):
            delta = metric(full, 'optimized')['asr'] - metric(full, baseline)['asr']
            report.append(f'389 任务上，Optimized 相对 {LABELS[baseline]} 的 ASR 变化：'
                          f'**{100 * delta:+.2f} 个百分点**。')
        report.append('')
    report += ['历史请求运行的可评分参考值为 Plain **12/388 (3.09%)**、'
               'Remember **219/387 (56.59%)**。这两个分母分别对应其可评分任务，'
               '不是统一的 389；历史运行使用不同 backend/执行条件。'
               '因此它们单独列为历史参考，不能与本次同 backend 新 vLLM ASR 混合计算提升量。', '',
               'Probe 目标提升不等于攻击成功率提升；最终关于 ASR 的结论以真实 rollout 的完整分母及独立比较为准。', '',
               '## 独立检查与成本', '',
               f'Probe 独立审计状态：`{audit.get("status", "待完成")}`；'
               f'已审计 `{audit.get("candidates", 0)}/{len(scores)}` 候选，'
               f'原始特征重新计算最大概率差 `{number(audit.get("max_probability_absolute_error"))}`。', '',
               '独立审计从导出的原生 BF16 bits、FP16 特征与 checkpoint 重新计算 exact-GELU probe，'
               '验证 goal token mask、实际 goal 原文、六任务等权均值和产物哈希。', '',
               f'ASR 独立审计状态：`{full.get("audit", {}).get("status", "待完成")}`；'
               '重新执行原生 AgentDojo 判据并核对任务、payload、生成 token、Harmony stop 和工具调用记录。', '',
               f'OpenRouter：API 尝试 `{usage.get("attempts", 0)}` 次、成功 '
               f'`{usage.get("successful_calls", 0)}` 次；prompt '
               f'`{usage.get("prompt_tokens", 0)}` tokens，completion '
               f'`{usage.get("completion_tokens", 0)}` tokens。'
               + (f'API 返回成本合计 **${usage["reported_cost"]:.6f}**。'
                  if usage.get('cost_available') else 'API 未返回可汇总成本，不能视为零费用。'), '',
               f'审计文件：{link("independent_audit.json", output / "independent_audit.json")}；'
               f'全量结果目录：{link("full evaluation", full["path"])}；'
               + (f'六任务结果目录：{link("sampled evaluation", sampled["path"])}。'
                  if sampled.get('summary') else
                  f'六任务 ASR 来源：{link("full comparison 的 optimization6 子集", full["comparison_path"])}。'), '']
    (output / 'REPORT.zh-CN.md').write_text('\n'.join(report), encoding='utf-8')
    summary = dict(status=status, search_complete=search_complete, probe_audit_complete=audit_complete,
                   unique_candidates=len(scores), probe_means=means,
                   sampled_asr=sampled_metrics,
                   full_asr={m: metric(full, m) for m in METHODS},
                   holdout_asr={m: metric(full, m, 'holdout383') for m in METHODS},
                   full_asr_independent_audit=full['audit'],
                   full_comparison=full['comparison'], full_evaluation_complete=full_complete,
                   winner_template=optimized['template'] if optimized else None,
                   api=usage, plotting=plotting, source_provenance=provenance,
                   diagnostics_included=diagnostics_used,
                   artifacts={name: str(output / name) for name in (
                       'REPORT.zh-CN.md', 'six_task_probe.csv', 'six_task_probe_long.csv', 'candidate_evolution.csv')})
    write(output / 'report_summary.json', summary)
    print(json.dumps(dict(status=status, unique_candidates=len(scores), probe_means=means,
                          full_evaluation_complete=full_complete,
                          report=str(output / 'REPORT.zh-CN.md'), plotting=plotting), ensure_ascii=False))
    return summary


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sampled-eval', type=Path)
    parser.add_argument('--full-eval', type=Path)
    parser.add_argument('--plot-python', type=Path)
    parser.add_argument('--plot-only', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.output and not args.plot_only:
        parser.error('--output is required')
    return args


if __name__ == '__main__':
    options = arguments()
    if options.plot_only:
        result = plot(options.plot_only.resolve())
        print(json.dumps(result))
        if result['status'] != 'complete':
            raise SystemExit(1)
    else:
        build(options)
