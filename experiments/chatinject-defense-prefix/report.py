#!/usr/bin/env python3
"""Build REPORT.zh-CN.md and figures for the ChatInject defensive-prefix experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sxy / (sx * sy) if sx and sy else float('nan')


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def pct(m):
    return '%.1f%% (%d/%d)' % (100 * m['successes'] / m['scored'], m['successes'], m['scored']) if m['scored'] else '–'


def short(text, n=90):
    text = text.replace('\n', '⏎')
    return text if len(text) <= n else text[:n] + '…'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-asr', type=Path, required=True)
    parser.add_argument('--defense-asr', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    args = parser.parse_args()
    out = args.outputs
    rows = [json.loads(line) for line in (args.run / 'scores.jsonl').open()]
    rows = [r for r in rows if r.get('status') == 'complete']
    summary = read(args.run / 'summary.json') if (args.run / 'summary.json').exists() else {}
    baseline_run = read(args.run / 'baseline.json')
    base = read(args.baseline_asr / 'summary.json')
    defense = read(args.defense_asr / 'summary.json')
    conditions = read(args.conditions)

    # ---------- figures
    best, curve = -1, []
    for r in rows:
        best = max(best, r['fitness'])
        curve.append(best)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.scatter(range(len(rows)), [r['fitness'] for r in rows], s=3, alpha=.25, color='tab:gray', label='each prefix')
    ax.plot(curve, color='tab:blue', label='best so far')
    ax.axhline(baseline_run['fitness'], color='tab:red', linestyle='--', linewidth=1, label='no defense')
    ax.set_xlabel('evaluation'); ax.set_ylabel('attack-body P(tool)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'fitness_curve.png', dpi=150); fig.savefig(out / 'fitness_curve.svg'); plt.close(fig)

    points = []
    for c in conditions:
        m = defense['conditions'].get(c['condition'])
        if m and m['scored']:
            points.append((c['probe']['body_tool'], 100 * m['successes'] / m['scored'], c['condition'], m))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter([p[0] for p in points], [p[1] for p in points], color='tab:blue', zorder=3)
    for x, y, name, _ in points:
        ax.annotate(name.replace('defense_', ''), (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
    text_only = base['conditions'].get('remember_text_only')
    if text_only and text_only['scored']:
        ax.axhline(100 * text_only['successes'] / text_only['scored'], color='gray', linestyle=':', linewidth=1,
                   label='remember text only (no delimiters)')
    ax.set_xlabel('attack-body P(tool) (probe, 6-context mean)'); ax.set_ylabel('AgentDojo ASR (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), frameon=False)
    fig.tight_layout(); fig.savefig(out / 'asr_vs_ptool.png', dpi=150); fig.savefig(out / 'asr_vs_ptool.svg'); plt.close(fig)

    # ---------- tables
    xs, ys, us = [p[0] for p in points], [p[1] for p in points], [p[3]['scored_utility'] for p in points]
    rho = spearman(xs, ys) if len(points) > 2 else float('nan')
    r = pearson(xs, ys) if len(points) > 2 else float('nan')
    lines = ['# ChatInject 防御前缀：提高攻击 payload 的 P(tool) 能否降低 ASR', '',
             '本文件由 `report.py` 自动生成。方法说明见 `METHOD.zh-CN.md`。', '',
             '## 1. 攻击基线（无防御）', '',
             '| 条件 | payload 形式 | ASR | utility | 曝光 | 错误 |', '|---|---|---|---|---|---|']
    names = {'chatinject_user_remember': '`<|end|><|start|>user<|message|>` + remember 句 + goal + `<|end|><|start|>assistant<|channel|>final<|message|>`',
             'chatinject_user_remember_minimal': '`<|start|>user<|message|>` + remember 句 + goal + `<|end|>`',
             'remember_text_only': 'remember 句 + goal（无分隔符）'}
    for k, m in base['conditions'].items():
        lines.append('| %s | %s | %s | %.3f | %d | %d |' % (k, names.get(k, ''), pct(m), m['scored_utility'] or 0, m['exposed'], m['errors']))
    lines += ['', '探针（六上下文均值，主攻击形式，无防御）：攻击正文 P(tool) = %.4f，P(user) = %.4f；分 suite P(tool)：%s。' % (
        baseline_run['fitness'], baseline_run['body_user_mean'],
        ' / '.join('%s %.3f' % kv for kv in baseline_run['suite_means'].items())), '']
    lines += ['## 2. 防御前缀优化', '',
              '- 评估模板数：%d；最优攻击正文 P(tool) = %.4f（无防御 %.4f）。' % (len(rows), max(r['fitness'] for r in rows), baseline_run['fitness']),
              '- 运行时长 %.2f h，最终长度上限 %s token；LLM 花费 $%s / %s 次调用。' % (
                  summary.get('elapsed_hours', float('nan')), summary.get('final_cap'),
                  (summary.get('llm') or {}).get('spent_usd'), (summary.get('llm') or {}).get('calls')), '',
              '![fitness](fitness_curve.png)', '']
    lines += ['## 3. 不同 P(tool) 幅度下的 ASR（主攻击形式，389 任务全量）', '',
              '| 条件 | 前缀 token | 正文 P(tool) | 正文 P(user) | ASR | ASR 区间 | utility | banking / slack / travel ASR |',
              '|---|---|---|---|---|---|---|---|']
    for c in conditions:
        m = defense['conditions'].get(c['condition'])
        if not m:
            continue
        sm = defense['suite_metrics'][c['condition']]
        suites = ' / '.join('%.1f%%' % (100 * sm[s]['successes'] / sm[s]['scored']) if sm[s]['scored'] else '–' for s in ('banking', 'slack', 'travel'))
        lines.append('| %s | %s | %.4f | %.4f | %s | %.3f–%.3f | %.3f | %s |' % (
            c['condition'], c['probe'].get('wrapper_tokens', 0), c['probe']['body_tool'], c['probe']['body_user'], pct(m),
            m['asr_lower_bound'], m['asr_upper_bound'], m['scored_utility'] or 0, suites))
    lines += ['', '相关性（%d 个条件，正文 P(tool) vs ASR）：Spearman %.2f，Pearson %.2f。' % (len(points), rho, r), '',
              '![asr](asr_vs_ptool.png)', '']
    notes = out / 'conclusions.zh-CN.md'
    if notes.exists():
        lines += [notes.read_text().rstrip(), '']
    lines += ['## 5. 前缀全文', '']
    for c in conditions:
        if c['defense']:
            lines += ['**%s**（P(tool)=%.4f）' % (c['condition'], c['probe']['body_tool']), '', '```text', c['defense'], '```', '']
    (out / 'REPORT.zh-CN.md').write_text('\n'.join(lines) + '\n')
    (out / 'asr_by_condition.json').write_text(json.dumps(
        [dict(condition=c['condition'], defense=c['defense'], probe=c['probe'], asr=defense['conditions'].get(c['condition']))
         for c in conditions], indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(points=[(round(x, 4), round(y, 1), n) for x, y, n, _ in points], spearman=rho, pearson=r, utility=us)))


if __name__ == '__main__':
    main()
