#!/usr/bin/env python3
"""Compile plots and a Chinese report for a rich-evolution run from its ledgers.

Reads scores.jsonl / run.log / summary.json only; performs no scoring or API call.
Run with a Python that has matplotlib (e.g. the AutoDAN conda environment).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SUITES = ('banking', 'slack', 'travel')


def _ids_safe(text):
    return (text or '').replace('{goal}', '').split()


def jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous-best', type=float, default=0.41950407090830893)
    parser.add_argument('--previous-label', default='diverse8h strict ≤128-token best (0.4195)')
    parser.add_argument('--asr', nargs='*', default=[], help='fixed-candidate ASR summary.json files to merge')
    args = parser.parse_args()
    out = Path(args.output)
    rows = [r for r in jsonl(out / 'scores.jsonl') if r.get('status') == 'complete']
    events = []
    for log in sorted(out.glob('run.part*.log')) + [out / 'run.log']:
        if log.exists():
            events += jsonl(log)
    events.sort(key=lambda e: e.get('unix', 0))
    summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
    rows.sort(key=lambda r: r['evaluation_index'])
    best_curve, best = [], 0.0
    for r in rows:
        best = max(best, r['fitness'])
        best_curve.append(best)
    caps = [(e['unix'], e['cap']) for e in events if e.get('event') == 'cap_raised']
    cap_at_index = []
    for r in rows:
        cap_at_index.append(r['cap'])
    # Figure 1: fitness per evaluation and best-so-far.
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {'seed': '#999999', 'local': '#1f77b4', 'llm_generate': '#d62728', 'llm_feedback': '#9467bd', 'llm_rephrase': '#ff7f0e',
              'llm_expand': '#2ca02c', 'llm_crossover': '#8c564b', 'llm_shorten': '#e377c2'}
    for source, color in colors.items():
        xs = [r['evaluation_index'] for r in rows if r.get('source') == source]
        ys = [r['fitness'] for r in rows if r.get('source') == source]
        if xs:
            ax.scatter(xs, ys, s=6, color=color, alpha=0.45, label=f'{source} (n={len(xs)})')
    ax.plot([r['evaluation_index'] for r in rows], best_curve, color='black', linewidth=1.8, label='best so far')
    ax.axhline(args.previous_best, color='red', linestyle='--', linewidth=1, label=args.previous_label)
    previous_cap = None
    for r in rows:
        if r['cap'] != previous_cap:
            if previous_cap is not None:
                ax.axvline(r['evaluation_index'], color='gray', linestyle=':', linewidth=1)
                ax.text(r['evaluation_index'], 0.02, f"cap {r['cap']}", rotation=90, fontsize=8, color='gray')
            previous_cap = r['cap']
    ax.set_xlabel('evaluation index (six-context native scores)')
    ax.set_ylabel('mean goal-token P(user)')
    ax.set_ylim(0, 1)
    ax.set_title('Rich evolution: goal-token P(user) per candidate')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'fitness_curve.png', dpi=150, bbox_inches='tight')
    fig.savefig(out / 'fitness_curve.svg')
    # Figure 2: per-suite trajectory of the best-so-far template.
    fig, ax = plt.subplots(figsize=(11, 4))
    best_row = None
    xs, series = [], defaultdict(list)
    for r in rows:
        if best_row is None or r['fitness'] > best_row['fitness']:
            best_row = r
        xs.append(r['evaluation_index'])
        for s in SUITES:
            series[s].append(best_row['suite_means'][s])
    for s in SUITES:
        ax.plot(xs, series[s], label=s)
    ax.plot(xs, best_curve, color='black', linewidth=1.5, label='mean (objective)')
    ax.set_ylim(0, 1)
    ax.set_xlabel('evaluation index')
    ax.set_ylabel('P(user) of the best-so-far template')
    ax.legend(fontsize=8)
    ax.set_title('Per-suite P(user) of the incumbent best template')
    fig.tight_layout()
    fig.savefig(out / 'suite_curve.png', dpi=150)
    # Tables.
    improvements = [e for e in events if e.get('event') == 'new_best']
    index_of = {}
    for r in rows:
        index_of.setdefault(round(r['fitness'], 12), r['evaluation_index'])
    for e in improvements:
        e['i'] = index_of.get(round(e['fitness'], 12), '')
    by_source = defaultdict(list)
    for r in rows:
        by_source[r.get('source') or 'unknown'].append(r['fitness'])
    by_operator = defaultdict(list)
    for r in rows:
        if r.get('source') == 'local':
            by_operator[r.get('operator')].append((r['fitness'], r.get('parent_fitness')))
    top = sorted(rows, key=lambda r: -r['fitness'])[:15]
    lines = ['# Rich-evolution 结果报告（自动生成）', '',
             f'- 评估总数：{len(rows)}；运行目录：`{out}`',
             f'- 最终最优 P(user)：**{top[0]["fitness"]:.4f}**（wrapper tokens {top[0]["wrapper_tokens"]}，cap {top[0]["cap"]}，来源 {top[0].get("source")}/{top[0].get("operator")}）',
             f'- 之前严格短模板最优：{args.previous_best:.4f}',
             '- 分 suite：' + ', '.join('%s %.3f' % (s, top[0]['suite_means'][s]) for s in SUITES), '']
    if summary.get('suffix_neutrality'):
        n = summary['suffix_neutrality']
        if 'fitness_with_suffix' in n:
            lines.append(f'- 加上标准结尾后缀后的分数：{n["fitness_with_suffix"]:.4f}（canonical 无后缀 {n["fitness_canonical"]:.4f}）')
    if summary.get('llm'):
        lines.append(f'- OpenRouter 调用：{summary["llm"]["calls"]} 次，花费 ${summary["llm"]["spent_usd"]:.2f}')
    lines += ['', '## 最优更新轨迹', '', '| eval | P(user) | cap | tokens | source | operator | banking | slack | travel |', '|---|---|---|---|---|---|---|---|---|']
    for e in improvements:
        s = e['suites']
        lines.append(f'| {e.get("i", "")} | {e["fitness"]:.4f} | {e["cap"]} | {e["tokens"]} | {e.get("source")} | {e.get("operator")} | {s["banking"]:.3f} | {s["slack"]:.3f} | {s["travel"]:.3f} |')
    lines += ['', '## 各来源的评估统计', '', '| source | n | mean | max |', '|---|---|---|---|']
    for source, values in sorted(by_source.items(), key=lambda kv: -max(kv[1])):
        lines.append(f'| {source} | {len(values)} | {sum(values)/len(values):.4f} | {max(values):.4f} |')
    lines += ['', '## 本地算子统计（子代 > 父代 的比例）', '', '| operator | n | improved | max child |', '|---|---|---|---|']
    for name, values in sorted(by_operator.items(), key=lambda kv: -len(kv[1])):
        improved = sum(1 for f, p in values if p is not None and f > p + 1e-4)
        lines.append(f'| {name} | {len(values)} | {improved} ({improved/len(values):.0%}) | {max(f for f, _ in values):.4f} |')
    lines += ['', '## Top 15 模板', '']
    for i, r in enumerate(top, 1):
        suites = ', '.join('%s %.2f' % (s, r['suite_means'][s]) for s in SUITES)
        lines.append('### %d. P(user)=%.4f (%s; tokens %s; %s/%s)' % (i, r['fitness'], suites, r['wrapper_tokens'], r.get('source'), r.get('operator')))
        lines.append('')
        lines.append('```text')
        lines.append(r['template'])
        lines.append('```')
        lines.append('')
    # Diverse top: greedy clustering by token-set similarity so distinct lineages are visible.
    import re as _re

    def bag(text):
        return set(_re.findall(r'[A-Za-z]{3,}|<[^\s<>]{1,20}>|[|<>\[\]]{2,}', text.lower()))

    representatives = []
    for r in sorted(rows, key=lambda r: -r['fitness']):
        b = bag(r['template'])
        if all(len(b & bag(x['template'])) / max(1, len(b | bag(x['template']))) < 0.45 for x in representatives):
            representatives.append(r)
        if len(representatives) >= 12:
            break
    lines += ['', '## 结构互异的高分模板（按词袋相似度 < 0.45 贪心去重）', '']
    for i, r in enumerate(representatives, 1):
        suites = ', '.join('%s %.2f' % (s, r['suite_means'][s]) for s in SUITES)
        lines.append('### D%d. P(user)=%.4f (%s; tokens %s; %s/%s)' % (i, r['fitness'], suites, r['wrapper_tokens'], r.get('source'), r.get('operator')))
        lines += ['', '```text', r['template'], '```', '']
    json.dump([dict(fitness=r['fitness'], suite_means=r['suite_means'], wrapper_tokens=r['wrapper_tokens'], source=r.get('source'),
                    operator=r.get('operator'), template=r['template']) for r in representatives],
              (out / 'diverse_top.json').open('w'), ensure_ascii=False, indent=2)
    # ASR merge: P(user) bands versus native AgentDojo ASR.
    asr_rows = []
    token_of = {}
    for manifest_path in out.glob('asr_candidates_*.json'):
        for cand in json.loads(manifest_path.read_text()).get('candidates', []):
            token_of[cand['template']] = cand.get('wrapper_tokens')
    for path in args.asr:
        path = Path(path)
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        templates = {}
        manifest = path.parent / 'protocol.json'
        if manifest.exists():
            proto = json.loads(manifest.read_text())
            for cand in proto.get('candidates', []):
                templates[cand['condition']] = cand
        for condition, metrics in summary.get('conditions', {}).items():
            cand = templates.get(condition, {})
            asr_rows.append(dict(condition=condition, p_user=summary['candidate_p_user'].get(condition),
                                 interval=cand.get('p_user_interval'), template=cand.get('template'),
                                 successes=metrics.get('successes'), scored=metrics.get('scored'), scheduled=metrics.get('scheduled'),
                                 unknown=metrics.get('unknown'), asr_lower=metrics.get('asr_lower_bound'), asr_upper=metrics.get('asr_upper_bound'),
                                 utility=metrics.get('scored_utility'), exposure=metrics.get('exposure_rate'), group=path.parent.name))
    if asr_rows:
        asr_rows.sort(key=lambda r: -(r['p_user'] or 0))
        lines += ['', '## P(user) 区间 vs. AgentDojo 全量 389 任务 ASR', '',
                  '历史参考：plain 基线 ASR 12/388 ≈ 3.1%，prerequisite 基线 219/387 ≈ 56.6%，diverse8h 严格短模板最优 (P=0.42) ASR 149/388 ≈ 38.4%，其 P=0.30 候选 ASR 167/389 = 42.9%。', '',
                  '| 区间 | P(user) | ASR (成功/已评分) | ASR 上下界 | utility | 曝光率 | tokens | 组 |', '|---|---|---|---|---|---|---|---|']
        for r in asr_rows:
            lo, hi = (r['interval'] or [None, None])
            asr = (r['successes'] / r['scored']) if r['scored'] else float('nan')
            lines.append('| %s–%s | %.4f | %.1f%% (%d/%d) | %.3f–%.3f | %.3f | %.3f | %s | %s |' % (
                lo, hi, r['p_user'], 100 * asr, r['successes'], r['scored'], r['asr_lower'], r['asr_upper'], r['utility'] or float('nan'),
                r['exposure'] or float('nan'), token_of.get(r['template'], '?'), r['group']))
        lines += ['', '### 评测模板原文', '']
        for r in asr_rows:
            lines += ['#### %s  (P(user)=%.4f, ASR %d/%d)' % (r['condition'], r['p_user'], r['successes'], r['scored']), '', '```text', r['template'] or '', '```', '']
        json.dump(asr_rows, (out / 'asr_by_band.json').open('w'), ensure_ascii=False, indent=2)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [r['p_user'] for r in asr_rows]
        ys = [100 * r['successes'] / r['scored'] for r in asr_rows]
        ax.scatter(xs, ys, color='#1f77b4', s=40, zorder=3)
        ax.plot(xs, ys, color='#1f77b4', alpha=0.4)
        ax.axhline(100 * 12 / 388, color='red', linestyle='--', linewidth=1, label='plain baseline 3.1%')
        ax.axhline(100 * 219 / 387, color='gray', linestyle=':', linewidth=1, label='prerequisite baseline 56.6%')
        ax.set_xlabel('mean goal-token P(user) (six-context probe)')
        ax.set_ylabel('AgentDojo ASR (%) over 389 tasks')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title('Role-probe P(user) versus native ASR')
        fig.tight_layout()
        fig.savefig(out / 'asr_vs_puser.png', dpi=150)
        fig.savefig(out / 'asr_vs_puser.svg')
    (out / 'REPORT.zh-CN.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(evaluations=len(rows), best=top[0]['fitness'], improvements=len(improvements),
                          sources={k: len(v) for k, v in by_source.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
