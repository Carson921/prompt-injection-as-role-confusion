"""Cross-defender overview of the phase-0 runs: one table (ASR per template per defender) plus a figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_json, write_json

RUNS = [('gpt-oss-120b', 'run-phase0-20261111'), ('gpt-oss-120b + 防御系统提示', 'run-phase0-defended-20261111'),
        ('gemini-2.5-flash', 'run-phase0-gemini25flash-20261111'), ('claude-sonnet-4.6', 'run-phase0-claudesonnet46-20261111')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=Path(__file__).with_name('outputs'))
    args = parser.parse_args()
    runs = {label: read_json(args.outputs / d / 'summary.json')['conditions'] for label, d in RUNS if (args.outputs / d / 'summary.json').exists()}
    probes = {label: read_json(args.outputs / d / 'probe_summary.json') for label, d in RUNS if (args.outputs / d / 'probe_summary.json').exists()}
    conditions = read_json(args.outputs / RUNS[0][1] / 'conditions.json')
    order = list(conditions)
    table = {c: {label: runs[label][c]['asr'] for label in runs} for c in order}
    write_json(args.outputs / 'phase0_overview.json', dict(asr=table, probes={l: {c: p[c]['p_user_mean'] for c in order} for l, p in probes.items()}))
    lines = ['# 阶段 0 总览：同一批 44 个良性目标场景、7 个模板、4 种目标模型', '',
             '| 模板 | AgentDojo P(user) | IPI P(user) | IPI ASR | 本集 P(user)（gpt-oss） | ' + ' | '.join(runs) + ' |',
             '|---|---|---|---|---|' + '---|' * len(runs)]
    base = list(runs)[0]
    for c in order:
        spec = conditions[c]
        pu = probes.get(base, {}).get(c, {}).get('p_user_mean')
        lines.append('| %s | %s | %.3f | %.1f%% | %s | %s |' % (
            c, '-' if spec.get('p_user_agentdojo') is None else '%.3f' % spec['p_user_agentdojo'], spec['p_user_ipi'], 100 * spec['asr_ipi'],
            '-' if pu is None else '%.3f' % pu, ' | '.join('%.1f%% (%d/%d)' % (100 * runs[l][c]['asr'], runs[l][c]['successes'], runs[l][c]['complete']) for l in runs)))
    (args.outputs / 'PHASE0-OVERVIEW.zh-CN.md').write_text('\n'.join(lines) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = list(range(len(order)))
        english = {'gpt-oss-120b + 防御系统提示': 'gpt-oss-120b + defense system prompt'}
        for label in runs:
            ax.plot(x, [100 * runs[label][c]['asr'] for c in order], marker='o', label=english.get(label, label))
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('band_', '') for c in order], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('single-turn ASR (%)')
        ax.set_title('benign-goal synthetic set: ASR per template and defender')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outputs / 'phase0_overview.png', dpi=150)
    except Exception as error:
        print('figure skipped:', error)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
