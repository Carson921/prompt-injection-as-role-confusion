"""
Step 3/3 of the agent prompt-injection evaluation.
Python port of experiments/cot-forgery-agent-evals/03-plot-injections.ipynb.

Output: experiments/cot-forgery-agent-evals/plots/*.png and asr-summary.csv
"""
import pandas as pd

from common import AGENT_DIR
from plotstyle import grouped_bars, percent_axis, plt

PLOTS = AGENT_DIR / 'plots'
CLASSES = ['ATTEMPTED_SUCCESSFUL', 'ATTEMPTED_UNSUCCESSFUL', 'DENIED', 'UNNOTICED']

VARIANTS = ['base-injection', 'cot-forgery-injection']
COLORS = {'base-injection': '#90a1b9', 'cot-forgery-injection': '#ff637e'}
LABELS = {'base-injection': 'Standard prompt injection', 'cot-forgery-injection': 'CoT Forgery'}


def load() -> pd.DataFrame:
    paths = sorted(AGENT_DIR.glob('agent-outputs-classified-*.csv'))
    if not paths:
        raise SystemExit('no agent-outputs-classified-*.csv found; run agent_01_run_injections.py first')
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.replace({'target_model': {'gptoss20': 'gpt-oss-20b', 'gptoss120': 'gpt-oss-120b'}})


def counts_table(df) -> pd.DataFrame:
    t = (
        df.groupby(['target_model', 'variant', 'output_class'], as_index=False)
        .agg(n=('redteam_prompt_ix', 'count'))
        .pivot(columns='output_class', index=['target_model', 'variant'], values='n')
        .fillna(0).reset_index()
    )
    for c in CLASSES:
        if c not in t:
            t[c] = 0.0
    t['n'] = t[CLASSES].sum(axis=1)
    t['asr'] = (t['ATTEMPTED_SUCCESSFUL'] + t['ATTEMPTED_UNSUCCESSFUL']) / t['n']
    t['exfil_completed_rate'] = t['ATTEMPTED_SUCCESSFUL'] / t['n']
    return t


def plot_asr(t):
    groups = list(t['target_model'].unique())
    values = {(r.target_model, r.variant): r.asr for r in t.itertuples()}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    grouped_bars(ax, groups, VARIANTS, values, COLORS, LABELS, width=0.6, label_size=9)
    percent_axis(ax, top=0.8)
    ax.set_ylabel('Exfiltration Attack Success Rate')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=8)
    fig.savefig(PLOTS / 'agent-eval-result.png')
    plt.close(fig)


def plot_breakdown(df):
    """Stacked outcome mix per variant: the ASR bar hides how the failures split."""
    t = (
        df.groupby(['variant', 'output_class'], as_index=False)
        .agg(n=('redteam_prompt_ix', 'count'))
        .pivot(columns='output_class', index='variant', values='n').fillna(0)
    )
    for c in CLASSES:
        if c not in t:
            t[c] = 0.0
    t = t[CLASSES].reindex([v for v in VARIANTS if v in t.index])
    frac = t.div(t.sum(axis=1), axis=0)

    palette = {'ATTEMPTED_SUCCESSFUL': '#c70036', 'ATTEMPTED_UNSUCCESSFUL': '#ff9aa9',
               'DENIED': '#00a6f4', 'UNNOTICED': '#cbd5e1'}
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    left = [0.0] * len(frac)
    for c in CLASSES:
        ax.barh(range(len(frac)), frac[c], left=left, color=palette[c], label=c.replace('_', ' ').title(),
                height=0.6, zorder=3)
        for i, (v, l) in enumerate(zip(frac[c], left)):
            if v > 0.04:
                ax.text(l + v / 2, i, f'{v:.0%}', ha='center', va='center', fontsize=7.5,
                        color='white' if c == 'ATTEMPTED_SUCCESSFUL' else '#1f2937')
        left = [a + b for a, b in zip(left, frac[c])]
    ax.set_yticks(range(len(frac)))
    ax.set_yticklabels([LABELS[v] for v in frac.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Share of episodes')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, fontsize=7)
    fig.savefig(PLOTS / 'agent-eval-outcome-breakdown.png')
    plt.close(fig)
    return t


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f'{len(df)} agent episodes')
    t = counts_table(df)
    plot_asr(t)
    breakdown = plot_breakdown(df)
    print(t.to_string(index=False))
    print()
    print(breakdown.to_string())
    t.to_csv(PLOTS / 'asr-summary.csv', index=False)
    print(f'\nplots + asr-summary.csv -> {PLOTS}')


if __name__ == '__main__':
    main()
