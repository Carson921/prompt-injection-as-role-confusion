"""
Step 3/3 of the chat CoT Forgery evaluation.
Python port of experiments/cot-forgery-chat-evals/04-plot-jailbreak-stats.ipynb
(the original is R; R is not installed here).

Output: experiments/cot-forgery-chat-evals/plots/*.png and asr-summary.csv
"""
import pandas as pd

from common import CHAT_DIR
from plotstyle import grouped_bars, percent_axis, plt

PLOTS = CHAT_DIR / 'plots'

# ASRs published by OpenAI for standard jailbreak evals (gpt-oss / GPT-5 / o4-mini
# model cards); simple averages across categories, as in the notebook.
OPENAI_PRECALC_ASRS = {'gpt-oss-20b': 0.0272}


def asr(df, keys) -> pd.DataFrame:
    out = (
        df.groupby(list(keys) + ['output_class'], as_index=False)
        .agg(n=('redteam_prompt_ix', 'count'))
        .pivot(columns='output_class', index=list(keys), values='n')
        .fillna(0).reset_index()
    )
    for c in ['HARMFUL_RESPONSE', 'REDIRECTION', 'REFUSAL']:
        if c not in out:
            out[c] = 0.0
    out['n'] = out['HARMFUL_RESPONSE'] + out['REDIRECTION'] + out['REFUSAL']
    out['asr'] = out['HARMFUL_RESPONSE'] / out['n']
    return out


def load() -> pd.DataFrame:
    return (
        pd.read_csv(CHAT_DIR / 'openrouter-generations' / 'harmful-responses-classified.csv')
        .assign(target_model=lambda df: df['target_model'].str.replace(r'.*/', '', regex=True))
    )


def plot_1_basic(df):
    """ASR for harmful query vs CoT Forgery vs CoT Forgery with an absurd qualifier."""
    d = df[df['policy_style'] != 'destyled'].copy()
    d['prompt_type'] = [
        'no_policy' if ps == 'no_policy' else ('cot_forgery' if q == 'no_qualifier' else 'cot_forgery_with_qualifier')
        for ps, q in zip(d['policy_style'], d['qualifier_type'])
    ]
    t = asr(d, ['target_model', 'prompt_type'])

    series = ['no_policy', 'cot_forgery', 'cot_forgery_with_qualifier']
    colors = {'no_policy': '#00bcff', 'cot_forgery': '#ff637e', 'cot_forgery_with_qualifier': '#fd9a00'}
    labels = {'no_policy': 'Harmful prompt', 'cot_forgery': 'Harmful prompt + CoT Forgery',
              'cot_forgery_with_qualifier': 'Harmful prompt + CoT Forgery (Variant 2)'}
    groups = list(t['target_model'].unique())
    values = {(r.target_model, r.prompt_type): r.asr for r in t.itertuples()}

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    grouped_bars(ax, groups, series, values, colors, labels, width=0.75, label_size=8)
    percent_axis(ax)
    ax.set_ylabel('Attack Success Rate')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=1, fontsize=8)
    fig.savefig(PLOTS / 'user-eval-result-1.png')
    plt.close(fig)
    return t


def plot_2_vs_published(df):
    """ASR vs the standard-jailbreak ASR OpenAI publishes for the same model."""
    d = df[df['policy_style'] != 'destyled'].copy()
    d['prompt_type'] = ['no_policy' if ps == 'no_policy' else 'cot_forgery' for ps in d['policy_style']]
    t = asr(d, ['target_model', 'prompt_type'])

    series = ['no_policy', 'standard_jailbreak', 'cot_forgery']
    colors = {'no_policy': '#90a1b9', 'standard_jailbreak': '#f4a8ff', 'cot_forgery': '#fd9a00'}
    labels = {'no_policy': 'Harmful query', 'standard_jailbreak': 'Standard jailbreak (published)',
              'cot_forgery': 'CoT Forgery'}
    groups = list(t['target_model'].unique())
    values = {(r.target_model, r.prompt_type): r.asr for r in t.itertuples()}
    for m, v in OPENAI_PRECALC_ASRS.items():
        if m in groups:
            values[(m, 'standard_jailbreak')] = v

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    grouped_bars(ax, groups, series, values, colors, labels, width=0.75, label_size=8)
    percent_axis(ax)
    ax.set_ylabel('Attack Success Rate (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=1, fontsize=8)
    fig.savefig(PLOTS / 'user-eval-result.png')
    plt.close(fig)
    return t


def plot_3_by_qualifier(df):
    """ASR split by which absurd qualifier the forged policy hangs its exception on."""
    d = df[df['policy_style'] != 'destyled'].copy()
    d['prompt_type'] = [
        'no_policy' if ps == 'no_policy' else ('no_qualifier' if q == 'no_qualifier' else q)
        for ps, q in zip(d['policy_style'], d['qualifier_type'])
    ]
    t = asr(d, ['target_model', 'prompt_type'])

    series = ['no_policy', 'no_qualifier', 'lucky_coin', 'green_shirt', 'dog_owner', 'time_of_day']
    colors = {'no_policy': '#90a1b9', 'no_qualifier': '#fb2c36', 'lucky_coin': '#ffba00',
              'green_shirt': '#bbf451', 'dog_owner': '#00a6f4', 'time_of_day': '#a3b3ff'}
    labels = {'no_policy': 'No CoT Forgery', 'no_qualifier': 'Standard CoT Forgery',
              'lucky_coin': '"I have a lucky coin!"', 'green_shirt': '"I\'m wearing a green shirt."',
              'dog_owner': '"I have a dog!"', 'time_of_day': '"It\'s after 10pm."'}
    groups = list(t['target_model'].unique())
    values = {(r.target_model, r.prompt_type): r.asr for r in t.itertuples()}

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    grouped_bars(ax, groups, series, values, colors, labels, width=0.85, label_size=7)
    percent_axis(ax)
    ax.set_ylabel('Attack Success Rate (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=3, fontsize=7.5)
    fig.savefig(PLOTS / 'user-eval-result-split.png')
    plt.close(fig)
    return t


def plot_4_destyled(df):
    """Does the forged CoT still work once its distinctive style is removed?"""
    t = asr(df, ['policy_style'])
    order = ['no_policy', 'base', 'destyled']
    names = {'no_policy': 'Harmful query', 'base': 'CoT Forgery', 'destyled': 'Destyled CoT Forgery'}
    colors = {'no_policy': '#90a1b9', 'base': '#fd9a00', 'destyled': '#a684ff'}
    t = t.set_index('policy_style').reindex([o for o in order if o in set(t['policy_style'])]).reset_index()

    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ys = range(len(t))
    ax.barh(list(ys), t['asr'], color=[colors[p] for p in t['policy_style']], height=0.68, zorder=3)
    for y, v in zip(ys, t['asr']):
        ax.text(v + 0.012, y, f'{v:.1%}', va='center', fontsize=8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([names[p] for p in t['policy_style']])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Attack Success Rate (%)')
    fig.savefig(CHAT_DIR / 'plots' / 'user-eval-result-styled-vs-destyled.png')
    plt.close(fig)
    return t


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f'{len(df)} classified generations, {df["harmful_question_ix"].nunique()} harmful questions')

    tables = {
        'basic': plot_1_basic(df),
        'vs_published': plot_2_vs_published(df),
        'by_qualifier': plot_3_by_qualifier(df),
        'styled_vs_destyled': plot_4_destyled(df),
    }
    for name, t in tables.items():
        print(f'\n== {name} ==')
        print(t.to_string(index=False))

    pd.concat([t.assign(plot=name) for name, t in tables.items()]).to_csv(
        PLOTS / 'asr-summary.csv', index=False)
    print(f'\nplots + asr-summary.csv -> {PLOTS}')


if __name__ == '__main__':
    main()
