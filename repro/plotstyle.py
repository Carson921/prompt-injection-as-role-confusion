"""Matplotlib approximation of the paper's ggplot theme (r-utils/plots.r::theme_iclr)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 10,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.6,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#d9d9d9',
    'grid.linewidth': 0.4,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'legend.frameon': False,
})


def grouped_bars(ax, groups, series, values, colors, labels, width=0.8, fmt='{:.1%}', label_size=7):
    """
    groups : x categories
    series : bar keys within each group
    values : dict[(group, serie)] -> value or None
    """
    n = len(series)
    slot = width / n
    for i, s in enumerate(series):
        xs, ys = [], []
        for g_ix, g in enumerate(groups):
            v = values.get((g, s))
            if v is None:
                continue
            xs.append(g_ix - width / 2 + slot * (i + 0.5))
            ys.append(v)
        ax.bar(xs, ys, width=slot * 0.95, color=colors[s], label=labels[s], zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, fmt.format(y), ha='center', va='bottom', fontsize=label_size, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontweight='bold')
    ax.grid(axis='x', visible=False)


def percent_axis(ax, top=1.0):
    ax.set_ylim(0, top)
    ax.set_yticks([t / 100 for t in range(0, int(top * 100) + 1, 25 if top > 0.5 else 10)])
    ax.set_yticklabels([f'{int(t * 100)}%' for t in ax.get_yticks()])
