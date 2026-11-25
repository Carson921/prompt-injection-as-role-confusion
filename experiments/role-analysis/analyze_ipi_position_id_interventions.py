"""Summarize position-id interventions on the sealed IPI Arena no-payload traces.

Reads the readouts written by run_ipi_position_id_interventions.py plus the sealed
baseline / farthest-move summary of outputs/ipi-benign-trace-interventions-20261117.
Per trace the six-class softmax is averaged over the fixed first-role body tokens,
then traces are weighted equally. Intervals are pointwise 95% percentile intervals
from 2,000 paired whole-trace bootstrap draws (seed 20261112, as in the sealed run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_ipi_position_id_interventions as runner

ROLES = list(runner.ROLES)
LABEL = dict(system='System', developer='Developer', user='User', cot='CoT', assistant='Assistant', tool='Tool')
BLOCKS = [f'block{runner.signed(s)}' for s in sorted(runner.BLOCK_SHIFTS)]
GAP, GLOBAL = f'gap{runner.signed(runner.GAP_SHIFT)}', f'global{runner.signed(runner.GLOBAL_SHIFT)}'
TARGET_CONDITIONS = ['move', 'posid_move', 'order_move', *BLOCKS, GAP]
DRAWS, BOOT_SEED = 2000, 20261112


def load(output):
    """One row per (input, read role): mean six-class probability over that role's body."""
    variants, _ = runner.build_variants()
    geometry = {v['input_id']: v for v in variants}
    rows = []
    for input_id, variant in geometry.items():
        with np.load(output / 'readouts' / (input_id + '.npz')) as z:
            positions, probability = z['positions'], z['probabilities'].astype(np.float64)
            assert np.array_equal(positions, variant['target_positions'])
            for role in z['role_names']:
                mask = z['mask_' + str(role)]
                index = np.searchsorted(positions, mask)
                assert np.array_equal(positions[index], mask)
                p = probability[index].mean(0)
                rows.append(dict(trace_id=variant['trace_id'], condition=variant['condition'],
                    source_role=variant['source_role'] or '', read_role=str(role), body_tokens=len(mask),
                    read_body_start=int(variant['role_target_positions'][role][0]),
                    block_start=variant.get('block_start'), block_end=variant.get('block_end'),
                    move_destination=variant.get('move_destination', ''),
                    move_displacement_tokens=variant.get('move_displacement_tokens'),
                    input_tokens=variant['input_tokens'], p_original=p[ROLES.index(role)],
                    **{'p_' + r: p[i] for i, r in enumerate(ROLES)}))
    return pd.DataFrame(rows)


def sealed(traces):
    table = pd.read_csv(runner.SOURCE / 'all/input-summary.csv')
    table = table[table.operation.isin(['baseline', 'move'])]
    assert set(table.trace_id) == set(traces)
    return pd.DataFrame(dict(trace_id=table.trace_id, condition=table.operation,
        source_role=np.where(table.operation == 'baseline', '', table.source_role), read_role=table.source_role,
        body_tokens=table.body_tokens, p_original=table.mean_original_role_probability,
        **{'p_' + r: table['p_' + r] for r in ROLES}))


def matrix(frame, traces, role):
    """traces x 1 vector of P(original role); NaN where the role is absent."""
    return frame[frame.read_role == role].set_index('trace_id').p_original.reindex(traces).to_numpy()


def interval(values, draws):
    means = np.nanmean(values[draws], axis=1)
    return np.nanmean(values), *np.percentile(means, [2.5, 97.5])


def analyze(args):
    out = args.output
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    new = load(out)
    traces = sorted(new.trace_id.unique())
    assert len(traces) == 41
    old = sealed(traces)
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(traces), (DRAWS, len(traces)))

    # Explicit arange ids reproduce the sealed baseline summary.
    check = new[new.condition == 'baseline_arange'].merge(old[old.condition == 'baseline'], on=['trace_id', 'read_role'])
    assert len(check) == 239
    baseline_difference = float(np.abs(check.p_original_x - check.p_original_y).max())
    assert baseline_difference < 1e-9

    target = pd.concat([new[new.source_role == new.read_role], old[old.condition == 'move'].assign(source_role=lambda t: t.read_role)])
    base = old[old.condition == 'baseline']
    effects, per_trace = [], []
    for role in ROLES:
        b = matrix(base, traces, role)
        effects.append(dict(condition='baseline', role=role, n=int(np.isfinite(b).sum()),
            **dict(zip(['mean_pct', 'ci_low_pct', 'ci_high_pct'], 100 * np.array(interval(b, draws)))),
            delta_pp=0.0, delta_ci_low_pp=0.0, delta_ci_high_pp=0.0))
        for condition in TARGET_CONDITIONS:
            v = matrix(target[target.condition == condition], traces, role)
            assert np.array_equal(np.isfinite(v), np.isfinite(b))
            effects.append(dict(condition=condition, role=role, n=int(np.isfinite(v).sum()),
                **dict(zip(['mean_pct', 'ci_low_pct', 'ci_high_pct'], 100 * np.array(interval(v, draws)))),
                **dict(zip(['delta_pp', 'delta_ci_low_pp', 'delta_ci_high_pp'], 100 * np.array(interval(v - b, draws))))))
            per_trace.extend(dict(trace_id=t, role=role, condition=condition, baseline_pct=100 * x, value_pct=100 * y)
                             for t, x, y in zip(traces, b, v) if np.isfinite(x))
        g = matrix(new[new.condition == GLOBAL], traces, role)
        effects.append(dict(condition=GLOBAL, role=role, n=int(np.isfinite(g).sum()),
            **dict(zip(['mean_pct', 'ci_low_pct', 'ci_high_pct'], 100 * np.array(interval(g, draws)))),
            **dict(zip(['delta_pp', 'delta_ci_low_pp', 'delta_ci_high_pp'], 100 * np.array(interval(g - b, draws))))))
    effects = pd.DataFrame(effects)
    effects.to_csv(analysis / 'effects.csv', index=False)
    per_trace = pd.DataFrame(per_trace)
    geometry = new[(new.condition == 'posid_move') & (new.source_role == new.read_role)][
        ['trace_id', 'read_role', 'move_destination', 'move_displacement_tokens', 'block_start', 'block_end', 'body_tokens', 'input_tokens']
    ].rename(columns=dict(read_role='role'))
    per_trace = per_trace.merge(geometry, on=['trace_id', 'role'])
    per_trace.to_csv(analysis / 'per-trace-target-role.csv', index=False)

    # Where the probability mass goes: mean six-class readout of the manipulated body.
    destination = (pd.concat([target, base.assign(condition='baseline')])
                   .groupby(['condition', 'read_role'])[['p_' + r for r in ROLES]].mean() * 100).reset_index()
    destination.to_csv(analysis / 'six-class-destination.csv', index=False)

    # Split by farthest-move direction (start vs end of the trace).
    wide = per_trace.pivot_table(index=['trace_id', 'role', 'move_destination', 'baseline_pct'], columns='condition', values='value_pct').reset_index()
    direction = wide.groupby(['role', 'move_destination']).agg(n=('trace_id', 'size'), baseline=('baseline_pct', 'mean'),
        **{c: (c, 'mean') for c in TARGET_CONDITIONS}).reset_index()
    direction['role'] = pd.Categorical(direction.role, ROLES)
    direction.sort_values(['role', 'move_destination']).to_csv(analysis / 'by-move-direction.csv', index=False)

    # Spillover: bodies of the other first roles when one block's ids are changed.
    other = new[(new.source_role != '') & (new.source_role != new.read_role) & (new.condition != 'order_move')].merge(
        base[['trace_id', 'read_role', 'p_original']].rename(columns=dict(p_original='baseline')), on=['trace_id', 'read_role'])
    other['delta_pp'] = 100 * (other.p_original - other.baseline)
    other['side'] = np.where(other.read_body_start >= other.block_end, 'after_block', 'before_block')
    upstream = other[other.side == 'before_block'].assign(uniform=lambda t: t.condition == 'posid_move')
    upstream_local = float(upstream[~upstream.uniform].delta_pp.abs().max())      # earlier ids untouched: exact
    upstream_uniform = float(upstream[upstream.uniform].delta_pp.abs().max())     # earlier ids shifted together
    spillover = other[other.side == 'after_block'].groupby(['condition', 'source_role', 'read_role']).delta_pp.agg(
        n_traces='size', mean_delta_pp='mean', min_delta_pp='min', max_delta_pp='max').reset_index()
    spillover.to_csv(analysis / 'spillover-downstream-roles.csv', index=False)
    control = new[new.condition == GLOBAL].merge(base[['trace_id', 'read_role', 'p_original']].rename(
        columns=dict(p_original='baseline')), on=['trace_id', 'read_role'])
    control['delta_pp'] = 100 * (control.p_original - control.baseline)
    control[['trace_id', 'read_role', 'delta_pp']].to_csv(analysis / 'global-shift-control-per-trace.csv', index=False)

    # Identities implied by causal attention + relative RoPE.
    pair = target[target.condition == GAP].merge(target[target.condition == 'block+1000'], on=['trace_id', 'read_role'])
    system = target[(target.read_role == 'system') & target.condition.isin(['posid_move', *BLOCKS, GAP])].merge(
        base[base.read_role == 'system'], on='trace_id')
    validation = dict(status='passed', traces=len(traces), inputs=int(new.groupby(['trace_id', 'condition', 'source_role']).ngroups),
        baseline_arange_vs_sealed_summary_max_abs=baseline_difference,
        gap_vs_block_same_shift_target_role_max_abs_pp=float(100 * np.abs(pair.p_original_x - pair.p_original_y).max()),
        system_block_starts_at_zero=bool((new[(new.source_role == 'system')].block_start == 0).all()),
        system_position_only_vs_baseline_max_abs_pp=float(100 * np.abs(system.p_original_x - system.p_original_y).max()),
        upstream_bodies_ids_untouched_max_abs_delta_pp=upstream_local,
        upstream_bodies_uniformly_shifted_max_abs_delta_pp=upstream_uniform,
        global_shift_per_trace_abs_delta_pp=dict(median=float(control.delta_pp.abs().median()),
            p95=float(control.delta_pp.abs().quantile(0.95)), max=float(control.delta_pp.abs().max())),
        global_shift_role_mean_max_abs_delta_pp=float(effects[effects.condition == GLOBAL].delta_pp.abs().max()),
        bootstrap=dict(draws=DRAWS, seed=BOOT_SEED, unit='trace', interval='pointwise 95% percentile'))
    (analysis / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    plot(effects, analysis)
    print(json.dumps(validation, indent=2))
    print(effects.pivot(index='role', columns='condition', values='mean_pct').reindex(ROLES)[['baseline', GLOBAL, *TARGET_CONDITIONS]].round(2).to_string())


def plot(effects, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ink, muted, surface, grid = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
        'xtick.color': muted, 'ytick.color': muted, 'text.color': ink, 'figure.facecolor': surface, 'axes.facecolor': surface,
        'savefig.facecolor': surface, 'svg.fonttype': 'none'})
    table = effects.set_index(['condition', 'role'])

    # Figure 1: 2x2 of token order x position ids.
    series = [('baseline', 'Original order, original ids (baseline)', '#9a9992'),
              ('move', 'Moved order, moved ids (physical move)', '#2a78d6'),
              ('posid_move', 'Original order, moved ids (position ids only)', '#eb6834'),
              ('order_move', 'Moved order, original ids (order only)', '#1baf7a')]
    figure, axis = plt.subplots(figsize=(9.6, 4.2))
    width = 0.19
    for k, (condition, label, color) in enumerate(series):
        x = np.arange(len(ROLES)) + (k - 1.5) * (width + 0.015)
        mean = np.array([table.loc[(condition, r), 'mean_pct'] for r in ROLES])
        low = np.array([table.loc[(condition, r), 'ci_low_pct'] for r in ROLES])
        high = np.array([table.loc[(condition, r), 'ci_high_pct'] for r in ROLES])
        axis.bar(x, mean, width, color=color, label=label, linewidth=0)
        axis.errorbar(x, mean, yerr=[mean - low, high - mean], fmt='none', ecolor=ink, elinewidth=0.8, capsize=1.5)
        for xi, m, h in zip(x, mean, high):
            axis.text(xi, max(m, h) + 1.5, f'{m:.0f}', ha='center', va='bottom', fontsize=7, color=muted)
    axis.set_xticks(np.arange(len(ROLES)), [f'{LABEL[r]}\n(n={int(table.loc[("baseline", r), "n"])})' for r in ROLES])
    axis.set_ylim(0, 112)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel('P(original role) on the moved body, %')
    axis.yaxis.grid(True, color=grid, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8)
    axis.set_title('Farthest move of a full role block: token order vs. position ids\n'
                   'GPT-OSS-120B, 41 clean IPI Arena traces; bars = trace means, whiskers = 95% bootstrap interval',
                   loc='left', fontsize=10)
    figure.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        figure.savefig(analysis / f'order-vs-position-ids.{suffix}', dpi=200)
    plt.close(figure)

    # Figure 2: dose response of shifting only the block's position ids.
    columns = ['baseline', *BLOCKS]
    names = ['baseline\n(no shift)'] + [c.replace('block', 'block ids ').replace('-', '−') for c in BLOCKS]
    values = np.array([[table.loc[(c, r), 'mean_pct'] for c in columns] for r in ROLES])
    ramp = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    axis.imshow(values, cmap=ramp, vmin=0, vmax=100, aspect='auto')
    for i in range(len(ROLES)):
        for j in range(len(columns)):
            axis.text(j, i, f'{values[i, j]:.1f}', ha='center', va='center', fontsize=9,
                      color='#ffffff' if values[i, j] > 45 else ink)
    axis.set_xticks(range(len(columns)), names)
    axis.set_yticks(range(len(ROLES)), [LABEL[r] for r in ROLES])
    axis.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ROLES)), minor=True)
    axis.grid(which='minor', color=surface, linewidth=2)
    axis.tick_params(which='both', length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.set_title('Shifting only the position ids of one role block (token order unchanged)\n'
                   'Cell = mean P(original role) on that block’s body, %; darker = role signal kept', loc='left', fontsize=10)
    figure.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        figure.savefig(analysis / f'block-id-shift-dose.{suffix}', dpi=200)
    plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=runner.DEFAULT_OUTPUT)
    analyze(p.parse_args())


if __name__ == '__main__':
    main()
