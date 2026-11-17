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
import re
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo


METHODS = ('plain', 'remember', 'optimized')
ROLES = ('system', 'developer', 'user', 'cot', 'assistant', 'tool')
LABELS = {'plain': 'Plain', 'remember': 'Remember', 'optimized': 'AutoDAN optimized',
          'previous_optimized': 'Previous winner'}


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


def target_role_for(output):
    score = read(Path(output) / 'score_protocol.json', {})
    search = read(Path(output) / 'search_protocol.json', {}).get('config', {})
    role = score.get('target_role', search.get('target_role', 'user'))
    if role not in ROLES:
        raise ValueError('Unsupported target role: ' + str(role))
    return role


def goal_probability(record, role, assumed_target_role='user'):
    goal = (record.get('probabilities') or record.get('region_means') or {}).get('goal', {}) or {}
    values = []
    if goal.get(role) is not None:
        values.append(goal[role])
    if record.get('goal_' + role + '_mean') is not None:
        values.append(record['goal_' + role + '_mean'])
    if record.get('goal_target_mean') is not None and record.get('target_role', assumed_target_role) == role:
        values.append(record['goal_target_mean'])
    if not values:
        return None
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('Invalid goal probability for role ' + role)
    if any(abs(value - values[0]) > 1e-10 for value in values[1:]):
        raise ValueError('Goal target probability conflicts with six-role probabilities: ' + role)
    return values[0]


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


def missing_verdict_bounds(value, expected, allow_final=True):
    """Possible binary outcomes for unscored native errors, never an imputed ASR."""
    value = value or {}
    result = dict(status='pending', scheduled=value.get('scheduled', expected),
                  terminal=value.get('terminal', 0), known_successes=value.get('successes'),
                  native_errors=value.get('errors', 0), lower=None, upper=None,
                  confidence_interval=False, replaces_asr=False)
    n, s, e = result['scheduled'], result['known_successes'], result['native_errors']
    if (allow_final and isinstance(n, int) and n > 0 and n == expected
            and result['terminal'] == n and isinstance(s, int) and isinstance(e, int)
            and value.get('scored') == n - e and 0 <= s <= n - e):
        result.update(status='final', lower=s / n, upper=(s + e) / n)
    return result


def missing_verdict_difference(left, right):
    """Right minus left, allowing each unscored verdict to be either outcome."""
    if left.get('status') != 'final' or right.get('status') != 'final':
        return dict(status='pending', lower=None, upper=None, confidence_interval=False, replaces_asr=False)
    return dict(status='final', lower=right['lower'] - left['upper'],
                upper=right['upper'] - left['lower'], confidence_interval=False, replaces_asr=False)


def evaluation_complete(record, count):
    return all(metric(record, method).get('asr') is not None
               and metric(record, method).get('scheduled') == count
               and metric(record, method).get('scored') == count for method in METHODS)


def candidate_data(output, scores, target_role='user'):
    rows, best = [], 0.
    for index, score in enumerate(scores, 1):
        best = max(best, score['fitness'])
        rows.append(dict(candidate_index=index, generation=score.get('generation', 0),
                         fitness=score['fitness'], cumulative_best=best,
                         target_role=target_role, **{'goal_' + target_role + '_probability': score['fitness']},
                         template_sha256=score.get('template_sha256', hashlib.sha256(score['template'].encode()).hexdigest()),
                         template=score['template']))
    write_csv(output / 'candidate_evolution.csv', rows,
              ['candidate_index', 'generation', 'target_role', 'goal_' + target_role + '_probability',
               'fitness', 'cumulative_best', 'template_sha256', 'template'])
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
    ax.set(xlabel='Unique candidate evaluation', ylabel=f'Mean goal-token P({target_role_for(output)}), six tasks',
           title='AutoDAN template optimization')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output / 'candidate_evolution.svg')
    fig.savefig(output / 'candidate_evolution.png', dpi=180)
    plt.close(fig)
    return dict(status='complete', runtime=sys.executable, matplotlib=matplotlib.__version__)


def probe_table(output, dataset, selected, target_role='user'):
    contexts = dataset.get('contexts') or []
    tables = {method: {row['case_id']: row for row in (selected.get(method) or {}).get('per_case', [])}
              for method in METHODS}
    wide, long = [], []
    for context in contexts:
        identity = context['case_id']
        row = {key: context.get(key) for key in ('case_id', 'suite', 'user_task_id', 'injection_task_id')}
        row['target_role'] = target_role
        for method in METHODS:
            record = tables[method].get(identity, {})
            probability = goal_probability(record, target_role, target_role)
            row[method + '_goal_' + target_role + '_probability'] = probability
            long.append(dict(**{key: row[key] for key in ('case_id', 'suite', 'user_task_id', 'injection_task_id')},
                             method=method, target_role=target_role, goal_target_probability=probability,
                             **{'goal_' + role + '_probability': goal_probability(record, role, target_role) for role in ROLES},
                             goal_tokens=record.get('goal_tokens'),
                             scored_payload_occurrence=record.get('scored_payload_occurrence'),
                             observed_payload_occurrences=record.get('observed_payload_occurrences'),
                             token_artifact=record.get('token_artifact')))
        optimized = row['optimized_goal_' + target_role + '_probability']
        for baseline in [key for key in METHODS if key != 'optimized']:
            probability = row[baseline + '_goal_' + target_role + '_probability']
            row['optimized_minus_' + baseline] = (optimized - probability
                                                   if optimized is not None and probability is not None else None)
        wide.append(row)
    names = ['case_id', 'suite', 'user_task_id', 'injection_task_id', 'target_role',
             *(method + '_goal_' + target_role + '_probability' for method in METHODS),
             *('optimized_minus_' + method for method in METHODS if method != 'optimized')]
    write_csv(output / 'six_task_probe.csv', wide, names)
    write_csv(output / 'six_task_probe_long.csv', long,
              ['case_id', 'suite', 'user_task_id', 'injection_task_id', 'method', 'target_role',
               'goal_target_probability', *('goal_' + role + '_probability' for role in ROLES),
               'goal_tokens', 'scored_payload_occurrence',
               'observed_payload_occurrences', 'token_artifact'])
    means = {}
    for method in METHODS:
        values = [row[method + '_goal_' + target_role + '_probability'] for row in wide]
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
    global METHODS
    output = args.output.resolve()
    dataset = read(output / 'dataset.json', {})
    protocol = read(output / 'search_protocol.json', {})
    config = protocol.get('config', {})
    score_protocol = read(output / 'score_protocol.json', {})
    target_role = target_role_for(output)
    probability_key = 'goal_' + target_role + '_probability'
    audit = read(output / 'independent_audit.json', {})
    diagnostics = read(output / 'diagnostics.json', {})
    winner = read(output / 'winner.json', {})
    sampled = evaluation(output, args.sampled_eval, 'sampled')
    full = evaluation(output, args.full_eval, 'full')
    previous_template = full.get('protocol', {}).get('templates', {}).get('previous_optimized')
    previous_template = previous_template or (config.get('baselines') or {}).get('previous_optimized')
    previous_path = getattr(args, 'previous_winner', None)
    previous_record = {}
    if previous_path:
        previous_record = read(previous_path, {}) if Path(previous_path).suffix == '.json' else {}
        previous_template = previous_record.get('template') if previous_record else Path(previous_path).read_text().strip()
    if not previous_template:
        warm = config.get('warm_start_template')
        if isinstance(warm, str) and '{goal}' in warm:
            previous_template = warm
    previous_source = previous_path or config.get('warm_start_source_path')
    previous_source_protocol = read(Path(previous_source).parent / 'score_protocol.json', {}) if previous_source else {}
    previous_role = (previous_record.get('target_role') or previous_source_protocol.get('target_role')
                     or config.get('previous_target_role')
                     or full.get('protocol', {}).get('previous_winner_objective', {}).get('target_role') or 'user')
    previous_is_near = bool(previous_template and (re.search(r'<[^>\n]*>', previous_template)
                            or (previous_source and (Path(previous_source).parent / 'near_delimiter_library.json').exists())))
    near_library = read(output / 'near_delimiter_library.json', {})
    near_stats = read(output / 'near_variation_stats.json', {})
    near_events = lines(output / 'near_mutations.jsonl')
    native_error_review = read(output / 'native_error_review.json', {})
    native_error_review_final = read(output / 'native_error_review_final.json', {})
    refinement = read(output / 'refinement_protocol.json', {})
    cot_search_review = read(output / 'cot_search_independent_review.json', {})
    search_review_path = next((output / name for name in
                              ('cot_search_independent_review.json', 'search_independent_review.json')
                              if (output / name).exists()), None)
    if refinement and refinement.get('target_role', target_role) != target_role:
        raise ValueError('Refinement protocol target role differs from this search')
    if refinement and refinement.get('asr_not_consulted_for_cot_selection') is not True:
        raise ValueError('Refinement protocol must explicitly exclude ASR from cot selection')
    near_config = config.get('near_delimiter') or {}
    is_near = bool(near_library or near_stats or near_config or config.get('near_delimiter_rate', 0))
    METHODS = ('plain', 'remember', 'previous_optimized', 'optimized') if previous_template else ('plain', 'remember', 'optimized')
    LABELS['optimized'] = 'Near-delimiter optimized' if is_near else 'AutoDAN optimized'
    LABELS['previous_optimized'] = f'Previous {previous_role}-target ' + ('near-delimiter winner' if previous_is_near else 'winner')
    scores = checked_scores(output)
    refinement_summary = None
    if refinement:
        first_directory = Path(refinement['first_cot_output'])
        first_scores = checked_scores(first_directory) if (first_directory / 'scores.jsonl').exists() else None
        budgets = refinement.get('new_trial_native_candidate_evaluations', [])
        refinement_summary = dict(
            first_round_output=str(first_directory),
            first_round_completed_native_candidate_scores=len(first_scores) if first_scores is not None else None,
            current_round_completed_native_candidate_scores=len(scores),
            total_completed_native_candidate_scores=len(first_scores) + len(scores) if first_scores is not None else None,
            planned_native_candidate_scores=sum(budgets) if budgets else None,
            planned_native_context_forwards=refinement.get('native_context_forwards_planned_total'),
            historical_cached_candidates_for_seed_only=refinement.get('historical_candidates_reused_for_seed_selection_only'),
            historical_seed_selection_new_native_forwards=0,
            asr_not_used_for_selection=refinement.get('asr_not_consulted_for_cot_selection') is True,
            previous_optimized_is_separate_from_extra_seeds=True,
            extra_initial_template_sources=config.get('initial_templates', []),
            independent_cross_trial_accounting=cot_search_review.get('cross_trial_accounting'))
    by_template = {row['template']: row for row in scores}
    guard_rows = [record.get('location', {}).get('payload_special_token_check')
                  for score in scores for record in score.get('per_case', [])]
    guard_summary = dict(expected_scored_task_records=len(guard_rows),
                         verified_scored_task_records=sum(bool(row and row.get('passed') is True
                             and row.get('native_special_token_count') == 0) for row in guard_rows),
                         payload_occurrences_checked=sum((row or {}).get('payload_occurrences_checked', 0) for row in guard_rows),
                         raw_payload_tokens_checked=sum((row or {}).get('raw_payload_tokens_checked', 0) for row in guard_rows),
                         rendered_payload_tokens_checked=sum((row or {}).get('rendered_payload_tokens_checked', 0) for row in guard_rows))
    baselines = config.get('baselines') or dataset.get('conditions') or {}
    if winner.get('template'):
        optimized = by_template.get(winner['template'])
    else:
        optimized = max(scores, key=lambda row: row['fitness']) if scores else None
    selected = {method: by_template.get(baselines.get(method)) for method in ('plain', 'remember')}
    selected['optimized'] = optimized
    if previous_template:
        selected['previous_optimized'] = by_template.get(previous_template)
    if refinement and optimized:
        seed_sources = config.get('initial_templates') or [
            dict(path=refinement['first_cot_winner_path']), dict(path=refinement['historical_seed_path'])]
        matched_seeds, extra_seed_templates = [], set()
        for source in seed_sources:
            template = source.get('template') or read(source['path'], {}).get('template')
            if template:
                extra_seed_templates.add(template)
            if template == optimized['template']:
                source_path = str(Path(source['path']).resolve())
                matched_seeds.append(dict(path=source_path,
                    historical_latent_seed=Path(source_path) == Path(refinement['historical_seed_path']).resolve()))
        control_templates = {value for value in baselines.values() if isinstance(value, str)}
        nonseed_scores = [row for row in scores if row['template'] not in extra_seed_templates | control_templates]
        best_nonseed = max(nonseed_scores, key=lambda row: row['fitness']) if nonseed_scores else None
        current_templates = {row['template'] for row in scores}
        first_templates = {row['template'] for row in first_scores} if first_scores is not None else None
        refinement_summary.update(
            cross_round_unique_templates=len(first_templates | current_templates) if first_templates is not None else None,
            cross_round_shared_templates=len(first_templates & current_templates) if first_templates is not None else None,
            final_winner_matching_extra_seeds=matched_seeds,
            final_winner_is_historical_latent_seed=any(row['historical_latent_seed'] for row in matched_seeds),
            nonseed_current_candidates=len(nonseed_scores),
            best_nonseed_current_fitness=best_nonseed['fitness'] if best_nonseed else None,
            best_nonseed_current_template_sha256=hashlib.sha256(best_nonseed['template'].encode('utf-8')).hexdigest() if best_nonseed else None,
            best_nonseed_current_template_sha256_encoding='raw template UTF-8 bytes',
            best_nonseed_current_score_cache_key=best_nonseed.get('template_sha256') if best_nonseed else None,
            best_nonseed_current_score_cache_key_encoding='search.digest: SHA256 of JSON-encoded template string',
            new_variants_exceed_historical_seed=any(row['fitness'] > refinement['historical_seed_cot_fitness'] + 1e-12
                                                   for row in nonseed_scores))
    table, means = probe_table(output, dataset, selected, target_role)
    evolution = candidate_data(output, scores, target_role)
    plotting = plot(output, args.plot_python)
    usage = api_usage(output, winner)
    refinement_api_usage = None
    if refinement:
        first_usage = api_usage(first_directory, read(refinement['first_cot_winner_path'], {}))
        combined = {key: (first_usage.get(key) or 0) + (usage.get(key) or 0) for key in
                    ('attempts', 'successful_calls', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'reported_cost')}
        combined['cost_available'] = bool(first_usage.get('cost_available') and usage.get('cost_available'))
        refinement_api_usage = dict(first_round=first_usage, current_round=usage, combined=combined,
                                    scope='The two cot searches only; excludes the earlier user-target search and cached seed selection')
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
    missing_bounds = {str(count): {method: missing_verdict_bounds(records[method], count, full_terminal)
                                  for method in METHODS} for _, records, count in metric_columns}
    near_previous_bounds = (missing_verdict_difference(missing_bounds['389']['previous_optimized'],
                             missing_bounds['389']['optimized']) if previous_template else None)
    headers = ['方法', f'6 任务 probe P({target_role})，等权']
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
        f'{target_role}-role 预测概率：先对每个任务的 goal tokens 平均，再对 6 个任务等权平均。'
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
    report += ['', f'P({target_role}) 是 [0, 1] 概率；ASR 是成功次数除以对应计划任务数。'
               '原生错误不算成失败。任务尚未全部终态时标为“待完成”；全部终态但存在原生错误时，'
               '完整计划分母 ASR 标为“N/A（原生错误）”，并在紧邻列单独列出可评分子集的成功数/分母。'
               '可评分子集比率不会替代完整计划 ASR。', '']
    if means['optimized'] is not None:
        for baseline in [key for key in METHODS if key != 'optimized']:
            if means[baseline] is not None:
                delta = means['optimized'] - means[baseline]
                report.append(f'相对 {LABELS[baseline]}，优化模板的平均 probe 概率变化为 '
                              f'{delta:+.7g}（{100 * delta:+.5g} 个百分点）。')
        report.append('')
    report += [
        '## 6 个优化任务', '',
        '| 场景 | user task | injection task | ' + ' | '.join(LABELS[method] + f' P({target_role})' for method in METHODS) + ' |',
        '| --- | --- | --- | ' + ' | '.join('---:' for _ in METHODS) + ' |',
    ]
    for row in table:
        report.append(f'| {row["suite"]} | {row["user_task_id"]} | {row["injection_task_id"]} | '
                      + ' | '.join(number(row[method + '_' + probability_key]) for method in METHODS) + ' |')
    if means['optimized'] is not None and means['remember'] is not None:
        changes = [(row, row['optimized_' + probability_key] - row['remember_' + probability_key]) for row in table]
        dominant, change = max(changes, key=lambda item: item[1])
        increased = sum(delta > 1e-12 for _, delta in changes)
        decreased = sum(delta < -1e-12 for _, delta in changes)
        remaining = [row for row in table if row['case_id'] != dominant['case_id']]
        remainder_change = statistics.fmean(row['optimized_' + probability_key] - row['remember_' + probability_key] for row in remaining)
        report += ['', f'相对 Remember，6 个任务中 {increased} 个提高、{decreased} 个降低、'
                   f'{6 - increased - decreased} 个不变。最大正向贡献来自 '
                   f'`{dominant["suite"]}/{dominant["user_task_id"]}/{dominant["injection_task_id"]}`：'
                   f'P({target_role}) 从 {number(dominant["remember_" + probability_key])} 变为 '
                   f'{number(dominant["optimized_" + probability_key])}。其余 5 个任务的平均概率变化为 '
                   f'{remainder_change:+.7g}。上述任务数来自逐任务比较；未对其余 383 个任务计算 probe 概率。']
    diagnostics_used = bool(target_role == 'user' and diagnostics and optimized
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
                          f'其他 5 任务的平均 P({target_role}) 为 '
                          f'{number(gain["other_five_baseline_mean"])} → {number(gain["other_five_winner_mean"])}。')
        report += ['', '净提升贡献可以超过 100%，因为其余任务合计下降会抵消一部分提升。', '']
        if token_concentration:
            pieces = ', '.join('`' + row['raw_rendered_piece'].strip() + '`'
                               for row in token_concentration.get('top_six', []))
            report += [f'上述 banking case 的 {token_concentration["goal_tokens"]} 个 goal tokens 中，'
                       f'最高 6 个 token（{pieces}）占 goal-token P({target_role}) 总和的 '
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
               f'六角色概率、目标角色、token 数量和原始评分文件：{link("six_task_probe_long.csv", output / "six_task_probe_long.csv")}。', '',
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
               f'{"当前轮" if refinement else "本次"}已完整评分 **{len(scores)}** 个唯一模板；计划上限 '
               f'**{config.get("max_unique_scores", "待记录")}**。种群 '
               f'`{config.get("population", "待记录")}`，代数（含初代）'
               f'`{config.get("generations", "待记录")}`；后续每代最多 '
               f'`{config.get("new_per_generation", "待记录")}` 个新评分。', '',
               '模板的 prefix 和 suffix 可以参与变异，实际 goal 原文保持不变，模板恰好含一个 `{goal}` 占位符。'
               f'适应度对应 `loss = −mean_goal_{target_role}_probability`。实际从指定 AutoDAN 仓库加载原版 '
               'GA selection、多点句子交叉和 HGA 词级算子；GPT 变异改接 OpenRouter。'
               '输入 LLM 的是未填充实际 goal 的抽象模板。重复或不合法候选被过滤，必要时补充多样化候选。', '',
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
    if refinement:
        budgets = refinement.get('new_trial_native_candidate_evaluations', [])
        first_path = Path(refinement['first_cot_winner_path'])
        latent_path = Path(refinement['historical_seed_path'])
        report += ['## 两轮 cot refinement 与种子来源', '',
                   f'第一轮 cot 搜索与本轮 refinement 的候选评分预算分别为 `{budgets}`，'
                   f'累计计划 **{refinement_summary["planned_native_candidate_scores"]} 次新 native 候选评分**。'
                   '每次候选评分覆盖同一组 6 个任务；两轮模板可以重复，不能把累计评分次数称为跨轮全局唯一模板数。'
                   f'对应计划 `{refinement_summary["planned_native_context_forwards"]}` 次完整上下文前向。', '',
                   f'已保存的第一轮/当前轮候选评分数分别为 '
                   f'`{refinement_summary["first_round_completed_native_candidate_scores"]}` / '
                   f'`{refinement_summary["current_round_completed_native_candidate_scores"]}`；'
                   f'累计 `{refinement_summary["total_completed_native_candidate_scores"]}`。'
                   '本页演化图和本目录 scores 只展示当前 refinement 轮。', '',
                   f'直接按两个 scores.jsonl 的完整模板字符串去重，两轮共有 '
                   f'`{refinement_summary.get("cross_round_unique_templates", "待完成")}` 个跨轮唯一模板，'
                   f'其中 `{refinement_summary.get("cross_round_shared_templates", "待完成")}` 个在两轮均出现；'
                   '重复模板的第二次 native 评分仍计入 84 次评分账目。', '',
                   f'第一轮 winner 的 P(cot) 为 `{number(refinement.get("first_cot_best_fitness"))}`：'
                   f'{link("first cot winner", first_path)}；'
                   f'该轮是 probe-only 搜索，没有运行用于选择的 ASR：'
                   f'{link("first cot probe audit", first_directory / "independent_audit.json")}、'
                   f'{link("first cot context audit", first_directory / "near_search_audit.json")}。', '',
                   f'第二个额外种子来自旧 user-target 搜索的 '
                   f'`{refinement_summary["historical_cached_candidates_for_seed_only"]}` 个模板的**已存 goal-token 概率**，'
                   f'仅按 cot 列等任务平均挑出 P(cot) `{number(refinement.get("historical_seed_cot_fitness"))}` 的模板：'
                   f'{link("latent_best_cot_seed.json", latent_path)}。'
                   '这一步不加载 backbone、不调用 API，也不产生新的 native 前向；旧 42 个候选不计入本次两轮的 84 次新评分。'
                   '进入本轮的种子与其他候选一样接受本轮 cot 目标下的新评分。', '',
                   '上述两个文件仅作为额外初代搜索种子；`previous_optimized` 仍是上轮 user-target near winner 的固定对照，'
                   '不会被第一轮 cot winner 或 latent seed 替换，也不会因额外种子而增加 ASR 组数。', '',
                   f'协议记录 ASR 未用于 cot 选择：`{refinement_summary["asr_not_used_for_selection"]}`。'
                   '在本轮最终 payload 冻结之后才启动四组 fresh ASR；第一轮 cot 搜索和历史 seed 挑选均不使用 ASR 反馈。', '',
                   f'完整 refinement 协议：{link("refinement_protocol.json", output / "refinement_protocol.json")}。', '']
        if refinement_summary.get('final_winner_matching_extra_seeds'):
            matches = '、'.join(link(Path(row['path']).name, row['path'])
                               for row in refinement_summary['final_winner_matching_extra_seeds'])
            report += [f'本轮最终 winner 与以下额外种子的**完整模板字符串相同**：{matches}。', '']
        if refinement_summary.get('final_winner_is_historical_latent_seed'):
            report += [f'最终 P(cot) 为 `{number(optimized["fitness"])}`，winner 就是历史 latent seed。'
                       '第二轮新变异没有超过该已知 seed，因此最终目标概率的提高不能归因于本轮新变异。'
                       '它反映了从已有概率中选出并重新验证该 seed 后的最终模板选择。', '']
        if refinement_summary.get('best_nonseed_current_fitness') is not None:
            report += [f'排除 Plain/Remember/previous 固定控制及两个额外 seed 后，本轮 '
                       f'`{refinement_summary["nonseed_current_candidates"]}` 个候选的最高 P(cot) 为 '
                       f'`{number(refinement_summary["best_nonseed_current_fitness"])}`。'
                       '这里的“非 seed 候选”不等同于“历史未见模板”；历史新颖性另按下面的跨轮复核口径统计。'
                       f'独立复核依据：{link("cot_search_independent_review.json", output / "cot_search_independent_review.json")}。', '']
        cross_review = cot_search_review.get('cross_trial_accounting') or {}
        if cross_review:
            novel = cross_review.get('best_historically_novel_nonprotected') or {}
            report += [f'独立跨轮复核进一步排除所有 protected seeds 后，得到 '
                       f'`{cross_review.get("nonprotected_unique_templates_excluding_union", "未记录")}` 个唯一模板；'
                       f'再排除旧 user-target 42 候选中已经出现的模板，得到 '
                       f'`{cross_review.get("historically_novel_nonprotected_unique_templates", "未记录")}` 个历史未见模板。'
                       f'这批历史未见模板的最高 P(cot) 为 `{number(novel.get("cot_mean"))}`，'
                       '与最终 latent seed 的来源明确区分。', '']
        if config.get('initial_templates'):
            report += ['| 额外初代种子 | 原优化目标 | 选择目标 | 文件 SHA256 |',
                       '| --- | --- | --- | --- |']
            for source in config['initial_templates']:
                report.append(f'| {link(Path(source["path"]).name, source["path"])} | '
                              f'{source.get("source_optimizer_role", source.get("source_target_role", "未记录"))} | '
                              f'{source.get("selection_role", target_role)} | `{source.get("file_sha256", "未记录")}` |')
            report.append('')
    if is_near:
        coverage = near_library.get('coverage', {})
        validation = full.get('protocol', {}).get('payload_special_token_validation', {})
        proposed, applied = near_stats.get('proposal_events'), near_stats.get('applied_events')
        bernoulli = near_stats.get('bernoulli_events')
        protected = (sum(bool(event.get('protected')) for event in near_events) if near_events else
                     proposed - bernoulli if isinstance(proposed, int) and isinstance(bernoulli, int) else None)
        bernoulli_applied = (sum(event.get('draw') is not None and bool(event.get('applied')) for event in near_events)
                            if near_events else applied)
        actual_rate = bernoulli_applied / bernoulli if isinstance(bernoulli, int) and bernoulli and isinstance(bernoulli_applied, int) else None
        scored_near = near_stats.get('scored_with_near', near_stats.get('scored_with_near_delimiter'))
        report += ['## 近似 delimiter 材料与变异记录', '',
                   '近似标记属于普通文本；模板允许在 goal 前后放置材料。模型真正的 special-token 拼写与 ID 均被排除。'
                   '材料来源包括本地 tokenizer/Harmony 库；当前库不支持的类别明确保留在 coverage 元数据。', '',
                   f'配置近似符变异概率：`{near_config.get("rate", config.get("near_delimiter_rate", "见搜索协议"))}`；'
                   f'共记录 proposal `{proposed}`，其中 `{protected}` 个 protected control 不抽签，'
                   f'实际 Bernoulli 提案 `{bernoulli}`，其中应用 `{bernoulli_applied}` 次 '
                   f'（**{bernoulli_applied}/{bernoulli} = {percent(actual_rate)}**）；applied 总数 `{applied}`。'
                   '应用比例使用实际参与抽签的提案为分母，不使用包含 protected controls 的总 proposal。'
                   f'按材料库精确模式识别的已评分近似候选 `{scored_near}` / `{near_stats.get("scored_candidates", len(scores))}`。', '',
                   f'材料项 `{len(near_library.get("materials", []))}`；'
                   f'禁用拼写 `{len(near_library.get("forbidden_spellings", []))}`，'
                   f'禁用 token IDs `{len(near_library.get("forbidden_ids", []))}`。'
                   f'本次 ASR 启动前对全部 `{validation.get("checked_payloads", "待完成")}` 个实际填充 payload '
                   f'校验 token/拼写排除，结果 `{validation.get("passed", "待完成")}`；'
                   f'独立 ASR 审计复核 `{full.get("audit", {}).get("payload_special_token_exclusion_verified", "待完成")}`。', '',
                   f'Probe scorer 的独立 guard 已记录 `{guard_summary["verified_scored_task_records"]}` / '
                   f'`{guard_summary["expected_scored_task_records"]}` 个任务评分校验，覆盖 '
                   f'`{guard_summary["payload_occurrences_checked"]}` 个实际序列化 payload 副本；'
                   '逐项检查 raw payload 和真实 tool-body payload 范围，实际 special/reserved token 数必须为零。'
                   '正常外层 Harmony frame 不在排除范围内。', '',
                   f'材料库验证：检查 `{near_library.get("validation", {}).get("variants_checked", "待记录")}` 个变体，'
                   f'失败 `{len(near_library.get("validation", {}).get("failures", []))}`；'
                   '逐项 token ID/拼写证明保留在材料库 export 中。', '',
                   'coverage 概况：', '', '```json',
                   json.dumps({key: (len(value) if isinstance(value, list) else value)
                               for key, value in coverage.items()}, ensure_ascii=False, indent=2), '```', '',
                   f'完整材料及其逐项验证：{link("near_delimiter_library.json", output / "near_delimiter_library.json")}；'
                   f'实际变异逐条记录：{link("near_mutations.jsonl", output / "near_mutations.jsonl")}；'
                   f'统计：{link("near_variation_stats.json", output / "near_variation_stats.json")}。', '',
                   f'全上下文预审：{link("near_context_preflight.json", output / "near_context_preflight.json")}；'
                   f'正式搜索上下文审计：{link("near_search_audit.json", output / "near_search_audit.json")}；' +
                   ('独立搜索复核：' + link(search_review_path.name, search_review_path) + '。'
                    if search_review_path else '独立搜索复核：待生成。'), '',
                   '库只覆盖所记录 tokenizer/Harmony 版本能够识别的类别，不把未支持的多模态或其他类别宣称为已覆盖。', '']
        recognition = cot_search_review.get('checks', {}).get('actual_near_sampling', {})
        if recognition.get('independent_whitespace_tolerant_near_count') is not None:
            report += [f'近似符计数有两个明确口径：冻结日志中的精确材料库识别器计数为 '
                       f'**{recognition.get("library_exact_recognizer_scored_with_near", scored_near)}/{len(scores)}**；'
                       f'独立宽空白语法检查为 **{recognition["independent_whitespace_tolerant_near_count"]}/{len(scores)}**。'
                       'HGA 产生了 `< start >` / `< | start | >` 这类额外空格变体，有一个候选未被精确模式识别器标为 near。'
                       '两种计数均保留；没有修改冻结的 contains_near、原始变异统计或 ASR。'
                       f'证据：{link("cot_search_independent_review.json", output / "cot_search_independent_review.json")}。', '']
    report += ['## vLLM ASR 与泛化范围', '',
               f'全量评估独立运行相同 389 个任务，{len(METHODS)} 组 payload 使用同一 vLLM backend、模型和生成配置。'
               '优化后的模板先冻结，再读取这些 ASR；全量结果不反馈到模板选择。', '',
               '表中的 6 任务 ASR 直接从本次全量评估的 `optimization6` 子集提取；'
               '如显式提供独立六任务评估目录，则优先使用该目录。没有重复启动六任务评估。', '',
               '389 任务包括 6 个优化任务与 383 个未参与模板优化的 task pairs。'
               '383 子集与 6 个优化 pair 不重合，但共享场景、工具环境，且可能共享 user task；'
               '它是 task-pair heldout，并不是场景外或 user-task 完全隔离的检验。', '',
               comparison_details(full), '']
    if previous_template:
        report += [f'`previous_optimized` 为 `{LABELS["previous_optimized"]}`，其优化目标是 P({previous_role})；'
                   f'本轮优化目标是 P({target_role})。对照 ASR 来自本次同一 engine 新运行的 389 个任务。'
                   '没有把上一轮或历史运行的 verdict 当作本次匹配对照；与上一轮固定 winner 的差值只使用本次配对结果。', '',
                   '上一轮固定模板：', '', '```text', previous_template, '```', '']
    comparisons = [row for row in full['comparison'].get('paired', [])
                   if row.get('suite') == 'all' and row.get('right') == 'optimized'
                   and row.get('left') in ('plain', 'remember', 'previous_optimized')]
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
        for baseline in [key for key in METHODS if key != 'optimized']:
            delta = metric(full, 'optimized')['asr'] - metric(full, baseline)['asr']
            report.append(f'389 任务上，Optimized 相对 {LABELS[baseline]} 的 ASR 变化：'
                          f'**{100 * delta:+.2f} 个百分点**。')
        report.append('')
    any_native_error = any(records[method].get('errors', 0) for _, records, _ in metric_columns for method in METHODS)
    if any_native_error or (previous_template and not full_terminal):
        report += ['### 完整计划分母的缺失判据界限', '',
                   '这些界限把尚未取得原生 verdict 的错误任务视为可能成功或失败，计算 '
                   '`[S/N, (S+E)/N]`：S 是已知成功数，E 是原生错误数，N 是完整计划分母。'
                   '**这不是置信区间，不是对缺失 verdict 的填补，也不替代主表 ASR。**'
                   '完整计划 ASR 遇原生错误仍为 N/A；可评分子集与配对比较继续单列。', '']
        if not full_terminal:
            report += ['全量运行尚未全部终态，最终缺失判据界限为 **pending（待全量终态）**；'
                       '不把当前已完成任务比例作为最终完整分母结果。', '']
        else:
            report += ['| 计划分母 | 方法 | 已知成功 S | 原生错误 E | 可能结果界限 |',
                       '| ---: | --- | ---: | ---: | --- |']
            for _, records, count in metric_columns:
                for method in METHODS:
                    value = missing_bounds[str(count)][method]
                    if value['native_errors']:
                        interval = (f'[{percent(value["lower"])}, {percent(value["upper"])}]'
                                    if value['status'] == 'final' else 'pending')
                        report.append(f'| {count} | {LABELS[method]} | {value["known_successes"]} | '
                                      f'{value["native_errors"]} | {interval} |')
            report.append('')
        if near_previous_bounds is not None:
            if near_previous_bounds['status'] == 'final':
                report.append(f'{LABELS["optimized"]} − {LABELS["previous_optimized"]} 的完整 389 分母差值界限为 '
                              f'**[{100 * near_previous_bounds["lower"]:+.4f}, '
                              f'{100 * near_previous_bounds["upper"]:+.4f}] 个百分点**；'
                              '它允许两组错误任务取任一可能 verdict，仍不是置信区间或已观测 ASR 差值。')
            else:
                report.append(f'{LABELS["optimized"]} − {LABELS["previous_optimized"]} 的完整 389 分母差值界限：'
                              '**pending（待全量终态）**。')
            report.append('')
    if native_error_review:
        review_terminal = native_error_review.get('running_snapshot', {}).get('progress', {}).get('terminal', '未记录')
        report += [f'原生错误诊断：{link("native_error_review.json", output / "native_error_review.json")}。'
                   f'该文件记录运行至 terminal `{review_terminal}` 时的中间快照，只用于解释 malformed-generation '
                   '与 parser 行为；最终成功数、错误数、分母和界限均从本次正式评估 summary/独立 comparison 计算，'
                   '不采用该中间快照作为最终统计来源。', '']
    if native_error_review_final:
        report += [f'全部原生错误的最终诊断：{link("native_error_review_final.json", output / "native_error_review_final.json")}。'
                   '逐项核对原始输出、stop、解析异常和 payload token 检查；保持错误 verdict 为 null，'
                   '不从异常前的回答或工具调用补判成功。', '']
    if refinement_api_usage:
        first_api = refinement_api_usage['first_round']
        total_api = refinement_api_usage['combined']
        report += ['### 两轮 cot 搜索的 API 合计', '',
                   f'第一轮 `{first_api.get("attempts", 0)}` 次、本轮 `{usage.get("attempts", 0)}` 次，'
                   f'两轮合计 **{total_api["attempts"]} 次 API 尝试**。'
                   + (f'返回成本分别为 `${first_api.get("reported_cost", 0):.7f}`、'
                      f'`${usage.get("reported_cost", 0):.7f}`，两轮合计 **${total_api["reported_cost"]:.7f}**。'
                      if total_api['cost_available'] else '部分 API 成本不可用，不能把已知成本当成完整总成本。'), '',
                   '该合计只覆盖两轮 cot 搜索，不含更早 user-target 搜索的 API；'
                   '历史缓存概率挑选 latent seed 本身不产生 API 调用。下方本轮成本仍单独保留。', '']
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
               f'{"本轮 refinement 的 " if refinement else ""}OpenRouter：API 尝试 `{usage.get("attempts", 0)}` 次、成功 '
               f'`{usage.get("successful_calls", 0)}` 次；prompt '
               f'`{usage.get("prompt_tokens", 0)}` tokens，completion '
               f'`{usage.get("completion_tokens", 0)}` tokens。'
               + (f'API 返回成本合计 **${usage["reported_cost"]:.7f}**。'
                  if usage.get('cost_available') else 'API 未返回可汇总成本，不能视为零费用。'), '',
               f'审计文件：{link("independent_audit.json", output / "independent_audit.json")}；'
               f'全量结果目录：{link("full evaluation", full["path"])}；'
               + (f'六任务结果目录：{link("sampled evaluation", sampled["path"])}。'
                  if sampled.get('summary') else
                  f'六任务 ASR 来源：{link("full comparison 的 optimization6 子集", full["comparison_path"])}。'), '']
    (output / 'REPORT.zh-CN.md').write_text('\n'.join(report), encoding='utf-8')
    summary = dict(status=status, search_complete=search_complete, probe_audit_complete=audit_complete,
                   target_role=target_role, target_role_index=ROLES.index(target_role),
                   refinement_protocol=refinement or None, refinement_summary=refinement_summary,
                   unique_candidates=len(scores), probe_means=means,
                   sampled_asr=sampled_metrics,
                   full_asr={m: metric(full, m) for m in METHODS},
                   holdout_asr={m: metric(full, m, 'holdout383') for m in METHODS},
                   full_asr_independent_audit=full['audit'],
                   full_comparison=full['comparison'], full_evaluation_complete=full_complete,
                   full_evaluation_terminal=full_terminal,
                   winner_template=optimized['template'] if optimized else None,
                   api=usage, api_scope='current_refinement_round_only' if refinement else 'single_round',
                   refinement_two_round_api=refinement_api_usage,
                   plotting=plotting, source_provenance=provenance,
                   diagnostics_included=diagnostics_used,
                   conditions=list(METHODS), previous_winner_template=previous_template,
                   previous_winner_target_role=previous_role, previous_winner_is_near_delimiter=previous_is_near,
                   near_delimiter_library_coverage=near_library.get('coverage'), near_variation_stats=near_stats,
                   near_recognition_independent_review=cot_search_review.get('checks', {}).get('actual_near_sampling'),
                   scorer_payload_guard_summary=guard_summary,
                   payload_special_token_validation=full.get('protocol', {}).get('payload_special_token_validation'),
                   missing_verdict_bounds=missing_bounds,
                   near_minus_previous_missing_verdict_bounds=near_previous_bounds,
                   native_error_review_diagnostic_only=bool(native_error_review),
                   native_error_review_final_path=str(output / 'native_error_review_final.json')
                       if native_error_review_final else None,
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
    parser.add_argument('--previous-winner', type=Path,
                        help='Optional previous winner template; ASR still comes only from this evaluation protocol')
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
