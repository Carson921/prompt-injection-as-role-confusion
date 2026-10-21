#!/usr/bin/env python3
"""Read-only interpretation of terminal frozen DEVELOPMENT tables; never open p/NPZ/test/LM."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/root/prompt-injection-as-role-confusion')
HERE = ROOT / 'experiments/role-analysis'
OUT = HERE / 'outputs/local-gptoss120b/systemness-binary-residual'
RESULT = Path('/tmp/feature_review_binary_residual_interpretation.json')
SOURCE_SHA = 'fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9'
FILES = [OUT / p for p in ['status.json', 'complete.json', 'records.json', 'selection.json',
                          'validation-metrics.csv', 'zero-controls.json', 'protocol.json']]
FILES += [OUT / folder / (arm + '.json') for folder in ['training', 'candidates'] for arm in ['new', 'matched']]
FILES += [HERE / 'systemness_binary_residual.py']

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

assert read(OUT / 'status.json')['stage'] == 'validation_complete'
assert digest(HERE / 'systemness_binary_residual.py') == SOURCE_SHA
provenance = {str(p): digest(p) for p in FILES}
complete = read(OUT / 'complete.json')
assert complete['source_sha256'] == SOURCE_SHA
for p in FILES:
    if p.parent == OUT and p.name in complete['artifact_sha256']:
        assert digest(p) == complete['artifact_sha256'][p.name]
for arm in ['new', 'matched']:
    assert digest(OUT / 'candidates' / (arm + '.json')) == complete['candidate_sha256'][arm]

records = read(OUT / 'records.json')
index = {x['record']: x['validation'] for x in records}
selection = read(OUT / 'selection.json')
assert len(records) == 10 and selection['promoted'] is False and selection['primary_changed'] is False
assert selection['records_sha256'] == digest(OUT / 'records.json')
with (OUT / 'validation-metrics.csv').open(newline='') as stream:
    csv_rows = {(x['record'], x['group']): x for x in csv.DictReader(stream)}
assert len(csv_rows) == 120

KEYS = ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy',
        'system_false_positive_rate', 'accuracy']
CHOSEN = ['new-max33', 'matched-max33', 'oldhead-oldmean', 'oldhead-max33',
          'ensemble-max33', 'naturalCE-oldmean', 'naturalCE-max33']
GROUPS = ['old-long-full', 'old-long-full-tail', 'new-C4-long-full', 'new-C4-long-full-tail', 'old-natural']
summary = {}
for name in CHOSEN:
    v = index[name]
    summary[name] = {'system_threshold': v['system_threshold'], 'min33': v['min33'], 'groups': {}}
    for g in GROUPS:
        m = v['groups'][g]
        for key in KEYS:
            assert math.isclose(float(csv_rows[name, g][key]), m[key], rel_tol=0., abs_tol=2e-14)
        one = {'tokens': m['tokens'], **{key: m[key] for key in KEYS}}
        if g == 'old-natural':
            confusion = m['confusion']; cot = confusion[2]
            one.update(cot_support=sum(cot), cot_correct=cot[2], cot_recall=m['recall']['cot'],
                       cot_to_assistant=cot[3], cot_to_system=cot[0],
                       five_correct=sum(confusion[i][i] for i in range(5)), tool_truth_support=sum(confusion[4]))
            assert one['cot_support'] == 8087 and one['tool_truth_support'] == 0
        summary[name]['groups'][g] = one

def differences(control):
    candidate = index['new-max33']; old = index[control]
    result = {'min33_change': candidate['min33'] - old['min33'], 'group_change_percentage_points': {}}
    for g in GROUPS:
        result['group_change_percentage_points'][g] = {key: 100 * (candidate['groups'][g][key] - old['groups'][g][key]) for key in KEYS}
        if g == 'old-natural':
            result['group_change_percentage_points'][g]['cot_recall'] = 100 * (candidate['groups'][g]['recall']['cot'] - old['groups'][g]['recall']['cot'])
    return result

traces = {}; budgets = {}
for arm in ['new', 'matched']:
    candidate = read(OUT / 'candidates' / (arm + '.json'))
    training = read(OUT / 'training' / (arm + '.json'))
    assert candidate['training_sha256'] == digest(OUT / 'training' / (arm + '.json'))
    budgets[arm] = {key: training[key] for key in ['best_epoch', 'best_steps', 'actual_steps']}
    budgets[arm]['epochs'] = len(training['history'])
    traces[arm] = {}
    for bank, proof in candidate['raw_invariance'].items():
        assert proof['raw_non_system_sha256'] == proof['base_raw_non_system_sha256']
        assert proof['raw_non_system_argmax_changes'] == proof['non_system_to_different_non_system_p_argmax'] == proof['CoT_to_assistant_p_argmax'] == 0
        traces[arm][bank] = {key: proof[key] for key in ['tokens', 'raw_non_system_sha256',
            'base_raw_non_system_sha256', 'raw_non_system_argmax_changes',
            'non_system_to_different_non_system_p_argmax', 'CoT_to_assistant_p_argmax',
            'p_conditional_non_system_argmax_changes', 'p_non_system_tie_rows',
            'delta_min', 'delta_max', 'delta_mean']}
assert budgets['new'] == budgets['matched'] == {'best_epoch': 3, 'best_steps': 471, 'actual_steps': 1727, 'epochs': 11}
minimum = index['new-max33']['min33']
limiting = sorted([{'group': g, 'statistic': k, 'value': m[k]} for g, m in index['new-max33']['groups'].items()
                   for k in ['system_precision', 'system_recall', 'system_balanced_accuracy']], key=lambda x: x['value'])[:3]

report = dict(
    stage='read_only_binary_residual_frozen_development_interpretation',
    independent_results_audit=False, production_or_independent_verifiers_invoked_by_this_script=False,
    probabilities_or_weights_read=False, test_data_or_results_read=False, gpu_used=False,
    formal_outputs_modified=False, source_sha256=SOURCE_SHA,
    provenance_sha256=provenance | {str(Path(__file__).resolve()): digest(__file__)},
    selected_candidate='new-max33', promoted=False, primary_changed=False,
    min33=minimum, shortfall_to_90_percentage_points=100 * (.9 - minimum),
    minimum_statistics=limiting,
    all_ten_record_threshold_and_min33=[{'record': x['record'], 'system_threshold': x['validation']['system_threshold'],
        'min33': x['validation']['min33']} for x in records],
    selected_group_comparisons=summary,
    new_minus_controls={name: differences(name) for name in CHOSEN if name != 'new-max33'},
    exact_training_budget=budgets, raw_invariance_evidence=traces,
    interpretation=[
        'The bounded residual avoids the prior full-CE damage to legacy CoT/five-way discrimination. Versus the original frozen head, the CoT gain is small; this chiefly preserves existing discrimination.',
        'Frozen raw non-system logits and zero observed non-system-to-different-non-system winner swaps do not imply all truth-CoT errors stay unchanged: transitions through the system class remain possible.',
        'At the predeclared global max33 threshold, new-C4 tail system precision and recall are both below90%. This trial does not repair the system separation bottleneck.',
        'Old-full/old-tail F1 regresses versus the original controls, while new-full/new-tail regresses versus the same-budget matched residual and the fixed ensemble. A gain over one weaker operating point is not promotion.',
        'Both arms have identical initialization design, role-slot/update budgets and actual early-stop totals here. Added natural training does not beat its matched control in min33.',
        'These are reused development results, not independent confirmation or confidence bounds. No threshold/model/head/source was reselected.'
        ,'Failure of this fixed residual experiment is not a proof that the frozen features or all possible readouts cannot separate system roles.'
    ],
    next_step_judgment=[
        'Preserve this negative result and all current frozen-primary files; do not launch fresh confirmation for this failed candidate.',
        'Prioritize a prospective fit-only complementary-head signal experiment with an information-free duplicate-head capacity control, or a separately declared causal context readout; the existing two-arm result gives no evidence that another threshold-only adjustment solves the separation gap.',
        'Any future candidate must pass the unchanged development floors and then new disjoint C4/full-tail plus natural confirmation; existing test observations cannot tune its feature, threshold or architecture.'
    ])
assert {str(p): digest(p) for p in FILES} == provenance
RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'report': str(RESULT), 'report_sha256': digest(RESULT), 'script_sha256': digest(__file__),
                  'min33': minimum, 'promoted': False, 'training': budgets, 'limiting': limiting}))
