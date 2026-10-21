"""Summarize completed dense and paired validation trials without rerunning them.

Default: help. --preflight checks synthetic schemas in CPU memory. --write
creates only validation-report.txt and validation-summary.csv in the existing
systemness-final-readout folder. No activation, model, probability, test or
confirmation data is opened, and no selection or frozen metadata is modified.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / 'outputs/local-gptoss120b'
OUT = BASE / 'systemness-final-readout'
PROFILES = ['single', 'multi', 'natural', 'long-single', 'long-full']
DENSE = ['base-mlp64', 'resampled-mlp64', 'dense-mlp64', 'dense-mlp128', 'dense-linear']
PAIRED = ['lambda-0', 'lambda-0.1', 'lambda-0.5']
TRIALS = {'dense': (BASE / 'systemness-long-full', DENSE),
          'paired': (BASE / 'systemness-paired-pilot', PAIRED)}
FIELDS = ['trial', 'candidate', 'profile', 'tokens', 'system_threshold',
    'mean_five_profile_system_f1', 'original_four_profile_system_f1',
    'original_four_profile_accuracy', 'system_precision', 'system_recall',
    'system_f1', 'system_balanced_accuracy', 'system_false_positive_rate',
    'five_role_accuracy', 'natural_cot_recall', 'best_epoch', 'epochs',
    'best_steps', 'actual_steps', 'instances_per_epoch', 'steps_per_epoch',
    'tail_documents', 'model_sha256', 'source_candidate_sha256']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=1e-12)


def validate_record(record):
    require(record['tests_read'] is False, 'A candidate was not declared validation-only.')
    result = record['validation']
    require(set(result['profiles']) == set(PROFILES), 'Candidate validation profiles differ.')
    require(close(result['mean_system_f1'], sum(result['profiles'][p]['system_f1'] for p in PROFILES) / 5),
            'Five-profile mean does not match its profile metrics.')
    require(close(result['original_mean_system_f1'], sum(result['profiles'][p]['system_f1'] for p in PROFILES[:4]) / 4),
            'Original four-profile F1 mean does not match.')
    require(close(result['original_mean_accuracy'], sum(result['profiles'][p]['accuracy'] for p in PROFILES[:4]) / 4),
            'Original four-profile accuracy mean does not match.')
    for metrics in [*result['profiles'].values(), result['long_full_tail']]:
        require(close(metrics['system_threshold'], result['system_threshold']), 'Per-profile threshold changed.')
        for key in ['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy',
                    'system_false_positive_rate', 'accuracy']:
            value = metrics[key]
            require(value is None or (math.isfinite(value) and 0 <= value <= 1), f'Invalid {key}.')
    require(0 <= result['profiles']['natural']['recall']['cot'] <= 1, 'Missing natural CoT recall.')


def load_completed():
    records = []
    selections = {}
    provenance = {}
    for trial, (folder, candidates) in TRIALS.items():
        selection_path = folder / 'selection.json'
        require(selection_path.is_file(), f'{trial} validation selection is incomplete; wait before writing this report.')
        selection = read(selection_path)
        require(selection['test_used_for_selection'] is False, f'{trial} selection was not validation-only.')
        plan_path = folder / 'search-plan.json'
        require(selection['search_plan_sha256'] == digest(plan_path), f'{trial} plan changed.')
        require(set(selection['candidates']) == set(candidates), f'{trial} candidate membership changed.')
        for candidate in candidates:
            path = folder / 'candidates' / f'{candidate}.json'
            value = read(path)
            require(value['candidate'] == candidate and selection['candidates'][candidate] == digest(path),
                    f'{trial}/{candidate} candidate identity/hash changed.')
            require(value['search_plan_sha256'] == selection['search_plan_sha256']
                    and value['source_sha256'] == selection['source_sha256'], f'{trial}/{candidate} provenance differs.')
            validate_record(value)
            records.append(dict(trial=trial, record=value, source_candidate_sha256=digest(path)))
        chosen = next(item['record'] for item in records
                      if item['trial'] == trial and item['record']['candidate'] == selection['candidate'])
        require(selection['validation'] == chosen['validation'] and selection['model_sha256'] == chosen['model_sha256'],
                f'{trial} selection does not match its chosen candidate.')
        selections[trial] = selection
        provenance[trial] = digest(selection_path)
    long_data = read(BASE / 'systemness-long-full/data-complete.json')
    original = read(BASE / 'systemness-expansion/plan.json')
    cohort = dict(full_fit=long_data['selected_document_counts']['fit'],
        full_validation=long_data['selected_document_counts']['validation'],
        previous_full_fit=long_data['previous_long_single_eligible_documents']['fit'],
        paired_long_validation=long_data['previous_long_single_eligible_documents']['validation'],
        original_validation=original['c4_documents']['validation'],
        natural_validation=original['natural_validation_conversations'])
    require(cohort == dict(full_fit=132, full_validation=31, previous_full_fit=26,
        paired_long_validation=10, original_validation=200, natural_validation=40), 'Document-count protocol changed.')
    final_path = OUT / 'selection.json'
    final = read(final_path) if final_path.exists() else None
    if final is not None:
        require(final['test_used_for_selection'] is False and final['trial_selection_sha256'] == provenance,
                'Final selection does not match these frozen validation trials.')
        chosen = selections[final['originating_trial']]
        require(all(final[key] == chosen[key] for key in ['candidate', 'model_path', 'model_sha256',
                    'system_threshold', 'validation']), 'Final selection differs from its originating trial.')
    return records, selections, cohort, final


def summary_rows(records):
    rows = []
    for item in records:
        value = item['record']; result = value['validation']
        natural_cot = result['profiles']['natural']['recall']['cot']
        for profile, metrics in [*result['profiles'].items(), ('long-full-tail', result['long_full_tail'])]:
            rows.append(dict(trial=item['trial'], candidate=value['candidate'], profile=profile,
                tokens=metrics['tokens'], system_threshold=result['system_threshold'],
                mean_five_profile_system_f1=result['mean_system_f1'],
                original_four_profile_system_f1=result['original_mean_system_f1'],
                original_four_profile_accuracy=result['original_mean_accuracy'],
                system_precision=metrics['system_precision'], system_recall=metrics['system_recall'],
                system_f1=metrics['system_f1'], system_balanced_accuracy=metrics['system_balanced_accuracy'],
                system_false_positive_rate=metrics['system_false_positive_rate'], five_role_accuracy=metrics['accuracy'],
                natural_cot_recall=natural_cot, best_epoch=value.get('best_epoch'), epochs=value.get('epochs'),
                best_steps=value.get('best_steps'), actual_steps=value.get('actual_steps'),
                instances_per_epoch=value.get('instances_per_epoch'), steps_per_epoch=value.get('steps_per_epoch'),
                tail_documents=metrics.get('documents'), model_sha256=value['model_sha256'],
                source_candidate_sha256=item['source_candidate_sha256']))
    return rows


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.2f}'


def number(value):
    return 'N/A' if value is None else str(value)


def render(records, selections, cohort, final):
    rows = summary_rows(records)
    lines = ['GPT-OSS-120B follow-up readout experiments — VALIDATION ONLY', '',
        'Frozen native MXFP4 language model; pointwise attention readouts. Values below are percentages;',
        'differences are percentage points (pp). No independent test/confirmation accuracy is claimed.',
        'This report reads completed candidate JSON only, and does not open the new confirmation cohort.', '',
        f'Full-length coverage: {cohort["full_fit"]} existing fit documents, versus {cohort["previous_full_fit"]} genuinely1024-token documents in the prior sparse long-fit cohort.',
        f'The {cohort["full_validation"]} long-full validation documents are a subset of the original {cohort["original_validation"]} C4 validation documents;',
        f'{cohort["paired_long_validation"]} also have old long-single versions. These are paired variants, not additional independent documents.',
        f'Natural metrics use the unchanged {cohort["natural_validation"]}-conversation validation profile. Tokens and role versions within documents are dependent.', '',
        'Mean5 averages single/multi/natural/long-single/long-full with equal profile weight.',
        'Original4 averages only the first four. Do not compare Mean5 directly with the earlier four-profile 89% mean.',
        'Each candidate has one global threshold calibrated across all five profiles; Original4 uses that same threshold,',
        'so it also differs in calibration protocol from the earlier four-profile search.', '',
        'Candidate overview',
        'trial  candidate              Mean5  Original4 fullF1 tailF1 natural-CoT  best/actual steps']
    for item in records:
        record = item['record']; value = record['validation']
        lines.append(f'{item["trial"]:6s} {record["candidate"]:22s} {pct(value["mean_system_f1"]):>6s} '
            f'{pct(value["original_mean_system_f1"]):>9s} {pct(value["profiles"]["long-full"]["system_f1"]):>6s} '
            f'{pct(value["long_full_tail"]["system_f1"]):>6s} {pct(value["profiles"]["natural"]["recall"]["cot"]):>11s} '
            f'{number(record.get("best_steps"))}/{number(record.get("actual_steps"))}')
    control = next(item['record'] for item in records if item['trial'] == 'dense' and item['record']['candidate'] == 'resampled-mlp64')
    same = next(item['record'] for item in records if item['trial'] == 'dense' and item['record']['candidate'] == 'dense-mlp64')
    selected128 = next(item['record'] for item in records if item['trial'] == 'dense' and item['record']['candidate'] == 'dense-mlp128')
    lines += ['', 'Primary same-architecture comparison: dense-mlp64 versus resampled-mlp64',
        'Both use MLP64, original train-only scaler, seed and per-epoch CE/update budget; dense-mlp128 is a separate architecture search candidate.',
        'profile          resampled-F1 dense64-F1 delta-pp dense128-F1 delta128-pp']
    for profile in [*PROFILES, 'long-full-tail']:
        def metric(record):
            result = record['validation']
            return result['long_full_tail'] if profile == 'long-full-tail' else result['profiles'][profile]
        a = metric(control)['system_f1']; b = metric(same)['system_f1']; c = metric(selected128)['system_f1']
        lines.append(f'{profile:16s} {pct(a):>12s} {pct(b):>10s} {100 * (b-a):>8.2f} {pct(c):>11s} {100 * (c-a):>11.2f}')
    lines += ['', 'Interpretation and limitations',
        'Dense fit replaces4160 old sparse rows with168960 long-full rows:413290-4160+168960=578090.',
        'Resampled control repeats original413290 rows to578090 instances. Both dense and resampled use142 updates/epoch (last batch554).',
        'Coverage, long-text/profile weighting and distinct examples change together; this comparison does not isolate coverage alone.',
        'Independent early stopping/checkpoint selection gives different best and actual update counts, despite matched per-epoch/max budgets.',
        'Base-mlp64 uses101 updates/epoch and is a secondary control. Dense-linear uses L-BFGS; neural update counts are N/A.',
        'Paired candidates add128 five-role groups per step only to the ranking term, preserving578090 CE instances and142 updates/epoch.',
        'The auxiliary term uses long-full fit only; gains cannot separate matched-pair structure from ordinary long-text reweighting.',
        'Same initial weights, CE permutations and isolated auxiliary-dropout RNG are checked by the paired trial.',
        'Repeated validation searches, threshold calibration and early stopping make these development results; independent confirmation remains necessary.',
        'Report natural CoT recall and five-role accuracy alongside systemness so role tradeoffs remain visible.', '',
        'Frozen trial selections']
    for trial in ['dense', 'paired']:
        selection = selections[trial]
        lines.append(f'{trial}: {selection["candidate"]}; promoted={selection["promoted"]}; threshold={selection["system_threshold"]:.12g}')
    if final is None:
        lines += ['', 'Final selection: pending. This report does not choose or modify a model.']
    else:
        lines += ['', f'Final selection: {final["originating_trial"]}/{final["candidate"]}',
            f'Model: {final["model_path"]}', f'Recorded model SHA256: {final["model_sha256"]}',
            f'Threshold: {final["system_threshold"]:.12g}',
            'Final gates require paired matched-control promotion, higher Mean5 than dense, no full/tail F1 regression,',
            'at most2pp F1 loss in each original profile, original four-profile accuracy floor, and each-profile FPR<=5%.',
            'Applied gates: ' + json.dumps(final['comparison_gates'], sort_keys=True),
            'Paired gains versus dense: ' + json.dumps(final['paired_gains_vs_dense'], sort_keys=True)]
    lines += ['', 'Detailed profile metrics (same values are available as full-precision CSV)',
        'trial/candidate/profile                   tokens      P      R     F1   BAcc fiveAcc naturalCoT']
    for row in rows:
        name = f'{row["trial"]}/{row["candidate"]}/{row["profile"]}'
        lines.append(f'{name:42s} {row["tokens"]:7d} {pct(row["system_precision"]):>6s} {pct(row["system_recall"]):>6s} '
            f'{pct(row["system_f1"]):>6s} {pct(row["system_balanced_accuracy"]):>6s} {pct(row["five_role_accuracy"]):>7s} '
            f'{pct(row["natural_cot_recall"]):>10s}')
    lines += ['', 'The naturalCoT column repeats each candidate\'s original natural-profile CoT recall for reference;',
        'it is not the CoT recall of the other listed profiles. CSV best/actual step fields repeat candidate-level values.',
        'Only candidate/plan/selection JSON provenance is checked here; this lightweight report does not recompute probabilities or model hashes.']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(rows)
    return '\n'.join(lines) + '\n', stream.getvalue()


def write():
    records, selections, cohort, final = load_completed()
    text, summary = render(records, selections, cohort, final)
    require(OUT.is_dir(), 'The final-readout folder must already exist; report does not declare a trial.')
    for name, content in [('validation-report.txt', text), ('validation-summary.csv', summary)]:
        path = OUT / name; temporary = OUT / (name + '.tmp')
        temporary.write_text(content); temporary.replace(path)
    print(json.dumps(dict(report=str(OUT / 'validation-report.txt'), summary=str(OUT / 'validation-summary.csv'),
        candidates=len(records), profile_rows=48, validation_only=True, selections_modified=False)))


def preflight():
    def metrics():
        return dict(tokens=100, system_threshold=.6, system_precision=.82, system_recall=.78,
            system_f1=.8, system_balanced_accuracy=.86, system_false_positive_rate=.03,
            accuracy=.84, recall=dict(cot=.55))
    records = []; selections = {}
    for trial, (_, candidates) in TRIALS.items():
        for candidate in candidates:
            result = dict(system_threshold=.6, mean_system_f1=.8, original_mean_system_f1=.8,
                original_mean_accuracy=.84, profiles={profile: metrics() for profile in PROFILES},
                long_full_tail=dict(metrics(), documents=31))
            record = dict(candidate=candidate, tests_read=False, validation=result, model_sha256='synthetic',
                best_steps=None if candidate == 'dense-linear' else 1420,
                actual_steps=None if candidate == 'dense-linear' else 2556)
            validate_record(record)
            records.append(dict(trial=trial, record=record, source_candidate_sha256='synthetic'))
        selections[trial] = dict(candidate=candidates[-1], promoted=False, system_threshold=.6)
    cohort = dict(full_fit=132, previous_full_fit=26, full_validation=31, original_validation=200,
                  paired_long_validation=10, natural_validation=40)
    text, summary = render(records, selections, cohort, None)
    rows = list(csv.DictReader(io.StringIO(summary)))
    require(len(records) == 8 and len(rows) == 48 and len({(row['trial'], row['candidate']) for row in rows}) == 8,
            'Synthetic report schema/count smoke failed.')
    require('dense-mlp64 versus resampled-mlp64' in text and 'Final selection: pending' in text
            and 'Do not compare Mean5' in text and '0.55' == rows[0]['natural_cot_recall'], 'Report warning/metric smoke failed.')
    print(json.dumps(dict(cpu_only=True, synthetic_candidates=8, synthetic_profile_rows=48,
        formal_outputs_written=False, source_sha256=digest(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Synthetic CPU-only report schema smoke; no outputs.')
    action.add_argument('--write', action='store_true', help='Write the two validation reports after both trials finish.')
    args = parser.parse_args()
    if args.write: write()
    elif args.preflight: preflight()
    else: parser.print_help()
