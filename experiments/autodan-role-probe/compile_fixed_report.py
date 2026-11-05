"""Compile verified fixed-candidate ASR evidence; never generate candidate text."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_report(run_root, audit_root, experiment_root, output):
    run_root, audit_root = Path(run_root).resolve(), Path(audit_root).resolve()
    experiment_root, output = Path(experiment_root).resolve(), Path(output).resolve()
    audit = json.loads((audit_root / 'audit.json').read_text())
    if (audit.get('status') != 'verified' or audit.get('audited_episodes') != audit.get('expected_episodes')
            or audit.get('producer_summary_independently_matched') is not True
            or audit.get('audited_native_artifacts_sha_verified') is not True
            or audit.get('all_audited_completed_episodes_official_utility_security_predicates_match') is not True
            or audit.get('native_errors_preserve_null') is not True):
        raise ValueError('A complete independently verified ASR audit is required')
    comparison_path = audit_root / 'comparison.json'
    if sha(comparison_path) != audit['outputs']['comparison.json']:
        raise ValueError('Audited comparison changed')
    comparison = json.loads(comparison_path.read_text())
    protocol_path = run_root / 'protocol.json'
    if sha(protocol_path) != audit['protocol_file_sha256']:
        raise ValueError('The independently audited run protocol changed')
    protocol = json.loads(protocol_path.read_text())
    metadata = {row['condition']: row for row in comparison['candidates']}
    templates = {row['condition']: row for row in protocol['candidates']}
    metrics = {row['condition']: row for row in comparison['conditions']
               if row['cohort'] == 'all' and row['suite'] == 'all'}
    if set(metadata) != set(templates) or set(metadata) != set(metrics):
        raise ValueError('Candidate condition inventories differ')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (condition, candidate) in enumerate(sorted(metadata.items(),
            key=lambda item: item[1]['goal_user_mean'])):
        original = templates[condition]
        text = original['template']
        if hashlib.sha256(text.encode()).hexdigest() != candidate['template_raw_sha256']:
            raise ValueError('Template bytes changed')
        score = metrics[condition]
        if score['scheduled'] != 389 or score['terminal'] != 389:
            raise ValueError('Each candidate requires all389 terminal episodes')
        name = f"{chr(65+index)}_{condition}.txt"
        path = output / 'templates' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        rows.append(dict(label=chr(65+index), condition=condition,
            p_user=candidate['goal_user_mean'], p_user_interval=candidate['p_user_interval'],
            wrapper_tokens=candidate['wrapper_tokens'], successes=score['successes'],
            scheduled=score['scheduled'], scored=score['scored'], unknown=score['unknown'],
            asr=score['asr'], asr_lower_bound=score['asr_lower_bound'],
            asr_upper_bound=score['asr_upper_bound'], scored_asr=score['scored_asr'],
            scored_asr_wilson_95=score['scored_asr_wilson_95'], exposed=score['exposed'],
            template_path=str(path), template_raw_sha256=candidate['template_raw_sha256'],
            native_score_path=original['probe_evidence']['score_path']))
    if sum(row['scored'] for row in rows) != audit['official_predicates_rechecked']:
        raise ValueError('The official verdict replay does not cover every scored episode')
    scope_path = experiment_root / 'scope_status.json'
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    report = dict(schema='verified-fixed-candidate-report-v1', created_unix=time.time(),
        fixed_candidate_ASR_evaluation_complete=True,
        allowed_evaluation_complete=scope.get('all_allowed_work_complete', False),
        full_requested_optimization_complete=False,
        candidates=rows, native_model_calls_replayed=audit['model_calls_replayed'],
        official_predicates_rechecked=audit['official_predicates_rechecked'],
        unknown_policy='Initial errors remain unknown; no retry, replacement, or imputed failure.',
        probe_scope='Equal-six-task mean over all immutable goal tokens in the first original tool occurrence.',
        probe_pairs_are_used_for_candidate_selection=True,
        holdout_scope=comparison['pair_holdout_note'],
        new_seed_generation_API_calls=scope.get('new_wrapper_generation_API_calls'),
        implementation_limitation=scope.get('generation_branch_status'),
        unimplemented_requested_branches=scope.get('not_claimed_completed'),
        supplemental_existing_evaluation=dict(
            format60=scope.get('format60_existing_literal_supplement'),
            format2=scope.get('format2_existing_literal_supplement')),
        selection_coverage=scope.get('selection_coverage'),
        historical_reference_comparison=scope.get('historical_reference_comparison'),
        evidence=dict(run_root=str(run_root),audit_root=str(audit_root),
            audit_sha256=sha(audit_root/'audit.json'),comparison_sha256=sha(comparison_path),
            protocol_sha256=sha(protocol_path),
            scope_status_path=str(scope_path),
            scope_status_sha256=sha(scope_path) if scope_path.exists() else None,
            manifest_sha256=protocol['candidates_manifest_sha256'],
            baseline_reruns=False))
    (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    fields = ['label','condition','p_user','wrapper_tokens','successes','scheduled','scored','unknown',
              'asr','asr_lower_bound','asr_upper_bound','scored_asr','exposed','template_path','template_raw_sha256']
    with (output / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in rows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','audit-root','experiment-root','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = compile_report(args.run_root,args.audit_root,args.experiment_root,args.output)
    print(json.dumps(dict(candidates=len(result['candidates']),
                         verified=True,full_requested_optimization_complete=False)))
