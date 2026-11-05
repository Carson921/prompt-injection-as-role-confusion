"""Merge audited existing-score references; never generate or mutate templates."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compile_pool(root, format2_audit_sha256):
    root = Path(root).resolve()
    inputs = {}

    def bound(relative, expected=None):
        path = root / relative
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('An audited source changed: ' + str(path))
        inputs[str(path)] = digest
        return path

    original = bound('probe_short128_supplement/score_records.jsonl',
                     'dc24aa8eaa2d48805bad049021ea5ae87655fe4b6c1d7a4c29094cd930482c27')
    original_audit = read(bound('audit_short128_supplement_final/audit.json',
                     'c435b8eef97f6dfb626ba0f5fde598bd63c6c4e4af144ff630394aa3b140d547'))
    if original_audit['status'] != 'passed':
        raise ValueError('The initial finite short catalog lacks its final audit')
    summary112 = read(bound('readonly_strict_rejection_diagnostic/native_cache_compatibility_v3/summary.json',
                     'da67e18c9b47d7439e50d8f10d9bef7db2f45c0447553d3c6a743125aedbe67d'))
    bound('readonly_strict_rejection_diagnostic/final_report.json',
          'cd0dc7a487fa86b3d40549e924e6e81f2f0098764a92c0e92e369e16638694e1')
    bound('readonly_strict_rejection_diagnostic/native_cache_compatibility_v3/coverage_report.json',
          'd2ab47fa3bf678e5795db323a5f1aa00b515bdb6973dac73f69d3929749c01c8')
    cache112_path = Path(summary112['cache_records_path'])
    if sha(cache112_path) != summary112['cache_records_sha256']:
        raise ValueError('The audited cache inventory changed')
    inputs[str(cache112_path)] = sha(cache112_path)
    cache112 = read(cache112_path)['records']
    cache20 = read(bound('audit_remaining39_independent/cache_coverage.json',
                     '6f3852f2c5e539ef659927ac47824d82a44c694e2c37d3c11b04ab8107de39e4'))
    bound('audit_remaining39_independent/classification.json',
          'a8a0c46cb9c62e71a019532997ee842f47e682b2561a6adacdecd10fd5497a7c')
    diagnostic_path = bound('readonly_strict_rejection_diagnostic/reviewed_literal_index_v2.jsonl',
                     'fc94c6040402ef0bccd84a934b85e3e24bfc06ac11c216f107afef5e1e887569')
    diagnostic = {row['raw_sha256']: row for row in lines(diagnostic_path)}
    for relative, expected in [('audit_format60_final/audit.json',
            'cc2c608ddf02e7f97956fb1249bc1c3688b66757fe746363444b8a774a1ee8d2'),
            ('audit_format2_final/audit.json', format2_audit_sha256)]:
        if read(bound(relative, expected))['status'] != 'passed':
            raise ValueError('A supplemental native score audit is incomplete')
    groups = {'strict_short_archive': lines(original),
              'format60_fresh': lines(bound('probe_format60_short128/score_records.jsonl')),
              'format2_fresh': lines(bound('probe_format2_short128/score_records.jsonl'))}
    grouped112 = {}
    for row in cache112:
        if row['protocol_input_mask_probability_aggregation_compatible'] is True:
            grouped112.setdefault(row['raw_sha256'], []).append(row)
    groups['format112_existing_cache'] = [
        dict(sorted(rows, key=lambda r:r['source_score_path'])[0], raw_sha256=raw,
             wrapper_tokens=diagnostic[raw]['wrapper_tokens'])
        for raw, rows in grouped112.items()]
    groups['format20_existing_cache'] = [dict(
        sorted([r for r in row['records'] if r['compatible_existing_complete_user2_score']],
               key=lambda r:r['source_score_path'])[0], raw_sha256=row['raw_sha256'],
        wrapper_tokens=diagnostic[row['raw_sha256']]['wrapper_tokens'])
        for row in cache20['records'] if row['compatible_existing_full_user2_score']]
    expected_counts = {'strict_short_archive':1586, 'format60_fresh':60, 'format2_fresh':2,
                       'format112_existing_cache':112, 'format20_existing_cache':20}
    if {key:len(rows) for key,rows in groups.items()} != expected_counts:
        raise ValueError('An audited population is missing or changed')
    contexts = read(bound('dataset.json',
                    'ef2f03449bc49f0dfc535c6ec3aacf40236fafb730fcfba7b0026c7c414927af'))['contexts']
    expected_cases = [row['case_id'] for row in contexts]
    records = {}
    for group, rows in groups.items():
        for row in rows:
            path = Path(row['source_score_path'])
            if sha(path) != row['source_score_sha256']:
                raise ValueError('An audited native score file changed')
            score = read(path)
            raw = hashlib.sha256(score['template'].encode()).hexdigest()
            cases = score['per_case']
            if (raw != row['raw_sha256'] or raw in records or score['status'] != 'complete'
                    or score['complete'] is not True or score['target_role'] != 'user'
                    or score['target_role_index'] != 2 or len(cases) != 6
                    or [case['case_id'] for case in cases] != expected_cases
                    or not 0 <= row['wrapper_tokens'] <= 128):
                raise ValueError('The merged score is not a unique complete short user2 score')
            for case, context in zip(cases, contexts):
                if (case['actual_goal_sha256'] != hashlib.sha256(context['actual_goal'].encode()).hexdigest()
                        or case['target_role'] != 'user' or case['target_role_index'] != 2):
                    raise ValueError('The immutable goal changed')
            mean = math.fsum(case['probabilities']['goal']['user'] for case in cases) / 6
            if (not math.isfinite(mean) or not 0 <= mean <= 1
                    or abs(mean - score['fitness']) > 1e-12
                    or abs(mean - score['goal_role_means']['user']) > 1e-12):
                raise ValueError('The saved score is not the equal-six complete-goal user mean')
            records[raw] = dict(raw_sha256=raw, wrapper_tokens=row['wrapper_tokens'],
                p_user=mean, origin_group=group, source_score_path=str(path),
                source_score_sha256=row['source_score_sha256'])
    if len(records) != 1780:
        raise ValueError('Unexpected merged population')
    output = root / 'existing_short_pool'
    output.mkdir(exist_ok=False)
    index = output / 'score_index.jsonl'
    index.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for raw,row in sorted(records.items())))
    maximum = max(row['p_user'] for row in records.values())
    summary = dict(schema='audited-existing-short-score-pool-v1', status='complete',
        unique_complete_candidates=len(records), group_counts=expected_counts,
        fresh_scores_after_latest_short_constraint=957, maximum_wrapper_tokens=128,
        maximum_p_user=maximum,
        maximum_candidates=[r for r in records.values() if abs(r['p_user']-maximum)<1e-12],
        false_rejections_corrected=194, confirmed_repeated_body_or_empty_stacks_excluded=340,
        new_templates_generated_or_modified=False, compiler_started_ASR_runs=0,
        input_bindings=inputs, score_index_path=str(index), score_index_sha256=sha(index),
        limitations=['Finite pre-round archive only; no global optimum or new-mutation search claimed.',
                     'Compatible existing native probabilities were audited and reaggregated; not every cached checkpoint prediction was independently recomputed.'])
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--format2-audit-sha256', required=True)
    args = parser.parse_args()
    result = compile_pool(args.root, args.format2_audit_sha256)
    print(json.dumps({key:result[key] for key in ('status','unique_complete_candidates','maximum_p_user')}))
