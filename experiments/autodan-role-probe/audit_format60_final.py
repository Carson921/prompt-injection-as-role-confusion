"""Independent terminal CPU audit of the frozen format60 native supplement.

Never constructs a native model or calls forward/score.  The exact allowlist's
extra strict-header exception is retained; all original diverse checks remain.
Only after all 60 new native scores pass does one highest-score literal receive
the already authorized CPU checkpoint recomputation via audit_diverse_native.
Incomplete/error producer outcomes are preserved without retrying any scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from evaluate_static_catalog import pointer_value
from precise_scorer import PreciseScorer
from scorer import load_module

HERE = Path(__file__).resolve().parent
B = HERE / 'outputs/diverse8h-20261105'
PREP = B / 'format60_preparation'
CONTRACT = PREP / 'contract_v2.json'
CONTRACT_SHA = 'e56d08616e68be7f7884dc2307ee34623e3b3063493bc05300f0a3dd942afcb1'
OUT = B / 'audit_format60_final'
COUNT, CAP = 60, 128


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n')


def same_array(actual, expected, name):
    expected = np.asarray(expected)
    require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and actual.tobytes() == expected.tobytes(), name + ' dtype/shape/bytes differ')


def process_live(pid, start_ticks):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return False
    fields = path.read_text().rsplit(')', 1)[1].split()
    return int(fields[19]) == int(start_ticks) and fields[0] != 'Z'


def audit(args, report):
    run = args.run.resolve()
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, 'Frozen file changed: ' + str(path))
        if str(path) in bindings:
            require(bindings[str(path)] == digest, 'Source changed during audit: ' + str(path))
        bindings[str(path)] = digest
        return digest

    bind(CONTRACT, CONTRACT_SHA)
    c = read(CONTRACT)
    require(run == Path(c['output']).resolve(), 'Only the frozen format60 output may be audited')
    require(c['count'] == COUNT and c['caps'] == [CAP] and c['extra_catalog'] == [], 'Frozen scope differs')
    for name in ('catalog', 'allowlist', 'cache_index', 'dataset', 'budget', 'probe',
                 'diagnosis', 'coverage', 'literal_coverage', 'diagnostic_index', 'reference_score_protocol'):
        bind(c[name]['path'], c[name]['sha256'])
    for path, digest in c['sources'].items():
        bind(path, digest)
    bind(__file__)
    for name in ('audit_diverse_native.py', 'audit_diverse_catalog.py', 'audit.py'):
        bind(HERE / name)
    decision_path = PREP / 'root_execution_decision.json'
    bind(decision_path)
    decision = read(decision_path)
    require(decision['conditional_execution_authorized'] is True
            and decision['contract_sha256'] == CONTRACT_SHA
            and decision['new_ASR_authorized'] is False, 'Owner authorization scope differs')
    # The frozen execute path rewrites this derived CPU report. Its old hash in
    # prepared_v2 is historical evidence, not a present-file equality claim.
    prepared_path = PREP / 'prepared_v2.json'
    bind(prepared_path, decision['prepared_sha256'])
    prepared = read(prepared_path)
    note_path = B / 'format60_preflight_evidence_note_v2.json'
    bind(note_path, 'ce6c1ff8478c21950f4a51082ac49d71f1aeb96e9b3c555df018ae230340fdfc')
    note = read(note_path)
    bind(note['prior_note_path'], note['prior_note_sha256'])
    current_report = Path(note['preparation_report_path'])
    snapshot = Path(note['runtime_report_snapshot_path'])
    current_sha = bind(current_report, note['current_runtime_report_sha256'])
    snapshot_sha = bind(snapshot, note['runtime_report_snapshot_sha256'])
    historical_sha = prepared['CPU_preflight']['sha256']
    require(Path(prepared['CPU_preflight']['path']).resolve() == current_report.resolve()
            and historical_sha == note['preparation_report_sha256_recorded_before_launch']
            and historical_sha != current_sha == snapshot_sha
            and note['prepared_report_original_full_bytes_available'] is False
            and note['old_prepared_report_SHA_not_claimed_as_current_match'] is True,
            'Derived preflight overwrite history differs from the preserved note')
    require(current_report.stat().st_mtime == note['current_runtime_report_source_mtime_unix'],
            'Runtime CPU report mtime differs from the preserved note')
    runtime_preflight = read(snapshot)
    require(runtime_preflight['contract_sha256'] == CONTRACT_SHA and runtime_preflight['status'] == 'passed'
            and runtime_preflight['full_contexts_prepared'] == 360
            and runtime_preflight['payload_occurrences_prepared'] == 540
            and runtime_preflight['model_forward_calls'] == 0, 'Runtime derived CPU evidence binding differs')
    report['derived_preflight_evidence_overwrite'] = dict(
        note_path=str(note_path), note_sha256=sha(note_path), prepared_sidecar_sha256=sha(prepared_path),
        historical_preparation_report_sha256=historical_sha, historical_original_full_bytes_available=False,
        historical_sha_claimed_as_current_match=False, current_report_path=str(current_report),
        current_report_sha256=current_sha, current_report_mtime_unix=current_report.stat().st_mtime,
        runtime_snapshot_path=str(snapshot), runtime_snapshot_sha256=snapshot_sha,
        runtime_snapshot_mtime_unix=snapshot.stat().st_mtime,
        cause=note['cause'], core_frozen_input_hashes_still_required=True)
    bind(args.terminal_evidence)
    outcome = read(args.terminal_evidence)
    require(Path(outcome['run_output']).resolve() == run and outcome['producer_live'] is False
            and isinstance(outcome['exit_code'], int), 'Terminal outcome lacks the actual stopped producer result')
    require(not process_live(outcome['pid'], outcome['start_ticks']), 'The known native producer is still live')
    report['producer_outcome'] = outcome
    report['producer_outcome_sha256'] = sha(args.terminal_evidence)
    report['contract_sha256'] = CONTRACT_SHA
    require(read(c['cache_index']['path'])['entries'] == [], 'Cache imports were not empty')
    allowed_list = read(c['allowlist']['path'])['raw_sha256']
    allowed = set(allowed_list)
    require(len(allowed_list) == len(allowed) == COUNT, 'Allowlist must contain exactly 60 unique identities')
    catalog_rows = lines(c['catalog']['path'])
    catalog = {r['raw_sha256']: r for r in catalog_rows}
    require(len(catalog_rows) == COUNT and set(catalog) == allowed
            and all('template' not in r for r in catalog_rows), 'Reference catalog scope differs')
    diagnostics = {r['raw_sha256']: r for r in lines(c['diagnostic_index']['path'])}
    coverage = {r['raw_sha256']: r for r in read(c['literal_coverage']['path'])['literals']}
    require(allowed == {raw for raw, row in coverage.items() if row['existing_score_records'] == 0
                        and not row['compatible_complete_current_user_native_cache']}, 'Original unscored set differs')
    dataset = read(c['dataset']['path'])
    contexts = dataset['contexts']
    require(len(contexts) == len({r['case_id'] for r in contexts}) == 6
            and Counter(r['suite'] for r in contexts) == {'banking': 2, 'slack': 2, 'travel': 2}, 'Fixed panel differs')
    ref = read(c['reference_score_protocol']['path'])
    require(ref['dataset_sha256'] == c['dataset']['sha256'] and ref['probe_sha256'] == c['probe']['sha256']
            and ref['target_role'] == 'user' and ref['target_role_index'] == 2, 'Current probe or panel differs')
    protocol_path = run / 'native/score_protocol.json'
    bind(protocol_path)
    require(read(protocol_path) == ref, 'Actual native numerical protocol differs')
    protocol_digest = hashlib.sha256(json.dumps(ref, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    run_protocol_path = run / 'run_protocol.json'
    bind(run_protocol_path)
    rp = read(run_protocol_path)
    require(rp['caps'] == [CAP] and rp['extra_catalog_paths'] == []
            and rp['catalog_sha256'] == c['catalog']['sha256']
            and rp['legacy_cache_index_sha256'] == c['cache_index']['sha256']
            and rp['dataset_sha256'] == c['dataset']['sha256'] and rp['probe_sha256'] == c['probe']['sha256']
            and rp['no_new_text_generation'] is True and rp['no_API_generation'] is True
            and rp['full_six_scores_only'] is True and rp['genuine_special_tokens_allowed'] is False,
            'Actual static-run scope differs')
    for path, digest in rp['sources'].items():
        require(c['sources'].get(path) == digest, 'Run source is outside the frozen source contract')
        bind(path, digest)
    constraint_path = run / 'format60_constraint_protocol.json'
    bind(constraint_path)
    constraint = read(constraint_path)
    require(constraint['contract_sha256'] == CONTRACT_SHA and constraint['sources'] == c['sources']
            and constraint['exact_allowlist'] == sorted(allowed) and constraint['maximum_wrapper_tokens'] == CAP
            and constraint['cache_imports'] == 0 and constraint['extra_catalogs'] == []
            and all(constraint[k] is False for k in ('generation', 'mutation', 'API', 'ASR')),
            'Actual exception boundary differs')
    controls = {}
    for name in ('checkpoint.json', 'format60_terminal_integrity.json', 'score_records.jsonl',
                 'rejections.jsonl', 'winner.json', 'interval_candidates.json'):
        path = run / name
        if path.exists():
            bind(path)
            if path.suffix == '.json':
                controls[name] = read(path)
    records = lines(run / 'score_records.jsonl')
    rejects = lines(run / 'rejections.jsonl')
    require(len({r['raw_sha256'] for r in records}) == len(records), 'Duplicate complete score records')
    require(all(r['raw_sha256'] in allowed for r in records + rejects), 'Record outside the exact allowlist')
    record_map = {r['raw_sha256']: r for r in records}
    report['producer_controls'] = controls
    report['rejections_preserved'] = rejects
    report['recorded_complete_count'] = len(records)
    report['missing_record_raw_sha256'] = sorted(allowed - set(record_map))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    library = BroadDelimiterLibrary(c['model'])
    for name, digest in ref['model_binding']['assets_sha256'].items():
        bind(Path(c['model']) / name, digest)
    spans_path = HERE.parents[1] / 'experiments/agentdojo-first-tool/spans.py'
    preparer = object.__new__(PreciseScorer)
    preparer.protocol, preparer.tokenizer = ref, tokenizer
    preparer.spans = load_module('_independent_format60_final_spans', spans_path)
    preparer.native_special_id_floor = 199998
    preparer.native_special_spellings = ref['payload_delimiter_policy']['reject_real_special_spellings']
    config = read(Path(c['model']) / 'config.json')
    preparer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=config['max_position_embeddings']))
    source_docs, verified, problems = {}, [], []
    for raw in sorted(allowed):
        entry = catalog[raw]
        require(diagnostics[raw]['diagnostic']['category'] == 'format_punctuation_only_repetition', 'Non-format literal exception')
        source = Path(entry['source_path'])
        bind(source, entry['source_sha256'])
        if str(source) not in source_docs:
            source_docs[str(source)] = read(source)
        original = pointer_value(source_docs[str(source)], entry['json_pointer'])
        require(raw_sha(original) == raw and original.encode() == diagnostics[raw]['template'].encode(), 'Original source bytes differ')
        policy = validate_diverse_template(original, tokenizer, library, maximum=CAP)
        require(policy['allowed'] and policy['metrics']['wrapper_tokens'] <= CAP, 'Original diverse policy failed')
        native_path = run / 'native/scores' / raw[:24] / 'score.json'
        if raw not in record_map:
            preserved = {'raw_sha256': raw, 'status': 'no_complete_logged_native_measurement', 'fitness': None}
            if native_path.exists():
                bind(native_path)
                partial = read(native_path)
                preserved.update(native_status=partial.get('status'), native_error=partial.get('error'),
                                 saved_case_count=len(partial.get('per_case', [])), source_score_path=str(native_path))
            report.setdefault('incomplete_native_records_preserved', []).append(preserved)
            continue
        try:
            row = record_map[raw]
            bind(native_path, row['source_score_sha256'])
            require(Path(row['source_score_path']).resolve() == native_path.resolve(), 'Foreign/imported native score path')
            score = read(native_path)
            require(all(row.get(k) == v for k, v in score.items()), 'Logged record changed native score fields')
            require(score['template'].encode() == original.encode() and score['status'] == 'complete'
                    and score['complete'] is True and score['expected_cases'] == score['completed_cases'] == 6
                    and score['cache_hit'] is False and score['target_role'] == 'user' and score['target_role_index'] == 2
                    and score['protocol_sha256'] == protocol_digest and len(score['per_case']) == 6,
                    'Incomplete/non-native score measurement')
            require(row['origin'] == 'fresh_fixed_catalog' and row['evaluated_cap'] == CAP
                    and row['wrapper_tokens'] == policy['metrics']['wrapper_tokens'] and row['policy'] == policy,
                    'Fresh/cap/diverse evidence differs')
            require(row['original_catalog_source'] == {k: entry[k] for k in ('source_path', 'source_sha256', 'json_pointer')},
                    'Original source pointer binding differs')
            means, case_evidence, copies = [], [], 0
            for index, (context, case) in enumerate(zip(contexts, score['per_case'])):
                require(case['case_id'] == context['case_id'] and case['suite'] == context['suite']
                        and case['status'] == 'complete' and case['template'].encode() == original.encode()
                        and case['actual_goal_sha256'] == raw_sha(context['actual_goal'])
                        and case['target_role'] == 'user' and case['target_role_index'] == 2
                        and case['protocol_sha256'] == protocol_digest and case['scored_payload_occurrence'] == 0,
                        'Complete case identity/goal/protocol differs')
                request, prompt, ids, offsets, positions, goal, location, input_sha = preparer.prepare(context, original)
                folder = native_path.parent / f'case_{index:02d}'
                artifact = folder / 'tokens.npz'
                require(Path(case['token_artifact']).resolve() == artifact.resolve(), 'Foreign native token artifact path')
                require(set(case['artifact_sha256']) == {'tokens.npz', 'request.json'}, 'Native artifact binding set differs')
                for name, digest in case['artifact_sha256'].items():
                    bind(folder / name, digest)
                bind(folder / 'score.json')
                require(read(folder / 'score.json') == case, 'Native per-case record differs')
                saved = read(folder / 'request.json')
                require(all(saved[k] == v for k, v in request.items()) and saved['rendered_prompt'] == prompt
                        and saved['location'] == case['location'] == location
                        and saved['input_sha256'] == case['input_sha256'] == input_sha
                        and saved['input_tokens'] == case['input_tokens'] == len(ids), 'Full native input/request/location differs')
                with np.load(artifact, allow_pickle=False) as a:
                    for name, expected in [('input_ids', ids), ('input_token_offsets', offsets),
                            ('payload_token_indices', positions), ('goal_mask', goal), ('template_mask', ~goal),
                            ('head_ids', ref['head_ids']), ('roles', ref['roles']),
                            ('payload_char_span', location['payload_char_span']), ('goal_char_span', location['goal_char_span']),
                            ('payload_char_spans', [o['payload_char_span'] for o in location['occurrences']]),
                            ('goal_char_spans', [o['goal_char_span'] for o in location['occurrences']])]:
                        same_array(a[name], expected, name)
                    require(str(a['target_role'].item()) == 'user' and int(a['target_role_index']) == 2, 'Array role differs')
                    p, bits, features = a['probabilities'], a['native_bf16_bits'], a['features_fp16']
                    require(p.dtype == np.float32 and p.shape == (len(positions), 6) and np.isfinite(p).all()
                            and ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(1), 1, atol=2e-6, rtol=0), 'Invalid saved probabilities')
                    require(bits.dtype == np.uint16 and bits.shape == (len(positions), 16, 64)
                            and features.dtype == np.float16 and features.shape == (len(positions), 1024), 'Native feature shape/dtype differs')
                    recovered = (bits.astype(np.uint32) << 16).view(np.float32).reshape(len(positions), 1024).astype(np.float16)
                    same_array(features, recovered, 'native BF16 to FP16 features')
                    require(np.isfinite(features).all(), 'Nonfinite saved native feature')
                    role_means = p[goal].mean(0, dtype=np.float64)
                require(case['goal_tokens'] == int(goal.sum()) and case['observed_payload_occurrences'] == len(location['occurrences']),
                        'Full goal/copy count differs')
                for role, value in zip(ref['roles'], role_means):
                    require(abs(case['probabilities']['goal'][role] - float(value)) <= 1e-12, 'Saved goal-role mean differs')
                require(abs(case['goal_user_mean'] - role_means[2]) <= 1e-12
                        and abs(case['goal_target_mean'] - role_means[2]) <= 1e-12, 'Saved user/index2 mean differs')
                means.append(role_means)
                copies += len(location['occurrences'])
                case_evidence.append(dict(case_id=context['case_id'], input_sha256=input_sha, input_tokens=len(ids),
                    goal_tokens=int(goal.sum()), payload_copies=len(location['occurrences']),
                    native_goal_user_mean=float(role_means[2]), full_input_offsets_masks_spans_bitwise_equal=True,
                    native_payload_special_token_count=location['payload_special_token_check']['native_special_token_count'],
                    token_artifact_sha256=case['artifact_sha256']['tokens.npz'], request_sha256=case['artifact_sha256']['request.json']))
            require(len(means) == 6 and copies == 9, 'Complete six contexts/nine copies required')
            aggregate = np.mean(means, axis=0)
            require(abs(score['fitness'] - aggregate[2]) <= 1e-12 and abs(score['goal_target_mean'] - aggregate[2]) <= 1e-12,
                    'Complete equal-six user aggregate differs')
            for role, value in zip(ref['roles'], aggregate):
                require(abs(score['goal_role_means'][role] - float(value)) <= 1e-12, 'Equal-six role aggregate differs')
            verified.append(dict(raw_sha256=raw, native_mean_p_user=float(aggregate[2]), wrapper_tokens=row['wrapper_tokens'],
                source_score_path=str(native_path), source_score_sha256=sha(native_path), original_source=row['original_catalog_source'],
                complete_six_input_mask_aggregation_verified=True, payload_copies=copies, per_case=case_evidence))
            print(json.dumps(dict(stage='verified_native_literal', verified=len(verified), raw_sha256=raw)), flush=True)
        except Exception as error:
            problems.append(dict(raw_sha256=raw, error=type(error).__name__ + ': ' + str(error)))
    expected_native = {run / 'native/scores' / raw[:24] / 'score.json' for raw in allowed}
    actual_native = set((run / 'native/scores').glob('*/score.json'))
    require(actual_native <= expected_native, 'Unexpected native candidate directory outside the allowlist')
    native_folders = list((run / 'native/scores').iterdir()) if (run / 'native/scores').exists() else []
    require(all(path.name in {raw[:24] for raw in allowed} and not path.is_symlink()
                for path in native_folders), 'Foreign native candidate folder or cache symlink')
    checkpoint = controls.get('checkpoint.json', {})
    terminal = controls.get('format60_terminal_integrity.json', {})
    all_complete = (outcome['exit_code'] == 0 and not problems and len(verified) == COUNT and not rejects
                    and set(record_map) == allowed and terminal.get('status') == 'passed'
                    and terminal.get('exact60_complete') is True and terminal.get('contract_sha256') == CONTRACT_SHA
                    and checkpoint.get('status') == 'audited_finite_catalog_exhausted' and checkpoint.get('cap') == CAP
                    and checkpoint.get('phase_index') == 0
                    and checkpoint.get('complete_scores') == checkpoint.get('fresh_scores') == COUNT)
    report.update(verified_native_scores=verified, verification_problems=problems, verified_complete_count=len(verified),
                  original_source_files=len(source_docs), exact60_complete_structure_passed=all_complete,
                  strict_exception_applied_only_to_frozen_allowlist=True, strict_header_filter_reapplied=False)
    if all_complete:
        best = sorted(records, key=lambda r: (-r['fitness'], r['raw_sha256']))[0]
        winner = controls.get('winner.json', {})
        require(winner.get('raw_sha256') == best['raw_sha256'] and winner.get('fitness') == best['fitness'], 'Saved winner differs from highest new raw score')
        record_path = OUT / 'best_original_record.json'
        write(record_path, best)
        child = OUT / 'best_native_checkpoint_recompute'
        report['checkpoint_probability_recomputations'] = 1
        write(OUT / 'single_checkpoint_attempt.json', dict(raw_sha256=best['raw_sha256'], record_sha256=sha(record_path), attempts=1))
        print(json.dumps(dict(stage='single_best_CPU_checkpoint_recompute', raw_sha256=best['raw_sha256'])), flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
        result = subprocess.run([sys.executable, str(HERE / 'audit_diverse_native.py'), '--record', str(record_path),
            '--dataset', c['dataset']['path'], '--run-protocol', str(run_protocol_path), '--out', str(child),
            '--model', c['model'], '--probe', c['probe']['path']], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (OUT / 'single_checkpoint_audit.log').write_text(result.stdout)
        require(result.returncode == 0, 'One authorized CPU checkpoint audit failed: ' + result.stdout[-2000:])
        proof = read(child / 'audit.json')
        require(proof['status'] == 'passed' and proof['raw_sha256'] == best['raw_sha256']
                and proof['native_score_sha256'] == sha(best['source_score_path'])
                and proof['probe_sha256'] == c['probe']['sha256'] and proof['dataset_sha256'] == c['dataset']['sha256']
                and proof['independent_native_feature_probe_recompute']['candidates'] == 1, 'Highest raw checkpoint audit binding differs')
        report.update(status='passed', highest_new_raw_sha256=best['raw_sha256'], highest_new_mean_p_user=best['fitness'],
                      single_checkpoint_audit_path=str(child / 'audit.json'), single_checkpoint_audit_sha256=sha(child / 'audit.json'),
                      maximum_probe_probability_error=proof['independent_native_feature_probe_recompute']['max_probability_absolute_error'])
    else:
        report['status'] = 'failed' if problems else 'incomplete'
        report['checkpoint_recompute_skipped_reason'] = 'No exact60 fully verified successful producer outcome; no partial objective or scoring retry.'
    for path, digest in bindings.items():
        require(sha(path) == digest, 'Bound source/native artifact changed during CPU audit: ' + path)
    report['source_and_native_artifact_bindings'] = bindings
    report['all_bound_sources_and_artifacts_unchanged'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--terminal-evidence', type=Path, required=True,
        help='Owner JSON with run_output, exit_code, producer_live=false, pid and start_ticks from the actual stopped native producer')
    args = parser.parse_args()
    require(args.run.resolve() == (B / 'probe_format60_short128').resolve(), 'Unexpected native supplement path')
    outcome = read(args.terminal_evidence)
    require(outcome.get('producer_live') is False and not process_live(outcome['pid'], outcome['start_ticks']),
            'Audit is restricted to an actually stopped native producer')
    with (B / 'asr_stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        (OUT / 'snapshots').mkdir()
        shutil.copyfile(args.terminal_evidence, OUT / 'snapshots/producer_outcome.json')
        # Preserve the stopped producer's own terminal/error controls even if an
        # early frozen-binding check fails. No producer artifact is rewritten.
        for name in ('checkpoint.json', 'format60_terminal_integrity.json', 'score_records.jsonl',
                     'rejections.jsonl', 'winner.json', 'interval_candidates.json', 'run_protocol.json',
                     'format60_constraint_protocol.json'):
            source = args.run / name
            if source.exists():
                shutil.copyfile(source, OUT / 'snapshots' / name)
        report = dict(schema='independent-format60-terminal-native-audit-v1', status='running', started_unix=time.time(),
            run_output=str(args.run.resolve()), exact_allowlist_count=COUNT, maximum_wrapper_tokens=CAP,
            audit_source_sha256=sha(__file__), backbone_forward_calls=0, new_native_scores=0,
            checkpoint_probability_recomputations=0, API_calls=0, ASR_audits_or_runs=0, GPU_used=False,
            source_artifacts_modified=False, new_payload_generation=False, native_retries=0)
        try:
            audit(args, report)
        except Exception as error:
            report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        report['completed_unix'] = time.time()
        write(OUT / 'audit.json', report)
        print(json.dumps({k: report.get(k) for k in ('status', 'verified_complete_count', 'exact60_complete_structure_passed',
            'highest_new_raw_sha256', 'highest_new_mean_p_user', 'checkpoint_probability_recomputations', 'error')}), flush=True)
        return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
