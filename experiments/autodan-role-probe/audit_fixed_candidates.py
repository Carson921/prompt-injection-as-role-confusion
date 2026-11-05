#!/usr/bin/env python3
"""CPU-only integrity and official-predicate replay of fixed-candidate ASR.

No model is loaded, no generation is requested, and no task environment is
advanced. The frozen auditor restores saved native snapshots and checks the
official predicates. Missing/error verdicts remain unknown in all summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
from itertools import combinations
import json
import math
from pathlib import Path
import time

import audit_asr_optimized as frozen
import evaluate_fixed_candidates as evaluator
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from prepare import COUNTS, ROOT, SOURCE, read, sha


require = frozen.require
serial = frozen.serial
write = frozen.write


def wilson_interval(successes, n, z=1.959963984540054):
    if not n:
        return None
    require(0 <= successes <= n, 'Invalid Wilson interval counts')
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [0.0 if successes==0 else max(0.0, center-half),
            1.0 if successes==n else min(1.0, center+half)]


def condition_metrics(rows, expected):
    result = frozen.condition_metrics(rows, expected)
    result['unknown'] = result['null_verdicts']
    result['scored_asr_wilson_95'] = wilson_interval(result['successes'], result['scored'])
    result['wilson_scope'] = ('Descriptive interval among scored episodes only; initial errors/missing '
                              'episodes are excluded here and remain unknown in ASR identification bounds')
    return result


def verify_sources(root, protocol):
    require(protocol['evaluator_source_sha256'] == sha(ROOT/'evaluate_fixed_candidates.py'),
            'Fixed-candidate evaluator bytes differ from the frozen protocol')
    require(protocol['native_episode_runner_source_sha256'] == sha(ROOT/'evaluate_optimized.py'),
            'Frozen native episode runner bytes differ')
    bindings = protocol.get('source_bindings')
    require(isinstance(bindings, dict) and len(bindings) >= 10, 'Source binding inventory is incomplete')
    required = [ROOT/'evaluate_fixed_candidates.py',ROOT/'evaluate_optimized.py',ROOT/'evaluate_precise.py',
                ROOT/'prepare.py',ROOT/'diverse_policy.py',ROOT/'broad_validation.py',
                ROOT/'near_delimiters.py',ROOT/'deterministic_spans.py']
    for path in required:
        require(bindings.get(str(path.resolve())) == sha(path), 'Required source binding differs: '+path.name)
    for source, digest in bindings.items():
        require(sha(Path(source)) == digest, 'A protocol-bound implementation file changed: '+source)
    manifest = read(root/'provenance/source_manifest.json')
    require(len(manifest) == len(bindings)
            and {row['source'] for row in manifest} == set(bindings), 'Frozen source-copy inventory differs')
    for row in manifest:
        relative = Path(row['frozen_copy']).name
        copied = root/'provenance'/relative
        require(row['sha256'] == bindings[row['source']] and sha(copied) == row['sha256'],
                'Frozen provenance source bytes differ')
    return dict(bound_implementation_files=len(bindings),frozen_source_copies=len(manifest),
                all_source_hashes_verified=True)


def verify_candidates(protocol, tokenizer, library, dataset, dataset_path, manifest_path=None):
    candidates = protocol.get('candidates')
    require(isinstance(candidates,list) and 3<=len(candidates)<=6,
            'Expected three to six frozen fixed candidates')
    require(len({row['condition'] for row in candidates})==len(candidates), 'Duplicate candidate condition')
    require(set(protocol['templates']) == {row['condition'] for row in candidates},
            'Condition/template inventory differs from frozen candidates')
    checkpoint_shas, intervals, records = set(), [], []
    for candidate in candidates:
        template = candidate['template']
        digest = evaluator.text_sha(template)
        require(digest == candidate['template_raw_sha256']
                and candidate['condition']=='candidate_'+digest[:16]
                and template == protocol['templates'][candidate['condition']],
                'Stable condition ID, exact template, or raw SHA differs')
        admission = candidate['admission']
        guard = validate_diverse_template(template,tokenizer,library,
                                           maximum=admission['maximum_wrapper_tokens'])
        require(guard == admission and guard['allowed'], 'Independent nonrepeat/special-token admission differs')
        prior = candidate['probe_evidence']
        independent = evaluator.bind_probe_score(dict(score_path=prior['score_path'],
            score_protocol_path=prior['score_protocol_path']),template,Path.cwd(),dataset,sha(dataset_path))
        require(prior == independent, 'Prior complete native probe evidence differs')
        lower,upper = candidate['p_user_interval']
        value = prior['goal_user_mean']
        require(0<=lower<upper<=1 and lower<=value and (value<upper or value==upper==1),
                'Candidate measured P(user) is outside its declared interval')
        require(all(upper<=a or lower>=b for a,b in intervals),'Candidate P(user) intervals overlap')
        intervals.append((lower,upper))
        checkpoint_shas.add(prior['probe_checkpoint_sha256'])
        records.append(dict(condition=candidate['condition'],template_raw_sha256=digest,
                            goal_user_mean=value,p_user_interval=[lower,upper],
                            wrapper_tokens=guard['metrics']['wrapper_tokens'],
                            score_file_sha256=prior['score_file_sha256']))
    require(len(checkpoint_shas)==1 and None not in checkpoint_shas,'Candidate role-probe checkpoint differs')
    manifest_verified = False
    if manifest_path is not None:
        require(sha(manifest_path)==protocol['candidates_manifest_sha256'],'Original fixed-candidate manifest SHA differs')
        recomputed=evaluator.load_candidates(manifest_path,tokenizer,library,dataset,sha(dataset_path),
                    maximum=max(row['admission']['maximum_wrapper_tokens'] for row in candidates))
        require(recomputed==candidates,'Original manifest no longer resolves to frozen fixed candidates')
        manifest_verified=True
    return records,manifest_verified


def verify_engine_configuration(protocol):
    config = protocol['engine_configuration']
    expected = dict(model=config['model'],tokenizer=config['model'],dtype='bfloat16',quantization='mxfp4',
                    tensor_parallel_size=2,pipeline_parallel_size=1,distributed_executor_backend='mp',
                    max_model_len=65536,gpu_memory_utilization=.85,max_num_seqs=32,
                    max_num_batched_tokens=8192,enable_chunked_prefill=True,enable_prefix_caching=False,
                    enforce_eager=True,generation_config='vllm',seed=0,trust_remote_code=False)
    require(config==expected,'Fixed-candidate vLLM configuration differs from the frozen optimized-only configuration')
    require(protocol['max_tokens']==1024 and protocol['reasoning_effort']=='medium'
            and protocol['prompt_date']=='2026-10-21' and protocol['seed']==0
            and protocol['temperature']==0. and protocol['stop_ids']==frozen.STOP_IDS,
            'Generation/prompt configuration differs')
    require(protocol.get('candidate_generation') is False and protocol.get('candidate_optimization') is False
            and protocol.get('external_llm_api_calls') is False,
            'This auditor accepts only fixed-candidate evaluations')


def replay_unparsed_error_calls(root, record, native, encoding, protocol):
    """Verify failed parser outcomes without changing their null verdicts."""
    events=[]
    if record['status']!='error':
        return events
    for reference in record['requests']:
        if 'raw_response_sha256' not in reference or 'parsed_response_sha256' in reference:
            continue
        call_id=reference['call_id']
        raw_path=root/'calls'/call_id/'raw_response.json'
        response=read(raw_path)
        require(sha(raw_path)==reference['raw_response_sha256'],'Error raw-response SHA differs')
        generated=frozen.validate_raw(response,protocol['max_tokens'],call_id)
        try:
            native['safe_parse_completion'](encoding,generated,call_id,max_tokens=protocol['max_tokens'],
                                            stop_ids=frozen.STOP_IDS)
        except Exception as error:
            observed=f'{type(error).__name__}: {error}'
            require(observed==record['error'],'Error native parser replay produced a different exception')
            events.append(dict(case_id=record['case_id'],call_id=call_id,status='parser_error_reproduced',
                               error=observed,generated_token_ids=len(generated),verdict_preserved_null=True))
        else:
            # A saved I/O/generation/infrastructure error may occur after a
            # successful parse. A successful replay cannot manufacture a verdict.
            events.append(dict(case_id=record['case_id'],call_id=call_id,
                               status='parser_succeeded_initial_error_remains_unknown',
                               recorded_error=record['error'],generated_token_ids=len(generated),
                               verdict_preserved_null=True))
    return events


def verify_episode_artifact_inventory(root, rows, cases_by_id, *, partial):
    expected={name:set() for name in ('request','raw_response','parsed_response','trajectory','injection')}
    injection_hashes=[]
    for row in rows:
        case_id=row['case_id']
        record=read(root/'cases'/(case_id+'.json'))
        for reference in record['requests']:
            call_id=reference['call_id']
            expected['request'].add(call_id)
            if 'raw_response_sha256' in reference: expected['raw_response'].add(call_id)
            if 'parsed_response_sha256' in reference: expected['parsed_response'].add(call_id)
        for index,_reference in enumerate(record['trajectories']):
            expected['trajectory'].add(case_id+f'/pipeline_{index:02d}')
        injection_path=root/'injections'/(case_id+'.json')
        if injection_path.exists():
            values=read(injection_path)
            require(isinstance(values,dict),'Saved injection map is not a mapping')
            expected_payload=cases_by_id[case_id]['payload_text'].replace('\\','\\\\').replace('"','\\"')
            require(all(isinstance(value,str) and value==expected_payload for value in values.values()),
                    'Saved injection content differs from the exact immutable goal rendering')
            expected['injection'].add(case_id)
            injection_hashes.append(dict(case_id=case_id,sha256=sha(injection_path),injection_fields=len(values)))
        else:
            require(record['status']=='error','Completed native episode lacks its injection map')
    actual={}
    for name,filename in [('request','request.json'),('raw_response','raw_response.json'),('parsed_response','parsed_response.json')]:
        actual[name]={path.parent.relative_to(root/'calls').as_posix() for path in (root/'calls').rglob(filename)}
    actual['trajectory']={path.relative_to(root/'trajectories').as_posix()[:-5]
                          for path in (root/'trajectories').rglob('pipeline_*.json')}
    actual['injection']={path.relative_to(root/'injections').as_posix()[:-5]
                         for path in (root/'injections').rglob('*.json')}
    for name in expected:
        require(expected[name].issubset(actual[name]),'Referenced native artifact missing: '+name)
        if not partial:
            require(expected[name]==actual[name],'Unexpected/unreferenced native artifacts: '+name)
    return dict(referenced={name:len(value) for name,value in expected.items()},
                observed={name:len(value) for name,value in actual.items()},
                full_inventory_exact=not partial,injection_hashes=injection_hashes)


def verify_summary(root, rows, conditions):
    path=root/'summary.json'
    require(path.exists(),'Final producer summary is missing')
    summary=read(path)
    require(summary['tasks_per_condition']==389 and summary['scheduled_episodes']==len(conditions)*389
            and summary['terminal']==len(rows),'Producer summary scheduled/terminal counts differ')
    require(set(summary['conditions'])==set(conditions),'Producer condition summary inventory differs')
    for condition in conditions:
        observed=summary['conditions'][condition]
        independent=condition_metrics([row for row in rows if row['condition']==condition],389)
        for key in ('scheduled','terminal','scored','errors','successes','asr','scored_asr',
                    'known_successes_over_scheduled','utility','scored_utility','exposed','exposure_rate',
                    'unknown','asr_lower_bound','asr_upper_bound'):
            require(observed[key]==independent[key],'Producer summary disagrees with native replay: '+condition+'/'+key)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'outputs/near-delimiter-20261104/dataset.json')
    parser.add_argument('--candidates',type=Path,help='Optionally rebind the exact original candidate manifest')
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--model',type=Path,help='Tokenizer path override after model staging relocation')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--partial',action='store_true',help='Missing episodes remain unknown; audit available terminal episodes only')
    parser.add_argument('--protocol-only',action='store_true',help='Validate CPU protocol only; explicitly replay zero episode/model calls')
    args=parser.parse_args()
    root=args.run_root.resolve()
    output=args.output.resolve() if args.output else root/'audit_fixed'
    protocol=read(root/'protocol.json')
    binding=hashlib.sha256(serial(protocol).encode()).hexdigest()
    require(protocol.get('schema')=='fixed-candidates-native-vllm-eval-v1' and protocol['scope']=='full',
            'Unknown fixed-candidate full-evaluation protocol')
    source_check=verify_sources(root,protocol)
    verify_engine_configuration(protocol)
    require(protocol['source_identity_sha256']==sha(args.source/'agentdojo-source.json'),'Native source identity differs')
    require(protocol['dataset_sha256']==sha(args.dataset),'Fixed numerical dataset binding differs')
    dataset=read(args.dataset)
    require(protocol['normal_system']==dataset['fixed_normal_system'],'Normal system message differs')
    native=frozen.load_dependencies(args.source)
    validation=protocol.get('serialized_payload_validation',{})
    require(validation.get('method')=='deterministic-native-string-origins-v1'
            and validation.get('unchanged_native_prompt_and_goal') is True,'Unknown rendered-payload policy')
    for key,path in [('deterministic_locator_source_sha256',ROOT/'deterministic_spans.py'),
                     ('trusted_outer_spans_source_sha256',ROOT.parent/'agentdojo-first-tool/spans.py'),
                     ('native_recipe_source_sha256',Path(inspect.getsourcefile(native['recipes'])))]:
        require(validation.get(key)==sha(path),'Rendered-payload implementation binding differs: '+key)
    model=args.model or Path(protocol['engine_configuration']['model'])
    for name,digest in protocol['model_configuration_sha256'].items():
        require(sha(model/name)==digest,'Authoritative model/tokenizer asset SHA differs: '+name)
    require(sha(model/'staging-proof.json')==protocol['model_staging_proof_sha256'],'Staged model proof SHA differs')
    tokenizer=native['AutoTokenizer'].from_pretrained(model,local_files_only=True)
    library=BroadDelimiterLibrary(model)
    candidate_rows,manifest_verified=verify_candidates(protocol,tokenizer,library,dataset,args.dataset,args.candidates)
    conditions=tuple(row['condition'] for row in protocol['candidates'])
    suites=native['get_suites']('v1.2.1')
    panel=evaluator.full_panel(suites)
    expected_cases=evaluator.fixed_cases(panel,suites,protocol['candidates'])
    require(protocol['cases']==expected_cases,'Scheduled full population, native immutable goal, or fixed payload differs')
    cases_by_id={case['case_id']:case for case in expected_cases}
    expected_episode_count=len(conditions)*389
    require(len(expected_cases)==expected_episode_count,'Full candidate-by-task population differs')
    training_pairs={row['pair_id'] for row in dataset['sample_manifest']['selected_pairs']}
    require(len(training_pairs)==6 and training_pairs.issubset({row['pair_id'] for row in panel}),
            'Original six-case optimization cohort differs')
    token_guard=evaluator.frozen.validate_payload_tokens(tokenizer,expected_cases,dict(
        forbidden_ids=list(library.forbidden_token_ids),forbidden_spellings=list(library.forbidden_spellings)))
    require(token_guard==protocol['payload_special_token_validation'],'Filled-payload special-token exclusion differs')
    observed_paths=list((root/'cases').rglob('*.json'))
    observed_ids={path.relative_to(root/'cases').as_posix()[:-5] for path in observed_paths}
    require(observed_ids.issubset(cases_by_id),'Unexpected extra episode files')
    common=dict(run_root=str(root),expected_episodes=expected_episode_count,conditions=list(conditions),
                candidates=candidate_rows,original_candidate_manifest_sha_verified=manifest_verified,
                protocol_file_sha256=sha(root/'protocol.json'),protocol_binding_sha256=binding,
                dataset_sha256=sha(args.dataset),source_check=source_check,
                audit_source_sha256=sha(__file__),frozen_audit_case_source_sha256=sha(ROOT/'audit_asr_optimized.py'),
                payload_special_token_exclusion_verified=True,language_model_loaded=False,
                generation_calls=0,gpu_required=False)
    if args.protocol_only:
        report=dict(schema='fixed-candidates-cpu-protocol-audit-v1',status='protocol_verified_only',
                    **common,audited_episodes=0,model_calls_replayed=0,official_predicates_rechecked=0,
                    observed_episode_files=len(observed_ids),
                    note='CPU protocol validation only; no request, generation, trajectory, or verdict replay is claimed.')
        write(output/'protocol_integrity.json',report)
        print(serial(dict(event='protocol_audit_complete',output=str(output),status=report['status'],
                          audited_episodes=0,model_calls_replayed=0)),flush=True)
        return
    if not args.partial:
        require(observed_ids==set(cases_by_id),'Scheduled episodes are missing; use --partial while running')
    encoding=native['load_harmony_encoding'](native['HarmonyEncodingName'].HARMONY_GPT_OSS)
    started=time.time()
    rows,calls,error_replays=[],[],[]
    for case in expected_cases:
        if case['case_id'] not in observed_ids: continue
        record=read(root/'cases'/(case['case_id']+'.json'))
        if args.partial and record.get('status') not in ('complete','error'): continue
        require('import_provenance' not in record,'Fixed-candidate run unexpectedly imported an episode')
        row,case_calls=frozen.audit_case(root,case,protocol,binding,native,suites,tokenizer,encoding)
        row['cohort']='optimization6' if row['pair_id'] in training_pairs else 'holdout383'
        rows.append(row)
        calls.extend(case_calls)
        error_replays.extend(replay_unparsed_error_calls(root,record,native,encoding,protocol))
        if len(rows)%50==0:
            print(serial(dict(event='audit_progress',episodes=len(rows),model_calls=len(calls))),flush=True)
    inventory=verify_episode_artifact_inventory(root,rows,cases_by_id,partial=args.partial)
    condition_rows,paired_rows=[],[]
    for cohort,pairs in [('all',{row['pair_id'] for row in panel}),('optimization6',training_pairs),
                         ('holdout383',{row['pair_id'] for row in panel}-training_pairs)]:
        for suite in ('all',*COUNTS):
            selected={pair for pair in pairs if suite=='all' or pair.startswith(suite+'/')}
            subset=[row for row in rows if row['pair_id'] in selected]
            for condition in conditions:
                condition_rows.append(dict(cohort=cohort,suite=suite,condition=condition,
                    **condition_metrics([row for row in subset if row['condition']==condition],len(selected))))
            for left,right in combinations(conditions,2):
                paired_rows.append(dict(cohort=cohort,suite=suite,
                    **frozen.paired_metrics(subset,sorted(selected),left,right)))
    complete=len(rows)==expected_episode_count
    summary_verified=verify_summary(root,rows,conditions) if complete and not args.partial else False
    comparison=dict(schema='fixed-candidates-native-asr-results-v1',scope='full',conditions=condition_rows,
                    paired=paired_rows,candidates=candidate_rows,
                    error_policy='Initial errors and missing episodes have null verdicts; report ASR bounds and scored-only rates separately.',
                    pair_holdout_note='383 pair holdouts can share user tasks or injection objectives with the six optimization pairs.',
                    difference_direction='right minus left; positive favors right',
                    matched_backend_policy='Fixed candidate conditions generated by one shared native vLLM configuration; no baseline replay.')
    write(output/'comparison.json',comparison)
    frozen.write_csv(output/'case_results.csv',rows)
    frozen.write_csv(output/'condition_summary.csv',condition_rows)
    frozen.write_csv(output/'paired_comparisons.csv',paired_rows)
    write(output/'error_parse_replay.json',error_replays)
    report=dict(schema='fixed-candidates-native-asr-audit-v1',
                status='verified' if complete else 'partial_verified',**common,
                audited_episodes=len(rows),missing_or_nonterminal_episodes=expected_episode_count-len(rows),
                condition_counts=dict(Counter(row['condition'] for row in rows)),
                official_predicates_rechecked=sum(row['official_predicates_rechecked'] for row in rows),
                native_error_episodes=sum(row['status']=='error' for row in rows),
                native_errors_preserve_null=True,model_calls_replayed=len(calls),
                complete_prompt_tokens_reencoded=sum(row['prompt_tokens'] for row in rows),
                generated_token_ids_checked=sum(row['generated_tokens'] for row in rows),
                audited_native_artifacts_sha_verified=True,
                all_audited_completed_episodes_official_utility_security_predicates_match=True,
                error_parser_replays=len(error_replays),artifact_inventory=inventory,
                producer_summary_independently_matched=summary_verified,
                snapshot_restore_source_sha256=sha(native['snapshot_path']),
                predicate_replay_scope='Official predicates on saved first-pre/last-final typed snapshots and final function stack; tools and ground-truth pipelines are not rerun.',
                elapsed_seconds=time.time()-started,
                errors=[dict(case_id=row['case_id'],error=row['error']) for row in rows if row['status']=='error'],
                outputs={name:sha(output/name) for name in ['comparison.json','case_results.csv',
                    'condition_summary.csv','paired_comparisons.csv','error_parse_replay.json']})
    write(output/'audit.json',report)
    print(serial(dict(event='fixed_candidates_audit_complete',output=str(output),status=report['status'],
                      expected=expected_episode_count,audited=len(rows),model_calls_replayed=len(calls))),flush=True)


if __name__=='__main__':
    main()
