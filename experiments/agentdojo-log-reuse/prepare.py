"""Choose one existing scored episode per case, prepare only its logged input.

Selection never uses attack outcome. Priority: completed current native episode,
old complete native episode, reference episode, complete Table 1 episode, already
scored unexposed current episode. Only remaining cases need new ASR generation.
"""
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from transformers import AutoTokenizer

from reuse_common import *


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE / 'provenance'))
    sys.path.insert(0, str(BASE / 'postprocessing'))
    from spans import render, select_rows
    from spans_v2 import locate
    identity = read(BASE / 'agentdojo-source.json')
    source = Path(identity['source'])
    assert all(sha(source / name) == digest for name, digest in identity['files'].items())
    sys.path.insert(0, str(source))
    from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai, _function_to_openai
    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites
    suites = get_suites('v1.2.1')
    tok = AutoTokenizer.from_pretrained('/scratch/wz55191/models/gpt-oss-120b', local_files_only=True)
    protocol = read(BASE / 'manifest.json')
    schedule = read(BASE / 'schedule.json')
    canonical = {r['case_id']: r for r in read(BASE / 'canonical_cases.json')}
    features = {r['case_id']: r for r in read(BASE / 'feature_index.json')}
    refcases = {r['case_id']: r for r in read(REFERENCE / 'cases.json')}
    nativecases = {r['case_id']: (r, p) for p in (NATIVE_OLD / 'cases/chatinject').rglob('*.json')
                   if (r := read(p))['status'] == 'complete' and type(r.get('security')) is bool}
    inventory = read(OLD_ASR / 'log_reuse_inspection/table1_plain389_inventory.json')
    classic = {'plain_delimiter/' + '/'.join(r[k] for k in ('suite', 'user_task_id', 'injection_task_id')): r
               for r in inventory['cases']}
    plan, counts = [], Counter()

    def copy_source(path, target, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        shutil.copyfile(path, target)
        assert sha(target) == digest
        return {'original_path': str(path), 'path': str(target.relative_to(ROOT)), 'sha256': digest}

    def save_input(case, record, request, prompt, date, ids=None, original_exact=False):
        encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        if ids is not None:
            assert encoded['input_ids'] == ids, 'Saved original token IDs differ'
        ids = encoded['input_ids']
        assert len(ids) <= 131072
        assert render(tok, request, date) == prompt, 'Rendering differs from recorded prompt'
        location = locate(tok, request, prompt, date)
        if location is None:
            return False
        positions, payload, goal, removed = select_rows(np.asarray(ids), encoded['offset_mapping'], location, protocol['delimiter_ids'])
        directory = ROOT / 'logged_inputs' / case['case_id']
        directory.mkdir(parents=True, exist_ok=True)
        value = {'case_id': case['case_id'], 'request': request, 'rendered_prompt': prompt,
                 'input_ids': ids, 'prompt_date': date, 'location': location,
                 'positions': positions.tolist(), 'payload_mask': payload.tolist(), 'goal_mask': goal.tolist(),
                 'excluded_delimiter_positions': removed.tolist(), 'original_saved_prompt_and_ids': original_exact,
                 'system_order': 'original normal', 'input_reconstructed': not original_exact}
        path = directory / 'input.json'
        write(path, value)
        record.update(payload_exposed=True, input_path=str(path.relative_to(ROOT)), input_sha256=sha(path),
                      input_tokens=len(ids), rows=len(positions), original_saved_prompt_and_ids=original_exact,
                      feature_method='forward_logged_input')
        origin = canonical[case['case_id']]
        if origin['status'] == 'captured':
            current = read(phase(origin) / 'calls' / origin['first_exposure_call'] / 'input.json')
            if current['input_ids'] == ids and current['rendered_prompt'] == prompt:
                entry = features[case['case_id']]
                # Masks, positions and rows must match too, not only the prompt.
                from safetensors import safe_open
                with safe_open(str(BASE / entry['states']), framework='numpy') as state:
                    assert np.array_equal(state.get_tensor('positions'), positions)
                    assert np.array_equal(state.get_tensor('payload_mask'), payload)
                    assert np.array_equal(state.get_tensor('goal_mask'), goal)
                record.update(feature_method='reuse_exact_input_states', states=str(BASE / entry['states']),
                              states_sha256=entry['states_sha256'])
        return True

    for index, case in enumerate(schedule):
        cid = case['case_id']
        origin = canonical[cid]
        directory = ROOT / 'sources' / cid
        directory.mkdir(parents=True, exist_ok=True)
        record = {k: case[k] for k in ('case_id', 'condition', 'suite', 'user_task_id', 'injection_task_id')}
        record.update(case_index=index, attacker_success=None, utility=None, payload_exposed=None,
                      feature_method=None, sources=[])
        completed = OLD_ASR / 'cases' / (cid + '.json')
        native_id = cid.replace('plain_delimiter/', 'chatinject/', 1)
        ref_id = cid.replace('plain_delimiter/', 'chatinject/', 1).replace('remember_delimiter/', 'prerequisite_chatinject/', 1)
        if completed.exists() and (result := read(completed))['status'] == 'completed':
            record.update(asr_source='current_completed_native', attacker_success=result['attacker_success'], utility=result['utility'],
                          native_result=str(completed), native_result_sha256=sha(completed), payload_exposed=origin['status']=='captured')
            record['sources'].append(copy_source(completed, directory / 'case.json'))
            if origin['status'] == 'captured':
                entry = features[cid]
                record.update(feature_method='reuse_current_native_states', states=str(BASE / entry['states']),
                              states_sha256=entry['states_sha256'], rows=entry['rows'])
        elif native_id in nativecases:
            result, result_path = nativecases[native_id]
            record.update(asr_source='historical_native_complete', attacker_success=result['security'], utility=result['utility'])
            record['sources'].append(copy_source(result_path, directory / 'case.json'))
            exposed = False
            for item in sorted(result['requests'], key=lambda x:x['call_index']):
                for artifact in item['artifacts']:
                    name = Path(artifact['path']).name
                    path = NATIVE_OLD / 'calls' / item['call_id'] / name
                    target = directory / f"call_{item['call_index']:04d}_{name}"
                    record['sources'].append(copy_source(path, target, artifact['sha256']))
                request_path = NATIVE_OLD / 'calls' / item['call_id'] / 'request.json'
                old = read(request_path)
                request = {**case, 'messages': old['normalized_messages'], 'tools': old['tools']}
                date = re.search(r'Current date: (\d{4}-\d\d-\d\d)', old['rendered_prompt']).group(1)
                with np.load(request_path.with_name('input.npz'), allow_pickle=False) as data:
                    ids = data['input_ids'].reshape(-1).tolist()
                if not exposed and save_input(case, record, request, old['rendered_prompt'], date, ids, True):
                    exposed = True
            record['payload_exposed'] = exposed
        elif ref_id in refcases:
            result = refcases[ref_id]
            assert result['status'] == 'complete' and type(result['security']) is bool
            record.update(asr_source='historical_reference', attacker_success=result['security'], utility=result['utility'],
                          historical_selection=result.get('selection'), historical_exposure_conditioned=result.get('exposure_conditioned',False))
            source_record = REFERENCE / result['source_record']['path']
            record['sources'].append(copy_source(source_record, directory / 'case.json', result['source_record']['sha256']))
            trace = REFERENCE / result['raw_trace']['path']
            record['sources'].append(copy_source(trace, directory / 'trace.json', result['raw_trace']['sha256']))
            assert read(trace)['security'] == read(source_record)['security'] == result['security']
            exposed = False
            for item in sorted(result['requests'], key=lambda x:x['call_index']):
                for artifact in item['artifacts']:
                    name = Path(artifact['path']).name
                    record['sources'].append(copy_source(REFERENCE / artifact['path'], directory / f"call_{item['call_index']:04d}_{name}", artifact['sha256']))
                request_path = REFERENCE / next(a['path'] for a in item['artifacts'] if a['path'].endswith('/request.json'))
                old = read(request_path)
                request = {**case, 'messages': old['normalized_messages'], 'tools': old['tools']}
                date = re.search(r'Current date: (\d{4}-\d\d-\d\d)', old['rendered_prompt']).group(1)
                with np.load(request_path.with_name('input.npz'), allow_pickle=False) as data:
                    ids = data['input_ids'].reshape(-1).tolist()
                if not exposed and save_input(case, record, request, old['rendered_prompt'], date, ids, True):
                    exposed = True
            record['payload_exposed'] = exposed
        elif cid in classic:
            info = classic[cid]
            path = Path(info['path'])
            result = read(path)
            assert result['error'] is None and result['injections'] == case['injections']
            record.update(asr_source='historical_table1', attacker_success=result['security'], utility=result['utility'],
                          reconstruction_note='Dialogue copied verbatim; tools from recorded benchmark v1.2.1; header date from evaluation_timestamp; medium reasoning. Original server token IDs unavailable.')
            record['sources'].append(copy_source(path, directory / 'trace.json'))
            tools = [_function_to_openai(f) for f in FunctionsRuntime(suites[case['suite']].tools).functions.values()]
            messages = []
            for raw in result['messages']:
                raw = dict(raw)
                if raw.get('tool_calls'):
                    raw['tool_calls'] = [FunctionCall.model_validate(v) for v in raw['tool_calls']]
                if raw.get('tool_call'):
                    raw['tool_call'] = FunctionCall.model_validate(raw['tool_call'])
                m = _message_to_openai(raw, 'gpt-oss-120b')
                if isinstance(m.get('content'), list):
                    m['content'] = ''.join(b['text'] for b in m['content'])
                if m.get('content') is None:
                    m['content'] = ''
                for call in m.get('tool_calls', []):
                    if isinstance(call['function']['arguments'],str):
                        call['function']['arguments'] = json.loads(call['function']['arguments'])
                messages.append(m)
            date = result['evaluation_timestamp'][:10]
            exposed = False
            for i,m in enumerate(messages):
                if m['role'] != 'assistant' or i == 0 or messages[i-1]['role'] != 'tool':
                    continue
                request = {**case, 'messages': messages[:i], 'tools': tools}
                prompt = render(tok,request,date)
                if save_input(case,record,request,prompt,date):
                    exposed = True
                    break
            record['payload_exposed'] = exposed
        elif origin['status'] == 'unexposed':
            path = BASE / origin['source_result']
            result = read(path)
            assert type(result['security']) is bool
            record.update(asr_source='current_completed_unexposed', attacker_success=result['security'], utility=result['utility'], payload_exposed=False)
            record['sources'].append(copy_source(path, directory / 'case.json'))
        else:
            record.update(asr_source='needs_native_continuation', payload_exposed=True,
                          feature_method='reuse_current_native_states')
            entry = features[cid]
            record.update(states=str(BASE / entry['states']), states_sha256=entry['states_sha256'], rows=entry['rows'])
        assert type(record['utility']) is bool or record['asr_source']=='needs_native_continuation'
        counts[record['asr_source']] += 1
        counts['feature:' + str(record['feature_method'])] += 1
        plan.append(record)
        if (index+1)%100 == 0:
            print(json.dumps({'prepared':index+1,'counts':dict(counts)}),flush=True)
    missing = [r['case_id'] for r in plan if r['asr_source']=='needs_native_continuation']
    write(ROOT / 'plan.json', plan)
    write(ROOT / 'missing_case_ids.json', missing)
    write(ROOT / 'manifest.json', {'format':'agentdojo-log-first-merge-v1','total_cases':len(plan),
          'schedule_sha256':sha(BASE/'schedule.json'),'base_manifest_sha256':sha(BASE/'manifest.json'),
          'plan_sha256':sha(ROOT/'plan.json'),'missing_case_ids_sha256':sha(ROOT/'missing_case_ids.json'),
          'system_order':'original normal','counts':dict(counts), 'missing_asr_cases':len(missing),
          'selection_priority':['current_completed_native','historical_native_complete','historical_reference','historical_table1','current_completed_unexposed','needs_native_continuation'],
          'selection_uses_outcome':False,'feature_heads':protocol['head_ids_final_v7_16'],
          'feature_head_ids_union':protocol['head_ids_union'], 'state_dtype':'bfloat16',
          'delimiter_ids':protocol['delimiter_ids'],
          'source_code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
    print(json.dumps({'status':'prepared','cases':len(plan),'counts':dict(counts),'missing':len(missing)}),flush=True)


if __name__=='__main__':
    main()
