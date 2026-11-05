"""Audit every case/call and export raw float32 inputs in both frozen head orders."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from spans import render, locate, select_rows, require
from outer_frames import independent_frames_from_messages


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write(p, v):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def verify(root, model, *, partial=False, export=False):
    torch.set_num_threads(1)
    m, schedule = read(root / 'manifest.json'), read(root / 'schedule.json')
    require(sha(root / 'scope.json') == m['scope_sha256'] and sha(root / 'schedule.json') == m['schedule_sha256']
            and sha(root / 'agentdojo-source.json') == m['source_sha256'], 'Frozen scope changed')
    binding = signature(m)
    require(read(root / 'validation.json')['status'] == 'passed', 'Native validation missing')
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    counts, entries, errors, missing, rows, calls = Counter(), [], [], [], 0, 0
    cache_verified = set()
    for case_index, case in enumerate(schedule):
        path = root / 'cases' / (case['case_id'] + '.json')
        if not path.exists():
            missing.append(case['case_id'])
            continue
        result = read(path)
        require(result['case_id'] == case['case_id'] and result['status'] in ('captured', 'unexposed', 'error'), 'Bad terminal case')
        counts[(case['condition'], case['suite'], result['status'])] += 1
        if result['status'] == 'error':
            errors.append({'case_id': case['case_id'], 'error': result['error']})
        observed = []
        for i, request_row in enumerate(result['requests']):
            call_id = case['case_id'] + f'/call_{i:04d}'
            require(request_row['call_id'] == call_id and request_row['call_index'] == i, 'Call ordering differs')
            directory = root / 'calls' / call_id
            raw, inp = read(directory / 'original_request.json'), read(directory / 'input.json')
            require(sha(directory / 'original_request.json') == request_row['request_sha256'], 'Original request changed')
            require(all(raw[k] == v for k, v in case.items()), 'Case/payload identity changed')
            prompt = render(tok, raw, m['generation']['prompt_date'])
            require(prompt == inp['rendered_prompt'] and inp['binding'] == binding, 'Full native rendered input differs')
            enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            require(enc['input_ids'] == inp['input_ids'] and enc['offset_mapping'] == [tuple(x) for x in inp['token_char_offsets']],
                    'Whole original input retokenization differs')
            location = locate(tok, raw, prompt, m['generation']['prompt_date'])
            require(location == inp['location'], 'Trusted first-exposure location differs')
            independent = independent_frames_from_messages({'messages': raw['messages'],
                'normalized_messages': raw['messages'], 'rendered_prompt': prompt,
                'tools': raw['tools'], 'generation': m['generation']}, tok)
            if location:
                require(tuple(location['tool_char_span']) in {(x['start'], x['end']) for x in independent if x['tag'] == 'tool'},
                        'Independent sequential outer-message reconstruction differs')
            calls += 1
            if location:
                observed.append(i)
                require(result['status'] == 'captured' and i == len(result['requests']) - 1, 'Episode continued after capture')
                receipt = read(directory / 'receipt.json')
                require(receipt['binding'] == binding and receipt['case_id'] == case['case_id']
                        and receipt['actual_root_input_ids_verified'] and receipt['all_selected_layer_hooks_seen'], 'Capture binding missing')
                require(all(sha(directory / name) == digest for name, digest in receipt['files'].items()), 'Capture checksum differs')
                positions, payload, goal, excluded = select_rows(enc['input_ids'], enc['offset_mapping'], location, m['delimiter_ids'])
                with safe_open(str(directory / 'states.safetensors'), framework='pt', device='cpu') as s:
                    features = s.get_tensor('features_union')
                    require(features.dtype == torch.bfloat16 and features.shape == (len(positions), len(m['head_ids_union']), 64)
                            and bool(torch.isfinite(features).all()), 'Feature shape/dtype/finite check failed')
                    expected = {'positions': positions, 'token_ids': np.asarray(enc['input_ids'])[positions],
                        'char_offsets': np.asarray(enc['offset_mapping'])[positions], 'payload_mask': payload,
                        'goal_mask': goal, 'template_mask': payload & ~goal, 'context_mask': ~payload,
                        'excluded_delimiter_positions': excluded}
                    for name, value in expected.items():
                        require(np.array_equal(s.get_tensor(name).numpy(), value), f'Incorrect complete tool token selection: {name}')
                    require(s.metadata()['binding'] == binding and json.loads(s.metadata()['head_ids_union']) == m['head_ids_union'],
                            'Tensor head order binding differs')
                require(receipt['tool_rows'] == len(positions), 'Receipt row count differs')
                entries.append({'case_index': case_index, 'case_id': case['case_id'], 'call_id': call_id,
                    'condition': case['condition'], 'suite': case['suite'], 'rows': len(positions),
                    'row_start': rows, 'row_end': rows + len(positions), 'location': location,
                    'states': str((directory / 'states.safetensors').relative_to(root)),
                    'states_sha256': sha(directory / 'states.safetensors')})
                rows += len(positions)
            else:
                require(not (directory / 'states.safetensors').exists(), 'Hidden states saved for an unexposed input')
                response_path = directory / 'response.json'
                if not response_path.exists():
                    require(result['status'] == 'error' and i == len(result['requests']) - 1, 'Missing prefix response')
                    continue
                response = read(response_path)
                require(sha(response_path) == request_row['response_sha256'], 'Recorded native response changed')
                if 'parsed_response_sha256' in request_row:
                    parsed_path = directory / 'parsed_response.json'
                    parsed = read(parsed_path)
                    require(sha(parsed_path) == request_row['parsed_response_sha256']
                            and parsed['native_generated_ids'] == response['generated_ids']
                            and parsed['source_response_sha256'] == sha(response_path), 'Native parser response binding differs')
                cache_path = root / 'prefix_cache' / (response['cache_key'] + '.json')
                require(sha(cache_path) == response['cache_sha256'], 'Native prefix cache changed')
                cache = read(cache_path)
                require(cache['binding'] == binding and cache['input_ids'] == enc['input_ids']
                        and cache['generated_ids'] == response['generated_ids'], 'Prefix input/generation differs')
                require(signature({'binding': binding, 'input_ids': enc['input_ids']}) == response['cache_key'], 'Incorrect exact input cache key')
                require(tok.decode(response['generated_ids'], skip_special_tokens=False) == response['generated_text'], 'Raw native generation differs')
                cache_verified.add(response['cache_key'])
        require(len(observed) == int(result['status'] == 'captured'), 'First-exposure accounting differs')
        if (case_index + 1) % 100 == 0:
            print(json.dumps({'audited_cases': case_index + 1, 'rows': rows}), flush=True)
    if not partial:
        require(not missing, f'{len(missing)} cases missing')
    report = {'status': 'partial_verified' if missing else 'full_scope_verified', 'expected_cases': len(schedule),
        'terminal_cases': len(schedule) - len(missing), 'captured_cases': len(entries), 'rows': rows,
        'calls': calls, 'unique_native_prefixes': len(cache_verified), 'missing_cases': missing, 'errors': errors,
        'counts': [{'condition': c, 'suite': s, 'status': t, 'cases': n} for (c, s, t), n in sorted(counts.items())],
        'binding': binding, 'every_actual_input_retokenized': True, 'earliest_exposure_verified': True,
        'entire_trusted_outer_tool_body_verified': True, 'all_saved_features_finite': True,
        'all_delimiters_excluded_and_other_tool_tokens_retained': True, 'verified_at_unix': time.time()}
    write(root / 'audit.json', report)
    write(root / 'feature_index.json', entries)
    if export:
        require(not missing, 'Refusing incomplete export')
        out = root / 'probe_inputs'
        out.mkdir(exist_ok=True)
        shapes = {}
        for label, head_key in [('final_v7_16', 'head_ids_final_v7_16'), ('reference_10', 'head_ids_reference_10')]:
            indices = [m['head_ids_union'].index(h) for h in m[head_key]]
            shape = (rows, len(indices) * 64)
            x = np.lib.format.open_memmap(out / f'X_{label}.npy', mode='w+', dtype=np.float32, shape=shape)
            for entry in entries:
                with safe_open(str(root / entry['states']), framework='pt', device='cpu') as f:
                    block = f.get_tensor('features_union')[:, indices].reshape(entry['rows'], -1).float().numpy()
                x[entry['row_start']:entry['row_end']] = block
                require(np.array_equal(x[entry['row_start']:entry['row_end']], block), 'Export roundtrip differs')
            x.flush()
            shapes[label] = list(shape)
            del x
        aligned = {name: np.lib.format.open_memmap(out / (name + '.npy'), mode='w+', dtype=dtype, shape=(rows,))
            for name, dtype in [('token_ids', np.int64), ('positions', np.int64), ('case_index', np.int32),
                                ('payload_mask', bool), ('goal_mask', bool), ('template_mask', bool), ('context_mask', bool)]}
        for entry in entries:
            a, b = entry['row_start'], entry['row_end']
            with safe_open(str(root / entry['states']), framework='pt', device='cpu') as f:
                for name, arr in aligned.items():
                    arr[a:b] = entry['case_index'] if name == 'case_index' else f.get_tensor(name).numpy()
        for arr in aligned.values():
            arr.flush()
        write(out / 'manifest.json', {'status': 'complete', 'binding': binding, 'shapes': shapes,
            'raw_features': 'float32 exact BF16 conversion, unstandardized and unpooled',
            'head_ids_final_v7_16': m['head_ids_final_v7_16'], 'head_ids_reference_10': m['head_ids_reference_10'],
            'index': '../feature_index.json', 'expected_cases': len(schedule), 'captured_cases': len(entries),
            'unexposed_and_error_cases': '../audit.json', 'SHA256': {f.name: sha(f) for f in out.glob('*.npy')}})
        (out / 'USAGE.txt').write_text('X_final_v7_16.npy: raw N x 1024 float32 input in final-v7 ordered 16 heads.\n'
            'X_reference_10.npy: raw N x 640 float32 input in the reference dataset ordered 10 heads.\n'
            'np.load(path, mmap_mode="r", allow_pickle=False) reads each input. No standardization applied.\n'
            'All rows come from the WHOLE first-exposure trusted outer tool body, excluding native delimiters.\n'
            'payload_mask includes template and actual goal. goal_mask takes overlap priority.\n'
            'template_mask = payload_mask & ~goal_mask; context_mask = ~payload_mask.\n'
            'case_index joins schedule.json; feature_index.json provides per-case row ranges.\n'
            'Unexposed/error cases have no fabricated feature rows and remain in audit.json accounting.\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103'))
    p.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    p.add_argument('--partial', action='store_true')
    p.add_argument('--export', action='store_true')
    a = p.parse_args()
    verify(a.output.resolve(), a.model, partial=a.partial, export=a.export)
