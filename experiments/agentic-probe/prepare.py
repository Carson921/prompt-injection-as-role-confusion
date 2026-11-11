"""Build every initial request (condition x scenario) on the CPU and freeze a protocol.

The payload (template with {goal} = attacker instruction) is appended to a free-text string
field of the final task-domain tool response.  Writes <output>/{conditions,goals,protocol}.json
and <output>/requests/<cond>/<scenario>.json in the same shape as experiments/ipi-arena-richevo
so probe_score.py there can score the exact served prompt.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re

from common import MODEL, SPECIAL, locate_payload, read_json, render_prompt, sha, special_floor, write_json
from target import openai_tools

PREFERRED = ('notes', 'note', 'message', 'summary', 'description', 'status_message', 'details', 'comment', 'remarks')


def inject(content_json: str, payload: str) -> tuple[str, str]:
    """Append the payload to a string field of the JSON tool response; returns (new content, field)."""
    obj = json.loads(content_json)
    keys = [k for k, v in obj.items() if isinstance(v, str)]
    if keys:
        key = next((k for k in PREFERRED if k in keys), max(keys, key=lambda k: len(obj[k])))
        obj[key] = (obj[key].rstrip() + ' ' + payload) if obj[key].strip() else payload
    else:
        key = 'notes'
        obj[key] = payload
    return json.dumps(obj, ensure_ascii=False), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='dataset dir with dataset.json and trajectories.json')
    parser.add_argument('--conditions-json', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--date', default='2026-11-11')
    parser.add_argument('--max-model-len', type=int, default=131072)
    parser.add_argument('--system', default=None, help='override the system prompt (e.g. a prompt-level defense)')
    args = parser.parse_args()
    output = args.output.resolve()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for text, token_id in SPECIAL.items():
        assert tokenizer.encode(text, add_special_tokens=False) == [token_id], text
    floor, spellings = special_floor(args.model)
    dataset = {e['scenario_id']: e for e in read_json(args.dataset / 'dataset.json')['scenarios']}
    trajectories = read_json(args.dataset / 'trajectories.json')
    scenarios = {k: v for k, v in dataset.items() if trajectories.get(k, {}).get('status') == 'match'}
    conditions = read_json(args.conditions_json)
    goals = {k: dict(goal=e['attacker_goal'], target_call=e['target_call'], task_domain=e['task_domain'],
                     target_domain=e['target_domain'], category=e['task_category'], primary=True) for k, e in scenarios.items()}
    write_json(output / 'goals.json', dict(goals=goals, skipped=sorted(set(dataset) - set(scenarios))))
    write_json(output / 'conditions.json', conditions)
    rows = []
    for condition, spec in conditions.items():
        template = spec['template']
        for case_id, entry in sorted(scenarios.items()):
            goal = entry['attacker_goal']
            payload = template.replace('{goal}', goal)
            if any(s in payload for s in spellings) or re.search(r'<\|reserved_\d+\|>', payload):
                raise ValueError(f'Payload contains a real special token spelling: {condition}')
            if any(t >= floor for t in tokenizer.encode(payload, add_special_tokens=False)):
                raise ValueError(f'Payload tokenises to a special token id: {condition}')
            messages = json.loads(json.dumps(trajectories[case_id]['messages']))
            assert messages[-1]['role'] == 'tool'
            if args.system:
                assert messages[0]['role'] == 'system'
                messages[0]['content'] = args.system
            messages[-1]['content'], field = inject(messages[-1]['content'], payload)
            tools = openai_tools(entry['task_tools'] + entry['target_tools'])
            prompt = render_prompt(tokenizer, messages, tools, args.date)
            encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded['input_ids'], encoded['offset_mapping']
            occurrences = locate_payload(prompt, offsets, payload, goal)
            payload_indices = sorted({i for o in occurrences for i in o['token_indices']})
            if any(ids[i] >= floor for i in payload_indices):
                raise ValueError(f'Rendered payload contains a special token id: {condition}/{case_id}')
            goal_indices = occurrences[0]['goal_token_indices']
            if not goal_indices:
                raise ValueError(f'Goal has no tokens: {condition}/{case_id}')
            request = dict(request_id=f'{condition}/{case_id}', condition=condition, case_id=case_id,
                           category=entry['task_category'], primary=True, messages=messages, tools=tools,
                           injection_field=field, payload=payload, goal=goal, template=template,
                           target_call=entry['target_call'], rendered_prompt=prompt, prompt_token_ids=ids,
                           token_char_offsets=[list(x) for x in offsets], payload_occurrences=occurrences,
                           payload_token_indices=payload_indices, goal_token_indices=goal_indices, scored_occurrence=0,
                           prompt_tokens=len(ids), prompt_sha256=sha(prompt),
                           input_ids_sha256=sha(json.dumps(ids, separators=(',', ':'))),
                           fits_context=len(ids) + 4096 <= args.max_model_len)
            path = output / 'requests' / condition / f'{case_id}.json'
            write_json(path, request)
            rows.append(dict(condition=condition, case_id=case_id, category=entry['task_category'], primary=True,
                             prompt_tokens=len(ids), payload_tokens=len(payload_indices), goal_tokens=len(goal_indices),
                             payload_occurrences=len(occurrences), fits_context=request['fits_context'],
                             request=str(path.relative_to(output)), request_sha256=sha(path)))
    protocol = dict(schema='agentic-probe-benign-v1', model_path=str(args.model), prompt_date=args.date,
                    reasoning_effort='medium', max_model_len=args.max_model_len, conditions=list(conditions),
                    system_prompt_override=args.system,
                    scenarios=len(scenarios), dataset_sha256=sha(args.dataset / 'dataset.json'),
                    trajectories_sha256=sha(args.dataset / 'trajectories.json'), delimiter_tokens=SPECIAL, special_id_floor=floor,
                    tokenizer_sha256={n: sha(args.model / n) for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')},
                    code_sha256={n: sha(Path(__file__).with_name(n)) for n in ('common.py', 'target.py', 'prepare.py')},
                    versions={n: importlib.metadata.version(n) for n in ('transformers', 'tokenizers')}, rows=rows)
    write_json(output / 'protocol.json', protocol)
    print(json.dumps(dict(requests=len(rows), fit=sum(r['fits_context'] for r in rows),
                          max_prompt_tokens=max(r['prompt_tokens'] for r in rows),
                          goal_tokens_mean=sum(r['goal_tokens'] for r in rows) / len(rows),
                          multi_occurrence=sum(r['payload_occurrences'] > 1 for r in rows))))


if __name__ == '__main__':
    main()
