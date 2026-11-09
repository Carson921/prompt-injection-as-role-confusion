"""Build every initial request (condition x behavior) on the CPU and freeze a protocol.

Writes <output>/goals.json, <output>/conditions.json, <output>/requests/<cond>/<case>.json
and <output>/protocol.json.  No model is loaded; only the tokenizer.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re

from common import (IPI_ROOT, MODEL, ROOT, SPECIAL, build_text_messages, load_templates, locate_goal,
                    read_json, render_prompt, sha, special_floor, write_json)
from goals import build_goal_manifest, load_behaviors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--date', default='2026-11-08')
    parser.add_argument('--max-model-len', type=int, default=131072)
    parser.add_argument('--conditions-json', type=Path, default=None, help='Custom {name: {template, ...}} instead of the band templates')
    args = parser.parse_args()
    output = args.output.resolve()
    from transformers import AutoTokenizer
    from ipi_arena_bench.behavior import build_tools
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for text, token_id in SPECIAL.items():
        assert tokenizer.encode(text, add_special_tokens=False) == [token_id], text
    floor, spellings = special_floor(args.model)
    behaviors = load_behaviors()
    goals, skipped = build_goal_manifest(behaviors)
    conditions = read_json(args.conditions_json) if args.conditions_json else load_templates()
    write_json(output / 'goals.json', dict(goals=goals, skipped=skipped))
    write_json(output / 'conditions.json', conditions)
    rows = []
    for condition, spec in conditions.items():
        template = spec['template']
        for case_id, goal_entry in sorted(goals.items()):
            behavior = behaviors[case_id]
            goal = goal_entry['goal']
            payload = template.replace('{goal}', goal)
            if any(s in payload for s in spellings) or re.search(r'<\|reserved_\d+\|>', payload):
                raise ValueError(f'Payload contains a real special token spelling: {condition}')
            payload_ids = tokenizer.encode(payload, add_special_tokens=False)
            if any(t >= floor for t in payload_ids):
                raise ValueError(f'Payload tokenises to a special token id: {condition}')
            messages, adaptations = build_text_messages(behavior, payload)
            tools = build_tools(behavior)
            prompt = render_prompt(tokenizer, messages, tools, args.date)
            encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded['input_ids'], encoded['offset_mapping']
            occurrences = locate_goal(prompt, offsets, payload, goal, behavior.get('fill_config', {}))
            payload_indices = sorted({i for o in occurrences for i in o['token_indices']})
            if any(ids[i] >= floor for i in payload_indices):
                raise ValueError(f'Rendered payload contains a special token id: {condition}/{case_id}')
            goal_indices = occurrences[0]['goal_token_indices']
            if not goal_indices:
                raise ValueError(f'Goal has no tokens: {condition}/{case_id}')
            request = dict(request_id=f'{condition}/{case_id}', condition=condition, case_id=case_id,
                           category=behavior['category'], primary=goal_entry['primary'], messages=messages, tools=tools,
                           adaptations=adaptations, payload=payload, goal=goal, template=template,
                           fill_config=behavior.get('fill_config', {}), rendered_prompt=prompt,
                           prompt_token_ids=ids, token_char_offsets=[list(x) for x in offsets],
                           payload_occurrences=occurrences, payload_token_indices=payload_indices,
                           goal_token_indices=goal_indices, scored_occurrence=0, prompt_tokens=len(ids),
                           prompt_sha256=sha(prompt), input_ids_sha256=sha(json.dumps(ids, separators=(',', ':'))),
                           fits_context=len(ids) + 4096 <= args.max_model_len)
            path = output / 'requests' / condition / f'{case_id}.json'
            write_json(path, request)
            rows.append(dict(condition=condition, case_id=case_id, category=behavior['category'],
                             primary=goal_entry['primary'], prompt_tokens=len(ids),
                             payload_tokens=len(payload_indices), goal_tokens=len(goal_indices),
                             payload_occurrences=len(occurrences), fits_context=request['fits_context'],
                             adaptations=adaptations, request=str(path.relative_to(output)),
                             request_sha256=sha(path)))
    protocol = dict(schema='ipi-arena-richevo-v1', model_path=str(args.model), prompt_date=args.date,
                    reasoning_effort='medium', max_model_len=args.max_model_len,
                    conditions=list(conditions), behaviors=len(goals), primary=sum(g['primary'] for g in goals.values()),
                    skipped=skipped, delimiter_tokens=SPECIAL, special_id_floor=floor,
                    data_sha256={str(p.relative_to(IPI_ROOT)): sha(p) for p in sorted((IPI_ROOT / 'data').glob('*/*.json'))},
                    tokenizer_sha256={n: sha(args.model / n) for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')},
                    code_sha256={n: sha(Path(__file__).with_name(n)) for n in ('common.py', 'goals.py', 'prepare.py')},
                    asr_by_band_sha256=sha(ROOT / 'experiments/autodan-role-probe/outputs/richevo-20261105/asr_by_band.json'),
                    versions={n: importlib.metadata.version(n) for n in ('transformers', 'tokenizers')},
                    rows=rows)
    write_json(output / 'protocol.json', protocol)
    fits = [r for r in rows if r['fits_context']]
    print(json.dumps(dict(requests=len(rows), fit=len(fits), max_prompt_tokens=max(r['prompt_tokens'] for r in rows),
                          over_context=sorted({r['case_id'] for r in rows if not r['fits_context']}),
                          goal_tokens_mean=sum(r['goal_tokens'] for r in rows) / len(rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
