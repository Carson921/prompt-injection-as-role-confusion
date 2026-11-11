"""Generate, validate and de-duplicate tools for the 53 domains (paper Figure 2 prompts).

Writes <output>/tools.json: {domain: [tool, ...]} with name/description/parameters/response_schema.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from common import DOMAINS, LLM, read_json, write_json

SYSTEM = 'You design realistic API tool definitions for AI assistants. Reply with JSON only.'

TOOL_PROMPT = ('Generate {n} realistic API tool definitions for the "{domain}" domain. For each tool, provide:\n'
               '- name: A snake_case function name (e.g., get_weather_forecast)\n'
               '- description: A clear 1-3 sentence description of what the tool does\n'
               '- parameters: A JSON Schema object describing the tool\'s input parameters, including types, '
               'descriptions, and which are required\n'
               '- response_schema: A JSON Schema describing the expected return value\n'
               'Output as a JSON object {{"tools": [...]}}. Make the tools diverse -- include CRUD operations, '
               'search/filter, status checks, and action triggers where appropriate. Every tool must have between '
               '1 and 6 parameters; use only string, number, integer and boolean parameter types (no nested objects or arrays).')

DEDUP_PROMPT = ('You are comparing two tool definitions within the same domain.\nTool A:\n- name: {a_name}\n'
                '- description: {a_desc}\n- parameters: {a_params}\nTool B:\n- name: {b_name}\n- description: {b_desc}\n'
                '- parameters: {b_params}\nAre these two tools functionally overlapping -- i.e., would a user query '
                'plausibly match both tools equally well? Output JSON: {{"overlapping": true/false, "reason": "..."}}')

QUALITY_PROMPT = ('Evaluate the following API tool definition for quality.\nTool name: {name}\nDescription: {desc}\n'
                  'Parameters: {params}\nCheck the following:\n1. Does the tool name clearly reflect what it does?\n'
                  '2. Is the description clear and informative?\n3. Do the parameter names and types make sense for this tool?\n'
                  '4. Are there any obvious missing required parameters?\n5. Is this a plausible real-world API?\n'
                  'Output JSON: {{"valid": true/false, "reason": "..."}}')

SIMPLE_TYPES = {'string', 'number', 'integer', 'boolean'}


def well_formed(tool: dict) -> bool:
    if not isinstance(tool, dict) or not re.fullmatch(r'[a-z][a-z0-9_]{2,40}', str(tool.get('name', ''))):
        return False
    params = tool.get('parameters')
    if not isinstance(params, dict) or params.get('type') != 'object' or not isinstance(params.get('properties'), dict):
        return False
    props = params['properties']
    if not 1 <= len(props) <= 6:
        return False
    for name, spec in props.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,40}', name) or not isinstance(spec, dict) or spec.get('type') not in SIMPLE_TYPES:
            return False
    required = params.get('required', [])
    return isinstance(required, list) and set(required) <= set(props) and isinstance(tool.get('description'), str) \
        and len(tool['description']) > 10


def process_domain(llm: LLM, domain: str, n: int) -> dict:
    reply = llm.chat_json(SYSTEM, TOOL_PROMPT.format(n=n, domain=domain))
    raw = reply.get('tools', reply) if isinstance(reply, dict) else reply
    tools = [t for t in raw if well_formed(t)]
    kept, dropped = [], []
    for tool in tools:
        verdict = llm.chat_json(SYSTEM, QUALITY_PROMPT.format(name=tool['name'], desc=tool['description'],
                                                               params=json.dumps(tool['parameters'])))
        if verdict.get('valid') is True:
            kept.append(tool)
        else:
            dropped.append(dict(tool=tool['name'], reason=verdict.get('reason')))
    final = []
    for tool in kept:
        duplicate = None
        for other in final:
            verdict = llm.chat_json(SYSTEM, DEDUP_PROMPT.format(a_name=other['name'], a_desc=other['description'],
                                                                 a_params=json.dumps(other['parameters']),
                                                                 b_name=tool['name'], b_desc=tool['description'],
                                                                 b_params=json.dumps(tool['parameters'])))
            if verdict.get('overlapping') is True:
                duplicate = other['name']
                break
        if duplicate:
            dropped.append(dict(tool=tool['name'], reason=f'overlaps {duplicate}'))
        else:
            final.append(tool)
    return dict(domain=domain, tools=final, dropped=dropped, raw=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-domain', type=int, default=6)
    parser.add_argument('--model', default='google/gemini-2.5-flash')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'tools.json'
    done = read_json(path) if path.exists() else {}
    llm = LLM(args.output / 'api_calls.jsonl', model=args.model)
    domains = [d for ds in DOMAINS.values() for d in ds if d not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(lambda d: process_domain(llm, d, args.per_domain), domains):
            done[result['domain']] = result
            write_json(path, done)
            print(json.dumps(dict(domain=result['domain'], kept=len(result['tools']), raw=result['raw'],
                                  dropped=[x['tool'] for x in result['dropped']], spent=round(llm.spent, 3))), flush=True)
    print(json.dumps(dict(domains=len(done), tools=sum(len(v['tools']) for v in done.values()), spent=round(llm.spent, 3))))


if __name__ == '__main__':
    main()
