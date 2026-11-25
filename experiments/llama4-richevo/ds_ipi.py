"""IPI Arena (ipi_arena_os) for Llama-4-Scout-17B-16E-Instruct: prepare requests, run tool-check ASR.

Reuses the GPT-OSS study's goal table, text-only browser adaptation, five-step runner, Gemini
WorldSim and deterministic tool judge (experiments/ipi-arena-richevo).  Llama 4 specifics:
* prompts are rendered by the checkpoint's chat template (ds_common.Codec) and sent to vLLM as token ids;
* the fixture's JSON escaping of the fill keeps non-ASCII characters literal (json.dumps(..., ensure_ascii=False)),
  the same rule as the DeepSeek run;
* the chat template JSON-encodes every tool result, so a payload inside a tool message is escaped once more.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

from ds_common import Codec, GENERATION, MODEL, SERVED_MODEL, SLOT, read_json, sha, write_json

HERE = Path(__file__).resolve().parent
IPI_EXPERIMENT = HERE.parent / 'ipi-arena-richevo'
IPI_ROOT = Path('/scratch/wz55191/Github/ipi_arena_os')


def ipi_modules():
    sys.path.insert(0, str(IPI_ROOT / 'src'))
    sys.path.insert(0, str(IPI_EXPERIMENT))
    import ipi_arena_bench.behavior as behavior

    def prepare_fill_text(attack_text, fill_config):
        if fill_config.get('json_escape', False):
            escaped = json.dumps(attack_text, ensure_ascii=False)
            return escaped[1:-1] if fill_config.get('fill_in_quotes', False) else escaped
        return attack_text

    behavior._prepare_fill_text = prepare_fill_text
    import common
    import goals
    return behavior, common, goals, prepare_fill_text


def encoder_messages(messages):
    """Normalised fixture messages (tool-call arguments as objects) -> official encoder input."""
    result = []
    for message in messages:
        row = dict(role=message['role'], content=message.get('content') or '')
        if message.get('tool_calls'):
            row['tool_calls'] = [dict(id=c['id'], type='function', function=dict(
                name=c['function']['name'],
                arguments=c['function']['arguments'] if isinstance(c['function']['arguments'], str)
                else json.dumps(c['function']['arguments'], ensure_ascii=False))) for c in message['tool_calls']]
        if message['role'] == 'tool':
            row['tool_call_id'] = message.get('tool_call_id', '')
        result.append(row)
    return result


def locate(prompt, offsets, prepared, prepared_goal):
    """Every rendered payload occurrence (verbatim, or JSON-escaped once or twice more) with its goal tokens."""
    escape = lambda text: json.dumps(text, ensure_ascii=False)[1:-1]
    variants = [(prepared, prepared_goal), (escape(prepared), escape(prepared_goal)),
                (escape(escape(prepared)), escape(escape(prepared_goal)))]
    occupied, occurrences = [], []
    for rendered, rendered_goal in dict.fromkeys(variants):
        relative = rendered.find(rendered_goal)
        if relative < 0:
            raise ValueError('Escaped goal missing from escaped payload')
        position = 0
        while True:
            start = prompt.find(rendered, position)
            if start < 0:
                break
            end = position = start + len(rendered)
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            goal_start, goal_end = start + relative, start + relative + len(rendered_goal)
            indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
            occurrences.append(dict(start_char=start, end_char=end, goal_start_char=goal_start, goal_end_char=goal_end,
                                    token_indices=indices,
                                    goal_token_indices=[i for i in indices if offsets[i][0] < goal_end and offsets[i][1] > goal_start]))
    occurrences.sort(key=lambda row: row['start_char'])
    if not occurrences:
        raise ValueError('Payload not exposed in rendered prompt')
    return occurrences


def prepare(args):
    behavior_module, common, goals_module, prepare_fill_text = ipi_modules()
    output = args.output.resolve()
    codec = Codec(MODEL)
    behaviors = goals_module.load_behaviors()
    goals, skipped = goals_module.build_goal_manifest(behaviors)
    conditions = read_json(args.conditions_json)
    write_json(output / 'goals.json', dict(goals=goals, skipped=skipped))
    write_json(output / 'conditions.json', conditions)
    rows = []
    for condition, spec in conditions.items():
        template = spec['template']
        assert template.count(SLOT) == 1, condition
        if not spec.get('allow_special') and codec.has_special(template.replace(SLOT, ' ')):
            raise ValueError(f'Template contains a genuine special token: {condition}')
        for case_id, entry in sorted(goals.items()):
            behavior = behaviors[case_id]
            goal, fill_config = entry['goal'], behavior.get('fill_config', {})
            payload = template.replace(SLOT, goal)
            messages, adaptations = common.build_text_messages(behavior, payload)
            tools = behavior_module.build_tools(behavior)
            encoded = codec.encode(encoder_messages(messages), tools)
            prepared = prepare_fill_text(payload, fill_config)
            prepared_goal = json.dumps(goal, ensure_ascii=False)[1:-1] if fill_config.get('json_escape') else goal
            occurrences = locate(encoded['prompt'], encoded['offsets'], prepared, prepared_goal)
            ids = encoded['input_ids']
            payload_indices = sorted({i for o in occurrences for i in o['token_indices']})
            specials = sorted({ids[i] for i in payload_indices if ids[i] in codec.special_ids})
            if specials and not spec.get('allow_special'):
                raise ValueError(f'Rendered payload contains special ids {specials}: {condition}/{case_id}')
            request = dict(request_id=f'{condition}/{case_id}', condition=condition, case_id=case_id,
                           category=behavior['category'], primary=entry['primary'], messages=messages, tools=tools,
                           adaptations=adaptations, payload=payload, goal=goal, template=template, fill_config=fill_config,
                           rendered_prompt=encoded['prompt'], prompt_token_ids=ids, payload_occurrences=occurrences,
                           payload_token_indices=payload_indices, goal_token_indices=occurrences[0]['goal_token_indices'],
                           payload_special_ids=specials, scored_occurrence=0, prompt_tokens=len(ids),
                           fits_context=len(ids) + GENERATION['max_tokens'] <= args.max_model_len)
            path = output / 'requests' / condition / f'{case_id}.json'
            write_json(path, request)
            rows.append(dict(condition=condition, case_id=case_id, category=behavior['category'], primary=entry['primary'],
                             prompt_tokens=len(ids), payload_tokens=len(payload_indices),
                             goal_tokens=len(request['goal_token_indices']), payload_occurrences=len(occurrences),
                             payload_special_ids=specials, fits_context=request['fits_context'], adaptations=adaptations,
                             request=str(path.relative_to(output)), request_sha256=sha(path)))
    protocol = dict(schema='ipi-arena-llama4-scout-v1', model_path=str(MODEL), generation=GENERATION, max_model_len=args.max_model_len,
                    conditions=list(conditions), behaviors=len(goals), primary=sum(g['primary'] for g in goals.values()),
                    skipped=skipped, codec=codec.binding, fill_escaping='json.dumps(ensure_ascii=False) for json_escape fixtures',
                    data_sha256={str(p.relative_to(IPI_ROOT)): sha(p) for p in sorted((IPI_ROOT / 'data').glob('*/*.json'))},
                    rows=rows)
    write_json(output / 'protocol.json', protocol)
    print(json.dumps(dict(requests=len(rows), fit=sum(r['fits_context'] for r in rows),
                          max_prompt_tokens=max(r['prompt_tokens'] for r in rows),
                          goal_tokens_mean=sum(r['goal_tokens'] for r in rows) / len(rows),
                          special_by_condition={c: sum(bool(r['payload_special_ids']) for r in rows if r['condition'] == c)
                                                for c in conditions}), ensure_ascii=False))


class LlamaTarget:
    """Same interface as run_asr.Target, backed by the chat-template Codec and vLLM completions."""
    keep_reasoning = False  # no reasoning channel

    def __init__(self, base_url, model, max_model_len):
        from ds_serving import Vllm
        self.codec = Codec(MODEL)
        self.client = Vllm(base_url, model, max_model_len)
        self.model = model

    def tokenize(self, messages, tools):
        return self.codec.encode(encoder_messages(messages), tools)['input_ids']

    def chat(self, messages, tools):
        encoded = self.codec.encode(encoder_messages(messages), tools)
        response = self.client.complete(encoded['input_ids'], self.codec.stop_ids)
        text, finish, calls, malformed = response['text'], response['finish_reason'], [], None
        try:
            if finish != 'stop':
                raise ValueError(f'generation ended with finish_reason={finish}')
            parsed = self.codec.parse(text, tools)
            content, reasoning = parsed.get('content') or '', ''
            for index, call in enumerate(parsed.get('tool_calls') or []):
                raw = call['function']['arguments']
                arguments = json.loads(raw)
                if not isinstance(arguments, dict):
                    arguments = {'_raw': raw}
                calls.append(dict(id='call_' + sha(text)[:16] + f'_{index}', name=call['function']['name'],
                                  arguments=arguments, raw_arguments=raw))
        except (ValueError, AssertionError, KeyError, TypeError) as error:
            malformed = f'{type(error).__name__}: {str(error)[:200]}'
            content, reasoning, calls = text, '', []
        return dict(content=content, reasoning=reasoning, tool_calls=calls,
                    finish_reason='length' if finish == 'length' else ('malformed' if malformed else finish),
                    usage=dict(prompt_tokens=len(encoded['input_ids']),
                               completion_tokens=(response.get('usage') or {}).get('completion_tokens')),
                    model=self.model, malformed=malformed, completion=text)


def run(args):
    ipi_modules()
    import run_asr
    output = args.output.resolve()
    protocol = read_json(output / 'protocol.json')
    behavior_module, common, goals_module, _ = ipi_modules()
    behaviors = goals_module.load_behaviors()
    from ipi_arena_bench.llm_client import LLMClient
    worldsim = LLMClient(provider='openrouter', model=args.worldsim_model, api_key=run_asr.openrouter_key())
    target = LlamaTarget(args.base_url, args.served_model, args.max_model_len)
    settings = dict(protocol_sha256=sha(output / 'protocol.json'), target=args.served_model, worldsim=args.worldsim_model,
                    generation=GENERATION, worldsim_max_tokens=run_asr.WORLDSIM_MAX_TOKENS, max_steps=run_asr.MAX_STEPS,
                    worldsim_block_fallback='{"status": "success"}', keep_reasoning=False)
    identity = sha(json.dumps(settings, sort_keys=True))
    write_json(output / 'asr_identity.json', dict(settings, identity_sha256=identity,
                                                  judge='ipi_arena_bench.judges.tool_usage.judge_tool_usage only'))
    pending = []
    for row in protocol['rows']:
        if args.conditions and row['condition'] not in args.conditions:
            continue
        case_path = output / 'cases' / row['condition'] / f"{row['case_id']}.json"
        if case_path.exists():
            existing = read_json(case_path)
            if existing.get('status') == 'complete' and existing.get('identity_sha256') == identity:
                continue
        request = read_json(output / row['request'])
        request['_sha256'] = row['request_sha256']
        pending.append(request)
    print(json.dumps(dict(stage='start', pending=len(pending), workers=args.workers)), flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_asr.run_episode, r, behaviors[r['case_id']], target, worldsim, output, identity) for r in pending]
        for future in as_completed(futures):
            record = future.result()
            done += 1
            print(json.dumps(dict(stage='case', done=done, total=len(pending), condition=record['condition'],
                                  case_id=record['case_id'], status=record['status'], success=record.get('success'),
                                  steps=record.get('n_steps'), tokenize_match=record.get('tokenize', {}).get('match'),
                                  error=record.get('error'), seconds=round(record['duration_seconds'], 1))), flush=True)
    summary = run_asr.summarize(output, protocol)
    print(json.dumps({k: dict(asr=v['asr'], complete=v['complete'], errors=v['errors']) for k, v in summary['conditions'].items()}), flush=True)


def probe(args):
    from ds_scorer import Prober
    probe_requests(args.output.resolve(), Prober(args.output.resolve() / 'probe_backend'))


def probe_requests(output, prober):
    import numpy as np
    from ds_common import ROLES
    protocol = read_json(output / 'protocol.json')
    write_json(output / 'probe_protocol.json', prober.protocol)
    results = []
    for index, row in enumerate(protocol['rows']):
        destination = output / 'probe' / row['condition'] / f"{row['case_id']}.json"
        if destination.exists():
            existing = read_json(destination)
            if existing.get('status') == 'complete' and existing.get('request_sha256') == row['request_sha256']:
                results.append(existing)
                continue
        request = read_json(output / row['request'])
        started = time.monotonic()
        occurrence = request['payload_occurrences'][request['scored_occurrence']]
        positions = np.asarray(occurrence['token_indices'], dtype=np.int64)
        goal_mask = np.isin(positions, np.asarray(occurrence['goal_token_indices'], dtype=np.int64))
        probabilities = prober.probabilities(request['prompt_token_ids'], positions)
        means = dict(goal=dict(zip(ROLES, map(float, probabilities[goal_mask].mean(0, dtype=np.float64)))))
        if (~goal_mask).any():
            means['template'] = dict(zip(ROLES, map(float, probabilities[~goal_mask].mean(0, dtype=np.float64))))
        archive = destination.with_suffix('.npz')
        archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, payload_token_indices=positions, goal_mask=goal_mask, probabilities=probabilities)
        record = dict(status='complete', condition=row['condition'], case_id=row['case_id'], category=row['category'],
                      primary=row['primary'], request_sha256=row['request_sha256'], input_tokens=len(request['prompt_token_ids']),
                      goal_tokens=int(goal_mask.sum()), payload_tokens=int(len(positions)),
                      goal_user_mean=means['goal']['user'], region_means=means, elapsed_seconds=time.monotonic() - started)
        write_json(destination, record)
        results.append(record)
        print(json.dumps(dict(stage='probe', index=index + 1, total=len(protocol['rows']), condition=row['condition'],
                              case_id=row['case_id'], p_user=round(record['goal_user_mean'], 4),
                              seconds=round(record['elapsed_seconds'], 1))), flush=True)
    summary = {}
    for condition in protocol['conditions']:
        primary = [r for r in results if r['condition'] == condition and r['primary']]
        summary[condition] = dict(
            p_user_mean=float(np.mean([r['goal_user_mean'] for r in primary])), cases=len(primary),
            goal_role_means={role: float(np.mean([r['region_means']['goal'][role] for r in primary])) for role in ROLES},
            per_category={c: float(np.mean([r['goal_user_mean'] for r in primary if r['category'] == c]))
                          for c in ('browser', 'coding', 'tool')})
    write_json(output / 'probe_summary.json', summary)
    print(json.dumps({k: round(v['p_user_mean'], 4) for k, v in summary.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'probe'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions-json', type=Path)
    parser.add_argument('--conditions', nargs='*')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--served-model', default=SERVED_MODEL)
    parser.add_argument('--max-model-len', type=int, default=65536)
    parser.add_argument('--worldsim-model', default='google/gemini-2.5-flash')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    dict(prepare=prepare, run=run, probe=probe)[args.command](args)


if __name__ == '__main__':
    main()
