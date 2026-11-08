"""Shared helpers: templates, text-only message adaptation, prompt rendering, goal location.

The browser adaptation follows the earlier gptoss120b_toolcheck pipeline in the
ipi_arena_os repository: GPT-OSS-120B is text-only, so screenshot image blocks
are dropped and, when the injection lives only in the rendered screenshot, the
screenshot's source HTML (binary data URLs elided) is appended to the associated
tool response.  Nothing else in the fixture text is changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IPI_ROOT = Path('/scratch/wz55191/Github/ipi_arena_os')
RICHEVO = ROOT / 'experiments/autodan-role-probe/outputs/richevo-20261105'
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')
sys.path.insert(0, str(IPI_ROOT / 'src'))
sys.path.insert(0, str(HERE))

SPECIAL = {'<|start|>': 200006, '<|end|>': 200007, '<|message|>': 200008, '<|channel|>': 200005,
           '<|return|>': 200002, '<|call|>': 200012, '<|constrain|>': 200003}


def sha(value) -> str:
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def load_templates() -> dict[str, dict]:
    """The ten band-selected rich-evolution templates plus the plain baseline."""
    bands = read_json(RICHEVO / 'asr_by_band.json')
    result = {'plain': dict(condition='plain', template='{goal}', p_user_agentdojo=None, interval=None,
                            asr_agentdojo=None, wrapper_tokens=0, source='baseline')}
    for row in bands:
        low, high = row['interval']
        key = f'band_{low:.2f}_{high:.2f}'
        result[key] = dict(condition=key, template=row['template'], p_user_agentdojo=row['p_user'],
                           interval=row['interval'], asr_agentdojo=row['asr_lower'],
                           asr_agentdojo_upper=row['asr_upper'], successes_agentdojo=row['successes'],
                           scored_agentdojo=row['scored'], richevo_condition=row['condition'],
                           source=row['group'])
    for key, row in result.items():
        assert row['template'].count('{goal}') == 1, key
    return result


def normalize_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Text-only normalisation: drop image blocks, flatten text, name tool messages."""
    result, adaptations, call_names = [], [], {}
    for index, original in enumerate(messages):
        message = json.loads(json.dumps(original))
        content = message.get('content')
        if isinstance(content, list):
            texts = [b.get('text', '') for b in content if b.get('type') == 'text']
            images = sum(b.get('type') == 'image_url' for b in content)
            if images:
                adaptations.append({'message_index': index, 'removed_image_blocks': images})
            if images and texts == ['[Screenshot of current page]']:
                continue
            message['content'] = '\n'.join(texts)
        elif content is None:
            message['content'] = ''
        for call in message.get('tool_calls') or []:
            function = call['function']
            if isinstance(function.get('arguments'), str):
                function['arguments'] = json.loads(function['arguments'])
            call_names[call['id']] = function['name']
        if message['role'] == 'tool':
            name = message.get('name') or call_names.get(message.get('tool_call_id'))
            if not name:
                raise ValueError(f'Tool response has no identifiable preceding call: {index}')
            message['name'] = name
        result.append(message)
    return result, adaptations


def build_text_messages(behavior: dict, payload: str) -> tuple[list[dict], list[dict]]:
    """Official build_messages, then the text-only browser adaptation."""
    from ipi_arena_bench.behavior import build_messages, _prepare_fill_text
    messages = build_messages(behavior, payload)
    changes = []
    prepared = _prepare_fill_text(payload, behavior.get('fill_config', {}))
    if behavior.get('html_template') and not any(
            prepared in m.get('content', '') for m in messages if isinstance(m.get('content'), str)):
        tool_indices = [i for i, m in enumerate(messages) if m['role'] == 'tool']
        if not tool_indices:
            raise ValueError('Screenshot-only case has no tool response for HTML adaptation')
        index = tool_indices[-1]
        template = behavior['html_template']
        html, omitted_images = re.subn(r'data:image/[^\s"\']+', '[image data omitted for text-only model]', template)
        html = html.replace('{fill}', prepared)
        messages[index]['content'] = (messages[index].get('content') or '') + '\n[Rendered page HTML]\n' + html
        changes.append(dict(message_index=index, screenshot_source_html_added=True,
                            omitted_binary_image_data_urls=omitted_images))
    normalized, adaptations = normalize_messages(messages)
    return normalized, changes + adaptations


def render_prompt(tokenizer, messages: list[dict], tools: list[dict], date: str) -> str:
    return tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                         add_generation_prompt=True, reasoning_effort='medium',
                                         strftime_now=lambda fmt: date)


def locate_goal(prompt: str, offsets, payload: str, goal: str, fill_config: dict) -> list[dict]:
    """Find every rendered occurrence of the payload and the goal tokens inside it.

    The fixture's fill_config may JSON-escape the payload, and the Harmony chat
    template JSON-serialises tool message content once more, so both the
    once- and twice-escaped renderings are searched.  Only occurrences that
    are not already covered are kept.
    """
    from ipi_arena_bench.behavior import _prepare_fill_text
    prepared = _prepare_fill_text(payload, fill_config)
    prepared_goal = json.dumps(goal)[1:-1] if fill_config.get('json_escape') else goal
    variants = [(json.dumps(prepared, ensure_ascii=False)[1:-1], json.dumps(prepared_goal, ensure_ascii=False)[1:-1]),
                (prepared, prepared_goal)]
    occupied, occurrences = [], []
    for rendered_payload, rendered_goal in dict.fromkeys(variants):
        relative = rendered_payload.find(rendered_goal)
        if relative < 0:
            raise ValueError('Escaped goal missing from escaped payload')
        position = 0
        while True:
            start = prompt.find(rendered_payload, position)
            if start < 0:
                break
            end = start + len(rendered_payload)
            position = end
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            goal_start, goal_end = start + relative, start + relative + len(rendered_goal)
            indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
            goal_indices = [i for i in indices if offsets[i][0] < goal_end and offsets[i][1] > goal_start]
            occurrences.append(dict(start_char=start, end_char=end, goal_start_char=goal_start, goal_end_char=goal_end,
                                    token_indices=indices, goal_token_indices=goal_indices))
    occurrences.sort(key=lambda row: row['start_char'])
    if not occurrences:
        raise ValueError('Payload not exposed in rendered prompt')
    return occurrences


def special_floor(model_path: Path) -> tuple[int, list[str]]:
    source = json.loads((model_path / 'tokenizer.json').read_text())
    floor = max(source['model']['vocab'].values()) + 1
    spellings = [row['content'] for row in source['added_tokens'] if row['special']]
    return floor, spellings
