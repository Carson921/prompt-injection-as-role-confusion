"""Recover trusted outer tool bodies without parsing injected Harmony frames."""
import json
import re
import numpy as np

BEGIN, END = '\ue000FIRST_TOOL_BEGIN\ue001', '\ue000FIRST_TOOL_END\ue001'


def require(value, message):
    if not value:
        raise ValueError(message)


def flexible_text(text):
    # Exact policy from delimeter-token/role_probe_consolidation/build.py.
    parts, previous_space = [], False
    for character in text:
        if character.isspace():
            if not previous_space:
                parts.append(r'(?:\s|\\+[nrt])+')
            previous_space = True
        else:
            previous_space = False
            parts.append("'{1,2}" if character == "'" else re.escape(character))
    return ''.join(parts)


def render(tok, request, date):
    return tok.apply_chat_template(request['messages'], tools=request['tools'] or None,
            tokenize=False, add_generation_prompt=True, reasoning_effort='medium',
            strftime_now=lambda fmt: date)


def tool_spans(tok, request, prompt, date):
    require(BEGIN not in prompt and END not in prompt, 'Sentinel collision')
    count = 0

    def mark(match):
        nonlocal count
        expression = match.group()
        if 'message.content|tojson' not in expression:
            return expression
        require('" to=assistant<|channel|>commentary<|message|>"' in expression
                and expression.count('<|message|>') == expression.count('<|end|>') == 1,
                'Unexpected native tool template')
        count += 1
        return expression.replace('<|message|>', '<|message|>' + BEGIN).replace('<|end|>', END + '<|end|>')

    template = re.sub(r'{{[\s\S]*?}}', mark, tok.chat_template)
    require(count == 1, 'Missing unique official tool expression')
    marked = tok.apply_chat_template(request['messages'], tools=request['tools'] or None,
            chat_template=template, tokenize=False, add_generation_prompt=True, reasoning_effort='medium',
            strftime_now=lambda fmt: date)
    require(marked.replace(BEGIN, '').replace(END, '') == prompt, 'Instrumented rendering differs')
    spans, pos, cursor, pending = [], 0, 0, None
    for m in re.finditer(re.escape(BEGIN) + '|' + re.escape(END), marked):
        pos += m.start() - cursor
        if m.group() == BEGIN:
            require(pending is None, 'Nested outer body')
            pending = pos
        else:
            require(pending is not None, 'Unmatched outer end')
            spans.append((pending, pos))
            pending = None
        cursor = m.end()
    require(pending is None, 'Unclosed outer tool')
    messages = [(i, m) for i, m in enumerate(request['messages']) if m['role'] == 'tool']
    require(len(spans) == len(messages), 'Outer message/span count differs')
    for (start, end), (_, message) in zip(spans, messages):
        require(prompt[start:end] == json.dumps(message['content'], ensure_ascii=False),
                'Outer body differs from full original JSON tool string')
    return [(a, b, index, message) for (a, b), (index, message) in zip(spans, messages)]


def locate(tok, request, prompt, date):
    if not any(m['role'] == 'tool' for m in request['messages']):
        return None
    prefix, suffix = request['payload_template'].split('{goal}')
    pattern = re.compile(flexible_text(prefix) + '(?P<goal>' + flexible_text(request['actual_goal']) + ')'
                         + flexible_text(suffix))
    for start, end, index, message in tool_spans(tok, request, prompt, date):
        match = pattern.search(message['content'])
        if match is None:
            continue
        mapping, used = [], 0
        for character in message['content']:
            part = json.dumps(character, ensure_ascii=False)[1:-1]
            mapping.append((used, used + len(part)))
            used += len(part)

        def mapped(a, b):
            return [start + 1 + mapping[a][0], start + 1 + mapping[b - 1][1]]

        return {'tool_char_span': [start, end], 'tool_message_index': index,
                'tool_name': message.get('name'), 'payload_char_span': mapped(*match.span()),
                'goal_char_span': mapped(*match.span('goal'))}
    return None


def overlap(offsets, span):
    offsets = np.asarray(offsets)
    return (offsets[:, 1] > span[0]) & (offsets[:, 0] < span[1]) & (offsets[:, 1] > offsets[:, 0])


def select_rows(ids, offsets, location, delimiter_ids):
    body = overlap(offsets, location['tool_char_span'])
    delimiters = np.isin(ids, delimiter_ids)
    positions = np.flatnonzero(body & ~delimiters)
    payload = overlap(offsets, location['payload_char_span'])[positions]
    goal = overlap(offsets, location['goal_char_span'])[positions]
    require(len(positions) > 0 and goal.any() and (payload & ~goal).any(), 'Empty required region')
    require(not np.any(goal & ~payload), 'Goal outside payload')
    return positions, payload, goal, np.flatnonzero(body & delimiters)
