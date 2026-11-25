"""Scoring contexts for the Qwen3.5 role probe, built from real Qwen3.5 AgentDojo episodes.

A context is the exact request of the first model call that saw the injection (system, user,
the model's own tool calls and the tool results).  For a new template the recorded tool calls are
re-executed in a freshly injected AgentDojo environment, so every tool result is formatted by
AgentDojo itself (YAML quoting, folding and escaping included) rather than by string surgery.
"""
from __future__ import annotations

from ast import literal_eval
import json
import re

from ds_common import SLOT, TOOL_CLOSE, TOOL_OPEN


def flexible_pattern(text):
    """Regex for ``text`` as it may appear inside a YAML-dumped tool result (copy of the
    richevo locator v2: folded/escaped whitespace, doubled quotes, backslash and \\u escapes)."""
    parts, run = [], []

    def flush():
        if not run:
            return
        alternatives = [r'\s', r'\\+[nrtfv]', r'\\[ \t]', r'\\\r?\n[ \t]*', r'\\_', r'\\N', r'\\L', r'\\P']
        for character in sorted(set(run)):
            number = ord(character)
            alternatives += [re.escape(f'\\u{number:04x}'), re.escape(f'\\u{number:04X}'),
                             re.escape(f'\\U{number:08x}'), re.escape(f'\\U{number:08X}')]
            if number <= 255:
                alternatives += [re.escape(f'\\x{number:02x}'), re.escape(f'\\x{number:02X}')]
        parts.append('(?:' + '|'.join(alternatives) + ')+')
        run.clear()

    for character in text:
        if character.isspace():
            run.append(character)
            continue
        flush()
        if character == "'":
            parts.append(r"(?:'{2}|\\*')")
        elif character == '"':
            parts.append(r'\\*"')
        elif character == '\\':
            parts.append(r'\\+')
        else:
            number = ord(character)
            choices = [re.escape(character), re.escape(f'\\u{number:04x}'), re.escape(f'\\U{number:08x}'),
                       re.escape(f'\\u{number:04X}'), re.escape(f'\\U{number:08X}')]
            if number <= 255:
                choices += [re.escape(f'\\x{number:02x}'), re.escape(f'\\x{number:02X}')]
            parts.append('(?:' + '|'.join(choices) + ')')
    flush()
    return r'(?:\\\r?\n[ \t]*)?'.join(parts)


def payload_regex(template, goal):
    prefix, suffix = template.split(SLOT)
    fold = r'(?:\\\r?\n[ \t]*)?'
    return re.compile(flexible_pattern(prefix) + fold + '(?P<goal>' + flexible_pattern(goal) + ')' + fold + flexible_pattern(suffix))


def first_exposure(record):
    """Index of the first model call whose request already contains the goal inside a tool result."""
    goal = re.compile(flexible_pattern(record['goal']))
    for index, call in enumerate(record['calls']):
        request = record['transcripts'][call['query_index']][:call['n_messages']]
        if any(m['role'] == 'tool' and goal.search(m['content'] or '') for m in request):
            return index
    return None


def replay(deps, suite, context, template):
    """Re-run the recorded tool calls with ``template`` injected; returns the request messages."""
    from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
    from agentdojo.agent_pipeline.tool_execution import is_string_list
    payload = template.replace(SLOT, context['goal'])
    from ds_agentdojo import escape_for_environment
    environment = suite.load_and_inject_default_environment(
        {v: escape_for_environment(payload) for v in context['injection_vectors']})
    user_task = suite.user_tasks[context['user_task_id']]
    environment = user_task.init_environment(environment)
    runtime = deps['FunctionsRuntime'](suite.tools)
    names = {tool.name for tool in runtime.functions.values()}
    messages, pending = [], []
    for original in context['messages']:
        message = dict(original)
        if message['role'] == 'assistant':
            pending = []
            for call in message.get('tool_calls') or []:
                name, arguments = call['function']['name'], json.loads(call['function']['arguments'])
                if name == EMPTY_FUNCTION_NAME:
                    pending.append('Empty function name provided. Provide a valid function name.')
                    continue
                if name not in names:
                    pending.append(f'Invalid tool {name} provided.')
                    continue
                for key, value in arguments.items():
                    if isinstance(value, str) and is_string_list(value):
                        arguments[key] = literal_eval(value)
                result, error = runtime.run_function(environment, name, arguments)
                pending.append(error if error else deps['formatter'](result))
        elif message['role'] == 'tool':
            message['content'] = pending.pop(0)
        messages.append(message)
    assert not pending
    return messages, payload


def locate(codec, messages, tools, template, goal):
    """Render the request and return token ids plus goal / payload token indices of the first payload."""
    encoded = codec.encode(messages, tools)
    prompt, ids, offsets = encoded['prompt'], encoded['input_ids'], encoded['offsets']
    regex = payload_regex(template, goal)
    cursor = 0
    for message in messages:
        if message['role'] != 'tool':
            continue
        content = message['content'] or ''
        # the chat template trims each tool result and wraps it as <tool_response>\nBODY\n</tool_response>
        body = content.strip()
        start = prompt.find(TOOL_OPEN + '\n' + body + '\n' + TOOL_CLOSE, cursor)
        if start < 0:
            raise ValueError('Tool result is not verbatim in the rendered prompt')
        start += len(TOOL_OPEN) + 1
        cursor = start + len(body)
        content = body
        match = regex.search(content)
        if not match:
            continue
        payload_span = (start + match.start(), start + match.end())
        goal_span = (start + match.start('goal'), start + match.end('goal'))
        inside = lambda span: [i for i, (a, b) in enumerate(offsets) if b > a and a < span[1] and b > span[0]]
        payload_indices, goal_indices = inside(payload_span), inside(goal_span)
        if not goal_indices:
            raise ValueError('Goal has no tokens')
        return dict(prompt=prompt, input_ids=ids, payload_token_indices=payload_indices, goal_token_indices=goal_indices,
                    payload_char_span=list(payload_span), goal_char_span=list(goal_span))
    raise ValueError('No complete payload with the goal inside a tool result')


def context_from_record(record):
    index = first_exposure(record)
    if index is None:
        return None
    return dict(case_id=record['base_case_id'], suite=record['suite'], user_task_id=record['user_task_id'],
                injection_task_id=record['injection_task_id'], goal=record['goal'],
                injection_vectors=record['injection_vectors'], tools=record['tools'],
                messages=record['transcripts'][record['calls'][index]['query_index']][:record['calls'][index]['n_messages']],
                exposure_call=index,
                source_condition=record['condition'])
