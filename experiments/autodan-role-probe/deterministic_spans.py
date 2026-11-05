"""Recover payload spans through native tool serialization without regex search.

The native prompt is never changed.  Each character of a decoded string field
is mapped to the characters that encode it in YAML, Python repr, or plain text;
that map is then composed with the outer JSON tool-body encoding.  Recipe paths
select the actual replaced fields, so quoted JSON in a payload is ordinary data.
"""
from __future__ import annotations

import ast
import json


def require(value, message):
    if not value:
        raise ValueError(message)


def _get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _occurrences(text, needle):
    """Non-overlapping, complete semantic occurrences, matching str.replace."""
    require(bool(needle), 'Empty payload is not permitted')
    cursor = 0
    while True:
        index = text.find(needle, cursor)
        if index < 0:
            return
        yield index, index + len(needle)
        cursor = index + len(needle)


def _python_string(value, start):
    encoded = repr(value)
    quote = encoded[0]
    require(quote in ('\"', "'") and encoded[-1] == quote,
            'Unexpected native Python string representation')
    parts, origins, cursor = [], [], start + 1
    for char in value:
        if char in ('\\', quote):
            part = '\\' + char
        else:
            part = repr(char)[1:-1]
        parts.append(part)
        origins.append((cursor, cursor + len(part)))
        cursor += len(part)
    require(''.join(parts) == encoded[1:-1], 'Python repr character reconstruction differs')
    return encoded, origins


def _repr_fields(text):
    """Rebuild canonical native repr and retain path-indexed string origins."""
    tree = ast.literal_eval(text)
    fields, chunks, cursor = {}, [], 0

    def append(value):
        nonlocal cursor
        chunks.append(value)
        cursor += len(value)

    def visit(value, path):
        if isinstance(value, str):
            encoded, origins = _python_string(value, cursor)
            fields[tuple(path)] = (value, origins)
            append(encoded)
        elif isinstance(value, dict):
            append('{')
            for index, (key, child) in enumerate(value.items()):
                if index:
                    append(', ')
                append(repr(key))
                append(': ')
                visit(child, path + [key])
            append('}')
        elif isinstance(value, (list, tuple)):
            append('[' if isinstance(value, list) else '(')
            for index, child in enumerate(value):
                if index:
                    append(', ')
                visit(child, path + [index])
            if isinstance(value, tuple) and len(value) == 1:
                append(',')
            append(']' if isinstance(value, list) else ')')
        else:
            # Native scalar leaves (numbers, None, bool, bytes) have no string
            # payload slots.  Their exact canonical repr is still verified.
            append(repr(value))

    visit(tree, [])
    require(''.join(chunks) == text, 'Native Python repr round trip differs')
    return tree, fields


_BREAKS = '\r\n\x85\u2028\u2029'


def _line_break(text, index):
    start = index
    char = text[index]
    require(char in _BREAKS, 'Expected YAML line break')
    index += 1
    if char == '\r' and index < len(text) and text[index] == '\n':
        index += 1
    return ('\n' if char in '\r\n\x85' else char), (start, index), index


def _yaml_flow_string(text, start, end, style):
    """Decode a quoted/plain scalar with the scanner's deterministic folding."""
    import yaml.scanner

    quoted = style in ("'", '\"')
    double = style == '\"'
    index, limit = (start + 1, end - 1) if quoted else (start, end)
    require(not quoted or (text[start] == style and text[end - 1] == style),
            'YAML quote marks differ from scalar style')
    chars, origins = [], []

    def emit(value, interval):
        for char in value:
            chars.append(char)
            origins.append(interval)

    def continuation_breaks(position):
        """Discard indentation, preserving the additional physical breaks."""
        output = []
        while position < limit:
            while position < limit and text[position] in ' \t':
                position += 1
            if position < limit and text[position] in _BREAKS:
                value, interval, position = _line_break(text, position)
                output.append((value, interval))
            else:
                break
        return output, position

    while index < limit:
        char = text[index]
        if quoted and not double and char == "'":
            require(index + 1 < limit and text[index + 1] == "'", 'Unescaped YAML single quote')
            emit("'", (index, index + 2))
            index += 2
        elif double and char == '\\':
            begin = index
            index += 1
            require(index < limit, 'Incomplete YAML escape')
            char = text[index]
            if char in yaml.scanner.Scanner.ESCAPE_REPLACEMENTS:
                emit(yaml.scanner.Scanner.ESCAPE_REPLACEMENTS[char], (begin, index + 1))
                index += 1
            elif char in yaml.scanner.Scanner.ESCAPE_CODES:
                count = yaml.scanner.Scanner.ESCAPE_CODES[char]
                require(index + 1 + count <= limit, 'Incomplete YAML Unicode escape')
                emit(chr(int(text[index + 1:index + 1 + count], 16)), (begin, index + 1 + count))
                index += 1 + count
            elif char in _BREAKS:
                _, _, index = _line_break(text, index)
                breaks, index = continuation_breaks(index)
                for value, interval in breaks:
                    emit(value, interval)
            else:
                raise ValueError('Unknown YAML escape in native scalar')
        elif char in ' \t' + _BREAKS:
            begin = index
            while index < limit and text[index] in ' \t':
                index += 1
            if index < limit and text[index] in _BREAKS:
                first, first_interval, index = _line_break(text, index)
                breaks, index = continuation_breaks(index)
                if first != '\n':
                    emit(first, (begin, index if not breaks else first_interval[1]))
                elif not breaks:
                    emit(' ', (begin, index))
                for offset, (value, interval) in enumerate(breaks):
                    # Attribute discarded first break/indentation to the first
                    # preserved break; the decoded scalar remains exact.
                    emit(value, (begin if offset == 0 and first == '\n' else interval[0],
                                 index if offset == len(breaks) - 1 else interval[1]))
            else:
                for offset in range(begin, index):
                    emit(text[offset], (offset, offset + 1))
        else:
            emit(char, (index, index + 1))
            index += 1
    return ''.join(chars), origins


def _yaml_block_string(text, node, start=None, end=None):
    """Map block scalar content; native safe_dump normally uses flow strings.

    The YAML parser has already established exact scalar semantics.  Lexical
    non-whitespace characters occur in order; folding/chomping affects only
    whitespace, whose origins are aligned in one pass below.
    """
    start = node.start_mark.index if start is None else start
    end = node.end_mark.index if end is None else end
    header_end = text.find('\n', start, end)
    require(header_end >= 0, 'Block scalar has no header line break')
    lines = text[header_end + 1:end].splitlines(keepends=True)
    nonempty = [line for line in lines if line.rstrip('\r\n \t\x85\u2028\u2029')]
    explicit = [int(char) for char in text[start + 1:header_end].split('#', 1)[0] if char.isdigit()]
    if explicit:
        line_start = text.rfind('\n', 0, start) + 1
        prefix = text[line_start:start]
        indent = len(prefix) - len(prefix.lstrip(' ')) + explicit[0]
    else:
        indent = min((len(line) - len(line.lstrip(' ')) for line in nonempty), default=0)
    chars, origins, cursor = [], [], header_end + 1
    for line in lines:
        remove = min(indent, len(line) - len(line.lstrip(' ')))
        for offset, char in enumerate(line[remove:], remove):
            chars.append(char)
            origins.append((cursor + offset, cursor + offset + 1))
        cursor += len(line)
    return _align_whitespace(node.value, ''.join(chars), origins)


def _align_whitespace(expected, decoded, origins):
    """Linear fallback only for parser-verified YAML whitespace folding.

    Exact decoded node equality is checked by the caller.  Non-whitespace must
    agree byte for byte; each complete whitespace run retains all source
    origins.  This cannot create matches by dropping quotes or braces.
    """
    target, source, result = 0, 0, []
    while target < len(expected):
        if expected[target].isspace():
            target_end = target
            while target_end < len(expected) and expected[target_end].isspace():
                target_end += 1
            source_end = source
            while source_end < len(decoded) and decoded[source_end].isspace():
                source_end += 1
            require(source_end > source, 'YAML whitespace has no lexical origin')
            interval = (origins[source][0], origins[source_end - 1][1])
            result.extend([interval] * (target_end - target))
            target, source = target_end, source_end
        else:
            # Block chomping can discard only leading/trailing whitespace, not
            # non-whitespace; permit a leading empty block line if necessary.
            if target == 0:
                while source < len(decoded) and decoded[source].isspace():
                    source += 1
            require(source < len(decoded) and expected[target] == decoded[source],
                    'YAML lexical content differs from decoded scalar')
            result.append(origins[source])
            target += 1
            source += 1
    require(all(char.isspace() for char in decoded[source:]), 'Unmapped YAML lexical content')
    return expected, result


def _yaml_fields(text, paths):
    import yaml
    from yaml.nodes import ScalarNode, SequenceNode, MappingNode
    from yaml.tokens import ScalarToken

    loader = yaml.SafeLoader(text)
    try:
        root = loader.get_single_node()
        tree = loader.construct_object(root, deep=True)
        # A ScalarNode's start can include an anchor/tag.  ScalarToken marks
        # delimit the actual lexical scalar and also handle aliases correctly.
        tokens = {token.end_mark.index: token for token in yaml.scan(text, Loader=yaml.SafeLoader)
                  if isinstance(token, ScalarToken)}
        fields = {}
        for path in paths:
            node = root
            for key in path:
                if isinstance(node, SequenceNode):
                    require(isinstance(key, int) and 0 <= key < len(node.value), 'Invalid YAML sequence path')
                    node = node.value[key]
                elif isinstance(node, MappingNode):
                    matches = [value for key_node, value in node.value
                               if loader.construct_object(key_node, deep=True) == key]
                    require(len(matches) == 1, 'YAML mapping path is not unique')
                    node = matches[0]
                else:
                    raise ValueError('YAML recipe path does not select a scalar field')
            actual = _get(tree, path)
            require(isinstance(node, ScalarNode) and isinstance(actual, str)
                    and node.tag == 'tag:yaml.org,2002:str' and node.value == actual,
                    'YAML scalar semantics differ from actual native field')
            token = tokens.get(node.end_mark.index)
            require(token is not None and token.value == actual and token.style == node.style,
                    'YAML scalar token differs from composed string field')
            if node.style in ('|', '>'):
                decoded, origins = _yaml_block_string(text, node, token.start_mark.index, token.end_mark.index)
            else:
                decoded, origins = _yaml_flow_string(text, token.start_mark.index, token.end_mark.index, node.style)
                if decoded != actual:
                    decoded, origins = _align_whitespace(actual, decoded, origins)
            require(decoded == actual and len(origins) == len(actual), 'YAML string origin reconstruction differs')
            fields[tuple(path)] = (actual, origins)
        return tree, fields
    finally:
        loader.dispose()


def _field_maps(text, formatter, paths):
    if formatter == 'plain':
        require(all(not path for path in paths), 'Plain recipe must select the whole string')
        return text, {(): (text, [(index, index + 1) for index in range(len(text))])}
    if formatter == 'repr':
        return _repr_fields(text)
    if formatter == 'yaml':
        return _yaml_fields(text, paths)
    raise ValueError(f'Unknown native formatter: {formatter}')


def locate_payload(tokenizer, spans_module, request, prompt, date, allowed_messages=None, context=None):
    """Return exact first/all complete payload spans in trusted native fields.

    ``context`` is the frozen source context whose slot_recipes rendered this
    request.  For backwards-compatible plain bodies, omitting recipes searches
    whole tool strings literally.  Structured recipes are required to invert
    structured serialization safely.
    """
    template, goal = request['payload_template'], request['actual_goal']
    require(isinstance(template, str) and template.count('{goal}') == 1,
            'A candidate must contain exactly one immutable goal slot')
    prefix, suffix = template.split('{goal}')
    require(isinstance(goal, str) and bool(goal), 'Missing immutable complete goal')
    payload = prefix + goal + suffix
    require(request.get('payload_text', payload) == payload, 'Request payload differs from immutable template/goal')
    recipes = (context or {}).get('slot_recipes') or request.get('slot_recipes')
    by_message = {}
    if recipes:
        for recipe in recipes:
            index = recipe['message_index']
            require(index not in by_message, 'Duplicate native recipe message')
            by_message[index] = recipe
    found, proofs = [], []
    for start, end, index, message in spans_module.tool_spans(tokenizer, request, prompt, date):
        if allowed_messages is not None and index not in allowed_messages:
            continue
        if recipes and index not in by_message:
            continue
        text = message['content']
        require(isinstance(text, str), 'Native tool body must be a string')
        recipe = by_message.get(index)
        slots = recipe['slots'] if recipe else [{'path': []}]
        formatter = recipe['formatter'] if recipe else 'plain'
        _, fields = _field_maps(text, formatter, [slot['path'] for slot in slots])
        outer_map, used = [], 0
        for char in text:
            part = json.dumps(char, ensure_ascii=False)[1:-1]
            outer_map.append((used, used + len(part)))
            used += len(part)
        require(prompt[start:end] == json.dumps(text, ensure_ascii=False), 'Outer JSON tool body differs')

        def mapped(origins, begin, finish):
            require(0 <= begin < finish <= len(origins), 'Empty or invalid semantic span')
            a, b = origins[begin][0], origins[finish - 1][1]
            require(0 <= a < b <= len(text), 'Invalid native serialization origin span')
            return [start + 1 + outer_map[a][0], start + 1 + outer_map[b - 1][1]]

        for slot in slots:
            path = tuple(slot['path'])
            require(path in fields, 'Native recipe string field not found')
            value, origins = fields[path]
            matches = list(_occurrences(value, payload))
            if 'occurrences' in slot:
                require(len(matches) == slot['occurrences'], 'Complete payload occurrence count differs from recipe')
            require(matches, 'No complete payload in native recipe field')
            for begin, finish in matches:
                goal_begin, goal_end = begin + len(prefix), begin + len(prefix) + len(goal)
                require(value[begin:finish] == payload and value[goal_begin:goal_end] == goal,
                        'Immutable payload/goal semantic validation differs')
                found.append({'tool_char_span': [start, end], 'tool_message_index': index,
                              'tool_name': message.get('name'),
                              'payload_char_span': mapped(origins, begin, finish),
                              'goal_char_span': mapped(origins, goal_begin, goal_end),
                              'serialization_proof': {'method': 'deterministic-native-string-origins-v1',
                                                      'formatter': formatter, 'field_path': list(path),
                                                      'decoded_payload_exact': True, 'decoded_goal_exact': True,
                                                      'decoded_field_length': len(value),
                                                      'semantic_payload_span': [begin, finish],
                                                      'semantic_goal_span': [goal_begin, goal_end]}})
            proofs.append({'tool_message_index': index, 'formatter': formatter,
                           'field_path': list(path), 'complete_occurrences': len(matches),
                           'decoded_field_verified': True})
    require(found, 'No complete payload plus immutable goal in a trusted original tool field')
    found.sort(key=lambda row: (row['payload_char_span'][0], row['goal_char_span'][0]))
    return {**found[0], 'occurrences': found,
            'mapping_proof': {'format': 'deterministic-native-string-origins-v1',
                              'no_regex_payload_matching': True, 'native_prompt_unchanged': True,
                              'recipe_fields': proofs, 'complete_occurrences': len(found)}}
