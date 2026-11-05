"""Review existing rejection diagnostics without changing any candidate/policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

import audit_strict_rejections_readonly as scan
from diverse_policy import _MARKUP, _ROLE_HEADER
from strict_header_policy import ROLES, normalized, strict_header_metrics

ROLE_SET = set(ROLES.split('|'))
BOUNDARIES = {'end', 'imend', 'return', 'call'}
KNOWN_NAMES = ROLE_SET | BOUNDARIES | {
    'start', 'imstart', 'begin', 'message', 'imsep', 'content', 'channel',
    'metasep', 'metaend', 'analysis', 'commentary', 'final', 'summary'}
JSON_ROLE = re.compile(r'''["']role["']\s*:\s*["']('''+ROLES+r''')["']''', re.I)
JSON_CONTENT_KEY = re.compile(r'''["']content["']\s*:''', re.I)
PLAIN = re.compile(r'(?m)^[ \t]*(?:#{1,6}[ \t]*)?('+ROLES+r')[ \t]*(?::|(?=\r?$))', re.I)


def marker_name(marker):
    return ''.join(c for c in unicodedata.normalize('NFKC', marker.group()).casefold() if c.isalpha())


def is_format_marker(marker):
    raw = marker.group()
    return (marker_name(marker) in KNOWN_NAMES and bool(
        re.fullmatch(r'[<>|/ \t\r\nA-Za-z_]+',raw))) or bool(
        re.fullmatch(r'</?[A-Za-z_][A-Za-z0-9_.:-]*\s*/?>', raw))


def remove_known(value):
    return _MARKUP.sub(lambda m: ' ' if is_format_marker(m) else m.group(), value)


def parsed_json_messages(template):
    def visit(node, result):
        if isinstance(node, dict):
            if isinstance(node.get('role'), str) and node['role'].casefold() in ROLE_SET:
                if not isinstance(node.get('content'), str):
                    return False
                value = node['content']
                result.append(dict(kind='json_role_header', role=node['role'].casefold(),
                                   body=normalized(value), body_has_goal='{goal}' in value))
            return all(visit(v, result) for v in node.values() if isinstance(v, (dict,list)))
        if isinstance(node, list):
            return all(visit(v, result) for v in node)
        return True
    try:
        document = json.loads(template)
        result = []
        if visit(document, result) and result:
            return result, 'complete_JSON_role_content'
    except (ValueError, TypeError):
        pass
    # Existing wrappers sometimes contain only closing punctuation before a
    # complete JSON messages object. Decode the unchanged embedded substring.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'[\[{]', template):
        prefix = remove_known(template[:match.start()])
        if any(c.isalnum() for c in prefix):
            continue
        try:
            document, end = decoder.raw_decode(template, match.start())
        except ValueError:
            continue
        if any(c.isalnum() for c in remove_known(template[end:])):
            continue
        result = []
        if visit(document, result) and result:
            return result, 'complete_embedded_JSON_with_only_format_prefix_suffix'
    return None, None


def parse_headers(template):
    messages, method = parsed_json_messages(template)
    if messages is not None:
        return messages, method, []
    if JSON_ROLE.search(template):
        return None, 'unresolved_JSON_role_content', []
    markers = list(_MARKUP.finditer(template))
    names = [marker_name(m) for m in markers]
    if any(name in ROLE_SET and not is_format_marker(marker)
           for marker,name in zip(markers,names)):
        return None, 'unrecognized_role_like_markup', []
    headers = []
    for i, (marker, name) in enumerate(zip(markers, names)):
        if name in ('start','imstart','begin'):
            for j in range(i+1, min(len(markers),i+8)):
                if names[j] in ('start','imstart','begin','end','imend','call','return'):
                    break
                if names[j] in ('message','imsep','content'):
                    label = normalized(remove_known(template[marker.end():markers[j].start()]))
                    words = label.split()
                    if words and words[0] in ROLE_SET and all(
                            v in ('analysis','commentary','final','summary') or re.fullmatch(r'to=\S+',v)
                            for v in words[1:]):
                        headers.append((marker.start(),markers[j].end(),'near_message_header',words[0]))
                    break
        if name in ROLE_SET and not marker.group().lstrip().startswith('</'):
            headers.append((marker.start(),marker.end(),'xml_role_header',name))
    for match in PLAIN.finditer(template):
        headers.append((match.start(),match.end(),'plain_role_header',match.group(1).casefold()))
    # The old line policy recognizes bracketed role-only labels too. Recover
    # those exact existing spans, rather than calling an unparsed label prose.
    offset = 0
    for line in template.splitlines(keepends=True):
        label = normalized(remove_known(line))
        if len(label) <= 32 and _ROLE_HEADER.fullmatch(label):
            words = re.findall(r'[A-Za-z]+', label.casefold())
            if len(words) == 1 and words[0] in ROLE_SET and not any(
                    start <= offset < end or offset <= start < offset+len(line)
                    for start,end,_,_ in headers):
                headers.append((offset,offset+len(line),'bracketed_or_role_only_line',words[0]))
        offset += len(line)
    headers.sort()
    if any(previous[1] > current[0] for previous,current in zip(headers,headers[1:])):
        return None, 'overlapping_header_spans', []
    result = []
    for i, (start,end,kind,role) in enumerate(headers):
        bound = headers[i+1][0] if i+1 < len(headers) else len(template)
        for marker in markers:
            if not (end <= marker.start() < bound):
                continue
            name = marker_name(marker)
            if name in BOUNDARIES or (name == role and marker.group().lstrip().startswith('</')):
                bound = marker.start()
                break
        body_raw = template[end:bound]
        assert end <= bound
        body = normalized(remove_known(body_raw))
        result.append(dict(kind=kind,role=role,body=body,body_has_goal='{goal}' in body_raw,
                           header_offset=[start,end],body_offset=[end,bound]))
    return result, 'recognized_header_span_with_closing_boundary', headers


def readable_repeated_units(template, strict, header_spans):
    # Preserve offsets while marking only proven formatting. This handles
    # duplicate lines that are fragments of one multiline existing marker.
    value = template.replace('{goal}','')
    mask = [False]*len(value)
    def mark(start,end):
        mask[start:end] = [True]*(end-start)
    for marker in _MARKUP.finditer(value):
        if is_format_marker(marker):
            mark(marker.start(),marker.end())
    for pattern in (JSON_ROLE, JSON_CONTENT_KEY, PLAIN):
        for match in pattern.finditer(value):
            mark(match.start(),match.end())
    # Header offsets above are from the full template. Reparse the slot-free
    # diagnostic string to avoid an offset shift after the immutable goal slot.
    _,_,spans = parse_headers(value)
    for start,end,_,_ in spans:
        mark(start,end)
    repeated = []
    for name, pattern in [('exact_duplicate_lines',r'[^\r\n]+'),
                          ('exact_duplicate_paragraphs',r'[^\n]*(?:\n(?![ \t\r]*\n)[^\n]*)*')]:
        duplicated = {d['text']:d['copies'] for d in strict[name]}
        readable = {}
        occurrences = Counter()
        for match in re.finditer(pattern,value):
            unit = normalized(match.group())
            if unit not in duplicated:
                continue
            occurrences[unit] += 1
            uncovered = ''.join(c for j,c in enumerate(value[match.start():match.end()],match.start())
                                if not mask[j])
            if any(c.isalnum() for c in uncovered):
                readable.setdefault(unit,[]).append(normalized(uncovered))
        for unit,texts in readable.items():
            repeated.append(dict(unit=name,normalized_unit=unit,copies=duplicated[unit],
                                 parsed_occurrences=occurrences[unit],readable_occurrences=len(texts),
                                 uncovered_readable_texts=texts))
    return repeated


def classify(template, strict):
    messages, method, spans = parse_headers(template)
    repeated = readable_repeated_units(template,strict,spans)
    empty = []
    completion = []
    identical = []
    repeated_blocks = []
    if messages is not None:
        for i,m in enumerate(messages):
            if not m['body_has_goal'] and not any(c.isalnum() for c in m['body']):
                if i == len(messages)-1 and m['role'] == 'assistant' and method.startswith('recognized_'):
                    completion.append(i)
                else:
                    empty.append(i)
        body_sequences = [normalized(remove_known(m['body']).replace('{goal}','')) for m in messages]
        counts = Counter(body for body in body_sequences if body)
        identical = [dict(normalized_body=b,copies=n) for b,n in counts.items() if n>1]
        seen_blocks = {}
        found_blocks = set()
        for index,body in enumerate(body_sequences):
            for offset in range(max(0,len(body)-31)):
                block = body[offset:offset+32]
                if len(re.findall(r'\w+',block)) < 4:
                    continue
                if block in seen_blocks:
                    previous_index,previous_offset = seen_blocks[block]
                    if previous_index != index or abs(previous_offset-offset)>=32:
                        if block not in found_blocks and len(repeated_blocks)<8:
                            repeated_blocks.append(dict(normalized_exact_block=block,characters=32,
                                first_message=previous_index,first_offset=previous_offset,
                                repeated_message=index,repeated_offset=offset))
                            found_blocks.add(block)
                else:
                    seen_blocks[block] = (index,offset)
    mixed = [item for item in repeated if item['readable_occurrences']<2]
    repeated_prose = [item for item in repeated if item['readable_occurrences']>=2]
    if method in ('unrecognized_role_like_markup','overlapping_header_spans'):
        category = 'unresolved_header_body_parse'
    elif repeated_prose or identical or repeated_blocks:
        category = 'actual_repeated_readable_body'
    elif empty:
        category = 'empty_role_header_stacking'
    elif messages is None or mixed:
        category = 'unresolved_header_body_parse'
    elif strict['exact_duplicate_headers'] and len(messages)<strict['header_count']:
        category = 'unresolved_header_body_parse'
    elif strict['exact_duplicate_headers'] or len(messages)>1:
        category = 'format_or_role_repetition_with_distinct_bodies'
    else:
        category = 'format_punctuation_only_repetition'
    return dict(category=category,diagnostic_only=True,eligible_for_any_future_evaluation=None,
                repeated_readable_units=repeated,parsed_message_count=None if messages is None else len(messages),
                message_diagnostics=messages,parse_method=method,empty_message_indices=empty,
                optional_single_terminal_assistant_prefix_indices=completion,
                identical_nonempty_message_bodies_excluding_goal=identical,
                exact_repeated_body_blocks_excluding_goal=repeated_blocks,
                repeated_block_diagnostic_minimum_characters=32,
                repeated_block_diagnostic_minimum_word_fragments=4,
                semantic_paraphrase_detection=False)


def main():
    out=scan.OUT
    original_summary=json.loads((out/'summary.json').read_text())
    assert original_summary['status']=='complete'
    original=out/'rejected_literal_index.jsonl'
    assert scan.sha(original.read_bytes())==original_summary['rejected_literal_index_sha256']
    rows=[json.loads(s) for s in original.read_text().splitlines()]
    documents={};source_hashes={};verified=0
    for row in rows:
        assert scan.sha(row['template'].encode())==row['raw_sha256']
        strict=strict_header_metrics(row['template'])
        assert strict==row['strict_metrics'] and not strict['allowed']
        assert row['original_diverse_policy']['allowed'] and row['wrapper_tokens']<=128
        for ref in [row,*row['source_aliases']]:
            path=Path(ref['source_path'])
            if str(path) not in documents:
                data=path.read_bytes();source_hashes[str(path)]=scan.sha(data)
                documents[str(path)]=json.loads(data) if ref['source_format']=='json' else [
                    json.loads(line) if line.strip() else None for line in data.splitlines()]
            assert source_hashes[str(path)]==ref['source_sha256']
            document=documents[str(path)]
            if ref['source_format']=='jsonl':
                document=document[ref['source_line_number']-1]
            assert scan.point_value(document,ref['json_pointer']).encode()==row['template'].encode()
            verified+=1
        row['diagnostic']=classify(row['template'],strict)
        row['classification_revised_after_independent_code_review']=True
    target=out/'reviewed_literal_index_v2.jsonl'
    with target.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
    categories=Counter(row['diagnostic']['category'] for row in rows)
    report=dict(schema='readonly-strict-rejection-reviewed-classification-v1',status='complete',
        completed_unix=time.time(),pre_round_cutoff_unix=scan.CUTOFF,
        original_scan_summary_path=str(out/'summary.json'),original_scan_summary_sha256=scan.sha((out/'summary.json').read_bytes()),
        original_index_sha256=scan.sha(original.read_bytes()),reviewed_index_path=str(target),
        reviewed_index_sha256=scan.sha(target.read_bytes()),unique_literals=len(rows),category_counts=dict(categories),
        source_files_reverified=len(documents),original_source_pointer_occurrences_reverified=verified,
        literal_text_changed=False,new_candidates_generated=False,API_calls=0,GPU_used=False,
        source_sha256={str(Path(__file__).resolve()):scan.sha(Path(__file__).read_bytes())},
        old_policy_modified=False,active_ASR_modified=False,future_admission_decision=False,
        caution='Exact formatting proof only; no semantic paraphrase judge. A single final empty assistant completion prefix is recorded separately from stacked empty headers. Unresolved parsing does not count as format-only.')
    scan.write(out/'reviewed_summary_v2.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
