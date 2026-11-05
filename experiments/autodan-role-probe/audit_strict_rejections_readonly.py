"""CPU-only diagnostic of strict-filter rejections among existing literals.

This script never changes a template, admission policy, score, or running input.
Diagnostic normalization and parsing are not candidate text transformations.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template, _MARKUP, _ROLE_HEADER
from strict_header_policy import strict_header_metrics, normalized, ROLES

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'outputs'
OUT = ROOT / 'diverse8h-20261105' / 'readonly_strict_rejection_diagnostic'
CUTOFF = 1793858050.0
MODEL = '/tmp/wz55191-agentdojo-first-tool-model'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n')


def walk(value, pointer=''):
    if isinstance(value, dict):
        for key, child in value.items():
            point = pointer + '/' + key.replace('~', '~0').replace('/', '~1')
            if key == 'template' and isinstance(child, str):
                yield child, point
            elif isinstance(child, (dict, list)):
                yield from walk(child, point)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            point = pointer + '/' + str(i)
            if isinstance(child, str) and pointer.endswith(('/templates', '/records')):
                yield child, point
            elif isinstance(child, (dict, list)):
                yield from walk(child, point)


def point_value(value, pointer):
    for part in pointer[1:].split('/') if pointer else []:
        key = part.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def marker_name(marker):
    return ''.join(c for c in unicodedata.normalize('NFKC', marker.group()).casefold()
                   if c.isalpha())


def prose(value):
    """Diagnostic body only; the original template is never overwritten."""
    text = _MARKUP.sub(' ', value)
    return normalized(text)


def structural(value):
    """Prove an exact repeated unit is only punctuation/markup/a role label."""
    value = _MARKUP.sub(' ', value).strip()
    if not any(c.isalnum() for c in value):
        return True
    if _ROLE_HEADER.fullmatch(value) and len(value) <= 32:
        return True
    # A JSON role declaration carries a role tag but no message body.
    if re.fullmatch(r'''[\s,{\[]*["']role["']\s*:\s*["'](?:'''+ROLES+r''')["'][\s,}\]]*''', value, re.I):
        return True
    return False


def json_messages(value):
    """Return complete parsed role/content objects, or None for non-JSON."""
    try:
        document = json.loads(value)
    except (ValueError, TypeError):
        return None
    found = []
    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get('role'), str) and re.fullmatch(ROLES, node['role'], re.I):
                body = node.get('content')
                if not isinstance(body, str):
                    return False
                found.append(dict(kind='json_role_header', role=node['role'].casefold(),
                                  body=normalized(body), body_has_goal='{goal}' in body))
            for child in node.values():
                if isinstance(child, (dict, list)) and visit(child) is False:
                    return False
        elif isinstance(node, list):
            for child in node:
                if visit(child) is False:
                    return False
        return True
    return found if visit(document) and found else None


def header_messages(template):
    parsed = json_messages(template)
    if parsed is not None:
        return parsed, 'full_json_role_content_parse'
    if re.search(r'''["']role["']\s*:''', template, re.I):
        return None, 'partial_or_unrecognized_JSON'
    headers = []
    markers = list(_MARKUP.finditer(template))
    names = [marker_name(m) for m in markers]
    for i, (marker, kind) in enumerate(zip(markers, names)):
        if kind in ('start', 'imstart', 'begin'):
            for j in range(i+1, min(len(markers), i+8)):
                if names[j] in ('start', 'imstart', 'begin', 'end', 'imend', 'call', 'return'):
                    break
                if names[j] in ('message', 'imsep', 'content'):
                    between = template[marker.end():markers[j].start()].strip()
                    role = between.casefold()
                    if re.fullmatch(ROLES, role):
                        headers.append((marker.start(), markers[j].end(), 'near_message_header', role))
                    break
        if not marker.group().lstrip().startswith('</') and re.fullmatch(ROLES, kind):
            headers.append((marker.start(), marker.end(), 'xml_role_header', kind))
    pattern = re.compile(r'(?m)^[ \t]*(?:#{1,6}[ \t]*)?('+ROLES+r')[ \t]*(?::|(?=\r?$))', re.I)
    for match in pattern.finditer(template):
        headers.append((match.start(), match.end(), 'plain_role_header', match.group(1).casefold()))
    headers.sort()
    if not headers:
        return [], 'no_parsed_role_headers'
    messages = []
    for index, (start, end, kind, role) in enumerate(headers):
        bound = headers[index+1][0] if index+1 < len(headers) else len(template)
        raw_body = template[end:bound]
        body = prose(raw_body)
        messages.append(dict(kind=kind, role=role, body=body,
                             body_has_goal='{goal}' in raw_body,
                             header_offset=[start, end], body_offset=[end, bound]))
    return messages, 'recognized_header_span_parse'


def classify(template, strict):
    repeated_readable = []
    for kind in ('exact_duplicate_lines', 'exact_duplicate_paragraphs'):
        for item in strict[kind]:
            if not structural(item['text']):
                repeated_readable.append(dict(unit=kind, **item))
    messages, parse_method = header_messages(template)
    empty = []
    same_bodies = []
    if messages is not None:
        for i, message in enumerate(messages):
            if not message['body_has_goal'] and not any(c.isalnum() for c in message['body']):
                empty.append(i)
        counts = Counter(m['body'] for m in messages if m['body'])
        same_bodies = [dict(normalized_body=b, copies=n) for b,n in counts.items() if n > 1]
    if repeated_readable or same_bodies:
        category = 'actual_repeated_readable_body'
    elif empty:
        category = 'empty_role_header_stacking'
    elif strict['exact_duplicate_headers'] and messages is None:
        category = 'unresolved_header_body_parse'
    elif strict['exact_duplicate_headers'] and (not messages or len(messages) < strict['header_count']):
        category = 'unresolved_header_body_parse'
    elif strict['exact_duplicate_headers']:
        category = 'format_or_role_repetition_with_distinct_nonempty_bodies'
    else:
        category = 'format_punctuation_only_repetition'
    return dict(category=category, diagnostic_only=True, eligible_for_any_future_evaluation=None,
                repeated_readable_units=repeated_readable,
                parsed_message_count=None if messages is None else len(messages),
                message_diagnostics=messages, parse_method=parse_method,
                empty_message_indices=empty, identical_nonempty_message_bodies=same_bodies,
                semantic_paraphrase_detection=False)


def main():
    from transformers import AutoTokenizer
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    library = BroadDelimiterLibrary(MODEL)
    command = ['rg','--files','--hidden','--no-ignore',str(ROOT),'-g','*.json','-g','*.jsonl']
    paths = sorted(Path(p) for p in subprocess.check_output(command, text=True).splitlines())
    counts = Counter(rg_files_inventoried=len(paths)); seen = set(); retained = {}
    sources = {}; policy_reasons = Counter(); examples = []
    for path in paths:
        if 'diverse8h-20261105' in path.parts or path.stat().st_mtime > CUTOFF:
            counts['excluded_current_or_post_cutoff_files'] += 1
            continue
        before = path.stat()
        data = path.read_bytes()
        digest = sha(data)
        try:
            documents = [(None, json.loads(data))] if path.suffix == '.json' else [
                (i+1,json.loads(line)) for i,line in enumerate(data.splitlines()) if line.strip()]
        except (ValueError, UnicodeError):
            counts['invalid_source_JSON_files'] += 1
            continue
        counts['pre_round_files_scanned'] += 1
        file_refs = 0
        for line_number, document in documents:
            for template, pointer in walk(document):
                counts['template_nodes_seen'] += 1
                if template.count('{goal}') != 1:
                    counts['non_single_goal_slot'] += 1
                    continue
                raw = sha(template.encode('utf-8', errors='strict'))
                source = dict(source_path=str(path), source_sha256=digest, json_pointer=pointer,
                              source_format=path.suffix[1:], source_mtime_unix=before.st_mtime)
                if line_number is not None:
                    source['source_line_number'] = line_number
                if raw in seen:
                    counts['duplicate_template_occurrences'] += 1
                    if raw in retained:
                        retained[raw]['source_aliases'].append(source)
                        file_refs += 1
                    continue
                seen.add(raw)
                # Tokenize the short bound before expensive repeated-block metrics.
                size = len(tokenizer.encode(template.replace('{goal}',''), add_special_tokens=False))
                if size > 128:
                    counts['over_128_unique'] += 1
                    continue
                counts['within_128_unique'] += 1
                strict = strict_header_metrics(template)
                if strict['allowed']:
                    counts['strict_allowed_short_unique'] += 1
                    continue
                admission = validate_diverse_template(template, tokenizer, library, maximum=128)
                if not admission['allowed']:
                    counts['strict_rejected_and_diverse_rejected_unique'] += 1
                    policy_reasons.update(admission['reason_codes'])
                    continue
                assert point_value(document, pointer).encode('utf-8') == template.encode('utf-8')
                row = dict(schema='readonly-existing-strict-rejection-diagnostic-v1',
                           template=template, raw_sha256=raw, wrapper_tokens=size,
                           **source, source_aliases=[], strict_metrics=strict,
                           original_diverse_policy=admission,
                           diagnostic=classify(template, strict),
                           original_literal_bytes_unchanged=True, prediction_or_fitness=False,
                           new_text_generation=False, template_mutation=False,
                           source_is_derived_copy=False)
                retained[raw] = row
                file_refs += 1
                counts['strict_rejected_diverse_allowed_short_unique'] += 1
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise RuntimeError('Read-only source changed while scanning: ' + str(path))
        if file_refs:
            sources[str(path)] = dict(sha256=digest, pre_round_source=True,
                                     source_mtime_unix=before.st_mtime, indexed_occurrences=file_refs)
        if counts['pre_round_files_scanned'] % 10000 == 0:
            print(json.dumps(dict(stage='scan', counts=dict(counts), elapsed=time.time()-started)), flush=True)
    rows = [retained[k] for k in sorted(retained)]
    catalog = OUT/'rejected_literal_index.jsonl'
    with catalog.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n')
    categories = Counter(r['diagnostic']['category'] for r in rows)
    sources_path = OUT/'sources.json'
    write(sources_path, dict(sources=sources))
    source_code = [Path(__file__).resolve(), HERE/'strict_header_policy.py', HERE/'diverse_policy.py', HERE/'broad_validation.py']
    summary = dict(schema='readonly-existing-strict-rejection-inventory-v1', status='complete',
                   started_unix=started, completed_unix=time.time(),
                   pre_round_cutoff_unix=CUTOFF, command=command, counts=dict(counts),
                   category_counts=dict(categories), other_policy_rejection_reasons=dict(policy_reasons),
                   rejected_literal_index_path=str(catalog), rejected_literal_index_sha256=sha(catalog.read_bytes()),
                   sources_path=str(sources_path), sources_sha256=sha(sources_path.read_bytes()),
                   source_bindings={str(p):sha(p.read_bytes()) for p in source_code},
                   new_text_generation=False, template_mutation=False, GPU_used=False, API_calls=0,
                   frozen_inputs_or_ASR_modified=False, future_admission_decision=False,
                   caution='Diagnostic categories are not an admission policy. Unresolved parsing is not counted as a proven format-only rejection. Exact repeated readable units are retained as repetition, never relabeled as formatting.')
    write(OUT/'summary.json',summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
