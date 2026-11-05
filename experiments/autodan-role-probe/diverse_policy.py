"""CPU admission rules for diverse, non-repetitive abstract goal templates.

Admission evidence is never probe fitness.  The unchanged native six-case
scorer is the only source of P(user).  Parent templates receive no exemption.
Short role headers and near-delimiter markup may recur; copied prose and long
copied blocks may not.  All measurements leave the candidate text unchanged.
"""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

SCHEMA = 'diverse-template-admission-v1'
SLOT = '{goal}'
BASE_BPE_BOUND = 199998
DEFAULT_MAXIMUM = 1024
THRESHOLDS = {
    'maximum_identical_body_line_copies': 2,
    'maximum_identical_body_sentence_copies': 2,
    'character_ngram_sizes': [96, 192],
    'token_ngram_sizes': [24, 48],
    'full_template_character_ngram_sizes': [256, 512],
    'full_template_token_ngram_sizes': [64, 128],
    'maximum_redundant_character_ngram_fraction': 0.25,
    'maximum_redundant_token_ngram_fraction': 0.25,
    'maximum_identical_character_run': 24,
    'maximum_unusual_unicode_fraction': 0.15,
}

# These patterns are used only to avoid charging fixed markup/short headers as
# repeated *prose*. They have no influence on the independent special-token ban.
_MARKUP = re.compile(r'<[^<>]{1,256}>|(?<!<)\|[^<>|\r\n]{1,64}\|>|<\|[^<>|\r\n]{1,64}\|(?=[^>]|$)')
_ROLE_HEADER = re.compile(
    r'^(?:[#*\[({<\s]*)(?:system|developer|user|assistant|tool|browser|python|analysis|commentary|final|summary)'
    r'(?:[\s\]})>*:#|=\-]*)$', re.I)
_SENTENCE = re.compile(r'(?<=[.!?。！？])\s+')


def _ids(tokenizer, text):
    values = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(values, 'ids'):
        values = values.ids
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _normal(value):
    return ' '.join(unicodedata.normalize('NFC', value).casefold().split())


def _body(text):
    value = _MARKUP.sub(' ', text.replace(SLOT, ' '))
    lines = []
    ignored = 0
    for line in value.splitlines():
        normalized = _normal(line)
        if not normalized:
            continue
        if len(normalized) <= 32 and _ROLE_HEADER.fullmatch(normalized):
            ignored += 1
        else:
            lines.append(normalized)
    return lines, ignored


def _ngram_metrics(sequence, n):
    """Count the union of all later occurrences of exact length-n windows.

    Overlapping redundant windows count each position once.  Dictionary keys
    are the actual strings/tuples, so there is no hash-collision approximation.
    """
    length = len(sequence)
    if length < n:
        return dict(size=n, sequence_units=length, windows=0, repeated_windows=0,
                    redundant_units=0, redundant_fraction=0.0, maximum_copies=0)
    counts = {}
    marked = bytearray(length)
    repeated = maximum = 0
    token_sequence = not isinstance(sequence, str)
    fill = bytes([1]) * n
    for index in range(length - n + 1):
        key = tuple(sequence[index:index+n]) if token_sequence else sequence[index:index+n]
        count = counts.get(key, 0) + 1
        counts[key] = count
        maximum = max(maximum, count)
        if count > 1:
            repeated += 1
            marked[index:index+n] = fill
    redundant = sum(marked)
    return dict(size=n, sequence_units=length, windows=length-n+1,
                repeated_windows=repeated, redundant_units=redundant,
                redundant_fraction=redundant/length if length else 0.0,
                maximum_copies=maximum)


def repetition_metrics(text, tokenizer=None):
    """JSON-safe text diagnostics. Unicode letters remain ordinary content."""
    if not isinstance(text, str):
        return None
    lines, ignored = _body(text)
    body = ' '.join(lines)
    sentences = [_normal(value) for value in _SENTENCE.split(body) if _normal(value)]
    line_counts, sentence_counts = Counter(lines), Counter(sentences)
    maximum_run = current_run = 0
    previous = None
    for char in body:
        current_run = current_run + 1 if char == previous else 1
        if not char.isspace():
            maximum_run = max(maximum_run, current_run)
        previous = char
    result = dict(
        body_characters=len(body), body_nonempty_lines=len(lines),
        ignored_short_role_headers=ignored,
        body_sentence_units=len(sentences),
        max_identical_body_line_copies=max(line_counts.values(), default=0),
        max_identical_body_sentence_copies=max(sentence_counts.values(), default=0),
        redundant_body_line_characters=sum(len(value)*(count-1) for value,count in line_counts.items()),
        maximum_identical_character_run=maximum_run,
        character_ngrams=[_ngram_metrics(body, n) for n in THRESHOLDS['character_ngram_sizes']],
        full_template_character_ngrams=[_ngram_metrics(_normal(text.replace(SLOT,' ')),n)
                                       for n in THRESHOLDS['full_template_character_ngram_sizes']],
        normalization='NFC, casefold, collapse whitespace; remove approximate markup and short role-only headers',
    )
    if tokenizer is not None:
        body_ids = _ids(tokenizer, body)
        result['body_tokens'] = len(body_ids)
        result['token_ngrams'] = [_ngram_metrics(body_ids, n) for n in THRESHOLDS['token_ngram_sizes']]
        raw_ids = _ids(tokenizer,_normal(text.replace(SLOT,' ')))
        result['full_template_token_ngrams'] = [_ngram_metrics(raw_ids,n)
                                               for n in THRESHOLDS['full_template_token_ngram_sizes']]
    return result


def _unicode_metrics(text):
    if not isinstance(text, str):
        return None
    controls, surrogates, noncharacters, formats = [], [], [], []
    unusual = 0
    nonspace = 0
    for index, char in enumerate(text):
        codepoint, category = ord(char), unicodedata.category(char)
        if not char.isspace():
            nonspace += 1
        if category == 'Cc' and char not in '\n\r\t':
            controls.append(index)
        if category == 'Cs':
            surrogates.append(index)
        if category == 'Cf':
            formats.append(index)
        if 0xFDD0 <= codepoint <= 0xFDEF or codepoint & 0xFFFF in (0xFFFE, 0xFFFF):
            noncharacters.append(index)
        # Visible random Unicode letters, punctuation and symbols are allowed.
        # Private-use/unassigned characters and combining marks are the limited
        # noise budget; this is a measurable proxy, not a semantic garble judge.
        if category in ('Co', 'Cn', 'Mn', 'Mc', 'Me', 'Cf'):
            unusual += 1
    return dict(control_character_count=len(controls), format_control_count=len(formats),
                surrogate_count=len(surrogates), noncharacter_count=len(noncharacters),
                invalid_position_examples=(controls+surrogates+formats+noncharacters)[:16],
                unusual_unicode_characters=unusual, nonspace_characters=nonspace,
                unusual_unicode_fraction=unusual/max(1,nonspace),
                unusual_unicode_definition='categories Co,Cn,Mn,Mc,Me,Cf; visible letters and symbols are allowed; format characters are measured, not categorically banned',
                structural_whitespace_allowed=['LF','CR','TAB'])


def measure_template(template, tokenizer):
    if not isinstance(template, str):
        return dict(text_type=type(template).__name__, goal_slots=None, wrapper_tokens=None,
                    whole_template_tokens=None, repetition=None, unicode=None)
    # UTF-8 encoding rejects lone surrogates before a native tokenizer can emit
    # an opaque backend error. No replacement decoding is permitted.
    utf8_valid = True
    try:
        template.encode('utf-8', errors='strict')
    except UnicodeEncodeError:
        utf8_valid = False
    safe_to_tokenize = utf8_valid
    return dict(text_type='str', goal_slots=template.count(SLOT), utf8_valid=utf8_valid,
                wrapper_tokens=len(_ids(tokenizer,template.replace(SLOT,''))) if safe_to_tokenize else None,
                whole_template_tokens=len(_ids(tokenizer,template)) if safe_to_tokenize else None,
                repetition=repetition_metrics(template,tokenizer if safe_to_tokenize else None),
                unicode=_unicode_metrics(template))


def validate_diverse_template(template, tokenizer, library, *, maximum=DEFAULT_MAXIMUM, parent=None):
    """Return admission evidence; parent is recorded but never grants a waiver.

    ``maximum`` is the current stage token budget, excluding the immutable slot.
    Richer content, wrapper count and ordering may vary freely within this cap.
    The special spelling/ID bans use the authoritative local delimiter library.
    """
    reasons = []
    def reject(code, **detail):
        reasons.append(dict(code=code, **detail))
    if not isinstance(maximum,int) or isinstance(maximum,bool) or maximum < 1:
        reject('invalid_maximum', supplied=repr(maximum))
        maximum = DEFAULT_MAXIMUM
    try:
        metrics = measure_template(template,tokenizer)
    except (ValueError, TypeError, RuntimeError, OverflowError) as error:
        return dict(schema=SCHEMA, allowed=False, reason_codes=['tokenization_failed'],
                    reasons=[dict(code='tokenization_failed',error_type=type(error).__name__)],
                    metrics=None,maximum_wrapper_tokens=maximum,thresholds=dict(THRESHOLDS),
                    prediction_or_fitness=False,parent_repetition_exemption=False)
    if not isinstance(template,str):
        reject('not_string')
    else:
        if metrics['goal_slots'] != 1:
            reject('goal_slot_count',actual=metrics['goal_slots'],expected=1)
        if not template.replace(SLOT,'').strip():
            reject('empty_wrapper')
        if 'AUTODANIMMUTABLEGOALSLOT' in template:
            reject('unrestored_protected_marker')
        if not metrics['utf8_valid']:
            reject('invalid_utf8_or_surrogate')
        for key,code in [('control_character_count','control_character')]:
            if metrics['unicode'][key]:
                reject(code,count=metrics['unicode'][key])
        fraction = metrics['unicode']['unusual_unicode_fraction']
        if fraction > THRESHOLDS['maximum_unusual_unicode_fraction']:
            reject('unusual_unicode_budget',actual=fraction,maximum=THRESHOLDS['maximum_unusual_unicode_fraction'])
        if metrics['wrapper_tokens'] is not None and metrics['wrapper_tokens'] > maximum:
            reject('wrapper_budget_exceeded',actual=metrics['wrapper_tokens'],maximum=maximum)
        if library is None:
            reject('missing_delimiter_library')
        elif metrics['utf8_valid'] and not library.validate(template,tokenizer=tokenizer):
            reject('special_spelling_or_goal_policy')
        if metrics['utf8_valid']:
            ids = _ids(tokenizer,template)
            forbidden = set(getattr(library,'forbidden_token_ids',())) | set(getattr(tokenizer,'all_special_ids',()))
            bad = sorted({value for value in ids if value >= BASE_BPE_BOUND or value in forbidden})
            if bad:
                reject('actual_special_token_ids',token_ids=bad)
        repeat = metrics['repetition']
        for metric,threshold,code in [
                ('max_identical_body_line_copies','maximum_identical_body_line_copies','identical_body_line_limit'),
                ('max_identical_body_sentence_copies','maximum_identical_body_sentence_copies','identical_body_sentence_limit'),
                ('maximum_identical_character_run','maximum_identical_character_run','identical_character_run')]:
            if repeat[metric] > THRESHOLDS[threshold]:
                reject(code,actual=repeat[metric],maximum=THRESHOLDS[threshold])
        for kind,threshold in [('character','maximum_redundant_character_ngram_fraction'),
                               ('token','maximum_redundant_token_ngram_fraction')]:
            for scope in ('','full_template_'):
                for item in repeat.get(scope+kind+'_ngrams',[]):
                    if item['redundant_fraction'] > THRESHOLDS[threshold]:
                        reject('repeated_'+scope+kind+'_block_coverage',size=item['size'],
                               actual=item['redundant_fraction'],maximum=THRESHOLDS[threshold],
                               redundant_units=item['redundant_units'])
    return dict(schema=SCHEMA,allowed=not reasons,reasons=reasons,
                reason_codes=[reason['code'] for reason in reasons],metrics=metrics,
                maximum_wrapper_tokens=maximum,thresholds=dict(THRESHOLDS),
                prediction_or_fitness=False,parent_repetition_exemption=False,has_parent=parent is not None)


validate = validate_diverse_template
measure = measure_template
