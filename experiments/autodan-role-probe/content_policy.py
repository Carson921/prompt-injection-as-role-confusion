"""CPU-only content-search admission policy for abstract goal templates.

Wrapper length excludes the sole literal {goal} placeholder.  Repetition is
measured without changing text: repeated line/sentence spans contribute a union
of content atoms, avoiding double-counting the same duplicated sentence as a
line. Each approximate markup delimiter is one atom, while its exact spelling
is retained in equality keys. Thus different delimiter styles are not merged,
and a longer spelling alone does not inflate repeated-content weight.
"""
from __future__ import annotations

from collections import Counter
import re


HARD_WRAPPER_CAP = 3165
MAX_NEW_SEED_LINE_COPIES = 2
MAX_NEW_SEED_SENTENCE_COPIES = 2
BASE_BPE_BOUND = 199998
SLOT = '{goal}'

# Bounded markup recognition affects repetition accounting only, never special
# token validation. Include embedded-newline near spellings as single units.
_MARKUP = re.compile(r'<[^<>]{1,256}>|(?<!<)\|[^<>|\r\n]{1,64}\|>|<\|[^<>|\r\n]{1,64}\|(?=[^>]|$)')


def _ids(tokenizer, text):
    values = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(values, 'ids'):
        values = values.ids
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def repetition_metrics(text):
    """JSON-safe duplicate metrics; no tokenizer, probe, or model required."""
    if not isinstance(text, str):
        return None
    markup = [(match.start(), match.end()) for match in _MARKUP.finditer(text)]
    owner = {}
    atoms = set()
    for start, end in markup:
        atoms.add(start)
        for index in range(start, end):
            owner[index] = start
    atoms.update(index for index, char in enumerate(text)
                 if not char.isspace() and index not in owner)

    def trim(start, end):
        while start < end and text[start].isspace() and start not in owner:
            start += 1
        while end > start and text[end - 1].isspace() and end - 1 not in owner:
            end -= 1
        return start, end

    def key(start, end):
        chunks, position, space = [], start, False
        while position < end:
            if position in owner:
                marker_start = owner[position]
                marker_end = next_finish[marker_start]
                if space and chunks:
                    chunks.append(' ')
                chunks.append(text[marker_start:marker_end])
                position, space = marker_end, False
            elif text[position].isspace():
                space = True
                position += 1
            else:
                if space and chunks:
                    chunks.append(' ')
                chunks.append(text[position])
                position, space = position + 1, False
        return ''.join(chunks)

    next_finish = dict(markup)
    lines, sentences = [], []
    start = index = 0
    while index < len(text):
        if text[index] in '\r\n' and index not in owner:
            a, b = trim(start, index)
            if a < b:
                lines.append((a, b))
            if text[index] == '\r' and index + 1 < len(text) and text[index + 1] == '\n':
                index += 1
            start = index + 1
        index += 1
    a, b = trim(start, len(text))
    if a < b:
        lines.append((a, b))
    start = index = 0
    while index < len(text):
        if text[index] in '.!?' and index not in owner:
            end = index + 1
            while end < len(text) and text[end] in '.!?' and end not in owner:
                end += 1
            if end == len(text) or text[end].isspace():
                a, b = trim(start, end)
                if a < b:
                    sentences.append((a, b))
                start = end
            index = end
        else:
            index += 1
    a, b = trim(start, len(text))
    if a < b:
        sentences.append((a, b))

    redundant_atoms = set()
    counters = []
    for spans in (lines, sentences):
        counts = Counter()
        for a, b in spans:
            value = key(a, b)
            if not value:
                continue
            if counts[value]:
                redundant_atoms.update(index for index in range(a, b) if index in atoms)
            counts[value] += 1
        counters.append(counts)
    line_counts, sentence_counts = counters
    return dict(total_redundancy=len(redundant_atoms),
                redundancy_unit='union of duplicated content characters and one atom per literal markup delimiter',
                line_redundant_units=sum(max(0, count - 1) for count in line_counts.values()),
                sentence_redundant_units=sum(max(0, count - 1) for count in sentence_counts.values()),
                max_identical_nonempty_line_copies=max(line_counts.values(), default=0),
                max_identical_sentence_copies=max(sentence_counts.values(), default=0),
                nonempty_lines=len(lines), sentence_units=len(sentences),
                distinct_nonempty_lines=len(line_counts), distinct_sentence_units=len(sentence_counts),
                delimiter_spellings_preserved=True, delimiter_spelling_length_neutral=True,
                line_sentence_overlap_counted_once=True)


def measure_template(template, tokenizer):
    """Measure abstract template text only; never fills the protected goal."""
    if not isinstance(template, str):
        return dict(goal_slots=None, wrapper_tokens=None, whole_template_tokens=None,
                    repetition=None, text_type=type(template).__name__)
    return dict(goal_slots=template.count(SLOT),
                wrapper_tokens=len(_ids(tokenizer, template.replace(SLOT, ''))),
                whole_template_tokens=len(_ids(tokenizer, template)),
                repetition=repetition_metrics(template), text_type='str')


def validate_content_template(template, tokenizer, library, parent=None, hard_cap=HARD_WRAPPER_CAP):
    """Return JSON-safe admission evidence, not a score or optimization fitness.

    ``library`` must be the existing BroadDelimiterLibrary used by the driver.
    A supplied parent may retain any existing repetition count, including384;
    neither total duplicate content nor maximum identical line/sentence copies
    may increase. New seeds allow at most two copies of any line or sentence.
    ``hard_cap`` may tighten, never relax, the fixed3165-wrapper-token ceiling.
    """
    reasons = []

    def reason(code, **values):
        reasons.append(dict(code=code, **values))

    if not isinstance(hard_cap, int) or isinstance(hard_cap, bool) or not 1 <= hard_cap <= HARD_WRAPPER_CAP:
        reason('invalid_hard_cap', supplied=repr(hard_cap), maximum=HARD_WRAPPER_CAP)
        hard_cap = HARD_WRAPPER_CAP
    try:
        child_metrics = measure_template(template, tokenizer)
        parent_metrics = measure_template(parent, tokenizer) if parent is not None else None
    except (ValueError, TypeError, RuntimeError) as error:
        return dict(allowed=False, reasons=[dict(code='tokenization_failed', error_type=type(error).__name__)],
                    reason_codes=['tokenization_failed'], child_metrics=None, parent_metrics=None,
                    hard_wrapper_cap=hard_cap, prediction_or_fitness=False)

    def basic(value, metrics, label):
        if not isinstance(value, str):
            reason(label + '_not_string')
            return
        if metrics['goal_slots'] != 1:
            reason(label + '_goal_slot_count', actual=metrics['goal_slots'], expected=1)
        wrapper = value.replace(SLOT, '')
        if not wrapper.strip():
            reason(label + '_empty_wrapper')
        if 'AUTODANIMMUTABLEGOALSLOT' in value:
            reason(label + '_unrestored_protected_marker')
        if any(ord(char) < 32 and char not in '\n\r\t' for char in wrapper):
            reason(label + '_control_character')
        if library is None:
            reason(label + '_missing_broad_delimiter_library')
        elif not library.validate(value, tokenizer=tokenizer):
            reason(label + '_broad_special_or_goal_policy')
        ids = _ids(tokenizer, value)
        forbidden = set(getattr(library, 'forbidden_token_ids', ())) | set(getattr(tokenizer, 'all_special_ids', ()))
        bad = sorted({token for token in ids if token >= BASE_BPE_BOUND or token in forbidden})
        if bad:
            reason(label + '_actual_special_token_ids', token_ids=bad)

    basic(template, child_metrics, 'child')
    if isinstance(template, str) and child_metrics['wrapper_tokens'] > hard_cap:
        reason('hard_wrapper_cap_exceeded', actual=child_metrics['wrapper_tokens'], maximum=hard_cap)
    if parent is not None:
        basic(parent, parent_metrics, 'parent')
        if isinstance(template, str) and isinstance(parent, str):
            if child_metrics['wrapper_tokens'] > parent_metrics['wrapper_tokens']:
                reason('parent_wrapper_length_increased', child=child_metrics['wrapper_tokens'],
                       parent=parent_metrics['wrapper_tokens'])
            child_repeat = child_metrics['repetition']['total_redundancy']
            parent_repeat = parent_metrics['repetition']['total_redundancy']
            if child_repeat > parent_repeat:
                reason('parent_total_redundancy_increased', child=child_repeat, parent=parent_repeat)
            for metric, code in (
                    ('max_identical_nonempty_line_copies', 'parent_max_identical_line_copies_increased'),
                    ('max_identical_sentence_copies', 'parent_max_identical_sentence_copies_increased')):
                child_copies = child_metrics['repetition'][metric]
                parent_copies = parent_metrics['repetition'][metric]
                if child_copies > parent_copies:
                    reason(code, child=child_copies, parent=parent_copies)
    elif isinstance(template, str):
        copies = child_metrics['repetition']['max_identical_nonempty_line_copies']
        if copies > MAX_NEW_SEED_LINE_COPIES:
            reason('new_seed_repeated_line_limit', actual=copies, maximum=MAX_NEW_SEED_LINE_COPIES)
        sentence_copies = child_metrics['repetition']['max_identical_sentence_copies']
        if sentence_copies > MAX_NEW_SEED_SENTENCE_COPIES:
            reason('new_seed_repeated_sentence_limit', actual=sentence_copies,
                   maximum=MAX_NEW_SEED_SENTENCE_COPIES)
    return dict(allowed=not reasons, reasons=reasons, reason_codes=[item['code'] for item in reasons],
                child_metrics=child_metrics, parent_metrics=parent_metrics,
                hard_wrapper_cap=hard_cap, has_parent=parent is not None,
                maximum_new_seed_identical_nonempty_line_copies=MAX_NEW_SEED_LINE_COPIES,
                maximum_new_seed_identical_sentence_copies=MAX_NEW_SEED_SENTENCE_COPIES,
                prediction_or_fitness=False)


# Short aliases for callers that already carry a policy namespace.
validate = validate_content_template
measure = measure_template
