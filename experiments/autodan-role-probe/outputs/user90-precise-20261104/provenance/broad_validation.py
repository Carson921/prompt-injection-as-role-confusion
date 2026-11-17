"""Literal braces are ordinary text; only genuine delimiters are excluded.

The old brace restriction was a compatibility rule for str.format. Native
rendering and ExactGoalAttack now replace only the exact immutable goal slot.
"""
from __future__ import annotations

import re

from near_delimiters import NearDelimiterLibrary


class BroadDelimiterLibrary(NearDelimiterLibrary):
    def validate(self, template, tokenizer=None):
        if not isinstance(template, str) or template.count('{goal}') != 1:
            return False
        if self._forbidden_re.search(template) or re.search(r'<\|reserved_\d+\|>', template):
            return False
        if tokenizer is not None:
            ids = tokenizer.encode(template, add_special_tokens=False)
            if hasattr(ids, 'ids'):
                ids = ids.ids
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if any(int(i) >= 199998 for i in ids):
                return False
            if self.forbidden_token_ids.intersection(ids):
                return False
        return True

    def export(self, path, tokenizer=None):
        result = super().export(path, tokenizer)
        result['validation']['other_literal_braces_allowed'] = True
        result['validation']['ordinary_id_upper_bound_exclusive'] = 199998
        result['validation']['literal_goal_replacement_only'] = True
        from scorer import write
        write(path, result)
        return result


def valid_template(template, tokenizer=None, minimum=1, maximum=8192, library=None):
    if not isinstance(template, str) or template.count('{goal}') != 1:
        return False
    if 'AUTODANIMMUTABLEGOALSLOT' in template:
        return False
    wrapper = template.replace('{goal}', '')
    if not wrapper.strip():
        return False
    if any(ord(c) < 32 and c not in '\n\r\t' for c in wrapper):
        return False
    if library is None or not library.validate(template, tokenizer=tokenizer):
        return False
    size = len(tokenizer.encode(wrapper, add_special_tokens=False)) if tokenizer else len(wrapper.split())
    return minimum <= size <= maximum


def install_empty_join_guard(operators, provenance):
    """Preserve the original upstream function for every nonempty word list."""
    original = operators['join_words_with_punctuation']

    def guarded(words):
        return original(words) if words else ''

    operators['join_words_with_punctuation'] = guarded
    provenance['adaptations'].append(
        'Empty upstream HGA word lists return empty text instead of IndexError; '
        'marker-invalid children are discarded, nonempty lists use original join.')
