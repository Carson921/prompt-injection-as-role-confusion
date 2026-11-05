"""Native scorer with deterministic serialized-string goal origins.

Only span recovery differs from the frozen native Scorer. Model loading, full
prompts, hidden states, probe columns, arithmetic and score files are inherited.
The base score protocol binds that native numerical path; the campaign protocol
additionally binds this implementation and deterministic_spans.py. Cached scores
are reusable only after the independent exact-input/mask compatibility audit.
"""
import re
import numpy as np
from scorer import Scorer, render_messages, require, overlap, sha
from deterministic_spans import locate_payload


class PreciseScorer(Scorer):
    def prepare(self, context, template):
        request = render_messages(context, template)
        date = context.get('prompt_date', self.protocol['prompt_date'])
        prompt = self.spans.render(self.tokenizer, request, date)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.asarray(encoded['input_ids'], dtype=np.int64)
        offsets = np.asarray(encoded['offset_mapping'], dtype=np.int64)
        payload_ids = self.tokenizer.encode(request['payload_text'], add_special_tokens=False)
        require(not any(token >= self.native_special_id_floor for token in payload_ids),
                'Payload contains an actual native special/reserved token ID')
        require(not any(token in request['payload_text'] for token in self.native_special_spellings)
                and re.search(r'<\|reserved_\d+\|>', request['payload_text']) is None,
                'Payload contains an exact native special/reserved token spelling')
        require(len(ids) <= int(self.model.config.max_position_embeddings), 'Complete context exceeds native model capacity; no truncation allowed')
        allowed = {row['message_index'] for row in context['slot_recipes']} if context.get('slot_recipes') else None
        location = locate_payload(self.tokenizer, self.spans, request, prompt, date, allowed, context=context)
        payload_rows = overlap(offsets, location['payload_char_span'])
        goal_rows = overlap(offsets, location['goal_char_span'])
        expected_occurrences = sum(slot['occurrences'] for recipe in context.get('slot_recipes', []) for slot in recipe['slots'])
        if expected_occurrences:
            require(len(location['occurrences']) == expected_occurrences, 'Payload occurrence count differs from native field recipes')
        all_payload_rows = np.zeros(len(ids), dtype=bool)
        for occurrence in location['occurrences']:
            all_payload_rows |= overlap(offsets, occurrence['payload_char_span'])
        require(not np.any(ids[all_payload_rows] >= self.native_special_id_floor),
                'A rendered payload occurrence contains a real special/reserved token ID')
        location['payload_special_token_check'] = {
            'passed': True, 'raw_payload_tokens_checked': len(payload_ids),
            'rendered_payload_tokens_checked': int(all_payload_rows.sum()),
            'payload_occurrences_checked': len(location['occurrences']),
            'native_special_token_count': 0,
            'native_special_id_floor': self.native_special_id_floor}
        positions = np.flatnonzero(payload_rows)
        require(len(positions) > 0, 'Payload has no original input token rows')
        goal = goal_rows[positions]
        require(goal.any(), 'Complete immutable goal has no original token rows')
        require(offsets[positions[0], 0] <= location['payload_char_span'][0]
                and offsets[positions[-1], 1] >= location['payload_char_span'][1], 'Token offsets do not cover complete payload')
        input_sha256 = sha(ids.tobytes())
        original = context.get('original_template', context.get('payload_template'))
        if template == original and context.get('source_input_sha256'):
            require(input_sha256 == context['source_input_sha256'], 'Original source template fails full-input token reconstruction')
        return request, prompt, ids, offsets, positions, goal, location, input_sha256
