"""Independent native-parser checks for deterministic span recovery (CPU only).

No payload regex is executed.  A located byte interval is checked by replacing
only that interval with a benign ASCII sentinel, decoding the *whole* original
native tool body, and requiring exactly the corresponding semantic field edit.
This independently tests the origin map, including escaping and line folding.
Codec fixtures below are tests, never search seeds or scored proposals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import deterministic_spans as locator
import scorer


OUTPUT = HERE / 'outputs/user90-broad-20261104'
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')
TIMEOUT = 5.0
REPORT = {
    'schema': 'deterministic-span-locator-independent-audit-v1',
    'entire_campaign_complete': False,
    'new_model_forwards': 0,
    'api_calls': 0,
    'codec_fixtures_are_not_search_proposals': True,
    'saved_cases': [], 'fixture_cases': [], 'compatibility_differences': [],
}
SENTINEL = 'AUDIT_SENTINEL_7f8c920a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def deadline(seconds=TIMEOUT):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(*_):
        raise TimeoutError('Deterministic span fixture exceeded its wall-clock limit')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    entered = time.monotonic()
    try:
        yield
    finally:
        remaining, interval = previous_timer
        if remaining:
            remaining = max(.000001, remaining - (time.monotonic() - entered))
        signal.setitimer(signal.ITIMER_REAL, remaining, interval)
        signal.signal(signal.SIGALRM, previous_handler)


def get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def set_value(tree, path, value):
    if not path:
        return value
    parent = tree
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return tree


def find_occurrences(text, needle):
    begin = 0
    while True:
        start = text.find(needle, begin)
        if start < 0:
            return
        yield start
        begin = start + len(needle)


def inner_span(text, outer_start, span):
    # The real rendered tool body is a JSON string.  Build inverse *boundaries*,
    # so partial escapes and partial Unicode characters cannot pass this check.
    boundaries = {0: 0}
    used = 0
    for index, char in enumerate(text):
        used += len(json.dumps(char, ensure_ascii=False)[1:-1])
        boundaries[used] = index + 1
    a, b = span[0] - outer_start - 1, span[1] - outer_start - 1
    if a not in boundaries or b not in boundaries:
        raise AssertionError('Span cuts an outer JSON character escape')
    return boundaries[a], boundaries[b]


def independent_semantics(request, context, prompt, spans_module, tokenizer, date, location):
    """Check exact goal/payload spans through full native parse substitution."""
    payload, goal = request['payload_text'], request['actual_goal']
    prefix = request['payload_template'].split('{goal}')[0]
    outer = {index: (a, b, message) for a, b, index, message
             in spans_module.tool_spans(tokenizer, request, prompt, date)}
    recipes = context.get('slot_recipes') or [
        {'message_index': index, 'formatter': 'plain', 'slots': [{'path': []}]}
        for index in outer]
    expected = {}
    styles = Counter()
    for recipe in recipes:
        index = recipe['message_index']
        text = request['messages'][index]['content']
        tree = scorer.decode_tool(text, recipe['formatter'])
        edits = []
        for slot in recipe['slots']:
            path = slot['path']
            value = get(tree, path)
            if not isinstance(value, str):
                raise AssertionError('Native recipe does not select a string')
            starts = list(find_occurrences(value, payload))
            if 'occurrences' in slot and len(starts) != slot['occurrences']:
                raise AssertionError('Semantic occurrence denominator changed')
            for start in starts:
                a, b = start + len(prefix), start + len(prefix) + len(goal)
                if value[a:b] != goal:
                    raise AssertionError('The complete original goal bytes differ')
                goal_tree = set_value(copy.deepcopy(tree), path,
                                      value[:a] + SENTINEL + value[b:])
                payload_tree = set_value(copy.deepcopy(tree), path,
                                         value[:start] + SENTINEL + value[start + len(payload):])
                edits.append({'field_path': path, 'semantic_goal_span': [a, b],
                              'semantic_payload_span': [start, start + len(payload)],
                              'goal_tree': goal_tree, 'payload_tree': payload_tree})
        expected[index] = {'tree': tree, 'formatter': recipe['formatter'], 'edits': edits}
        if recipe['formatter'] == 'yaml':
            import yaml
            for token in yaml.scan(text):
                if isinstance(token, yaml.tokens.ScalarToken):
                    styles['plain' if token.style is None else token.style] += 1
    matched = set()
    occurrences = location['occurrences']
    if len(occurrences) != sum(len(row['edits']) for row in expected.values()):
        raise AssertionError('Not all semantic payload occurrences were located')
    for occurrence in occurrences:
        index = occurrence['tool_message_index']
        a, b, message = outer[index]
        if occurrence['tool_char_span'] != [a, b]:
            raise AssertionError('Trusted original outer tool span differs')
        if prompt[a:b] != json.dumps(message['content'], ensure_ascii=False):
            raise AssertionError('Trusted original outer tool body differs')
        row = expected[index]
        text = message['content']
        ga, gb = inner_span(text, a, occurrence['goal_char_span'])
        pa, pb = inner_span(text, a, occurrence['payload_char_span'])
        if not 0 <= pa <= ga < gb <= pb <= len(text):
            raise AssertionError('Complete goal is outside its complete payload')
        try:
            goal_tree = scorer.decode_tool(text[:ga] + SENTINEL + text[gb:], row['formatter'])
            payload_tree = scorer.decode_tool(text[:pa] + SENTINEL + text[pb:], row['formatter'])
        except Exception as exc:
            raise AssertionError('Mapped span cuts native scalar syntax: ' + type(exc).__name__) from exc
        matches = [offset for offset, edit in enumerate(row['edits'])
                   if goal_tree == edit['goal_tree'] and payload_tree == edit['payload_tree']]
        if len(matches) != 1:
            raise AssertionError('Span replacement does not edit exactly one complete native goal/payload')
        identity = (index, matches[0])
        if identity in matched:
            raise AssertionError('One physical occurrence was counted twice')
        matched.add(identity)
    ordered = sorted(occurrences, key=lambda value: value['payload_char_span'][0])
    if occurrences != ordered or location['goal_char_span'] != ordered[0]['goal_char_span']:
        raise AssertionError('Scored goal is not the first complete payload occurrence')
    return {'native_parse_goal_substitution_verified': True,
            'native_parse_payload_substitution_verified': True,
            'all_occurrences_checked': len(occurrences), 'yaml_scalar_styles_seen': dict(styles)}


def common_location(location):
    names = ('tool_char_span', 'tool_message_index', 'tool_name',
             'payload_char_span', 'goal_char_span')
    return {**{name: location.get(name) for name in names},
            'occurrences': [{name: row.get(name) for name in names}
                            for row in location['occurrences']]}


class DeterministicSpanContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        cls.spans = load('independent_deterministic_outer_spans',
                         ROOT / 'experiments/agentdojo-first-tool/spans.py')
        cls.dataset = json.loads((OUTPUT / 'dataset.json').read_text())
        cls.contexts = cls.dataset['contexts']
        cls.protocol = json.loads((OUTPUT / 'score_protocol.json').read_text())
        cls.source_hashes = {str(path): sha(path) for path in (
            HERE / 'scorer.py', HERE / 'broad_campaign.py', HERE / 'broad_validation.py')}
        cls.locator_sha256 = sha(HERE / 'deterministic_spans.py')
        cls.saved = []
        for path in sorted((OUTPUT / 'scores').glob('*/score.json')):
            result = json.loads(path.read_text())
            if result.get('status') == 'complete':
                cls.saved.append((path, sha(path), result))
        if len(cls.saved) != 139:
            raise AssertionError('Expected the stopped 139-complete-candidate snapshot')
        REPORT['snapshot_complete_candidates'] = len(cls.saved)
        REPORT['source_dataset_sha256'] = sha(OUTPUT / 'dataset.json')
        REPORT['locator_sha256'] = cls.locator_sha256
        REPORT['source_score_snapshot'] = [{'path': str(path), 'sha256': digest}
                                          for path, digest, _ in cls.saved]

    @classmethod
    def tearDownClass(cls):
        REPORT['frozen_core_files_unchanged'] = all(sha(path) == digest
                                                   for path, digest in cls.source_hashes.items())
        REPORT['locator_unchanged_during_audit'] = sha(HERE / 'deterministic_spans.py') == cls.locator_sha256
        if not REPORT['frozen_core_files_unchanged'] or not REPORT['locator_unchanged_during_audit']:
            raise AssertionError('Source changed during the independent audit; rerun after freeze')

    def locate(self, request, context, prompt, date):
        allowed = {row['message_index'] for row in context['slot_recipes']} if context.get('slot_recipes') else None
        with deadline(TIMEOUT):
            began = time.monotonic()
            location = locator.locate_payload(self.tokenizer, self.spans, request, prompt,
                                               date, allowed_messages=allowed, context=context)
            elapsed = time.monotonic() - began
        return location, elapsed

    def check_tokens(self, request, prompt, location, encoded=None):
        encoded = encoded or self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.asarray(encoded['input_ids'], dtype=np.int64)
        offsets = np.asarray(encoded['offset_mapping'], dtype=np.int64)
        self.assertTrue(all(value < 199998 for value in self.tokenizer.encode(request['payload_text'], add_special_tokens=False)))
        all_payload = np.zeros(len(ids), dtype=bool)
        for occurrence in location['occurrences']:
            all_payload |= scorer.overlap(offsets, occurrence['payload_char_span'])
        self.assertFalse(np.any(ids[all_payload] >= 199998))
        positions = np.flatnonzero(scorer.overlap(offsets, location['payload_char_span']))
        goal = scorer.overlap(offsets, location['goal_char_span'])[positions]
        self.assertTrue(goal.any())
        return ids, offsets, positions, goal

    def test_01_saved_139_six_contexts(self):
        for path, expected_sha, result in self.saved:
            self.assertEqual(len(result['per_case']), 6)
            self.assertEqual([row['case_id'] for row in result['per_case']],
                             [row['case_id'] for row in self.contexts])
            for context, record in zip(self.contexts, result['per_case']):
                with self.subTest(score=path.parent.name, case=context['case_id']):
                    artifact = Path(record['token_artifact'])
                    for name, digest in record['artifact_sha256'].items():
                        self.assertEqual(sha(artifact.parent / name), digest)
                    request = json.loads((artifact.parent / 'request.json').read_text())
                    self.assertEqual(request['actual_goal'], context['actual_goal'])
                    self.assertEqual(raw_sha(context['actual_goal']), record['actual_goal_sha256'])
                    self.assertEqual(request['payload_text'], result['template'].replace('{goal}', context['actual_goal']))
                    date = context.get('prompt_date', self.protocol['prompt_date'])
                    prompt = self.spans.render(self.tokenizer, request, date)
                    self.assertEqual(prompt, request['rendered_prompt'])
                    location, elapsed = self.locate(request, context, prompt, date)
                    semantics = independent_semantics(request, context, prompt, self.spans,
                                                      self.tokenizer, date, location)
                    ids, offsets, positions, goal = self.check_tokens(request, prompt, location)
                    with np.load(artifact, allow_pickle=False) as archive:
                        self.assertTrue(np.array_equal(ids, archive['input_ids']))
                        self.assertTrue(np.array_equal(offsets, archive['input_token_offsets']))
                        original_positions = archive['payload_token_indices']
                        original_goal = archive['goal_mask']
                        same_positions = np.array_equal(positions, original_positions)
                        same_goal = same_positions and np.array_equal(goal, original_goal)
                        old_global_goal = original_positions[original_goal]
                        new_global_goal = positions[goal]
                        self.assertTrue(np.array_equal(old_global_goal, new_global_goal))
                        source_rows = np.searchsorted(original_positions, positions)
                        self.assertTrue(np.all(source_rows < len(original_positions)))
                        self.assertTrue(np.array_equal(original_positions[source_rows], positions))
                        # A correct payload-boundary fix may drop a benign row.
                        # The immutable goal's native features and probability
                        # rows must still be exactly the original captured rows.
                        for name in ('native_bf16_bits', 'features_fp16', 'probabilities'):
                            self.assertTrue(np.array_equal(archive[name][original_goal],
                                                           archive[name][source_rows][goal]))
                        before_means = archive['probabilities'][original_goal].mean(0, dtype=np.float64)
                        after_means = archive['probabilities'][source_rows][goal].mean(0, dtype=np.float64)
                        self.assertTrue(np.array_equal(before_means, after_means))
                        self.assertEqual(float(after_means[2]), record['goal_user_mean'])
                        same_span = common_location(location) == common_location(record['location'])
                        if not (same_positions and same_goal and same_span):
                            # Diagnose the old result semantically; never force a known-wrong mask.
                            try:
                                old_semantics = independent_semantics(request, context, prompt, self.spans,
                                                                      self.tokenizer, date, record['location'])
                                old_semantically_valid = True
                            except Exception as exc:
                                old_semantics = {'error': type(exc).__name__ + ': ' + str(exc)}
                                old_semantically_valid = False
                            REPORT['compatibility_differences'].append({
                                'score_id': path.parent.name, 'case_id': context['case_id'],
                                'same_payload_token_positions': same_positions,
                                'same_relative_goal_mask': same_goal,
                                'same_global_goal_token_indices': True,
                                'same_goal_raw_features_and_probabilities': True,
                                'same_all_occurrence_goal_char_spans': all(
                                    old['goal_char_span'] == new['goal_char_span']
                                    for old, new in zip(record['location']['occurrences'], location['occurrences'])),
                                'source_npz_rows_for_precise_payload': source_rows.tolist(),
                                'precise_payload_token_indices': positions.tolist(),
                                'old_user_mean': float(before_means[2]),
                                'new_user_mean': float(after_means[2]),
                                'goal_role_mean_max_absolute_delta': float(np.max(np.abs(before_means - after_means))),
                                'old_extra_prefix_prompt_text_repr': repr(prompt[
                                    record['location']['payload_char_span'][0]:location['payload_char_span'][0]]),
                                'old_location': common_location(record['location']),
                                'new_location': common_location(location),
                                'new_native_semantics': semantics, 'old_native_semantics': old_semantics,
                                'old_semantically_valid': old_semantically_valid})
                            if old_semantically_valid:
                                self.fail('Both span interpretations parse correctly but differ; requires diagnosis')
                    REPORT['saved_cases'].append({
                        'score_id': path.parent.name, 'case_id': context['case_id'],
                        'request_sha256': sha(artifact.parent / 'request.json'), 'tokens_npz_sha256': sha(artifact),
                        'input_sha256_unchanged': hashlib.sha256(ids.tobytes()).hexdigest(),
                        'same_old_payload_positions': same_positions, 'same_old_goal_mask': same_goal,
                        'same_old_spans': same_span,
                        'same_global_goal_token_indices': True,
                        'same_goal_raw_features_and_probabilities': True,
                        'same_complete_goal_role_means': True,
                        'same_all_occurrence_goal_char_spans': all(
                            old['goal_char_span'] == new['goal_char_span']
                            for old, new in zip(record['location']['occurrences'], location['occurrences'])),
                        'locate_seconds': elapsed, **semantics})
            self.assertEqual(sha(path), expected_sha)

    def fixture(self, name, context, template, request=None):
        with deadline(TIMEOUT):
            request = request or scorer.render_messages(context, template)
            date = context.get('prompt_date', self.protocol['prompt_date'])
            prompt = self.spans.render(self.tokenizer, request, date)
            location, elapsed = self.locate(request, context, prompt, date)
            semantics = independent_semantics(request, context, prompt, self.spans,
                                              self.tokenizer, date, location)
            _, _, positions, goal = self.check_tokens(request, prompt, location)
        REPORT['fixture_cases'].append({'fixture': name, 'case_id': context.get('case_id', name),
                                        'locate_seconds': elapsed, 'goal_token_count': int(goal.sum()),
                                        'payload_token_count': len(positions), **semantics})

    def test_02_pretty_json_pending_all_six_with_deadline(self):
        template = json.loads((OUTPUT / 'debug_pending_candidate.json').read_text())['template']
        for context in self.contexts:
            with self.subTest(case=context['case_id']):
                self.fixture('previously_regex_hanging_pretty_json', context, template)

    def test_03_native_codec_edges_all_six(self):
        fixtures = {
            'ordinary_plain': 'Codec plain prefix {goal} ordinary suffix',
            'single_and_double_quotes': 'Codec single\'s and "double" prefix {goal} suffix\'s "quoted"',
            'newlines_and_linefold': 'Codec first line\n\nsecond line\n{goal}\nlast line\n',
            'long_folded_line': 'Codec ' + 'ordinary words '*100 + '{goal} ' + 'trailing words '*100,
            'unicode_controls': 'Codec Ω café 中文 \b\t\r\x07\x1b\u0085\u2028\u2029\n{goal}\nUnicode Ω',
            'backslashes_and_repr_quotes': 'Codec \\ path \' " \\\"\n{goal}\nEnd \\ \' "',
            'literal_json_braces': '{"role":"user","content":"{goal}","meta":{"kind":"codec"}}',
        }
        for name, template in fixtures.items():
            for context in self.contexts:
                with self.subTest(fixture=name, case=context['case_id']):
                    self.fixture(name, context, template)

    def test_04_explicit_yaml_scalar_styles(self):
        import yaml
        goal = 'Keep the entire supplied goal text intact'
        def run(name, template, content):
            payload = template.replace('{goal}', goal)
            self.assertEqual(yaml.safe_load(content)['value'], payload)
            base = self.contexts[3]
            messages = scorer.normalize(base['messages'])
            index = base['slot_recipes'][0]['message_index']
            messages[index]['content'] = content
            context = {'case_id': 'codec/' + name, 'actual_goal': goal,
                       'slot_recipes': [{'message_index': index, 'formatter': 'yaml',
                                         'slots': [{'path': ['value'], 'occurrences': 1}]}]}
            request = {'messages': messages, 'tools': copy.deepcopy(base['tools']), 'actual_goal': goal,
                       'payload_template': template, 'payload_text': payload}
            self.fixture(name, context, template, request)
        plain = 'Codec prefix {goal} codec suffix'
        run('yaml_plain', plain, 'value: ' + plain.replace('{goal}', goal) + '\n')
        for style, name in [("'", 'yaml_single'), ('"', 'yaml_double')]:
            template = 'Codec quote\'s "quoted" ' + 'fold words '*20 + '{goal} end'
            content = yaml.safe_dump({'value': template.replace('{goal}', goal)},
                                     default_style=style, width=35, allow_unicode=False)
            run(name, template, content)
        run('yaml_plain_fold', 'Codec first line second line {goal} last line',
            'value: Codec first line\n  second line ' + goal + '\n  last line\n')
        run('yaml_double_escaped_break', 'Codec{goal}suffix',
            'value: "Codec\\\n  ' + goal + 'suffix"\n')
        run('yaml_literal_strip', 'Codec first line\n{goal}\nlast line',
            'value: |-\n  Codec first line\n  ' + goal + '\n  last line\n')
        run('yaml_folded_strip', 'Codec first line {goal} last line',
            'value: >-\n  Codec first line\n  ' + goal + '\n  last line\n')
        run('yaml_literal_explicit_indent', 'Codec first line\n{goal}\nlast line',
            'value: |2-\n  Codec first line\n  ' + goal + '\n  last line\n')
        run('yaml_anchor_and_tag', plain,
            'value: &codec !!str "' + plain.replace('{goal}', goal) + '"\n')


def main():
    global OUTPUT, MODEL, TIMEOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--timeout-per-case', type=float, default=5.)
    args = parser.parse_args()
    OUTPUT, MODEL, TIMEOUT = args.output.resolve(), args.model.resolve(), args.timeout_per_case
    began = time.monotonic()
    REPORT['snapshot_started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeterministicSpanContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT['status'] = ('passed_with_legacy_payload_boundary_corrections'
                        if result.wasSuccessful() and REPORT['compatibility_differences']
                        else 'passed' if result.wasSuccessful() else 'failed')
    REPORT['algorithm_correctness_passed'] = result.wasSuccessful()
    REPORT['tests_run'] = result.testsRun
    REPORT['test_failures'] = [{'test': str(test), 'details': error} for test, error in result.failures + result.errors]
    REPORT['saved_case_count'] = len(REPORT['saved_cases'])
    REPORT['fixture_case_count'] = len(REPORT['fixture_cases'])
    REPORT['source_output'] = str(OUTPUT)
    REPORT['dataset_sha256'] = sha(OUTPUT / 'dataset.json')
    REPORT['cached_complete_candidates'] = REPORT.get('snapshot_complete_candidates', 0)
    REPORT['cached_contexts'] = len(REPORT['saved_cases'])
    REPORT['cached_exact_input_offsets_indices_masks_spans'] = (
        REPORT['cached_complete_candidates'] == 139 and REPORT['cached_contexts'] == 834
        and not REPORT['compatibility_differences']
        and all(row['same_old_payload_positions'] and row['same_old_goal_mask']
                and row['same_old_spans'] for row in REPORT['saved_cases']))
    REPORT['passed'] = (result.wasSuccessful()
                        and REPORT['cached_exact_input_offsets_indices_masks_spans']
                        and REPORT.get('frozen_core_files_unchanged', False)
                        and REPORT.get('locator_unchanged_during_audit', False))
    REPORT['cached_goal_objective_equivalent'] = (
        result.wasSuccessful() and REPORT['cached_contexts'] == 834
        and all(row['same_global_goal_token_indices']
                and row['same_goal_raw_features_and_probabilities']
                and row['same_complete_goal_role_means']
                and row['same_all_occurrence_goal_char_spans'] for row in REPORT['saved_cases']))
    REPORT['cached_input_ids_and_offsets_equal'] = result.wasSuccessful() and REPORT['cached_contexts'] == 834
    REPORT['cached_global_goal_indices_features_probabilities_equal'] = REPORT['cached_goal_objective_equivalent']
    REPORT['cached_goal_role_mean_max_absolute_delta'] = 0. if REPORT['cached_goal_objective_equivalent'] else None
    REPORT['legacy_payload_boundary_correction_case_count'] = len(REPORT['compatibility_differences'])
    REPORT['legacy_payload_boundary_correction_candidate_count'] = len({
        row['score_id'] for row in REPORT['compatibility_differences']})
    REPORT['strict_cache_compatibility_passed'] = REPORT['passed']
    REPORT['cache_migration_requires_payload_metadata_correction'] = bool(REPORT['compatibility_differences'])
    REPORT['maximum_locator_seconds'] = max([v['locate_seconds'] for v in
                                             REPORT['saved_cases'] + REPORT['fixture_cases']], default=0.)
    REPORT['elapsed_seconds'] = time.monotonic() - began
    REPORT['test_source_sha256'] = sha(__file__)
    REPORT['snapshot_ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    scorer.write(OUTPUT / 'span_locator_audit.json', REPORT)
    print(json.dumps({key: REPORT[key] for key in ('status', 'saved_case_count', 'fixture_case_count',
                                                  'maximum_locator_seconds', 'elapsed_seconds')}), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
