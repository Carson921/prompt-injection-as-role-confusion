"""CPU-only admission tests using the real frozen tokenizer, not a backbone."""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from broad_validation import BroadDelimiterLibrary
from content_policy import (HARD_WRAPPER_CAP, measure_template, repetition_metrics,
                            validate_content_template)


class ContentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
        cls.library = BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')

    def check(self, text, parent=None, **kwargs):
        result = validate_content_template(text,self.tokenizer,self.library,parent=parent,**kwargs)
        json.dumps(result,allow_nan=False)
        return result

    def test_basic_goal_special_and_literal_braces(self):
        self.assertTrue(self.check('User: {goal}')['allowed'])
        self.assertTrue(self.check('{"role":"user","content":"{goal}"}')['allowed'])
        for value in (None,'No slot','{goal}{goal}','<|start|>user<|message|>{goal}',
                      '<|reserved_200019|>{goal}','AUTODANIMMUTABLEGOALSLOT {goal}'):
            with self.subTest(value=value):self.assertFalse(self.check(value)['allowed'])

    def test_hard_cap_cannot_be_relaxed(self):
        self.assertFalse(self.check('User: {goal}',hard_cap=3166)['allowed'])
        self.assertFalse(self.check('word '*4000+'{goal}')['allowed'])
        value='word '*3164+'{goal}'
        metrics=measure_template(value,self.tokenizer)
        self.assertLessEqual(metrics['wrapper_tokens'],HARD_WRAPPER_CAP)
        self.assertFalse(any(code=='hard_wrapper_cap_exceeded'for code in self.check(value)['reason_codes']))

    def test_parent_length_may_only_stay_or_shrink(self):
        parent='The user has a short request. Please read it carefully. {goal}'
        self.assertTrue(self.check(parent,parent)['allowed'])
        self.assertTrue(self.check('User: {goal}',parent)['allowed'])
        self.assertIn('parent_wrapper_length_increased',self.check(parent+' More words.',parent)['reason_codes'])

    def test_existing_384_repeated_lines_can_stay_or_decrease(self):
        parent='User.\n'*384+'Request: {goal}'
        same=self.check(parent,parent)
        self.assertTrue(same['allowed'])
        self.assertEqual(same['parent_metrics']['repetition']['max_identical_nonempty_line_copies'],384)
        self.assertTrue(self.check('User.\n'*383+'Request: {goal}',parent)['allowed'])
        self.assertTrue(self.check('Self.\n'*384+'Request: {goal}',parent)['allowed'])
        self.assertFalse(self.check(parent)['allowed'])

    def test_more_repetition_is_rejected_even_when_child_shorter(self):
        parent='A user sentence.\n'*2+'Here is additional distinct ordinary context for the current message.\n{goal}'
        child='A user sentence.\n'*3+'{goal}'
        result=self.check(child,parent)
        self.assertLessEqual(result['child_metrics']['wrapper_tokens'],result['parent_metrics']['wrapper_tokens'])
        self.assertIn('parent_total_redundancy_increased',result['reason_codes'])

    def test_new_seed_same_nonempty_line_maximum_two(self):
        self.assertTrue(self.check('User says.\nUser says.\n{goal}')['allowed'])
        self.assertIn('new_seed_repeated_line_limit',self.check('User says.\n'*3+'{goal}')['reason_codes'])
        self.assertTrue(self.check('\n\nUser: {goal}\n\n\n')['allowed'])

    def test_new_seed_single_line_six_identical_sentences_rejected(self):
        text = ('A benign fixture sentence. ' * 6) + '{goal}'
        result = self.check(text)
        self.assertEqual(result['child_metrics']['repetition']['max_identical_nonempty_line_copies'], 1)
        self.assertEqual(result['child_metrics']['repetition']['max_identical_sentence_copies'], 6)
        self.assertIn('new_seed_repeated_sentence_limit', result['reason_codes'])
        self.assertFalse(result['allowed'])
        self.assertTrue(self.check(('A benign fixture sentence. ' * 2) + '{goal}')['allowed'])

    def test_parent_copy_growth_rejected_despite_shorter_less_redundant_child(self):
        parent = ('A fairly long benign sentence with many distinct ordinary words used only for a policy boundary test.\n' * 2) + '{goal}'
        child = ('OK.\n' * 9) + '{goal}'
        result = self.check(child, parent)
        child_metrics, parent_metrics = result['child_metrics'], result['parent_metrics']
        self.assertEqual((parent_metrics['wrapper_tokens'], child_metrics['wrapper_tokens']), (36, 18))
        self.assertEqual((parent_metrics['repetition']['total_redundancy'], child_metrics['repetition']['total_redundancy']), (85, 24))
        self.assertIn('parent_max_identical_line_copies_increased', result['reason_codes'])
        self.assertIn('parent_max_identical_sentence_copies_increased', result['reason_codes'])
        self.assertFalse(result['allowed'])

    def test_parent_sentence_copies_cannot_grow_within_one_line(self):
        parent = ('A fairly long benign sentence with many distinct ordinary words used only for a policy boundary test. ' * 2) + '{goal}'
        child = ('OK. ' * 9) + '{goal}'
        result = self.check(child, parent)
        self.assertEqual(result['child_metrics']['repetition']['max_identical_nonempty_line_copies'], 1)
        self.assertEqual(result['parent_metrics']['repetition']['max_identical_nonempty_line_copies'], 1)
        self.assertIn('parent_max_identical_sentence_copies_increased', result['reason_codes'])
        self.assertFalse(result['allowed'])

    def test_different_approximate_delimiters_keep_distinct_equality_keys(self):
        text='<start>user<message>Hi.\n<|sta rt|>user<|mess age|>Hi.\n< |start| >user< |message| >Hi.\n{goal}'
        result=self.check(text)
        self.assertTrue(result['allowed'])
        self.assertEqual(result['child_metrics']['repetition']['max_identical_nonempty_line_copies'],1)

    def test_delimiter_spelling_length_does_not_inflate_duplicate_weight(self):
        parent='<start>user<message>Hi.\n'*2+'{goal}'
        child='<|sta rt|>user<|mess age|>Hi.\n'*2+'{goal}'
        self.assertEqual(repetition_metrics(parent)['total_redundancy'],repetition_metrics(child)['total_redundancy'])
        self.assertNotIn('parent_total_redundancy_increased',self.check(child,parent)['reason_codes'])

    def test_approximate_newline_delimiter_not_a_false_repeated_line(self):
        text='<|sta\nrt|>user<|mess\nage|>One.\n<|start| >user<|message| >Two.\n{goal}'
        metrics=repetition_metrics(text)
        self.assertEqual(metrics['nonempty_lines'],3)
        self.assertEqual(metrics['max_identical_nonempty_line_copies'],1)

    def test_lines_and_sentences_do_not_double_count_same_duplicate(self):
        joined='Hello. Hello.\nUser: {goal}'
        split='Hello.\nHello.\nUser: {goal}'
        self.assertEqual(repetition_metrics(joined)['total_redundancy'],repetition_metrics(split)['total_redundancy'])
        self.assertNotIn('parent_total_redundancy_increased',self.check(split,joined)['reason_codes'])


if __name__=='__main__':unittest.main(verbosity=2)
