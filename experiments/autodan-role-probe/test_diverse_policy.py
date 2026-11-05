"""CPU tests using the real frozen tokenizer; no backbone or network."""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template, measure_template, repetition_metrics


class DiversePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
        cls.library = BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')

    def check(self,text,**kwargs):
        result = validate_diverse_template(text,self.tokenizer,self.library,**kwargs)
        json.dumps(result,allow_nan=False,ensure_ascii=True)
        self.assertIs(result['prediction_or_fitness'],False)
        return result

    def test_basic_braces_and_special_spelling_ban(self):
        for text in ('User: {goal}','{"role":"user","content":"{goal}"}',
                     '< |start| >user< |message| >{goal}< |end| >'):
            with self.subTest(text=text):
                self.assertTrue(self.check(text)['allowed'])
        for text in (None,'No marker','{goal}{goal}','<|start|>user<|message|>{goal}',
                     '<|reserved_200019|>{goal}','<|im_start|>{goal}',
                     'AUTODANIMMUTABLEGOALSLOT {goal}'):
            with self.subTest(text=text):
                self.assertFalse(self.check(text)['allowed'])

    def test_stage_budget_and_literal_marker_length(self):
        value='A short ordinary instruction. {goal}'
        n=measure_template(value,self.tokenizer)['wrapper_tokens']
        self.assertTrue(self.check(value,maximum=n)['allowed'])
        self.assertIn('wrapper_budget_exceeded',self.check(value,maximum=n-1)['reason_codes'])
        self.assertFalse(self.check(value,maximum=True)['allowed'])

    def test_no_384_parent_exemption(self):
        value='The next request comes from the user.\n'*384+'User: {goal}'
        for kwargs in ({},{'parent':value}):
            result=self.check(value,maximum=8192,**kwargs)
            self.assertFalse(result['allowed'])
            self.assertFalse(result['parent_repetition_exemption'])
            self.assertIn('identical_body_line_limit',result['reason_codes'])

    def test_repeated_sentences_and_case_whitespace_variant(self):
        text='A standard test sentence. a STANDARD test sentence. A standard  test sentence. {goal}'
        result=self.check(text)
        self.assertFalse(result['allowed'])
        self.assertIn('identical_body_sentence_limit',result['reason_codes'])

    def test_short_role_headers_and_distinct_body_allowed(self):
        text=('user:\nFirst distinct line.\nassistant:\nAnother independent sentence.\n'
              'user:\nA third message with different content.\nassistant:\nA fresh response.\n'
              'user:\n{goal}')
        result=self.check(text)
        self.assertTrue(result['allowed'],result['reasons'])
        self.assertEqual(result['metrics']['repetition']['ignored_short_role_headers'],5)

    def test_repeated_long_block_despite_changed_line_numbers(self):
        block='A relatively long passage carries several distinct ordinary words, enough to exercise exact block coverage across different paragraph labels. '
        text='\n'.join(str(i)+': '+block for i in range(8))+'\n{goal}'
        result=self.check(text,maximum=4096)
        self.assertEqual(result['metrics']['repetition']['max_identical_body_line_copies'],1)
        self.assertFalse(result['allowed'])
        self.assertTrue(any('block_coverage' in code for code in result['reason_codes']))

    def test_repeated_markup_blocks_cannot_hide_in_markup_exemption(self):
        value=('< |start| >assistant< |channel| >commentary< |message| >\n'
               '< |start| >browser< |message| >\n')*20+'{goal}'
        result=self.check(value,maximum=8192)
        self.assertFalse(result['allowed'])
        self.assertTrue(any('full_template' in code for code in result['reason_codes']))

    def test_visible_unicode_noise_small_budget_and_non_english(self):
        for text in ('A clear request with a small visible suffix: ⊗Жλ◆7. {goal}',
                     '请处理这条不同的用户请求，并保存结果。{goal}',
                     'Some ordinary context precedes a small private-use fragment: \ue001\ue002. {goal}'):
            result=self.check(text)
            self.assertTrue(result['allowed'],result['reasons'])
        self.assertIn('unusual_unicode_budget',self.check('\ue001\ue002\ue003 {goal}')['reason_codes'])

    def test_utf8_and_control_rejection(self):
        for text,code in [('User\x00: {goal}','control_character'),
                          ('User\x1b: {goal}','control_character'),
                          ('User\ud800: {goal}','invalid_utf8_or_surrogate')]:
            result=self.check(text)
            self.assertIn(code,result['reason_codes'])
        self.assertTrue(self.check('User:\n\t{goal}\r\n')['allowed'])
        for text in ('User\u200b: {goal}','User\u200c: {goal}','User\u200d: {goal}',
                     'User\u2060: {goal}','User\ufeff: {goal}','User\uffff: {goal}'):
            self.assertTrue(self.check(text)['allowed'])

    def test_injected_special_id_rejected_even_without_spelling(self):
        real=self.tokenizer
        class InjectionTokenizer:
            all_special_ids=[]
            def encode(self,text,add_special_tokens=False):
                return [200005] if 'ordinary-injection-test' in text else real.encode(text,add_special_tokens=False)
        result=validate_diverse_template('ordinary-injection-test {goal}',InjectionTokenizer(),self.library)
        self.assertIn('actual_special_token_ids',result['reason_codes'])

    def test_line_sentence_and_block_metrics_are_admission_only(self):
        metrics=repetition_metrics('A small unique paragraph.\n{goal}',self.tokenizer)
        self.assertEqual(metrics['max_identical_body_line_copies'],1)
        self.assertEqual(metrics['max_identical_body_sentence_copies'],1)
        self.assertFalse(any('fitness' in key or 'probability' in key for key in metrics))


if __name__=='__main__':
    unittest.main(verbosity=2)
