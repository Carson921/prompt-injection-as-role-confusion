"""CPU campaign contract checks; no backbone construction or HTTP request.

Real local tokenizer, delimiter validation, structural seed generation and the
upstream AutoDAN operators run end to end. FakeScorer only supplies explicitly
labelled six-case probabilities, never experimental evidence.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import broad_campaign as campaign
import scorer

DATASET = HERE / 'outputs/near-delimiter-20261104/dataset.json'


def result_for(contexts, template, fitness=.3):
    probabilities = {role: (1-fitness)/5 for role in scorer.ROLES}
    probabilities['user'] = fitness
    cases = [dict(case_id=context['case_id'], suite=context['suite'],
                  goal_user_mean=fitness, goal_target_mean=fitness,
                  target_role='user', target_role_index=2,
                  probabilities=dict(goal=probabilities)) for context in contexts]
    return dict(status='complete', complete=True, template=template,
                target_role='user', target_role_index=2, fitness=fitness,
                goal_target_mean=fitness, goal_role_means=probabilities,
                per_case=cases, completed_cases=6, expected_cases=6,
                protocol_sha256='CPU-FAKE-NO-BACKBONE')


class FakeScorer:
    calls = []

    def __init__(self, dataset, output, **kwargs):
        assert kwargs['target_role'] == 'user'
        self.contexts = json.loads(Path(dataset).read_text())['contexts']
        self.closed = False

    def score(self, template):
        self.calls.append(template)
        value = .1 + int(campaign.raw_sha(template)[:6], 16) / (2**24-1) * .5
        return result_for(self.contexts, template, value)

    def prepare(self, context, template):
        assert template.count('{goal}') == 1
        return None

    def close(self):
        self.closed = True


class CampaignContracts(unittest.TestCase):
    def setUp(self):
        self.contexts = json.loads(DATASET.read_text())['contexts']
        self.case_ids = [row['case_id'] for row in self.contexts]

    def test_actual_user_column_and_all_six_are_required(self):
        result = result_for(self.contexts, 'Before {goal}', .32)
        self.assertAlmostEqual(campaign.verify_result(result, self.case_ids), .32)
        for modification in ('target', 'partial', 'case_order', 'wrong_aggregate', 'wrong_column'):
            row = json.loads(json.dumps(result))
            if modification == 'target':
                row['target_role'], row['target_role_index'] = 'cot', 3
            elif modification == 'partial':
                row['per_case'].pop()
            elif modification == 'case_order':
                row['per_case'].reverse()
            elif modification == 'wrong_aggregate':
                row['fitness'] = .99
            else:
                row['per_case'][0]['goal_target_mean'] = .99
            with self.assertRaises(ValueError, msg=modification):
                campaign.verify_result(row, self.case_ids)

    def test_bulk_seed_formats_and_exact_text_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            text = '\n  Before {goal}\nAfter.\n'
            variants = [dict(templates=[dict(template=text, family='bulk', meta={'x': 1})]),
                        [text], [dict(template=text)], dict(template=text, target_role='cot')]
            for index, value in enumerate(variants):
                path = folder / f'format{index}.json'
                path.write_text(json.dumps(value))
                records = campaign.load_initial_records([path])
                self.assertEqual(records[0]['template'], text)
                self.assertEqual(records[0]['file_sha256'], campaign.search.digest(path.read_bytes()))
            path = folder / 'plain.txt'
            path.write_text(text)
            self.assertEqual(campaign.load_initial_records([path])[0]['template'], text)

    def test_openrouter_length_contract_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as folder:
            api = campaign.CampaignMutator(folder, 'unit-test-secret', 'unused', 4, target_role='user')
            api.wrapper_limit = 2048
            marker = campaign.search.GOAL_MARKER
            template = 'Before ' + marker + '\nAfter.'
            response = dict(choices=[dict(message=dict(content=json.dumps(dict(templates=[template]))))])
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self): return json.dumps(response).encode()
            with patch.object(campaign.request, 'urlopen', return_value=Response()) as mocked:
                values = api.propose([template], 1)
            self.assertEqual(values, [template])
            body = json.loads(mocked.call_args.args[0].data)
            self.assertIn('1 to 2048', body['messages'][0]['content'])
            self.assertNotIn('1 to 128', body['messages'][0]['content'])
            self.assertIn('mean P(user)', body['messages'][0]['content'])
            self.assertIn(marker, body['messages'][1]['content'])
            self.assertNotIn('{goal}', body['messages'][1]['content'])
            self.assertGreater(body['max_tokens'], 2048)
            self.assertNotIn('unit-test-secret', (Path(folder) / 'api_calls.jsonl').read_text())
            with self.assertRaisesRegex(ValueError, 'protected abstract'):
                api.propose(['Before {goal}'])

    def test_prefix_mutation_prefers_causally_preceding_changes(self):
        class Advanced:
            calls = 0
            @classmethod
            def mutate(cls, template, *args, **kwargs):
                cls.calls += 1
                if cls.calls < 3:
                    return template + ' After.', dict(operator='suffix')
                return 'New prefix. ' + template, dict(operator='prefix')
        value, event = campaign.prefix_mutate(Advanced, 'Before {goal}', random.Random(1), None, 128, None)
        self.assertTrue(value.startswith('New prefix.'))
        self.assertTrue(event['prefix_changed'])
        self.assertEqual(event['causal_prefix_attempts'], 3)

    def test_complete_campaign_resume_and_preserved_intermediate_templates(self):
        FakeScorer.calls = []
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            folder = Path(folder)
            args = campaign.arguments(['--dataset', str(DATASET), '--output', str(folder),
                '--offline-mutation', '--seeds', '7', '--stage-lengths', '128', '--population', '4',
                '--batch-size', '2', '--scores-per-stage', '4', '--cycles', '1',
                '--breadth-per-stage', '0', '--max-new-scores', '4'])
            with patch.object(scorer, 'Scorer', FakeScorer):
                first = campaign.run(args)
            self.assertEqual(first['unique_scores'], 4)
            self.assertFalse(first['target_reached'])
            self.assertEqual(first['status'], 'score_budget')
            self.assertEqual(len(FakeScorer.calls), 4)
            bank = next((folder / 'seed_banks').glob('*.json'))
            bank_before = bank.read_bytes()
            args.resume, args.cycles, args.max_new_scores = True, 2, 6
            with patch.object(scorer, 'Scorer', FakeScorer):
                second = campaign.run(args)
            self.assertEqual(second['unique_scores'], 6)
            self.assertEqual(len(FakeScorer.calls), 6)
            self.assertEqual(len(set(FakeScorer.calls)), 6)
            self.assertEqual(bank_before, bank.read_bytes())
            saved = json.loads((folder / 'asr_candidates.json').read_text())
            self.assertTrue(saved['templates'])
            self.assertEqual(saved['target_role'], 'user')
            for item in saved['templates']:
                metadata = json.loads(Path(item['metadata_path']).read_text())
                exact_text = Path(item['template_path']).read_text()
                self.assertEqual(exact_text, metadata['template'])
                self.assertEqual(campaign.raw_sha(exact_text), metadata['template_raw_sha256'])
                self.assertEqual([row['case_id'] for row in metadata['per_case']], self.case_ids)
                self.assertEqual(exact_text.count('{goal}'), 1)
                self.assertEqual(metadata['target_role_index'], 2)
            rows = campaign.search.read_lines(folder / 'scores.jsonl')
            self.assertEqual(len(rows), 6)
            self.assertTrue(any(row['origin'].startswith('autodan_') for row in rows))
            args.crossover = .123
            with patch.object(scorer, 'Scorer', FakeScorer), self.assertRaisesRegex(ValueError, 'different frozen protocol'):
                campaign.run(args)

    def test_prepare_failure_is_unscored_recorded_and_not_retried(self):
        class RejectOnce(FakeScorer):
            attempts = []
            rejected = False
            def prepare(self, context, template):
                self.attempts.append(template)
                if not self.rejected:
                    type(self).rejected = True
                    raise ValueError('CPU fixture: native serialization/span is unrepresentable')
                return None
        FakeScorer.calls = []
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            folder = Path(folder)
            args = campaign.arguments(['--dataset', str(DATASET), '--output', str(folder),
                '--offline-mutation', '--seeds', '3', '--stage-lengths', '128', '--population', '4',
                '--batch-size', '2', '--scores-per-stage', '3', '--cycles', '1',
                '--breadth-per-stage', '0', '--max-new-scores', '3'])
            with patch.object(scorer, 'Scorer', RejectOnce):
                result = campaign.run(args)
            rejections = campaign.search.read_lines(folder / 'proposal_rejections.jsonl')
            self.assertEqual(len(rejections), 1)
            rejected = rejections[0]
            self.assertEqual(rejected['completed_native_forwards'], 0)
            self.assertFalse(rejected['counted_as_fitness'])
            self.assertNotIn(rejected['template'], FakeScorer.calls)
            self.assertEqual(RejectOnce.attempts.count(rejected['template']), 1)
            self.assertEqual(result['unique_scores'], 3)
            self.assertEqual(result['rejected_unscored_proposals'], 1)
            self.assertEqual(len(FakeScorer.calls), 3)


if __name__ == '__main__':
    unittest.main()
