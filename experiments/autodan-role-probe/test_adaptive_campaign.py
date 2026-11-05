"""CPU-only adaptive-driver contracts; all synthetic captures stay temporary.

Real tokenizer, delimiter validation, upstream AutoDAN operators and source
proposal files are used.  FakeBoundedScorer returns labelled synthetic results
solely for control-flow tests. They are never experimental fitness evidence.
"""
from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adaptive_campaign as campaign
import bounded_scorer
import scorer

DATASET = HERE / 'outputs/user90-precise-20261104/dataset.json'


def complete_result(contexts, template, values=None):
    values = values or [.75, .80, .85, .90, .95, .55]
    cases = []
    for context, value in zip(contexts, values):
        probabilities = {role: ((1-value)/5 if role != 'user' else value) for role in scorer.ROLES}
        cases.append(dict(status='complete', case_id=context['case_id'], suite=context['suite'],
                          goal_user_mean=value, goal_target_mean=value,
                          target_role='user', target_role_index=2,
                          probabilities={'goal': probabilities}))
    means = {role: float(np.mean([row['probabilities']['goal'][role] for row in cases]))
             for role in scorer.ROLES}
    return dict(status='complete', complete=True, template=template, fitness=means['user'],
                target_role='user', target_role_index=2, goal_target_mean=means['user'],
                goal_role_means=means, per_case=cases, expected_cases=6, completed_cases=6,
                protocol_sha256='SYNTHETIC-CPU-TEST-NOT-NATIVE-EVIDENCE', cache_hit=False)


def pruned_result(contexts, template, threshold):
    cases = complete_result(contexts, template, [.0] * 6)['per_case'][:2]
    return dict(status='pruned_upper_bound', complete=False, template=template, fitness=None,
                target_role='user', target_role_index=2, per_case=cases,
                completed_cases=2, expected_cases=6, threshold=threshold, upper_bound=4/6,
                protocol_sha256='SYNTHETIC-CPU-TEST-NOT-NATIVE-EVIDENCE')


class FakeBoundedScorer:
    calls = []
    prepare_calls = []

    def __init__(self, dataset, output, **kwargs):
        assert kwargs['target_role'] == 'user'
        self.contexts = json.loads(Path(dataset).read_text())['contexts']
        self.output = Path(output)
        self.closed = False

    def prepare(self, context, template):
        assert template.count('{goal}') == 1
        type(self).prepare_calls.append((context['case_id'], template))

    def score_if_can_improve(self, template, threshold):
        assert not self.closed
        if threshold == 0:
            result = complete_result(self.contexts, template)
        else:
            result = pruned_result(self.contexts, template, threshold)
            path = self.output / 'screening' / campaign.raw_sha(template)[:24] / 'synthetic_test_only.json'
            campaign.search.write(path, dict(result, synthetic_test_only=True))
            result['artifact'] = str(path)
        type(self).calls.append((template, threshold, result['status']))
        return result

    def close(self):
        self.closed = True


class HardCrash(BaseException):
    """Bypass graceful interruption to retain the last on-disk checkpoint."""


class AdaptiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts = json.loads(DATASET.read_text())['contexts']
        cls.case_ids = [row['case_id'] for row in cls.contexts]

    def setUp(self):
        FakeBoundedScorer.calls = []
        FakeBoundedScorer.prepare_calls = []

    def args(self, folder):
        return campaign.arguments(['--dataset', str(DATASET), '--output', str(folder),
            '--offline-mutation', '--seeds', '7', '--stage-lengths', '128',
            '--population', '4', '--batch-size', '2', '--scores-per-stage', '4',
            '--long-stage-scores', '4', '--cycles', '1', '--breadth-per-stage', '0',
            '--max-new-scores', '20', '--selection-temperature', '.1'])

    def run_fake(self, args):
        with patch.object(bounded_scorer, 'BoundedScorer', FakeBoundedScorer), \
             patch.object(campaign.request, 'urlopen', side_effect=AssertionError('No HTTP in CPU tests')), \
             redirect_stdout(io.StringIO()):
            return campaign.run(args)

    def test_actual_user_column_full_denominator_and_exact_mean(self):
        result = complete_result(self.contexts, 'User: {goal}')
        self.assertEqual(campaign.verify_result(result, self.case_ids),
                         float(np.mean([.75, .80, .85, .90, .95, .55])))
        for modification in ('top_role', 'case_role', 'wrong_column', 'weighted_mean', 'partial', 'order'):
            row = copy.deepcopy(result)
            if modification == 'top_role': row.update(target_role='cot', target_role_index=3)
            elif modification == 'case_role': row['per_case'][0].update(target_role='cot', target_role_index=3)
            elif modification == 'wrong_column': row['per_case'][0]['goal_target_mean'] = .99
            elif modification == 'weighted_mean': row['fitness'] = .99
            elif modification == 'partial': row['per_case'].pop()
            else: row['per_case'].reverse()
            with self.assertRaises(ValueError, msg=modification):
                campaign.verify_result(row, self.case_ids)

    def test_pruned_gate_rejects_wrong_role_column_count_and_bound(self):
        valid = pruned_result(self.contexts, 'User: {goal}', .8)
        campaign.verify_pruned_result(valid, self.case_ids, .8)
        for modification in ('top_role', 'case_role', 'case_column', 'aggregate', 'fitness',
                             'complete', 'count', 'order', 'bound', 'threshold', 'equality'):
            row = copy.deepcopy(valid)
            threshold = .8
            if modification == 'top_role': row.update(target_role='cot', target_role_index=3)
            elif modification == 'case_role': row['per_case'][0].update(target_role='cot', target_role_index=3)
            elif modification == 'case_column': row['per_case'][0]['goal_target_mean'] = .5
            elif modification == 'aggregate': row['goal_target_mean'] = .0
            elif modification == 'fitness': row['fitness'] = .0
            elif modification == 'complete': row['complete'] = True
            elif modification == 'count': row['completed_cases'] = 3
            elif modification == 'order': row['per_case'].reverse()
            elif modification == 'bound': row['upper_bound'] = .1
            elif modification == 'threshold': row['threshold'] = .9
            else: threshold = row['upper_bound']; row['threshold'] = threshold
            with self.assertRaises(ValueError, msg=modification):
                campaign.verify_pruned_result(row, self.case_ids, threshold)

    def test_phase_counters_prunes_never_ranked_and_resume(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); args = self.args(folder)
            first = self.run_fake(args)
            self.assertEqual(first['status'], 'planned_phases_exhausted')
            self.assertEqual(first['unique_scores'], 1)
            self.assertEqual(first['native_proposals_evaluated'], 4)
            self.assertEqual(first['phases'][0]['scored'], 1)
            self.assertEqual(first['phases'][0]['evaluated'], 4)
            self.assertEqual(first['pruned_without_full_fitness'], 3)
            self.assertEqual(len(FakeBoundedScorer.calls), 4)
            self.assertEqual(len(FakeBoundedScorer.prepare_calls), 24)
            scores = campaign.search.read_lines(folder/'scores.jsonl')
            prunes = campaign.search.read_lines(folder/'pruned_upper_bounds.jsonl')
            self.assertEqual(len(scores), 1); self.assertEqual(len(prunes), 3)
            self.assertTrue(all(row['fitness'] is None and not row['counted_as_fitness'] for row in prunes))
            pruned_ids = {row['template_sha256'] for row in prunes}
            winner = json.loads((folder/'winner.json').read_text())
            self.assertNotIn(winner['template_sha256'], pruned_ids)
            self.assertEqual(winner['fitness'], scores[0]['fitness'])
            retained = json.loads((folder/'asr_candidates.json').read_text())['templates']
            self.assertTrue(all(row['template_sha256'] not in pruned_ids for row in retained))
            bank = next((folder/'seed_banks').glob('*.json')); before = bank.read_bytes()
            args.resume, args.cycles = True, 2
            second = self.run_fake(args)
            self.assertEqual(second['unique_scores'], 1)
            self.assertEqual(second['native_proposals_evaluated'], 8)
            self.assertEqual(second['pruned_without_full_fitness'], 7)
            self.assertEqual([row['scored'] for row in second['phases']], [1, 0])
            self.assertEqual([row['evaluated'] for row in second['phases']], [4, 4])
            self.assertEqual(len({row[0] for row in FakeBoundedScorer.calls}), 8)
            self.assertEqual(before, bank.read_bytes())

    def test_hard_crash_after_prune_append_reconciles_pending_exactly_once(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); args = self.args(folder)
            original = campaign.search.append
            crashed = False

            def append_then_crash(path, value):
                nonlocal crashed
                original(path, value)
                if Path(path).name == 'pruned_upper_bounds.jsonl' and not crashed:
                    crashed = True
                    raise HardCrash('Synthetic crash after durable prune log before checkpoint')

            with patch.object(campaign.search, 'append', side_effect=append_then_crash):
                with self.assertRaises(HardCrash): self.run_fake(args)
            state = json.loads((folder/'checkpoint.json').read_text())
            self.assertEqual(state['phases'][0]['evaluated'], 1)
            self.assertEqual(state['phases'][0]['scored'], 1)
            crashed_template = state['pending'][0]['template']
            prunes_before = campaign.search.read_lines(folder/'pruned_upper_bounds.jsonl')
            self.assertEqual(len(prunes_before), 1)
            self.assertEqual(prunes_before[0]['template'], crashed_template)
            self.assertEqual(len(FakeBoundedScorer.calls), 2)
            args.resume = True
            final = self.run_fake(args)
            self.assertEqual(final['native_proposals_evaluated'], 4)
            self.assertEqual(final['phases'][0]['scored'], 1)
            self.assertEqual(final['pruned_without_full_fitness'], 3)
            self.assertEqual(len(FakeBoundedScorer.calls), 4)
            self.assertEqual(sum(text == crashed_template for text, _, _ in FakeBoundedScorer.calls), 1)

    def test_migrated_cache_gate_and_native_source_role(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            campaign.search.write(folder/'native_cache_import.json',dict(source='synthetic-test-only',complete_six_task_results=1))
            with self.assertRaisesRegex(ValueError,'compatibility audit'):
                campaign.verify_native_cache_compatibility(folder,DATASET)
            audit = dict(passed=True,cached_exact_input_offsets_indices_masks_spans=True,
                         cached_complete_candidates=1,cached_contexts=6,source_output='synthetic-test-only',
                         locator_sha256=campaign.search.digest((HERE/'deterministic_spans.py').read_bytes()),
                         dataset_sha256=campaign.search.digest(DATASET.read_bytes()))
            campaign.search.write(folder/'native_cache_compatibility.json',audit)
            campaign.verify_native_cache_compatibility(folder,DATASET)
            audit['locator_sha256']='wrong'
            campaign.search.write(folder/'native_cache_compatibility.json',audit)
            with self.assertRaisesRegex(ValueError,'audit gate'):
                campaign.verify_native_cache_compatibility(folder,DATASET)
            campaign.search.write(folder/'score_protocol.json',dict(target_role='cot',target_role_index=3))
            with self.assertRaisesRegex(ValueError,'user column'):
                campaign.import_native_bootstrap(folder,{},None,None,self.case_ids,16384)

    def test_invalid_upstream_children_rejected_before_advanced_mutation(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(campaign.DEFAULT_MODEL,local_files_only=True)
        library=campaign.NearDelimiterLibrary(campaign.DEFAULT_MODEL)
        args=self.args(Path('/unused-cpu-fixture-only'))
        args.advanced_mutation_rate=1.; args.near_delimiter_rate=0.
        parents=['User: {goal}','Human: {goal}']
        cache={}
        for text in parents:
            row=complete_result(self.contexts,text)
            row.update(template_sha256=campaign.search.digest(text),wrapper_tokens=4)
            cache[row['template_sha256']]=row
        marker=campaign.search.GOAL_MARKER
        children=[campaign.search.encode_template(text)for text in parents]+[
            'BROKEN child without protected slot', marker+' '+marker,
            '<|start|>user '+marker, campaign.search.encode_template('Fresh user: {goal}')]
        seen=[]
        validator=lambda text:isinstance(text,str)and text.count('{goal}')==1 and '<|start|>'not in text
        class Advanced:
            @staticmethod
            def mutate(parent,*unused,**kwargs):
                if not validator(parent):raise AssertionError('Invalid upstream child reached advanced.mutate')
                seen.append(parent)
                return f'New prefix {len(seen)}: '+parent,dict(operator='synthetic-CPU-test-only')
        phase=dict(max_wrapper_tokens=128,evaluated=0,batch_index=0)
        proposals,_,_=campaign.propose_batch(args,phase,cache,[],[],0,Advanced,tokenizer,library,
            types.SimpleNamespace(key=''),{'autodan_sample_control':lambda **kwargs:children},
            {},random.Random(7),validator)
        self.assertEqual(len(proposals),args.batch_size)
        self.assertTrue(seen)
        self.assertTrue(all(validator(parent)for parent in seen))
        self.assertTrue(all(validator(row['template'])for row in proposals))
        self.assertNotIn(None,seen)

    def test_real_40_full_budget_scaling_parents_mutate_without_marker_budget_crash(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(campaign.DEFAULT_MODEL,local_files_only=True)
        library=campaign.NearDelimiterLibrary(campaign.DEFAULT_MODEL)
        advanced=campaign.load_advanced()
        source=HERE/'outputs/user90-20261104/length_scaling_seed_templates.json'
        records=json.loads(source.read_text())['templates']
        self.assertEqual(len(records),40)
        original_budget_failures=0
        for index,row in enumerate(records):
            parent=row['template'];maximum=row['metadata']['target_budget']
            with self.subTest(index=index,maximum=maximum,variant=row['metadata']['variant']):
                self.assertTrue(campaign.valid_template(parent,tokenizer,1,maximum,library))
                whole=advanced.token_count(parent,tokenizer)
                original_budget_failures+=not advanced._valid(parent,tokenizer,maximum,library)
                child,event=campaign.prefix_mutate(advanced,parent,random.Random(20261104+index),
                                                   tokenizer,maximum,library)
                self.assertGreaterEqual(event['upstream_maximum_tokens'],whole)
                self.assertEqual(event['wrapper_maximum_tokens'],maximum)
                self.assertTrue(event['driver_validation_passed'])
                self.assertTrue(campaign.valid_template(child,tokenizer,1,maximum,library))
                self.assertEqual(child.count('{goal}'),1)
                self.assertLessEqual(campaign.tokens(tokenizer,child),maximum)
                self.assertTrue(all(token<199998 for token in tokenizer.encode(child,add_special_tokens=False)))
        self.assertGreater(original_budget_failures,0,'Regression fixtures must exercise the original mismatch')

    def test_mutator_bridge_rejects_overbudget_special_and_broken_children(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(campaign.DEFAULT_MODEL,local_files_only=True)
        library=campaign.NearDelimiterLibrary(campaign.DEFAULT_MODEL)
        parent='User: {goal}';maximum=16
        responses=[('word '*100+'{goal}'),'<|start|>user {goal}','{goal} {goal}',None,'New user: {goal}']
        class Advanced:
            @staticmethod
            def mutate(template,*unused,**kwargs):
                return responses.pop(0),dict(operator='synthetic-admission-bridge-test')
        child,event=campaign.prefix_mutate(Advanced,parent,random.Random(7),tokenizer,maximum,library)
        self.assertEqual(child,'New user: {goal}')
        self.assertEqual(len(event['upstream_rejections']),4)
        self.assertTrue(campaign.valid_template(child,tokenizer,1,maximum,library))

    def test_short_parent_advanced_minimum_uses_validated_safe_fallback(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(campaign.DEFAULT_MODEL,local_files_only=True)
        library=campaign.NearDelimiterLibrary(campaign.DEFAULT_MODEL)
        parent='U{goal}'
        self.assertLess(campaign.tokens(tokenizer,parent),8)
        class RejectShort:
            @staticmethod
            def mutate(*unused,**kwargs):
                raise ValueError('Invalid parent: synthetic upstream minimum8')
        for maximum in (1,32):
            with self.subTest(maximum=maximum):
                # For a one-token budget, preserving the unchanged valid parent
                # is the only safe fallback if every introduction would grow it.
                child,event=campaign.prefix_mutate(RejectShort,parent,random.Random(7),tokenizer,maximum,library)
                self.assertTrue(campaign.valid_template(child,tokenizer,1,maximum,library))
                self.assertTrue(event['driver_validation_passed'])
                self.assertTrue(event['upstream_rejections'])

    def test_actual_proposal_files_priority_at_8192_and_16384(self):
        from transformers import AutoTokenizer
        paths = [HERE/'outputs/user90-20261104/length_scaling_seed_templates.json',
                 HERE/'outputs/user90-precise-20261104/factorial_tail_seed_templates.json',
                 HERE/'outputs/user90-precise-20261104/surrogate_shortlist_templates.json']
        records=campaign.load_initial_records(paths)
        initial=[dict(template=row['template'],family='external:'+row['source_family'],metadata=row)
                 for row in records]
        tokenizer=AutoTokenizer.from_pretrained(campaign.DEFAULT_MODEL,local_files_only=True)
        library=campaign.NearDelimiterLibrary(campaign.DEFAULT_MODEL)
        advanced=campaign.load_advanced()
        for maximum in (8192,16384):
            rows=campaign.phase_seed_records(advanced,tokenizer,library,maximum,20261104,initial)
            first=rows[:8]
            self.assertTrue(all(row['family']=='external:priority_length_scaling' for row in first))
            self.assertEqual(len({row['template']for row in first}),8)
            self.assertTrue(all(.9*maximum<=campaign.tokens(tokenizer,row['template'])<=maximum for row in first))
            priority=[row for row in rows[:80]if row['family'].endswith('surrogate_diversified_prediction_only')or'factorial_'in row['family']]
            self.assertTrue(any('factorial_'in row['family']for row in priority))
            self.assertTrue(any(row['family'].endswith('surrogate_diversified_prediction_only')for row in priority))


if __name__=='__main__':unittest.main(verbosity=2)
