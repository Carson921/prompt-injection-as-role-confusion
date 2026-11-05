"""CPU-only synthetic unit tests. Temporary fixtures are not native fitness."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from bounded_scorer import BoundedScorer, BOUND_TOLERANCE, native_upper_bound
from scorer import Scorer, ROLES, sha, write


class SyntheticScorer(BoundedScorer):
    """Never loads a tokenizer, backbone, actual goal, probe, GPU, or API."""
    def __init__(self, output, values):
        self.output = Path(output)
        self.values = values
        self.calls = []
        self.closed = False
        self.target_role, self.target_role_index = 'user', 2
        self.protocol_sha256 = 'synthetic-unit-test-protocol-not-a-native-score'
        self.contexts = [dict(case_id=f'synthetic_case_{i}', suite=suite)
                         for i, suite in enumerate(('banking', 'banking', 'slack', 'slack', 'travel', 'travel'))]

    def score_case(self, context, template, directory):
        index = int(context['case_id'].split('_')[-1])
        self.calls.append(index)
        if isinstance(self.values[index], Exception):
            raise self.values[index]
        p = self.values[index]
        probabilities = {role: (p if role == 'user' else (1-p)/5) for role in ROLES}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'tokens.npz').write_bytes(b'SYNTHETIC TEST ONLY ' + str(index).encode())
        write(directory / 'request.json', dict(synthetic_unit_test_only=True, case_id=context['case_id']))
        result = dict(status='complete', case_id=context['case_id'], suite=context['suite'], template=template,
                      goal_user_mean=p, goal_target_mean=p, target_role='user', target_role_index=2,
                      probabilities={'goal': probabilities}, protocol_sha256=self.protocol_sha256,
                      token_artifact=str(directory / 'tokens.npz'),
                      artifact_sha256={name: sha(directory / name) for name in ('tokens.npz', 'request.json')})
        write(directory / 'score.json', result)
        return result


class BoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.template = 'User: {goal}'

    def tearDown(self):
        self.temp.cleanup()

    def test_ordinary_score_is_inherited_unchanged(self):
        self.assertIs(BoundedScorer.score, Scorer.score)

    def test_prune_preserves_partial_artifacts_without_fitness(self):
        scorer = SyntheticScorer(self.output, [0.] * 6)
        result = scorer.score_if_can_improve(self.template, .9)
        self.assertEqual(result['status'], 'pruned_upper_bound')
        self.assertEqual(scorer.calls, [0])
        self.assertIsNone(result['fitness'])
        self.assertNotIn('goal_target_mean', result)
        self.assertNotIn('goal_role_means', result)
        self.assertFalse((self.output / 'scores').exists())
        self.assertTrue(Path(result['artifact']).is_file())
        self.assertTrue(Path(result['per_case'][0]['token_artifact']).is_file())

    def test_equality_and_tolerance_never_prune(self):
        scorer = SyntheticScorer(self.output, [0., 1., 1., 1., 1., 1.])
        result = scorer.score_if_can_improve(self.template, 5/6 + BOUND_TOLERANCE/2)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(scorer.calls, list(range(6)))
        self.assertEqual(result['fitness'], float(np.mean(scorer.values)))

    def test_strict_conservative_inequality_prunes(self):
        scorer = SyntheticScorer(self.output, [0., 1., 1., 1., 1., 1.])
        result = scorer.score_if_can_improve(self.template, 5/6 + 10*BOUND_TOLERANCE)
        self.assertEqual(result['status'], 'pruned_upper_bound')
        self.assertEqual(result['completed_cases'], 1)

    def test_complete_publication_reuses_each_case_once_and_exact_base_schema(self):
        values = [.2, .4, .6, .8, .9, 1.]
        scorer = SyntheticScorer(self.output, values)
        result = scorer.score_if_can_improve(self.template, 0.)
        self.assertEqual(scorer.calls, list(range(6)))
        self.assertEqual(result['fitness'], float(np.mean(values)))
        self.assertEqual(result['goal_target_mean'], result['fitness'])
        self.assertEqual(result['goal_role_means']['user'], result['fitness'])
        self.assertEqual(result['suite_means'], {'banking': .30000000000000004, 'slack': .7, 'travel': .95})
        expected={'status','complete','fitness','target_role','target_role_index','goal_target_mean',
                  'goal_role_means','template','per_case','expected_cases','completed_cases','suite_means',
                  'protocol_sha256','cache_hit','elapsed_seconds','artifact'}
        self.assertEqual(set(result), expected)
        for row in result['per_case']:
            artifact=Path(row['token_artifact'])
            self.assertTrue(artifact.is_relative_to(self.output / 'scores'))
            self.assertTrue(artifact.is_file())
            self.assertEqual(sha(artifact), row['artifact_sha256']['tokens.npz'])
            self.assertEqual(json.loads((artifact.parent/'score.json').read_text())['token_artifact'],str(artifact))
        self.assertFalse(any((self.output/'screening').glob('*/attempt_*/native_cases')))
        again=scorer.score_if_can_improve(self.template,1.)
        self.assertTrue(again['cache_hit'])
        self.assertEqual(scorer.calls,list(range(6)))

    def test_sixth_case_always_publishes_even_when_below_threshold(self):
        scorer=SyntheticScorer(self.output,[1.,1.,1.,1.,1.,0.])
        result=scorer.score_if_can_improve(self.template,1.)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['fitness'],5/6)

    def test_cache_corruption_recomputed_without_destroying_previous(self):
        scorer=SyntheticScorer(self.output,[.8]*6)
        first=scorer.score_if_can_improve(self.template,0.)
        Path(first['per_case'][0]['token_artifact']).write_bytes(b'CORRUPTED SYNTHETIC')
        again=scorer.score_if_can_improve(self.template,0.)
        self.assertFalse(again['cache_hit'])
        self.assertEqual(scorer.calls,list(range(6))*2)
        self.assertEqual(len(list((self.output/'screening').glob('*/attempt_*/previous_invalid_main'))),1)

    def test_error_never_enters_main_scores(self):
        scorer=SyntheticScorer(self.output,[1.,RuntimeError('synthetic failure'),1.,1.,1.,1.])
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            scorer.score_if_can_improve(self.template,.9)
        self.assertFalse((self.output/'scores').exists())
        records=list((self.output/'screening').glob('*/attempt_*/screening.json'))
        self.assertEqual(len(records),1)
        result=json.loads(records[0].read_text())
        self.assertEqual(result['status'],'error');self.assertIsNone(result['fitness'])

    def test_endpoint_probabilities_and_thresholds(self):
        for i,(values,threshold,expected) in enumerate((([0.]*6,0.,'complete'),
                    ([1.]*6,1.,'complete'),([0.]*6,1.,'pruned_upper_bound'))):
            scorer=SyntheticScorer(self.output/str(i),values)
            self.assertEqual(scorer.score_if_can_improve(self.template,threshold)['status'],expected)

    def test_random_bounds_cannot_reject_a_winning_full_mean(self):
        rng=random.Random(20261104)
        for _ in range(10000):
            values=[rng.random()for _ in range(6)];threshold=rng.random()
            actual=float(np.mean(values))
            for n in range(1,6):
                upper=native_upper_bound(values[:n])
                self.assertGreaterEqual(upper+BOUND_TOLERANCE,actual)
                if upper+BOUND_TOLERANCE<threshold:self.assertLess(actual,threshold)

    def test_invalid_threshold_probability_role_and_count_rejected(self):
        for threshold in (float('nan'),float('inf'),-.01,1.01,'0.9'):
            scorer=SyntheticScorer(self.output,[.5]*6)
            with self.assertRaises(ValueError):scorer.score_if_can_improve(self.template,threshold)
        for values in ([float('nan')],[1.01],[-.01]):
            with self.assertRaises(ValueError):native_upper_bound(values)
        scorer=SyntheticScorer(self.output,[.5]*6);scorer.target_role='cot'
        with self.assertRaises(ValueError):scorer.score_if_can_improve(self.template,.9)
        scorer=SyntheticScorer(self.output,[.5]*6);scorer.contexts=scorer.contexts[:5]
        with self.assertRaises(ValueError):scorer.score_if_can_improve(self.template,.9)


if __name__=='__main__':unittest.main(verbosity=2)
