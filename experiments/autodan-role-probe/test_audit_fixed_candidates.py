"""CPU-only audit regressions using explicit non-model-generated fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import audit_fixed_candidates as audit
from prepare import sha


class FixedCandidateAuditTests(unittest.TestCase):
    def test_audit_candidate_count_limits(self):
        for count in [2,7]:
            with self.assertRaisesRegex(ValueError,'three to six'):
                audit.verify_candidates({'candidates':[{}]*count},None,None,{},Path('unused'))

    def test_five_condition_summary_population(self):
        conditions=[f'fixture_{n}' for n in range(5)]
        rows=[dict(condition=condition,status='complete',security=index%2==0,
                   utility=index%3==0,payload_exposed=True)
              for condition in conditions for index in range(389)]
        summary=dict(tasks_per_condition=389,scheduled_episodes=1945,terminal=1945,
                     conditions={condition:audit.condition_metrics(
                         [row for row in rows if row['condition']==condition],389) for condition in conditions})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            path.write_text(json.dumps(summary))
            self.assertTrue(audit.verify_summary(Path(directory),rows,conditions))
            summary['scheduled_episodes']=1167
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'scheduled/terminal'):
                audit.verify_summary(Path(directory),rows,conditions)

    def test_unknown_verdicts_are_bounds_not_imputed_failures(self):
        rows=[dict(status='complete',security=True,utility=False,payload_exposed=True)
              for _ in range(12)]
        rows += [dict(status='complete',security=False,utility=True,payload_exposed=True)
                 for _ in range(375)]
        rows += [dict(status='error',security=None,utility=None,payload_exposed=False)]
        result=audit.condition_metrics(rows,389)
        self.assertEqual((result['scheduled'],result['scored'],result['errors'],result['missing']),
                         (389,387,1,1))
        self.assertEqual(result['unknown'],2)
        self.assertIsNone(result['asr'])
        self.assertAlmostEqual(result['asr_lower_bound'],12/389)
        self.assertAlmostEqual(result['asr_upper_bound'],14/389)
        self.assertAlmostEqual(result['scored_asr'],12/387)
        self.assertEqual(len(result['scored_asr_wilson_95']),2)

    def test_wilson_zero_count_and_extreme_samples(self):
        self.assertIsNone(audit.wilson_interval(0,0))
        for successes in [0,10]:
            lower,upper=audit.wilson_interval(successes,10)
            self.assertLessEqual(lower,successes/10)
            self.assertGreaterEqual(upper,successes/10)
            self.assertTrue(0<=lower<=upper<=1)

    def test_failed_parser_replay_preserves_null_and_rejects_different_error(self):
        def fail(*_args,**_kwargs):
            raise ValueError('explicit CPU parser fixture')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            call='fixture/user/injection/call_0000'
            raw_path=root/'calls'/call/'raw_response.json'
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text(json.dumps(dict(generated_ids=[1,2,200002],finish_reason='stop',stop_reason=200002)))
            record=dict(case_id='fixture/user/injection',status='error',security=None,utility=None,
                        error='ValueError: explicit CPU parser fixture',requests=[dict(
                            call_id=call,raw_response_sha256=sha(raw_path))])
            original=json.dumps(record,sort_keys=True)
            rows=audit.replay_unparsed_error_calls(root,record,{'safe_parse_completion':fail},None,{'max_tokens':3})
            self.assertEqual(rows[0]['status'],'parser_error_reproduced')
            self.assertTrue(rows[0]['verdict_preserved_null'])
            self.assertEqual(original,json.dumps(record,sort_keys=True))
            record['error']='ValueError: different fixture'
            with self.assertRaisesRegex(ValueError,'different exception'):
                audit.replay_unparsed_error_calls(root,record,{'safe_parse_completion':fail},None,{'max_tokens':3})

    def test_unreferenced_artifact_is_rejected_by_full_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'calls/fixture/orphan/request.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unexpected/unreferenced'):
                audit.verify_episode_artifact_inventory(root,[],{},partial=False)
            partial=audit.verify_episode_artifact_inventory(root,[],{},partial=True)
            self.assertEqual(partial['observed']['request'],1)
            self.assertFalse(partial['full_inventory_exact'])


if __name__=='__main__':
    unittest.main()
