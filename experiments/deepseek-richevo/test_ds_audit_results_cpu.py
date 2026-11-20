"""CPU-only tests of the explicit user-stop audit and unchanged ASR validation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ds_audit_results import Audit, read


class CurrentBestAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.groups = {f'{suite}/{i}': suite for suite in ('banking', 'slack', 'travel', 'browser', 'coding', 'tool')
                       for i in range(2)}
        vector = dict(system=0.0, user=0.4, cot=0.0, assistant=0.0, tool=0.6)
        self.row = dict(status='complete', template='{goal}', evaluation_index=7, fitness=0.4,
                        suite_means={suite: 0.4 for suite in self.groups.values()}, goal_role_means=vector,
                        per_case=[dict(case_id=case, goal_user_mean=0.4, probabilities=dict(goal=vector))
                                  for case in self.groups])
        raw = (json.dumps(self.row) + '\n').encode()
        self.ledger = self.root / 'scores.jsonl'
        # A complete score after receipt must not change the immutable selection.
        self.ledger.write_bytes(raw + (json.dumps(dict(self.row, evaluation_index=8, fitness=0.9)) + '\n').encode())
        self.snapshot = self.root / 'user_stop_snapshot.json'
        self.snapshot.write_text(json.dumps(dict(schema='deepseek-user-stop-v1', best=self.row,
                                                completed_at_snapshot=1, ledger_snapshot_sha256=hashlib.sha256(raw).hexdigest())))
        self.conditions = self.root / 'conditions.json'
        self.conditions.write_text(json.dumps(dict(opt_top1=dict(template='{goal}',
            template_sha256=hashlib.sha256(b'{goal}').hexdigest(), evaluation_index=7, p_user_optimization=0.4))))

    def audit(self, selection=None):
        audit = Audit()
        audit.panel_groups = self.groups
        audit.current_best(read(self.conditions) if selection is None else selection,
                           self.snapshot, self.ledger, set(self.groups))
        return audit

    def test_exact_snapshot_choice_accepts_newer_ledger_tail(self):
        self.assertTrue(all(c['status'] == 'pass' for c in self.audit().checks))

    def test_other_condition_or_score_is_rejected(self):
        selection = read(self.conditions)
        selection['opt_top2'] = copy.deepcopy(selection['opt_top1'])
        self.assertIn('selection.current_best_only', [c['name'] for c in self.audit(selection).checks if c['status'] == 'fail'])
        selection.pop('opt_top2')
        selection['opt_top1']['p_user_optimization'] = 0.40000000001
        self.assertIn('selection.current_best_score', [c['name'] for c in self.audit(selection).checks if c['status'] == 'fail'])

    def test_ledger_prefix_tamper_or_missing_snapshot_is_rejected(self):
        raw = self.ledger.read_bytes().replace(b'"evaluation_index": 7', b'"evaluation_index": 6', 1)
        self.ledger.write_bytes(raw)
        failures = [c['name'] for c in self.audit().checks if c['status'] == 'fail']
        self.assertIn('user_stop_snapshot.ledger_sha256', failures)
        self.assertIn('user_stop_snapshot.best_ledger_identity', failures)
        self.snapshot.unlink()
        self.assertIn('user_stop_snapshot.exists', [c['name'] for c in self.audit().checks if c['status'] == 'fail'])

    def test_skip_reason_is_explicit_and_not_pending(self):
        audit = Audit()
        audit.skip_asr_only('final_condition_scores')
        self.assertEqual(audit.checks[0]['status'], 'skipped-by-user-ASR-only')
        self.assertIn('User stopped', audit.checks[0]['reason'])

    def test_reused_score_is_still_bound_to_every_snapshot_case(self):
        best = read(self.snapshot)['best']
        score = dict(p_user=best['fitness'], per_case={r['case_id']: r['goal_user_mean'] for r in best['per_case']},
                     suite_means=best['suite_means'], goal_role_means=best['goal_role_means'],
                     template_sha256_observed=best['template_sha256_observed'])
        audit = Audit()
        audit.reused_best_score(score, best)
        self.assertTrue(all(c['status'] == 'pass' for c in audit.checks))
        score['per_case'][next(iter(self.groups))] += 0.01
        audit = Audit()
        audit.reused_best_score(score, best)
        self.assertIn('final_condition_scores.opt_top1.snapshot_per_case',
                      [c['name'] for c in audit.checks if c['status'] == 'fail'])

    def test_ipi_asr_request_and_record_hashes_cannot_drift(self):
        directory = self.root / 'bound'
        ipi = directory / 'ipi'
        (ipi / 'cases/opt_top1').mkdir(parents=True)
        (ipi / 'conditions.json').write_bytes(self.conditions.read_bytes())
        request = ipi / 'request.json'
        request.write_bytes(b'{}\n')
        request_hash = hashlib.sha256(request.read_bytes()).hexdigest()
        protocol = ipi / 'protocol.json'
        protocol.write_text(json.dumps(dict(rows=[dict(condition='opt_top1', case_id='case-0',
            request='request.json', request_sha256=request_hash)])))
        settings = dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(), target='frozen')
        identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        (ipi / 'asr_identity.json').write_text(json.dumps(dict(settings, identity_sha256=identity, judge='fixed')))
        (ipi / 'cases/opt_top1/case-0.json').write_text(json.dumps(dict(identity_sha256=identity, request_sha256=request_hash)))
        audit = Audit()
        audit.asr_bindings('optimized.ipi', directory, 'ipi', read(self.conditions))
        self.assertTrue(all(c['status'] == 'pass' for c in audit.checks))
        request.write_bytes(b'{"changed":true}\n')
        audit = Audit()
        audit.asr_bindings('optimized.ipi', directory, 'ipi', read(self.conditions))
        self.assertIn('optimized.ipi.opt_top1.case-0.request_hash',
                      [c['name'] for c in audit.checks if c['status'] == 'fail'])

    def test_ipi_complete_38_counts_summary_and_tamper(self):
        directory = self.root / 'bench'
        cases = directory / 'ipi/cases/opt_top1'
        cases.mkdir(parents=True)
        plan = [dict(condition='opt_top1', case_id=f'case-{i}', primary=i < 36) for i in range(38)]
        for i, entry in enumerate(plan):
            (cases / (entry['case_id'] + '.json')).write_text(json.dumps(dict(entry, status='error' if i == 35 else 'complete', success=i < 7)))
        (directory / 'ipi/protocol.json').write_text(json.dumps(dict(rows=plan)))
        (directory / 'ipi/conditions.json').write_text(json.dumps(dict(opt_top1={})))
        summary = dict(conditions=dict(opt_top1=dict(scheduled=36, complete=35, errors=1, successes=7,
                    asr=7 / 35, asr_over_scheduled=7 / 36, supplementary=dict(complete=2, successes=0))))
        summary_path = directory / 'ipi/summary.json'
        summary_path.write_text(json.dumps(summary))
        audit = Audit()
        audit.benchmark('optimized.ipi', directory, 'ipi', {'opt_top1'}, {'case-0'})
        self.assertTrue(all(c['status'] == 'pass' for c in audit.checks))
        self.assertEqual(audit.metrics['optimized.ipi.opt_top1']['full']['errors'], 1)
        self.assertEqual(audit.metrics['optimized.ipi.opt_top1']['heldout']['successes'], 6)
        summary['conditions']['opt_top1']['successes'] = 8
        summary_path.write_text(json.dumps(summary))
        audit = Audit()
        audit.benchmark('optimized.ipi', directory, 'ipi', {'opt_top1'}, set())
        self.assertIn('optimized.ipi.opt_top1.summary.successes', [c['name'] for c in audit.checks if c['status'] == 'fail'])

    def test_agentdojo_389_error_counts_and_missing_remains_pending(self):
        directory = self.root / 'bench'
        cases = directory / 'agentdojo/cases/opt_top1/banking/user'
        cases.mkdir(parents=True)
        expected = {f'banking/user/case-{i}' for i in range(389)}
        for i in range(389):
            (cases / f'case-{i}.json').write_text(json.dumps(dict(condition='opt_top1',
                base_case_id=f'banking/user/case-{i}', status='error' if i == 388 else 'complete',
                security=i < 10, utility=i < 300)))
        (directory / 'agentdojo/conditions.json').write_text(json.dumps(dict(opt_top1={})))
        (directory / 'agentdojo/summary.json').write_text(json.dumps(dict(conditions=dict(opt_top1=dict(
            expected=389, scored=388, errors=1, successes=10, utility_successes=300,
            asr_scored=10 / 388, utility_scored=300 / 388, asr=None)))))
        audit = Audit()
        audit.agentdojo_cases = expected
        audit.benchmark('optimized.agentdojo', directory, 'agentdojo', {'opt_top1'}, set())
        self.assertTrue(all(c['status'] == 'pass' for c in audit.checks))
        self.assertEqual(audit.metrics['optimized.agentdojo.opt_top1']['full']['asr_lower'], 10 / 389)
        (cases / 'case-388.json').unlink()
        audit = Audit()
        audit.agentdojo_cases = expected
        audit.benchmark('optimized.agentdojo', directory, 'agentdojo', {'opt_top1'}, set())
        self.assertIn('optimized.agentdojo.opt_top1.records', [c['name'] for c in audit.checks if c['status'] == 'pending'])


if __name__ == '__main__':
    unittest.main()
