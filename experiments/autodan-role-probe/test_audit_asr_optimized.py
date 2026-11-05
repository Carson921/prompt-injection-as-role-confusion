"""CPU-only checks for optimized episode provenance and null-verdict bounds."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_asr_optimized as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(audit.serial(value) + '\n')


class ImportedEpisodeIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old, self.new = self.base / 'source', self.base / 'destination'
        self.case_id = 'optimized/demo/user_task_0/injection_task_0'
        case = dict(case_id=self.case_id, condition='optimized', pair_id='demo/user_task_0/injection_task_0',
                    suite='demo', user_task_id='user_task_0', injection_task_id='injection_task_0',
                    actual_goal='benign immutable fixture', payload_template='{goal}',
                    payload_text='benign immutable fixture')
        fields = ('scope', 'dataset_sha256', 'winner_sha256', 'source_identity_sha256',
                  'engine_configuration', 'max_tokens', 'reasoning_effort', 'prompt_date',
                  'seed', 'temperature', 'stop_ids', 'benchmark', 'normal_system', 'initial_errors',
                  'model_staging_proof_sha256', 'model_configuration_sha256', 'package_versions',
                  'numerical_equivalence_to_historical_backend_claimed', 'serialized_payload_validation',
                  'optimization_objective', 'engine_environment')
        old_protocol = {name: 'fixed-fixture-' + name for name in fields}
        old_protocol.update(templates={'plain': '{goal}', 'remember': 'Fixture {goal}', 'optimized': '{goal}'},
                            cases=[case], evaluator_source_sha256=audit.sha(audit.ROOT / 'evaluate_precise.py'))
        write(self.old / 'protocol.json', old_protocol)
        old_binding = hashlib.sha256(audit.serial(old_protocol).encode()).hexdigest()
        call_id = self.case_id + '/call_0000'
        artifacts = {}
        for name in ('request.json', 'raw_response.json', 'parsed_response.json'):
            relative = Path('calls') / call_id / name
            write(self.old / relative, dict(benign_fixture=name))
            artifacts[relative.as_posix()] = audit.sha(self.old / relative)
        trajectory = Path('trajectories') / self.case_id / 'pipeline_00.json'
        injection = Path('injections') / (self.case_id + '.json')
        for relative in (trajectory, injection):
            write(self.old / relative, dict(benign_fixture=relative.name))
            artifacts[relative.as_posix()] = audit.sha(self.old / relative)
        ref = dict(call_id=call_id, status='complete',
                   request_sha256=artifacts[(Path('calls') / call_id / 'request.json').as_posix()],
                   raw_response_sha256=artifacts[(Path('calls') / call_id / 'raw_response.json').as_posix()],
                   parsed_response_sha256=artifacts[(Path('calls') / call_id / 'parsed_response.json').as_posix()])
        source = dict(case=case, case_id=self.case_id, condition='optimized', status='complete',
                      security=True, utility=False, protocol_sha256=old_binding, requests=[ref],
                      trajectories=[dict(path=str(self.old / trajectory), sha256=artifacts[trajectory.as_posix()])])
        self.source_case = self.old / 'cases' / (self.case_id + '.json')
        write(self.source_case, source)
        entry = dict(case_id=self.case_id, source_case_path=str(self.source_case),
                     source_case_sha256=audit.sha(self.source_case), artifacts=[])
        for relative, digest in artifacts.items():
            path = self.new / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.old / relative).read_bytes())
            entry['artifacts'].append(dict(kind='fixture', source_path=str(self.old / relative),
                                           destination_relative_path=relative, sha256=digest))
        self.protocol = copy.deepcopy(old_protocol)
        self.protocol['templates'] = {'optimized': '{goal}'}
        self.protocol['completed_episode_import'] = dict(schema='optimized-completed-native-episode-import-v1', source_output=str(self.old),
            imported_conditions=['optimized'], baseline_imports=0,
            episode_record_changes=['protocol_sha256','import_provenance'], native_artifact_bytes_unchanged=True,
            source_protocol_path=str(self.old / 'protocol.json'),
            source_protocol_physical_sha256=audit.sha(self.old / 'protocol.json'),
            source_protocol_canonical_sha256=old_binding,
            source_evaluator_source_sha256=old_protocol['evaluator_source_sha256'],
            compatibility_fields_checked=list(fields), cases=[entry])
        self.binding = hashlib.sha256(audit.serial(self.protocol).encode()).hexdigest()
        self.destination_case = self.new / 'cases' / (self.case_id + '.json')
        identity = self.protocol['completed_episode_import']
        derived = copy.deepcopy(source)
        derived['protocol_sha256'] = self.binding
        derived['import_provenance'] = {key: identity[key] for key in (
            'source_protocol_physical_sha256', 'source_protocol_canonical_sha256', 'source_output')}
        derived['import_provenance'].update(source_case_path=str(self.source_case),
                                           source_case_sha256=entry['source_case_sha256'])
        write(self.destination_case, derived)
        write(self.new / 'completed_episode_import_manifest.json', dict(
            schema='optimized-completed-native-episode-import-manifest-v1',
            destination_protocol_canonical_sha256=self.binding,
            source_protocol_physical_sha256=identity['source_protocol_physical_sha256'],
            source_protocol_canonical_sha256=old_binding, completed_optimized_imports=1, baseline_imports=0,
            native_artifacts_copied_and_sha_verified=5, cases=[dict(case_id=self.case_id,
                source_case_path=str(self.source_case), source_case_sha256=entry['source_case_sha256'],
                destination_case_path=str(self.destination_case), destination_case_sha256=audit.sha(self.destination_case),
                native_artifact_count=5)]))

    def test_exact_complete_import_passes(self):
        result = audit.verify_completed_episode_import(self.new, self.protocol, self.binding)
        self.assertEqual(result['imported_completed_optimized_episodes'], 1)
        self.assertEqual(result['imported_artifacts_byte_identical'], 5)
        self.assertEqual(result['baseline_episodes_imported'], 0)

    def test_artifact_tamper_is_rejected(self):
        artifact = self.protocol['completed_episode_import']['cases'][0]['artifacts'][0]
        (self.new / artifact['destination_relative_path']).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'artifact bytes differ'):
            audit.verify_completed_episode_import(self.new, self.protocol, self.binding)

    def test_changed_verdict_is_rejected(self):
        record = json.loads(self.destination_case.read_text())
        record['security'] = False
        write(self.destination_case, record)
        with self.assertRaisesRegex(ValueError, 'changed beyond'):
            audit.verify_completed_episode_import(self.new, self.protocol, self.binding)

    def test_engine_metadata_mismatch_is_rejected(self):
        self.protocol['max_tokens'] = 'different'
        with self.assertRaisesRegex(ValueError, 'configuration differs: max_tokens'):
            audit.verify_completed_episode_import(self.new, self.protocol, self.binding)

    def test_undeclared_import_is_rejected(self):
        self.protocol['completed_episode_import']['cases'] = []
        with self.assertRaisesRegex(ValueError, 'missing or undeclared'):
            audit.verify_completed_episode_import(self.new, self.protocol, self.binding)

    def test_manifest_tamper_is_rejected(self):
        path = self.new / 'completed_episode_import_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['native_artifacts_copied_and_sha_verified'] = 99
        write(path, manifest)
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            audit.verify_completed_episode_import(self.new, self.protocol, self.binding)

    def test_source_error_cannot_be_imported(self):
        source = json.loads(self.source_case.read_text())
        source.update(status='error', security=None, utility=None)
        write(self.source_case, source)
        self.protocol['completed_episode_import']['cases'][0]['source_case_sha256'] = audit.sha(self.source_case)
        with self.assertRaisesRegex(ValueError, 'Only matching complete'):
            audit.verify_completed_episode_import(self.new, self.protocol, self.binding)

    def test_null_bounds_do_not_turn_errors_into_failures(self):
        rows = [dict(status='complete', security=True, utility=False, payload_exposed=True),
                dict(status='complete', security=False, utility=True, payload_exposed=False),
                dict(status='error', security=None, utility=None, payload_exposed=False)]
        result = audit.condition_metrics(rows, 4)
        self.assertIsNone(result['asr'])
        self.assertEqual(result['null_verdicts'], 2)
        self.assertEqual(result['asr_lower_bound'], .25)
        self.assertEqual(result['asr_upper_bound'], .75)
        self.assertEqual(result['scored_asr'], .5)


if __name__ == '__main__':
    unittest.main()
