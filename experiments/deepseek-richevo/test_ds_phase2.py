"""CPU-only regression checks for process safety and continuation failure handling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ds_phase2 as phase


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.args = argparse.Namespace(evolution=root / 'e', baseline=root / 'b', optimized=root / 'o',
                                       hours=6, check_only=False)
        self.pipeline = phase.Pipeline(self.args)
        self.addCleanup(self.pipeline.cleanup)

    def test_nonzero_child_is_recorded_and_not_validated(self):
        validated = []
        with self.assertRaisesRegex(ValueError, 'exited 7'):
            self.pipeline.stage('bad', [sys.executable, '-c', 'raise SystemExit(7)'],
                                self.pipeline.o / 'bad.log', lambda: validated.append(True))
        self.assertEqual(validated, [])
        state = phase.read_json(self.pipeline.state_path)['stages']['bad']
        self.assertEqual((state['status'], state['exit_code']), ('failed', 7))
        self.assertIn('actual exit 7', (self.pipeline.o / 'bad.log').read_text())

    def test_zero_exit_with_incomplete_output_does_not_complete(self):
        def incomplete():
            raise ValueError('missing episodes')
        with self.assertRaisesRegex(ValueError, 'missing episodes'):
            self.pipeline.stage('incomplete', [sys.executable, '-c', 'pass'],
                                self.pipeline.o / 'incomplete.log', incomplete)
        self.assertEqual(self.pipeline.state['stages']['incomplete']['status'], 'failed')

    def test_completed_stage_is_revalidated_without_rerunning(self):
        log = self.pipeline.o / 'once.log'
        self.pipeline.stage('once', [sys.executable, '-c', 'print("ran")'], log, lambda: None)
        original = log.read_text()
        self.pipeline.stage('once', ['does-not-exist'], log, lambda: None)
        self.assertEqual(log.read_text(), original)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.pipeline.completed('once', lambda: phase.require(False, 'stale'))

    def test_parallel_failure_is_not_hidden_behind_other_child(self):
        self.pipeline.launch('slow', [sys.executable, '-c', 'import time; time.sleep(30)'], self.pipeline.o / 'slow.log')
        self.pipeline.launch('bad', [sys.executable, '-c', 'raise SystemExit(11)'], self.pipeline.o / 'bad.log')
        with self.assertRaisesRegex(ValueError, 'exited 11'):
            self.pipeline.finish_parallel([('slow', lambda: None), ('bad', lambda: None)])
        self.assertIsNone(self.pipeline.active['slow'][0].poll())
        self.assertEqual(self.pipeline.state['stages']['bad']['exit_code'], 11)

    def test_existing_stage_pid_prevents_duplicate_execution(self):
        with patch.object(phase, 'process_table', return_value={123: {'start': 'bound'}}):
            self.pipeline.state['stages']['native'] = dict(status='running', pid=123, process_start='bound')
            with self.assertRaisesRegex(ValueError, 'still running'):
                self.pipeline.launch('native', ['must-not-run'], self.pipeline.o / 'native.log')

    def test_reuse_healthy_server_never_spawns_or_stops_it(self):
        with patch.object(phase, 'server_ready', return_value=True), patch.object(phase.subprocess, 'Popen') as spawn:
            self.pipeline.ensure_server()
        spawn.assert_not_called()

    def test_loading_existing_server_is_waited_for_without_duplicate(self):
        with patch.object(phase, 'server_ready', side_effect=[False, False, True]), \
                patch.object(phase, 'server_roots', return_value=[123]), \
                patch.object(phase, 'process_table', return_value={}), \
                patch.object(phase.time, 'sleep'), patch.object(phase.subprocess, 'Popen') as spawn:
            self.pipeline.ensure_server()
        spawn.assert_not_called()

    def test_unowned_server_is_never_signalled(self):
        with patch.object(phase, 'server_ready', return_value=True), \
                patch.object(phase, 'process_table', return_value={}), patch.object(phase.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'no matching owned local PID'):
                self.pipeline.stop_for_native()
        kill.assert_not_called()

    def test_release_only_exact_server_and_descendants(self):
        rows = {
            10: dict(ppid=1, start='10', state='S', argv=['python', '/venv/bin/vllm', 'serve', str(phase.MODEL), '--port', '8000']),
            11: dict(ppid=10, start='11', state='S', argv=['engine']),
            99: dict(ppid=1, start='99', state='S', argv=['unrelated']),
        }
        calls = []
        def snapshots():
            return {99: rows[99]} if len(calls) == 2 else rows
        with patch.object(phase, 'process_table', side_effect=snapshots), \
                patch.object(phase, 'server_ready', return_value=True), \
                patch.object(phase.os, 'kill', side_effect=lambda pid, sig: calls.append(pid)), \
                patch.object(phase.subprocess, 'check_output', return_value='NVIDIA H100, 80000\n' * 4):
            self.pipeline.stop_for_native()
        self.assertEqual(set(calls), {10, 11})
        self.pipeline.restore_service = False  # Do not restore the mocked service in cleanup.

    def test_deadline_and_handoff_survive_process_restart(self):
        self.pipeline.state.update(evolution_deadline_unix=123.5, evolution_hours=6,
                                   service_handoff={'restored': False})
        self.pipeline.save()
        restored = phase.Pipeline(self.args)
        self.assertEqual(restored.state['evolution_deadline_unix'], 123.5)
        self.assertTrue(restored.restore_service)

    def test_missing_child_is_resumed_with_history_and_original_deadline(self):
        self.pipeline.state.update(evolution_deadline_unix=123.5)
        self.pipeline.state['stages']['evolution'] = dict(status='running', pid=123, process_start='old')
        with patch.object(phase, 'process_table', return_value={}):
            self.pipeline.recover_stale_stages()
        self.assertEqual(self.pipeline.state['stages']['evolution']['status'], 'interrupted')
        self.pipeline.stage('evolution', [sys.executable, '-c', 'pass'], self.pipeline.o / 'resume.log', lambda: None)
        state = self.pipeline.state['stages']['evolution']
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['previous_attempts'][0]['status'], 'interrupted')
        self.assertNotIn('exit_code', state['previous_attempts'][0])
        self.assertEqual(self.pipeline.state['evolution_deadline_unix'], 123.5)

    def test_zombie_or_reused_pid_does_not_block_recovery(self):
        for current in ({'start': 'old', 'state': 'Z'}, {'start': 'new', 'state': 'R'}):
            self.pipeline.state['stages']['evolution'] = dict(status='running', pid=123, process_start='old')
            with patch.object(phase, 'process_table', return_value={123: current}):
                self.pipeline.recover_stale_stages()
            self.assertEqual(self.pipeline.state['stages']['evolution']['status'], 'interrupted')

    def test_orphan_child_prevents_gpu_handoff_and_service_restoration(self):
        self.pipeline.restore_service = True
        self.pipeline.state['stages']['evolution'] = dict(status='running', pid=123, process_start='live')
        with patch.object(phase, 'process_table', return_value={123: {'start': 'live', 'state': 'S'}}), \
                patch.object(self.pipeline, 'check_baseline') as baseline, \
                patch.object(self.pipeline, 'stop_for_native') as handoff, \
                patch.object(self.pipeline, 'ensure_server') as restore:
            with self.assertRaisesRegex(RuntimeError, 'still running'):
                self.pipeline.execute()
        baseline.assert_not_called()
        handoff.assert_not_called()
        restore.assert_not_called()

    def test_model_initialization_failure_restores_service_and_keeps_failure(self):
        failure = RuntimeError('native model initialization failed')
        self.pipeline.restore_service = True
        def restored():
            self.pipeline.restore_service = False
        with patch.object(self.pipeline, 'run', side_effect=failure), \
                patch.object(self.pipeline, 'ensure_server', side_effect=restored) as restore:
            with self.assertRaises(RuntimeError) as observed:
                self.pipeline.execute()
        self.assertIs(observed.exception, failure)
        restore.assert_called_once()
        state = phase.read_json(self.pipeline.state_path)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('native model initialization failed', state['error'])

    def test_restore_failure_cannot_mask_model_initialization_failure(self):
        failure = RuntimeError('original native failure')
        self.pipeline.restore_service = True
        def failed_restore():
            self.pipeline.restore_service = False
            raise OSError('vLLM startup also failed')
        with patch.object(self.pipeline, 'run', side_effect=failure), \
                patch.object(self.pipeline, 'ensure_server', side_effect=failed_restore):
            with self.assertRaises(RuntimeError) as observed:
                self.pipeline.execute()
        self.assertIs(observed.exception, failure)
        self.assertIn('vLLM startup also failed', '\n'.join(observed.exception.__notes__))
        state = phase.read_json(self.pipeline.state_path)
        self.assertIn('original native failure', state['error'])
        self.assertIn('vLLM startup also failed', state['cleanup_error'])

    def test_cleanup_failure_after_success_is_reported_as_failure(self):
        with patch.object(self.pipeline, 'run'), patch.object(self.pipeline, 'cleanup', side_effect=OSError('cleanup failed')):
            with self.assertRaisesRegex(OSError, 'cleanup failed'):
                self.pipeline.execute()
        self.assertEqual(self.pipeline.state['status'], 'failed')


class MatchingTests(unittest.TestCase):
    def test_shell_wrappers_different_models_and_ports_are_excluded(self):
        table = {
            1: {'argv': ['python', '/venv/bin/vllm', 'serve', str(phase.MODEL), '--port', '8000']},
            2: {'argv': ['bash', '-c', f'vllm serve {phase.MODEL}']},
            3: {'argv': ['vllm', 'serve', '/other-model']},
            4: {'argv': ['vllm', 'serve', str(phase.MODEL), '--port', '8001']},
        }
        self.assertEqual(phase.server_roots(table), [1])


if __name__ == '__main__':
    unittest.main()
