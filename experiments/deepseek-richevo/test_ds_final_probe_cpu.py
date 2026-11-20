"""Backend parity guard regressions; uses metadata only, no model or API."""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ds_common import PROBE_RUN, read_json, write_json
import ds_final_probe
from ds_final_probe import audit_backend_parity


class BackendParityTests(unittest.TestCase):
    def setUp(self):
        self.training_backend = read_json(PROBE_RUN / 'model_backend.json')

    def test_training_backend_matches_with_different_load_statistics(self):
        current = copy.deepcopy(self.training_backend)
        current['load_seconds'] += 100
        current['gpu_memory'] = []
        with tempfile.TemporaryDirectory() as output:
            audit = audit_backend_parity(current, output)
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['mismatches'], [])
            self.assertEqual(read_json(Path(output) / 'final_probe_model_backend.json'), current)
            self.assertTrue(read_json(Path(output) / 'final_probe_backend_audit.json')['passed'])

    def test_changed_kernel_revision_fails_and_preserves_evidence(self):
        current = copy.deepcopy(self.training_backend)
        current['loaded_kernels'][0]['revision'] = '0' * 40
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, 'loaded_kernels.repository/version/revision'):
                audit_backend_parity(current, output)
            audit = read_json(Path(output) / 'final_probe_backend_audit.json')
            self.assertFalse(audit['passed'])
            self.assertEqual(audit['mismatches'][0]['actual'][0]['revision'], '0' * 40)
            self.assertEqual(read_json(Path(output) / 'final_probe_model_backend.json'), current)

    def test_equivalent_runtime_tuples_are_normalized_to_json_lists(self):
        current = copy.deepcopy(self.training_backend)
        current['quantization_config']['weight_block_size'] = tuple(
            current['quantization_config']['weight_block_size'])
        current['official_checkpoint_quantization_config']['weight_block_size'] = tuple(
            current['official_checkpoint_quantization_config']['weight_block_size'])
        with tempfile.TemporaryDirectory() as output:
            self.assertTrue(audit_backend_parity(current, output)['passed'])

    def test_missing_numeric_implementation_field_fails(self):
        current = copy.deepcopy(self.training_backend)
        del current['attention_query_tile_size']
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, 'attention_query_tile_size'):
                audit_backend_parity(current, output)

    def test_exact_long_ipi_lengths_are_prefilled_before_model_loading(self):
        import ds_ipi
        import ds_prefill_cache_cpu
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            ipi_run, output = root / 'ipi', root / 'out'
            write_json(ipi_run / 'protocol.json', dict(rows=[dict(prompt_tokens=17513)]))
            scorer = Mock()
            scorer.prober = SimpleNamespace(model=SimpleNamespace(_deepseek_role_probe_backend=self.training_backend))
            events = Mock()
            events.attach_mock(Mock(), 'prefill')
            events.attach_mock(Mock(return_value=scorer), 'load')
            argv = ['ds_final_probe.py', '--dataset', str(root / 'dataset.json'), '--output', str(output),
                    '--scores-out', str(root / 'scores.json'), '--ipi-runs', str(ipi_run)]
            with patch.object(sys, 'argv', argv), patch.object(ds_final_probe, 'Scorer', events.load), \
                    patch.object(ds_prefill_cache_cpu, 'prefill_lengths', events.prefill), \
                    patch.object(ds_ipi, 'probe_requests'), \
                    patch.object(ds_final_probe, 'prefetch_during_load', return_value=nullcontext()):
                ds_final_probe.main()
            self.assertEqual(events.mock_calls[0][0], 'prefill')
            self.assertEqual(events.mock_calls[0].args[0], {17513})
            self.assertTrue(events.mock_calls[0].kwargs['apply'])
            self.assertEqual(events.mock_calls[1][0], 'load')
            scorer.score.assert_not_called()
            scorer.close.assert_called_once()

    def test_optional_prefetch_startup_failure_keeps_original_exception(self):
        with tempfile.TemporaryDirectory() as root:
            failure = RuntimeError('primary model failure')
            with patch.object(ds_final_probe.subprocess, 'Popen', side_effect=OSError('reader unavailable')):
                with self.assertRaises(RuntimeError) as observed:
                    with ds_final_probe.prefetch_during_load(Path(root), Path(root) / 'scores.json'):
                        raise failure
            self.assertIs(observed.exception, failure)

    def test_optional_prefetch_cleanup_failure_keeps_original_exception(self):
        with tempfile.TemporaryDirectory() as root:
            failure = RuntimeError('primary model failure')
            helper = Mock(pid=123)
            helper.poll.return_value = None
            helper.terminate.side_effect = OSError('reader already gone')
            with patch.object(ds_final_probe.subprocess, 'Popen', return_value=helper):
                with self.assertRaises(RuntimeError) as observed:
                    with ds_final_probe.prefetch_during_load(Path(root), Path(root) / 'scores.json'):
                        raise failure
            self.assertIs(observed.exception, failure)

    def test_reader_receives_fresh_backend_cutoff_and_its_own_status_directory(self):
        with tempfile.TemporaryDirectory() as root:
            helper = Mock(pid=123)
            helper.poll.return_value = 0
            with patch.object(ds_final_probe.subprocess, 'Popen', return_value=helper) as launch:
                with ds_final_probe.prefetch_during_load(Path(root) / 'e', Path(root) / 'o/scores.json'):
                    pass
            args = launch.call_args.args[0]
            self.assertEqual(args[args.index('--kind') + 1], 'final-probe')
            self.assertIn('/o/final_probe_prefetch/', args[args.index('--status-dir') + 1])
            self.assertIn('--backend-after-unix', args)
            self.assertEqual(args[-1], '--already-read')


if __name__ == '__main__':
    unittest.main()
