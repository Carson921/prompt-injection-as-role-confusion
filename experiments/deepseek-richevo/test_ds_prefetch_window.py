"""The auxiliary reader must follow native PTE references, not cache residency."""
from pathlib import Path
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import ds_prefetch_window as window
from ds_prefetch_window import referenced_shards, window_for, read_shard, backend_finished, matches_process


class WindowTests(unittest.TestCase):
    def test_ignores_global_mtp_foreign_paths_and_unreferenced_maps(self):
        model = Path('/model')
        def mapped(index, referenced, parent='/model'):
            return f'1000-2000 rw-p 0000 00:01 123 {parent}/model-{index:05d}-of-00048.safetensors\nRss: 999999 kB\nReferenced: {referenced} kB\n'
        text = mapped(1, 90000) + mapped(27, 10000) + mapped(28, 4096) + mapped(29, 0) + mapped(44, 50000, '/other')
        text += mapped(45, 99999) + mapped(46, 99999) + mapped(47, 99999) + mapped(48, 99999)
        references = referenced_shards(text, model)
        self.assertEqual(references, {1: 90000, 27: 10000})
        self.assertEqual(window_for(references), (27, {27, 28}))

    def test_never_reads_beyond_decoder_shards(self):
        self.assertEqual(window_for({44: 10000}), (44, {44}))
        self.assertEqual(window_for({}), (None, set()))

    def test_reader_does_not_modify_file_and_reports_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            value = b'unchanged source bytes'
            path.write_bytes(value)
            stop, cancel = threading.Event(), threading.Event()
            result = read_shard(path, cancel, stop)
            self.assertEqual((result['status'], result['bytes']), ('complete', len(value)))
            self.assertEqual(path.read_bytes(), value)
            cancel.set()
            result = read_shard(path, cancel, stop)
            self.assertEqual((result['status'], result['bytes']), ('cancelled', 0))

    def test_old_backend_does_not_stop_a_later_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_backend.json'
            self.assertFalse(backend_finished(path))
            path.write_text('{}')
            os.utime(path, (100, 100))
            self.assertTrue(backend_finished(path))
            self.assertFalse(backend_finished(path, 101))
            os.utime(path, (102, 102))
            self.assertTrue(backend_finished(path, 101))

    def test_final_probe_pid_matches_output_without_a_model_cli_flag(self):
        model = window.DEFAULT_MODEL.resolve()
        args = ['python', '-u', 'ds_final_probe.py', '--output', 'outputs/e']
        self.assertTrue(matches_process(args, Path('/work'), model, Path('/work/outputs/e'), 'final-probe'))
        self.assertFalse(matches_process(args, Path('/work'), model, Path('/other'), 'final-probe'))
        self.assertFalse(matches_process(args, Path('/work'), Path('/other'), Path('/work/outputs/e'), 'final-probe'))

    def test_vllm_pid_requires_exact_serve_model(self):
        model = window.DEFAULT_MODEL.resolve()
        args = ['python', '/venv/bin/vllm', 'serve', str(model)]
        self.assertTrue(matches_process(args, Path('/work'), model, Path('/logs'), 'vllm'))
        self.assertFalse(matches_process(['bash', '-c', ' '.join(args)], Path('/work'), model, Path('/logs'), 'vllm'))

    def test_vllm_window_follows_slowest_mapped_rank(self):
        model = window.DEFAULT_MODEL.resolve()
        def contents(path, *args, **kwargs):
            pid, name = path.parts[-2:]
            if pid == '12':
                return ''
            if name == 'maps':
                return str(model) + '/model-'
            index = 27 if pid == '10' else 30
            return f'1000-2000 rw-p 0000 00:01 123 {model}/model-{index:05d}-of-00048.safetensors\nReferenced: 9000 kB\n'
        with patch.object(window, 'owned_descendants', return_value={10, 11, 12}), \
                patch.object(Path, 'read_text', contents):
            _, current, targets, frontiers = window.observed_window(1, model, 'vllm')
        self.assertEqual(current, 27)
        self.assertEqual(targets, {27, 28})
        self.assertEqual(frontiers, {10: 27, 11: 30})


if __name__ == '__main__':
    unittest.main()
