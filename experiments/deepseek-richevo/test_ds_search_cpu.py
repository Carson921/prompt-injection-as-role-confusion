"""CPU regressions for cache admission and resumable DeepSeek optimization.

No model weights are loaded and no API calls are made. Uses the local tokenizer
and frozen experiment inputs, as the production search does.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import ds_evolution as evolution
from ds_common import Codec, MODEL, ORIGINAL_DELIMITER, read_json, sha, write_json
from ds_scorer import Scorer


class CacheAdmissionTests(unittest.TestCase):
    def make_scorer(self, directory):
        scorer = Scorer.__new__(Scorer)
        scorer.output = Path(directory)
        scorer.codec = Codec(MODEL)
        scorer.protocol_sha256 = 'unit-test-protocol'
        scorer.allow_special = False
        return scorer

    def test_allowed_baseline_cache_cannot_bypass_strict_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self.make_scorer(directory)
            write_json(scorer.output / 'scores' / sha(ORIGINAL_DELIMITER)[:24] / 'score.json',
                       dict(status='complete', template=ORIGINAL_DELIMITER,
                            protocol_sha256=scorer.protocol_sha256, allow_special=True, fitness=.99))
            with self.assertRaisesRegex(ValueError, 'special-token'):
                scorer.score(ORIGINAL_DELIMITER)

    def test_allowed_cache_is_reused_only_for_allowed_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self.make_scorer(directory)
            scorer.allow_special = True
            write_json(scorer.output / 'scores' / sha(ORIGINAL_DELIMITER)[:24] / 'score.json',
                       dict(status='complete', template=ORIGINAL_DELIMITER,
                            protocol_sha256=scorer.protocol_sha256, allow_special=True, fitness=.99))
            self.assertTrue(scorer.score(ORIGINAL_DELIMITER)['cache_hit'])


class ResumeTests(unittest.TestCase):
    def arguments(self, root):
        dataset = Path(root) / 'dataset.json'
        write_json(dataset, dict(suites=['banking']))
        return evolution.R.arguments(['--output', str(Path(root) / 'run'), '--dataset', str(dataset),
                                      '--caps', '128,256,512'])

    def test_dataset_change_cannot_reuse_old_fitness(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.arguments(root)
            evolution.DeepSeekRun(args)
            write_json(args.dataset, dict(suites=['banking'], changed_context=True))
            with self.assertRaisesRegex(ValueError, 'Resume binding differs'):
                evolution.DeepSeekRun(args)

    def test_unbound_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.arguments(root)
            output = Path(args.output)
            output.mkdir()
            (output / 'scores.jsonl').write_text(json.dumps(dict(status='complete', fitness=.99)) + '\n')
            with self.assertRaisesRegex(ValueError, 'lack a DeepSeek resume binding'):
                evolution.DeepSeekRun(args)

    def test_cap_stage_and_stall_count_survive_restart(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.arguments(root)
            evolution.DeepSeekRun(args)
            output = Path(args.output)
            rows = [dict(status='complete', template=template, fitness=fitness, wrapper_tokens=5,
                         cap_index=1, source='seed') for template, fitness in
                    [('<|User|>One: {goal}', .6), ('<|User|>Two: {goal}', .5)]]
            (output / 'scores.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            # A cap transition may be persisted before the first score at that cap.
            with (output / 'run.log').open('a') as stream:
                stream.write(json.dumps(dict(event='cap_raised', cap=512, unix=1234)) + '\n')
            resumed = evolution.DeepSeekRun(args)
            self.assertEqual(resumed.cap, 512)
            self.assertEqual(resumed.cap_started, 1234)
            self.assertEqual(resumed.evals_since_best, 1)
            self.assertEqual(len(resumed.records), 2)


class NativeImportTests(unittest.TestCase):
    def test_real_scorer_initialization_keeps_native_model_imports(self):
        # A fresh process is essential: Python/Transformers cache successful
        # imports and optional-dependency availability checks.
        here = Path(__file__).resolve().parent
        script = """
import json, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import ds_evolution
import ds_scorer
sys.path.insert(0, str(ds_scorer.ROLE_ANALYSIS))
import deepseek_role_probe_model as backbone
def fake_load(*args, **kwargs):
    import torch
    from transformers import DeepseekV4ForCausalLM
    from transformers.utils import is_torchvision_available
    assert not is_torchvision_available()
    assert 'torchvision' not in sys.modules
    assert not torch.cuda.is_initialized()
    return object()
with tempfile.TemporaryDirectory() as output, patch.object(backbone, 'load_model', side_effect=fake_load) as load:
    scorer = ds_scorer.Scorer(Path('outputs/dsevo-20261118/dataset.json'), output)
    assert load.call_count == 1
    assert len(scorer.contexts) == 6 and len(scorer.ipi_cases) == 6
    print('real-scorer-native-import-ok')
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=here, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('real-scorer-native-import-ok', result.stdout)

    def test_initialization_failure_cancels_queued_llm_work(self):
        run = evolution.DeepSeekRun.__new__(evolution.DeepSeekRun)
        run.stop = Mock()
        run.client = Mock()
        # The shared loop can fail after constructing its client but before
        # entering its own try/finally (for example while loading the model).
        with patch.object(evolution.ds_scorer, 'prepare_native_runtime'), \
                patch.object(evolution.DeepSeekRun.__mro__[1], 'run', side_effect=RuntimeError('load failed')):
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                run.run()
        run.stop.set.assert_called_once()
        run.client.pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


if __name__ == '__main__':
    unittest.main()
