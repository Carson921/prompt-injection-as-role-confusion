"""CPU-only cache publication tests; all writes stay in a temporary directory."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ds_common import read_json, write_json
import ds_prefill_cache_cpu as prefill


class PublicationTests(unittest.TestCase):
    def test_exact_lengths_preserve_existing_and_protected_entries(self):
        reference = read_json(prefill.REFERENCE)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cache = root / 'cache'
            for n in (8, 1200, 8309):
                for projection in prefill.performance.PROJECTIONS:
                    path = prefill.performance.native_cache_path(
                        cache, reference['native_identity'], prefill.performance.tuning_key(projection, n))
                    write_json(path, dict(original=n, projection=projection))
            with patch.object(prefill, 'validate_reference', return_value=(reference, dict(test=True))):
                result = prefill.prefill_lengths([1200, 1201, 1201], root / 'audit.json', apply=True, cache_dir=cache)
                self.assertEqual(result['application']['created'], 2)
                self.assertEqual(result['application']['skipped_existing'], 2)
                self.assertEqual(result['application']['skipped_protected'], 4)
                self.assertEqual(result['existing_changed_externally'], [])
                self.assertEqual(result['seeded_replaced_by_native'], [])
                repeated = prefill.prefill_lengths([1200, 1201], root / 'repeat.json', apply=True, cache_dir=cache)
                self.assertNotIn('created', repeated['application'])
            for target in result['plan']['targets']:
                value = read_json(target['path'])
                if target['tokens'] == 1201:
                    self.assertEqual(value['timings'], [])
                else:
                    self.assertEqual(value['original'], target['tokens'])

    def test_ipi_protocol_import_keeps_actual_long_lengths(self):
        with tempfile.TemporaryDirectory() as root:
            write_json(Path(root) / 'protocol.json', dict(rows=[dict(prompt_tokens=n) for n in (1174, 17513, 17513)]))
            lengths, inputs = prefill.lengths_from_ipi_runs([root])
            self.assertEqual(lengths, {1174, 17513})
            self.assertEqual(inputs[0]['requests'], 3)


if __name__ == '__main__':
    unittest.main()
