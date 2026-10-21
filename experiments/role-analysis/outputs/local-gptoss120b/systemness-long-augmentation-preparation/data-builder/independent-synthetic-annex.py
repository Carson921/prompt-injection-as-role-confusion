"""Source/toy-only annex. Never reads archive text or calls real prepare."""
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0, str(ROOT / 'experiments/role-analysis'))
import systemness_long_augmentation_data as b


def main():
    source_sha = b.data.digest(b.__file__); checks = {}
    def expect_fail(call):
        try: call()
        except RuntimeError: return
        raise AssertionError('Expected failure was accepted')
    folder = Path(tempfile.mkdtemp(prefix='feature-review-long-augmentation-builder-', dir='/tmp'))
    # Missing freeze must prevent opening even a mocked source archive.
    with patch.object(b, 'OUT', folder / 'not-declared'), patch.object(b.gzip, 'open', side_effect=AssertionError('source opened')):
        expect_fail(lambda: b.historical_exclusions({}, None))
        expect_fail(lambda: b.scan_window({}, None, set(), set(), set()))
    checks['missing_plan_prevents_any_source_open'] = True
    with patch.object(b, 'OUT', folder), patch.object(b, 'declare', return_value={}), \
            patch.object(b.shutil, 'disk_usage', return_value=SimpleNamespace(free=128 * 2**20 - 1)), \
            patch.object(b.old, 'load_tokenizer', side_effect=AssertionError('tokenizer/source reached')):
        expect_fail(b.prepare)
    checks['low_disk_fails_before_tokenizer_or_source'] = True
    assert len(b.MANIFESTS) == 5 and 'systemness-head-confirmation/document-keys.json' in b.MANIFESTS
    assert b.FIRST == 4111 and b.LAST == 12302 and b.LAST - b.FIRST + 1 == 8192
    for name, digest in b.MANIFESTS.items():
        assert b.data.digest(b.data.runtime.OUT / name) == digest
    for path, digest in b.ORIGINAL.items(): assert b.data.digest(path) == digest
    checks['five_pinned_manifest_and_two_original249_file_shas'] = True
    # Independently recover toy full/192/1024 fingerprints from all historical
    # lines and both249 inventories. All strings below are locally synthetic.
    class Characters:
        def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
        def decode(self, ids, skip_special_tokens=False): return ''.join(map(chr, ids))
    tokenizer = Characters()
    manifest = {f'fixture-{i}': [dict(shard_line=i, source_url=f'https://OLD{i}.invalid:443/p#f',
        full_body_sha256=b.sha('stored-' + str(i)))] for i in range(5)}
    prior = ['prior-' + str(i) + 'x' * 1300 for i in range(249)]
    raw = [dict(text='raw-' + str(i) + 'y' * 1300) for i in range(249)]
    articles = [dict(text='article-' + str(i) + 'z' * 1400, url=f'https://ARTICLE{i}.invalid:443/a#fragment') for i in range(5)]
    original_read = b.read
    def fake_read(path):
        if str(path) in manifest: return manifest[str(path)]
        if Path(path).name == 'raw_training_data.json': return raw
        raise AssertionError('Unexpected read in toy historical recovery: ' + str(path))
    with patch.object(b, 'guarded_source_access', return_value=None), patch.object(b, 'read', side_effect=fake_read), \
            patch.object(b.pd, 'read_parquet', return_value=pd.DataFrame(dict(question=prior))), \
            patch.object(b.gzip, 'open', side_effect=lambda *a, **k: io.StringIO(''.join(json.dumps(item) + '\n' for item in articles))):
        hashes, urls, lines, proof = b.historical_exclusions(dict(source_signature=dict(manifests=manifest)), tokenizer)
    assert lines == set(range(5)) and proof['original249_inventory_counts'] == [249, 249]
    for text in prior + [row['text'] for row in raw] + [row['text'] for row in articles]:
        assert {b.sha(text), b.sha(text[:192]), b.sha(text[:1024])} <= hashes
    for i in range(5):
        assert b.sha('stored-' + str(i)) in hashes
        assert f'https://old{i}.invalid/p' in urls and f'https://article{i}.invalid/a' in urls
    checks['all_old_lines_and_both249_full_192_1024_hashes_recovered_on_toys'] = True
    # All352 source ranks fixed before an independently seeded split; supplied
    # source iteration order cannot change the selected sources or partition.
    candidates = []
    for index in range(352):
        candidates.append(dict(shard_line=b.FIRST + index, rank_sha256=b.sha('rank-' + str(index)),
            full_body_sha256=b.sha('full-' + str(index)), body_sha256=b.sha('body-' + str(index)),
            prefix192_sha256=b.sha('short-' + str(index)), source_url=f'https://toy.invalid/{index}',
            canonical_source_url=f'https://toy.invalid/{index}', _body='synthetic-only'))
    selected, _ = b.select_candidates(candidates, set(), set())
    reversed_selection, _ = b.select_candidates(list(reversed(candidates)), set(), set())
    assert selected == reversed_selection
    expected_split_order = sorted(candidates, key=lambda row: b.sha('\0'.join(map(str,
        [b.SPLIT_SEED, b.REVISION, b.SHARD, row['shard_line'], row['full_body_sha256']]))))
    expected_fit = {row['shard_line'] for row in expected_split_order[:256]}
    assert {row['shard_line'] for row in selected if row['split'] == 'fit'} == expected_fit
    assert [row['document_id'] for row in selected] == list(range(6000000, 6000352))
    assert [row['new_document_ix'] for row in selected] == list(range(352))
    assert len({row['source_rank'] for row in selected}) == 352
    assert any(row['source_rank'] != row['new_document_ix'] for row in selected)
    expect_fail(lambda: b.select_candidates(candidates[:351], set(), set()))
    checks['exact352_or_fail_and_independent_seeded_256_96_split'] = True
    checks['source_rank_distinct_from_render_document_namespace'] = True
    for hash_field in ['full_body_sha256', 'body_sha256', 'prefix192_sha256']:
        expect_fail(lambda field=hash_field: b.select_candidates(candidates, {candidates[0][field]}, set()))
    expect_fail(lambda: b.select_candidates(candidates, set(), {candidates[0]['source_url']}))
    checks['full_body_prefix_or_url_collision_causes_fixed_count_failure'] = True
    # A fixed short synthetic window cannot be silently extended.
    with patch.object(b, 'guarded_source_access', return_value=None), patch.object(b, 'FIRST', 1), patch.object(b, 'LAST', 4), \
            patch.object(b.gzip, 'open', side_effect=lambda *a, **k: io.StringIO('{}\n{}\n{}\n')), \
            patch.object(b, 'make_candidate', side_effect=lambda article, line, tok: (dict(shard_line=line), None)), \
            patch.object(b, 'select_candidates', side_effect=AssertionError('Truncated window reached selection')):
        expect_fail(lambda: b.scan_window({}, None, set(), set(), set()))
    checks['truncated_fixed_window_rejected_before_selection'] = True
    for split, count in [('fit', 256), ('validation', 128)]:
        positions = b.data.uniform_indices(1024, count)
        assert positions[0] == 0 and positions[-1] == 1023 and len(set(positions)) == count
        assert np.sum(positions >= 512) == count // 2
    assert 256 * 5 * 256 == 327680 and 96 * 5 * 128 == 61440
    render = inspect.getsource(b.render); verify = inspect.getsource(b.verify_content)
    assert 'old.parse_native_labels' in render and 'len(entries)==len(actual)==BODY_CAP' in render
    assert 'data.serialize' in render and "for role in data.ROLES" in render
    assert 'old.parse_native_labels' in verify and "body_positions=sorted(actual)" in verify
    checks['uniform_offsets_tail_counts_and_independent_parser_source_path'] = True
    assert b.data.digest(b.__file__) == source_sha
    report = dict(source_sha256_before=source_sha, source_sha256_after=source_sha, checks=checks,
        actual_source_text_read=False, original_source_text_read=False, synthetic_only=True,
        actual_prepare_called=False, formal_outputs_written=False, gpu_work_run=False,
        model_probabilities_read=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = Path('/tmp/feature_review_long_augmentation_builder_audit.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(target), sha256=b.data.digest(target), **report)))


if __name__ == '__main__': main()
