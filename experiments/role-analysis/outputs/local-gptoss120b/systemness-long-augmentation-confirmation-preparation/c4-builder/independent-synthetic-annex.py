"""Independent source/toy-cache audit; no real new C4 source body or IDs."""
import ast
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0, str(ROOT / 'experiments/role-analysis'))
import systemness_aug_confirmation_data as b
import systemness_head_confirm as h


def main():
    expected = 'cb31b133382044453f87057590f3c71068902f7fccfb5fbcf2889dd736a7d4c2'
    assert b.data.digest(b.__file__) == expected
    checks = {}; folder = Path('/tmp/systemness-aug-confirmation-data-preflight-r9he5o3h')
    plan = json.loads((folder / 'roundtrip/plan.json').read_text())
    def reject(fn):
        try: fn()
        except (RuntimeError, KeyError, FileNotFoundError): return
        raise AssertionError('Expected fault was accepted')
    tree = ast.parse(Path(b.__file__).read_text())
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert 'rglob' not in calls and 'verified_completion' not in calls and 'selections' not in calls and 'frozen_snapshot' not in calls
    assert b.FIRST == 12303 and b.LAST == 20494 and b.LAST - b.FIRST + 1 == 8192
    assert (b.GENERAL, b.FULL, b.PAIRED) == (400, 128, 64) and len(b.MANIFESTS) == 6
    checks['static_closed_inventory_no_dynamic_verifier_or_rglob'] = True
    checks['fixed8192_window_and400_plus128_disjoint_with64_paired'] = True
    with patch.object(b, 'OUT', folder / 'never-declared'), patch.object(b.gzip, 'open', side_effect=AssertionError('New archive source opened')):
        reject(lambda: b.scan_window({}, None, set(), set(), set()))
        reject(b.prepare)
    checks['source_decode_and_prepare_blocked_before_declaration'] = True
    candidates = []
    for index in range(8):
        body_tokens = 96 if index in [1, 3] else 1024
        candidates.append(dict(shard_line=b.FIRST + index, rank_sha256=f'{index:064x}',
            full_body_sha256=b.sha('full-' + str(index)), body_sha256=b.sha('body-' + str(index)),
            prefix192_sha256=b.sha('short-' + str(index)), source_url=f'https://fixture.invalid/{index}',
            canonical_source_url=f'https://fixture.invalid/{index}', body_tokens=body_tokens, _body='synthetic fixture'))
    chosen, _ = b.select_candidates(candidates, set(), set(), (4, 2, 2))
    again, _ = b.select_candidates(candidates[::-1], set(), set(), (4, 2, 2))
    assert chosen == again and [row['shard_line'] for row in chosen] == [b.FIRST + i for i in range(6)]
    assert [row['cohort'] for row in chosen] == ['general'] * 4 + ['full-length'] * 2
    assert sum(row['up_to1024_variant'] for row in chosen) == 2 and all(row['body_tokens'] == 1024 for row in chosen[4:])
    assert not {row['shard_line'] for row in chosen[:4]} & {row['shard_line'] for row in chosen[4:]}
    replacement, _ = b.select_candidates(candidates, {candidates[4]['prefix192_sha256']}, set(), (4, 2, 2))
    assert [row['shard_line'] for row in replacement[4:]] == [b.FIRST + 5, b.FIRST + 6]
    reject(lambda: b.select_candidates(candidates[:5], set(), set(), (4, 2, 2)))
    reject(lambda: b.select_candidates(candidates, set(), set(), (9, 2, 2)))
    checks['independent_greedy_general_then_full_prefix_dedupe_and_no_extension'] = True
    tokenizer = b.native.load_tokenizer()
    # Reuse only the author's synthetic /tmp rows; forbid source scans or model
    # selection functions. Completed prepare must remain read-only.
    with patch.multiple(b, OUT=folder / 'roundtrip', CACHE=folder / 'roundtrip-cache', GENERAL=4, FULL=2, PAIRED=2), \
            patch.object(b, 'plan_value', return_value=plan), \
            patch.object(b, 'scan_window', side_effect=AssertionError('Source scan in completed load')), \
            patch.object(b, 'historical_exclusions', side_effect=AssertionError('Historical body replay in completed load')), \
            patch.object(h, 'selections', side_effect=AssertionError('Model/probability access')), \
            patch.object(h.head_probes, 'verified_completion', side_effect=AssertionError('Probability completion access')):
        before = {name: b.data.digest(b.CACHE / name) for name in b.FILES}
        metadata, inputs, rows, tokens = b.load_verified_data()
        assert b.prepare() == metadata and before == {name: b.data.digest(b.CACHE / name) for name in b.FILES}
        keys = b.read(b.OUT / 'document-keys.json'); bodies = b.read(b.CACHE / 'selected-bodies.json')
        assert len(keys) == 6 and len(inputs) == 44 and len(rows) == 26053
        assert np.array_equal(rows.cache_ix, np.arange(len(rows)))
        assert np.array_equal(rows.sample_ix, b.SAMPLE_START + np.arange(len(rows)))
        lookup = {int(index): group for index, group in rows.groupby('prompt_ix', sort=False)}
        positions, structural = 0, 0
        for entry in inputs.itertuples():
            ids = tokenizer.encode(entry.prompt, add_special_tokens=False)
            labels = h.parse_labels(ids, tokenizer); group = lookup[entry.prompt_ix]
            expected_labels = {position: role for position, role in labels.items() if role != 'tool_call'}
            assert ids == tokens[entry.prompt_ix] and dict(zip(group.token_ix, group.role)) == expected_labels
            assert np.array_equal(group.token_id, np.asarray(ids)[group.token_ix.to_numpy()])
            for _, message in group.groupby('message_ix', sort=False): assert np.array_equal(message.body_offset, np.arange(len(message)))
            if entry.profile == 'long-full': assert len(group) == 1024 and entry.role in b.data.ROLES
            positions += len(expected_labels); structural += sum(role == 'tool_call' for role in labels.values())
        assert positions == len(rows) and structural > 0 and 'tool_call' not in set(rows.role)
        checks['public_loader_completed_prepare_readonly_owned44inputs26053rows'] = True
        checks['independent_h_parser_every_normal_content_position_and_structural_calls'] = True
        # Independent reproduction of the original _prepare_data sequence rules.
        def old_specs(item):
            body, ix = item['_body'], item['source_index']; ids = tokenizer.encode(body, add_special_tokens=False); bos = bool(ix % 2)
            if item['cohort'] == 'full-length': return [([(role, body)], 'long-full', bos) for role in b.data.ROLES]
            short = tokenizer.decode(ids[:192], skip_special_tokens=False)
            if ix % 4 == 0: short = b.INSTRUCTIONS[(ix // 4) % 4] + '\n\nReference text:\n' + short
            result = [([(role, short)], 'single', bos) for role in b.data.ROLES]
            width = min(48, max(16, len(ids) // 5)); shift = ix % 5 * width; chunks = []
            for j in range(5):
                offset = (j * width + shift) % max(width, len(ids) - width)
                chunk = tokenizer.decode(ids[offset:offset + width], skip_special_tokens=False)
                if ix % 4 == 0: chunk = b.INSTRUCTIONS[(ix // 4) % 4] + '\n\n' + chunk
                chunks.append(chunk)
            messages = list(zip(b.data.ROLES[:4], chunks[:4])) + [('user', chunks[1]), ('tool_call', '{"query":"context"}'),
                ('tool', chunks[4]), ('cot', chunks[2]), ('assistant', chunks[3])]
            result.append((messages, 'multi', True))
            if item['up_to1024_variant']: result.extend([([(role, body)], 'long-single', bos) for role in b.data.ROLES])
            return result
        for key, body in zip(keys, bodies):
            item = dict(key, _body=body); assert b.message_specs(item, tokenizer) == old_specs(item)
        gallery = b.gallery_index(keys, (4, 2, 2)); assert gallery['selected_before_probabilities'] and not gallery['score_based_selection']
        assert [entry['cohort_rank'] for entry in gallery['general_multi']] == list(range(4))
        assert len(gallery['full_system']) == 2
        checks['old_single_multi_long_sequence_rules_and_fixed_gallery'] = True
        bad = rows.copy(); bad.loc[0, 'body_offset'] = 17
        reject(lambda: b.verify_content(inputs, bad, tokens, keys, bodies, tokenizer, (4, 2, 2)))
        checks['actual_content_math_rejects_wrong_offset_beyond_filehash'] = True
    assert b.data.digest(b.__file__) == expected
    report = dict(source_sha256_before=expected, source_sha256_after=expected, checks=checks,
        synthetic_inputs=44, synthetic_rows=26053, actual_test_sources_or_IDs_read=False,
        actual_declare_prepare_or_verify_called=False, real_classifier_probabilities_read=False,
        gpu_used=False, formal_source_or_outputs_modified=False, replay_script_sha256=b.data.digest(__file__))
    target = Path('/tmp/feature_review_aug_c4_builder_audit.json'); target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(target), report_sha256=b.data.digest(target), **report)))


if __name__ == '__main__': main()
