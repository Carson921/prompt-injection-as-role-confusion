"""Independent pure mock annex; only owner /tmp certificate fixtures are read."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

PATH = Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/systemness_aug_confirm_snapshot.py')
spec = importlib.util.spec_from_file_location('closed_annex_target', PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    expected = '382449444d5963ca7902b437df0bccc5f89bbcf33c5a44eb296b0f9dde374b95'
    assert m.digest(PATH) == expected
    report = json.loads(Path('/tmp/systemness-aug-confirm-snapshot-preflight-xz0b3nhn/report.json').read_text())
    folder = Path(report['fixture_directory']); audit_source = folder / 'synthetic-auditor.py'
    values = dict(HERE=folder / 'sources', BASE=folder / 'metadata', OUT=folder / 'snapshot',
        TRAINER=folder / 'metadata/systemness-long-augmentation-probes', OLD_HEAD=folder / 'metadata/systemness-head-probes',
        NEW_DATA=folder / 'metadata/systemness-long-augmentation', OLD_BANK=folder / 'old-bank', NEW_BANK=folder / 'new-bank',
        CHECKPOINT=folder / 'checkpoint', FUTURE_TEST_OUT=folder / 'future-test', FUTURE_TEST_CACHE=folder / 'future-cache',
        OLD_MODEL_SHA=m.digest(folder / 'metadata/systemness-head-probes/models/joint-mlp128.npz'),
        APPROVED_AUDITOR_SOURCE_SHA=m.digest(audit_source))
    checks = {}
    with patch.multiple(m, **values):
        witness = json.loads((folder / 'snapshot/readout-snapshot.json').read_text())
        original_read = m.read
        def read_metadata_only(path):
            if Path(path).name == 'raw_training_data.json': raise AssertionError('Original249 text JSON was decoded')
            return original_read(path)
        original_load = np.load
        allowed = {str(path) for path in m.models_and_scaler()[0].values()} | {str(m.models_and_scaler()[1])}
        def load_portable_only(path, *args, **kwargs):
            assert str(path) in allowed, 'Unexpected array/probability read: ' + str(path)
            return original_load(path, *args, **kwargs)
        with patch.object(m, 'read', side_effect=read_metadata_only), patch.object(Path, 'rglob', side_effect=AssertionError('Global discovery forbidden')), \
                patch.object(np, 'load', side_effect=load_portable_only):
            paths, thresholds, heads = m.verify_closed_snapshot(witness)
            rebuilt = m.build_conditional_snapshot(folder / 'synthetic-actual-audit-witness.json')
            assert rebuilt['physical_model_sha256'] == witness['physical_model_sha256']
            assert len(paths) == 3 and len(thresholds) == 4 and len(heads) == 8
            assert thresholds['old-head-original'] == m.OLD_THRESHOLD
            checks['valid_three_physical_four_record_closed_snapshot'] = True
            checks['build_verify_no_raw249_JSON_or_rglob_or_probabilities'] = True
        original_new = paths['new']
        with original_load(original_new, allow_pickle=False) as archive: payload = {key: archive[key].copy() for key in archive.files}
        payload['weight::head.bias'][0] += 1
        forged = folder / 'annex-forged-new.npz'; np.savez_compressed(forged, **payload)
        changed = copy.deepcopy(witness)
        changed['physical_model_paths']['new'] = str(forged)
        changed['physical_model_sha256']['new'] = m.digest(forged)
        changed['runtime_files_sha256'][str(forged)] = m.digest(forged)
        changed['portable_metadata']['new'] = m.portable_metadata(forged, payload['mean'], payload['scale'], heads, arm='new', plan=m.read(m.TRAINER / 'search-plan.json'))
        content = dict(changed); content.pop('snapshot_sha256'); changed['snapshot_sha256'] = m.canonical(content)
        try: m.verify_closed_snapshot(changed)
        except RuntimeError as error:
            assert 'independently audited readout' in str(error), str(error)
        else: raise AssertionError('Rehashed but unaudited physical replacement passed')
        checks['recomputed_contenthash_unaudited_new_weights_rejected'] = True
        changed = copy.deepcopy(witness); changed['record_thresholds']['old-head-original'] = .5
        content = dict(changed); content.pop('snapshot_sha256'); changed['snapshot_sha256'] = m.canonical(content)
        try: m.verify_closed_snapshot(changed)
        except RuntimeError: pass
        else: raise AssertionError('Rehashed frozen-threshold alteration passed')
        checks['recomputed_contenthash_old_threshold_tamper_rejected'] = True
    assert m.digest(PATH) == expected
    output = dict(source_sha256_before=expected, source_sha256_after=expected, checks=checks,
        synthetic_owner_fixture_only=True, actual_models_or_development_results_read=False,
        new_test_body_or_IDs_read=False, formal_source_or_outputs_modified=False, gpu_used=False,
        replay_script_sha256=m.digest(__file__))
    target = Path('/tmp/feature_review_aug_snapshot_audit.json'); target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(report=str(target), report_sha256=m.digest(target), **output)))


if __name__ == '__main__': main()
