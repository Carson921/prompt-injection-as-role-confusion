"""Read-only actual feature-bank audit. No LM/GPU/training/probability cache."""
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT = Path('/root/prompt-injection-as-role-confusion')
HERE = ROOT / 'experiments/role-analysis'
OUT = HERE / 'outputs/local-gptoss120b/systemness-long-augmentation'
BANK = Path('/dev/shm/role-analysis-long-augmentation-heads')
ROWS, DIMENSIONS, BATCH = 389120, 512, 4
FEATURE_SHA = '69e3db049f3de168c4c0922cfd1a935bb3bf19f91d697da75af2fe3752269cd8'
ROWS_SHA = '4baf4436eae8c2666630b38cc89ad4dca3ed3ee53393a489407bbf198d63b481'
SOURCE_SHA = '907204a31cea06bd3885d7148bd2684aaa77b9d3a322737cb8b3ad0c4629fd86'
DATA_COMPLETE_SHA = '6fe07499112eb816fb116905d7db245211297ec07b99cc45df5ec7be8efb847e'


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def frame_sha(frame, fields): return hashlib.sha256(frame[fields].to_csv(index=False).encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''): value.update(block)
    return value.hexdigest()


def main():
    require(not Path('/proc/296570').exists(), 'Extractor PID is still live; partial bank inspection forbidden.')
    require(read(OUT / 'extraction-status.json')['stage'] == 'extraction_complete', 'Feature extraction is not terminal complete.')
    bindings = {}
    def bind(path, expected=None):
        actual = digest(path)
        if expected is not None: require(actual == expected, 'File SHA mismatch: ' + str(path))
        bindings[str(path)] = actual; return actual
    bind(__file__); bind(HERE / 'systemness_long_augmentation_extract.py', SOURCE_SHA)
    plan, complete, metadata = read(OUT / 'extraction-plan.json'), read(OUT / 'extraction-complete.json'), read(OUT / 'data-complete.json')
    plan_sha = bind(OUT / 'extraction-plan.json'); complete_sha = bind(OUT / 'extraction-complete.json')
    bind(OUT / 'data-complete.json', DATA_COMPLETE_SHA)
    require(complete['plan_sha256'] == plan_sha and complete['signature'] == plan['signature']
        and complete['source_sha256'] == plan['source_sha256'] == SOURCE_SHA
        and complete['dependencies_sha256'] == plan['dependencies_sha256'] == plan['signature']['dependencies_sha256'], 'Extraction source/plan/completion binding mismatch.')
    for name, expected in plan['dependencies_sha256'].items(): bind(HERE / name, expected)
    require(complete['rows'] == ROWS and complete['dimensions'] == DIMENSIONS and complete['feature_shape'] == [ROWS, DIMENSIONS]
        and complete['dtype'] == 'float16' and complete['all_rows_written'] is True and complete['all_finite'] is True
        and complete['tests_read'] is False and complete['readout_training'] is False and complete['frozen_primary_changed'] is False,
        'Feature-bank completeness/geometry/development-only contract mismatch.')
    require(complete['rows_path'] == str(BANK / 'rows.parquet') and complete['features_path'] == str(BANK / 'features.f16'), 'Feature-bank path namespace mismatch.')
    require((BANK / 'features.f16').stat().st_size == ROWS * DIMENSIONS * 2 == 398458880, 'Feature-bank file is truncated or has an unexpected header.')
    bind(BANK / 'features.f16', FEATURE_SHA); bind(BANK / 'rows.parquet', ROWS_SHA)
    require(complete['features_sha256'] == FEATURE_SHA and complete['rows_sha256'] == ROWS_SHA
        == metadata['cache_sha256']['rows.parquet'] == plan['signature']['input_cache_sha256']['rows.parquet'], 'Raw bank/source row SHA mismatch.')
    source_cache = Path(metadata['cache_path'])
    require(plan['signature']['data_complete_sha256'] == DATA_COMPLETE_SHA
        and plan['signature']['input_cache_sha256'] == metadata['cache_sha256'], 'Native source cache declaration mismatch.')
    for name, expected in metadata['cache_sha256'].items(): bind(source_cache / name, expected)
    for name, expected in metadata['metadata_sha256'].items(): bind(OUT / name, expected)
    bind(OUT / 'plan.json', metadata['plan_sha256'])
    require(metadata['independent_native_labels_match'] is True and metadata['all_body_tokens1024'] is True
        and metadata['test_inputs_prepared'] is False and metadata['test_probabilities_read'] is False
        and metadata['test_metrics_read'] is False, 'Native source/parser proof was not development-only.')
    prior_source_audit = Path('/tmp/feature_review_long_augmentation_actual_audit.json')
    bind(prior_source_audit, 'a912e883f3778b81482ba28b2b740e35ef5f0c0c7f1f971be20e312e9054e248')
    source_proof = read(prior_source_audit)
    require(source_proof['bindings']['data_complete_sha256'] == DATA_COMPLETE_SHA
        and source_proof['checks']['native_independent_parser'] and source_proof['checks']['all_five_roles_share_actual1024_body'], 'Independent actual native source replay certificate mismatch.')
    rows = pd.read_parquet(BANK / 'rows.parquet')
    inputs = pd.read_parquet(source_cache / 'inputs.parquet', columns=['prompt_ix', 'uid', 'length', 'split', 'role', 'document_id'])
    tokens = read(source_cache / 'token-ids.json')
    require(len(rows) == ROWS and len(inputs) == len(tokens) == 1760
        and np.array_equal(rows.cache_ix, np.arange(ROWS))
        and np.array_equal(rows.sample_ix, 6000000000000 + np.arange(ROWS))
        and rows.groupby('split').size().to_dict() == {'fit': 327680, 'validation': 61440}
        and rows.groupby('split').document_id.nunique().to_dict() == {'fit': 256, 'validation': 96}
        and set(rows.wrapper) == {'native'} and set(rows.source_set) == {'long-augmentation'} and set(rows.profile) == {'long-full'},
        'Bank-local row namespace, order or partition mismatch.')
    require(not set(rows.loc[rows.split.eq('fit'), 'group_id']) & set(rows.loc[rows.split.eq('validation'), 'group_id']), 'Feature rows leak across development source splits.')
    lookup = {uid: group for uid, group in rows.groupby('uid', sort=False)}
    for entry in inputs.itertuples():
        group = lookup[entry.uid]; ids = np.asarray(tokens[entry.prompt_ix])
        sample = np.linspace(0, 1023, 256 if entry.split == 'fit' else 128, dtype=int)
        require(len(ids) == entry.length and np.array_equal(group.body_offset, sample)
            and np.array_equal(ids[group.token_ix.to_numpy()], group.token_id.to_numpy())
            and group.role.eq(entry.role).all() and group.prompt_ix.eq(entry.prompt_ix).all(), 'Native token/body-offset-to-feature row mapping mismatch.')
    ordered = inputs.sort_values(['length', 'uid']).reset_index(drop=True)
    parts = [ordered.iloc[start:start + BATCH] for start in range(0, len(ordered), BATCH)]
    targets = [np.concatenate([lookup[entry.uid].cache_ix.to_numpy(dtype=np.int64) for entry in part.itertuples()]) for part in parts]
    full_order = np.concatenate(targets)
    require(len(parts) == 440 and len(np.unique(full_order)) == ROWS and set(full_order) == set(range(ROWS)), 'Batch ownership does not cover each bank row exactly once.')
    progress = read(BANK / 'progress.json'); bind(BANK / 'progress.json', complete['progress_sha256'])
    initializer = read(BANK / 'initialized.json'); bind(BANK / 'initialized.json')
    require(initializer == dict(plan_sha256=plan_sha, rows=ROWS, dimensions=DIMENSIONS, dtype='float16', rows_sha256=ROWS_SHA), 'Bank initialization ownership mismatch.')
    require(progress['plan_sha256'] == plan_sha and progress['next_batch'] == 440 and progress['rows_written'] == ROWS
        and len(progress['batch_features_sha256']) == 440
        and progress['cache_ix_sha256'] == array_sha(full_order)
        and progress['sample_ix_sha256'] == array_sha(rows.iloc[full_order].sample_ix.to_numpy())
        and progress['prompt_prefix_sha256'] == frame_sha(ordered, ['uid', 'length']), 'Committed complete prompt-prefix proof mismatch.')
    features = np.memmap(BANK / 'features.f16', mode='r', dtype=np.float16, shape=(ROWS, DIMENSIONS))
    zeros, head_zeros, minimum, maximum = 0, np.zeros(8, np.int64), float('inf'), -float('inf')
    for start in range(0, ROWS, 4096):
        block = np.asarray(features[start:start + 4096]); require(np.isfinite(block).all(), 'Feature bank contains NaN/Inf/unwritten rows.')
        zeros += int(np.all(block == 0, axis=1).sum())
        head_zeros += np.all(block.reshape(-1, 8, 64) == 0, axis=2).sum(0)
        minimum = min(minimum, float(block.min())); maximum = max(maximum, float(block.max()))
    require(complete['zero_row_count'] == zeros and complete['all_rows_nonzero'] == (zeros == 0), 'Finite/nonzero diagnostics differ from actual bank.')
    for index, target in enumerate(targets):
        require(array_sha(features[target]) == progress['batch_features_sha256'][index], 'A committed batch differs from its actual feature SHA.')
    descriptor = dict(rows=ROWS, prompts=1760, batches=440, split_rows={'fit':327680,'validation':61440},
        source_rows={'long-augmentation': ROWS}, input_tokens=int(ordered.length.sum()), max_batch_rows=max(map(len, targets)),
        inputs_sha256=frame_sha(ordered, ['uid', 'prompt_ix', 'length']),
        row_mapping_sha256=frame_sha(rows, ['cache_ix', 'sample_ix', 'uid', 'token_ix', 'token_id', 'role', 'body_offset']),
        sample_ix_sha256=array_sha(rows.sample_ix.to_numpy()))
    require(descriptor == plan['data'] == complete['data'], 'Complete bank row descriptor differs from actual ordering.')
    selected = complete['selected_head_ids']; require(selected == plan['selected_head_ids'] == plan['signature']['selection']['selected_head_ids']
        and len(selected) == len(set(selected)) == 8 and complete['head_dim'] == 64
        and complete['ordered_layers_heads'] == [[value // 64, value % 64] for value in selected], 'Frozen unsorted head rank/column layout changed.')
    hooks, backend = read(OUT / 'extraction-hook-validation.json'), read(OUT / 'augmentation-model_backend.json')
    bind(OUT / 'extraction-hook-validation.json', complete['hook_validation_sha256'])
    bind(OUT / 'augmentation-model_backend.json', complete['model_backend_sha256'])
    require(complete['hook_check_disabled_writes'] is True and complete['hidden_states_bitexact'] is True
        and hooks['writes_disabled'] is True and hooks['hidden_states_bitexact'] is True and hooks['feature_rows_unchanged'] is True
        and hooks['sample_rows'] == len(targets[0]) and hooks['layers'] == sorted(set(value // 64 for value in selected)), 'Disabled-write/bitexact selected-head hook proof mismatch.')
    expected_map = {'model.embed_tokens':0, 'model.rotary_emb':0, 'model.norm':3, 'lm_head':3,
        **{f'model.layers.{i}':i//9 for i in range(36)}}
    require(backend['expert_backends'] == ['Mxfp4GptOssExperts'] and backend['device_map'] == expected_map
        and len(backend['gpus']) == 4 and all('A40' in value for value in backend['gpus'])
        and backend['load_path'] == plan['signature']['checkpoint_path'], 'Actual nativeMXFP4 four-sharded-A40 backend differs.')
    checkpoint = Path(backend['load_path']); require(not str(checkpoint).startswith('/workspace'), 'Do not access unavailable original mount.')
    config = read(checkpoint / 'config.json')
    bind(checkpoint / 'config.json', plan['signature']['config_sha256'])
    bind(checkpoint / 'verified-weights.json', plan['signature']['verified_weights_sha256'])
    require(config['quantization_config']['quant_method'] == 'mxfp4' and (config['num_hidden_layers'], config['num_attention_heads'], config['head_dim']) == (36,64,64), 'Native checkpoint geometry/quantization differs.')
    parity = complete['portable_consumer']; first = targets[0][:256]
    require(parity == progress['portable_consumer'] and parity['rows'] == 256 and parity['same_saved_f16_vectors'] is True
        and parity['saved_vectors_sha256'] == array_sha(features[first]) and parity['classifier_device'] == 'cuda:0'
        and np.isfinite(parity['max_difference']) and 0 <= parity['max_difference'] <= parity['tolerance'] == 5e-5
        and parity['max_difference'] == complete['portable_probability_max_difference'], 'Saved F16 vectors or actual portable parity proof differs.')
    del features
    # The trainer may read these immutable arrays concurrently. It is never
    # inspected, stopped, or joined; candidate/probability paths are not opened.
    for path, expected in bindings.items(): require(digest(path) == expected, 'Extraction audit input changed: ' + path)
    report = dict(stage='actual_completed_long_augmentation_extraction_audit', audit_passed=True,
        source_sha256=digest(__file__), provenance_sha256=bindings, extraction_complete_sha256=complete_sha,
        features_sha256=FEATURE_SHA, rows_sha256=ROWS_SHA, feature_shape=[ROWS,DIMENSIONS], dtype='float16', feature_bytes=398458880,
        row_descriptor=descriptor, complete_batches_rehashed=440, all_rows_written=True, all_finite=True,
        all_rows_nonzero=zeros==0, zero_row_count=zeros, per_head_zero_row_counts=head_zeros.tolist(), minimum_feature=minimum, maximum_feature=maximum,
        selected_head_ids=selected, selected_layer_head_pairs=complete['ordered_layers_heads'],
        native_parser_source_replay_bound=True, actual_four_A40_native_MXFP4_metadata_verified=True,
        disabled_writes_hidden_bitexact=True, first256_saved_vectors_sha256=parity['saved_vectors_sha256'],
        recorded_same_saved_vector_GPU_CPU_probability_max_difference=parity['max_difference'], portable_tolerance=5e-5,
        limitation='No LM/GPU replay: raw activations and GPU parity are authenticated producer evidence, with captured saved-vector SHA independently verified. Original /workspace mount is not accessed or byte-compared.',
        model_or_candidate_probabilities_read=False, LM_loaded=False, gpu_used=False, training_run=False, formal_outputs_modified=False)
    target = Path('/tmp/feature_review_long_augmentation_extraction_audit.json'); target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(target), report_sha256=digest(target), features_sha256=FEATURE_SHA,
        zero_row_count=zeros, portable_max_difference=parity['max_difference'], source_sha256=report['source_sha256'])), flush=True)


if __name__ == '__main__': main()
