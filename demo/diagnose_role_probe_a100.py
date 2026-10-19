"""Check layer-12 cuML convergence on the same cached demo data and split."""

import json
import os
from pathlib import Path
import shutil
import time

import nbformat

import run_role_probe_a100 as runner


BASE = runner.OUT
DIAG = BASE / 'solver-diagnostics'
DIAG.mkdir(parents=True, exist_ok=True)
shutil.copyfile(BASE / 'raw_data.json', DIAG / 'raw_data.json')
runner.OUT = DIAG
nb = runner.prepare_notebook(batch_size=8)
nb.cells = nb.cells[:17]
nb.cells[0].source = (
    'Layer-12 solver diagnostic: original cached samples, seed, prompt split, '
    'C=0.005 and max_iter=2000. Compare the original float32 probe with a '
    'float64 refit. Original demo results are preserved in the parent directory.'
)
nb.cells[11].source = nb.cells[11].source.replace('list(range(0, 24, 4))', '[12]')
nb.cells[12].source = nb.cells[12].source.replace('all_probe_hs[0]', 'all_probe_hs[12]')
nb.cells[16].source = f'''
ROLE_COMBINATION = ('system', 'user', 'cot', 'assistant')
roles_map = {{r: i for i, r in enumerate(ROLE_COMBINATION)}}
valid = probe_sample_df[probe_sample_df.role.isin(ROLE_COMBINATION)].reset_index(drop=True)
train_prompts, test_prompts = cuml.train_test_split(valid.prompt_ix.unique(), test_size=.1, random_state=seed)
train = valid[valid.prompt_ix.isin(train_prompts)]
test = valid[valid.prompt_ix.isin(test_prompts)]
x_train = cupy.asarray(all_probe_hs[12][train.sample_ix.tolist()].float().numpy())
x_test = cupy.asarray(all_probe_hs[12][test.sample_ix.tolist()].float().numpy())
y_train = cupy.asarray([roles_map[r] for r in train.role])
y_test = cupy.asarray([roles_map[r] for r in test.role])
with open({str(BASE / 'probes.pkl')!r}, 'rb') as f:
    original = next(p['probe'] for p in pickle.load(f) if p['layer_ix'] == 12)
reproduced_accuracy = float(original.score(x_test, y_test))
expected = json.loads(Path({str(BASE / 'metrics.json')!r}).read_text())['accuracy_by_layer'][3]['accuracy']
assert abs(reproduced_accuracy - expected) < 1e-5, (reproduced_accuracy, expected)
results = {{
    'layer': 12, 'seed': seed, 'C': .005, 'max_iter': 2000,
    'training_tokens': len(train), 'test_tokens': len(test),
    'original_test_accuracy': reproduced_accuracy,
    'original_train_accuracy': float(original.score(x_train, y_train)),
    'original_iterations': int(original.named_steps['clf'].n_iter_[0]),
    'max_abs_feature': float(cupy.abs(x_train).max()),
    'nonfinite_features': int((~cupy.isfinite(x_train)).sum()),
}}
print('Original probe reproduced:', results)
lr64 = cuml.linear_model.LogisticRegression(
    penalty='l2', max_iter=2000, fit_intercept=True, C=.005,
)
started = time.time()
lr64.fit(x_train.astype(cupy.float64), y_train)
results.update({{
    'float64_train_accuracy': float(lr64.score(x_train.astype(cupy.float64), y_train)),
    'float64_test_accuracy': float(lr64.score(x_test.astype(cupy.float64), y_test)),
    'float64_iterations': int(lr64.n_iter_[0]),
    'float64_fitting_seconds': time.time() - started,
}})
with open(OUT / 'probe_float64.pkl', 'wb') as f:
    pickle.dump(lr64, f)
print('Float64 comparison:', results)
del x_train, x_test, res, all_probe_hs
gc.collect()
torch.cuda.empty_cache()
templated_input = Path({str(BASE / 'example_conversation.txt')!r}).read_text()
test_dl = DataLoader(
    ReconstructableTextDataset([templated_input], tokenizer,
        max_length=len(tokenizer(templated_input, add_special_tokens=False).input_ids), prompt_ix=[0]),
    batch_size=1, shuffle=False, collate_fn=stack_collate,
)
test_outputs = run_and_export_states(model, tokenizer, run_model_return_states=run_gptoss_custom,
    dl=test_dl, layers_to_keep_acts=[12])
test_df = label_gptoss_content_roles(test_outputs['sample_df'].assign(sample_ix=lambda d: range(len(d))))
test_valid = test_df[test_df.role.notna()]
acts = test_outputs['all_hs'][:, 0, :].to(torch.float16)
features = cupy.asarray(acts[test_valid.sample_ix.tolist()].float().numpy()).astype(cupy.float64)
probs = cupy.asnumpy(lr64.predict_proba(features))
projection = pd.DataFrame(probs, columns=ROLE_COMBINATION)
projection['sample_ix'] = test_valid.sample_ix.to_numpy()
projection = projection.melt(id_vars='sample_ix', var_name='target_role', value_name='prob')
projection = projection.merge(test_df[['sample_ix', 'token', 'role']], on='sample_ix')
projection.to_csv(OUT / 'token_role_projections_float64.csv', index=False)
cached = json.loads(Path({str(BASE / 'cached_example.json')!r}).read_text())
import re
forgery = re.search(r'<SYNTHETIC_POLICY\\b[^>]*>(.*?)</SYNTHETIC_POLICY>', cached['response'], re.DOTALL).group(1)
start = templated_input.index(forgery)
end = start + len(forgery)
offsets = tokenizer(templated_input, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
cot = projection[projection.target_role == 'cot'].copy()
def region(row):
    if row.role == 'user':
        a, b = offsets[row.sample_ix]
        return 'forged_cot_in_user' if a < end and b > start else 'ordinary_user_text'
    return {{'cot': 'model_analysis', 'assistant': 'model_final'}}.get(row.role, f'{{row.role}}_context')
cot['region'] = cot.apply(region, axis=1)
summary = cot.groupby('region').agg(tokens=('prob','size'), mean_cot_probability=('prob','mean'))
summary.to_csv(OUT / 'cotness_by_region_float64.csv')
results['float64_cotness_by_region'] = summary.reset_index().to_dict('records')
(OUT / 'diagnostics.json').write_text(json.dumps(results, indent=2))
display(summary)
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
os.environ['HF_HUB_CACHE'] = '/root/.cache/huggingface-role-probe/hub'
os.environ['HF_DATASETS_CACHE'] = '/root/.cache/huggingface-role-probe/datasets'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
client = runner.LoggingClient(nb, timeout=None, kernel_name='role-probe-uv')
with client.setup_kernel(cwd=str(runner.REPO)):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            print(f'>>> DIAGNOSTIC CELL {i} START', flush=True)
            started = time.time()
            try:
                client.execute_cell(cell, i)
            finally:
                nbformat.write(nb, DIAG / 'solver-diagnostics.executed.ipynb')
            print(f'>>> DIAGNOSTIC CELL {i} DONE ({{:.1f}}s)'.format(time.time() - started), flush=True)
