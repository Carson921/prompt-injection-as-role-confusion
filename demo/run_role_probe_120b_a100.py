"""Run the cached role-probe demo on local GPT-OSS-120B in MXFP4 on an A100.

The 20B run used BF16/eager experts. This run keeps MXFP4 expert weights to fit
80GB, and explicitly records that backend difference in its notebook and report.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time

import nbformat
import pynvml

import run_role_probe_a100 as base


REPO = base.REPO
OUT = REPO / 'demo/outputs/role-probe-120b-a100'
PREVIOUS = REPO / 'demo/outputs/role-probe-a100'
LAYERS = [0, 4, 8, 12, 16, 18, 20, 24, 28, 32]
TEST_LAYER = 18
KERNEL_REVISION = '9fbc1b48e710ac8abf18ea2650ee690d52dc3859'


def prepare_notebook(batch_size):
    base.OUT = OUT
    nb = base.prepare_notebook(batch_size)
    nb.cells[0].source = (
        'GPT-OSS-120B/A100 adaptation. Uses the exact cached 150 C4/Dolma3 texts, '
        'seed=123, original prompt_ix split, C=0.005, and cached forgery example '
        'from the preceding 20B run. MXFP4 Triton expert kernels replace the '
        '20B run\'s BF16/eager experts to fit 80GB. Attention stays eager. '
        f'Probe layers: {LAYERS}; example projection uses middle layer {TEST_LAYER}. '
        'Layer numbers are zero-based. All adaptations are included below.'
    )
    nb.cells[4].source = f'''
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from transformers.integrations import hub_kernels
from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
MODEL_PATH = '/workspace/gpt-oss-120b'
# Pin the downloaded kernel source without changing the installed transformers.
original_get_kernel = hub_kernels.get_kernel
def pinned_get_kernel(name, *args, **kwargs):
    if name == 'kernels-community/gpt-oss-triton-kernels':
        kwargs['revision'] = {KERNEL_REVISION!r}
    return original_get_kernel(name, *args, **kwargs)
hub_kernels.get_kernel = pinned_get_kernel
# A fallback to full BF16 would exceed this GPU's memory: fail before loading it.
original_validate = Mxfp4HfQuantizer.validate_environment
def require_mxfp4(self, *args, **kwargs):
    original_validate(self, *args, **kwargs)
    if self.quantization_config.dequantize:
        raise RuntimeError('MXFP4 kernels are unavailable; full BF16 cannot fit 80GB')
Mxfp4HfQuantizer.validate_environment = require_mxfp4
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=False),
    attn_implementation='eager', device_map={{'': main_device}}, use_kernels=False,
).eval()
assert model.config.num_hidden_layers == 36
assert type(model.model.layers[0].mlp.experts).__name__ == 'Mxfp4GptOssExperts'
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, add_eos_token=False,
    add_bos_token=False, padding_side='left',
)
print('Quantized expert backend:', type(model.model.layers[0].mlp.experts).__name__)
check_memory()
'''
    nb.cells[11].source = nb.cells[11].source.replace('list(range(0, 24, 4))', repr(LAYERS))
    # Store the token map and only the selected CPU activations for safe resumption.
    nb.cells[11].source += '''
res['sample_df'].to_parquet(OUT / 'training_tokens.parquet', index=False)
torch.save(res['all_hs'], OUT / 'training_hidden_states.pt')
print('Activation checkpoint saved:', res['all_hs'].shape)
'''
    nb.cells[16].source = nb.cells[16].source.replace(
        'del res, all_probe_hs',
        "del res, all_probe_hs\ncupy.get_default_memory_pool().free_all_blocks()",
    )
    # A solver diagnostic is cheap while the same activations are still available.
    marker = "with open(OUT / 'probes.pkl', 'wb') as f:"
    position = nb.cells[16].source.index(marker)
    nb.cells[16].source = nb.cells[16].source[:position] + f'''
selected = probe_sample_df[probe_sample_df.role.isin(ROLE_COMBINATION)].reset_index(drop=True)
train_ids, test_ids = cuml.train_test_split(selected.prompt_ix.unique(), test_size=.1, random_state=seed)
train_rows = selected[selected.prompt_ix.isin(train_ids)]
test_rows = selected[selected.prompt_ix.isin(test_ids)]
role_map = {{r: i for i, r in enumerate(ROLE_COMBINATION)}}
train_y = cupy.asarray([role_map[r] for r in train_rows.role])
test_y = cupy.asarray([role_map[r] for r in test_rows.role])
train_x = cupy.asarray(all_probe_hs[{TEST_LAYER}][train_rows.sample_ix.tolist()].float().numpy()).astype(cupy.float64)
test_x = cupy.asarray(all_probe_hs[{TEST_LAYER}][test_rows.sample_ix.tolist()].float().numpy()).astype(cupy.float64)
diagnostic_probe = cuml.linear_model.LogisticRegression(penalty='l2', max_iter=2000, fit_intercept=True, C=.005)
diagnostic_probe.fit(train_x, train_y)
solver_diagnostic = {{
    'layer': {TEST_LAYER}, 'precision': 'float64',
    'train_tokens': len(train_rows), 'test_tokens': len(test_rows),
    'test_accuracy': float(diagnostic_probe.score(test_x, test_y)),
    'iterations': int(diagnostic_probe.n_iter_[0]),
}}
print('Middle-layer float64 solver diagnostic:', solver_diagnostic)
with open(OUT / 'middle_probe_float64.pkl', 'wb') as f:
    pickle.dump(diagnostic_probe, f)
(OUT / 'solver_diagnostic.json').write_text(json.dumps(solver_diagnostic, indent=2))
del train_x, test_x, train_y, test_y
''' + nb.cells[16].source[position:]
    nb.cells[20].source = nb.cells[20].source.replace('TEST_LAYER_IX = 12', f'TEST_LAYER_IX = {TEST_LAYER}')
    nb.cells[21].source = nb.cells[21].source.replace('layer 12', f'layer {TEST_LAYER}').replace('gpt-oss-20b', 'gpt-oss-120b')
    nb.cells[21].source += f'''
metrics.update({{
    'model_path': MODEL_PATH, 'model_name': 'gpt-oss-120b',
    'weight_format': 'MXFP4', 'attention_implementation': 'eager',
    'expert_backend': 'Triton MXFP4', 'kernel_revision': {KERNEL_REVISION!r},
    'test_layer': TEST_LAYER_IX,
    'solver_diagnostic': solver_diagnostic,
    'backend_difference_from_20b': '20B used BF16/eager experts; 120B keeps MXFP4/Triton experts',
}})
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
# Evaluate the saved float64 probe on exactly the same local conversation.
valid_example = test_sample_df[test_sample_df.role.notna()]
example_x64 = cupy.asarray(test_hs[TEST_LAYER_IX][valid_example.sample_ix.tolist()].float().numpy()).astype(cupy.float64)
example_proba64 = cupy.asnumpy(diagnostic_probe.predict_proba(example_x64))
projection64 = pd.DataFrame(example_proba64, columns=ROLE_COMBINATION)
projection64['sample_ix'] = valid_example.sample_ix.to_numpy()
projection64 = projection64.melt(id_vars='sample_ix', var_name='target_role', value_name='prob')
projection64 = projection64.merge(valid_example[['sample_ix','role','token']], on='sample_ix')
projection64.to_csv(OUT / 'token_role_projections_float64.csv', index=False)
cot64 = projection64[projection64.target_role == 'cot'].merge(cotness_projections[['sample_ix','region']], on='sample_ix')
cot64.groupby('region').agg(tokens=('prob','size'), mean_cot_probability=('prob','mean')).to_csv(OUT / 'cotness_by_region_float64.csv')
'''
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--resume-activations', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ('raw_data.json', 'cached_example.json'):
        if not (OUT / filename).exists():
            shutil.copyfile(PREVIOUS / filename, OUT / filename)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
    os.environ['HF_HUB_CACHE'] = '/root/.cache/huggingface-role-probe/hub'
    os.environ['HF_DATASETS_CACHE'] = '/root/.cache/huggingface-role-probe/datasets'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    nb = prepare_notebook(args.batch_size)
    if args.resume_activations:
        assert (OUT / 'training_hidden_states.pt').exists()
        nb.cells[11].source = f'''
LAYERS_TO_PROBE = {LAYERS!r}
from demo.simple_test_helpers import run_and_export_states
res = {{'sample_df': pd.read_parquet(OUT / 'training_tokens.parquet'),
       'all_hs': torch.load(OUT / 'training_hidden_states.pt', map_location='cpu', weights_only=True)}}
'''
    client = base.LoggingClient(nb, timeout=None, kernel_name='role-probe-uv')
    stop = threading.Event()
    samples = []
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def monitor():
        while not stop.wait(1):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            samples.append({'elapsed_seconds': time.time() - started, 'used_gib': memory.used / 2**30})
    started = time.time()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    timings = []
    try:
        with client.setup_kernel(cwd=str(REPO)):
            for index, cell in enumerate(nb.cells):
                if cell.cell_type != 'code':
                    continue
                print(f'>>> ORIGINAL CELL {index - 1} START', flush=True)
                cell_started = time.time()
                try:
                    client.execute_cell(cell, index)
                finally:
                    timings.append({'original_cell': index - 1, 'seconds': time.time() - cell_started})
                    nbformat.write(nb, OUT / 'role-probe-demo.executed.ipynb')
                    (OUT / 'cell_timings.json').write_text(json.dumps(timings, indent=2))
                print(f'>>> ORIGINAL CELL {index - 1} DONE ({time.time() - cell_started:.1f}s)', flush=True)
    finally:
        stop.set()
        thread.join(timeout=2)
        (OUT / 'gpu_memory_samples.json').write_text(json.dumps(samples))
        memory_stats = {'nvml_peak_used_gib': max((s['used_gib'] for s in samples), default=0),
                        'nvml_total_gib': pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30,
                        'wall_seconds': time.time() - started}
        (OUT / 'memory_stats.json').write_text(json.dumps(memory_stats, indent=2))
        print('Memory measurements:', memory_stats, flush=True)
        pynvml.nvmlShutdown()
    print('Completed:', OUT, flush=True)


if __name__ == '__main__':
    main()
