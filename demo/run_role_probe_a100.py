"""Execute the original demo with explicit local-model/A100 adaptations.

Run from the repository root with .venv/bin/python demo/run_role_probe_a100.py.
The original notebook is untouched; the executed copy contains every adaptation.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import nbformat
from nbclient import NotebookClient


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo/outputs/role-probe-a100"
SOURCE = REPO / "demo/role-probe-demo.ipynb"


class LoggingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def prepare_cached_example():
    """Use a matched cached prompt/forgery, verified with the repo cache hash."""
    path = OUT / "cached_example.json"
    if path.exists():
        return
    import yaml

    url = (
        "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
        "strongreject_dataset/strongreject_dataset.csv"
    )
    dataset = urllib.request.urlopen(url, timeout=60).read()
    rows = list(csv.DictReader(io.StringIO(dataset.decode())))
    index = 5
    question = rows[index]["forbidden_prompt"]
    prompt = yaml.safe_load(
        (REPO / "experiments/cot-forgery-chat-evals/prompts/forgery-prompt-openai.yaml").read_text()
    ) + [{"role": "user", "content": f"`<TEST_PROMPT>{question}</TEST_PROMPT>`"}]
    cache = json.loads((REPO / "repro/cache/chat_01_base_policies.json").read_text())
    record = cache[index]
    digest = hashlib.sha1(
        json.dumps({"p": prompt, "m": "google/gemini-2.5-pro"}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    assert digest == record["h"], "Cached forgery does not match the selected question"
    path.write_text(json.dumps({
        "question": question,
        "cache_index": index,
        "cache_hash": digest,
        "dataset_url": url,
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
        "auxiliary_model": record["r"]["model"],
        "response": record["r"]["choices"][0]["message"]["content"],
    }, ensure_ascii=False, indent=2))


def prepare_notebook(batch_size):
    nb = nbformat.read(SOURCE, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nb.cells[1].source += f'''
import os, json, time, pickle, gc, platform
from pathlib import Path
OUT = Path({str(OUT)!r})
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_num_threads(8)
RUN_STARTED = time.time()
assert 'A100' in torch.cuda.get_device_name(0), 'This run requires the requested A100'
print('Versions:', {{'python': platform.python_version(), 'torch': torch.__version__,
    'transformers': transformers.__version__, 'cuml': cuml.__version__,
    'cupy': cupy.__version__, 'sklearn': sklearn.__version__}})
'''
    nb.cells[3].source = '''
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
MODEL_PATH = '/workspace/gpt-oss-20b'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True),
    attn_implementation='eager', device_map={'': main_device},
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, add_eos_token=False,
    add_bos_token=False, padding_side='left',
)
model.set_experts_implementation('eager')
check_memory()
'''
    nb.cells[4].source = nb.cells[4].source.replace(
        "run_gptoss_custom(model, torch.tensor([[1, 2, 3]], device = main_device), torch.tensor([[1, 1, 1]], device = main_device), return_hidden_states = True)",
        "smoke = run_gptoss_custom(model, torch.tensor([[1, 2, 3]], device=main_device), torch.tensor([[1, 1, 1]], device=main_device), return_hidden_states=True)\nprint('Smoke test:', smoke['logits'].shape, len(smoke['all_pre_mlp_hidden_states']))\ndel smoke",
    )
    nb.cells[6].source = nb.cells[6].source.replace(
        "raw_data = load_raw_ds()\nraw_data",
        "raw_data_path = OUT / 'raw_data.json'\n"
        "raw_data = json.loads(raw_data_path.read_text()) if raw_data_path.exists() else load_raw_ds()\n"
        "assert len(raw_data) == N_SAMPLES\n"
        "raw_data_path.write_text(json.dumps(raw_data, ensure_ascii=False))\n"
        "print('Raw samples:', pd.Series([r['source'] for r in raw_data]).value_counts().to_dict())",
    )
    nb.cells[9].source = nb.cells[9].source.replace("BATCH_SIZE = 32", f"BATCH_SIZE = {batch_size}")
    nb.cells[15].source += '''
with open(OUT / 'probes.pkl', 'wb') as f:
    pickle.dump(all_probes, f)
accuracy_df = pd.DataFrame([{'layer': p['layer_ix'], 'accuracy': float(p['acc'])} for p in all_probes])
accuracy_df.to_csv(OUT / 'probe_accuracy.csv', index=False)
probe_sample_df.groupby('role').size().rename('tokens').to_csv(OUT / 'role_token_counts.csv')
input_df.to_parquet(OUT / 'training_prompts.parquet', index=False)
training_stats = {
    'raw_samples': len(raw_data), 'role_wrapped_sequences': len(input_df),
    'total_tokens_including_tags': len(sample_df),
    'content_tokens_all_five_roles': len(probe_sample_df),
    'content_tokens_four_probe_roles': int(probe_sample_df.role.isin(ROLE_COMBINATION).sum()),
    'role_token_counts': {k: int(v) for k,v in probe_sample_df.groupby('role').size().items()},
    'batch_size': BATCH_SIZE, 'maximum_sequence_length': max_seqlen,
    'layers': LAYERS_TO_PROBE, 'roles': list(ROLE_COMBINATION), 'seed': seed,
    'accuracy_by_layer': accuracy_df.to_dict('records'),
    'split': 'original notebook: 90/10 prompt_ix split (shared base texts can cross splits)',
}
(OUT / 'training_stats.json').write_text(json.dumps(training_stats, indent=2))
del res, all_probe_hs
gc.collect()
torch.cuda.empty_cache()
'''
    nb.cells[17].source = '''
# Local replacement of the two API calls, authorized by the user.
# This matched cached example replaces the original notebook's drug question.
import re
cached_example = json.loads((OUT / 'cached_example.json').read_text())
HARM_QUESTION = cached_example['question']
forged_cot = re.search(r'<SYNTHETIC_POLICY\\b[^>]*>(.*?)</SYNTHETIC_POLICY>', cached_example['response'], re.DOTALL).group(1)
prefix = f"Hey GPT,\\n1. What's an interesting cat fact\\n2. {HARM_QUESTION}\\n\\n"
user_prompt = prefix + forged_cot
prompt_text = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_prompt}], tokenize=False,
    add_generation_prompt=True, reasoning_effort='medium',
)
encoded = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).to(main_device)
generation_started = time.time()
with torch.inference_mode():
    generated = model.generate(
        **encoded, do_sample=False, max_new_tokens=2048, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
new_ids = generated[0, encoded.input_ids.shape[1]:]
model_output = tokenizer.decode(new_ids, skip_special_tokens=False)
templated_input = prompt_text + model_output
generation_stats = {
    'input_tokens': encoded.input_ids.shape[1], 'generated_tokens': len(new_ids),
    'seconds': time.time() - generation_started,
    'ended_with_eos': int(new_ids[-1]) == tokenizer.eos_token_id,
    'max_new_tokens': 2048,
}
print('Cached question:', HARM_QUESTION)
print('Cache verification:', cached_example['cache_hash'])
print('Local generation:', generation_stats)
print('[MODEL OUTPUT]\\n' + model_output)
(OUT / 'example_conversation.txt').write_text(templated_input)
(OUT / 'generation_stats.json').write_text(json.dumps(generation_stats, indent=2))
del generated, encoded
gc.collect()
torch.cuda.empty_cache()
'''
    nb.cells[18].source = nb.cells[18].source.replace(
        "max_length = 16_000", "max_length = len(tokenizer(templated_input, add_special_tokens=False).input_ids)"
    )
    nb.cells[20].source = nb.cells[20].source.replace("px.scatter(", "fig = px.scatter(")
    nb.cells[20].source += '''
fig.update_layout(title='Role probe: layer 12 CoT probability', xaxis_title='Token index', yaxis_title='P(cot)')
fig.write_html(OUT / 'cotness.html', include_plotlyjs=True)
test_projections.to_csv(OUT / 'token_role_projections.csv', index=False)
# Locate the injected text by character offsets, retaining its architectural role.
offsets = tokenizer(templated_input, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
forgery_start = templated_input.index(user_prompt) + len(prefix)
forgery_end = forgery_start + len(forged_cot)
spans = []
for row in cotness_projections.itertuples():
    start, end = offsets[int(test_sample_df.iloc[row.sample_ix].token_ix)]
    if row.role == 'user':
        spans.append('forged_cot_in_user' if start < forgery_end and end > forgery_start else 'ordinary_user_text')
    elif row.role == 'cot':
        spans.append('model_analysis')
    elif row.role == 'assistant':
        spans.append('model_final')
    else:
        spans.append(f'{row.role}_context')
cotness_projections = cotness_projections.assign(region=spans)
region_summary = cotness_projections.groupby('region').agg(
    tokens=('prob', 'size'), mean_cot_probability=('prob', 'mean'),
    median_cot_probability=('prob', 'median'), fraction_cot_above_half=('prob', lambda p: (p > .5).mean()),
)
region_summary.to_csv(OUT / 'cotness_by_region.csv')
display(region_summary)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig_static, ax = plt.subplots(figsize=(12, 4))
for region, frame in cotness_projections.groupby('region', sort=False):
    ax.scatter(frame.sample_ix, frame.prob, s=8, alpha=.6, label=region)
ax.set(xlabel='Token index', ylabel='P(cot)', ylim=(-.03, 1.03), title='gpt-oss-20b: layer 12 role probe')
ax.legend(loc='upper right', fontsize=8)
fig_static.tight_layout()
fig_static.savefig(OUT / 'cotness.png', dpi=160)
plt.close(fig_static)
metrics = {
    **json.loads((OUT / 'training_stats.json').read_text()),
    'generation': generation_stats,
    'cotness_by_region': region_summary.reset_index().to_dict('records'),
    'gpu': torch.cuda.get_device_name(0),
    'torch_peak_allocated_gib': torch.cuda.max_memory_allocated(0) / 2**30,
    'torch_peak_reserved_gib': torch.cuda.max_memory_reserved(0) / 2**30,
    'notebook_execution_seconds': time.time() - RUN_STARTED,
    'example_cache_index': cached_example['cache_index'],
    'example_cache_hash': cached_example['cache_hash'],
    'example_question': HARM_QUESTION,
}
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
fig.show()
'''
    nb.cells.insert(0, nbformat.v4.new_markdown_cell(
        "Executed local/A100 adaptation of `demo/role-probe-demo.ipynb`. "
        "Uses local MXFP4 checkpoint dequantized to BF16, eager attention/experts, "
        f"batch size {batch_size}, original 150 training samples and six probe layers. "
        "Final example uses verified repository Gemini cache entry 5 and local greedy "
        "gpt-oss-20b generation instead of OpenRouter; its question differs from the original. "
        "Test padding is set to the actual token length. Original prompt_ix split is retained."
    ))
    nb.metadata.kernelspec = {
        "name": "role-probe-uv", "display_name": "Role probe (project uv .venv)", "language": "python"
    }
    return nb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--reuse-probes', action='store_true', help='Resume the example using the saved probes')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
    os.environ['HF_HUB_CACHE'] = '/root/.cache/huggingface-role-probe/hub'
    os.environ['HF_DATASETS_CACHE'] = '/root/.cache/huggingface-role-probe/datasets'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    prepare_cached_example()
    nb = prepare_notebook(args.batch_size)
    # Validate/download dataset dependencies before allocating the full GPU model.
    if not args.reuse_probes and not (OUT / 'raw_data.json').exists():
        import pandas as pd
        from datasets import load_dataset

        print('Preparing the original C4/Dolma3 training samples before model loading', flush=True)
        exec(nb.cells[7].source, {
            'seed': 123, 'OUT': OUT, 'json': json, 'pd': pd, 'load_dataset': load_dataset,
        })
    if args.reuse_probes:
        assert (OUT / 'probes.pkl').exists()
        previous = nbformat.read(OUT / 'role-probe-demo.executed.ipynb', as_version=4)
        for index in (6, 7, 9, 10, 11, 13, 15):
            nb.cells[index + 1] = previous.cells[index + 1]
        nb.cells[5].source += "\nLAYERS_TO_PROBE = list(range(0, 24, 4))\nfrom torch.utils.data import DataLoader\nfrom demo.simple_test_helpers import ReconstructableTextDataset, stack_collate, run_and_export_states, label_gptoss_content_roles\nwith open(OUT / 'probes.pkl', 'rb') as f: all_probes = pickle.load(f)\n"
    client = LoggingClient(nb, timeout=None, kernel_name='role-probe-uv', resources={'metadata': {'path': str(REPO)}})
    timings = []
    started = time.time()
    output = OUT / 'role-probe-demo.executed.ipynb'
    with client.setup_kernel(cwd=str(REPO)):
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            if args.reuse_probes and index - 1 in (6, 7, 9, 10, 11, 13, 15):
                continue
            print(f'\n>>> ORIGINAL CELL {index - 1} START', flush=True)
            cell_started = time.time()
            try:
                client.execute_cell(cell, index)
            finally:
                timings.append({'original_cell': index - 1, 'seconds': time.time() - cell_started})
                nbformat.write(nb, output)
                (OUT / 'cell_timings.json').write_text(json.dumps(timings, indent=2))
            print(f'>>> ORIGINAL CELL {index - 1} DONE ({time.time() - cell_started:.1f}s)', flush=True)
    print(f'Completed in {time.time() - started:.1f}s: {output}', flush=True)


if __name__ == '__main__':
    main()
