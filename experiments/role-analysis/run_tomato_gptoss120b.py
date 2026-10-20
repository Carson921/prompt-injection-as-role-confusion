"""Apply the frozen MXFP4 120B model and all 72 probes to the original tomato text.

Run from the repository root, with the training environment and four visible GPUs:
  HF_HOME=/root/.cache/huggingface-role-probe \
    ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b \
    .venv/bin/python experiments/role-analysis/run_tomato_gptoss120b.py

The three prompts and token/message assignments follow notebook 02 exactly.
The original R notebook renders both layer 12 (the 20B plotting setting) and
layer 16 (an available middle layer of the 36-layer 120B model).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import run_local_gptoss120b as runtime

HERE, ROOT = runtime.HERE, runtime.ROOT
OUT = runtime.OUT / 'tomato'
PROJECTIONS = HERE / 'outputs/probe-projections/tomato-role-projections-gptoss-120b.csv.gz'
atomic_json = runtime.atomic_json


def sha256(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def status(stage, **fields):
    value = dict(stage=stage, unix_time=time.time(), **fields)
    atomic_json(OUT / 'status.json', value)
    print(json.dumps(value), flush=True)


def prepare():
    import pandas as pd
    import yaml
    from utils.dataset import ReconstructableTextDataset
    from utils.role_templates import load_chat_template
    from utils.substring_assignments import flag_message_types

    OUT.mkdir(parents=True, exist_ok=True)
    tok = runtime.tokenizer()
    # Notebook 02 has already installed this template before its tomato cells.
    tok.chat_template = load_chat_template(str(ROOT / 'utils/chat_templates'), 'gptoss-120b')
    ns = dict(tokenizer=tok, ws=str(ROOT), model_prefix='gptoss-120b', yaml=yaml, pd=pd)
    cells = json.loads((HERE / '02-train-role-probes.ipynb').read_text())['cells']
    for i in (48, 49):
        source = ''.join(cells[i]['source']).replace("print(tomato_prompts['proper_tags'])", '')
        exec(compile(source, f'02-train-role-probes.ipynb:cell{i}', 'exec'), ns)
    inputs = ns['tomato_input_df']
    lengths = [len(tok.encode(p, add_special_tokens=False)) for p in inputs.prompt]
    ds = ReconstructableTextDataset(inputs.prompt.tolist(), tok, max_length=max(lengths),
                                   prompt_ix=inputs.prompt_ix.tolist())
    records = []
    for i in range(len(ds)):
        for j in ds.attention_mask[i].nonzero().flatten().tolist():
            records.append((i, j, int(ds.input_ids[i, j]), ds.original_tokens[i][j]))
    tokens = pd.DataFrame(records, columns=['prompt_ix', 'token_ix', 'token_id', 'token'])
    tokens = flag_message_types(tokens, ns['base_messages'])
    tokens['sample_ix'] = range(len(tokens))
    tokens['token_in_prompt_ix'] = tokens.groupby('prompt_ix').cumcount()
    labeled = tokens.merge(inputs, on='prompt_ix', validate='many_to_one').merge(
        pd.DataFrame(dict(base_message_ix=range(7), base_message_type=ns['base_message_types'])),
        on='base_message_ix', how='inner', validate='many_to_one')
    assert len(tokens) == sum(lengths)
    for i, prompt in enumerate(inputs.prompt):
        assert ''.join(tokens[tokens.prompt_ix == i].token) == prompt
        assert set(labeled[labeled.prompt_ix == i].base_message_ix) == set(range(7))
    # A one-to-one comparison against the existing 20B experiment makes the
    # formatting and Unicode boundary handling independently reviewable.
    old = pd.read_csv(HERE / 'outputs/probe-projections/tomato-role-projections-gptoss-20b.csv.gz',
                      keep_default_na=False)
    cols = ['sample_ix', 'prompt_ix', 'prompt_key', 'base_message_ix',
            'base_message_type', 'token_in_prompt_ix', 'token']
    old_tokens = old[cols].drop_duplicates().sort_values('sample_ix').reset_index(drop=True)
    new_tokens = labeled[cols].sort_values('sample_ix').reset_index(drop=True)
    pd.testing.assert_frame_equal(old_tokens, new_tokens, check_dtype=False)
    inputs.to_parquet(OUT / 'inputs.parquet', index=False)
    tokens.to_parquet(OUT / 'tokens.parquet', index=False)
    labeled.to_parquet(OUT / 'labels.parquet', index=False)
    atomic_json(OUT / 'inputs.json', inputs.to_dict('records'))
    manifest = dict(model='gptoss-120b', weight_format='MXFP4',
                    capture_point='post_attention_layernorm output', stored_activation_dtype='float16',
                    layers=runtime.LAYERS, n_probes=72,
                    text_source='config/tomato.yaml:gptoss-20b (the exact fixed 20B example)',
                    generated_new_conversation=False, formats=inputs.prompt_key.tolist(),
                    lengths=lengths, tokens=len(tokens), content_tokens=len(labeled),
                    original_20b_token_and_message_mapping_identical=True,
                    tomato_config_sha256=sha256(HERE / 'config/tomato.yaml'),
                    probes_sha256=sha256(HERE / 'outputs/probes/gptoss-120b.pkl'),
                    plotting_role_space='suca', plotting_layers=[12, 16],
                    layer_12_reason='same zero-based index as the 20B tomato notebook',
                    layer_16_reason='available middle layer of the 36-layer 120B model')
    atomic_json(OUT / 'manifest.json', manifest)
    return tok, ds, manifest


def extract():
    import torch
    tok, ds, manifest = prepare()
    if (OUT / 'activations.pt').exists() and (OUT / 'extraction.json').exists():
        old = json.loads((OUT / 'extraction.json').read_text())
        assert old['manifest_sha256'] == sha256(OUT / 'manifest.json')
        status('activations_cached')
        return
    # Keep the completed training status and backend records untouched.
    runtime.status = status
    runtime.atomic_json = lambda path, value: atomic_json(OUT / Path(path).name, value)
    model = runtime.load_model()
    captures = {}
    handles = []
    def capture(index):
        def hook(module, args, output):
            captures[index] = output.detach().to('cpu', torch.float16).reshape(-1, output.shape[-1])
        return hook
    chunks = {i: [] for i in runtime.LAYERS}
    started = time.time()
    with torch.inference_mode():
        short = tok('A short forward validation.', return_tensors='pt').to('cuda:0')
        ref = model(**short, use_cache=False).logits.float().cpu()
        handles = [model.model.layers[i].post_attention_layernorm.register_forward_hook(capture(i))
                   for i in runtime.LAYERS]
        hooked = model(**short, use_cache=False).logits.float().cpu()
        assert torch.equal(ref, hooked) and torch.isfinite(ref).all() and ref.std() > .01
        atomic_json(OUT / 'forward_validation.json', dict(exact_logits=True,
                    max_absolute_difference=float((ref-hooked).abs().max()), logit_std=float(ref.std())))
        del ref, hooked
        for i in range(len(ds)):
            captures.clear()
            batch = ds[i]
            result = model.model(input_ids=batch['input_ids'][None].to('cuda:0'),
                                 attention_mask=batch['attention_mask'][None].to('cuda:0'),
                                 use_cache=False, return_dict=True)
            del result
            mask = batch['attention_mask'].bool()
            assert set(captures) == set(runtime.LAYERS)
            for layer in runtime.LAYERS:
                values = captures[layer][mask]
                assert torch.isfinite(values).all()
                chunks[layer].append(values)
            status('extracting', prompt=i+1, total_prompts=len(ds), seconds=time.time()-started)
    for handle in handles:
        handle.remove()
    activations = {i: torch.cat(chunks[i]) for i in runtime.LAYERS}
    assert all(v.shape == (manifest['tokens'], 2880) for v in activations.values())
    torch.save(activations, OUT / 'activations.pt')
    atomic_json(OUT / 'extraction.json', dict(manifest_sha256=sha256(OUT / 'manifest.json'),
                tokens=manifest['tokens'], layers=runtime.LAYERS, seconds=time.time()-started,
                peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(4)]))
    status('extraction_complete', tokens=manifest['tokens'])


def project():
    import numpy as np
    import pandas as pd
    import torch
    from utils.probes import run_projections
    labels = pd.read_parquet(OUT / 'labels.parquet')
    activations = torch.load(OUT / 'activations.pt', map_location='cpu', weights_only=True)
    with (HERE / 'outputs/probes/gptoss-120b.pkl').open('rb') as f:
        probes = pickle.load(f)
    assert len(probes) == 72
    results = []
    for probe in probes:
        valid = labels[labels.base_message_type.isin(probe['role_space'])]
        df = run_projections(valid, activations[probe['layer_ix']], probe)
        assert len(df) == len(valid) * len(probe['role_space'])
        assert np.isfinite(df.prob).all() and df.prob.between(0, 1).all()
        np.testing.assert_allclose(df.groupby('sample_ix').prob.sum(), 1, atol=2e-6)
        df = df.assign(layer_ix=probe['layer_ix'], role_space=''.join(r[0] for r in probe['role_space']))
        results.append(df)
    joined = pd.concat(results, ignore_index=True).merge(labels[
        ['sample_ix', 'prompt_ix', 'prompt_key', 'base_message_ix', 'base_message_type',
         'token_in_prompt_ix', 'token']], on='sample_ix', validate='many_to_one')
    assert not joined.duplicated(['sample_ix', 'layer_ix', 'role_space', 'target_role']).any()
    PROJECTIONS.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(PROJECTIONS, index=False, compression=dict(method='gzip', mtime=0))
    selected = joined[(joined.role_space == 'suca') & joined.layer_ix.isin([12, 16])]
    selected.to_csv(OUT / 'selected_projections.csv', index=False)
    means = selected[selected.target_role == 'cot'].groupby(
        ['layer_ix', 'prompt_key', 'base_message_ix', 'base_message_type'], as_index=False).agg(
            tokens=('prob', 'size'), mean_cot_probability=('prob', 'mean'))
    means.to_csv(OUT / 'cotness-by-message.csv', index=False)
    atomic_json(OUT / 'projection_validation.json', dict(n_probes=len(probes),
                rows=len(joined), formats=sorted(joined.prompt_key.unique()),
                all_finite=True, probability_sums_checked=True, unique_token_role_layer_keys=True,
                original_run_projections_interface=True, projections_sha256=sha256(PROJECTIONS)))
    status('projections_complete', rows=len(joined))


def finalize():
    from PIL import Image
    plots = runtime.OUT / 'plots'
    artifacts = []
    for layer in (12, 16):
        directory = plots / f'layer-{layer}'
        pngs, pdfs = sorted(directory.glob('*.png')), sorted(directory.glob('*.pdf'))
        assert len(pngs) == len(pdfs) == 10
        assert {p.stem for p in pngs} == {p.stem for p in pdfs}
        for path in pngs:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                assert min(img.size) >= 800
        for path in pdfs:
            with path.open('rb') as f:
                assert f.read(5) == b'%PDF-'
        artifacts.extend(dict(file=str(p.relative_to(runtime.OUT)), sha256=sha256(p)) for p in pngs + pdfs)
    projection_check = json.loads((OUT / 'projection_validation.json').read_text())
    assert projection_check['projections_sha256'] == sha256(PROJECTIONS)
    validation = dict(complete=True, n_png=20, n_pdf=20, n_probes=72,
                      projection_rows=projection_check['rows'], plotting_layers=[12, 16],
                      plotting_role_space='suca', artifacts=artifacts,
                      source_sha256={p.name: sha256(p) for p in [Path(__file__),
                          HERE / 'plot_tomato_gptoss120b.R', HERE / '04-tomato-probe-results.ipynb']})
    atomic_json(OUT / 'complete.json', validation)
    status('complete', n_png=20, n_pdf=20, projection_rows=projection_check['rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'extract', 'project', 'plot'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all':
        for stage in ['extract', 'project', 'plot']:
            env = dict(os.environ)
            # cuML workers need only one GPU, after model extraction has exited.
            if stage == 'project':
                env['CUDA_VISIBLE_DEVICES'] = '3'
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', stage],
                           env=env, check=True)
    elif args.stage == 'extract':
        extract()
    elif args.stage == 'project':
        project()
    else:
        subprocess.run(['Rscript', str(HERE / 'plot_tomato_gptoss120b.R')], cwd=ROOT, check=True)
        finalize()


if __name__ == '__main__':
    main()
