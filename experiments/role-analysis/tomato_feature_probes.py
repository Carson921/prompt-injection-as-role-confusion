"""Apply validation-selected document-split feature probes to the fixed tomato text.

HF_HOME=/root/.cache/huggingface-role-probe CUDA_VISIBLE_DEVICES=0,1,2,3 \
ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b \
.venv/bin/python experiments/role-analysis/tomato_feature_probes.py
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import search_feature_variants as run

OUT=run.OUT/'tomato'


def extract():
    import pandas as pd
    import torch
    import run_local_gptoss120b as runtime
    from feature_pipeline_runtime import install_hooks
    from utils.dataset import ReconstructableTextDataset
    selection=json.loads((run.OUT/'selection.json').read_text())['document']
    specs={'norm-20':dict(point='norm',layer=20)}
    for choice in selection.values():
        for spec in run.FEATURES[choice['feature']]:specs[f'{spec["point"]}-{spec["layer"]}']=spec
    inputs=pd.read_parquet(run.SOURCE/'tomato/inputs.parquet')
    tok=runtime.tokenizer()
    max_length=max(len(tok.encode(p,add_special_tokens=False)) for p in inputs.prompt)
    ds=ReconstructableTextDataset(inputs.prompt.tolist(),tok,max_length=max_length,prompt_ix=inputs.prompt_ix.tolist())
    if (OUT/'activations.pt').exists() and (OUT/'extraction.json').exists():
        assert json.loads((OUT/'extraction.json').read_text())['selection_sha256']==run.engine.digest(run.OUT/'selection.json')
        return
    runtime.atomic_json=lambda path,value:run.atomic_json(OUT/Path(path).name,value)
    model=runtime.load_model()
    captures={}
    def receive(name,value):captures[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    results={key:[] for key in specs}
    with torch.inference_mode():
        short=tok('A short forward validation.',return_tensors='pt').to('cuda:0')
        logits=model(**short,use_cache=False).logits.float().cpu()
        handles=install_hooks(model,specs,receive)
        hooked=model(**short,use_cache=False).logits.float().cpu()
        assert torch.equal(logits,hooked) and logits.std()>.01
        for i in range(len(ds)):
            captures.clear()
            batch=ds[i]
            output=model.model(input_ids=batch['input_ids'][None].to('cuda:0'),
                               attention_mask=batch['attention_mask'][None].to('cuda:0'),use_cache=False)
            del output
            for key,value in captures.items():
                value=value[batch['attention_mask'].bool()]
                assert torch.isfinite(value).all()
                results[key].append(value)
        for handle in handles:handle.remove()
    results={key:torch.cat(parts) for key,parts in results.items()}
    original=torch.load(run.SOURCE/'tomato/activations.pt',map_location='cpu',weights_only=True)
    assert torch.equal(results['norm-20'],original[20])
    torch.save(results,OUT/'activations.pt')
    run.atomic_json(OUT/'extraction.json',dict(selection_sha256=run.engine.digest(run.OUT/'selection.json'),
                    feature_specs=specs,tokens=len(results['norm-20']),exact_original_norm20=True,exact_logits=True,
                    inputs_sha256=run.engine.digest(run.SOURCE/'tomato/inputs.parquet')))


def project():
    import cupy as cp
    import numpy as np
    import pandas as pd
    import torch
    from utils.probes import run_projections
    states=torch.load(OUT/'activations.pt',map_location='cpu',weights_only=True)
    labels=pd.read_parquet(run.SOURCE/'tomato/labels.parquet')
    selection=json.loads((run.OUT/'selection.json').read_text())['document']
    all_results=[]
    for name,choice in selection.items():
        for variant,feature,C in [('baseline','norm-20',.005),('selected',choice['feature'],choice['C'])]:
            path=run.OUT/'models'/f'final-document-{name}-{feature}-{C:g}.pkl'
            with path.open('rb') as f:probe=pickle.load(f)
            tensor=torch.cat([states[f'{s["point"]}-{s["layer"]}'] for s in probe['feature_spec']],dim=1)
            valid=labels[labels.base_message_type.isin(probe['role_space'])]
            projected=run_projections(valid,tensor,probe)
            assert np.isfinite(projected.prob).all()
            np.testing.assert_allclose(projected.groupby('sample_ix').prob.sum(),1,atol=2e-6)
            projected=projected.merge(labels[['sample_ix','prompt_ix','prompt_key','base_message_ix',
                'base_message_type','token_in_prompt_ix','token']],on='sample_ix',validate='many_to_one')
            all_results.append(projected.assign(role_space=name,layer_ix='feature-model',model_variant=variant,
                                                feature=feature,protocol='document'))
    table=pd.concat(all_results,ignore_index=True)
    table.to_csv(OUT/'projections.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    means=table[table.target_role=='cot'].groupby(['model_variant','role_space','prompt_key','base_message_ix','base_message_type'],as_index=False).agg(
        tokens=('prob','size'),mean_cot_probability=('prob','mean'))
    means.to_csv(OUT/'cotness-by-message.csv',index=False)
    run.atomic_json(OUT/'projection-validation.json',dict(rows=len(table),all_finite=True,probability_sums_checked=True,
                    selected_without_tomato_tuning=True,protocol='document',roles_include_system=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','extract','project','plot'],default='all')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=='all':
        for stage in ['extract','project','plot']:
            env=dict(os.environ)
            if stage=='project':env['CUDA_VISIBLE_DEVICES']='3'
            subprocess.run([sys.executable,'-u',__file__,'--stage',stage],env=env,check=True)
    elif args.stage=='extract':extract()
    elif args.stage=='project':project()
    else:subprocess.run(['Rscript',str(run.engine.HERE/'plot_feature_tomato.R')],cwd=run.engine.ROOT,check=True)


if __name__=='__main__':main()
