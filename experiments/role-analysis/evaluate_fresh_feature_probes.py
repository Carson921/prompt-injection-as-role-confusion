"""Confirm frozen validation-selected feature probes on 100 new C4 documents.

Inputs/downloads are local caches; published artifacts contain hashes and metrics.
The original 249 training/validation/test document bodies are excluded by hash.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import search_feature_variants as run

OUT=run.OUT/'fresh'
CACHE=OUT/'cache'
ACTS=Path('/dev/shm/role-analysis-feature-fresh/activations.pt')
REVISION='1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
SHARD='en/c4-validation.00000-of-00008.json.gz'
SHA256='1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f'
URL=f'https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{SHARD}'
atomic_json=run.atomic_json


def prepare():
    import pandas as pd
    import requests
    import run_local_gptoss120b as runtime
    from utils.role_templates import render_single_message
    from utils.dataset import ReconstructableTextDataset
    from utils.role_assignments import label_content_roles
    CACHE.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'evaluation-plan.json',dict(created_unix_time=time.time(),documents=100,
                protocol='document',variants=['norm-20 C=.005 baseline','validation-selected feature probe'],
                source_revision=REVISION,source_shard=SHARD,
                rule='First 100 novel nonempty truncated document bodies; exclude every original document body. No parameter or feature selection using this evaluation.'))
    archive=CACHE/'c4-validation.json.gz'
    if not archive.exists():
        temporary=archive.with_suffix('.part')
        with requests.get(URL,stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for chunk in response.iter_content(2**20):f.write(chunk)
        assert run.engine.digest(temporary)==SHA256
        temporary.replace(archive)
    assert run.engine.digest(archive)==SHA256
    tok=runtime.tokenizer()
    old=pd.read_parquet(run.SOURCE/'training_inputs.parquet')
    digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
    excluded={digest(s) for s in old.question.unique()}
    seen=set(excluded)
    records=[]
    keys=[]
    roles=['system','user','tool','cot','assistant']
    with gzip.open(archive,'rt',encoding='utf-8') as f:
        for line_ix,line in enumerate(f):
            row=json.loads(line)
            ids=tok.encode(row['text'],add_special_tokens=False)[:1024]
            if not ids:continue
            body=tok.decode(ids,skip_special_tokens=False)
            key=digest(body)
            if key in seen:continue
            seen.add(key)
            question_ix=len(keys)
            keys.append(dict(question_ix=question_ix,shard_line=line_ix,truncated_body_sha256=key,
                             full_body_sha256=digest(row['text']),tokens=len(ids),source_url=row.get('url')))
            for role in roles:
                records.append(dict(question_ix=question_ix,role=role,prompt_ix=len(records),
                                    prompt=render_single_message('gptoss-120b',role,body)))
            if len(keys)==100:break
    assert len(keys)==100 and not excluded&{r['truncated_body_sha256'] for r in keys}
    inputs=pd.DataFrame(records)
    max_length=max(len(tok.encode(p,add_special_tokens=False)) for p in inputs.prompt)
    ds=ReconstructableTextDataset(inputs.prompt.tolist(),tok,max_length=max_length,prompt_ix=inputs.prompt_ix.tolist())
    tokens=[]
    for i in range(len(ds)):
        for j in ds.attention_mask[i].nonzero().flatten().tolist():tokens.append((i,j,int(ds.input_ids[i,j]),ds.original_tokens[i][j]))
    tokens=pd.DataFrame(tokens,columns=['prompt_ix','token_ix','token_id','token'])
    tokens['sample_ix']=range(len(tokens))
    labels=label_content_roles('gptoss-120b',tokens).merge(inputs[['prompt_ix','role','question_ix']].rename(columns={'role':'target_role'}),on='prompt_ix')
    labels=labels[labels.is_content&(labels.role==labels.target_role)].copy()
    counts=labels.groupby('role').size().to_dict()
    assert set(counts)==set(roles) and len(set(counts.values()))==1
    inputs.to_parquet(CACHE/'inputs.parquet',index=False)
    labels.to_parquet(CACHE/'labels.parquet',index=False)
    atomic_json(OUT/'document-keys.json',keys)
    atomic_json(OUT/'source.json',dict(dataset='allenai/c4',revision=REVISION,url=URL,shard_sha256=SHA256,
                source='C4 English validation; new document bodies relative to all 249 original texts',
                n_documents=100,n_prompts=500,tokens=len(tokens),content_tokens_by_role=counts,max_length=max_length,
                no_original_body_overlap=True,original_dataset_documents=249,cache_texts_not_published=True))
    print(json.dumps(dict(stage='fresh_data_ready',documents=100,tokens=len(tokens))),flush=True)


def extract():
    import pandas as pd
    import torch
    import run_local_gptoss120b as runtime
    from torch.utils.data import DataLoader
    from utils.dataset import ReconstructableTextDataset,stack_collate
    from feature_pipeline_runtime import install_hooks,feature_batches
    meta=json.loads((OUT/'source.json').read_text())
    selection=json.loads((run.OUT/'selection.json').read_text())['document']
    specs={'norm-20':dict(point='norm',layer=20)}
    for choice in selection.values():
        for s in run.FEATURES[choice['feature']]:specs[f'{s["point"]}-{s["layer"]}']=s
    if ACTS.exists() and (OUT/'extraction.json').exists():
        saved=json.loads((OUT/'extraction.json').read_text())
        assert saved['selection_sha256']==run.engine.digest(run.OUT/'selection.json')
        assert saved['inputs_sha256']==run.engine.digest(CACHE/'inputs.parquet')
        return
    inputs=pd.read_parquet(CACHE/'inputs.parquet')
    tok=runtime.tokenizer()
    ds=ReconstructableTextDataset(inputs.prompt.tolist(),tok,max_length=meta['max_length'],prompt_ix=inputs.prompt_ix.tolist())
    assert int(ds.attention_mask.sum())==meta['tokens']
    runtime.atomic_json=lambda path,value:atomic_json(OUT/Path(path).name,value)
    model=runtime.load_model()
    captures={}
    def receive(name,value):captures[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    maps={key:torch.empty((meta['tokens'],2880),dtype=torch.float16) for key in specs}
    with torch.inference_mode():
        short=tok('A short forward validation.',return_tensors='pt').to('cuda:0')
        logits=model(**short,use_cache=False).logits.float().cpu()
        handles=install_hooks(model,specs,receive)
        hooked=model(**short,use_cache=False).logits.float().cpu()
        assert torch.equal(logits,hooked) and logits.std()>.01
        validation=dict(captures)
        for handle in handles:handle.remove()
        dl=DataLoader(ds,batch_size=8,shuffle=False,collate_fn=stack_collate)
        cursor=0
        started=time.time()
        for bi,batch,states in feature_batches(model,dl,specs,0,(short,validation),lambda r:atomic_json(OUT/'pipeline-validation.json',r)):
            mask=batch['attention_mask'].reshape(-1).bool()
            count=int(mask.sum())
            for key in specs:
                values=states[key][mask]
                assert torch.isfinite(values).all()
                maps[key][cursor:cursor+count]=values
            cursor+=count
            states.clear()
            if (bi+1)%8==0 or bi+1==len(dl):print(json.dumps(dict(stage='fresh_extracting',batch=bi+1,batches=len(dl),seconds=time.time()-started)),flush=True)
        assert cursor==meta['tokens']
    ACTS.parent.mkdir(exist_ok=True)
    torch.save(maps,ACTS)
    atomic_json(OUT/'extraction.json',dict(tokens=cursor,features=specs,seconds=time.time()-started,
                selection_sha256=run.engine.digest(run.OUT/'selection.json'),exact_logits=True,
                inputs_sha256=run.engine.digest(CACHE/'inputs.parquet'),
                model_sha256={path.name:run.engine.digest(path) for path in (run.OUT/'models').glob('*.pkl')}))


def evaluate():
    import cupy as cp
    import numpy as np
    import pandas as pd
    import torch
    states=torch.load(ACTS,map_location='cpu',weights_only=True)
    extraction=json.loads((OUT/'extraction.json').read_text())
    assert extraction['selection_sha256']==run.engine.digest(run.OUT/'selection.json')
    assert extraction['inputs_sha256']==run.engine.digest(CACHE/'inputs.parquet')
    assert all(run.engine.digest(run.OUT/'models'/name)==sha for name,sha in extraction['model_sha256'].items())
    labels=pd.read_parquet(CACHE/'labels.parquet')
    choices=json.loads((run.OUT/'selection.json').read_text())['document']
    rows=[]
    doc_tables=[]
    for name,choice in choices.items():
        roles=run.engine.ROLE_SPACES[name]
        valid=labels[labels.role.isin(roles)]
        truth=valid.role.map({r:i for i,r in enumerate(roles)}).to_numpy()
        corrects={}
        for variant,feature,C in [('baseline','norm-20',.005),('selected',choice['feature'],choice['C'])]:
            with (run.OUT/'models'/f'final-document-{name}-{feature}-{C:g}.pkl').open('rb') as f:probe=pickle.load(f)
            probabilities=[]
            indices=valid.sample_ix.tolist()
            for start in range(0,len(valid),8192):
                ids=indices[start:start+8192]
                tensor=torch.cat([states[f'{s["point"]}-{s["layer"]}'][ids] for s in probe['feature_spec']],dim=1).float()
                p=probe['probe'].predict_proba(cp.asarray(tensor.numpy()))
                assert cp.isfinite(p).all() and cp.allclose(p.sum(axis=1),1,atol=2e-6)
                probabilities.append(cp.asnumpy(p))
            p=np.concatenate(probabilities)
            pred=p.argmax(axis=1)
            correct=pred==truth
            corrects[variant]=correct
            k=len(roles)
            matrix=np.bincount(truth*k+pred,minlength=k*k).reshape(k,k)
            recall=matrix.diagonal()/matrix.sum(axis=1)
            record=dict(role_space=name,variant=variant,feature=feature,C=C,accuracy=float(correct.mean()),
                        nll=float(-np.log(np.clip(p[np.arange(len(truth)),truth],1e-15,1)).mean()),
                        recall=dict(zip(roles,map(float,recall))),confusion=matrix.tolist(),tokens=len(valid))
            rows.append(record)
            atomic_json(OUT/f'{name}-{variant}.json',record)
        docs=valid.assign(baseline=corrects['baseline'],selected=corrects['selected']).groupby('question_ix').agg(
            tokens=('sample_ix','size'),baseline_correct=('baseline','sum'),selected_correct=('selected','sum')).reset_index()
        doc_tables.append(docs.assign(role_space=name))
        rng=np.random.RandomState(20261020)
        ix=rng.randint(0,len(docs),size=(2000,len(docs)))
        gains=(docs.selected_correct.to_numpy()-docs.baseline_correct.to_numpy())[ix].sum(axis=1)/docs.tokens.to_numpy()[ix].sum(axis=1)*100
        lo,hi=map(float,np.quantile(gains,[.025,.975]))
        atomic_json(OUT/f'{name}-comparison.json',dict(role_space=name,
                    gain_percentage_points=float(100*(corrects['selected'].mean()-corrects['baseline'].mean())),
                    gain_ci95_lower=lo,gain_ci95_upper=hi,test_documents=len(docs)))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['recall','confusion']}|{f'recall_{k}':v for k,v in r['recall'].items()} for r in rows]).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(doc_tables,ignore_index=True).to_csv(OUT/'document-counts.csv',index=False)
    atomic_json(OUT/'complete.json',dict(new_documents=100,no_original_body_overlap=True,protocol='document',
                all_finite=True,probability_sums_checked=True,selection_unchanged=True,
                selection_sha256=run.engine.digest(run.OUT/'selection.json'),results=rows))
    print(json.dumps(rows),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','extract','evaluate'],default='all')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=='all':
        for stage in ['prepare','extract','evaluate']:
            env=dict(os.environ)
            if stage=='evaluate':env['CUDA_VISIBLE_DEVICES']='3'
            subprocess.run([sys.executable,'-u',__file__,'--stage',stage],env=env,check=True)
    elif args.stage=='prepare':prepare()
    elif args.stage=='extract':extract()
    else:evaluate()


if __name__=='__main__':main()
