"""Expand the validation-selected attention-output pilot to all fit documents.

Only the original fit/validation partition is extracted here. The published
48-candidate residual benchmark remains unchanged. No independent test is read.
"""
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

import systemness_components as components
import systemness_data as data
import systemness_probes as probes

OUT = data.runtime.OUT / 'systemness-attention'
CACHE = data.CACHE / 'attention'
PILOT = components.OUT
SPECS = {f'attention-{layer}':dict(point='attention',layer=layer) for layer in [16,20,24]}


def initialize():
    rows, _ = probes.development()
    assert set(rows.split) == {'fit','validation'} and set(rows.wrapper) == {'native'}
    assert rows[rows.split=='fit'].document_id.nunique() == 1200
    rows_path = CACHE/'rows.parquet'
    rows.to_parquet(rows_path,index=False)
    signature = dict(rows_sha256=data.digest(rows_path),features=SPECS,
                     pilot_decision_sha256=data.digest(PILOT/'promotion-decision.json'))
    cache_path = CACHE/'features.f16'
    initialized_path = CACHE/'initialized.json'
    if initialized_path.exists():
        previous=json.loads(initialized_path.read_text())
        assert previous['signature']==signature and cache_path.stat().st_size==len(rows)*8640*2
        return rows,np.memmap(cache_path,mode='r+',dtype=np.float16,shape=(len(rows),8640)),previous['reused_indices']
    assert not cache_path.exists(),'Inspect incomplete cache initialization before restarting.'
    decision=json.loads((PILOT/'promotion-decision.json').read_text())
    assert decision['promoted']=='attention' and decision['validation_only']
    pilot_rows=pd.read_parquet(components.CACHE/'rows.parquet')
    reused=pilot_rows.cache_ix.to_numpy(dtype=np.int64)
    assert np.array_equal(rows.iloc[reused].sample_ix.to_numpy(),pilot_rows.sample_ix.to_numpy())
    assert np.array_equal(rows.iloc[reused].role.to_numpy(),pilot_rows.role.to_numpy())
    original=components.CACHE/'features.f16'
    compact=CACHE/'pilot-attention.f16'
    if not compact.exists():
        capture=json.loads((PILOT/'capture-complete.json').read_text())
        assert data.digest(original)==capture['sha256']
        for name,_,_ in components.JOBS:
            record=json.loads((PILOT/'candidates'/f'{name}.json').read_text())
            assert data.digest(PILOT/'models'/f'{name}.npz')==record['model_sha256']
        mapped=np.memmap(original,mode='r',dtype=np.float16,shape=(len(pilot_rows),25920))
        attention=np.array(mapped[:,8640:17280],copy=True)
        for start in range(0,len(attention),4096):assert np.isfinite(attention[start:start+4096]).all()
        mapped._mmap.close();del mapped
        # All four portable models and probabilities are already validated. Retire
        # only this disposable, Git-ignored pilot activation file to make space.
        data.atomic_json(PILOT/'cache-retirement.json',dict(
            retired_cache=str(original),original_sha256=capture['sha256'],
            retained_columns=[8640,17280],retained_sha256=components.digest_array(attention),
            retained_rows_sha256=data.digest(components.CACHE/'rows.parquet'),
            reason='Completed pilot; retain selected attention columns and reuse them in the full fit/validation cache.',
            completed_models_unchanged=True,created_unix_time=time.time()))
        original.unlink()
        temporary=compact.with_suffix('.part')
        attention.tofile(temporary);os.replace(temporary,compact)
        del attention;gc.collect()
    retained=json.loads((PILOT/'cache-retirement.json').read_text())
    assert data.digest(compact)==retained['retained_sha256']
    needed=len(rows)*8640*2
    filesystem=os.statvfs(CACHE)
    assert filesystem.f_bavail*filesystem.f_frsize>needed+512*2**20,'Insufficient full-cache space.'
    storage=np.memmap(cache_path,mode='w+',dtype=np.float16,shape=(len(rows),8640))
    saved=np.memmap(compact,mode='r',dtype=np.float16,shape=(len(pilot_rows),8640))
    for start in range(0,len(saved),4096):storage[reused[start:start+4096]]=saved[start:start+4096]
    storage.flush();saved._mmap.close();del saved
    data.atomic_json(initialized_path,dict(signature=signature,reused_indices=reused.tolist(),
        reused_rows=len(reused),retained_pilot_sha256=retained['retained_sha256']))
    # Full cache plus durable initialization now contain these exact rows.
    compact.unlink()
    return rows,storage,reused.tolist()


def missing_inputs(rows,reused):
    missing=rows[~rows.cache_ix.isin(reused)].copy()
    missing['source_set']=np.where(missing.sample_ix>=10**9,'supplement','main')
    missing['original_prompt_ix']=missing.prompt_ix
    tokens={};inputs=[];mapping={}
    for source,folder in [('main',data.CACHE),('supplement',data.CACHE/'supplement')]:
        source_inputs=pd.read_parquet(folder/'inputs.parquet')
        source_tokens=json.loads((folder/'token-ids.json').read_text())
        wanted=set(missing[missing.source_set==source].original_prompt_ix)
        for row in source_inputs[source_inputs.prompt_ix.isin(wanted)].itertuples():
            ix=len(inputs);mapping[(source,int(row.prompt_ix))]=ix
            tokens[ix]=source_tokens[row.prompt_ix]
            inputs.append(dict(prompt_ix=ix,length=row.length))
    missing['prompt_ix']=[mapping[(s,int(p))] for s,p in zip(missing.source_set,missing.original_prompt_ix)]
    assert set(missing.split)=={'fit'} and set(missing.source_set)=={'main'}
    missing.to_parquet(CACHE/'missing-rows.parquet',index=False)
    return missing,pd.DataFrame(inputs),tokens


def main():
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts,collate
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (OUT/'extraction-complete.json').exists(),'Full attention extraction is complete.'
    rows,storage,reused=initialize()
    missing,inputs,tokens=missing_inputs(rows,reused)
    signature=dict(rows_sha256=data.digest(CACHE/'rows.parquet'),
        missing_rows_sha256=data.digest(CACHE/'missing-rows.parquet'),features=SPECS,batch_size=8,
        verified_weights_sha256=data.digest(Path(os.environ['ROLE_ANALYSIS_LOAD_PATH'])/'verified-weights.json'))
    plan=dict(fit_documents=1200,features=SPECS,dimensions=8640,
        reused_rows=len(reused),new_rows=len(missing),new_input_tokens=int(inputs.length.sum()),
        new_inputs=len(inputs),signature=signature,source_sha256=data.digest(__file__),
        pilot_source_sha256=data.digest(data.HERE/'systemness_components.py'),
        pilot_decision_sha256=data.digest(PILOT/'promotion-decision.json'),
        language_model='Frozen native MXFP4, four A40s',test_used=False,
        published_tests_already_reported=True,fresh_confirmation_required=True,
        inference='Pointwise attention output features; no header labels, token IDs or history features.')
    if (OUT/'plan.json').exists():
        previous=json.loads((OUT/'plan.json').read_text());assert previous==plan
    else:data.atomic_json(OUT/'plan.json',plan)
    progress_path=CACHE/'progress.json';start_batch=0;written=0
    if progress_path.exists():
        previous=json.loads(progress_path.read_text());assert previous['signature']==signature
        start_batch=previous['next_batch'];written=previous['new_rows_written']
    runtime=data.runtime
    runtime.atomic_json=lambda path,value:data.atomic_json(OUT/Path(path).name,value)
    runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
    model=runtime.load_model()
    loader=DataLoader(Prompts(inputs,tokens),batch_size=8,shuffle=False,collate_fn=collate)
    lookup={int(p):g for p,g in missing.groupby('prompt_ix',sort=False)}
    captured={}
    def receive(name,value):captured[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started=time.time()
    with torch.inference_mode():
        batch=next(iter(loader))
        sample=SimpleNamespace(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0))
        before=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        handles=components.hooks(model,SPECS,receive)
        after=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        assert torch.equal(before,after) and torch.isfinite(after).all()
        reference=dict(captured)
        for handle in handles:handle.remove()
        del before,after
        original=pipeline.install_hooks;pipeline.install_hooks=components.hooks
        try:
            for bi,batch,states in pipeline.feature_batches(model,loader,SPECS,start_batch,(sample,reference),
                    lambda result:data.atomic_json(OUT/'pipeline-validation.json',result)):
                width=batch['input_ids'].shape[1];positions=[];targets=[]
                for b,prompt in enumerate(batch['prompt_ix']):
                    group=lookup[prompt];offset=b*width+width-len(tokens[prompt])
                    positions.extend((group.token_ix.to_numpy()+offset).tolist());targets.extend(group.cache_ix.tolist())
                indices=torch.tensor(positions,dtype=torch.long)
                values=torch.cat([states[key][indices] for key in SPECS],dim=1)
                assert values.shape==(len(targets),8640) and torch.isfinite(values).all()
                storage[targets]=values.numpy();written+=len(targets);states.clear()
                if (bi+1)%16==0 or bi+1==len(loader):
                    storage.flush()
                    data.atomic_json(progress_path,dict(signature=signature,next_batch=bi+1,
                        new_rows_written=written,complete=bi+1==len(loader)))
                    status=dict(stage='extracting',batch=bi+1,batches=len(loader),new_rows_written=written,
                        reused_rows=len(reused),seconds=time.time()-started)
                    data.atomic_json(OUT/'status.json',status);print(json.dumps(status),flush=True)
        finally:pipeline.install_hooks=original
    assert written==len(missing)
    for start in range(0,len(storage),4096):
        block=storage[start:start+4096]
        assert np.isfinite(block).all() and (block!=0).any(axis=1).all()
    assert data.digest(__file__)==plan['source_sha256']
    data.atomic_json(OUT/'extraction-complete.json',dict(rows=len(rows),reused_rows=len(reused),
        newly_extracted_rows=written,sha256=data.digest(CACHE/'features.f16'),
        rows_sha256=data.digest(CACHE/'rows.parquet'),seconds=time.time()-started,
        source_sha256=data.digest(__file__),signature=signature,all_finite=True,tests_read=False))
    data.atomic_json(OUT/'status.json',dict(stage='features_complete',rows=len(rows)))


if __name__=='__main__':main()
