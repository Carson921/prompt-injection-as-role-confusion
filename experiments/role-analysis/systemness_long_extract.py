"""Reuse the frozen local-attention bank and extract denser long-body rows.

No action runs by default. --preflight performs read-only CPU checks; --extract
explicitly authorizes this entry point's two documented disposable-cache
retirements and the frozen four-A40 forward pass. Existing published scripts,
models, source documents and original row indices remain unchanged.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_components as components
import systemness_data as data
import systemness_global_attention as global_attention
import systemness_long_data as long_data
import systemness_probes as probes

OUT=data.runtime.OUT/'systemness-long-full'
CACHE=data.CACHE/'long-full-bank'
SPECS={f'attention-{layer}':dict(point='attention',layer=layer) for layer in [16,20,24]}
DIMENSIONS=8640
BASE_ROWS=566482
LONG_ROWS=188800
BATCH_SIZE=8
DEPENDENCIES=['systemness_long_extract.py','systemness_long_data.py','systemness_data.py',
    'systemness_attention.py','systemness_components.py','systemness_global_attention.py',
    'systemness_probes.py','systemness_extract.py','feature_pipeline_runtime.py','run_local_gptoss120b.py']


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint():
    value=os.environ.get('ROLE_ANALYSIS_LOAD_PATH')
    require(bool(value),'Set ROLE_ANALYSIS_LOAD_PATH to the verified local model checkpoint.')
    require(not (value=='/workspace' or value.startswith('/workspace/')),
            'The inaccessible /workspace mount must not be accessed.')
    path=Path(value).resolve()
    require(not (str(path)=='/workspace' or str(path).startswith('/workspace/')),
            'Model path resolves to forbidden /workspace.')
    require((path/'verified-weights.json').is_file(),'Verified weight manifest is missing.')
    return path


def check_file(path,rows,columns=DIMENSIONS):
    expected=rows*columns*np.dtype(np.float16).itemsize
    require(Path(path).is_file(),f'Missing feature bank: {path}; refusing to resume partial state.')
    require(Path(path).stat().st_size==expected,
            f'Incorrect feature-bank size: {path}; refusing to resume partial state.')


def check_values(array,block_size=4096):
    for start in range(0,len(array),block_size):
        block=array[start:start+block_size]
        require(np.isfinite(block).all() and (block!=0).any(axis=1).all(),
                f'Nonfinite or unwritten feature rows {start}:{start+len(block)}.')


def prefix_digest(path,byte_count):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining=byte_count
        while remaining:
            block=stream.read(min(16*2**20,remaining))
            require(bool(block),'Feature bank is truncated while verifying the reused prefix.')
            digest.update(block);remaining-=len(block)
    return digest.hexdigest()


def input_sources():
    result={}
    for name,folder,meta_path in [
        ('main',data.CACHE,data.OUT/'data-complete.json'),
        ('supplement',data.CACHE/'supplement',data.OUT/'supplement-data-complete.json'),
        ('long-full',long_data.CACHE,OUT/'data-complete.json')]:
        metadata=read_json(meta_path)
        hashes=metadata['cache_sha256']
        require(all(data.digest(folder/key)==sha for key,sha in hashes.items()),
                f'{name} source input cache changed.')
        result[name]=dict(folder=str(folder),metadata_sha256=data.digest(meta_path),cache_sha256=hashes)
    long=read_json(OUT/'data-complete.json');plan=read_json(OUT/'plan.json')
    require(long['plan_sha256']==data.digest(OUT/'plan.json'),'Long-full preparation plan changed.')
    require(long['document_keys_sha256']==data.digest(OUT/'document-keys.json'),'Long-full document keys changed.')
    require(long['independent_validation']['all_labels_match'] and not long['test_inputs_read'],
            'Long-full data lacks independent native-label validation or read test inputs.')
    require(long['sampled_rows']==LONG_ROWS and long['selected_document_counts']=={'fit':132,'validation':31},
            'Long-full prepared row/document counts changed.')
    require(long['source_sha256']==plan['source_sha256'],'Long-full source declarations disagree.')
    require(all(data.digest(data.HERE/name)==sha for name,sha in long['source_sha256'].items()),
            'Long-full preparation source changed after preparation.')
    return result


def assemble_rows():
    sources=input_sources()
    base,bank=probes.development()
    # Only original row tables are needed. Close the original residual mappings;
    # this entry point never copies, deletes or uses their activation contents.
    for array in bank.arrays:
        array._mmap.close()
    del bank
    require(len(base)==BASE_ROWS and base.cache_ix.tolist()==list(range(BASE_ROWS)),
            'Original development bank membership/order changed.')
    local_rows=pd.read_parquet(attention.CACHE/'rows.parquet')
    pd.testing.assert_frame_equal(base,local_rows,check_exact=True)
    extra=pd.read_parquet(long_data.CACHE/'rows.parquet')
    require(len(extra)==LONG_ROWS and extra.sample_ix.tolist()==list(range(2_000_000_000,2_000_000_000+LONG_ROWS)),
            'Long-full sample identity/order changed.')
    require(extra.groupby('split').size().to_dict()=={'fit':168960,'validation':19840},
            'Long-full fit/validation sampling counts changed.')
    for name,frame in [('base',base),('long-full',extra)]:
        require(set(frame.split)=={'fit','validation'} and set(frame.wrapper)=={'native'},
                f'{name} bank contains test rows or non-native wrappers.')
        require(not set(frame[frame.split=='fit'].group_id)&set(frame[frame.split=='validation'].group_id),
                f'{name} bank leaks source groups between fit and validation.')
    require(not set(base.sample_ix)&set(extra.sample_ix),'Base and long-full sample identities overlap.')
    base=base.copy();extra=extra.copy()
    base['source_set']=np.where(base.sample_ix>=10**9,'supplement','main')
    extra['source_set']='long-full'
    rows=pd.concat([base,extra],ignore_index=True)
    rows['cache_ix']=np.arange(len(rows),dtype=np.int64)
    rows['original_prompt_ix']=rows.prompt_ix
    require(not set(rows[rows.split=='fit'].group_id)&set(rows[rows.split=='validation'].group_id),
            'Combined bank leaks source groups between fit and validation.')
    inputs=pd.read_parquet(long_data.CACHE/'inputs.parquet')
    tokens=read_json(long_data.CACHE/'token-ids.json')
    require(inputs.prompt_ix.tolist()==list(range(len(inputs))) and len(tokens)==len(inputs)==815,
            'Long-full inputs and token IDs are not aligned.')
    require(set(extra.prompt_ix)==set(inputs.prompt_ix),'Long-full sampled rows do not cover all prompts.')
    lengths=dict(zip(inputs.prompt_ix,inputs.length))
    require(all(len(tokens[ix])==length for ix,length in lengths.items()),'Input-token lengths changed.')
    require(all(0<=row.token_ix<lengths[row.prompt_ix] and row.token_id==tokens[row.prompt_ix][row.token_ix]
                for row in extra.itertuples()),'Sampled token IDs or positions do not match prepared inputs.')
    return rows,inputs,tokens,sources


def verify_published_models():
    """CPU-only hash/portable-export checks before either disposable file retires."""
    local_capture=read_json(attention.OUT/'extraction-complete.json')
    local_selection=read_json(attention.OUT/'selection.json')
    require(local_capture['rows']==BASE_ROWS and local_capture['all_finite'] and not local_capture['tests_read'],
            'Local full extraction is incomplete or not fit/validation-only.')
    require(local_capture['rows_sha256']==data.digest(attention.CACHE/'rows.parquet'),
            'Local full bank row table changed.')
    require(local_capture['signature']['features']==SPECS,'Local reusable bank has different features.')
    require(local_capture['source_sha256']==data.digest(attention.__file__),'Local extraction source changed.')
    require(local_selection['search_plan_sha256']==data.digest(attention.OUT/'search-plan.json'),
            'Published local search plan changed.')
    models={}
    for name in ['linear-C0.0005','linear-C0.005','mlp-h64','mlp-h128']:
        record_path=attention.OUT/'candidates'/f'{name}.json'
        record=read_json(record_path);path=attention.OUT/'models'/f'{name}.npz'
        require(data.digest(record_path)==local_selection['candidates'][name],f'Published candidate changed: {name}.')
        require(data.digest(path)==record['model_sha256'],f'Published portable local model changed: {name}.')
        require(record['portable_probability_max_difference']<5e-4 and not record['tests_read'],
                f'Local portable model was not validated: {name}.')
        models[f'local/{name}']=record['model_sha256']
    require(data.digest(Path(local_selection['model_path']))==local_selection['model_sha256'],
            'Frozen selected local model changed.')
    complete=read_json(global_attention.OUT/'complete.json')
    capture=read_json(global_attention.OUT/'capture-complete.json')
    plan=read_json(global_attention.OUT/'plan.json')
    record=read_json(global_attention.OUT/'candidates/attention-global.json')
    require(not complete['tests_read'] and complete['plan_sha256']==data.digest(global_attention.OUT/'plan.json'),
            'Global pilot did not complete its declared validation-only protocol.')
    require(complete['promoted'] is False,'Only the completed, rejected global-pilot cache is retired here.')
    require(complete['source_sha256']==capture['source_sha256']==data.digest(global_attention.__file__),
            'Global pilot extraction source changed.')
    require(capture['all_finite'] and capture['hooks_leave_hidden_states_exact'] and capture['rows']==248874,
            'Global pilot capture is incomplete or was not verified.')
    require(capture['features']==global_attention.SPECS,'The completed global-pilot feature definition changed.')
    require(plan['rows_sha256']==capture['signature']['rows_sha256']==record['rows_sha256']==
            data.digest(global_attention.CACHE/'rows.parquet'),'Global pilot row identities changed.')
    require(plan['linear_trainer_sha256']==data.digest(components.__file__), 'Global pilot trainer source changed.')
    path=global_attention.OUT/'models/attention-global.npz'
    require(data.digest(path)==record['model_sha256'] and record['portable_probability_max_difference']<5e-4
            and not record['test_used_for_selection'],'Global portable model lacks its verified export.')
    models['global/attention-global']=record['model_sha256']
    return local_capture,capture,models


def signature(rows_path,sources):
    model_path=checkpoint()
    config=read_json(model_path/'config.json')
    require(all(config['layer_types'][i]=='sliding_attention' for i in [16,20,24]),
            'Expected local-attention layer types changed.')
    return dict(rows_sha256=data.digest(rows_path),input_sources=sources,
        long_plan_sha256=data.digest(OUT/'plan.json'),long_data_sha256=data.digest(OUT/'data-complete.json'),
        features=SPECS,batch_size=BATCH_SIZE,dimensions=DIMENSIONS,
        model_path=str(model_path),verified_weights_sha256=data.digest(model_path/'verified-weights.json'),
        config_sha256=data.digest(model_path/'config.json'),
        dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        local_capture_sha256=data.digest(attention.OUT/'extraction-complete.json'),
        local_selection_sha256=data.digest(attention.OUT/'selection.json'),
        global_complete_sha256=data.digest(global_attention.OUT/'complete.json'),
        global_capture_sha256=data.digest(global_attention.OUT/'capture-complete.json'))


def save_rows(rows):
    destination=CACHE/'rows.parquet';temporary=CACHE/'rows.parquet.tmp'
    require(not temporary.exists(),'Partial row-table initialization exists; inspect it before restarting.')
    rows.to_parquet(temporary,index=False)
    if destination.exists():
        try:
            require(data.digest(temporary)==data.digest(destination),'Existing long-full bank rows changed.')
        finally:temporary.unlink()
    else:temporary.replace(destination)


def initialize(rows,current_signature):
    path=CACHE/'features.f16';initialized_path=CACHE/'initialized.json'
    retirement_path=OUT/'cache-retirement.json'
    if initialized_path.exists():
        saved=read_json(initialized_path)
        require(saved['signature']==current_signature,'Initialized feature bank signature changed.')
        require(saved['retirement_sha256']==data.digest(retirement_path),
                'Initialized feature bank lost its matching cache-retirement audit.')
        check_file(path,len(rows))
        require(prefix_digest(path,BASE_ROWS*DIMENSIONS*2)==saved['reused_prefix_sha256'],
                'The reused local feature prefix changed after initialization.')
        return np.memmap(path,mode='r+',dtype=np.float16,shape=(len(rows),DIMENSIONS))
    require(not path.exists() and not retirement_path.exists(),
            'Partial cache retirement/initialization detected. Refusing to reuse or overwrite it; inspect retained files explicitly.')
    local_capture,global_capture,models=verify_published_models()
    local_path=attention.CACHE/'features.f16';global_path=global_attention.CACHE/'features.f16'
    check_file(local_path,BASE_ROWS);check_file(global_path,global_capture['rows'])
    require(data.digest(local_path)==local_capture['sha256'],'Local full feature SHA mismatch; refusing retirement.')
    require(data.digest(global_path)==global_capture['sha256'],'Global pilot feature SHA mismatch; refusing retirement.')
    expected_weights=current_signature['verified_weights_sha256']
    require(local_capture['signature']['verified_weights_sha256']==global_capture['signature']['verified_weights_sha256']==expected_weights,
            'Reusable features do not use the current verified frozen model weights.')
    # Keep every reused original float16 bit in anonymous CPU memory before
    # deleting its disposable disk copy. No other activation cache is retired.
    mapped=np.memmap(local_path,mode='r',dtype=np.float16,shape=(BASE_ROWS,DIMENSIONS))
    retained=np.array(mapped,copy=True,order='C');mapped._mmap.close();del mapped
    check_values(retained)
    retained_sha=components.digest_array(retained)
    require(retained_sha==local_capture['sha256'],'Anonymous retained local features failed their bitwise SHA check.')
    needed=len(rows)*DIMENSIONS*2
    filesystem=os.statvfs(CACHE)
    available=filesystem.f_bavail*filesystem.f_frsize
    require(available+local_path.stat().st_size+global_path.stat().st_size>needed+512*2**20,
            'Insufficient space even after retiring exactly the two declared disposable files.')
    # This durable marker makes an interruption after unlink explicit. Such an
    # incomplete initialization is rejected on restart rather than treated as a
    # recoverable feature bank with missing original rows.
    data.atomic_json(retirement_path,dict(signature=current_signature,models_sha256=models,
        retired_files={str(local_path):local_capture['sha256'],str(global_path):global_capture['sha256']},
        retained_rows=BASE_ROWS,retained_dimensions=DIMENSIONS,retained_dtype='float16',
        retained_cpu_sha256=retained_sha,retained_in='anonymous CPU RAM until durable bank initialization',
        reason='Completed local full and rejected global pilot; reuse every local row and add denser existing-document long inputs.',
        interrupted_initialization_requires_explicit_recovery=True,created_unix_time=time.time()))
    global_path.unlink();local_path.unlink()
    filesystem=os.statvfs(CACHE)
    require(filesystem.f_bavail*filesystem.f_frsize>needed+512*2**20,'Retired filesystem space is not available.')
    storage=np.memmap(path,mode='w+',dtype=np.float16,shape=(len(rows),DIMENSIONS))
    for start in range(0,BASE_ROWS,4096):storage[start:start+min(4096,BASE_ROWS-start)]=retained[start:start+4096]
    storage.flush()
    require(prefix_digest(path,BASE_ROWS*DIMENSIONS*2)==retained_sha,'Durable reused prefix SHA differs from retained CPU data.')
    data.atomic_json(initialized_path,dict(signature=current_signature,reused_rows=BASE_ROWS,
        total_rows=len(rows),reused_prefix_sha256=retained_sha,cache_bytes=needed,
        retirement_sha256=data.digest(retirement_path)))
    del retained;gc.collect()
    return storage


def prepare_extraction():
    rows,inputs,tokens,sources=assemble_rows()
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    save_rows(rows)
    current_signature=signature(CACHE/'rows.parquet',sources)
    plan=dict(signature=current_signature,source_sha256=data.digest(__file__),
        rows=len(rows),base_rows=BASE_ROWS,added_rows=LONG_ROWS,dimensions=DIMENSIONS,
        reused_rows=BASE_ROWS,new_inputs=len(inputs),new_input_tokens=int(inputs.length.sum()),
        language_model='Frozen native MXFP4 on four A40s',features=SPECS,
        input_mapping='Base prompt_ix values preserved with source_set main/supplement; new inputs use source_set long-full and its original prompt_ix.',
        training_policy='Bank preserves all original base rows. Readout training replaces overlapping old long-single rows as declared in plan.json.',
        tests_read=False,inference='Pointwise local attention outputs only; no token IDs, role labels or history supplied to readout.')
    plan_path=OUT/'extraction-plan.json'
    if plan_path.exists():require(read_json(plan_path)==plan,'Extraction plan/signature changed.')
    else:data.atomic_json(plan_path,plan)
    return rows,inputs,tokens,sources,current_signature,plan


def extract():
    rows,inputs,tokens,sources,current_signature,plan=prepare_extraction()
    path=CACHE/'features.f16';done_path=OUT/'extraction-complete.json'
    if done_path.exists():
        complete=read_json(done_path)
        require(complete['signature']==current_signature,'Completed extraction signature changed.')
        check_file(path,len(rows))
        require(data.digest(path)==complete['sha256'],'Completed feature bank SHA changed.')
        print(json.dumps(dict(stage='already_complete_verified',rows=len(rows))),flush=True)
        return
    # Validate saved progress and cache existence before any destructive action
    # or model allocation. A missing initialized bank cannot be reconstructed by
    # silently creating a zero-filled file and skipping completed batches.
    progress_path=CACHE/'progress.json';start_batch=0;written=0
    if progress_path.exists():
        progress=read_json(progress_path)
        require(progress['signature']==current_signature,'Extraction progress signature changed.')
        require((CACHE/'initialized.json').is_file(),'Progress exists without durable cache initialization.')
        check_file(path,len(rows))
        start_batch=progress['next_batch'];written=progress['new_rows_written']
    import torch
    require(torch.cuda.device_count()==4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)),
            'This experiment requires exactly four A40 GPUs before any cache is retired.')
    storage=initialize(rows,current_signature)
    missing=rows.iloc[BASE_ROWS:].copy()
    require(set(missing.source_set)=={'long-full'} and len(missing)==LONG_ROWS,'Only prepared long-full rows may be newly extracted.')
    batches=(len(inputs)+BATCH_SIZE-1)//BATCH_SIZE
    require(0<=start_batch<=batches and 0<=written<=LONG_ROWS,'Invalid resumed batch/row counters.')
    # Match Prompts' length order so the checkpoint counter itself is checked
    # against exact document/token identities, not only a plausible row total.
    ordered=inputs.sort_values(['length','prompt_ix'])
    covered=set(ordered.iloc[:min(start_batch*BATCH_SIZE,len(ordered))].prompt_ix)
    require(int(missing.prompt_ix.isin(covered).sum())==written,'Saved progress rows do not match its completed batches.')
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts,collate
    runtime=data.runtime
    runtime.atomic_json=lambda output,value:data.atomic_json(OUT/Path(output).name,value)
    runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
    model=runtime.load_model()
    loader=DataLoader(Prompts(inputs,tokens),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)
    lookup={int(prompt):group for prompt,group in missing.groupby('prompt_ix',sort=False)}
    captured={}
    def receive(name,value):captured[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started=time.time()
    with torch.inference_mode():
        batch=next(iter(loader))
        sample=SimpleNamespace(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0))
        before=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        handles=components.hooks(model,SPECS,receive)
        try:
            after=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
            require(torch.equal(before,after) and bool(torch.isfinite(after).all()),'Capture hooks changed model hidden states.')
            reference=dict(captured)
        finally:
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
                values=torch.cat([states[name][indices] for name in SPECS],dim=1)
                require(values.shape==(len(targets),DIMENSIONS) and bool(torch.isfinite(values).all()),'Invalid extracted feature batch.')
                storage[targets]=values.numpy();written+=len(targets);states.clear()
                if (bi+1)%16==0 or bi+1==len(loader):
                    storage.flush()
                    data.atomic_json(progress_path,dict(signature=current_signature,next_batch=bi+1,new_rows_written=written))
                    status=dict(stage='extracting',batch=bi+1,batches=len(loader),new_rows_written=written,
                        reused_rows=BASE_ROWS,seconds=time.time()-started)
                    data.atomic_json(OUT/'status.json',status);print(json.dumps(status),flush=True)
        finally:pipeline.install_hooks=original
    require(written==LONG_ROWS,'Extraction did not write every declared sampled row.')
    storage.flush();check_values(storage)
    validation=read_json(OUT/'pipeline-validation.json')
    require(validation['exact_activations'] and set(validation['max_absolute_difference'])==set(SPECS)
            and all(value==0 for value in validation['max_absolute_difference'].values()),
            'Sequential and pipelined features were not bitwise identical at every capture point.')
    require(signature(CACHE/'rows.parquet',input_sources())==current_signature,'Input, model or source dependency changed during extraction.')
    require(prefix_digest(path,BASE_ROWS*DIMENSIONS*2)==read_json(CACHE/'initialized.json')['reused_prefix_sha256'],
            'Original reused feature rows changed during extraction.')
    data.atomic_json(done_path,dict(rows=len(rows),reused_rows=BASE_ROWS,newly_extracted_rows=written,
        dimensions=DIMENSIONS,features=SPECS,sha256=data.digest(path),signature=current_signature,
        rows_sha256=data.digest(CACHE/'rows.parquet'),source_sha256=data.digest(__file__),
        hooks_leave_hidden_states_exact=True,pipeline_all_features_bitexact=True,all_finite=True,
        all_rows_nonzero=True,tests_read=False,seconds=time.time()-started))
    data.atomic_json(OUT/'status.json',dict(stage='features_complete',rows=len(rows)))


def preflight():
    rows,inputs,tokens,sources=assemble_rows()
    _,_,models=verify_published_models()
    checkpoint()
    print(json.dumps(dict(stage='cpu_preflight_only',rows=len(rows),base_rows=BASE_ROWS,
        added_rows=LONG_ROWS,dimensions=DIMENSIONS,features=SPECS,new_inputs=len(inputs),
        input_tokens=sum(map(len,tokens)),models_sha256=models,source_sets=list(sources),
        no_model_loaded=True,no_cache_retired=True,no_output_written=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='Read-only CPU provenance/schema checks.')
    action.add_argument('--extract',action='store_true',help='Retire the two verified disposable caches, reuse local features, and extract new rows.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.extract:extract()
    else:parser.print_help()
