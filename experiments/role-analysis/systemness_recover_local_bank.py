"""Explicitly reconstruct the retired local prefix in its original two stages.

Default invocation only prints help. --preflight replays historical CPU input
construction under /tmp. Only --recover can allocate the replacement bank or
load the frozen model; it never deletes a cache or starts long-full extraction.
Recovery is accepted only when its float16 hashes exactly equal the old hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_components as components
import systemness_data as data
import systemness_long_extract as long_extract
import systemness_probes as probes

OUT=long_extract.OUT
CACHE=long_extract.CACHE
RECOVERY=CACHE/'recovery-local-bank'
WIDTH=8640
BASE_ROWS=566482
TOTAL_ROWS=755282
BATCH_SIZE=8
STAGE_COUNTS={'pilot':(248874,3690),'remaining':(317608,6000)}


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path,value):
    if path.exists():
        require(read_json(path)==value,f'Recovery {path.name} changed; refusing to overwrite it.')
    else:data.atomic_json(path,value)


def verify_context():
    require((OUT/'cache-retirement.json').is_file(),'Explicit original cache-retirement audit is required.')
    require(not (CACHE/'initialized.json').exists(),
            'The bank is already initialized; this recovery entry point must not overwrite it.')
    require(not (OUT/'extraction-complete.json').exists() and not (CACHE/'progress.json').exists(),
            'Normal long extraction has already started or completed; recovery must not overwrite its bank.')
    retirement=read_json(OUT/'cache-retirement.json')
    extraction_plan=read_json(OUT/'extraction-plan.json')
    require(retirement['signature']==extraction_plan['signature'],'Retirement and frozen extraction plan disagree.')
    expected_paths={str(attention.CACHE/'features.f16'),
        str(data.CACHE/'global-attention-pilot/features.f16')}
    require(set(retirement['retired_files'])==expected_paths,'Unexpected files in the retirement declaration.')
    require(all(not Path(path).exists() for path in expected_paths),
            'Both explicitly retired raw caches must be absent before this recovery is permitted.')
    require(retirement['retained_rows']==BASE_ROWS and retirement['retained_dimensions']==WIDTH
            and retirement['retained_dtype']=='float16','Retirement has a different original bank shape/type.')
    require(retirement['retained_cpu_sha256']==retirement['retired_files'][str(attention.CACHE/'features.f16')],
            'The original retained CPU hash differs from the retired local-file hash.')
    require(extraction_plan['rows']==TOTAL_ROWS and extraction_plan['base_rows']==BASE_ROWS,
            'Frozen destination bank dimensions changed.')
    require(long_extract.signature(CACHE/'rows.parquet',long_extract.input_sources())==retirement['signature'],
            'Original frozen extraction dependencies, inputs, model manifest or row bank changed.')
    _,_,models=long_extract.verify_published_models()
    require(models==retirement['models_sha256'],'Published portable models differ from the retirement audit.')
    return retirement


def close_bank(bank):
    for array in bank.arrays:array._mmap.close()


def build_stages(folder):
    """Call the unchanged historical CPU selectors, writing only under folder."""
    pilot_folder=folder/'pilot';remaining_folder=folder/'remaining'
    pilot_folder.mkdir(parents=True,exist_ok=True);remaining_folder.mkdir(parents=True,exist_ok=True)
    original=components.CACHE
    try:
        components.CACHE=pilot_folder
        pilot,first_inputs,first_tokens=components.inputs_and_rows()
    finally:components.CACHE=original
    require(data.digest(pilot_folder/'rows.parquet')==data.digest(original/'rows.parquet'),
            'Recreated component-pilot row parquet is not byte-identical to the historical input.')
    retained=read_json(components.OUT/'cache-retirement.json')
    require(data.digest(pilot_folder/'rows.parquet')==retained['retained_rows_sha256'],
            'Recreated component-pilot row identity differs from its retained-attention audit.')
    base,bank=probes.development();close_bank(bank);del bank
    require(len(base)==BASE_ROWS and base.cache_ix.tolist()==list(range(BASE_ROWS)),
            'Original pristine development-bank row order changed.')
    require(data.digest(attention.CACHE/'rows.parquet')==
            read_json(attention.OUT/'plan.json')['signature']['rows_sha256'],'Historical full-attention rows changed.')
    pd.testing.assert_frame_equal(base,pd.read_parquet(attention.CACHE/'rows.parquet'),check_exact=True)
    reused=pilot.cache_ix.to_numpy(dtype=np.int64)
    require(np.array_equal(base.iloc[reused].sample_ix.to_numpy(),pilot.sample_ix.to_numpy()),
            'Original pilot-to-base sample mapping changed.')
    original=attention.CACHE
    try:
        attention.CACHE=remaining_folder
        remaining,second_inputs,second_tokens=attention.missing_inputs(base,reused)
    finally:attention.CACHE=original
    historical=read_json(attention.OUT/'plan.json')
    require(data.digest(remaining_folder/'missing-rows.parquet')==historical['signature']['missing_rows_sha256'],
            'Recreated remaining-row parquet differs from the exact historical full-attention input.')
    require(data.digest(remaining_folder/'missing-rows.parquet')==data.digest(original/'missing-rows.parquet'),
            'The retained historical remaining-row parquet changed.')
    joined=np.r_[reused,remaining.cache_ix.to_numpy(dtype=np.int64)]
    require(len(joined)==BASE_ROWS and np.array_equal(np.sort(joined),np.arange(BASE_ROWS)),
            'The two historical stages do not cover every original row exactly once.')
    result=[]
    for name,rows,inputs,tokens,specs in [
        ('pilot',pilot,first_inputs,first_tokens,components.SPECS),
        ('remaining',remaining,second_inputs,second_tokens,attention.SPECS)]:
        require((len(rows),len(inputs))==STAGE_COUNTS[name],f'{name} historical row/input counts changed.')
        require(inputs.prompt_ix.tolist()==list(range(len(inputs))) and set(rows.prompt_ix)==set(inputs.prompt_ix),
                f'{name} prompt indices changed.')
        require(all(len(tokens[ix])==length for ix,length in zip(inputs.prompt_ix,inputs.length)),
                f'{name} historical token lengths changed.')
        require(set(rows.split)<= {'fit','validation'} and set(rows.wrapper)=={'native'},
                f'{name} reconstruction contains test or non-native input rows.')
        inputs.to_parquet(folder/name/'inputs.parquet',index=False)
        immutable_json(folder/name/'token-ids.json',[tokens[ix] for ix in range(len(inputs))])
        result.append(dict(name=name,rows=rows,inputs=inputs,tokens=tokens,specs=specs,
            rows_path=folder/name/('rows.parquet' if name=='pilot' else 'missing-rows.parquet')))
    return result


def recovery_signature(retirement,stages):
    return dict(original_signature=retirement['signature'],runner_source_sha256=data.digest(__file__),
        retirement_sha256=data.digest(OUT/'cache-retirement.json'),
        component_retirement_sha256=data.digest(components.OUT/'cache-retirement.json'),
        historical_attention_plan_sha256=data.digest(attention.OUT/'plan.json'),
        stages={stage['name']:dict(rows_sha256=data.digest(stage['rows_path']),
            inputs_sha256=data.digest(stage['rows_path'].parent/'inputs.parquet'),
            tokens_sha256=data.digest(stage['rows_path'].parent/'token-ids.json'),specs=stage['specs'],
            rows=len(stage['rows']),inputs=len(stage['inputs']),batch_size=BATCH_SIZE) for stage in stages})


def check_bank(path):
    require(path.is_file(),'Recovery bank is missing; refusing to recreate a file with saved recovery progress.')
    require(path.stat().st_size==TOTAL_ROWS*WIDTH*2,
            'Recovery bank has the wrong size; refusing to trust saved recovery progress.')


def open_bank(signature):
    path=CACHE/'features.f16';started=RECOVERY/'started.json'
    has_progress=any((RECOVERY/f'{name}-progress.json').exists() or
                     (RECOVERY/f'{name}-complete.json').exists() for name in STAGE_COUNTS)
    if started.exists():
        require(read_json(started)['signature']==signature,'Existing recovery bank has a different signature.')
        check_bank(path)
    else:
        require(not path.exists() and not has_progress,
                'Unknown partial bank/progress exists without this recovery runner allocation audit.')
        filesystem=os.statvfs(CACHE);needed=TOTAL_ROWS*WIDTH*2
        require(filesystem.f_bavail*filesystem.f_frsize>needed+512*2**20,
                'Insufficient currently available space for the replacement bank; no files will be deleted.')
        data.atomic_json(started,dict(signature=signature,rows=TOTAL_ROWS,dimensions=WIDTH,
            cache_bytes=needed,sparse_file=True,created_unix_time=time.time()))
        # Exclusive sparse allocation never truncates an unknown existing file.
        with path.open('xb') as stream:stream.truncate(needed)
    return np.memmap(path,mode='r+',dtype=np.float16,shape=(TOTAL_ROWS,WIDTH))


def gather_digest(array,indices,block_size=4096):
    digest=hashlib.sha256()
    for start in range(0,len(indices),block_size):
        block=np.ascontiguousarray(array[indices[start:start+block_size]])
        long_extract.check_values(block)
        digest.update(memoryview(block).cast('B'))
    return digest.hexdigest()


def validate_progress(stage,progress,signature):
    require(progress['signature']==signature,'Saved recovery progress signature changed.')
    next_batch=progress['next_batch'];written=progress['rows_written']
    ordered=stage['inputs'].sort_values(['length','prompt_ix'])
    batches=(len(ordered)+BATCH_SIZE-1)//BATCH_SIZE
    require(0<=next_batch<=batches,'Saved recovery batch counter is invalid.')
    completed=set(ordered.iloc[:min(next_batch*BATCH_SIZE,len(ordered))].prompt_ix)
    require(int(stage['rows'].prompt_ix.isin(completed).sum())==written,
            'Saved recovery row counter does not match the exact original completed batches.')
    return next_batch,written


def run_stage(stage,array,signature,get_model):
    name=stage['name'];rows=stage['rows'];inputs=stage['inputs'];tokens=stage['tokens'];specs=stage['specs']
    progress_path=RECOVERY/f'{name}-progress.json';complete_path=RECOVERY/f'{name}-complete.json'
    targets=rows.cache_ix.to_numpy(dtype=np.int64)
    expected=read_json(components.OUT/'cache-retirement.json')['retained_sha256'] if name=='pilot' else None
    if complete_path.exists():
        complete=read_json(complete_path)
        require(complete['signature']==signature,'Completed recovery-stage signature changed.')
        digest=gather_digest(array,targets)
        require(digest==complete['attention_sha256'] and (expected is None or digest==expected),
                'A completed recovery-stage feature hash changed.')
        return
    start_batch=0;written=0
    if progress_path.exists():start_batch,written=validate_progress(stage,read_json(progress_path),signature)
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts,collate
    model=get_model();loader=DataLoader(Prompts(inputs,tokens),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)
    lookup={int(prompt):group for prompt,group in rows.groupby('prompt_ix',sort=False)}
    captured={}
    def receive(key,value):captured[key]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started=time.time()
    with torch.inference_mode():
        batch=next(iter(loader))
        sample=SimpleNamespace(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0))
        before=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        handles=components.hooks(model,specs,receive)
        try:
            after=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
            require(torch.equal(before,after) and bool(torch.isfinite(after).all()),'Recovery capture hooks changed model hidden states.')
            reference=dict(captured)
        finally:
            for handle in handles:handle.remove()
        del before,after
        original=pipeline.install_hooks;pipeline.install_hooks=components.hooks
        try:
            for bi,batch,states in pipeline.feature_batches(model,loader,specs,start_batch,(sample,reference),
                    lambda result:data.atomic_json(RECOVERY/f'{name}-pipeline-validation.json',result)):
                width=batch['input_ids'].shape[1];positions=[];destination=[]
                for b,prompt in enumerate(batch['prompt_ix']):
                    group=lookup[prompt];offset=b*width+width-len(tokens[prompt])
                    positions.extend((group.token_ix.to_numpy()+offset).tolist());destination.extend(group.cache_ix.tolist())
                indices=torch.tensor(positions,dtype=torch.long)
                values=torch.cat([states[key][indices] for key in attention.SPECS],dim=1)
                require(values.shape==(len(destination),WIDTH) and bool(torch.isfinite(values).all()),
                        'Recovery produced an invalid attention batch.')
                array[destination]=values.numpy();written+=len(destination);states.clear()
                if (bi+1)%16==0 or bi+1==len(loader):
                    array.flush()
                    data.atomic_json(progress_path,dict(signature=signature,next_batch=bi+1,rows_written=written))
                    status=dict(stage='recovering_original_local_bank',part=name,batch=bi+1,batches=len(loader),
                        rows_written=written,rows=len(rows),seconds=time.time()-started)
                    data.atomic_json(OUT/'cache-recovery-status.json',status);print(json.dumps(status),flush=True)
        finally:pipeline.install_hooks=original
    require(written==len(rows),'Recovery stage did not write every original row.')
    array.flush()
    digest=gather_digest(array,targets)
    require(expected is None or digest==expected,'Recomputed pilot attention hash is not bit-identical to the original.')
    validation=read_json(RECOVERY/f'{name}-pipeline-validation.json')
    require(validation['exact_activations'] and set(validation['max_absolute_difference'])==set(specs)
            and all(value==0 for value in validation['max_absolute_difference'].values()),
            'Recovery sequential/pipeline features are not bit-identical at all original capture points.')
    data.atomic_json(complete_path,dict(signature=signature,rows=written,attention_sha256=digest,
        historical_attention_sha256=expected,hooks_leave_hidden_states_exact=True,
        pipeline_all_original_capture_points_bitexact=True,seconds=time.time()-started))


def finalize(array,retirement,signature):
    long_extract.check_values(array[:BASE_ROWS])
    actual=long_extract.prefix_digest(CACHE/'features.f16',BASE_ROWS*WIDTH*2)
    require(actual==retirement['retained_cpu_sha256'],
            'Recomputed full original local prefix is not bit-identical; initialized.json will not be written.')
    for start in range(BASE_ROWS,TOTAL_ROWS,4096):
        require((array[start:start+4096]==0).all(),'Recovery unexpectedly wrote unextracted long-full rows.')
    for name in STAGE_COUNTS:
        complete=read_json(RECOVERY/f'{name}-complete.json')
        require(complete['signature']==signature,'Recovery-stage signature changed before finalization.')
    audit=dict(signature=signature,original_signature=retirement['signature'],
        original_retirement_sha256=data.digest(OUT/'cache-retirement.json'),
        restored_rows=BASE_ROWS,destination_rows=TOTAL_ROWS,dimensions=WIDTH,dtype='float16',
        actual_prefix_sha256=actual,expected_prefix_sha256=retirement['retained_cpu_sha256'],
        exact_original_prefix_restored=True,new_long_rows_not_extracted=True,
        method='Recomputed original component pilot with nine capture points, then original full-attention missing inputs; original batches and float16 bits verified.',
        reason='Original long-full initialization retired the raw files, then exited before replacement-bank allocation because statvfs had not yet reflected freed space.',
        no_remaining_cache_files_deleted=True,long_full_extraction_started=False,
        source_sha256=data.digest(__file__),created_unix_time=time.time())
    audit_path=OUT/'cache-recovery.json'
    if audit_path.exists():
        previous=read_json(audit_path)
        require({key:value for key,value in previous.items() if key!='created_unix_time'}==
                {key:value for key,value in audit.items() if key!='created_unix_time'},
                'Existing completed recovery audit differs; refusing to replace it.')
        audit=previous
    else:data.atomic_json(audit_path,audit)
    # Only a verified original prefix may unlock the unchanged long extractor.
    initialized=dict(signature=retirement['signature'],reused_rows=BASE_ROWS,total_rows=TOTAL_ROWS,
        reused_prefix_sha256=actual,cache_bytes=TOTAL_ROWS*WIDTH*2,
        retirement_sha256=data.digest(OUT/'cache-retirement.json'),
        recovered_by_recomputation=True,recovery_audit_sha256=data.digest(audit_path))
    require(not (CACHE/'initialized.json').exists(),'Bank initialization appeared while recovery was running.')
    data.atomic_json(CACHE/'initialized.json',initialized)
    data.atomic_json(OUT/'cache-recovery-status.json',dict(stage='complete',restored_rows=BASE_ROWS,
        original_prefix_bitexact=True,long_full_extraction_started=False))
    print(json.dumps(audit),flush=True)


def recover():
    retirement=verify_context()
    RECOVERY.mkdir(exist_ok=True)
    stages=build_stages(RECOVERY)
    signature=recovery_signature(retirement,stages)
    immutable_json(OUT/'cache-recovery-plan.json',dict(signature=signature,
        purpose='Explicit recomputation of the exact retired original local-attention prefix.',
        no_cache_deletion=True,long_full_extraction_started=False))
    # Validate checkpoint signatures before even allocating/opening a bank.
    for stage in stages:
        path=RECOVERY/f'{stage["name"]}-progress.json'
        if path.exists():validate_progress(stage,read_json(path),signature)
    array=open_bank(signature)
    model=None
    def get_model():
        nonlocal model
        if model is None:
            import torch
            require(torch.cuda.device_count()==4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)),
                    'Recovery requires four A40 GPUs and the original frozen native MXFP4 checkpoint.')
            runtime=data.runtime
            runtime.atomic_json=lambda output,value:data.atomic_json(RECOVERY/Path(output).name,value)
            runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
            model=runtime.load_model()
        return model
    for stage in stages:run_stage(stage,array,signature,get_model)
    array.flush()
    require(verify_context()==retirement and recovery_signature(retirement,stages)==signature,
            'Recovery source, input, model or frozen metadata changed while it ran.')
    finalize(array,retirement,signature)


def preflight():
    retirement=verify_context()
    with tempfile.TemporaryDirectory(prefix='local-bank-recovery-preflight-',dir='/tmp') as temporary:
        stages=build_stages(Path(temporary))
        signature=recovery_signature(retirement,stages)
        print(json.dumps(dict(stage='cpu_preflight_only',stages=signature['stages'],
            original_prefix_sha256=retirement['retained_cpu_sha256'],rows=TOTAL_ROWS,
            no_model_loaded=True,no_cache_deleted=True,no_formal_output_written=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='Replay CPU input construction in /tmp; never load weights or alter the bank.')
    action.add_argument('--recover',action='store_true',help='Explicitly reconstruct and verify the original local prefix; do not start long-full extraction.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.recover:recover()
    else:parser.print_help()
