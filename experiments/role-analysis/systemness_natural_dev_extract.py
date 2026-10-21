"""Extract fixed512-head features for clean natural development transcripts.

Default help. --preflight uses CPU synthetic arrays/hooks and projected existing
dev row descriptors only. Explicit --stage declare hashes metadata and never
loads a prompt, model or probability. Root-only --stage extract requires that
declaration. No old128 test reply, raw public source, current seed result or
global historical-inventory verifier is consulted. No readout is trained here.
"""
import argparse
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as native
import systemness_natural_transcripts as dev
import systemness_head_probes as portable
import systemness_head_extract as bank_helpers
import systemness_aug_confirmation_infer as shared


OUT=data.runtime.OUT/'systemness-natural-dev-features'
CACHE=Path('/dev/shm/role-analysis-natural-dev-heads')
ROWS,PROMPTS,FIT_ROWS,VALID_ROWS,DIMENSIONS=80116,319,63403,16713,512
FIT_PROMPTS,VALID_PROMPTS=255,64
SOURCE_SET='natural-dev-clean'
DEV_COMPLETE_SHA256='b6c411c920f583a128401c0d787869e2eba17b632136a5cec9613fd227bb9dfd'
OLD_MODEL_SHA256='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
HEAD_IDS=[458,462,2045,460,459,2047,456,997]
OLD_MODEL=portable.OUT/'models/joint-mlp128.npz'
SCALER=portable.OUT/'standardization.npz'
PLAN_NAME='extraction-plan.json'
BATCH,CHECKPOINT_INTERVAL,RESERVE_BYTES,MIN_GPU_FREE=4,16,512*2**20,22*2**30
PARITY_TOLERANCE=5e-4
ROLES=['system','user','assistant']
ROLE_COUNTS={'fit':{'system':7935,'user':12660,'assistant':42808},
    'validation':{'system':2032,'user':3596,'assistant':11085}}
INPUT_COLUMNS=['prompt_ix','length','turns','message_count','wrapper','profile','group_id','source_uid',
    'dataset','native_id','split','template_id','family_id','eligible_fit','eligible_clean_validation',
    'old_generated_collision','cross_split_assistant_collision','dependency_component_id']
ROW_COLUMNS=['cache_ix','sample_ix','prompt_ix','wrapper','profile','group_id','source_uid','dataset',
    'native_id','split','template_id','family_id','token_ix','token_id','message_ix','role','semantic_role',
    'eligible_fit','eligible_clean_validation','old_generated_collision','cross_split_assistant_collision','dependency_component_id']
DEPENDENCIES=['systemness_natural_dev_extract.py','systemness_natural_transcripts.py','systemness_natural_sources.py',
    'systemness_aug_confirmation_infer.py','systemness_aug_confirm_snapshot.py','systemness_head_extract.py','systemness_head_probes.py',
    'systemness_long_data.py','systemness_data.py','run_local_gptoss120b.py']
require,read=native.require,dev.read
array_sha=bank_helpers.array_digest
frame_sha=bank_helpers.frame_digest


def immutable_json(path,value):
    if Path(path).exists():require(read(path)==value,'Immutable extraction artifact differs: '+str(path))
    else:data.atomic_json(path,value)


def protocol_spec():
    return dict(version=1,experiment='clean-natural-dev-fixed-head-extraction',
        source='Existing frozen systemness_natural_transcripts dev320 prepared cache, not public-source replay.',
        filter='All255 prepared fit conversations with eligible_fit=True plus all64 validation conversations; quarantine the entire one contaminated fit conversation including all320 of its body rows. No validation enters fit.',
        prompts=PROMPTS,rows=ROWS,fit_prompts=FIT_PROMPTS,validation_prompts=VALID_PROMPTS,
        fit_rows=FIT_ROWS,validation_rows=VALID_ROWS,role_counts=ROLE_COUNTS,
        truth_roles=ROLES,no_cot_tool_truth=True,all_native_content_positions=True,
        row_identity='Stable original prompt/sample order. Preserve original4e12 sample_ix with quarantine gap; new contiguous cache_ix/prompt_ix and original_cache_ix/original_prompt_ix. Never synthesize C4 line/document IDs.',
        feature_shape=[ROWS,DIMENSIONS],dtype='float16',raw_feature_bytes=ROWS*DIMENSIONS*2,
        selected_head_ids=HEAD_IDS,head_dim=64,selected_head_order='Identical original frozen top8 order; pre-o_proj query heads.',
        identity='Fixed original joint-mlp128 SHA27a22 only as portable parity consumer; metadata/config/tokenizer/verified weights closure. No current seed/augmentation results or old rglob verifier.',
        backend='Exactly4 A40; frozen nativeMXFP4 sequential shared forward; transient current-batch selected-head vectors only.',
        batch_size=BATCH,checkpoint_interval=CHECKPOINT_INTERVAL,
        batch_order='Prioritize prompts containing fixed first256 clean FIT cache_ix, then remaining prompts by(length,prompt_ix). First batch covers the fixed FIT parity targets. Never compute validation classifier probabilities during extraction.',
        hook_check='Capture disabled: no features allocated or written; final hidden states bitexact before/after prehook registration. Hooks return None.',
        parity='Only fixed first256 clean FIT cache_ix: save exact bank F16 vectors, sample_ix and CPU/GPU original portable probabilities; maxabs<5e-4. No validation model probability is computed. Numerical parity, not evaluation or selection.',
        restart='Atomic NaN bank initialization and exact selected-row owner. Every committed batch has byteSHA, target/sample/prompt prefix hashes. Partial uncommitted rows are recomputed; initialized missing/truncated bank fails. Completion read-only.',
        cache_path=str(CACHE),minimum_shared_memory_reserve=RESERVE_BYTES,minimum_free_per_A40_before_load=MIN_GPU_FREE,
        prepared_dev_only=True,test_replies_read=False,test_probabilities_read=False,readout_training=False,
        source_replay=False,old_banks_deleted=False,primary_changed=False)


def verified_source_metadata():
    """Hash prepared dev cache and metadata only; never call dev.verify_dev()."""
    path=dev.OUT/'data-complete.json';require(data.digest(path)==DEV_COMPLETE_SHA256,'Fixed natural dev completion changed.')
    complete=read(path);protocol=read(dev.OUT/'protocol.json')
    require(complete['source_sha256']==data.digest(dev.__file__) and complete['protocol_sha256']==data.digest(dev.OUT/'protocol.json')
        and complete['cache_path']==str(dev.CACHE) and complete['fixed_root_counts']=={'fit':256,'validation':64}
        and complete['prepared_prompts']==320 and complete['content_rows']==80436 and complete['quality_failures']==0,
        'Prepared320 natural dev source identity changed.')
    require(set(complete['cache_sha256'])==set(dev.FILES)
        and set(complete['metadata_sha256'])=={'source-read-audit.json','dev-ledger.json','duplicate-audit.json','coverage.json'},
        'Prepared dev cache/audit map incomplete.')
    files={str(path):DEV_COMPLETE_SHA256,str(dev.OUT/'protocol.json'):complete['protocol_sha256']}
    for name,value in complete['cache_sha256'].items():
        require(data.digest(dev.CACHE/name)==value,'Prepared dev cache changed: '+name);files[str(dev.CACHE/name)]=value
    for name,value in complete['metadata_sha256'].items():
        require(data.digest(dev.OUT/name)==value,'Prepared dev audit changed: '+name);files[str(dev.OUT/name)]=value
    require(read(dev.CACHE/'initialized.json')==dict(protocol_sha256=complete['protocol_sha256'],source_sha256=complete['source_sha256'],
        dev_keys_sha256=complete['source_binding']['dev_keys_sha256']),'Prepared dev cache owner changed.')
    files[str(dev.CACHE/'initialized.json')]=data.digest(dev.CACHE/'initialized.json')
    require(complete['test_assistant_decoded'] is False and complete['test_transcripts_prepared'] is False
        and complete['classifier_probabilities_read'] is False and complete['gpu_work_run'] is False
        and complete['model_generation_run'] is False and complete['downloads_run'] is False
        and complete['coverage']['all_content_tokens_scored'] is True and complete['coverage']['independent_native_labels_match'] is True,
        'Prepared cache crosses dev-only/no-generation boundary.')
    # Explicit declared source dependencies, not the old dynamic source builder.
    for name,value in protocol['dependencies_sha256'].items():
        require(data.digest(data.HERE/name)==value,'Frozen dev dependency changed.');files[str(data.HERE/name)]=value
    return complete,files


def fixed_identity():
    """Original portable identity only; never call any selection verifier."""
    require(data.digest(OLD_MODEL)==OLD_MODEL_SHA256,'Fixed original portable head model changed.')
    head_path=data.runtime.OUT/'systemness-head-scan/head-selection.json'
    feature_path=portable.FEATURE_OUT/'extraction-complete.json'
    selection=read(head_path);feature=read(feature_path)
    require(selection['selected_head_ids']==feature['selected_head_ids']==HEAD_IDS,'Original selected head rank differs.')
    with np.load(SCALER,allow_pickle=False) as z:mean,scale=z['mean'].copy(),z['scale'].copy()
    with np.load(OLD_MODEL,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        require(meta['roles']==data.ROLES and meta['kind']=='mlp' and meta['hidden']==128
            and meta['dimensions']==512 and meta['selected_head_ids']==HEAD_IDS
            and np.array_equal(z['mean'],mean) and np.array_equal(z['scale'],scale),
            'Original portable geometry/scaler/head order differs.')
    checkpoint,tokenizer_sha=native.tokenizer_assets();config=read(checkpoint/'config.json')
    require(config['quantization_config']['quant_method']=='mxfp4'
        and (config['num_hidden_layers'],config['num_attention_heads'],config['head_dim'])==(36,64,64),
        'Native MXFP4 geometry differs.')
    expected=feature['signature']['scanner_signature']
    require(data.digest(checkpoint/'config.json')==expected['config_sha256']
        and data.digest(checkpoint/'verified-weights.json')==expected['verified_weights_sha256']
        and tokenizer_sha==expected['tokenizer_sha256'],'Checkpoint metadata/tokenizer differs from original extraction.')
    files={str(p):data.digest(p) for p in [OLD_MODEL,SCALER,head_path,feature_path,
        checkpoint/'config.json',checkpoint/'verified-weights.json',data.runtime.OUT/'checkpoint-cache-verification.json']}
    files.update({str(checkpoint/name):value for name,value in tokenizer_sha.items()})
    return dict(portable_model_path=str(OLD_MODEL),portable_model_sha256=OLD_MODEL_SHA256,
        selected_head_ids=HEAD_IDS,ordered_layers_heads=[[h//64,h%64] for h in HEAD_IDS],
        checkpoint_path=str(checkpoint),config_sha256=files[str(checkpoint/'config.json')],tokenizer_sha256=tokenizer_sha,
        verified_weights_sha256=files[str(checkpoint/'verified-weights.json')],files_sha256=files)


def source_signature():
    complete,files=verified_source_metadata();identity=fixed_identity()
    require(identity['tokenizer_sha256']==complete['source_binding']['source_signature']['tokenizer_sha256'],
        'Prepared dev tokenizer differs from original model tokenizer.')
    return dict(source_sha256=data.digest(__file__),dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        prepared_dev_files_sha256=files,fixed_identity=identity,protocol=protocol_spec(),
        package_versions={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','pandas']},
        kernel_revision=data.runtime.KERNEL_REVISION,cache_path=str(CACHE),source_cache_path=str(dev.CACHE))


def filter_layout(inputs,rows,expected=None):
    expected=expected or dict(source_prompts=320,source_rows=80436,prompts=PROMPTS,rows=ROWS,
        fit_prompts=FIT_PROMPTS,validation_prompts=VALID_PROMPTS,fit_rows=FIT_ROWS,validation_rows=VALID_ROWS,
        removed_fit_prompts=1,removed_fit_rows=320,role_counts=ROLE_COUNTS)
    require(len(inputs)==expected['source_prompts'] and len(rows)==expected['source_rows']
        and inputs.prompt_ix.tolist()==list(range(len(inputs))) and rows.cache_ix.tolist()==list(range(len(rows)))
        and rows.sample_ix.is_unique and set(inputs.split)==set(rows.split)=={'fit','validation'}
        and set(rows.role)<=set(ROLES) and rows.semantic_role.eq(rows.role).all(), 'Prepared dev row layout differs.')
    keep=(inputs.split.eq('fit')&inputs.eligible_fit.eq(True))|inputs.split.eq('validation')
    chosen=inputs[keep].copy();removed=inputs[~keep]
    require(removed.split.eq('fit').all() and len(removed)==expected['removed_fit_prompts']
        and chosen[chosen.split.eq('fit')].eligible_fit.all()
        and chosen[chosen.split.eq('validation')].eligible_clean_validation.all(), 'Quarantine/clean validation policy changed.')
    old_prompts=chosen.prompt_ix.to_numpy(dtype=np.int64);indices={int(old):new for new,old in enumerate(old_prompts)}
    selected=rows[rows.prompt_ix.isin(indices)].copy()
    require(len(rows)-len(selected)==expected['removed_fit_rows']
        and not set(selected.source_uid)&set(removed.source_uid),'Contaminated conversation content escaped quarantine.')
    chosen['original_prompt_ix']=chosen.prompt_ix;chosen['prompt_ix']=np.arange(len(chosen))
    selected['original_prompt_ix']=selected.prompt_ix;selected['original_cache_ix']=selected.cache_ix
    selected['prompt_ix']=selected.prompt_ix.map(indices);selected['cache_ix']=np.arange(len(selected))
    chosen=chosen.reset_index(drop=True);selected=selected.reset_index(drop=True)
    for frame in [chosen,selected]:
        frame['source_set']=SOURCE_SET;frame['uid']='natural-dev:'+frame.source_uid
    selected['body_offset']=selected.token_ix-selected.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    require(len(chosen)==expected['prompts'] and len(selected)==expected['rows']
        and chosen.split.value_counts().to_dict()=={'fit':expected['fit_prompts'],'validation':expected['validation_prompts']}
        and selected.split.value_counts().to_dict()=={'fit':expected['fit_rows'],'validation':expected['validation_rows']},
        'Clean319 input/row split counts differ.')
    actual={split:{role:int((selected.split.eq(split)&selected.role.eq(role)).sum()) for role in ROLES}
        for split in ['fit','validation']}
    require(actual==expected['role_counts'],'Selected native S/U/A role counts changed.')
    require(not set(selected[selected.split.eq('fit')].source_uid)&set(selected[selected.split.eq('validation')].source_uid)
        and selected.sample_ix.is_monotonic_increasing and selected.original_cache_ix.is_monotonic_increasing,
        'Fit/validation leakage or original sample order changed.')
    return chosen,selected,dict(removed_source_uids=removed.source_uid.tolist(),removed_original_prompt_ix=removed.prompt_ix.tolist(),
        removed_fit_body_rows=expected['removed_fit_rows'],role_counts=actual,source_body_token_spelling_read=False)


class Bundle:
    def __init__(self,inputs,rows,tokens=None):
        self.inputs,self.rows,self.tokens=inputs,rows,tokens
        self.lookup={int(p):g for p,g in rows.groupby('prompt_ix',sort=False)}
        self.parity_targets=rows.loc[rows.split.eq('fit'),'cache_ix'].to_numpy(dtype=np.int64)[:256]
        require(len(self.parity_targets)>0,'No clean fit rows for portable parity.')
        priority=rows.iloc[self.parity_targets].prompt_ix.drop_duplicates().to_numpy(dtype=np.int64).tolist()
        self.parity_prompt_ix=priority
        require(len(priority)<=BATCH,'Fixed256 FIT parity rows span more than one initial batch; review rather than use validation.')
        order=inputs.sort_values(['length','prompt_ix']).prompt_ix.to_numpy(dtype=np.int64)
        order=np.asarray(priority+[int(p) for p in order if int(p) not in set(priority)],dtype=np.int64)
        self.parts=[order[i:i+BATCH].tolist() for i in range(0,len(order),BATCH)]
        require(set(self.parity_targets)<=set(self.targets(self.parts[0])),'First batch omits fixed FIT parity rows.')
    def targets(self,prompts):return np.concatenate([self.lookup[p].cache_ix.to_numpy(dtype=np.int64) for p in prompts])
    def collate(self,prompts):require(self.tokens is not None,'Full prompts were not loaded.');return shared.collate(self,prompts)
    def descriptor(self):return dict(inputs=len(self.inputs),rows=len(self.rows),batches=len(self.parts),
        fit_prompts=int(self.inputs.split.eq('fit').sum()),validation_prompts=int(self.inputs.split.eq('validation').sum()),
        split_rows=self.rows.split.value_counts().to_dict(),
        input_mapping_sha256=frame_sha(self.inputs,['prompt_ix','original_prompt_ix','uid','length','split','eligible_fit']),
        row_mapping_sha256=frame_sha(self.rows,['cache_ix','original_cache_ix','sample_ix','prompt_ix','uid','token_ix','token_id','role','body_offset']),
        sample_ix_sha256=array_sha(self.rows.sample_ix.to_numpy()),
        parity_fit_rows=len(self.parity_targets),parity_fit_cache_ix_sha256=array_sha(self.parity_targets),
        parity_fit_sample_ix_sha256=array_sha(self.rows.iloc[self.parity_targets].sample_ix.to_numpy()),
        parity_fit_prompt_ix=self.parity_prompt_ix,
        batch_prompt_order_sha256=array_sha(np.asarray([p for part in self.parts for p in part],dtype=np.int64)),
        max_batch_rows=max(len(self.targets(part)) for part in self.parts))


def metadata_bundle():
    verified_source_metadata()
    inputs=pd.read_parquet(dev.CACHE/'inputs.parquet',columns=INPUT_COLUMNS)
    rows=pd.read_parquet(dev.CACHE/'rows.parquet',columns=ROW_COLUMNS)
    selected_inputs,selected_rows,quarantine=filter_layout(inputs,rows)
    return Bundle(selected_inputs,selected_rows),quarantine


def validate_native(bundle,tokenizer,ledger):
    by_id={r['source_uid']:r for r in ledger};pattern=re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>)',re.S)
    for entry in bundle.inputs.itertuples():
        group=bundle.lookup[entry.prompt_ix];ids=bundle.tokens[entry.prompt_ix]
        require(len(ids)==entry.length and dict(zip(group.token_ix,group.role))==native.parse_native_labels(ids,tokenizer)
            and all(ids[r.token_ix]==r.token_id for r in group.itertuples()), 'Prepared S/U/A native body labels/token order differ.')
        record=by_id[entry.source_uid];messages=pattern.findall(entry.prompt)
        roles=[m['role'] for m in record['messages']]
        require(roles in [['system','user','assistant'],['system','user','assistant','user','assistant']]
            and len(messages)==entry.message_count==len(roles),'Native genuine S/U/A path changed.')
        for index,(head,body,end) in enumerate(messages):
            role=roles[index];part=group[group.message_ix.eq(index)]
            require(head==('assistant<|channel|>final' if role=='assistant' else role)
                and end==('<|return|>' if role=='assistant' else '<|end|>')
                and data.text_digest(body)==record['messages'][index]['serialized_body_sha256']
                and len(part)==record['messages'][index]['body_tokens']<=192
                and part.body_offset.tolist()==list(range(len(part))), 'Native message body hash/cap/coverage changed.')
    return dict(all_native_SUA_content_positions=True,independent_native_labels_match=True,
        content_rows=len(bundle.rows),normal_system_tag=True,cot_tool_truth_support=0,old128_test_reply_decoded=False)


def load_data():
    bundle,quarantine=metadata_bundle()
    # Filter at Parquet scan before prompt strings reach Python. The token JSON
    # is hash-bound but not decoded; re-tokenize only the319 selected dev prompts.
    import pyarrow.dataset as ds
    wanted=bundle.inputs.original_prompt_ix.tolist()
    selected=ds.dataset(dev.CACHE/'inputs.parquet',format='parquet').to_table(
        filter=ds.field('prompt_ix').isin(wanted),columns=['prompt_ix','prompt']).to_pandas().set_index('prompt_ix')
    require(set(selected.index)==set(wanted),'Selected prepared prompt scope escaped319 IDs.')
    bundle.inputs['prompt']=[selected.loc[p,'prompt'] for p in wanted]
    tokenizer=native.load_tokenizer();bundle.tokens=[tokenizer.encode(p,add_special_tokens=False) for p in bundle.inputs.prompt]
    proof=validate_native(bundle,tokenizer,read(dev.OUT/'dev-ledger.json')['records'])
    return bundle,quarantine,proof


def plan_value():
    signature=source_signature();bundle,quarantine=metadata_bundle()
    return dict(source_sha256=signature['source_sha256'],dependencies_sha256=signature['dependencies_sha256'],
        signature=signature,protocol=protocol_spec(),data=bundle.descriptor(),quarantine=quarantine,
        selected_head_ids=HEAD_IDS,ordered_layers_heads=[[h//64,h%64] for h in HEAD_IDS],
        rows=ROWS,dimensions=DIMENSIONS,feature_shape=[ROWS,DIMENSIONS],dtype='float16',roles=ROLES)


def declare():
    value=plan_value()
    if not (OUT/PLAN_NAME).exists():
        require(not CACHE.exists() or not any(CACHE.iterdir()),'Bank state predates metadata declaration; inspect instead of reconstructing a retrospective plan.')
    OUT.mkdir(parents=True,exist_ok=True)
    immutable_json(OUT/PLAN_NAME,value)
    bundle,_=metadata_bundle();dev.save_parquet(OUT/'selected-rows.parquet',bundle.rows)
    with (OUT/PLAN_NAME).open('rb') as stream:os.fsync(stream.fileno())
    descriptor=os.open(OUT,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return value


def verify_plan(plan):
    require(read(OUT/PLAN_NAME)==plan==plan_value(),'Natural dev extraction source/selection/cache/model identity changed.')
    bundle,_=metadata_bundle()
    pd.testing.assert_frame_equal(pd.read_parquet(OUT/'selected-rows.parquet'),bundle.rows,check_exact=True)


def prefix_state(bundle,next_batch):
    require(type(next_batch) is int and 0<=next_batch<=len(bundle.parts),'Invalid committed batch prefix.')
    prompts=[p for part in bundle.parts[:next_batch] for p in part]
    targets=bundle.targets(prompts) if prompts else np.empty(0,np.int64)
    require(len(np.unique(targets))==len(targets),'Completed prefix repeats target rows.')
    return dict(next_batch=next_batch,rows_written=len(targets),target_cache_ix_sha256=array_sha(targets),
        source_sample_ix_sha256=array_sha(bundle.rows.iloc[targets].sample_ix.to_numpy()),
        prompt_order_sha256=frame_sha(bundle.inputs.iloc[prompts],['prompt_ix','original_prompt_ix','uid','length']))


def save_progress(bundle,plan_sha,hashes,parity):
    bank_helpers.fsync_file(CACHE/'features.f16')
    value=dict(plan_sha256=plan_sha,**prefix_state(bundle,len(hashes)),batch_features_sha256=hashes,portable=parity)
    data.atomic_json(CACHE/'progress.json',value);return value


def restore_progress(bundle,plan_sha,array):
    if not (CACHE/'progress.json').exists():return [],None
    value=read(CACHE/'progress.json');hashes=value['batch_features_sha256']
    require(value['plan_sha256']==plan_sha and all(value[key]==item for key,item in prefix_state(bundle,len(hashes)).items()),
        'Committed prompt/target/sample prefix changed.')
    for part,expected in zip(bundle.parts,hashes):require(bank_helpers.batch_digest(array,bundle.targets(part))==expected,'Committed feature bytes changed.')
    require(not hashes or value['portable'] is not None,'Committed feature prefix lacks same-vector parity.')
    if hashes:validate_parity_scope(bundle,value['portable'])
    return list(hashes),value['portable']


def validate_parity_scope(bundle,parity):
    require(parity['fit_only'] is True and parity['validation_probabilities_computed'] is False
        and parity['sampled_rows']==len(bundle.parity_targets)
        and parity['target_cache_ix_sha256']==array_sha(bundle.parity_targets)
        and parity['source_sample_ix_sha256']==array_sha(bundle.rows.iloc[bundle.parity_targets].sample_ix.to_numpy()),
        'Portable parity does not match predetermined clean FIT rows.')


@contextmanager
def checked_hooks(model,collector,bundle,signature,device='cuda:0'):
    import torch
    batch=bundle.collate(bundle.parts[0]);handles=[]
    with torch.inference_mode():
        before=model.model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),use_cache=False).last_hidden_state.cpu()
        try:
            for layer in collector.layers:handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)))
            collector.begin(batch['positions'],False)
            after=model.model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),use_cache=False).last_hidden_state.cpu();collector.finish()
            require(torch.equal(before,after) and torch.isfinite(after).all() and collector.features is None,
                'Disabled hooks changed hidden states or retained/wrote features.')
            data.atomic_json(OUT/'hook-validation.json',dict(signature=signature,hidden_states_bitexact=True,
                all_capture_writes_disabled=True,features_allocated=False,selected_head_ids=HEAD_IDS,sample_rows=len(batch['targets'])))
            del before,after
            yield
        finally:
            collector.release()
            for handle in handles:handle.remove()


def same_saved_vector_parity(array,targets,signature,bundle):
    import torch
    require(np.array_equal(targets,bundle.parity_targets) and bundle.rows.iloc[targets].split.eq('fit').all(),
        'Portable parity must use only the predetermined first256 clean FIT cache_ix, never validation.')
    count=len(targets);require(0<count<=256,'No saved FIT F16 vectors for parity.')
    saved=np.asarray(array[targets],dtype=np.float16).copy()
    net,mean,scale,_=portable.load_model(OLD_MODEL,'cpu');net.requires_grad_(False)
    with torch.inference_mode():
        cpu=net((torch.from_numpy(saved).float()-mean)/scale).exp().numpy()
        gpu=copy.deepcopy(net).to('cuda:0').eval()
        actual=gpu((torch.from_numpy(saved).to('cuda:0').float()-mean.cuda(0))/scale.cuda(0)).exp().cpu().numpy()
    difference=float(np.abs(cpu-actual).max());require(np.isfinite(difference) and difference<PARITY_TOLERANCE,'Saved F16 CPU/GPU portable probabilities differ.')
    path=OUT/'portable-sample.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:require(np.array_equal(z['features'],saved) and np.array_equal(z['target_cache_ix'],targets)
            and np.array_equal(z['sample_ix'],bundle.rows.iloc[targets].sample_ix.to_numpy()),'Existing portable FIT sample differs.')
    else:np.savez_compressed(path,features=saved,target_cache_ix=targets,
        sample_ix=bundle.rows.iloc[targets].sample_ix.to_numpy(),cpu_probabilities=cpu,gpu_probabilities=actual)
    result=dict(signature=signature,sampled_rows=count,same_saved_f16_vectors=True,captured_dtype='float16',
        portable_model_sha256=OLD_MODEL_SHA256,max_probability_difference=difference,tolerance=PARITY_TOLERANCE,
        sample_sha256=data.digest(path),features_sha256=array_sha(saved),no_performance_metric=True,
        fit_only=True,validation_probabilities_computed=False,target_cache_ix_sha256=array_sha(targets),
        source_sample_ix_sha256=array_sha(bundle.rows.iloc[targets].sample_ix.to_numpy()))
    immutable_json(OUT/'portable-validation.json',result);del net,gpu;return result


def no_competing_jobs():
    names={'systemness_natural_dev_extract.py','systemness_aug_confirmation_infer.py','systemness_long_augmentation_extract.py',
        'systemness_long_augmentation_probes.py','systemness_multiseed_ensemble.py','systemness_head_extract.py',
        'systemness_head_scan.py','systemness_head_probes.py','systemness_natural_test_confirm.py','systemness_head_confirm.py'}
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg):active.append(int(entry.name))
    require(not active,'Other experiment remains active; no natural extraction: '+str(active))


def verified_extraction():
    """Read-only full feature/prefix verification; never loads model or prompts."""
    plan=read(OUT/PLAN_NAME);verify_plan(plan);complete=read(OUT/'extraction-complete.json');plan_sha=data.digest(OUT/PLAN_NAME)
    require(complete['plan_sha256']==plan_sha and complete['signature']==plan['signature'] and complete['source_sha256']==data.digest(__file__)
        and complete['rows']==ROWS and complete['feature_shape']==[ROWS,DIMENSIONS] and complete['dimensions']==512
        and complete['dtype']=='float16' and complete['selected_head_ids']==HEAD_IDS and complete['all_rows_written'] is True
        and complete['all_finite'] is True and complete['tests_read'] is False and complete['readout_training'] is False,
        'Natural dev feature completion differs from declaration.')
    require(complete['features_path']==str(CACHE/'features.f16') and complete['rows_path']==str(CACHE/'rows.parquet')
        and complete['rows_sha256']==data.digest(CACHE/'rows.parquet')==data.digest(OUT/'selected-rows.parquet')
        and complete['features_sha256']==data.digest(CACHE/'features.f16'),'Completed bank paths/bytes changed.')
    require(read(CACHE/'initialized.json')==dict(plan_sha256=plan_sha,rows=ROWS,dimensions=DIMENSIONS,dtype='float16',rows_sha256=complete['rows_sha256'])
        and (CACHE/'features.f16').stat().st_size==ROWS*DIMENSIONS*2,'Feature ownership/size changed.')
    bundle,_=metadata_bundle();array=np.memmap(CACHE/'features.f16',mode='r',dtype=np.float16,shape=(ROWS,DIMENSIONS))
    hashes,parity=restore_progress(bundle,plan_sha,array)
    require(len(hashes)==len(bundle.parts) and prefix_state(bundle,len(hashes))['rows_written']==ROWS
        and parity==read(OUT/'portable-validation.json')==complete['portable']
        and parity['sample_sha256']==data.digest(OUT/'portable-sample.npz')
        and parity['max_probability_difference']<PARITY_TOLERANCE,'Completed exact prefix/parity missing.')
    validate_parity_scope(bundle,parity)
    require(len(bundle.parity_targets)==256,'Completed FIT parity sample count changed.')
    require(complete['progress_sha256']==data.digest(CACHE/'progress.json')
        and set(complete['audit_sha256'])=={'hook-validation.json','portable-validation.json','portable-sample.npz','model_backend.json'},
        'Completion omitted required prefix/backend/portable audit files.')
    for name,value in complete['audit_sha256'].items():require(data.digest(OUT/name)==value,'Completed audit changed: '+name)
    require(bank_helpers.inspect_features(array)=={key:complete[key] for key in ['all_finite','all_rows_nonzero','zero_row_count']},'Completed finite/zero diagnostics changed.')
    del array;return complete


def extract():
    no_competing_jobs();require((OUT/PLAN_NAME).is_file(),'Explicit metadata declaration required before extract.')
    plan=read(OUT/PLAN_NAME);verify_plan(plan)
    if (OUT/'extraction-complete.json').exists():return verified_extraction()
    import fcntl
    with (OUT/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another natural-dev extractor holds this output.') from error
        bundle,quarantine,proof=load_data();require(bundle.descriptor()==plan['data'] and quarantine==plan['quarantine'],'Loaded319 data differs from metadata declaration.')
        require(shutil.disk_usage(CACHE.parent).free>=ROWS*DIMENSIONS*2+RESERVE_BYTES if not (CACHE/'initialized.json').exists()
            else shutil.disk_usage(CACHE.parent).free>=RESERVE_BYTES,'Insufficient shared memory; do not delete old banks.')
        plan_sha=data.digest(OUT/PLAN_NAME)
        array=bank_helpers.initialize(CACHE,OUT/'selected-rows.parquet',plan_sha,rows=ROWS,dimensions=DIMENSIONS)
        hashes,parity=restore_progress(bundle,plan_sha,array)
        if len(hashes)==len(bundle.parts):raise RuntimeError('Full feature prefix without extraction completion; inspect rather than fabricate backend completion.')
        import torch
        require(torch.cuda.device_count()==4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)), 'Exactly4 A40 required.')
        require(min(torch.cuda.mem_get_info(i)[0] for i in range(4))>=MIN_GPU_FREE,'Fewer22GiB free per A40.')
        os.environ['ROLE_ANALYSIS_LOAD_PATH']=plan['signature']['fixed_identity']['checkpoint_path']
        runtime=data.runtime;old_atomic,old_status=runtime.atomic_json,runtime.status
        runtime.atomic_json=lambda path,value:data.atomic_json(OUT/Path(path).name,value)
        runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
        try:model=runtime.load_model()
        finally:runtime.atomic_json,runtime.status=old_atomic,old_status
        require(not model.training and not any(p.requires_grad for p in model.parameters()),'Native LM must remain frozen/eval.')
        collector=shared.Collector(HEAD_IDS)
        try:
            with torch.inference_mode(),checked_hooks(model,collector,bundle,plan['signature']):
                for bi,part in enumerate(bundle.parts):
                    if bi<len(hashes):continue
                    batch=bundle.collate(part);collector.begin(batch['positions'],True)
                    output=model.model(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0),use_cache=False).last_hidden_state
                    del output;collector.finish();array[batch['targets']]=collector.features;collector.release()
                    if parity is None:parity=same_saved_vector_parity(array,bundle.parity_targets,plan['signature'],bundle)
                    hashes.append(bank_helpers.batch_digest(array,batch['targets']))
                    if (bi+1)%CHECKPOINT_INTERVAL==0 or bi+1==len(bundle.parts):
                        verify_plan(plan);array.flush();save_progress(bundle,plan_sha,hashes,parity)
                        data.atomic_json(OUT/'status.json',dict(stage='natural-dev-extraction',batch=bi+1,batches=len(bundle.parts),rows=prefix_state(bundle,bi+1)['rows_written']))
            verify_plan(plan);inspection=bank_helpers.inspect_features(array);bank_helpers.fsync_file(CACHE/'features.f16')
            hooks=read(OUT/'hook-validation.json');backend=read(OUT/'model_backend.json')
            require(hooks['hidden_states_bitexact'] and hooks['all_capture_writes_disabled'] and backend['expert_backends']==['Mxfp4GptOssExperts'],
                'Actual disabled-hook/native-MXFP4 proof missing.')
            complete=dict(signature=plan['signature'],source_sha256=data.digest(__file__),dependencies_sha256=plan['dependencies_sha256'],plan_sha256=plan_sha,
                rows=ROWS,dimensions=DIMENSIONS,feature_shape=[ROWS,DIMENSIONS],dtype='float16',head_dim=64,selected_head_ids=HEAD_IDS,
                ordered_layers_heads=[[h//64,h%64] for h in HEAD_IDS],features_path=str(CACHE/'features.f16'),features_sha256=data.digest(CACHE/'features.f16'),
                rows_path=str(CACHE/'rows.parquet'),rows_sha256=data.digest(CACHE/'rows.parquet'),data=plan['data'],quarantine=quarantine,
                **inspection,all_rows_written=True,portable=parity,coverage=proof,
                audit_sha256={name:data.digest(OUT/name) for name in ['hook-validation.json','portable-validation.json','portable-sample.npz','model_backend.json']},
                progress_sha256=data.digest(CACHE/'progress.json'),tests_read=False,readout_training=False,primary_changed=False)
            immutable_json(OUT/'extraction-complete.json',complete);return verified_extraction()
        finally:collector.release();del model,array;gc.collect();torch.cuda.empty_cache()


@contextmanager
def patched(**values):
    original={name:globals()[name] for name in values};globals().update(values)
    try:yield
    finally:globals().update(original)


def preflight():
    """Metadata projection plus small CPU fixtures; no real prompt/model/p load."""
    folder=Path(tempfile.mkdtemp(prefix='systemness-natural-dev-extract-preflight-',dir='/tmp'));checks=[]
    source_sha=data.digest(__file__);formal=(OUT.exists(),CACHE.exists())
    def check(name,value=True):require(value,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,FileNotFoundError,KeyError,ValueError,AssertionError):checks.append(name)
        else:raise RuntimeError('Fault accepted: '+name)
    bundle,quarantine=metadata_bundle();identity=fixed_identity()
    check('Actual metadata-only255fit64val80116rows exact role counts',len(bundle.inputs)==319 and len(bundle.rows)==80116
        and quarantine['removed_fit_body_rows']==320 and quarantine['role_counts']==ROLE_COUNTS)
    check('Original sample order preserved with quarantine gap',bundle.rows.sample_ix.is_monotonic_increasing
        and np.array_equal(bundle.rows.cache_ix,np.arange(ROWS)) and bundle.rows.original_cache_ix.max()==80435)
    check('No val/test in fit; whole contaminated conversation absent',bundle.inputs[bundle.inputs.split.eq('fit')].eligible_fit.all()
        and not set(bundle.rows.source_uid)&set(quarantine['removed_source_uids']) and set(bundle.rows.split)=={'fit','validation'})
    check('Fixed original model/heads only',identity['portable_model_sha256']==OLD_MODEL_SHA256 and identity['selected_head_ids']==HEAD_IDS)
    check('Portable parity fixed first256clean FIT, firstbatch covered and no val',len(bundle.parity_targets)==256
        and bundle.rows.iloc[bundle.parity_targets].split.eq('fit').all()
        and set(bundle.parity_targets)<=set(bundle.targets(bundle.parts[0])))
    reject('Val parity targets rejected before any model prediction',lambda:same_saved_vector_parity(
        np.empty((0,512),np.float16),bundle.rows[bundle.rows.split.eq('validation')].cache_ix.to_numpy()[:256],{},bundle))
    import torch
    # Small actual native serialization and parser fixture, with one quarantined
    # fit conversation; no existing dev prompt string or public reply is loaded.
    toy_inputs=[];toy_rows=[];toy_tokens=[];ledger=[];tokenizer=native.load_tokenizer()
    for index,split in enumerate(['fit','fit','validation']):
        messages=[('system','A fabricated concise instruction.'),('user',f'A synthetic question number {index}.'),('assistant','A fabricated reply.')]
        prompt,spans=dev.render_native(messages);ids,content=dev.content_rows(prompt,spans,tokenizer)
        common=dict(source_uid=f'fixture:{index}',group_id=f'fixture:{index}',split=split,eligible_fit=split=='fit' and index==0,
            eligible_clean_validation=split=='validation',wrapper='native',profile='natural-public-sua')
        toy_inputs.append(dict(common,prompt_ix=index,length=len(ids),message_count=3,turns=1,prompt=prompt))
        for row in content:toy_rows.append(dict(common,cache_ix=len(toy_rows),sample_ix=4_000_000_000_000+len(toy_rows),prompt_ix=index,**row))
        ledger.append(dict(source_uid=common['source_uid'],messages=[dict(role=role,serialized_body_sha256=data.text_digest(body),body_tokens=len(tokenizer.encode(body,add_special_tokens=False))) for role,body in messages]))
        toy_tokens.append(ids)
    a=pd.DataFrame(toy_inputs);b=pd.DataFrame(toy_rows);role_counts={s:{r:int((b.split.eq(s)&b.role.eq(r)&b.source_uid.ne('fixture:1')).sum()) for r in ROLES} for s in ['fit','validation']}
    expected=dict(source_prompts=3,source_rows=len(b),prompts=2,rows=int(b.source_uid.ne('fixture:1').sum()),fit_prompts=1,validation_prompts=1,
        fit_rows=int(b.source_uid.eq('fixture:0').sum()),validation_rows=int(b.source_uid.eq('fixture:2').sum()),removed_fit_prompts=1,
        removed_fit_rows=int(b.source_uid.eq('fixture:1').sum()),role_counts=role_counts)
    ti,tr,_=filter_layout(a,b,expected);toy=Bundle(ti,tr,[toy_tokens[p] for p in ti.original_prompt_ix])
    # Deliberately non-row-order prefix, as length sorting can cause in real
    # extraction; a cursor must bind actual prompt/sample targets, not first N.
    toy.parts=[[1],[0]]
    validate_native(toy,tokenizer,ledger);check('Toy native SUA labels/body/sample order preserved',set(toy.rows.role)==set(ROLES))
    with patched(CACHE=folder/'bank'):
        source_rows=folder/'rows.parquet';dev.save_parquet(source_rows,toy.rows)
        array=bank_helpers.initialize(CACHE,source_rows,'toy-plan',rows=len(toy.rows),dimensions=512)
        check('NaN initialization has no fabricated rows',np.isnan(array).all())
        array[:2]=0;hashes,parity=restore_progress(toy,'toy-plan',array)
        check('Uncommitted halfbatch does not advance',hashes==[] and parity is None)
        targets=toy.targets(toy.parts[0]);array[targets]=np.arange(len(targets)*512,dtype=np.float32).reshape(len(targets),512)/128
        digest=bank_helpers.batch_digest(array,targets)
        toy_parity=dict(synthetic_only=True,fit_only=True,validation_probabilities_computed=False,sampled_rows=len(toy.parity_targets),
            target_cache_ix_sha256=array_sha(toy.parity_targets),source_sample_ix_sha256=array_sha(toy.rows.iloc[toy.parity_targets].sample_ix.to_numpy()))
        save_progress(toy,'toy-plan',[digest],toy_parity)
        check('Committed prefix roundtrip',restore_progress(toy,'toy-plan',array)[0]==[digest])
        reject('Wrong plan owner rejected',lambda:restore_progress(toy,'wrong-plan',array))
        array[targets[0],0]+=1
        reject('Committed feature byte tamper rejected',lambda:restore_progress(toy,'toy-plan',array))
        array[targets[0],0]=np.nan
        reject('Committed singlecolumn NaN rejected',lambda:restore_progress(toy,'toy-plan',array))
        del array;(CACHE/'features.f16').unlink()
        reject('Missing initialized bank not recreated',lambda:bank_helpers.initialize(CACHE,source_rows,'toy-plan',rows=len(toy.rows),dimensions=512))
    # Shared collector head ordering and disabled-write bitexact CPU fixture.
    collector=shared.Collector(HEAD_IDS);positions=np.array([1,7],np.int64);source=torch.arange(2*4*4096,dtype=torch.float32).reshape(2,4,4096)/256
    collector.begin(positions,False)
    for layer in collector.layers:require(collector.hook(layer)(None,(source,)) is None,'Hook altered input.')
    collector.finish();check('Disabled collector allocates/writes nothing',collector.features is None)
    collector.begin(positions,True)
    for layer in collector.layers:collector.hook(layer)(None,(source+layer,))
    collector.finish()
    expected_features=np.concatenate([(source+int(h//64)).reshape(-1,64,64)[positions,int(h%64)].numpy().astype(np.float16) for h in HEAD_IDS],axis=1)
    check('Ordered512F16 blocks match independent selected-head gather',np.array_equal(collector.features,expected_features))
    from torch import nn
    class Layer(nn.Module):
        def __init__(self):super().__init__();self.self_attn=SimpleNamespace(o_proj=nn.Identity())
    class Body(nn.Module):
        def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer() for _ in range(36)])
        def forward(self,input_ids,attention_mask,use_cache=False):
            value=input_ids.float().unsqueeze(-1)+torch.arange(4096).float()[None,None,:]/1024
            for index,layer in enumerate(self.layers):value=layer.self_attn.o_proj(value+index/128)
            return SimpleNamespace(last_hidden_state=value)
    fake=SimpleNamespace(model=Body());audit_collector=shared.Collector(HEAD_IDS)
    with patched(OUT=folder/'hook-audit'):
        OUT.mkdir()
        with checked_hooks(fake,audit_collector,toy,dict(synthetic_only=True),device='cpu'):
            require(audit_collector.features is None,'Disabled toy hook retained features.')
        proof=read(OUT/'hook-validation.json')
        check('Actual checked_hooks toy hiddenstates bitexact and exception-safe cleanup',proof['hidden_states_bitexact']
            and proof['all_capture_writes_disabled'] and all(not fake.model.layers[layer].self_attn.o_proj._forward_pre_hooks for layer in audit_collector.layers))
    batch=toy.collate(toy.parts[0])
    for target,position in zip(batch['targets'],batch['positions']):
        require(int(batch['input_ids'].reshape(-1)[position])==int(toy.rows.iloc[target].token_id),'Leftpad token mapping differs.')
    check('Leftpadding exact content targets/token identities')
    check('Metadata/preflight do not create formal state',formal==(OUT.exists(),CACHE.exists()) and data.digest(__file__)==source_sha)
    report=dict(source_sha256=source_sha,checks=checks,checks_passed=len(checks),metadata_descriptor=bundle.descriptor(),quarantine=quarantine,
        fixed_identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),bank_bytes=ROWS*DIMENSIONS*2,
        prepared_dev_prompt_text_materialized=False,old128_test_reply_materialized=False,public_source_replayed=False,
        classifier_probability_artifact_read=False,real_model_loaded=False,gpu_work_run=False,formal_declaration_run=False,formal_extraction_run=False)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report_path=str(path),report_sha256=data.digest(path),**report)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='Projected dev metadata + synthetic CPU/tmp fixtures only; no real model/prompt/probability.')
    action.add_argument('--stage',choices=['declare','extract','verify'],help='Explicit metadata declaration or root-only dev extraction/completion verification.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.stage=='declare':declare();print(json.dumps(dict(plan_sha256=data.digest(OUT/PLAN_NAME))),flush=True)
    elif args.stage=='extract':print(json.dumps(extract()),flush=True)
    elif args.stage=='verify':print(json.dumps(verified_extraction()),flush=True)
    else:parser.print_help()


if __name__=='__main__':main()
