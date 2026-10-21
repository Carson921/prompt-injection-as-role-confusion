"""Conditional two-head append-only feature bank, frozen native MXFP4.

Default help. --preflight uses synthetic CPU/tmp only. --stage declare/extract
require hard-pinned completed scanner and independent actual audit; those pins
are deliberately unknown until root reviews the scan. Only two new64-dim heads
are persisted; original512 banks remain separate and unchanged. No readout/test.
"""
import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import systemness_data as data
import systemness_long_data as native
import systemness_head_extract as legacy
import systemness_head_scan as original
import systemness_complementary_head_scan as scanner

OUT=data.runtime.OUT/'systemness-complementary-head-features'
CACHE=Path('/dev/shm/role-analysis-complementary-heads')
SCAN_SOURCE_SHA='70db2015b90a496dc465c63631911da81d641614fcf0b1d48b04313a078ce156'
SCAN_COMPLETE_SHA='dd538fb5d3cc20d6652ca2fd214bbadd7477d124978b50809561673e34de4b95'
SCAN_SELECTION_SHA='77563f6dfacab0c7bf42a72090e64496bd4029d2143b2c9f21f13472f5762551'
AUDIT_PATH=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/local-gptoss120b/systemness-complementary-head-scan/actual-audit/actual-scan-audit.json')
AUDIT_SHA='a71c052c38947b9c2a4f1ab73f58a3414533b43e013bac624322b0b836e67ff0'
ROWS,DIMENSIONS,HEAD_DIM=896838,128,64
COUNTS={'old':755282,'c4':61440,'natural':80116}
ORIGINAL=[458,462,2045,460,459,2047,456,997]
ADDED=[247,2248]
ORIGINAL_MODEL_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
FEATURE_COMPLETE_SHA={'old':'b970591a6c82d8818bec589b58e3e96ce65e5d8a6daa4d0710c42b2e909abc89','c4':'ef266142d54ccd1ccd61c93fb08e7a8f7484702e874c3f508d7cac2ca958f6f7','natural':'4f020979dc1e337924d6f8ef6b7a576fd36a7f629c72a85431b8caf68d01f9f4'}
NEW_DATA_SHA='6fe07499112eb816fb116905d7db245211297ec07b99cc45df5ec7be8efb847e'
NAT_DATA_SHA='b6c411c920f583a128401c0d787869e2eba17b632136a5cec9613fd227bb9dfd'
BATCH,INTERVAL=4,16
PARITY_TOLERANCE=5e-4
DEPENDENCIES=['systemness_complementary_head_extract.py','systemness_complementary_head_scan.py','systemness_head_extract.py','systemness_head_scan.py','systemness_long_data.py','systemness_data.py','systemness_probes.py','systemness_long_probes.py','run_local_gptoss120b.py']
require=native.require
require_checkpoint_path=scanner.require_checkpoint_path
read=legacy.read
array_sha=legacy.array_digest
frame_sha=legacy.frame_digest
NAT_CACHE=data.CACHE/'natural-transcripts-dev'
NAT_BANK=Path('/dev/shm/role-analysis-natural-dev-heads')
NEW_BANK=Path('/dev/shm/role-analysis-long-augmentation-heads')
OLD_BANK=Path('/dev/shm/role-analysis-top8-heads')


def protocol():
    return dict(version=1,kind='conditional-added-two-head-feature-bank',original_head_ids=ORIGINAL,
        prerequisite='Admissible exact2 complementary heads; completed scan and independent actual audit pins before any source text/model.',
        ordered_rows='Old755282 + existing newC4 VALIDATION61440 + clean-natural80116. Keep parent_bank,parent_cache_ix and original sample_ix; new consecutive cache_ix.',
        shape=[ROWS,DIMENSIONS],dtype='float16',feature_bytes=ROWS*DIMENSIONS*2,heads_persisted=2,head_dim=64,
        old512='No duplicate512 bank; fixed first256 original long-full FIT rows only are transiently captured for parity, then released.',
        parity='Deterministic first256 old long-full FIT rows, never chosen by probabilities; original8 transient/reference and sameF16 NumPy/GPU LDA parity, extra2 savedF16 parity; no validation probabilities.',
        checkpoint='Owned atomic NaN initialization; flush/fsync exact complete-batch prefix; independent all128-column batch byte SHA; uncommitted partial writes overwritten; missing/wrong-size files fail.',
        batch_size=4,interval=16,native_mxfp4=True,gpus=4,old_banks_deleted=False,readout_training=False,
        source_reader='Selected prompt IDs AND native wrapper AND fit/validation predicates before Arrow text materialization; C4augmentation validation-only. Mixed token JSON never decoded.',
        tests_read=False,primary_changed=False)


def immutable(path,value):
    if path.exists():require(read(path)==value,'Immutable extra-head artifact changed: '+str(path))
    else:data.atomic_json(path,value)


def gate():
    require(all(v is not None for v in [SCAN_COMPLETE_SHA,SCAN_SELECTION_SHA,AUDIT_PATH,AUDIT_SHA]),'Reviewed admissible scanner/audit pins unknown: actual preparation forbidden.')
    require(data.digest(scanner.__file__)==SCAN_SOURCE_SHA and data.digest(scanner.OUT/'scan-complete.json')==SCAN_COMPLETE_SHA
        and data.digest(scanner.OUT/'head-selection.json')==SCAN_SELECTION_SHA and data.digest(AUDIT_PATH)==AUDIT_SHA,'Frozen scanner or independent audit certificate changed.')
    complete=read(scanner.OUT/'scan-complete.json');selection=read(scanner.OUT/'head-selection.json');audit=read(AUDIT_PATH)
    require(audit['stage']=='actual_completed_complementary_head_scan_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False
        and audit['admissible'] is True and audit['scanner_source_sha256']==SCAN_SOURCE_SHA and audit['scanner_complete_sha256']==SCAN_COMPLETE_SHA
        and audit['head_selection_sha256']==SCAN_SELECTION_SHA and complete['admissible'] is True and complete['tests_read'] is False,'No audited admissible2head result.')
    selected=selection['selected_head_ids'];extra=selected[8:] if selected is not None else []
    require(selected==complete['selected_head_ids']==audit['selected_head_ids'] and selected[:8]==ORIGINAL and len(selected)==10 and len(set(selected))==10
        and extra==ADDED and len(extra)==2 and extra==sorted(extra) and all(0<=h<2304 and h not in ORIGINAL for h in extra),'Exact2 ascending complementary head identity changed.')
    files={str(scanner.OUT/'scan-complete.json'):SCAN_COMPLETE_SHA,str(scanner.OUT/'head-selection.json'):SCAN_SELECTION_SHA,str(AUDIT_PATH):AUDIT_SHA}
    for name,want in complete['artifact_sha256'].items():
        path=scanner.OUT/name;require(data.digest(path)==want,'Closed scan artifact changed: '+name);files[str(path)]=want
    require(data.digest(scanner.OUT/'plan.json')==complete['plan_sha256']==audit['scanner_plan_sha256'],'Scanner plan/audit closure changed.')
    plan=read(scanner.OUT/'plan.json');scanner.require_checkpoint_path(plan['signature']['checkpoint_path'])
    # The actual auditor attests the large discovery/moment cache; no head p
    # values are consumed by this extractor and no global verifier is called.
    for path,want in plan['signature']['closed_files_sha256'].items():
        require(data.digest(path)==want,'Scanner explicit input dependency changed: '+path);files[path]=want
    return dict(selected_head_ids=selected,added_head_ids=extra,files_sha256=files,checkpoint_path=plan['signature']['checkpoint_path'],
        kernel_revision=plan['signature']['kernel_revision'],scanner_plan_sha256=complete['plan_sha256'],audit_sha256=AUDIT_SHA)


def source_metadata():
    """Own explicit native development source list; no historical verifier."""
    sources={};files={}
    old_complete_path=data.runtime.OUT/'systemness-head-features/extraction-complete.json'
    require(data.digest(old_complete_path)==FEATURE_COMPLETE_SHA['old'],'Original audited feature-completion pin changed.')
    old_complete=read(old_complete_path);old_sources=old_complete['signature']['input_sources']
    entries=[('old:main',data.CACHE,data.OUT/'data-complete.json'),('old:supplement',data.CACHE/'supplement',data.OUT/'supplement-data-complete.json'),
        ('old:long-full',native.CACHE,native.OUT/'data-complete.json'),('c4',data.CACHE/'long-augmentation',data.runtime.OUT/'systemness-long-augmentation/data-complete.json'),
        ('natural',NAT_CACHE,data.runtime.OUT/'systemness-natural-transcripts/data-complete.json')]
    for name,folder,path in entries:
        meta=read(path);files[str(path)]=data.digest(path)
        expected=(old_sources[name[4:]]['metadata_sha256'] if name.startswith('old:') else NEW_DATA_SHA if name=='c4' else NAT_DATA_SHA)
        require(files[str(path)]==expected,'Prepared source completion differs from previously audited parent bank: '+name)
        for fn in ['inputs.parquet','rows.parquet']:
            require(data.digest(folder/fn)==meta['cache_sha256'][fn],'Prepared source cache changed: '+name+'/'+fn);files[str(folder/fn)]=meta['cache_sha256'][fn]
        sources[name]=dict(cache=str(folder),metadata_sha256=files[str(path)])
    for name,folder,done_path in [('old',OLD_BANK,data.runtime.OUT/'systemness-head-features/extraction-complete.json'),
        ('c4',NEW_BANK,data.runtime.OUT/'systemness-long-augmentation/extraction-complete.json'),('natural',NAT_BANK,data.runtime.OUT/'systemness-natural-dev-features/extraction-complete.json')]:
        require(data.digest(done_path)==FEATURE_COMPLETE_SHA[name],'Parent feature-completion pin changed: '+name)
        done=read(done_path);require(done['all_rows_written'] is True and done['all_finite'] is True and done['selected_head_ids']==ORIGINAL,'Existing8head bank ownership changed.')
        require(data.digest(folder/'rows.parquet')==done['rows_sha256'],'Existing head rows changed.');files[str(done_path)]=data.digest(done_path);files[str(folder/'rows.parquet')]=done['rows_sha256']
    original_model=data.runtime.OUT/'systemness-head-probes/models/joint-mlp128.npz';require(data.digest(original_model)==ORIGINAL_MODEL_SHA,'Original512 head readout identity changed.');files[str(original_model)]=ORIGINAL_MODEL_SHA
    # Original8 LDA arrays are only a FIT portability consumer, not new scores.
    for path,want in [(original.OUT/'models/top8-lda.npz',old_complete['signature']['selection']['model_sha256']),(original.OUT/'head-selection.json',old_complete['signature']['selection']['head_selection_sha256'])]:
        require(data.digest(path)==want,'Original8 FIT portability identity changed.');files[str(path)]=want
    require((OLD_BANK/'features.f16').stat().st_size==755282*512*2,'Original512 parent bank truncated; fail before model loading.')
    return sources,files


def project_rows(old,c4,natural,*,expected=COUNTS):
    pieces=[]
    for name,source in [('old',old),('c4',c4),('natural',natural)]:
        frame=source.copy();require(len(frame)==expected[name] and frame.cache_ix.is_unique and frame.sample_ix.is_unique and frame.wrapper.eq('native').all(),'Bad parent native bank: '+name)
        require(frame.split.isin(['fit','validation']).all() and (name!='c4' or frame.split.eq('validation').all()),'Test or new C4fit escaped extraction scope.')
        if name=='natural':require(set(frame.role)<={'system','user','assistant'} and frame.source_set.eq('natural-dev-clean').all(),'Invented natural role/source.')
        frame['parent_bank']=name;frame['parent_cache_ix']=frame.cache_ix
        frame['source_prompt_ix']=frame.original_prompt_ix if name=='natural' else frame.prompt_ix
        frame['input_source']=('old:'+frame.source_set if name=='old' else name)
        frame['uid']=frame.input_source+':'+frame.source_prompt_ix.astype(str)
        pieces.append(frame)
    rows=pd.concat(pieces,ignore_index=True);rows['cache_ix']=np.arange(len(rows),dtype=np.int64)
    require(rows.sample_ix.is_unique,'Cross-bank sample namespace collision.')
    return rows


class Bundle(legacy.Bundle):
    def __init__(self,inputs,rows):
        self.rows=rows.copy().reset_index(drop=True);self.parity_targets=self.rows.loc[self.rows.parent_bank.eq('old')&self.rows.split.eq('fit')&self.rows.source_set.eq('long-full')&self.rows.profile.eq('long-full'),'cache_ix'].to_numpy(dtype=np.int64)[:256]
        require(len(self.parity_targets)==256,'Fixed original long-full FIT parity scope absent.')
        priority=self.rows.iloc[self.parity_targets].uid.drop_duplicates().tolist();require(len(priority)<=BATCH,'Fixed FIT parity spans more than one batch; do not substitute validation.')
        ordered=inputs.sort_values(['length','uid']);self.inputs=pd.concat([ordered[ordered.uid.isin(priority)].sort_values('uid'),ordered[~ordered.uid.isin(priority)]]).reset_index(drop=True)
        require(self.inputs.uid.is_unique and set(self.inputs.uid)==set(self.rows.uid),'Unified extra-bank prompt ownership changed.')
        self.lookup={u:g for u,g in self.rows.groupby('uid',sort=False)};self.tokens=None
        self.parts=[self.inputs.iloc[i:i+BATCH] for i in range(0,len(self.inputs),BATCH)]
        require(set(self.parity_targets)<=set(self.targets(self.parts[0])),'First batch excludes predetermined FIT parity.')
    def descriptor(self):
        return dict(rows=len(self.rows),prompts=len(self.inputs),batches=len(self.parts),input_tokens=int(self.inputs.length.sum()),
            row_mapping_sha256=frame_sha(self.rows,['cache_ix','parent_bank','parent_cache_ix','sample_ix','uid','token_ix','token_id','role','split']),
            input_mapping_sha256=frame_sha(self.inputs,['uid','source_prompt_ix','input_source','length','split']),sample_ix_sha256=array_sha(self.rows.sample_ix.to_numpy()),
            parity_fit_targets_sha256=array_sha(self.parity_targets),parity_fit_sample_ix_sha256=array_sha(self.rows.iloc[self.parity_targets].sample_ix.to_numpy()),parity_prompts=self.rows.iloc[self.parity_targets].uid.drop_duplicates().tolist())


def metadata_bundle(sources):
    old=pd.read_parquet(OLD_BANK/'rows.parquet');c4=pd.read_parquet(NEW_BANK/'rows.parquet',filters=[('split','=','validation')]);natural=pd.read_parquet(NAT_BANK/'rows.parquet')
    require(len(old)==755282 and len(c4)==61440 and len(natural)==80116,'Expected original755282+newC4val61440+clean80116.')
    rows=project_rows(old,c4,natural);frames=[]
    for name,source in sources.items():
        wanted=rows[rows.input_source.eq(name)];frame=pd.read_parquet(Path(source['cache'])/'inputs.parquet',columns=['prompt_ix','length','split','wrapper'],filters=[('split','in',['validation'] if name=='c4' else ['fit','validation']),('wrapper','=','native')])
        frame=frame[frame.prompt_ix.isin(wanted.source_prompt_ix)].copy();require(frame.prompt_ix.is_unique and set(frame.prompt_ix)==set(wanted.source_prompt_ix),'Source metadata input IDs changed: '+name)
        lookup=frame.set_index('prompt_ix');require(np.array_equal(lookup.loc[wanted.source_prompt_ix,'split'],wanted.split),'Parent row/prompt split changed.')
        frame['source_prompt_ix']=frame.prompt_ix;frame['input_source']=name;frame['uid']=name+':'+frame.prompt_ix.astype(str);frames.append(frame)
    return Bundle(pd.concat(frames,ignore_index=True),rows)


def signature():
    selected=gate();sources,files=source_metadata();return dict(source_sha256=data.digest(__file__),dependencies_sha256={n:data.digest(data.HERE/n) for n in DEPENDENCIES},
        selection=selected,source_files_sha256=files,sources=sources,checkpoint_path=selected['checkpoint_path'],kernel_revision=selected['kernel_revision'],packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','pandas','pyarrow']})


def declare():
    sig=signature();bundle=metadata_bundle(sig['sources']);value=dict(protocol=protocol(),signature=sig,data=bundle.descriptor(),added_head_ids=sig['selection']['added_head_ids'],rows=ROWS,dimensions=128,feature_shape=[ROWS,128],dtype='float16')
    if not (OUT/'plan.json').exists():require(not CACHE.exists(),'Extra-bank state predates declaration.')
    OUT.mkdir(parents=True,exist_ok=True);immutable(OUT/'plan.json',value);return value


def verify_plan(plan):
    require(plan['protocol']==protocol() and plan['signature']==signature(),'Conditional source/selection/audit changed.');bundle=metadata_bundle(plan['signature']['sources']);require(bundle.descriptor()==plan['data'],'Conditional feature row/prompt mapping changed.');return bundle


def materialize(bundle,sources,checkpoint):
    scanner.require_checkpoint_path(checkpoint)
    import pyarrow.dataset as ds
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,add_eos_token=False,add_bos_token=False,padding_side='left');texts={}
    for name,source in sources.items():
        wanted=bundle.inputs[bundle.inputs.input_source.eq(name)];predicate=ds.field('prompt_ix').isin(wanted.source_prompt_ix.tolist())&(ds.field('wrapper')=='native')
        predicate=predicate&(ds.field('split').isin(['validation'] if name=='c4' else ['fit','validation']))
        if name=='natural':predicate=predicate&((ds.field('split')=='validation')|(ds.field('eligible_fit')==True))
        selected=ds.dataset(Path(source['cache'])/'inputs.parquet',format='parquet').to_table(filter=predicate,columns=['prompt_ix','prompt']).to_pandas().set_index('prompt_ix')
        require(selected.index.is_unique and set(selected.index)==set(wanted.source_prompt_ix),'Selected native development filter changed.')
        for r in wanted.itertuples():texts[r.uid]=selected.loc[r.source_prompt_ix,'prompt']
    bundle.inputs['prompt']=bundle.inputs.uid.map(texts);bundle.tokens={}
    for r in bundle.inputs.itertuples():
        ids=tokenizer.encode(r.prompt,add_special_tokens=False);actual=native.parse_native_labels(ids,tokenizer);group=bundle.lookup[r.uid]
        require(len(ids)==r.length and all(ids[t.token_ix]==t.token_id and actual.get(t.token_ix)==t.role for t in group.itertuples()),'Native sampled content-role/token identity changed.')
        bundle.tokens[r.uid]=ids
    bundle.parts=[bundle.inputs.iloc[i:i+BATCH] for i in range(0,len(bundle.inputs),BATCH)];return bundle


@contextmanager
def model_context(plan):
    """Own loader: no writes or reassignment to any frozen runner globals."""
    import torch
    from transformers import AutoModelForCausalLM,Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    import accelerate.hooks
    import accelerate.utils.operations as operations
    checkpoint=plan['signature']['checkpoint_path']
    require_checkpoint_path(checkpoint)
    require(torch.cuda.device_count()==4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)),'Exactly4 visible A40s required.')
    require(all(torch.cuda.mem_get_info(i)[0]>=22*2**30 for i in range(4)),'Insufficient free assigned GPU memory; no process is killed/reset.')
    original_get=hub_kernels.get_kernel;original_validate=Mxfp4HfQuantizer.validate_environment;original_send=operations.send_to_device;original_hook_send=accelerate.hooks.send_to_device
    def pinned(name,*args,**kwargs):
        if name=='kernels-community/gpt-oss-triton-kernels':kwargs['revision']=plan['signature']['kernel_revision']
        return original_get(name,*args,**kwargs)
    def strict(self,*args,**kwargs):
        original_validate(self,*args,**kwargs);require(not self.quantization_config.dequantize,'Native MXFP4 required; no BF16 dequantization.')
    def staged(tensor,device,non_blocking=False,skip_keys=None):
        if isinstance(tensor,torch.Tensor) and tensor.device.type=='cuda':
            target=torch.device(f'cuda:{device}' if isinstance(device,int) else device)
            if target.type=='cuda' and tensor.device!=target:return tensor.cpu().to(target,non_blocking=False)
        return original_send(tensor,device,non_blocking=non_blocking,skip_keys=skip_keys)
    device_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3};device_map.update({f'model.layers.{i}':i//9 for i in range(36)})
    model=None
    try:
        hub_kernels.get_kernel=pinned;Mxfp4HfQuantizer.validate_environment=strict;operations.send_to_device=staged;accelerate.hooks.send_to_device=staged
        model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,dtype=torch.bfloat16,quantization_config=Mxfp4Config(dequantize=False),device_map=device_map,attn_implementation='eager',use_kernels=False).eval().requires_grad_(False)
        backends={type(layer.mlp.experts).__name__ for layer in model.model.layers}
        require(backends=={'Mxfp4GptOssExperts'} and not model.hf_quantizer.quantization_config.dequantize and not any(p.requires_grad for p in model.parameters()),'Model backend/quantization/freeze differs.')
        immutable(OUT/'model-backend.json',dict(expert_backends=sorted(backends),device_map=model.hf_device_map,checkpoint_path=checkpoint,native_mxfp4=True,parameters_frozen=True,
            gpu_transfer='Synchronous CPU staging between different CUDA devices',gpus=[torch.cuda.get_device_name(i) for i in range(4)],compute_capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(4)]))
        yield model
    finally:
        hub_kernels.get_kernel=original_get;Mxfp4HfQuantizer.validate_environment=original_validate;operations.send_to_device=original_send;accelerate.hooks.send_to_device=original_hook_send
        if model is not None:del model
        gc.collect()


def initialize(bundle,plan_sha,*,cache=CACHE):
    owner=dict(plan_sha256=plan_sha,rows=len(bundle.rows),dimensions=128,dtype='float16',row_mapping_sha256=bundle.descriptor()['row_mapping_sha256'])
    if (cache/'initialized.json').exists():
        saved=read(cache/'initialized.json');require(all(saved[k]==v for k,v in owner.items()) and data.digest(cache/'rows.parquet')==saved['rows_sha256'],'Wrong extra-bank owner or row table.')
        legacy.check_feature_file(cache/'features.f16',len(bundle.rows),128)
    else:
        require(not cache.exists() or not any(cache.iterdir()),'Unowned partial extra bank; no auto-deletion/reconstruction.');cache.mkdir(exist_ok=True)
        path=cache/'rows.parquet.initializing';bundle.rows.to_parquet(path,index=False);legacy.fsync_file(path);path.replace(cache/'rows.parquet')
        path=cache/'features.f16.initializing';array=np.memmap(path,mode='w+',dtype=np.float16,shape=(len(bundle.rows),128))
        for start in range(0,len(array),4096):array[start:start+4096]=np.nan
        array.flush();del array;legacy.fsync_file(path);path.replace(cache/'features.f16');data.atomic_json(cache/'initialized.json',owner|dict(rows_sha256=data.digest(cache/'rows.parquet')))
    return np.memmap(cache/'features.f16',mode='r+',dtype=np.float16,shape=(len(bundle.rows),128))


def parity_scope(bundle,proof):
    require(proof['fit_only'] is True and proof['validation_probabilities_computed'] is False and proof['rows']==256
        and proof['target_cache_ix_sha256']==array_sha(bundle.parity_targets) and proof['sample_ix_sha256']==array_sha(bundle.rows.iloc[bundle.parity_targets].sample_ix.to_numpy())
        and bundle.rows.iloc[bundle.parity_targets].split.eq('fit').all() and bundle.rows.iloc[bundle.parity_targets].parent_bank.eq('old').all(),'Portable scope is not the predetermined original256 FIT rows.')


def restore(bundle,array,plan_sha,*,cache=CACHE):
    if not (cache/'progress.json').exists():return [],[],None
    value=read(cache/'progress.json');hashes=value['batch_features_sha256'];state=legacy.prefix_state(bundle,len(hashes))
    require(value['plan_sha256']==plan_sha and all(value[k]==v for k,v in state.items()) and len(value['batch_f16_roundtrip_max'])==len(hashes),'Progress is not exact complete prompt prefix.')
    for part,want in zip(bundle.parts,hashes):require(legacy.batch_digest(array,bundle.targets(part))==want,'Previously committed extra128 feature bytes changed.')
    proof=value['portable_consumer'];require(not hashes or proof is not None,'Committed rows have no FIT-only parity proof.')
    if proof is not None:parity_scope(bundle,proof)
    return hashes.copy(),value['batch_f16_roundtrip_max'].copy(),proof


def checkpoint(bundle,array,plan_sha,hashes,roundtrips,proof,*,cache=CACHE):
    array.flush();legacy.fsync_file(cache/'features.f16');data.atomic_json(cache/'progress.json',dict(plan_sha256=plan_sha,**legacy.prefix_state(bundle,len(hashes)),
        batch_features_sha256=hashes,batch_f16_roundtrip_max=roundtrips,portable_consumer=proof))


def dual_forward(model,extra,transient,batch,bundle,*,device='cuda:0'):
    extra.begin(batch);wanted={int(t):i for i,t in enumerate(batch['targets'])};targets=bundle.parity_targets
    positions=np.asarray([batch['positions'][wanted[int(t)]] for t in targets if int(t) in wanted],np.int64)
    transient.begin(dict(positions=positions,targets=np.arange(len(positions),dtype=np.int64)))
    output=model.model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),use_cache=False).last_hidden_state
    extra.finish();transient.finish();return output


@contextmanager
def checked_hooks(model,extra,transient,bundle,device='cuda:0'):
    import torch
    sample=bundle.collate(bundle.parts[0]);old=extra.array[sample['targets']].copy();old_transient=transient.array.copy();handles=[]
    before=model.model(input_ids=sample['input_ids'].to(device),attention_mask=sample['attention_mask'].to(device),use_cache=False).last_hidden_state.cpu()
    try:
        for layer in sorted(set(extra.layers+transient.layers)):
            for collector in [extra,transient]:
                if layer in collector.layers:handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)))
        extra.enabled=False;transient.enabled=False;after=dual_forward(model,extra,transient,sample,bundle,device=device).cpu()
        require(torch.equal(before,after) and torch.isfinite(after).all() and np.array_equal(old,extra.array[sample['targets']],equal_nan=True)
            and np.array_equal(old_transient,transient.array,equal_nan=True),'Disabled hooks changed hidden states or wrote vectors.')
        immutable(OUT/'hook-validation.json',dict(hidden_states_bitexact=True,writes_disabled=True,feature_rows_unchanged=True,transient_rows_unchanged=True,layers=sorted(set(extra.layers+transient.layers)),sample_rows=len(sample['targets'])))
        del before,after,old,old_transient;extra.enabled=True;yield
    finally:
        for h in handles:h.remove()


def portable_parity(bundle,array,transient,added):
    targets=bundle.parity_targets;require(bundle.rows.iloc[targets].split.eq('fit').all(),'Validation parity forbidden.')
    with np.load(original.OUT/'models/top8-lda.npz',allow_pickle=False) as z:w,b=z['weight'].copy(),z['bias'].copy()
    require(w.shape==(8,5,64) and b.shape==(8,5),'Original8 LDA consumer shape changed.')
    first=legacy.portable_consumer(transient,np.arange(256),w,b,'cuda:0')
    reference=np.memmap(OLD_BANK/'features.f16',mode='r',dtype=np.float16,shape=(755282,512))[bundle.rows.iloc[targets].parent_cache_ix.to_numpy()]
    expected=original.predict_heads(reference.astype(np.float64).reshape(-1,8,64),w,b).mean(1)
    current=original.predict_heads(np.asarray(transient,np.float64).reshape(-1,8,64),w,b).mean(1);difference=float(np.abs(current-expected).max())
    require(difference<PARITY_TOLERANCE,'Transient original8 versus saved parent FIT probability drift exceeds fixed tolerance.')
    with np.load(scanner.OUT/'models/selected10-proxy-lda.npz',allow_pickle=False) as z:
        require(z['selected_head_ids'].tolist()==ORIGINAL+added,'Extra2 proxy model order changed.');ew,eb=z['weight'][8:].copy(),z['bias'][8:].copy()
    second=legacy.portable_consumer(array,targets,ew,eb,'cuda:0')
    path=OUT/'portable-fit-sample.npz';np.savez_compressed(path,extra_features=np.asarray(array[targets]),transient_original8=np.asarray(transient),target_cache_ix=targets,sample_ix=bundle.rows.iloc[targets].sample_ix.to_numpy())
    proof=dict(rows=256,fit_only=True,validation_probabilities_computed=False,target_cache_ix_sha256=array_sha(targets),sample_ix_sha256=array_sha(bundle.rows.iloc[targets].sample_ix.to_numpy()),
        original8_savedF16_CPU_GPU=first,extra2_savedF16_CPU_GPU=second,original8_reference_probability_max_difference=difference,
        original8_reference_vectors_bitexact=bool(np.array_equal(reference.view(np.uint16),transient.view(np.uint16))),tolerance=PARITY_TOLERANCE,
        original8_transient_only=True,no_new_readout=True,sample_sha256=data.digest(path))
    parity_scope(bundle,proof);immutable(OUT/'portable-validation.json',proof);return proof


def verified_extraction():
    gate()
    plan=read(OUT/'plan.json');bundle=verify_plan(plan);complete=read(OUT/'extraction-complete.json')
    require(complete['plan_sha256']==data.digest(OUT/'plan.json') and complete['source_sha256']==data.digest(__file__) and complete['signature']==plan['signature']
        and complete['rows']==ROWS and complete['feature_shape']==[ROWS,128] and complete['dimensions']==128 and complete['dtype']=='float16'
        and complete['all_rows_written'] is True and complete['all_finite'] is True and complete['tests_read'] is False and complete['added_head_ids']==plan['added_head_ids'],'Completed extra-bank identity changed.')
    require(complete['features_path']==str(CACHE/'features.f16') and complete['rows_path']==str(CACHE/'rows.parquet') and complete['rows_sha256']==data.digest(CACHE/'rows.parquet'),'Completed extra-bank path/row bytes changed.')
    legacy.check_feature_file(CACHE/'features.f16',ROWS,128);require(data.digest(CACHE/'features.f16')==complete['features_sha256'],'Completed extra feature bytes changed.')
    owner=read(CACHE/'initialized.json');require(owner['plan_sha256']==complete['plan_sha256'] and owner['row_mapping_sha256']==bundle.descriptor()['row_mapping_sha256'] and owner['rows_sha256']==complete['rows_sha256'],'Completed cache owner changed.')
    array=np.memmap(CACHE/'features.f16',mode='r',dtype=np.float16,shape=(ROWS,128));inspect=legacy.inspect_features(array)
    hashes,roundtrips,proof=restore(bundle,array,complete['plan_sha256']);require(len(hashes)==len(bundle.parts) and complete['portable_consumer']==proof and proof==read(OUT/'portable-validation.json'),'Completed prefix/parity changed.')
    require(data.digest(OUT/'portable-fit-sample.npz')==proof['sample_sha256'] and complete['progress_sha256']==data.digest(CACHE/'progress.json')
        and complete['hook_validation_sha256']==data.digest(OUT/'hook-validation.json') and complete['model_backend_sha256']==data.digest(OUT/'model-backend.json'),'Completed small evidence SHA changed.')
    for k,v in inspect.items():require(complete[k]==v,'Completed finite/zero report differs.')
    hook=read(OUT/'hook-validation.json');require(all(hook[k] is True for k in ['hidden_states_bitexact','writes_disabled','feature_rows_unchanged','transient_rows_unchanged']),'Hook self-check proof changed.')
    return complete


def _extract_locked():
    require((OUT/'plan.json').is_file(),'Root must declare/freeze after admissible scanner independent audit before extraction.')
    gate();no_competing_jobs();plan=read(OUT/'plan.json');bundle=verify_plan(plan)
    if (OUT/'extraction-complete.json').exists():return verified_extraction()
    require(shutil.disk_usage(CACHE.parent).free>=ROWS*128*2+512*2**20,'Need extra128 bank bytes plus512MiB reserve; old banks remain unchanged.')
    bundle=materialize(bundle,plan['signature']['sources'],plan['signature']['checkpoint_path']);plan_sha=data.digest(OUT/'plan.json');array=initialize(bundle,plan_sha)
    hashes,roundtrips,proof=restore(bundle,array,plan_sha)
    if proof is not None:require(data.digest(OUT/'portable-fit-sample.npz')==proof['sample_sha256'],'Saved FIT portable proof changed.')
    if len(hashes)<len(bundle.parts):
        import torch
        extra=legacy.Collector(array,plan['added_head_ids']);transient=legacy.Collector(np.full((256,512),np.nan,np.float16),ORIGINAL)
        with model_context(plan) as model,torch.inference_mode(),checked_hooks(model,extra,transient,bundle):
            for i,part in enumerate(bundle.parts):
                if i<len(hashes):continue
                extra.maximum_f16_roundtrip_difference=0.;transient.enabled=proof is None;output=dual_forward(model,extra,transient,bundle.collate(part),bundle);del output
                if proof is None:
                    require(i==0 and np.isfinite(transient.array).all(),'First committed batch must contain complete fixed FIT parity.')
                    proof=portable_parity(bundle,array,transient.array,plan['added_head_ids']);transient.enabled=False
                hashes.append(legacy.batch_digest(array,bundle.targets(part)));roundtrips.append(extra.maximum_f16_roundtrip_difference)
                if (i+1)%INTERVAL==0 or i+1==len(bundle.parts):
                    checkpoint(bundle,array,plan_sha,hashes,roundtrips,proof);data.atomic_json(OUT/'status.json',dict(stage='extracting-added2',batch=i+1,batches=len(bundle.parts)))
        del extra,transient;gc.collect()
    inspect=legacy.inspect_features(array);require(proof is not None,'No fixed FIT portability proof.')
    complete=dict(plan_sha256=plan_sha,source_sha256=data.digest(__file__),signature=plan['signature'],rows=ROWS,dimensions=128,feature_shape=[ROWS,128],dtype='float16',
        added_head_ids=plan['added_head_ids'],original_head_ids=ORIGINAL,head_dim=64,features_path=str(CACHE/'features.f16'),features_sha256=data.digest(CACHE/'features.f16'),rows_path=str(CACHE/'rows.parquet'),rows_sha256=data.digest(CACHE/'rows.parquet'),
        all_rows_written=True,**inspect,portable_consumer=proof,maximum_f16_roundtrip_difference=float(max(roundtrips,default=0)),
        hook_validation_sha256=data.digest(OUT/'hook-validation.json'),model_backend_sha256=data.digest(OUT/'model-backend.json'),progress_sha256=data.digest(CACHE/'progress.json'),tests_read=False,readout_trained=False,old_banks_deleted=False)
    immutable(OUT/'extraction-complete.json',complete);data.atomic_json(OUT/'status.json',dict(stage='extraction_complete',rows=ROWS));return complete


def no_competing_jobs():
    scanner.no_competing_jobs();active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name==Path(__file__).name for a in args if a):active.append(int(entry.name))
    require(not active,'Another added-head extractor live: '+str(active))


@contextmanager
def run_lock():
    import fcntl
    with (OUT/'.run.lock').open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another process owns this extra bank.') from e
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def extract():
    gate();require((OUT/'plan.json').is_file(),'Publish reviewed conditional declaration before extraction.')
    with run_lock():return _extract_locked()


def preflight():
    import torch
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-extra-heads-preflight-',dir='/tmp'));before=data.digest(__file__);checks=[]
    old_globals=(legacy.OUT,legacy.CACHE,scanner.OUT,scanner.CACHE,original.OUT,data.runtime.atomic_json,data.runtime.status)
    def check(name,value=True):require(bool(value),'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,AssertionError,KeyError):checks.append(name);return
        raise RuntimeError('Expected rejection: '+name)
    # Keep preflight synthetic after formal pins are filled: patch only this
    # module's two pins, then restore them without touching frozen modules.
    saved_pins={name:globals()[name] for name in ['AUDIT_PATH','AUDIT_SHA']}
    try:
        globals().update(AUDIT_PATH=None,AUDIT_SHA=None)
        reject('unknown_actual_audit_pins_block_before_any_source_or_headp_read',gate)
    finally:globals().update(saved_pins)
    old=pd.DataFrame(dict(cache_ix=np.arange(256),sample_ix=2_000_000_000+np.arange(256),wrapper='native',split='fit',profile='long-full',source_set='long-full',prompt_ix=0,
        token_ix=np.arange(256),token_id=11+np.arange(256),role='system',group_id='old-long',document_id=3))
    c4=pd.DataFrame(dict(cache_ix=[100,101],sample_ix=[6_000_000_000_000,6_000_000_000_001],wrapper='native',split='validation',profile='long-full',source_set='long-augmentation',prompt_ix=5,token_ix=[0,1],token_id=[21,22],role='user',group_id='new-val'))
    natural=pd.DataFrame(dict(cache_ix=[0,1,2],sample_ix=4_000_000_000_000+np.arange(3),wrapper='native',split='fit',profile='natural',source_set='natural-dev-clean',prompt_ix=0,original_prompt_ix=10,token_ix=[0,1,2],token_id=[31,32,33],role=['system','user','assistant'],group_id='natural-clean'))
    rows=project_rows(old,c4,natural,expected=dict(old=256,c4=2,natural=3))
    check('parent_bank_cache_mapping_and_new_consecutive_indices',len(rows)==261 and rows.cache_ix.tolist()==list(range(261)) and rows.loc[256:257,'parent_cache_ix'].tolist()==[100,101] and rows.iloc[-1].source_prompt_ix==10)
    bad=c4.copy();bad.loc[0,'split']='fit';reject('new_C4fit_never_enters_extra_bank',lambda:project_rows(old,bad,natural,expected=dict(old=256,c4=2,natural=3)))
    bad=natural.copy();bad.loc[0,'role']='cot';reject('no_fabricated_public_CoT_support',lambda:project_rows(old,c4,bad,expected=dict(old=256,c4=2,natural=3)))
    bad=natural.copy();bad.loc[0,'sample_ix']=old.sample_ix.iloc[0];reject('cross_bank_sample_namespace_overlap_rejected',lambda:project_rows(old,c4,bad,expected=dict(old=256,c4=2,natural=3)))
    inputs=rows[['uid','source_prompt_ix','input_source','split']].drop_duplicates().copy();inputs['length']=[256,4,4]
    bundle=Bundle(inputs,rows);bundle.tokens={}
    for r in bundle.inputs.itertuples():
        tokens=list(range(r.length));g=bundle.lookup[r.uid]
        for t in g.itertuples():tokens[t.token_ix]=int(t.token_id)
        bundle.tokens[r.uid]=tokens
    batch=bundle.collate(bundle.parts[0]);check('metadata_fixed_first256_oldFIT_priority_scope',bundle.parity_targets.tolist()==list(range(256)) and bundle.rows.iloc[bundle.parity_targets].split.eq('fit').all())
    check('all_unified_sampled_rows_leftpad_token_mapping',np.array_equal(batch['input_ids'].reshape(-1)[batch['positions']].numpy(),bundle.rows.iloc[batch['targets']].token_id))
    cache=folder/'bank';array=initialize(bundle,'synthetic-plan',cache=cache);check('NaN_initialize_exact128_without512_copy',array.shape==(261,128) and np.isnan(array).all() and (cache/'features.f16').stat().st_size==261*128*2)
    hashes,roundtrips,proof=restore(bundle,array,'synthetic-plan',cache=cache);check('uncommitted_partial_bytes_do_not_advance_cursor',not hashes and proof is None)
    array[0]=1;array.flush();hashes,_,_=restore(bundle,array,'synthetic-plan',cache=cache);check('finite_partial_write_still_uncommitted',not hashes)
    reject('wrong_owner_plan_rejected',lambda:initialize(bundle,'different-plan',cache=cache))
    array[:]=np.arange(128,dtype=np.float16);array.flush();proof=dict(rows=256,fit_only=True,validation_probabilities_computed=False,target_cache_ix_sha256=array_sha(bundle.parity_targets),sample_ix_sha256=array_sha(bundle.rows.iloc[bundle.parity_targets].sample_ix.to_numpy()))
    hashes=[legacy.batch_digest(array,bundle.targets(bundle.parts[0]))];checkpoint(bundle,array,'synthetic-plan',hashes,[0.],proof,cache=cache)
    restored,_,saved=restore(bundle,array,'synthetic-plan',cache=cache);check('complete_prefix_all128_bytes_and_FIT_scope_restore',restored==hashes and saved==proof)
    previous=float(array[0,0]);array[0,0]=1;array.flush();reject('committed_finite_byte_tamper_rejected',lambda:restore(bundle,array,'synthetic-plan',cache=cache));array[0,0]=previous
    array[0,1]=np.nan;array.flush();reject('committed_one_head_column_NaN_rejected',lambda:restore(bundle,array,'synthetic-plan',cache=cache));array[0,1]=1
    reject('portable_scope_cannot_be_validation',lambda:parity_scope(bundle,proof|dict(validation_probabilities_computed=True)))
    # Explicit selected-layer/rank scatter with two unrelated layers and an
    # unsorted sampled-row mapping; no model or source activation is used.
    small=np.full((2,128),np.nan,np.float16);collector=legacy.Collector(small,ADDED);collector.begin(dict(positions=np.array([0,1]),targets=np.array([1,0])));collector.enabled=False
    values={layer:torch.arange(layer*64,(layer+1)*64,dtype=torch.float32).reshape(1,1,64,1).expand(1,2,64,64).reshape(1,2,4096).clone() for layer in collector.layers}
    for layer in collector.layers:
        original_x=values[layer].clone();require(collector.hook(layer)(None,(values[layer],)) is None,'Hook returned replacement activations.');require(torch.equal(original_x,values[layer]),'Disabled hook modified input.')
    collector.finish();check('disabled_extra_hook_zero_writes_bitexact_input',np.isnan(small).all())
    collector.begin(dict(positions=np.array([0,1]),targets=np.array([1,0])));collector.enabled=True
    for layer in collector.layers:collector.hook(layer)(None,(values[layer],))
    collector.finish();check('ascending_two_head_columns_and_full128_scatter',np.all(small[:,:64]==247) and np.all(small[:,64:]==2248))
    inspect=legacy.inspect_features(np.zeros((2,128),np.float16));check('legal_zero_heads_not_rejected',inspect['all_finite'] and inspect['zero_row_count']==2 and not inspect['all_rows_nonzero'])
    truncate=folder/'truncate';ta=initialize(bundle,'synthetic-plan',cache=truncate);del ta
    with (truncate/'features.f16').open('r+b') as stream:stream.truncate(7)
    reject('truncated_cache_fails_instead_of_rebuild',lambda:initialize(bundle,'synthetic-plan',cache=truncate))
    check('legacy_and_scanner_globals_unchanged',old_globals==(legacy.OUT,legacy.CACHE,scanner.OUT,scanner.CACHE,original.OUT,data.runtime.atomic_json,data.runtime.status))
    check('preflight_source_frozen',before==data.digest(__file__))
    report=dict(stage='synthetic_conditional_added_two_head_extraction',source_sha256=before,checks=checks,check_count=len(checks),
        conditional_actual_audit_sha256=AUDIT_SHA,actual_enabled=False,real_source_rows_read=False,real_head_probabilities_read=False,real_model_or_checkpoint_read=False,
        source_text_materialized=False,formal_OUT_or_CACHE_created=False,model_loaded=False,gpu_used=False,tests_read=False)
    output=folder/'report.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=data.digest(output),source_sha256=before,checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--stage',choices=['declare','extract','verify']);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.stage=='declare':declare()
    elif args.stage=='extract':extract()
    elif args.stage=='verify':verified_extraction()
    else:parser.print_help()

if __name__=='__main__':main()
