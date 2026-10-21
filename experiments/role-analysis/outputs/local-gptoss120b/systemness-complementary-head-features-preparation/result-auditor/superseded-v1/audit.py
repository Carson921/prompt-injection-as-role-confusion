"""Independent extra2 F16 bank audit. Default help; --preflight is tiny CPU/tmp.

Only root may run --actual after extractor terminal0. No producer imports,
LM/GPU, text decoding, probabilities from scan/test, or raw LM replay.
"""
import argparse,copy,hashlib,json,os,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis');BASE=HERE/'outputs/local-gptoss120b'
SOURCE=HERE/'systemness_complementary_head_extract.py';SOURCE_SHA='964335b1dbdbea19219cc4e3db15409195376cd92dfd7125f7d4370758a34509'
PLAN_SHA='a64a8e697cde2dafbb9155113fcfbb1876227ad7aab8f785ac3879d4c09c5291'
SCAN_SOURCE_SHA='70db2015b90a496dc465c63631911da81d641614fcf0b1d48b04313a078ce156'
SCAN_COMPLETE_SHA='dd538fb5d3cc20d6652ca2fd214bbadd7477d124978b50809561673e34de4b95';SCAN_SELECTION_SHA='77563f6dfacab0c7bf42a72090e64496bd4029d2143b2c9f21f13472f5762551'
SCAN_AUDIT_SHA='a71c052c38947b9c2a4f1ab73f58a3414533b43e013bac624322b0b836e67ff0'
OUT=BASE/'systemness-complementary-head-features';CACHE=Path('/dev/shm/role-analysis-complementary-heads');SCAN_OUT=BASE/'systemness-complementary-head-scan'
BANKS={'old':Path('/dev/shm/role-analysis-top8-heads'),'c4':Path('/dev/shm/role-analysis-long-augmentation-heads'),'natural':Path('/dev/shm/role-analysis-natural-dev-heads')}
PARENT_COMPLETES={'old':BASE/'systemness-head-features/extraction-complete.json','c4':BASE/'systemness-long-augmentation/extraction-complete.json','natural':BASE/'systemness-natural-dev-features/extraction-complete.json'}
PARENT_COMPLETE_SHA={'old':'b970591a6c82d8818bec589b58e3e96ce65e5d8a6daa4d0710c42b2e909abc89','c4':'ef266142d54ccd1ccd61c93fb08e7a8f7484702e874c3f508d7cac2ca958f6f7','natural':'4f020979dc1e337924d6f8ef6b7a576fd36a7f629c72a85431b8caf68d01f9f4'}
ORIGINAL=[458,462,2045,460,459,2047,456,997];ADDED=[247,2248];COUNTS={'old':755282,'c4':61440,'natural':80116};ROLES=['system','user','cot','assistant','tool'];ROWS=896838;TOL=5e-4
META_COLUMNS=['cache_ix','sample_ix','prompt_ix','original_prompt_ix','original_cache_ix','token_ix','token_id','message_ix','role','semantic_role','wrapper','profile','source_set','split','group_id','document_id','source_uid','dataset','family_id','dependency_component_id','eligible_fit','eligible_clean_validation','old_generated_collision','cross_split_assistant_collision','body_offset']
MAP_COLUMNS=['cache_ix','parent_bank','parent_cache_ix','sample_ix','uid','token_ix','token_id','role','split']
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fsha(f,c):return hashlib.sha256(f[c].to_csv(index=False).encode()).hexdigest()
def close(a,b,s):require(np.isfinite(a) and np.isfinite(b) and abs(float(a)-float(b))<=2e-11,s)
def processes():
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name in {'systemness_complementary_head_extract.py','systemness_complementary_head_scan.py'} for a in args if a):active.append(int(p.name))
    return active
def terminal_gate():
    require(not processes(),'Producer/scanner live; no partial artifact reads.')
    require((OUT/'extraction-complete.json').is_file() and (OUT/'status.json').is_file(),'Terminal completion absent.')
    require(read(OUT/'status.json').get('stage')=='extraction_complete','Producer not terminal complete.')
def metadata(path,filters=None):
    import pyarrow.parquet as pq
    columns=[c for c in META_COLUMNS if c in pq.ParquetFile(path).schema.names]
    return pd.read_parquet(path,columns=columns,filters=filters)
def project(parents,counts=COUNTS):
    pieces=[]
    for name in ['old','c4','natural']:
        r=parents[name].copy();require(len(r)==counts[name] and r.cache_ix.is_unique and r.sample_ix.is_unique and r.wrapper.eq('native').all(),'Wrong native parent scope: '+name)
        require(r.split.isin(['fit','validation']).all() and (name!='c4' or r.split.eq('validation').all()),'Test/new C4 fit entered bank.')
        if name=='natural':require(r.source_set.eq('natural-dev-clean').all() and set(r.role)<={'system','user','assistant'},'Non-clean public role/source.')
        r['parent_bank']=name;r['parent_cache_ix']=r.cache_ix;r['source_prompt_ix']=r.original_prompt_ix if name=='natural' else r.prompt_ix
        r['input_source']='old:'+r.source_set if name=='old' else name;r['uid']=r.input_source+':'+r.source_prompt_ix.astype(str);pieces.append(r)
    rows=pd.concat(pieces,ignore_index=True);rows['cache_ix']=np.arange(len(rows),dtype=np.int64);require(rows.sample_ix.is_unique,'Cross-parent sample collision.');return rows
class Layout:
    def __init__(self,rows,inputs):
        self.rows=rows.reset_index(drop=True);r=self.rows
        self.parity=r.index[r.parent_bank.eq('old')&r.split.eq('fit')&r.source_set.eq('long-full')&r.profile.eq('long-full')].to_numpy(np.int64)[:256]
        require(len(self.parity)==256,'Deterministic original256 FIT scope absent.');priority=set(r.iloc[self.parity].uid);require(len(priority)<=4,'FIT parity spans more than first batch.')
        sorted_inputs=inputs.sort_values(['length','uid']);self.inputs=pd.concat([sorted_inputs[sorted_inputs.uid.isin(priority)].sort_values('uid'),sorted_inputs[~sorted_inputs.uid.isin(priority)]]).reset_index(drop=True)
        require(self.inputs.uid.is_unique and set(self.inputs.uid)==set(r.uid),'Prompt mapping mismatch.');self.lookup={u:g for u,g in r.groupby('uid',sort=False)};self.parts=[self.inputs.iloc[s:s+4] for s in range(0,len(self.inputs),4)]
        require(set(self.parity)<=set(self.targets(self.parts[0])),'Predetermined FIT vectors not in first batch.')
    def targets(self,part):return np.concatenate([self.lookup[r.uid].cache_ix.to_numpy(np.int64) for r in part.itertuples()])
    def descriptor(self):
        r=self.rows;return dict(rows=len(r),prompts=len(self.inputs),batches=len(self.parts),input_tokens=int(self.inputs.length.sum()),row_mapping_sha256=fsha(r,MAP_COLUMNS),input_mapping_sha256=fsha(self.inputs,['uid','source_prompt_ix','input_source','length','split']),sample_ix_sha256=arrsha(r.sample_ix.to_numpy()),parity_fit_targets_sha256=arrsha(self.parity),parity_fit_sample_ix_sha256=arrsha(r.iloc[self.parity].sample_ix.to_numpy()),parity_prompts=r.iloc[self.parity].uid.drop_duplicates().tolist())
def prefix(layout,n):
    require(0<=n<=len(layout.parts),'Out-of-range prefix.');ix=np.concatenate([layout.targets(p) for p in layout.parts[:n]]) if n else np.empty(0,np.int64)
    require(len(np.unique(ix))==len(ix),'Duplicate completed row.');prompts=layout.inputs.iloc[:min(n*4,len(layout.inputs))]
    return dict(next_batch=n,rows_written=len(ix),cache_ix_sha256=arrsha(ix),sample_ix_sha256=arrsha(layout.rows.iloc[ix].sample_ix.to_numpy()),prompt_prefix_sha256=fsha(prompts,['uid','length']))
def batch_sha(a,ix):
    h=hashlib.sha256()
    for start in range(0,len(ix),4096):
        block=np.ascontiguousarray(a[ix[start:start+4096]]);require(np.isfinite(block).all(),'Unwritten/nonfinite batch feature column.');h.update(block.tobytes())
    return h.hexdigest()
def verify_progress(layout,a,p,plan_sha):
    require(p['plan_sha256']==plan_sha,'Progress plan changed.');n=len(p['batch_features_sha256']);want=prefix(layout,n);require(all(p[k]==v for k,v in want.items()),'Complete-batch prefix identity changed.')
    require(n==len(layout.parts) and want['rows_written']==len(layout.rows) and len(p['batch_f16_roundtrip_max'])==n,'Incomplete batch prefix.')
    require(all(np.isfinite(x) and x>=0 for x in p['batch_f16_roundtrip_max']),'Invalid cast-roundtrip evidence.')
    for part,h in zip(layout.parts,p['batch_features_sha256']):require(batch_sha(a,layout.targets(part))==h,'Committed all128-column batch hash changed.')
    return want

def predict(x,w,b):
    z=np.einsum('nhd,hkd->nhk',x,w)+b[None];z-=z.max(2,keepdims=True);v=np.exp(z);return (v/v.sum(2,keepdims=True)).mean(1)
def consumer_evidence(e,saved):
    require(e['rows']==256 and e['same_saved_f16_vectors'] is True and e['classifier_device']=='cuda:0' and e['saved_vectors_sha256']==arrsha(saved),'Portable saved-F16 identity differs.')
    require(e['tolerance']==TOL and np.isfinite(e['max_difference']) and 0<=e['max_difference']<=TOL,'Portable sampled CPU/GPU error evidence invalid.')
def parity(layout,a,proof,sample,oldfile,oldw,oldb,extraw,extrab):
    targets=layout.parity;r=layout.rows.iloc[targets]
    require(proof['rows']==256 and proof['fit_only'] is True and proof['validation_probabilities_computed'] is False and proof['original8_transient_only'] is True and proof['no_new_readout'] is True
        and proof['tolerance']==TOL and proof['target_cache_ix_sha256']==arrsha(targets) and proof['sample_ix_sha256']==arrsha(r.sample_ix.to_numpy()),'Fixed FIT parity identity/scope changed.')
    require(set(sample)=={'extra_features','transient_original8','target_cache_ix','sample_ix'} and np.array_equal(sample['target_cache_ix'],targets) and np.array_equal(sample['sample_ix'],r.sample_ix),'FIT sample row identities changed.')
    extra=sample['extra_features'];old=sample['transient_original8'];require(extra.shape==(256,128) and old.shape==(256,512) and extra.dtype==old.dtype==np.float16 and np.isfinite(extra).all() and np.isfinite(old).all(),'FIT F16 geometry/finite check failed.')
    require(np.array_equal(extra.view(np.uint16),np.asarray(a[targets]).view(np.uint16)),'Extra2 sample is not saved feature bank bytes.')
    reference=np.asarray(oldfile[r.parent_cache_ix.to_numpy(np.int64)]);require(reference.shape==(256,512),'Parent256 rows absent.')
    difference=float(np.abs(predict(old.astype(np.float64).reshape(-1,8,64),oldw,oldb)-predict(reference.astype(np.float64).reshape(-1,8,64),oldw,oldb)).max())
    close(difference,proof['original8_reference_probability_max_difference'],'Original8 reference probability difference changed.');require(difference<TOL,'Original8 FIT drift too large.')
    require(proof['original8_reference_vectors_bitexact']==bool(np.array_equal(reference.view(np.uint16),old.view(np.uint16))),'Reference bitexact diagnosis changed.')
    consumer_evidence(proof['original8_savedF16_CPU_GPU'],old);consumer_evidence(proof['extra2_savedF16_CPU_GPU'],extra)
    originalp=predict(old.astype(np.float64).reshape(-1,8,64),oldw,oldb);extrap=predict(extra.astype(np.float64).reshape(-1,2,64),extraw,extrab)
    require(np.isfinite(originalp).all() and np.isfinite(extrap).all() and np.allclose(originalp.sum(1),1,atol=1e-12) and np.allclose(extrap.sum(1),1,atol=1e-12),'Independent saved-F16 LDA consumer invalid.')
    return dict(rows=256,fit_only=True,original8_reference_probability_maximum_difference=difference,original8_savedF16_CPU_probability_sha256=arrsha(originalp),extra2_savedF16_CPU_probability_sha256=arrsha(extrap),producer_GPU_parity_evidence_bound=True,GPU_independently_replayed=False)

def actual(output):
    terminal_gate();require(Path(output).resolve().is_relative_to(Path('/tmp')),'Audit writes /tmp only.');bindings={}
    def bind(p,want=None):
        p=Path(p);require(not str(p).startswith('/workspace'),'Failed mount forbidden.');got=digest(p);require(want is None or got==want,'SHA mismatch: '+str(p));bindings[str(p.resolve())]=got;return got
    bind(__file__);bind(SOURCE,SOURCE_SHA);bind(OUT/'plan.json',PLAN_SHA);plan=read(OUT/'plan.json');done_sha=bind(OUT/'extraction-complete.json');done=read(OUT/'extraction-complete.json');bind(OUT/'status.json')
    require(plan['rows']==ROWS and plan['dimensions']==128 and plan['feature_shape']==[ROWS,128] and plan['dtype']=='float16' and plan['added_head_ids']==ADDED,'Frozen extra2 plan geometry differs.')
    sig=plan['signature'];require(sig['source_sha256']==SOURCE_SHA and sig['checkpoint_path']=='/dev/shm/role-analysis-gptoss120b' and plan['protocol']['native_mxfp4'] is True and plan['protocol']['tests_read'] is False,'Frozen source/native checkpoint scope differs.')
    for n,want in sig['dependencies_sha256'].items():bind(HERE/n,want)
    for p,want in sig['source_files_sha256'].items():bind(p,want)
    selected=sig['selection'];require(selected['selected_head_ids']==ORIGINAL+ADDED and selected['added_head_ids']==ADDED and selected['checkpoint_path']==sig['checkpoint_path'] and selected['kernel_revision']==sig['kernel_revision'] and selected['audit_sha256']==SCAN_AUDIT_SHA,'Scanner dependency identity differs.')
    for p,want in selected['files_sha256'].items():bind(p,want)
    bind(SCAN_OUT/'scan-complete.json',SCAN_COMPLETE_SHA);bind(SCAN_OUT/'head-selection.json',SCAN_SELECTION_SHA);audit_path=SCAN_OUT/'actual-audit/actual-scan-audit.json';bind(audit_path,SCAN_AUDIT_SHA);scan_audit=read(audit_path)
    require(scan_audit['stage']=='actual_completed_complementary_head_scan_audit' and scan_audit['audit_passed'] is True and scan_audit['scanner_source_sha256']==SCAN_SOURCE_SHA and scan_audit['scanner_complete_sha256']==SCAN_COMPLETE_SHA and scan_audit['head_selection_sha256']==SCAN_SELECTION_SHA
        and scan_audit['scanner_plan_sha256']==selected['scanner_plan_sha256'] and scan_audit['selected_head_ids']==ORIGINAL+ADDED and scan_audit['admissible'] is True and scan_audit['test_data_or_results_read'] is False,'Independent scan audit not exact approved result.')
    checkpoint=Path(sig['checkpoint_path']);require(read(checkpoint/'config.json')['quantization_config']['quant_method']=='mxfp4','Checkpoint is not native MXFP4.')
    require(done['plan_sha256']==PLAN_SHA and done['source_sha256']==SOURCE_SHA and done['signature']==sig and done['rows']==ROWS and done['dimensions']==128 and done['feature_shape']==[ROWS,128] and done['dtype']=='float16'
        and done['added_head_ids']==ADDED and done['original_head_ids']==ORIGINAL and done['head_dim']==64 and done['all_rows_written'] is True and done['all_finite'] is True and done['tests_read'] is False and done['readout_trained'] is False and done['old_banks_deleted'] is False,'Complete geometry/ownership/scope differs.')
    require(done['features_path']==str(CACHE/'features.f16') and done['rows_path']==str(CACHE/'rows.parquet'),'Output cache paths changed.')
    parent_tables={};parent_feature_hashes={}
    for name in ['old','c4','natural']:
        parentpath=PARENT_COMPLETES[name];bind(parentpath,PARENT_COMPLETE_SHA[name]);parent=read(parentpath);require(parent['selected_head_ids']==ORIGINAL and parent['all_rows_written'] is True and parent['all_finite'] is True,'Parent original8 identity differs.')
        bind(BANKS[name]/'rows.parquet',parent['rows_sha256']);parent_feature_hashes[name]=bind(BANKS[name]/'features.f16',parent['features_sha256'])
        parent_tables[name]=metadata(BANKS[name]/'rows.parquet',filters=[('split','=','validation')] if name=='c4' else None)
    require(np.array_equal(parent_tables['old'].cache_ix,np.arange(755282)) and np.array_equal(parent_tables['c4'].cache_ix,np.arange(327680,389120)) and np.array_equal(parent_tables['natural'].cache_ix,np.arange(80116)),'Parent cache_ix order/range changed.')
    rows=project(parent_tables);inputs=[]
    for name,source in sig['sources'].items():
        wanted=rows[rows.input_source.eq(name)];frame=pd.read_parquet(Path(source['cache'])/'inputs.parquet',columns=['prompt_ix','length','split','wrapper'],filters=[('split','in',['validation'] if name=='c4' else ['fit','validation']),('wrapper','=','native')]);frame=frame[frame.prompt_ix.isin(wanted.source_prompt_ix)].copy()
        require(frame.prompt_ix.is_unique and set(frame.prompt_ix)==set(wanted.source_prompt_ix),'Native source prompt membership changed.')
        require(np.array_equal(frame.set_index('prompt_ix').loc[wanted.source_prompt_ix,'split'],wanted.split),'Native source row/prompt split differs.')
        frame['source_prompt_ix']=frame.prompt_ix;frame['input_source']=name;frame['uid']=name+':'+frame.prompt_ix.astype(str);inputs.append(frame)
    layout=Layout(rows,pd.concat(inputs,ignore_index=True));require(layout.descriptor()==plan['data'],'Independent full row/prompt/batch descriptor differs.')
    row_sha=bind(CACHE/'rows.parquet',done['rows_sha256']);saved=metadata(CACHE/'rows.parquet')
    mapping=pd.read_parquet(CACHE/'rows.parquet',columns=['parent_bank','parent_cache_ix','source_prompt_ix','input_source','uid'])
    for c in mapping:saved[c]=mapping[c]
    require(len(saved)==ROWS and set(rows.columns)==set(saved.columns),'Saved metadata fields or row count changed.')
    pd.testing.assert_frame_equal(saved[rows.columns],rows,check_dtype=False,check_exact=True)
    owner=read(CACHE/'initialized.json');bind(CACHE/'initialized.json');require(owner==dict(plan_sha256=PLAN_SHA,rows=ROWS,dimensions=128,dtype='float16',row_mapping_sha256=layout.descriptor()['row_mapping_sha256'],rows_sha256=row_sha),'Initialized row owner differs.')
    fp=CACHE/'features.f16';require(fp.stat().st_size==ROWS*128*2,'Missing/truncated/oversized feature bank.');feature_sha=bind(fp,done['features_sha256']);a=np.memmap(fp,mode='r',dtype=np.float16,shape=(ROWS,128));zero=0
    for start in range(0,ROWS,4096):block=a[start:start+4096];require(np.isfinite(block).all(),'Incomplete/nonfinite bank.');zero+=int(np.all(block==0,axis=1).sum())
    require(done['zero_row_count']==zero and done['all_rows_nonzero']==(zero==0),'Zero-row diagnostic differs; zero heads remain legal.')
    bind(CACHE/'progress.json',done['progress_sha256']);progress=read(CACHE/'progress.json');state=verify_progress(layout,a,progress,PLAN_SHA);proof=progress['portable_consumer']
    require(proof==done['portable_consumer']==read(OUT/'portable-validation.json'),'FIT portability proof differs.');bind(OUT/'portable-validation.json')
    close(max(progress['batch_f16_roundtrip_max'],default=0),done['maximum_f16_roundtrip_difference'],'Full cast-roundtrip summary differs.')
    for name,key in [('hook-validation.json','hook_validation_sha256'),('model-backend.json','model_backend_sha256')]:bind(OUT/name,done[key])
    hook=read(OUT/'hook-validation.json');require(all(hook[k] is True for k in ['hidden_states_bitexact','writes_disabled','feature_rows_unchanged','transient_rows_unchanged']) and hook['layers']==sorted(set(h//64 for h in ORIGINAL+ADDED)) and hook['sample_rows']==len(layout.targets(layout.parts[0])),'Disabled-hook first batch proof differs.')
    backend=read(OUT/'model-backend.json');device_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}}
    require(backend['expert_backends']==['Mxfp4GptOssExperts'] and backend['native_mxfp4'] is True and backend['parameters_frozen'] is True and backend['checkpoint_path']==str(checkpoint) and backend['device_map']==device_map
        and len(backend['gpus'])==len(backend['compute_capabilities'])==4 and all('A40' in g for g in backend['gpus']) and all(c==[8,6] for c in backend['compute_capabilities']) and backend['gpu_transfer']=='Synchronous CPU staging between different CUDA devices','Native frozen4A40 backend evidence differs.')
    samplepath=OUT/'portable-fit-sample.npz';bind(samplepath,proof['sample_sha256'])
    with np.load(samplepath,allow_pickle=False) as z:sample={k:z[k].copy() for k in z.files}
    oldmodel=BASE/'systemness-head-scan/models/top8-lda.npz';bind(oldmodel)
    with np.load(oldmodel,allow_pickle=False) as z:oldw,oldb=z['weight'].copy(),z['bias'].copy()
    proxy=SCAN_OUT/'models/selected10-proxy-lda.npz';bind(proxy)
    with np.load(proxy,allow_pickle=False) as z:
        require(z['selected_head_ids'].tolist()==ORIGINAL+ADDED and z['weight'].shape==(10,5,64) and z['bias'].shape==(10,5),'Proxy head order/dimensions differ.');ew,eb=z['weight'][8:].copy(),z['bias'][8:].copy()
    require(oldw.shape==(8,5,64) and oldb.shape==(8,5),'Original8 LDA consumer malformed.')
    oldfile=np.memmap(BANKS['old']/'features.f16',mode='r',dtype=np.float16,shape=(755282,512));evidence=parity(layout,a,proof,sample,oldfile,oldw,oldb,ew,eb)
    terminal_gate()
    for p,want in bindings.items():require(digest(p)==want,'Artifact changed during audit: '+p)
    report=dict(stage='actual_completed_complementary_head_feature_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),extractor_source_sha256=SOURCE_SHA,extractor_plan_sha256=PLAN_SHA,extractor_complete_sha256=done_sha,
        features_sha256=feature_sha,rows_sha256=row_sha,rows=ROWS,dimensions=128,shape=[ROWS,128],dtype='float16',original_head_ids=ORIGINAL,added_head_ids=ADDED,parent_counts=COUNTS,parent_feature_sha256=parent_feature_hashes,data=layout.descriptor(),prefix=state,all128_batch_hashes_verified=len(layout.parts),all_finite=True,zero_row_count=zero,portable_fit_evidence=evidence,
        scanner_complete_sha256=SCAN_COMPLETE_SHA,scanner_selection_sha256=SCAN_SELECTION_SHA,scanner_actual_audit_sha256=SCAN_AUDIT_SHA,provenance_sha256=bindings,native_MXFP4_four_A40_producer_evidence_bound=True,native_LM_independently_replayed=False,producer_GPU_parity_independently_replayed=False,
        limitation='No original LM activations were persisted. Independent CPU audit verifies full F16 bytes/metadata/batch prefix and saved FIT256 LDA consumer; native forward/hook/GPU parity evidence is producer-attested and hash-bound.',test_data_or_results_read=False,source_text_materialized=False,LM_loaded=False,gpu_used=False,formal_outputs_written=False)
    p=Path(output);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),source_sha256=report['auditor_source_sha256'],rows=ROWS,batches=len(layout.parts))))


def preflight():
    folder=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-feature-audit-'));start=digest(__file__);checks=[]
    def check(name,v=True):require(v,'CPU check failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except RuntimeError:checks.append(name);return
        raise RuntimeError('Expected rejection: '+name)
    def forbidden(*a,**k):raise LookupError('Production artifact read forbidden in CPU fixture.')
    with patch.dict(globals(),dict(processes=lambda:[123],read=forbidden,digest=forbidden)),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden):
        reject('live_producer_rejected_before_artifact_or_array_IO',lambda:actual(folder/'forbidden.json'))
    fake=folder/'not-complete';fake.mkdir();(fake/'extraction-complete.json').write_text('{}');(fake/'status.json').write_text('{"stage":"extracting-added2"}')
    with patch.dict(globals(),dict(OUT=fake,processes=lambda:[])),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden):
        reject('nonterminal_status_rejected_before_parent_metadata',lambda:actual(folder/'forbidden.json'))
    old=pd.DataFrame(dict(cache_ix=np.arange(257),sample_ix=2_000_000_000+np.arange(257),wrapper='native',split='fit',profile='long-full',source_set='long-full',prompt_ix=7,token_ix=np.arange(257),token_id=30+np.arange(257),role='system'))
    old.loc[256,['source_set','prompt_ix','token_ix','profile']]=['main',1,3,'single']
    c4=pd.DataFrame(dict(cache_ix=[327680,327681],sample_ix=[6_000_000_327680,6_000_000_327681],wrapper='native',split='validation',profile='long-full',source_set='long-augmentation',prompt_ix=[1280,1281],token_ix=[1,3],token_id=[20,22],role='user'))
    natural=pd.DataFrame(dict(cache_ix=[0,1,2],sample_ix=4_000_000_000_000+np.arange(3),wrapper='native',split='fit',profile='natural-public-sua',source_set='natural-dev-clean',prompt_ix=[0,1,2],original_prompt_ix=[11,12,13],token_ix=[0,1,2],token_id=[31,32,33],role=['system','user','assistant']))
    parents=dict(old=old,c4=c4,natural=natural);counts=dict(old=257,c4=2,natural=3);rows=project(parents,counts)
    check('all_parent_indices_samples_and_new_contiguous_indices',rows.cache_ix.tolist()==list(range(262)) and rows.iloc[257:259].parent_cache_ix.tolist()==[327680,327681] and rows.iloc[-1].sample_ix==natural.sample_ix.iloc[-1] and rows.iloc[-1].uid=='natural:13')
    bad=copy.deepcopy(parents);bad['c4'].loc[0,'split']='fit';reject('new_C4_fit_and_test_excluded',lambda:project(bad,counts))
    bad=copy.deepcopy(parents);bad['natural'].loc[0,'role']='cot';reject('public_CoT_truth_never_invented',lambda:project(bad,counts))
    bad=copy.deepcopy(parents);bad['natural'].loc[0,'sample_ix']=old.sample_ix.iloc[0];reject('cross_parent_sample_alias_rejected',lambda:project(bad,counts))
    inputs=rows[['uid','source_prompt_ix','input_source','split']].drop_duplicates().copy();inputs['length']=[600,10,30,40,50,60,70];layout=Layout(rows,inputs)
    check('fixed256FIT_priority_and_two_complete_batches',np.array_equal(layout.parity,np.arange(256)) and layout.inputs.uid.iloc[0]=='old:long-full:7' and len(layout.parts)==2)
    require(layout.descriptor()['sample_ix_sha256']==arrsha(rows.sample_ix.to_numpy()),'Row mapping sample digest failed.')
    a=np.broadcast_to(np.arange(128,dtype=np.float16)/np.float16(128),(262,128)).copy();hashes=[batch_sha(a,layout.targets(p)) for p in layout.parts]
    progress=dict(plan_sha256='toy',**prefix(layout,2),batch_features_sha256=hashes,batch_f16_roundtrip_max=[0.,0.])
    check('full128_bytes_two_batch_prefix_verified',verify_progress(layout,a,progress,'toy')['rows_written']==262)
    wrong=copy.deepcopy(progress);wrong['sample_ix_sha256']='0'*64;reject('prefix_sample_order_digest_tamper_rejected',lambda:verify_progress(layout,a,wrong,'toy'))
    wrong=copy.deepcopy(progress);wrong['batch_features_sha256']=hashes[:1];reject('partial_prefix_cannot_claim_complete',lambda:verify_progress(layout,a,wrong,'toy'))
    a[261,127]+=1;reject('last_extra_head_last_column_finite_byte_tamper',lambda:verify_progress(layout,a,progress,'toy'));a[261,127]=np.nan
    reject('NaN_single_column_cannot_commit',lambda:batch_sha(a,np.array([261])));a[261,127]=np.float16(127/128)
    zero=np.zeros((2,128),np.float16);check('legal_zero_head_vectors_finite_and_hashable',batch_sha(zero,np.arange(2))==arrsha(zero))
    rng=np.random.RandomState(2026);oldw=rng.normal(size=(8,5,64));oldb=rng.normal(size=(8,5));ew=rng.normal(size=(2,5,64));eb=rng.normal(size=(2,5));reference=np.zeros((257,512),np.float16)
    sample=dict(extra_features=a[layout.parity].copy(),transient_original8=reference[:256].copy(),target_cache_ix=layout.parity.copy(),sample_ix=rows.iloc[layout.parity].sample_ix.to_numpy().copy())
    def evidence(saved):return dict(rows=256,same_saved_f16_vectors=True,classifier_device='cuda:0',saved_vectors_sha256=arrsha(saved),tolerance=TOL,max_difference=0.)
    proof=dict(rows=256,fit_only=True,validation_probabilities_computed=False,original8_transient_only=True,no_new_readout=True,tolerance=TOL,target_cache_ix_sha256=arrsha(layout.parity),sample_ix_sha256=arrsha(rows.iloc[layout.parity].sample_ix.to_numpy()),original8_reference_probability_max_difference=0.,original8_reference_vectors_bitexact=True,original8_savedF16_CPU_GPU=evidence(sample['transient_original8']),extra2_savedF16_CPU_GPU=evidence(sample['extra_features']))
    e=parity(layout,a,proof,sample,reference,oldw,oldb,ew,eb);check('independent_savedF16_FIT256_old8_extra2_LDA_consumers',e['rows']==256 and e['original8_reference_probability_maximum_difference']==0 and not e['GPU_independently_replayed'])
    wrong=copy.deepcopy(proof);wrong['validation_probabilities_computed']=True;reject('parity_scope_cannot_use_validation_probabilities',lambda:parity(layout,a,wrong,sample,reference,oldw,oldb,ew,eb))
    wrong=copy.deepcopy(sample);wrong['extra_features'][0,127]+=1;reject('FIT_sample_must_equal_committed_extra_bank_bytes',lambda:parity(layout,a,proof,wrong,reference,oldw,oldb,ew,eb))
    wrong=copy.deepcopy(proof);wrong['original8_savedF16_CPU_GPU']['max_difference']=np.nan;reject('nonfinite_producer_GPU_parity_evidence_rejected',lambda:parity(layout,a,wrong,sample,reference,oldw,oldb,ew,eb))
    check('source_unchanged',digest(__file__)==start)
    report=dict(stage='synthetic_CPU_complementary_feature_auditor_preflight',audit_passed=True,source_sha256=start,checks=checks,check_count=len(checks),actual_artifacts_read=False,real_layout_or_feature_values_read=False,real_probabilities_or_weights_read=False,production_modules_imported=False,source_text_materialized=False,LM_loaded=False,gpu_used=False,actual_run=False)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),source_sha256=start,checks=len(checks))))

def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group();g.add_argument('--preflight',action='store_true');g.add_argument('--actual',action='store_true');parser.add_argument('--output',default='/tmp/confirmation_review_complementary_feature_results_audit.json');a=parser.parse_args()
    if a.preflight:preflight()
    elif a.actual:actual(a.output)
    else:parser.print_help()
if __name__=='__main__':main()
