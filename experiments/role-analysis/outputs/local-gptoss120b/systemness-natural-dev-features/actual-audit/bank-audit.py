"""Independent clean-natural development bank audit. Default help; --preflight is synthetic.

--actual is permitted only after root confirms extractor terminal exit0. Never
loads an LM/readout, computes probabilities, reads validation/test p, or writes
production files. Existing FIT-only portable sample p are checked numerically.
"""
import argparse,hashlib,json,os
from pathlib import Path
from unittest.mock import patch
import tempfile
import numpy as np
import pandas as pd
ROOT=Path('/root/prompt-injection-as-role-confusion');HERE=ROOT/'experiments/role-analysis';BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-natural-dev-features';BANK=Path('/dev/shm/role-analysis-natural-dev-heads')
DEV=BASE/'systemness-natural-transcripts';CACHE=BASE/'systemness-expansion/cache/natural-transcripts-dev'
SOURCE=HERE/'systemness_natural_dev_extract.py'
SOURCE_SHA='fb1ad8dc5f0642a64c674fb66f1dd8f77c854cb0e6ec941fcdace29b3334c4a1'
PLAN_SHA='0b52adae0244dbe17bc4c17562847ef27637e4d7f35c310ad2a9faa30767d3b5'
DEV_SHA='b6c411c920f583a128401c0d787869e2eba17b632136a5cec9613fd227bb9dfd'
MODEL_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
HEADS=[458,462,2045,460,459,2047,456,997]
ROWS,DIMS,PROMPTS=80116,512,319
ROLES=['system','user','assistant']
ROLE_COUNTS={'fit':{'system':7935,'user':12660,'assistant':42808},'validation':{'system':2032,'user':3596,'assistant':11085}}
INPUT_COLUMNS=['prompt_ix','length','turns','message_count','wrapper','profile','group_id','source_uid','dataset','native_id','split','template_id','family_id','eligible_fit','eligible_clean_validation','old_generated_collision','cross_split_assistant_collision','dependency_component_id']
ROW_COLUMNS=['cache_ix','sample_ix','prompt_ix','wrapper','profile','group_id','source_uid','dataset','native_id','split','template_id','family_id','token_ix','token_id','message_ix','role','semantic_role','eligible_fit','eligible_clean_validation','old_generated_collision','cross_split_assistant_collision','dependency_component_id']

def require(x,message):
 if not x:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def frame_sha(frame,cols):return hashlib.sha256(frame[cols].to_csv(index=False).encode()).hexdigest()

def live_extractors():
 active=[]
 for e in Path('/proc').iterdir():
  if not e.name.isdigit() or int(e.name)==os.getpid():continue
  try:args=(e/'cmdline').read_bytes().split(b'\0')
  except(FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(Path(a.decode(errors='replace')).name=='systemness_natural_dev_extract.py' for a in args if a):active.append(int(e.name))
 return active

def terminal_gate():
 require(not live_extractors(),'Extractor remains live; no partial-bank/results inspection.')
 require((OUT/'extraction-complete.json').is_file(),'Extractor completion absent.')
 # Source intentionally leaves its final checkpoint status at this stage.
 status=read(OUT/'status.json')
 require(status==dict(stage='natural-dev-extraction',batch=80,batches=80,rows=80116),'Extractor final80batch status incomplete.')

def select_rows(inputs,rows):
 require(len(inputs)==320 and len(rows)==80436 and np.array_equal(inputs.prompt_ix,np.arange(320))
  and np.array_equal(rows.cache_ix,np.arange(80436)) and rows.sample_ix.is_unique,'Original dev metadata identity changed.')
 selected=inputs.loc[(inputs.split.eq('fit')&inputs.eligible_fit.eq(True))|inputs.split.eq('validation')].copy()
 removed=inputs.loc[~inputs.prompt_ix.isin(selected.prompt_ix)]
 require(len(removed)==1 and removed.split.eq('fit').all() and selected[selected.split.eq('validation')].eligible_clean_validation.all(),'Whole-fit quarantine/clean validation changed.')
 mapping={int(old):new for new,old in enumerate(selected.prompt_ix)}
 actual=rows.loc[rows.prompt_ix.isin(mapping)].copy();require(len(rows)-len(actual)==320,'Whole contaminated conversation320rows were not removed.')
 selected['original_prompt_ix']=selected.prompt_ix;selected['prompt_ix']=np.arange(len(selected))
 actual['original_prompt_ix']=actual.prompt_ix;actual['original_cache_ix']=actual.cache_ix;actual['prompt_ix']=actual.prompt_ix.map(mapping);actual['cache_ix']=np.arange(len(actual))
 for f in [selected,actual]:f['source_set']='natural-dev-clean';f['uid']='natural-dev:'+f.source_uid
 selected=selected.reset_index(drop=True);actual=actual.reset_index(drop=True)
 actual['body_offset']=actual.token_ix-actual.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
 require(len(selected)==319 and len(actual)==80116 and selected.split.value_counts().to_dict()=={'fit':255,'validation':64}
  and actual.split.value_counts().to_dict()=={'fit':63403,'validation':16713} and set(actual.role)==set(ROLES)
  and actual.semantic_role.eq(actual.role).all() and not set(actual.source_uid)&set(removed.source_uid)
  and not set(actual[actual.split.eq('fit')].source_uid)&set(actual[actual.split.eq('validation')].source_uid),'Selected319 namespace/roles/split changed.')
 counts={split:{role:int((actual.split.eq(split)&actual.role.eq(role)).sum()) for role in ROLES} for split in ['fit','validation']}
 require(counts==ROLE_COUNTS and actual.sample_ix.is_monotonic_increasing and actual.original_cache_ix.is_monotonic_increasing,'SUA labels/order/support changed.')
 for (p,m),g in actual.groupby(['prompt_ix','message_ix'],sort=False):
  require(np.array_equal(g.body_offset,np.arange(len(g))) and len(g)<=192 and g.role.nunique()==1,'Native message content positions/cap changed.')
  expected='system' if m==0 else 'user' if m%2 else 'assistant';require(g.role.eq(expected).all(),'S/U/A message-index mapping changed.')
 return selected,actual,dict(removed_source_uids=removed.source_uid.tolist(),removed_original_prompt_ix=removed.prompt_ix.tolist(),removed_fit_body_rows=320,role_counts=counts,source_body_token_spelling_read=False)

def batches(inputs,rows):
 parity=rows.loc[rows.split.eq('fit'),'cache_ix'].to_numpy(np.int64)[:256]
 priority=rows.iloc[parity].prompt_ix.drop_duplicates().tolist()
 require(len(parity)==256 and len(priority)<=4,'Fixed FIT256 parity does not fit first batch.')
 order=inputs.sort_values(['length','prompt_ix']).prompt_ix.tolist();order=priority+[p for p in order if p not in set(priority)]
 parts=[order[i:i+4] for i in range(0,len(order),4)]
 lookup={int(p):g.cache_ix.to_numpy(np.int64) for p,g in rows.groupby('prompt_ix',sort=False)}
 targets=[np.concatenate([lookup[p] for p in part]) for part in parts]
 whole=np.concatenate(targets)
 require(len(parts)==80 and np.array_equal(np.sort(whole),np.arange(80116)) and set(parity)<=set(targets[0]),'Exactly-once80batch feature ownership/parity priority changed.')
 return parts,targets,parity,priority

def inspect_features(value):
 zeros=0;head_zeros=np.zeros(8,np.int64);low=float('inf');high=-float('inf')
 for start in range(0,len(value),4096):
  block=np.asarray(value[start:start+4096]);require(np.isfinite(block).all(),'NaN/Inf/unwritten feature row.')
  zeros+=int(np.all(block==0,axis=1).sum());head_zeros+=np.all(block.reshape(-1,8,64)==0,axis=2).sum(0)
  low=min(low,float(block.min()));high=max(high,float(block.max()))
 return dict(all_finite=True,all_rows_nonzero=zeros==0,zero_row_count=zeros,per_head_zero_row_counts=head_zeros.tolist(),minimum_feature=low,maximum_feature=high)

def validate_fit_sample(rows,features,targets,parity,source_ids,cpu,gpu):
 # Scope is verified before any probability array is inspected by the caller.
 require(np.array_equal(targets,parity) and rows.iloc[targets].split.eq('fit').all() and np.array_equal(source_ids,rows.iloc[targets].sample_ix),'Portable sample includes validation/wrong fit row IDs.')
 require(features.shape==(256,512) and features.dtype==np.float16 and np.isfinite(features).all(),'Saved FIT F16 vectors malformed.')
 for p in [cpu,gpu]:require(p.shape==(256,5) and p.dtype==np.float32 and np.isfinite(p).all() and (p>=0).all() and (p<=1.00001).all() and np.allclose(p.sum(1),1,atol=2e-5,rtol=0),'Invalid FIT-only portable probability sample.')
 delta=float(np.abs(cpu-gpu).max());require(0<=delta<5e-4,'FIT saved-vector CPU/GPU technical parity failed.');return delta

def actual(output):
 terminal_gate() # No bank/result/source metadata reads before terminal proof.
 bindings={}
 def bind(p,want=None):
  p=Path(p);value=digest(p)
  if want is not None:require(value==want,'Bound file SHA differs: '+str(p))
  bindings[str(p.resolve())]=value;return value
 bind(__file__);bind(SOURCE,SOURCE_SHA);bind(OUT/'extraction-plan.json',PLAN_SHA);bind(OUT/'status.json');bind(DEV/'data-complete.json',DEV_SHA)
 plan,complete,dev=read(OUT/'extraction-plan.json'),read(OUT/'extraction-complete.json'),read(DEV/'data-complete.json');complete_sha=bind(OUT/'extraction-complete.json')
 require(plan['source_sha256']==SOURCE_SHA==complete['source_sha256'] and complete['signature']==plan['signature']
  and complete['dependencies_sha256']==plan['dependencies_sha256']==plan['signature']['dependencies_sha256']
  and complete['plan_sha256']==PLAN_SHA and complete['rows']==80116 and complete['dimensions']==512 and complete['head_dim']==64
  and complete['feature_shape']==[80116,512] and complete['dtype']=='float16' and complete['all_rows_written'] is True
  and complete['all_finite'] is True and complete['tests_read'] is False and complete['readout_training'] is False and complete['primary_changed'] is False,'Completion geometry/dev-only source contract changed.')
 for name,want in plan['dependencies_sha256'].items():bind(HERE/name,want)
 sig=plan['signature'];identity=sig['fixed_identity']
 for name,want in sig['prepared_dev_files_sha256'].items():bind(name,want)
 for name,want in identity['files_sha256'].items():bind(name,want)
 require(identity['portable_model_sha256']==MODEL_SHA and identity['selected_head_ids']==plan['selected_head_ids']==complete['selected_head_ids']==HEADS
  and complete['ordered_layers_heads']==plan['ordered_layers_heads']==[[h//64,h%64] for h in HEADS],'Fixed original portable model/head rank changed.')
 require(dev['cache_path']==str(CACHE) and dev['source_sha256']==plan['dependencies_sha256']['systemness_natural_transcripts.py']
  and dev['classifier_probabilities_read'] is False and dev['test_assistant_decoded'] is False and dev['test_transcripts_prepared'] is False
  and dev['coverage']['all_content_tokens_scored'] is True and dev['coverage']['independent_native_labels_match'] is True,'Prepared native dev parser/source proof changed.')
 inputs=pd.read_parquet(CACHE/'inputs.parquet',columns=INPUT_COLUMNS);source_rows=pd.read_parquet(CACHE/'rows.parquet',columns=ROW_COLUMNS)
 selected,expected_rows,quarantine=select_rows(inputs,source_rows)
 require(quarantine==plan['quarantine']==complete['quarantine'],'Whole-conversation quarantine proof changed.')
 require(complete['features_path']==str(BANK/'features.f16') and complete['rows_path']==str(BANK/'rows.parquet')
  and (BANK/'features.f16').stat().st_size==80116*512*2==82038784,'Raw bank path/size/header mismatch.')
 features_sha=bind(BANK/'features.f16',complete['features_sha256']);rows_sha=bind(BANK/'rows.parquet',complete['rows_sha256'])
 bind(OUT/'selected-rows.parquet',rows_sha);actual_rows=pd.read_parquet(BANK/'rows.parquet')
 require(set(actual_rows)==set(expected_rows),'Selected row fields changed.')
 pd.testing.assert_frame_equal(actual_rows,expected_rows[actual_rows.columns],check_exact=True)
 parts,targets,parity_targets,priority=batches(selected,actual_rows);full_targets=np.concatenate(targets);prompt_order=[p for part in parts for p in part]
 descriptor=dict(inputs=319,rows=80116,batches=80,fit_prompts=255,validation_prompts=64,split_rows={'fit':63403,'validation':16713},
  input_mapping_sha256=frame_sha(selected,['prompt_ix','original_prompt_ix','uid','length','split','eligible_fit']),
  row_mapping_sha256=frame_sha(actual_rows,['cache_ix','original_cache_ix','sample_ix','prompt_ix','uid','token_ix','token_id','role','body_offset']),
  sample_ix_sha256=array_sha(actual_rows.sample_ix.to_numpy()),parity_fit_rows=256,parity_fit_cache_ix_sha256=array_sha(parity_targets),
  parity_fit_sample_ix_sha256=array_sha(actual_rows.iloc[parity_targets].sample_ix.to_numpy()),parity_fit_prompt_ix=priority,
  batch_prompt_order_sha256=array_sha(np.asarray(prompt_order,np.int64)),max_batch_rows=max(map(len,targets)))
 require(descriptor==plan['data']==complete['data'],'Independent feature/batch descriptor differs.')
 owner=read(BANK/'initialized.json');bind(BANK/'initialized.json')
 require(owner==dict(plan_sha256=PLAN_SHA,rows=80116,dimensions=512,dtype='float16',rows_sha256=rows_sha),'Initialized bank owner changed.')
 progress=read(BANK/'progress.json');bind(BANK/'progress.json',complete['progress_sha256'])
 require(progress['plan_sha256']==PLAN_SHA and progress['next_batch']==80 and progress['rows_written']==80116 and len(progress['batch_features_sha256'])==80
  and progress['target_cache_ix_sha256']==array_sha(full_targets) and progress['source_sample_ix_sha256']==array_sha(actual_rows.iloc[full_targets].sample_ix.to_numpy())
  and progress['prompt_order_sha256']==frame_sha(selected.iloc[prompt_order],['prompt_ix','original_prompt_ix','uid','length']),'Complete exact-prefix ownership changed.')
 features=np.memmap(BANK/'features.f16',mode='r',dtype=np.float16,shape=(80116,512));inspection=inspect_features(features)
 require(all(complete[k]==inspection[k] for k in ['all_finite','all_rows_nonzero','zero_row_count']),'Saved finite/zero diagnostics differ.')
 for i,ix in enumerate(targets):require(array_sha(features[ix])==progress['batch_features_sha256'][i],'Committed batch feature bytes differ: '+str(i))
 audit_names={'hook-validation.json','portable-validation.json','portable-sample.npz','model_backend.json'}
 require(set(complete['audit_sha256'])==audit_names,'Backend/hook/portable evidence scope changed.')
 for name,want in complete['audit_sha256'].items():bind(OUT/name,want)
 hooks,backend,parity=read(OUT/'hook-validation.json'),read(OUT/'model_backend.json'),read(OUT/'portable-validation.json')
 require(hooks['signature']==sig and hooks['hidden_states_bitexact'] is True and hooks['all_capture_writes_disabled'] is True
  and hooks['features_allocated'] is False and hooks['selected_head_ids']==HEADS and hooks['sample_rows']==len(targets[0]),'Disabled-write bitexact hook evidence changed.')
 device_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}}
 require(backend['expert_backends']==['Mxfp4GptOssExperts'] and backend['device_map']==device_map and backend['load_path']==identity['checkpoint_path']
  and len(backend['gpus'])==4 and all('A40' in value for value in backend['gpus']),'Actual fourA40 nativeMXFP4 backend differs.')
 checkpoint=Path(identity['checkpoint_path']);require(not str(checkpoint).startswith('/workspace'),'Forbidden original mount.')
 config=read(checkpoint/'config.json');require(config['quantization_config']['quant_method']=='mxfp4' and (config['num_hidden_layers'],config['num_attention_heads'],config['head_dim'])==(36,64,64),'Checkpoint quantization/geometry differs.')
 require(parity==complete['portable']==progress['portable'] and parity['signature']==sig and parity['fit_only'] is True
  and parity['validation_probabilities_computed'] is False and parity['sampled_rows']==256 and parity['same_saved_f16_vectors'] is True
  and parity['captured_dtype']=='float16' and parity['portable_model_sha256']==MODEL_SHA and parity['no_performance_metric'] is True
  and parity['target_cache_ix_sha256']==array_sha(parity_targets) and parity['source_sample_ix_sha256']==array_sha(actual_rows.iloc[parity_targets].sample_ix.to_numpy())
  and parity['sample_sha256']==complete['audit_sha256']['portable-sample.npz'] and parity['tolerance']==5e-4,'Fixed FIT-only portable sample proof changed.')
 with np.load(OUT/'portable-sample.npz',allow_pickle=False) as z:
  require(set(z.files)=={'features','target_cache_ix','sample_ix','cpu_probabilities','gpu_probabilities'},'FIT-only parity artifact keys differ.')
  fixed=z['target_cache_ix'];source_ids=z['sample_ix'];vectors=z['features']
  # Validate scope before accessing any probability array.
  require(np.array_equal(fixed,parity_targets) and actual_rows.iloc[fixed].split.eq('fit').all() and np.array_equal(source_ids,actual_rows.iloc[fixed].sample_ix),'FIT-only portable artifact points at validation.')
  require(np.array_equal(vectors,features[fixed]) and array_sha(vectors)==parity['features_sha256'],'Saved FIT vectors differ from actual bank.')
  delta=validate_fit_sample(actual_rows,vectors,fixed,parity_targets,source_ids,z['cpu_probabilities'],z['gpu_probabilities'])
 require(delta==parity['max_probability_difference'],'Recorded versus independently computed FIT technical parity differs.')
 coverage=complete['coverage'];require(coverage['all_native_SUA_content_positions'] is True and coverage['independent_native_labels_match'] is True
  and coverage['content_rows']==80116 and coverage['normal_system_tag'] is True and coverage['cot_tool_truth_support']==0 and coverage['old128_test_reply_decoded'] is False,'Producer actual native S/U/A coverage proof differs.')
 del features;terminal_gate()
 for path,want in bindings.items():require(digest(path)==want,'Audited input changed: '+path)
 report=dict(stage='actual_completed_natural_dev_extraction_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),
  extractor_source_sha256=SOURCE_SHA,extraction_plan_sha256=PLAN_SHA,extraction_complete_sha256=complete_sha,features_sha256=features_sha,rows_sha256=rows_sha,
  shape=[80116,512],dtype='float16',feature_bytes=82038784,descriptor=descriptor,quarantine=quarantine,complete_batches_independently_rehashed=80,
  feature_inspection=inspection,selected_head_ids=HEADS,fit_only_portable_rows=256,fit_only_portable_features_sha256=parity['features_sha256'],
  independently_recomputed_fit_CPU_GPU_max_difference=delta,actual_four_A40_MXFP4_metadata_verified=True,disabled_writes_hidden_bitexact_metadata_verified=True,
  native_full_SUA_content_mapping_matches_frozen_dev_cache=True,provenance_sha256=bindings,
  fit_only_technical_parity_probabilities_read=True,validation_probabilities_read=False,test_data_or_results_read=False,LM_or_readout_loaded=False,gpu_used=False,production_files_modified=False,
  limitation='No LM/GPU replay or native retokenization: every content role/token/offset maps to the frozen prepared dev cache, and actual parser/bitexact/GPU evidence is authenticated producer metadata. Only saved training-row technical parity probabilities were read, never validation probabilities. Original /workspace is not accessed.')
 output=Path(output);require(str(output.resolve()).startswith('/tmp/'),'Audit output must remain under /tmp.');output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=digest(__file__),features_sha256=features_sha,zero_row_count=inspection['zero_row_count'],fit_portable_max_difference=delta)),flush=True)

def preflight():
 import py_compile
 source=digest(__file__);checks=[];folder=Path(tempfile.mkdtemp(prefix='feature-review-natural-dev-bank-preflight-',dir='/tmp'))
 def check(name,x=True):require(bool(x),name);checks.append(name)
 def reject(name,fn):
  try:fn()
  except(RuntimeError,ValueError,KeyError,AssertionError,FileNotFoundError):checks.append(name)
  else:raise RuntimeError('Accepted synthetic fault: '+name)
 def banned(*a,**k):raise AssertionError('Actual metadata/p/bank access forbidden in preflight.')
 with patch(__name__+'.live_extractors',return_value=[123]),patch(__name__+'.read',side_effect=banned),patch(__name__+'.digest',side_effect=banned):
  reject('Live extractor rejected before all metadata/feature/p reads',lambda:actual(folder/'never.json'))
 with patch(__name__+'.OUT',folder/'missing'),patch(__name__+'.live_extractors',return_value=[]),patch(__name__+'.digest',side_effect=banned):
  reject('Missing completion rejected before bank read',lambda:actual(folder/'never.json'))
 fitcounts=np.full(255,248);fitcounts[:163]+=1;valcounts=np.full(64,261);valcounts[:9]+=1
 count=np.r_[fitcounts,valcounts];prompt=np.repeat(np.arange(319),count);rows=pd.DataFrame(dict(cache_ix=np.arange(80116),sample_ix=4_000_000_000_000+np.arange(80116),prompt_ix=prompt,split=np.where(prompt<255,'fit','validation')))
 inputs=pd.DataFrame(dict(prompt_ix=np.arange(319),length=1000-np.arange(319)))
 parts,targets,fixed,priority=batches(inputs,rows)
 check('Synthetic80116 rows /319 prompts exactly80batches once',len(parts)==80 and sum(map(len,targets))==80116 and len(np.unique(np.concatenate(targets)))==80116)
 check('256FIT scope has two priority prompts despite short validation',priority==[0,1] and len(fixed)==256 and rows.iloc[fixed].split.eq('fit').all() and parts[0][:2]==[0,1])
 vectors=np.zeros((256,512),np.float16);p=np.full((256,5),.2,np.float32);source_ids=rows.iloc[fixed].sample_ix.to_numpy()
 check('Exact savedFIT numerical parity zero',validate_fit_sample(rows,vectors,fixed,fixed,source_ids,p,p)==0)
 wrong=rows.index[rows.split.eq('validation')].to_numpy()[:256]
 reject('Validation targets refused before FIT probability shape checks',lambda:validate_fit_sample(rows,None,wrong,fixed,None,None,None))
 reject('Wrong original sample namespace refused',lambda:validate_fit_sample(rows,vectors,fixed,fixed,source_ids+1,p,p))
 bad=p.copy();bad[255,0]=np.nan;reject('NaN last savedFIT probability refused',lambda:validate_fit_sample(rows,vectors,fixed,fixed,source_ids,p,bad))
 bad=p.copy();bad[255,0]+=.001;bad[255,1]-=.001;reject('FIT technical parity tolerance enforced',lambda:validate_fit_sample(rows,vectors,fixed,fixed,source_ids,p,bad))
 f=np.vstack([np.zeros((1,512),np.float16),np.ones((2,512),np.float16)]);inspection=inspect_features(f)
 check('Legitimate zero features counted rather than rejected',inspection['zero_row_count']==1 and not inspection['all_rows_nonzero'] and inspection['per_head_zero_row_counts']==[1]*8)
 f[-1,-1]=np.nan;reject('Nonfinite final feature column refused',lambda:inspect_features(f))
 target=folder/'saved-fit-sample.npz';np.savez_compressed(target,features=vectors,target_cache_ix=fixed,sample_ix=source_ids,cpu_probabilities=p,gpu_probabilities=p)
 with np.load(target,allow_pickle=False) as z:
  require(np.array_equal(z['target_cache_ix'],fixed) and np.array_equal(z['sample_ix'],source_ids),'Toy scope ownership changed.')
  check('FIT-only sample byte/array hashes bind roundtrip',array_sha(z['features'])==array_sha(vectors) and digest(target))
 py_compile.compile(__file__,cfile=str(folder/'auditor.pyc'),doraise=True);require(digest(__file__)==source,'Preflight changed auditor.')
 result=dict(stage='natural_dev_bank_auditor_synthetic_preflight',audit_passed=True,auditor_source_sha256=source,source_sha256_after=digest(__file__),checks=checks,checks_passed=len(checks),
  actual_bank_or_result_read=False,actual_dev_prompt_or_source_read=False,actual_probability_artifact_read=False,LM_or_readout_loaded=False,gpu_used=False,formal_files_modified=False)
 target=folder/'report.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(target),report_sha256=digest(target),**result)),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--preflight',action='store_true');a.add_argument('--actual',action='store_true');p.add_argument('--output',default='/tmp/feature_review_natural_dev_bank_audit.json');args=p.parse_args()
 if args.preflight:preflight()
 elif args.actual:actual(args.output)
 else:p.print_help()
