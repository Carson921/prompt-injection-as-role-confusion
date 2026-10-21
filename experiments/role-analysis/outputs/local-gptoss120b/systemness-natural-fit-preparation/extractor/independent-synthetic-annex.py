#!/usr/bin/env python3
"""Independent projected-dev / CPU synthetic audit. Never actual declare/extract or probabilities."""
import copy,hashlib,json,pathlib,sys,tempfile,inspect
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=pathlib.Path('/root/prompt-injection-as-role-confusion');sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_natural_dev_extract as s
PATH=pathlib.Path(s.__file__)
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
SOURCE=digest(PATH);CHECKS=[];BINDINGS={}
def bind(p,want=None):
 p=pathlib.Path(p);d=digest(p);assert want is None or d==want;BINDINGS[str(p.resolve())]=d;return d
def check(name,x=True):assert bool(x),name;CHECKS.append(name)
def reject(name,fn):
 try:fn()
 except(RuntimeError,ValueError,KeyError,AssertionError,FileNotFoundError):CHECKS.append(name)
 else:raise AssertionError('accepted fault: '+name)
def banned(*a,**k):raise AssertionError('Actual source/p/model/declare/extract forbidden')
declare_impl=s.declare
formal=(s.OUT.exists(),s.CACHE.exists());folder=pathlib.Path(tempfile.mkdtemp(prefix='feature-review-natural-dev-extract-audit-',dir='/tmp'))
with patch.object(s,'declare',side_effect=banned),patch.object(s,'extract',side_effect=banned),patch.object(s,'load_data',side_effect=banned),patch.object(s.dev,'verify_dev',side_effect=banned),patch.object(s.data.runtime,'load_model',side_effect=banned),patch.object(s.portable,'predict_raw',side_effect=banned):
 bind(s.OLD_MODEL,s.OLD_MODEL_SHA256);bind(s.SCALER)
 head=pathlib.Path(s.data.runtime.OUT)/'systemness-head-scan/head-selection.json';bind(head)
 assert json.loads(head.read_text())['selected_head_ids']==s.HEAD_IDS
 complete=json.loads((s.dev.OUT/'data-complete.json').read_text());bind(s.dev.OUT/'data-complete.json',s.DEV_COMPLETE_SHA256)
 for name,want in complete['cache_sha256'].items():bind(s.dev.CACHE/name,want)
 for name,want in complete['metadata_sha256'].items():bind(s.dev.OUT/name,want)
 inputs=pd.read_parquet(s.dev.CACHE/'inputs.parquet',columns=s.INPUT_COLUMNS);rows=pd.read_parquet(s.dev.CACHE/'rows.parquet',columns=s.ROW_COLUMNS)
 a,b,q=s.filter_layout(inputs,rows)
 wanted=inputs[(inputs.split.eq('fit')&inputs.eligible_fit.eq(True))|inputs.split.eq('validation')]
 independent=rows[rows.prompt_ix.isin(wanted.prompt_ix)].copy().reset_index(drop=True)
 check('Exact319conversations/80116nativecontentrows',len(a)==319 and len(b)==80116 and len(wanted)==319)
 check('Whole contaminated fit conversation320rows excluded',len(inputs)-len(wanted)==1 and len(rows)-len(independent)==320 and set(q['removed_source_uids']).isdisjoint(b.source_uid))
 check('255fit63403rows plus64validation16713rows',a.split.value_counts().to_dict()=={'fit':255,'validation':64} and b.split.value_counts().to_dict()=={'fit':63403,'validation':16713})
 for column in ['sample_ix','token_ix','token_id','role','semantic_role','message_ix','source_uid','split']:
  check('Original row identity unchanged '+column,np.array_equal(b[column].to_numpy(),independent[column].to_numpy()))
 check('Continuous newcache and preserved original gap',np.array_equal(b.cache_ix,np.arange(80116)) and np.array_equal(b.original_cache_ix,independent.cache_ix) and b.sample_ix.min()>=4_000_000_000_000 and int(b.original_cache_ix.max())==80435)
 check('Clean split/SUA support and no invented C4 IDs',set(b.role)=={'system','user','assistant'} and set(b.source_set)=={'natural-dev-clean'} and 'document_id' not in b and not set(b[b.split.eq('fit')].source_uid)&set(b[b.split.eq('validation')].source_uid))
 families={split:dict(families=int(a[a.split.eq(split)].family_id.nunique()),components=int(a[a.split.eq(split)].dependency_component_id.nunique())) for split in ['fit','validation']}
 check('Actual family/component counts32fit16validation',families=={'fit':{'families':32,'components':32},'validation':{'families':16,'components':16}})
 bundle=s.Bundle(a,b);alltargets=[]
 fixed_fit=b.loc[b.split.eq('fit'),'cache_ix'].to_numpy(np.int64)[:256]
 priority=b.iloc[fixed_fit].prompt_ix.drop_duplicates().tolist()
 ordinary=a.sort_values(['length','prompt_ix']).prompt_ix.tolist()
 expected_order=priority+[p for p in ordinary if p not in set(priority)]
 check('Fixed256FITparity and metadata-only priority independently exact',np.array_equal(bundle.parity_targets,fixed_fit) and len(priority)<=4 and [p for part in bundle.parts for p in part]==expected_order and set(fixed_fit)<=set(bundle.targets(bundle.parts[0])))
 with patch.object(s.portable,'load_model',side_effect=banned):
  reject('Validation parity rejected before portablemodel/pGPU',lambda:s.same_saved_vector_parity(None,b.loc[b.split.eq('validation'),'cache_ix'].to_numpy(np.int64)[:256],{},bundle))
 # Synthetic token arrays have real lengths and IDs only at projected content positions;
 # they do not read/decode actual prompt strings or any public/source assistant.
 tokens=[[999]*int(length) for length in a.length]
 for r in b.itertuples():tokens[r.prompt_ix][r.token_ix]=int(r.token_id)
 bundle.tokens=tokens
 for part in bundle.parts:
  batch=bundle.collate(part);targets=batch['targets'];alltargets.extend(targets.tolist())
  got=batch['input_ids'].reshape(-1)[batch['positions']].numpy()
  assert np.array_equal(got,b.iloc[targets].token_id.to_numpy())
  assert np.array_equal(targets,bundle.targets(part))
  assert np.all(batch['attention_mask'].reshape(-1)[batch['positions']].numpy()==1)
 check('All80batches/80116leftpadded mapping exact',len(bundle.parts)==80 and sorted(alltargets)==list(range(80116)))
 prefixes={}
 for n in [0,1,16,len(bundle.parts)]:
  prompts=[p for part in bundle.parts[:n] for p in part];ix=np.r_[tuple(bundle.targets([p]) for p in prompts)] if prompts else np.empty(0,np.int64)
  value=s.prefix_state(bundle,n);assert value['rows_written']==len(ix) and value['target_cache_ix_sha256']==s.array_sha(ix) and value['source_sample_ix_sha256']==s.array_sha(b.iloc[ix].sample_ix.to_numpy())
  prefixes[str(n)]=value['rows_written']
 check('Independent exact0/1/16/full batch prefix ownership')
 # A pre-existing feature artifact must not gain a retrospective declaration.
 prior=folder/'predated-bank';prior.mkdir();(prior/'features.f16').write_bytes(b'not-owned')
 with patch.object(s,'OUT',folder/'not-declared'),patch.object(s,'CACHE',prior),patch.object(s,'plan_value',return_value=dict(synthetic=True)):
  reject('Bank-before-plan refused without creating output',declare_impl)
  assert not (folder/'not-declared').exists()
 # Small fixture tests persisted recovery and head geometry without actual features/model.
 ti=[];tr=[]
 for p,split in enumerate(['fit','fit','validation']):
  common=dict(source_uid='toy:'+str(p),group_id='toy:'+str(p),split=split,eligible_fit=p==0,eligible_clean_validation=p==2,wrapper='native',profile='natural-public-sua')
  ti.append(dict(common,prompt_ix=p,length=10))
  for m,role in enumerate(['system','user','assistant']):tr.append(dict(common,cache_ix=len(tr),sample_ix=4_000_000_000_000+len(tr),prompt_ix=p,token_ix=3+3*m,token_id=20+m,message_ix=m,role=role,semantic_role=role))
 ti,tr=pd.DataFrame(ti),pd.DataFrame(tr);expected=dict(source_prompts=3,source_rows=9,prompts=2,rows=6,fit_prompts=1,validation_prompts=1,fit_rows=3,validation_rows=3,removed_fit_prompts=1,removed_fit_rows=3,role_counts={split:{role:1 for role in s.ROLES} for split in ['fit','validation']})
 ta,tb,_=s.filter_layout(ti,tr,expected);toy=s.Bundle(ta,tb)
 wrong=ti.copy();wrong.loc[2,'eligible_clean_validation']=False;reject('Unclean validation rejected rather than silently dropped',lambda:s.filter_layout(wrong,tr,expected))
 with patch.object(s,'CACHE',folder/'bank'):
  p=folder/'rows.parquet';tb.to_parquet(p,index=False);array=s.bank_helpers.initialize(s.CACHE,p,'toy',rows=6,dimensions=512)
  check('New bank initialized NaN',np.isnan(array).all())
  array[0]=0;check('Uncommitted partialbatch gives no completion',s.restore_progress(toy,'toy',array)==([],None))
  target=toy.targets(toy.parts[0]);array[target]=np.arange(6*512,dtype=np.float32).reshape(6,512)/64
  h=s.bank_helpers.batch_digest(array,target);s.save_progress(toy,'toy',[h],dict(fit_only=True,validation_probabilities_computed=False,sampled_rows=len(toy.parity_targets),target_cache_ix_sha256=s.array_sha(toy.parity_targets),source_sample_ix_sha256=s.array_sha(toy.rows.iloc[toy.parity_targets].sample_ix.to_numpy())))
  check('Saved completeprefix restores exact batchhash',s.restore_progress(toy,'toy',array)[0]==[h])
  bad_parity=dict(fit_only=True,validation_probabilities_computed=True,sampled_rows=len(toy.parity_targets),target_cache_ix_sha256=s.array_sha(toy.parity_targets),source_sample_ix_sha256=s.array_sha(toy.rows.iloc[toy.parity_targets].sample_ix.to_numpy()))
  reject('Resume parity rejects validationprobability flag',lambda:s.validate_parity_scope(toy,bad_parity))
  reject('Wrong plan refuses progress',lambda:s.restore_progress(toy,'wrong',array))
  array[target[0],0]+=1;reject('Committed byte tamper rejected',lambda:s.restore_progress(toy,'toy',array))
  del array;(s.CACHE/'features.f16').unlink();reject('Missing initialized bank cannot be recreated',lambda:s.bank_helpers.initialize(s.CACHE,p,'toy',rows=6,dimensions=512))
 import torch
 collector=s.shared.Collector(s.HEAD_IDS);positions=np.array([1,7],np.int64);v=torch.arange(8*4096,dtype=torch.float32).reshape(2,4,4096)/512
 collector.begin(positions,False)
 for layer in collector.layers:assert collector.hook(layer)(None,(v,)) is None
 collector.finish();check('Disabled prehooks returnNone and allocate no features',collector.features is None)
 collector.begin(positions,True)
 for layer in reversed(collector.layers):assert collector.hook(layer)(None,(v+layer,)) is None
 collector.finish();want=np.concatenate([(v+h//64).reshape(-1,64,64)[positions,h%64].numpy().astype(np.float16) for h in s.HEAD_IDS],axis=1)
 check('Unsorted selectedrank survives reversed layer execution',np.array_equal(collector.features,want))
 collector.begin(positions,True);overflow=torch.full_like(v,65536,dtype=torch.bfloat16)
 reject('BF16 outsideF16finite range rejected without clipping',lambda:collector.hook(collector.layers[0])(None,(overflow,)))
 verify_source=inspect.getsource(s.verified_extraction)
 check('Completed route contains no prompt/model/GPU load',all(x not in verify_source for x in ['load_data(','.load_model(','.cuda(',"load_tokenizer("]))
 check('No dynamic result verifiers or newseed/aug model results in signature',all(x not in inspect.getsource(s.source_signature)+inspect.getsource(s.fixed_identity)+inspect.getsource(s.verified_source_metadata) for x in ['verified_completion(','verified_extraction(','rglob(','.selection(']))
 check('Formal output/cache untouched',formal==(s.OUT.exists(),s.CACHE.exists()))
 check('Audited source unchanged',digest(PATH)==SOURCE)
result=dict(stage='independent_natural_dev_extractor_source_CPU_audit',audit_passed=True,source_sha256=SOURCE,source_sha256_after=digest(PATH),checks_passed=len(CHECKS),checks=CHECKS,
 metadata=dict(prompts=319,content_rows=80116,fit_prompts=255,fit_rows=63403,validation_prompts=64,validation_rows=16713,excluded_fit_prompts=1,excluded_fit_rows=320,role_counts=q['role_counts'],families=families,lm_input_tokens=int(a.length.sum()),features_f16_bytes=80116*512*2,parity_fit_rows=256,parity_fit_cache_ix_sha256=s.array_sha(fixed_fit),parity_fit_sample_ix_sha256=s.array_sha(b.iloc[fixed_fit].sample_ix.to_numpy()),parity_fit_prompt_count=len(priority),prefix_rows=prefixes),
 provenance_sha256=BINDINGS,real_dev_prompt_strings_read=False,source_or_test_assistant_decoded=False,probability_artifacts_read=False,LM_loaded=False,gpu_used=False,formal_declare_or_extract_run=False,
 limitation='Full committed feature prefix lacking final extraction completion is deliberately refused; automatic completion recovery is not claimed. This is extraction readiness, not training/accuracy evidence.')
p=folder/'report.json';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),auditor_source_sha256=digest(__file__),**result)))
