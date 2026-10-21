#!/usr/bin/env python3
"""CPU metadata-only full-sequence sizing; no prompt/token-array/p decoding or source selection."""
import hashlib,json,math,pathlib,shutil
import numpy as np
import pandas as pd
ROOT=pathlib.Path('/root/prompt-injection-as-role-confusion');HERE=ROOT/'experiments/role-analysis';BASE=HERE/'outputs/local-gptoss120b'
REPORT=pathlib.Path('/tmp/feature_review_context_cache_feasibility.json');BINDINGS={}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p,w=None):
 p=pathlib.Path(p);d=sha(p);assert w is None or d==w;BINDINGS[str(p.resolve())]=d;return d
def read(p):bind(p);return json.loads(pathlib.Path(p).read_text())
def projected(p,cols,filters=None):bind(p);return pd.read_parquet(p,columns=cols,filters=filters)
def array_sha(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def group_stats(rows,inputs,bank,cohort):
 assert rows.prompt_ix.isin(inputs.prompt_ix).all() and inputs.prompt_ix.is_unique
 result=[]
 for profile,g in inputs.groupby('profile',sort=True):
  r=rows[rows.prompt_ix.isin(g.prompt_ix)]
  result.append(dict(cohort=cohort,bank=bank,profile=str(profile),prompts=len(g),full_nonpadding_input_tokens=int(g.length.sum()),
   minimum_input_length=int(g.length.min()),maximum_input_length=int(g.length.max()),current_sampled_or_content_rows=len(r),
   full_input_minus_current_rows=int(g.length.sum())-len(r),prompt_membership_sha256=array_sha(g.prompt_ix.to_numpy(np.int64))))
 return result

def main():
 bind(__file__)
 plan=read(BASE/'systemness-head-features/plan.json');sources=plan['signature']['input_sources']
 old=projected('/dev/shm/role-analysis-top8-heads/rows.parquet',['cache_ix','sample_ix','prompt_ix','source_set','split','profile','role','document_id','group_id'])
 assert len(old)==755282
 raw=old[old.split.eq('fit')];full=set(raw[raw.profile.eq('long-full')].document_id)
 fit=raw[~(raw.profile.eq('long-single')&raw.document_id.isin(full))];valid=old[old.split.eq('validation')]
 removed=raw[~raw.cache_ix.isin(fit.cache_ix)];assert len(fit)==578090 and len(valid)==173032 and len(removed)==4160
 tables=[];selected_inputs=[]
 for bank,s in sources.items():
  path=pathlib.Path(s['folder'])/'inputs.parquet';bind(path,s['cache_sha256']['inputs.parquet'])
  # Projection excludes prompt bodies; filtering excludes every test/non-native input before metadata enters Python.
  inp=projected(path,['prompt_ix','split','profile','wrapper','length'],[('split','in',['fit','validation']),('wrapper','=','native')])
  for cohort,frame in [('original-fit',fit),('original-old-validation',valid)]:
   r=frame[frame.source_set.eq(bank)];i=inp[inp.prompt_ix.isin(r.prompt_ix)].copy()
   assert set(i.prompt_ix)==set(r.prompt_ix) and i.split.eq('fit' if cohort=='original-fit' else 'validation').all()
   if len(i):tables+=group_stats(r,i,bank,cohort);selected_inputs.append((cohort,bank,i))
  # All original755282 membership is only a check against the already frozen original input-token descriptor.
  before=inp[inp.prompt_ix.isin(old[old.source_set.eq(bank)].prompt_ix)]
  assert len(before)==old[old.source_set.eq(bank)].prompt_ix.nunique()
  if bank=='supplement':
   rem=before[before.prompt_ix.isin(removed[removed.source_set.eq(bank)].prompt_ix)]
   removed_prompts,removed_tokens=len(rem),int(rem.length.sum())
 old_full_tokens=sum(int(i.length.sum()) for c,b,i in selected_inputs)
 assert old_full_tokens+removed_tokens==plan['data']['input_tokens'] and sum(len(i) for c,b,i in selected_inputs)+removed_prompts==plan['data']['prompts']
 c4meta=read(BASE/'systemness-long-augmentation/data-complete.json');c4cache=pathlib.Path(c4meta['cache_path'])
 for n in ['inputs.parquet','rows.parquet']:bind(c4cache/n,c4meta['cache_sha256'][n])
 c4rows=projected(c4cache/'rows.parquet',['cache_ix','sample_ix','prompt_ix','split','profile','role','group_id'],[('split','=','validation')])
 c4inputs=projected(c4cache/'inputs.parquet',['prompt_ix','split','profile','wrapper','length','bos'],[('split','=','validation'),('wrapper','=','native')])
 assert len(c4rows)==61440 and len(c4inputs)==480 and set(c4inputs.bos)=={False,True}
 tables+=group_stats(c4rows,c4inputs,'new-C4','existing-new-C4-validation');selected_inputs.append(('existing-new-C4-validation','new-C4',c4inputs))
 natmeta=read(BASE/'systemness-natural-transcripts/data-complete.json');natcache=pathlib.Path(natmeta['cache_path'])
 for n in ['inputs.parquet','rows.parquet']:bind(natcache/n,natmeta['cache_sha256'][n])
 natrows=projected(natcache/'rows.parquet',['cache_ix','sample_ix','prompt_ix','split','profile','role','eligible_fit','eligible_clean_validation','source_uid'])
 natinputs=projected(natcache/'inputs.parquet',['prompt_ix','split','profile','wrapper','length','eligible_fit','eligible_clean_validation','source_uid'])
 assert len(natinputs)==320 and len(natrows)==80436
 for cohort,mask in [('clean-natural-fit',natinputs.split.eq('fit')&natinputs.eligible_fit.eq(True)),('clean-natural-validation',natinputs.split.eq('validation')&natinputs.eligible_clean_validation.eq(True))]:
  i=natinputs.loc[mask];r=natrows[natrows.prompt_ix.isin(i.prompt_ix)];tables+=group_stats(r,i,'clean-natural',cohort);selected_inputs.append((cohort,'clean-natural',i))
 excluded=natinputs[natinputs.split.eq('fit')&~natinputs.eligible_fit.eq(True)]
 assert len(excluded)==1 and int(excluded.length.sum())==345
 total_prompts=sum(len(i) for c,b,i in selected_inputs);total_tokens=sum(int(i.length.sum()) for c,b,i in selected_inputs)
 total_rows=len(fit)+len(valid)+len(c4rows)+80116
 assert total_prompts==11174 and total_rows==892678
 cohort={}
 for c,b,i in selected_inputs:
  q=cohort.setdefault(c,dict(prompts=0,full_nonpadding_input_tokens=0));q['prompts']+=len(i);q['full_nonpadding_input_tokens']+=int(i.length.sum())
 n,m=total_tokens,total_prompts
 offsets=8*(m+1);one_packed_mask=math.ceil(n/8);two_packed_masks=2*one_packed_mask
 descriptors=16*m # optional fixed-size existing-source/prompt mapping, not model inputs
 storage=dict(tokens=n,prompts=m,p_f32_five_classes_bytes=20*n,prompt_offsets_u64_bytes=offsets,
  one_bitpacked_loss_or_score_mask_bytes=one_packed_mask,two_bitpacked_masks_bytes=two_packed_masks,
  minimal_p_plus_offsets_one_mask_bytes=20*n+offsets+one_packed_mask,
  practical_p_offsets_body_role_int8_flags_uint8_prompt_descriptor_bytes=22*n+offsets+descriptors,
  hypothetical_dense_raw512_f16_bytes=1024*n,
  practical_note='One mask plus existing frozen row labels/maps is the lower bound. Portable dense scoring can add int8 body-role sidecar(-1 special) and uint8 loss/score flags; these are loss/evaluation-only, never readout inputs. Original source/prompt mapping is optionally16bytes per prompt.')
 actual_free={k:dict(free_bytes=shutil.disk_usage(path).free,total_bytes=shutil.disk_usage(path).total) for k,path in [('workspace',ROOT),('dev_shm',pathlib.Path('/dev/shm'))]}
 budgets=dict(workspace_719_MiB=719*2**20,dev_shm_2_7_GiB=int(2.7*2**30))
 practical=storage['practical_p_offsets_body_role_int8_flags_uint8_prompt_descriptor_bytes'];staging=2*practical+512*2**20
 feasibility=dict(declared_budgets_bytes=budgets,actual_filesystem_usage=actual_free,
  practical_cache_bytes=practical,atomic_double_copy_plus_512MiB_reserve_bytes=staging,
  fits_declared_disk_with_atomic_double_copy_and_reserve=staging<budgets['workspace_719_MiB'],
  fits_declared_shm_with_atomic_double_copy_and_reserve=staging<budgets['dev_shm_2_7_GiB'],
  fits_current_workspace_with_atomic_double_copy_and_reserve=staging<actual_free['workspace']['free_bytes'],
  fits_current_shm_with_atomic_double_copy_and_reserve=staging<actual_free['dev_shm']['free_bytes'],
  dense_raw512_exceeds_declared_shm=storage['hypothetical_dense_raw512_f16_bytes']>budgets['dev_shm_2_7_GiB'],
  limitation='Capacity only, not performance/runtime or kernel correctness. Model/runtime/current-batch activations add RAM/GPU use; no additional LM/GPU was loaded. Producer should recheck free bytes at actual start, stream files, avoid extra full copies, and retain old banks.')
 # Metadata/code evidence that cached lengths include complete serialization, not sampled row count.
 for name in ['systemness_head_extract.py','systemness_data.py','systemness_long_data.py','systemness_long_augmentation_data.py','systemness_natural_transcripts.py','systemness_natural_dev_extract.py']:bind(HERE/name)
 result=dict(stage='metadata_only_causal_context_dense_probability_feasibility',audit_passed=True,
  authorization='用户持续授权继续实验。本次子任务仅CPU/元数据估算；没有实施/执行新路线，不代表缺少用户实验授权。',
  groups=tables,cohort_totals=cohort,total=dict(prompts=m,full_nonpadding_input_tokens=n,current_supervised_or_scored_rows=total_rows,full_tokens_per_current_row=n/total_rows),
  original_oldbank_crosscheck=dict(original755282_feature_rows_full_input_tokens=plan['data']['input_tokens'],original_prompts=plan['data']['prompts'],
   densefit_removed_sparse_long_single_prompts=removed_prompts,densefit_removed_prompt_full_tokens=removed_tokens,densefit_removed_sampled_rows=4160),
  new_C4_BOS_counts={str(k):int(v) for k,v in c4inputs.bos.value_counts().items()},quarantine=dict(excluded_natural_fit_prompts=1,excluded_natural_fit_full_input_tokens=345,excluded_natural_fit_content_rows=320),
  sequence_reuse=dict(full_sequences_include_headers_and_BOS_if_present=True,
   source_evidence='Prepared inputs.parquet length is computed from the complete serialized native prompt with add_special_tokens=False. Old extractor re-encodes entry.prompt and requires len(ids)==entry.length, then separately gathers sparse content positions. BOS is explicit in strings where selected. No whole token-ids JSON or prompt strings were decoded for this estimate.',
   relevant_code={'systemness_data.py':'serialize/add builds complete native text+encoding; stores len(ids).','systemness_head_extract.py':'Bundle tokenizes full cached prompt; sampled positions only affect targets.','systemness_long_augmentation_data.py':'prepared full1024 body serializations; input length includes native tags.','systemness_natural_dev_extract.py':'load_data filters selected319 prompt IDs before decoding; re-tokenizes complete prompt.'},
   future_safe_reuse='Filter existing prepared Parquet by exact frozen source/prompt IDs, native wrapper and fit/validation before reading prompt column; re-tokenize locally and enforce cached length/token mappings. Do not wholesale decode main token-ids.json if it also contains old-test entries. No raw C4/public source or new assistant generation is required.',
   logical_prompt_limit='Counts are existing(source_cache,prompt_ix) logical forward instances, not independent documents/stimuli. Cross-profile identical serialized inputs may exist; no new body hash/dedup/source-set version was generated, so sizing retains those views as a conservative upper bound.'),
  storage=storage,feasibility=feasibility,provenance_sha256=BINDINGS,
  input_columns_exclude_prompt_and_token_text=True,prepared_token_arrays_decoded=False,current_natural_candidate_probabilities_or_results_read=False,
  old_or_new_test_data_or_results_read=False,LM_loaded=False,gpu_used=False,new_source_or_body_set_generated=False,production_outputs_modified=False)
 REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps(dict(report=str(REPORT),report_sha256=sha(REPORT),script_sha256=sha(__file__),total=result['total'],cohorts=cohort,storage=storage,feasibility=feasibility),ensure_ascii=False))
if __name__=='__main__':main()
