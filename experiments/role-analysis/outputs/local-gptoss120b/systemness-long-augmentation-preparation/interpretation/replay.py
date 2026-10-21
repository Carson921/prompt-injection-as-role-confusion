#!/usr/bin/env python3
"""Read completed development metadata/tables only; never load probabilities or model arrays."""
import hashlib,json,pathlib,math
import numpy as np
import pandas as pd
ROOT=pathlib.Path('/root/prompt-injection-as-role-confusion')
BASE=ROOT/'experiments/role-analysis/outputs/local-gptoss120b'
OUT=BASE/'systemness-long-augmentation-probes'
NAT=BASE/'systemness-natural-transcripts'
REPORT=pathlib.Path('/tmp/feature_review_long_augmentation_interpretation.json')
PROVENANCE={}
def sha(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=pathlib.Path(p).resolve();d=sha(p);PROVENANCE[str(p)]=d;return d
def read(p):
 bind(p);return json.loads(pathlib.Path(p).read_text())
def projected(p,columns):
 bind(p);return pd.read_parquet(p,columns=columns)
def counts(s):
 return {str(k):int(v) for k,v in s.value_counts().sort_index().items()}
def native_count(rows,inputs,mask):
 r=rows.loc[mask];i=inputs.loc[inputs.prompt_ix.isin(r.prompt_ix)]
 assert r.source_uid.nunique()==i.prompt_ix.nunique()
 return dict(content_rows=len(r),prompts=int(r.prompt_ix.nunique()),role_counts=counts(r.role),
  system_families=int(r.family_id.nunique()),dependency_components=int(r.dependency_component_id.nunique()),
  lm_input_tokens=int(i.length.sum()),maximum_input_tokens=int(i.length.max()) if len(i) else None,
  eight_head_f16_bytes=len(r)*512*2)

def main():
 bind(__file__)
 complete=read(OUT/'complete.json')
 for name,want in complete['artifact_sha256'].items():assert bind(OUT/name)==want
 records=read(OUT/'records.json');by={r['record']:r for r in records};assert len(by)==6
 csv_path=OUT/'validation-metrics.csv';bind(csv_path);csv=pd.read_csv(csv_path)
 assert len(csv)==54
 metrics=['system_precision','system_recall','system_f1','system_balanced_accuracy','system_false_positive_rate','accuracy','system_auroc','system_average_precision']
 for r in records:
  for g,m in r['validation']['groups'].items():
   c=csv.loc[csv.record.eq(r['record']) & csv.group.eq(g)];assert len(c)==1
   for k in metrics:assert abs(float(c.iloc[0][k])-m[k])<2e-14
 selection=read(OUT/'selection.json');assert selection['promoted'] is False and selection['primary_changed'] is False and selection['no_fallback'] is True
 source_counts=read(OUT/'source-counts.json')
 training={k:read(OUT/'training'/f'{k}.json') for k in ['new','matched']}
 old_rows=projected('/dev/shm/role-analysis-top8-heads/rows.parquet',['cache_ix','sample_ix','document_id','group_id','split','profile','role'])
 assert len(old_rows)==755282
 raw=old_rows.loc[old_rows.split.eq('fit')];full_ids=set(raw.loc[raw.profile.eq('long-full'),'document_id']);assert len(full_ids)==132
 fit=raw.loc[~(raw.profile.eq('long-single') & raw.document_id.isin(full_ids))]
 fit_profile=counts(fit.profile);assert len(fit)==578090 and fit_profile.get('natural',0)==0
 assert len(raw)-len(fit)==4160
 data=read(NAT/'data-complete.json');cache=pathlib.Path(data['cache_path'])
 for name,want in data['cache_sha256'].items():assert bind(cache/name)==want
 for name,want in data['metadata_sha256'].items():assert bind(NAT/name)==want
 assert bind(NAT/'protocol.json')==data['protocol_sha256']
 rows=projected(cache/'rows.parquet',['cache_ix','sample_ix','prompt_ix','source_uid','split','role','eligible_fit','eligible_clean_validation','family_id','dependency_component_id'])
 inputs=projected(cache/'inputs.parquet',['prompt_ix','length','split','eligible_fit','eligible_clean_validation','source_uid','family_id','dependency_component_id'])
 assert len(rows)==80436 and len(inputs)==320
 assert np.array_equal(rows.cache_ix,np.arange(len(rows))) and rows.sample_ix.is_unique
 assert set(rows.role)=={'system','user','assistant'} and set(rows.split)=={'fit','validation'}
 fmask=rows.eligible_fit.eq(True);vmask=rows.eligible_clean_validation.eq(True)
 assert rows.loc[fmask].split.eq('fit').all() and rows.loc[vmask].split.eq('validation').all()
 selected=rows.loc[fmask|vmask];assert len(selected)==80116 and selected.prompt_ix.nunique()==319
 assert not set(rows.loc[fmask].source_uid)&set(rows.loc[vmask].source_uid)
 natural=dict(all_prepared=native_count(rows,inputs,pd.Series(True,index=rows.index)),
  eligible_fit=native_count(rows,inputs,fmask),clean_validation=native_count(rows,inputs,vmask),
  excluded_fit=native_count(rows,inputs,rows.split.eq('fit') & ~fmask),
  selected_content_rows=len(selected),selected_prompts=319,selected_lm_input_tokens=int(inputs.loc[inputs.prompt_ix.isin(selected.prompt_ix),'length'].sum()),
  proposed_raw_f16_feature_bytes=len(selected)*512*2,
  absent_roles={'cot':{'support':0,'recall':None},'tool':{'support':0,'recall':None}},
  columns_read_only_metadata=True,text_token_strings_not_read=True)
 summaries={k:dict(threshold=r['validation']['system_threshold'],min24=r['validation']['min24'],old5_mean_f1=r['validation']['old5_mean_f1'],model_sha256=r['model_sha256']) for k,r in by.items()}
 diffs={}
 for control in ['old-head-old-mean','old-head-maximin','matched-maximin']:
  a=by['new-maximin']['validation'];b=by[control]['validation']
  diffs[control]=dict(min24_pp=100*(a['min24']-b['min24']),old5_mean_f1_pp=100*(a['old5_mean_f1']-b['old5_mean_f1']),groups={g:{k:100*(a['groups'][g][k]-b['groups'][g][k]) for k in metrics} for g in a['groups']})
 exact_natural={k:dict(tokens=by[k]['validation']['groups']['old-natural']['tokens'],system_confusion=by[k]['validation']['groups']['old-natural']['system_confusion'],system_precision=by[k]['validation']['groups']['old-natural']['system_precision'],system_recall=by[k]['validation']['groups']['old-natural']['system_recall']) for k in by}
 budget={k:{f:v[f] for f in ['initial_weight_sha256','best_epoch','best_steps','epochs','actual_steps','actual_instances','instances_per_epoch','steps_per_epoch','last_batch_instances']} for k,v in training.items()}
 mix_total=578090+natural['eligible_fit']['content_rows']
 plan=dict(status='proposal_only_not_extraction_or_training_authorization',
  objective='Add genuinely prepared natural SUA training support, which current densefit entirely lacks; isolate this change from long-source upweighting.',
  source_selection='Use exactly cached eligible_fit=True 255 conversations and eligible_clean_validation=True 64. Exclude the complete one contaminated fit conversation, preserve original sample_ix and source IDs; no replacement.',
  proposed_features=dict(shape=[80116,512],dtype='float16',bytes=80116*1024,lm_prompts=319,lm_input_tokens=natural['selected_lm_input_tokens'],
   selected_heads=[458,462,2045,460,459,2047,456,997],
   method='New separate metadata-only declaration/source signature before GPU; exactly four A40 native frozen MXFP4 forward, eight ranked pre-o_proj heads only, no legacy captures, no LM training.',
   integrity='Bind natural protocol/cache/ledger/duplicate hashes, tokenizer/config/weight manifests, head rank/old scaler and extractor source; local cache_ix plus original sample_ix; >=512MiB reserve; NaN/prefix/byte-hash restart, disabled-write bitexact and saved-F16 portable parity. Do not alter old banks.'),
  minimal_training_pair=dict(architecture='Existing pointwise MLP128, unchanged original fit-only 512-scaler; same initial seed, optimizer and early-stopping rule. Training protocol must be frozen separately before use.',
   treatment='578090 original densefit C4 rows plus each of the 63403 eligible natural fit tokens exactly once per epoch.',
   matched_control='Same per-epoch instances/updates and maximum budget; replace the 63403 natural instances by a predeclared deterministic sample from original C4 fit only. Both arms retain the original 578090 rows.',
   instances_per_epoch=mix_total,batch_size=4096,updates_per_epoch=math.ceil(mix_total/4096),last_batch_instances=mix_total%4096,
   natural_instance_fraction=natural['eligible_fit']['content_rows']/mix_total,
   roles='Retain all five C4 roles and normal SUA natural roles. Public natural CoT/tool have zero support/N/A; never relabel assistant replies as CoT/tool.',
   early_stopping='Freeze best epoch using the same original old-five validation helper for both arms; only after best checkpoints are fixed, evaluate/calibrate on the predeclared development groups.',
   validation='Keep all 64 clean natural conversations entirely out of optimizer and scaler. Treat their 16 families/dependency components as development, not an independent test. Reuse original old-five/old-tail and new-long96/full-tail as development; no test probabilities.',
   selection='Before any new training fix the global F32-consumer threshold objective and retention guards against unchanged old control and matched control; publish both negative and positive arms. No per-profile threshold or test-driven fallback.'),
  parallel_multiseed='Use the root separately frozen a1ad395c345ec8af91d5782987a95f0b9cd9034da92b7f40debc2dce1ec521e6 design: three new seeds 20261022/23/24 plus original 20261021, fixed equal four-model probability mean, original 578090 fit/scaler, full old CPU/GPU-saved drift controls. This proposal does not change that design.',
  confirmation='No currently seen C4/natural test may select a model or threshold. Fresh natural128 sources are not yet selected/materialized in this round; any new independent confirmation requires another frozen protocol.',
  limitations=['Public assistant replies are OASST human or recorded ToxicChat/Vicuna replies, not GPT-OSS-120B generation under newly assigned system instructions.','255 fit conversations have 32 system wording families; 64 validation conversations have 16 families/components, not 64 independent system stimuli.','SUA augmentation may improve natural system false positives; it supplies no genuine CoT/tool training supervision and does not guarantee 90% on fresh long documents.'])
 result=dict(stage='completed_long_augmentation_development_interpretation_metadata_only',
  gpu_used=False,raw_candidate_probabilities_read=False,test_data_or_results_read=False,
  source_counts=source_counts,densefit_profile_rows=fit_profile,densefit_natural_rows=0,
  long_full_fraction_before=fit_profile['long-full']/len(fit),long_full_fraction_after=(fit_profile['long-full']+327680)/905770,
  old_multi_fraction_before=fit_profile['multi']/len(fit),old_multi_fraction_after=fit_profile['multi']/905770,
  record_summaries=summaries,record_groups={k:{g:dict({m:v[m] for m in metrics},tokens=v['tokens'],role_recall=v['recall'],role_support=[sum(row) for row in v['confusion']],system_confusion=v['system_confusion']) for g,v in r['validation']['groups'].items()} for k,r in by.items()},new_maximin_minus_controls_pp=diffs,old_natural_system_counts=exact_natural,
  training_budgets=budget,natural_development_cache=natural,
  decision=dict(promoted=False,no_fallback=True,primary_changed=False,reasons=selection['promotion_rejection_reasons']),
  interpretation=['New-long-tail development ranking and five-way accuracy improve, while natural system precision and old long-tail system F1 regress; the result is a tradeoff, not a promoted model.',
   'Old-natural system TP=662/FN=0 for original and new-maximin; false positives rise from 26 to114. The natural failure is precision, despite near-perfect AUROC/BAcc.',
   'Long-full training instance share rises from 29.23% to54.83% in both augmentation arms, with zero natural fit examples in either arm. Source diversity and weighting cannot be separated causally by this result alone.',
   'The two arms match per-epoch instance count and maximum budget, but independent early stopping produces different best and actual total updates; do not claim actual-step matching.',
   'Matched-maximin diagnostic min24 exceeds90%, but it fails retention/no-fallback criteria and is not an eligible promoted replacement. Reused validation does not establish fresh-test generalization.'],
  prospective_natural_fitmix_plan=plan,
  provenance_sha256=PROVENANCE)
 REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps(dict(report=str(REPORT),sha256=sha(REPORT),auditor_source_sha256=sha(__file__),natural_selected_rows=len(selected),natural_selected_prompts=319,natural_features_bytes=80116*1024,new_maximin_promoted=False)))
if __name__=='__main__':main()
