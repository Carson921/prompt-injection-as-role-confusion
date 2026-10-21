"""Independent v3 bank-auditor source/tiny CPU annex; no actual artifacts."""
import ast,copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
SOURCE=Path('/tmp/confirmation_review_complementary_feature_results_audit.py')
EXPECTED='f5741cdf8ea9e11f0f3058ef30b1f3ed63d97d9af1076ab8b8fef05970929c13'
ROOT=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
spec=importlib.util.spec_from_file_location('review_bank_auditor',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(n,v=True):assert bool(v),n;checks.append(n)
def reject(n,f):
    try:f()
    except (RuntimeError,AssertionError,ValueError):checks.append(n);return
    raise AssertionError(n)
def forbidden(*a,**k):raise LookupError('Actual artifacts forbidden')
# Independent source-interface tolerance check, never call producer function.
legacy=ast.parse((ROOT/'systemness_head_extract.py').read_text());legacy_tol=next(ast.literal_eval(n.value) for n in legacy.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARITY_TOLERANCE' for t in n.targets))
check('nested5e5_matches_actual_legacy_constant_not_outer5e4',m.CONSUMER_TOL==legacy_tol==5e-5 and m.REFERENCE_TOL==5e-4)
with patch.object(m,'processes',lambda:[678]),patch.object(m,'digest',forbidden),patch.object(m,'read',forbidden),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden):
    reject('live_process_rejects_before_any_artifact_IO',lambda:m.actual('/tmp/never-written-actual.json'))
# Dynamic fake schema verifies ALL safe fields rather than repeating allowlist.
cols=['cache_ix','token','token_id','token_ix','body','body_offset','body_sha256','prompt','native_id','template_id','cohort','new_document_ix','source_rank','future_metadata','assistant_text','raw_text']
calls=[]
import pyarrow.parquet as pq
with patch.object(pq,'ParquetFile',lambda p:SimpleNamespace(schema=SimpleNamespace(names=cols))),patch.object(pd,'read_parquet',lambda p,**kw:calls.append(kw) or pd.DataFrame()):m.metadata('/tmp/no-such-actual-parquet',filters=[('split','=','validation')])
want=[x for x in cols if x not in {'token','body','prompt','assistant_text','raw_text'}]
check('dynamic_schema_preserves_all_nonpayload_future_fields',calls==[{'columns':want,'filters':[('split','=','validation')]}])
# Tiny independent parent projection with C4 parent indices nonzero.
old=pd.DataFrame(dict(cache_ix=np.arange(256),sample_ix=10000+np.arange(256),wrapper='native',split='fit',profile='long-full',source_set='long-full',prompt_ix=3,token_ix=np.arange(256)+4,token_id=np.arange(256),role='system',cohort='original'))
c4=pd.DataFrame(dict(cache_ix=[327680,389119],sample_ix=[6000000327680,6000000389119],wrapper='native',split='validation',profile='long-full',source_set='long-augmentation',prompt_ix=[1280,1759],token_ix=[4,1027],token_id=[7,8],role='user',cohort='new-long',new_document_ix=[256,351],source_rank=[140,33]))
natural=pd.DataFrame(dict(cache_ix=[0,1,2],sample_ix=[4000000000000,4000000000001,4000000000002],wrapper='native',split='fit',profile='natural-public-sua',source_set='natural-dev-clean',prompt_ix=[0,0,0],original_prompt_ix=[17,17,17],token_ix=[4,16,22],token_id=[9,10,11],role=['system','user','assistant'],native_id='native42',template_id='temp42',future_metadata='checked'))
rows=m.project(dict(old=old,c4=c4,natural=natural),dict(old=256,c4=2,natural=3))
check('rebuild_keeps_union_safe_metadata_and_exact_parent_indices',len(rows)==261 and rows.parent_cache_ix.iloc[256:258].tolist()==[327680,389119] and rows.uid.iloc[-1]=='natural:17' and set(['cohort','new_document_ix','source_rank','native_id','template_id','future_metadata'])<=set(rows))
m.same_metadata(rows.copy(),rows);bad=rows.copy();bad.loc[260,'future_metadata']='tampered'
reject('nonmapping_harmless_metadata_tamper_rejected',lambda:m.same_metadata(bad,rows))
reject('dropped_nonpayload_column_rejected',lambda:m.same_metadata(rows.drop(columns='cohort'),rows))
bad=c4.copy();bad.loc[0,'split']='fit';reject('newC4_fit_never_enters_projected_bank',lambda:m.project(dict(old=old,c4=bad,natural=natural),dict(old=256,c4=2,natural=3)))
inputs=rows[['uid','source_prompt_ix','input_source','split']].drop_duplicates().copy();inputs['length']=[1100,1101,1102,40];layout=m.Layout(rows,inputs)
check('FIT256_priority_then_exact_complete4prompt_batch',np.array_equal(layout.parity,np.arange(256)) and layout.inputs.uid.iloc[0]=='old:long-full:3' and len(layout.parts)==1)
features=np.arange(261*128,dtype=np.float32).reshape(261,128).astype(np.float16);hashes=[m.batch_sha(features,layout.targets(p)) for p in layout.parts]
progress=dict(plan_sha256='tiny',**m.prefix(layout,len(layout.parts)),batch_features_sha256=hashes,batch_f16_roundtrip_max=[0.])
check('all128columns_complete_prefix_bytehash_verified',m.verify_progress(layout,features,progress,'tiny')['rows_written']==261)
changed=features.copy();changed[-1,-1]=np.nextafter(changed[-1,-1],np.float16(np.inf));reject('last_row_last_head_column_singlebit_change_rejected',lambda:m.verify_progress(layout,changed,progress,'tiny'))
changed=features.copy();changed[-1,-1]=np.nan;reject('unwritten_final_column_rejected',lambda:m.verify_progress(layout,changed,progress,'tiny'))
bad=copy.deepcopy(progress);bad['sample_ix_sha256']='0'*64;reject('prefix_sample_order_tamper_rejected',lambda:m.verify_progress(layout,features,bad,'tiny'))
bad=copy.deepcopy(progress);bad['batch_features_sha256']=[];reject('partial_prefix_cannot_be_completed',lambda:m.verify_progress(layout,features,bad,'tiny'))
# Saved F16 evidence tolerances independently constructed from legacy schema.
saved=np.zeros((256,128),np.float16)
evidence=dict(rows=256,same_saved_f16_vectors=True,classifier_device='cuda:0',saved_vectors_sha256=m.arrsha(saved),tolerance=5e-5,max_difference=4e-5)
m.consumer_evidence(evidence,saved);check('valid_nested5e5_evidence_accepted')
bad=dict(evidence,tolerance=5e-4);reject('nested_cannot_silently_use_outer5e4_tolerance',lambda:m.consumer_evidence(bad,saved))
bad=dict(evidence,max_difference=5.01e-5);reject('nested_error_above_legacy_bound_rejected',lambda:m.consumer_evidence(bad,saved))
zero=np.zeros((261,128),np.float16);oldvectors=np.zeros((256,512),np.float16);sample=dict(extra_features=zero[:256].copy(),transient_original8=oldvectors.copy(),target_cache_ix=layout.parity.copy(),sample_ix=rows.iloc[layout.parity].sample_ix.to_numpy())
def ev(a):return dict(rows=256,same_saved_f16_vectors=True,classifier_device='cuda:0',saved_vectors_sha256=m.arrsha(a),tolerance=5e-5,max_difference=0.)
proof=dict(rows=256,fit_only=True,validation_probabilities_computed=False,original8_transient_only=True,no_new_readout=True,tolerance=5e-4,target_cache_ix_sha256=m.arrsha(layout.parity),sample_ix_sha256=m.arrsha(rows.iloc[layout.parity].sample_ix.to_numpy()),original8_reference_probability_max_difference=0.,original8_reference_vectors_bitexact=True,original8_savedF16_CPU_GPU=ev(oldvectors),extra2_savedF16_CPU_GPU=ev(sample['extra_features']))
w=np.zeros((8,5,64));b=np.zeros((8,5));ew=np.zeros((2,5,64));eb=np.zeros((2,5));r=m.parity(layout,zero,proof,sample,oldvectors,w,b,ew,eb)
check('FIT256_savedF16_CPU_replay_distinguished_from_GPU_attestation',r['rows']==256 and not r['GPU_independently_replayed'])
bad=copy.deepcopy(proof);bad['tolerance']=5e-5;reject('outer_reference_scope_does_not_inherit_nested_tolerance',lambda:m.parity(layout,zero,bad,sample,oldvectors,w,b,ew,eb))
bad=copy.deepcopy(proof);bad['validation_probabilities_computed']=True;reject('parity_cannot_expand_to_validation',lambda:m.parity(layout,zero,bad,sample,oldvectors,w,b,ew,eb))
# Source closure checks only, no formal metadata load.
tree=ast.parse(SOURCE.read_text());actual=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='actual');code=ast.unparse(actual)
check('actual_terminal_first_and_final_with_full_source_feature_hash_closure',ast.unparse(actual.body[0])=='terminal_gate()' and code.count('terminal_gate()')==2 and "bind(SOURCE, SOURCE_SHA)" in code and "bind(fp, done['features_sha256'])" in code and "digest(p) == want" in code)
check('explicit_fixed_parent_ranges_and_complete896838x128',"np.arange(327680, 389120)" in code and m.ROWS==755282+61440+80116==896838 and m.COUNTS==dict(old=755282,c4=61440,natural=80116))
check('native_gpu_hook_parity_attestation_boundary_honest',all(s in code for s in ['native_LM_independently_replayed=False','producer_GPU_parity_independently_replayed=False','hidden_states_bitexact','writes_disabled','Mxfp4GptOssExperts']))
check('source_unchanged',sha(SOURCE)==EXPECTED)
folder=Path(tempfile.mkdtemp(prefix='feature-review-complementary-feature-auditor-',dir='/tmp'));report=dict(stage='independent_source_tiny_CPU_extra2_bank_auditor_review',passed=True,mustfix=[],source_path=str(SOURCE),source_sha256_before=EXPECTED,source_sha256_after=sha(SOURCE),checks=checks,check_count=len(checks),superseded_versions_never_actual=True,actual_artifacts_read=False,actual_layout_or_bank_read=False,real_probabilities_or_weights_read=False,production_modules_imported=False,source_text_materialized=False,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False)
p=folder/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=sha(p),script_sha256=sha(__file__),source_sha256=sha(SOURCE),checks=len(checks))))
