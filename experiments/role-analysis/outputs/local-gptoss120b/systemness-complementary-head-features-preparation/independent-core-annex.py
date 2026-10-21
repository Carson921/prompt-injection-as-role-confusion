"""Synthetic AST-only annex. Never imports the extractor or reads real data."""
import ast,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
SOURCE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/systemness_complementary_head_extract.py')
EXPECTED='964335b1dbdbea19219cc4e3db15409195376cd92dfd7125f7d4370758a34509'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def require(v,s):
    if not bool(v):raise RuntimeError(s)
root=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-extract-source-'));start=digest(SOURCE);require(start==EXPECTED,'Candidate changed.');text=SOURCE.read_text()
prior=Path('/tmp/confirmation-review-complementary-extract-10102382.py').read_text()
reverse=text.replace("AUDIT_PATH=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/local-gptoss120b/systemness-complementary-head-scan/actual-audit/actual-scan-audit.json')",'AUDIT_PATH=None').replace("AUDIT_SHA='a71c052c38947b9c2a4f1ab73f58a3414533b43e013bac624322b0b836e67ff0'",'AUDIT_SHA=None')
patched="""    # Keep preflight synthetic after formal pins are filled: patch only this
    # module's two pins, then restore them without touching frozen modules.
    saved_pins={name:globals()[name] for name in ['AUDIT_PATH','AUDIT_SHA']}
    try:
        globals().update(AUDIT_PATH=None,AUDIT_SHA=None)
        reject('unknown_actual_audit_pins_block_before_any_source_or_headp_read',gate)
    finally:globals().update(saved_pins)"""
reverse=reverse.replace(patched,"    reject('unknown_actual_audit_pins_block_before_any_source_or_headp_read',gate)")
require(reverse==prior and hashlib.sha256(prior.encode()).hexdigest()=='10102382e9c7a9f908fb0a0729014a4c92ff6cf4be4c3178f2976168ec4ef3c5','Unexpected changes beyond exact pins + own preflight isolation.')
tree=ast.parse(text);nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};checks=[]
def check(name,v=True):require(v,name);checks.append(name)
def reject(name,fn):
    try:fn()
    except RuntimeError:checks.append(name);return
    raise RuntimeError('No rejection: '+name)
def forbidden(*a,**k):raise LookupError('Real artifact/body/model access forbidden.')
class ParentBundle:
    def targets(self,part):return np.concatenate([self.lookup[r.uid].cache_ix.to_numpy(np.int64) for r in part.itertuples()])
def batch_digest(a,targets):
    v=np.ascontiguousarray(a[targets]);require(np.isfinite(v).all(),'Unwritten128-column batch.');return sha(v)
def prefix(b,n):
    require(0<=n<=len(b.parts),'Out of range.');targets=np.concatenate([b.targets(p) for p in b.parts[:n]]) if n else np.empty(0,np.int64)
    return dict(next_batch=n,rows_written=len(targets),cache_ix_sha256=sha(targets),sample_ix_sha256=sha(b.rows.iloc[targets].sample_ix.to_numpy()))
ns=dict(np=np,pd=pd,Path=Path,require=require,COUNTS={},BATCH=4,CACHE=root/'unused',array_sha=sha,frame_sha=lambda f,c:hashlib.sha256(f[c].to_csv(index=False).encode()).hexdigest(),legacy=SimpleNamespace(Bundle=ParentBundle,prefix_state=prefix,batch_digest=batch_digest),
    SCAN_COMPLETE_SHA='complete',SCAN_SELECTION_SHA='selection',AUDIT_PATH=None,AUDIT_SHA=None,data=SimpleNamespace(digest=forbidden),read=forbidden)
for name in ['gate','project_rows','Bundle','parity_scope','restore']:
    exec(compile(ast.fix_missing_locations(ast.Module(body=[nodes[name]],type_ignores=[])),str(SOURCE),'exec'),ns)
reject('unknown_audit_gate_before_any_SHA_or_metadata_IO',ns['gate'])
old=pd.DataFrame(dict(cache_ix=np.arange(256),sample_ix=2_000_000_000+np.arange(256),wrapper='native',split='fit',profile='long-full',source_set='long-full',prompt_ix=7,token_ix=np.arange(256),token_id=30+np.arange(256),role='system'))
c4=pd.DataFrame(dict(cache_ix=[327680,327681],sample_ix=[6_000_000_327680,6_000_000_327681],wrapper='native',split='validation',profile='long-full',source_set='long-augmentation',prompt_ix=1280,token_ix=[1,3],token_id=[20,22],role='user'))
nat=pd.DataFrame(dict(cache_ix=[0,1,2],sample_ix=4_000_000_000_000+np.arange(3),wrapper='native',split='fit',profile='natural-public-sua',source_set='natural-dev-clean',prompt_ix=0,original_prompt_ix=11,token_ix=[0,1,2],token_id=[31,32,33],role=['system','user','assistant']))
project=ns['project_rows'];kw=dict(expected={'old':256,'c4':2,'natural':3});rows=project(old,c4,nat,**kw)
check('c4_parent_indices_not_rebased_after_validation_filter',rows.iloc[256:258].parent_cache_ix.tolist()==[327680,327681] and rows.cache_ix.tolist()==list(range(261)))
check('natural_original_prompt_and_original_sample_identity',rows.iloc[-1].source_prompt_ix==11 and rows.iloc[-1].uid=='natural:11' and rows.iloc[-1].sample_ix==nat.sample_ix.iloc[-1])
bad=c4.copy();bad['split']='fit';reject('C4_fit_parent_rejected',lambda:project(old,bad,nat,**kw))
bad=nat.copy();bad.loc[0,'sample_ix']=old.sample_ix.iloc[0];reject('cross_parent_sample_alias_rejected',lambda:project(old,c4,bad,**kw))
i=rows[['uid','source_prompt_ix','input_source','split']].drop_duplicates().copy();i['length']=[500,10,20];bundle=ns['Bundle'](i,rows)
check('deterministic256_original_longfull_fit_before_length_sort',bundle.parity_targets.tolist()==list(range(256)) and bundle.inputs.uid.iloc[0]=='old:long-full:7' and bundle.descriptor()['parity_prompts']==['old:long-full:7'])
proof=dict(fit_only=True,validation_probabilities_computed=False,rows=256,target_cache_ix_sha256=sha(bundle.parity_targets),sample_ix_sha256=sha(bundle.rows.iloc[bundle.parity_targets].sample_ix.to_numpy()))
ns['parity_scope'](bundle,proof);reject('validation_probability_parity_proof_rejected',lambda:ns['parity_scope'](bundle,proof|{'validation_probabilities_computed':True}))
a=np.zeros((261,128),np.float16);targets=bundle.targets(bundle.parts[0]);h=batch_digest(a,targets);progress=dict(plan_sha256='plan',**prefix(bundle,1),batch_features_sha256=[h],batch_f16_roundtrip_max=[0.],portable_consumer=proof)
cache=root/'progress';cache.mkdir();(cache/'progress.json').touch();ns['read']=lambda p:progress
check('exact_complete128column_batch_restore',ns['restore'](bundle,a,'plan',cache=cache)[0]==[h])
a[256,127]=1;reject('finite_second_extra_head_byte_change_rejected',lambda:ns['restore'](bundle,a,'plan',cache=cache));a[256,127]=np.nan
reject('NaN_in_one_extra_column_rejected',lambda:ns['restore'](bundle,a,'plan',cache=cache))
model=ast.get_source_segment(text,nodes['model_context']);mat=ast.get_source_segment(text,nodes['materialize']);initial=ast.get_source_segment(text,nodes['initialize']);complete=ast.get_source_segment(text,nodes['verified_extraction'])
check('own_model_context_native_frozen4A40_and_old_globals_unassigned',"Mxfp4Config(dequantize=False)" in model and '.eval().requires_grad_(False)' in model and 'model_context' in model and 'scanner.OUT=' not in text and 'legacy.OUT=' not in text)
check('Arrow_ID_native_split_filters_precede_prompt_materialization',mat.index("ds.field('prompt_ix')")<mat.index('.to_table(') and "ds.field('wrapper')=='native'" in mat and "['validation'] if name=='c4'" in mat and "ds.field('eligible_fit')==True" in mat)
check('NaN128_atomic_initialization_and_complete_byte_prefix_proof','shape=(len(bundle.rows),128)' in initial and '=np.nan' in initial and 'rows.parquet.initializing' in initial and 'legacy.check_feature_file' in initial and "restore(bundle,array" in complete and "data.digest(CACHE/'features.f16')" in complete)
check('source_unchanged',digest(SOURCE)==start)
report=dict(stage='independent_synthetic_complementary_extract_source_audit',audit_passed=True,extractor_source_sha256=start,checks=checks,check_count=len(checks),minimal_diff_pins_and_own_fixture_only=True,prior_extractor_source_sha256='10102382e9c7a9f908fb0a0729014a4c92ff6cf4be4c3178f2976168ec4ef3c5',production_module_imported=False,real_layout_or_features_read=False,real_probabilities_read=False,real_weights_or_text_read=False,gpu_used=False,formal_files_written=False)
p=root/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),script_sha256=digest(__file__),source_sha256=start,checks=len(checks))))
