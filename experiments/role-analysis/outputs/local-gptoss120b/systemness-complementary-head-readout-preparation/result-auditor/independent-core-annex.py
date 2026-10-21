"""Independent tiny/source review of unpinned640 result adapter. No actual IO."""
import ast,copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
P=Path('/tmp/confirmation_review_complementary_readout_results_audit.py');PIN='fbd87f808d0913821dfa4391464a658c1ca6069a4566d03b824b1080e27a19c0'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)==PIN
spec=importlib.util.spec_from_file_location('annex640audit',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(n,v=True):assert bool(v),n;checks.append(n)
def reject(n,f):
    try:f()
    except (RuntimeError,ValueError,AssertionError):checks.append(n);return
    raise AssertionError(n)
def forbidden(*a,**k):raise LookupError('Actual artifact access forbidden')
# Runtime spies prove prerequisite failures occur before all actual IO/library loads.
with patch.object(m,'libraries',forbidden),patch.object(m,'read',forbidden),patch.object(m,'digest',forbidden),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden),patch.object(np,'memmap',forbidden):
    reject('unknown_SOURCE_PROTOCOL_pins_refuse_before_actualIO',lambda:m.actual('/tmp/no-audit-written.json'))
with patch.object(m,'SOURCE_SHA','mock'),patch.object(m,'PROTOCOL_SHA','mock'),patch.object(m,'processes',lambda:[999]),patch.object(m,'libraries',forbidden),patch.object(m,'read',forbidden),patch.object(m,'digest',forbidden):
    reject('live_readout_or_extractor_refuses_before_metadata',lambda:m.actual('/tmp/no-audit-written.json'))
# Dup control must never read added data, even if added getter is an exception.
rng=np.random.RandomState(901);basebank=rng.normal(size=(7,512)).astype('f2');extra=rng.normal(size=(12,128)).astype('f2');lookup=np.array([6,4,2,0,1,3,5]);ix=np.array([6,0,4,1])
class NoAdded:
    def __getitem__(self,ix):raise AssertionError('Duplicate control touched actual-added bank')
dup=m.Joined(basebank,NoAdded(),lookup,True);new=m.Joined(basebank,extra,lookup)
check('duplicate640_never_reads_real_added128_and_repeats_rank01',np.array_equal(dup[ix][:,512:].view('u2'),basebank[ix,:128].view('u2')))
check('new640_joins_exact_lookup_and_retains_original512_bits',np.array_equal(new[ix][:,:512].view('u2'),basebank[ix].view('u2')) and np.array_equal(new[ix][:,512:].view('u2'),extra[lookup[ix]].view('u2')))
bad=lookup.copy();bad[0]=-1;reject('unmapped_C4fit_rejected_for_duplicate_too',lambda:m.Joined(basebank,NoAdded(),bad,True)[np.array([0])])
# Sparse parent adapter has real nonzero cache indices but only two fake rows.
old=pd.DataFrame(dict(sample_ix=[1,2],role=['system','user'],split='fit'));nat=pd.DataFrame(dict(sample_ix=[40,41],role=['system','assistant'],split='fit'))
cv=pd.DataFrame(dict(sample_ix=[60,61],role=['system','tool'],split='validation'),index=[327680,389119])
class Iloc:
    def __getitem__(self,i):return cv.loc[i]
class Parent:
    iloc=Iloc()
    def __len__(self):return 389120
pieces=[]
for name,f in [('old',old),('c4',cv),('natural',nat)]:
    for i,r in f.iterrows():pieces.append(dict(parent_bank=name,parent_cache_ix=i,sample_ix=r.sample_ix,role=r.role,split=r.split))
erows=pd.DataFrame(pieces);erows['cache_ix']=np.arange(len(erows));maps=m.mapping(erows,dict(old=old,c4=Parent(),natural=nat),expected=6)
check('independent_mapping_uses_parent327680_and389119_not_filtered_offsets',maps['c4'][327680]==2 and maps['c4'][389119]==3 and (maps['c4'][:327680]==-1).all())
bad=erows.copy();bad.loc[2,'sample_ix']=0;reject('wrong_parent_sample_id_rejected',lambda:m.mapping(bad,dict(old=old,c4=Parent(),natural=nat),expected=6))
# Fit128 statistic independent direct NumPy formula, poison all unused rows.
a=rng.normal(size=(17,128)).astype('f2');a[:,3]=2.;a[10:]=50000;look=np.array([9,3,1,7,5,0,2,4,6,8]);fit=np.array([0,1,2,3,4,5]);mm,ss=m.fit128(a,look,fit,chunk=2);expected=a[look[fit]].astype('f8')
check('F64_fit128_population_mean_std_floor_no_unused_rows',np.array_equal(mm,expected.mean(0).astype('f4')) and np.allclose(ss,np.maximum(expected.std(0),1e-6).astype('f4'),rtol=1e-6) and ss[3]==np.float32(1e-6))
# Independent torch module parity for full rawF16 functional consumer.
base={'encoder.0.weight':rng.normal(0,.02,(128,512)).astype('f4'),'encoder.0.bias':rng.normal(0,.01,128).astype('f4'),'head.weight':rng.normal(0,.03,(5,128)).astype('f4'),'head.bias':np.array([.2,.5,.1,-.1,-.4],'f4')}
res={'encoder.0.weight':rng.normal(0,.04,(32,640)).astype('f4'),'encoder.0.bias':rng.normal(0,.02,32).astype('f4'),'head.weight':rng.normal(0,.04,(1,32)).astype('f4'),'head.bias':np.array([.08],'f4')}
mean=rng.normal(0,.01,640).astype('f4');scale=rng.uniform(.9,1.1,640).astype('f4');frame=pd.DataFrame(dict(cache_ix=np.arange(7),role=[m.ROLES[i%5] for i in range(7)]))
seen=[];linear=torch.nn.functional.linear
def spy(x,w,b=None):seen.append((x.shape[-1],x.is_contiguous()));return linear(x,w,b)
with patch.object(torch.nn.functional,'linear',spy):p,bp,proof=m.replay640(base,res,mean,scale,new,frame)
check('functional_base512_and_g640_receive_correct_contiguous_widths',seen==[(512,True),(128,True),(640,True),(32,True)])
class Direct(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.GELU(),torch.nn.Linear(128,5));self.delta=torch.nn.Sequential(torch.nn.Linear(640,32),torch.nn.GELU(),torch.nn.Linear(32,1))
        for layer,w,b in [(self.base[0],base['encoder.0.weight'],base['encoder.0.bias']),(self.base[2],base['head.weight'],base['head.bias']),(self.delta[0],res['encoder.0.weight'],res['encoder.0.bias']),(self.delta[2],res['head.weight'],res['head.bias'])]:layer.weight.data.copy_(torch.from_numpy(w));layer.bias.data.copy_(torch.from_numpy(b))
    def forward(self,x):
        z=self.base(x[:,:512].contiguous());raw=z.clone();raw[:,0]+=4*torch.tanh(self.delta(x))[:,0];return raw
net=Direct().eval()
with torch.inference_mode():x=(torch.from_numpy(new[np.arange(7)].astype('f4'))-torch.from_numpy(mean))/torch.from_numpy(scale);expectedp=net(x).log_softmax(1).exp().numpy()
check('independent_F32_functional_replay_bitexact_tiny_module',np.array_equal(p.view('u4'),expectedp.view('u4')) and proof['raw_non_system_bitexact'] and proof['p_non_system_different_class_swaps']==0)
zero={k:v.copy() for k,v in res.items()};zero['head.weight'][:]=0;zero['head.bias'][:]=0;pz,bz,zproof=m.replay640(base,zero,mean,scale,new,frame)
check('same640_zero_path_actual_p_bitexact_base',np.array_equal(pz.view('u4'),bz.view('u4')) and zproof['delta_min']==zproof['delta_max']==0)
# Fault probability layer only: raw invariance cannot excuse a different non-system winner.
calls=[0]
def wrong_p_logits(z,dim):
    calls[0]+=1;v=[.1,.2,.5,.15,.05] if calls[0]%2 else [.1,.5,.2,.15,.05];return torch.log(torch.tensor(v,dtype=torch.float32).repeat(len(z),1))
with patch.object(torch,'log_softmax',wrong_p_logits):reject('observed_non_system_probability_class_swap_is_hard_failure',lambda:m.replay640(base,res,mean,scale,new,frame))
# Policy wiring independently mocked, not production thresholds/guards.
def guard(c,v,gpu=False):return {'rejection_reasons':v.get('fail',[])}
math=SimpleNamespace(guard=guard);records=[dict(record=n,validation={'min33':.91 if n=='new-max33' else .9}) for n in m.RECORDS];zero_map={n:{'min33':.9} for n in m.ZERO};legacy={'min33':.8}
s=m.select_records(math,records,zero_map,legacy)
check('14records_5strict_8guards_savedGPU_no_failed_model_floor',s['promoted'] and len(s['guards'])==9 and len(s['min33_gains'])==5 and not set(m.STRICT+m.GUARDS)&{'priorLongMix-max33','naturalCE-max33'})
bad=copy.deepcopy(records);next(r for r in bad if r['record']=='new-max33')['validation']['min33']=.89999;next(r for r in bad if r['record']=='new-oldmean')['validation']['min33']=.99
check('absolute90_failure_cannot_use_high_oldmean_fallback',not m.select_records(math,bad,zero_map,legacy)['promoted'])
bad=copy.deepcopy(records);next(r for r in bad if r['record']=='priorBinary-max33')['validation']['min33']=.91
check('strict_control_tie_blocks_promotion',not m.select_records(math,bad,zero_map,legacy)['promoted'])
# Read complete actual adapter statically; all read/replay expressions remain unexecuted.
tree=ast.parse(P.read_text());actual=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='actual');code=ast.unparse(actual)
check('actual_gate_first_and_final_then_digest_recheck',ast.unparse(actual.body[0])=='terminal_gate()' and code.count('terminal_gate()')==2 and 'digest(p) == want' in code)
check('all3cohorts_new_matched_zero_replayed_raw_evidence_bound',"for arm in ['new', 'matched']" in code and "replay('zero'" in code and 'for b, frame in rows.items()' in code and 'lib.raw_evidence(rawproof[b], inv' in code)
check('fit578090_scaler_bits_duplicated_scaler_and_sample_hashes_bound',all(t in code for t in ["fit_rows=578090",'natural_rows_used=0','validation_rows_used=0','np.r_[om, om[:128]]','np.r_[oscale, oscale[:128]]',"static['sample_ix_sha256'][n]","np.array_equal(z[arm + '_' + k].view(np.uint32), v.view(np.uint32))"]))
check('oldgrid_legacy_kept_newgrid_independent_f32_and192CSV',all(t in code for t in ['math.direct_grid(rows, p)','math.legacy_mean','lib.verify_csv','combined_csv_rows=192']) and len(m.RECORDS)==14 and len(m.ZERO)==2)
check('history_budget_best_before_new_and_originalGPU_checked',all(t in code for t in ['math.history_audit',"training['new_C4_predictions_read'] is False","training['public_validation_predictions_read'] is False",'original_savedGPU_validation','training_sha']))
check('no_production_import_or_global_monkeypatch_actual_adapter',not any(isinstance(n,ast.Import) and any(a.name.startswith('systemness_') for a in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').startswith('systemness_') for n in tree.body) and 'setattr(' not in code and 'math.actual(' not in code and 'lib.actual(' not in code)
check('source_unchanged',sha(P)==PIN)
folder=Path(tempfile.mkdtemp(prefix='feature-review-complementary640-auditor-',dir='/tmp'));out=folder/'report.json';r=dict(stage='independent_source_tiny_CPU_complementary640_result_auditor_review',passed=True,mustfix=[],source_path=str(P),source_sha256_before=PIN,source_sha256_after=sha(P),checks=checks,check_count=len(checks),final_source_and_protocol_pins_pending=True,actual_enabled=False,actual_layout_or_feature_bank_read=False,real_probabilities_or_model_weights_read=False,production_modules_imported=False,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False)
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(report=str(out),report_sha256=sha(out),script_sha256=sha(__file__),source_sha256=sha(P),checks=len(checks))))
