"""Independent source/AST-isolated tiny CPU review; never imports production modules."""
import ast,copy,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
SOURCE=ROOT/'systemness_complementary_head_readout.py'
EXPECTED='9c9565f75190f8b81dc97e470dcb54080a9b1229b0b22e174299e8cf138c0a97'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
text=SOURCE.read_text();tree=ast.parse(text);checks=[]
def check(n,v=True):
    assert bool(v),n
    checks.append(n)
def reject(n,f):
    try:f()
    except (RuntimeError,ValueError,AssertionError,KeyError):checks.append(n);return
    raise AssertionError(n)
def require(v,m):
    if not bool(v):raise RuntimeError(m)
def isolate(path,names,ns):
    t=ast.parse(Path(path).read_text());nodes=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
roles=['system','user','cot','assistant','tool']
ns=dict(np=np,pd=pd,torch=torch,Path=Path,json=json,require=require)
hpns=dict(np=np,require=require,DIMENSIONS=512,HEADS=8,HEAD_DIM=64)
isolate(ROOT/'systemness_head_probes.py',{'make_net','fit_scaler'},hpns)
ns['hp']=SimpleNamespace(make_net=hpns['make_net']);ns['data']=SimpleNamespace(ROLES=roles)
isolate(SOURCE,{'bank_gate','make_model','CombinedBank','ExtraFitBank','extra_mapping','select_records','flat_metrics','read_scaler','verified_completion','zero_controls','declare'},ns)
# Frozen null prerequisite fails before any hash/read/mkdir.
def forbidden(*a,**k):raise AssertionError('Formal IO forbidden in synthetic annex')
ns.update(EXTRA_COMPLETE_SHA=None,EXTRA_AUDIT_PATH=None,EXTRA_AUDIT_SHA=None,read=forbidden)
ns['hp'].require_hash=forbidden
reject('unknown_bank_certificates_reject_before_hash_or_read',ns['bank_gate'])
# Real tiny torch model with synthetic base only, no saved weights or predictions.
torch.set_num_threads(2);rng=np.random.RandomState(90)
base={'encoder.0.weight':rng.normal(0,.015,(128,512)).astype('f4'),'encoder.0.bias':rng.normal(0,.01,128).astype('f4'),'head.weight':rng.normal(0,.02,(5,128)).astype('f4'),'head.bias':np.array([0,.3,.1,-.2,.15],'f4')}
a=ns['make_model'](base);b=ns['make_model'](base)
check('trainable_parameters20545_and_all_initial_neural_arrays_identical',sum(p.numel() for p in a.parameters() if p.requires_grad)==20545 and all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
a.train();check('recursive_train_keeps_frozen_base_eval_no_grad',not a.base.training and not any(p.requires_grad for p in a.base.parameters()))
x=torch.from_numpy(rng.normal(size=(9,1280)).astype('f4'))[:,::2];seen=[]
handle=a.base.encoder[0].register_forward_pre_hook(lambda m,args:seen.append((args[0].shape,args[0].is_contiguous())))
raw,rb,delta=a.components(x);handle.remove()
check('noncontiguous640_input_base_gets_only_contiguous512',seen==[(torch.Size([9,512]),True)])
check('actual_zero_correction_raw_is_bitexact_base',torch.equal(raw,rb) and torch.equal(delta,torch.zeros_like(delta)))
loss=torch.nn.functional.binary_cross_entropy_with_logits(a.binary_logit(x),torch.tensor([0.,1.,0.,1.,1.,0.,1.,0.,1.]));loss.backward()
check('only_correction_receives_BCE_gradient',all(p.grad is None for p in a.base.parameters()) and a.correction[-1].weight.grad.abs().sum()>0)
a.eval()
with torch.no_grad():a.correction[-1].weight.fill_(.2);r,rb,d=a.components(x)
check('nonzero_delta_preserves_raw_other4_bits_and_is_bounded',torch.equal(r[:,1:].view(torch.int32),rb[:,1:].view(torch.int32)) and d.abs().max()<=4)
# Exact bank mapping with nonzero full C4 parent offsets; no actual layout read.
old=pd.DataFrame(dict(sample_ix=[10,11,12,13],role=roles[:4],split='fit'))
natural=pd.DataFrame(dict(sample_ix=[40,41,42],role=['system','user','assistant'],split='fit'))
c4frame=pd.DataFrame(dict(sample_ix=[60,61],role=['system','user'],split='validation'),index=[327680,389119])
class At:
    def __getitem__(self,ix):return c4frame.loc[ix]
class SparseParent:
    iloc=At()
    def __len__(self):return 389120
parents=dict(old=old,c4=SparseParent(),natural=natural);parts=[]
for name,frame in [('old',old),('c4',c4frame),('natural',natural)]:
    for ix,r in frame.iterrows():parts.append(dict(parent_bank=name,parent_cache_ix=ix,sample_ix=r.sample_ix,role=r.role,split=r.split))
rows=pd.DataFrame(parts);rows['cache_ix']=np.arange(len(rows));lookup=ns['extra_mapping'](rows,parents,expected=len(rows))
check('C4_parent327680_and389119_map_without_zero_offset_assumption',lookup['c4'][327680]==4 and lookup['c4'][389119]==5 and (lookup['c4'][:327680]==-1).all())
bad=rows.copy();bad.loc[4,'sample_ix']=0;reject('wrong_parent_sample_ix_is_not_silently_mapped',lambda:ns['extra_mapping'](bad,parents,expected=len(bad)))
original=rng.normal(size=(4,512)).astype('f2');extra=rng.normal(size=(9,128)).astype('f2');ix=np.array([3,1,0]);new=ns['CombinedBank'](original,extra,lookup['old']);dup=ns['CombinedBank'](original,extra,lookup['old'],True)
check('virtual640_keeps512_order_and_duplicate_exact_rank01',np.array_equal(new[ix][:,:512],original[ix]) and np.array_equal(dup[ix][:,512:],original[ix,:128]) and np.array_equal(new[ix][:,512:],extra[lookup['old'][ix]]))
reject('new_C4_fit_rejected_even_for_duplicate_control',lambda:ns['CombinedBank'](np.zeros((389120,512),np.float16),extra,lookup['c4'],True)[np.array([0])])
# Fit scaler observes only original-fit indices, with poisoned held-out extras.
values=rng.normal(size=(12,128)).astype('f2');values[:,0]=3.;values[6:]=60000
fitbank=ns['ExtraFitBank'](values,np.arange(12));fitix=np.array([5,1,3,0,4,2]);m,s=hpns['fit_scaler'](fitbank,fitix,chunk=2)
check('F64_Welford_population_scale_uses_only_explicit_fit_and_floor',np.array_equal(m,values[fitix].astype('f8').mean(0).astype('f4')) and np.allclose(s,np.maximum(values[fitix].astype('f8').std(0),1e-6).astype('f4'),rtol=1e-6) and s[0]==np.float32(1e-6))
# Independently mock guard outcomes to test exact unchanged policy wiring.
recordnames=[m+'-'+o for m in ['new','matched','oldhead','ensemble','naturalCE','priorBinary','priorLongMix'] for o in ['oldmean','max33']]
recordnames[recordnames.index('ensemble-oldmean')]='ensemble-original'
guards=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33','priorBinary-max33']
strict=['matched-max33','oldhead-max33','ensemble-max33','zero-max33','priorBinary-max33'];calls=[]
def guard(c,v,gpu=False):calls.append((v['id'],gpu));return {'rejection_reasons':v.get('fail',[])}
ns.update(RECORDS=recordnames,GUARD_CONTROLS=guards,STRICT_CONTROLS=strict,nf=SimpleNamespace(guard=guard))
records=[{'record':n,'validation':{'id':n,'min33':.91 if n=='new-max33' else .9}} for n in recordnames]
zero={n:{'id':n,'min33':.9} for n in ['zero-oldmean','zero-max33']};legacy={'id':'GPU','min33':.89}
result=ns['select_records'](records,legacy,zero)
check('sole_newmax33_requires_exact5strict8guards_plus_savedGPU',result['promoted'] and len(result['min33_gains'])==5 and len(result['guards'])==9 and calls==[(n,False) for n in guards]+[('GPU',True)])
for key in ['priorLongMix-oldmean','priorLongMix-max33','naturalCE-oldmean','naturalCE-max33']:
    assert key not in guards+strict
check('failed_priorLongMix_and_CE_add_no_new_hardfloor')
recs=copy.deepcopy(records);recs[0]['validation']['min33']=.99;recs[1]['validation']['min33']=.899
reject_value=ns['select_records'](recs,legacy,zero)
check('failed_newmax33_no_fallback_to_high_oldmean',not reject_value['promoted'] and reject_value['candidate']=='new-max33')
recs=copy.deepcopy(records);next(r for r in recs if r['record']=='priorBinary-max33')['validation']['min33']=.91
check('strict_tie_to_any_one_control_blocks_promotion',not ns['select_records'](recs,legacy,zero)['promoted'])
for r in records:r['validation'].update(groups={f'g{i}':{'recall':{x:None for x in roles},'support':0} for i in range(11)},pooled_descriptive_only={'recall':{x:None for x in roles},'support':0})
for v in zero.values():v.update(groups=records[0]['validation']['groups'],pooled_descriptive_only=records[0]['validation']['pooled_descriptive_only'])
check('14main_plus_twozero_flatten_to192_CSV_rows',len(ns['flat_metrics'](records))==168 and len(ns['flat_metrics']([dict(record=n,validation=v) for n,v in zero.items()]))==24)
# Static ordering/ownership invariants from the frozen actual code.
def fun(n):return next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==n)
def code(n):return ast.unparse(fun(n))
check('scaler_scope578090_only_and_owner_hash_before_resume_return','len(indices) == 578090' in code('prepare_scalers') and "natural_rows_used=0" in code('prepare_scalers') and "validation_rows_used=0" in code('prepare_scalers') and "['standardization_sha256'] == data.digest(path)" in code('prepare_scalers'))
check('zero_requires_both_frozen_before_predict_and_readonly_never_creates','frozen = both_frozen(plan)'==ast.unparse(fun('zero_controls').body[0]) and 'allow_create and' in code('zero_controls') and 'calibrate(plan, False)' in code('verified_completion') and 'Missing inherited640 controls in readonly path.' in code('inherited_controls'))
check('worker_freezes_training_before_any_new_cohort_predict',code('worker').index("data.atomic_json(OUT / 'training'")<code('worker').index('for b, frame in rows.items()'))
check('fixed641493_157_and_last2517',641493==578090+63403 and (641493+4095)//4096==157 and 641493%4096==2517)
check('source_unchanged',sha(SOURCE)==EXPECTED)
folder=Path(tempfile.mkdtemp(prefix='feature-review-complementary-readout-annex-',dir='/tmp'))
report=dict(stage='independent_AST_tiny_CPU_complementary640_source_review',passed=True,mustfix=[],source_path=str(SOURCE),source_sha256_before=EXPECTED,source_sha256_after=sha(SOURCE),checks=checks,check_count=len(checks),production_modules_imported=False,actual_layout_or_bank_read=False,real_probabilities_or_weights_read=False,source_text_materialized=False,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False)
p=folder/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=sha(p),script_sha256=sha(__file__),source_sha256=sha(SOURCE),checks=len(checks))))
