"""Read-only source/tiny CPU annex; never calls actual against production paths."""
import ast,copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
AUDITOR=Path('/tmp/long_data_binary_residual_results_audit.py')
EXPECTED='87c0e47c0ab9904fa3b77f477abe2bb3d437c66789773914d1b339d0194776c6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(AUDITOR)==EXPECTED
spec=importlib.util.spec_from_file_location('annex_binary_auditor',AUDITOR);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-binary-auditor-',dir='/tmp'));checks=[]
def check(name,value=True):
    assert bool(value),name
    checks.append(name)
def reject(name,fn):
    try:fn()
    except (RuntimeError,ValueError,AssertionError,KeyError):checks.append(name);return
    raise AssertionError('Fault accepted: '+name)
class UnauthorizedAccess(LookupError):pass
def forbidden(*args,**kwargs):raise UnauthorizedAccess('Real metadata/bank/p/GPU path forbidden')
tree=ast.parse(AUDITOR.read_text());imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
check('No production module imports',not any((isinstance(n,ast.Import) and any(v.name.startswith('systemness') for v in n.names)) or (isinstance(n,ast.ImportFrom) and str(n.module).startswith('systemness')) for n in imports))
check('Exact frozen trainer/protocol pin',a.SOURCE_SHA=='fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9' and a.PROTOCOL_SHA=='84ebfc2dff0138dab829cb7756660b819e4d734823508c6219ec9de6c93cba1a')
actual=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='actual')
check('Actual code invokes no production verifier/math',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in {'hp','probes','production','nf','aug'} for n in ast.walk(actual)))
check('Actual prior certificate path matches frozen published layout',"prior_out/'preparation/audits/actual-development-audit.json'" in AUDITOR.read_text())
# A LookupError sentinel is deliberately outside reject()'s caught exceptions.
with patch.object(a,'OUT',folder/'gate'),patch.object(a,'processes',return_value=[98765]),patch.object(a,'read',side_effect=forbidden),patch.object(a,'digest',side_effect=forbidden),patch.object(np,'load',side_effect=forbidden),patch.object(pd,'read_parquet',side_effect=forbidden):
    reject('Live trainer rejects before metadata/hash/probability access',lambda:a.actual(folder/'forbidden.json'))
with patch.object(a,'OUT',folder/'missing'),patch.object(a,'processes',return_value=[]),patch.object(a,'read',side_effect=forbidden),patch.object(a,'digest',side_effect=forbidden),patch.object(np,'load',side_effect=forbidden):
    reject('Missing completion rejects before metadata/hash/probability access',lambda:a.actual(folder/'forbidden.json'))
gate=folder/'notterminal';gate.mkdir();(gate/'complete.json').write_text('{}');(gate/'status.json').write_text('{}')
with patch.object(a,'OUT',gate),patch.object(a,'processes',return_value=[]),patch.object(a,'read',return_value={'stage':'training'}),patch.object(a,'digest',side_effect=forbidden),patch.object(np,'load',side_effect=forbidden):
    reject('Nonterminal status rejects before completion contents/probability access',lambda:a.actual(folder/'forbidden.json'))
# Functional network assembled independently; only synthetic F16 inputs and weights.
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');rng=np.random.RandomState(901)
base={'encoder.0.weight':(rng.normal(size=(128,512))*.02).astype(np.float32),'encoder.0.bias':np.zeros(128,np.float32),'head.weight':(rng.normal(size=(5,128))*.02).astype(np.float32),'head.bias':np.array([-.3,-.2,.3,.02,-.7],np.float32)}
res={'encoder.0.weight':(rng.normal(size=(32,512))*.03).astype(np.float32),'encoder.0.bias':np.zeros(32,np.float32),'head.weight':np.zeros((1,32),np.float32),'head.bias':np.array([.3],np.float32)}
raw=rng.normal(size=(9,512)).astype(np.float16);mean=np.full(512,.01,np.float32);scale=np.full(512,.7,np.float32)
rows=pd.DataFrame(dict(cache_ix=[8,2,4,0,7],role=a.ROLES))
x=(torch.from_numpy(raw[rows.cache_ix.to_numpy()].astype(np.float32))-torch.from_numpy(mean))/torch.from_numpy(scale)
with torch.inference_mode():
    z=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,torch.from_numpy(base['encoder.0.weight']),torch.from_numpy(base['encoder.0.bias']))),torch.from_numpy(base['head.weight']),torch.from_numpy(base['head.bias']))
    d=4*torch.tanh(torch.full((5,),.3));changed=z.clone();changed[:,0]+=d
    expected=torch.log_softmax(changed,1).exp().numpy();basep=torch.log_softmax(z,1).exp().numpy()
p,bp,proof=a.residual_replay(base,res,mean,scale,raw,rows)
check('Independent full rawF16 functional replay exact same p/basep',np.array_equal(p,expected) and np.array_equal(bp,basep))
check('Raw tail identity and observed winner preservation',proof['non_system_raw_before_sha256']==proof['non_system_raw_after_sha256']==hashlib.sha256(np.ascontiguousarray(z.numpy()[:,1:]).tobytes()).hexdigest() and proof['conditional_raw_argmax_changes']==proof['p_non_system_different_class_swaps']==0)
check('delta exactly4tanh without dividing input',proof['maximum_abs_delta']==float(d.max()) and abs(proof['maximum_abs_delta']-float(4*torch.tanh(torch.tensor(.3/4))))>.5)
check('Truth-role transition matrices add to overall5x5',np.array_equal(np.sum([np.asarray(v) for v in proof['transitions_by_true_role'].values()],axis=0),np.asarray(proof['five_class_transition'])))
# Portable mismatch must fail even if tampered file describes its own new hash.
state={**{'base.'+k:v for k,v in base.items()},'correction.0.weight':res['encoder.0.weight'],'correction.0.bias':res['encoder.0.bias'],'correction.3.weight':res['head.weight'],'correction.3.bias':res['head.bias']}
meta=dict(kind='binary-residual',dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=a.ROLES,base_model_sha256=a.CURRENT_SHA,embedded_base_weight_sha256=a.weight_state_sha(base,True),mean_sha256=a.array_sha(mean),scale_sha256=a.array_sha(scale))
portable=folder/'toy.npz'
def save(s=state,m=meta):np.savez(portable,mean=mean,scale=scale,metadata=np.array(json.dumps(m)),**{'weight::'+k:v for k,v in s.items()})
save();_,bb,rr=a.binary_model_arrays(portable,base,mean,scale)
check('Portable unpack maps full base+correction exactly',all(np.array_equal(bb[k],base[k]) for k in base) and all(np.array_equal(rr[k],res[k]) for k in res))
bad=copy.deepcopy(state);bad['base.head.bias'][0]+=.1;badbase={k[5:]:v for k,v in bad.items() if k.startswith('base.')};save(bad,dict(meta,embedded_base_weight_sha256=a.weight_state_sha(badbase,True)))
reject('Selfconsistent forged portable frozen base still rejected against original arrays',lambda:a.binary_model_arrays(portable,base,mean,scale))
# Saved evidence includes the fixed semantic definition field that was missed before.
t=np.asarray(proof['five_class_transition'])
saved=dict(tokens=proof['rows'],raw_non_system_sha256=proof['non_system_raw_after_sha256'],base_raw_non_system_sha256=proof['non_system_raw_before_sha256'],raw_non_system_argmax_changes=proof['conditional_raw_argmax_changes'],delta_min=proof['delta_min'],delta_max=proof['delta_max'],delta_mean=proof['delta_mean'],base_to_residual_p_argmax=t.tolist(),p_conditional_non_system_argmax_changes=proof['p_conditional_non_system_argmax_changes'],p_non_system_tie_rows=proof['p_non_system_tie_rows'],p_non_system_underflow_values=proof['p_non_system_underflow_values'],CoT_to_assistant_p_argmax=int(t[2,3]),non_system_to_system_p_argmax=int(t[1:,0].sum()),system_to_non_system_p_argmax=int(t[0,1:].sum()),non_system_to_different_non_system_p_argmax=int(t[1:,1:].sum()-np.trace(t[1:,1:])),metric_convention='Unmodified savedF32 p.argmax; raw-tail bitidentity does not guarantee p rounding preserves non-system argmax.',transition_definition='Rows index original base p.argmax; columns index residual p.argmax, regardless of ground-truth role. CoT_to_assistant is a newly introduced winner transition, not total truth-CoT misclassification.')
a.compare_raw_evidence(saved,proof,'annex');check('Raw proof consumes full frozen schema including transition semantics')
bad=copy.deepcopy(saved);bad['transition_definition']='truth CoT mistakes'
reject('Misleading truth-versus-winner description rejected',lambda:a.compare_raw_evidence(bad,proof,'annex'))
badproof=copy.deepcopy(proof);badproof['five_class_transition'][2][3]+=1
reject('Non-system winner swap hard fails independent replay proof',lambda:a.compare_raw_evidence(saved,badproof,'annex'))
cond=copy.deepcopy(proof);cond['p_conditional_non_system_argmax_changes']=2;condsave=copy.deepcopy(saved);condsave['p_conditional_non_system_argmax_changes']=2
a.compare_raw_evidence(condsave,cond,'conditional-diagnostic');check('Conditional-only p argmax changes do not expand global winner hardgate')
def val(n):return dict(min33=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in a.GROUPS})
records=[dict(record=n,validation=val(.95 if n=='new-max33' else .93)) for n in a.RECORDS];zeros=[dict(record=n,validation=val(.93)) for n in a.ZERO_RECORDS]
check('Exactly10main+2zero4strict7guard and immutableGPU candidate allowed',a.selection(records,zeros,val(.93))['promoted'] and len(a.GUARD_CONTROLS)==7 and len(a.STRICT_CONTROLS)==4)
for control in a.STRICT_CONTROLS:
    r,z=copy.deepcopy(records),copy.deepcopy(zeros);next(x for x in r+z if x['record']==control)['validation']['min33']=.95
    check('Strict control tie rejected '+control,not a.selection(r,z,val(.93))['promoted'])
bad=copy.deepcopy(records);next(x for x in bad if x['record']=='new-max33')['validation']['min33']=.9-1e-13
check('Absolute90 is not lowered by1e-12 comparison tolerance',not a.selection(bad,zeros,val(.8))['promoted'])
high=val(.93);high['groups']['old-natural']['system_f1']=.99
check('Original savedGPU floor rejects apparently improved CPU selection',not a.selection(records,zeros,high)['promoted'])
for x in records:
    if x['record'].startswith('naturalCE-'):x['validation']=val(.999)
check('NaturalCE two strong records remain diagnostic and not fallback',a.selection(records,zeros,val(.93))['promoted'])
# A single tiny full33 grid is sufficient; no production datasets or model search.
r,p=a.synthetic();tau,g,annex=a.direct_grid(r,p);v=a.evaluate(r,p,tau)
manual=[]
for name,(bank,ix) in a.group_indices(r).items():
    yy=a.labels(r[bank].iloc[ix])==0;chosen=p[bank][ix,0]>=float(tau);tp=int((yy&chosen).sum());fp=int((~yy&chosen).sum());fn=int((yy&~chosen).sum());tn=int((~yy&~chosen).sum());manual.extend([tp/(tp+fp),tp/(tp+fn),(tp/(tp+fn)+tn/(tn+fp))/2])
check('Selected min33 directly equals own11group binary counts',len(manual)==33 and min(manual)==v['min33']==g['min33'] and g['groups']==11 and len(a.FEASIBLE)==9)
check('Legacy CoT support separate from SUA public NAs',v['groups']['old-natural']['recall']['cot'] is not None and all(v['groups'][k]['recall']['cot'] is None and v['groups'][k]['recall']['tool'] is None for k in a.PUBLIC))
def flat(entries):
    result=[]
    for entry in entries:
        for group,m in dict(entry['validation']['groups'],pooled_descriptive_only=entry['validation']['pooled_descriptive_only']).items():result.append(dict(record=entry['record'],group=group,**{k:x for k,x in m.items() if not isinstance(x,(dict,list))},**{'role_recall_'+k:x for k,x in m['recall'].items()}))
    return pd.DataFrame(result)
rs=[dict(record=n,validation=v) for n in a.RECORDS];zs=[dict(record=n,validation=v) for n in a.ZERO_RECORDS];mf,zf=flat(rs),flat(zs)
a.verify_csv(mf,rs);a.verify_csv(zf,zs,True);check('Main120=110+10 and zero24=22+2 exact coverage',len(mf)==120 and len(zf)==24)
bad=mf.copy();bad.loc[0,'record']=bad.loc[1,'record'];bad.loc[0,'group']=bad.loc[1,'group'];reject('Duplicate CSV row cannot replace missing pair',lambda:a.verify_csv(bad,rs))
assert sha(AUDITOR)==EXPECTED
report=dict(stage='source_and_synthetic_binary_auditor_annex',audit_passed=True,auditor_source_sha256=EXPECTED,annex_source_sha256=sha(__file__),checks=checks,checks_passed=len(checks),maximum_functional_probability_difference=float(np.abs(expected-p if False else expected-a.residual_replay(base,res,mean,scale,raw,rows)[0]).max()),actual_probability_model_or_bank_reads=False,test_data_read=False,gpu_used=False,formal_outputs_written=False,fixture_directory=str(folder))
path=folder/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(path),report_sha256=sha(path),**report)))
