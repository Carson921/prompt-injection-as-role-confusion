"""Source/tiny CPU annex only; never invoke real natural-fit audit/data."""
import ast,hashlib,importlib.util,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.special import erf
source=Path('/tmp/long_data_natural_fit_results_audit.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
start=sha(source);assert start=='0d755e9491ecf65f5477714140260e4ebc45319d413c3a64ee72a18bc426ad57'
spec=importlib.util.spec_from_file_location('independent_natural_fit_auditor',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-natural-fit-auditor-',dir='/tmp'));checks=[]
def check(name,value=True):assert bool(value),name;checks.append(name)
class UnexpectedArtifactRead(LookupError):pass
def forbidden(*a,**k):raise UnexpectedArtifactRead('Actual artifact reader forbidden')
def rejects(fn):
    try:fn()
    except (RuntimeError,AssertionError):return True
    return False
with patch.object(np,'load',side_effect=forbidden),patch.object(pd,'read_parquet',side_effect=forbidden):
    with patch.object(m,'processes',return_value=[123]),patch.object(m,'read',side_effect=forbidden),patch.object(m,'digest',side_effect=forbidden):
        check('Live gate before every formal read/hash',rejects(lambda:m.actual(folder/'never.json')))
    with patch.object(m,'OUT',folder/'absent'),patch.object(m,'processes',return_value=[]),patch.object(m,'read',side_effect=forbidden),patch.object(m,'digest',side_effect=forbidden):
        check('Missing complete gate before every formal read/hash',rejects(lambda:m.actual(folder/'never.json')))
    fake=folder/'fake';fake.mkdir();(fake/'complete.json').write_text('{}');(fake/'status.json').write_text('{}')
    with patch.object(m,'OUT',fake),patch.object(m,'processes',return_value=[]),patch.object(m,'read',return_value={'stage':'validation_complete'}),patch.object(m,'digest',side_effect=forbidden):
        check('Output restricted to tmp before any source/model/p read',rejects(lambda:m.actual('/root/forbidden-audit.json')))
    tree=ast.parse(source.read_text());imports=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):imports += [a.name for a in n.names]
        if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
    check('No production imports; pinned trainer and formal protocol',not any(n.startswith('systemness') for n in imports)
        and m.SOURCE_SHA=='7c0674adbce3ef4980a1f29077a6ca36396b93aa992c32ad848ff9e56be4aea3'
        and m.PROTOCOL_SHA=='1edc202c1f8f2651e730a650590489c5a8f1f8da1b9da4ab730f0b949b59c104')
    check('Independent tied-score rank AUC/AP',m.rankings(np.array([.5,.5,.1,.1],np.float32),np.array([True,False,True,False]))==(.5,.5))
    p=np.array([[.4,.5,.04,.04,.02],[.3,.2,.4,.08,.02],[.6,.1,.05,.2,.05],[.1,.2,.1,.2,.4],[.5,.1,.1,.2,.1]],np.float32);y=np.array([0,1,3,3,0]);v=m.metrics(p,y,.35)
    matrix=np.zeros((5,5),int)
    for a,b in zip(y,p.argmax(1)):matrix[a,b]+=1
    diag=matrix.diagonal();sup=matrix.sum(1);macro=np.mean([2*diag[i]/(sup[i]+matrix.sum(0)[i]) for i in range(5) if sup[i]])
    check('Five-way errors / truth-supported macro / threshold vs argmax separated',v['confusion']==matrix.tolist() and v['macro_f1']==macro
        and v['system_recall']!=v['recall']['system'] and v['recall']['cot'] is v['recall']['tool'] is None)
    # Tiny role-slot recipe exercises private RNG, labels and leakage rejection.
    fit=pd.DataFrame(dict(cache_ix=np.arange(15),sample_ix=100+np.arange(15),role=np.tile(m.ROLES,3),split='fit'))
    nf=pd.DataFrame(dict(cache_ix=np.arange(9),sample_ix=4_000_000_000_000+np.arange(9),source_uid=['nf']*9,
        role=['assistant','system','user']*3,split='fit'))
    np.random.seed(7);state=np.random.get_state();ts=torch.get_rng_state().clone();draw,proof=m.matched_draw(fit,nf);after=np.random.get_state()
    check('Slot labels exact / private NumPy and Torch RNG preserved',np.array_equal(fit.set_index('cache_ix').loc[draw.matched_old_cache_ix].role,nf.role)
        and all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(state,after)) and torch.equal(ts,torch.get_rng_state())
        and proof['no_validation_rows_sampled'] and not proof['torch_rng_consumed'])
    bad=fit.copy();bad.loc[0,'split']='validation';check('Role draw refuses validation contamination',rejects(lambda:m.matched_draw(bad,nf)))
    rng=np.random.default_rng(4);members=rng.uniform(size=(4,7,5)).astype(np.float32);members/=members.sum(2,keepdims=True)
    check('Fixed four F64 mean to F32 / missing member fails',np.array_equal(m.fixed_mean(list(members)),members.astype(np.float64).mean(0).astype(np.float32))
        and rejects(lambda:m.fixed_mean(list(members[:3]))))
    def value(n):return dict(min33=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in m.GROUPS})
    records=[dict(record=n,validation=value(.94 if n=='new-max33' else .92)) for n in m.RECORDS];legacy=value(.92)
    check('8records / five guards / only newmax33 eligible',m.selection(records,legacy)['promoted'] and len(m.selection(records,legacy)['guards'])==6)
    bad=json.loads(json.dumps(records));next(r for r in bad if r['record']=='new-max33')['validation']['min33']=.899999999999
    check('Absolute90 rejects even subpicounit shortfall',not m.selection(bad,legacy)['promoted'])
    old=json.loads(json.dumps(legacy));old['groups']['old-long-full-tail']['system_f1']=1.
    check('SavedGPU full-tail floor independent of all CPU controls',not m.selection(records,old)['promoted']
        and 'original-savedGPU:C4_full_tail_f1_regression' in m.selection(records,old)['rejection_reasons'])
    h=[dict(epoch=e,steps=157,instances=641493,cumulative_steps=e*157,loss=.5,old5_mean_f1=.8) for e in range(1,11)];b=m.history_audit(h,1)
    check('157-step epoch /2517 last /best vsactual budget',b['best_steps']==157 and b['actual_steps']==1570 and b['last_batch_instances']==2517)
    rows,ps=m.synthetic();groups=m.group_indices(rows)
    check('Eleven groups/nine FPR / public SUA vs legacy CoT',len(groups)==11 and len(m.FEASIBLE)==9
        and (rows['old'].loc[rows['old'].profile.eq('natural'),'role']=='cot').any() and set(rows['public'].role)=={'system','user','assistant'})
    # Full-row portable recomputation math, small temporary RAM fixture only.
    bank=rng.normal(size=(12,512)).astype(np.float16);frame=pd.DataFrame(dict(cache_ix=[11,2,7]));mean=np.full(512,.13,np.float32);scale=np.linspace(.3,1.7,512,dtype=np.float32)
    w={'encoder.0.weight':rng.normal(0,.02,(128,512)).astype(np.float32),'encoder.0.bias':rng.normal(0,.02,128).astype(np.float32),
       'head.weight':rng.normal(0,.02,(5,128)).astype(np.float32),'head.bias':rng.normal(0,.02,5).astype(np.float32)}
    actual=m.independent_predict(w,mean,scale,bank,frame)
    x=(bank[frame.cache_ix].astype(np.float32)-mean)/scale;z=x.astype(np.float64)@w['encoder.0.weight'].T+w['encoder.0.bias'];z=.5*z*(1+erf(z/np.sqrt(2)))
    logits=z@w['head.weight'].T+w['head.bias'];expected=np.exp(logits-logits.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True);error=float(np.abs(actual-expected).max())
    check('RawF16/scaler/GELU/full5prob independent NumPy/SciPy agreement',error<1e-6 and torch.get_num_threads()==4 and torch.get_float32_matmul_precision()=='highest')
assert sha(source)==start
result={'auditor_source_sha256':start,'source_sha256_after':sha(source),'annex_source_sha256':sha(__file__),'checks':checks,'checks_passed':len(checks),
    'maximum_toy_portable_difference':error,'actual_probabilities_or_artifacts_read':False,'gpu_used':False,'formal_outputs_written':False}
(folder/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'report':str(folder/'report.json'),'report_sha256':sha(folder/'report.json'),**result}))
