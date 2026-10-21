"""Independent fusion synthetic CPU audit. Never reads experiment rows/probabilities."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0, str(ROOT/'experiments/role-analysis'))
import numpy as np
import pandas as pd
import torch
f = importlib.import_module('systemness_head_fusion')
torch.set_num_threads(1)
source = Path(f.__file__)
start_sha = hashlib.sha256(source.read_bytes()).hexdigest()
work = Path(tempfile.mkdtemp(prefix='confirmation-review-fusion-'))
checks = []
formal_out,formal_cache=Path(f.OUT),Path(f.CACHE)
initial_formal_state=dict(out_path=str(formal_out),cache_path=str(formal_cache),out_exists=formal_out.exists(),cache_exists=formal_cache.exists())

def checked(name):
    checks.append(name)

def rejected(action, name):
    try:
        action()
    except (AssertionError, RuntimeError, ValueError, KeyError, FileNotFoundError):
        checked(name)
    else:
        raise AssertionError('Expected rejection: '+name)

# Declaration metadata is safe to read; no upstream completed-search/rows call.
assert f.check_design()['source_implemented'] is False
checked('actual pinned pre-implementation design')
saved_design, saved_digest = f.DESIGN, f.DESIGN_SHA256
fake_design = work/'changed-design.json'
fake_design.write_text(saved_design.read_text()+'\n')
f.DESIGN = fake_design
rejected(f.check_design, 'pre-implementation declaration byte tamper rejected')
f.DESIGN = saved_design

# Pure metadata fallback guard with a /tmp namespace.
f.HEAD_CONFIRM_OUT = work/'absent-head-confirmation'
selection = dict(promoted=False,test_used_for_selection=False,primary_changed=False,confirmation_required=True)
completion = dict(candidate_sha256={key:'synthetic' for key in f.heads.CONFIGS},tests_read=False,primary_changed=False)
f.require_fallback(selection,completion)
checked('all-four unpromoted fallback accepted')
for bad in [True, None, 0, 'false']:
    changed=dict(selection,promoted=bad)
    rejected(lambda changed=changed:f.require_fallback(changed,completion),'strict promoted False rejects '+repr(bad))
changed=dict(completion,candidate_sha256={key:'x' for key in list(f.heads.CONFIGS)[:3]})
rejected(lambda:f.require_fallback(selection,changed),'missing fourth completed head candidate rejected')
f.HEAD_CONFIRM_OUT.mkdir()
rejected(lambda:f.require_fallback(selection,completion),'any head-confirmation OUT forbids fallback')
f.HEAD_CONFIRM_OUT.rmdir()

# Independent numerical identities, including the exact implemented network.
torch.manual_seed(19)
z=torch.randn(37,5,dtype=torch.float64)
x=torch.randn(37,11,dtype=torch.float64)
max_odds_error=0.
for hidden in [0,32]:
    delta=f.make_delta(hidden,dimensions=11).double().eval()
    assert torch.count_nonzero(delta.head.weight)==torch.count_nonzero(delta.head.bias)==0
    assert torch.equal(delta.raw(z,x),z)
    checked('zero final layer exact raw-logit identity hidden='+str(hidden))
    with torch.no_grad():delta.head.weight.normal_(0,.3);delta.head.bias.fill_(.2)
    raw=delta.raw(z,x)
    d=delta.delta(x).squeeze(1)
    assert torch.equal(raw[:,1:],z[:,1:])
    assert torch.equal(raw[:,1:].argmax(1),z[:,1:].argmax(1))
    p=raw.softmax(1);q=z.softmax(1)
    conditional=p[:,1:]/p[:,1:].sum(1,keepdim=True)
    assert torch.allclose(conditional,z[:,1:].softmax(1),atol=4e-16,rtol=0)
    error=(torch.logit(p[:,0])-torch.logit(q[:,0])-d).abs().max().item()
    max_odds_error=max(max_odds_error,error)
    assert error<5e-15
    labels=torch.arange(len(z))%5
    ce=torch.nn.functional.cross_entropy(raw,labels)
    binary=torch.nn.functional.binary_cross_entropy_with_logits(z[:,0]-torch.logsumexp(z[:,1:],dim=1)+d,(labels==0).double())
    a=torch.autograd.grad(ce,tuple(delta.parameters()),retain_graph=True)
    b=torch.autograd.grad(binary,tuple(delta.parameters()))
    assert all(torch.allclose(u,v,atol=2e-16,rtol=0) for u,v in zip(a,b))
    with torch.no_grad():delta.head.weight.fill_(1e6);delta.head.bias.fill_(1e6)
    assert torch.isfinite(delta.delta(x)).all() and (delta.delta(x).abs()<=4).all()
    checked('implemented bound/odds/non-system/CE-gradient identities hidden='+str(hidden))

# Standalone portable: a synthetic 8640/128 frozen base, no real model loaded.
base=f.probes.make_net('mlp',128).eval()
base_mean=np.linspace(-.2,.2,8640,dtype=np.float32)
base_scale=np.linspace(.9,1.1,8640,dtype=np.float32)
base_path=work/'synthetic-base.npz'
f.probes.save_model(base_path,base,base_mean,base_scale,dict(kind='mlp',hidden=128,roles=f.data.ROLES,feature_spec=f.attention.SPECS))
base_sha=f.data.digest(base_path)
head_mean=np.linspace(-.1,.1,512,dtype=np.float32)
head_scale=np.linspace(.8,1.2,512,dtype=np.float32)
raw_base=torch.randn(31,8640)
raw_head=torch.randn(31,512)
for key,cfg in f.CONFIGS.items():
    delta=f.make_delta(cfg['hidden']).eval()
    with torch.no_grad():delta.head.weight.normal_(0,.02);delta.head.bias.fill_(.03)
    meta=dict(candidate=key,configuration=cfg,roles=f.data.ROLES,head_dimensions=512,base_dimensions=8640,bound=4.,
        base_model_path=str(base_path),base_model_sha256=base_sha,
        head_scaler_mean_sha256=f.array_digest(head_mean),head_scaler_scale_sha256=f.array_digest(head_scale),
        base_scaler_mean_sha256=f.array_digest(base_mean),base_scaler_scale_sha256=f.array_digest(base_scale))
    path=work/(key+'.npz')
    f.save_model(path,delta,head_mean,head_scale,meta)
    portable,_=f.load_model(path,'cpu')
    with torch.no_grad():
        expected=delta(base.raw((raw_base-torch.from_numpy(base_mean))/torch.from_numpy(base_scale)),
                       (raw_head-torch.from_numpy(head_mean))/torch.from_numpy(head_scale))
        actual=portable(raw_base,raw_head)
    assert torch.equal(actual,expected)
    portable.train()
    assert not portable.base.training and portable.correction.training
    assert all(not param.requires_grad for param in portable.base.parameters())
    checked('portable raw8640/raw512 parity plus train-mode frozen base '+key)
    with np.load(path,allow_pickle=False) as archive:arrays={name:archive[name].copy() for name in archive.files}
    arrays['scale'][0]*=2
    bad_path=work/('changed-scale-'+key+'.npz');f.atomic_npz(bad_path,**arrays)
    rejected(lambda:f.load_model(bad_path,'cpu'),'portable head scaler digest tamper rejected '+key)
    changed_base=work/('changed-base-'+key+'.npz');changed_base.write_bytes(base_path.read_bytes()+b'altered')
    rejected(lambda:f.load_model(path,'cpu',base_path=changed_base),'portable base model digest tamper rejected '+key)

# Exact prefix recovery and explicit failed-cache guards, all synthetic /tmp.
cache=work/'logit-cache';count=23;chunk=7
rows=pd.DataFrame(dict(cache_ix=np.arange(count),sample_ix=np.arange(count)+9000))
array=f.initialize_logits(cache,'synthetic-plan','synthetic-rows',rows=count)
assert np.isnan(array).all()
for i in range(2):array[i*chunk:(i+1)*chunk]=np.arange(chunk*5,dtype=np.float32).reshape(chunk,5)+i
array.flush()
hashes=[f.array_digest(np.asarray(array[i*chunk:(i+1)*chunk])) for i in range(2)]
f.data.atomic_json(cache/'base-logits-progress.json',f.logit_progress(array,rows,2,hashes,'synthetic-plan',chunk))
assert f.restore_logits(cache,array,rows,'synthetic-plan',chunk)==hashes
checked('NaN initialization and exact committed-prefix recovery')
array[0,0]+=1;array.flush()
rejected(lambda:f.restore_logits(cache,array,rows,'synthetic-plan',chunk),'committed baseline logit tamper rejected')
del array
raw=cache/'base-logits.f32';saved=raw.read_bytes();raw.write_bytes(saved[:-4])
rejected(lambda:f.initialize_logits(cache,'synthetic-plan','synthetic-rows',rows=count),'truncated logit cache with old progress rejected')
raw.unlink()
rejected(lambda:f.initialize_logits(cache,'synthetic-plan','synthetic-rows',rows=count),'missing logit cache with old progress rejected')

# Pure selection: numerical drift control cannot be claimed as delta improvement.
def result(old=.8,full=None,tail=None,mean=None):
    full=old if full is None else full;tail=full if tail is None else tail
    profiles={p:dict(system_f1=(full if p=='long-full' else old),accuracy=.85,system_false_positive_rate=.01) for p in f.long_probes.PROFILES}
    return dict(mean_system_f1=(4*old+full)/5 if mean is None else mean,profiles=profiles,long_full_tail=dict(system_f1=tail))
original=result(.8);zero=result(.82)
drift_only=result(.815,full=.82,tail=.82)
chosen,best,guards,promoted,reasons=f.choose_candidate([dict(candidate='delta-linear',validation=drift_only)],original,zero)
assert guards['delta-linear']['eligible'] and not promoted and 'mean_f1_not_above_both_controls_plus1e-12' in reasons
checked('dual-control gate rejects apparent gain explained by delta0 drift')
gain=result(.821,full=.83,tail=.83)
assert f.choose_candidate([dict(candidate='delta-linear',validation=gain)],original,zero)[3]
checked('positive gain passing both controls can promote')
boundary=result(.82,mean=.82+1e-12)
assert not f.choose_candidate([dict(candidate='delta-linear',validation=boundary)],original,zero)[3]
checked('strict mean gain boundary plus1e-12 enforced')

# Downstream evidence must independently require all full/CPU checks and budgets.
if hasattr(f, 'require_candidate_evidence'):
    history=[dict(epoch=e,instances=f.FIT_ROWS,steps=f.STEPS,cumulative_steps=e*f.STEPS,
                  loss=1.,mean_system_f1=(.7 if e==1 else .8 if e==2 else .79)) for e in range(1,11)]
    evidence=dict(zero_full_validation_rows=f.VALID_ROWS,zero_portable_full_validation_rows=f.VALID_ROWS,
        portable_full_validation_rows=f.VALID_ROWS,portable_cpu_rows=256,
        zero_max_probability_difference_to_original=0.,zero_portable_max_probability_difference_to_original=0.,
        portable_full_max_probability_difference=0.,portable_cpu_max_probability_difference=0.,
        history=history,epochs=10,best_epoch=2,best_steps=2*f.STEPS,actual_steps=10*f.STEPS,
        actual_instances=10*f.FIT_ROWS,instances_per_epoch=f.FIT_ROWS,steps_per_epoch=f.STEPS,
        last_batch_instances=554,validation=dict(mean_system_f1=.8))
    f.require_candidate_evidence(evidence)
    checked('valid full portable/CPU and early-stopping history evidence accepted')
    for field,value in [('portable_cpu_rows',255),('zero_portable_full_validation_rows',256),
        ('portable_full_max_probability_difference',float('nan')),
        ('portable_cpu_max_probability_difference',-1.),('actual_steps',10*f.STEPS+1),
        ('best_epoch',3)]:
        changed=copy.deepcopy(evidence);changed[field]=value
        rejected(lambda changed=changed:f.require_candidate_evidence(changed),'invalid candidate evidence rejected '+field)
    changed=copy.deepcopy(evidence)
    changed['history'].append(dict(epoch=11,instances=f.FIT_ROWS,steps=f.STEPS,cumulative_steps=11*f.STEPS,loss=1.,mean_system_f1=.79))
    changed.update(epochs=11,actual_steps=11*f.STEPS,actual_instances=11*f.FIT_ROWS)
    rejected(lambda:f.require_candidate_evidence(changed),'continued after early-stop condition rejected')

# Formal entry must fail on missing upstream before creating namespace/bank read.
if hasattr(f,'train'):
    f.OUT=work/'never-created-out';f.CACHE=work/'never-created-cache'
    f.no_competing_jobs=lambda *a,**kw:None
    seen=[]
    def unavailable():
        seen.append('signature')
        raise RuntimeError('synthetic missing upstream completion')
    f.source_signature=unavailable
    def forbidden(*a,**kw):
        raise AssertionError('feature bank accessed before gate')
    f.dataset=forbidden
    rejected(f.train,'train missing upstream rejected before namespace or bank access')
    assert seen==['signature'] and not f.OUT.exists() and not f.CACHE.exists()

assert initial_formal_state['out_exists']==formal_out.exists() and initial_formal_state['cache_exists']==formal_cache.exists()
checked('actual formal OUT and CACHE existence unchanged')
end_sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert start_sha==end_sha,'Source changed during audit; repeat after owner freezes'
report=dict(source=str(source),source_sha256=start_sha,checks=checks,check_count=len(checks),
            maximum_log_odds_error=max_odds_error,fixture_directory=str(work),
            scope='synthetic CPU source audit plus actual design declaration only',
            gpu_used=False,experiment_rows_or_probabilities_read=False,formal_outputs_created=False,formal_paths=initial_formal_state)
path=Path('/tmp/confirmation_review_fusion_audit.json');path.write_text(json.dumps(report,indent=2)+'\n')
print(path);print(json.dumps(report,sort_keys=True))
