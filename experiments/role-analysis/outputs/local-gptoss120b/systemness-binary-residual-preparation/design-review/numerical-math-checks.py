#!/usr/bin/env python3
"""CPU synthetic design checks only. No production imports, weights, data, probabilities or GPU."""
import hashlib,json,pathlib
import numpy as np
import torch
import torch.nn.functional as F
CHECKS=[]
def check(n,x):assert bool(x),n;CHECKS.append(n)
torch.set_num_threads(2)
gen=torch.Generator().manual_seed(20261021)
z=torch.randn((31,5),generator=gen,dtype=torch.float64);g=torch.linspace(-1.2,1.2,31,dtype=torch.float64,requires_grad=True)
delta=4*torch.tanh(g);changed=z.clone();changed[:,0]+=delta
p=F.softmax(z,1);q=F.softmax(changed,1)
check('Onlyraw systemlogit modified',torch.equal(z[:,1:],changed[:,1:]))
check('Rawnon-system argmax bitwise invariant',torch.equal(z[:,1:].argmax(1),changed[:,1:].argmax(1)))
conditional_error=float((F.softmax(z[:,1:],1)-q[:,1:]/q[:,1:].sum(1,keepdim=True)).abs().max().detach())
check('Real-valued conditionalnon-system distribution unchanged',conditional_error<1e-14)
logodds_error=float(((torch.log(q[:,0])-torch.log(q[:,1:].sum(1)))-(torch.log(p[:,0])-torch.log(p[:,1:].sum(1)))-delta).abs().max().detach())
check('System logodds increment equalsdelta',logodds_error<1e-14)
y=torch.arange(31)%5
ce=F.cross_entropy(changed,y,reduction='sum');grad_ce=torch.autograd.grad(ce,g,retain_graph=True)[0]
d=z[:,0]-torch.logsumexp(z[:,1:],dim=1)+delta
bce=F.binary_cross_entropy_with_logits(d,y.eq(0).double(),reduction='sum');grad_bce=torch.autograd.grad(bce,g,retain_graph=True)[0]
grad_error=float((grad_ce-grad_bce).abs().max().detach())
check('Full5CE andbinaryCE gradients equal forfixed others',grad_error<1e-14)
y2=y.clone();y2[y2.ne(0)]=4
ce2=F.cross_entropy(changed,y2,reduction='sum');grad2=torch.autograd.grad(ce2,g)[0]
check('Non-system label identity cannot trainrelative ranking',float((grad_ce-grad2).abs().max().detach())<1e-14)
u=torch.tensor(0.,dtype=torch.float64,requires_grad=True);zero_grad=float(torch.autograd.grad(4*torch.tanh(u),u)[0]);check('Current4tanh(g) derivativezero is4 notolder4tanh(g/4)',zero_grad==4.)
check('Delta bounded by4',bool((4*torch.tanh(torch.tensor([-1e6,1e6],dtype=torch.float64))).abs().le(4).all()))
z32=torch.randn((19,5),generator=gen,dtype=torch.float32);zero=z32.clone();zero[:,0]+=torch.zeros(19)
check('Delta0 same rawconsumer has identicalF32 probabilities',torch.equal(F.log_softmax(z32,1).exp(),F.log_softmax(zero,1).exp()))
saturated=torch.tensor([[20.,0.,-1.,-2.,-3.]],dtype=torch.float32);sp=F.log_softmax(saturated,1).exp();raw_d=saturated[:,0]-torch.logsumexp(saturated[:,1:],1)
check('Savedp0 saturates1 while rawlogodds remainfinite',float(sp[0,0])==1. and torch.isinf(torch.logit(sp[:,0])).all() and torch.isfinite(raw_d).all())
check('BCEWithLogits from raw is stable forboth targets',torch.isfinite(F.binary_cross_entropy_with_logits(raw_d,torch.zeros_like(raw_d))) and torch.isfinite(F.binary_cross_entropy_with_logits(raw_d,torch.ones_like(raw_d))))
base=sp.numpy()[0];mult=np.float32(np.exp(np.float32(-4)));denom=np.float32(1)+base[0]*(mult-np.float32(1));reweighted=base/denom;reweighted[0]=base[0]*mult/denom
raw_changed=saturated.clone();raw_changed[:,0]-=4;raw_probability=F.log_softmax(raw_changed,1).exp().numpy()[0]
check('Multiplicative F32 savedp reconstruction differsfromraw',not np.array_equal(reweighted,raw_probability))
# Strictly ordered non-system adjacent F32 probabilities can collapse after common rescaling.
lo=np.float32(.495);hi=np.nextafter(lo,np.float32(1));tiny=np.float32(1e-7)
p0=np.float32(1-float(lo)-float(hi)-2*float(tiny));original=np.array([p0,lo,tiny,hi,tiny],np.float32)
grid=np.linspace(.001,3.,10000,dtype=np.float32);m=np.exp(grid).astype(np.float32);D=np.float32(1)+p0*(m-np.float32(1));a=lo/D;b=hi/D;s=p0*m/D
hits=np.flatnonzero((a==b)&(a>s));check('Positive commonfactor canmerge adjacentF32 non-system probabilities intoargmax tie',len(hits)>0)
i=int(hits[0]);new=original/D[i];new[0]=original[0]*m[i]/D[i]
check('F32p argmax canchange non-system role solelydue tie',int(original.argmax())==3 and int(new.argmax())==1)
observed_non_swap=int(original.argmax())!=0 and int(new.argmax())!=0 and int(original.argmax())!=int(new.argmax())
check('Frozen observednon-system preservation policy rejects tieexample',observed_non_swap)
result=dict(stage='binary_system_residual_independent_design_math',checks_passed=len(CHECKS),checks=CHECKS,
 realvalued_max_error=dict(conditional_non_system=conditional_error,system_logodds=logodds_error,binary_vs_full5CE_gradient=grad_error),
 delta_formula='4*tanh(g)',delta_zero_derivative=zero_grad,
 saturated_example=dict(raw_logits=saturated.tolist(),saved_probabilities=sp.tolist(),raw_logodds=raw_d.tolist(),reconstructed_logit_savedp0='inf',
  delta=-4.,raw_consumer_probability=raw_probability.tolist(),multiplicative_F32_probability=reweighted.tolist(),multiplicative_sum=float(reweighted.sum()),max_difference=float(np.abs(raw_probability-reweighted).max())),
 tie_example=dict(original_F32_probability=original.tolist(),delta=float(grid[i]),changed_F32_probability=new.tolist(),old_argmax=3,new_argmax=1,explanation='Mathematical ordering cannot reverse; floating rounding merges two distinct values and index-priority argmax chooses an earlier class.'),
 root_design_sha256='c68a32d55e732f678f77a7ef91d8d966a416f118d43663f913b3c9f2d14e09c5',root_policy='Observed p.argmax non-system to different-non-system changes must be zero; the constructed tie example is a technical rejection, never repaired by epsilon.',model_or_data_loaded=False,actual_probabilities_read=False,gpu_used=False,production_sources_modified=False)
pth=pathlib.Path('/tmp/feature_review_binary_residual_math.json');pth.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(pth),report_sha256=hashlib.sha256(pth.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),**result)))
