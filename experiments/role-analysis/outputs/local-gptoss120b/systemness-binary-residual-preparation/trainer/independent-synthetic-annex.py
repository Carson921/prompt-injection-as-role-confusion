#!/usr/bin/env python3
"""Independent source/CPU toy audit only; no real portable weights/data/p/GPU."""
import ast,copy,hashlib,inspect,json,pathlib,sys,tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=pathlib.Path('/root/prompt-injection-as-role-confusion');sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_binary_residual as s
PATH=pathlib.Path(s.__file__)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
START=sha(PATH);CHECKS=[]
EXPECTED_SOURCE='fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9'
def check(n,x=True):assert bool(x),n;CHECKS.append(n)
def reject(n,fn):
 try:fn()
 except(RuntimeError,ValueError,AssertionError,KeyError,FileNotFoundError):CHECKS.append(n)
 else:raise AssertionError('accepted '+n)
def banned(*a,**k):raise AssertionError('Actual model/data/probability/GPU API forbidden')
with patch.object(np,'load',side_effect=banned),patch.object(pd,'read_parquet',side_effect=banned),patch.object(s,'read_base',side_effect=banned),patch.object(s.data.runtime,'load_model',side_effect=banned),patch.object(s.nf,'dataset',side_effect=banned),patch.object(s.nf,'static_snapshot',side_effect=banned):
 check('Final owner-held source fd49 exact',START==EXPECTED_SOURCE)
 torch.set_num_threads(2);gen=torch.Generator().manual_seed(195)
 base={'encoder.0.weight':(torch.randn((128,512),generator=gen)*.025).numpy(),'encoder.0.bias':(torch.randn(128,generator=gen)*.01).numpy(),'head.weight':(torch.randn((5,128),generator=gen)*.05).numpy(),'head.bias':(torch.randn(5,generator=gen)*.01).numpy()}
 state=torch.get_rng_state().clone();net=s.make_model(base)
 check('Construction preserves callerCPU RNG',torch.equal(state,torch.get_rng_state()))
 check('Identical fullinitial hashes across two arms',s.initial_hash(base)==s.initial_hash(base))
 check('Residual lastW/b exactzero',torch.count_nonzero(net.correction[-1].weight)==0 and torch.count_nonzero(net.correction[-1].bias)==0)
 check('Residual trainable parameter budget16449',sum(p.numel() for p in net.correction.parameters())==16449)
 net.train(True)
 check('Parenttrain keeps frozenbase eval including dropout',net.training and net.correction[2].training and not net.base.training and not net.base.encoder[2].training and not any(p.requires_grad for p in net.base.parameters()))
 x=torch.randn((37,512),generator=gen);raw,b,delta=net.components(x)
 check('Delta0 rawconsumer exactly oldbase',torch.equal(raw,b) and torch.count_nonzero(delta)==0 and torch.equal(F.log_softmax(raw,1).exp(),F.log_softmax(b,1).exp()))
 initial={k:v.detach().cpu().numpy().copy() for k,v in net.base.state_dict().items()};y=torch.arange(37)%5
 optimizer=torch.optim.AdamW(net.correction.parameters(),lr=.001,weight_decay=.001)
 for _ in range(3):
  raw,b,d=net.components(x);loss=F.binary_cross_entropy_with_logits(raw[:,0]-torch.logsumexp(raw[:,1:],1),y.eq(0).float());optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.correction.parameters(),5);optimizer.step()
 check('Three toyBCE updates nevercreate basegrad or changebasebytes',all(p.grad is None for p in net.base.parameters()) and all(np.array_equal(v.detach().numpy().view(np.uint32),initial[k].view(np.uint32)) for k,v in net.base.state_dict().items()))
 net.eval();raw,b,delta=net.components(x)
 check('Traineddelta remainsbound4 and rawnonlogits bitexact',bool(delta.abs().le(4).all()) and np.array_equal(raw[:,1:].detach().numpy().view(np.uint32),b[:,1:].detach().numpy().view(np.uint32)))
 check('Rawconditional logits/argmax unchanged',torch.equal(raw[:,1:].argmax(1),b[:,1:].argmax(1)) and torch.equal(F.softmax(raw[:,1:],1),F.softmax(b[:,1:],1)))
 double=copy.deepcopy(net).double().eval();dx=x.double();ar,ab,ad=double.components(dx);params=list(double.correction.parameters())
 grad_ce=torch.autograd.grad(F.cross_entropy(ar,y),params,retain_graph=True);grad_bce=torch.autograd.grad(F.binary_cross_entropy_with_logits(ar[:,0]-torch.logsumexp(ar[:,1:],1),y.eq(0).double()),params)
 gradient_error=max(float((a-b).abs().max()) for a,b in zip(grad_ce,grad_bce))
 check('Actualsource CE/BCE residual gradients same atF64',gradient_error<1e-13)
 rows=pd.DataFrame(dict(cache_ix=np.arange(len(x)),sample_ix=4_000_000_000_000+np.arange(len(x)),split='fit'))
 p,evidence=s.predict_net(net,torch.zeros(512),torch.ones(512),x.numpy().astype(np.float16),rows)
 with torch.no_grad():reference=net.base.raw(torch.from_numpy(x.numpy().astype(np.float16).astype(np.float32))).numpy()
 check('Sourcepredictor rawtail digest matches independentbytes',evidence['raw_non_system_sha256']==evidence['base_raw_non_system_sha256']==ah(np.ascontiguousarray(reference[:,1:]).view(np.uint32)))
 check('Sourcepredictor keeps original5way probability convention',evidence['raw_non_system_argmax_changes']==0 and np.isfinite(p).all() and p.dtype==np.float32 and np.allclose(p.sum(1),1,atol=2e-5,rtol=0))
 val=pd.concat([rows]*8,ignore_index=True).iloc[:256].copy();val['split']='validation'
 reject('Validationparity fails before anyportableprediction',lambda:s.parity(None,None,val))
 s.verify_raw_evidence(evidence,len(rows));bad=copy.deepcopy(evidence)
 bad['base_to_residual_p_argmax']=np.diag([0,len(rows),0,0,0]).tolist();bad['base_to_residual_p_argmax'][1][1]-=1;bad['base_to_residual_p_argmax'][1][3]+=1
 bad['non_system_to_different_non_system_p_argmax']=1
 reject('Full winner-transition evidence rejects raw-preserving user toassistant swap',lambda:s.verify_raw_evidence(bad,len(rows)))
 def v(minimum=.92):return dict(min33=minimum,old5_mean_f1=.92,groups={g:dict(system_f1=.92,accuracy=.92,system_false_positive_rate=.01) for g in s.GROUPS})
 records=[dict(record=n,validation=v(.94 if n=='new-max33' else .92)) for n in s.RECORDS];zero={n:v() for n in ['zero-oldmean','zero-max33']};legacy=v()
 positive=s.select_records(records,legacy,zero)
 check('Ten records plus two separatezero/fourstrict/sevenguards plus originalGPU',len(records)==10 and len(zero)==2 and len(s.STRICT_CONTROLS)==4 and len(s.GUARD_CONTROLS)==7 and len(positive['guards'])==8 and positive['promoted'] and positive['candidate']=='new-max33')
 for n in s.STRICT_CONTROLS:
  rr=copy.deepcopy(records);zz=copy.deepcopy(zero)
  if n in zz:zz[n]['min33']=.94
  else:next(r['validation'] for r in rr if r['record']==n)['min33']=.94
  reject_reason=s.select_records(rr,legacy,zz)
  check('Strict min33 tie rejects '+n,not reject_reason['promoted'] and n+':min33_not_strictly_improved' in reject_reason['rejection_reasons'])
 for n in s.GUARD_CONTROLS:
  rr=copy.deepcopy(records);zz=copy.deepcopy(zero)
  value=zz[n] if n in zz else next(r['validation'] for r in rr if r['record']==n)
  value['groups'][s.PUBLIC[1]]['system_f1']=.99
  selected=s.select_records(rr,legacy,zz)
  check('Every public-systemF1 floor enforced '+n,not selected['promoted'] and n+':public_system_f1_loss' in selected['rejection_reasons'])
 for metric,value,reason in [('accuracy',.90,'public_five_accuracy_loss'),('system_false_positive_rate',.051,'FPR_above_limit')]:
  rr=copy.deepcopy(records);next(r['validation'] for r in rr if r['record']=='new-max33')['groups'][s.PUBLIC[2]][metric]=value
  check('Candidate public '+reason+' rejects',not s.select_records(rr,legacy,zero)['promoted'])
 rr=copy.deepcopy(records);next(r['validation'] for r in rr if r['record']=='new-max33')['groups']['old-long-full-tail']['system_f1']=.919
 check('Legacy tail F1 may not decline even ifmin33 rises',not s.select_records(rr,legacy,zero)['promoted'])
 high=copy.deepcopy(legacy);high['groups']['old-natural']['system_f1']=.99
 selected=s.select_records(records,high,zero)
 check('Original savedGPU floor separately hardrejects',not selected['promoted'] and 'original-savedGPU:old4_system_f1_loss' in selected['rejection_reasons'])
 rr=copy.deepcopy(records)
 for r in rr:
  if r['record'].startswith('naturalCE'):r['validation']=v(.999)
 check('Strong naturalCE diagnostic neitherhardfloor norfallback',s.select_records(rr,legacy,zero)['promoted'])
 rr=copy.deepcopy(records);next(r['validation'] for r in rr if r['record']=='new-max33')['min33']=.899999
 selected=s.select_records(rr,legacy,zero)
 check('Absolute90 failure neverselects diagnostic fallback',not selected['promoted'] and selected['no_fallback'] and selected['candidate']=='new-max33')
 # Call the real calibrator with all physical/probability APIs replaced by toys.
 order=[];toy_p={bank:np.tile(np.array([[.8,.05,.05,.05,.05]],np.float32),(len(rows),1)) for bank in ['old','c4','public']}
 def toy_candidate(arm,plan,frames,nrows):order.append(('verified-candidate',arm));return {},toy_p
 def toy_inherit(plan,frames,allow_create=False):order.append(('inherited',allow_create));return {n:toy_p for n in ['oldhead','ensemble','naturalCE']},legacy
 def toy_zero(plan,banks,frames,nrows,old,allow_create=False):
  assert order[:2]==[('verified-candidate','new'),('verified-candidate','matched')]
  assert old is toy_p
  order.append(('zero',allow_create));return zero
 metric=v(.94);metric['system_threshold']=.6008080808080808
 with patch.object(s,'dataset',return_value=({},rows,rows,rows,rows,rows,{})),patch.object(s,'candidate_record',side_effect=toy_candidate),patch.object(s,'inherited_controls',side_effect=toy_inherit),patch.object(s,'zero_controls',side_effect=toy_zero),patch.object(s,'metrics_at',side_effect=lambda *a:metric),patch.object(s,'max33',return_value=(.6008080808080808,dict(min33=.94,old5_mean_f1=.92))),patch.object(s.lp,'validation_metrics',return_value={'system_threshold':.6}):
  calibrated=s.calibrate({},False)
 check('Readonly calibrate verifies BOTH candidates beforezero and propagates createFalse',len(calibrated[0])==10 and order==[('verified-candidate','new'),('verified-candidate','matched'),('inherited',False),('zero',False)])
 # Consumer-level base substitution with its own internally valid metadata still fails against original fixed arrays.
 frozen=s.make_model(base);mean=torch.zeros(512);scale=torch.ones(512);static=dict(embedded_base_weight_sha256=s.state_hash(base));fakeplan={}
 impostor=copy.deepcopy(frozen);impostor.base.head.bias.data[0]+=.25
 with patch.object(s,'load_model',return_value=(impostor,mean,scale,{})),patch.object(s,'read',return_value={'static':static}),patch.object(s,'read_base',return_value=(base,mean.numpy(),scale.numpy(),{})):
  reject('Self-consistent substituted embeddedbase still rejected against originalarray',lambda:s.verify_portable('/tmp/not-read',fakeplan,'new'))
 folder=pathlib.Path(tempfile.mkdtemp(prefix='feature-review-binary-residual-source-audit-',dir='/tmp'))
 with patch.object(s,'OUT',folder),patch.object(s,'both_frozen',return_value={'new':'n','matched':'m'}),patch.object(s,'predict_raw',side_effect=banned),patch.object(s,'parity',side_effect=banned):
  reject('Readonly missingzero record nevercreates modelorprobability',lambda:s.zero_controls({},None,dict(old=rows),rows,None,False))
  check('Readonly missingzero leaves directoryempty',not list(folder.iterdir()))
 # Source-order checks cover the worker/supervisor lifecycle without CUDA or launching a child.
 def calls(fn,name):
  tree=ast.parse(inspect.getsource(fn));return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id==name) or (isinstance(n.func,ast.Attribute) and n.func.attr==name))]
 worker_text=inspect.getsource(s.worker);freeze_pos=worker_text.index("data.atomic_json(OUT/'training'");predict_pos=worker_text.index('predict_raw(path')
 check('Per-arm training/model/history/GPUold freeze precedes allnew-cohort predictions',freeze_pos<predict_pos and "new_C4_predictions_read=False,public_validation_predictions_read=False" in worker_text)
 check('Zero routine checks bothfrozen as itsfirst operation',ast.unparse(ast.parse(inspect.getsource(s.zero_controls)).body[0].body[0].value)=='both_frozen(plan)')
 check('Completed verifier invokes readonly calibrateFalse',any(len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value is False for n in calls(s.verified_completion,'calibrate')))
 check('Parent competingjob guard precedes signature andworkers',inspect.getsource(s.train).index('no_competing_jobs()')<inspect.getsource(s.train).index('source_signature()')<inspect.getsource(s.train).index('run_workers(pending)'))
 static_text=inspect.getsource(s.static_snapshot)
 check('Static prior b8de cert bound through fixed c68a design plus actual sourcecompleteprotocolselectionrecords',"hp.require_hash(PRIOR_AUDIT,design['background']['latest_actual_audit_sha256']" in static_text and all("audit['"+n+"']" in static_text for n in ['trainer_source_sha256','trainer_complete_sha256','trainer_protocol_sha256','trainer_selection_sha256','trainer_records_sha256']) and "audit['model_sha256']['new']==NATURAL_CE_SHA" in static_text)
 check('Static declaration references hashes/sampleIDs withoutopening p arrays',not calls(s.static_snapshot,'read_probability') and not calls(s.static_snapshot,'load') and "sample_ix_sha256=samples[b]" in static_text and "validation_probability_arrays_read=False" in static_text)
 publish=inspect.getsource(s.publish)
 check('CSV coverage separate120main24zero is mandatory',"len(flat)==120 and len(zf)==24" in publish and "zero-control-metrics.csv" in publish)
 process=inspect.getsource(s.run_workers)
 check('Supervisor childlease/terminatekillwait/reap and bound2arms retained',all(t in process for t in ['len(pending)<=2','ROLE_BINARY_RESIDUAL_SUPERVISOR_PID','p.terminate()','p.kill()','p.wait(timeout=10)']))
 if hasattr(s,'DESIGN_SHA') and s.DESIGN_SHA:
  check('Design matches root c68a',s.DESIGN_SHA=='c68a32d55e732f678f77a7ef91d8d966a416f118d43663f913b3c9f2d14e09c5')
 check('Source unchanged duringCPU audit',sha(PATH)==START)
folder=pathlib.Path(tempfile.mkdtemp(prefix='feature-review-binary-residual-source-audit-',dir='/tmp'));out=folder/'report.json'
r=dict(stage='independent_binary_residual_source_synthetic_audit',audit_passed=True,source_sha256=START,source_sha256_after=sha(PATH),checks_passed=len(CHECKS),checks=CHECKS,binary_full5CE_gradient_max_difference=gradient_error,
 actual_weights_data_or_probabilities_read=False,gpu_used=False,formal_outputs_modified=False)
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(out),report_sha256=sha(out),auditor_source_sha256=sha(__file__),**r)))
