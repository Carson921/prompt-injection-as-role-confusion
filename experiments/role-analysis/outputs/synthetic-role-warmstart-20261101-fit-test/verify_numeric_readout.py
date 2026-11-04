from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch
sys.path.insert(0,'/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis')
from synthetic_agent_classifier import Probe
from validate_synthetic_role_warmstart_results import ExactTorchProbe

torch.set_num_threads(4)
base=Path('/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/synthetic-role-warmstart-20261101-fit-test')
task=Path('/home/wentian/Github/SPAR/runpod-backups/role-warmstart-20261027-6h-final/task')
old=Path('/home/wentian/Github/SPAR/runpod-backups/role-retraining-20261026-20261027/workspace-task/runs-v7/formal/features/formal')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
published=read(base/'metrics.json');started=time.time()
paths={'parent':task/'parent/best.npz','new':task/'run/best.npz'}
np_probes={k:Probe(p) for k,p in paths.items()};torch_probes={k:ExactTorchProbe(p) for k,p in paths.items()}
for k,p in paths.items():assert sha(p)==published['checkpoint_sha256'][k]
summary={}
for dataset,root,split in [('article_TEST',task/'features','test'),('new_article_FIT',task/'features','fit'),('old_FIT_replay',old,'fit')]:
 replay=np.load(task/'run/replay-indices.npy',allow_pickle=False) if dataset=='old_FIT_replay' else None
 manifest=read(root/'manifest.json');offset=0;rows=0
 counts={k:np.zeros((6,6),np.int64) for k in paths};torch_counts={k:np.zeros((6,6),np.int64) for k in paths}
 max_error={k:0. for k in paths};deltas={k:0 for k in paths};margins={k:0. for k in paths}
 for entry in manifest['splits'][split]:
  folder=(root/entry['path']).resolve();assert folder.is_relative_to(root.resolve())
  assert sha(folder/'complete.json')==published['evidence'][str(folder/'complete.json')]
  n=int(entry['rows']);selected=np.arange(n) if replay is None else replay[(replay>=offset)&(replay<offset+n)]-offset;offset+=n
  if not len(selected):continue
  for filename in ['x.npy','y.npy']:assert sha(folder/filename)==published['evidence'][str(folder/filename)]
  x=np.load(folder/'x.npy',mmap_mode='r',allow_pickle=False);y=np.load(folder/'y.npy',mmap_mode='r',allow_pickle=False)
  assert x.dtype==np.float16 and x.shape==(n,1024)
  for start in range(0,len(selected),8192):
   ix=selected[start:start+8192];features=x[ix];truth=np.asarray(y[ix],dtype=np.int64)
   for name in paths:
    a=np_probes[name].predict_proba(features);b=torch_probes[name].predict(features)
    err=float(np.max(abs(a-b)));max_error[name]=max(max_error[name],err);assert err<5e-4
    pa,pb=a.argmax(1),b.argmax(1);different=pa!=pb;deltas[name]+=int(different.sum())
    if different.any():
     top=np.sort(a[different],axis=1);margin=float(np.max(top[:,-1]-top[:,-2]));margins[name]=max(margins[name],margin);assert margin<=2*err+1e-7
    counts[name]+=np.bincount(truth*6+pa,minlength=36).reshape(6,6)
    torch_counts[name]+=np.bincount(truth*6+pb,minlength=36).reshape(6,6)
   rows+=len(ix)
 assert rows==published['datasets'][dataset]['rows']
 for name,matrix in counts.items():
  actual=published['datasets'][dataset]['models'][name];assert matrix.tolist()==actual['confusion']
  tp=np.diag(matrix);support=matrix.sum(1);predicted=matrix.sum(0)
  assert abs(float(tp.sum()/matrix.sum())-actual['accuracy'])<1e-12
  assert abs(float(np.mean(2*tp/(support+predicted)))-actual['macro_f1'])<1e-12
  for i,role in enumerate(published['roles']):
   assert abs(float(tp[i]/support[i])-actual['roles'][role]['recall'])<1e-12
 summary[dataset]={'rows':rows,'numpy_manual_confusion_matches':True,'torch_maximum_probability_error':max_error,'argmax_disagreements':deltas,'largest_numpy_margin_at_backend_disagreement':margins,'torch_errors':{k:rows-int(np.trace(v)) for k,v in torch_counts.items()},'torch_perfect_separation':{k:bool(np.trace(v)==rows) for k,v in torch_counts.items()}}
 print(json.dumps({'dataset':dataset,**summary[dataset]}),flush=True)
proof={'status':'passed','method':'Independent Torch CPU exact GELU and manually accumulated six-role confusion; full row-wise comparison to NumPy, with bounded numerical-margin policy.','script_sha256':sha(Path(__file__)),'source_metrics_sha256':sha(base/'metrics.json'),'torch_version':torch.__version__,'cpu_only':True,'datasets':summary,'elapsed_seconds':time.time()-started}
(base/'independent-numeric-validation.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','seconds':proof['elapsed_seconds']}),flush=True)
