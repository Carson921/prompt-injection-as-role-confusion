from pathlib import Path
import hashlib,json,struct,time
import numpy as np
import torch

started=time.time();torch.set_num_threads(4)
repo=Path('/home/wentian/Github/SPAR/prompt-injection-as-role-confusion')
base=repo/'experiments/role-analysis/outputs/agentdojo-system-swap-57-scratch11-20261102'
private=Path('/home/wentian/Github/SPAR/runpod-backups/agentdojo-gpt-oss-120b-system-swapped-57-20261102')
root=private/'inputs';checkpoint=repo/'experiments/role-analysis/outputs/synthetic-role-warmstart-20261027-6h/checkpoints/best.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=json.loads((base/'metrics.json').read_text())
assert sha(checkpoint)==result['checkpoint_sha256']=='45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df'
with np.load(checkpoint,allow_pickle=False) as a:
 metadata=json.loads(str(a['metadata'].item()));mean=torch.from_numpy(a['mean'].copy());scale=torch.from_numpy(a['scale'].copy());w={k[8:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('weight::')}
assert metadata['hidden']==128 and metadata['activation']=='gelu_exact'
x=np.load(root/'X.npy',mmap_mode='r',allow_pickle=False);published=np.load(private/'readout/probabilities.npy',allow_pickle=False)
with (root/'system_states.safetensors').open('rb') as stream:
 size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size));f=header['features'];stream.seek(8+size+f['data_offsets'][0]);raw=bytearray(stream.read(f['data_offsets'][1]-f['data_offsets'][0]))
assert f['dtype']=='BF16' and f['shape']==[len(x),1024]
decoded=torch.frombuffer(raw,dtype=torch.bfloat16).reshape(len(x),1024).float().numpy();assert np.array_equal(decoded,x)
p=[]
with torch.inference_mode():
 for start in range(0,len(x),2048):
  z=(torch.tensor(np.asarray(x[start:start+2048]),dtype=torch.float32)-mean)/scale
  z=torch.nn.functional.linear(z,w['0.weight'],w['0.bias']);z=torch.nn.functional.gelu(z,approximate='none')
  z=torch.nn.functional.linear(z,w['3.weight'],w['3.bias']);p.append(torch.softmax(z,dim=1).numpy())
p=np.concatenate(p);maximum=float(np.max(abs(p-published)));delta=int(np.sum(p.argmax(1)!=published.argmax(1)))
assert maximum<5e-4 and delta==0
roles=tuple(result['scopes']['initial_calls']['channel_rules']['mean_probability'])
assert roles==('system','developer','user','cot','assistant','tool')
groups=np.load(private/'readout/segment_ids.npy',allow_pickle=False);initial=np.load(root/'initial_call_mask.npy',allow_pickle=False)
max_mean_difference=0.
for scope,mask in [('initial_calls',initial),('all_calls',np.ones(len(x),bool))]:
 for i,segment in enumerate(result['segment_order']):
  selection=mask&(groups==i);row=result['scopes'][scope][segment];assert selection.sum()==row['tokens']
  means=published[selection].mean(0,dtype=np.float64);torch_means=p[selection].mean(0,dtype=np.float64);counts=np.bincount(published[selection].argmax(1),minlength=6)
  for j,role in enumerate(roles):
   assert abs(means[j]-row['mean_probability'][role])<1e-12
   assert int(counts[j])==row['argmax_counts'][role]
   assert abs(counts[j]/selection.sum()-row['argmax_fraction'][role])<1e-12
  max_mean_difference=max(max_mean_difference,float(np.max(abs(means-torch_means))))
proof={'status':'passed','verification':'Independent Torch CPU exact-GELU readout and BF16 safetensors decoding; manual segment mean/count recomputation',
 'script_sha256':sha(Path(__file__)),'source_metrics_sha256':sha(base/'metrics.json'),'checkpoint_sha256':sha(checkpoint),
 'rows':len(x),'maximum_probability_error':maximum,'argmax_disagreements':delta,'maximum_segment_mean_difference':max_mean_difference,
 'all_raw_BF16_values_match_X':True,'initial_and_all_call_segment_metrics_match':True,'torch_version':torch.__version__,'seconds':time.time()-started}
(base/'independent-numeric-validation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
