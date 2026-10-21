"""Independent synthetic-only audit of the frozen complementary inference source.
No actual producer/builder/snapshot/cache/tokenizer/checkpoint/GPU is used.
"""
import ast, copy, hashlib, importlib.util, json, os, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import erf, logsumexp

SOURCE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/systemness_complementary_confirmation_infer.py')
EXPECTED='7915a7e53fe40263aabd5ab202e812be1a65e7a0f0df971d6bf43c4b010c346c'
ROOT=Path(tempfile.mkdtemp(prefix='long-data-complementary-infer-audit-',dir='/tmp'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('independent_toy_complementary_infer',SOURCE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
CHECKS=[];DETAIL={}
def ok(name,condition=True):
    assert bool(condition),name
    CHECKS.append(name)
def refuses(name,fn):
    try:fn()
    except (RuntimeError,ValueError,KeyError,FileNotFoundError,AssertionError) as e:
        CHECKS.append(name);return str(e)
    raise AssertionError('Expected rejection: '+name)
class UnexpectedRead(BaseException):pass
def forbidden(*a,**k):raise UnexpectedRead('Forbidden non-synthetic API or IO')
class Bomb:
    def __getattribute__(self,k):raise UnexpectedRead('Unexpected context access: '+k)

def toy_context(folder='cache',empty=False):
    base=ROOT/folder;out=base/'out';cache=base/'cache';out.mkdir(parents=True,exist_ok=True)
    lengths=[] if empty else [8,4,7,3,10,5]
    tokens=[list(range(100*p,100*p+n)) for p,n in enumerate(lengths)]
    inputs=pd.DataFrame(dict(prompt_ix=np.arange(len(lengths),dtype=np.int64),length=np.asarray(lengths,np.int64)))
    records=[]
    for p,n in enumerate(lengths):
        for pos in [1,n-1]:records.append(dict(prompt_ix=p,token_ix=pos,token_id=tokens[p][pos],cache_ix=len(records),sample_ix=10**13+len(records)))
    rows=pd.DataFrame(records,columns=['prompt_ix','token_ix','token_id','cache_ix','sample_ix']).astype(np.int64)
    snap=dict(physical_model_sha256={name:hashlib.sha256(name.encode()).hexdigest() for name in m.PHYSICAL},checkpoint_path=m.CHECKPOINT)
    return m.Context('natural' if empty else 'c4',None,{},inputs,rows,tokens,snap,{}, {},list(m.closed.CAPTURE),out,cache,dict(toy=True,seed=7041),{})

# Real unknown pin gate must precede both function-level and context-manager IO.
with patch.object(m,'digest',forbidden),patch.object(m,'read',forbidden),patch.object(m,'dataset_module',forbidden),patch.object(m,'no_competing_jobs',forbidden):
    for name,fn in [('load-context',lambda:m.load_context('c4',Bomb())),('portable-load',lambda:m.load_portables(Bomb())),
                    ('completed-p',lambda:m.completed_probabilities('natural',Bomb())),('infer',lambda:m.infer('natural',Bomb())),
                    ('model-context',lambda:m.model_context(Bomb()).__enter__())]:
        refuses('unknown-pins-before-IO:'+name,fn)
# Pure string rejection also precedes process scan/model imports.
with patch.object(m,'actual_gate',lambda:None),patch.object(m,'no_competing_jobs',forbidden):
    refuses('invalid-checkpoint-before-process-or-import',lambda:m.model_context(SimpleNamespace(snapshot={'checkpoint_path':'/workspace/forbidden'})).__enter__())

# Terminal condition precedes builder loading. Only our temporary files exist.
tctx=toy_context('terminal')
module=SimpleNamespace(OUT=tctx.out.parent/'builder')
with patch.object(m,'actual_gate',lambda:None),patch.object(m,'load_context',forbidden):
    refuses('missing-terminal-before-builder',lambda:m.completed_probabilities('c4',module))
    inferout=module.OUT/'inference';inferout.mkdir(parents=True)
    m.atomic_json(inferout/'inference-complete.json',{})
    m.atomic_json(inferout/'status.json',{'stage':'inference'})
    refuses('partial-terminal-before-builder',lambda:m.completed_probabilities('c4',module))

# Independent pre-o_proj head/block projection, with deliberately unsorted rank order.
heads=[2303,2,65,0,127,2048,458,997,2045,247]
collector=m.Collector(heads);positions=np.array([7,1,9],np.int64);inputs={}
for layer in collector.layers:
    inputs[layer]=(torch.arange(2*6*4096,dtype=torch.float32).reshape(2,6,4096)/64+layer).to(torch.bfloat16)
collector.begin(positions)
for layer in reversed(collector.layers):
    before=inputs[layer].clone();assert collector.hook(layer)(None,(inputs[layer],)) is None
    assert torch.equal(inputs[layer],before)
collector.finish()
expected=np.stack([inputs[h//64].float().numpy().reshape(-1,64,64)[positions,h%64].astype(np.float16) for h in heads],axis=1).reshape(3,640)
ok('unsorted-ten-head-sample-rank-blocks-bitexact',np.array_equal(collector.features.view(np.uint16),expected.view(np.uint16)))
collector.begin(positions,enabled=False)
for layer in collector.layers:
    before=inputs[layer].clone();collector.hook(layer)(None,(inputs[layer],));assert torch.equal(before,inputs[layer])
collector.finish();ok('disabled-hooks-never-write-or-change-input',collector.features is None)
collector.begin(positions);collector.hook(collector.layers[0])(None,(inputs[collector.layers[0]],))
refuses('missing-hook-layer-rejected',collector.finish)
refuses('duplicate-hook-layer-rejected',lambda:collector.hook(collector.layers[0])(None,(inputs[collector.layers[0]],)))
for bad,name in [(float('nan'),'nan'),(70000.,'finite-f16-overflow')]:
    collector.begin(positions);layer=collector.layers[0];x=inputs[layer].float().clone();x.reshape(-1,64,64)[positions[0],heads[np.flatnonzero(np.asarray(heads)//64==layer)[0]]%64,0]=bad
    refuses(name+'-activation-rejected',lambda:m.Collector.hook(collector,layer)(None,(x,)))
removed=[]
class Projection:
    def __init__(self,fail=False):self.fail=fail
    def register_forward_pre_hook(self,hook):
        if self.fail:raise ValueError('synthetic registration failure')
        return SimpleNamespace(remove=lambda:removed.append(True))
firstlayer=collector.layers[0];layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=Projection(i==collector.layers[1]))) for i in range(36)]
refuses('partial-hook-registration-cleans-handles',lambda:m.install_hooks(SimpleNamespace(model=SimpleNamespace(layers=layers)),collector))
ok('partial-hook-previous-handle-removed',removed==[True])

# Synthetic NPZ models and independent F32/F64 mathematics. No real model path.
rng=np.random.default_rng(70821)
def arr(shape):return rng.normal(0,.025,size=shape).astype(np.float32)
old={'encoder.0.weight':arr((128,512)),'encoder.0.bias':arr((128,)),'head.weight':arr((5,128)),'head.bias':arr((5,))}
mean=arr((512,));scale=np.exp(arr((512,))).astype(np.float32)
states={'old-head':old};means={'old-head':mean};scales={'old-head':scale}
for name in ['new','matched']:
    states[name]={**{'base.'+k:v.copy() for k,v in old.items()},'correction.0.weight':arr((32,640)),'correction.0.bias':arr((32,)),'correction.3.weight':arr((1,32)),'correction.3.bias':arr((1,))}
    means[name]=np.r_[mean,arr((128,))] if name=='new' else np.r_[mean,mean[:128]]
    scales[name]=np.r_[scale,np.exp(arr((128,))).astype(np.float32)] if name=='new' else np.r_[scale,scale[:128]]
ctx=toy_context('models');ctx.paths={};ctx.snapshot['portable_metadata']={}
def write_model(name,state=None,mu=None,sig=None,meta_update=None):
    state=states[name] if state is None else state;mu=means[name] if mu is None else mu;sig=scales[name] if sig is None else sig
    view=m.closed.VIEWS[name]
    meta=dict(roles=list(m.closed.ROLES),selected_head_ids=list(view['selected_head_ids']),kind=view['kind'],dimensions=view['dimensions'])
    if name!='old-head':meta.update(base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',base_model_sha256=m.closed.OLD_MODEL_SHA)
    if meta_update:meta.update(meta_update)
    path=ROOT/'models'/(name+'.npz');path.parent.mkdir(exist_ok=True)
    np.savez(path,metadata=json.dumps(meta),mean=mu,scale=sig,**{'weight::'+k:v for k,v in state.items()})
    ctx.paths[name]=path;ctx.snapshot['physical_model_sha256'][name]=sha(path)
    ctx.snapshot['portable_metadata'][name]=dict(metadata_sha256=m.closed.canonical(meta),mean_sha256=ah(mu),scale_sha256=ah(sig),weight_sha256={k:ah(v) for k,v in state.items()})
for name in m.PHYSICAL:write_model(name)
with patch.object(m,'actual_gate',lambda:None):nets=m.load_portables(ctx)
ok('three-independent-frozen-portable-identities',set(nets)==set(m.PHYSICAL) and all(not net.training and not any(p.requires_grad for p in net.parameters()) for net,_,_ in nets.values()))
features=(rng.normal(size=(11,640))*3).astype(np.float16)
other=features.copy();other[:,512:]=np.float16(100)
oldview=features[:,:512].copy();dupview=np.c_[oldview,oldview[:,:128]]
ok('dup640-view-repeats-first128-and-ignores-extra',np.array_equal(m.feature_view('matched',features),dupview) and np.array_equal(m.feature_view('matched',features),m.feature_view('matched',other)))
ok('old512-view-ignores-extra-and-is-contiguous',m.feature_view('old-head',features).flags.c_contiguous and np.array_equal(m.feature_view('old-head',features),oldview))
seencont=[]
hooks=[]
for name,(net,_,_) in nets.items():
    base=net if name=='old-head' else net.base
    hooks.append(base.encoder[0].register_forward_pre_hook(lambda mod,args:seencont.append((args[0].shape[1],args[0].is_contiguous()))))
    if name!='old-head':hooks.append(net.correction[0].register_forward_pre_hook(lambda mod,args:seencont.append((args[0].shape[1],args[0].is_contiguous()))))
got=m.cpu_probabilities(nets,features)
for h in hooks:h.remove()
ok('base512-and-correction640-are-contiguous',all(cont for _,cont in seencont) and {dim for dim,_ in seencont}=={512,640})
f64_errors={};f32_errors={}
for name in m.PHYSICAL:
    rawfeature=features.copy() if name=='new' else (dupview if name=='matched' else oldview)
    standardized=(torch.tensor(rawfeature,dtype=torch.float32)-torch.tensor(means[name]))/torch.tensor(scales[name]);standardized=standardized.contiguous()
    st=states[name];prefix='' if name=='old-head' else 'base.'
    def linear(x,key):return F.linear(x,torch.tensor(st[key+'.weight']),torch.tensor(st[key+'.bias']))
    z=linear(F.gelu(linear(standardized[:,:512].contiguous(),prefix+'encoder.0')),prefix+'head')
    base=z.clone()
    if name!='old-head':
        delta=4*torch.tanh(linear(F.gelu(linear(standardized.contiguous(),'correction.0')),'correction.3'))
        z[:,0]+=delta[:,0]
        assert torch.equal(z[:,1:].view(torch.int32),base[:,1:].contiguous().view(torch.int32)) and bool((delta.abs()<=4).all())
    want=torch.exp(torch.log_softmax(z,1)).numpy();f32_errors[name]=float(np.abs(got[name]-want).max())
    x=(rawfeature.astype(np.float64)-means[name].astype(np.float64))/scales[name].astype(np.float64)
    def dense64(t,key):return t@st[key+'.weight'].astype(np.float64).T+st[key+'.bias'].astype(np.float64)
    def gelu64(t):return .5*t*(1+erf(t/np.sqrt(2)))
    z64=dense64(gelu64(dense64(x[:,:512],prefix+'encoder.0')),prefix+'head')
    if name!='old-head':z64[:,0]+=4*np.tanh(dense64(gelu64(dense64(x,'correction.0')),'correction.3')[:,0])
    p64=np.exp(z64-logsumexp(z64,axis=1)[:,None]);f64_errors[name]=float(np.abs(got[name]-p64).max())
ok('independent-functional-F32-portable-bitexact',all(v==0 for v in f32_errors.values()))
ok('independent-numpy-scipy-F64-probabilities',max(f64_errors.values())<2e-6)
DETAIL['independent_F32_max_difference']=f32_errors;DETAIL['independent_F64_max_difference']=f64_errors
aliases=m.records_from_physical(got)
ok('old-two-threshold-records-share-identical-array',aliases['oldhead-max33'] is aliases['oldhead-original'] and aliases['oldhead-max33'] is got['old-head'])
# Rehashing altered fake NPZ/proof cannot bypass semantic identity checks.
bad=copy.deepcopy(states['new']);bad['base.head.bias'][2]+=np.float32(.1);write_model('new',state=bad)
with patch.object(m,'actual_gate',lambda:None):refuses('altered-embedded-base-even-with-updated-byte-hashes',lambda:m.load_portables(ctx))
write_model('new');badscale=scales['matched'].copy();badscale[512]+=np.float32(.125);write_model('matched',sig=badscale)
with patch.object(m,'actual_gate',lambda:None):refuses('duplicate-tail-scaler-semantic-tamper-rejected',lambda:m.load_portables(ctx))
write_model('matched');write_model('new',meta_update={'selected_head_ids':list(reversed(m.closed.CAPTURE))})
with patch.object(m,'actual_gate',lambda:None):refuses('wrong-portable-head-order-even-with-updated-proof',lambda:m.load_portables(ctx))
write_model('new')

# Leftpad and canonical-row prefix are independently reconstructed.
pctx=toy_context('probabilities');parts=m.batches(pctx)
ok('deterministic-length-prompt-batches',parts==[[3,1,5,2],[0,4]])
for prompts in parts:
    batch=m.collate(pctx,prompts);width=max(len(pctx.tokens[p]) for p in prompts);epos=[];etarget=[]
    for i,p in enumerate(prompts):
        gap=width-len(pctx.tokens[p]);assert batch['input_ids'][i].tolist()==[0]*gap+pctx.tokens[p]
        assert batch['attention_mask'][i].tolist()==[0]*gap+[1]*len(pctx.tokens[p])
        for ri,row in pctx.rows[pctx.rows.prompt_ix==p].iterrows():epos.append(i*width+gap+int(row.token_ix));etarget.append(ri)
    assert batch['positions'].tolist()==epos and batch['targets'].tolist()==etarget
ok('leftpad-target-and-cache-scatter-independent')
indices=np.asarray([i for i,r in pctx.rows.iterrows() if r.prompt_ix in set(parts[0])],np.int64)
ok('prefix-is-canonical-row-order-not-forward-order',np.array_equal(indices,m.completed_indices(pctx,1)) and not np.array_equal(indices,m.collate(pctx,parts[0])['targets']))
values=m.initialize_probabilities(pctx)
ok('owner-creates-exact-three-files-all-columns-NaN',all(v.shape==(12,5) and v.dtype==np.float32 and np.isnan(v).all() and (pctx.cache/(n+'-probabilities.f32')).stat().st_size==240 for n,v in values.items()))
ok('new-NaN-cache-resumes-only-zero',m.restore_progress(pctx,values)==0)
# Incomplete uncommitted writes do not advance prefix.
values['new'][indices[0]]=np.float32(.2)
ok('uncommitted-partial-write-never-advances',m.restore_progress(pctx,values)==0)
for name in m.PHYSICAL:
    r=rng.uniform(.1,1,size=(len(indices),5));r=(r/r.sum(1,keepdims=True)).astype(np.float32);values[name][indices]=r
m.checkpoint(pctx,values,1)
ok('committed-prefix-one-restores',m.restore_progress(pctx,values)==1)
saved=json.loads((pctx.cache/'progress.json').read_text())
indsha={n:hashlib.sha256(np.ascontiguousarray(values[n][indices]).tobytes()).hexdigest() for n in m.PHYSICAL}
ok('prefix-hash-covers-all-five-columns-and-sample-ids',saved['completed_probability_sha256']==indsha and saved['sample_ix_sha256']==ah(pctx.rows.iloc[indices].sample_ix.to_numpy()) and saved['cache_ix_sha256']==ah(indices))
rest=np.setdiff1d(np.arange(len(pctx.rows)),indices);values['old-head'][rest]=np.float32(.2)
ok('uncommitted-tail-finite-bytes-not-falsely-completed',m.restore_progress(pctx,values)==1)
oldrow=values['new'][indices[0]].copy();values['new'][indices[0],0]-=np.float32(.001);values['new'][indices[0],4]+=np.float32(.001)
refuses('normalized-fifth-column-committed-byte-tamper',lambda:m.restore_progress(pctx,values));values['new'][indices[0]]=oldrow
savedrows=pctx.rows.copy();pctx.rows.loc[indices[0],'sample_ix']+=1
refuses('sample-identity-tamper-rejected',lambda:m.restore_progress(pctx,values));pctx.rows=savedrows
ownerpath=pctx.cache/'initialized.json';owner=json.loads(ownerpath.read_text());badowner=copy.deepcopy(owner);badowner['signature']={'wrong':True};m.atomic_json(ownerpath,badowner)
refuses('wrong-owner-refuses-resume',lambda:m.initialize_probabilities(pctx));m.atomic_json(ownerpath,owner)
missing=pctx.cache/'matched-probabilities.f32';savedbytes=missing.read_bytes();missing.unlink()
refuses('missing-owned-file-refused-not-recreated',lambda:m.initialize_probabilities(pctx));assert not missing.exists();missing.write_bytes(savedbytes[:-4])
refuses('truncated-owned-file-refused-before-mapping',lambda:m.initialize_probabilities(pctx));missing.write_bytes(savedbytes)
# Close old mapping then reopen the restored fake files; no real cache is touched.
del values;values=m.initialize_probabilities(pctx)
# Complete all rows, then test complete/read-only/proof chain using synthetic evidence.
for name in m.PHYSICAL:
    block=rng.uniform(.1,1,size=(len(pctx.rows),5));values[name][:]=(block/block.sum(1,keepdims=True)).astype(np.float32)
m.checkpoint(pctx,values,len(parts))
first=m.collate(pctx,parts[0]);count=min(32,len(first['targets']));dm={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}}
portable=dict(signature=pctx.signature,model_sha256=pctx.snapshot['physical_model_sha256'],sampled_tokens=count,tolerance=5e-4,max_probability_difference={n:0. for n in m.PHYSICAL},same_vectors=True,captured_dtype='float16',selected_head_ids=pctx.heads,physical_feature_views=copy.deepcopy(m.closed.VIEWS),first_original_batch=True,no_features_persisted=True,CPU_outputs_are_saved=True,target_cache_ix_sha256=ah(first['targets'][:count]),sample_ix_sha256=ah(pctx.rows.iloc[first['targets'][:count]].sample_ix.to_numpy()),feature_sha256=hashlib.sha256(b'synthetic-no-actual-vector').hexdigest())
hook=dict(signature=pctx.signature,hidden_states_bitexact=True,all_capture_writes_disabled=True,only_pre_o_proj_hooks=True,legacy_capture=False,selected_head_ids=pctx.heads,dimensions=640,sample_rows=len(first['targets']),sample_ix_sha256=ah(pctx.rows.iloc[first['targets']].sample_ix.to_numpy()),input_ids_sha256=ah(first['input_ids'].numpy()),attention_mask_sha256=ah(first['attention_mask'].numpy()))
backend=dict(signature=pctx.signature,expert_backends=['Mxfp4GptOssExperts'],checkpoint_path=m.CHECKPOINT,native_mxfp4=True,parameters_frozen=True,device_map=dm,gpus=['synthetic A40']*4,compute_capabilities=[[8,6]]*4,gpu_transfer='Synchronous CPU staging between different CUDA devices')
for filename,data in [('portable-validation.json',portable),('hook-validation.json',hook),('model-backend.json',backend)]:m.atomic_json(pctx.out/filename,data)
m.verify_evidence(pctx);ok('synthetic-first-original-batch-evidence-scope')
bad=copy.deepcopy(portable);bad['target_cache_ix_sha256']=ah(m.collate(pctx,parts[1])['targets']);m.atomic_json(pctx.out/'portable-validation.json',bad)
refuses('resume-proof-for-later-batch-rejected',lambda:m.verify_evidence(pctx));m.atomic_json(pctx.out/'portable-validation.json',portable)
(pctx.out/'model-backend.json').rename(pctx.out/'wrong-backend-name.json')
refuses('backend-evidence-name-exact',lambda:m.verify_evidence(pctx));(pctx.out/'wrong-backend-name.json').rename(pctx.out/'model-backend.json')
with patch.object(m,'actual_gate',lambda:None),patch.object(m,'fast_integrity',lambda ctx:None):
    complete=m.finish_completion(pctx,values,0)
    _,readonly=m.completed_probabilities('c4',context=pctx)
ok('synthetic-completion-readonly-all-three-cache-hashes',all(v.mode=='r' for v in readonly.values()) and complete['probability_sha256']=={n:sha(pctx.cache/(n+'-probabilities.f32')) for n in m.PHYSICAL})
ok('four-record-completion-alias-hashes',complete['record_probability_sha256']['oldhead-original']==complete['record_probability_sha256']['oldhead-max33'] and complete['record_argmax_sha256']['oldhead-original']==complete['record_argmax_sha256']['oldhead-max33'])
empty=toy_context('empty',empty=True)
with patch.object(m,'actual_gate',lambda:None),patch.object(m,'no_competing_jobs',lambda:None),patch.object(m,'load_context',lambda *a,**k:empty),patch.object(m,'fast_integrity',lambda ctx:None),patch.object(m,'model_context',forbidden),patch.object(m,'load_portables',forbidden):
    m.infer('natural');ec,ep=m.completed_probabilities('natural',context=empty)
ok('all-failed-natural-no-LM-model-probability-files',ec['no_observations'] is True and ec['gpu_run'] is False and ec['model_forward_run'] is False and all(v.shape==(0,5) for v in ep.values()) and not empty.cache.exists())

# Execute an AST-isolated exact model_context with fake third-party imports.
# This exercises finally restoration without importing transformers, touching a
# checkpoint, calling CUDA, or mutating any frozen producer's module globals.
tree=ast.parse(SOURCE.read_text());fn=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='model_context')
class FakeDevice:
    def __init__(self,x):self.value=str(x);self.type=self.value.split(':')[0]
    def __eq__(self,o):return isinstance(o,FakeDevice) and o.value==self.value
class FakeTensor:
    def __init__(self,where='cuda:0',events=None):self.device=FakeDevice(where);self.events=[] if events is None else events
    def cpu(self):self.events.append('cpu');return FakeTensor('cpu',self.events)
    def to(self,target,non_blocking=False):self.events.append(('to',target.value,non_blocking));return FakeTensor(target.value,self.events)
class Mxfp4GptOssExperts:pass
for failwhen in ['load','inside-with']:
    events=[];oldget=lambda *a,**k:('kernel',a,k);oldvalidate=lambda *a,**k:events.append('validate');oldsend=lambda *a,**k:('oldsend',a,k);oldhook=lambda *a,**k:None
    hub=SimpleNamespace(get_kernel=oldget)
    class Quant:validate_environment=oldvalidate
    ops=SimpleNamespace(send_to_device=oldsend);hooks=SimpleNamespace(send_to_device=oldhook)
    acc=SimpleNamespace(hooks=hooks,utils=SimpleNamespace(operations=ops))
    fakecuda=SimpleNamespace(device_count=lambda:4,get_device_name=lambda i:'synthetic A40',mem_get_info=lambda i:(32*2**30,48*2**30),get_device_capability=lambda i:(8,6))
    faketorch=SimpleNamespace(cuda=fakecuda,Tensor=FakeTensor,device=FakeDevice,bfloat16='synthetic-bfloat16')
    fake_model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=Mxfp4GptOssExperts())) for _ in range(36)]),hf_quantizer=SimpleNamespace(quantization_config=SimpleNamespace(dequantize=False)),hf_device_map=dm,parameters=lambda:iter([SimpleNamespace(requires_grad=False)]))
    fake_model.eval=lambda:fake_model;fake_model.requires_grad_=lambda v:fake_model
    def pretrained(*args,**kwargs):
        events.append(('load',args,kwargs))
        assert args==(m.CHECKPOINT,) and kwargs['local_files_only'] is True and kwargs['attn_implementation']=='eager' and kwargs['use_kernels'] is False and not kwargs['quantization_config'].dequantize and kwargs['device_map']==dm
        if failwhen=='load':raise ValueError('synthetic load failure')
        return fake_model
    transformers=SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=pretrained),Mxfp4Config=lambda **k:SimpleNamespace(**k))
    modules={'torch':faketorch,'transformers':transformers,'transformers.integrations':SimpleNamespace(hub_kernels=hub),'transformers.quantizers.quantizer_mxfp4':SimpleNamespace(Mxfp4HfQuantizer=Quant),'accelerate.hooks':acc,'accelerate.utils.operations':acc}
    def fakeimport(name,globals=None,locals=None,fromlist=(),level=0):
        assert name in modules,'Unexpected import '+name
        return modules[name]
    built=vars(__import__('builtins')).copy();built['__import__']=fakeimport
    saved=[];ns=dict(__builtins__=built,contextmanager=contextmanager,actual_gate=lambda:None,require_checkpoint_path=m.require_checkpoint_path,no_competing_jobs=lambda:None,require=m.require,CHECKPOINT=m.CHECKPOINT,immutable_json=lambda path,value:saved.append((path,value)),gc=SimpleNamespace(collect=lambda:None))
    exec(compile(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[]),str(SOURCE),'exec'),ns)
    c=SimpleNamespace(snapshot={'checkpoint_path':m.CHECKPOINT,'kernel_revision':'synthetic-fixed-rev'},out=ROOT/'fake-backend',signature={'synthetic':True})
    try:
        with ns['model_context'](c):
            assert failwhen=='inside-with'
            assert hub.get_kernel('kernels-community/gpt-oss-triton-kernels')[2]['revision']=='synthetic-fixed-rev'
            Quant.validate_environment(SimpleNamespace(quantization_config=SimpleNamespace(dequantize=False)))
            tensor=FakeTensor();staged=ops.send_to_device(tensor,1,non_blocking=True)
            assert tensor.events==['cpu',('to','cuda:1',False)] and staged.device.value=='cuda:1'
            assert hooks.send_to_device is ops.send_to_device
            raise ValueError('synthetic caller failure')
    except ValueError:pass
    assert hub.get_kernel is oldget and Quant.validate_environment is oldvalidate and ops.send_to_device is oldsend and hooks.send_to_device is oldhook
    if failwhen=='inside-with':assert saved[0][0].name=='model-backend.json' and saved[0][1]['native_mxfp4'] is True
ok('own-context-restores-all-four-third-party-functions-on-load-and-body-errors')
# Structural assurance: no imports or assignments to frozen runtime/probe modules.
imports=[x.module or '' for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]+[a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
ok('no-old-runtime-predictor-or-verifier-import',not any(('feature_pipeline_runtime' in s or 'systemness_head_extract'==s or 'systemness_complementary_head_readout'==s or 'systemness_aug_confirmation_infer'==s) for s in imports))
cp=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='checkpoint')
loop=cp.body[0];commit=cp.body[1]
ok('checkpoint-fsync-loop-precedes-atomic-progress-commit',isinstance(loop,ast.For) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='flush' for x in ast.walk(loop)) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='fsync' for x in ast.walk(loop)) and isinstance(commit,ast.Expr) and isinstance(commit.value,ast.Call) and commit.value.func.id=='atomic_json')
assert sha(SOURCE)==EXPECTED
report=dict(audit_passed=True,scope='Source plus synthetic CPU/tmp only; no actual builder, snapshot, cached source, IDs/body, tokenizer, checkpoint, portable model, probability, bank, LM or CUDA access.',source_path=str(SOURCE),source_sha256=EXPECTED,source_sha256_end=sha(SOURCE),checks=len(CHECKS),check_names=CHECKS,details=DETAIL,actual_gpu_hook_or_portable_parity_tested=False,actual_data_or_probabilities_read=False,real_closed_pins_remain_unknown=True,output=str(ROOT))
path=ROOT/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(report=str(path),report_sha256=sha(path),script_sha256=sha(__file__),checks=len(CHECKS),source_sha256=EXPECTED),indent=2))
