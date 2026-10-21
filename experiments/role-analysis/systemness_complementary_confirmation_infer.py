"""Shared native640 probability-only confirmation inference.

Default help; --preflight uses synthetic CPU/tmp only. Actual public entrypoints
first require a closed independently audited promoted readout snapshot. Ten
ranked pre-o_proj heads are transient F16, consumed by new640, duplicated640
and original512 views/scalers. Four fixed records share three F32 probability
files. No source/model/threshold selection, persisted hidden bank or report.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
import systemness_complementary_confirm_snapshot as closed

HERE=Path(__file__).resolve().parent
BATCH,INTERVAL,DIMENSIONS,PAD_ID=4,16,640,0
CHECKPOINT='/dev/shm/role-analysis-gptoss120b'
PHYSICAL=['new','matched','old-head']
RECORD_TO_PHYSICAL=dict(closed.RECORD_TO_MODEL)
BUILDERS={'c4':'systemness_complementary_confirmation_data','natural':'systemness_complementary_natural_data'}
require,read,digest=closed.require,closed.read,closed.digest


def protocol():
    return dict(version=1,batch_size=BATCH,checkpoint_interval=INTERVAL,capture_dimensions=640,
        capture='Exactly10 unique ranked pre-o_proj heads, transient F16 perbatch; no legacy capture or persisted hidden vectors.',
        feature_views=copy.deepcopy(closed.VIEWS),physical_models=PHYSICAL,record_to_physical=RECORD_TO_PHYSICAL,
        residual='Only raw z_system +=4*tanh(g640), base512 contiguous and correction640 contiguous; eval/no-grad.',
        backend='Native MXFP4 frozen120B, exactly4 A40,36layers9/card,eager,use_cacheFalse,synchronousCPU staging between different CUDA devices.',
        parity='First original input batch: hidden bitexact with all capture writes disabled and actual same-F16 CPU/GPU3consumer probability error<5e-4.',
        padding='Left0 with explicit attention mask and true perprompt token positions.',
        no_features_persisted=True,no_model_or_threshold_selection=True,empty_natural_allowed=True)


def actual_gate():closed.pin_gate()


def immutable_json(path,value):
    if Path(path).exists():require(read(path)==value,'Frozen inference evidence changed: '+str(path))
    else:atomic_json(path,value)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_probabilities(values, indices=None):
    for name, value in values.items():
        require(name in PHYSICAL and value.ndim == 2 and value.shape[1] == 5 and value.dtype == np.float32, 'Invalid probability geometry/dtype.')
        for start in range(0, len(value) if indices is None else len(indices), 65536):
            block = value[start:start + 65536] if indices is None else value[indices[start:start + 65536]]
            require(np.isfinite(block).all() and (block >= 0).all() and (block <= 1.00001).all()
                and np.allclose(block.sum(1), 1., atol=2e-5, rtol=0), 'Nonfinite or unnormalized probability rows.')


def records_from_physical(values):
    require(set(values) == set(PHYSICAL), 'Exactly three physical probability arrays are required.')
    return {record: values[physical] for record, physical in RECORD_TO_PHYSICAL.items()}


@dataclass
class Context:
    dataset: str
    module: object
    metadata: dict
    inputs: object
    rows: object
    tokens: list
    snapshot: dict
    paths: dict
    thresholds: dict
    heads: list
    out: Path
    cache: Path
    signature: dict
    bindings: dict


def batches(ctx):
    order = ctx.inputs.sort_values(['length', 'prompt_ix']).prompt_ix.to_numpy(dtype=np.int64)
    return [order[i:i + BATCH].tolist() for i in range(0, len(order), BATCH)]


def collate(ctx, prompts):
    import torch
    width = max(len(ctx.tokens[p]) for p in prompts)
    ids = torch.full((len(prompts), width), PAD_ID, dtype=torch.long); mask = torch.zeros_like(ids)
    positions, targets = [], []
    lookup = getattr(ctx, '_lookup', None)
    if lookup is None:
        lookup = {int(p): g for p, g in ctx.rows.groupby('prompt_ix', sort=False)}; ctx._lookup = lookup
    for batch_ix, prompt in enumerate(prompts):
        tokens = ctx.tokens[prompt]; offset = width - len(tokens)
        ids[batch_ix, offset:] = torch.as_tensor(tokens, dtype=torch.long); mask[batch_ix, offset:] = 1
        group = lookup[prompt]; positions.extend((group.token_ix.to_numpy() + offset + batch_ix * width).tolist()); targets.extend(group.index.tolist())
    require(len(set(positions)) == len(positions) and len(set(targets)) == len(targets), 'Duplicate content targets in shared forward.')
    return dict(input_ids=ids, attention_mask=mask, positions=np.asarray(positions, np.int64), targets=np.asarray(targets, np.int64))


def completed_indices(ctx, next_batch):
    parts = batches(ctx); require(type(next_batch) is int and 0 <= next_batch <= len(parts), 'Invalid saved batch cursor.')
    prompts = [p for group in parts[:next_batch] for p in group]
    return np.flatnonzero(ctx.rows.prompt_ix.isin(prompts).to_numpy())


def probability_hashes(values, indices):
    output = {}
    for name in PHYSICAL:
        h = hashlib.sha256()
        for start in range(0, len(indices), 65536): h.update(np.ascontiguousarray(values[name][indices[start:start + 65536]]).tobytes())
        output[name] = h.hexdigest()
    return output


def progress_record(ctx, values, next_batch):
    indices = completed_indices(ctx, next_batch); validate_probabilities(values, indices)
    return dict(signature=ctx.signature, next_batch=next_batch, rows_written=len(indices),
        sample_ix_sha256=array_sha(ctx.rows.iloc[indices].sample_ix.to_numpy()),
        cache_ix_sha256=array_sha(ctx.rows.iloc[indices].cache_ix.to_numpy()),
        completed_probability_sha256=probability_hashes(values, indices))


def initialize_probabilities(ctx, *, readonly=False):
    require(len(ctx.rows) > 0, 'Empty datasets have no probability memmap.')
    expected = dict(signature=ctx.signature, models=PHYSICAL, rows=len(ctx.rows), classes=5, dtype='float32')
    owner = ctx.cache / 'initialized.json'; paths = {name: ctx.cache / (name + '-probabilities.f32') for name in PHYSICAL}
    if owner.exists():
        require(read(owner) == expected and (ctx.cache / 'progress.json').is_file(), 'Probability ownership or committed progress is missing/changed.')
    else:
        require(not readonly and not (ctx.cache / 'progress.json').exists() and not any(p.exists() for p in paths.values()), 'Unowned/partial probability cache; refusing silent reuse.')
        ctx.cache.mkdir(parents=True, exist_ok=True)
        for path in paths.values():
            array = np.memmap(path, mode='w+', dtype=np.float32, shape=(len(ctx.rows), 5)); array[:] = np.nan; array.flush(); del array
            with path.open('rb') as stream: os.fsync(stream.fileno())
        atomic_json(owner, expected)
        initial = {name: np.memmap(path, mode='r+', dtype=np.float32, shape=(len(ctx.rows), 5)) for name, path in paths.items()}
        atomic_json(ctx.cache / 'progress.json', progress_record(ctx, initial, 0)); del initial
    require(all(p.is_file() and p.stat().st_size == len(ctx.rows) * 20 for p in paths.values()), 'Saved probability file is missing/truncated; refusing old progress.')
    return {name: np.memmap(path, mode='r' if readonly else 'r+', dtype=np.float32, shape=(len(ctx.rows), 5)) for name, path in paths.items()}


def restore_progress(ctx, values):
    saved = read(ctx.cache / 'progress.json'); expected = progress_record(ctx, values, saved['next_batch'])
    require(saved == expected, 'Committed probability/sample/cache prefix changed; refusing resume.')
    return saved['next_batch']


def checkpoint(ctx, values, next_batch):
    for name, value in values.items():
        value.flush()
        with (ctx.cache / (name + '-probabilities.f32')).open('rb') as stream: os.fsync(stream.fileno())
    atomic_json(ctx.cache / 'progress.json', progress_record(ctx, values, next_batch))


@contextmanager
def run_lock(ctx):
    import fcntl
    with (ctx.out / '.run.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('This inference namespace is locked by another process.') from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def record_identity(physical_sha, argmax_sha):
    return dict(record_probability_sha256={name: physical_sha[physical] for name, physical in RECORD_TO_PHYSICAL.items()},
        record_argmax_sha256={name: argmax_sha[physical] for name, physical in RECORD_TO_PHYSICAL.items()},
        record_to_physical=RECORD_TO_PHYSICAL, old_head_two_records_share_probability_and_argmax=True)


def argmax_hashes(values):
    output = {}
    for name, value in values.items():
        h = hashlib.sha256()
        for start in range(0, len(value), 65536): h.update(value[start:start + 65536].argmax(1).astype(np.int64).tobytes())
        output[name] = h.hexdigest()
    return output


def empty_completion(ctx):
    require(ctx.dataset == 'natural' and not len(ctx.inputs) and not len(ctx.rows), 'Only all-failed natural data may have no observations.')
    require(not any((ctx.cache / name).exists() for name in ['initialized.json', 'progress.json'] + [p + '-probabilities.f32' for p in PHYSICAL]), 'A no-observation inference cannot reuse nonempty cache files.')
    empty = {name: None for name in PHYSICAL}
    complete = dict(signature=ctx.signature, rows=0, inputs=0, no_observations=True, gpu_run=False,
        probability_sha256=empty, argmax_sha256=empty, **record_identity(empty, empty),
        all_finite=True, all_normalized=True, no_features_persisted=True, model_forward_run=False,
        portable_validation_sha256=None, hook_validation_sha256=None, model_backend_sha256=None, progress_sha256=None)
    atomic_json(ctx.out / 'inference-complete.json', complete)
    atomic_json(ctx.out / 'status.json', dict(stage='inference_complete', rows=0, no_observations=True, gpu_run=False))
    return complete


def dataset_module(dataset,module=None):
    require(dataset in BUILDERS,'Unknown confirmation dataset.')
    return module if module is not None else importlib.import_module(BUILDERS[dataset])


def load_context(dataset,module=None):
    actual_gate()
    module=dataset_module(dataset,module);metadata,inputs,rows,tokens=module.load_verified_data()
    snapshot=metadata['frozen'];paths,thresholds,heads=closed.verify_closed_snapshot(snapshot)
    require(heads==closed.CAPTURE and len(set(heads))==10 and set(paths)==set(PHYSICAL) and set(thresholds)==set(RECORD_TO_PHYSICAL)
        and snapshot['physical_feature_views']==closed.VIEWS and snapshot['record_to_physical']==RECORD_TO_PHYSICAL,'Frozen feature views/model/record order differs.')
    runtime=snapshot['runtime_files_sha256']
    for source in [Path(__file__).resolve(),Path(module.__file__).resolve(),Path(closed.__file__).resolve()]:
        require(runtime.get(str(source))==digest(source),'Inference/builder/helper was not frozen before source preparation: '+str(source))
    require(metadata['frozen_sha256']==snapshot['snapshot_sha256'] and digest(metadata['snapshot_path'])==metadata['snapshot_file_sha256'] and read(metadata['snapshot_path'])==snapshot,'Builder snapshot statement differs.')
    require(metadata['coverage']['all_content_tokens_scored'] is True and metadata['coverage']['independent_native_labels_match'] is True,'All-content native parser coverage proof absent.')
    require(np.array_equal(inputs.prompt_ix,np.arange(len(inputs))) and len(tokens)==len(inputs) and np.array_equal(rows.cache_ix,np.arange(len(rows)))
        and np.array_equal(rows.index,np.arange(len(rows))) and rows.sample_ix.is_unique and set(rows.role)<=set(closed.ROLES) and set(rows.split)<={'test'}
        and set(rows.wrapper)<={'native'} and set(rows.prompt_ix)==set(inputs.prompt_ix) and bool(len(inputs))==bool(len(rows)),'Native test cache identity/ownership/scope differs.')
    if dataset=='natural':require(set(rows.role)<={'system','user','assistant'},'Natural truth must be SUA only.')
    else:require(len(rows)>0,'C4 confirmation cannot be empty.')
    for prompt,part in rows.groupby('prompt_ix',sort=False):
        ids=np.asarray(tokens[int(prompt)],np.int64);pos=part.token_ix.to_numpy(np.int64)
        require(len(ids)==int(inputs.iloc[int(prompt)].length) and len(np.unique(pos))==len(pos) and (pos>=0).all() and (pos<len(ids)).all()
            and np.array_equal(ids[pos],part.token_id.to_numpy(np.int64)),'Prepared content token ID/position changed.')
    bindings={str(Path(module.OUT)/'data-complete.json'):digest(Path(module.OUT)/'data-complete.json'),str(Path(metadata['snapshot_path'])):metadata['snapshot_file_sha256']}
    for name,want in metadata['cache_sha256'].items():
        path=Path(module.CACHE)/name;require(digest(path)==want,'Builder cache changed: '+name);bindings[str(path)]=want
    for name,want in metadata['dependencies_sha256'].items():
        path=HERE/name;require(digest(path)==want,'Builder dependency changed: '+name);bindings[str(path)]=want
    require(metadata['source_sha256']==digest(module.__file__),'Builder source SHA changed.')
    for path in [Path(__file__).resolve(),Path(module.__file__).resolve(),Path(closed.__file__).resolve()]:bindings[str(path)]=digest(path)
    signature=dict(source_sha256=digest(__file__),builder_source_sha256=digest(module.__file__),dataset=dataset,protocol=protocol(),
        data_complete_sha256=bindings[str(Path(module.OUT)/'data-complete.json')],snapshot_sha256=snapshot['snapshot_sha256'],snapshot_file_sha256=metadata['snapshot_file_sha256'],
        model_sha256=snapshot['physical_model_sha256'],physical_feature_views=copy.deepcopy(closed.VIEWS),record_thresholds=thresholds,selected_head_ids=heads,
        cache_sha256=metadata['cache_sha256'],cache_ix_sha256=array_sha(rows.cache_ix.to_numpy()),sample_ix_sha256=array_sha(rows.sample_ix.to_numpy()),rows=len(rows),inputs=len(inputs),
        batch_prompt_order_sha256=closed.canonical(inputs.sort_values(['length','prompt_ix']).prompt_ix.to_list()),checkpoint_path=snapshot['checkpoint_path'],
        package_versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','numpy','pandas']})
    return Context(dataset,module,metadata,inputs,rows,tokens,snapshot,paths,thresholds,heads,Path(module.OUT)/'inference',Path(module.CACHE)/'inference',signature,bindings)


def fast_integrity(ctx):
    actual_gate();closed.verify_closed_snapshot(ctx.snapshot)
    for path,want in ctx.bindings.items():require(digest(path)==want,'Closed source/input cache changed: '+path)


class Collector:
    """Transient ranked10 query-head features; hook returns None and never edits input."""
    def __init__(self,selected):
        self.selected=np.asarray(selected,np.int64)
        require(self.selected.shape==(10,) and len(set(self.selected))==10 and ((self.selected>=0)&(self.selected<2304)).all(),'Invalid10head rank order.')
        self.layers=sorted(set((self.selected//64).tolist()));self.features=None;self.enabled=False;self.indices={}
    def begin(self,positions,enabled=True):
        self.positions=np.asarray(positions,np.int64)
        require(self.positions.ndim==1 and len(np.unique(self.positions))==len(self.positions) and (self.positions>=0).all(),'Duplicate/negative content position.')
        self.enabled=enabled;self.seen=set();self.indices={};self.features=np.full((len(self.positions),640),np.nan,np.float16) if enabled else None
    def hook(self,layer):
        def receive(module,args):
            import torch
            value=args[0];require(layer not in self.seen and value.ndim==3 and value.shape[-1]==4096,'Duplicate/malformed pre-o_proj hook.')
            self.seen.add(layer)
            if not self.enabled:return None
            key=str(value.device)
            if key not in self.indices:self.indices[key]=torch.as_tensor(self.positions,dtype=torch.long,device=value.device)
            ranks=np.flatnonzero(self.selected//64==layer);heads=torch.as_tensor(self.selected[ranks]%64,dtype=torch.long,device=value.device)
            sample=value.detach().reshape(-1,64,64).index_select(0,self.indices[key]).index_select(1,heads)
            require(bool(torch.isfinite(sample).all()) and bool((sample.float().abs()<=np.finfo(np.float16).max).all()),'Nonfinite/F16overflow activation; no clipping permitted.')
            columns=(ranks[:,None]*64+np.arange(64)).reshape(-1);self.features[:,columns]=sample.to('cpu',torch.float16).numpy().reshape(len(self.positions),-1)
            return None
        return receive
    def finish(self):
        require(self.seen==set(self.layers),'Selected layer hook missed.')
        if self.enabled:require(np.isfinite(self.features).all(),'Unwritten/nonfinite selected column.')
    def release(self):self.features=None;self.indices={}


def install_hooks(model,collector):
    handles=[]
    try:
        for layer in collector.layers:handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)))
        return handles
    except BaseException:
        for h in handles:h.remove()
        raise


def feature_view(name,features):
    require(name in PHYSICAL and features.ndim==2 and features.shape[1]==640 and features.dtype==np.float16 and np.isfinite(features).all(),'Exact finite transient640F16 required.')
    if name=='new':return np.ascontiguousarray(features)
    if name=='matched':return np.concatenate([features[:,:512],features[:,:128]],axis=1)
    return np.ascontiguousarray(features[:,:512])


def make_portable(name):
    import torch
    from torch import nn
    require(name in PHYSICAL,'Unknown physical model.')
    class Base(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Dropout(.1));self.head=nn.Linear(128,5)
        def raw(self,x):return self.head(self.encoder(x.contiguous()))
        def forward(self,x):return torch.log_softmax(self.raw(x),1)
    class Residual(nn.Module):
        def __init__(self):
            super().__init__();self.base=Base();self.correction=nn.Sequential(nn.Linear(640,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
        def raw(self,x):
            require(x.ndim==2 and x.shape[1]==640,'Residual requires640 standardized features.')
            base=self.base.raw(x[:,:512].contiguous());delta=4*torch.tanh(self.correction(x.contiguous()));raw=base.clone();raw[:,0]+=delta[:,0];return raw
        def forward(self,x):return torch.log_softmax(self.raw(x),1)
    return (Base() if name=='old-head' else Residual()).eval().requires_grad_(False)


def load_portables(ctx):
    actual_gate()
    import torch
    output={};original={}
    for name in ['old-head','new','matched']:
        path=ctx.paths[name];require(digest(path)==ctx.snapshot['physical_model_sha256'][name],'Portable bytes changed before load.')
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));mean,scale=z['mean'].copy(),z['scale'].copy();state={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')}
        proof=ctx.snapshot['portable_metadata'][name];view=closed.VIEWS[name];dim=view['dimensions']
        require(closed.canonical(meta)==proof['metadata_sha256'] and meta['roles']==closed.ROLES and meta['selected_head_ids']==view['selected_head_ids'] and meta['kind']==view['kind'] and meta['dimensions']==dim,'Portable metadata/order/view changed.')
        require(mean.shape==scale.shape==(dim,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all()
            and array_sha(mean)==proof['mean_sha256'] and array_sha(scale)==proof['scale_sha256'],'Portable scaler identity changed.')
        require(set(state)==set(proof['weight_sha256']) and all(v.dtype==np.float32 and np.isfinite(v).all() and array_sha(v)==proof['weight_sha256'][k] for k,v in state.items()),'Portable weight array identity changed.')
        if name=='old-head':original=state;oldmean,oldscale=mean,scale
        else:
            require(meta['base_dimensions']==512 and meta['base_hidden']==128 and meta['residual_hidden']==32 and meta['parameters']==20545 and meta['delta_formula']=='4*tanh(g)' and meta['base_model_sha256']==closed.OLD_MODEL_SHA,'Residual formula/base changed.')
            base={k[5:]:v for k,v in state.items() if k.startswith('base.')}
            require(set(base)==set(original) and all(np.array_equal(v.view(np.uint32),original[k].view(np.uint32)) for k,v in base.items()),'Residual base is not bitexact original512.')
            require(np.array_equal(mean[:512].view(np.uint32),oldmean.view(np.uint32)) and np.array_equal(scale[:512].view(np.uint32),oldscale.view(np.uint32)),'Original512 scaler changed.')
            if name=='matched':require(np.array_equal(mean[512:].view(np.uint32),oldmean[:128].view(np.uint32)) and np.array_equal(scale[512:].view(np.uint32),oldscale[:128].view(np.uint32)),'Duplicate128 scaler changed.')
        net=make_portable(name);net.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()},strict=True);output[name]=(net,torch.from_numpy(mean),torch.from_numpy(scale))
    return output


def cpu_probabilities(nets,features):
    import torch
    require(set(nets)==set(PHYSICAL),'Three cached physical consumers required.');result={}
    with torch.inference_mode():
        for name,(net,mean,scale) in nets.items():
            parts=[]
            for start in range(0,len(features),4096):
                raw=torch.from_numpy(feature_view(name,features[start:start+4096])).float();x=((raw-mean)/scale).contiguous()
                require(torch.isfinite(x).all() and not net.training and not any(p.requires_grad for p in net.parameters()),'Consumer must remain frozen/eval with finite standardized input.')
                parts.append(net(x).exp().numpy().astype(np.float32))
            result[name]=np.concatenate(parts) if parts else np.empty((0,5),np.float32)
    validate_probabilities(result);return result


def same_vector_gpu_parity(ctx,nets,features,cpu,targets):
    import torch
    count=min(32,len(features));require(count>0 and len(targets)==len(features),'No exact saved-vector parity scope.')
    differences={}
    with torch.inference_mode():
        for name,(net,mean,scale) in nets.items():
            raw=torch.from_numpy(feature_view(name,features[:count])).to('cuda:0').float();gpu=copy.deepcopy(net).to('cuda:0').eval().requires_grad_(False)
            got=gpu(((raw-mean.cuda(0))/scale.cuda(0)).contiguous()).exp().cpu().numpy();error=float(np.abs(got-cpu[name][:count]).max())
            require(np.isfinite(error) and 0<=error<5e-4,'Same-F16 CPU/GPU consumer parity failed: '+name);differences[name]=error;del gpu
    return dict(signature=ctx.signature,model_sha256=ctx.snapshot['physical_model_sha256'],sampled_tokens=count,max_probability_difference=differences,
        selected_head_ids=ctx.heads,physical_feature_views=copy.deepcopy(closed.VIEWS),captured_dtype='float16',same_vectors=True,CPU_outputs_are_saved=True,
        first_original_batch=True,feature_sha256=array_sha(features[:count]),target_cache_ix_sha256=array_sha(targets[:count]),
        sample_ix_sha256=array_sha(ctx.rows.iloc[targets[:count]].sample_ix.to_numpy()),no_features_persisted=True,tolerance=5e-4)


def require_checkpoint_path(value):
    require(isinstance(value,str) and value==CHECKPOINT,'Only exact verified /dev/shm checkpoint is permitted before filesystem/model access.')


def no_competing_jobs():
    names=set(BUILDERS.values())|{Path(__file__).name,'systemness_complementary_head_extract.py','systemness_complementary_head_readout.py',
        'systemness_complementary_head_scan.py','systemness_aug_confirmation_infer.py','systemness_natural_fit_probes.py','systemness_binary_residual.py',
        'systemness_long_mix_residual.py','systemness_long_augmentation_extract.py','systemness_head_extract.py','systemness_natural_dev_extract.py',
        'systemness_multiseed_ensemble.py','systemness_head_probes.py','systemness_head_scan.py','systemness_aug_confirmation_data.py','systemness_aug_natural_data.py'}
    names|={n+'.py' for n in BUILDERS.values()};active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg):active.append(int(entry.name))
    require(not active,'A competing extraction/training/builder/inference process is live: '+str(active))


@contextmanager
def model_context(ctx):
    """Own fixed4A40 loader; only third-party functions are temporarily patched."""
    actual_gate();require_checkpoint_path(ctx.snapshot['checkpoint_path']);no_competing_jobs()
    import torch
    from transformers import AutoModelForCausalLM,Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    import accelerate.hooks
    import accelerate.utils.operations as operations
    require(torch.cuda.device_count()==4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)),'Exactly4 visible A40 required.')
    require(all(torch.cuda.mem_get_info(i)[0]>=22*2**30 for i in range(4)),'Each A40 requires22GiB free; no process is killed/reset.')
    old_get,old_validate,old_send,old_hook_send=hub_kernels.get_kernel,Mxfp4HfQuantizer.validate_environment,operations.send_to_device,accelerate.hooks.send_to_device
    def pinned(name,*args,**kwargs):
        if name=='kernels-community/gpt-oss-triton-kernels':kwargs['revision']=ctx.snapshot['kernel_revision']
        return old_get(name,*args,**kwargs)
    def strict(self,*args,**kwargs):
        old_validate(self,*args,**kwargs);require(not self.quantization_config.dequantize,'Native MXFP4 required.')
    def staged(tensor,device,non_blocking=False,skip_keys=None):
        if isinstance(tensor,torch.Tensor) and tensor.device.type=='cuda':
            target=torch.device(f'cuda:{device}' if isinstance(device,int) else device)
            if target.type=='cuda' and tensor.device!=target:return tensor.cpu().to(target,non_blocking=False)
        return old_send(tensor,device,non_blocking=non_blocking,skip_keys=skip_keys)
    device_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}};model=None
    try:
        hub_kernels.get_kernel=pinned;Mxfp4HfQuantizer.validate_environment=strict;operations.send_to_device=staged;accelerate.hooks.send_to_device=staged
        model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,dtype=torch.bfloat16,quantization_config=Mxfp4Config(dequantize=False),device_map=device_map,attn_implementation='eager',use_kernels=False).eval().requires_grad_(False)
        backends=sorted({type(layer.mlp.experts).__name__ for layer in model.model.layers})
        require(backends==['Mxfp4GptOssExperts'] and len(model.model.layers)==36 and not model.hf_quantizer.quantization_config.dequantize
            and not any(p.requires_grad for p in model.parameters()) and model.hf_device_map==device_map,'Native4A40 frozen backend differs.')
        immutable_json(ctx.out/'model-backend.json',dict(signature=ctx.signature,expert_backends=backends,device_map=device_map,checkpoint_path=CHECKPOINT,native_mxfp4=True,parameters_frozen=True,
            gpu_transfer='Synchronous CPU staging between different CUDA devices',gpus=[torch.cuda.get_device_name(i) for i in range(4)],compute_capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(4)]))
        yield model
    finally:
        hub_kernels.get_kernel=old_get;Mxfp4HfQuantizer.validate_environment=old_validate;operations.send_to_device=old_send;accelerate.hooks.send_to_device=old_hook_send
        if model is not None:del model
        gc.collect()


def forward_capture(model,collector,batch,enabled=True):
    collector.begin(batch['positions'],enabled=enabled)
    hidden=model.model(input_ids=batch['input_ids'].to('cuda:0'),attention_mask=batch['attention_mask'].to('cuda:0'),use_cache=False).last_hidden_state
    collector.finish();return hidden


def checked_hooks(model,collector,ctx,batch):
    import torch
    kwargs=dict(input_ids=batch['input_ids'].to('cuda:0'),attention_mask=batch['attention_mask'].to('cuda:0'),use_cache=False)
    before=model.model(**kwargs).last_hidden_state.cpu();handles=install_hooks(model,collector)
    try:
        after=forward_capture(model,collector,batch,enabled=False).cpu()
        require(torch.equal(before,after) and bool(torch.isfinite(after).all()) and collector.features is None,'Disabled pre-o_proj capture changed hidden states or wrote features.')
        proof=dict(signature=ctx.signature,hidden_states_bitexact=True,all_capture_writes_disabled=True,selected_head_ids=ctx.heads,only_pre_o_proj_hooks=True,
            legacy_capture=False,dimensions=640,sample_rows=len(batch['targets']),sample_ix_sha256=array_sha(ctx.rows.iloc[batch['targets']].sample_ix.to_numpy()),
            input_ids_sha256=array_sha(batch['input_ids'].numpy()),attention_mask_sha256=array_sha(batch['attention_mask'].numpy()))
        immutable_json(ctx.out/'hook-validation.json',proof);return handles
    except BaseException:
        for h in handles:h.remove()
        raise


def verify_evidence(ctx):
    portable,hooks,backend=[read(ctx.out/n) for n in ['portable-validation.json','hook-validation.json','model-backend.json']]
    first=collate(ctx,batches(ctx)[0]);count=min(32,len(first['targets']))
    require(portable['signature']==hooks['signature']==backend['signature']==ctx.signature and portable['model_sha256']==ctx.snapshot['physical_model_sha256']
        and set(portable['max_probability_difference'])==set(PHYSICAL) and portable['sampled_tokens']==count and portable['tolerance']==5e-4
        and all(np.isfinite(v) and 0<=v<5e-4 for v in portable['max_probability_difference'].values()) and portable['same_vectors'] is True
        and portable['captured_dtype']=='float16' and portable['selected_head_ids']==ctx.heads and portable['physical_feature_views']==closed.VIEWS
        and portable['first_original_batch'] is True and portable['no_features_persisted'] is True and portable['CPU_outputs_are_saved'] is True
        and portable['target_cache_ix_sha256']==array_sha(first['targets'][:count]) and portable['sample_ix_sha256']==array_sha(ctx.rows.iloc[first['targets'][:count]].sample_ix.to_numpy()),'Same-saved-F16 portable evidence scope/identity differs.')
    require(closed.valid_sha(portable['feature_sha256']) and hooks['hidden_states_bitexact'] is True and hooks['all_capture_writes_disabled'] is True
        and hooks['only_pre_o_proj_hooks'] is True and hooks['legacy_capture'] is False and hooks['selected_head_ids']==ctx.heads and hooks['dimensions']==640
        and hooks['sample_rows']==len(first['targets']) and hooks['sample_ix_sha256']==array_sha(ctx.rows.iloc[first['targets']].sample_ix.to_numpy())
        and hooks['input_ids_sha256']==array_sha(first['input_ids'].numpy()) and hooks['attention_mask_sha256']==array_sha(first['attention_mask'].numpy()),'Disabled prehook evidence differs.')
    dm={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}}
    require(backend['expert_backends']==['Mxfp4GptOssExperts'] and backend['checkpoint_path']==CHECKPOINT==ctx.snapshot['checkpoint_path'] and backend['native_mxfp4'] is True
        and backend['parameters_frozen'] is True and backend['device_map']==dm and len(backend['gpus'])==len(backend['compute_capabilities'])==4
        and all('A40' in name for name in backend['gpus']) and all(cap==[8,6] for cap in backend['compute_capabilities'])
        and backend['gpu_transfer']=='Synchronous CPU staging between different CUDA devices','Frozen native4A40 backend proof differs.')


def finish_completion(ctx,values,started):
    fast_integrity(ctx);validate_probabilities(values);require(restore_progress(ctx,values)==len(batches(ctx)),'Content-token coverage not completely committed.');verify_evidence(ctx)
    psha={n:digest(ctx.cache/(n+'-probabilities.f32')) for n in PHYSICAL};asha=argmax_hashes(values)
    complete=dict(signature=ctx.signature,rows=len(ctx.rows),inputs=len(ctx.inputs),probability_sha256=psha,argmax_sha256=asha,**record_identity(psha,asha),
        no_observations=False,gpu_run=True,model_forward_run=True,all_finite=True,all_normalized=True,no_features_persisted=True,
        portable_validation_sha256=digest(ctx.out/'portable-validation.json'),hook_validation_sha256=digest(ctx.out/'hook-validation.json'),model_backend_sha256=digest(ctx.out/'model-backend.json'),
        progress_sha256=digest(ctx.cache/'progress.json'),seconds=time.time()-started)
    atomic_json(ctx.out/'inference-complete.json',complete);atomic_json(ctx.out/'status.json',dict(stage='inference_complete',rows=len(ctx.rows),gpu_run=True));return complete


def completed_probabilities(dataset,module=None,*,context=None):
    actual_gate()
    if context is None:
        module=dataset_module(dataset,module);out=Path(module.OUT)/'inference'
        require((out/'inference-complete.json').is_file() and (out/'status.json').is_file() and read(out/'status.json')['stage']=='inference_complete','Inference not terminal complete; partial probability/builder reads forbidden.')
        ctx=load_context(dataset,module)
    else:ctx=context
    require(ctx.dataset==dataset and (ctx.out/'inference-complete.json').is_file() and read(ctx.out/'status.json')['stage']=='inference_complete','Inference is not terminal complete.')
    complete=read(ctx.out/'inference-complete.json');require(complete['signature']==ctx.signature and complete['rows']==len(ctx.rows) and complete['inputs']==len(ctx.inputs)
        and complete['all_finite'] is True and complete['all_normalized'] is True and complete['no_features_persisted'] is True,'Completed probability signature/coverage differs.')
    require(set(complete['probability_sha256'])==set(complete['argmax_sha256'])==set(PHYSICAL),'Three physical probability caches required.')
    if not len(ctx.rows):
        require(ctx.dataset=='natural' and not len(ctx.inputs) and complete['no_observations'] is True and complete['gpu_run'] is False and complete['model_forward_run'] is False
            and complete['probability_sha256']==complete['argmax_sha256']=={n:None for n in PHYSICAL}
            and all(complete[k] is None for k in ['portable_validation_sha256','hook_validation_sha256','model_backend_sha256','progress_sha256']),'Empty natural invented observations/evidence.')
        values={n:np.empty((0,5),np.float32) for n in PHYSICAL}
    else:
        require(complete['no_observations'] is False and complete['gpu_run'] is True and complete['model_forward_run'] is True,'Nonempty backend evidence absent.')
        for name,key in [('portable-validation.json','portable_validation_sha256'),('hook-validation.json','hook_validation_sha256'),('model-backend.json','model_backend_sha256')]:require(digest(ctx.out/name)==complete[key],'Inference evidence SHA differs: '+name)
        require(digest(ctx.cache/'progress.json')==complete['progress_sha256'],'Completed progress SHA differs.');verify_evidence(ctx)
        values=initialize_probabilities(ctx,readonly=True)
        for n in PHYSICAL:require(digest(ctx.cache/(n+'-probabilities.f32'))==complete['probability_sha256'][n],'Saved probability file SHA differs: '+n)
        validate_probabilities(values);require(restore_progress(ctx,values)==len(batches(ctx)) and argmax_hashes(values)==complete['argmax_sha256'],'Completed prefix/argmax differs.')
    require(all(complete[k]==v for k,v in record_identity(complete['probability_sha256'],complete['argmax_sha256']).items()),'Oldhead two-threshold probability/argmax alias changed.')
    fast_integrity(ctx);return complete,values


def infer(dataset,module=None):
    actual_gate();no_competing_jobs();ctx=load_context(dataset,module);ctx.out.mkdir(parents=True,exist_ok=True)
    with run_lock(ctx):
        if (ctx.out/'inference-complete.json').exists():completed_probabilities(dataset,context=ctx);return
        if not len(ctx.rows):fast_integrity(ctx);empty_completion(ctx);return
        values=initialize_probabilities(ctx);start=restore_progress(ctx,values);parts=batches(ctx);started=time.time()
        if start:verify_evidence(ctx)
        if start==len(parts):finish_completion(ctx,values,started);completed_probabilities(dataset,context=ctx);return
        import torch
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');nets=load_portables(ctx);collector=Collector(ctx.heads);handles=[];model=None
        try:
            with model_context(ctx) as model,torch.inference_mode():
                first=collate(ctx,parts[0]);handles=checked_hooks(model,collector,ctx,first)
                hidden=forward_capture(model,collector,first);del hidden
                firstp=cpu_probabilities(nets,collector.features)
                if start:require(all(np.array_equal(firstp[n].view(np.uint32),np.asarray(values[n][first['targets']]).view(np.uint32)) for n in PHYSICAL),'Resumed first-batch CPU outputs differ from saved committed probabilities.')
                proof=same_vector_gpu_parity(ctx,nets,collector.features,firstp,first['targets'])
                path=ctx.out/'portable-validation.json'
                if path.exists():
                    previous=read(path);require(previous['signature']==ctx.signature and previous['feature_sha256']==proof['feature_sha256'] and previous['target_cache_ix_sha256']==proof['target_cache_ix_sha256'],'Original first-batch feature/parity identity changed on resume.');verify_evidence(ctx)
                else:atomic_json(path,proof)
                for bi in range(start,len(parts)):
                    if bi==0:batch,p=first,firstp
                    else:
                        collector.release();batch=collate(ctx,parts[bi]);hidden=forward_capture(model,collector,batch);del hidden;p=cpu_probabilities(nets,collector.features)
                    for n in PHYSICAL:values[n][batch['targets']]=p[n]
                    collector.release();del p
                    if (bi+1)%INTERVAL==0 or bi+1==len(parts):
                        fast_integrity(ctx);checkpoint(ctx,values,bi+1);status=dict(stage='inference',batch=bi+1,batches=len(parts),rows=progress_record(ctx,values,bi+1)['rows_written'],seconds=time.time()-started)
                        atomic_json(ctx.out/'status.json',status);print(json.dumps(status),flush=True)
                del firstp,first
            finish_completion(ctx,values,started);completed_probabilities(dataset,context=ctx)
        except BaseException:
            if not (ctx.out/'inference-complete.json').exists():atomic_json(ctx.out/'status.json',dict(stage='failed',next_batch=read(ctx.cache/'progress.json')['next_batch']))
            raise
        finally:
            for h in handles:h.remove()
            collector.release()
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()


def preflight():
    """Synthetic CPU/tmp only, never a native LM/GPU/tokenizer validation."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import ast
    import py_compile
    import torch
    from torch import nn
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-infer-preflight-',dir='/tmp'));source=digest(__file__);checks=[]
    def check(name,value=True):require(bool(value),'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,KeyError,FileNotFoundError):checks.append(name);return
        raise RuntimeError('Expected CPU fixture refusal: '+name)
    def forbidden(*a,**k):raise LookupError('Real builder/model/snapshot/checkpoint/tokenizer API forbidden in CPU fixture.')
    # The real closed pin gate, not a simulated gate, must run before formal IO.
    with patch.object(closed,'TRAINER_SOURCE_SHA',None),patch.object(closed,'read',forbidden),patch.object(closed,'digest',forbidden),patch.object(pd,'read_parquet',forbidden),patch.object(np,'load',forbidden):
        reject('unknown_pins_load_context_before_builder_IO',lambda:load_context('c4',SimpleNamespace(load_verified_data=forbidden)))
        reject('unknown_pins_infer_before_context_or_GPU',lambda:infer('natural'))
        reject('unknown_pins_completed_before_builder_or_probability_IO',lambda:completed_probabilities('c4'))
        reject('unknown_pins_portable_loader_before_weights',lambda:load_portables(None))
    reject('checkpoint_workspace_lexical_guard',lambda:require_checkpoint_path('/workspace'))
    reject('checkpoint_alias_or_other_path_rejected',lambda:require_checkpoint_path('/dev/shm/role-analysis-gptoss120b/'))
    # Hook-only synthetic36layer model, no Transformers/model construction.
    selected=[2248,458,247,462,2045,460,459,2047,456,997]
    class Attention(nn.Module):
        def __init__(self):super().__init__();self.o_proj=nn.Identity()
        def forward(self,x):return self.o_proj(x)
    class Layer(nn.Module):
        def __init__(self,i):super().__init__();self.self_attn=Attention();self.i=i
        def forward(self,x):return self.self_attn(x)+self.i/1000
    layers=nn.ModuleList([Layer(i) for i in range(36)]);fake=SimpleNamespace(model=SimpleNamespace(layers=layers));raw=torch.arange(2*5*4096).reshape(2,5,4096).float()/2000
    def forward():
        x=raw.clone()
        for layer in layers:x=layer(x)
        return x
    before=forward();collector=Collector(selected);positions=np.array([0,4,7,9]);handles=install_hooks(fake,collector)
    try:
        collector.begin(positions,enabled=False);after=forward();collector.finish();check('disabled_hook_writes_none_and_hidden_bitexact',torch.equal(before,after) and collector.features is None)
        collector.begin(positions);after=forward();collector.finish();x=raw.clone();expected={}
        for i in range(36):expected[i]=x.clone();x=x+i/1000
        wanted=np.concatenate([expected[h//64].reshape(-1,64,64)[positions,h%64].half().numpy() for h in selected],1)
        check('unsorted10head640_every_column_and_position_exact',np.array_equal(collector.features,wanted) and collector.features.shape==(4,640) and torch.equal(before,after))
        reject('duplicate_layer_hook_rejected',lambda:layers[collector.layers[0]].self_attn(raw))
        collector.begin(positions);reject('missing_layer_or_unwritten_column_rejected',collector.finish)
        reject('duplicate_sampled_token_position_rejected',lambda:collector.begin([1,1]))
    finally:
        for h in handles:h.remove()
        collector.release()
    rng=np.random.RandomState(20261021);original=[458,462,2045,460,459,2047,456,997];heads=original+[247,2248]
    oldnet=make_portable('old-head');base={k:v.detach().numpy().copy() for k,v in oldnet.state_dict().items()};paths={};portable={}
    oldmean=rng.normal(0,.02,512).astype(np.float32);oldscale=rng.uniform(.8,1.2,512).astype(np.float32)
    newmean=np.r_[oldmean,np.zeros(128,np.float32)];newscale=np.r_[oldscale,np.ones(128,np.float32)];dupmean=np.r_[oldmean,oldmean[:128]];dupscale=np.r_[oldscale,oldscale[:128]]
    for name,mean,scale in [('old-head',oldmean,oldscale),('new',newmean,newscale),('matched',dupmean,dupscale)]:
        net=make_portable(name)
        if name=='old-head':net.load_state_dict(oldnet.state_dict())
        else:
            net.base.load_state_dict(oldnet.state_dict())
            with torch.no_grad():
                net.correction[0].weight.zero_();net.correction[0].bias.zero_();net.correction[0].weight[0,512:]=.02;net.correction[3].weight.zero_();net.correction[3].weight[0,0]=.5;net.correction[3].bias.zero_()
        meta=dict(kind=closed.VIEWS[name]['kind'],roles=closed.ROLES,selected_head_ids=closed.VIEWS[name]['selected_head_ids'],dimensions=closed.VIEWS[name]['dimensions'])
        if name!='old-head':meta.update(base_dimensions=512,base_hidden=128,residual_hidden=32,parameters=20545,delta_formula='4*tanh(g)',base_model_sha256=closed.OLD_MODEL_SHA)
        weights={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};path=folder/(name+'.npz');np.savez_compressed(path,metadata=np.array(json.dumps(meta)),mean=mean,scale=scale,**{'weight::'+k:v for k,v in weights.items()});paths[name]=path
        portable[name]=dict(metadata_sha256=closed.canonical(meta),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale),weight_sha256={k:array_sha(v) for k,v in weights.items()})
    frames=[];inputs=[];tokens=[]
    for prompt,length in enumerate([5,8,6,9,7,10]):
        ids=list(range(100+prompt*20,100+prompt*20+length));tokens.append(ids);inputs.append(dict(prompt_ix=prompt,length=length))
        for t in range(1,length-1):frames.append(dict(cache_ix=len(frames),sample_ix=12_000_000_000_000+len(frames),prompt_ix=prompt,token_ix=t,token_id=ids[t],role=closed.ROLES[t%5],split='test',wrapper='native'))
    inputs,rows=pd.DataFrame(inputs),pd.DataFrame(frames);builder=folder/'fake-builder.py';builder.write_text('# synthetic builder only\n');out,cache=folder/'data',folder/'cache';out.mkdir();cache.mkdir()
    for n in ['inputs.parquet','rows.parquet','token-ids.json']:(cache/n).write_text('synthetic identity only')
    snapshot=dict(snapshot_sha256='f'*64,physical_model_sha256={n:digest(p) for n,p in paths.items()},physical_feature_views=copy.deepcopy(closed.VIEWS),record_to_physical=RECORD_TO_PHYSICAL,
        portable_metadata=portable,checkpoint_path=CHECKPOINT,kernel_revision='synthetic',runtime_files_sha256={str(Path(p).resolve()):digest(p) for p in [__file__,builder,closed.__file__]})
    statement=folder/'snapshot.json';atomic_json(statement,snapshot);metadata=dict(frozen=snapshot,frozen_sha256=snapshot['snapshot_sha256'],snapshot_path=str(statement),snapshot_file_sha256=digest(statement),
        coverage=dict(all_content_tokens_scored=True,independent_native_labels_match=True),cache_sha256={n:digest(cache/n) for n in ['inputs.parquet','rows.parquet','token-ids.json']},dependencies_sha256={},source_sha256=digest(builder))
    atomic_json(out/'data-complete.json',metadata);module=SimpleNamespace(OUT=out,CACHE=cache,__file__=str(builder),load_verified_data=lambda:(metadata,inputs,rows,tokens));thresholds={r:.5 for r in RECORD_TO_PHYSICAL}
    with patch.object(closed,'pin_gate',lambda:None),patch.object(closed,'verify_closed_snapshot',lambda value:(paths,thresholds,heads)),patch.dict(BUILDERS,{'c4':'never_import_real_builder'}):
        ctx=load_context('c4',module);nets=load_portables(ctx);feature=rng.normal(size=(len(rows),640)).astype(np.float16);p=cpu_probabilities(nets,feature)
        # Each model checked against independent functional F32 computation.
        maximum=0.
        with torch.inference_mode():
            for name in PHYSICAL:
                net,mean,scale=nets[name];v=feature_view(name,feature);x=((torch.from_numpy(v).float()-mean)/scale).contiguous();s=net.state_dict()
                b={k:v for k,v in s.items()} if name=='old-head' else {k[5:]:v for k,v in s.items() if k.startswith('base.')}
                rawref=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x[:,:512].contiguous(),b['encoder.0.weight'],b['encoder.0.bias'])),b['head.weight'],b['head.bias'])
                if name!='old-head':
                    g=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,s['correction.0.weight'],s['correction.0.bias'])),s['correction.3.weight'],s['correction.3.bias']).reshape(-1);rawref[:,0]+=4*torch.tanh(g)
                expectedp=rawref.log_softmax(1).exp().numpy();maximum=max(maximum,float(np.abs(expectedp-p[name]).max()))
        check('three_portable_specific_F32_functional_reference',maximum==0)
        changed=feature.copy();changed[:,512:]=changed[:,512:]+np.float16(2);q=cpu_probabilities(nets,changed)
        check('real_extra128_changes_only_new_not_duplicate_or_old',not np.array_equal(p['new'],q['new']) and np.array_equal(p['matched'].view(np.uint32),q['matched'].view(np.uint32)) and np.array_equal(p['old-head'].view(np.uint32),q['old-head'].view(np.uint32)))
        seen=[];h=nets['new'][0].base.encoder[0].register_forward_pre_hook(lambda n,args:seen.append((args[0].shape[1],args[0].is_contiguous())))
        with torch.inference_mode():nets['new'][0](torch.from_numpy(rng.normal(size=(3,1280)).astype('f4'))[:,::2])
        h.remove();check('noncontiguous640_input_still_base_contiguous512',seen==[(512,True)])
        alias=records_from_physical(p);check('oldhead_two_thresholds_same_array_and_argmax_object',alias['oldhead-max33'] is alias['oldhead-original'] and np.array_equal(alias['oldhead-max33'].argmax(1),alias['oldhead-original'].argmax(1)))
        bad=snapshot['portable_metadata']['matched']['mean_sha256'];snapshot['portable_metadata']['matched']['mean_sha256']='0'*64;reject('wrong_arm_embedded_scaler_hash_rejected',lambda:load_portables(ctx));snapshot['portable_metadata']['matched']['mean_sha256']=bad
        batch=collate(ctx,[0,1]);check('leftpadding_positions_preserve_true_token_ids',np.array_equal(batch['input_ids'].numpy().reshape(-1)[batch['positions']],rows.iloc[batch['targets']].token_id) and batches(ctx)==[[0,2,4,1],[3,5]])
        ctx.out.mkdir();values=initialize_probabilities(ctx);check('NaN_owner_initialized_zero_prefix',restore_progress(ctx,values)==0 and all(np.isnan(v).all() for v in values.values()))
        targets=completed_indices(ctx,1)
        for n in PHYSICAL:values[n][targets]=p[n][targets]
        checkpoint(ctx,values,1);check('complete_batch_prefix_sample_allprobbytes_restores',restore_progress(ctx,values)==1)
        held=values['new'][targets[-1]].copy();values['new'][targets[-1]]=np.roll(held,1);reject('normalized_changed_committed_last_row_rejected',lambda:restore_progress(ctx,values));values['new'][targets[-1]]=held
        values['new'][targets[-1],4]=np.nan;reject('committed_last_column_NaN_rejected',lambda:restore_progress(ctx,values));values['new'][targets[-1]]=held
        progress=read(ctx.cache/'progress.json');wrong=copy.deepcopy(progress);wrong['sample_ix_sha256']='0'*64;atomic_json(ctx.cache/'progress.json',wrong);reject('wrong_sample_order_prefix_rejected',lambda:restore_progress(ctx,values));atomic_json(ctx.cache/'progress.json',progress)
        del values;gc.collect();file=ctx.cache/'new-probabilities.f32';payload=file.read_bytes();file.write_bytes(payload[:-4]);reject('truncated_probability_cannot_restore_old_progress',lambda:initialize_probabilities(ctx,readonly=True));file.write_bytes(payload)
        parked=ctx.cache/'held.f32';file.rename(parked);reject('missing_probability_not_silently_rebuilt',lambda:initialize_probabilities(ctx,readonly=True));parked.rename(file)
        old=module.load_verified_data;module.load_verified_data=forbidden;reject('terminal_marker_before_builder_or_probability_read',lambda:completed_probabilities('c4',module));module.load_verified_data=old
        bad={n:a.copy() for n,a in p.items()};bad['new'][0]*=.5;reject('unnormalized_probability_rejected',lambda:validate_probabilities(bad))
        empty=Context('natural',module,metadata,inputs.iloc[:0].copy(),rows.iloc[:0].copy(),[],snapshot,paths,thresholds,heads,folder/'empty-out',folder/'empty-cache',dict(synthetic=True,rows=0),{})
        with patch.dict(globals(),dict(load_context=lambda *a,**k:empty,no_competing_jobs=lambda:None,fast_integrity=lambda value:None,model_context=forbidden,load_portables=forbidden)):
            infer('natural',module);complete,ep=completed_probabilities('natural',context=empty)
        check('empty_natural_infer_and_completed_noGPU_no_model_no_cache',complete['gpu_run'] is False and complete['model_forward_run'] is False and not empty.cache.exists() and all(a.shape==(0,5) for a in ep.values()))
        check('empty_oldhead_alias_no_invented_probability_SHA',records_from_physical(ep)['oldhead-max33'] is records_from_physical(ep)['oldhead-original'] and complete['record_probability_sha256']['oldhead-original'] is None)
    code=Path(__file__).read_text();tree=ast.parse(code);loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='model_context');loadercode=ast.unparse(loader)
    check('own_loader_checkpoint_guard_precedes_thirdparty_load',loadercode.index('require_checkpoint_path(')<loadercode.index('from transformers import') and not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load_model' for n in ast.walk(tree)) and not any(isinstance(n,ast.Import) and any(a.name=='systemness_data' for a in n.names) for n in tree.body))
    check('own_loader_restores_all_thirdparty_patches_in_finally',all(s in loadercode for s in ['hub_kernels.get_kernel = old_get','Mxfp4HfQuantizer.validate_environment = old_validate','operations.send_to_device = old_send','accelerate.hooks.send_to_device = old_hook_send']) and bool(loader.body[-1].finalbody))
    check('source_unchanged',source==digest(__file__));py_compile.compile(__file__,cfile=str(folder/'infer.pyc'),doraise=True)
    report=dict(stage='synthetic_CPU_complementary640_inference_preflight',source_sha256=source,checks=checks,checks_passed=len(checks),maximum_CPU_functional_probability_difference=maximum,
        fixture_directory=str(folder),physical_models=3,records=4,actual_builder_or_snapshot_read=False,actual_source_IDs_or_body_read=False,real_portable_or_LM_or_tokenizer_loaded=False,
        raw_feature_bank_or_actual_probability_arrays_read=False,raw_features_persisted=False,gpu_used=False,formal_outputs_written=False,
        native_evidence='Native4A40 hidden bitexact and same-saved-F16 CPU/GPU parity remain mandatory future actual checks; these CPU fixtures do not claim either.')
    path=folder/'report.json';atomic_json(path,report);print(json.dumps(dict(report=str(path),report_sha256=digest(path),source_sha256=source,checks=len(checks))),flush=True);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group();action.add_argument('--preflight',action='store_true');action.add_argument('--infer',action='store_true');parser.add_argument('--dataset',choices=sorted(BUILDERS));args=parser.parse_args()
    if args.preflight:preflight()
    elif args.infer:require(args.dataset is not None,'--infer requires --dataset.');infer(args.dataset)
    else:parser.print_help()


if __name__=='__main__':main()
