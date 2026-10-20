"""Extract additional frozen MXFP4 120B feature locations on four A40s.

HF_HOME=/root/.cache/huggingface-role-probe CUDA_VISIBLE_DEVICES=0,1,2,3 \
ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b \
.venv/bin/python experiments/role-analysis/extract_feature_variants.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import run_local_gptoss120b as base

OUT=base.OUT/'feature-search'
SPECS={**{f'{point}-{layer}':dict(point=point,layer=layer) for point in ['mid','out'] for layer in [16,20,24]},
       **{f'norm-{layer}':dict(point='norm',layer=layer) for layer in [18,22]}}
atomic_json=base.atomic_json


def status(stage,**fields):
    value=dict(stage=stage,unix_time=time.time(),**fields)
    atomic_json(OUT/'extraction-status.json',value)
    print(json.dumps(value),flush=True)


def main():
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from utils.dataset import ReconstructableTextDataset,stack_collate
    from feature_pipeline_runtime import install_hooks,feature_batches
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    cache=OUT/'activations'
    cache.mkdir(exist_ok=True)
    meta=json.loads((base.OUT/'manifest.json').read_text())
    inputs=pd.read_parquet(base.OUT/'training_inputs.parquet')
    tok=base.tokenizer()
    ds=ReconstructableTextDataset(inputs.prompt.tolist(),tok,max_length=meta['max_length'],prompt_ix=inputs.prompt_ix.tolist())
    assert int(ds.attention_mask.sum())==meta['tokens']
    n,width=meta['tokens'],2880
    config=dict(specs=SPECS,source_manifest_sha256=hashlib.sha256((base.OUT/'manifest.json').read_bytes()).hexdigest(),
                stored_dtype='float16',weight_format='MXFP4',batch_size=args.batch_size,tokens=n,width=width)
    if (OUT/'extraction-config.json').exists():
        assert json.loads((OUT/'extraction-config.json').read_text())==config
    atomic_json(OUT/'extraction-config.json',config)
    checkpoint=cache/'progress.json'
    p=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(cursor=0,next_batch=0)
    if p.get('complete'):
        status('cached',tokens=n)
        return
    if not p['cursor']:
        assert shutil.disk_usage(OUT).free>len(SPECS)*n*width*2+5*2**30,'Insufficient space for eight feature maps'
    maps={name:np.memmap(cache/f'{name}.f16',mode='r+' if p['cursor'] else 'w+',dtype=np.float16,shape=(n,width)) for name in SPECS}
    control=np.memmap(base.OUT/'activations/layer-16.f16',mode='r',dtype=np.float16,shape=(n,width))
    base.status=status
    base.atomic_json=lambda path,value:atomic_json(OUT/Path(path).name,value)
    model=base.load_model()
    captures={}
    def receive(name,output):
        captures[name]=output.detach().to('cpu',torch.float16).reshape(-1,width)
    specs={**SPECS,'control-norm-16':dict(point='norm',layer=16)}
    with torch.inference_mode():
        sample=tok('A short forward validation.',return_tensors='pt').to('cuda:0')
        reference=model(**sample,use_cache=False).logits.float().cpu()
        handles=install_hooks(model,specs,receive)
        hooked=model(**sample,use_cache=False).logits.float().cpu()
        assert torch.equal(reference,hooked) and reference.std()>.01
        atomic_json(OUT/'forward-validation.json',dict(exact_logits=True,logit_std=float(reference.std()),
                    features=list(specs),all_finite=all(bool(torch.isfinite(v).all()) for v in captures.values())))
        validation=dict(captures)
        for handle in handles:
            handle.remove()
        captures.clear()
        dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=stack_collate)
        batches=feature_batches(model,dl,specs,p['next_batch'],(sample,validation),
                    lambda result:atomic_json(OUT/'pipeline-validation.json',result))
        cursor=p['cursor']
        started=time.time()
        for bi,batch,states in batches:
            mask=batch['attention_mask'].reshape(-1).bool()
            count=int(mask.sum())
            existing=states['control-norm-16'][mask].numpy()
            # Every token is checked, rather than relying only on a short input.
            assert np.array_equal(existing,control[cursor:cursor+count]),(bi,'Existing cache mismatch')
            for name,mm in maps.items():
                values=states[name][mask].numpy()
                assert np.isfinite(values).all(),(name,bi)
                mm[cursor:cursor+count]=values
            cursor+=count
            states.clear()
            if (bi+1)%4==0 or bi+1==len(dl):
                for mm in maps.values():
                    mm.flush()
                atomic_json(checkpoint,dict(cursor=cursor,next_batch=bi+1,batch_size=args.batch_size,complete=bi+1==len(dl)))
                status('extracting',batch=bi+1,total_batches=len(dl),tokens=cursor,seconds=time.time()-started)
        assert cursor==n
    atomic_json(OUT/'extraction-complete.json',dict(features=list(SPECS),tokens=n,seconds=time.time()-started,
                all_tokens_match_original_norm16=True,peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(4)]))
    status('complete',tokens=n,features=len(SPECS))


if __name__=='__main__':
    main()
