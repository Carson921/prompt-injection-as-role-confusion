"""Extract frozen MXFP4 features on four A40s, keeping only scored content rows."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import systemness_data as data
import run_local_gptoss120b as runtime
from feature_pipeline_runtime import feature_batches, install_hooks


class Prompts(Dataset):
    def __init__(self, inputs, tokens):
        self.inputs = inputs.sort_values(['length','prompt_ix']).reset_index(drop=True)
        self.tokens = tokens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        ix = int(self.inputs.iloc[i].prompt_ix)
        return dict(prompt_ix=ix, input_ids=self.tokens[ix])


def collate(batch):
    width = max(len(row['input_ids']) for row in batch)
    ids = torch.zeros((len(batch),width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i,row in enumerate(batch):
        n = len(row['input_ids'])
        ids[i,-n:] = torch.tensor(row['input_ids'])
        mask[i,-n:] = 1
    return dict(input_ids=ids, attention_mask=mask, prompt_ix=[row['prompt_ix'] for row in batch])


def extract(part, batch_size=8):
    data.plan()
    supplemental = part in ['augmentation','challenge']
    source = data.CACHE/'supplement' if supplemental else data.CACHE
    meta_path = data.OUT/('supplement-data-complete.json' if supplemental else 'data-complete.json')
    meta = json.loads(meta_path.read_text())
    assert all(data.digest(source/name)==sha for name,sha in meta['cache_sha256'].items())
    if part in ['test','challenge']:
        frozen = json.loads((data.OUT/'selection.json').read_text())
        assert frozen['test_used_for_selection'] is False
        assert frozen['selected_model_sha256'] == data.digest(data.OUT/'models/selected.npz')
    folder = data.FEATURE_CACHE / part
    folder.mkdir(exist_ok=True)
    done = data.OUT / f'extraction-{part}.json'
    if done.exists():
        old = json.loads(done.read_text())
        assert old['rows_sha256'] == data.digest(folder/'rows.parquet')
        assert (folder/'features.f16').stat().st_size == old['rows']*8640*2
        print(f'{part} features already complete.', flush=True)
        return
    inputs = pd.read_parquet(source/'inputs.parquet')
    rows = pd.read_parquet(source/'rows.parquet')
    wanted = ['fit','validation'] if part in ['development','augmentation'] else ['test']
    inputs = inputs[inputs.split.isin(wanted)]
    rows = rows[rows.split.isin(wanted)].copy().reset_index(drop=True)
    rows['cache_ix'] = np.arange(len(rows))
    rows.to_parquet(folder/'rows.parquet',index=False)
    lookup = {int(p):group for p,group in rows.groupby('prompt_ix',sort=False)}
    with (source/'token-ids.json').open() as f:
        tokens = json.load(f)
    ds = Prompts(inputs,tokens)
    loader = DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=collate)
    path = folder/'features.f16'
    progress = folder/'progress.json'
    signature = dict(rows_sha256=data.digest(folder/'rows.parquet'), inputs_sha256=meta['cache_sha256']['inputs.parquet'],
                     specs=data.SPECS,batch_size=batch_size)
    start_batch = 0
    if progress.exists():
        previous = json.loads(progress.read_text())
        assert previous['signature']==signature
        start_batch = previous['next_batch']
    array = np.memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=(len(rows),8640))
    runtime.atomic_json = lambda p,v: data.atomic_json(data.OUT/f'{part}-{p.name}',v)
    runtime.status = lambda stage,**fields: print(json.dumps(dict(stage=stage,part=part,**fields)),flush=True)
    model = runtime.load_model()
    captured = {}
    def receive(name,value):
        captured[name] = value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started = time.time()
    with torch.inference_mode():
        # Check hooks and the four-stage implementation on an actual padded batch.
        batch = next(iter(loader))
        short = dict(input_ids=batch['input_ids'].to('cuda:0'),attention_mask=batch['attention_mask'].to('cuda:0'))
        before = model.model(**short,use_cache=False).last_hidden_state.float().cpu()
        handles = install_hooks(model,data.SPECS,receive)
        after = model.model(**short,use_cache=False).last_hidden_state.float().cpu()
        assert torch.equal(before,after) and torch.isfinite(after).all()
        reference = dict(captured)
        for handle in handles:
            handle.remove()
        del before,after
        from types import SimpleNamespace
        sample = SimpleNamespace(**short)
        completed_rows = 0
        for bi,batch,states in feature_batches(model,loader,data.SPECS,start_batch,(sample,reference),
                lambda v:data.atomic_json(data.OUT/f'pipeline-validation-{part}.json',v)):
            length = batch['input_ids'].shape[1]
            positions = []
            targets = []
            for b,prompt in enumerate(batch['prompt_ix']):
                group = lookup[prompt]
                offset = b*length+length-len(tokens[prompt])
                positions.extend((group.token_ix.to_numpy()+offset).tolist())
                targets.extend(group.cache_ix.tolist())
            indices = torch.tensor(positions,dtype=torch.long)
            values = torch.cat([states[key][indices] for key in data.SPECS],dim=1)
            assert values.shape==(len(targets),8640) and torch.isfinite(values).all()
            array[targets] = values.numpy()
            completed_rows += len(targets)
            del values,indices
            states.clear()
            if (bi+1)%16==0 or bi+1==len(loader):
                array.flush()
                data.atomic_json(progress,dict(signature=signature,next_batch=bi+1,batches=len(loader),complete=bi+1==len(loader)))
                status = dict(stage='extracting',part=part,batch=bi+1,batches=len(loader),
                              rows_this_run=completed_rows,seconds=time.time()-started)
                data.atomic_json(data.OUT/'status.json',status)
                print(json.dumps(status),flush=True)
        array.flush()
    # Spot checking the complete mapped file catches corrupt/unwritten values.
    assert np.isfinite(array).all()
    record = dict(part=part,rows=len(rows),features=8640,dtype='float16',prompts=len(inputs),
                  input_tokens=int(inputs.length.sum()),seconds=time.time()-started,
                  source_data_sha256=data.digest(meta_path),
                  rows_sha256=data.digest(folder/'rows.parquet'),features_sha256=data.digest(path),
                  hook_leaves_hidden_states_exact=True,feature_spec=data.SPECS,all_finite=True,
                  runner_sha256=data.digest(__file__))
    data.atomic_json(done,record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=['development','augmentation','test','challenge'],required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    extract(args.part,args.batch_size)
