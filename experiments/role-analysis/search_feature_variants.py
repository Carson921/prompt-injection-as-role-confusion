"""Compare frozen-model feature locations and linear multi-layer probes.

Run after extract_feature_variants.py, with the existing four-A40 environment:
  HF_HOME=/root/.cache/huggingface-role-probe .venv/bin/python -u \
    experiments/role-analysis/search_feature_variants.py

Screen all 22 feature sets on at most 256 uniformly spaced content tokens per
training prompt. Refit the three best feature sets and the original norm-16/20
controls on all fit tokens before validation selection. Tests are evaluated only
after selection; original test splits have been reported in earlier experiments.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

import tune_gptoss120b_probes as engine
from extract_feature_variants import SPECS

SOURCE=engine.SOURCE
OUT=SOURCE/'feature-search'
engine.OUT=OUT
CS=[.0005,.005]
FEATURES={f'norm-{l}':[dict(point='norm',layer=l)] for l in [12,16,18,20,22,24]}
FEATURES.update({f'{p}-{l}':[dict(point=p,layer=l)] for p in ['mid','out'] for l in [16,20,24]})
for point in ['norm','mid','out']:
    for layers in [(16,20),(20,24),(16,20,24)]:
        FEATURES[point+'-'+ '+'.join(map(str,layers))]=[dict(point=point,layer=l) for l in layers]
FEATURES['norm-18+20+22']=[dict(point='norm',layer=l) for l in [18,20,22]]
atomic_json=engine.atomic_json


def status(stage,**fields):
    value=dict(stage=stage,unix_time=time.time(),**fields)
    atomic_json(OUT/'search-status.json',value)
    print(json.dumps(value),flush=True)


def feature_path(spec):
    key=f'{spec["point"]}-{spec["layer"]}'
    if key in SPECS:
        return OUT/'activations'/f'{key}.f16'
    assert spec['point']=='norm'
    return SOURCE/'activations'/f'layer-{spec["layer"]}.f16'


def prepare():
    assert (OUT/'extraction-complete.json').exists()
    assert json.loads((OUT/'extraction-complete.json').read_text())['all_tokens_match_original_norm16']
    assert len(FEATURES)==22
    config=dict(features=FEATURES,role_spaces=engine.ROLE_SPACES,protocols=engine.PROTOCOLS,
                C=CS,scaling=True,screen_max_tokens_per_prompt=256,screen_candidates=176,
                shortlist_unique_features=3,refinement_anchors=['norm-16','norm-20'],
                solver=dict(penalty='l2',max_iter=5000,linesearch_max_iter=100,tol=1e-4,fit_intercept=True),
                source_probes_sha256=engine.digest(engine.HERE/'outputs/probes/gptoss-120b.pkl'),
                source_manifest_sha256=engine.digest(SOURCE/'manifest.json'),
                source_splits_sha256=engine.digest(SOURCE/'tuning/splits.json'),
                source_runner_sha256=engine.digest(Path(__file__)),
                source_pipeline_sha256=engine.digest(engine.HERE/'feature_pipeline_runtime.py'),
                note='All candidates are linear probes. Same fixed texts, system tags, role labels and original document/prompt splits. Previously reported tests are reused; no new external blind test.')
    atomic_json(OUT/'config.json',config)
    shutil.copyfile(SOURCE/'tuning/splits.json',OUT/'splits.json')
    status('prepared',features=len(FEATURES),screen_candidates=176)


def feature_array(feature,rows):
    import cupy as cp
    import numpy as np
    import torch
    n=json.loads((SOURCE/'manifest.json').read_text())['tokens']
    specs=FEATURES[feature]
    x=cp.empty((len(rows),2880*len(specs)),dtype=cp.float32,order='F')
    indices=rows.sample_ix.tolist()
    for col,spec in enumerate(specs):
        mmap=torch.from_file(str(feature_path(spec)),shared=False,size=n*2880,dtype=torch.float16).view(-1,2880)
        for start in range(0,len(rows),32768):
            stop=min(start+32768,len(rows))
            value=mmap[indices[start:stop]].float().to('cuda:0')
            block=cp.from_dlpack(value)
            x[start:stop,col*2880:(col+1)*2880]=block
            cp.cuda.get_current_stream().synchronize()
            del value,block
            torch.cuda.empty_cache()
        del mmap
    assert x.flags.f_contiguous and bool(cp.isfinite(x).all())
    return x


def data(feature,name,protocol,stage):
    import cupy as cp
    import numpy as np
    import pandas as pd
    from cuml.preprocessing import StandardScaler
    train,test=engine.datasets(name,protocol,'final' if stage=='final' else 'search')
    if stage=='screen':
        ids=[]
        for _,group in train.groupby('prompt_ix',sort=False):
            positions=np.linspace(0,len(group)-1,min(256,len(group)),dtype=int)
            ids.extend(group.index.to_numpy()[positions].tolist())
        train=train.loc[ids].sort_values('sample_ix')
    x=feature_array(feature,train)
    engine.cleanup()
    # Bound the temporary variance arrays for the 8640-dimensional probes.
    scaler=StandardScaler(copy=False)
    for start in range(0,len(train),16384):
        # cuML 25.10 reduces this count to a Python int after each chunk,
        # although its next partial_fit calls cupy.repeat on that value.
        if hasattr(scaler,'n_samples_seen_'):
            scaler.n_samples_seen_=cp.asarray(scaler.n_samples_seen_)
        scaler.partial_fit(x[start:start+16384])
    scaler.transform(x,copy=False)
    engine.cleanup()
    xt=feature_array(feature,test)
    scaler.transform(xt,copy=False)
    assert int(scaler.n_samples_seen_)==len(train)
    return train,test,x,xt,scaler


def fit_block(feature,name,protocol,stage,cs):
    folder=OUT/stage
    pending=[C for C in cs if not (folder/f'{protocol}-{name}-{feature}-{C:g}.json').exists()]
    if not pending:
        return
    train,test,x,xt,scaler=data(feature,name,protocol,stage)
    for C in pending:
        tag=f'{stage}-{protocol}-{name}-{feature}-{C:g}'
        result=engine.train_one(name,protocol,feature,True,C,'float32',
                               'feature-selected' if stage=='final' else 'search',
                               train,test,x,xt,scaler,tag)
        result.update(feature=feature,feature_spec=FEATURES[feature],search_stage=stage)
        if stage=='final':
            path=OUT/'models'/f'{tag}.pkl'
            with path.open('rb') as f:artifact=pickle.load(f)
            artifact.update(feature_name=feature,feature_spec=FEATURES[feature],tuning=result,
                            feature_note='Requires the named concatenated feature blocks; do not substitute the original single-layer normalized activations.')
            with path.open('wb') as f:pickle.dump(artifact,f)
        atomic_json(folder/f'{protocol}-{name}-{feature}-{C:g}.json',result)
        print(json.dumps(dict(stage=stage,protocol=protocol,role_space=name,feature=feature,C=C,
                              accuracy=result['evaluation']['accuracy'],seconds=result['seconds'])),flush=True)
        del result
        engine.cleanup()
    del x,xt,scaler
    engine.cleanup()


def screen_worker(job):
    for feature in list(FEATURES)[job::4]:
        for protocol in engine.PROTOCOLS:
            for name in engine.ROLE_SPACES:
                fit_block(feature,name,protocol,'screen',CS)


def shortlist():
    records=[json.loads(p.read_text()) for p in (OUT/'screen').glob('*.json')]
    assert len(records)==176
    selected={}
    for protocol in engine.PROTOCOLS:
        selected[protocol]={}
        for name in engine.ROLE_SPACES:
            candidates=[r for r in records if r['protocol']==protocol and r['role_space']==name]
            ranked=sorted(candidates,key=lambda r:(-r['evaluation']['accuracy'],r['evaluation']['nll'],len(r['feature_spec'])))
            features=[]
            for r in ranked:
                if r['feature'] not in features:
                    features.append(r['feature'])
                if len(features)==3:
                    break
            selected[protocol][name]=list(dict.fromkeys(features+['norm-16','norm-20']))
    atomic_json(OUT/'shortlist.json',selected)
    status('shortlisted',features=selected)


def refine_worker(job):
    protocol,name=[(p,n) for p in engine.PROTOCOLS for n in engine.ROLE_SPACES][job]
    candidates=json.loads((OUT/'shortlist.json').read_text())[protocol][name]
    for feature in candidates:
        subprocess.run([sys.executable,'-u',__file__,'--stage','block','--protocol',protocol,
                        '--rolespace',name,'--feature',feature,'--block-stage','refine'],check=True)


def select():
    selected={}
    records=[json.loads(p.read_text()) for p in (OUT/'refine').glob('*.json')]
    for protocol in engine.PROTOCOLS:
        selected[protocol]={}
        for name in engine.ROLE_SPACES:
            candidates=[r for r in records if r['protocol']==protocol and r['role_space']==name]
            best=min(candidates,key=lambda r:(-r['evaluation']['accuracy'],r['evaluation']['nll'],len(r['feature_spec'])))
            selected[protocol][name]=dict(feature=best['feature'],C=best['C'],validation=best['evaluation'])
    atomic_json(OUT/'selection.json',selected)
    status('selected',selected=selected)


def final_worker(job):
    protocol,name=[(p,n) for p in engine.PROTOCOLS for n in engine.ROLE_SPACES][job]
    selected=json.loads((OUT/'selection.json').read_text())[protocol][name]
    for feature,C in [(selected['feature'],selected['C']),('norm-20',.005)]:
        subprocess.run([sys.executable,'-u',__file__,'--stage','block','--protocol',protocol,
                        '--rolespace',name,'--feature',feature,'--block-stage','final','--C',str(C)],check=True)


def parallel(stage):
    workers=[]
    for job in range(4):
        log=(OUT/'logs'/f'{stage}-{job}.log').open('a')
        proc=subprocess.Popen([sys.executable,'-u',__file__,'--stage',stage,'--job',str(job)],
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job),HF_HOME='/root/.cache/huggingface-role-probe'),
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append((proc,log))
    try:
        while any(p.poll() is None for p,_ in workers):
            if any(p.returncode not in [None,0] for p,_ in workers):
                raise RuntimeError(f'{stage} failed; see feature-search/logs')
            time.sleep(3)
        assert all(p.returncode==0 for p,_ in workers)
    except BaseException:
        status('worker_failed',worker_stage=stage)
        for p,_ in workers:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise
    finally:
        for p,log in workers:
            if p.poll() is None:p.terminate()
            log.close()


def report():
    import pandas as pd
    rows=[]
    for stage in ['screen','refine','final']:
        for p in (OUT/stage).glob('*.json'):
            r=json.loads(p.read_text())
            rows.append(dict(stage=stage,protocol=r['protocol'],role_space=r['role_space'],feature=r['feature'],
                             dimensions=2880*len(r['feature_spec']),C=r['C'],train_accuracy=r['train']['accuracy'],
                             accuracy=r['evaluation']['accuracy'],nll=r['evaluation']['nll'],macro_f1=r['evaluation']['macro_f1'],
                             cot_recall=r['evaluation']['recall']['cot'],system_recall=r['evaluation']['recall']['system'],
                             seconds=r['seconds'],optimizer_warnings=len(r['optimizer_warnings'])))
    pd.DataFrame(rows).sort_values(['stage','protocol','role_space','feature','C']).to_csv(OUT/'metrics.csv',index=False)
    assert engine.digest(engine.HERE/'outputs/probes/gptoss-120b.pkl')==json.loads((OUT/'config.json').read_text())['source_probes_sha256']
    atomic_json(OUT/'complete.json',dict(screen_candidates=176,refinement_candidates=sum(r['stage']=='refine' for r in rows),
                final_fits=sum(r['stage']=='final' for r in rows),selected_from_validation=True,original_probes_unchanged=True))
    status('complete',records=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','screen-worker','shortlist','refine-worker','select','final-worker','report','block'],default='all')
    parser.add_argument('--job',type=int,default=0)
    parser.add_argument('--feature')
    parser.add_argument('--protocol',choices=engine.PROTOCOLS)
    parser.add_argument('--rolespace',choices=list(engine.ROLE_SPACES))
    parser.add_argument('--block-stage',choices=['refine','final'])
    parser.add_argument('--C',type=float)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for folder in ['screen','refine','final','models','logs']:(OUT/folder).mkdir(exist_ok=True)
    if args.stage=='all':
        if not (OUT/'config.json').exists():prepare()
        status('screening',workers=4,candidates=176)
        parallel('screen-worker')
        shortlist()
        status('refining',workers=4)
        parallel('refine-worker')
        select()
        status('final_refit',workers=4)
        parallel('final-worker')
        report()
    elif args.stage=='prepare':prepare()
    elif args.stage=='screen-worker':screen_worker(args.job)
    elif args.stage=='shortlist':shortlist()
    elif args.stage=='refine-worker':refine_worker(args.job)
    elif args.stage=='select':select()
    elif args.stage=='final-worker':final_worker(args.job)
    elif args.stage=='block':fit_block(args.feature,args.rolespace,args.protocol,args.block_stage,CS if args.block_stage=='refine' else [args.C])
    else:report()


if __name__=='__main__':main()
