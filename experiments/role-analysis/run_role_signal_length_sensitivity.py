"""Post-hoc native-runtime sequence-length sensitivity, primary data unchanged."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import run_role_signal_factorial as run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;dest=out/'numerical-length-sensitivity';dest.mkdir(exist_ok=True)
    original=json.loads((out/'protocol.json').read_text())
    assert run.digest(run.PROBE)==run.SHA==original['probe_sha256']
    assert run.digest(out/'inputs.jsonl')==original['inputs_sha256']
    rows=[r for r in run.jsonlines(out/'inputs.jsonl') if r['split']=='formal']
    assert len(rows)==1200
    altered=[]
    for r in rows:
        assert r['input_ids'][-1]==200007 and r['body_end']==len(r['input_ids'])-1
        changed=dict(r);changed['input_ids']=r['input_ids'][:-1];changed['rendered_prompt']=r['rendered_prompt'][:-len('<|end|>')]
        altered.append(changed)
    run.dump_lines(dest/'inputs.jsonl',altered)
    run.write(dest/'protocol.json',dict(
        primary_protocol_sha256=run.digest(out/'protocol.json'),primary_inputs_sha256=run.digest(out/'inputs.jsonl'),
        altered_inputs_sha256=run.digest(dest/'inputs.jsonl'),probe_sha256=run.SHA,
        code_sha256=run.digest(__file__),analysis_code_sha256=run.digest(run.HERE/'analyze_role_signal_length_sensitivity.py'),
        purpose='Post-hoc diagnostic prompted by numerical shape differences when removing the single future ending token.',
        operation='Remove exactly final<|end|> ID200007 after body from every formal input; all body IDs/positions and preceding tokens fixed.',
        primary_unchanged=True,formal_inputs=1200,model_backend='Same frozen native MXFP4 / tiled eager / batch1',
        bootstrap='Same2000 paired semantic-unit resamples, seed20261112, for primary-vs-alternate differences.'))
    scorer=run.load_module('_factorial_length_scorer',run.ROOT/'experiments/autodan-role-probe/scorer.py')
    collect,_,attention,classifier=scorer.dependencies()
    probe=classifier.load_probe(run.PROBE)
    os.environ['HF_HOME']=str(run.ROOT/'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR']='/tmp/wz55191-role-signal-triton'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    torch.set_num_threads(2);torch.manual_seed(run.SEED)
    net=probe.torch(device='cpu').eval();mean,scale=torch.from_numpy(probe.mean),torch.from_numpy(probe.scale)
    started=time.monotonic();model,_=collect.load_model(run.MODEL);attention.install_tiled_attention(256)
    summaries=[];(dest/'readouts').mkdir(exist_ok=True)
    for i,r in enumerate(altered):
        path=dest/'readouts'/(r['input_id']+'.npz');positions=np.arange(r['body_start'],r['body_end'])
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                logits=z['logits'];probability=z['probabilities']
        else:
            native,_=collect.forward_capture(model,probe.metadata['heads'],r['input_ids'],positions)
            features=native.to(torch.float16).reshape(-1,1024).numpy()
            with torch.inference_mode():
                logits=net((torch.from_numpy(features).float()-mean)/scale).numpy()
                probability=torch.from_numpy(logits).softmax(1).numpy()
            assert np.isfinite(logits).all() and np.allclose(probability.sum(1),1,atol=1e-6)
            temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,positions=positions,logits=logits,probabilities=probability,roles=np.asarray(run.ROLES),
                                **({'features':features} if i%120==0 else {}));temp.replace(path)
        summaries.append({k:r[k] for k in ['input_id','pair_id','split','filler_id','text_style','wrapper','position','body_tokens']}|
            dict(mean_margin=float((logits[:,3].astype(float)-logits[:,2]).mean()),
                 **{'p_'+role:float(probability[:,j].astype(float).mean()) for j,role in enumerate(run.ROLES)},
                 **{'z_'+role:float(logits[:,j].astype(float).mean()) for j,role in enumerate(run.ROLES)}))
        if (i+1)%60==0 or i+1==len(altered):
            pd.DataFrame(summaries).to_csv(dest/'input-summary.csv',index=False)
            status=dict(done=i+1,total=1200,seconds=time.monotonic()-started)
            run.write(dest/'status.json',status);print(json.dumps(status),flush=True)
    run.write(dest/'run-manifest.json',dict(status='complete',inputs=1200,seconds=time.monotonic()-started,
        summary_sha256=run.digest(dest/'input-summary.csv'),inputs_sha256=run.digest(dest/'inputs.jsonl'),
        probe_sha256=run.SHA,runner_sha256=run.digest(__file__),
        readout_sha256={p.name:run.digest(p) for p in sorted((dest/'readouts').glob('*.npz'))}))


if __name__=='__main__':main()
