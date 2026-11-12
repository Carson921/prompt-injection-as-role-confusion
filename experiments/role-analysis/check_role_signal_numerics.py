"""Post-run numerical diagnostics; never select formal samples or a probe."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import run_role_signal_factorial as run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    scorer=run.load_module('_factorial_numerics_scorer',run.ROOT/'experiments/autodan-role-probe/scorer.py')
    collect,_,attention,classifier=scorer.dependencies()
    probe=classifier.load_probe(run.PROBE);assert run.digest(run.PROBE)==run.SHA
    os.environ['HF_HOME']=str(run.ROOT/'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR']='/tmp/wz55191-role-signal-triton'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    torch.set_num_threads(2);torch.manual_seed(run.SEED)
    net=probe.torch(device='cpu').eval()
    mean,scale=torch.from_numpy(probe.mean),torch.from_numpy(probe.scale)
    model,_=collect.load_model(run.MODEL);attention.install_tiled_attention(256)
    inputs={r['input_id']:r for r in run.jsonlines(out/'inputs.jsonl')}
    checks=json.loads((out/'forward-validation.json').read_text())['checks']
    results=[]
    started=time.monotonic()
    for check in checks:
        row=inputs[check['input_id']]
        with np.load(out/'readouts'/(row['input_id']+'.npz')) as z:
            ref={k:z[k] for k in ['features','logits','probabilities']}
        variants={'repeat':row['input_ids'],
                  'remove_end':row['input_ids'][:row['body_end']],
                  'replace_end_with_return':row['input_ids'][:-1]+[200002],
                  'replace_end_with_ordinary_token':row['input_ids'][:-1]+[87]}
        positions=np.arange(row['body_start'],row['body_end'])
        for label,ids in variants.items():
            native,_=collect.forward_capture(model,probe.metadata['heads'],ids,positions)
            features=native.to(torch.float16).reshape(-1,1024).numpy()
            with torch.inference_mode():
                logits=net((torch.from_numpy(features).float()-mean)/scale).numpy()
                probability=torch.from_numpy(logits).softmax(1).numpy()
            delta=features.astype(float)-ref['features'].astype(float)
            ldelta=logits.astype(float)-ref['logits'].astype(float)
            results.append(dict(input_id=row['input_id'],position=row['position'],variant=label,
                feature_max_abs=float(abs(delta).max()),feature_rmse=float(np.sqrt((delta**2).mean())),
                reference_feature_rms=float(np.sqrt((ref['features'].astype(float)**2).mean())),
                relative_feature_l2=float(np.linalg.norm(delta)/np.linalg.norm(ref['features'].astype(float))),
                logit_max_abs=float(abs(ldelta).max()),
                mean_margin_delta=float((ldelta[:,3]-ldelta[:,2]).mean()),
                probability_max_abs=float(abs(probability-ref['probabilities']).max()),
                mean_probability_delta={r:float((probability[:,j]-ref['probabilities'][:,j]).mean()) for j,r in enumerate(run.ROLES)},
                argmax_disagreement=float((probability.argmax(1)!=ref['probabilities'].argmax(1)).mean())))
            print(json.dumps(results[-1]),flush=True)
    run.write(out/'numerical-stability.json',dict(probe_sha256=run.SHA,diagnostic_code_sha256=run.digest(__file__),
        selection='Same three inputs fixed by original forward-validation, one per position; never used to select primary analysis.',
        seconds=time.monotonic()-started,checks=results))


if __name__=='__main__':main()
