"""Frozen six-role probe on first-role interventions of genuine benign traces.

Preparation and inference are separate. The pilot must be reported before all
traces are prepared. Inference never generates text or executes archived tools.
All 36 residual layers are saved for the scored target body token positions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

import agentdojo_trace_interventions_data as data
import run_role_signal_factorial as shared

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ROLES=data.ROLES
DEFAULT_OUTPUT=HERE/'outputs/agentdojo-trace-interventions-20261112'
PROBE=shared.PROBE
MODEL=shared.MODEL
SHA=shared.SHA
SEED=20261112


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def write(path,value):
    shared.write(path,value)


def prepare(args):
    out=args.output
    stage=out/args.stage
    stage.mkdir(parents=True,exist_ok=True)
    assert not (stage/'protocol.json').exists(),'Prepared inputs already exist; use run to resume'
    assert digest(args.probe)==SHA
    records=shared.jsonlines(out/'sources/traces.jsonl')
    assert len({r['trace_id'] for r in records})==len(records)
    tok=shared.tokenizer(args.model)
    inventories=[]
    for trace in records:
        blocks=data.parse_native_blocks(trace['input_ids'],tok,allow_terminal_body=trace.get('allow_terminal_body',False))
        targets=data.first_role_targets(blocks)
        assert set(targets)==set(ROLES)
        inventories.append(dict(trace_id=trace['trace_id'],source_split=trace['source_split'],
            tokens=len(trace['input_ids']),blocks=len(blocks),
            first_body_tokens={role:len(block['body_ids']) for role,block in targets.items()},
            allow_terminal_body=trace.get('allow_terminal_body',False)))
    candidates=[r for r in inventories if r['source_split']=='fit']
    pilot=min(candidates,key=lambda r:(r['tokens'],r['trace_id']))['trace_id']
    if args.trace_id:
        assert args.stage=='pilot'
        assert args.trace_id in {r['trace_id'] for r in candidates}
        pilot=args.trace_id
    if args.stage=='pilot':
        selected=[r for r in records if r['trace_id']==pilot]
    else:
        feedback=out/'pilot/feedback.json'
        assert feedback.is_file(),'Report the completed single-trace pilot before preparing all traces'
        note=json.loads(feedback.read_text())
        assert note['pilot_run_manifest_sha256']==digest(out/'pilot/run-manifest.json')
        assert note['feedback_delivered'] is True
        assert len(records)==57 and Counter(r['source_split'] for r in records)=={'fit':40,'dev':8,'test':9}
        selected=sorted(records,key=lambda r:(r['trace_id']!=pilot,r['source_split']!='fit',r['trace_id']))
    rows=[]
    for trace in selected:
        variants=data.build_interventions(trace,tok)
        assert len(variants)==43
        for row in variants:
            row['source_split']=trace['source_split']
            row['trace_sha256']=object_sha(trace['input_ids'])
            assert len(row['input_ids'])<=131072,'Do not truncate long traces'
            assert len(row['target_positions'])>0
        rows.extend(variants)
    shared.dump_lines(stage/'inputs.jsonl',rows)
    write(stage/'protocol.json',dict(
        stage=args.stage,created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        probe_sha256=digest(args.probe),expected_probe_sha256=SHA,probe=str(args.probe.resolve()),
        backbone='openai/gpt-oss-120b',model_path=str(args.model),roles=list(ROLES),
        source_records_sha256=digest(out/'sources/traces.jsonl'),inputs_sha256=digest(stage/'inputs.jsonl'),
        parser_sha256=digest(HERE/'agentdojo_trace_interventions_data.py'),runner_sha256=digest(__file__),
        runtime_code_sha256={str(path.relative_to(ROOT)):digest(path) for path in [
            ROOT/'experiments/agentdojo-first-tool/collect.py',ROOT/'experiments/agentdojo-first-tool/attention.py',
            ROOT/'experiments/autodan-role-probe/scorer.py',HERE/'synthetic_agent_classifier.py']},
        pilot_trace_id=pilot,selected_traces=[r['trace_id'] for r in selected],
        trace_split_counts=dict(Counter(r['source_split'] for r in selected)),inputs=len(rows),
        interventions_per_trace=dict(baseline=1,strip=6,swap=30,move=6),
        first_occurrence='First observed ordinary body per six-role class; assistant tool-call JSON remains context, never an assistant target. The single audited length-censored terminal assistant body is kept without adding a closing token.',
        strip='Remove complete native opening wrapper and its closing marker, retaining identical body IDs.',
        swap='Copy exact native header and closing marker from first destination-role body of the same trace; five other roles.',
        move='Move original whole wrapper+body to farthest end by absolute body-start token displacement; tie to beginning.',
        context='Full chronological retrospective trajectory including raw generated CoT; not a claim that later live requests retained previous CoT.',
        measurement='Body token mean six-class softmax and direct logits; same body tokens across interventions; trace-equal aggregate.',
        hiddenstates=dict(scope='Scored target body positions; baseline union of first six roles',
            layers=list(range(36)),embedding=True,final_norm=True,native_dtype='BF16',format='safetensors',
            probe_head_features='Ordered16 pre-o_proj heads; native BF16 and FP16 checkpoint input both saved'),
        execution=dict(seed=SEED,batch_size=1,attention='original eager with256-query tiling',eval=True,dropout=False,
                       truncation=False,tool_execution=False,text_generation=False),
        numerical_limit='Fixed native MXFP4/BF16 readouts may depend on input shape; no claim of backend-independent numerical invariance.',
        inventory=inventories))
    assert digest(args.probe)==SHA
    print(json.dumps(dict(stage='prepared',selection=args.stage,inputs=len(rows),pilot_trace_id=pilot)),flush=True)


class ResidualCapture:
    """Passive hooks store body rows without materializing full-sequence histories."""
    def __init__(self,model,positions):
        self.model,self.positions=model,np.asarray(positions,dtype=np.int64)
        self.values,self.handles={},[]

    def hook(self,name):
        def receive(module,args,output):
            import torch
            assert name not in self.values
            value=output[0] if isinstance(output,tuple) else output
            assert isinstance(value,torch.Tensor) and value.ndim==3 and value.shape[0]==1 and value.shape[2]==2880
            indices=torch.as_tensor(self.positions,device=value.device,dtype=torch.long)
            selected=value.detach()[0].index_select(0,indices).cpu().contiguous()
            assert selected.dtype==torch.bfloat16 and bool(torch.isfinite(selected).all())
            self.values[name]=selected
        return receive

    def __enter__(self):
        self.handles.append(self.model.model.embed_tokens.register_forward_hook(self.hook('embedding')))
        for layer,module in enumerate(self.model.model.layers):
            self.handles.append(module.register_forward_hook(self.hook(f'residual_layer_{layer:02d}')))
        self.handles.append(self.model.model.norm.register_forward_hook(self.hook('final_norm')))
        return self

    def __exit__(self,*error):
        for handle in self.handles:handle.remove()

    def finish(self):
        expected={'embedding','final_norm'}|{f'residual_layer_{i:02d}' for i in range(36)}
        assert set(self.values)==expected,'Not all hiddenstate hooks fired'
        return self.values


def run(args):
    import torch
    import pandas as pd
    from safetensors.torch import save_file
    out=args.output;stage=out/args.stage
    protocol=json.loads((stage/'protocol.json').read_text())
    assert digest(stage/'inputs.jsonl')==protocol['inputs_sha256']
    assert digest(__file__)==protocol['runner_sha256']
    assert digest(HERE/'agentdojo_trace_interventions_data.py')==protocol['parser_sha256']
    assert all(digest(ROOT/name)==sha for name,sha in protocol['runtime_code_sha256'].items())
    assert digest(args.probe)==protocol['probe_sha256']==SHA
    assert digest(out/'sources/traces.jsonl')==protocol['source_records_sha256']
    scorer=shared.load_module('_trace_interventions_scorer',ROOT/'experiments/autodan-role-probe/scorer.py')
    collect,_,attention,classifier=scorer.dependencies()
    probe=classifier.load_probe(args.probe)
    assert tuple(probe.metadata['roles'])==ROLES
    binding=probe.metadata['model_binding']
    for name,sha in binding['assets_sha256'].items():assert digest(args.model/name)==sha
    proof=json.loads((args.model/'staging-proof.json').read_text())
    assert proof['all_shards_verified'] and {r['name']:r['sha256'] for r in proof['shards']}==binding['weight_shards_sha256']
    os.environ['HF_HOME']=str(ROOT/'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR']='/tmp/wz55191-agentdojo-trace-interventions-triton'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    torch.set_num_threads(2);torch.manual_seed(SEED)
    net=probe.torch(device='cpu').eval()
    mean,scale=torch.from_numpy(probe.mean),torch.from_numpy(probe.scale)
    rows=shared.jsonlines(stage/'inputs.jsonl')
    started=time.monotonic();model,device_map=collect.load_model(args.model)
    attention.install_tiled_attention(256)
    write(stage/'runtime.json',dict(device_map=device_map,hardware=[torch.cuda.get_device_name(i) for i in range(2)],
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','kernels','triton','safetensors','numpy','pandas']},
        model_load_seconds=time.monotonic()-started,staging_proof_sha256=digest(args.model/'staging-proof.json')))
    for folder in ['readouts','hiddenstates','records']:(out/folder).mkdir(exist_ok=True)
    summaries=[];cached=0;records=[]
    for index,row in enumerate(rows):
        input_id=row['input_id'];record_path=out/'records'/(input_id+'.json')
        readout_path=out/'readouts'/(input_id+'.npz');hidden_path=out/'hiddenstates'/(input_id+'.safetensors')
        signature=dict(input_ids_sha256=object_sha(row['input_ids']),positions_sha256=object_sha(row['target_positions']),
                       original_positions_sha256=object_sha(row['target_original_positions']),trace_sha256=row['trace_sha256'],
                       runtime_code_sha256=protocol['runtime_code_sha256'],
                       probe_sha256=SHA,parser_sha256=protocol['parser_sha256'],runner_sha256=protocol['runner_sha256'])
        t=time.monotonic()
        if record_path.exists():
            record=json.loads(record_path.read_text())
            assert record['signature']==signature and record['status']=='complete'
            assert hidden_path.is_file() and digest(readout_path)==record['readout_sha256']
            assert hidden_path.stat().st_size==record['hiddenstate_bytes']
            with np.load(readout_path,allow_pickle=False) as z:
                logits=z['logits'];probabilities=z['probabilities'];positions=z['positions']
            cached+=1
        else:
            positions=np.asarray(row['target_positions'],np.int64)
            with torch.inference_mode(),ResidualCapture(model,positions) as states:
                native,_=collect.forward_capture(model,probe.metadata['heads'],row['input_ids'],positions)
                residuals=states.finish()
            assert native.dtype==torch.bfloat16
            features=native.to(torch.float16).reshape(len(positions),1024).numpy()
            with torch.inference_mode():
                logits=net((torch.from_numpy(features).float()-mean)/scale).numpy()
                probabilities=torch.from_numpy(logits).softmax(1).numpy()
            assert np.isfinite(logits).all() and np.allclose(probabilities.sum(1),1,atol=1e-6)
            if args.stage=='pilot' and row['operation']=='baseline':
                repeat,_=collect.forward_capture(model,probe.metadata['heads'],row['input_ids'],positions)
                assert torch.equal(native,repeat),'Passive residual hooks changed feature values'
                write(stage/'passive-hook-validation.json',dict(status='passed',input_id=input_id,
                    selected_head_values_bit_exact=True,hiddenstate_tensors=38,body_tokens=len(positions)))
            target_ids=np.asarray(row['input_ids'],np.int64)[positions]
            original_positions=np.asarray(row['target_original_positions'],np.int64)
            temp=readout_path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,positions=positions,original_positions=original_positions,
                token_ids=target_ids,head_features=features,logits=logits,probabilities=probabilities,roles=np.asarray(ROLES))
            temp.replace(readout_path)
            residuals.update(pre_o_proj_heads_bf16=native.contiguous(),
                probe_features_fp16=torch.from_numpy(features),positions=torch.from_numpy(positions.copy()),
                original_positions=torch.from_numpy(original_positions.copy()),token_ids=torch.from_numpy(target_ids.copy()))
            temp_hidden=hidden_path.with_suffix('.tmp.safetensors')
            save_file(residuals,str(temp_hidden),metadata=dict(
                input_id=input_id,trace_id=row['trace_id'],scope='Scored body positions; residual output after each block',
                native_dtype='BF16',signature=json.dumps(signature,sort_keys=True)))
            temp_hidden.replace(hidden_path)
            record=dict(status='complete',input_id=input_id,trace_id=row['trace_id'],signature=signature,
                readout_sha256=digest(readout_path),hiddenstates_sha256=digest(hidden_path),
                hiddenstate_bytes=hidden_path.stat().st_size,hiddenstate_tensors=len(residuals),
                hiddenstate_layers=36,body_tokens=len(positions),input_tokens=len(row['input_ids']),
                seconds=time.monotonic()-t,hiddenstates=str(hidden_path.relative_to(out)),readout=str(readout_path.relative_to(out)))
            write(record_path,record)
            del residuals,states,native,features
        records.append(record)
        for role,role_positions in row['role_target_positions'].items():
            offsets=np.searchsorted(positions,np.asarray(role_positions))
            assert np.array_equal(positions[offsets],role_positions)
            p=probabilities[offsets].astype(float);z=logits[offsets].astype(float)
            original_positions=row['original_role_target_positions'][role]
            destination=row['destination_role'] if row['operation']=='swap' else role if row['operation'] in ('baseline','move') else ''
            summaries.append(dict(input_id=input_id,trace_id=row['trace_id'],source_split=row['source_split'],
                operation=row['operation'],source_role=role,destination_role=destination,
                body_tokens=len(offsets),original_body_start=original_positions[0],new_body_start=role_positions[0],
                total_input_tokens=len(row['input_ids']),mean_original_role_probability=float(p[:,ROLES.index(role)].mean()),
                **{'p_'+r:float(p[:,i].mean()) for i,r in enumerate(ROLES)},
                **{'z_'+r:float(z[:,i].mean()) for i,r in enumerate(ROLES)},
                argmax_original_role_fraction=float((p.argmax(1)==ROLES.index(role)).mean())))
        if args.stage=='pilot' or (index+1)%43==0 or index+1==len(rows):
            pd.DataFrame(summaries).to_csv(stage/'input-summary.csv',index=False)
        status=dict(stage='forward',selection=args.stage,done=index+1,total=len(rows),trace_id=row['trace_id'],
                    operation=row['operation'],source_role=row['source_role'],cached=cached,seconds=time.monotonic()-started)
        write(stage/'status.json',status)
        print(json.dumps(status),flush=True)
    write(stage/'run-manifest.json',dict(status='complete',stage=args.stage,inputs=len(rows),cached_inputs=cached,
        traces=len(protocol['selected_traces']),summary_rows=len(summaries),seconds=time.monotonic()-started,
        summary_sha256=digest(stage/'input-summary.csv'),protocol_sha256=digest(stage/'protocol.json'),
        inputs_sha256=digest(stage/'inputs.jsonl'),probe_sha256=SHA,
        hiddenstate_bytes=sum(r['hiddenstate_bytes'] for r in records),
        records={r['input_id']:dict(readout_sha256=r['readout_sha256'],hiddenstates_sha256=r['hiddenstates_sha256']) for r in records}))
    print(json.dumps(dict(stage='complete',selection=args.stage,inputs=len(rows))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--stage',choices=['pilot','all'],required=True)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--probe',type=Path,default=PROBE)
    p.add_argument('--trace-id')
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a)


if __name__=='__main__':main()
