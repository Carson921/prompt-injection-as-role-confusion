"""Collect all 41 no-payload IPI continuations as original native token streams.

Scenario tools are never executed. The target runs locally in the checkpoint's
native two-H100 backend; only the clean WorldSim LLM receives tool requests.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import ipi_benign_source_data as source
import run_role_signal_factorial as shared
from run_agentdojo_trace_interventions import digest, object_sha, write

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DEFAULT=HERE/'outputs/ipi-benign-trace-interventions-20261117'
MODEL=Path('/tmp/wz55191-agentdojo-first-tool-model')


def emit(**data):
    print(json.dumps(data,ensure_ascii=False),flush=True)


def trace_from_record(record):
    ids=[];segments=[]
    for step in record['steps']:
        if not ids:
            incoming=step['input_ids']
            segments.append(dict(kind='initial_request',step=step['step'],start=0,end=len(incoming),
                                 request_start=0,request_end=len(incoming)))
            ids.extend(incoming)
        elif step.get('new_request_tail'):
            incoming=step['new_request_tail']
            start=len(ids);ids.extend(incoming)
            segments.append(dict(kind='new_tool_and_open_header',step=step['step'],start=start,end=len(ids),
                                 request_start=step['new_request_tail_start'],request_end=len(step['input_ids'])))
        start=len(ids);ids.extend(step['generated_ids'])
        segments.append(dict(kind='raw_generated',step=step['step'],start=start,end=len(ids),
                             generated_start=0,generated_end=len(step['generated_ids'])))
    assert sum(s['end']-s['start'] for s in segments)==len(ids)
    last=record['steps'][-1]
    return dict(trace_id=record['trace_id'],source_split='external',category=record['category'],
                input_ids=ids,allow_terminal_body=last['finish_reason']=='length',
                eof_finish_reason=last['finish_reason'],
                provenance=dict(collection_case=record['case_id'],collection_signature=record['signature'],
                    prepared_sha256=record['prepared_sha256'],source_commit=source.SOURCE_COMMIT,
                    segments=segments,raw_generated_tokens=sum(len(x['generated_ids']) for x in record['steps']),
                    raw_generated_tokens_preserved_exactly_once=True,terminal_reason=record['terminal_reason'],
                    context='Retrospective fixture-prefix plus raw native continuation and first-visible tool returns; not the final live request.',
                    real_tool_execution=False,empty_injection_slot=True))


def collect_case(prepared,record_path,model,tok,collect,config,simulator):
    import ipi_trace_interventions_data as data
    signature=object_sha(dict(config=config,prepared_sha256=digest(prepared)))
    if record_path.exists():
        cached=json.loads(record_path.read_text())
        assert cached['signature']==signature,'Collection specification changed; preserve the old run.'
        if cached['status']=='complete':return cached,simulator
        # Preserve the unsuccessful attempt before retrying the same fixed case.
        backup=record_path.with_name(record_path.stem+f'.attempt-{time.time_ns()}.json')
        record_path.replace(backup)
    item=json.loads(prepared.read_text());messages=copy.deepcopy(item['messages']);tools=item['tools']
    record=dict(status='running',case_id=item['case_id'],trace_id=item['trace_id'],category=item['category'],
                signature=signature,prepared_sha256=digest(prepared),steps=[],terminal_reason=None,
                initial_adaptations=item['adaptations'],real_tool_execution=False,error=None)
    previous_prompt=None;started=time.monotonic()
    try:
        for step_number in range(1,config['max_steps']+1):
            request=source.render_input(tok,messages,tools,config['date']);ids=request['initial_input_ids']
            assert len(ids)+config['max_new_tokens']<=131072,'Refuse context truncation'
            if step_number==1:assert ids==item['initial_input_ids']
            blocks=data.parse_native_blocks(ids,tok)
            assert not blocks[-1]['complete'],'Expected the actual native generation header'
            prefix=ids[blocks[-1]['start']:]
            step=dict(step=step_number,input_ids=ids,input_ids_sha256=object_sha(ids),
                      generation_prefix_ids=prefix,worldsim=[],messages=copy.deepcopy(messages))
            if previous_prompt is not None:
                assert blocks[-2]['role']=='tool','The newly visible frame must be the simulated tool response'
                start=blocks[-2]['start'];step.update(new_request_tail_start=start,new_request_tail=ids[start:])
            emit(event='decode_start',case_id=item['case_id'],step=step_number,input_tokens=len(ids))
            t=time.monotonic();generated=collect.generate(model,ids,config['max_new_tokens'])
            finish='stop' if generated and generated[-1] in (200002,200012,199999) else 'length'
            step.update(generated_ids=generated,generated_text=tok.decode(generated,skip_special_tokens=False),
                        finish_reason=finish,decode_seconds=time.monotonic()-t)
            record['steps'].append(step);write(record_path,record)
            emit(event='decode_complete',case_id=item['case_id'],step=step_number,
                 generated_tokens=len(generated),seconds=step['decode_seconds'],finish_reason=finish)
            parsed=source.parse_native_response(generated,tok,generation_prefix_ids=prefix,finish_reason=finish)
            step['parsed_response']=parsed
            calls=parsed['tool_calls']
            if finish=='length':
                record['terminal_reason']='max_new_tokens';break
            if not calls:
                record['terminal_reason']='final_response';break
            assert len(calls)==1,'The native template supports one tool call per message; never drop parallel calls'
            call=calls[0];names={t['function']['name'] for t in tools}
            if call['name'] not in names:
                record['terminal_reason']='unknown_tool_name';break
            call['id']=f"call_{step_number:04d}"
            prior_ids={prior['id'] for message in messages for prior in message.get('tool_calls',[])}
            assert call['id'] not in prior_ids,'Opaque continuation call ID collides with a history ID'
            assistant=dict(role='assistant',content='',tool_calls=[dict(id=call['id'],type='function',
                function=dict(name=call['name'],arguments=call['arguments']))])
            thinking='\n'.join(x for x in (parsed.get('reasoning'),parsed.get('content')) if x)
            if thinking:assistant['thinking']=thinking
            messages.append(assistant)
            if step_number==config['max_steps']:
                record['terminal_reason']='max_steps';break
            if simulator is None:
                simulator=source.WorldSimClient(model=config['worldsim_model'],max_tokens=config['worldsim_max_tokens'],
                    output_char_budget=config['worldsim_requested_output_character_budget'])
            # A simulated environment sees visible history, not the target's private analysis.
            visible=[{k:v for k,v in message.items() if k not in ('thinking','reasoning','reasoning_content')}
                     for message in messages]
            result=None
            for attempt in range(4):
                result=simulator.simulate(tools,visible,call);step['worldsim'].append(result);write(record_path,record)
                if result['status']=='complete':break
                if attempt<3:time.sleep(2*(attempt+1))
            if result is None or result['status']!='complete':
                raise RuntimeError('Clean WorldSim failed; no fabricated success response is allowed')
            text=result['tool_response']
            assert not any(x in text for x in source.SPECIAL),'WorldSim emitted a native control spelling'
            messages.append(dict(role='tool',name=call['name'],tool_call_id=call['id'],content=text))
            previous_prompt=ids
        record.update(status='complete',duration_seconds=time.monotonic()-started,messages=messages)
        trace=trace_from_record(record)
        data.parse_native_blocks(trace['input_ids'],tok,allow_terminal_body=trace['allow_terminal_body'])
        record['trace']=trace;write(record_path,record)
    except Exception as error:
        record.update(status='error',duration_seconds=time.monotonic()-started,
                      error=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc())
        write(record_path,record)
        raise
    return record,simulator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['pilot','all'],required=True)
    p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--max-new-tokens',type=int,default=4096)
    p.add_argument('--max-steps',type=int,default=5);p.add_argument('--worldsim-model',default=source.WORLDSIM_MODEL)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared=json.loads((out/'prepared/manifest.json').read_text());assert prepared['cases']==41
    selection_path=out/'prepared/pilot-selection.json'
    selection=json.loads(selection_path.read_text())
    pilot=selection['case_id']
    assert selection['prepared_manifest_sha256']==digest(out/'prepared/manifest.json')
    if a.stage=='all':
        feedback=json.loads((out/'pilot/feedback.json').read_text())
        assert feedback['feedback_delivered'] and feedback['pilot_run_manifest_sha256']==digest(out/'pilot/run-manifest.json')
    config=dict(date=prepared['date'],max_steps=a.max_steps,max_new_tokens=a.max_new_tokens,
                target='native MXFP4 GPT-OSS-120B greedy',worldsim_model=a.worldsim_model,
                worldsim_max_tokens=source.WORLDSIM_MAX_TOKENS,
                worldsim_requested_output_character_budget=source.WORLDSIM_OUTPUT_CHAR_BUDGET,
                worldsim_policy=source.worldsim_policy(),
                tool_call_id_policy='Opaque call_0001..call_0005 by step; no case names; collisions with fixture/history IDs are rejected.',
                prepared_manifest_sha256=digest(out/'prepared/manifest.json'),probe_sha256=shared.SHA,
                pilot_case_id=pilot,pilot_selection_sha256=digest(selection_path),
                code_sha256={str(f.relative_to(REPO)):digest(f) for f in [Path(__file__),HERE/'ipi_benign_source_data.py',
                    HERE/'ipi_trace_interventions_data.py',REPO/'experiments/agentdojo-first-tool/collect.py',
                    REPO/'experiments/agentdojo-first-tool/attention.py']},
                live_history='Original native template with parsed tool calls and thinking; completed previous analysis may be omitted by template.',
                worldsim='Visible clean history only, no fixture attack metadata, no real tool execution, no success fallback')
    root=out/'collection';root.mkdir(exist_ok=True)
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==config
    else:write(root/'protocol.json',config)
    os.environ['HF_HOME']=str(REPO/'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR']='/tmp/wz55191-ipi-benign-triton';os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    import torch
    torch.set_num_threads(2);torch.manual_seed(0)
    scorer=shared.load_module('_ipi_collect_scorer',REPO/'experiments/autodan-role-probe/scorer.py')
    collect,_,attention,classifier=scorer.dependencies()
    probe=classifier.load_probe(shared.PROBE);assert digest(shared.PROBE)==shared.SHA
    binding=probe.metadata['model_binding'];proof=json.loads((a.model/'staging-proof.json').read_text())
    assert proof['all_shards_verified'] and {r['name']:r['sha256'] for r in proof['shards']}==binding['weight_shards_sha256']
    for name,sha in binding['assets_sha256'].items():assert digest(a.model/name)==sha
    tok=shared.tokenizer(a.model);model,device_map=collect.load_model(a.model)
    backend=root/'backend-validation';backend.mkdir(exist_ok=True)
    collect.validate(model,tok,probe.metadata['heads'],backend,256)
    write(root/'runtime.json',dict(device_map=device_map,staging_proof_sha256=digest(a.model/'staging-proof.json'),
        gpus=[torch.cuda.get_device_name(i) for i in range(2)],
        packages={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','kernels','triton']}))
    selected=[r for r in prepared['rows'] if a.stage=='all' or r['case_id']==pilot]
    selected.sort(key=lambda r:(r['case_id']!=pilot,r['case_id']))
    records=[];simulator=None
    try:
        for i,item in enumerate(selected):
            case_path=root/'cases'/(item['case_id']+'.json');case_path.parent.mkdir(exist_ok=True)
            assert digest(out/item['path'])==item['sha256']
            record,simulator=collect_case(out/item['path'],case_path,model,tok,collect,config,simulator)
            records.append(record)
            emit(event='case_complete',stage=a.stage,done=i+1,total=len(selected),case_id=item['case_id'],
                 trace_tokens=len(record['trace']['input_ids']),terminal_reason=record['terminal_reason'])
    finally:
        if simulator is not None:simulator.close()
    sources=out/'sources';sources.mkdir(exist_ok=True)
    path=sources/('pilot-traces.jsonl' if a.stage=='pilot' else 'traces.jsonl')
    shared.dump_lines(path,[r['trace'] for r in records])
    write(sources/(a.stage+'-collection-manifest.json'),dict(status='complete',stage=a.stage,
        traces=len(records),pilot_case_id=pilot,source_records_path=str(path.relative_to(out)),
        source_records_sha256=digest(path),collection_protocol_sha256=digest(root/'protocol.json'),
        cases={r['case_id']:dict(sha256=digest(root/'cases'/(r['case_id']+'.json')),
            trace_tokens=len(r['trace']['input_ids']),terminal_reason=r['terminal_reason']) for r in records}))
    emit(event='collection_complete',stage=a.stage,traces=len(records),source_file=str(path))


if __name__=='__main__':main()
