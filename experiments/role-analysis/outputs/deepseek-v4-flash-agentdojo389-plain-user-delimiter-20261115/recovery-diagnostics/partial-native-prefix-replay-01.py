"""Independent CPU recovery diagnostic; stop at first uncommitted model request."""
from __future__ import annotations
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO/'experiments/role-analysis'))
import deepseek_agentdojo_native_audit as audit
import deepseek_agentdojo_protocol as protocol


class PendingBoundary(BaseException):
    pass


def run(root):
    root = Path(root).resolve()
    identity, cases = audit.read(root/'run_identity.json'), audit.read(root/'schedule.json')
    bindings = {str(p.relative_to(root)): audit.sha(p) for p in
                (root/'run_identity.json', root/'schedule.json', root/'server_identity.json', root/'generation-backend.json')}
    audit.require(identity['mode']=='full' and len(cases)==778, 'Expected original formal 778 scope')
    audit.require(audit.object_sha(cases)==identity['schedule_sha256'], 'Canonical schedule identity changed')
    audit.require(bindings['server_identity.json']==identity['server_identity_sha256'], 'Server identity changed')
    native = audit.dependencies(Path.home()/'Github/delimeter-token')
    audit.require(cases==protocol.native_cases(native['get_suite']), 'Pinned canonical cases changed')
    source = Path(native['agentdojo'].__file__).resolve().parent
    for name,digest in identity['agentdojo_source']['files_sha256'].items():
        audit.require(audit.sha(source.parent/name)==digest, 'Pinned native source changed: '+name)
    audit.require(audit.sha(Path(protocol.__file__))==identity['protocol_sha256'], 'Protocol source changed')
    selected = []
    for case in cases:
        first=root/'bridge_calls'/case['case_id']/'call_0000/receipt.json'
        if first.is_file():
            audit.require(not (root/'cases'/(case['case_id']+'.json')).exists(), 'Diagnostic expects incomplete cases')
            selected.append(case)
    audit.require(len(selected)==6, 'Expected the six originally committed partial episodes')
    result = {'schema':'deepseek-independent-partial-recovery-v1','status':'incomplete','complete':False,
        'full_benchmark_passed':False,'scheduled_cases':778,'terminal_cases_at_snapshot':0,
        'partial_episodes_checked':0,'committed_model_completions_replayed':0,
        'model_generation_performed':False,'network_calls_attempted':0,'started_unix':time.time(),
        'diagnostic_source_sha256':audit.sha(__file__),'frozen_native_audit_source_sha256':audit.sha(audit.__file__),
        'protocol_source_sha256':audit.sha(protocol.__file__),'episodes':[]}
    for case in selected:
        case_id=case['case_id'];base=root/'bridge_calls'/case_id
        committed=[]
        index=0
        while (base/f'call_{index:04d}/receipt.json').is_file():
            p=base/f'call_{index:04d}/receipt.json'
            committed.append({'call_index':index,'call_id':case_id+f'/call_{index:04d}',
                'bridge_receipt':{'path':str(p.relative_to(root)),'sha256':audit.sha(p)}})
            index+=1
        requests=audit.saved_calls(root,case,{'spec_sha256':audit.object_sha(identity),'requests':committed},identity,bindings)
        for entry in committed:
            receipt=audit.read(root/entry['bridge_receipt']['path'])
            for ref in receipt['artifacts']:
                audit.bound_path(root,ref,bindings)
        pending_path=base/f'call_{index:04d}/original_request.json'
        audit.require(pending_path.is_file(),'Expected original pending boundary request')
        bindings[str(pending_path.relative_to(root))]=audit.sha(pending_path)
        pending=audit.read(pending_path)
        audit.require(not (pending_path.parent/'receipt.json').exists(),'Pending boundary became committed')
        trace_path=root/'raw/deepseek_v4_flash_role_probe'/case['suite']/case['user_task_id']/protocol.ATTACK_NAMES[case['condition']]/(case['injection_task_id']+'.json')
        trace=audit.read(trace_path) if trace_path.is_file() else None
        if trace is not None:bindings[str(trace_path.relative_to(root))]=audit.sha(trace_path)
        evidence,passes=audit.email_evidence([*requests,pending],trace)
        clocks=[];tool_counts={};boundary={}
        suite=native['get_suite'](protocol.BENCHMARK_VERSION,case['suite'])
        task=suite.user_tasks[case['user_task_id']]
        injection=suite.get_injection_task_by_id(case['injection_task_id'])
        audit.require(injection.GOAL==case['actual_goal'],'Native injection goal changed')

        class Capture(native['Logger']):
            def __init__(self):self.messages=[];self.errors=[];self.logdir=None
            def log(self,messages,**kwargs):self.messages=json.loads(json.dumps(messages,default=lambda value:value.model_dump()))
            def log_error(self,error):self.errors.append(str(error))

        class Replay(native['BasePipelineElement']):
            name='independent_partial_native_prefix_replay'
            def __init__(self):self.index=0;self.current=None;self.email_index=0
            def query(self,query,runtime,env,messages=(),extra_args=None):
                i=self.index;call_id=case_id+f'/call_{i:04d}'
                metadata={**case,'call_index':i,'call_id':call_id,'recording_dir':str(root/'calls'/call_id),
                    **identity['generation']}
                tools=[{'type':'function','function':{'name':tool.name,'description':tool.description,
                    'parameters':tool.parameters.model_json_schema()}}for tool in runtime.functions.values()]
                reconstructed={'messages':audit.independent_messages(messages),'tools':tools,'metadata':metadata}
                expected=requests[i] if i<len(requests) else pending
                original={key:expected[key]for key in ('messages','tools','metadata')}
                audit.require(reconstructed==original,f'Resumed semantic original request differs at {case_id} call {i}')
                audit.require(audit.canonical(reconstructed).encode()==audit.canonical(original).encode(),'Canonical request bytes differ')
                original_path=base/f'call_{i:04d}/original_request.json'
                pretty=(json.dumps(reconstructed,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
                audit.require(pretty==original_path.read_bytes(),'Original request serialized bytes differ')
                if i==len(requests):
                    boundary.update(call_index=i,call_id=call_id,original_request_path=str(original_path.relative_to(root)),
                        original_request_sha256=audit.sha(original_path),raw_request_sha256=audit.object_sha(reconstructed),
                        semantic_object_equal=True,canonical_bytes_equal=True,original_pretty_json_bytes_equal=True,
                        response_committed=False,stopped_before_generation=True)
                    raise PendingBoundary()
                self.index+=1;self.current=i;self.email_index=0
                audit.validate_model_error(expected,identity)
                response=audit.independent_assistant(copy.deepcopy(expected['response']),call_id,native)
                return query,runtime,env,[*messages,response],extra_args or {}

        llm=Replay()
        class Runtime(native['FunctionsRuntime']):
            def run_function(self,env,function,kwargs,raise_on_error=False):
                tool_counts[function]=tool_counts.get(function,0)+1
                if function!='send_email':return super().run_function(env,function,kwargs,raise_on_error=raise_on_error)
                audit.require(llm.current is not None,'Email outside saved completion')
                response=audit.independent_assistant(copy.deepcopy(requests[llm.current]['response']),requests[llm.current]['call_id'],native)
                emails=[call for call in response['tool_calls']or[]if call.function=='send_email']
                audit.require(llm.email_index<len(emails),'Extra native email call')
                call=emails[llm.email_index];llm.email_index+=1;key=(passes[llm.current],call.id)
                audit.require(key in evidence,'Missing original email timestamp')
                stamp=audit.email_timestamp(evidence[key]['message']);clock=audit.BoundEmailClock(stamp)
                with patch.object(native['email_client'],'datetime',SimpleNamespace(datetime=clock)):
                    value,error=super().run_function(env,function,kwargs,raise_on_error=raise_on_error)
                audit.require(not clock.denied and clock.calls==int(stamp is not None),'Native email clock use changed')
                clocks.append({'tool_call_id':call.id,'timestamp':None if stamp is None else stamp.isoformat(),'tool_result_replaced':False})
                return value,error
        pipeline=native['AgentPipeline'].from_config(native['PipelineConfig'](llm=llm,model_id=None,defense=None,
            tool_delimiter='tool',system_message_name=None,system_message=None,tool_output_format=None,provider_config=None))
        import yaml
        formatters=0
        for element in pipeline.elements:
            for child in getattr(element,'elements',()):
                if isinstance(child,native['ToolsExecutor']):
                    child.output_formatter=lambda value:native['tool_result_to_str'](value,dump_fn=partial(yaml.safe_dump,allow_unicode=True))
                    formatters+=1
        audit.require(formatters==1 and identity['tool_formatter']['allow_unicode']is True,'Native formatter changed')
        attack=native['FixedJailbreakAttack'](case['payload_template'],suite,pipeline)
        injections=attack.attack(task,injection)
        audit.require(injections and all(v==case['payload_text']for v in injections.values()),'Native attack injection differs')
        if trace is not None:audit.require(trace['injections']==injections,'Saved injection slots differ')
        capture=Capture();token=native['LOGGER_STACK'].set([]);reached=False
        try:
            with capture:
                try:suite.run_task_with_pipeline(pipeline,task,injection,injections,runtime_class=Runtime)
                except PendingBoundary:reached=True
        finally:native['LOGGER_STACK'].reset(token)
        audit.require(reached and boundary and llm.index==len(requests),'Replay did not stop exactly at first pending request')
        result['episodes'].append({'case_id':case_id,'status':'incomplete_at_verified_pending_boundary',
            'native_prefix_valid':True,'committed_completions_replayed':llm.index,'actual_native_tools_executed':tool_counts,
            'clock_bindings':clocks,'tool_results_replaced':False,'utility':None,'security':None,'boundary':boundary})
        result['partial_episodes_checked']+=1;result['committed_model_completions_replayed']+=llm.index
    for rel,digest in bindings.items():audit.require(audit.sha(root/rel)==digest,'Source changed during diagnostic: '+rel)
    result.update(diagnostic_replay_valid=True,safe_to_resume_existing_plan=True,finished_unix=time.time(),
                  request_receipt_trace_sources_sha256=bindings)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();attempted=[]
    def reject_network(*args,**kwargs):attempted.append(True);raise audit.AuditError('Network forbidden in partial native replay')
    try:
        with patch.object(socket.socket,'connect',reject_network):result=run(args.run)
    except Exception as error:
        result={'status':'failed','complete':False,'diagnostic_replay_valid':False,'error':f'{type(error).__name__}: {error}',
                'network_calls_attempted':len(attempted),'model_generation_performed':False}
    audit.write(args.output,result)
    print(json.dumps({k:result.get(k)for k in ('status','complete','diagnostic_replay_valid','partial_episodes_checked',
        'committed_model_completions_replayed','safe_to_resume_existing_plan','error')},ensure_ascii=False))
    if not result.get('diagnostic_replay_valid'):raise SystemExit(1)


if __name__=='__main__':main()
