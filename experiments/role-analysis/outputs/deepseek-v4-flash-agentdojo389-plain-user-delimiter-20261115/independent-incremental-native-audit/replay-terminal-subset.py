"""CPU-only immutable terminal-subset replay; never marks full778 complete."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[5]/'experiments/role-analysis'
sys.path.insert(0,str(SOURCE))
import deepseek_agentdojo_native_audit as audit
import deepseek_agentdojo_protocol as protocol


def run(root, case_ids):
    root=Path(root).resolve();started=time.time()
    identity,cases=audit.read(root/'run_identity.json'),audit.read(root/'schedule.json')
    audit.require(identity['mode']=='full' and len(cases)==778,'Expected complete formal schedule')
    audit.require(identity['schedule_sha256']==audit.object_sha(cases),'Schedule object binding changed')
    audit.require(len(set(case_ids))==len(case_ids)>0,'Subset case IDs must be unique and nonempty')
    by_id={case['case_id']:case for case in cases}
    audit.require(set(case_ids)<=set(by_id),'Subset contains unscheduled case')
    snapshot={case_id:audit.sha(root/'cases'/(case_id+'.json')) for case_id in case_ids}
    records={case_id:audit.read(root/'cases'/(case_id+'.json')) for case_id in case_ids}
    original_full_audit=audit.sha(root/'native-evaluation-audit.json') if(root/'native-evaluation-audit.json').exists()else None
    identity_sha,schedule_sha=audit.sha(root/'run_identity.json'),audit.sha(root/'schedule.json')
    audit.require(audit.sha(root/'server_identity.json')==identity['server_identity_sha256'],'Server identity changed')
    native=audit.dependencies(Path.home()/'Github/delimeter-token')
    package=Path(native['agentdojo'].__file__).resolve().parent
    audit.require(subprocess.check_output(['git','-C',str(package),'rev-parse','HEAD'],text=True).strip()==audit.REVISION,'Pinned source revision changed')
    audit.require(not subprocess.check_output(['git','-C',str(package),'status','--porcelain','--untracked-files=no'],text=True).strip(),'Pinned tracked source changed')
    for name,digest in identity['agentdojo_source']['files_sha256'].items():
        audit.require(audit.sha(package.parent/name)==digest,'Native source changed: '+name)
    audit.require(cases==protocol.native_cases(native['get_suite']),'Canonical native schedule differs')
    audit.require(audit.sha(protocol.__file__)==identity['protocol_sha256'],'Protocol source changed')
    bindings={};rows=[]
    for case_id in case_ids:
        row=audit.replay_case(root,by_id[case_id],records[case_id],identity,native,protocol,bindings)
        audit.require(row['valid'],'Native subset replay did not validate')
        for call in records[case_id]['requests']:
            receipt=audit.read(audit.bound_path(root,call['bridge_receipt'],bindings))
            for ref in receipt['artifacts']:
                audit.bound_path(root,ref,bindings)
        rows.append(row)
    audit.require(all(audit.sha(root/'cases'/(case_id+'.json'))==digest for case_id,digest in snapshot.items()),'Terminal case mutated during replay')
    audit.require(audit.sha(root/'run_identity.json')==identity_sha and audit.sha(root/'schedule.json')==schedule_sha,'Frozen run identity mutated')
    for name,digest in bindings.items():audit.require(audit.sha(root/name)==digest,'Call or trace changed during replay: '+name)
    current_full_audit=audit.sha(root/'native-evaluation-audit.json')if(root/'native-evaluation-audit.json').exists()else None
    audit.require(current_full_audit==original_full_audit,'Full native-audit artifact changed during subset check')
    return {'schema':'deepseek-independent-incremental-native-audit-v1','status':'subset_verified',
        'scope':'explicit immutable terminal subset only','complete':False,'full778_complete':False,
        'scheduled_cases':778,'audited_subset_cases':len(rows),'model_requests_replayed':sum(row['native_calls_replayed']for row in rows),
        'model_generation_performed':False,'network_calls_attempted':0,'full_native_audit_unchanged':True,
        'started_unix':started,'finished_unix':time.time(),'run_identity_sha256':identity_sha,'schedule_sha256':schedule_sha,
        'server_identity_sha256':identity['server_identity_sha256'],'cases_sha256':snapshot,'native_results':rows,
        'score_comparison':[{k:row[k]for k in ('case_id','utility','security','valid')}for row in rows],
        'score_comparison_equal_to_original_cases':True,'request_and_trace_files_sha256':bindings,
        'source':{'diagnostic_sha256':audit.sha(__file__),'frozen_native_audit_sha256':audit.sha(audit.__file__),
            'protocol_sha256':audit.sha(protocol.__file__),'agentdojo_revision':audit.REVISION,
            'agentdojo_files_sha256':identity['agentdojo_source']['files_sha256']}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case-id',action='append',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();attempted=[]
    def deny_network(*args,**kwargs):attempted.append(True);raise audit.AuditError('Network forbidden in independent subset replay')
    try:
        with patch.object(socket.socket,'connect',deny_network):result=run(args.run,args.case_id)
    except Exception as error:
        result={'status':'failed','complete':False,'full778_complete':False,'model_generation_performed':False,
                'error':f'{type(error).__name__}: {error}','network_calls_attempted':len(attempted)}
    audit.write(args.output,result)
    print(json.dumps({k:result.get(k)for k in ('status','complete','full778_complete','audited_subset_cases','model_requests_replayed','error')},ensure_ascii=False))
    if result['status']!='subset_verified':raise SystemExit(1)


if __name__=='__main__':main()
