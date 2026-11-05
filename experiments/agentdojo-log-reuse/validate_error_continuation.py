"""Verify errors remain null and do not abort other cases or trigger a decode."""
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from reuse_common import *
from continue_after_errors import events


def main():
    bad='remember_delimiter/slack/user_task_7/injection_task_1'
    other='remember_delimiter/slack/user_task_7/injection_task_2'
    request={'event':'request','call_id':bad+'/call_0000','path':'unused'}
    error={'event':'case_complete','case_id':bad,'status':'error','attacker_success':None,'utility':None,'error':'original error'}
    next_request={'event':'request','call_id':other+'/call_0000','path':'unused'}
    complete={'event':'case_complete','case_id':other,'status':'completed','attacker_success':False,'utility':True}
    lines=[serial(v)+'\n' for v in [request,error,next_request,complete]]
    process=SimpleNamespace(stdout=StringIO(''.join(lines)),stdin=StringIO())
    failures=[];skipped=[]
    result=list(events(process,{bad:{}},failures.append,skipped.append))
    reply=read_event(process.stdin.getvalue())
    assert reply['call_id']==request['call_id'] and 'error' in reply and 'generated_ids' not in reply
    assert len(skipped)==1 and failures==[error]
    converted=read_event(result[0]);assert converted['event']=='case_error_preserved'
    assert converted['attacker_success'] is None and converted['utility'] is None
    assert result[1:]==lines[2:],'Unrelated requests and completed outcomes must pass through byte-for-byte'
    # A newly encountered error is also retained as an error event; it never
    # becomes a completed case or consumes the remaining batch.
    process=SimpleNamespace(stdout=StringIO(lines[1]+lines[2]+lines[3]),stdin=StringIO())
    result=list(events(process,{},failures.append,skipped.append))
    assert len(result)==3 and read_event(result[0])['event']=='case_error_preserved'
    assert result[1:]==lines[2:] and not process.stdin.getvalue()
    write(ROOT/'error_continuation_validation.json',{'status':'passed','source_sha256':sha(Path(__file__).with_name('continue_after_errors.py')),
           'tests':['previous native error is deferred without invoking inference',
                    'new case errors preserve null scores and allow subsequent cases',
                    'unrelated requests and completed outcomes pass through byte-for-byte'],
           'frozen_inference_files_modified':False})
    print(serial({'status':'passed','tests':3}))


def read_event(line):
    import json
    return json.loads(line)


if __name__=='__main__':main()
