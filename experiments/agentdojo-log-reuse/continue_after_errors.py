"""Run the frozen native engine, retaining unscored errors and continuing others.

The frozen run/bridge, inference functions, prompts, sampling limit and original
generation binding remain unchanged. This outer orchestration layer consumes
case-error events instead of letting them abort the entire selected-case batch.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import traceback

from reuse_common import *


def events(process,deferred,on_error,on_defer):
    for line in process.stdout:
        event=read_event(line)
        if event['event']=='request':
            cid=event['call_id'].rsplit('/',1)[0]
            if cid in deferred:
                on_defer(event)
                process.stdin.write(serial({'call_id':event['call_id'],
                    'error':'Deferred unscored case; original native error and token-limit evidence are preserved'})+'\n')
                process.stdin.flush()
                continue
        if event['event']=='case_complete' and event['status']=='error':
            on_error(event)
            yield serial({**event,'event':'case_error_preserved'})+'\n'
        else:yield line


def read_event(line):
    import json
    return json.loads(line)


class Process:
    def __init__(self,process,deferred,on_error,on_defer):
        self.process=process
        self.stdout=events(process,deferred,on_error,on_defer)

    def __getattr__(self,name):return getattr(self.process,name)


def main():
    binding=read(MISSING/'manifest.json')
    assert all(sha(MISSING/'provenance'/name)==digest for name,digest in binding['continuation_code_sha256'].items())
    deferred_path=ROOT/'deferred_native_errors.json'
    state=read(deferred_path)
    deferred=state['cases']
    selected=set(read(MISSING/'selected_case_ids.json'))
    assert set(deferred)<=selected
    validation=read(ROOT/'error_continuation_validation.json')
    assert validation['status']=='passed' and validation['source_sha256']==sha(Path(__file__))
    receipt={'format':'native-error-control-plane-v1','started_at':time.time(),
             'wrapper_sha256':sha(Path(__file__)),'frozen_run_sha256':sha(MISSING/'provenance/run.py'),
             'frozen_bridge_sha256':sha(MISSING/'provenance/continuation_bridge.py'),
             'native_manifest_sha256':sha(MISSING/'manifest.json'),'generation_binding':signature(binding),
             'policy':'Preserve every case error with null security/utility, defer previously failed cases without a model call, and continue the next case. No model output repair, new seed, token-limit increase or failed-score substitution.',
             'model_generation_functions_modified':False}
    write(ROOT/'error_continuation.json',receipt)

    def log(event):
        with (ROOT/'continuation_errors.log').open('a') as stream:stream.write(serial({'time':time.time(),**event})+'\n')

    def on_defer(event):
        path=Path(event['path'])
        log({'event':'previous_error_deferred','case_id':event['call_id'].rsplit('/',1)[0],
             'request_path':str(path.relative_to(ROOT)),'request_sha256':sha(path),'asr_label':None})

    def on_error(event):
        cid=event['case_id'];assert cid in selected
        path=MISSING/'cases'/(cid+'.json')
        record=read(path)
        assert record['status']=='error' and record['attacker_success'] is None and record['utility'] is None
        if cid not in deferred:
            deferred[cid]={'error':record['error'],'first_record_path':str(path.relative_to(ROOT)),
                           'first_record_sha256':sha(path),'asr_label':None,'observed_at':time.time()}
            preserved=ROOT/'native_error_evidence'/cid/'case.json'
            write(preserved,record)
            deferred[cid]['preserved_record_path']=str(preserved.relative_to(ROOT))
            deferred[cid]['preserved_record_sha256']=sha(preserved)
            write(deferred_path,{'policy':'Deferred native technical errors remain unscored; no output repair or failure labels',
                                 'cases':deferred})
        log({'event':'case_error_preserved','case_id':cid,'error':record['error'],'asr_label':None})

    class Subprocess:
        def __getattr__(self,name):return getattr(subprocess,name)

        def Popen(self,command,*args,**kwargs):
            process=subprocess.Popen(command,*args,**kwargs)
            if len(command)>1 and command[1]==str(MISSING/'provenance/continuation_bridge.py'):
                return Process(process,deferred,on_error,on_defer)
            return process

    # Import and call the actual unchanged frozen main. Only its subprocess
    # event transport is wrapped; the source and generation cache binding stay
    # verifiable against every already-completed episode and model call.
    sys.path.insert(0,str(MISSING/'provenance'))
    spec=importlib.util.spec_from_file_location('frozen_native_error_continuation',MISSING/'provenance/run.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    frozen.subprocess=Subprocess()
    frozen.main()


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
