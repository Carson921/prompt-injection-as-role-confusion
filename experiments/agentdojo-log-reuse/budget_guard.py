"""Node-side deadline guard; checkpointing does not depend on app wake-up timing."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from reuse_common import *
from checkpoint_state import process_info,stop_bound_process,native_status,verify_frozen_binding,create_checkpoint,idle_worker_lock


def update(stage,**fields):
    value={'stage':stage,'pid':os.getpid(),'updated_at':time.time(),**fields}
    write(ROOT/'budget_guard_status.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    config=read(ROOT/'gpu_budget.json')
    binding=verify_frozen_binding()
    validation=read(ROOT/'checkpoint_guard_validation.json')
    assert validation['status']=='passed' and validation['checkpoint_state_sha256']==sha(Path(__file__).with_name('checkpoint_state.py'))
    assert config['checkpoint_start_epoch']<config['budget_deadline_epoch']
    worker=config['worker']
    if args.validate:
        print(serial({'status':'valid','worker':process_info(worker),'binding':binding,
              'seconds_until_checkpoint':config['checkpoint_start_epoch']-time.time(),
              'checkpoint_utc':config['checkpoint_utc'],'deadline_utc':config['deadline_utc']}));return
    lock=(ROOT/'budget_guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=config['budget_deadline_epoch']
    while True:
        status=native_status()
        info=process_info(worker)
        if not status['pending']:
            update('all_native_cases_complete',completed_asr=778,pending_asr=0)
            print(serial({'event':'all_native_complete','time':time.time()}),flush=True);return
        if info is None or time.time()>=config['checkpoint_start_epoch']:break
        update('watching',checkpoint_start_epoch=config['checkpoint_start_epoch'],budget_deadline_epoch=deadline,
               completed_asr=558+len(status['completed']),pending_asr=len(status['pending']),worker_pid=worker['pid'])
        time.sleep(min(15,max(0,config['checkpoint_start_epoch']-time.time())))
    reason='GPU time budget reached' if info else 'Native worker exited before completion'
    update('checkpoint_requested',reason=reason,budget_deadline_epoch=deadline)
    write(ROOT/'checkpoint_requested.json',{'reason':reason,'time':time.time(),'worker':worker,'deadline':deadline})
    # Allow the currently active case up to one minute to commit its verdict.
    initial=set(status['completed']);end=min(time.time()+config.get('graceful_case_wait_seconds',60),deadline-180)
    while process_info(worker) is not None and time.time()<end:
        now=native_status()
        if set(now['completed'])!=initial or not now['pending']:break
        time.sleep(1)
    stopped=stop_bound_process(worker,group=True,timeout=config.get('force_stop_after_seconds',15))
    finisher_path=ROOT/'finisher-launch.json'
    if finisher_path.exists():
        stop_bound_process(read(finisher_path),timeout=5)
    with idle_worker_lock():
        path,checkpoint=create_checkpoint(reason,idle=True,deep=True)
    update('checkpoint_saved',checkpoint=str(path.relative_to(ROOT)),completed_asr=checkpoint['completed_asr'],
           pending_asr=checkpoint['pending_asr'],worker_stop=stopped)
    # The resumable checkpoint is already committed. CPU verification/export may
    # improve the current labels, but cannot prevent the checkpoint from existing.
    post={}
    commands=[('native_partial_audit',[BENCHMARK_PYTHON,'-u',str(MISSING/'postprocessing/audit_export.py'),'--partial']),
              ('partial_export',[NATIVE_PYTHON,'-u',str(Path(__file__).with_name('export_dataset.py')),'--partial'])]
    for name,command in commands:
        remaining=deadline-time.time()-15
        if remaining<=5:post[name]={'status':'deferred','reason':'checkpoint already saved; preserve budget'};continue
        try:
            with (path.parent/(name+'.log')).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=min(config.get('postprocess_timeout_seconds',180),remaining),check=False)
            post[name]={'status':'passed' if result.returncode==0 else 'error','exit_code':result.returncode}
        except subprocess.TimeoutExpired:post[name]={'status':'deferred','reason':'time budget; checkpoint unaffected'}
    write(path.parent/'postprocessing.json',post)
    final={'stage':'checkpoint_saved','completed_asr':checkpoint['completed_asr'],'pending_asr':checkpoint['pending_asr'],
           'total':778,'checkpoint':str(path.relative_to(ROOT)),'postprocessing':post,'worker_stopped':True,
           'automatic_continuation_held':True,'updated_at':time.time(),'resume_command':checkpoint['resume_command'],'stop_reason':reason}
    write(ROOT/'progress.json',final)
    update('checkpoint_saved',**{k:v for k,v in final.items() if k not in ('stage','updated_at')})
    print(serial(final),flush=True)


if __name__=='__main__':main()
