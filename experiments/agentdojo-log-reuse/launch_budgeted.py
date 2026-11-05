"""Resume within a new allocation, with a new explicit GPU time budget."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import subprocess
import time

from reuse_common import *
from checkpoint_state import process_info,native_status,prepare_resume


def remaining_allocation():
    job=os.environ.get('SLURM_JOB_ID')
    if not job:return None
    raw=subprocess.check_output(['squeue','-h','-j',job,'-o','%L'],text=True).strip()
    split=raw.split('-');days=int(split[0]) if len(split)>1 else 0
    fields=list(map(int,split[-1].split(':')))
    return days*86400+sum(v*m for v,m in zip(reversed(fields),(1,60,3600)))


def launch(script,arguments,logname,receiptname):
    with (ROOT/logname).open('a') as output:
        process=subprocess.Popen([NATIVE_PYTHON,'-u',str(script),*arguments],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    fields=Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()
    record={'pid':process.pid,'start_ticks':int(fields[19]),'script':str(script),
            'slurm_job':os.environ.get('SLURM_JOB_ID'),'started_at':time.time()}
    write(ROOT/receiptname,record)
    return process,record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget-minutes',type=int,default=120)
    parser.add_argument('--keep-deadline',action='store_true',help='Continue within the already-authorized absolute deadline')
    args=parser.parse_args()
    assert args.budget_minutes>0
    lock=(ROOT/'launch-management.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if ROOT.with_suffix('.tar.gz.complete.json').exists():
        print('Verified full result already exists.');return
    for name in ('launch.json','budget_guard-launch.json','finisher-launch.json'):
        path=ROOT/name
        if path.exists() and process_info(read(path)) is not None:
            raise RuntimeError(f'An existing managed process is still active: {name}')
    now=time.time();remaining=remaining_allocation()
    deadline=min(now+args.budget_minutes*60,now+remaining) if remaining is not None else now+args.budget_minutes*60
    if args.keep_deadline:
        deadline=min(deadline,read(ROOT/'gpu_budget.json')['budget_deadline_epoch'])
    assert deadline-now>600,'Less than ten minutes remain; obtain a fresh GPU allocation first'
    checkpoint,value=prepare_resume()
    if not value['pending_asr']:
        # All actual case verdicts exist; audit them before reconstructing a
        # terminal receipt if interruption happened after the last case commit.
        subprocess.run([BENCHMARK_PYTHON,'-u',str(MISSING/'postprocessing/audit_export.py')],check=True)
        write(MISSING/'producer-terminal.json',{'status':'terminal','completed':220,'requested':220,
              'recovered_from_independently_audited_case_files':True,'finished_at':time.time()})
        subprocess.run([NATIVE_PYTHON,'-u',str(ROOT/'provenance/finish.py')],check=True);return
    if not (MODEL/'staging-proof.json').exists():
        subprocess.run([NATIVE_PYTHON,str(BASE/'provenance/stage_model.py'),
                        '--selection',str(BASE/'provenance/head-selection-16.json')],check=True)
    assert time.time()<deadline-300,'Model staging used the available time; checkpoint remains safe'
    entry=ROOT/'provenance/continue_after_errors.py' if (ROOT/'deferred_native_errors.json').exists() else MISSING/'provenance/run.py'
    native,worker=launch(entry,[],'run.log','launch.json')
    config={'format':'agentdojo-gpu-budget-v1','worker':worker,'slurm_job':os.environ.get('SLURM_JOB_ID'),
            'budget_deadline_epoch':deadline,'checkpoint_start_epoch':deadline-300,
            'graceful_case_wait_seconds':60,'force_stop_after_seconds':15,'postprocess_timeout_seconds':180,
            'automatic_continuation_held':True,'created_at':now,'checkpoint_utc':datetime.fromtimestamp(deadline-300,timezone.utc).isoformat(),
            'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),'time_basis':'Earlier of this explicitly requested budget and the current Slurm allocation expiry'}
    write(ROOT/'gpu_budget.json',config)
    finisher,_=launch(ROOT/'provenance/finish.py',['--watch'],'finish.log','finisher-launch.json')
    guard,_=launch(ROOT/'provenance/budget_guard.py',[],'budget_guard.log','budget_guard-launch.json')
    print(serial({'stage':'resumed','worker':worker,'deadline_utc':config['deadline_utc'],
                  'checkpoint_utc':config['checkpoint_utc'],'pending_asr':value['pending_asr']}),flush=True)
    result=native.wait()
    if result==0 and not native_status()['pending']:
        assert finisher.wait()==0,'Final audit/packaging failed; inspect finish.log'
        guard.wait(timeout=40)
    else:
        guard.wait(timeout=300)
        state=read(ROOT/'budget_guard_status.json')
        assert state['stage']=='checkpoint_saved',state
        print(serial(state),flush=True)
        if state['stop_reason']!='GPU time budget reached':
            raise RuntimeError('Native worker exited early; its checkpoint and error evidence were retained')


if __name__=='__main__':main()
