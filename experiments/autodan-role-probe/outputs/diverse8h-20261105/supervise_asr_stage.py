#!/usr/bin/env python3
"""One-owner local ASR stage supervisor. No payload generation or mutation."""
import fcntl, hashlib, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[3]
RUNTIME=Path('/scratch/wz55191/venvs/chatinject-table1/bin/python')
NATIVE_PID=2996055
NATIVE_TICKS='548524111'
ALLOWED_TERMINAL={'audited_finite_catalog_exhausted','ASR_reserved_deadline'}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)
def emit(event,**values):
    print(json.dumps(dict(event=event,observed_unix=time.time(),**values),ensure_ascii=False),flush=True)
def process_state(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    tail=stat[stat.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=tail[19],state=tail[0])
def gpu_state():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=30)
    rows=[]
    for line in values.splitlines():
        index,used,total=map(int,line.split(','))
        rows.append(dict(index=index,used_MiB=used,total_MiB=total))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=30)
    pids=[int(line.strip()) for line in apps.splitlines() if line.strip()]
    return dict(gpus=rows,compute_pids=pids,released=len(rows)==2 and all(row['used_MiB']<1024 for row in rows) and not pids)
def note(status,**values):
    record=dict(status=status,supervisor_pid=os.getpid(),supervisor_process=process_state(os.getpid()),native_pid=NATIVE_PID,native_start_ticks=NATIVE_TICKS,updated_unix=time.time(),**values)
    write(BASE/'asr_supervisor_status.json',record)
    return record

def choose_final(checkpoint):
    source=BASE/'probe/interval_candidates.json'
    data=read(source)
    available=[]
    intervals=[]
    for item in data['candidates']:
        path=Path(item['score_path']).resolve()
        score=read(path)
        if score.get('status')!='complete' or score.get('complete') is not True or score.get('completed_cases')!=6:
            raise ValueError('An interval candidate is not a full complete native score')
        template=score['template']
        if template.count('{goal}')!=1:
            raise ValueError('Candidate immutable goal slot is invalid')
        value=float(score['fitness'])
        lower,upper=map(float,item['p_user_interval'])
        if not(0<=lower<upper<=1 and lower<=value and (value<upper or value==upper==1)):
            raise ValueError('Candidate P(user) lies outside its native interval')
        if any(not(upper<=a or lower>=b) for a,b in intervals):
            raise ValueError('Candidate intervals overlap')
        intervals.append((lower,upper))
        available.append(dict(template_path=str(path),score_path=str(path),p_user_interval=[lower,upper],native_fitness=value,raw_sha256=hashlib.sha256(template.encode()).hexdigest()))
    if len(available)<3:
        raise ValueError('Fewer than three nonempty native intervals are available')
    selected=sorted(sorted(available,key=lambda row:row['p_user_interval'][0],reverse=True)[:6],key=lambda row:row['p_user_interval'][0])
    snapshot=BASE/'asr_selection'
    snapshot.mkdir(exist_ok=True)
    for name,payload in [('interval_candidates.snapshot.json',source.read_bytes()),('probe_checkpoint.snapshot.json',(BASE/'probe/checkpoint.json').read_bytes())]:
        target=snapshot/name
        if target.exists() and target.read_bytes()!=payload:
            raise ValueError('ASR final selection snapshot already exists with different bytes')
        target.write_bytes(payload)
    final=BASE/'asr_candidates_final.json'
    if final.exists():
        raise ValueError('An existing final selection must be inspected; supervisor will not silently replace it')
    manifest=dict(schema='fixed-candidates-final-selection-v1',final=True,selection_metric='complete six-context P(user), highest six available nonoverlapping native bin maxima',selected_unix=time.time(),no_ASR_feedback=True,baseline_conditions_scheduled=False,native_terminal_status=checkpoint['status'],native_process=dict(pid=NATIVE_PID,start_ticks=NATIVE_TICKS),source_interval_snapshot=str(snapshot/'interval_candidates.snapshot.json'),source_interval_snapshot_sha256=sha(snapshot/'interval_candidates.snapshot.json'),checkpoint_snapshot_sha256=sha(snapshot/'probe_checkpoint.snapshot.json'),candidates=selected)
    write(final,manifest)
    selection=dict(status='final_candidates_frozen',selected_unix=manifest['selected_unix'],manifest=str(final),manifest_sha256=sha(final),conditions=len(selected),scheduled_episodes=389*len(selected),no_ASR_feedback=True,source_interval_sha256=sha(source),probe_terminal_status=checkpoint['status'])
    write(BASE/'asr_selection.json',selection)
    emit('asr_candidates_frozen',**selection)
    return final,selection

def run_cpu_step(command,logfile,label):
    emit('cpu_preflight_started',step=label,logfile=str(logfile))
    with Path(logfile).open('w') as stream:
        result=subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} failed, exit={result.returncode}; see {logfile}')
    emit('cpu_preflight_complete',step=label)

def main():
    lock=(BASE/'asr_stage.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (BASE/'asr_final/launch.json').exists():
        note('existing_asr_launch_detected_no_restart',launch=read(BASE/'asr_final/launch.json'))
        emit('existing_asr_launch_detected_no_restart')
        return 4
    previous=None
    while True:
        state=process_state(NATIVE_PID)
        same=state is not None and state['start_ticks']==NATIVE_TICKS and state['state'] not in ('Z','X')
        checkpoint=read(BASE/'probe/checkpoint.json')
        signature=(same,checkpoint.get('status'),checkpoint.get('complete_scores'),checkpoint.get('best_fitness'))
        if same:
            note('waiting_live_native_probe',known_native_process=state,checkpoint=checkpoint)
            if signature!=previous:
                emit('waiting_live_native_probe',native_process=state,checkpoint=checkpoint)
            previous=signature
            time.sleep(60)
            continue
        if checkpoint.get('status') not in ALLOWED_TERMINAL:
            note('native_ended_unexpected_status_no_asr',known_native_process=state,checkpoint=checkpoint)
            emit('native_ended_unexpected_status_no_asr',checkpoint=checkpoint,native_process=state)
            return 5
        gpus=gpu_state()
        if not gpus['released']:
            note('waiting_gpu_release_no_asr',known_native_process=state,checkpoint=checkpoint,gpus=gpus)
            emit('waiting_gpu_release_no_asr',gpus=gpus,checkpoint_status=checkpoint['status'])
            time.sleep(60)
            continue
        break
    final,selection=choose_final(checkpoint)
    out=BASE/'final_selection_preflight'
    out.mkdir(exist_ok=True)
    note('cpu_preflight',selection=selection)
    run_cpu_step([str(RUNTIME),str(REPO/'experiments/autodan-role-probe/evaluate_fixed_candidates.py'),'--candidates',str(final),'--output',str(out/'dry_run'),'--maximum-wrapper-tokens','16384','--dry-run'],out/'evaluate.log','fixed_candidates_dry_run')
    run_cpu_step([str(RUNTIME),str(REPO/'experiments/autodan-role-probe/audit_fixed_candidates.py'),'--run-root',str(out/'dry_run'),'--candidates',str(final),'--protocol-only','--output',str(out/'protocol_audit')],out/'audit.log','fixed_candidates_protocol_audit')
    if not gpu_state()['released']:
        note('gpu_occupied_after_preflight_no_asr',selection=selection,gpus=gpu_state())
        emit('gpu_occupied_after_preflight_no_asr')
        return 6
    if (BASE/'asr_final/launch.json').exists():
        note('existing_asr_launch_detected_no_restart',selection=selection)
        emit('existing_asr_launch_detected_no_restart')
        return 4
    script=BASE/'run_asr.sh'
    child=subprocess.Popen(['bash',str(script)],cwd=REPO)
    time.sleep(.2)
    asr_process=process_state(child.pid)
    started=dict(pid=child.pid,process=asr_process,started_unix=time.time(),manifest=str(final),manifest_sha256=sha(final),conditions=selection['conditions'],scheduled_episodes=selection['scheduled_episodes'],run_script_sha256=sha(script),generation_api_calls=0,no_baseline_rerun=True)
    write(BASE/'asr_process.json',started)
    note('asr_started',asr=started,selection=selection)
    emit('asr_started',**started)
    last_terminal=None
    while child.poll() is None:
        progress=read(BASE/'asr_final/progress.json') if (BASE/'asr_final/progress.json').exists() else None
        if progress and progress['terminal']!=last_terminal:
            last_terminal=progress['terminal']
            emit('asr_progress',progress=progress)
        note('asr_running',asr=started,progress=progress)
        time.sleep(60)
    code=child.returncode
    note('asr_process_terminal',asr=started,exit_code=code)
    emit('asr_process_terminal',exit_code=code,pid=child.pid)
    if code!=0:
        return code or 7
    run_cpu_step([str(RUNTIME),str(REPO/'experiments/autodan-role-probe/audit_fixed_candidates.py'),'--run-root',str(BASE/'asr_final'),'--candidates',str(final),'--output',str(BASE/'asr_final/audit_fixed')],BASE/'asr_final_audit.log','final_native_predicate_audit')
    note('asr_and_independent_audit_complete',asr=started,summary_path=str(BASE/'asr_final/summary.json'),audit_path=str(BASE/'asr_final/audit_fixed/audit.json'))
    emit('asr_and_independent_audit_complete',pid=child.pid)
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        note('supervisor_error_no_automatic_restart',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        emit('supervisor_error_no_automatic_restart',error=f'{type(error).__name__}: {error}')
        raise
