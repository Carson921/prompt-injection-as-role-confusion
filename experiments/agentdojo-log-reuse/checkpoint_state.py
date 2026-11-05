"""Persistent case/call checkpoints without changing the frozen inference binding."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import time

from reuse_common import BASE, ROOT, MISSING, read, write, sha, serial, signature


def process_info(record):
    path=Path('/proc')/str(record['pid'])
    if not path.exists():return None
    try:
        fields=(path/'stat').read_text().rpartition(') ')[2].split()
        ticks=int(fields[19])
        if ticks!=record['start_ticks'] or fields[0]=='Z':return None
        raw=(path/'cmdline').read_bytes()
        if not raw.rstrip(b'\0'):return None  # exited between stat and cmdline reads
        command=raw.split(b'\0')
        assert str(record['script']).encode() in command, 'PID command does not match the launch receipt'
        return {'pid':record['pid'],'start_ticks':ticks,'state':fields[0],'pgid':os.getpgid(record['pid'])}
    except FileNotFoundError:return None


def stop_bound_process(record,group=False,timeout=15):
    info=process_info(record)
    if info is None:return {'already_exited':True}
    if group:
        assert info['pgid']==record['pid'],'Worker must own its process group'
        os.killpg(record['pid'],signal.SIGTERM)
        if info['state']=='T':os.killpg(record['pid'],signal.SIGCONT)
    else:
        os.kill(record['pid'],signal.SIGTERM)
    end=time.monotonic()+timeout
    while process_info(record) is not None and time.monotonic()<end:time.sleep(.2)
    forced=process_info(record) is not None
    if forced:
        if group:os.killpg(record['pid'],signal.SIGKILL)
        else:os.kill(record['pid'],signal.SIGKILL)
    return {'terminated':True,'forced_kill':forced,'pid':record['pid']}


def native_status(root=ROOT):
    missing=root/'missing_asr'
    selected=read(missing/'selected_case_ids.json')
    completed={}
    errors={}
    for cid in selected:
        path=missing/'cases'/(cid+'.json')
        if not path.exists():continue
        result=read(path)
        assert result['case_id']==cid
        if result['status']=='completed':
            assert type(result['attacker_success']) is bool and type(result['utility']) is bool
            completed[cid]={'path':str(path.relative_to(root)),'sha256':sha(path),
                            'attacker_success':result['attacker_success'],'utility':result['utility']}
        else:errors[cid]={'status':result['status'],'error':result.get('error')}
    return {'selected':selected,'completed':completed,'pending':[c for c in selected if c not in completed],
            'errors':errors}


def verify_frozen_binding(root=ROOT):
    missing=root/'missing_asr';manifest=read(missing/'manifest.json')
    assert manifest['selected_case_ids_sha256']==sha(missing/'selected_case_ids.json')
    assert all(sha(missing/'provenance'/name)==digest for name,digest in manifest['continuation_code_sha256'].items())
    merged=read(root/'manifest.json')
    assert merged['total_cases']==778 and merged['delimiter_only']
    assert merged['plan_sha256']==sha(root/'plan.json')
    assert set(read(missing/'selected_case_ids.json'))==set(read(root/'missing_case_ids.json'))
    assert len(read(missing/'selected_case_ids.json'))==merged['missing_asr_cases']==220
    return signature(manifest)


def quarantine_pending(root,status,checkpoint_dir):
    """Keep unfinished traces out of active audit paths; retain exact-input caches."""
    missing=root/'missing_asr';moved=[]
    for cid in status['pending']:
        for category in ('calls','trajectories'):
            source=missing/category/cid
            if source.exists():
                target=checkpoint_dir/'unfinished'/cid/category
                target.parent.mkdir(parents=True,exist_ok=True)
                source.rename(target)
                moved.append({'from':str(source.relative_to(root)),'to':str(target.relative_to(root))})
        source=missing/'cases'/(cid+'.json')
        if source.exists():
            assert read(source)['status']!='completed'
            target=checkpoint_dir/'unfinished'/cid/'incomplete_case.json'
            target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
            moved.append({'from':str(source.relative_to(root)),'to':str(target.relative_to(root))})
    return moved


@contextmanager
def idle_worker_lock(root=ROOT):
    lock=(root/'missing_asr/run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('Inference worker is still active; refusing to move its files')
    try:yield
    finally:lock.close()


def create_checkpoint(reason,*,idle=False,deep=False,root=ROOT):
    binding=verify_frozen_binding(root)
    status=native_status(root)
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(time.time_ns()%1000000000)
    directory=root/'checkpoints'/name;directory.mkdir(parents=True)
    moved=quarantine_pending(root,status,directory) if idle else []
    cache={}
    for path in sorted((root/'missing_asr/generation_cache').glob('*.json')):
        value=read(path)
        assert value['binding']==binding
        assert path.stem==signature({'binding':binding,'input_ids':value['input_ids']})
        assert 0<len(value['generated_ids'])<=1024
        cache[str(path.relative_to(root))]=sha(path)
    files={}
    if deep:
        assert idle,'Deep file snapshot requires an idle worker'
        for category in ('cases','calls','trajectories'):
            for path in sorted((root/'missing_asr'/category).rglob('*.json')):
                files[str(path.relative_to(root))]=sha(path)
        for path in sorted((directory/'unfinished').rglob('*.json')):
            files[str(path.relative_to(root))]=sha(path)
    merged=read(root/'manifest.json');total=merged['total_cases'];reused=total-merged['missing_asr_cases']
    value={'format':'agentdojo-case-call-checkpoint-v1','status':'resume_ready' if idle else 'live_snapshot',
           'reason':reason,'saved_at':time.time(),'root':str(root),'total_cases':total,'reused_asr_cases':reused,
           'new_completed':len(status['completed']),'completed_asr':reused+len(status['completed']),
           'pending_asr':len(status['pending']),'pending_case_ids':status['pending'],'completed_native':status['completed'],
           'unresolved_errors':status['errors'],'incomplete_paths_preserved':moved,
           'native_binding':binding,'native_manifest_sha256':sha(root/'missing_asr/manifest.json'),
           'plan_sha256':sha(root/'plan.json'),'generation_cache_sha256':cache,'committed_files_sha256':files,
           'features_receipt_sha256':sha(root/'probe_inputs/features.json'),
           'resume_command':'bash '+str(root/'provenance/resume.sh'),
           'resume_policy':'Skip every completed case. Reexecute unfinished cases with native real tools; reuse generated outputs only for exactly equal full input IDs and frozen binding. An interrupted, uncommitted decode is regenerated.',
           'no_scores_fabricated':True,'model_weights_unchanged':True}
    path=directory/'checkpoint.json';write(path,value)
    write(root/'checkpoint-latest.json',{'path':str(path.relative_to(root)),'sha256':sha(path),
          **{k:value[k] for k in ('status','saved_at','total_cases','completed_asr','pending_asr','new_completed','resume_command')}})
    return path,value


def prepare_resume():
    with idle_worker_lock():
        path,value=create_checkpoint('Prepare a new allocation; preserve incomplete traces and exact-input caches',idle=True,deep=True)
    return path,value
