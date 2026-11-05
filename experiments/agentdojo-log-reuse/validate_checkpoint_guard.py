"""Exercise checkpoint preservation, run locks, and PID-bound process-group stop."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from reuse_common import *
from checkpoint_state import native_status,quarantine_pending,idle_worker_lock,process_info,stop_bound_process


def main():
    parent=ROOT/'checkpoints/validation';parent.mkdir(parents=True,exist_ok=True)
    passed=[]
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        root=Path(temp);missing=root/'missing_asr';missing.mkdir()
        done='remember_delimiter/banking/user_task_0/injection_task_0'
        pending='remember_delimiter/banking/user_task_0/injection_task_1'
        write(missing/'selected_case_ids.json',[done,pending])
        write(missing/'cases'/(done+'.json'),{'case_id':done,'status':'completed','attacker_success':False,'utility':True})
        write(missing/'calls'/done/'call_0000/response.json',{'completed':'keep'})
        write(missing/'calls'/pending/'call_0000/response.json',{'completed_call_in_unfinished_episode':'preserve'})
        write(missing/'calls'/pending/'call_0001/original_request.json',{'unfinished':'preserve'})
        write(missing/'trajectories'/pending/'pipeline_00.json',{'partial_pipeline':'preserve'})
        write(missing/'generation_cache/cache.json',{'exact_input_cache':'keep'})
        preserved={str(p.relative_to(root)):sha(p) for p in (missing/'cases').rglob('*.json')}
        cache_sha=sha(missing/'generation_cache/cache.json')
        state=native_status(root);assert list(state['completed'])==[done] and state['pending']==[pending]
        target=root/'checkpoint'
        moved=quarantine_pending(root,state,target)
        assert len(moved)==2
        assert (missing/'calls'/done/'call_0000/response.json').exists()
        assert not (missing/'calls'/pending).exists()
        assert read(target/'unfinished'/pending/'calls/call_0000/response.json')['completed_call_in_unfinished_episode']=='preserve'
        assert read(target/'unfinished'/pending/'calls/call_0001/original_request.json')['unfinished']=='preserve'
        assert sha(missing/'generation_cache/cache.json')==cache_sha
        assert all(sha(root/path)==digest for path,digest in preserved.items())
        passed.append('completed cases and exact-input caches unchanged; unfinished traces preserved out of active audit paths')
        with idle_worker_lock(root):
            try:
                with idle_worker_lock(root):raise AssertionError('Concurrent lock was incorrectly allowed')
            except RuntimeError:pass
        passed.append('active run lock prevents checkpoint relocation')
        script=root/'harmless_test_worker.py';pidfile=root/'child.json'
        script.write_text('import subprocess,sys,time,json\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\nPath(sys.argv[1]).write_text(json.dumps({"pid":p.pid}))\ntime.sleep(60)\n')
        process=subprocess.Popen([sys.executable,str(script),str(pidfile)],start_new_session=True)
        try:
            record={'pid':process.pid,'start_ticks':int(Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()[19]),'script':str(script)}
            end=time.monotonic()+5
            while not pidfile.exists() and time.monotonic()<end:time.sleep(.05)
            assert pidfile.exists()
            wrong={**record,'start_ticks':record['start_ticks']+1}
            assert stop_bound_process(wrong,group=True)['already_exited'] and process.poll() is None
            child=read(pidfile)['pid']
            assert process_info(record) is not None
            assert stop_bound_process(record,group=True,timeout=3)['terminated']
            process.wait(timeout=3)
            path=Path(f'/proc/{child}/stat')
            assert not path.exists() or path.read_text().rpartition(') ')[2].split()[0]=='Z'
        finally:
            if process.poll() is None:
                os.killpg(process.pid,9);process.wait(timeout=3)
        passed.append('stale PID identity cannot signal a process; valid stop terminates parent and child process group')
    write(ROOT/'checkpoint_guard_validation.json',{'status':'passed','tests':passed,
          'checkpoint_state_sha256':sha(Path(__file__).with_name('checkpoint_state.py')),
          'validation_source_sha256':sha(Path(__file__)),'native_inference_files_modified':False})
    print(serial({'status':'passed','tests':len(passed)}))


if __name__=='__main__':main()
