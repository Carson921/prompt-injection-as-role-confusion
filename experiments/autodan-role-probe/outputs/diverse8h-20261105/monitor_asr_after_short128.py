"""Read-only supervision of one known ASR process; never launches or retries."""
import json,time,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
B=Path(__file__).resolve().parent
PID=3053238
TICKS='549017219'
EXPECTED=2334

def read(path):return json.loads(path.read_text())
def process():
 try:
  value=Path(f'/proc/{PID}/stat').read_text()
 except FileNotFoundError:return None
 tail=value[value.rfind(')')+2:].split()
 return dict(pid=PID,start_ticks=tail[19],state=tail[0])
def write(value):
 path=B/'asr_after_short128_monitor.json';temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def local(value):return datetime.datetime.fromtimestamp(value,ZoneInfo('America/New_York')).isoformat()
last=None
while True:
 now=time.time();proc=process();same=proc is not None and proc['start_ticks']==TICKS and proc['state']not in('Z','X')
 progress=read(B/'asr_after_short128/progress.json')if(B/'asr_after_short128/progress.json').exists()else None
 summary=read(B/'asr_after_short128/summary.json')if(B/'asr_after_short128/summary.json').exists()else None
 terminal=progress['terminal']if progress else 0
 record=dict(observed_unix=now,observed_local=local(now),main_process=proc,known_process_live=same,
             terminal=terminal,expected_episodes=EXPECTED,progress=progress,summary_available=summary is not None,
             ASR_restarts=0,error_episode_reruns=0,baseline_reruns=0,full_predicate_audit_owner='/root/numerical_audit')
 if progress and terminal:
  remaining=progress['elapsed_seconds']/terminal*(EXPECTED-terminal)
  record.update(observed_mean_throughput_remaining_seconds=remaining,observed_mean_throughput_predicted_finish_local=local(now+remaining),
                ETA_caveat='Descriptive completed-episode throughput; long pending native trajectories may change it.')
 if summary is not None:
  record.update(status='all_ASR_terminal_main_teardown_pending'if same else'ASR_process_terminal_summary_available',
                terminal=summary['terminal'],conditions=summary['conditions'],native_errors=summary['errors'])
  assert summary['terminal']==summary['scheduled_episodes']==EXPECTED
 else:record['status']='ASR_running'if same else'ASR_process_missing_without_final_summary'
 write(record)
 signature=(record['status'],record['terminal'])
 if signature!=last:
  print(json.dumps(record,ensure_ascii=False),flush=True);last=signature
 if not same:break
 time.sleep(60)
