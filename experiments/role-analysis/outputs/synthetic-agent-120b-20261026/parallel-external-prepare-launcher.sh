#!/usr/bin/env bash
# Review candidate only: root deploys/starts it. No model forward or core edits.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
task_root=/root/role-retraining-20261026
task_python=/root/role-retraining-venv/bin/python
task_log=$task_root/parallel-external-prepare-v7.log

if [[ $# -eq 0 ]]; then
  task_seconds=$("$task_python" - <<'DEADLINE_PY'
from datetime import datetime, timezone
import hashlib,json,time
from pathlib import Path
root=Path('/workspace/role-retraining-20261026/runs-v7')
ends=[datetime(2026,10,27,13,19,17,tzinfo=timezone.utc).timestamp()]
for phase in ('pilot','formal'):
    raw=(root/phase/'config.json').read_bytes()
    if phase=='formal' and hashlib.sha256(raw).hexdigest()!='4c32c9eba68304f5b7b7ed798262c349c385aa1d2a7a48b1840f5c4afcdbd1cf':
        raise SystemExit('formal_config_sha_mismatch')
    c=json.loads(raw);b=c['budget']
    start=datetime.fromisoformat(b['started_utc'].replace('Z','+00:00'))
    end=datetime.fromisoformat(b['deadline_utc'].replace('Z','+00:00'))
    hours=float(b['authorized_hours'])
    if c['phase']!=phase or not(start.tzinfo and end.tzinfo and 0<hours<=24 and 0<(end-start).total_seconds()<=hours*3600):
        raise SystemExit('invalid_authorized_budget')
    ends.append(end.timestamp())
remaining=int(min(ends)-time.time())
if remaining<=0:
    raise SystemExit('authorized_deadline_exceeded')
print(min(1800,remaining))
DEADLINE_PY
)
  # GNU timeout owns a separate process group. Do not add --foreground.
  exec /usr/bin/timeout --signal=KILL "${task_seconds}s" /bin/bash "${BASH_SOURCE[0]}" --bounded
fi
if [[ $# -ne 1 || $1 != --bounded ]]; then
  printf '%s\n' '{"status":"failed","error_code":"invalid_launcher_arguments"}' >&2
  exit 64
fi

# Only this launcher uses this local lock; root must keep the campaign from
# reaching its own external writer until this bounded process has exited.
exec 9>"$task_root/parallel-external-prepare-v7.lock"
if ! /usr/bin/flock -n 9; then
  printf '%s\n' '{"status":"failed","error_code":"external_writer_lock_busy"}' >&2
  exit 73
fi
cd "$task_root/code-v7"
exec "$task_python" -u - >>"$task_log" 2>&1 <<'PREPARE_PY'
from __future__ import annotations
import contextlib,copy,hashlib,json,os,signal,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path

TASK=Path('/root/role-retraining-20261026')
CODE=TASK/'code-v7'
RUNS=Path('/workspace/role-retraining-20261026/runs-v7')
RUN=RUNS/'formal'
DATA=Path('/workspace/role-retraining-20261026/data-v7')
EXTERNAL=DATA/'external'
MODEL=Path('/root/role-retraining-model')
STATE=TASK/'parallel-external-prepare-v7.json'
CACHE=Path('/root/role-retraining-cache/external-downloads')
PIN_CONFIG='4c32c9eba68304f5b7b7ed798262c349c385aa1d2a7a48b1840f5c4afcdbd1cf'
PIN_FREEZE='ecaf97a27e5cd8f456279360493e619b0e3de2c9efff4a451b53028d7d94eb97'
PIN_EXTERNAL='0d5496063e6f9a28501ff0c98b215a3f45bc7afc90efc212b9292607f21615b3'
PIN_LEDGER='7a672d743bc0028078ed6e98c8174e76f4af2c0587bd115decb7e1fad36263ef'
PROTECTED={'formal_config':RUN/'config.json','pilot_config':RUNS/'pilot/config.json',
 'formal_status':RUN/'status.json','freeze':RUN/'freeze.json','best':RUN/'best.npz',
 'selection':RUN/'selection.json','protocol_pointer':RUN/'external-protocol.json',
 'protocol':EXTERNAL/'protocol.json','history_index':EXTERNAL/'history-index.json',
 'legacy_ledger':EXTERNAL/'legacy-hashes.json','external_implementation':CODE/'synthetic_agent_external.py'}

class GuardError(Exception):
    pass

def guard(condition,code):
    if not condition:
        raise GuardError(code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def save(value):
    # Only the dedicated audit sidecar is written, never formal/status.json.
    temporary=STATE.with_name(STATE.name+'.tmp.'+str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,STATE)

def hashes():
    return {name:sha(path) for name,path in PROTECTED.items()}

def merged_binding(protected,source_pins):
    return hashlib.sha256(canonical({'protected_sha256':protected,'source_pins':source_pins}).encode()).hexdigest()

def budget():
    ends=[datetime(2026,10,27,13,19,17,tzinfo=timezone.utc).timestamp()]
    for phase in ('pilot','formal'):
        c=read(RUNS/phase/'config.json');b=c['budget']
        start=datetime.fromisoformat(b['started_utc'].replace('Z','+00:00'))
        end=datetime.fromisoformat(b['deadline_utc'].replace('Z','+00:00'));hours=float(b['authorized_hours'])
        guard(c['phase']==phase and start.tzinfo is not None and end.tzinfo is not None and
              0<hours<=24 and 0<(end-start).total_seconds()<=hours*3600,'invalid_authorized_budget')
        ends.append(end.timestamp())
    return min(ends)

started=time.time()
proc_tail=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
state={'status':'running','pid':os.getpid(),'start_ticks':int(proc_tail[19]),'started_unix':started,
       'stage':'guards','cache_override':{'external.cache_dir':str(CACHE)},
       'log':str(TASK/'parallel-external-prepare-v7.log'),
       'hard_kill_note':'SIGKILL is uncatchable: a running sidecar without this live PID/start_ticks is incomplete, never success.'}
before=None
timer=None
try:
    guard(os.environ.get('CUDA_VISIBLE_DEVICES')=='','cuda_mask_not_empty')
    deadline=budget();guard(deadline>time.time(),'authorized_deadline_exceeded')
    state['deadline_unix']=deadline
    # Absolute safety bound in addition to GNU timeout's <=1800-second bound.
    timer=threading.Timer(max(0,deadline-time.time()),lambda:os.kill(os.getpid(),signal.SIGKILL))
    timer.daemon=True;timer.start()
    before=hashes();state['protected_sha256_before']=before
    guard(before['formal_config']==PIN_CONFIG,'formal_config_sha_mismatch')
    guard(before['freeze']==PIN_FREEZE,'freeze_sha_mismatch')
    guard(before['external_implementation']==PIN_EXTERNAL,'external_implementation_sha_mismatch')
    guard(before['legacy_ledger']==PIN_LEDGER,'legacy_ledger_sha_mismatch')
    raw_config=(RUN/'config.json').read_bytes()
    guard(hashlib.sha256(raw_config).hexdigest()==PIN_CONFIG,'formal_config_read_changed')
    original=json.loads(raw_config);effective=copy.deepcopy(original)
    effective.setdefault('external',{})['cache_dir']=str(CACHE)
    protocol=read(EXTERNAL/'protocol.json');frozen=read(RUN/'freeze.json')
    guard(frozen['probe_sha256']==before['best'] and frozen['selection_sha256']==before['selection'] and
          frozen['external_protocol_sha256']==before['protocol'] and
          frozen['external_protocol_pointer_sha256']==before['protocol_pointer'] and
          frozen['external_history_index_sha256']==before['history_index'],'frozen_binding_mismatch')
    sys.path.insert(0,str(CODE))
    import synthetic_agent_external as external
    guard(Path(external.__file__).resolve()==PROTECTED['external_implementation'].resolve() and sha(external.__file__)==PIN_EXTERNAL,
          'imported_external_implementation_mismatch')
    guard(protocol['source_pins']==external.PINS and protocol['prefixes']==external.PREFIXES and
          protocol['implementation_sha256']==PIN_EXTERNAL,'source_protocol_binding_mismatch')
    state['source_pins']=copy.deepcopy(external.PINS)
    state['binding_sha256_before']=merged_binding(before,state['source_pins'])
    save(state);print(canonical({'status':'running','pid':state['pid'],'start_ticks':state['start_ticks'],'started_unix':started}),flush=True)
    # Library progress and exceptions cannot print public/historical source text.
    # Only the curated numeric/hash audit below reaches the dedicated log.
    with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
        state['stage']='tokenizer'
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        state['stage']='prepare_external'
        result=external.prepare_external(effective,tokenizer,DATA,RUN)
    state['stage']='validate_commits'
    manifest=read(EXTERNAL/'manifest.json');sidecar=read(RUN/'external-preparation.json')
    guard(result==manifest==sidecar,'external_committed_dictionaries_differ')
    guard(manifest['counts_by_source']=={'oasst':128,'toxicchat':128},'external_quotas_mismatch')
    inputs_sha=sha(EXTERNAL/'inputs-test.jsonl')
    guard(manifest['inputs_sha256']==inputs_sha,'external_inputs_sha_mismatch')
    guard(manifest['freeze_sha256']==PIN_FREEZE and manifest['probe_sha256']==before['best'] and
          manifest['protocol_sha256']==before['protocol'] and manifest['historical_metadata_sha256']==before['history_index'] and
          manifest['historical_de_duplication']['augmentation_sha256']==PIN_LEDGER,'external_manifest_binding_mismatch')
    guard(set(manifest['source_snapshots'])==set(external.PINS),'source_snapshot_set_mismatch')
    for dataset,pin in external.PINS.items():
        snapshot=manifest['source_snapshots'][dataset]
        guard({key:snapshot[key] for key in pin}==pin and sha(Path(snapshot['path']))==pin['sha256'],'cached_source_pin_mismatch')
    after=hashes();state['protected_sha256_after']=after
    guard(before==after,'protected_original_file_changed')
    state['binding_sha256_after']=merged_binding(after,read(EXTERNAL/'protocol.json')['source_pins'])
    guard(state['binding_sha256_before']==state['binding_sha256_after'],'merged_source_frozen_binding_changed')
    guard(time.time()<deadline,'authorized_deadline_exceeded')
    state.update(status='complete',stage='complete',finished_unix=time.time(),elapsed_seconds=time.time()-started,
                 counts_by_source=manifest['counts_by_source'],scored_body_tokens_by_role=manifest['scored_body_tokens_by_role'],
                 output_sha256={'inputs':inputs_sha,'manifest':sha(EXTERNAL/'manifest.json'),'sidecar':sha(RUN/'external-preparation.json')})
    save(state)
    print(canonical({'status':'complete','pid':state['pid'],'elapsed_seconds':state['elapsed_seconds'],
                     'counts_by_source':state['counts_by_source'],'output_sha256':state['output_sha256']}),flush=True)
except BaseException as error:
    # No traceback, source snippet, token IDs, input records, or digest-set dump.
    state.update(status='failed',finished_unix=time.time(),elapsed_seconds=time.time()-started,
                 error_type=type(error).__name__,error_code=str(error) if isinstance(error,GuardError) else 'api_or_io_failure')
    if before is not None:
        try:
            after=hashes();state['protected_sha256_after']=after
            state['protected_changed']=[name for name in before if before[name]!=after[name]]
        except BaseException:
            state['protected_postcheck']='unavailable_after_failure'
    save(state)
    print(canonical({'status':'failed','pid':state['pid'],'stage':state['stage'],'error_type':state['error_type'],
                     'error_code':state['error_code'],'elapsed_seconds':state['elapsed_seconds']}),flush=True)
    raise SystemExit(1)
finally:
    if timer is not None:
        timer.cancel()
PREPARE_PY
