"""Observe a fixed ASR producer, then perform exactly one CPU-only replay audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

B = Path(__file__).resolve().parent
P = B / 'asr_after_short128'
PID = 3053238
START = '549017219'
MANIFEST_SHA = 'ac079fa32bc39b1b2d0a34613fc2add7ace44ae414841ad5c77859cf8cc8e11a'
PROTOCOL_SHA = 'f78bd41e9781c76486b63d735edb8b56ac0fae9c46feb401ae688841485d9ccc'
AUDITOR_SHA = 'e0f049470e698d3231d72a60639170e30a2b3baa55849caa5a052047b74d5720'
PLAN = B / 'independent_asr_audit_plan.json'
OUT = B / 'audit_asr_terminal_supervision'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n')


def process_state():
    try:
        value = Path(f'/proc/{PID}/stat').read_text()
    except FileNotFoundError:
        return {'same_process_alive': False, 'process_state': 'absent'}
    fields = value[value.rfind(')') + 2:].split()
    same = fields[19] == START
    return {'same_process_alive': same and fields[0] != 'Z',
            'process_state': fields[0] if same else 'pid_reused',
            'observed_process_start_ticks': fields[19]}


def payload_inventory():
    rows = []
    for name in ('cases', 'calls', 'trajectories', 'injections', 'provenance'):
        root = P / name
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    st = path.stat()
                    rows.append([path.relative_to(P).as_posix(), st.st_size, st.st_mtime_ns])
    for name in ('protocol.json', 'launch.json', 'progress.json', 'summary.json'):
        path = P / name
        if path.exists():
            st = path.stat()
            rows.append([name, st.st_size, st.st_mtime_ns, sha(path)])
    return {'files': len(rows), 'stat_inventory_sha256': hashlib.sha256(
        json.dumps(rows, separators=(',', ':')).encode()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'full_audit_launch.json').exists(), 'One-shot audit was already launched'
    plan = read(PLAN)
    plan_sha = sha(PLAN)
    manifest = B / 'asr_candidates_after_short128.json'
    auditor = Path(plan['auditor_source_path'])
    prior_terminal_inventory = None
    previous_terminal = -1
    while True:
        assert sha(PLAN) == plan_sha
        assert sha(manifest) == MANIFEST_SHA == plan['manifest_sha256']
        assert sha(P / 'protocol.json') == PROTOCOL_SHA
        assert sha(auditor) == AUDITOR_SHA == plan['auditor_source_sha256']
        state = process_state()
        progress = read(P / 'progress.json') if (P / 'progress.json').exists() else {}
        summary = read(P / 'summary.json') if (P / 'summary.json').exists() else None
        terminal = progress.get('terminal', 0)
        assert previous_terminal <= terminal <= 2334
        previous_terminal = terminal
        assert progress.get('scheduled', 2334) == 2334
        observation = {'observed_unix': time.time(), 'producer_pid': PID,
                       'producer_start_ticks': START, **state,
                       'terminal_episodes': terminal, 'scheduled_episodes': 2334,
                       'summary_present': summary is not None,
                       'protocol_and_manifest_unchanged': True}
        ready = False
        if not state['same_process_alive']:
            assert summary is not None, 'Producer ended without a final summary'
            assert summary['terminal'] == summary['scheduled_episodes'] == terminal == 2334
            assert summary['tasks_per_condition'] == 389 and len(summary['conditions']) == 6
            inventory = payload_inventory()
            observation.update(inventory)
            ready = inventory == prior_terminal_inventory
            observation['stable_for_two_terminal_observations'] = ready
            prior_terminal_inventory = inventory
        with (OUT / 'observations.jsonl').open('a') as handle:
            handle.write(json.dumps(observation, sort_keys=True) + '\n')
        print(json.dumps(observation, sort_keys=True), flush=True)
        if ready:
            report = {'schema': 'independent-fixed-asr-terminal-gate-v1', 'status': 'ready',
                      'last_observation': observation, 'poll_seconds': 60,
                      'plan_sha256': plan_sha, 'manifest_sha256': MANIFEST_SHA,
                      'protocol_sha256': PROTOCOL_SHA, 'auditor_sha256': AUDITOR_SHA,
                      'summary_sha256': sha(P / 'summary.json'),
                      'progress_sha256': sha(P / 'progress.json'),
                      'watcher_sha256': sha(__file__),
                      'model_generation_by_auditor': 0, 'API_calls_by_auditor': 0,
                      'episode_retries_by_auditor': 0,
                      'unknown_verdicts_remain_null': True}
            write(OUT / 'readiness.json', report)
            launch = {'schema': 'independent-fixed-asr-full-audit-launch-v1',
                      'started_unix': time.time(), 'argv': plan['full_argv'],
                      'environment': plan['environment'],
                      'readiness_sha256': sha(OUT / 'readiness.json'),
                      'one_shot_cpu_only': True}
            write(OUT / 'full_audit_launch.json', launch)
            env = dict(os.environ)
            env.update(plan['environment'])
            with (B / 'asr_full_independent_audit.log').open('xb') as log:
                result = subprocess.run(plan['full_argv'], env=env, stdout=log, stderr=subprocess.STDOUT)
            result_record = {'schema': 'independent-fixed-asr-full-audit-result-v1',
                             'completed_unix': time.time(), 'exit_code': result.returncode,
                             'launch_sha256': sha(OUT / 'full_audit_launch.json'),
                             'audit_log_sha256': sha(B / 'asr_full_independent_audit.log')}
            audit = P / 'audit_fixed/audit.json'
            if audit.exists():
                result_record.update(audit_status=read(audit).get('status'), audit_sha256=sha(audit))
            write(OUT / 'full_audit_result.json', result_record)
            print(json.dumps(result_record, sort_keys=True), flush=True)
            raise SystemExit(result.returncode)
        time.sleep(60)


if __name__ == '__main__':
    main()
