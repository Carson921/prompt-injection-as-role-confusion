"""Checked continuation of the DeepSeek evolution, probe, ASR and report stages.

Native extraction needs the same four H100s as vLLM. The exact owned DeepSeek
server tree is stopped for native work and restored afterwards; a healthy
server is reused for ASR and left running. --check-only touches no processes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from ds_common import MODEL, PROBE_RUN, PROBE_SHA256, read_json, sha, write_json

HERE = Path(__file__).resolve().parent
PYA = '/scratch/wz55191/venvs/chatinject-table1/bin/python'
PYP = '/scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python'
BASE_URL = 'http://127.0.0.1:8000/v1'
SERVED_MODEL = 'deepseek-v4-flash'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_agentdojo(directory, specs):
    require(read_json(directory / 'conditions.json') == specs, f'{directory}: conditions differ')
    summary = read_json(directory / 'summary.json')['conditions']
    require(set(summary) == set(specs), f'{directory}: summary conditions differ')
    for name in specs:
        row = summary[name]
        require(row['expected'] == row['scored'] == 389 and row['errors'] == 0,
                f'{directory}/{name}: incomplete AgentDojo: {row["scored"]}/389, errors={row["errors"]}')
        paths = list((directory / 'cases' / name).glob('*/*/*.json'))
        require(len(paths) == 389 and all(read_json(p)['status'] == 'complete' for p in paths),
                f'{directory}/{name}: incomplete raw AgentDojo records')


def check_ipi_prepare(directory, specs):
    require(read_json(directory / 'conditions.json') == specs, f'{directory}: conditions differ')
    protocol = read_json(directory / 'protocol.json')
    require(set(protocol['conditions']) == set(specs) and protocol['behaviors'] == 38 and protocol['primary'] == 36,
            f'{directory}: unexpected IPI panel')
    rows = protocol['rows']
    require(len(rows) == 38 * len(specs), f'{directory}: missing IPI requests')
    for row in rows:
        require(row['fits_context'], f'{directory}: request too long: {row["request"]}')
        require(sha(directory / row['request']) == row['request_sha256'], f'{directory}: request hash differs')
    return protocol


def check_ipi_asr(directory, specs):
    protocol = check_ipi_prepare(directory, specs)
    identity = read_json(directory / 'asr_identity.json')['identity_sha256']
    summary = read_json(directory / 'summary.json')['conditions']
    require(set(summary) == set(specs), f'{directory}: summary conditions differ')
    for name in specs:
        row = summary[name]
        require(row['scheduled'] == row['complete'] == 36 and row['errors'] == 0
                and row['supplementary']['complete'] == 2 and row['tokenize_match'] == 38,
                f'{directory}/{name}: incomplete IPI ASR or tokenization mismatch')
    for row in protocol['rows']:
        record = read_json(directory / 'cases' / row['condition'] / f'{row["case_id"]}.json')
        require(record['status'] == 'complete' and record['identity_sha256'] == identity
                and record['request_sha256'] == row['request_sha256'],
                f'{directory}: stale or incomplete IPI case {row["condition"]}/{row["case_id"]}')


def check_ipi_probe(directory, specs):
    protocol = check_ipi_prepare(directory, specs)
    summary = read_json(directory / 'probe_summary.json')
    require(set(summary) == set(specs), f'{directory}: probe summary conditions differ')
    for name in specs:
        require(summary[name]['cases'] == 36 and math.isfinite(summary[name]['p_user_mean']),
                f'{directory}/{name}: incomplete IPI probe summary')
    for row in protocol['rows']:
        destination = directory / 'probe' / row['condition'] / f'{row["case_id"]}.json'
        record = read_json(destination)
        require(record['status'] == 'complete' and record['request_sha256'] == row['request_sha256']
                and destination.with_suffix('.npz').is_file(), f'{directory}: missing or stale probe result')


def process_table():
    """PID reuse guards and argv for processes owned by this user only."""
    result = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            result[int(path.name)] = dict(ppid=int(fields[1]), start=fields[19], argv=argv, state=fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def server_roots(table):
    roots = []
    for pid, process in table.items():
        args = process['argv']
        for index, arg in enumerate(args[:-2]):
            if Path(arg).name == 'vllm' and args[index + 1] == 'serve' and Path(args[index + 2]) == MODEL:
                port = args[args.index('--port') + 1] if '--port' in args else '8000'
                if port == '8000':
                    roots.append(pid)
    return roots


def server_ready():
    try:
        with urllib.request.urlopen(BASE_URL + '/models', timeout=5) as response:
            models = json.load(response)['data']
        require(any(m['id'] == SERVED_MODEL for m in models), 'Port 8000 serves a different model; refusing to interfere')
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.e, self.b, self.o = args.evolution.resolve(), args.baseline.resolve(), args.optimized.resolve()
        self.o.mkdir(parents=True, exist_ok=True)
        self.state_path = self.o / 'phase2_state.json'
        self.active = {}
        self.server = None
        self.reload_state()

    def reload_state(self):
        self.state = read_json(self.state_path) if self.state_path.exists() else dict(stages={})
        self.restore_service = bool(self.state.get('service_handoff')) and not self.state['service_handoff'].get('restored')

    def save(self):
        write_json(self.state_path, self.state)

    def event(self, stage, **values):
        print(json.dumps(dict(time=time.time(), stage=stage, **values)), flush=True)

    def completed(self, name, validator):
        if self.state['stages'].get(name, {}).get('status') == 'complete':
            validator()
            self.event(name, status='reused')
            return True
        return False

    def recover_stale_stages(self):
        """A killed wrapper may leave its independent child process running.

        Check this before any GPU handoff. Never restore vLLM over an orphaned
        native model, and never treat a missing child as a successful stage.
        """
        table = process_table()
        running = [(name, row) for name, row in self.state['stages'].items() if row.get('status') == 'running']
        for name, row in running:
            process = table.get(row.get('pid'))
            if process and process['start'] == row.get('process_start') and process.get('state') != 'Z':
                self.restore_service = False
                raise RuntimeError(f'Previous {name} process {row["pid"]} is still running; '
                                   'refusing a duplicate or a competing vLLM restoration')
        for name, row in running:
            row.update(status='interrupted', interrupted_unix=time.time(),
                       interruption='Recorded process is gone, a zombie, or its PID was reused; exit code unavailable')
            self.event(name, status='interrupted', previous_pid=row.get('pid'))
        if running:
            self.save()

    def launch(self, name, command, log):
        old = self.state['stages'].get(name, {})
        if old.get('status') == 'running':
            process = process_table().get(old.get('pid'))
            require(not process or process['start'] != old.get('process_start') or process.get('state') == 'Z',
                    f'Previous {name} process {old["pid"]} is still running; refusing a duplicate')
        handle = log.open('a')
        handle.write(f'\n=== phase2 {time.time()} {name}\n')
        handle.flush()
        try:
            process = subprocess.Popen(command, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            handle.close()
            raise
        info = process_table().get(process.pid, {})
        attempts = list(old.get('previous_attempts', []))
        if old:
            attempts.append({key: value for key, value in old.items() if key != 'previous_attempts'})
        self.state['stages'][name] = dict(status='running', pid=process.pid, process_start=info.get('start'),
                                         started_unix=time.time(), command=list(map(str, command)), log=str(log),
                                         previous_attempts=attempts)
        self.active[name] = (process, handle)
        self.save()
        self.event(name, status='running', pid=process.pid, log=str(log))

    def finish(self, name, validator):
        process, handle = self.active[name]
        code = process.wait()
        handle.write(f'\n=== phase2 actual exit {code}\n')
        handle.close()
        del self.active[name]
        row = self.state['stages'][name]
        row.update(exit_code=code, finished_unix=time.time(), status='failed')
        self.save()
        require(code == 0, f'{name} exited {code}; see {row["log"]}')
        validator()
        row['status'] = 'complete'
        self.save()
        self.event(name, status='complete', exit_code=code)

    def stage(self, name, command, log, validator):
        if not self.completed(name, validator):
            self.launch(name, command, log)
            self.finish(name, validator)

    def finish_parallel(self, pending):
        """Notice either ASR child's failure promptly; never wait on the server."""
        pending = dict(pending)
        while pending:
            for name, validator in list(pending.items()):
                if self.active[name][0].poll() is not None:
                    self.finish(name, validator)
                    del pending[name]
            if pending:
                time.sleep(1)

    def stop_for_native(self):
        table = process_table()
        roots = server_roots(table)
        require(len(roots) <= 1, 'Multiple matching vLLM servers found; refusing an ambiguous stop')
        ready = server_ready()
        require(not ready or roots, 'Ready server has no matching owned local PID; cannot release its GPUs safely')
        if roots:
            targets = set(roots)
            while True:
                descendants = {pid for pid, row in table.items() if row['ppid'] in targets}
                if descendants <= targets:
                    break
                targets |= descendants
            self.restore_service = True
            self.state['service_handoff'] = dict(stopped_pids=sorted(targets), stopped_unix=time.time(), restored=False)
            self.save()
            self.event('vllm-release', pids=sorted(targets), reason='native probe needs all four H100s')
            for signum, delay in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
                current = process_table()
                for pid in targets:
                    if pid in current and current[pid]['start'] == table[pid]['start']:
                        try:
                            os.kill(pid, signum)
                        except ProcessLookupError:
                            pass
                end = time.monotonic() + delay
                while time.monotonic() < end:
                    current = process_table()
                    if not any(pid in current and current[pid]['start'] == table[pid]['start']
                               and current[pid]['state'] != 'Z' for pid in targets):
                        break
                    time.sleep(2)
                else:
                    continue
                break
            else:
                raise RuntimeError('Matching vLLM process tree did not stop')
        end = time.monotonic() + 90
        while True:
            report = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free',
                                              '--format=csv,noheader,nounits'], text=True)
            rows = [line.rsplit(',', 1) for line in report.strip().splitlines()]
            require(len(rows) == 4 and all('H100' in row[0] for row in rows), 'Native protocol requires exactly four H100s')
            if all(int(row[1].strip()) >= 60000 for row in rows):
                break
            require(time.monotonic() < end, f'GPUs were not released for native model: {report.strip()}')
            time.sleep(5)

    def ensure_server(self):
        # A failed startup is reported once; a later invocation may retry it.
        self.restore_service = False
        if server_ready():
            self.event('vllm', status='reused')
        else:
            if not server_roots(process_table()):
                handle = (self.o / 'vllm.log').open('a')
                self.server = subprocess.Popen(['bash', 'serve_vllm.sh'], cwd=HERE, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                handle.close()
                self.state['vllm_pid'] = self.server.pid
                self.save()
                self.event('vllm', status='starting', pid=self.server.pid)
            end = time.monotonic() + 4000
            while not server_ready():
                require(self.server is None or self.server.poll() is None,
                        f'vLLM startup exited {getattr(self.server, "returncode", None)}; see {self.o / "vllm.log"}')
                require(time.monotonic() < end, f'vLLM readiness timed out; see {self.o / "vllm.log"}')
                time.sleep(10)
        self.restore_service = False
        if 'service_handoff' in self.state:
            self.state['service_handoff'].update(restored=True, restored_unix=time.time())
            self.save()

    def check_baseline(self):
        check_agentdojo(self.b / 'agentdojo', read_json(self.b / 'conditions_agentdojo.json'))
        check_ipi_asr(self.b / 'ipi', read_json(self.b / 'conditions_ipi.json'))
        require(sha(PROBE_RUN / 'checkpoint.npz') == PROBE_SHA256, 'Frozen checkpoint hash differs')
        dataset = read_json(self.e / 'dataset.json')
        require(len(dataset['contexts']) == len(dataset['ipi_cases']) == 6, 'Expected six AgentDojo plus six IPI optimization cases')

    def check_evolution(self):
        summary = read_json(self.e / 'summary.json')
        require(summary['status'] == 'complete' and summary['scored'] > 0
                and math.isfinite(summary['best']['fitness']), 'No complete evolution result')
        rows = [json.loads(line) for line in (self.e / 'scores.jsonl').read_text().splitlines() if line.strip()]
        require(any(r.get('status') == 'complete' and r['template'] == summary['best']['template'] for r in rows),
                'Evolution best is missing from score ledger')

    def check_selection(self):
        specs = read_json(self.o / 'conditions.json')
        require('opt_top1' in specs and all(s['template'].count('{goal}') == 1 for s in specs.values()),
                'Missing optimized template selection')

    def check_probe(self):
        specs = read_json(self.o / 'conditions.json')
        scores = read_json(self.o / 'condition_scores.json')
        names = set(read_json(self.b / 'conditions_agentdojo.json')) | set(specs)
        require(set(scores) == names, 'Condition probe scores are incomplete')
        for name, row in scores.items():
            require(len(row['per_case']) == 12 and math.isfinite(row['p_user']) and 0 <= row['p_user'] <= 1,
                    f'Incomplete condition score: {name}')
        check_ipi_probe(self.b / 'ipi', read_json(self.b / 'conditions_ipi.json'))
        check_ipi_probe(self.o / 'ipi', specs)

    def run(self):
        if not self.args.check_only:
            self.recover_stale_stages()
        self.check_baseline()
        self.event('baseline', status='validated')
        if self.args.check_only:
            return
        binding = {str(p): sha(p) for p in (self.e / 'dataset.json', self.b / 'conditions_agentdojo.json',
                                            self.b / 'conditions_ipi.json')}
        require('inputs' not in self.state or self.state['inputs'] == binding, 'Phase2 inputs changed since the previous run')
        self.state['inputs'] = binding
        self.state['status'] = 'running'
        self.state.pop('error', None)
        self.state.pop('cleanup_error', None)
        self.save()
        if not self.completed('evolution', self.check_evolution):
            self.stop_for_native()
            if 'evolution_deadline_unix' not in self.state:
                self.state['evolution_deadline_unix'] = time.time() + self.args.hours * 3600
                self.state['evolution_hours'] = self.args.hours
                self.save()
            self.stage('evolution', ['bash', str(self.e / 'run.sh'), '--deadline-unix', str(self.state['evolution_deadline_unix'])],
                       self.e / 'stdout.log', self.check_evolution)
        self.stage('select', [PYA, 'ds_select.py', '--evolution', str(self.e), '--out', str(self.o / 'conditions.json'),
                              '--top', '2', '--bands', '0.2,0.4,0.6,0.8'], self.o / 'selection.log', self.check_selection)
        specs = read_json(self.o / 'conditions.json')
        self.stage('ipi-prepare', [PYA, 'ds_ipi.py', 'prepare', '--output', str(self.o / 'ipi'),
                                   '--conditions-json', str(self.o / 'conditions.json')], self.o / 'ipi_prepare.log',
                   lambda: check_ipi_prepare(self.o / 'ipi', specs))
        if not self.completed('final-probe', self.check_probe):
            self.stop_for_native()
            self.stage('final-probe', [PYP, '-u', 'ds_final_probe.py', '--dataset', str(self.e / 'dataset.json'), '--output', str(self.e),
                                       '--conditions-json', str(self.b / 'conditions_agentdojo.json'), str(self.o / 'conditions.json'),
                                       '--scores-out', str(self.o / 'condition_scores.json'), '--ipi-runs', str(self.b / 'ipi'), str(self.o / 'ipi')],
                       self.o / 'final_probe.log', self.check_probe)
        self.ensure_server()
        jobs = [('ipi-asr', [PYA, '-u', 'ds_ipi.py', 'run', '--output', str(self.o / 'ipi'), '--workers', '8'],
                 self.o / 'ipi_run.log', lambda: check_ipi_asr(self.o / 'ipi', specs)),
                ('agentdojo-asr', [PYA, '-u', 'ds_agentdojo.py', '--output', str(self.o / 'agentdojo'),
                                  '--conditions-json', str(self.o / 'conditions.json'), '--workers', '24'],
                 self.o / 'agentdojo_run.log', lambda: check_agentdojo(self.o / 'agentdojo', specs))]
        pending = []
        for name, command, log, validator in jobs:
            if not self.completed(name, validator):
                self.launch(name, command, log)
                pending.append((name, validator))
        self.finish_parallel(pending)
        self.stage('report', [PYA, 'ds_report.py', '--baseline', str(self.b), '--optimized', str(self.o),
                              '--evolution', str(self.e), '--out', str(self.o / 'REPORT.zh-CN.md')], self.o / 'report.log',
                   lambda: require((self.o / 'REPORT.zh-CN.md').stat().st_size > 1000, 'Final report is missing or empty'))
        self.state['status'] = 'complete'
        self.save()
        self.event('done', report=str(self.o / 'REPORT.zh-CN.md'), vllm='left running')

    def cleanup(self):
        for name, (process, handle) in list(self.active.items()):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            self.state['stages'][name].update(status='failed', exit_code=process.returncode, finished_unix=time.time())
            handle.close()
        self.active.clear()
        self.save()
        if self.restore_service:
            self.ensure_server()

    def execute(self):
        """Retain the experiment failure even if service restoration also fails."""
        primary_error = None
        try:
            self.run()
        except BaseException as error:
            primary_error = error
            self.state.update(status='failed', error=f'{type(error).__name__}: {error}')
            try:
                self.save()
            except BaseException as save_error:
                error.add_note(f'Could not persist failure state: {save_error}')
            raise
        finally:
            if not self.args.check_only:
                try:
                    self.cleanup()
                except BaseException as cleanup_error:
                    description = f'{type(cleanup_error).__name__}: {cleanup_error}'
                    self.state['cleanup_error'] = description
                    self.event('cleanup', status='failed', error=description)
                    if primary_error is None:
                        self.state.update(status='failed', error=description)
                    try:
                        self.save()
                    except BaseException as save_error:
                        cleanup_error.add_note(f'Could not persist cleanup failure: {save_error}')
                    if primary_error is not None:
                        primary_error.add_note(f'Cleanup or vLLM restoration also failed: {description}')
                    else:
                        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=HERE / 'outputs/baseline-20261118')
    parser.add_argument('--evolution', type=Path, default=HERE / 'outputs/dsevo-20261118')
    parser.add_argument('--optimized', type=Path, default=HERE / 'outputs/optimized-20261118')
    parser.add_argument('--hours', type=float, default=float(os.environ.get('DS_EVO_HOURS', '6')))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    require(args.hours > 0, 'Evolution hours must be positive')
    pipeline = Pipeline(args)
    with (pipeline.o / 'phase2.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another phase2 pipeline already holds the lock') from error
        # The previous owner may have saved a final receipt between construction
        # and lock acquisition. Resume from the state protected by this lock.
        pipeline.reload_state()
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'phase2 received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        pipeline.execute()


if __name__ == '__main__':
    main()
