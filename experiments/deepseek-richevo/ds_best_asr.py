"""Evaluate the recorded current best after a user-requested search stop.

This entry point does not start evolution or load the native probe model.
It reuses the existing checked benchmark runners and leaves vLLM serving.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time

from ds_common import read_json, sha, write_json
from ds_phase2 import (HERE, PYA, Pipeline, check_agentdojo, check_ipi_asr,
                       check_ipi_prepare, process_table, require, server_roots)
from ds_select import panel_scores


def prepare_best(evolution, output):
    receipt = read_json(evolution / 'user_stop_snapshot.json')
    best = receipt['best']
    rows = [json.loads(line) for line in (evolution / 'scores.jsonl').read_text().splitlines() if line.strip()]
    require(any(r.get('status') == 'complete' and r['evaluation_index'] == best['evaluation_index']
                and sha(r['template']) == sha(best['template']) and r['fitness'] == best['fitness'] for r in rows),
            'Stopped best is absent from completed score ledger')
    dataset = read_json(evolution / 'dataset.json')
    ad_ids = {r['case_id'] for r in dataset['contexts']}
    ipi_ids = set(dataset['ipi_cases'])
    scores = panel_scores(best, ad_ids, ipi_ids)
    specs = {'opt_top1': dict(template=best['template'], template_sha256=sha(best['template']),
                             p_user_optimization=best['fitness'], suite_means=best['suite_means'],
                             wrapper_tokens=best['wrapper_tokens'], source='user_stopped_current_best',
                             evolution_source=best.get('source'), operator=best.get('operator'),
                             evaluation_index=best['evaluation_index'], selection_labels=['user_stopped_current_best'],
                             selection_scores=scores, selection_population='completed ledger at explicit user stop',
                             stop_receipt_sha256=sha(evolution / 'user_stop_snapshot.json'))}
    path = output / 'conditions.json'
    if path.exists():
        previous = read_json(path)
        require(set(previous) == {'opt_top1'} and previous['opt_top1']['template_sha256'] == sha(best['template']),
                'Existing conditions differ from the single stopped best')
        # Keep provenance stable on a resume after the receipt gains finalization metadata.
        specs = previous
    else:
        write_json(path, specs)
    score = dict(p_user=best['fitness'], suite_means=best['suite_means'], goal_role_means=best['goal_role_means'],
                 per_case={r['case_id']: r['goal_user_mean'] for r in best['per_case']},
                 agentdojo_mean=scores['agentdojo_mean'], template=best['template'],
                 provenance='existing completed optimization score; no new native forward')
    write_json(output / 'condition_scores.json', {'opt_top1': score})
    return specs


class BestASR(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.state_path = self.o / 'best_asr_state.json'
        self.reload_state()

    def wait_native_exit(self):
        receipt = read_json(self.e / 'user_stop_snapshot.json')
        pid, start = receipt['native_pid'], receipt['native_start_ticks']
        end = time.monotonic() + 300
        while True:
            row = process_table().get(pid)
            if not row or row['start'] != start or row['state'] == 'Z':
                break
            require(time.monotonic() < end, 'Stopped native process has not finished finalization; no competing server started')
            time.sleep(5)
        # A missing process alone does not establish successful native finalization.
        summary = read_json(self.e / 'summary.json')
        require(summary['best']['fitness'] == receipt['best']['fitness']
                and sha(summary['best']['template']) == sha(receipt['best']['template']),
                'Native finalization changed the recorded stopped best')
        if summary['status'] != 'stopped_by_user':
            require(summary['status'] == 'complete', 'Native finalization summary is incomplete')
            write_json(self.e / 'summary.native_finalization.json', summary)
            summary.update(status='stopped_by_user', stop_requested_unix=receipt['requested_unix'],
                           stopping_reason=receipt['request'],
                           selection_boundary='best among complete records at user stop',
                           native_finalization='current in-flight score and suffix check drained; no further search')
            write_json(self.e / 'summary.json', summary)
        # Verify the actual resource handoff before starting the serving process.
        roots = server_roots(process_table())
        require(len(roots) <= 1, 'Multiple matching serving processes; refusing ambiguous reuse')
        end = time.monotonic() + 90
        while not roots:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            free = [int(line) for line in raw.splitlines() if line.strip()]
            if len(free) == 4 and all(value >= 60000 for value in free):
                break
            require(time.monotonic() < end, 'Native GPU memory has not been released')
            time.sleep(5)
        self.event('evolution', status='stopped_by_user', scored=summary['scored'], best=summary['best']['fitness'])

    def run(self):
        self.recover_stale_stages()
        self.check_baseline()
        specs = prepare_best(self.e, self.o)
        self.state.update(status='running', mode='user_stopped_current_best_asr_only',
                          native_final_probe='skipped: user requested current-best ASR directly')
        self.state.pop('error', None)
        self.save()
        self.stage('ipi-prepare', [PYA, 'ds_ipi.py', 'prepare', '--output', str(self.o / 'ipi'),
                                  '--conditions-json', str(self.o / 'conditions.json')], self.o / 'ipi_prepare.log',
                   lambda: check_ipi_prepare(self.o / 'ipi', specs))
        self.wait_native_exit()
        self.ensure_server()
        pending = []
        jobs = [('ipi-asr', [PYA, '-u', 'ds_ipi.py', 'run', '--output', str(self.o / 'ipi'), '--workers', '8'],
                 self.o / 'ipi_run.log', lambda: check_ipi_asr(self.o / 'ipi', specs)),
                ('agentdojo-asr', [PYA, '-u', 'ds_agentdojo.py', '--output', str(self.o / 'agentdojo'),
                 '--conditions-json', str(self.o / 'conditions.json'), '--workers', '24'],
                 self.o / 'agentdojo_run.log', lambda: check_agentdojo(self.o / 'agentdojo', specs))]
        for name, command, log, validator in jobs:
            if not self.completed(name, validator):
                self.launch(name, command, log)
                pending.append((name, validator))
        self.finish_parallel(pending)
        self.stage('report', [PYA, 'ds_report.py', '--baseline', str(self.b), '--optimized', str(self.o),
                             '--evolution', str(self.e), '--out', str(self.o / 'REPORT.zh-CN.md')], self.o / 'report.log',
                   lambda: require((self.o / 'REPORT.zh-CN.md').stat().st_size > 1000, 'Final report missing'))
        self.state.update(status='complete')
        self.save()
        self.event('done', report=str(self.o / 'REPORT.zh-CN.md'), vllm='left running')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=HERE / 'outputs/baseline-20261118')
    parser.add_argument('--evolution', type=Path, default=HERE / 'outputs/dsevo-20261118')
    parser.add_argument('--optimized', type=Path, default=HERE / 'outputs/optimized-20261118')
    args = parser.parse_args()
    args.check_only = False
    pipeline = BestASR(args)
    with (pipeline.o / 'phase2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline.reload_state()
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'best ASR received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        pipeline.execute()


if __name__ == '__main__':
    main()
