"""Audit and package original hidden states together with native ASR outcomes."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'provenance'))
from common import BASE, OUTPUT, BENCHMARK_PYTHON, read, write, sha


def live(receipt):
    p = Path(f"/proc/{receipt['pid']}/stat")
    try:
        fields = p.read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] not in ('Z', 'X') and fields[19] == str(receipt['proc_start_ticks'])


def progress(stage, **fields):
    write(OUTPUT.parent / (OUTPUT.name + '.completion-progress.json'),
          {'stage': stage, 'updated_at': time.time(), **fields})
    print(json.dumps({'stage': stage, **fields}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        launch = read(OUTPUT / 'launch.json')
        progress('waiting_for_native_producer', producer_pid=launch['pid'])
        while live(launch):
            time.sleep(30)
    terminal = read(OUTPUT / 'producer-terminal.json')
    assert terminal['completed'] == terminal['requested'] == 1898
    lock = (OUTPUT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress('reevaluating_all_cases_and_aligning_probe_labels')
    script = OUTPUT / 'postprocessing/audit_export.py'
    with (OUTPUT / 'audit.log').open('a') as log:
        subprocess.run([BENCHMARK_PYTHON, str(script)], stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = read(OUTPUT / 'audit.json')
    assert audit['status'] == 'all_cases_verified' and audit['verified_cases'] == 1898 and not audit['errors']
    summary = read(OUTPUT / 'asr_summary.json')
    assert summary['all_cases']['cases'] == 1898
    write(OUTPUT / 'progress.json', {'stage': 'completed_and_verified', 'completed': 1898, 'total': 1898,
        'attacker_successes': summary['all_cases']['attacker_successes'], 'asr': summary['all_cases']['asr'],
        'official_predicates_rechecked': True, 'updated_at': time.time()})
    assert read(BASE / 'audit.json')['status'] == 'full_scope_verified'
    assert read(BASE / 'probe_inputs/manifest.json')['status'] == 'complete'
    assert BASE.is_relative_to(Path('/home/wz55191/Github/delimeter-token').resolve())
    description = '''The ASR continuation is paired with the immutable full_20261103 hidden-state collection.

case_results.json / .csv: 1,898 canonical cases, their attacker_success (bool),
asr_label (0/1), utility, payload_exposed, exact first_exposure_call, and the
original hidden-state feature_row_start / feature_row_end. security means
attacker success in this AgentDojo version. ASR is mean(attacker_success).
asr_summary.json: overall and per-condition/per-suite ASR, both across all cases
and across the cases that actually exposed a complete payload.
cases/: native TaskSuite.run_task_with_pipeline outputs with request bindings.
trajectories/: actual pre/final environments, complete AgentDojo messages, final
model output and function stack traces; all official predicates are reevaluated.
calls/: original raw requests, full rendered prompts and input IDs, native output
IDs/text and Harmony-parsed messages. Prefixes replay previously saved outputs
only after exact raw-request, rendered-prompt and full-token-ID equality checks.
The first exposure is generated from the very same full input that produced the
saved hidden states. Subsequent tools execute normally in the real environment.
No prior reasoning text is inserted into later messages (same frozen adapter).

probe_labels/attacker_success_by_case.npy: uint8[1898], canonical case order.
probe_labels/attacker_success_by_token.npy: uint8[1879782], aligned exactly to
both matrices in ../full_20261103/probe_inputs/. It repeats the corresponding
case outcome for each saved row; it is not a token-local success measurement.
Also included: utility_by_case.npy / utility_by_token.npy and
payload_exposed_by_case.npy. Unexposed cases have case labels but no feature rows.
Example:
  X = np.load('../full_20261103/probe_inputs/X_final_v7_16.npy', mmap_mode='r')
  y = np.load('probe_labels/attacker_success_by_token.npy')
  assert len(X) == len(y)

The native generation/loop limits remain 1,024 new tokens per call, 15 tool-loop
iterations per AgentDojo pipeline invocation and up to three native task
attempts. Outcomes are the official predicates under these limits. Limit hits
are explicitly counted rather than silently called natural completions.

The combined archive contains full_20261103/ and full_20261103_asr/ together.
The original hidden-state directory and its earlier archive remain unchanged.
All delivered files reside in the requested delimeter-token repository.
'''
    (OUTPUT / 'DATA_FORMAT.txt').write_text(description)
    progress('hashing_and_packaging_hidden_states_with_asr')
    archive = BASE.with_name(BASE.name + '_with_asr.tar.gz')
    assert not archive.exists(), 'Refusing to replace an existing completed bundle'
    paths = {}
    for root in (BASE, OUTPUT):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name != 'run.lock' and not p.name.endswith(('.tmp', '.log')) and '__pycache__' not in p.parts:
                paths[root.name + '/' + str(p.relative_to(root))] = p
    hashes = {name: sha(p) for name, p in paths.items()}
    checksums = ''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())).encode()
    hashes['SHA256SUMS'] = hashlib.sha256(checksums).hexdigest()
    temp = archive.with_name(archive.name + '.tmp')
    with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
        for name, path in paths.items():
            tar.add(path, arcname=name, recursive=False)
        item = tarfile.TarInfo('SHA256SUMS')
        item.size, item.mode = len(checksums), 0o644
        tar.addfile(item, io.BytesIO(checksums))
    progress('reading_back_every_archive_member')
    seen = set()
    with tarfile.open(temp, 'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in hashes and member.name not in seen
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
            assert h.hexdigest() == hashes[member.name], member.name
            seen.add(member.name)
    assert seen == set(hashes)
    # Also prove no archived data file changed between hashing and archive creation.
    assert all(sha(path) == hashes[name] for name, path in paths.items())
    temp.replace(archive)
    checksum = sha(archive)
    archive.with_name(archive.name + '.sha256').write_text(f'{checksum}  {archive.name}\n')
    receipt = {'status': 'complete', 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
        'archive_sha256': checksum, 'all_archive_member_bytes_readback_verified': True, 'files': len(hashes),
        'original_hidden_state_cases': 1852, 'asr_evaluated_cases': 1898, 'case_labels': 1898, 'token_labels': 1879782,
        'audit_sha256': sha(OUTPUT / 'audit.json'), 'asr_summary': summary, 'finished_at': time.time()}
    write(archive.with_name(archive.name + '.complete.json'), receipt)
    progress('complete', archive=str(archive), sha256=checksum, asr_evaluated_cases=1898)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        progress('failed', error=f'{type(error).__name__}: {error}')
        raise
