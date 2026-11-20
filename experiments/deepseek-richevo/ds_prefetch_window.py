"""Read-only prefetch beside an existing native DeepSeek load; independent of its protocol.

Uses only the native process's existing smaps PTEs to locate its active shard.
Reads at most that shard and its immediate successor with two 16 MiB buffers.
No model tensors, files, mappings, settings or native process state are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = Path('/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731')
SHARD = re.compile(r'^model-(\d{5})-of-00048\.safetensors$')
BUFFER_BYTES = 16 * 1024 * 1024
THRESHOLD_KIB = 4 * 1024


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def referenced_shards(smaps, model, max_shard=44):
    """Only real native mappings of shards 1..44; global/MTP files cannot move the cursor."""
    result = {}
    current = None
    for line in smaps.splitlines():
        if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
            current = None
            fields = line.split(None, 5)
            if len(fields) != 6:
                continue
            path = Path(fields[5])
            match = SHARD.fullmatch(path.name)
            if path.parent == model and match and 1 <= int(match[1]) <= max_shard:
                current = int(match[1])
        elif current is not None and line.startswith('Referenced:'):
            result[current] = result.get(current, 0) + int(line.split()[1])
    return {index: kib for index, kib in result.items() if kib > THRESHOLD_KIB}


def window_for(references, max_shard=44):
    current = max(references, default=None)
    return current, set() if current is None else {i for i in (current, current + 1) if i <= max_shard}


def matches_process(args, cwd, model, output, kind):
    try:
        if kind == 'vllm':
            return any(Path(arg).name == 'vllm' and args[i + 1] == 'serve'
                       and Path(args[i + 2]).resolve() == model for i, arg in enumerate(args[:-2]))
        expected_script = 'ds_evolution.py' if kind == 'evolution' else 'ds_final_probe.py'
        if not any(Path(arg).name == expected_script for arg in args):
            return False
        if kind == 'evolution' and Path(args[args.index('--model') + 1]).resolve() != model:
            return False
        if kind == 'final-probe' and model != DEFAULT_MODEL.resolve():
            return False
        return (cwd / args[args.index('--output') + 1]).resolve() == output
    except (OSError, ValueError, IndexError):
        return False


def native_identity(pid, model, output, kind='evolution'):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        args = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        cwd = (root / 'cwd').readlink()
        return fields[19] if fields[0] != 'Z' and matches_process(args, cwd, model, output, kind) else None
    except (OSError, ValueError, IndexError):
        return None


def owned_descendants(root_pid):
    parents = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            if path.stat().st_uid == os.getuid():
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(path.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    result = {root_pid}
    while True:
        more = {pid for pid, parent in parents.items() if parent in result}
        if more <= result:
            return result
        result |= more


def observed_window(pid, model, kind):
    if kind != 'vllm':
        references = referenced_shards((Path('/proc') / str(pid) / 'smaps').read_text(), model)
        current, window = window_for(references)
        return references, current, window, {pid: current}
    frontiers, combined = {}, {}
    for child in owned_descendants(pid):
        try:
            root = Path('/proc') / str(child)
            if str(model) + '/model-' not in (root / 'maps').read_text():
                continue
            references = referenced_shards((root / 'smaps').read_text(), model, max_shard=45)
            if not references:
                continue
            frontiers[child] = max(references)
            for index, kib in references.items():
                combined[index] = max(combined.get(index, 0), kib)
        except (OSError, ValueError):
            continue
    # Keep up with the slowest rank that has begun reading. Never chase a fast
    # rank several shards ahead of another worker sharing the same cache.
    current = min(frontiers.values(), default=None)
    _, window = window_for({current: 1} if current is not None else {}, max_shard=45)
    return combined, current, window, frontiers


def backend_finished(path, after_unix=None):
    try:
        modified = path.stat().st_mtime
        return after_unix is None or modified >= after_unix
    except OSError:
        return False


def service_ready(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return any(row.get('id') == 'deepseek-v4-flash' for row in json.load(response).get('data', []))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def read_shard(path, cancel, stop):
    started = time.monotonic()
    count = 0
    buffer = bytearray(BUFFER_BYTES)
    with path.open('rb', buffering=0) as stream:
        while not (cancel.is_set() or stop.is_set()):
            size = stream.readinto(buffer)
            if size == 0:
                return dict(file=path.name, bytes=count, seconds=time.monotonic() - started, status='complete')
            count += size
    return dict(file=path.name, bytes=count, seconds=time.monotonic() - started, status='cancelled')


def run(args):
    model, output = args.model.resolve(), args.output.resolve()
    deadline = time.monotonic() + args.wait_identity_seconds
    identity = native_identity(args.pid, model, output, args.kind)
    while identity is None and time.monotonic() < deadline:
        time.sleep(0.2)
        identity = native_identity(args.pid, model, output, args.kind)
    if identity is None:
        raise RuntimeError('Native PID is absent or does not match the expected script, model and output')
    references, current, window, frontiers = observed_window(args.pid, model, args.kind)
    if args.inspect_only:
        print(json.dumps(dict(pid=args.pid, process_start=identity, current=current, window=sorted(window),
                              referenced_mib={k: round(v / 1024, 2) for k, v in references.items()})), flush=True)
        return
    backend_path = output / 'model_backend.json'
    finished = lambda: service_ready(args.ready_url) if args.kind == 'vllm' else backend_finished(backend_path, args.backend_after_unix)
    if finished():
        print(json.dumps(dict(status='stopped', reason='vLLM already ready' if args.kind == 'vllm'
                              else 'model_backend.json already exists')), flush=True)
        return
    status_dir = (args.status_dir or output).resolve()
    status_dir.mkdir(parents=True, exist_ok=True)
    state_path = status_dir / 'weight_prefetch_window_state.json'
    events_path = status_dir / 'weight_prefetch_window_events.jsonl'
    with (status_dir / 'weight_prefetch_window.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = json.loads(state_path.read_text()) if state_path.exists() else {}
        if existing and (existing['native_pid'] != args.pid or existing['process_start'] != identity):
            raise RuntimeError('Previous window state belongs to another native process')
        completed = {int(k): v for k, v in existing.get('completed', {}).items()}
        for index in args.already_read:
            completed.setdefault(index, dict(file=f'model-{index:05d}-of-00048.safetensors', status='complete',
                                            source='previous single-window prefetch'))
        state = dict(native_pid=args.pid, process_start=identity, helper_pid=os.getpid(), status='running',
                     mode='read-only process smaps window', kind=args.kind, max_workers=2, buffer_bytes=BUFFER_BYTES,
                     backend_after_unix=args.backend_after_unix,
                     poll_seconds=3, started_unix=time.time(), completed=completed)
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
        state['nice'] = os.getpriority(os.PRIO_PROCESS, 0)
        pending = {}
        def event(**row):
            row['unix'] = time.time()
            with events_path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        def collect():
            for index, (future, cancel) in list(pending.items()):
                if future.done():
                    row = future.result()
                    event(**row)
                    if row['status'] == 'complete':
                        completed[index] = row
                    del pending[index]
        reason = 'signal'
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                try:
                    while not stop.is_set():
                        if finished():
                            reason = 'vLLM ready' if args.kind == 'vllm' else 'model_backend.json written'
                            break
                        if native_identity(args.pid, model, output, args.kind) != identity:
                            reason = 'native PID disappeared or identity changed'
                            break
                        references, current, window, frontiers = observed_window(args.pid, model, args.kind)
                        collect()
                        for index, (_, cancel) in pending.items():
                            if index not in window:
                                cancel.set()
                        for index in sorted(window):
                            if len(pending) >= 2:
                                break
                            if index not in completed and index not in pending:
                                cancel = threading.Event()
                                path = model / f'model-{index:05d}-of-00048.safetensors'
                                pending[index] = (executor.submit(read_shard, path, cancel, stop), cancel)
                                event(status='reading', file=path.name, current=current)
                        state.update(updated_unix=time.time(), current=current, window=sorted(window),
                                     mapping_frontiers=frontiers,
                                     referenced_mib={k: round(v / 1024, 2) for k, v in references.items()},
                                     reading=sorted(pending), completed=completed)
                        atomic_json(state_path, state)
                        stop.wait(3)
                finally:
                    stop.set()
            collect()
            state.update(status='stopped', reason=reason, reading=[], finished_unix=time.time(), completed=completed)
            atomic_json(state_path, state)
            event(status='stopped', reason=reason)
        except BaseException as error:
            stop.set()
            state.update(status='error', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
            atomic_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('evolution', 'final-probe', 'vllm'), default='evolution')
    parser.add_argument('--status-dir', type=Path)
    parser.add_argument('--backend-after-unix', type=float)
    parser.add_argument('--wait-identity-seconds', type=float, default=0)
    parser.add_argument('--ready-url', default='http://127.0.0.1:8000/v1/models')
    parser.add_argument('--already-read', type=int, nargs='*', default=[27, 28])
    parser.add_argument('--inspect-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
