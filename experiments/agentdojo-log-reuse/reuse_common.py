"""Paths and immutable source bindings for the merged, log-first collection."""
import hashlib
import json
from pathlib import Path

BASE = Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103').resolve()
OLD_ASR = BASE.with_name(BASE.name + '_asr')
LEGACY_MERGED = BASE.with_name(BASE.name + '_log_reuse')
PREVIOUS_CONTINUATION = LEGACY_MERGED / 'missing_asr'
ROOT = BASE.with_name('paper778_20261103_log_reuse')
MISSING = ROOT / 'missing_asr'
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')
PROJECT = Path('/scratch/wz55191/Github/prompt-injection-as-role-confusion')
BENCHMARK_PYTHON = '/scratch/wz55191/venvs/chatinject-table1/bin/python'
NATIVE_PYTHON = str(PROJECT / '.venv/bin/python')
CHAT = BASE.parents[1]
NATIVE_OLD = CHAT / 'agentdojo_role_probe_gptoss120b/20261021T1610Z/full'
REFERENCE = CHAT / 'agentdojo_role_probe_gptoss120b/subdataset_100_seed20261021'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def phase(record):
    return BASE if record['phase'] == 'original' else BASE / 'serialization_recovery_v2'


def serial(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def signature(value):
    return hashlib.sha256(serial(value).encode()).hexdigest()
