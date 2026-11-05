"""Immutable collection binding and atomic records for native ASR continuations."""
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

BASE = Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103').resolve()
OUTPUT = BASE.with_name(BASE.name + '_asr')
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')
PROJECT = Path('/scratch/wz55191/Github/prompt-injection-as-role-confusion')
BENCHMARK_PYTHON = '/scratch/wz55191/venvs/chatinject-table1/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def default(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def serial(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=default)


def signature(value):
    return hashlib.sha256(serial(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=default) + '\n')
    tmp.replace(path)


def source_phase(record):
    return (BASE / record['source_result']).parents[4]


def phase(record):
    return BASE if record['phase'] == 'original' else BASE / 'serialization_recovery_v2'
