#!/usr/bin/env bash
set -euo pipefail
task_root=/root/role-retraining-20261026
task_py=/root/role-retraining-venv/bin/python
task_code=$task_root/code-v7
task_public=$task_root/public-gardening-v7
task_runs=/workspace/role-retraining-20261026/runs-v7
task_out=/workspace/role-retraining-20261026/diagnostics/gardening-pilot-vs-first-formal
if [ "${1:-}" != --bounded ]; then
  task_seconds=$("$task_py" - <<'DEADLINE'
import json,time
from datetime import datetime
from pathlib import Path
cap=datetime.fromisoformat('2026-10-27T13:19:17+00:00').timestamp()
ends=[]
for phase in ('pilot','formal'):
    b=json.loads((Path('/workspace/role-retraining-20261026/runs-v7')/phase/'config.json').read_text())['budget']
    start=datetime.fromisoformat(b['started_utc'].replace('Z','+00:00')).timestamp()
    end=datetime.fromisoformat(b['deadline_utc'].replace('Z','+00:00')).timestamp()
    hours=float(b['authorized_hours'])
    assert 0<hours<=24 and start<end<=start+hours*3600
    ends.append(end)
remaining=int(min(cap,*ends)-time.time())
assert remaining>0,'The authorized window has expired'
print(min(900,remaining))
DEADLINE
)
  exec timeout --signal=KILL "${task_seconds}s" bash "$0" --bounded
fi
task_args=(--source-inputs "$task_public/inputs.json" --source-config "$task_public/tomato.yaml"
  --source-template "$task_public/gptoss.j2" --model-dir /root/role-retraining-model
  --pilot-run-dir "$task_runs/pilot" --formal-run-dir "$task_runs/formal"
  --output-dir "$task_out" --devices 0,1,2,3 --cache-dir /root/role-retraining-cache)
mkdir -p "$task_out"
env CUDA_VISIBLE_DEVICES= "$task_py" -u "$task_code/evaluate_synthetic_gardening.py" --stage check "${task_args[@]}" > "$task_out/check.log" 2>&1
env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false "$task_py" -u "$task_code/evaluate_synthetic_gardening.py" --stage infer "${task_args[@]}" > "$task_out/infer.log" 2>&1
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false "$task_py" -u "$task_code/evaluate_synthetic_gardening.py" --stage report "${task_args[@]}" > "$task_out/report.log" 2>&1
