#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo"
task_python=${TASK_PYTHON:-"$task_repo/.venv/bin/python"}
task_out=${1:-"$task_repo/experiments/single-message-pre-mlp/outputs/gpt-oss-120b-c4-1000-dolma3-1000"}
task_code="$task_repo/experiments/single-message-pre-mlp/collect.py"
task_model=${TASK_MODEL:-"/scratch/wz55191/models/gpt-oss-120b"}
task_batch=${TASK_BATCH_SIZE:-4}
mkdir -p "$task_out"
export HF_HOME=${HF_HOME:-"$task_repo/experiments/single-message-pre-mlp/outputs/huggingface"}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$task_python" -u "$task_code" prepare --model "$task_model" --output "$task_out" > "$task_out/prepare.log" 2>&1
task_pids=()
trap 'for task_pid in "${task_pids[@]}"; do kill "$task_pid" 2>/dev/null || true; done' INT TERM
for task_worker in 0 1; do
    "$task_python" -u "$task_code" extract --model "$task_model" --output "$task_out" \
        --batch-size "$task_batch" \
        --device "$task_worker" --workers 2 --worker-index "$task_worker" \
        > "$task_out/worker-$task_worker.log" 2>&1 &
    task_pids+=("$!")
done
task_failed=0
for task_pid in "${task_pids[@]}"; do
    if ! wait "$task_pid"; then task_failed=1; fi
done
if (( task_failed )); then
    echo "An extraction worker failed; inspect worker logs and rerun to resume." >&2
    exit 1
fi
"$task_python" -u "$task_code" verify --output "$task_out" > "$task_out/verify.log" 2>&1
echo "Collection and verification complete: $task_out"
