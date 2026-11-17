#!/usr/bin/env bash
set -euo pipefail

task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo"
task_out=${TASK_OUTPUT:-"$task_repo/experiments/autodan-role-probe/outputs/near-delimiter-20261104"}
task_model=${TASK_MODEL:-/tmp/wz55191-agentdojo-first-tool-model}
task_probe="$task_repo/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz"
task_score_python="$task_repo/.venv/bin/python"
task_eval_python=/scratch/wz55191/venvs/chatinject-table1/bin/python
task_plot_python=${TASK_PLOT_PYTHON:-/home/wz55191/miniconda3/envs/AutoDAN/bin/python}
task_code="$task_repo/experiments/autodan-role-probe"
task_dataset=${TASK_DATASET:-"$task_code/outputs/pilot-20261104/dataset.json"}
task_previous=${TASK_PREVIOUS_WINNER:-"$task_code/outputs/pilot-20261104/winner.json"}
task_near_rate=${TASK_NEAR_RATE:-0.5}
mkdir -p "$task_out"

# Two H100s; native BF16/MXFP4 pre-o_proj features during optimization.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20261104
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
# Reuse the exact six-task panel, without sampling again.
if [[ "$task_dataset" != "$task_out/dataset.json" ]]; then
  if [[ -f "$task_out/dataset.json" ]]; then
    cmp "$task_dataset" "$task_out/dataset.json"
  else
    cp "$task_dataset" "$task_out/dataset.json"
  fi
fi
"$task_eval_python" "$task_code/audit_near.py" --output "$task_out" \
  --model "$task_model" --mode preflight --report "$task_out/near_context_preflight.json"
task_search_options=(--near-delimiter-rate "$task_near_rate" --warm-start-template "$task_previous")
if [[ -n "${TASK_NEAR_LIBRARY:-}" ]]; then
  task_search_options+=(--near-delimiter-library "$TASK_NEAR_LIBRARY")
fi
"$task_score_python" -u "$task_code/search.py" --dataset "$task_out/dataset.json" \
  --output "$task_out" --model "$task_model" --probe "$task_probe" \
  --population 12 --generations 4 --seed 20261104 "${task_search_options[@]}" > "$task_out/search.log" 2>&1
"$task_score_python" "$task_code/audit.py" --output "$task_out" \
  --dataset "$task_out/dataset.json" --probe "$task_probe"
"$task_eval_python" "$task_code/audit_near.py" --output "$task_out" \
  --model "$task_model" --mode search --report "$task_out/near_search_audit.json"

# The native scorer has exited before the vLLM engine is started.
export OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/tmp/wz55191-autodan-vllm-cache
"$task_eval_python" -u "$task_code/evaluate.py" --dataset "$task_out/dataset.json" \
  --winner "$task_out/winner.json" --output "$task_out/eval_full" --scope full \
  --previous-winner "$task_previous" --near-delimiter-library "$task_out/near_delimiter_library.json" \
  --model "$task_model" --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --workers 32 --max-num-seqs 32 --enforce-eager > "$task_out/eval_full.log" 2>&1
"$task_eval_python" "$task_code/audit_asr.py" --run-root "$task_out/eval_full" --dataset "$task_out/dataset.json"
"$task_score_python" "$task_code/report.py" --output "$task_out" --full-eval "$task_out/eval_full" \
  --previous-winner "$task_previous" --plot-python "$task_plot_python"
