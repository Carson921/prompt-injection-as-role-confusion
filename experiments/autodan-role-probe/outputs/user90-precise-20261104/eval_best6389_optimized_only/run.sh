#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u experiments/autodan-role-probe/evaluate_optimized.py \
 --dataset experiments/autodan-role-probe/outputs/user90-precise-20261104/dataset.json \
 --winner experiments/autodan-role-probe/outputs/user90-precise-20261104/asr_selected_best.json \
 --near-delimiter-library experiments/autodan-role-probe/outputs/user90-precise-20261104/near_delimiter_library.json \
 --output experiments/autodan-role-probe/outputs/user90-precise-20261104/eval_best6389_optimized_only \
 --import-completed-from experiments/autodan-role-probe/outputs/user90-precise-20261104/eval_best6389 \
 --scope full --tensor-parallel-size 2 --pipeline-parallel-size 1 \
 --gpu-memory-utilization .85 --max-model-len 65536 \
 --max-num-seqs 32 --workers 32 --max-num-batched-tokens 8192 \
 --max-tokens 1024 --prompt-date 2026-10-21 --enforce-eager \
 "$@" > experiments/autodan-role-probe/outputs/user90-precise-20261104/eval_best6389_optimized_only.log 2>&1
