#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u \
  experiments/autodan-role-probe/evaluate_fixed_candidates.py \
  --candidates experiments/autodan-role-probe/outputs/diverse8h-20261105/asr_candidates_final.json \
  --output experiments/autodan-role-probe/outputs/diverse8h-20261105/asr_final \
  --maximum-wrapper-tokens 128 \
  > experiments/autodan-role-probe/outputs/diverse8h-20261105/asr_final.log 2>&1
