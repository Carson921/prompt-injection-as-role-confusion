#!/usr/bin/env bash
set -uo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
OUT=experiments/autodan-role-probe/outputs/richevo-20261105
for G in A B; do
  /scratch/wz55191/venvs/chatinject-table1/bin/python -u experiments/autodan-role-probe/evaluate_fixed_candidates.py \
    --candidates $OUT/asr_candidates_$G.json --output $OUT/asr_$G \
    --dataset experiments/autodan-role-probe/outputs/diverse8h-20261105/dataset.json \
    --maximum-wrapper-tokens 256 > $OUT/asr_$G.log 2>&1
  echo "group $G exit $?" >> $OUT/asr_chain.log
done
