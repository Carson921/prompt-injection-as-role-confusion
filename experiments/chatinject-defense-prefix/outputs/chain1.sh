#!/usr/bin/env bash
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
OUT=experiments/chatinject-defense-prefix/outputs
env OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13 \
  /scratch/wz55191/venvs/chatinject-table1/bin/python -u experiments/chatinject-defense-prefix/evaluate_defense_asr.py \
  --conditions $OUT/baseline_conditions.json --output $OUT/asr_baseline > $OUT/asr_baseline.log 2>&1
echo "asr_baseline exit $? $(date)" >> $OUT/chain.log
sleep 20
HOURS=4 $OUT/run_evo.sh
echo "defevo exit $? $(date)" >> $OUT/chain.log
