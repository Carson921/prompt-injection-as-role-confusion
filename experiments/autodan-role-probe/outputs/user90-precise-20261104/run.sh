#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261104
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/precise_campaign.py \
  --dataset experiments/autodan-role-probe/outputs/user90-precise-20261104/dataset.json \
  --output experiments/autodan-role-probe/outputs/user90-precise-20261104 \
  --seeds 20261104,20261105,20261106 \
  --stage-lengths 4096,8192,16384,2048,1024,512,128 \
  --scores-per-stage 64 --long-stage-scores 16 --cycles 5 \
  --population 32 --batch-size 16 --breadth-per-stage 32 \
  --max-new-scores 1536 --max-seconds 8500 --target .90 \
  --min-wrapper-tokens 1 --snapshot-every 16 --api-max-calls 400 \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/bootstrap_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/embedding_seed_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/broad_seed_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/feedback/feedback_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/length_scaling_seed_templates.json \
  "$@" > experiments/autodan-role-probe/outputs/user90-precise-20261104/campaign.log 2>&1
