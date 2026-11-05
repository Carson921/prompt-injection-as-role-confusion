#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261104
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/adaptive_campaign.py \
  --dataset experiments/autodan-role-probe/outputs/user90-adaptive-20261104/dataset.json \
  --output experiments/autodan-role-probe/outputs/user90-adaptive-20261104 \
  --seeds 20261104,20261105,20261106 \
  --stage-lengths 8192,16384,24576,4096 \
  --scores-per-stage 24 --long-stage-scores 12 --cycles 5 \
  --population 32 --batch-size 8 --breadth-per-stage 4 \
  --max-new-scores 1536 --max-seconds 4200 --target .90 \
  --min-wrapper-tokens 1 --snapshot-every 8 --api-max-calls 100 \
  --api-immigrants 2 --ga-every 2 --api-every 2 \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/bootstrap_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/embedding_seed_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/broad_seed_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/feedback/feedback_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-20261104/length_scaling_seed_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-precise-20261104/factorial_tail_seed_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-precise-20261104/surrogate_shortlist_templates.json \
  --initial-template experiments/autodan-role-probe/outputs/user90-adaptive-20261104/length24576_seed_templates.json \
  "$@" > experiments/autodan-role-probe/outputs/user90-adaptive-20261104/campaign.log 2>&1
