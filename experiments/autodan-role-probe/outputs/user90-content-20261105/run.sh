#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261105
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/content_campaign.py \
  --dataset experiments/autodan-role-probe/outputs/user90-content-20261105/dataset.json \
  --output experiments/autodan-role-probe/outputs/user90-content-20261105 \
  --seeds 20261105,20261106,20261107 --stage-lengths 3165 \
  --scores-per-stage 24 --cycles 20 \
  --population 16 --batch-size 8 --breadth-per-stage 4 \
  --max-new-scores 1536 --max-seconds 3300 --target .90 \
  --min-wrapper-tokens 1 --snapshot-every 8 --api-max-calls 100 \
  --api-immigrants 2 --ga-every 2 --api-every 2 \
  --initial-template experiments/autodan-role-probe/outputs/user90-precise-20261104/asr_selected_best.json \
  "$@" > experiments/autodan-role-probe/outputs/user90-content-20261105/campaign.log 2>&1
