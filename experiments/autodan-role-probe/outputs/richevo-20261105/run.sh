#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261105
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/rich_evolution.py \
  --output experiments/autodan-role-probe/outputs/richevo-20261105 \
  --dataset experiments/autodan-role-probe/outputs/diverse8h-20261105/dataset.json \
  --deadline-unix 1793910721.2131538 --skip-initial-wave --hours 8 --caps 128,256,512,1024,2048 --first-cap-minutes 30 --cap-minutes 50 --stall-evals 450 \
  --population 40 --grammar-seeds 150 \
  --warm-start experiments/autodan-role-probe/outputs/richevo-20261105/warm_start_pool.jsonl --warm-start-limit 60 \
  --llm-workers 4 --llm-queue 20 --max-usd 28 --max-calls 1800 --seed 20261105
