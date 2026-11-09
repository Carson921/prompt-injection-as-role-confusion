#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261108
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/ipi-arena-richevo/ipi_evolution.py \
  --output experiments/ipi-arena-richevo/outputs/ipievo-20261108 \
  --dataset experiments/ipi-arena-richevo/outputs/ipievo-20261108/ipi_dataset.json \
  --hours 6 --caps 128,256,512 --first-cap-minutes 40 --cap-minutes 60 --stall-evals 300 \
  --population 40 --grammar-seeds 100 \
  --warm-start experiments/ipi-arena-richevo/outputs/ipievo-20261108/warm_start_pool.jsonl --warm-start-limit 80 \
  --llm-workers 4 --llm-queue 20 --max-usd 20 --max-calls 1500 --seed 20261108 "$@"
