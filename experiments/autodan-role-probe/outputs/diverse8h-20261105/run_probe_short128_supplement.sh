#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261105
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/evaluate_short_existing.py \
  --catalog experiments/autodan-role-probe/outputs/diverse8h-20261105/readonly_additional_existing_short128.jsonl \
  --extra-catalog experiments/autodan-role-probe/outputs/diverse8h-20261105/readonly_jsonl_literal_snapshot_catalog.jsonl \
  --cache-index experiments/autodan-role-probe/outputs/diverse8h-20261105/strict_complete_short128_phase2_cache_index.json \
  --dataset experiments/autodan-role-probe/outputs/diverse8h-20261105/dataset.json \
  --budget experiments/autodan-role-probe/outputs/diverse8h-20261105/budget.json \
  --output experiments/autodan-role-probe/outputs/diverse8h-20261105/probe_short128_supplement \
  > experiments/autodan-role-probe/outputs/diverse8h-20261105/probe_short128_supplement.log 2>&1
