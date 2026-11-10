#!/usr/bin/env bash
set -uo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261109 TRITON_CACHE_DIR=/tmp/wz55191-autodan-role-probe-triton
OUT=experiments/chatinject-defense-prefix/outputs/defevo-20261109
mkdir -p $OUT
.venv/bin/python -u experiments/chatinject-defense-prefix/defense_evolution.py --output $OUT \
  --hours ${HOURS:-4} ${DEADLINE:+--deadline-unix $DEADLINE} ${SKIPWAVE:+--skip-initial-wave} \
  --caps 128,256,512 --first-cap-minutes 60 --cap-minutes 45 --stall-evals 400 \
  --population 40 --grammar-seeds 150 --warm-start experiments/chatinject-defense-prefix/outputs/defense_seeds.jsonl \
  --warm-start-limit 60 --llm-workers 4 --llm-queue 20 --max-usd 15 --max-calls 1200 --seed 20261109 \
  >> $OUT/stdout.log 2>&1
echo "exit $?" >> $OUT/stdout.log
