#!/usr/bin/env bash
# Second DeepSeek evolution (fast cached-prefix surrogate via ds_probe_server.py on port 8791).
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261119
E=outputs/dsevo2-20261119
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_evolution2.py \
  --output $E --dataset $E/dataset.json --model /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731 \
  --hours "${DS_EVO_HOURS:-3.5}" --caps 128,256,512 --first-cap-minutes 75 --cap-minutes 45 --stall-evals 400 \
  --population 40 --grammar-seeds 150 --warm-start $E/warm_start.jsonl --warm-start-limit 600 \
  --llm-workers 4 --llm-queue 20 --max-usd 15 --max-calls 1500 --seed 20261119 "$@"
