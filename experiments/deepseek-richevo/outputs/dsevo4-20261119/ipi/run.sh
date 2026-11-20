#!/usr/bin/env bash
# Fourth DeepSeek evolution (corrected probe 66c100b8), ipi-only objective; cached-prefix surrogate via ds_probe_server.py (port 8791).
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
E=outputs/dsevo4-20261119/ipi
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261119 DS_TARGET=ipi DS_EXACT_STEP=0.03 DS_EXACT_FLOOR=0.45 DS_FACTS=$E/facts.txt
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_evolution4.py \
  --output $E --dataset $E/dataset.json --model /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731 \
  --hours "${DS_EVO_HOURS:-3.5}" --caps 48,96,160 --first-cap-minutes 75 --cap-minutes 60 --stall-evals 350 \
  --population 40 --grammar-seeds 50 --warm-start $E/warm_start.jsonl --warm-start-limit 400 \
  --llm-workers 3 --llm-queue 16 --max-usd 10 --max-calls 1000 --seed 2026111904 "$@"
