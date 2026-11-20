#!/usr/bin/env bash
# Third DeepSeek evolution, agentdojo-only objective (cached-prefix surrogate via ds_probe_server.py on port 8791).
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261119 DS_TARGET=agentdojo DS_EXACT_STEP=0.02
E=outputs/dsevo3-20261119/agentdojo
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_evolution3.py \
  --output $E --dataset $E/dataset.json --model /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731 \
  --hours "${DS_EVO_HOURS:-2.5}" --caps 24,40,64 --first-cap-minutes 60 --cap-minutes 45 --stall-evals 350 \
  --population 40 --grammar-seeds 30 --warm-start $E/warm_start.jsonl --warm-start-limit 400 \
  --llm-workers 3 --llm-queue 16 --max-usd 10 --max-calls 1000 --seed 2026111903 "$@"
