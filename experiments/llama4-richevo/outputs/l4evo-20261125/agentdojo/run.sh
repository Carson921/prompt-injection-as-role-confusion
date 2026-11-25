#!/usr/bin/env bash
# Llama 4 Scout evolution (probe llama4-scout-17b-16e-role-probe-20261125), agentdojo-only objective; cached-prefix surrogate via ds_probe_server.py (port 8791).
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
E=outputs/l4evo-20261125/agentdojo
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261125 DS_TARGET=agentdojo DS_EXACT_STEP=${DS_EXACT_STEP:-0.02} DS_EXACT_FLOOR=${DS_EXACT_FLOOR:-0.75} DS_FACTS=$E/facts.txt
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_evolution4.py \
  --output $E --dataset $E/dataset.json --model /scratch/wz55191/Hugginface/Llama-4-Scout-17B-16E-Instruct \
  --hours "${DS_EVO_HOURS:-3}" --caps 24,40,64 --first-cap-minutes 75 --cap-minutes 50 --grammar-seeds 60 --stall-evals 350 \
  --warm-start $E/warm_start.jsonl --warm-start-limit 80 --population 40 --llm-workers 3 --llm-queue 16 --max-usd 10 --max-calls 1000 --seed 2026112501 "$@"
