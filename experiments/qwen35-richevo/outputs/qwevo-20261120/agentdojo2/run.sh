#!/usr/bin/env bash
# Qwen3.5 evolution (probe agentloop-20261123, sha 5d804b0f), agentdojo-only objective, round 2 warm-started from round-1 IPI and AgentDojo templates; cached-prefix surrogate via ds_probe_server.py (port 8791).
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/qwen35-richevo
E=outputs/qwevo-20261120/agentdojo2
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261123 DS_TARGET=agentdojo DS_EXACT_STEP=0.02 DS_EXACT_FLOOR=0.30 DS_FACTS=$E/facts.txt
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_evolution4.py \
  --output $E --dataset $E/dataset.json --model /scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --hours "${DS_EVO_HOURS:-2.5}" --caps 48,64,96 --first-cap-minutes 70 --cap-minutes 45 --grammar-seeds 30 --warm-start $E/warm_start.jsonl --warm-start-limit 400 --stall-evals 350 \
  --population 40 --llm-workers 3 --llm-queue 16 --max-usd 10 --max-calls 1000 --seed 2026112402 "$@"
