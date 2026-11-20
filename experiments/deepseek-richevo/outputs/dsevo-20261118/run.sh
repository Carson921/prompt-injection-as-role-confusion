#!/usr/bin/env bash
# DeepSeek-V4-Flash rich evolution: 6 AgentDojo contexts + 6 IPI Arena behaviors, goal-token P(user).
set -euo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261118
E=outputs/dsevo-20261118
exec /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u ds_evolution.py \
  --output $E --dataset $E/dataset.json --model /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731 \
  --hours "${DS_EVO_HOURS:-6}" --caps 128,256,512 --first-cap-minutes 60 --cap-minutes 60 --stall-evals 250 \
  --population 40 --grammar-seeds 100 --warm-start $E/warm_start.jsonl --warm-start-limit 150 \
  --llm-workers 4 --llm-queue 20 --max-usd 20 --max-calls 1500 --seed 20261118 "$@"
