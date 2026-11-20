#!/usr/bin/env bash
# Retrain the DeepSeek role probe with developer -> system (GPT-OSS system dropped). Resumable.
set -uo pipefail
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/home/wz55191/.cache/triton
/scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/role-analysis/run_deepseek_role_probe.py --stage all \
  --model /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731 --output /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/deepseek-v4-flash-0731-role-probe-devsys-20261119 2>&1 | tee -a /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/deepseek-v4-flash-0731-role-probe-devsys-20261119/run.log
echo "exit ${PIPESTATUS[0]} $(date)" >> /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/deepseek-v4-flash-0731-role-probe-devsys-20261119/run.status
