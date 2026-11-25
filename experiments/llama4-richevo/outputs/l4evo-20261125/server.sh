#!/usr/bin/env bash
# persistent native probe server (all four GPUs); log outputs/l4evo-20261125/probe_server.log
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
exec /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u ds_probe_server.py --output outputs/l4evo-20261125/probe_server > outputs/l4evo-20261125/probe_server.log 2>&1
