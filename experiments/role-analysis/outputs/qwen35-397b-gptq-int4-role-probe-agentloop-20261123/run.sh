#!/usr/bin/env bash
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export QWEN_SCAN_STAT=48 QWEN_SCAN_INNER=24 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
N=experiments/role-analysis/outputs/qwen35-397b-gptq-int4-role-probe-agentloop-20261123
/scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u experiments/role-analysis/run_qwen35_role_probe.py --stage "${1:-all}" --output $N 2>&1 | tee -a $N/run.log
echo "exit ${PIPESTATUS[0]} $(date)" >> $N/run.status
