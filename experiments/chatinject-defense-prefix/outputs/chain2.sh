#!/bin/bash
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/chatinject-defense-prefix
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python
$PY evaluate_defense_asr.py --conditions outputs/defense_conditions.json --output outputs/asr_defense >> outputs/asr_defense.log 2>&1
echo "asr_defense exit $? $(date)" >> outputs/chain.log
