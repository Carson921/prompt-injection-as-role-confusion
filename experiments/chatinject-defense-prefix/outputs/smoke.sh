#!/usr/bin/env bash
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR=/tmp/wz55191-autodan-role-probe-triton
.venv/bin/python -u experiments/chatinject-defense-prefix/defense_scorer.py --output experiments/chatinject-defense-prefix/outputs/smoke > experiments/chatinject-defense-prefix/outputs/smoke.log 2>&1
echo "exit $?" >> experiments/chatinject-defense-prefix/outputs/smoke.log
