#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--execute-reviewed-format2" || "$#" -ne 1 ]]; then
  echo "Prepared only. Native execution requires the explicit reviewed launch flag. No model started." >&2
  exit 64
fi
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261105
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/evaluate_format2_existing.py \
  --contract-sha256 9a40d6b27a2aededdb82006cb3c7fe2732e032ecfc2ef5b863d82c42fa54e7d2 \
  --execute \
  > experiments/autodan-role-probe/outputs/diverse8h-20261105/probe_format2_short128.log 2>&1
