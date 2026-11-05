#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--execute-reviewed-format60" || "$#" -ne 1 ]]; then
  echo "Prepared only. Native execution requires the explicit reviewed launch flag. No model started." >&2
  exit 64
fi
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20261105
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
exec .venv/bin/python -u experiments/autodan-role-probe/evaluate_format60_existing.py \
  --contract-sha256 e56d08616e68be7f7884dc2307ee34623e3b3063493bc05300f0a3dd942afcb1 \
  --execute \
  > experiments/autodan-role-probe/outputs/diverse8h-20261105/probe_format60_short128.log 2>&1
