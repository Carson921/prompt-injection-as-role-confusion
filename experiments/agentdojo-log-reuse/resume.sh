#!/bin/bash
# Run this inside a fresh allocation with two suitable GPUs.
# Optional argument: GPU budget in minutes (default 120).
set -euo pipefail
TASK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec /scratch/wz55191/Github/prompt-injection-as-role-confusion/.venv/bin/python \
    -u "$TASK_DIR/launch_budgeted.py" --budget-minutes "${1:-120}"
