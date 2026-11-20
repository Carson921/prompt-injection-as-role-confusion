#!/usr/bin/env bash
# Checked, resumable evolution -> probe -> ASR -> report chain. Keeps vLLM running afterwards.
set -euo pipefail
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
exec /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_phase2.py "$@"
