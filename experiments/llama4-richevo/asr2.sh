#!/usr/bin/env bash
# ASR for a conditions.json: start vLLM (if not ready), IPI prepare, IPI + AgentDojo runs in parallel, stop vLLM.
# usage: asr2.sh OUT_DIR   (OUT_DIR/conditions.json must exist; native probing must be stopped: all four GPUs are needed)
set -uo pipefail
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
O="$(readlink -f "$1")"
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python
cd "$D"
ready() { curl -sf -m 5 http://127.0.0.1:8000/v1/models > /dev/null; }
SERVER=
if ! ready; then
  bash serve_vllm.sh > "$O/vllm.log" 2>&1 &
  SERVER=$!
  echo "vllm start $(date) pid $SERVER"
  until ready; do
    kill -0 "$SERVER" 2>/dev/null || { echo "vllm died"; tail -30 "$O/vllm.log"; exit 3; }
    sleep 20
  done
fi
echo "vllm ready $(date)"
$PY ds_ipi.py prepare --output "$O/ipi" --conditions-json "$O/conditions.json" > "$O/ipi_prepare.log" 2>&1 || { echo "ipi prepare failed"; tail -20 "$O/ipi_prepare.log"; }
$PY -u ds_ipi.py run --output "$O/ipi" --workers "${L4_IPI_WORKERS:-16}" > "$O/ipi_run.log" 2>&1 &
IPI=$!
$PY -u ds_agentdojo.py --output "$O/agentdojo" --conditions-json "$O/conditions.json" --workers "${L4_AD_WORKERS:-72}" > "$O/agentdojo_run.log" 2>&1
echo "agentdojo exit $? $(date)"
wait "$IPI"; echo "ipi exit $? $(date)"
tail -1 "$O/agentdojo_run.log"; tail -1 "$O/ipi_run.log"
if [[ -n "$SERVER" ]]; then
  pkill -TERM -P "$SERVER" 2>/dev/null; kill -TERM "$SERVER" 2>/dev/null
  sleep 30; pkill -KILL -f 'vllm serve /scratch/wz55191/Hugginface/Llama-4-Scout-17B-16E-Instruct' 2>/dev/null
  echo "vllm stopped $(date)"
fi
echo "exit 0"
