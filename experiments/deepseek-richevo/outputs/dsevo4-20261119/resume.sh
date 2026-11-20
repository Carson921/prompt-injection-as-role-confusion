#!/usr/bin/env bash
# Resume the dsevo4 runs after a GPU interruption: probe server (corrected probe), then both evolutions for their remaining time.
# Needs all four H100s free.  State is restored from scores.jsonl (already scored candidates are never re-scored).
set -euo pipefail
S=/scratch/wz55191/tmux-richevo.sock
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
D=outputs/dsevo4-20261119
if ! curl -sf -m 5 http://127.0.0.1:8791/health > /dev/null; then
  tmux -S $S new-session -d -s dsprobe "cd $PWD && export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/home/wz55191/.cache/triton && /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u ds_probe_server.py --output $D/probe_server >> $D/probe_server.log 2>&1"
  until curl -sf -m 5 http://127.0.0.1:8791/health > /dev/null; do tmux -S $S has-session -t dsprobe || { echo "probe server died"; exit 1; }; sleep 20; done
fi
echo "probe server ready $(date)"
for T in agentdojo ipi; do
  read H FC < <(/scratch/wz55191/venvs/chatinject-table1/bin/python -c "import json;s=json.load(open('$D/stop_state.json'))['$T'];print(s['remaining_hours'], s['remaining_first_cap_minutes'])")
  if /scratch/wz55191/venvs/chatinject-table1/bin/python -c "import sys;sys.exit(0 if float('$H')>0.05 else 1)"; then
    tmux -S $S new-session -d -s dsevo4${T:0:2} "DS_EVO_HOURS=$H bash $D/$T/run.sh --first-cap-minutes $FC >> $D/$T/stdout.log 2>&1"
    echo "$T resumed for $H h (first cap $FC min)"
  fi
done
