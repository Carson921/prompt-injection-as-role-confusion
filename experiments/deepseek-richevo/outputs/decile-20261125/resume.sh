#!/usr/bin/env bash
# Decile set (one dsevo4 search candidate per P(user) bin), continuation in a new GPU job.  Needs all four H100s free.
#   1. ASR on AgentDojo (389) and IPI Arena (36) for conditions.json -- finished case files are skipped.
#   2. Probe server (corrected probe) -> exact panel P(user) of the picks on both 12-context panels -> stop server.
# usage: bash outputs/decile-20261125/resume.sh [HH:MM deadline for step 1]     log: outputs/decile-20261125/resume.log
S=/scratch/wz55191/tmux-richevo.sock
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
O=outputs/decile-20261125; E=outputs/dsevo4-20261119; PY=/scratch/wz55191/venvs/chatinject-table1/bin/python
export TOKENIZERS_PARALLELISM=false
[ -n "${1:-}" ] && tmux -S $S new-session -d -s dswatch "bash $O/watchdog.sh $1 >> $O/watchdog.log 2>&1"
mv $O/asr2.log $O/asr2.$(date +%m%d-%H%M).log 2>/dev/null
bash asr2.sh $O > $O/asr2.log 2>&1; echo "asr exit $? $(date)"
tail -2 $O/asr2.log | cut -c1-2500
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
tmux -S $S new-session -d -s dsprobe "cd $PWD && export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/home/wz55191/.cache/triton && /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u ds_probe_server.py --output $O/probe_server >> $O/probe_server.log 2>&1"
until curl -sf -m 5 http://127.0.0.1:8791/health > /dev/null; do tmux -S $S has-session -t dsprobe 2>/dev/null || { echo "probe server died"; tail -3 $O/probe_server.log | cut -c1-400; exit 1; }; sleep 20; done
echo "probe server ready $(date)"
PICKS=$($PY -c "import json;print(' '.join(sorted(json.load(open('$O/conditions.json')))))")
export DS_EXTRA_CONDITIONS=$O/conditions.json
for t in agentdojo ipi; do $PY -u ds_refs4.py $E/$t/dataset.json $O/refs_$t.json $PICKS >> $O/refs.log 2>&1; echo "$t refs exit $? $(date)"; done
tmux -S $S kill-session -t dsprobe
echo "decile chain done $(date)"
