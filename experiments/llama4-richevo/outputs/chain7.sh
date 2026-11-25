#!/usr/bin/env bash
# decile set (one searched candidate per P(user) bin): probe server -> exact panel P(user) -> stop server -> ASR.  Log: outputs/chain7.log
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python; E=outputs/l4evo-20261125; O=outputs/decile-20261125; S=/scratch/wz55191/tmux-richevo.sock
export TOKENIZERS_PARALLELISM=false
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
mv $E/probe_server.log $E/probe_server.graded2.log
tmux -S $S new-session -d -s l4server "bash $E/server.sh"
until grep -q '"stage": "ready"' $E/probe_server.log 2>/dev/null; do
  tmux -S $S has-session -t l4server 2>/dev/null || { echo "probe server died"; tail -5 $E/probe_server.log | cut -c1-500; exit 1; }
  sleep 10
done
echo "probe server ready $(date)"
PICKS=$($PY -c "import json;print(' '.join(json.load(open('$O/conditions.json'))))")
export DS_EXTRA_CONDITIONS=$O/conditions.json
for t in agentdojo ipi; do $PY -u ds_refs4.py $E/$t/dataset.json $E/$t/refs.json $PICKS >> $E/refs.log 2>&1; echo "$t refs exit $? $(date)"; done
tmux -S $S kill-session -t l4server
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
bash asr2.sh $O > $O/asr2.log 2>&1; echo "decile asr exit $? $(date)"
tail -3 $O/asr2.log | cut -c1-2500
echo "chain7 done $(date)"
