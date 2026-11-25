#!/usr/bin/env bash
# baseline / reference ASR -> scoring contexts -> probe server -> reference P(user).  Log: outputs/chain2.log
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python; E=outputs/l4evo-20261125; S=/scratch/wz55191/tmux-richevo.sock
export TOKENIZERS_PARALLELISM=false
bash asr2.sh outputs/baseline-20261125 > outputs/baseline-20261125/asr2.log 2>&1
echo "baseline asr exit $? $(date)"; tail -3 outputs/baseline-20261125/asr2.log | cut -c1-1500
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
$PY ds_build_contexts.py --run outputs/baseline-20261125/agentdojo --condition original --per-suite 4 --ipi-cases '' --out $E/agentdojo/dataset.json > $E/agentdojo/build_contexts.log 2>&1
echo "contexts exit $? $(date)"; tail -2 $E/agentdojo/build_contexts.log | cut -c1-600
tmux -S $S new-session -d -s l4server "bash $E/server.sh"
until grep -q '"stage": "ready"' $E/probe_server.log 2>/dev/null; do
  tmux -S $S has-session -t l4server 2>/dev/null || { echo "probe server died"; tail -5 $E/probe_server.log | cut -c1-500; exit 1; }
  sleep 10
done
echo "probe server ready $(date)"
$PY -u ds_refs4.py $E/ipi/dataset.json $E/ipi/refs.json > $E/refs.log 2>&1; echo "ipi refs exit $? $(date)"
$PY -u ds_refs4.py $E/agentdojo/dataset.json $E/agentdojo/refs.json >> $E/refs.log 2>&1; echo "agentdojo refs exit $? $(date)"
echo "chain2 done $(date)"
