#!/usr/bin/env bash
# after both evolutions: exact ranking -> selection -> panel refs for the picks -> stop probe server -> ASR of the picks.  Log: outputs/chain3.log
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python; E=outputs/l4evo-20261125; O=outputs/optimized-20261125; S=/scratch/wz55191/tmux-richevo.sock
export TOKENIZERS_PARALLELISM=false
while tmux -S $S has-session -t l4evoag 2>/dev/null || tmux -S $S has-session -t l4evoip 2>/dev/null; do sleep 30; done
echo "evolutions ended $(date)"
for t in agentdojo ipi; do
  $PY -u ds_finalize4.py $E/$t 10 > $E/$t/finalize.log 2>&1; echo "$t finalize exit $? $(date)"
done
$PY ds_select4.py $E $O > $O/select.log 2>&1 || { echo "selection failed"; cat $O/select.log; exit 1; }
cat $O/select.log
PICKS=$($PY -c "import json;print(' '.join(json.load(open('$O/conditions.json'))))")
export DS_EXTRA_CONDITIONS=$O/conditions.json
for t in agentdojo ipi; do $PY -u ds_refs4.py $E/$t/dataset.json $E/$t/refs.json $PICKS >> $E/refs.log 2>&1; done
echo "panel refs done $(date)"
tmux -S $S kill-session -t l4server
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
bash asr2.sh $O > $O/asr2.log 2>&1; echo "optimized asr exit $? $(date)"
tail -3 $O/asr2.log | cut -c1-1500
echo "chain3 done $(date)"
