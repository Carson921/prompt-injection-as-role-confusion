#!/usr/bin/env bash
# finish the AgentDojo panel P(user) of the graded set (chain5 stopped at a template whose payload the YAML dump rewrites).  Log: outputs/chain6.log
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python; E=outputs/l4evo-20261125; O=outputs/graded-20261125; S=/scratch/wz55191/tmux-richevo.sock
export TOKENIZERS_PARALLELISM=false
mv $E/probe_server.log $E/probe_server.graded.log
tmux -S $S new-session -d -s l4server "bash $E/server.sh"
until grep -q '"stage": "ready"' $E/probe_server.log 2>/dev/null; do
  tmux -S $S has-session -t l4server 2>/dev/null || { echo "probe server died"; tail -5 $E/probe_server.log | cut -c1-500; exit 1; }
  sleep 10
done
echo "probe server ready $(date)"
PICKS=$($PY -c "import json;print(' '.join(json.load(open('$O/conditions.json'))))")
export DS_EXTRA_CONDITIONS=$O/conditions.json
$PY -u ds_refs4.py $E/agentdojo/dataset.json $E/agentdojo/refs.json $PICKS >> $E/refs.log 2>&1; echo "agentdojo refs exit $? $(date)"
tmux -S $S kill-session -t l4server
echo "chain6 done $(date)"
