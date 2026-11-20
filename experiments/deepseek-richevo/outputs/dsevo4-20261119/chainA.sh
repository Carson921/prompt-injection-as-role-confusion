#!/usr/bin/env bash
# After the AgentDojo evolution: exact ranking of its leading candidates and the remaining AgentDojo-panel references.
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo; export TOKENIZERS_PARALLELISM=false
while tmux -S /scratch/wz55191/tmux-richevo.sock has-session -t dsevo4ag 2>/dev/null; do sleep 30; done
echo "AD evolution ended $(date)"
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_finalize4.py outputs/dsevo4-20261119/agentdojo 10 > outputs/dsevo4-20261119/agentdojo/finalize.log 2>&1; echo "AD finalize exit $? $(date)"
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/agentdojo/dataset.json outputs/dsevo4-20261119/agentdojo/refs.json ctrl_leadin opt2_top2 opt2_assistant original_delimiter original goal_only >> outputs/dsevo4-20261119/refs.log 2>&1
echo "chainA done $(date)"
