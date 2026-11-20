#!/usr/bin/env bash
# Stop both evolutions and the probe server cleanly and record the state (scores.jsonl / bank.json / llm_seeds.jsonl are the resumable state).
S=/scratch/wz55191/tmux-richevo.sock
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
for name in dsevo4ad dsevo4ipi; do tmux -S $S send-keys -t $name C-c 2>/dev/null; done
sleep 25
for name in dsevo4ad dsevo4ipi; do tmux -S $S kill-session -t $name 2>/dev/null; done
/scratch/wz55191/venvs/chatinject-table1/bin/python outputs/dsevo4-20261119/save_state.py > outputs/dsevo4-20261119/stop_state.log 2>&1
tmux -S $S kill-session -t dsprobe 2>/dev/null
echo "stopped $(date)" >> outputs/dsevo4-20261119/stop_state.log
