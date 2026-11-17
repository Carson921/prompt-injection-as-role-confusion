#!/usr/bin/env bash
set -uo pipefail
D=/scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/agentic-probe/outputs/phase2-g38-20261111
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/agentic-probe
COMMON="--defender-model google/gemini-3.8-flash --attacker-model google/gemini-3.8-flash --teams 4 --agents 2 --evals 12 --max-usd-per-scenario 3 --workers 3"
run() { name=$1; shift; $PY -u run_swarm.py --output $D/$name --name $name $COMMON "$@" > $D/$name.log 2>&1; echo "exit $? $name" >> $D/done.log; }
run probe_diag --strategies strategies_seed.json --probe-url http://127.0.0.1:8765 --shaping 0.1 --diagnostics &
run probe_gate --strategies strategies_seed.json --probe-url http://127.0.0.1:8765 --gate 0.5 --shaping 0.1 --diagnostics &
run probe_full --strategies strategies_seed.json --c0 band_0.99_1.00 --probe-url http://127.0.0.1:8765 --gate 0.5 --shaping 0.1 --diagnostics &
wait
echo "probe conditions done" >> $D/done.log
