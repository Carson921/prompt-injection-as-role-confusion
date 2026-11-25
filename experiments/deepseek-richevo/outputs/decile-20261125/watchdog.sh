#!/usr/bin/env bash
# Stop the ASR runners before the GPU job ends (both runners skip finished case files on the next start; asr2.sh then stops vLLM).
# usage: watchdog.sh "HH:MM"
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
O=outputs/decile-20261125
until [ "$(date +%s)" -ge "$(date -d "$1" +%s)" ]; do
  grep -q '^exit 0' $O/asr2.log 2>/dev/null && { echo "asr finished before the deadline $(date)"; exit 0; }
  sleep 20
done
echo "deadline reached $(date)"
pkill -TERM -f "ds_agentdojo.py --output .*decile-20261125"
pkill -TERM -f "ds_ipi.py run --output .*decile-20261125"
until grep -q '^exit 0\|vllm stopped' $O/asr2.log; do sleep 5; done
echo "stopped $(date)"
