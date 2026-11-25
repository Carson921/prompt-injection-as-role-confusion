#!/usr/bin/env bash
# after chain3: ASR of the second batch of hand-written references.  Log: outputs/chain4.log
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
until grep -qE 'chain3 done|selection failed' outputs/chain3.log 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
bash asr2.sh outputs/reference2-20261125 > outputs/reference2-20261125/asr2.log 2>&1; echo "reference2 asr exit $? $(date)"
tail -3 outputs/reference2-20261125/asr2.log | cut -c1-1500
echo "chain4 done $(date)"
