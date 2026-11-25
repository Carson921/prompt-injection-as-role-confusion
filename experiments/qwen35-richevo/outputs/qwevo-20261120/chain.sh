#!/usr/bin/env bash
# AgentDojo evolution -> exact ranking -> IPI evolution -> exact ranking -> selection -> panel refs -> ASR (optimized, then remaining references).
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/qwen35-richevo; export TOKENIZERS_PARALLELISM=false; P=/scratch/wz55191/venvs/chatinject-table1/bin/python; E=outputs/qwevo-20261120; O=outputs/optimized-20261123
for t in agentdojo ipi; do
  echo "$t evolution start $(date)"; bash $E/$t/run.sh > $E/$t/stdout.log 2>&1; echo "$t evolution exit $? $(date)"
  $P -u ds_finalize4.py $E/$t 10 > $E/$t/finalize.log 2>&1; echo "$t finalize exit $? $(date)"
  $P -c "from ds_probe_client import call; print(call('drop'))"
done
$P ds_select4.py $E $O > $O/select.log 2>&1 || { echo "selection failed"; cat $O/select.log; exit 1; }
cat $O/select.log
PICKS=$($P -c "import json;print(' '.join(json.load(open('$O/conditions.json'))))")
export DS_EXTRA_CONDITIONS=$O/conditions.json
for t in agentdojo ipi; do $P -u ds_refs4.py $E/$t/dataset.json $E/$t/refs.json $PICKS original_delimiter >> $E/refs.log 2>&1; done
echo "panel refs done $(date)"
tmux -S /scratch/wz55191/tmux-richevo.sock kill-session -t qwserver
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
bash serve_vllm.sh > $O/vllm.log 2>&1 &
until curl -sf -m 5 http://127.0.0.1:8000/v1/models > /dev/null; do sleep 20; done
bash asr2.sh $O > $O/asr2.log 2>&1; echo "optimized asr exit $? $(date)"
bash asr2.sh outputs/baseline-20261120 > outputs/baseline-20261120/asr2_resume.log 2>&1; echo "baseline asr exit $? $(date)"
pkill -KILL -f 'vllm serve /scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4'
echo "chain done $(date)"
