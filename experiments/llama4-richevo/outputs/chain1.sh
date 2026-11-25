#!/usr/bin/env bash
# download -> role probe -> vLLM smoke test -> baseline / reference ASR.  Log: outputs/chain1.log
D=/scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/llama4-richevo
M=/scratch/wz55191/Hugginface/Llama-4-Scout-17B-16E-Instruct
N=/scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/llama4-scout-17b-16e-role-probe-20261125
PY=/scratch/wz55191/venvs/chatinject-table1/bin/python
cd "$D"
until grep -q "^exit" $M.download.log; do sleep 20; done
echo "download $(tail -1 $M.download.log) $(date)"
grep -q "^exit 0" $M.download.log || { echo "download failed"; exit 1; }
$PY - <<'PYEOF' || { echo "shard check failed"; exit 1; }
import json, os
M='/scratch/wz55191/Hugginface/Llama-4-Scout-17B-16E-Instruct'
from huggingface_hub import HfApi
info=HfApi().model_info('meta-llama/Llama-4-Scout-17B-16E-Instruct', files_metadata=True)
bad=[s.rfilename for s in info.siblings if not os.path.exists(f'{M}/{s.rfilename}') or os.path.getsize(f'{M}/{s.rfilename}')!=s.size]
print('files', len(info.siblings), 'mismatched', bad)
json.dump({s.rfilename: dict(size=s.size, sha256=(s.lfs or {}).get('sha256') if isinstance(s.lfs, dict) else getattr(s.lfs,'sha256',None)) for s in info.siblings}, open(M+'.files.json','w'), indent=1)
assert not bad
PYEOF
echo "probe start $(date)"
bash $N/run.sh all > /dev/null 2>&1
tail -1 $N/run.status
tail -1 $N/run.status | grep -q "^exit 0" || { echo "probe failed"; tail -5 $N/run.log | cut -c1-600; exit 1; }
echo "probe done $(date)"
bash serve_vllm.sh > outputs/smoke-20261125/vllm.log 2>&1 &
SERVER=$!
echo "vllm start $(date) pid $SERVER"
until curl -sf -m 5 http://127.0.0.1:8000/v1/models > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vllm died"; tail -30 outputs/smoke-20261125/vllm.log | cut -c1-400; exit 3; }
  sleep 20
done
echo "vllm ready $(date)"
$PY -u ds_agentdojo.py --output outputs/smoke-20261125/agentdojo --conditions-json outputs/baseline-20261125/conditions.json --conditions original,delim_close_user --limit-per-suite 3 --workers 18 > outputs/smoke-20261125/agentdojo_run.log 2>&1
echo "smoke exit $? $(date)"; tail -1 outputs/smoke-20261125/agentdojo_run.log
bash asr2.sh outputs/baseline-20261125 > outputs/baseline-20261125/asr2.log 2>&1
echo "baseline asr exit $? $(date)"
pkill -TERM -P $SERVER 2>/dev/null; kill -TERM $SERVER 2>/dev/null; sleep 30
for p in $(pgrep -f 'bin/[v]llm serve /scratch/wz55191/Hugginface/Llama-4-Scout'); do kill -KILL $p; done
echo "chain1 done $(date)"
