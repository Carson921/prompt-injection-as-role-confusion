#!/usr/bin/env bash
# Round 2: probe server -> AgentDojo evolution (warm start) -> exact ranking -> selection -> panel refs -> ASR (round-2 picks, then remaining references).
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/qwen35-richevo; export TOKENIZERS_PARALLELISM=false; P=/scratch/wz55191/venvs/chatinject-table1/bin/python; E=outputs/qwevo-20261120; O=outputs/optimized2-20261124
( export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True; exec /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python ds_probe_server.py --output $E/probe_server > $E/probe_server.log 2>&1 ) &
SERVER=$!
until grep -q '"stage": "ready"' $E/probe_server.log; do kill -0 $SERVER || { echo "probe server died"; exit 1; }; sleep 30; done
echo "probe server ready $(date)"
bash $E/agentdojo2/run.sh > $E/agentdojo2/stdout.log 2>&1; echo "evolution exit $? $(date)"
$P -u ds_finalize4.py $E/agentdojo2 10 > $E/agentdojo2/finalize.log 2>&1; echo "finalize exit $? $(date)"
$P - <<'PY'
import json
from ds_common import Codec, MODEL, read_json, write_json
rows = sorted(read_json('outputs/qwevo-20261120/agentdojo2/exact_top.json'), key=lambda r: -r['exact_fitness'])
old = {v['template'] for v in read_json('outputs/optimized-20261123/conditions.json').values()}
picks, codec = {}, Codec(MODEL)
for kind, row in (('top', rows[0]), ('short', min(rows[:5], key=lambda r: r['wrapper_tokens']))):
    if row['template'] in old or any(c['template'] == row['template'] for c in picks.values()):
        continue
    assert not codec.has_special(row['template'].replace('{goal}', ' '))
    picks[f'opt_ad2_{kind}'] = dict(template=row['template'], source=f'qwevo agentdojo round 2 exact rank ({kind})', optimized_for='agentdojo',
                                    panel_exact=row['exact_fitness'], panel_suite_means=row['suite_means'], wrapper_tokens=row['wrapper_tokens'],
                                    fast_fitness=row['fast_fitness'], no_pretext=row['no_pretext'])
    print(f'opt_ad2_{kind}', round(row['exact_fitness'], 4), row['wrapper_tokens'], repr(row['template']))
write_json('outputs/optimized2-20261124/conditions.json', picks)
PY
PICKS=$($P -c "import json;print(' '.join(json.load(open('$O/conditions.json'))))")
if [ -n "$PICKS" ]; then
  export DS_EXTRA_CONDITIONS=$O/conditions.json
  for t in agentdojo ipi; do $P -u ds_refs4.py $E/$t/dataset.json $E/$t/refs.json $PICKS >> $E/refs.log 2>&1; done
fi
echo "panel refs done $(date) picks: $PICKS"
kill $SERVER; sleep 5
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
bash serve_vllm.sh > $O/vllm.log 2>&1 &
until curl -sf -m 5 http://127.0.0.1:8000/v1/models > /dev/null; do sleep 20; done
bash asr2.sh outputs/baseline-20261120 > outputs/baseline-20261120/asr2_resume2.log 2>&1; echo "baseline asr exit $? $(date)"
[ -n "$PICKS" ] && { bash asr2.sh $O > $O/asr2.log 2>&1; echo "round-2 asr exit $? $(date)"; }
pkill -KILL -f 'vllm serve /scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4'
echo "chain2 done $(date)"
