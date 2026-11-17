#!/usr/bin/env bash
set -euo pipefail

task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo"
task_out=${TASK_OUTPUT:-"$task_repo/experiments/autodan-role-probe/outputs/cot-near-delimiter-refined-20261104"}
task_model=${TASK_MODEL:-/tmp/wz55191-agentdojo-first-tool-model}
task_probe="$task_repo/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz"
task_score_python="$task_repo/.venv/bin/python"
task_eval_python=/scratch/wz55191/venvs/chatinject-table1/bin/python
task_plot_python=${TASK_PLOT_PYTHON:-/home/wz55191/miniconda3/envs/AutoDAN/bin/python}
task_code="$task_repo/experiments/autodan-role-probe"
task_dataset=${TASK_DATASET:-"$task_code/outputs/near-delimiter-20261104/dataset.json"}
task_previous=${TASK_PREVIOUS_WINNER:-"$task_code/outputs/near-delimiter-20261104/winner.json"}
task_near_rate=${TASK_NEAR_RATE:-0.5}
task_target_role=${TASK_TARGET_ROLE:-cot}
task_first_cot=${TASK_FIRST_COT_WINNER:-"$task_code/outputs/cot-near-delimiter-20261104/winner.json"}
task_latent_seed=${TASK_LATENT_COT_SEED:-"$task_code/outputs/cot-near-delimiter-20261104/latent_best_cot_seed.json"}
mkdir -p "$task_out"

# Two H100s; native BF16/MXFP4 pre-o_proj features during optimization.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20261104
export TRITON_CACHE_DIR=/tmp/wz55191-agentdojo-first-tool-triton
# Reuse the exact six-task panel, without sampling again.
if [[ "$task_dataset" != "$task_out/dataset.json" ]]; then
  if [[ -f "$task_out/dataset.json" ]]; then
    cmp "$task_dataset" "$task_out/dataset.json"
  else
    cp "$task_dataset" "$task_out/dataset.json"
  fi
fi
# Freeze the refinement accounting separately from the fixed previous_optimized
# control. Reading the historical seed performs no backbone or API operation.
"$task_score_python" - "$task_out" "$task_first_cot" "$task_latent_seed" "$task_previous" "$task_target_role" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output, first, latent, previous = (Path(value).resolve() for value in sys.argv[1:5])
role = sys.argv[5]
path = output / 'refinement_protocol.json'
sha = lambda source: hashlib.sha256(source.read_bytes()).hexdigest()
if path.exists():
    data = json.loads(path.read_text())
    for key, source in [('first_cot_winner', first), ('historical_seed', latent)]:
        if Path(data[key + '_path']).resolve() != source or data[key + '_sha256'] != sha(source):
            raise ValueError('Existing refinement seed binding differs: ' + key)
    if data['target_role'] != role or data.get('asr_not_consulted_for_cot_selection') is not True:
        raise ValueError('Existing refinement objective/selection policy differs')
else:
    first_record, latent_record = (json.loads(source.read_text()) for source in (first, latent))
    data = dict(schema='cot-near-refinement-protocol-v1', target_role=role,
                target_role_index=('system', 'developer', 'user', 'cot', 'assistant', 'tool').index(role),
                reason='Refine using the first cot winner and a latent cot seed selected only from saved historical probabilities.',
                first_cot_output=str(first.parent), first_cot_winner_path=str(first),
                first_cot_winner_sha256=sha(first), first_cot_best_fitness=first_record['fitness'],
                historical_seed_path=str(latent), historical_seed_sha256=sha(latent),
                historical_seed_cot_fitness=latent_record['fitness'],
                historical_candidates_reused_for_seed_selection_only=42,
                new_trial_native_candidate_evaluations=[42, 42],
                cross_trial_templates_may_repeat=True, native_context_forwards_planned_total=504,
                previous_optimized_control='previous user-target near winner; separate from extra initial seeds',
                previous_optimized_control_path=str(previous), previous_optimized_control_sha256=sha(previous),
                asr_not_consulted_for_cot_selection=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
PY
"$task_eval_python" "$task_code/audit_near.py" --output "$task_out" \
  --model "$task_model" --target-role "$task_target_role" --mode preflight --report "$task_out/near_context_preflight.json"
task_search_options=(--target-role "$task_target_role" --near-delimiter-rate "$task_near_rate" \
  --warm-start-template "$task_previous" --initial-template "$task_first_cot" --initial-template "$task_latent_seed")
if [[ -n "${TASK_NEAR_LIBRARY:-}" ]]; then
  task_search_options+=(--near-delimiter-library "$TASK_NEAR_LIBRARY")
fi
"$task_score_python" -u "$task_code/search.py" --dataset "$task_out/dataset.json" \
  --output "$task_out" --model "$task_model" --probe "$task_probe" \
  --population 12 --generations 4 --seed 20261104 "${task_search_options[@]}" > "$task_out/search.log" 2>&1
"$task_score_python" "$task_code/audit.py" --output "$task_out" \
  --dataset "$task_out/dataset.json" --probe "$task_probe"
"$task_eval_python" "$task_code/audit_near.py" --output "$task_out" \
  --model "$task_model" --target-role "$task_target_role" --mode search --report "$task_out/near_search_audit.json"

# The native scorer has exited before the vLLM engine is started.
export OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/tmp/wz55191-autodan-vllm-cache
"$task_eval_python" -u "$task_code/evaluate.py" --dataset "$task_out/dataset.json" \
  --winner "$task_out/winner.json" --output "$task_out/eval_full" --scope full \
  --previous-winner "$task_previous" --near-delimiter-library "$task_out/near_delimiter_library.json" \
  --model "$task_model" --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --workers 32 --max-num-seqs 32 --enforce-eager > "$task_out/eval_full.log" 2>&1
"$task_eval_python" "$task_code/audit_asr.py" --run-root "$task_out/eval_full" --dataset "$task_out/dataset.json"
"$task_score_python" "$task_code/report.py" --output "$task_out" --full-eval "$task_out/eval_full" \
  --previous-winner "$task_previous" --plot-python "$task_plot_python"
