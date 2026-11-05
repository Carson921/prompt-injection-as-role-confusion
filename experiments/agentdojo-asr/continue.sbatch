#!/bin/bash
#SBATCH --job-name=gptoss120b-asr
#SBATCH --partition=gpu_p
#SBATCH --account=lglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:2
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103_asr/slurm-%j.log

set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
TASK_OUTPUT=/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103_asr
TASK_PYTHON=/scratch/wz55191/Github/prompt-injection-as-role-confusion/.venv/bin/python
if [ -f "$TASK_OUTPUT/../full_20261103_with_asr.tar.gz.complete.json" ]; then
    exit 0
fi
if "$TASK_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; sys.exit(0 if d.get("completed")==d.get("requested")==1898 else 1)' "$TASK_OUTPUT/producer-terminal.json"; then
    "$TASK_PYTHON" -u "$TASK_OUTPUT/postprocessing/finish.py" >> "$TASK_OUTPUT/finish.log" 2>&1
    exit 0
fi
# A different node needs the exact bound checkpoint staged and byte-verified.
if [ ! -f /tmp/wz55191-agentdojo-first-tool-model/staging-proof.json ]; then
    "$TASK_PYTHON" /home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103/provenance/stage_model.py \
        --selection /home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103/provenance/head-selection-16.json
fi
"$TASK_PYTHON" -u "$TASK_OUTPUT/provenance/run.py" >> "$TASK_OUTPUT/run.log" 2>&1
"$TASK_PYTHON" -u "$TASK_OUTPUT/postprocessing/finish.py" >> "$TASK_OUTPUT/finish.log" 2>&1
