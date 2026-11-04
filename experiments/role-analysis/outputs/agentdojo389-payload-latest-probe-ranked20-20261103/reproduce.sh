#!/usr/bin/env bash
set -euo pipefail
task_output="${1:?Provide a NEW output directory}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/latest-probe-ranked20-matplotlib
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-payload-latest-probe-ranked20-20261103/code/plot_latest_probe_ranked_asr.py --predictions /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-payload-mixed-probe-20261103 --output "$task_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-payload-latest-probe-ranked20-20261103/code/validate_latest_probe_ranked_asr.py --output "$task_output"
