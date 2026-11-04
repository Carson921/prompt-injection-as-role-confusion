#!/usr/bin/env bash
set -euo pipefail
task_output="${1:?Provide a NEW output directory}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/tmp/role-asr-trend-20261103/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-latest-probe-trends-20261103/analyze_probe_asr_trends.py --predictions /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-payload-mixed-probe-20261103 --output "$task_output"
