#!/usr/bin/env bash
set -euo pipefail
task_output="${1:?Provide a NEW output directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/three-method-asr-matplotlib
/tmp/role-asr-trend-20261103/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-three-methods-20261103/run_three_method_asr_analysis.py --predictions /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-payload-mixed-probe-20261103 --previous /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-latest-probe-trends-20261103 --output "$task_output"
/tmp/role-asr-trend-20261103/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/agentdojo389-three-methods-20261103/report_three_method_asr_analysis.py --output "$task_output"
