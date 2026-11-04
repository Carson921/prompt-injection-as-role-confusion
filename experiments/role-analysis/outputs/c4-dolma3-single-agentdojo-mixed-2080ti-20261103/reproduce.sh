#!/usr/bin/env bash
set -euo pipefail
task_output="${1:?Provide a NEW output directory}"
export CUDA_VISIBLE_DEVICES=GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mixed-role-probe-matplotlib
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_mixed_single_agentdojo_probe.py --output "$task_output" --stage prepare
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_mixed_single_agentdojo_probe.py --output "$task_output" --stage train
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_mixed_single_agentdojo_probe.py --output "$task_output" --stage report
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/validate_mixed_single_agentdojo_probe.py --output "$task_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/code/audit_exact_feature_overlap.py --output "$task_output"
