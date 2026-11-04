#!/usr/bin/env bash
set -euo pipefail
cd /home/wentian/Github/SPAR/prompt-injection-as-role-confusion
all_single_output=/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-all-single-role-probe-2080ti-20261101-rerun
export CUDA_VISIBLE_DEVICES=GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/role-probe-2080ti-matplotlib
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_all_single_message_probe.py train --source /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-role-probe-2080ti-20261101 --output "$all_single_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_single_message_probe.py agentdojo --agentdojo /home/wentian/Github/SPAR/delimeter-token/results/ChatInject/agentdojo_role_probe_gptoss120b/full57_no_payload_first_occurrence_probe_v7_20261027 --parent /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz --output "$all_single_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/validate_all_single_message_probe.py --output "$all_single_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_all_single_message_probe.py report --source /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-role-probe-2080ti-20261101 --output "$all_single_output"
