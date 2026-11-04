#!/usr/bin/env bash
set -euo pipefail
cd /home/wentian/Github/SPAR/prompt-injection-as-role-confusion
profile_probe_output=/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-plus-one-round-2080ti-20261101-rerun
export CUDA_VISIBLE_DEVICES=GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/role-probe-2080ti-matplotlib
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_single_message_probe.py prepare --profiles single one_round --features /home/wentian/Github/SPAR/runpod-backups/role-retraining-20261026-20261027/workspace-task/runs-v7/formal/features/formal --data /home/wentian/Github/SPAR/runpod-backups/role-retraining-20261026-20261027/workspace-task/data-v7/formal --output "$profile_probe_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_single_message_probe.py train --output "$profile_probe_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_single_message_probe.py agentdojo --agentdojo /home/wentian/Github/SPAR/delimeter-token/results/ChatInject/agentdojo_role_probe_gptoss120b/full57_no_payload_first_occurrence_probe_v7_20261027 --parent /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz --output "$profile_probe_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/validate_single_message_probe.py --output "$profile_probe_output"
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/run_single_message_probe.py report --output "$profile_probe_output"
