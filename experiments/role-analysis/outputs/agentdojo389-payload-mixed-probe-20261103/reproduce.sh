#!/usr/bin/env bash
set -euo pipefail
task_output="${1:?Provide a NEW output directory}"
cd /home/wentian/Github/SPAR/delimeter-token
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/tmp/agentdojo-payload-test-venv/bin/python -m experiments.ChatInject.role_probe_full_payload.rescore --run-root /home/wentian/Github/SPAR/delimeter-token/results/ChatInject/agentdojo_role_probe_gptoss120b/full389_plain_remember_first_occurrence_warmstart_20261027 --probe /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz --expected-probe-sha256 fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33 --output "$task_output"
export CUDA_VISIBLE_DEVICES=GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a
/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/.venv-single-probe2080ti/bin/python /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/report_full389_payload_probe.py --probe /home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz --output "$task_output"
