#!/usr/bin/env bash
set -euo pipefail
artifact_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo="$(git -C "$artifact_dir" rev-parse --show-toplevel)"
cd "$repo"
output="${1:-$repo/experiments/role-analysis/outputs/role-signal-factorial-reproduction}"
python="${ROLE_SIGNAL_PYTHON:-$repo/.venv/bin/python}"
model="${ROLE_SIGNAL_MODEL:-/tmp/wz55191-agentdojo-first-tool-model}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
"$python" experiments/role-analysis/run_role_signal_factorial.py prepare --output "$output" --model "$model"
"$python" experiments/role-analysis/run_role_signal_factorial.py run --output "$output" --model "$model"
"$python" experiments/role-analysis/analyze_role_signal_factorial.py --output "$output"
"$python" experiments/role-analysis/analyze_role_signal_family_sensitivity.py --output "$output"
"$python" experiments/role-analysis/polish_role_signal_figures.py --output "$output"
"$python" experiments/role-analysis/export_role_signal_readouts.py --output "$output"
"$python" experiments/role-analysis/validate_role_signal_factorial.py --output "$output"
"$python" experiments/role-analysis/check_role_signal_numerics.py --output "$output"
"$python" experiments/role-analysis/run_role_signal_length_sensitivity.py --output "$output"
"$python" experiments/role-analysis/analyze_role_signal_length_sensitivity.py --output "$output"
"$python" experiments/role-analysis/validate_role_signal_length_sensitivity.py --output "$output"
cp "$artifact_dir/pair-review-independent.json" "$output/pair-review-independent.json"
"$python" experiments/role-analysis/report_role_signal_factorial.py --output "$output"
