#!/usr/bin/env bash
# OpenAI-compatible vLLM service for DeepSeek-V4-Flash-0731 on four H100s (TP=4).
# Requests are sent to /v1/completions as token ids rendered by the official encoding_dsv4.
set -euo pipefail
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
export TRITON_CACHE_DIR=/tmp/wz55191-deepseek-vllm-triton
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# DeepGEMM JIT: the venv's nvcc (13.4) and CUDA runtime headers (13.0) differ; the wrapper disables CCCL's check.
export DG_JIT_NVCC_COMPILER="$(dirname "$(readlink -f "$0")")/nvcc_wrapper.sh"
export DG_JIT_CACHE_DIR=/tmp/wz55191-deepseek-vllm-dgjit
# FlashInfer JIT needs the same override, plus an unversioned libcudart.so for its link step.
SHIM=/tmp/wz55191-cuda-linkshim
mkdir -p "$SHIM" && ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$SHIM/libcudart.so"
export FLASHINFER_NVCC="$DG_JIT_NVCC_COMPILER" FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export FLASHINFER_EXTRA_LDFLAGS="-L$SHIM" VLLM_USE_FLASHINFER_SAMPLER=0
# Read-only helper follows the real TP worker mappings and stops at /v1/models
# readiness. It owns no CUDA context and cannot change the serving exit code.
DS_SERVE_DIR="$(dirname "$(readlink -f "$0")")"
if [[ "${DS_WINDOW_PREFETCH:-1}" != 0 ]]; then
  DS_PREFETCH_ROOT="${DS_WINDOW_PREFETCH_ROOT:-$DS_SERVE_DIR/outputs/optimized-20261118/vllm_prefetch}"
  DS_PREFETCH_RUN="$DS_PREFETCH_ROOT/$$"
  if mkdir -p "$DS_PREFETCH_RUN"; then
    /scratch/wz55191/venvs/chatinject-table1/bin/python -u "$DS_SERVE_DIR/ds_prefetch_window.py" \
      --pid "$$" --kind vllm --output "$DS_PREFETCH_RUN" --status-dir "$DS_PREFETCH_RUN" \
      --wait-identity-seconds 30 --ready-url "http://127.0.0.1:${DS_VLLM_PORT:-8000}/v1/models" \
      --already-read > "$DS_PREFETCH_RUN/prefetch.log" 2>&1 &
  fi
fi
exec /scratch/wz55191/venvs/chatinject-table1/bin/vllm serve /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 --port "${DS_VLLM_PORT:-8000}" \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "${DS_MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${DS_MAX_NUM_SEQS:-32}" \
  --gpu-memory-utilization 0.92 \
  --seed 0 --generation-config vllm \
  --no-enable-prefix-caching ${DS_VLLM_EXTRA:-}
