#!/usr/bin/env bash
# OpenAI-compatible vLLM service for Qwen3.5-397B-A17B-GPTQ-Int4 on four H100s (TP=4).
# Requests are sent to /v1/completions as token ids rendered by ds_common.Codec (the checkpoint's chat template).
set -euo pipefail
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
export TRITON_CACHE_DIR=/tmp/wz55191-qwen35-vllm-triton
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
# FlashInfer JIT: the venv's nvcc (13.4) and CUDA runtime headers (13.0) differ; same overrides as the DeepSeek service.
D="$(dirname "$(readlink -f "$0")")"
SHIM=/tmp/wz55191-cuda-linkshim
mkdir -p "$SHIM" && ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$SHIM/libcudart.so"
export FLASHINFER_NVCC="$D/../deepseek-richevo/nvcc_wrapper.sh" FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export FLASHINFER_EXTRA_LDFLAGS="-L$SHIM" VLLM_USE_FLASHINFER_SAMPLER=0
exec /scratch/wz55191/venvs/chatinject-table1/bin/vllm serve /scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --served-model-name qwen35-397b \
  --host 127.0.0.1 --port "${QW_VLLM_PORT:-8000}" \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --quantization moe_wna16 --language-model-only \
  --max-model-len "${QW_MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${QW_MAX_NUM_SEQS:-96}" \
  --gpu-memory-utilization 0.93 \
  --seed 0 --generation-config vllm \
  --enable-prefix-caching --mamba-cache-mode align ${QW_VLLM_EXTRA:-}
