#!/usr/bin/env bash
# OpenAI-compatible vLLM service for gpt-oss-120b on the two H100s (TP=2).
set -euo pipefail
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13
export TRITON_CACHE_DIR=/tmp/wz55191-ipi-vllm-triton
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec /scratch/wz55191/venvs/chatinject-table1/bin/vllm serve /tmp/wz55191-agentdojo-first-tool-model \
  --served-model-name gpt-oss-120b \
  --host 127.0.0.1 --port "${IPI_VLLM_PORT:-8000}" \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --dtype bfloat16 --quantization mxfp4 \
  --max-model-len "${IPI_MAX_MODEL_LEN:-131072}" \
  --max-num-seqs "${IPI_MAX_NUM_SEQS:-32}" \
  --gpu-memory-utilization 0.90 \
  --enforce-eager --seed 0 \
  --generation-config vllm \
  --tool-call-parser openai --enable-auto-tool-choice \
  --reasoning-parser openai_gptoss \
  --no-enable-prefix-caching
