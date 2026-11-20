#!/usr/bin/env bash
# The venv ships nvcc 13.4 with CUDA 13.0 runtime headers; CCCL refuses that pair unless told otherwise.
exec /scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK "$@"
