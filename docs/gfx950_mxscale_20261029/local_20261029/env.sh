#!/usr/bin/env bash
# Source from bash; isolate this run to the requested repository and GPUs 0–3.
export OPUS_RUN_REPO=/root/workspace/aiter-fyz-10-29-aiter
export OPUS_RUN_DIR="$OPUS_RUN_REPO/docs/gfx950_mxscale_20261029/local_20261029"
export OPUS_RUN_PYTHON=/root/workspace/.venvs/aiter-fyz-20261029/bin/python
export PYTHONPATH="$OPUS_RUN_REPO"
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCR_VISIBLE_DEVICES=0,1,2,3
unset CUDA_VISIBLE_DEVICES AITER_USE_SYSTEM_TRITON AITER_AOT_IMPORT
unset OPUS_GEN_CO_DIR
export OPUS_HIP_CLANG_PATH=/root/workspace/aiter-fyz-opus-blockscale-mxfp8-gemm/reports/opus_retune_20261020_183111/pin_agpr_llvm/bin
export HIP_CLANG_PATH="$OPUS_HIP_CLANG_PATH"
unset FLYDSL_RUNTIME_RUN_ONLY COMPILE_ONLY MAX_JOBS
export GPU_ARCHS=gfx950
export ENABLE_CK=0
export CK_DIR=/root/workspace/aiter/3rdparty/composable_kernel
export AITER_MAX_JOBS=32
export FLYDSL_RUNTIME_CACHE_DIR="$OPUS_RUN_DIR/.cache/flydsl"
export TRITON_CACHE_DIR="$OPUS_RUN_DIR/.cache/triton"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -c 0
cd "$OPUS_RUN_REPO" || return 1
