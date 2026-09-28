#!/usr/bin/env bash
set -uo pipefail
cd /root/workspace/aiter-opus-mxfp8-bpreshuffle
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OPUS_HIP_CLANG_PATH=/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin AITER_JIT_DIR=/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/jit AITER_AOT_IMPORT=1 GPU_ARCHS=gfx950 CU_NUM=256 OMP_NUM_THREADS=2 MAX_JOBS=16 AITER_REBUILD=0
date -u +%FT%TZ > /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/started_utc.txt
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune -i reports/opus_mxfp8_renumber_20260928/added_shapes.csv -o /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/tuned.csv -o2 /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/profile.csv --opus-kids 9030 --libtype opus --shape_grouped --mp 8 --warmup 5 --iters 51 --all > /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/tune.log 2>&1
run_rc=$?
printf "%s\n" "$run_rc" > /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/exit_code.txt
date -u +%FT%TZ > /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_large_output_20260928/ended_utc.txt
exit "$run_rc"
