#!/bin/bash
# perf_gemm_a4w4.sh : portable perf run of aiter.gemm_a4w4(bpreshuffle=True) -- asm default vs opus backend
# (AITER_A4W4_BACKEND=opus) vs Triton gemm_afp4wfp4.  Run on an otherwise idle machine.
#   AITER=<aiter checkout> (default: this file's checkout)   GPU=<HIP_VISIBLE_DEVICES index> (default 0)
#   SMI=<rocm-smi / amd-smi index of that card for an idle check before each shape; empty = no check>
#   REBUILD=1 drops a stale aiter/jit/module_deepgemm_opus.so   SHAPES="M,N,K ..." (default: perf_gemm_a4w4.py's list)   PY_ENV="..." (default AITER_USE_SYSTEM_TRITON=1)
H=$(cd $(dirname $0) && pwd); AITER=${AITER:-$(cd $H/../../../.. && pwd)}; GPU=${GPU:-0}; SMI=${SMI:-}
idle() { [ -z "$SMI" ] && return; local c=0; while [ $c -lt 3 ]; do
  o=$(amd-smi process -g $SMI 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | awk '{s+=$1} END{print s+0}')
  u=$(rocm-smi -d $SMI --showuse 2>/dev/null | grep -oE "use \(%\): [0-9]+" | grep -oE "[0-9]+$")
  if [ "${o:-0}" = "0" ] && [ "${u:-100}" -lt 5 ]; then c=$((c+1)); else c=0; fi; sleep 3; done; }
# Stale JIT module check: aiter does not rebuild module_deepgemm_opus.so when the opus sources change, so an old .so can
# lack entries such as opus_gemm_a4w4.  REBUILD=1 deletes it (the next import rebuilds it).
SO=$AITER/aiter/jit/module_deepgemm_opus.so
if [ -f "$SO" ]; then
  newer=$(find $AITER/csrc/opus_gemm/opus_gemm.cu $AITER/csrc/opus_gemm/gen_instances.py $AITER/csrc/opus_gemm/opus_gemm_common.py \
          $AITER/csrc/opus_gemm/codegen $AITER/csrc/opus_gemm/include $AITER/csrc/include/rocm_ops.hpp $AITER/csrc/pybind/opus_gemm_pybind.cu \
          -newer "$SO" -type f 2>/dev/null | head -3)
  if [ -n "$newer" ] || ! grep -qa "opus_gemm_a4w4" "$SO"; then
    if [ "${REBUILD:-0}" = 1 ]; then echo "# removing stale $SO (rebuilds on first use)"; rm -f "$SO"
    else echo "# WARNING: $SO is older than the opus sources (${newer:-missing opus_gemm_a4w4}); rerun with REBUILD=1 or rm it"; exit 1; fi
  fi
fi
echo "# $(date -u +%FT%TZ) host=$(hostname) GPU=$GPU SMI=$SMI aiter=$AITER $(git -C $AITER rev-parse --short HEAD 2>/dev/null)"
run() { env ${PY_ENV:-AITER_USE_SYSTEM_TRITON=1} PYTHONPATH=$AITER GPU_ARCHS=gfx950 HIP_VISIBLE_DEVICES=$GPU python3 $H/perf_gemm_a4w4.py "$@" 2>&1 | grep -E "^[0-9#]|Error"; }
if [ -z "$SHAPES" ]; then idle; run; else for s in $SHAPES; do idle; run $s; done; fi
