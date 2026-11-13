#!/bin/bash
# perf_gemm_a4w4.sh : portable perf run of aiter.gemm_a4w4(bpreshuffle=True) -- asm default vs opus backend
# (AITER_A4W4_BACKEND=opus) vs Triton gemm_afp4wfp4.  Run on an otherwise idle machine.
#   AITER=<aiter checkout> (default: this file's checkout)   GPU=<HIP_VISIBLE_DEVICES index> (default 0)
#   SMI=<rocm-smi / amd-smi index of that card for an idle check before each shape; empty = no check>
#   SHAPES="M,N,K ..." (default: perf_gemm_a4w4.py's list)   PY_ENV="..." (default AITER_USE_SYSTEM_TRITON=1)
H=$(cd $(dirname $0) && pwd); AITER=${AITER:-$(cd $H/../../../.. && pwd)}; GPU=${GPU:-0}; SMI=${SMI:-}
idle() { [ -z "$SMI" ] && return; local c=0; while [ $c -lt 3 ]; do
  o=$(amd-smi process -g $SMI 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | awk '{s+=$1} END{print s+0}')
  u=$(rocm-smi -d $SMI --showuse 2>/dev/null | grep -oE "use \(%\): [0-9]+" | grep -oE "[0-9]+$")
  if [ "${o:-0}" = "0" ] && [ "${u:-100}" -lt 5 ]; then c=$((c+1)); else c=0; fi; sleep 3; done; }
echo "# $(date -u +%FT%TZ) host=$(hostname) GPU=$GPU SMI=$SMI aiter=$AITER $(git -C $AITER rev-parse --short HEAD 2>/dev/null)"
run() { env ${PY_ENV:-AITER_USE_SYSTEM_TRITON=1} PYTHONPATH=$AITER GPU_ARCHS=gfx950 HIP_VISIBLE_DEVICES=$GPU python3 $H/perf_gemm_a4w4.py "$@" 2>&1 | grep -E "^[0-9#]|Error"; }
if [ -z "$SHAPES" ]; then idle; run; else for s in $SHAPES; do idle; run $s; done; fi
