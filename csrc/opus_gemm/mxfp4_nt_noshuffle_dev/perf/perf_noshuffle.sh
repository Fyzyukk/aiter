#!/bin/bash
# perf_noshuffle.sh : portable perf run for the no-shuffle MXFP4 path (and, optionally, the dev-harness comparison with
# candidate_m8).  Everything is parameterized; run it on an otherwise idle machine.
#   AITER=<aiter checkout with this branch>  (default: this file's checkout)
#   GPU=<HIP_VISIBLE_DEVICES index>          (default 0)
#   SMI=<rocm-smi / amd-smi index of the same card, for the idle check; empty = no check>   REBUILD=1: drop a stale JIT .so
#   M8_EXE=<candidate_m8 exe>  DEV_EXE=<mxfp4_nt_noshuffle_dev exe>   (optional: 8192^3 dev-harness alternation)
#   SHAPES="M,N,K ..."                       (default: perf_noshuffle.py's list)   KIDS=0 skips the per-kid sweep
#   PY_ENV="AITER_USE_SYSTEM_TRITON=1 ..."   (extra env for python; default AITER_USE_SYSTEM_TRITON=1)
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
if [ -n "$M8_EXE" ] && [ -n "$DEV_EXE" ]; then
  run() { env HIP_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16 "$@" -v 0 -w 1000 -i 200 2>&1 | grep -oE '[0-9.]+ us' | head -1 | cut -d' ' -f1; }
  echo "== 8192^3 dev harness: candidate_m8 | noshuffle 256x256 persistent | 256x256 no-boundary (us)"
  for r in 1 2 3 4 5 6; do idle; echo "r$r m8=$(run $M8_EXE -m 8192 -n 8192 -k 8192 -b 1) ns256=$(run $DEV_EXE -t 256x256 -p 0 -m 8192 -n 8192 -k 8192 -b 1) ns256_nob=$(run $DEV_EXE -t 256x256 -p 0 -o 0 -m 8192 -n 8192 -k 8192 -b 1)"; done
fi
echo "== CUDA graph, us: triton gemm_afp4wfp4 | gemm_a4w4_opus_noshuffle | ours/triton | noshuffle kids (KIDS=0: skip)"
if [ -n "$SHAPES" ]; then LIST="$SHAPES"; else LIST=$(python3 - <<'P'
print(" ".join([f"{m},{n},{k}" for (n, k) in ((4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192)) for m in (1, 16, 64, 128, 256, 512, 1024, 2048, 4096)] + ["8192,8192,8192"]))
P
); fi
for s in $LIST; do idle
  env ${PY_ENV:-AITER_USE_SYSTEM_TRITON=1} PYTHONPATH=$AITER GPU_ARCHS=gfx950 HIP_VISIBLE_DEVICES=$GPU python3 $H/perf_noshuffle.py $s 2>&1 | grep -E "^[0-9]+ |^#|Error" ; done
