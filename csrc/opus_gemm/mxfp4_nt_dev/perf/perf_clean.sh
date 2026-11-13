#!/bin/bash
# per shape: require no CU_OCCUPANCY on smi GPU6 before and after; retry up to 3 times
W=/root/workspace/aiter-fyz-opus-mxfp4-nt-gemm
occ() { amd-smi process -g 6 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | awk '{s+=$1} END{print s+0}'; }
for m in 128 256 512 1024 2048; do for nk in 4096,4096 2048,7168 3072,6144 4096,8192; do
  for try in 1 2 3; do
    while [ "$(occ)" != "0" ]; do sleep 5; done
    out=$(flock /tmp/gpu6.lock env AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=$W GPU_ARCHS=gfx950 HIP_VISIBLE_DEVICES=6 timeout 900 python3 $W/csrc/opus_gemm/mxfp4_nt_dev/perf/perf_vs_asm.py $m,$nk 2>&1 | grep -E "^[0-9]+ ")
    if [ "$(occ)" = "0" ] && [ -n "$out" ]; then echo "$out"; break; fi
    echo "# interfered: $m,$nk try $try" >&2
  done
done; done
