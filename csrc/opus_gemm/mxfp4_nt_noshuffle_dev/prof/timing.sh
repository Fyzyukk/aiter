#!/bin/bash
# timing.sh [hip] [smi] : idle-gated timing of the no-shuffle pipeline (exe built by build_var.sh base).
#  1. 8192^3 alternating: candidate_m8 vs 256x256 persistent (boundary) vs 256x256 no-boundary (-o 0)
#  2. six tiles x {persistent, one tile per block} on M in {128..4096} x (N, K) in four pairs
# Waits until CU_OCCUPANCY is empty and use < 5% three times in a row before each round; prints the occupancy after it.
G=${1:-4}; S=${2:-7}; E=/tmp/opus_mxfp4_ns/base/build/gemm_a4w4_mxfp4_4wave.exe; M8=${M8_EXE:-/tmp/bench_m/m8/build/gemm_a4w4_mxfp4_4wave.exe}
idle() { local c=0; while [ $c -lt 3 ]; do o=$(amd-smi process -g $S 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1); u=$(rocm-smi -d $S --showuse | grep -oE "use \(%\): [0-9]+" | grep -oE "[0-9]+$"); if [ "${o:-0}" = "0" ] && [ "${u:-100}" -lt 5 ]; then c=$((c+1)); else c=0; fi; sleep 3; done; }
run() { flock /tmp/gpu$G.lock env HIP_VISIBLE_DEVICES=$G OMP_NUM_THREADS=16 "$@" -v 0 -w ${W:-1000} -i ${I:-200} 2>&1 | grep -oE '[0-9.]+ us' | head -1 | cut -d' ' -f1; }
occ() { amd-smi process -g $S 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | paste -sd+; }
if [ -z "$SKIP1" ]; then
echo "== 8192^3: candidate_m8 vs ns 256x256 persistent (boundary) vs ns 256x256 no-boundary"
for r in 1 2 3 4 5 6; do idle; a=$(run $M8 -m 8192 -n 8192 -k 8192 -b 1); sleep 2; b=$(run $E -t 256x256 -p 0 -m 8192 -n 8192 -k 8192 -b 1); sleep 2
  c=$(run $E -t 256x256 -p 0 -o 0 -m 8192 -n 8192 -k 8192 -b 1); echo "r$r m8=$a ns256=$b ns256_nob=$c occ_after=$(occ)"; done
fi
echo "== shape sweep (us, median of 2): M N K | p0: 256x256 256x128 128x256 128x128 192x256 256x192 | p1: same order"
W=200; I=100
for nk in "4096 4096" "2048 7168" "3072 6144" "4096 8192"; do set -- $nk; N=$1; K=$2
 for M in 128 256 512 1024 2048 4096; do idle; line="$M $N $K |"
  for p in 0 1; do for t in 256x256 256x128 128x256 128x128 192x256 256x192; do
    x=$(run $E -t $t -p $p -m $M -n $N -k $K -b 1); y=$(run $E -t $t -p $p -m $M -n $N -k $K -b 1)
    line="$line $(python3 -c "print(f'{min($x,$y):.1f}')")"; done; line="$line |"; done
  echo "$line occ_after=$(occ)"
 done
done
