#!/bin/bash
# timing.sh [hip] [smi] : idle-gated alternating timing of the no-shuffle pipeline vs candidate_m8 (8192^3), then the six tiles
# on medium shapes.  Waits until the card shows CU_OCCUPANCY 0 and use < 5% three times in a row before each round.
G=${1:-2}; S=${2:-2}; E=/tmp/opus_mxfp4_ns/base/build/gemm_a4w4_mxfp4_4wave.exe; M8=${M8_EXE:-/tmp/bench_m/m8/build/gemm_a4w4_mxfp4_4wave.exe}
idle() { local c=0; while [ $c -lt 3 ]; do o=$(amd-smi process -g $S 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1); u=$(rocm-smi -d $S --showuse | grep -oE "use \(%\): [0-9]+" | grep -oE "[0-9]+$"); if [ "${o:-0}" = "0" ] && [ "${u:-100}" -lt 5 ]; then c=$((c+1)); else c=0; fi; sleep 5; done; }
run() { flock /tmp/gpu$G.lock env HIP_VISIBLE_DEVICES=$G OMP_NUM_THREADS=16 "$@" -v 0 -w 1000 -i 200 2>&1 | grep -oE '[0-9.]+ us' | head -1 | cut -d' ' -f1; }
occ() { amd-smi process -g $S 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | paste -sd+; }
echo "== 8192^3: candidate_m8 vs no-shuffle 256x256 (persistent, boundary) vs 256x256 no-boundary"
for r in 1 2 3 4 5 6; do idle; a=$(run $M8 -m 8192 -n 8192 -k 8192 -b 1); sleep 2; b=$(run $E -t 256x256 -p 0 -m 8192 -n 8192 -k 8192 -b 1); sleep 2
  c=$(run $E -t 256x256 -p 0 -o 0 -m 8192 -n 8192 -k 8192 -b 1); echo "r$r m8=$a ns256=$b ns256_nob=$c occ_after=$(occ)"; done
for shp in "1024 4096 8192" "2048 4096 8192" "4096 4096 4096" "1024 7168 2048"; do set -- $shp; echo "== ${1}x${2}x${3}"
  for r in 1 2 3; do idle; line="r$r"; for t in 256x256 256x128 128x256 128x128 192x256 256x192; do line="$line $t=$(run $E -t $t -m $1 -n $2 -k $3 -b 1)"; sleep 1; done
    echo "$line occ_after=$(occ)"; done
done
