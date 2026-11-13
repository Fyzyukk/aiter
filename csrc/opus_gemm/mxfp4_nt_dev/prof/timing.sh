#!/bin/bash
# waits for HIP 2 (smi 2) idle before each measurement; alternating rounds
G=2; S=2; E=/tmp/exp_tiles/t2/build/gemm_a4w4_mxfp4_4wave.exe; Z=/tmp/exp_m10/zc/build/gemm_a4w4_mxfp4_4wave.exe
idle() { local c=0; while [ $c -lt 3 ]; do o=$(amd-smi process -g $S 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1); u=$(rocm-smi -d $S --showuse | grep -oE "use \(%\): [0-9]+" | grep -oE "[0-9]+$"); if [ "${o:-0}" = "0" ] && [ "${u:-100}" -lt 5 ]; then c=$((c+1)); else c=0; fi; sleep 5; done; }
run() { flock /tmp/gpu$G.lock env HIP_VISIBLE_DEVICES=$G OMP_NUM_THREADS=16 "$@" -v 0 -w 1000 -i 200 2>&1 | grep -oE '[0-9.]+ us' | head -1 | cut -d' ' -f1; }
occ() { amd-smi process -g $S 2>/dev/null | grep -oE "CU_OCCUPANCY: [0-9]+" | grep -oE "[0-9]+$" | paste -sd+; }
echo "== 8192^3: zc vs 256x256"
for r in 1 2 3 4 5 6; do idle; a=$(run $Z -m 8192 -n 8192 -k 8192 -b 1); sleep 2; b=$(run $E -t 256x256 -m 8192 -n 8192 -k 8192 -b 1); echo "r$r zc=$a t256=$b occ_after=$(occ)"; done
for shp in "1024 4096 8192" "2048 4096 8192"; do set -- $shp; echo "== ${1}x${2}x${3}"
  for r in 1 2 3; do idle; line="r$r"; for t in 256x256 256x128 128x256 128x128; do line="$line $t=$(run $E -t $t -m $1 -n $2 -k $3 -b 1)"; sleep 1; done
    a=$(flock /tmp/gpu$G.lock env HIP_VISIBLE_DEVICES=$G timeout 600 python3 /tmp/exp_tiles/bench_asm_mnk.py $1 $2 $3 2>&1 | grep -oE "asm gemm_a4w4 bpreshuf +[0-9.]+" | grep -oE "[0-9.]+$"); echo "$line asm=$a occ_after=$(occ)"; done
done
