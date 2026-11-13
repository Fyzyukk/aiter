#!/bin/bash
# check.sh exe gpu tile : correctness matrix (original 6 shapes + unaligned shapes) x 4 envs; prints failures and a summary
E=$1; G=${2:-2}; TL=$3; n=0; ok=0
shapes=("-m 8192 -n 8192 -k 8192 -b 1" "-m 1024 -n 2048 -k 1024 -b 2" "-m 2048 -n 1024 -k 7168 -b 3" "-m 768 -n 1280 -k 1536 -b 1" "-m 256 -n 256 -k 512 -b 1" "-m 4096 -n 4096 -k 512 -b 2")
ks=(256 768 7168); i=0
for m in 1 16 100 300 1000; do for nn in 128 400 1280 4112; do k=${ks[$((i%3))]}; b=$((1 + i%2)); shapes+=("-m $m -n $nn -k $k -b $b"); i=$((i+1)); done; done
for a in "${shapes[@]}"; do for env in "" "MXFP4_SFA_K_PATTERN=1" "PERSIST_GRID=37" "PERSIST_GRID=1"; do
  r=$(flock /tmp/gpu$G.lock env $env HIP_VISIBLE_DEVICES=$G timeout 600 $E -t $TL $a -v 1 -w 0 -i 0 2>&1 | grep -oE 'errors=[0-9]+/|FAIL|VALID|need .*|HIP .*|Memory access fault.*' | paste -sd' ')
  n=$((n+1)); if echo "$r" | grep -q "VALID" && ! echo "$r" | grep -q FAIL; then ok=$((ok+1)); else echo "FAIL $TL $a $env: $r"; fi
done; done
echo "tile $TL: $ok/$n ok"
