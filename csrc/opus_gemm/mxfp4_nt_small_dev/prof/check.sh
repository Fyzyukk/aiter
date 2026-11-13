#!/bin/bash
# check.sh EXE GPU cfg... : full correctness matrix of the small-tile pipeline.
#   cfg = MODE:BMxBN:KS:D[:LAYOUT]   (MODE 0 = B_DIRECT, 1 = B_LDS (row-major only), 2 = A_LDS (shuffle only); KS ignored unless B_DIRECT; LAYOUT 0 shuffled (default), 1 row-major)
#   GPU = HIP_VISIBLE_DEVICES index; runs are serialized with flock /tmp/gpu<GPU>.lock (correctness is unaffected by other users).
# Matrix per cfg: (N,K) in {(4096,4096),(2048,7168),(3072,6144),(4096,8192),(4096,1024),(3072,1536)} x M {1,64,128,512} x split {1,2}
#   + unaligned M {3,50,100,300} x N {128,400,1040} (row-major: {136,400,1032}) x K {256,768,2304} x split {1,3,8} x batch {1, 2 + MXFP4_SFA_K_PATTERN}.
# Scale padding is poisoned with 0xFF (e8m0 NaN) by the host, so any padding reaching C is an error.
E=$1; G=$2; shift 2
run() { flock /tmp/gpu$G.lock env $1 HIP_VISIBLE_DEVICES=$G $E $2 -v 1 2>&1 | grep -oE 'errors=[0-9]+/' | head -1; }
for cfg in "$@"; do IFS=: read -r mode tile ks d lay <<< "$cfg"; lay=${lay:-0}; X="-l $lay -a $mode -t $tile -s $ks -d $d"; ok=0; tot=0
  NS="128 400 1040"; [ "$lay" = 1 ] && NS="136 400 1032"
  for nk in 4096:4096 2048:7168 3072:6144 4096:8192 4096:1024 3072:1536; do for m in 1 64 128 512; do for p in 1 2; do
    o=$(run "" "$X -p $p -m $m -n ${nk%%:*} -k ${nk##*:} -b 1"); tot=$((tot+1)); [ "$o" = "errors=0/" ] && ok=$((ok+1)) || echo "FAIL $cfg p$p m$m $nk: $o"; done; done; done
  for m in 3 50 100 300; do for n in $NS; do for k in 256 768 2304; do for p in 1 3 8; do for b in 1 2; do
    env=""; [ $b = 2 ] && env="MXFP4_SFA_K_PATTERN=1"
    o=$(run "$env" "$X -p $p -m $m -n $n -k $k -b $b"); tot=$((tot+1)); [ "$o" = "errors=0/" ] && ok=$((ok+1)) || echo "FAIL $cfg p$p m$m n$n k$k b$b: $o"; done; done; done; done; done
  echo "cfg $cfg: $ok/$tot ok"; done
