#!/bin/bash
E=/tmp/opus_mxfp4/sk4/build/gemm_a4w4_mxfp4_4wave.exe
for t in 128x128 192x256 256x192; do /tmp/opus_mxfp4/check.sh $E 6 $t "-p 1"; done
for s in 1 2 4 8; do KS="2048 4096 6144" /tmp/opus_mxfp4/check.sh $E 6 128x128 "-p 2 -s $s"; done
for s in 2 8; do for t in 256x128 128x256 192x256 256x192; do KS="2048 4096 6144" /tmp/opus_mxfp4/check.sh $E 6 $t "-p 2 -s $s"; done; done
/tmp/opus_mxfp4/check.sh $E 6 256x256 "-p 0 -o 0"
echo DONE
