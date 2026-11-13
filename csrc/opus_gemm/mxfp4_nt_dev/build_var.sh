#!/bin/bash
# build_var.sh name ["-DFOO ..."] : build the dev harness into /tmp/opus_mxfp4/<name>/build (exe: build/gemm_a4w4_mxfp4_4wave.exe)
n=$1; x=$2; D=$(cd $(dirname $0) && pwd); o=/tmp/opus_mxfp4/$n
rm -rf $o && mkdir -p $o && make -s -C $D BUILD=$o/build EXTRA="$x" >$o/build.log 2>&1 && echo "$n ok" || { echo "$n FAIL"; tail -20 $o/build.log; }
