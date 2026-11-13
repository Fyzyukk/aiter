#!/bin/bash
# att.sh name srcdir ["-DFOO"] : build device code object from srcdir (+defines), ATT-trace 8192^3 on GPU ${GPU:-6}, CU ${ATT_CU:-1}
name=$1; src=$2; X=$3; H=$(cd $(dirname $0) && pwd)
SYM=_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
I=/root/workspace/aiter-blockscale-mxfp8-nn-nt-tt/csrc/include
CC=/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++
d=/tmp/att_tiles/t_$name; rm -rf $d; mkdir -p $d/src
cp $src/gemm_a4w4_mxfp4_4wave_kernel.cc $src/gemm_a4w4_mxfp4_4wave_traits.hpp $src/gemm_a4w4_mxfp4_4wave_kernel_template.hpp $d/src/
cd $d/src
$CC -x hip -I$I -I. -std=c++17 -O3 -ffast-math $X --offload-arch=gfx950 --cuda-device-only gemm_a4w4_mxfp4_4wave_kernel.cc -o ../kernel.co || exit 1
/opt/rocm/bin/hipcc -O2 -I. $H/launch.cpp -o ../launch 2>/dev/null || exit 1
cd $d; HIP_VISIBLE_DEVICES=${GPU:-6} timeout 600 rocprofv3 --att 1 --att-target-cu ${ATT_CU:-1} --att-shader-engine-mask 0x1 --att-buffer-size 268435456 -d att -o r -- ./launch kernel.co ${SHAPE:-8192 8192 8192} 2 $SYM > att.log 2>&1
echo "$name: $(find $d/att -name code.json | head -1)"
