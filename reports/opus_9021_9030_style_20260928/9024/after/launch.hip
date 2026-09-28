#include <hip/hip_runtime.h>
#include "pipeline_runtime.cuh"
template __global__ void gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<
    opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
