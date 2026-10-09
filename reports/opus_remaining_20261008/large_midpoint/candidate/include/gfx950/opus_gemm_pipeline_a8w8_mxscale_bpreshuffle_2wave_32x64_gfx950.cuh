// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_2wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_2wave_32x64_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_2wave_32x64_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_scale_2wave_32x64_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    gemm_a8w8_mxfp8_scale_2wave<Traits>(kargs);
}
#endif
