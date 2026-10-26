// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_lds_panel_layout = opus_gemm_bpreshuffle_pingpong_lds_layout;

template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_lds_panel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    opus_gemm_bpreshuffle_pingpong_lds_body<Traits, opus_gemm_bpreshuffle_panel_schedule::midpoint>(
        kargs);
}

// Existing generated CO stubs retain their registered compute entry.
template <class Traits>
__device__ __forceinline__ void
opus_gemm_bpreshuffle_lds_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    opus_gemm_blockscale_mxscale_bpreshuffle_lds_panel<Traits>(kargs);
}
#endif

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU) void
opus_gemm_blockscale_mxscale_bpreshuffle_lds_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_lds_panel<Traits>(args);
#endif
}
