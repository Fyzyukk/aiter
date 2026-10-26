// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_xcd_ring_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_resident_ring(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    static_assert(Traits::SPLIT_K == 1);
    opus_gemm_blockscale_mxscale_bpreshuffle_ring<Traits, opus_gemm_bpreshuffle_ring_loop::unrolled>(
        kargs, reinterpret_cast<opus::bf16_t*>(kargs.ptr_c), nullptr);
}
#endif

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_blockscale_mxscale_bpreshuffle_resident_ring_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args)
{
    static_assert(Traits::SPLIT_K == 1);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_resident_ring<Traits>(args);
#endif
}
