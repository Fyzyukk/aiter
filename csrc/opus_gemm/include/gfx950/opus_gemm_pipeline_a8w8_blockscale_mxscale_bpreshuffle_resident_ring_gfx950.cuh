// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_xcd_ring_gfx950.cuh"

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_resident_ring_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    static_assert(Traits::SPLIT_K == 1);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_xcd_ring_detail::body<Traits>(
        args, reinterpret_cast<opus::bf16_t*>(args.ptr_c), nullptr);
#endif
}
