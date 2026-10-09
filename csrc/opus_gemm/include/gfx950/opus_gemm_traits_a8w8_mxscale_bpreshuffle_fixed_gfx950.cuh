// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"

// Opt-in FixedK experiments inherit exactly the original geometry, LDS ring,
// scale panel and output layout. Long K still refills the original SP32 panel;
// FixedK is the total reduction length, not the scale panel capacity.
template<int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x128_fixed_traits_gfx950
    : opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<3, 32, 0> {
    static constexpr int FIXED_K = FixedK;
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
};

template<int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_160x128_fixed_traits_gfx950
    : opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
    static constexpr int FIXED_K = FixedK;
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
};
