#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950
    : opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
    static constexpr bool PAD_M = true;
};

struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950
    : opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950 {
    static constexpr int LOOP_UNROLL = 4;
};
