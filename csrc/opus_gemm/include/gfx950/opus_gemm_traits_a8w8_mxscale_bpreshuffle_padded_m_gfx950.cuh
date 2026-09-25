// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// Preserve the 256x256, four-wave compute schedule. Only the final M tile
// gets zero-filled operand loads and bounded output stores, in one launch.
struct opus_gemm_mxscale_bpreshuffle_padded_m_traits_gfx950
    : opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
    static constexpr bool PAD_M = true;
};
