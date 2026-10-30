// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage14_chunk_scale_gfx950.cuh"

template<class Base, int Panel, int Unroll>
struct opus_stage15_pin_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<Base, Panel> {
    static_assert(Panel == 32 || Panel == 64);
    static_assert(Unroll == 2 || Unroll == 4 || Unroll == 8);
    static constexpr bool GUARD_SFB_LANE = true;
    static constexpr int LOOP_UNROLL = Unroll;
    static constexpr int SCHEDULE_POLICY = 0;
};
