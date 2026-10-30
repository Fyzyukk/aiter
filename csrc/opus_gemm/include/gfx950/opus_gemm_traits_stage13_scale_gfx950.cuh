// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage11_unroll_gfx950.cuh"

template<class Base, int Panel>
struct opus_stage13_scale_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<Base, Panel> {
    static_assert(Panel == 16 || Panel == 32 || Panel == 64);
    static constexpr bool GUARD_SFB_LANE = true;
    static constexpr int LOOP_UNROLL = 8;
    static constexpr int SCHEDULE_POLICY = 0;
};
