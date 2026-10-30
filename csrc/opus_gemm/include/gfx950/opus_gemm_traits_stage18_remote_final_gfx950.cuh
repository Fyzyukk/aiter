// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage15_pin_unroll_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh"

template<class Base, int Panel, bool Early>
struct opus_stage18_scale_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<Base, Panel> {
    static constexpr int LOOP_UNROLL = 2;
    static constexpr int SCHEDULE_POLICY = 0;
    static constexpr bool GUARD_SFB_LANE = true;
    static constexpr bool EARLY_SCALE_LOAD = Early;
};

template<class Base, bool SkipUnusedA, bool BFirst>
struct opus_stage18_interleave_traits : Base {
    static constexpr bool SKIP_UNUSED_A = SkipUnusedA;
    static constexpr bool B_FIRST = BFirst;
};

template<class Base, int Ring>
struct opus_stage18_m_repeat_traits : Base {
    static_assert(Ring >= 2 && Ring <= Base::E_M && Ring <= 3);
    static constexpr int A_RING = Ring;
};
