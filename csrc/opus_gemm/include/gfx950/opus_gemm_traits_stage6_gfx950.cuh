// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"

// Only explicitly scoped COs use these wrappers. Existing pipelines and traits
// retain their definitions, input addressing and output bounds.
template<class Base, int StaticM, int StaticN>
struct opus_stage6_static_shape_traits : Base {
    static_assert(StaticM >= 0 && StaticN > 0);
    static constexpr int STATIC_M = StaticM;
    static constexpr int STATIC_N = StaticN;
};

// Actual fragment queue depth, independent of the BD multiplier's clipping.
// B_SETS changes with B_AHEAD; the existing pipeline recomputes its ring period
// and prologue/steady-state/epilogue waits from these two values.
template<class Base, int Ahead>
struct opus_stage6_b_queue_traits : Base {
    static_assert(Ahead >= 1 && Ahead <= Base::B_MAX_AHEAD);
    static_assert(Ahead < Base::PANELS * Base::B_FRAGMENTS);
    static constexpr int B_AHEAD = Ahead;
    static constexpr int B_SETS = Ahead + 1;
};
