// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage7_gfx950.cuh"

template<class Base, int Unroll, int Policy>
struct opus_stage11_pin_traits : Base {
    static_assert(Unroll == 1 || Unroll == 8);
    static_assert(Policy == 0 || Policy == 2);
    static constexpr int LOOP_UNROLL = Unroll;
    static constexpr int SCHEDULE_POLICY = Policy;
};
