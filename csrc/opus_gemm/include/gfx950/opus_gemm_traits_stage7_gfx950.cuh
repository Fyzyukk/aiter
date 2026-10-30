// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage6_gfx950.cuh"
#include "opus_gemm_traits_stage7_b_base_gfx950.cuh"

template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int FixedK = 7168, int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false, int GroupM = 0, int PublishM = 0>
struct opus_stage7_prefetch_b_overlap_base_traits
    : opus_stage7_prefetch_b_base_traits<BM, BN, NB, BD, BC, XCD, FixedK, WN,
        WGPerCU, ACache, StoreCache, Output, BlockK, Contiguous, PinA, PinC, GroupM> {
    static constexpr int PUBLISH_M = PublishM;
    static_assert(PublishM >= 0 && PublishM < BM / 16 - 1);
};

template<class Base, bool StreamA, bool StaticRing, int Output>
struct opus_stage7_b_traits : Base {
    static constexpr bool STREAM_A = StreamA;
    static constexpr bool STATIC_B_RING = StaticRing;
    static constexpr int OUTPUT = Output < 0 ? Base::OUTPUT : Output;
    static_assert(OUTPUT == 0 || OUTPUT == 2 || OUTPUT == 3);
};

template<class Base, int Policy>
struct opus_stage7_pin_traits : Base {
    static constexpr int SCHEDULE_POLICY = Policy;
    static_assert(Policy >= 0 && Policy <= 2);
};
