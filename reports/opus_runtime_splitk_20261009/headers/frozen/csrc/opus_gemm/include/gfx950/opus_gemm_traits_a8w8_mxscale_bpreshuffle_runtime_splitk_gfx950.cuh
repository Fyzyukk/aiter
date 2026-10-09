// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_variants_gfx950.cuh"
#include "opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh"

// Capacity is bounded by split one, while each launch allocates only the LDS
// needed by its largest balanced partition. Split count does not alter MMA.
template<int BM, int BN, int WM, int WN>
struct opus_gemm_mxscale_bpreshuffle_narrow_fine_runtime_traits
    : opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<BM, BN, WM, WN, 1> {
    using Base = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<BM, BN, WM, WN, 1>;
    static constexpr int lds_bytes(int k, int split_k) {
        const int total_loops = k / Base::B_K;
        const int loops = total_loops / split_k + (total_loops % split_k != 0);
        const int stages = loops < Base::NUM_STAGES ? loops : Base::NUM_STAGES;
        return stages * (Base::A_STAGE + Base::B_STAGE) +
            (Base::REGISTER_SCALES ? 0 : (BM + Base::B_GROUPS) * loops);
    }
};

// Local K waves and operand queues remain static; global split is an argument.
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_runtime_traits
    : opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
    static constexpr int REDUCE_VEC = 16, REDUCE_BLOCK = 128;
    static_assert(WaveK == 1 || WaveK == 2);
    static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
                  (BM == 32 && (BN == 32 || BN == 64)));
};
