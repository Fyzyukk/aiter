#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"

// K128 arithmetic and raw scales retained. Each global partition completes its
// local K-wave reduction before writing exactly one FP32 partial per output.
template<int BM, int BN, int WaveK>
struct opus_private_register_split4_traits
    : opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
    static constexpr int GLOBAL_SPLIT_K = 4;
    static_assert(WaveK == 1 || WaveK == 2);
    static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
                  (BM == 32 && (BN == 32 || BN == 64)));
};

// The existing common fine-M LDS loader, scale handoff, conservative waits,
// output packing and separate reducer are reused with smaller new geometry.
template<int BM, int BN, int WM, int WN, int SplitK>
struct opus_private_narrow_fine_traits
    : opus_gemm_small_lds_traits_gfx950<BM, BN, WM, WN, 4, 1, 2,
        false, false, true, true, true, SplitK, 16, 128, 2, 0, true> {
    static_assert((BM == 48 && BN == 64 && WM == 1 && WN == 4) ||
                  ((BM == 64 || BM == 96) && BN == 128 && WM == 2 && WN == 2));
    static_assert(SplitK == 1 || SplitK == 2);
};
