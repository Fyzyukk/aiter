// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_a8w8_blockscale_mxscale_bpreshuffle_ring_common_gfx950.cuh"

template<class Base>
struct opus_gemm_bpreshuffle_bounded_thread_traits : Base {
    static constexpr bool THREAD_RANGE_KNOWN=true;
    static_assert(Base::BLOCK_SIZE>0 && Base::BLOCK_SIZE%64==0);
};

// Optional tail-row minimum for valid launched workgroups. Older
// configurations retain their original generated code through the default path.
template<class Base>
struct opus_gemm_bpreshuffle_simple_tail_traits : Base {
    static constexpr bool SIMPLE_TAIL_ROWS=true;
    static_assert(Base::BLOCK_SIZE>0 && Base::BLOCK_SIZE%64==0);
};


constexpr int opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(int bm, int bn,
                                                                   int pipeline_bytes) {
    for (int rows = bm; rows >= 16; rows -= 16)
        if (bm % rows == 0 && rows * (bn + 8) * 2 <= pipeline_bytes)
            return rows;
    return 16;
}
template <int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false, int FixedK = 7168,
          int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2, int Output = 0, int BlockK = 256,
          bool Contiguous = false, bool PinA = false, bool PinC = false, bool Grid2D = false, int SplitK = 1, bool ByteScale = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits {
    static constexpr int WG_PER_CU = WGPerCU, CACHECTL_A = ACache, STORE_CACHE = StoreCache;
    static constexpr int B_M = BM, B_N = BN, B_K = BlockK;
    static constexpr int T_M = 1, T_N = WN, T_K = 1, BLOCK_SIZE = WN * 64;
    static constexpr int TOTAL_K = FixedK, SPLIT_K = SplitK;
    static constexpr int K = FixedK / SplitK, PANELS = K / B_K, SCALE_COLS = K / 128;
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4);
    static_assert(FixedK % (SplitK * B_K) == 0);
    static_assert(SplitK == 1 || Grid2D);
    static constexpr int REDUCE_VEC = 8, REDUCE_BLOCK = 128;
    static constexpr int MR = BM / 16, NR = BN / (WN * 16);
    static constexpr int OUTPUT = Output, E_M = MR, E_N = NR, W_M = 16, W_N = 16;
    static constexpr bool CONTIGUOUS = Contiguous, PIN_A = PinA, PIN_C = PinC;
    static constexpr bool GRID_2D = Grid2D;
    static_assert(!Grid2D || !XCD);
    static_assert(!PinA || (B_K / 128) * MR * 8 <= 128);
    static_assert(!PinC || MR * NR * 4 <= 128);
    static_assert(Output == 0 || Output == 2);
    static constexpr int STAGES = NB, B_AHEAD = BD, B_SETS = BD + 1, B_CACHE = BC;
    static constexpr bool XCD_ORDER = XCD, SB_SLOTS = SCALE_COLS % 4 != 0, C_AGPR = false, PERSTAGE = false;

    // The A ring and resident byte-scale panels share LDS with the output tile.
    static constexpr int A_BYTES = BM * B_K, STAGE_BYTES = A_BYTES;
    static constexpr int A_COPIES = A_BYTES / (BLOCK_SIZE * 16);
    static constexpr int SA_ROWS = (BM + 15) / 16 * 16;
    // Byte copies allow native column-major scales even when M is not a
    // multiple of 16. Async byte loads occupy one dword slot per lane in LDS.
    static constexpr bool BYTE_SCALE = ByteScale;
    static constexpr int SA_SCALE = ByteScale ? 4 : 1;
    static constexpr int SA_COPY_ELEMS = ByteScale ? BLOCK_SIZE : BLOCK_SIZE * 16;
    static constexpr int SA_COPIES =
        (SA_ROWS * SCALE_COLS + SA_COPY_ELEMS - 1) / SA_COPY_ELEMS;
    static constexpr int SA_BYTES = SA_COPIES * BLOCK_SIZE * (ByteScale ? 4 : 16);
    static constexpr int SB_ROWS = (BN + 127) / 128;
    static constexpr int SB_BYTES = BLOCK_SIZE * 4;
    static constexpr int OFF_SA = NB * STAGE_BYTES, OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int PIPELINE_BYTES = OFF_SB + SB_BYTES;
    static constexpr int C_ROWS =
        opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(BM, BN, PIPELINE_BYTES);
    static constexpr int LDS_BYTES = PIPELINE_BYTES;

    static constexpr int INITIAL_A = NB - 1, INITIAL_B = BD;
    static constexpr int STEADY = PANELS - INITIAL_A;
    static constexpr int B_ISSUES = NR * 2 * (B_K / 128);
    static constexpr int SCALE_ISSUES = SA_COPIES + 1;

    // Counts follow the actual prologue: B, first A, scales, remaining A.
    static constexpr int wait_a(int t) {
        int issued = INITIAL_B * B_ISSUES;
        int ready = -1;
        issued += A_COPIES;
        if (t == 0)
            ready = issued;
        issued += SCALE_ISSUES;
        // K0 consumes the resident scales as well as A0. The first barrier
        // must publish those async LDS writes before any scale LDS reads.
        if (t == 0)
            ready = issued;
        for (int stage = 1; stage < INITIAL_A; ++stage) {
            issued += A_COPIES;
            if (stage == t)
                ready = issued;
        }
        for (int step = 0; step < t; ++step) {
            if (step + BD < PANELS)
                issued += B_ISSUES;
            if (step + NB - 1 < PANELS) {
                issued += A_COPIES;
                if (step + NB - 1 == t)
                    ready = issued;
            }
        }
        return issued - ready;
    }
    static constexpr int wait_b(int t) {
        int issued = 0, ready = -1;
        for (int set = 0; set < INITIAL_B; ++set) {
            issued += B_ISSUES;
            if (set == t)
                ready = issued;
        }
        issued += INITIAL_A * A_COPIES + SCALE_ISSUES;
        for (int step = 0; step <= t; ++step) {
            if (step + BD < PANELS) {
                issued += B_ISSUES;
                if (step + BD == t)
                    ready = issued;
            }
            if (step + NB - 1 < PANELS)
                issued += A_COPIES;
        }
        return issued - ready;
    }

    static_assert(BM > 0 && BM % 16 == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(NB >= 2 && BD >= 1 && BD < NB);
    static_assert((BC == 0 || BC == 1 || BC == 2 || BC == 17) && K > 0 && K % B_K == 0);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BN <= 256 && (WGPerCU == 1 || WGPerCU == 2));
    static_assert(B_K == 128 || B_K == 256);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0 && PANELS >= INITIAL_A);
    static_assert(SB_ROWS * (SB_SLOTS ? SCALE_COLS : SCALE_COLS / 4) <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(MR * NR * 4 <= 256);
};
