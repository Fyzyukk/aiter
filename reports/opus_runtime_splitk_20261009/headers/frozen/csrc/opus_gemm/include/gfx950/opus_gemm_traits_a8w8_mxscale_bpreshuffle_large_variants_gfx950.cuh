#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"

struct opus_gemm_mxscale_bpreshuffle_large_output_panel16_traits
    : opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
    static constexpr int FIXED_K = 1536;
    static constexpr int MAX_K = FIXED_K;
    static constexpr int SCALE_PANEL = 16;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
    static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SCALE_PANEL;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
    static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
    static constexpr int SFA_PASSES = (B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static_assert(FIXED_K / B_K == 12 && SCALE_PANEL >= FIXED_K / B_K);
    static_assert(SFA_PASSES == 1 && SFA_BYTES == 3072 && SFB_BYTES == 64);
    static_assert(MATRIX_LDS_BYTES == 118272 && LDS_BYTES == 121408);
    static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
};

struct opus_gemm_mxscale_bpreshuffle_large_output_direct_b_traits
    : opus_gemm_mxscale_bpreshuffle_large_output_panel16_traits {
    static constexpr int NUM_STAGES = 3;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * A_STAGE;
    static constexpr int B_DIRECT_SETS = 2;
    static constexpr int C_CHUNK_ROWS = 96;
    static constexpr int C_CHUNKS = B_M / C_CHUNK_ROWS;
    static constexpr int C_CHUNK_BYTES = C_CHUNK_ROWS * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t);
    static constexpr int CHUNK_OUTPUT_PASSES = C_CHUNK_ROWS * B_N / (BLOCK_SIZE * VEC_OUTPUT);
    static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static_assert(B_DIRECT_SETS == 2 && NUM_STAGES == 3 && A_VMEM_INSTRUCTIONS == 3);
    static_assert(MATRIX_LDS_BYTES == 76032 && LDS_BYTES == 79168 && LDS_BYTES <= 80 * 1024);
    static_assert(C_CHUNKS == 2 && C_CHUNK_BYTES == 50688 && CHUNK_OUTPUT_PASSES == 6);
    static_assert(C_CHUNK_BYTES <= LDS_BYTES && C_CHUNK_ROWS % W_M == 0);
};
