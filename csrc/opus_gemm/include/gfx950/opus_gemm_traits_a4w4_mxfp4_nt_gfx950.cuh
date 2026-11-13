#pragma once

// Kernel arguments.  Units: fp4 tensors in BYTES (fp4x2 per byte, even k in
// the low nibble), C in fp32 elements, scales in bytes.  A is row-major [M, K/2]; B is in the aiter
// shuffle_weight(w, layout=(16, 16)) layout (stride_b stays K/2, the row-major row stride).  SFA/SFB are UE8M0 in
// the aiter gfx950 e8m0_shuffle / shuffle_scale layout (rows padded to 32,
// K/32 padded to 8, [rows/32][K/256][k_lane 4][row 16][kk 2][row-pair 2]);
// stride_sfa/sfb is the byte stride of one 32-row block (32 * K/32 padded to 8).
struct opus_gemm_mxfp4_scale_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
    int grid_blocks;          // persistent kernel: number of launched blocks (avoids reading the dispatch packet)
};

__host__ __device__ inline int ceil_div_scale(int a, int b) { return (a + b - 1) / b; }

// MXFP4 x MXFP4 NT, 4 waves (2x2), tile 256x256x256 (elements), 1x32 UE8M0.
// K tile = 256 fp4 = 128 bytes per row: the gmem/LDS geometry of the fp8
// 4-wave kernel.  One fp4 MFMA covers K = W_K = 128, so a K tile is
// E_K = 2 MFMAs along K (kk = 0: bytes [16g, +16) = K [32g, +32);
// kk = 1: bytes [64+16g, +16) = K [128+32g, +32), g = lane / 16).
template<int BM_, int BN_>
struct gemm_a4w4_mxfp4_4wave_traits_t {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = BM_;   // 256 or 128
    static constexpr int B_N = BN_;   // 256 or 128
    static_assert((B_M == 256 || B_M == 128) && (B_N == 256 || B_N == 128));
    static constexpr int B_K = 256;                            // fp4 elements
    static constexpr int ELEMS_PER_BYTE = 2;
    static constexpr int B_K_BYTES = B_K / ELEMS_PER_BYTE;     // 128

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static_assert(NUM_WAVES == T_M * T_N * T_K);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;                            // fp4 elements per MFMA
    static constexpr int W_K_BYTES = W_K / ELEMS_PER_BYTE;     // 64

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);         // 4
    static constexpr int E_N = HALF_B_N / (W_N * T_N);         // 4
    static constexpr int E_K = B_K / (W_K * T_K);              // 2
    static_assert(E_K == 2);

    static constexpr int VEC_A = 16;                           // bytes
    static constexpr int VEC_B = 16;                           // bytes
    static constexpr int VEC_A_ELEMS = 32;                     // fp4 elements
    static constexpr int VEC_B_ELEMS = 32;
    static_assert(VEC_A * ELEMS_PER_BYTE == VEC_A_ELEMS && VEC_B * ELEMS_PER_BYTE == VEC_B_ELEMS);
    static_assert(VEC_A <= 16 && VEC_B <= 16, "buffer_load ... lds moves at most 16 bytes per lane");
    static_assert(VEC_A_ELEMS == W_K / (WARP_SIZE / W_M));     // one b128 = one fp4 MFMA operand
    static constexpr int VEC_C = 4;

    // LDS: one wave issue = 64 lanes x 16 bytes = 1024 bytes = 8 rows, +32 pad.
    static constexpr int smem_padding = 64;   // s1: chunk stride 1088 B; with the 16 B swizzle the A/B ds_read_b128 are bank-conflict free
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;            // 1024
    static constexpr int smem_linear_wave_b = WARP_SIZE * VEC_B;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K_BYTES;       // 8 rows per chunk
    static constexpr int smem_sub_b = smem_linear_wave_b / B_K_BYTES;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub_a;                // 16 chunks per half
    // buffer_load ... lds (b128) per wave per half tile per K tile: 128 rows x 128 B / (256 lanes x 16 B).
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K_BYTES / (BLOCK_SIZE * VEC_A);   // 4
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K_BYTES / (BLOCK_SIZE * VEC_B);   // 4
    static constexpr int smem_n_rep = HALF_B_N / smem_sub_b;
    // Per-wave LDS-DMA issues per half tile (each issue = 64 lanes x 16 B = 8 rows of A / half a 16-row B block).
    static_assert(a_buffer_load_insts >= 2 && a_buffer_load_insts % 2 == 0 && b_buffer_load_insts >= 2 && b_buffer_load_insts % 2 == 0);
    // Scale panels: 32-row blocks, 2 blocks (2 x 512 B = one 1 KB DMA) per wave; with fewer blocks than 2 x NUM_WAVES the
    // remaining waves repeat the copies of the first ones (same data to the same LDS bytes), so every wave issues one DMA.
    static constexpr int SF_WAVES_A = B_M / 64;
    static constexpr int SF_WAVES_B = B_N / 64;
    // Top-half C copies per thread (bf16 x 8 per copy) and whether the C staging row swizzle applies (B_N = 256 only).
    static constexpr int C_COPIES = HALF_B_M * B_N / (BLOCK_SIZE * 8);
    static constexpr bool C_SWZ = (B_N == 256);

    // Scales: native row-major [rows, K/32] UE8M0, 8 columns per K tile.
    static constexpr int GROUP_K = 32;
    static constexpr int NUM_KGROUPS = B_K / GROUP_K;                       // 8
    // Global -> LDS: one b128 per lane = 16 columns of one row (lane = row,
    // 4 waves x 64 lanes = 256 rows).  Panel = 16 columns = 2 K tiles, double
    // buffered; LDS image [row 0..255][16 B].
    static constexpr int VEC_GLOBAL_SCALE = 16;
    static constexpr int SCALE_PANEL_GROUPS = VEC_GLOBAL_SCALE;                 // 16
    static constexpr int SCALE_PANEL_TILES = SCALE_PANEL_GROUPS / NUM_KGROUPS;  // 2
    static constexpr int SFA_PANEL_BYTES = B_M * SCALE_PANEL_GROUPS;            // 4096
    static constexpr int SFB_PANEL_BYTES = B_N * SCALE_PANEL_GROUPS;
    // Three panel buffers: with the rotated main loop the scales of tile t+1 are read
    // in the middle of iteration t, so a panel is copied about two tiles ahead.
    static constexpr int SCALE_PANEL_BUFFERS = 3;
    // Consumer: per tile one b64 per half per tensor = the tile's 8 columns
    // = [kk0 dword, kk1 dword] (the tile's offset in the panel row is (tile % 2) * 8).
    static constexpr int VEC_LDS_SCALE = NUM_KGROUPS;                           // 8 bytes

    static constexpr int SMEM_A_BYTES = smem_m_rep * (smem_linear_wave_a + smem_padding) * 4;  // 67584
    static constexpr int SMEM_B_BYTES = smem_n_rep * (smem_linear_wave_b + smem_padding) * 4;
    static constexpr int LDS_BYTES = SMEM_A_BYTES + SMEM_B_BYTES + SCALE_PANEL_BUFFERS * (SFA_PANEL_BYTES + SFB_PANEL_BYTES);
    static_assert(LDS_BYTES <= 163840);   // s1: padding 64 -> exactly 160 KB
};
using gemm_a4w4_mxfp4_4wave_traits = gemm_a4w4_mxfp4_4wave_traits_t<256, 256>;
